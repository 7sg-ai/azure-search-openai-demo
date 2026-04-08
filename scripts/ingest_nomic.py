"""
Data ingestion script for Azure AI Search with Nomic embeddings (Triton).
Uploads PDFs to blob storage, creates a 768-dim vector search index,
computes embeddings via Nomic Triton, and indexes all documents.

Requires env vars:
  ARM_TENANT_ID, ARM_CLIENT_ID, ARM_CLIENT_SECRET, ARM_SUBSCRIPTION_ID
  AZURE_RESOURCE_GROUP
  NOMIC_EMBED_URL, CF_ACCESS_CLIENT_ID, CF_ACCESS_CLIENT_SECRET
"""

import os
import sys
import json
import glob
import time
import requests
import hashlib
from pypdf import PdfReader
from azure.storage.blob import BlobServiceClient
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
    VectorSearch,
    VectorSearchProfile,
    HnswAlgorithmConfiguration,
    HnswParameters,
    VectorSearchAlgorithmMetric,
)
from azure.core.credentials import AzureKeyCredential


def get_arm_token():
    tid = os.environ["ARM_TENANT_ID"]
    cid = os.environ["ARM_CLIENT_ID"]
    cs = os.environ["ARM_CLIENT_SECRET"]
    r = requests.post(
        f"https://login.microsoftonline.com/{tid}/oauth2/v2.0/token",
        data={"grant_type": "client_credentials", "client_id": cid,
              "client_secret": cs, "scope": "https://management.azure.com/.default"},
        timeout=15,
    )
    return r.json()["access_token"]


def discover_resources():
    arm = get_arm_token()
    ah = {"Authorization": f"Bearer {arm}", "Content-Type": "application/json"}
    sid = os.environ["ARM_SUBSCRIPTION_ID"]
    rg = os.environ["AZURE_RESOURCE_GROUP"]
    base = "https://management.azure.com"

    resources = requests.get(
        f"{base}/subscriptions/{sid}/resourceGroups/{rg}/resources?api-version=2021-04-01",
        headers=ah, timeout=30,
    ).json().get("value", [])

    # Storage
    sa = next(r for r in resources if "storageAccounts" in r["type"])
    sa_keys = requests.post(
        f'{base}{sa["id"]}/listKeys?api-version=2023-05-01', headers=ah, timeout=30
    ).json()
    sa_name = sa["name"]
    sa_key = sa_keys["keys"][0]["value"]

    # Search
    search = next(r for r in resources if "searchServices" in r["type"])
    search_name = search["name"]
    search_keys = requests.post(
        f'{base}{search["id"]}/listAdminKeys?api-version=2023-11-01', headers=ah, timeout=30
    ).json()
    search_key = search_keys["primaryKey"]

    return sa_name, sa_key, search_name, search_key


def upload_documents(sa_name, sa_key, data_dir="data"):
    print("--- Uploading documents to blob storage ---")
    svc = BlobServiceClient(f"https://{sa_name}.blob.core.windows.net", credential=sa_key)
    try:
        svc.create_container("content")
    except Exception:
        pass
    cc = svc.get_container_client("content")
    existing = [b.name for b in cc.list_blobs()]

    for filepath in glob.glob(os.path.join(data_dir, "*")):
        if os.path.isdir(filepath):
            continue
        fname = os.path.basename(filepath)
        if fname in existing:
            print(f"  Exists: {fname}")
        else:
            with open(filepath, "rb") as f:
                cc.upload_blob(fname, f, overwrite=True)
            print(f"  Uploaded: {fname}")


def create_search_index(search_name, search_key, index_name="gptkbindex"):
    print(f"--- Creating search index: {index_name} ---")
    client = SearchIndexClient(
        endpoint=f"https://{search_name}.search.windows.net",
        credential=AzureKeyCredential(search_key),
    )

    existing = [idx.name for idx in client.list_indexes()]
    if index_name in existing:
        print(f"  Index {index_name} already exists, deleting...")
        client.delete_index(index_name)

    index = SearchIndex(
        name=index_name,
        fields=[
            SimpleField(name="id", type=SearchFieldDataType.String, key=True, filterable=True),
            SearchableField(name="content", type=SearchFieldDataType.String),
            SimpleField(name="category", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="sourcepage", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="sourcefile", type=SearchFieldDataType.String, filterable=True),
            SearchField(
                name="embedding",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=768,
                vector_search_profile_name="vp",
            ),
        ],
        vector_search=VectorSearch(
            algorithms=[
                HnswAlgorithmConfiguration(
                    name="algo",
                    parameters=HnswParameters(metric=VectorSearchAlgorithmMetric.COSINE),
                )
            ],
            profiles=[VectorSearchProfile(name="vp", algorithm_configuration_name="algo")],
        ),
        semantic_search=SemanticSearch(
            configurations=[
                SemanticConfiguration(
                    name="default",
                    prioritized_fields=SemanticPrioritizedFields(
                        content_fields=[SemanticField(field_name="content")]
                    ),
                )
            ],
            default_configuration_name="default",
        ),
    )
    client.create_index(index)
    print(f"  Created index with 768-dim vector field")


def extract_text_from_pdf(filepath):
    reader = PdfReader(filepath)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append({"text": text, "page": i + 1})
    return pages


def chunk_text(text, chunk_size=2000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def compute_nomic_embeddings(texts):
    url = os.environ["NOMIC_EMBED_URL"]
    headers = {"Content-Type": "application/json"}
    if cf_id := os.environ.get("CF_ACCESS_CLIENT_ID"):
        headers["CF-Access-Client-Id"] = cf_id
    if cf_secret := os.environ.get("CF_ACCESS_CLIENT_SECRET"):
        headers["CF-Access-Client-Secret"] = cf_secret

    prefixed = [f"search_document: {t}" for t in texts]

    # Batch in groups of 32
    all_embeddings = []
    for i in range(0, len(prefixed), 32):
        batch = prefixed[i : i + 32]
        body = {
            "inputs": [{"name": "text", "datatype": "BYTES", "shape": [len(batch), 1], "data": batch}],
            "outputs": [{"name": "embeddings"}],
        }
        r = requests.post(url, headers=headers, json=body, timeout=60)
        if r.status_code != 200:
            raise Exception(f"Nomic embedding failed: {r.status_code} {r.text[:200]}")
        flat = r.json()["outputs"][0]["data"]
        dim = 768
        for j in range(len(batch)):
            all_embeddings.append(flat[j * dim : (j + 1) * dim])

    return all_embeddings


def index_documents(search_name, search_key, data_dir="data", index_name="gptkbindex"):
    print("--- Indexing documents with Nomic embeddings ---")
    client = SearchClient(
        endpoint=f"https://{search_name}.search.windows.net",
        index_name=index_name,
        credential=AzureKeyCredential(search_key),
    )

    total_chunks = 0
    for filepath in sorted(glob.glob(os.path.join(data_dir, "*.pdf"))):
        fname = os.path.basename(filepath)
        print(f"  Processing: {fname}")

        pages = extract_text_from_pdf(filepath)
        all_chunks = []
        for page_info in pages:
            chunks = chunk_text(page_info["text"])
            for ci, chunk in enumerate(chunks):
                chunk_id = hashlib.md5(f"{fname}-{page_info['page']}-{ci}".encode()).hexdigest()
                all_chunks.append({
                    "id": chunk_id,
                    "content": chunk,
                    "category": "",
                    "sourcepage": f"{fname}#page={page_info['page']}",
                    "sourcefile": fname,
                    "text_for_embedding": chunk,
                })

        if not all_chunks:
            print(f"    No text extracted, skipping")
            continue

        texts = [c["text_for_embedding"] for c in all_chunks]
        embeddings = compute_nomic_embeddings(texts)

        documents = []
        for chunk, emb in zip(all_chunks, embeddings):
            documents.append({
                "id": chunk["id"],
                "content": chunk["content"],
                "category": chunk["category"],
                "sourcepage": chunk["sourcepage"],
                "sourcefile": chunk["sourcefile"],
                "embedding": emb,
            })

        # Upload in batches of 100
        for i in range(0, len(documents), 100):
            batch = documents[i : i + 100]
            client.upload_documents(batch)

        print(f"    Indexed {len(documents)} chunks")
        total_chunks += len(documents)

    # Also index markdown files
    for filepath in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        fname = os.path.basename(filepath)
        print(f"  Processing: {fname}")
        with open(filepath, "r") as f:
            text = f.read()

        chunks = chunk_text(text)
        texts = [f"{fname}-{i}" for i in range(len(chunks))]
        chunk_ids = [hashlib.md5(t.encode()).hexdigest() for t in texts]

        embeddings = compute_nomic_embeddings(chunks)
        documents = [
            {
                "id": chunk_ids[i],
                "content": chunks[i],
                "category": "",
                "sourcepage": fname,
                "sourcefile": fname,
                "embedding": embeddings[i],
            }
            for i in range(len(chunks))
        ]
        client.upload_documents(documents)
        print(f"    Indexed {len(documents)} chunks")
        total_chunks += len(documents)

    return total_chunks


def verify_index(search_name, search_key, index_name="gptkbindex"):
    print("--- Verifying index ---")
    client = SearchClient(
        endpoint=f"https://{search_name}.search.windows.net",
        index_name=index_name,
        credential=AzureKeyCredential(search_key),
    )
    results = client.search("*", top=1, include_total_count=True)
    count = results.get_count()
    print(f"  Total documents in index: {count}")
    return count


def main():
    data_dir = os.environ.get("DATA_DIR", "data")
    if not os.path.exists(data_dir):
        # Try repo root
        repo_data = os.path.join(os.environ.get("HOME", ""), "repo", "data")
        if os.path.exists(repo_data):
            data_dir = repo_data
        else:
            print(f"ERROR: data directory not found at {data_dir} or {repo_data}")
            sys.exit(1)

    print(f"Data directory: {data_dir}")
    print(f"Files: {glob.glob(os.path.join(data_dir, '*'))}")

    sa_name, sa_key, search_name, search_key = discover_resources()
    print(f"Storage: {sa_name}")
    print(f"Search: {search_name}")

    upload_documents(sa_name, sa_key, data_dir)
    create_search_index(search_name, search_key)
    total = index_documents(search_name, search_key, data_dir)
    count = verify_index(search_name, search_key)

    print(f"\n=== INGESTION COMPLETE ===")
    print(f"  Documents indexed: {total}")
    print(f"  Index count: {count}")
    print(f"  Embedding model: Nomic (768-dim)")
    print(f"  Search index: gptkbindex")


if __name__ == "__main__":
    main()
