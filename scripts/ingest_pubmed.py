#!/usr/bin/env python3
"""Ingest CAR-T literature from PubMed into the cart_literature collection.

Fetches abstracts via NCBI E-utilities, classifies by CAR-T development
stage, generates BGE-small embeddings, and stores in Milvus.

Usage:
    python scripts/ingest_pubmed.py --max-results 5000
    python scripts/ingest_pubmed.py --query "CD19 CAR-T" --max-results 500
    python scripts/ingest_pubmed.py --dry-run  # Preview without storing

Author: Adam Jones
Date: February 2026
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    parser = argparse.ArgumentParser(
        description="Ingest CAR-T literature from PubMed"
    )
    parser.add_argument(
        "--query",
        default=(
            '"chimeric antigen receptor"[Title/Abstract] OR '
            '"CAR-T"[Title/Abstract] OR '
            '"CAR T cell"[Title/Abstract]'
        ),
        help="PubMed search query",
    )
    parser.add_argument(
        "--max-results", type=int, default=5000,
        help="Maximum number of abstracts to fetch",
    )
    parser.add_argument(
        "--batch-size", type=int, default=100,
        help="Embedding and insert batch size",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Fetch and parse but don't store in Milvus",
    )
    args = parser.parse_args()

    print("=" * 65)
    print("  CAR-T Intelligence Agent — PubMed Ingest")
    print("=" * 65)
    print(f"  Query: {args.query[:60]}...")
    print(f"  Max results: {args.max_results}")
    print(f"  Dry run: {args.dry_run}")
    print()

    # TODO: Implement when infrastructure is ready
    #
    # from src.utils.pubmed_client import PubMedClient
    # from src.ingest.literature_parser import PubMedIngestPipeline
    # from src.collections import CARTCollectionManager
    #
    # 1. client = PubMedClient(api_key=settings.NCBI_API_KEY)
    # 2. pmids = client.search(args.query, max_results=args.max_results)
    # 3. abstracts = client.fetch_abstracts(pmids)
    # 4. pipeline = PubMedIngestPipeline(collection_manager, embedder)
    # 5. records = pipeline.parse(abstracts)
    # 6. if not args.dry_run: pipeline.embed_and_store(records, "cart_literature")

    print("  [SCAFFOLD] PubMed ingest pipeline ready for implementation.")
    print("  Run with --dry-run to test parsing without Milvus.")
    print()


if __name__ == "__main__":
    main()
