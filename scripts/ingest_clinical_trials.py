#!/usr/bin/env python3
"""Ingest CAR-T clinical trials from ClinicalTrials.gov.

Fetches trial data via the ClinicalTrials.gov API v2, parses into
ClinicalTrial models, generates embeddings, and stores in Milvus.

Usage:
    python scripts/ingest_clinical_trials.py --max-results 1500
    python scripts/ingest_clinical_trials.py --antigen CD19
    python scripts/ingest_clinical_trials.py --dry-run

Author: Adam Jones
Date: February 2026
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    parser = argparse.ArgumentParser(
        description="Ingest CAR-T clinical trials from ClinicalTrials.gov"
    )
    parser.add_argument(
        "--max-results", type=int, default=1500,
        help="Maximum trials to fetch",
    )
    parser.add_argument(
        "--antigen", type=str, default=None,
        help="Filter by target antigen (e.g., CD19, BCMA)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Fetch and parse but don't store",
    )
    args = parser.parse_args()

    print("=" * 65)
    print("  CAR-T Intelligence Agent — ClinicalTrials.gov Ingest")
    print("=" * 65)
    print(f"  Max results: {args.max_results}")
    if args.antigen:
        print(f"  Antigen filter: {args.antigen}")
    print()

    # TODO: Implement
    # ClinicalTrials.gov API v2: GET https://clinicaltrials.gov/api/v2/studies
    # Query params: query.cond=CAR-T, query.intr=chimeric antigen receptor
    # Fields: NCTId, BriefTitle, BriefSummary, Phase, OverallStatus,
    #         LeadSponsorName, EnrollmentCount, StartDate, ResultsFirstPostDate

    print("  [SCAFFOLD] ClinicalTrials.gov ingest pipeline ready.")
    print()


if __name__ == "__main__":
    main()
