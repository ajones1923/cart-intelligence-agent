# CAR-T Intelligence Agent

Cross-functional intelligence across the CAR-T cell therapy development lifecycle. Part of the [HCLS AI Factory](https://github.com/ajones1923/hcls-ai-factory).

## Overview

The CAR-T Intelligence Agent breaks down data silos across the 5 stages of CAR-T development. It searches across **all data sources simultaneously** and synthesizes cross-functional insights powered by Claude.

| Collection | Records | Source |
|---|---|---|
| **Literature** | 4,995 | PubMed abstracts via NCBI E-utilities |
| **Clinical Trials** | 973 | ClinicalTrials.gov API v2 |
| **CAR Constructs** | 6 | 6 FDA-approved CAR-T products |
| **Assay Results** | 0 | Ready for data (schema + ingest pipeline built) |
| **Manufacturing** | 0 | Ready for data (schema + ingest pipeline built) |
| **Total** | **5,974 vectors** | |

### Example Queries

```
"Why do CD19 CAR-T therapies fail in relapsed B-ALL?"
"Compare 4-1BB vs CD28 costimulatory domains for DLBCL"
"What manufacturing parameters predict clinical response?"
"BCMA CAR-T resistance mechanisms in multiple myeloma"
"How does T-cell exhaustion affect CAR-T persistence?"
```

All queries return grounded, cross-collection answers with `[Literature:PMID]` and `[Trial:NCT...]` citations.

## Architecture

```
User Query
    |
    v
[BGE-small-en-v1.5 Embedding] (384-dim, asymmetric query prefix)
    |
    v
[Parallel Search: 5 Milvus Collections] (IVF_FLAT / COSINE)
    |               |               |               |               |
    v               v               v               v               v
Literature      Trials        Constructs        Assays       Manufacturing
 4,995           973              6               0               0
    |               |               |               |               |
    +-------+-------+-------+-------+-------+-------+
            |
            v
    [Query Expansion] (6 maps, 111 keywords -> 1,086 terms)
            |
            v
    [Knowledge Graph Augmentation]
    (25 antigens, 8 toxicities, 10 mfg processes)
            |
            v
    [Claude LLM] -> Grounded response with citations
```

Built on the HCLS AI Factory platform:

- **Vector DB:** Milvus 2.4 with IVF_FLAT/COSINE indexes (nlist=1024, nprobe=16)
- **Embeddings:** BGE-small-en-v1.5 (384-dim)
- **LLM:** Claude Sonnet 4.6 (Anthropic API)
- **UI:** Streamlit (port 8520)
- **Hardware target:** NVIDIA DGX Spark ($3,999)

## Setup

### Prerequisites

- Python 3.10+
- Milvus 2.4 running on `localhost:19530`
- `ANTHROPIC_API_KEY` environment variable (or in `rag-chat-pipeline/.env`)

### Install

```bash
cd ai_agent_adds/cart_intelligence_agent
pip install -r requirements.txt
```

### 1. Create Collections and Seed FDA Constructs

```bash
python3 scripts/setup_collections.py --seed-constructs
```

This creates 5 Milvus collections with IVF_FLAT indexes and inserts 6 FDA-approved CAR-T products (Kymriah, Yescarta, Tecartus, Breyanzi, Abecma, Carvykti).

### 2. Ingest PubMed Literature (~15 min)

```bash
python3 scripts/ingest_pubmed.py --max-results 5000
```

Fetches CAR-T abstracts via NCBI E-utilities (esearch + efetch), classifies by development stage, extracts target antigens, embeds with BGE-small, and stores in `cart_literature`.

### 3. Ingest Clinical Trials (~3 min)

```bash
python3 scripts/ingest_clinical_trials.py --max-results 1500
```

Fetches CAR-T trials via ClinicalTrials.gov API v2, extracts phase/status/sponsor/antigen/generation, embeds, and stores in `cart_trials`.

### 4. Validate

```bash
python3 scripts/validate_e2e.py
```

Runs 5 tests: collection stats, single-collection search, multi-collection `search_all()`, filtered search (`target_antigen == "CD19"`), and all demo queries.

### 5. Run Integration Test (requires API key)

```bash
python3 scripts/test_rag_pipeline.py
```

Tests the full RAG pipeline: embed -> search_all -> knowledge graph -> Claude LLM response generation. Validates both synchronous and streaming modes.

### 6. Launch UI

```bash
streamlit run app/cart_ui.py --server.port 8520
```

## Project Structure

```
cart_intelligence_agent/
├── Docs/
│   └── CART_Intelligence_Agent_Design.md  # Architecture design document
├── src/
│   ├── models.py                  # Pydantic data models (15 models + enums)
│   ├── collections.py             # 5 Milvus collection schemas + manager
│   ├── knowledge.py               # Knowledge graph (25 targets, 8 toxicities, 10 mfg)
│   ├── query_expansion.py         # 6 expansion maps (111 keywords -> 1,086 terms)
│   ├── rag_engine.py              # Multi-collection RAG engine + Claude
│   ├── agent.py                   # CAR-T Intelligence Agent (plan -> search -> synthesize)
│   ├── ingest/
│   │   ├── base.py                # Base ingest pipeline (fetch -> parse -> embed -> store)
│   │   ├── literature_parser.py   # PubMed NCBI E-utilities ingest
│   │   ├── clinical_trials_parser.py  # ClinicalTrials.gov API v2 ingest
│   │   ├── construct_parser.py    # CAR construct data parser
│   │   └── assay_parser.py        # Assay / manufacturing data parser
│   └── utils/
│       └── pubmed_client.py       # NCBI E-utilities HTTP client
├── app/
│   └── cart_ui.py                 # Streamlit chat interface (NVIDIA theme)
├── config/
│   └── settings.py                # Pydantic BaseSettings configuration
├── scripts/
│   ├── setup_collections.py       # Create collections + seed FDA constructs
│   ├── ingest_pubmed.py           # CLI: ingest PubMed CAR-T literature
│   ├── ingest_clinical_trials.py  # CLI: ingest ClinicalTrials.gov trials
│   ├── validate_e2e.py            # End-to-end data layer validation
│   ├── test_rag_pipeline.py       # Full RAG + LLM integration test
│   └── seed_knowledge.py          # Export knowledge graph to JSON
├── requirements.txt
└── LICENSE                        # Apache 2.0
```

**23 Python files | 6,836 lines | Apache 2.0**

## Knowledge Graph

| Component | Count | Examples |
|---|---|---|
| Target Antigens | 25 | CD19, BCMA, CD22, CD20, CD30, HER2, GPC3, EGFR, Mesothelin, GPRC5D, ... |
| FDA-Approved Products | 6 | Kymriah, Yescarta, Tecartus, Breyanzi, Abecma, Carvykti |
| Toxicity Profiles | 8 | CRS, ICANS, B-cell aplasia, HLH/MAS, cytopenias, TLS, GvHD, on-target/off-tumor |
| Manufacturing Processes | 10 | Transduction, expansion, leukapheresis, cryopreservation, release testing, ... |
| Query Expansion Maps | 6 | Target Antigen, Disease, Toxicity, Manufacturing, Mechanism, Construct |
| Expansion Keywords | 111 | Mapping to 1,086 related terms |

## Performance

Measured on NVIDIA DGX Spark (GB10 GPU, 128GB unified memory):

| Metric | Value |
|---|---|
| PubMed ingest (4,995 abstracts) | ~15 min |
| ClinicalTrials.gov ingest (973 trials) | ~3 min |
| Vector search (5 collections, top-5 each) | 12-16 ms (cached) |
| Full RAG query (search + Claude) | ~24 sec |
| Cosine similarity scores | 0.74 - 0.90 |

## Status

- **Week 1 (Scaffold)** -- Complete. Architecture, data models, collection schemas, knowledge graph, ingest pipelines, RAG engine, agent, and Streamlit UI.
- **Week 2 Days 1-3 (Data)** -- Complete. PubMed (4,995) + ClinicalTrials.gov (973) + FDA constructs (6) ingested. End-to-end validation passing.
- **Week 2 Days 4-5 (Integration)** -- Complete. Full RAG pipeline with Claude LLM generating grounded cross-functional answers. Streamlit UI working.

## Credits

- **Adam Jones** -- HCLS AI Factory, 14+ years genomic research
- **TJ Chen (NVIDIA)** -- "One Unified CAR-T Intelligence Platform" concept
- **Apache 2.0 License**
