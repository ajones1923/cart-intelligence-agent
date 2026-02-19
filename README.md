# CAR-T Intelligence Agent

Cross-functional intelligence across the CAR-T cell therapy development lifecycle. Part of the [HCLS AI Factory](https://github.com/your-org/hcls-ai-factory).

## Overview

The CAR-T Intelligence Agent enables cross-functional queries that search across **all stages** of CAR-T development simultaneously:

- **Literature** — PubMed abstracts on CAR-T research (~5,000)
- **Clinical Trials** — ClinicalTrials.gov records (~1,500)
- **CAR Constructs** — FDA-approved products + published designs
- **Assay Results** — In vitro / in vivo testing data
- **Manufacturing** — Vector production and CMC records
- **Genomic Evidence** — Existing ClinVar/AlphaMissense variants

### Example Queries

```
"Why do CD19 CAR-T therapies fail in relapsed B-ALL?"
"Compare 4-1BB vs CD28 costimulatory domains for DLBCL"
"What manufacturing parameters predict clinical response?"
"BCMA CAR-T resistance mechanisms in multiple myeloma"
"How does T-cell exhaustion affect CAR-T persistence?"
```

## Architecture

Built on the HCLS AI Factory platform:

- **Vector DB:** Milvus 2.4 (5 new collections + existing genomic_evidence)
- **Embeddings:** BGE-small-en-v1.5 (384-dim)
- **LLM:** Claude (Anthropic API)
- **UI:** Streamlit (port 8520)
- **Knowledge Graph:** 25 target antigens, 8 toxicity profiles, 10 manufacturing processes
- **Query Expansion:** 111 keywords expanding to 1,086 terms across 6 categories

## Quick Start

### Prerequisites

- Python 3.10+
- Milvus 2.4 running on port 19530
- Anthropic API key

### Install

```bash
cd ai_agent_adds/cart_intelligence_agent
pip install -r requirements.txt
```

### Verify Knowledge Graph

```bash
python scripts/seed_knowledge.py
```

### Run UI (Scaffold Mode)

```bash
streamlit run app/cart_ui.py --server.port 8520
```

### Ingest Data (when infrastructure ready)

```bash
# PubMed literature
python scripts/ingest_pubmed.py --max-results 5000

# ClinicalTrials.gov
python scripts/ingest_clinical_trials.py --max-results 1500

# Seed knowledge graph
python scripts/seed_knowledge.py --export data/reference/knowledge.json
```

## Project Structure

```
cart_intelligence_agent/
├── Docs/                          # Architecture design document
├── src/
│   ├── models.py                  # Pydantic data models
│   ├── collections.py             # Milvus collection schemas
│   ├── knowledge.py               # CAR-T knowledge graph
│   ├── query_expansion.py         # Term expansion maps
│   ├── rag_engine.py              # Multi-collection RAG engine
│   ├── agent.py                   # CAR-T Intelligence Agent
│   ├── ingest/                    # Data ingest pipelines
│   └── utils/                     # API clients
├── app/cart_ui.py                 # Streamlit chat interface
├── config/settings.py             # Configuration
├── scripts/                       # CLI tools
└── requirements.txt
```

**20 Python files | 5,481 lines | Apache 2.0**

## Knowledge Graph

| Component | Count |
|---|---|
| Target Antigens | 25 (CD19, BCMA, CD22, HER2, GPC3, ...) |
| Approved Products | 6 (Kymriah, Yescarta, Tecartus, Breyanzi, Abecma, Carvykti) |
| Toxicity Profiles | 8 (CRS, ICANS, B-cell aplasia, HLH/MAS, ...) |
| Manufacturing Processes | 10 (transduction, expansion, cryopreservation, ...) |
| Query Expansion Keywords | 111 |
| Expansion Terms | 1,086 |

## Status

**Phase 1 (Scaffold)** — Complete. Architecture design + Python scaffold with data models, collection schemas, knowledge graph, and stubs ready for implementation.

## Credits

- **Adam Jones** — HCLS AI Factory
- **TJ Chen (NVIDIA)** — CAR-T Intelligence Platform concept
- **Apache 2.0 License**
