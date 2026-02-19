# CAR-T Intelligence Agent — Architecture Design Document

**Author:** Adam Jones
**Date:** February 2026
**Version:** 0.1.0 (Scaffold)
**License:** Apache 2.0

---

## 1. Executive Summary

The CAR-T Intelligence Agent extends the HCLS AI Factory platform to support cross-functional intelligence across the CAR-T cell therapy development lifecycle. Inspired by TJ Chen's (NVIDIA) "One Unified CAR-T Intelligence Platform," this agent breaks down data silos between the four stages of CAR-T development:

1. **Target Identification** — Antigen biology, expression profiling, disease association
2. **CAR Design** — scFv selection, costimulatory domains, vector engineering
3. **Vector Engineering** — Transduction, expansion, manufacturing processes
4. **In Vitro / In Vivo Testing** — Cytotoxicity, cytokine assays, animal models, clinical trials

The platform enables cross-functional queries like *"Why do CD19 CAR-T therapies fail in relapsed B-ALL?"* that simultaneously search published literature, clinical trials, CAR construct data, assay results, and manufacturing records.

---

## 2. Architecture Overview

### 2.1 Mapping to TJ Chen's Architecture

| TJ's Component | HCLS AI Factory Implementation |
|---|---|
| Data Sources (bottom layer) | Ingest parsers: PubMed, ClinicalTrials.gov, UniProt, IMGT |
| Ingest Pipelines | `src/ingest/` module — parse, chunk, annotate by CAR-T stage |
| Data Stores (middle layer) | 5 Milvus collections + existing `genomic_evidence` |
| Vector Database (cuVS) | Milvus 2.4 with COSINE/IVF_FLAT indexing |
| Embed + Index | BGE-small-en-v1.5 (384-dim) via existing `embedder.py` |
| Users + Agents (top layer) | Streamlit UI + CAR-T RAG Engine + Claude |

### 2.2 Mapping to VAST AI OS

| VAST AI OS Component | CAR-T Agent Role |
|---|---|
| **DataStore** | Raw files: PDFs, CSVs, PDB structures, SDF files |
| **DataEngine** | Event-driven ingest functions (CT1-CT4 equivalent) |
| **DataBase** | 5 CAR-T Milvus collections + knowledge graph |
| **InsightEngine** | Embedding + multi-collection RAG + LLM synthesis |
| **AgentEngine** | CAR-T Intelligence Agent (plan → search → synthesize → report) |

### 2.3 System Diagram

```
                        ┌─────────────────────────────┐
                        │   Streamlit Chat UI (8520)   │
                        │   Cross-functional queries   │
                        └──────────────┬──────────────┘
                                       │
                        ┌──────────────▼──────────────┐
                        │  CAR-T Intelligence Agent    │
                        │  plan → search → synthesize  │
                        └──────────────┬──────────────┘
                                       │
                 ┌─────────────────────┼─────────────────────┐
                 │                     │                     │
        ┌────────▼────────┐  ┌────────▼────────┐  ┌────────▼────────┐
        │  Query Expansion │  │  Knowledge Graph │  │   Claude LLM    │
        │  111 keywords    │  │  25 targets      │  │   (Anthropic)   │
        │  1,086 terms     │  │  8 toxicities    │  │                 │
        │  6 categories    │  │  10 manufacturing│  │                 │
        └────────┬────────┘  └────────┬────────┘  └─────────────────┘
                 │                     │
        ┌────────▼─────────────────────▼────────┐
        │        Multi-Collection RAG Engine     │
        │   Parallel search across 6 collections │
        │   Weight: lit 0.3, trial 0.25, ...     │
        └───┬────┬────┬────┬────┬────┬──────────┘
            │    │    │    │    │    │
    ┌───────▼┐ ┌▼───┐│┌───▼┐┌──▼─┐┌▼──────┐┌───────┐
    │ cart_  ││ cart ││cart ││cart ││ cart_  ││genomic│
    │ liter- ││trials││cons-││assa││ manuf- ││eviden-│
    │ ature  ││     ││truct││ys  ││ actur- ││ ce    │
    └────────┘└─────┘└─────┘└────┘└────────┘└───────┘
         ▲        ▲       ▲      ▲       ▲
    ┌────┴────┐ ┌─┴──┐ ┌──┴──┐┌─┴──┐ ┌──┴───┐
    │ PubMed  │ │ CT │ │ FDA │ │CSV │ │ Lit  │
    │E-utils  │ │.gov│ │ +   │ │JSON│ │ ext. │
    │         │ │API │ │ Pub │ │    │ │      │
    └─────────┘ └────┘ └─────┘└────┘ └──────┘
```

---

## 3. Data Collections

### 3.1 Collection Schemas (5 new + 1 existing)

#### `cart_literature` — Published research + patents
- **Source:** PubMed E-utilities, PMC
- **Fields:** PMID, title, text_chunk, source_type, year, cart_stage, target_antigen, disease, keywords, journal
- **Embedding:** FLOAT_VECTOR(384), BGE-small-en-v1.5
- **Index:** IVF_FLAT, COSINE, nlist=1024
- **Expected volume:** ~5,000 abstracts

#### `cart_trials` — Clinical trial records
- **Source:** ClinicalTrials.gov API v2
- **Fields:** NCT ID, title, text_summary, phase, status, sponsor, target_antigen, car_generation, costimulatory, disease, enrollment, start_year, outcome_summary
- **Expected volume:** ~1,500 trials

#### `cart_constructs` — CAR design data
- **Source:** FDA-approved products (6) + published designs (~50)
- **Fields:** name, text_summary, target_antigen, scfv_origin, costimulatory_domain, signaling_domain, generation, hinge_tm, vector_type, fda_status, known_toxicities
- **Seed data:** 6 FDA-approved CAR-T products with real clinical data

#### `cart_assays` — In vitro / in vivo testing data
- **Source:** Literature extraction, lab records
- **Fields:** text_summary, assay_type, construct_id, target_antigen, cell_line, effector_ratio, key_metric, metric_value, outcome, notes

#### `cart_manufacturing` — Vector production + CMC data
- **Source:** Published manufacturing data, CMC records
- **Fields:** text_summary, process_step, vector_type, parameter, parameter_value, target_spec, met_spec, batch_id, notes

#### `genomic_evidence` — Existing collection (reused)
- **Source:** ClinVar + AlphaMissense (existing pipeline)
- **Relevance:** CD19 variants, BCMA mutations, pharmacogenomic variants

---

## 4. Knowledge Graph

The CAR-T knowledge graph extends the Clinker pattern from `rag-chat-pipeline/src/knowledge.py`:

| Component | Count | Examples |
|---|---|---|
| Target Antigens | 25 | CD19, BCMA, CD22, HER2, GPC3, Mesothelin |
| Toxicity Profiles | 8 | CRS, ICANS, B-cell aplasia, HLH/MAS |
| Manufacturing Processes | 10 | Lentiviral transduction, expansion, cryopreservation |

Each target antigen entry includes: protein name, UniProt ID, expression pattern, associated diseases, approved products, key clinical trials, known resistance mechanisms, toxicity profile, and normal tissue expression.

---

## 5. Query Expansion

Six expansion map categories with 111 keywords expanding to 1,086 unique terms:

| Category | Keywords | Terms | Coverage |
|---|---|---|---|
| Target Antigen | 26 | 196 | CD19, BCMA, CD22, HER2, PSMA, ... |
| Disease | 16 | 143 | B-ALL, DLBCL, Multiple Myeloma, ... |
| Toxicity | 14 | 136 | CRS, ICANS, B-cell aplasia, HLH, ... |
| Manufacturing | 16 | 181 | Transduction, expansion, release testing, ... |
| Mechanism | 19 | 224 | Resistance, exhaustion, persistence, ... |
| Construct | 20 | 206 | scFv, bispecific, safety switch, ... |

---

## 6. Multi-Collection RAG Engine

### 6.1 Search Flow

```
User Query: "Why do CD19 CAR-T therapies fail in relapsed B-ALL?"
    │
    ├── 1. Embed query (BGE-small with "Represent this sentence: " prefix)
    │
    ├── 2. Parallel search across 6 collections (top_k per collection)
    │   ├── genomic_evidence:  CD19 variants, B-ALL mutations
    │   ├── cart_literature:   PubMed papers on CD19 CAR-T failure
    │   ├── cart_trials:       Terminated/failed CD19 B-ALL trials
    │   ├── cart_constructs:   CD19 CAR designs + known resistance
    │   ├── cart_assays:       In vitro failure patterns
    │   └── cart_manufacturing: Production failure modes
    │
    ├── 3. Query expansion: "CD19" → [CD19, B-ALL, DLBCL, tisagenlecleucel, ...]
    │
    ├── 4. Expanded search with expanded terms
    │
    ├── 5. Merge + deduplicate + rank by weighted relevance
    │   (literature: 0.30, trials: 0.25, constructs: 0.20, assays: 0.15, manufacturing: 0.10)
    │
    ├── 6. Knowledge graph augmentation:
    │   CD19 → known_resistance → [CD19 loss, lineage switch, trogocytosis]
    │   CD19 → toxicity_profile → {CRS: 30-90%, ICANS: 20-65%}
    │
    ├── 7. Build prompt with evidence from ALL collections + knowledge context
    │
    └── 8. Stream Claude response (grounded, cross-functional answer with citations)
```

### 6.2 Collection Weights

| Collection | Weight | Rationale |
|---|---|---|
| cart_literature | 0.30 | Published evidence is the primary source |
| cart_trials | 0.25 | Clinical outcomes provide direct answers |
| cart_constructs | 0.20 | Design data explains mechanisms |
| cart_assays | 0.15 | Lab data supports mechanistic claims |
| cart_manufacturing | 0.10 | Manufacturing links to clinical outcomes |

### 6.3 System Prompt

The agent uses a specialized system prompt that instructs Claude to:
1. Cite specific evidence with source type (literature, trial, construct, assay)
2. Consider ALL stages of CAR-T development (not just one silo)
3. Identify cross-functional insights (e.g., manufacturing issues causing clinical failure)
4. Highlight known failure modes and resistance mechanisms
5. Suggest optimization strategies based on historical data

---

## 7. Data Sources

| Source | API | Volume | Collection | Cost |
|---|---|---|---|---|
| PubMed | NCBI E-utilities (free) | ~5,000 abstracts | cart_literature | Free |
| ClinicalTrials.gov | REST API v2 (free) | ~1,500 trials | cart_trials | Free |
| FDA Product Labels | Manual curation | 6 products | cart_constructs | Free |
| Published Designs | Literature extraction | ~50 constructs | cart_constructs | Free |
| Published Assays | Literature extraction | ~200 results | cart_assays | Free |

---

## 8. Infrastructure Reuse

| Existing Component | Reuse Pattern |
|---|---|
| Milvus 2.4 (port 19530) | Add 5 new collections alongside existing genomic_evidence |
| BGE-small-en-v1.5 | Import `EvidenceEmbedder` from rag-chat-pipeline |
| Claude API | Import `AnthropicClient` from rag-chat-pipeline |
| Streamlit | New UI on port 8520 |
| Docker | New container in docker-compose |
| Grafana/Prometheus | Extend existing monitoring |

---

## 9. Demo Scenarios

### Seed Queries
1. **"Why do CD19 CAR-T therapies fail in relapsed B-ALL patients?"**
   - Searches: literature (resistance mechanisms), trials (ELIANA failures), constructs (CD19 targeting), genomic (CD19 variants)
   - Knowledge: CD19 loss, lineage switch, trogocytosis, exon 2 deletion

2. **"Compare 4-1BB vs CD28 costimulatory domains for DLBCL"**
   - Searches: literature (head-to-head reviews), trials (ZUMA-1 vs TRANSCEND), constructs (Yescarta vs Breyanzi)
   - Knowledge: Persistence vs rapid expansion, metabolic profiles

3. **"What manufacturing parameters predict clinical response?"**
   - Searches: literature (correlative studies), manufacturing (process parameters), trials (responder analysis)
   - Knowledge: VCN, T-cell phenotype, expansion kinetics

4. **"BCMA CAR-T resistance mechanisms in multiple myeloma"**
   - Searches: literature (resistance reviews), constructs (BCMA designs), trials (KarMMa/CARTITUDE)
   - Knowledge: BCMA downregulation, biallelic loss, gamma-secretase shedding

5. **"How does T-cell exhaustion affect CAR-T persistence?"**
   - Searches: literature (exhaustion biology), assays (PD-1/LAG-3/TIM-3 data), constructs (4-1BB advantage)
   - Knowledge: Tonic signaling, checkpoint receptors, epigenetic remodeling

---

## 10. File Structure

```
cart_intelligence_agent/
├── Docs/
│   └── CART_Intelligence_Agent_Design.md    # This document
├── src/
│   ├── __init__.py
│   ├── models.py                            # Pydantic data models (285 lines)
│   ├── collections.py                       # Milvus collection schemas (758 lines)
│   ├── knowledge.py                         # CAR-T knowledge graph (930 lines)
│   ├── query_expansion.py                   # Term expansion maps (956 lines)
│   ├── rag_engine.py                        # Multi-collection RAG (372 lines)
│   ├── agent.py                             # Intelligence agent (263 lines)
│   ├── ingest/
│   │   ├── __init__.py
│   │   ├── base.py                          # Base ingest pipeline (149 lines)
│   │   ├── literature_parser.py             # PubMed ingest (311 lines)
│   │   ├── clinical_trials_parser.py        # ClinicalTrials.gov ingest (226 lines)
│   │   ├── construct_parser.py              # CAR construct ingest (287 lines)
│   │   └── assay_parser.py                  # Assay data ingest (140 lines)
│   └── utils/
│       ├── __init__.py
│       └── pubmed_client.py                 # NCBI E-utilities client (250 lines)
├── app/
│   └── cart_ui.py                           # Streamlit chat interface (214 lines)
├── config/
│   └── settings.py                          # Pydantic BaseSettings (70 lines)
├── data/
│   ├── reference/                           # Seed data files
│   └── cache/                               # Embedding cache
├── scripts/
│   ├── ingest_pubmed.py                     # CLI: PubMed ingest
│   ├── ingest_clinical_trials.py            # CLI: ClinicalTrials.gov ingest
│   └── seed_knowledge.py                    # CLI: Knowledge graph verification
├── requirements.txt
└── README.md
```

**Total: 20 Python files, 5,481 lines of code**

---

## 11. Implementation Roadmap

### Phase 1: Scaffold (Current) ✅
- [x] Data models (Pydantic)
- [x] Collection schemas (5 Milvus collections)
- [x] Knowledge graph (25 targets, 8 toxicities, 10 manufacturing)
- [x] Query expansion (111 keywords, 1,086 terms)
- [x] RAG engine stub (multi-collection search)
- [x] Agent stub (plan-search-synthesize)
- [x] Streamlit UI stub
- [x] Ingest pipeline stubs (4 parsers + PubMed client)
- [x] Design document

### Phase 2: Data Ingest
- [ ] Implement PubMed client (search + efetch)
- [ ] Implement ClinicalTrials.gov parser
- [ ] Seed cart_constructs with 6 FDA products + ~50 published designs
- [ ] Create collection creation script
- [ ] Ingest ~5,000 PubMed abstracts
- [ ] Ingest ~1,500 clinical trials

### Phase 3: RAG Integration
- [ ] Connect to existing Milvus instance
- [ ] Implement multi-collection parallel search
- [ ] Implement knowledge augmentation pipeline
- [ ] Connect to Claude API
- [ ] End-to-end query testing

### Phase 4: UI + Demo
- [ ] Connect Streamlit UI to live RAG engine
- [ ] Add collection status indicators
- [ ] Add evidence source visualization
- [ ] Demo with 5 seed queries
- [ ] Add to Nextflow orchestrator

---

## 12. Relationship to HCLS AI Factory

This agent demonstrates the **generalizability** of the HCLS AI Factory architecture. The same infrastructure that supports the VCP/Frontotemporal Dementia pipeline can be extended to support CAR-T cell therapy intelligence with:

- **Same Milvus instance** (just add collections)
- **Same embedding model** (BGE-small-en-v1.5)
- **Same LLM** (Claude via Anthropic API)
- **Same orchestration** (Nextflow DSL2)
- **Same hardware** (NVIDIA DGX Spark)

The key architectural insight: **the platform is not disease-specific**. By changing the knowledge graph, query expansion maps, and collection schemas, the same RAG architecture serves any therapeutic area.
