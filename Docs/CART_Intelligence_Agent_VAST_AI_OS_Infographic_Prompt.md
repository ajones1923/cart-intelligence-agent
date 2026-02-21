# Nano Banana Pro — HCLS AI Factory CAR-T Intelligence Agent with VAST AI OS Infographic

## IMPORTANT: Read this entire prompt before generating. This describes a single technical architecture infographic — NOT a slide deck. Every element below appears on ONE canvas. Keep text SHORT inside every box — prefer 2-3 word labels and single-line descriptions. Avoid paragraphs inside boxes. Use icons and arrows to carry meaning instead of text.

---

## OVERALL LAYOUT AND STYLE

Create a professional technical architecture infographic in landscape orientation (16:9 aspect ratio). Clean, structured, authoritative — matching VAST Data's white paper aesthetic with enterprise architecture precision.

**Canvas:** White background (#FFFFFF). Organized with clear section boundaries and generous spacing between components.

**Typography:**
- Title: Large, bold, sans-serif (Inter or Helvetica), deep navy (#1B2333)
- Subtitle: Smaller, medium gray (#666666)
- "VAST" uses VAST Data logo typography (letters slightly separated, angular)
- Section headers: Bold, navy (#1B2333) with teal (#1AAFCC) left-border accent
- Component labels: Medium bold, 2-5 words maximum per label
- Body text: 10pt minimum, dark gray (#333333), maximum 2 short lines per box
- Metric callouts: Bold inside rounded pill badges

**Color Palette:**
- Teal: #1AAFCC — primary accent for all VAST AI OS components
- Deep Navy: #1B2333 — titles, dark bars
- NVIDIA Green: #76B900 — NVIDIA components, DGX SuperPOD, infrastructure
- Purple: #8B5CF6 — LLM components (Llama 3.1-70B-Instruct for CAR-T Agent, Med42-70B for HCLS Factory)
- Amber: #F5A623 — knowledge graph, query expansion
- Emerald: #059669 — comparative mode, canonical data, outputs
- Light Gray: #F5F5F5 — card backgrounds
- White: #FFFFFF — canvas, text on dark backgrounds

**Visual Elements:**
- Rounded-corner rectangles (8px radius)
- Thin-line monochrome icons matching VAST white paper style (not emoji)
- Color-coded arrows: gray (data flow), teal (VAST component links), emerald (comparative)
- Metric badges: small rounded pills, white text on colored background
- VAST logo mark and NVIDIA logo mark where referenced

---

## CANVAS STRUCTURE (Top to Bottom, 7 bands)

### ━━━ BAND 1: TITLE BAR ━━━

**Left badge:** "CAR-T Intelligence Agent" in teal (#1AAFCC) pill

**Center:**
- **Title:** "CAR-T Intelligence Agent"
- **Subtitle:** "Cross-Functional CAR-T Intelligence on VAST AI OS + NVIDIA DGX SuperPOD"
- **Tagline:** "6,049+ Vectors | 25 Targets | 5 Development Stages"

**Right — Legend** (compact):
```
VAST Components (teal): DataStore | DataEngine
  DataBase | InsightEngine | AgentEngine
NVIDIA (green): DGX SuperPOD | BGE-small
→ Data Flow   - → VAST Link   ⇒ Comparative
```

---

### ━━━ BAND 2: DATA SOURCES (left column) ━━━

Vertical stack of 5 small cards with icon + bold label + one-line detail:

1. **PubMed** [journal] — 4,995 abstracts via NCBI E-utilities
2. **ClinicalTrials.gov** [clipboard] — 973 trials via API v2
3. **FDA Products** [shield] — 6 approved CAR-T therapies
4. **Landmark Papers** [star] — 45 assay records (ELIANA, ZUMA-1, KarMMa)
5. **Manufacturing** [factory] — 30 CMC/process records

Arrows flow right into DataStore labeled: "S3-compatible ingest"

---

### ━━━ BAND 3: VAST AI OS THREE-PLANE ARCHITECTURE (center, largest section) ━━━

Enclosed in a container with teal (#1AAFCC) border.

**Header bar (teal #1AAFCC, white text):** "VAST AI OS — CAR-T Intelligence Agent"

#### Sub-row 3A: DATA PLANE

**Left label bar (teal, vertical text):** "① Data Plane"

**Box: DataStore** [teal border, VAST badge]
- "Raw Data Repository"
- "PubMed XML | CTgov JSON | Seed Data"
- Badge: "S3 interface"

**Box: Canonical CAR-T Data** [emerald border, larger, light green tint]
- 5 mini-cards in a row, each showing only name + count:
  - Literature: 4,995
  - Trials: 973
  - Constructs: 6
  - Assays: 45
  - Manufacturing: 30
- Badge: "6,049 total"

#### Sub-row 3B: EXECUTION PLANE

**Left label bar (teal #1AAFCC, vertical text):** "② Execution Plane"

**Box: DataEngine** [teal border, VAST badge]
- "Event-driven ingest"
- "5 pipelines: fetch → embed → store"
- Badge: "168h refresh"

**Box: Query Expansion** [amber border]
- "111 keywords → 1,086 terms"
- Badge: "6 maps"

**Box: Knowledge Graph** [amber border]
- "25 targets | 8 toxicities | 10 mfg"
- Badge: "111+ entities"

**Box: Parallel Search** [teal border, wide]
- "5 collections searched simultaneously"
- 5 colored squares (one per collection)
- Badge: "< 20 ms"

**Box: Comparative Mode** [emerald border, below standard path]
- "Dual retrieval: Entity A vs Entity B"
- "Auto-detected from X vs Y queries"
- Badge: "~365 ms"

#### Sub-row 3C: REASONING PLANE

**Left label bar (navy #1B2333, vertical text):** "③ Reasoning Plane"

**Box: DataBase** [teal border, VAST badge]
- "Unified SQL + Vector"
- "384-dim vectors | COSINE similarity"
- Badges: "6,049 vectors" | "< 20 ms"

**Box: InsightEngine** [teal border, VAST badge]
- "Multi-collection RAG"
- "Query expansion + knowledge graph"
- Badge: "1,086 terms"

**Box: AgentEngine** [teal border, VAST badge, largest in this row]
- "Plan → Execute → Reflect → Report"
- Small circular diagram showing the 4-phase cycle
- Badge: "CARTIntelligenceAgent"

**Box: Llama 3.1-70B-Instruct** [purple border]
- "Local LLM via NIM"
- "Streaming RAG + citations"
- Badges: "70B params" | "128K context"

---

### ━━━ BAND 4: OUTPUT MODES (right-center column) ━━━

4 stacked cards [emerald border], each with bold label + one line:

1. **RAG Response** — Grounded narrative with PMID + NCT citations
2. **Comparative Analysis** — Side-by-side tables + pros/cons
3. **Knowledge Graph** — Interactive network visualization
4. **Image Verification** — Claude Vision claim checking

---

### ━━━ BAND 5: INTERFACES (right edge) ━━━

3 destination cards [purple border]:

1. **Web UI** — Interactive chat interface
2. **REST API** — Programmatic access, 7 endpoints
3. **Export** — Markdown, JSON, PDF

---

### ━━━ BAND 6: HCLS AI FACTORY INTEGRATION (bottom strip) ━━━

Light indigo background (#E0E7FF).

**Header:** "HCLS AI Factory — Cross-Agent Integration on VAST AI OS"

5 horizontal boxes, each with bold label + one line:

1. **Genomics** — Shared DataBase, Parabricks 4.6
2. **RAG/Chat** — Med42-70B, shared InsightEngine
3. **Drug Discovery** — DataEngine: BioNeMo NIMs
4. **Imaging Agent** — Cross-modal via DataSpace (future)
5. **Biomarker Agent** — Shared InsightEngine (future)

Dashed teal arrows connect upward.

---

### ━━━ BAND 7: VAST + NVIDIA INFRASTRUCTURE BAR (bottom) ━━━

Full-width bar split in two halves. White text throughout.

**Left half (teal #1AAFCC background) — VAST AI OS:**

5 columns, bold header + 2 short lines each:

| DataStore | DataEngine | DataBase | InsightEngine | AgentEngine |
|---|---|---|---|---|
| S3 file interface | 5 ingest pipelines | Unified SQL + Vector | RAG orchestration | Agent runtime |
| Immutable archive | Event-driven | 6,049+ vectors | Knowledge graph | Plan/Execute/Reflect |

**Right half (green #76B900 background) — NVIDIA:**

4 columns, bold header + 2 short lines each:

| DGX SuperPOD | BGE-small | Llama 3.1-70B-Instruct | Containerized |
|---|---|---|---|
| Multi-node GPU cluster | 384-dim vectors | Local LLM via NIM | Scalable services |
| H100/B200 GPUs | 33M params | 128K context | Enterprise deployment |

---

## ANNOTATIONS (keep to 3 maximum)

**Badge 1** (top-left of VAST section):
- "Every component maps to a VAST AI OS primitive"

**Badge 2** (center-right floating):
- "6 FDA Products: Kymriah, Yescarta, Tecartus, Breyanzi, Abecma, Carvykti"

**Badge 3** (bottom-right):
- "Enterprise Scale on NVIDIA DGX SuperPOD"

---

## TEXT DENSITY RULES

- **Maximum 2 lines of body text per box** — if you need more, use a badge instead
- **Labels are 2-5 words** — never full sentences inside component boxes
- **Tables have maximum 2 data rows** (header + 1-2 rows)
- **Metric badges replace inline numbers** — put numbers in pills, not in body text
- **No code blocks, no file paths, no directory trees** on the canvas
- **Prefer icons + arrows over text** to show relationships
- **VAST component names (DataStore, DataEngine, etc.) are always bold labels, never explained in paragraphs**

---

## WHAT THIS DIAGRAM MUST COMMUNICATE AT A GLANCE

1. VAST AI OS three-plane architecture organizes the entire agent
2. DataStore → DataEngine → DataBase → InsightEngine → AgentEngine flow is clear
3. 5 data sources feed 5 DataBase collections (6,049+ vectors)
4. Knowledge graph + query expansion enrich every search
5. Comparative mode auto-detects "X vs Y" queries
6. Llama 3.1-70B-Instruct generates grounded answers locally with citations
7. Part of the broader HCLS AI Factory with cross-agent integration on VAST AI OS
8. NVIDIA DGX SuperPOD infrastructure anchors everything

The overall impression: every component of the CAR-T Intelligence Agent maps cleanly to VAST AI OS primitives — demonstrating VAST as the natural platform for domain-specific RAG agents at enterprise scale. Clean enough to read every label at a glance.

---

*End of Prompt*
