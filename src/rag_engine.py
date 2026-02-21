"""Multi-collection RAG engine for CAR-T Intelligence Agent.

Searches across all 11 collections simultaneously using parallel ThreadPoolExecutor,
synthesizes findings with full knowledge graph augmentation (targets, toxicities,
manufacturing, biomarkers, regulatory), and generates grounded LLM responses.

Extends the pattern from: rag-chat-pipeline/src/rag_engine.py

Author: Adam Jones
Date: February 2026
"""

import re
import time
from typing import Dict, Generator, List, Optional

from config.settings import settings

from .models import (
    AgentQuery,
    CrossCollectionResult,
    SearchHit,
)

# ═══════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT
# ═══════════════════════════════════════════════════════════════════════

CART_SYSTEM_PROMPT = """You are a CAR-T cell therapy intelligence agent with deep expertise in:

1. **Target Identification** — antigen biology, expression profiling, tumor specificity
2. **CAR Design** — scFv selection, costimulatory domains (CD28 vs 4-1BB), signaling architecture
3. **Vector Engineering** — lentiviral/retroviral production, transduction efficiency, VCN optimization
4. **In Vitro & In Vivo Testing** — cytotoxicity assays, cytokine profiling, mouse models, persistence
5. **Clinical Development** — trial design, response rates, toxicity management (CRS, ICANS)
6. **Manufacturing** — leukapheresis, T-cell expansion, cryopreservation, release testing, CMC
7. **Safety & Pharmacovigilance** — post-market safety signals, REMS, long-term follow-up, FAERS
8. **Biomarkers** — CRS prediction (ferritin, CRP, IL-6), response biomarkers, MRD monitoring, exhaustion markers
9. **Regulatory Intelligence** — FDA approval pathways, BLA timelines, breakthrough therapy, RMAT, EMA
10. **Molecular Design** — scFv binding affinity, CDR sequences, humanization, nanobodies, structural data
11. **Real-World Evidence** — registry outcomes (CIBMTR), community vs academic, special populations, disparities
12. **Genomic Evidence** — patient variant data, clinical significance (ClinVar), AlphaMissense pathogenicity, gene-level variant analysis

You have access to evidence from MULTIPLE data sources spanning the entire CAR-T development lifecycle,
from patient genomics and molecular design through post-market pharmacovigilance.

When answering questions:
- **Cite evidence using clickable markdown links** provided in the evidence. Use the exact
  link format from the evidence, e.g. [Literature:PMID 12345678](https://pubmed.ncbi.nlm.nih.gov/12345678/)
  or [Trial:NCT12345678](https://clinicaltrials.gov/study/NCT12345678). For Assay, Construct,
  Manufacturing, Safety, Biomarker, Regulatory, Sequence, and RealWorld sources, use the
  format [Collection:record-id] (no URL needed).
- **Think cross-functionally** — connect insights across development stages
  (e.g., how manufacturing choices affect clinical outcomes, how biomarkers predict safety)
- **Highlight failure modes** and resistance mechanisms when relevant
- **Suggest optimization strategies** based on historical data and published results
- **Be specific** — cite trial names (ELIANA, ZUMA-1), products (Kymriah, Yescarta),
  and quantitative data when available
- **Include regulatory context** when discussing products or approvals
- **Reference real-world evidence** to complement clinical trial data
- **Acknowledge uncertainty** — distinguish established facts from emerging data

Your goal is to break down data silos and provide unified intelligence that accelerates
CAR-T development from target to clinical candidate."""

# ═══════════════════════════════════════════════════════════════════════
# COLLECTION CONFIGURATION (reads weights from settings)
# ═══════════════════════════════════════════════════════════════════════

COLLECTION_CONFIG = {
    "cart_literature":    {"weight": settings.WEIGHT_LITERATURE,    "label": "Literature",     "has_target_antigen": True,  "year_field": "year"},
    "cart_trials":        {"weight": settings.WEIGHT_TRIALS,        "label": "Trial",          "has_target_antigen": True,  "year_field": "start_year"},
    "cart_constructs":    {"weight": settings.WEIGHT_CONSTRUCTS,    "label": "Construct",      "has_target_antigen": True,  "year_field": None},
    "cart_assays":        {"weight": settings.WEIGHT_ASSAYS,        "label": "Assay",          "has_target_antigen": True,  "year_field": None},
    "cart_manufacturing": {"weight": settings.WEIGHT_MANUFACTURING, "label": "Manufacturing",  "has_target_antigen": False, "year_field": None},
    "cart_safety":        {"weight": settings.WEIGHT_SAFETY,        "label": "Safety",         "has_target_antigen": False, "year_field": "year"},
    "cart_biomarkers":    {"weight": settings.WEIGHT_BIOMARKERS,    "label": "Biomarker",      "has_target_antigen": True,  "year_field": None},
    "cart_regulatory":    {"weight": settings.WEIGHT_REGULATORY,    "label": "Regulatory",     "has_target_antigen": False, "year_field": None},
    "cart_sequences":     {"weight": settings.WEIGHT_SEQUENCES,     "label": "Sequence",       "has_target_antigen": True,  "year_field": None},
    "cart_realworld":     {"weight": settings.WEIGHT_REALWORLD,     "label": "RealWorld",      "has_target_antigen": False, "year_field": None},
    "genomic_evidence":   {"weight": settings.WEIGHT_GENOMIC,       "label": "Genomic",        "has_target_antigen": False, "year_field": None},
}

# Known target antigens for expansion term classification
_KNOWN_ANTIGENS = {
    "CD19", "BCMA", "CD22", "CD20", "CD30", "CD33", "CD38", "CD123",
    "GD2", "HER2", "GPC3", "EGFR", "EGFRVIII", "MESOTHELIN",
    "CLAUDIN18.2", "MUC1", "PSMA", "ROR1", "CD7", "GPRC5D",
    "FLT3", "CD70", "DLL3", "NY-ESO-1", "B7-H3", "NKG2D",
    "IL13RA2", "CD5",
}


class CARTRAGEngine:
    """Multi-collection RAG engine for CAR-T cross-functional queries.

    Searches across all CAR-T collections simultaneously using parallel
    ThreadPoolExecutor, merges results with knowledge graph context, and
    generates grounded LLM responses.

    Improvements in v2.0:
    - Parallel search via ThreadPoolExecutor (was sequential)
    - Settings-driven weights and parameters (was hardcoded)
    - Full knowledge graph augmentation (targets, toxicities, manufacturing,
      biomarkers, regulatory — was targets + toxicities only)
    - Fixed query expansion (semantic search, not field-filter)
    - Citation relevance scoring (high/medium/low)
    - Temporal date-range filtering
    - Collection selection filtering
    - Cross-collection entity linking
    - Conversation memory context injection
    """

    def __init__(self, collection_manager, embedder, llm_client,
                 knowledge=None, query_expander=None):
        self.collections = collection_manager
        self.embedder = embedder
        self.llm = llm_client
        self.knowledge = knowledge
        self.expander = query_expander

    def retrieve(self, query: AgentQuery,
                 top_k_per_collection: int = None,
                 collections_filter: List[str] = None,
                 year_min: int = None,
                 year_max: int = None,
                 conversation_context: str = None) -> CrossCollectionResult:
        """Retrieve evidence from collections for a query.

        Args:
            query: The agent query with question and optional filters
            top_k_per_collection: Max results per collection (default from settings)
            collections_filter: Optional list of collection names to search
            year_min: Optional minimum year filter
            year_max: Optional maximum year filter
            conversation_context: Optional prior conversation context for follow-ups
        """
        top_k = top_k_per_collection or settings.TOP_K_PER_COLLECTION
        start = time.time()

        # Optionally prepend conversation context for follow-up queries
        search_text = query.question
        if conversation_context:
            search_text = f"{conversation_context}\n\nCurrent question: {query.question}"

        # Step 1: Embed query
        query_embedding = self._embed_query(search_text)

        # Step 2: Determine collections to search
        collections_to_search = collections_filter or list(COLLECTION_CONFIG.keys())

        # Step 3: Build per-collection filters
        filter_exprs = {}
        for coll in collections_to_search:
            parts = []
            cfg = COLLECTION_CONFIG.get(coll, {})
            if query.target_antigen and cfg.get("has_target_antigen"):
                parts.append(f'target_antigen == "{query.target_antigen}"')
            year_field = cfg.get("year_field")
            if year_field:
                if year_min:
                    parts.append(f'{year_field} >= {year_min}')
                if year_max:
                    parts.append(f'{year_field} <= {year_max}')
            if parts:
                filter_exprs[coll] = " and ".join(parts)

        # Step 4: Parallel search across all collections
        all_hits = self._search_all_collections(
            query_embedding, collections_to_search, top_k, filter_exprs,
        )

        # Step 5: Query expansion (semantic search, not field-filter)
        if self.expander:
            expanded_hits = self._expanded_search(
                query.question, query_embedding, collections_to_search, top_k,
            )
            all_hits.extend(expanded_hits)

        # Step 6: Deduplicate, score citations, rank
        hits = self._merge_and_rank(all_hits)

        # Step 7: Full knowledge graph augmentation
        knowledge_context = ""
        if self.knowledge:
            knowledge_context = self._get_knowledge_context(query.question)

        elapsed = (time.time() - start) * 1000

        return CrossCollectionResult(
            query=query.question,
            hits=hits,
            knowledge_context=knowledge_context,
            total_collections_searched=len(collections_to_search),
            search_time_ms=elapsed,
        )

    def query(self, question: str, **kwargs) -> str:
        """Full RAG query: retrieve evidence + generate LLM response."""
        agent_query = AgentQuery(question=question, **kwargs)
        evidence = self.retrieve(agent_query)
        prompt = self._build_prompt(agent_query.question, evidence)
        return self.llm.generate(
            prompt=prompt,
            system_prompt=CART_SYSTEM_PROMPT,
            max_tokens=2048,
            temperature=0.7,
        )

    def query_stream(self, question: str,
                     **kwargs) -> Generator[Dict, None, None]:
        """Streaming RAG query — yields evidence then token chunks."""
        agent_query = AgentQuery(question=question, **kwargs)
        evidence = self.retrieve(agent_query)
        yield {"type": "evidence", "content": evidence}

        prompt = self._build_prompt(agent_query.question, evidence)
        full_answer = ""
        for token in self.llm.generate_stream(
            prompt=prompt,
            system_prompt=CART_SYSTEM_PROMPT,
            max_tokens=2048,
            temperature=0.7,
        ):
            full_answer += token
            yield {"type": "token", "content": token}
        yield {"type": "done", "content": full_answer}

    # ── Cross-Collection Entity Linking ─────────────────────────────

    def find_related(self, entity: str, top_k: int = 5) -> Dict[str, List[SearchHit]]:
        """Find all evidence related to an entity across all 10 collections.

        Enables "show me everything about Yescarta" spanning all collections.

        Args:
            entity: Product name, target antigen, trial ID, etc.
            top_k: Max results per collection

        Returns:
            Dict of collection_name -> List[SearchHit]
        """
        embedding = self._embed_query(entity)
        results = {}

        all_results = self.collections.search_all(
            embedding, top_k_per_collection=top_k,
            score_threshold=settings.SCORE_THRESHOLD,
        )
        for coll_name, hits in all_results.items():
            label = COLLECTION_CONFIG.get(coll_name, {}).get("label", coll_name)
            search_hits = []
            for r in hits:
                hit = SearchHit(
                    collection=label,
                    id=r.get("id", ""),
                    score=r.get("score", 0.0),
                    text=r.get("text_summary", r.get("text_chunk", "")),
                    metadata=r,
                )
                search_hits.append(hit)
            if search_hits:
                results[coll_name] = search_hits
        return results

    # ── Private Methods ──────────────────────────────────────────────

    def _embed_query(self, text: str):
        """Embed query text with BGE instruction prefix."""
        prefix = "Represent this sentence for searching relevant passages: "
        return self.embedder.embed_text(prefix + text)

    def _search_all_collections(
        self, query_embedding, collections: List[str],
        top_k: int, filter_exprs: Dict[str, str],
    ) -> List[SearchHit]:
        """Search all collections in parallel via ThreadPoolExecutor."""
        all_hits = []

        # Use the parallel search_all method from CARTCollectionManager
        parallel_results = self.collections.search_all(
            query_embedding,
            top_k_per_collection=top_k,
            filter_exprs=filter_exprs,
            score_threshold=settings.SCORE_THRESHOLD,
        )

        for coll_name, results in parallel_results.items():
            if coll_name not in [c for c in collections]:
                continue
            cfg = COLLECTION_CONFIG.get(coll_name, {})
            weight = cfg.get("weight", 0.1)
            label = cfg.get("label", coll_name)

            for r in results:
                raw_score = r.get("score", 0.0)
                weighted_score = raw_score * (1 + weight)

                # Citation relevance scoring
                if raw_score >= settings.CITATION_HIGH_THRESHOLD:
                    relevance = "high"
                elif raw_score >= settings.CITATION_MEDIUM_THRESHOLD:
                    relevance = "medium"
                else:
                    relevance = "low"

                metadata = dict(r)
                metadata["relevance"] = relevance

                hit = SearchHit(
                    collection=label,
                    id=r.get("id", ""),
                    score=weighted_score,
                    text=r.get("text_summary", r.get("text_chunk", "")),
                    metadata=metadata,
                )
                all_hits.append(hit)

        return all_hits

    def _expanded_search(
        self, query: str, query_embedding,
        collections: List[str], top_k: int,
    ) -> List[SearchHit]:
        """Use query expansion for additional coverage.

        FIX: Expansion terms that are target antigens use field filters.
        Non-antigen terms are re-embedded for semantic search across all collections.
        """
        if not self.expander:
            return []

        from .query_expansion import expand_query
        expanded_terms = expand_query(query)

        additional_hits = []
        for term in expanded_terms[:5]:
            term_upper = term.upper().replace("-", "").replace(" ", "")

            # Check if this term is a known target antigen
            is_antigen = any(
                term_upper == a.upper().replace("-", "").replace(" ", "")
                for a in _KNOWN_ANTIGENS
            )

            if is_antigen:
                # Use as field filter on target_antigen-capable collections
                for coll_name in collections:
                    if not COLLECTION_CONFIG.get(coll_name, {}).get("has_target_antigen"):
                        continue
                    try:
                        filter_expr = f'target_antigen == "{term}"'
                        results = self.collections.search(
                            coll_name, query_embedding, min(3, top_k), filter_expr,
                        )
                        label = COLLECTION_CONFIG.get(coll_name, {}).get("label", coll_name)
                        for r in results:
                            additional_hits.append(SearchHit(
                                collection=label,
                                id=r.get("id", ""),
                                score=r.get("score", 0.0) * 0.8,
                                text=r.get("text_summary", r.get("text_chunk", "")),
                                metadata=r,
                            ))
                    except Exception:
                        pass
            else:
                # Semantic search: re-embed the expansion term and search all collections
                try:
                    term_embedding = self._embed_query(term)
                    term_results = self.collections.search_all(
                        term_embedding, top_k_per_collection=2,
                        score_threshold=settings.SCORE_THRESHOLD,
                    )
                    for coll_name, results in term_results.items():
                        if coll_name not in collections:
                            continue
                        label = COLLECTION_CONFIG.get(coll_name, {}).get("label", coll_name)
                        for r in results:
                            additional_hits.append(SearchHit(
                                collection=label,
                                id=r.get("id", ""),
                                score=r.get("score", 0.0) * 0.7,
                                text=r.get("text_summary", r.get("text_chunk", "")),
                                metadata=r,
                            ))
                except Exception:
                    pass

        return additional_hits

    def _merge_and_rank(self, hits: List[SearchHit]) -> List[SearchHit]:
        """Deduplicate by ID, sort by score descending, cap at 30."""
        seen = set()
        unique = []
        for hit in hits:
            if hit.id not in seen:
                seen.add(hit.id)
                unique.append(hit)
        unique.sort(key=lambda h: h.score, reverse=True)
        return unique[:30]

    def _get_knowledge_context(self, query: str) -> str:
        """Extract knowledge graph context from ALL domains."""
        if not self.knowledge:
            return ""

        from .knowledge import (
            get_target_context,
            get_toxicity_context,
            get_manufacturing_context,
            get_biomarker_context,
            get_regulatory_context,
        )

        context_parts = []
        query_upper = query.upper()

        # Check for target antigen mentions
        for antigen in ["CD19", "BCMA", "CD22", "CD20", "CD30", "CD33",
                        "CD38", "CD123", "GD2", "HER2", "GPC3", "EGFR",
                        "MESOTHELIN", "CLAUDIN18.2", "PSMA", "ROR1",
                        "CD7", "GPRC5D", "FLT3", "DLL3"]:
            if antigen in query_upper:
                ctx = get_target_context(antigen)
                if ctx:
                    context_parts.append(ctx)

        # Check for toxicity mentions
        for tox in ["CRS", "ICANS", "NEUROTOXICITY", "HLH",
                     "CYTOPENIA", "GVHD"]:
            if tox in query_upper:
                ctx = get_toxicity_context(tox)
                if ctx:
                    context_parts.append(ctx)

        # Check for manufacturing mentions
        mfg_keywords = {
            "MANUFACTURING": "lentiviral_transduction",
            "TRANSDUCTION": "lentiviral_transduction",
            "LENTIVIRAL": "lentiviral_transduction",
            "EXPANSION": "ex_vivo_expansion",
            "LEUKAPHERESIS": "leukapheresis",
            "CRYOPRESERVATION": "cryopreservation",
            "RELEASE TESTING": "release_testing",
            "VEIN-TO-VEIN": "vein_to_vein_time",
            "LYMPHODEPLETION": "lymphodepletion",
        }
        for keyword, process in mfg_keywords.items():
            if keyword in query_upper:
                ctx = get_manufacturing_context(process)
                if ctx:
                    context_parts.append(ctx)
                break  # One manufacturing context is enough

        # Check for biomarker mentions
        biomarker_keywords = {
            "FERRITIN": "ferritin", "CRP": "crp", "IL-6": "il6",
            "IL6": "il6", "PD-1": "pd1", "PD1": "pd1",
            "LAG-3": "lag3", "LAG3": "lag3", "TIM-3": "tim3",
            "TIM3": "tim3", "MRD": "mrd_flow", "BIOMARKER": "ferritin",
            "EXHAUSTION": "pd1", "CTDNA": "ctdna",
        }
        for keyword, biomarker in biomarker_keywords.items():
            if keyword in query_upper:
                ctx = get_biomarker_context(biomarker)
                if ctx:
                    context_parts.append(ctx)
                break  # One biomarker context is enough

        # Check for regulatory / product mentions
        product_keywords = {
            "KYMRIAH": "Kymriah", "TISAGENLECLEUCEL": "Kymriah",
            "YESCARTA": "Yescarta", "AXICABTAGENE": "Yescarta",
            "TECARTUS": "Tecartus", "BREXUCABTAGENE": "Tecartus",
            "BREYANZI": "Breyanzi", "LISOCABTAGENE": "Breyanzi",
            "ABECMA": "Abecma", "IDECABTAGENE": "Abecma",
            "CARVYKTI": "Carvykti", "CILTACABTAGENE": "Carvykti",
        }
        for keyword, product in product_keywords.items():
            if keyword in query_upper:
                ctx = get_regulatory_context(product)
                if ctx:
                    context_parts.append(ctx)

        # Also check for general regulatory terms
        if any(term in query_upper for term in ["FDA", "BLA", "RMAT", "REGULATORY", "APPROVAL"]):
            if not any("Regulatory" in p for p in context_parts):
                # Add first product's regulatory context as general reference
                ctx = get_regulatory_context("Kymriah")
                if ctx:
                    context_parts.append(ctx)

        return "\n\n".join(context_parts)

    @staticmethod
    def _format_citation(collection: str, record_id: str) -> str:
        """Format a citation with clickable URL where possible."""
        if collection == "Literature" and record_id.isdigit():
            return (
                f"[Literature:PMID {record_id}]"
                f"(https://pubmed.ncbi.nlm.nih.gov/{record_id}/)"
            )
        if collection == "Trial" and record_id.upper().startswith("NCT"):
            return (
                f"[Trial:{record_id}]"
                f"(https://clinicaltrials.gov/study/{record_id})"
            )
        return f"[{collection}:{record_id}]"

    def _build_prompt(self, question: str,
                      evidence: CrossCollectionResult) -> str:
        """Build LLM prompt with evidence, knowledge context, and relevance tags."""
        sections = []
        by_coll = evidence.hits_by_collection()

        for coll_name, hits in by_coll.items():
            section_lines = [f"### Evidence from {coll_name}"]
            for i, hit in enumerate(hits[:5], 1):
                citation = self._format_citation(hit.collection, hit.id)
                relevance = hit.metadata.get("relevance", "")
                relevance_tag = f" [{relevance} relevance]" if relevance else ""
                section_lines.append(
                    f"{i}. {citation}{relevance_tag} "
                    f"(score={hit.score:.3f}) {hit.text[:500]}"
                )
            sections.append("\n".join(section_lines))

        evidence_text = "\n\n".join(sections) if sections else "No evidence found."

        knowledge_text = ""
        if evidence.knowledge_context:
            knowledge_text = (
                f"\n\n### Knowledge Graph Context\n"
                f"{evidence.knowledge_context}"
            )

        return (
            f"## Retrieved Evidence\n\n"
            f"{evidence_text}"
            f"{knowledge_text}\n\n"
            f"---\n\n"
            f"## Question\n\n"
            f"{question}\n\n"
            f"Please provide a comprehensive answer grounded in the evidence above. "
            f"Cite sources using the clickable markdown links provided in each evidence item. "
            f"Prioritize [high relevance] citations. "
            f"Consider cross-functional insights across all stages of CAR-T development."
        )

    # ── Comparative Analysis Methods ────────────────────────────────

    def _is_comparative(self, question: str) -> bool:
        q_upper = question.upper()
        return ("COMPARE" in q_upper or " VS " in q_upper
                or "VERSUS" in q_upper or "COMPARING" in q_upper)

    def _parse_comparison_entities(self, question: str):
        q = question.strip()

        match = re.search(
            r'(.+?)\s+(?:vs\.?|versus)\s+(.+)$',
            q, re.IGNORECASE,
        )
        if match:
            raw_a = match.group(1).strip()
            raw_b = match.group(2).strip()
            raw_a = re.sub(r'^(?:compare|comparing)\s+', '', raw_a, flags=re.IGNORECASE)
        else:
            match = re.search(
                r'(?:compare|comparing)\s+(.+?)\s+(?:and|with)\s+(.+?)(?:\s+(?:for|in)\b.*)?$',
                q, re.IGNORECASE,
            )
            if match:
                raw_a = match.group(1).strip()
                raw_b = match.group(2).strip()
            else:
                return None, None

        raw_a = raw_a.rstrip('?.,;:')
        raw_b = raw_b.rstrip('?.,;:')

        trailing = [
            'costimulatory domains', 'costimulatory domain',
            'domains', 'domain', 'signaling',
            'resistance mechanisms', 'resistance', 'mechanisms',
            'for .*', 'in .*', 'differences', 'comparison',
            'toxicity', 'efficacy', 'outcomes', 'safety',
        ]
        for pattern in trailing:
            raw_a = re.sub(rf'\s+{pattern}$', '', raw_a, flags=re.IGNORECASE)
            raw_b = re.sub(rf'\s+{pattern}$', '', raw_b, flags=re.IGNORECASE)

        if not self.knowledge:
            return None, None

        from .knowledge import resolve_comparison_entity
        entity_a = resolve_comparison_entity(raw_a)
        entity_b = resolve_comparison_entity(raw_b)
        return entity_a, entity_b

    def retrieve_comparative(self, question: str,
                             collections_filter: List[str] = None,
                             year_min: int = None,
                             year_max: int = None) -> Optional['CrossCollectionResult']:
        """Run comparative retrieval with two separate searches."""
        from .models import ComparativeResult

        entity_a, entity_b = self._parse_comparison_entities(question)
        if not entity_a or not entity_b:
            return None

        start = time.time()

        query_a = AgentQuery(question=question, target_antigen=entity_a.get("target"))
        query_b = AgentQuery(question=question, target_antigen=entity_b.get("target"))

        evidence_a = self.retrieve(query_a, collections_filter=collections_filter,
                                   year_min=year_min, year_max=year_max)
        evidence_b = self.retrieve(query_b, collections_filter=collections_filter,
                                   year_min=year_min, year_max=year_max)

        comparison_context = ""
        if self.knowledge:
            from .knowledge import get_comparison_context
            comparison_context = get_comparison_context(entity_a, entity_b)

        elapsed = (time.time() - start) * 1000

        return ComparativeResult(
            query=question,
            entity_a=entity_a["canonical"],
            entity_b=entity_b["canonical"],
            evidence_a=evidence_a,
            evidence_b=evidence_b,
            comparison_context=comparison_context,
            total_search_time_ms=elapsed,
        )

    def _build_comparative_prompt(self, question: str, comp) -> str:
        def _fmt(label: str, evidence) -> str:
            sections = []
            by_coll = evidence.hits_by_collection()
            for coll_name, hits in by_coll.items():
                lines = [f"#### {coll_name}"]
                for i, hit in enumerate(hits[:4], 1):
                    citation = self._format_citation(hit.collection, hit.id)
                    lines.append(
                        f"{i}. {citation} "
                        f"(score={hit.score:.3f}) {hit.text[:400]}"
                    )
                sections.append("\n".join(lines))
            if not sections:
                return f"### Evidence for {label}\nNo evidence found."
            return f"### Evidence for {label}\n\n" + "\n\n".join(sections)

        evidence_a_text = _fmt(comp.entity_a, comp.evidence_a)
        evidence_b_text = _fmt(comp.entity_b, comp.evidence_b)

        knowledge_text = ""
        if comp.comparison_context:
            knowledge_text = (
                f"\n\n### Knowledge Graph Comparison Data\n"
                f"{comp.comparison_context}"
            )

        return (
            f"## Comparative Analysis Evidence\n\n"
            f"{evidence_a_text}\n\n"
            f"---\n\n"
            f"{evidence_b_text}"
            f"{knowledge_text}\n\n"
            f"---\n\n"
            f"## Question\n\n{question}\n\n"
            f"## Instructions\n\n"
            f"Provide a structured comparison of **{comp.entity_a}** vs "
            f"**{comp.entity_b}**. Your response MUST include:\n\n"
            f"1. A **comparison table** in markdown format with key dimensions "
            f"as rows and the two entities as columns.\n"
            f"2. **Advantages** of each entity (bulleted list).\n"
            f"3. **Limitations** of each entity (bulleted list).\n"
            f"4. A **clinical context** paragraph explaining when each might "
            f"be preferred.\n\n"
            f"Cite sources using the clickable markdown links provided in "
            f"the evidence above."
        )
