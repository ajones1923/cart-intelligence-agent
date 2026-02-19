"""Multi-collection RAG engine for CAR-T Intelligence Agent.

The core innovation: search across ALL 6 collections simultaneously
(existing genomic_evidence + 5 new CAR-T collections), then synthesize
findings with knowledge graph augmentation.

Extends the pattern from: rag-chat-pipeline/src/rag_engine.py

Author: Adam Jones
Date: February 2026
"""

import sys
import time
from pathlib import Path
from typing import Dict, Generator, List, Optional

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

You have access to evidence from MULTIPLE data sources spanning the entire CAR-T development lifecycle.

When answering questions:
- **Cite evidence** with source type: [Literature], [Trial], [Construct], [Assay], [Manufacturing], [Genomic]
- **Think cross-functionally** — connect insights across development stages
  (e.g., how manufacturing choices affect clinical outcomes)
- **Highlight failure modes** and resistance mechanisms when relevant
- **Suggest optimization strategies** based on historical data and published results
- **Be specific** — cite trial names (ELIANA, ZUMA-1), products (Kymriah, Yescarta),
  and quantitative data when available
- **Acknowledge uncertainty** — distinguish established facts from emerging data

Your goal is to break down data silos and provide unified intelligence that accelerates
CAR-T development from target to clinical candidate."""

# ═══════════════════════════════════════════════════════════════════════
# COLLECTION CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

COLLECTION_CONFIG = {
    "cart_literature": {"weight": 0.30, "label": "Literature"},
    "cart_trials": {"weight": 0.25, "label": "Trial"},
    "cart_constructs": {"weight": 0.20, "label": "Construct"},
    "cart_assays": {"weight": 0.15, "label": "Assay"},
    "cart_manufacturing": {"weight": 0.10, "label": "Manufacturing"},
    "genomic_evidence": {"weight": 0.00, "label": "Genomic"},  # 0 = optional
}


class CARTRAGEngine:
    """Multi-collection RAG engine for CAR-T cross-functional queries.

    Searches across all CAR-T collections simultaneously, merges results
    with knowledge graph context, and generates grounded LLM responses.

    Usage:
        engine = CARTRAGEngine(collection_manager, embedder, llm_client)
        response = engine.query("Why do CD19 CAR-T therapies fail?")
    """

    def __init__(self, collection_manager, embedder, llm_client,
                 knowledge=None, query_expander=None):
        """Initialize the RAG engine.

        Args:
            collection_manager: CARTCollectionManager instance (from collections.py)
            embedder: EvidenceEmbedder instance (from rag-chat-pipeline/src/embedder.py)
            llm_client: LLM client instance (from rag-chat-pipeline/src/llm_client.py)
            knowledge: Knowledge graph module (from knowledge.py)
            query_expander: Query expansion module (from query_expansion.py)
        """
        self.collections = collection_manager
        self.embedder = embedder
        self.llm = llm_client
        self.knowledge = knowledge
        self.expander = query_expander

    def retrieve(self, query: AgentQuery,
                 top_k_per_collection: int = 5) -> CrossCollectionResult:
        """Retrieve evidence from all collections for a query.

        1. Embed the query with BGE instruction prefix
        2. Search each collection in parallel
        3. Apply query expansion for additional coverage
        4. Merge, deduplicate, and rank results
        5. Augment with knowledge graph context

        Args:
            query: The agent query with question and optional filters
            top_k_per_collection: Max results per collection

        Returns:
            CrossCollectionResult with merged hits and knowledge context
        """
        start = time.time()

        # Step 1: Embed query with BGE instruction prefix
        query_embedding = self._embed_query(query.question)

        # Step 2: Determine which collections to search
        collections_to_search = list(COLLECTION_CONFIG.keys())
        if not query.include_genomic:
            collections_to_search.remove("genomic_evidence")

        # Step 3: Build collection-specific filters
        filter_exprs = {}
        if query.target_antigen:
            for coll in collections_to_search:
                if coll != "genomic_evidence":
                    filter_exprs[coll] = (
                        f'target_antigen == "{query.target_antigen}"'
                    )

        # Step 4: Search all collections
        all_hits = self._search_all_collections(
            query_embedding, collections_to_search,
            top_k_per_collection, filter_exprs,
        )

        # Step 5: Query expansion for additional coverage
        if self.expander:
            expanded_hits = self._expanded_search(
                query.question, query_embedding, collections_to_search,
            )
            all_hits.extend(expanded_hits)

        # Step 6: Deduplicate and rank
        hits = self._merge_and_rank(all_hits)

        # Step 7: Knowledge graph augmentation
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
        """Full RAG query: retrieve evidence + generate LLM response.

        Args:
            question: Natural language question
            **kwargs: Additional AgentQuery fields

        Returns:
            LLM-generated answer grounded in evidence
        """
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
        """Streaming RAG query — yields evidence then token chunks.

        Yields:
            {"type": "evidence", "content": CrossCollectionResult}
            {"type": "token", "content": str}
            {"type": "done", "content": str}  # full answer
        """
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

    # ── Private Methods ──────────────────────────────────────────────

    def _embed_query(self, text: str):
        """Embed query text with BGE instruction prefix.

        CRITICAL: BGE-small-en-v1.5 uses asymmetric encoding.
        Queries MUST be prefixed for optimal retrieval.
        """
        prefix = "Represent this sentence for searching relevant passages: "
        return self.embedder.embed_text(prefix + text)

    def _search_all_collections(
        self, query_embedding, collections: List[str],
        top_k: int, filter_exprs: Dict[str, str],
    ) -> List[SearchHit]:
        """Search all specified collections and collect hits."""
        all_hits = []

        for coll_name in collections:
            filter_expr = filter_exprs.get(coll_name)
            try:
                results = self.collections.search(
                    coll_name, query_embedding, top_k, filter_expr,
                )
                weight = COLLECTION_CONFIG.get(
                    coll_name, {}
                ).get("weight", 0.1)
                label = COLLECTION_CONFIG.get(
                    coll_name, {}
                ).get("label", coll_name)

                for r in results:
                    hit = SearchHit(
                        collection=label,
                        id=r.get("id", ""),
                        score=r.get("score", 0.0) * (1 + weight),
                        text=r.get("text_summary", r.get("text_chunk", "")),
                        metadata=r,
                    )
                    all_hits.append(hit)
            except Exception:
                pass  # Collection may not exist yet

        return all_hits

    def _expanded_search(
        self, query: str, query_embedding,
        collections: List[str],
    ) -> List[SearchHit]:
        """Use query expansion to find additional relevant results."""
        if not self.expander:
            return []

        from .query_expansion import expand_query
        expanded_terms = expand_query(query)

        additional_hits = []
        for term in expanded_terms[:5]:  # Limit expansion breadth
            for coll_name in collections:
                if coll_name == "genomic_evidence":
                    continue
                try:
                    filter_expr = f'target_antigen == "{term}"'
                    results = self.collections.search(
                        coll_name, query_embedding, 3, filter_expr,
                    )
                    label = COLLECTION_CONFIG.get(
                        coll_name, {}
                    ).get("label", coll_name)
                    for r in results:
                        hit = SearchHit(
                            collection=label,
                            id=r.get("id", ""),
                            score=r.get("score", 0.0) * 0.8,
                            text=r.get("text_summary",
                                       r.get("text_chunk", "")),
                            metadata=r,
                        )
                        additional_hits.append(hit)
                except Exception:
                    pass

        return additional_hits

    def _merge_and_rank(self, hits: List[SearchHit]) -> List[SearchHit]:
        """Deduplicate by ID and sort by score descending."""
        seen = set()
        unique = []
        for hit in hits:
            if hit.id not in seen:
                seen.add(hit.id)
                unique.append(hit)
        unique.sort(key=lambda h: h.score, reverse=True)
        return unique[:30]  # Cap at 30 total results

    def _get_knowledge_context(self, query: str) -> str:
        """Extract knowledge graph context relevant to the query."""
        if not self.knowledge:
            return ""

        from .knowledge import get_target_context, get_toxicity_context

        context_parts = []

        # Check for target antigen mentions
        query_upper = query.upper()
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

        return "\n\n".join(context_parts)

    def _build_prompt(self, question: str,
                      evidence: CrossCollectionResult) -> str:
        """Build the LLM prompt with evidence and knowledge context."""
        sections = []

        # Evidence section — grouped by collection
        by_coll = evidence.hits_by_collection()
        for coll_name, hits in by_coll.items():
            section_lines = [f"### Evidence from {coll_name}"]
            for i, hit in enumerate(hits[:5], 1):
                section_lines.append(
                    f"{i}. [{hit.collection}:{hit.id}] "
                    f"(score={hit.score:.3f}) {hit.text[:500]}"
                )
            sections.append("\n".join(section_lines))

        evidence_text = "\n\n".join(sections) if sections else "No evidence found."

        # Knowledge section
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
            f"Cite sources using [Collection:ID] format. "
            f"Consider cross-functional insights across all stages of CAR-T development."
        )
