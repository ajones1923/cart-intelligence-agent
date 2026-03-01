"""CAR-T Intelligence Agent — autonomous reasoning across data silos.

Implements the plan → search → synthesize → report pattern from the
VAST AI OS AgentEngine model. The agent can:

1. Parse complex multi-part questions about CAR-T development
2. Plan a search strategy across relevant collections
3. Execute multi-collection retrieval via the RAG engine
4. Synthesize cross-functional insights
5. Generate structured reports

Mapping to VAST AI OS:
  - AgentEngine entry point: CARTIntelligenceAgent.run()
  - Plan → search_plan()
  - Execute → rag_engine.retrieve()
  - Reflect → evaluate_evidence()
  - Report → generate_report()

Author: Adam Jones
Date: February 2026
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from .knowledge import CART_TARGETS
from .models import AgentQuery, AgentResponse, CARTStage, CrossCollectionResult


@dataclass
class SearchPlan:
    """Agent's plan for answering a question."""
    question: str
    identified_topics: List[str] = field(default_factory=list)
    target_antigens: List[str] = field(default_factory=list)
    relevant_stages: List[CARTStage] = field(default_factory=list)
    search_strategy: str = "broad"  # broad, targeted, comparative
    sub_questions: List[str] = field(default_factory=list)


class CARTIntelligenceAgent:
    """Autonomous CAR-T Intelligence Agent.

    Wraps the multi-collection RAG engine with planning and reasoning
    capabilities. Designed to answer complex cross-functional questions
    about CAR-T cell therapy development.

    Example queries this agent handles:
    - "Why did CAR-T candidate X fail in vitro?"
    - "Compare 4-1BB vs CD28 costimulatory domains for DLBCL"
    - "What manufacturing parameters predict clinical response?"
    - "How does antigen density affect CAR-T efficacy?"
    - "What are the resistance mechanisms to BCMA-targeted CAR-T?"

    Usage:
        agent = CARTIntelligenceAgent(rag_engine)
        response = agent.run("Why do CD19 CAR-T therapies fail?")
    """

    def __init__(self, rag_engine):
        """Initialize agent with a configured RAG engine.

        Args:
            rag_engine: CARTRAGEngine instance with all collections connected
        """
        self.rag = rag_engine

    def run(self, question: str, **kwargs) -> AgentResponse:
        """Execute the full agent pipeline: plan → search → synthesize.

        Args:
            question: Natural language question about CAR-T therapy
            **kwargs: Additional query parameters (target_antigen, cart_stage)

        Returns:
            AgentResponse with answer, evidence, and metadata
        """
        # Phase 1: Plan
        plan = self.search_plan(question)

        # Phase 2: Search (via RAG engine)
        query = AgentQuery(
            question=question,
            target_antigen=kwargs.get(
                "target_antigen",
                plan.target_antigens[0] if plan.target_antigens else None,
            ),
            cart_stage=kwargs.get("cart_stage"),
            include_genomic=kwargs.get("include_genomic", True),
        )
        evidence = self.rag.retrieve(query)

        # Phase 3: Evaluate evidence quality
        quality = self.evaluate_evidence(evidence)

        # Phase 4: If evidence is thin, try sub-questions
        if quality == "insufficient" and plan.sub_questions:
            for sub_q in plan.sub_questions[:2]:
                sub_query = AgentQuery(question=sub_q, include_genomic=False)
                sub_evidence = self.rag.retrieve(sub_query)
                evidence.hits.extend(sub_evidence.hits)

        # Phase 5: Generate answer
        answer = self.rag.query(question, **kwargs)

        # Phase 6: Identify knowledge used
        knowledge_used = []
        if evidence.knowledge_context:
            for line in evidence.knowledge_context.split("\n"):
                if line.strip().startswith("Target:"):
                    knowledge_used.append(line.strip())

        return AgentResponse(
            question=question,
            answer=answer,
            evidence=evidence,
            knowledge_used=knowledge_used,
        )

    def search_plan(self, question: str) -> SearchPlan:
        """Analyze a question and plan the search strategy.

        Identifies target antigens, relevant development stages,
        and decomposes complex questions into sub-queries.

        Args:
            question: The user's question

        Returns:
            SearchPlan with identified topics and strategy
        """
        plan = SearchPlan(question=question)
        q_upper = question.upper()

        # Identify target antigens (single source from knowledge graph)
        plan.target_antigens = [a for a in CART_TARGETS if a in q_upper]

        # Identify relevant CAR-T development stages
        stage_keywords = {
            CARTStage.TARGET_ID: [
                "TARGET", "ANTIGEN", "EXPRESSION", "SPECIFICITY",
            ],
            CARTStage.CAR_DESIGN: [
                "CONSTRUCT", "SCFV", "COSTIMULAT", "4-1BB", "CD28",
                "DOMAIN", "GENERATION", "HINGE", "DESIGN",
            ],
            CARTStage.VECTOR_ENG: [
                "VECTOR", "LENTIVIR", "RETROVIR", "TRANSDUC", "VCN",
                "MANUFACTURING", "PRODUCTION", "CMC",
            ],
            CARTStage.TESTING: [
                "VITRO", "VIVO", "ASSAY", "CYTOTOX", "CYTOKINE",
                "MOUSE", "KILLING", "EXPANSION",
            ],
            CARTStage.CLINICAL: [
                "TRIAL", "PATIENT", "RESPONSE", "SURVIVAL", "TOXICITY",
                "CRS", "ICANS", "RELAPSE", "REMISSION", "FDA",
            ],
        }

        for stage, keywords in stage_keywords.items():
            if any(kw in q_upper for kw in keywords):
                plan.relevant_stages.append(stage)

        # Determine search strategy
        if "COMPARE" in q_upper or " VS " in q_upper or "VERSUS" in q_upper:
            plan.search_strategy = "comparative"
        elif plan.target_antigens and len(plan.relevant_stages) <= 1:
            plan.search_strategy = "targeted"
        else:
            plan.search_strategy = "broad"

        # Decompose complex questions into sub-queries
        if "WHY" in q_upper and "FAIL" in q_upper:
            plan.sub_questions = [
                f"What are the resistance mechanisms for "
                f"{plan.target_antigens[0] if plan.target_antigens else 'CAR-T'} therapy?",
                "What manufacturing issues lead to CAR-T therapy failure?",
                "What patient factors predict poor CAR-T response?",
            ]
        elif "COMPARE" in q_upper:
            plan.sub_questions = [
                f"What are the advantages of {plan.identified_topics[0] if plan.identified_topics else 'option A'}?",
                f"What are the limitations of each approach?",
            ]

        return plan

    def evaluate_evidence(self, evidence: CrossCollectionResult) -> str:
        """Evaluate the quality and coverage of retrieved evidence.

        Returns:
            "sufficient", "partial", or "insufficient"
        """
        if evidence.hit_count == 0:
            return "insufficient"

        by_coll = evidence.hits_by_collection()
        collections_with_hits = len(by_coll)

        if collections_with_hits >= 3 and evidence.hit_count >= 10:
            return "sufficient"
        elif collections_with_hits >= 2 and evidence.hit_count >= 5:
            return "partial"
        else:
            return "insufficient"

    def generate_report(self, response: AgentResponse) -> str:
        """Generate a structured cross-functional analysis report.

        Args:
            response: AgentResponse from a completed query

        Returns:
            Formatted markdown report
        """
        by_coll = response.evidence.hits_by_collection()

        report_lines = [
            f"# CAR-T Intelligence Report",
            f"**Query:** {response.question}",
            f"**Generated:** {response.timestamp}",
            f"**Collections Searched:** {response.evidence.total_collections_searched}",
            f"**Evidence Items:** {response.evidence.hit_count}",
            f"**Search Time:** {response.evidence.search_time_ms:.0f} ms",
            "",
            "---",
            "",
            "## Analysis",
            "",
            response.answer,
            "",
            "---",
            "",
            "## Evidence Sources",
            "",
        ]

        for coll_name, hits in by_coll.items():
            report_lines.append(f"### {coll_name} ({len(hits)} results)")
            for hit in hits[:5]:
                report_lines.append(
                    f"- **{hit.id}** (score: {hit.score:.3f}): "
                    f"{hit.text[:200]}..."
                )
            report_lines.append("")

        if response.knowledge_used:
            report_lines.extend([
                "## Knowledge Graph",
                "",
            ])
            for k in response.knowledge_used:
                report_lines.append(f"- {k}")

        return "\n".join(report_lines)
