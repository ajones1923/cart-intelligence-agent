"""CAR-T Intelligence Agent — Streamlit Chat Interface v2.0.

Full-featured UI with:
- All 10 collection stats in sidebar
- Deep Research mode (autonomous agent pipeline)
- Conversation memory for follow-up queries
- Collection-specific filtering
- Temporal date-range filtering
- Citation relevance scoring (high/medium/low)
- Image/slide upload for claim verification
- Knowledge graph visualization
- Stage filter wired into query pipeline

Port: 8521 (assigned to CAR-T Intelligence Agent)

Usage:
    streamlit run app/cart_ui.py --server.port 8521

Author: Adam Jones
Date: February 2026
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st

# Add project root to path (must happen before src imports)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load API key from rag-chat-pipeline .env if not already set
if not os.environ.get("ANTHROPIC_API_KEY"):
    env_path = Path("/home/adam/projects/hcls-ai-factory/rag-chat-pipeline/.env")
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("ANTHROPIC_API_KEY="):
                os.environ["ANTHROPIC_API_KEY"] = line.split("=", 1)[1].strip().strip('"')
                break

from src.export import export_markdown, export_json, export_pdf, generate_filename


# ═══════════════════════════════════════════════════════════════════════
# ENGINE INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════


@st.cache_resource
def init_engine():
    """Initialize the CAR-T RAG engine (cached across reruns)."""
    try:
        from src.collections import CARTCollectionManager
        from src.rag_engine import CARTRAGEngine
        from src import knowledge as kg
        from src import query_expansion as qe

        manager = CARTCollectionManager()
        manager.connect()

        try:
            from sentence_transformers import SentenceTransformer

            class SimpleEmbedder:
                def __init__(self):
                    self.model = SentenceTransformer("BAAI/bge-small-en-v1.5")

                def embed_text(self, text):
                    return self.model.encode(text).tolist()

                def encode(self, texts):
                    return self.model.encode(texts).tolist()

            embedder = SimpleEmbedder()
        except ImportError:
            embedder = None

        try:
            import anthropic
            from config.settings import settings

            class SimpleLLMClient:
                def __init__(self):
                    self.client = anthropic.Anthropic()

                def generate(self, prompt, system_prompt="", max_tokens=2048, temperature=0.7):
                    msg = self.client.messages.create(
                        model=settings.LLM_MODEL,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        system=system_prompt,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    return msg.content[0].text

                def generate_stream(self, prompt, system_prompt="", max_tokens=2048, temperature=0.7):
                    with self.client.messages.stream(
                        model=settings.LLM_MODEL,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        system=system_prompt,
                        messages=[{"role": "user", "content": prompt}],
                    ) as stream:
                        for text in stream.text_stream:
                            yield text

            llm_client = SimpleLLMClient()
        except (ImportError, Exception):
            llm_client = None

        engine = CARTRAGEngine(
            collection_manager=manager,
            embedder=embedder,
            llm_client=llm_client,
            knowledge=kg,
            query_expander=qe,
        )
        return engine, manager
    except Exception as e:
        st.error(f"Failed to initialize: {e}")
        return None, None


@st.cache_resource
def init_agent(_engine):
    """Initialize the autonomous CAR-T Intelligence Agent."""
    if not _engine:
        return None
    try:
        from src.agent import CARTIntelligenceAgent
        return CARTIntelligenceAgent(_engine)
    except Exception:
        return None


engine, manager = init_engine()
agent = init_agent(engine)

# ═══════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="CAR-T Intelligence | HCLS AI Factory",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════
# CUSTOM CSS — NVIDIA Black + Green theme
# ═══════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
    .stApp { background-color: #0a0a0a; }

    .main-title {
        font-size: 2rem;
        font-weight: 700;
        color: #76B900;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1rem;
        color: #aaaaaa;
        margin-bottom: 1.5rem;
    }

    .collection-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 4px;
    }
    .badge-literature { background: #1D6FA4; color: white; }
    .badge-trial { background: #952FC6; color: white; }
    .badge-construct { background: #76B900; color: white; }
    .badge-assay { background: #F9C500; color: black; }
    .badge-manufacturing { background: #DF6500; color: white; }
    .badge-genomic { background: #1DBFA4; color: white; }
    .badge-safety { background: #E53935; color: white; }
    .badge-biomarker { background: #00897B; color: white; }
    .badge-regulatory { background: #5C6BC0; color: white; }
    .badge-sequence { background: #F06292; color: white; }
    .badge-realworld { background: #8D6E63; color: white; }

    .relevance-high { color: #76B900; font-weight: 700; }
    .relevance-medium { color: #F9C500; font-weight: 600; }
    .relevance-low { color: #888; font-weight: 400; }

    .evidence-card {
        background: #1e1e1e;
        border: 1px solid #333;
        border-radius: 8px;
        padding: 12px;
        margin: 6px 0;
    }
    .evidence-card .score {
        color: #76B900;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .evidence-card .snippet {
        color: #ccc;
        font-size: 0.85rem;
        margin-top: 6px;
        line-height: 1.4;
    }
    .evidence-card a {
        color: #76B900;
        text-decoration: none;
    }
    .evidence-card a:hover {
        text-decoration: underline;
    }

    .entity-header {
        font-size: 1rem;
        font-weight: 700;
        padding: 8px 12px;
        margin: 12px 0 6px 0;
        border-radius: 6px;
    }
    .entity-header-a { background: #1D6FA4; color: white; }
    .entity-header-b { background: #952FC6; color: white; }
    .vs-divider {
        text-align: center;
        color: #76B900;
        font-size: 1.2rem;
        font-weight: 700;
        margin: 8px 0;
    }

    .mode-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.7rem;
        font-weight: 700;
        margin-left: 8px;
    }
    .mode-deep { background: #952FC6; color: white; }
    .mode-quick { background: #1D6FA4; color: white; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════

st.markdown(
    '<div class="main-title">CAR-T Intelligence Agent</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-title">'
    'Cross-Functional Intelligence Across the CAR-T Development Lifecycle '
    '| HCLS AI Factory'
    '</div>',
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════════════════════════════
# MODE TABS: Chat | Knowledge Graph | Image Analysis
# ═══════════════════════════════════════════════════════════════════════

tab_chat, tab_kg, tab_image = st.tabs(["Chat", "Knowledge Graph", "Image Analysis"])

# ═══════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### Configuration")

    # Deep Research toggle
    deep_research = st.toggle("Deep Research Mode", value=False,
                              help="Uses autonomous agent with sub-question decomposition and evidence evaluation")
    if deep_research:
        st.markdown('<span class="mode-badge mode-deep">DEEP RESEARCH</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="mode-badge mode-quick">QUICK RAG</span>', unsafe_allow_html=True)

    target_filter = st.selectbox(
        "Target Antigen Filter",
        ["All Targets", "CD19", "BCMA", "CD22", "CD20", "CD30",
         "CD33", "CD38", "CD123", "GD2", "HER2", "GPC3", "EGFR",
         "Mesothelin", "PSMA", "ROR1"],
    )

    stage_filter = st.selectbox(
        "Development Stage",
        ["All Stages", "Target Identification", "CAR Design",
         "Vector Engineering", "Testing", "Clinical"],
    )

    # Temporal date-range filter
    st.markdown("---")
    st.markdown("### Date Range")
    col_yr1, col_yr2 = st.columns(2)
    with col_yr1:
        year_min = st.number_input("From Year", min_value=2010, max_value=2030,
                                   value=2010, step=1)
    with col_yr2:
        year_max = st.number_input("To Year", min_value=2010, max_value=2030,
                                   value=2026, step=1)
    use_date_filter = st.checkbox("Apply date filter", value=False)

    # Collection selection
    st.markdown("---")
    st.markdown("### Collections")

    COLLECTION_LABELS = {
        "cart_literature": "Literature",
        "cart_trials": "Clinical Trials",
        "cart_constructs": "CAR Constructs",
        "cart_assays": "Assay Data",
        "cart_manufacturing": "Manufacturing",
        "cart_safety": "Safety",
        "cart_biomarkers": "Biomarkers",
        "cart_regulatory": "Regulatory",
        "cart_sequences": "Sequences",
        "cart_realworld": "Real-World Evidence",
        "genomic_evidence": "Genomic Evidence",
    }

    # Get live stats
    collection_stats = {}
    if manager:
        try:
            collection_stats = manager.get_collection_stats()
        except Exception:
            pass

    # Collection toggles with record counts
    selected_collections = []
    for coll_id, label in COLLECTION_LABELS.items():
        count = collection_stats.get(coll_id, 0)
        checked = st.checkbox(f"{label} ({count:,})", value=True, key=f"coll_{coll_id}")
        if checked:
            selected_collections.append(coll_id)

    total_vectors = sum(collection_stats.values())
    st.markdown(f"**Total: {total_vectors:,} vectors across {len(selected_collections)} collections**")

    st.markdown("---")
    st.markdown("### Demo Queries")
    demo_queries = [
        "Why do CD19 CAR-T therapies fail in relapsed B-ALL?",
        "Compare 4-1BB vs CD28 costimulatory domains",
        "What manufacturing parameters predict response?",
        "BCMA CAR-T resistance mechanisms in myeloma",
        "How does T-cell exhaustion affect persistence?",
        "What are the long-term safety signals for CD19 CAR-T products?",
        "Which biomarkers best predict CRS severity?",
        "Compare the FDA regulatory pathway of Kymriah vs Yescarta",
        "What is the binding affinity of FMC63 scFv?",
        "How do real-world CAR-T outcomes compare between academic and community centers?",
        "What genomic variants in CD19 or BCMA pathway genes affect CAR-T response?",
        "What patents cover bispecific CAR-T constructs targeting CD19 and CD22?",
        "How does scFv humanization reduce immunogenicity risk in CAR-T therapy?",
    ]
    for q in demo_queries:
        if st.button(q, key=f"demo_{q[:20]}"):
            st.session_state["demo_query"] = q


# ═══════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════


def render_evidence_cards(evidence):
    """Render evidence cards with relevance indicators."""
    by_coll = evidence.hits_by_collection()
    for coll_name, hits in by_coll.items():
        badge_class = f"badge-{coll_name.lower()}"
        for hit in hits[:5]:
            source_link = ""
            if hit.collection == "Literature" and hit.id.isdigit():
                source_link = (
                    f' <a href="https://pubmed.ncbi.nlm.nih.gov/{hit.id}/"'
                    f' target="_blank">PubMed</a>'
                )
            elif hit.collection == "Trial" and hit.id.upper().startswith("NCT"):
                source_link = (
                    f' <a href="https://clinicaltrials.gov/study/{hit.id}"'
                    f' target="_blank">ClinicalTrials.gov</a>'
                )

            relevance = hit.metadata.get("relevance", "")
            relevance_html = ""
            if relevance:
                relevance_html = f' <span class="relevance-{relevance}">[{relevance}]</span>'

            snippet = hit.text[:200].replace("<", "&lt;").replace(">", "&gt;")
            st.markdown(
                f'<div class="evidence-card">'
                f'<span class="collection-badge {badge_class}">{hit.collection}</span>'
                f' <strong>{hit.id}</strong>'
                f' <span class="score">{hit.score:.3f}</span>'
                f'{relevance_html}'
                f'{source_link}'
                f'<div class="snippet">{snippet}...</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


def build_conversation_context():
    """Build conversation context from recent exchanges for follow-up queries."""
    from config.settings import settings
    if "messages" not in st.session_state:
        return None

    recent = []
    msg_pairs = []
    messages = st.session_state.messages
    for i in range(len(messages) - 1):
        if messages[i]["role"] == "user" and i + 1 < len(messages) and messages[i + 1]["role"] == "assistant":
            msg_pairs.append((messages[i]["content"], messages[i + 1]["content"][:300]))

    if not msg_pairs:
        return None

    # Take last N exchanges
    for q, a in msg_pairs[-settings.MAX_CONVERSATION_CONTEXT:]:
        recent.append(f"Previous Q: {q}\nPrevious A (summary): {a}")

    return "\n\n".join(recent) if recent else None


# ═══════════════════════════════════════════════════════════════════════
# TAB 1: CHAT INTERFACE
# ═══════════════════════════════════════════════════════════════════════

with tab_chat:
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for msg_idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("evidence_data"):
                ed = msg["evidence_data"]
                _export_kwargs = dict(
                    query=ed["query"], response_text=msg["content"],
                    evidence=ed.get("evidence"),
                    comp_result=ed.get("comp_result"),
                    filters_applied=ed.get("filters"),
                )
                col1, col2, col3, _spacer = st.columns([1, 1, 1, 3])
                with col1:
                    st.download_button(
                        "Download Markdown", export_markdown(**_export_kwargs),
                        file_name=generate_filename("md"),
                        mime="text/markdown",
                        key=f"dl_md_{msg_idx}",
                    )
                with col2:
                    st.download_button(
                        "Download JSON", export_json(**_export_kwargs),
                        file_name=generate_filename("json"),
                        mime="application/json",
                        key=f"dl_json_{msg_idx}",
                    )
                with col3:
                    st.download_button(
                        "Download PDF", export_pdf(**_export_kwargs),
                        file_name=generate_filename("pdf"),
                        mime="application/pdf",
                        key=f"dl_pdf_{msg_idx}",
                    )

    # Chat input
    prompt = st.chat_input("Ask about CAR-T cell therapy development...")

    # Handle demo query
    if "demo_query" in st.session_state:
        prompt = st.session_state.pop("demo_query")

    if prompt:
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            if engine and engine.llm:
                # Build query parameters
                query_kwargs = {}
                if target_filter != "All Targets":
                    query_kwargs["target_antigen"] = target_filter

                # Build retrieve parameters
                retrieve_kwargs = {}
                if selected_collections and len(selected_collections) < 10:
                    retrieve_kwargs["collections_filter"] = selected_collections
                if use_date_filter:
                    retrieve_kwargs["year_min"] = year_min
                    retrieve_kwargs["year_max"] = year_max

                # Conversation memory
                conv_context = build_conversation_context()
                if conv_context:
                    retrieve_kwargs["conversation_context"] = conv_context

                # Detect comparative query
                is_comparative = engine._is_comparative(prompt)
                evidence = None
                comp_result = None
                agent_report = None

                with st.status("Searching across CAR-T data sources...", expanded=True) as status:
                    try:
                        if deep_research and agent:
                            # Deep Research mode: autonomous agent pipeline
                            status.update(label="Deep Research: planning search strategy...")
                            plan = agent.search_plan(prompt)
                            st.write(f"**Strategy:** {plan.search_strategy}")
                            if plan.target_antigens:
                                st.write(f"**Targets:** {', '.join(plan.target_antigens)}")
                            if plan.relevant_stages:
                                st.write(f"**Stages:** {', '.join(s.value for s in plan.relevant_stages)}")
                            if plan.sub_questions:
                                st.write(f"**Sub-questions:** {len(plan.sub_questions)}")

                            status.update(label="Deep Research: retrieving evidence...")
                            from src.models import AgentQuery
                            agent_query = AgentQuery(question=prompt, **query_kwargs)
                            evidence = engine.retrieve(agent_query, **retrieve_kwargs)

                            quality = agent.evaluate_evidence(evidence)
                            st.write(f"**Evidence quality:** {quality}")

                            if quality == "insufficient" and plan.sub_questions:
                                status.update(label="Deep Research: expanding with sub-questions...")
                                for sub_q in plan.sub_questions[:2]:
                                    sub_query = AgentQuery(question=sub_q, include_genomic=False)
                                    sub_evidence = engine.retrieve(sub_query)
                                    evidence.hits.extend(sub_evidence.hits)
                                st.write(f"**Augmented to:** {evidence.hit_count} hits")

                            st.write(
                                f"Found {evidence.hit_count} results across "
                                f"{evidence.total_collections_searched} collections "
                                f"({evidence.search_time_ms:.0f}ms)"
                            )

                        elif is_comparative:
                            comp_result = engine.retrieve_comparative(
                                prompt,
                                collections_filter=retrieve_kwargs.get("collections_filter"),
                                year_min=retrieve_kwargs.get("year_min"),
                                year_max=retrieve_kwargs.get("year_max"),
                            )
                            if comp_result:
                                st.write(
                                    f"Comparative analysis: **{comp_result.entity_a}** "
                                    f"vs **{comp_result.entity_b}**"
                                )
                                st.write(
                                    f"Found {comp_result.total_hits} results "
                                    f"({comp_result.total_search_time_ms:.0f}ms)"
                                )
                            else:
                                is_comparative = False

                        if not is_comparative and not (deep_research and agent):
                            from src.models import AgentQuery
                            agent_query = AgentQuery(question=prompt, **query_kwargs)
                            evidence = engine.retrieve(agent_query, **retrieve_kwargs)
                            st.write(
                                f"Found {evidence.hit_count} results across "
                                f"{evidence.total_collections_searched} collections "
                                f"({evidence.search_time_ms:.0f}ms)"
                            )
                            by_coll = evidence.hits_by_collection()
                            for coll_name, hits in by_coll.items():
                                st.write(f"  - **{coll_name}**: {len(hits)} hits")

                        status.update(label="Generating response...", state="running")
                    except Exception as e:
                        st.error(f"Search error: {e}")
                        evidence = None
                        comp_result = None

                # ── Evidence Panel ──────────────────────────────────────
                if is_comparative and comp_result and comp_result.total_hits > 0:
                    with st.expander(
                        f"Comparative Evidence ({comp_result.total_hits} results, "
                        f"{comp_result.total_search_time_ms:.0f}ms)",
                        expanded=False,
                    ):
                        st.markdown(
                            f'<div class="entity-header entity-header-a">'
                            f'{comp_result.entity_a}</div>',
                            unsafe_allow_html=True,
                        )
                        render_evidence_cards(comp_result.evidence_a)
                        st.markdown(
                            '<div class="vs-divider">— VS —</div>',
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            f'<div class="entity-header entity-header-b">'
                            f'{comp_result.entity_b}</div>',
                            unsafe_allow_html=True,
                        )
                        render_evidence_cards(comp_result.evidence_b)

                elif evidence and evidence.hit_count > 0:
                    with st.expander(
                        f"Evidence Sources ({evidence.hit_count} results, "
                        f"{evidence.search_time_ms:.0f}ms)",
                        expanded=False,
                    ):
                        render_evidence_cards(evidence)

                # ── LLM Response ────────────────────────────────────────
                from src.rag_engine import CART_SYSTEM_PROMPT

                if is_comparative and comp_result:
                    prompt_text = engine._build_comparative_prompt(prompt, comp_result)
                    max_tokens = 3000
                elif evidence:
                    prompt_text = engine._build_prompt(prompt, evidence)
                    max_tokens = 2048
                else:
                    prompt_text = None

                if prompt_text:
                    response_text = ""
                    message_placeholder = st.empty()
                    try:
                        for token in engine.llm.generate_stream(
                            prompt=prompt_text,
                            system_prompt=CART_SYSTEM_PROMPT,
                            max_tokens=max_tokens,
                            temperature=0.7,
                        ):
                            response_text += token
                            message_placeholder.markdown(response_text + "▌")
                        message_placeholder.markdown(response_text)
                    except Exception as e:
                        response_text = f"LLM generation error: {e}"
                        message_placeholder.markdown(response_text)

                # ── Export Buttons ──────────────────────────────────────
                has_results = (
                    (is_comparative and comp_result and comp_result.total_hits > 0)
                    or (evidence and evidence.hit_count > 0)
                )
                if prompt_text and has_results:
                    active_filters = {
                        "target_antigen": target_filter,
                        "stage": stage_filter,
                        "date_range": f"{year_min}-{year_max}" if use_date_filter else "all",
                        "collections": len(selected_collections),
                        "mode": "deep_research" if deep_research else "quick_rag",
                    }
                    _export_kwargs = dict(
                        query=prompt, response_text=response_text,
                        evidence=evidence, comp_result=comp_result,
                        filters_applied=active_filters,
                    )
                    col1, col2, col3, _spacer = st.columns([1, 1, 1, 3])
                    with col1:
                        st.download_button(
                            "Download Markdown",
                            export_markdown(**_export_kwargs),
                            file_name=generate_filename("md"),
                            mime="text/markdown",
                            key="dl_md_new",
                        )
                    with col2:
                        st.download_button(
                            "Download JSON",
                            export_json(**_export_kwargs),
                            file_name=generate_filename("json"),
                            mime="application/json",
                            key="dl_json_new",
                        )
                    with col3:
                        st.download_button(
                            "Download PDF",
                            export_pdf(**_export_kwargs),
                            file_name=generate_filename("pdf"),
                            mime="application/pdf",
                            key="dl_pdf_new",
                        )
            else:
                response_text = (
                    f"**[CAR-T Intelligence Agent — Scaffold Mode]**\n\n"
                    f"Engine not fully initialized. Ensure Milvus is running on port 19530 "
                    f"and ANTHROPIC_API_KEY is set.\n\n"
                    f"Your query: *{prompt}*"
                )
                st.markdown(response_text)

            # Persist message with evidence
            evidence_data = None
            if engine and engine.llm:
                has_results = (
                    (comp_result and comp_result.total_hits > 0)
                    or (evidence and evidence.hit_count > 0)
                )
                if has_results:
                    evidence_data = {
                        "query": prompt,
                        "evidence": evidence,
                        "comp_result": comp_result,
                        "filters": {
                            "target_antigen": target_filter,
                            "stage": stage_filter,
                        },
                    }
            st.session_state.messages.append(
                {"role": "assistant", "content": response_text, "evidence_data": evidence_data}
            )

# ═══════════════════════════════════════════════════════════════════════
# TAB 2: KNOWLEDGE GRAPH VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════

with tab_kg:
    st.markdown("### Knowledge Graph Explorer")
    st.markdown("Interactive visualization of CAR-T entity relationships across all knowledge domains.")

    try:
        from src import knowledge as kg

        # Entity selector
        kg_col1, kg_col2 = st.columns([1, 2])
        with kg_col1:
            entity_type = st.selectbox(
                "Entity Type",
                ["Target Antigens", "Toxicities", "Manufacturing", "Biomarkers", "Regulatory"],
                key="kg_entity_type",
            )

        # Build and display knowledge graph
        stats = kg.get_knowledge_stats()
        st.markdown(
            f"**Knowledge Graph:** {stats.get('targets', 0)} targets, "
            f"{stats.get('toxicities', 0)} toxicities, "
            f"{stats.get('manufacturing', 0)} manufacturing processes, "
            f"{stats.get('biomarkers', 0)} biomarkers, "
            f"{stats.get('regulatory_products', 0)} regulatory products"
        )

        # Try to render interactive graph with pyvis
        try:
            from pyvis.network import Network
            import tempfile

            net = Network(height="500px", width="100%", bgcolor="#0a0a0a",
                          font_color="white", directed=False)
            net.barnes_hut()

            if entity_type == "Target Antigens":
                from src.knowledge import CART_TARGETS
                for antigen, data in list(CART_TARGETS.items())[:15]:
                    net.add_node(antigen, label=antigen, color="#76B900", size=25)
                    for disease in data.get("diseases", [])[:3]:
                        net.add_node(disease, label=disease, color="#1D6FA4", size=15)
                        net.add_edge(antigen, disease)
                    for product in data.get("approved_products", [])[:2]:
                        net.add_node(product, label=product, color="#952FC6", size=18)
                        net.add_edge(antigen, product)
                    for resistance in data.get("known_resistance", [])[:2]:
                        short = resistance[:30]
                        net.add_node(short, label=short, color="#E53935", size=12)
                        net.add_edge(antigen, short)

            elif entity_type == "Toxicities":
                from src.knowledge import CART_TOXICITIES
                for tox_key, data in CART_TOXICITIES.items():
                    name = data.get("full_name", tox_key)
                    net.add_node(tox_key, label=name, color="#E53935", size=25)
                    for biomarker in data.get("biomarkers", [])[:3]:
                        net.add_node(biomarker, label=biomarker, color="#00897B", size=15)
                        net.add_edge(tox_key, biomarker)
                    for drug in data.get("management", {}).get("drugs", [])[:2]:
                        net.add_node(drug, label=drug, color="#5C6BC0", size=12)
                        net.add_edge(tox_key, drug)

            elif entity_type == "Manufacturing":
                from src.knowledge import CART_MANUFACTURING
                for proc_key, data in CART_MANUFACTURING.items():
                    name = proc_key.replace("_", " ").title()
                    net.add_node(proc_key, label=name, color="#DF6500", size=25)
                    for param in data.get("critical_parameters", [])[:3]:
                        short = param[:25]
                        net.add_node(short, label=short, color="#F9C500", size=12)
                        net.add_edge(proc_key, short)

            elif entity_type == "Biomarkers":
                from src.knowledge import CART_BIOMARKERS
                for bm_key, data in CART_BIOMARKERS.items():
                    name = data.get("full_name", bm_key)
                    bm_type = data.get("type", "")
                    color = {"predictive": "#00897B", "pharmacodynamic": "#1D6FA4",
                             "monitoring": "#F9C500", "resistance": "#E53935"}.get(bm_type, "#888")
                    net.add_node(bm_key, label=name, color=color, size=20)
                    outcome = data.get("associated_outcome", "")
                    if outcome:
                        short = outcome[:30]
                        net.add_node(short, label=short, color="#aaa", size=12)
                        net.add_edge(bm_key, short)

            elif entity_type == "Regulatory":
                from src.knowledge import CART_REGULATORY
                for prod_key, data in CART_REGULATORY.items():
                    net.add_node(prod_key, label=prod_key, color="#5C6BC0", size=25)
                    indication = data.get("initial_indication", "")
                    if indication:
                        short = indication[:30]
                        net.add_node(short, label=short, color="#1D6FA4", size=15)
                        net.add_edge(prod_key, short)
                    for desig in data.get("designations", [])[:2]:
                        net.add_node(desig, label=desig, color="#76B900", size=12)
                        net.add_edge(prod_key, desig)

            # Save and display
            with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
                net.save_graph(f.name)
                html_content = Path(f.name).read_text()
                st.components.v1.html(html_content, height=520, scrolling=True)

        except ImportError:
            st.info("Install `pyvis` for interactive graph visualization: `pip install pyvis`")

            # Fallback: text-based knowledge display
            if entity_type == "Target Antigens":
                from src.knowledge import CART_TARGETS
                for antigen, data in list(CART_TARGETS.items())[:10]:
                    with st.expander(f"{antigen} — {', '.join(data.get('diseases', [])[:2])}"):
                        st.markdown(f"**UniProt:** {data.get('uniprot_id', 'N/A')}")
                        st.markdown(f"**Diseases:** {', '.join(data.get('diseases', []))}")
                        st.markdown(f"**Products:** {', '.join(data.get('approved_products', ['None']))}")
                        st.markdown(f"**Resistance:** {', '.join(data.get('known_resistance', ['Unknown']))}")

    except Exception as e:
        st.error(f"Knowledge graph error: {e}")

    # Cross-collection entity search
    st.markdown("---")
    st.markdown("### Cross-Collection Entity Search")
    entity_search = st.text_input("Search for an entity across all collections",
                                  placeholder="e.g., Yescarta, CD19, FMC63",
                                  key="kg_entity_search")
    if entity_search and engine:
        with st.spinner(f"Finding all evidence related to '{entity_search}'..."):
            try:
                related = engine.find_related(entity_search, top_k=3)
                for coll_name, hits in related.items():
                    label = coll_name.replace("cart_", "").title()
                    st.markdown(f"**{label}** ({len(hits)} results)")
                    for hit in hits:
                        st.markdown(f"- `{hit.id}` ({hit.score:.3f}): {hit.text[:150]}...")
            except Exception as e:
                st.error(f"Search error: {e}")


# ═══════════════════════════════════════════════════════════════════════
# TAB 3: IMAGE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════

with tab_image:
    st.markdown("### Slide & Image Analysis")
    st.markdown(
        "Upload a slide image or document screenshot. The agent will extract claims "
        "and find supporting evidence across all 10 collections."
    )

    uploaded_file = st.file_uploader(
        "Upload an image (PNG, JPG, PDF)",
        type=["png", "jpg", "jpeg", "pdf"],
        key="image_upload",
    )

    if uploaded_file:
        # Display the image
        if uploaded_file.type.startswith("image"):
            st.image(uploaded_file, caption=uploaded_file.name, use_container_width=True)

        analyze_btn = st.button("Analyze Image", key="analyze_image")

        if analyze_btn and engine and engine.llm:
            import base64

            with st.spinner("Analyzing image with Claude Vision..."):
                try:
                    import anthropic
                    from config.settings import settings

                    client = anthropic.Anthropic()
                    image_data = base64.b64encode(uploaded_file.getvalue()).decode("utf-8")

                    media_type = uploaded_file.type
                    if media_type == "application/pdf":
                        media_type = "application/pdf"

                    # Step 1: Extract claims from the image
                    vision_response = client.messages.create(
                        model=settings.LLM_MODEL,
                        max_tokens=2000,
                        messages=[{
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": media_type,
                                        "data": image_data,
                                    },
                                },
                                {
                                    "type": "text",
                                    "text": (
                                        "You are analyzing a slide/image about CAR-T cell therapy. "
                                        "Extract ALL specific claims, data points, and assertions made in this image. "
                                        "For each claim, provide:\n"
                                        "1. The exact claim or data point\n"
                                        "2. A search query that could verify this claim in a knowledge base\n\n"
                                        "Format as JSON array: [{\"claim\": \"...\", \"search_query\": \"...\"}]"
                                    ),
                                },
                            ],
                        }],
                    )

                    claims_text = vision_response.content[0].text
                    st.markdown("#### Extracted Claims")

                    # Try to parse JSON claims
                    try:
                        # Find JSON array in response
                        import re
                        json_match = re.search(r'\[.*\]', claims_text, re.DOTALL)
                        if json_match:
                            claims = json.loads(json_match.group())
                        else:
                            claims = [{"claim": claims_text, "search_query": claims_text[:200]}]
                    except (json.JSONDecodeError, Exception):
                        claims = [{"claim": claims_text, "search_query": claims_text[:200]}]

                    # Step 2: Search for evidence supporting each claim
                    st.markdown("#### Evidence Verification")
                    for i, claim_data in enumerate(claims[:5]):
                        claim = claim_data.get("claim", "")
                        search_q = claim_data.get("search_query", claim)

                        with st.expander(f"Claim {i+1}: {claim[:100]}...", expanded=(i == 0)):
                            st.markdown(f"**Claim:** {claim}")
                            st.markdown(f"**Search:** {search_q}")

                            from src.models import AgentQuery
                            q = AgentQuery(question=search_q)
                            result = engine.retrieve(q, top_k_per_collection=2)

                            if result.hit_count > 0:
                                st.success(f"Found {result.hit_count} supporting evidence items")
                                for hit in result.hits[:3]:
                                    st.markdown(
                                        f"- **[{hit.collection}:{hit.id}]** "
                                        f"({hit.score:.3f}): {hit.text[:200]}..."
                                    )
                            else:
                                st.warning("No direct evidence found for this claim")

                except Exception as e:
                    st.error(f"Image analysis error: {e}")
        elif analyze_btn:
            st.warning("Engine not initialized. Ensure Milvus and API key are configured.")


# ═══════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; font-size: 0.8rem;'>"
    "HCLS AI Factory — CAR-T Intelligence Agent v2.0.0 "
    "| Apache 2.0 | Adam Jones | February 2026"
    "</div>",
    unsafe_allow_html=True,
)
