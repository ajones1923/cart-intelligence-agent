"""CAR-T Intelligence Agent — Streamlit Chat Interface.

Provides a cross-functional query interface for CAR-T cell therapy
intelligence. Follows the same Streamlit patterns as:
  - rag-chat-pipeline/app/chat_ui.py

Port: 8520 (assigned to CAR-T Intelligence Agent)

Usage:
    streamlit run app/cart_ui.py --server.port 8520

Author: Adam Jones
Date: February 2026
"""

import os
import sys
from pathlib import Path

import streamlit as st

# Add project root to path
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

        # Collection manager
        manager = CARTCollectionManager()
        manager.connect()

        # Embedder - try to import from rag-chat-pipeline, fallback to sentence-transformers
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

        # LLM client
        try:
            import anthropic

            class SimpleLLMClient:
                def __init__(self):
                    self.client = anthropic.Anthropic()

                def generate(self, prompt, system_prompt="", max_tokens=2048, temperature=0.7):
                    msg = self.client.messages.create(
                        model="claude-sonnet-4-6",
                        max_tokens=max_tokens,
                        temperature=temperature,
                        system=system_prompt,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    return msg.content[0].text

                def generate_stream(self, prompt, system_prompt="", max_tokens=2048, temperature=0.7):
                    with self.client.messages.stream(
                        model="claude-sonnet-4-6",
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


engine, manager = init_engine()

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
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### Configuration")

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

    st.markdown("---")
    st.markdown("### Collections")

    collection_status = {
        "cart_literature": ("Literature", 0),
        "cart_trials": ("Clinical Trials", 0),
        "cart_constructs": ("CAR Constructs", 0),
        "cart_assays": ("Assay Data", 0),
        "cart_manufacturing": ("Manufacturing", 0),
    }

    if manager:
        try:
            stats = manager.get_collection_stats()
            collection_status = {
                "cart_literature": ("Literature", stats.get("cart_literature", 0)),
                "cart_trials": ("Clinical Trials", stats.get("cart_trials", 0)),
                "cart_constructs": ("CAR Constructs", stats.get("cart_constructs", 0)),
                "cart_assays": ("Assay Data", stats.get("cart_assays", 0)),
                "cart_manufacturing": ("Manufacturing", stats.get("cart_manufacturing", 0)),
            }
        except Exception:
            pass  # Keep default zeros

    for coll_id, (label, count) in collection_status.items():
        st.markdown(f"**{label}**: {count:,} records")

    st.markdown("---")
    st.markdown("### Demo Queries")
    demo_queries = [
        "Why do CD19 CAR-T therapies fail in relapsed B-ALL?",
        "Compare 4-1BB vs CD28 costimulatory domains",
        "What manufacturing parameters predict response?",
        "BCMA CAR-T resistance mechanisms in myeloma",
        "How does T-cell exhaustion affect persistence?",
    ]
    for q in demo_queries:
        if st.button(q, key=f"demo_{q[:20]}"):
            st.session_state["demo_query"] = q

# ═══════════════════════════════════════════════════════════════════════
# CHAT INTERFACE
# ═══════════════════════════════════════════════════════════════════════

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

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

    # Generate response via RAG engine
    with st.chat_message("assistant"):
        if engine and engine.llm:
            # Build query kwargs from sidebar filters
            query_kwargs = {}
            if target_filter != "All Targets":
                query_kwargs["target_antigen"] = target_filter

            # Show evidence retrieval status
            with st.status("Searching across CAR-T data sources...", expanded=True) as status:
                try:
                    from src.models import AgentQuery

                    agent_query = AgentQuery(question=prompt, **query_kwargs)
                    evidence = engine.retrieve(agent_query)

                    st.write(f"Found {evidence.hit_count} results across {evidence.total_collections_searched} collections ({evidence.search_time_ms:.0f}ms)")

                    by_coll = evidence.hits_by_collection()
                    for coll_name, hits in by_coll.items():
                        st.write(f"  - **{coll_name}**: {len(hits)} hits")

                    status.update(label="Generating response...", state="running")
                except Exception as e:
                    st.error(f"Search error: {e}")
                    evidence = None

            # ── Evidence Panel ──────────────────────────────────────
            if evidence and evidence.hit_count > 0:
                with st.expander(
                    f"Evidence Sources ({evidence.hit_count} results, "
                    f"{evidence.search_time_ms:.0f}ms)",
                    expanded=False,
                ):
                    by_coll = evidence.hits_by_collection()
                    for coll_name, hits in by_coll.items():
                        badge_class = f"badge-{coll_name.lower()}"
                        for hit in hits[:5]:
                            # Build source link
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

                            snippet = hit.text[:200].replace("<", "&lt;").replace(">", "&gt;")
                            st.markdown(
                                f'<div class="evidence-card">'
                                f'<span class="collection-badge {badge_class}">{hit.collection}</span>'
                                f' <strong>{hit.id}</strong>'
                                f' <span class="score">{hit.score:.3f}</span>'
                                f'{source_link}'
                                f'<div class="snippet">{snippet}...</div>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )

            # ── LLM Response ────────────────────────────────────────
            if evidence:
                prompt_text = engine._build_prompt(prompt, evidence)
                from src.rag_engine import CART_SYSTEM_PROMPT

                response_text = ""
                message_placeholder = st.empty()
                try:
                    for token in engine.llm.generate_stream(
                        prompt=prompt_text,
                        system_prompt=CART_SYSTEM_PROMPT,
                        max_tokens=2048,
                        temperature=0.7,
                    ):
                        response_text += token
                        message_placeholder.markdown(response_text + "▌")
                    message_placeholder.markdown(response_text)
                except Exception as e:
                    response_text = f"LLM generation error: {e}"
                    message_placeholder.markdown(response_text)
        else:
            # Fallback scaffold mode
            response_text = (
                f"**[CAR-T Intelligence Agent — Scaffold Mode]**\n\n"
                f"Engine not fully initialized. Ensure Milvus is running on port 19530 "
                f"and ANTHROPIC_API_KEY is set.\n\n"
                f"Your query: *{prompt}*"
            )
            st.markdown(response_text)

        st.session_state.messages.append(
            {"role": "assistant", "content": response_text}
        )

# ═══════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; font-size: 0.8rem;'>"
    "HCLS AI Factory — CAR-T Intelligence Agent v1.1.0 "
    "| Apache 2.0 | Adam Jones | February 2026"
    "</div>",
    unsafe_allow_html=True,
)
