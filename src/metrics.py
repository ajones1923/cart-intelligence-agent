"""Prometheus metrics for the CAR-T Intelligence Agent.

Exposes counters, histograms, and gauges for query latency, collection hits,
LLM token usage, evidence counts, and ingest freshness.  Scraped by the
Grafana + Prometheus stack on port 9099.

All metrics use the ``cart_`` prefix so they are easily filterable in
Grafana dashboards alongside the existing HCLS AI Factory exporters
(node_exporter:9100, DCGM:9400).

If ``prometheus_client`` is not installed the module silently exports
no-op stubs so the rest of the application can import metrics helpers
without a hard dependency.

Author: Adam Jones
Date: February 2026
"""

from __future__ import annotations

from typing import Dict

try:
    from prometheus_client import Counter, Gauge, Histogram, generate_latest

    # ── Histograms ────────────────────────────────────────────────────
    QUERY_LATENCY = Histogram(
        "cart_query_latency_seconds",
        "Query processing time",
        ["query_type"],
        buckets=[0.1, 0.5, 1, 2, 5, 10, 30],
    )

    EVIDENCE_COUNT = Histogram(
        "cart_evidence_count",
        "Evidence items per query",
        buckets=[0, 5, 10, 15, 20, 25, 30],
    )

    # ── Counters ──────────────────────────────────────────────────────
    QUERY_COUNT = Counter(
        "cart_queries_total",
        "Total queries processed",
        ["query_type", "status"],
    )

    COLLECTION_HITS = Counter(
        "cart_collection_hits_total",
        "Hits by collection",
        ["collection"],
    )

    LLM_TOKENS = Counter(
        "cart_llm_tokens_total",
        "LLM tokens used",
        ["direction"],
    )

    # ── Gauges ────────────────────────────────────────────────────────
    ACTIVE_CONNECTIONS = Gauge(
        "cart_active_connections",
        "Active connections",
    )

    COLLECTION_SIZE = Gauge(
        "cart_collection_size",
        "Records per collection",
        ["collection"],
    )

    LAST_INGEST = Gauge(
        "cart_last_ingest_timestamp",
        "Last ingest timestamp",
        ["source"],
    )

    _PROMETHEUS_AVAILABLE = True

except ImportError:
    # ── No-op stubs when prometheus_client is not installed ────────────
    _PROMETHEUS_AVAILABLE = False

    class _NoOpLabeled:
        """Stub that silently ignores .labels().observe/inc/set calls."""

        def labels(self, *args, **kwargs):
            return self

        def observe(self, *args, **kwargs):
            pass

        def inc(self, *args, **kwargs):
            pass

        def dec(self, *args, **kwargs):
            pass

        def set(self, *args, **kwargs):
            pass

    class _NoOpGauge:
        """Stub for label-less Gauge (ACTIVE_CONNECTIONS)."""

        def inc(self, *args, **kwargs):
            pass

        def dec(self, *args, **kwargs):
            pass

        def set(self, *args, **kwargs):
            pass

    QUERY_LATENCY = _NoOpLabeled()        # type: ignore[assignment]
    EVIDENCE_COUNT = _NoOpLabeled()        # type: ignore[assignment]
    QUERY_COUNT = _NoOpLabeled()           # type: ignore[assignment]
    COLLECTION_HITS = _NoOpLabeled()       # type: ignore[assignment]
    LLM_TOKENS = _NoOpLabeled()            # type: ignore[assignment]
    ACTIVE_CONNECTIONS = _NoOpGauge()      # type: ignore[assignment]
    COLLECTION_SIZE = _NoOpLabeled()       # type: ignore[assignment]
    LAST_INGEST = _NoOpLabeled()           # type: ignore[assignment]

    def generate_latest() -> bytes:  # type: ignore[misc]
        return b""


# ═════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════


def record_query(
    query_type: str,
    latency: float,
    hit_count: int,
    status: str = "success",
) -> None:
    """Record metrics for a completed query.

    Call this once at the end of every RAG query or agent run to update
    latency, hit count, and total-query counters in a single place.

    Args:
        query_type: Query category — e.g. ``"rag"``, ``"agent"``,
            ``"comparative"``, ``"entity_link"``.
        latency: Wall-clock processing time in **seconds**.
        hit_count: Total evidence items returned to the user.
        status: Outcome label — ``"success"`` or ``"error"``.
    """
    QUERY_LATENCY.labels(query_type=query_type).observe(latency)
    QUERY_COUNT.labels(query_type=query_type, status=status).inc()
    EVIDENCE_COUNT.observe(hit_count)


def record_collection_hits(hits_by_collection: Dict[str, int]) -> None:
    """Increment per-collection hit counters.

    Args:
        hits_by_collection: Mapping of collection name (e.g.
            ``"cart_literature"``) to the number of hits returned from
            that collection in a single query.
    """
    for collection, count in hits_by_collection.items():
        COLLECTION_HITS.labels(collection=collection).inc(count)


def update_collection_sizes(stats: Dict[str, int]) -> None:
    """Set the current record count for each collection.

    Typically called after ``CARTCollectionManager.get_collection_stats()``
    returns its ``{collection_name: row_count}`` dict.

    Args:
        stats: Mapping of collection name to entity/row count.
    """
    for collection, size in stats.items():
        COLLECTION_SIZE.labels(collection=collection).set(size)


def get_metrics_text() -> str:
    """Return the current Prometheus metrics exposition in text format.

    This is the string you serve at ``/metrics`` (typically via FastAPI
    or a dedicated ``prometheus_client.start_http_server``).

    Returns:
        UTF-8 decoded Prometheus exposition text, or an empty string if
        ``prometheus_client`` is not installed.
    """
    return generate_latest().decode("utf-8")
