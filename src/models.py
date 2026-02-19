"""Pydantic data models for CAR-T Intelligence Agent.

Maps to the 5 Milvus collections + knowledge graph entities.
Follows the same dataclass/Pydantic pattern as:
  - rag-chat-pipeline/src/vcf_parser.py (VariantEvidence)
  - drug-discovery-pipeline/src/models.py (GeneratedMolecule, DockingResult)
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════


class CARTStage(str, Enum):
    """The 4 stages of CAR-T development (from TJ Chen's architecture)."""
    TARGET_ID = "target_id"
    CAR_DESIGN = "car_design"
    VECTOR_ENG = "vector_eng"
    TESTING = "testing"
    CLINICAL = "clinical"


class SourceType(str, Enum):
    PUBMED = "pubmed"
    PMC = "pmc"
    PATENT = "patent"
    PREPRINT = "preprint"
    MANUAL = "manual"


class TrialPhase(str, Enum):
    EARLY_1 = "Early Phase 1"
    PHASE_1 = "Phase 1"
    PHASE_1_2 = "Phase 1/Phase 2"
    PHASE_2 = "Phase 2"
    PHASE_2_3 = "Phase 2/Phase 3"
    PHASE_3 = "Phase 3"
    PHASE_4 = "Phase 4"
    NA = "N/A"


class TrialStatus(str, Enum):
    RECRUITING = "Recruiting"
    ACTIVE = "Active, not recruiting"
    COMPLETED = "Completed"
    TERMINATED = "Terminated"
    WITHDRAWN = "Withdrawn"
    SUSPENDED = "Suspended"
    NOT_YET = "Not yet recruiting"
    UNKNOWN = "Unknown status"


class CARGeneration(str, Enum):
    FIRST = "1st"
    SECOND = "2nd"
    THIRD = "3rd"
    FOURTH = "4th"
    ARMORED = "armored"
    UNIVERSAL = "universal"


class AssayType(str, Enum):
    CYTOTOXICITY = "cytotoxicity"
    CYTOKINE = "cytokine"
    FLOW_CYTOMETRY = "flow"
    PROLIFERATION = "proliferation"
    IN_VIVO = "in_vivo"
    PERSISTENCE = "persistence"
    EXHAUSTION = "exhaustion"


class ProcessStep(str, Enum):
    TRANSDUCTION = "transduction"
    EXPANSION = "expansion"
    HARVEST = "harvest"
    FORMULATION = "formulation"
    RELEASE_TESTING = "release"
    CRYOPRESERVATION = "cryo"


class FDAStatus(str, Enum):
    APPROVED = "approved"
    BLA_FILED = "bla_filed"
    PHASE_3 = "phase3"
    PHASE_2 = "phase2"
    PHASE_1 = "phase1"
    PRECLINICAL = "preclinical"
    DISCONTINUED = "discontinued"


# ═══════════════════════════════════════════════════════════════════════
# COLLECTION MODELS (map to Milvus schemas)
# ═══════════════════════════════════════════════════════════════════════


class CARTLiterature(BaseModel):
    """Published research paper or patent — maps to cart_literature collection."""
    id: str = Field(..., description="PMID or patent number")
    title: str = Field(..., max_length=500)
    text_chunk: str = Field(..., max_length=3000, description="Text chunk for embedding")
    source_type: SourceType = SourceType.PUBMED
    year: int = Field(..., ge=1990, le=2030)
    cart_stage: CARTStage = CARTStage.CLINICAL
    target_antigen: str = Field("", max_length=100)
    disease: str = Field("", max_length=200)
    keywords: str = Field("", max_length=1000)
    journal: str = Field("", max_length=200)

    def to_embedding_text(self) -> str:
        """Generate text for BGE-small embedding."""
        parts = [self.title]
        if self.text_chunk:
            parts.append(self.text_chunk)
        if self.target_antigen:
            parts.append(f"Target: {self.target_antigen}")
        if self.disease:
            parts.append(f"Disease: {self.disease}")
        return " ".join(parts)


class ClinicalTrial(BaseModel):
    """ClinicalTrials.gov record — maps to cart_trials collection."""
    id: str = Field(..., description="NCT number", pattern=r"^NCT\d{8}$")
    title: str = Field(..., max_length=500)
    text_summary: str = Field(..., max_length=3000)
    phase: TrialPhase = TrialPhase.NA
    status: TrialStatus = TrialStatus.UNKNOWN
    sponsor: str = Field("", max_length=200)
    target_antigen: str = Field("", max_length=100)
    car_generation: CARGeneration = CARGeneration.SECOND
    costimulatory: str = Field("", max_length=50, description="CD28, 4-1BB, dual")
    disease: str = Field("", max_length=200)
    enrollment: int = Field(0, ge=0)
    start_year: int = Field(0, ge=0, le=2030)
    outcome_summary: str = Field("", max_length=2000)

    def to_embedding_text(self) -> str:
        parts = [self.title, self.text_summary]
        if self.target_antigen:
            parts.append(f"Target antigen: {self.target_antigen}")
        if self.disease:
            parts.append(f"Indication: {self.disease}")
        if self.outcome_summary:
            parts.append(f"Outcome: {self.outcome_summary}")
        return " ".join(parts)


class CARConstruct(BaseModel):
    """CAR construct design — maps to cart_constructs collection."""
    id: str = Field(..., max_length=100)
    name: str = Field(..., max_length=200, description="Product or construct name")
    text_summary: str = Field(..., max_length=2000)
    target_antigen: str = Field(..., max_length=100)
    scfv_origin: str = Field("", max_length=200, description="Antibody clone/origin")
    costimulatory_domain: str = Field("", max_length=100)
    signaling_domain: str = Field("CD3-zeta", max_length=100)
    generation: CARGeneration = CARGeneration.SECOND
    hinge_tm: str = Field("", max_length=200, description="Hinge + transmembrane")
    vector_type: str = Field("lentiviral", max_length=50)
    fda_status: FDAStatus = FDAStatus.PRECLINICAL
    known_toxicities: str = Field("", max_length=500)

    def to_embedding_text(self) -> str:
        parts = [
            f"{self.name}: {self.text_summary}",
            f"Target: {self.target_antigen}",
            f"Generation: {self.generation.value}",
        ]
        if self.costimulatory_domain:
            parts.append(f"Costimulatory: {self.costimulatory_domain}")
        if self.known_toxicities:
            parts.append(f"Toxicities: {self.known_toxicities}")
        return " ".join(parts)


class AssayResult(BaseModel):
    """In vitro / in vivo assay result — maps to cart_assays collection."""
    id: str = Field(..., max_length=100)
    text_summary: str = Field(..., max_length=2000)
    assay_type: AssayType = AssayType.CYTOTOXICITY
    construct_id: str = Field("", max_length=100, description="Link to cart_constructs")
    target_antigen: str = Field("", max_length=100)
    cell_line: str = Field("", max_length=100, description="e.g., Nalm-6, Raji, K562")
    effector_ratio: str = Field("", max_length=20, description="E:T ratio")
    key_metric: str = Field("", max_length=50)
    metric_value: float = Field(0.0)
    outcome: str = Field("", max_length=20, description="success, partial, failure")
    notes: str = Field("", max_length=1000)

    def to_embedding_text(self) -> str:
        parts = [self.text_summary]
        if self.cell_line:
            parts.append(f"Cell line: {self.cell_line}")
        if self.key_metric and self.metric_value:
            parts.append(f"{self.key_metric}: {self.metric_value}")
        if self.outcome:
            parts.append(f"Outcome: {self.outcome}")
        return " ".join(parts)


class ManufacturingRecord(BaseModel):
    """Manufacturing / CMC record — maps to cart_manufacturing collection."""
    id: str = Field(..., max_length=100)
    text_summary: str = Field(..., max_length=2000)
    process_step: ProcessStep = ProcessStep.TRANSDUCTION
    vector_type: str = Field("lentiviral", max_length=50)
    parameter: str = Field("", max_length=100)
    parameter_value: str = Field("", max_length=50)
    target_spec: str = Field("", max_length=100, description="Acceptance criteria")
    met_spec: str = Field("", max_length=10, description="yes, no, borderline")
    batch_id: str = Field("", max_length=50)
    notes: str = Field("", max_length=1000)

    def to_embedding_text(self) -> str:
        parts = [self.text_summary]
        if self.parameter:
            parts.append(f"{self.parameter}: {self.parameter_value}")
        if self.target_spec:
            parts.append(f"Spec: {self.target_spec}")
        if self.met_spec:
            parts.append(f"Met spec: {self.met_spec}")
        return " ".join(parts)


# ═══════════════════════════════════════════════════════════════════════
# SEARCH RESULT MODELS
# ═══════════════════════════════════════════════════════════════════════


class SearchHit(BaseModel):
    """A single search result from any collection."""
    collection: str
    id: str
    score: float = Field(..., ge=0.0)
    text: str
    metadata: Dict = Field(default_factory=dict)


class CrossCollectionResult(BaseModel):
    """Merged results from multi-collection search."""
    query: str
    hits: List[SearchHit] = Field(default_factory=list)
    knowledge_context: str = ""
    total_collections_searched: int = 0
    search_time_ms: float = 0.0

    @property
    def hit_count(self) -> int:
        return len(self.hits)

    def hits_by_collection(self) -> Dict[str, List[SearchHit]]:
        grouped: Dict[str, List[SearchHit]] = {}
        for hit in self.hits:
            grouped.setdefault(hit.collection, []).append(hit)
        return grouped


class ComparativeResult(BaseModel):
    """Results from a comparative analysis query."""
    query: str
    entity_a: str
    entity_b: str
    evidence_a: CrossCollectionResult
    evidence_b: CrossCollectionResult
    comparison_context: str = ""
    total_search_time_ms: float = 0.0

    @property
    def total_hits(self) -> int:
        return self.evidence_a.hit_count + self.evidence_b.hit_count


# ═══════════════════════════════════════════════════════════════════════
# AGENT MODELS
# ═══════════════════════════════════════════════════════════════════════


class AgentQuery(BaseModel):
    """Input to the CAR-T Intelligence Agent."""
    question: str
    target_antigen: Optional[str] = None
    cart_stage: Optional[CARTStage] = None
    include_genomic: bool = True  # Also search genomic_evidence collection


class AgentResponse(BaseModel):
    """Output from the CAR-T Intelligence Agent."""
    question: str
    answer: str
    evidence: CrossCollectionResult
    knowledge_used: List[str] = Field(default_factory=list)
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
