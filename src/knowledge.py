"""CAR-T Intelligence Agent — Knowledge Graph.

Extends the Clinker pattern from rag-chat-pipeline/src/knowledge.py,
adapted for the CAR-T cell therapy domain. Contains:

1. CART_TARGETS: ~25 target antigens with clinical data
2. CART_TOXICITIES: ~8 toxicity profiles with grading/management
3. CART_MANUFACTURING: ~10 manufacturing process parameters

Author: Adam Jones
Date: February 2026
"""

from typing import Any, Dict, List, Optional


# =============================================================================
# 1. CART_TARGETS — Target antigen knowledge graph (~25 entries)
# =============================================================================

CART_TARGETS: Dict[str, Dict[str, Any]] = {
    "CD19": {
        "protein": "B-Lymphocyte Antigen CD19",
        "uniprot_id": "P15391",
        "expression": "B-cell lineage (pro-B to mature B, not plasma cells)",
        "diseases": ["B-ALL", "DLBCL", "FL", "MCL", "CLL"],
        "approved_products": [
            "Kymriah (tisagenlecleucel)",
            "Yescarta (axicabtagene ciloleucel)",
            "Tecartus (brexucabtagene autoleucel)",
            "Breyanzi (lisocabtagene maraleucel)",
        ],
        "key_trials": ["ELIANA", "ZUMA-1", "ZUMA-2", "TRANSFORM", "TRANSCEND"],
        "known_resistance": [
            "CD19 loss/mutation", "lineage switch", "trogocytosis",
            "alternative splicing (exon 2 deletion)",
        ],
        "toxicity_profile": {"CRS": "30-90%", "ICANS": "20-65%", "B_cell_aplasia": "expected"},
        "normal_tissue": "B-cells (acceptable on-target/off-tumor)",
    },
    "BCMA": {
        "protein": "B-Cell Maturation Antigen (TNFRSF17)",
        "uniprot_id": "Q02223",
        "expression": "Mature B-cells and plasma cells",
        "diseases": ["Multiple Myeloma", "Waldenstrom's Macroglobulinemia"],
        "approved_products": [
            "Abecma (idecabtagene vicleucel)",
            "Carvykti (ciltacabtagene autoleucel)",
        ],
        "key_trials": ["KarMMa", "CARTITUDE-1", "CARTITUDE-4"],
        "known_resistance": [
            "BCMA downregulation", "biallelic BCMA loss",
            "soluble BCMA shedding (gamma-secretase)",
        ],
        "toxicity_profile": {"CRS": "80-95%", "ICANS": "10-20%", "cytopenia": "common"},
        "normal_tissue": "Mature B-cells, plasma cells",
    },
    "CD22": {
        "protein": "B-cell receptor CD22 (SIGLEC-2)",
        "uniprot_id": "P20273",
        "expression": "B-cell lineage, broader than CD19",
        "diseases": ["B-ALL", "DLBCL", "Hairy cell leukemia"],
        "approved_products": [],
        "key_trials": ["NCT02315612", "NCT04088890"],
        "known_resistance": ["CD22 downregulation", "diminished site density"],
        "toxicity_profile": {"CRS": "moderate", "ICANS": "low"},
        "normal_tissue": "B-cells",
    },
    "CD20": {
        "protein": "B-lymphocyte antigen CD20 (MS4A1)",
        "uniprot_id": "P11836",
        "expression": "Pre-B to mature B-cells",
        "diseases": ["DLBCL", "FL", "MCL", "CLL"],
        "approved_products": [],
        "key_trials": ["NCT03277729"],
        "known_resistance": ["CD20 loss (rare)", "rituximab resistance"],
        "toxicity_profile": {"CRS": "moderate", "B_cell_aplasia": "expected"},
        "normal_tissue": "B-cells",
    },
    "CD30": {
        "protein": "TNFRSF8 (Ki-1 antigen)",
        "uniprot_id": "P28908",
        "expression": "Activated T/B cells, Reed-Sternberg cells",
        "diseases": ["Hodgkin Lymphoma", "ALCL", "PTCL"],
        "approved_products": [],
        "key_trials": ["RELY-30", "NCT02690545"],
        "known_resistance": ["CD30 shedding"],
        "toxicity_profile": {"CRS": "moderate"},
        "normal_tissue": "Activated lymphocytes (limited)",
    },
    "CD33": {
        "protein": "Siglec-3",
        "uniprot_id": "P20138",
        "expression": "Myeloid progenitors, AML blasts",
        "diseases": ["AML", "MDS"],
        "approved_products": [],
        "key_trials": ["NCT03126864", "NCT03971799"],
        "known_resistance": ["CD33 splice variants", "heterogeneous expression"],
        "toxicity_profile": {"CRS": "moderate", "myelosuppression": "significant"},
        "normal_tissue": "Myeloid progenitors (significant on-target/off-tumor)",
    },
    "CD38": {
        "protein": "ADP-ribosyl cyclase/cyclic ADP-ribose hydrolase 1",
        "uniprot_id": "P28907",
        "expression": "Plasma cells, activated lymphocytes",
        "diseases": ["Multiple Myeloma", "T-ALL"],
        "approved_products": [],
        "key_trials": ["NCT03464916"],
        "known_resistance": ["CD38 downregulation"],
        "toxicity_profile": {"CRS": "moderate"},
        "normal_tissue": "T-cells, NK cells, red blood cells (broad)",
    },
    "CD123": {
        "protein": "Interleukin-3 receptor alpha chain (IL3RA)",
        "uniprot_id": "P26951",
        "expression": "Myeloid progenitors, plasmacytoid DCs, AML blasts",
        "diseases": ["AML", "BPDCN", "MDS"],
        "approved_products": [],
        "key_trials": ["NCT02159495", "AMETHYST"],
        "known_resistance": ["Low/variable expression"],
        "toxicity_profile": {"CRS": "moderate-high", "myelosuppression": "significant"},
        "normal_tissue": "Normal hematopoietic progenitors",
    },
    "GD2": {
        "protein": "Disialoganglioside GD2",
        "uniprot_id": None,
        "expression": "Neuroectodermal tumors, limited normal tissue",
        "diseases": ["Neuroblastoma", "Osteosarcoma", "Melanoma", "DIPG"],
        "approved_products": [],
        "key_trials": ["NCT01953900", "NCT03635632"],
        "known_resistance": ["GD2 downregulation", "immunosuppressive TME"],
        "toxicity_profile": {"CRS": "moderate", "neurotoxicity": "pain-related"},
        "normal_tissue": "Peripheral nerves, brain (low levels)",
    },
    "HER2": {
        "protein": "Human Epidermal Growth Factor Receptor 2 (ERBB2)",
        "uniprot_id": "P04626",
        "expression": "Epithelial cells, overexpressed in tumors",
        "diseases": ["Breast Cancer", "Gastric Cancer", "Sarcoma", "GBM"],
        "approved_products": [],
        "key_trials": ["NCT00902044", "NCT03500991"],
        "known_resistance": ["HER2 heterogeneity", "solid tumor barriers"],
        "toxicity_profile": {"CRS": "moderate", "on_target_off_tumor": "cardiac/pulmonary risk"},
        "normal_tissue": "Heart, lung, GI epithelium (safety concern)",
    },
    "GPC3": {
        "protein": "Glypican-3",
        "uniprot_id": "P51654",
        "expression": "Hepatocellular carcinoma, some pediatric tumors",
        "diseases": ["HCC", "Hepatoblastoma", "Wilms tumor"],
        "approved_products": [],
        "key_trials": ["NCT02905188", "NCT03884751"],
        "known_resistance": ["Solid tumor microenvironment"],
        "toxicity_profile": {"CRS": "moderate"},
        "normal_tissue": "Fetal liver (low in adults)",
    },
    "EGFR": {
        "protein": "Epidermal Growth Factor Receptor",
        "uniprot_id": "P00533",
        "expression": "Epithelial cells, overexpressed in many solid tumors",
        "diseases": ["NSCLC", "GBM", "Head and Neck Cancer"],
        "approved_products": [],
        "key_trials": ["NCT03182816"],
        "known_resistance": ["Antigen heterogeneity", "solid tumor barriers"],
        "toxicity_profile": {"on_target_off_tumor": "skin/GI toxicity risk"},
        "normal_tissue": "Skin, GI tract, lung epithelium",
    },
    "EGFRvIII": {
        "protein": "EGFR variant III (tumor-specific mutant)",
        "uniprot_id": None,
        "expression": "GBM (20-30%), some NSCLC",
        "diseases": ["Glioblastoma"],
        "approved_products": [],
        "key_trials": ["NCT02209376", "NCT01454596"],
        "known_resistance": ["Antigen loss (heterogeneous expression)"],
        "toxicity_profile": {"CRS": "low-moderate"},
        "normal_tissue": "None (tumor-specific neoantigen)",
    },
    "Mesothelin": {
        "protein": "Mesothelin (MSLN)",
        "uniprot_id": "Q13421",
        "expression": "Mesothelial cells, overexpressed in mesothelioma/pancreatic/ovarian",
        "diseases": ["Mesothelioma", "Pancreatic Cancer", "Ovarian Cancer"],
        "approved_products": [],
        "key_trials": ["NCT02159716", "NCT03054298"],
        "known_resistance": ["Immunosuppressive TME", "antigen shedding"],
        "toxicity_profile": {"CRS": "moderate", "on_target_off_tumor": "pleuritis risk"},
        "normal_tissue": "Pleura, peritoneum, pericardium",
    },
    "Claudin18.2": {
        "protein": "Claudin 18 isoform 2 (CLDN18.2)",
        "uniprot_id": "P56856",
        "expression": "Gastric mucosa, overexpressed in gastric/pancreatic cancer",
        "diseases": ["Gastric Cancer", "Pancreatic Cancer"],
        "approved_products": [],
        "key_trials": ["NCT03874897", "CT041"],
        "known_resistance": ["Heterogeneous expression"],
        "toxicity_profile": {"CRS": "moderate", "GI_toxicity": "mucosal"},
        "normal_tissue": "Gastric epithelium (tight junctions)",
    },
    "MUC1": {
        "protein": "Mucin 1 (episialin)",
        "uniprot_id": "P15941",
        "expression": "Epithelial cells, aberrantly glycosylated in tumors",
        "diseases": ["Breast Cancer", "Pancreatic Cancer", "NSCLC"],
        "approved_products": [],
        "key_trials": ["NCT02587689"],
        "known_resistance": ["Glycosylation heterogeneity"],
        "toxicity_profile": {"on_target_off_tumor": "epithelial toxicity risk"},
        "normal_tissue": "All epithelial surfaces",
    },
    "PSMA": {
        "protein": "Prostate-Specific Membrane Antigen (FOLH1)",
        "uniprot_id": "Q04609",
        "expression": "Prostate epithelium, overexpressed in prostate cancer",
        "diseases": ["Prostate Cancer"],
        "approved_products": [],
        "key_trials": ["NCT04227275", "NCT03089203"],
        "known_resistance": ["Heterogeneous expression", "solid tumor barriers"],
        "toxicity_profile": {"CRS": "moderate"},
        "normal_tissue": "Prostate, kidney, small intestine, brain",
    },
    "ROR1": {
        "protein": "Receptor Tyrosine Kinase-Like Orphan Receptor 1",
        "uniprot_id": "Q01973",
        "expression": "Embryonic, re-expressed in CLL/MCL/TNBC",
        "diseases": ["CLL", "MCL", "TNBC", "NSCLC"],
        "approved_products": [],
        "key_trials": ["NCT02706392"],
        "known_resistance": ["Variable expression levels"],
        "toxicity_profile": {"CRS": "moderate"},
        "normal_tissue": "Minimal in adults (oncofetal antigen)",
    },
    "GPRC5D": {
        "protein": "G Protein-Coupled Receptor Class C Group 5 Member D",
        "uniprot_id": "Q9NZD1",
        "expression": "Plasma cells, hair follicles",
        "diseases": ["Multiple Myeloma"],
        "approved_products": [],
        "key_trials": ["NCT05016778"],
        "known_resistance": ["Limited data"],
        "toxicity_profile": {"CRS": "moderate", "skin_toxicity": "alopecia/nail changes"},
        "normal_tissue": "Hair follicle keratinocytes",
    },
    "IL13Ra2": {
        "protein": "Interleukin-13 Receptor Alpha 2",
        "uniprot_id": "Q14627",
        "expression": "GBM (overexpressed in >50% of tumors)",
        "diseases": ["Glioblastoma", "Medulloblastoma"],
        "approved_products": [],
        "key_trials": ["NCT02208362", "NCT04510051"],
        "known_resistance": ["Antigen heterogeneity within tumor"],
        "toxicity_profile": {"CRS": "low-moderate"},
        "normal_tissue": "Testis, minimal elsewhere",
    },
    "DLL3": {
        "protein": "Delta-Like Ligand 3",
        "uniprot_id": "Q9NYJ7",
        "expression": "Small cell lung cancer, neuroendocrine tumors",
        "diseases": ["SCLC", "Neuroendocrine tumors"],
        "approved_products": [],
        "key_trials": ["NCT05680922"],
        "known_resistance": ["Limited data"],
        "toxicity_profile": {"CRS": "moderate"},
        "normal_tissue": "Limited expression in adult tissue",
    },
    "B7-H3": {
        "protein": "CD276",
        "uniprot_id": "Q5ZPR3",
        "expression": "Broad solid tumor expression, limited normal tissue",
        "diseases": ["Neuroblastoma", "Sarcoma", "Pediatric Brain Tumors"],
        "approved_products": [],
        "key_trials": ["NCT04483778", "NCT04185038"],
        "known_resistance": ["Immunosuppressive TME"],
        "toxicity_profile": {"CRS": "moderate"},
        "normal_tissue": "Activated immune cells (limited)",
    },
    "NKG2D_ligands": {
        "protein": "NKG2D Ligands (MICA/MICB, ULBP1-6)",
        "uniprot_id": None,
        "expression": "Stress-induced on tumor cells",
        "diseases": ["AML", "Multiple Myeloma", "Various Solid Tumors"],
        "approved_products": [],
        "key_trials": ["NCT03018405"],
        "known_resistance": ["Ligand shedding", "variable expression"],
        "toxicity_profile": {"CRS": "moderate"},
        "normal_tissue": "Stress-induced (potential for autoimmunity)",
    },
    "CD7": {
        "protein": "T-cell antigen CD7",
        "uniprot_id": "P09564",
        "expression": "T-cells, NK cells, T-ALL blasts",
        "diseases": ["T-ALL", "T-lymphoma"],
        "approved_products": [],
        "key_trials": ["NCT04572308"],
        "known_resistance": ["Fratricide (T-cell target on T-cell product)"],
        "toxicity_profile": {"CRS": "moderate-high", "T_cell_aplasia": "risk"},
        "normal_tissue": "Normal T-cells and NK cells",
    },
    "CD5": {
        "protein": "Lymphocyte Antigen CD5",
        "uniprot_id": "P06127",
        "expression": "T-cells, subset of B-cells",
        "diseases": ["T-ALL", "T-lymphoma", "CLL"],
        "approved_products": [],
        "key_trials": ["NCT03081910"],
        "known_resistance": ["Fratricide"],
        "toxicity_profile": {"CRS": "moderate"},
        "normal_tissue": "Normal T-cells",
    },
}


# =============================================================================
# 2. CART_TOXICITIES — Toxicity knowledge graph (~8 profiles)
# =============================================================================

CART_TOXICITIES: Dict[str, Dict[str, Any]] = {
    "CRS": {
        "full_name": "Cytokine Release Syndrome",
        "mechanism": (
            "Massive cytokine release (IL-6, IFN-gamma, IL-2R, TNF-alpha, GM-CSF) "
            "from activated CAR-T cells and bystander monocytes/macrophages"
        ),
        "grading": {
            "1": "Fever only (>=38C)",
            "2": "Hypotension responsive to fluids; O2 by low-flow nasal cannula",
            "3": "Hypotension requiring vasopressors; O2 by high-flow or non-rebreather",
            "4": "Life-threatening; vasopressors + positive pressure ventilation",
        },
        "grading_system": "Lee 2014 / ASTCT 2019 consensus",
        "incidence": "50-95% any grade; 10-25% grade 3+",
        "timing": "Typically onset day 1-14 post-infusion, peak day 3-7",
        "management": [
            "Tocilizumab (IL-6R blockade) — first-line for grade 2+",
            "Corticosteroids (dexamethasone 10mg IV) — second-line or grade 3+",
            "Siltuximab (anti-IL-6) — if tocilizumab-refractory",
            "Anakinra (IL-1R antagonist) — emerging evidence",
            "Ruxolitinib (JAK inhibitor) — refractory cases",
            "Supportive care: fluids, vasopressors, O2",
        ],
        "biomarkers": ["Ferritin", "CRP", "IL-6", "IFN-gamma", "sIL-2R"],
        "risk_factors": [
            "High tumor burden", "High CAR-T cell dose",
            "CD28 costimulation (vs 4-1BB)", "ALL > lymphoma",
        ],
    },
    "ICANS": {
        "full_name": "Immune Effector Cell-Associated Neurotoxicity Syndrome",
        "mechanism": (
            "BBB disruption from endothelial activation by cytokines (IL-6, IFN-gamma); "
            "direct CAR-T cell CNS trafficking; cerebral edema"
        ),
        "grading": {
            "1": "ICE score 7-9 (mild impairment)",
            "2": "ICE score 3-6 (moderate impairment)",
            "3": "ICE score 0-2; any seizure; raised ICP not requiring intervention",
            "4": "Prolonged seizure; cerebral edema; coma",
        },
        "grading_system": "ASTCT 2019 consensus (ICE score)",
        "incidence": "20-65% any grade; 10-30% grade 3+",
        "timing": "Onset typically day 3-10, often after CRS peak",
        "management": [
            "Corticosteroids (dexamethasone 10mg IV q6h) — first-line",
            "Levetiracetam — seizure prophylaxis",
            "Tocilizumab — if concurrent CRS (may not help isolated ICANS)",
            "Siltuximab — if steroid-refractory",
            "ICU monitoring for grade 3+",
        ],
        "biomarkers": ["ICE score", "CSF protein", "CSF IL-6", "MRI (edema)"],
        "risk_factors": [
            "High tumor burden", "Pre-existing neurologic conditions",
            "Severe CRS", "Thrombocytopenia", "CD28 costimulation",
        ],
    },
    "B_CELL_APLASIA": {
        "full_name": "B-Cell Aplasia / Hypogammaglobulinemia",
        "mechanism": (
            "On-target, off-tumor effect of CD19/CD22-directed CAR-T cells; "
            "depletion of normal B-cell compartment leading to agammaglobulinemia"
        ),
        "grading": {
            "expected": "Anticipated pharmacodynamic effect of CD19 CAR-T",
        },
        "incidence": "Near 100% with CD19 CAR-T; duration correlates with persistence",
        "timing": "Onset with CAR-T engraftment; may persist months to years",
        "management": [
            "IVIG replacement (target IgG > 400 mg/dL)",
            "Infection prophylaxis (PJP, fungal, viral)",
            "Monitoring: quantitative immunoglobulins monthly",
            "Vaccination (live vaccines contraindicated)",
        ],
        "biomarkers": ["IgG levels", "B-cell count (CD19+ cells)", "CAR-T persistence (qPCR)"],
        "risk_factors": ["CD19 CAR-T (expected)", "Prolonged CAR-T persistence"],
    },
    "HLH_MAS": {
        "full_name": "Hemophagocytic Lymphohistiocytosis / Macrophage Activation Syndrome",
        "mechanism": (
            "Severe immune hyperactivation with uncontrolled macrophage/histiocyte "
            "proliferation and hemophagocytosis; overlaps with severe CRS"
        ),
        "grading": {
            "diagnostic_criteria": "Ferritin >10,000, cytopenias, hepatosplenomegaly, "
            "coagulopathy, elevated sIL-2R, elevated triglycerides",
        },
        "incidence": "1-5% of CAR-T recipients",
        "timing": "Typically day 5-20 post-infusion, during or after CRS",
        "management": [
            "Etoposide — severe/refractory cases",
            "Anakinra (IL-1R antagonist)",
            "Ruxolitinib (JAK inhibitor)",
            "High-dose corticosteroids",
            "Tocilizumab (if concurrent CRS)",
        ],
        "biomarkers": ["Ferritin (>10,000)", "Fibrinogen (decreased)", "Triglycerides", "sIL-2R"],
        "risk_factors": ["Severe CRS", "High tumor burden", "Prior HLH history"],
    },
    "CYTOPENIAS": {
        "full_name": "Prolonged Cytopenias",
        "mechanism": (
            "Bone marrow suppression from lymphodepletion, CRS-related inflammation, "
            "and direct CAR-T effects on hematopoietic progenitors"
        ),
        "grading": {
            "early": "Within 30 days (expected from lymphodepletion)",
            "prolonged": "Beyond 30 days (concerning)",
            "severe": "Grade 3-4 neutropenia, thrombocytopenia, or anemia >90 days",
        },
        "incidence": "30-50% prolonged cytopenias (>30 days)",
        "timing": "Onset day 0-7; prolonged in 30-50%",
        "management": [
            "G-CSF for severe neutropenia (>14 days post-infusion)",
            "Platelet/RBC transfusion support",
            "TPO-RA for prolonged thrombocytopenia",
            "Infection monitoring and prophylaxis",
        ],
        "biomarkers": ["CBC with differential", "Reticulocyte count", "Bone marrow biopsy if prolonged"],
        "risk_factors": [
            "Severe CRS", "High Flu/Cy dose", "Pre-existing cytopenias",
            "Prior transplant", "Heavy prior therapy",
        ],
    },
    "TLS": {
        "full_name": "Tumor Lysis Syndrome",
        "mechanism": "Rapid tumor cell destruction releasing intracellular contents",
        "grading": {
            "laboratory": "2+ of: uric acid, K+, phosphate elevated; Ca2+ decreased",
            "clinical": "Laboratory TLS + renal/cardiac/neurologic sequelae",
        },
        "incidence": "Rare with CAR-T (<5%); more common in high tumor burden",
        "timing": "Day 1-7 post-infusion",
        "management": [
            "Rasburicase for hyperuricemia",
            "Allopurinol prophylaxis",
            "Aggressive hydration",
            "Electrolyte monitoring and correction",
        ],
        "biomarkers": ["Uric acid", "Potassium", "Phosphate", "Calcium", "LDH", "Creatinine"],
        "risk_factors": ["High tumor burden", "ALL", "Bulky disease"],
    },
    "GVHD": {
        "full_name": "Graft-versus-Host Disease",
        "mechanism": (
            "Allogeneic CAR-T cells with intact TCR attacking host tissues; "
            "relevant primarily for donor-derived or allogeneic CAR-T products"
        ),
        "grading": {
            "acute": "Skin, GI, liver involvement (Glucksberg criteria)",
            "chronic": "Skin, mouth, liver, lung, joints",
        },
        "incidence": "Rare with autologous CAR-T; risk with allogeneic products",
        "timing": "Variable; typically >2 weeks post-infusion",
        "management": [
            "TCR knockout in allogeneic products (prevention)",
            "Corticosteroids",
            "Ruxolitinib for steroid-refractory",
            "Ibrutinib for chronic GVHD",
        ],
        "biomarkers": ["Skin biopsy", "Liver function tests", "GI biopsy"],
        "risk_factors": [
            "Allogeneic (donor-derived) CAR-T",
            "Post-transplant donor lymphocyte-derived CAR-T",
            "Intact TCR on product",
        ],
    },
    "ON_TARGET_OFF_TUMOR": {
        "full_name": "On-Target, Off-Tumor Toxicity",
        "mechanism": (
            "CAR-T cells attacking normal tissues that express the target antigen; "
            "severity depends on target expression pattern in normal tissues"
        ),
        "grading": {
            "low_risk": "B-cell aplasia with CD19 (manageable with IVIG)",
            "moderate_risk": "Skin/nail toxicity with GPRC5D; mucosal with Claudin18.2",
            "high_risk": "Cardiac/pulmonary with HER2; myelosuppression with CD33",
        },
        "examples": [
            "CD19 → B-cell aplasia (manageable)",
            "BCMA → plasma cell depletion (manageable)",
            "HER2 → cardiac toxicity (fatal case reported 2010)",
            "CD33 → myelosuppression",
            "GPRC5D → skin/nail/taste changes",
            "Claudin18.2 → GI mucosal toxicity",
        ],
        "management": [
            "Target selection: tumor-restricted antigens preferred",
            "Affinity tuning: lower-affinity CARs spare low-expression normal tissue",
            "Logic gates: AND-gate CARs requiring two antigens",
            "Safety switches: iCasp9, EGFRt for rapid elimination",
        ],
        "biomarkers": ["Target-antigen dependent"],
        "risk_factors": ["Broad normal tissue expression", "High-affinity scFv"],
    },
}


# =============================================================================
# 3. CART_MANUFACTURING — Manufacturing process knowledge (~10 entries)
# =============================================================================

CART_MANUFACTURING: Dict[str, Dict[str, Any]] = {
    "lentiviral_transduction": {
        "description": "Lentiviral vector transduction of T-cells to introduce CAR transgene",
        "typical_efficiency": "20-60%",
        "target_vcn": "0.5-5.0 copies/cell",
        "critical_parameters": [
            "Multiplicity of infection (MOI)",
            "Viral titer (TU/mL)",
            "Polybrene/RetroNectin concentration",
            "Transduction duration (16-24 hours)",
            "Spinoculation protocol",
        ],
        "failure_modes": [
            "Low viral titer", "Insertional mutagenesis risk",
            "RCL (replication-competent lentivirus) contamination",
            "Low transduction efficiency", "High VCN (safety concern)",
        ],
        "release_criteria": {
            "CAR_expression": ">10-20% CAR+ by flow cytometry",
            "VCN": "<5 copies/cell (FDA guideline)",
            "RCL": "Negative",
        },
        "products_using": ["Kymriah", "Breyanzi", "Abecma", "Carvykti"],
    },
    "retroviral_transduction": {
        "description": "Gamma-retroviral vector transduction (requires actively dividing cells)",
        "typical_efficiency": "40-80%",
        "target_vcn": "1-5 copies/cell",
        "critical_parameters": [
            "Cell activation status (must be actively dividing)",
            "MOI", "Viral titer", "RetroNectin coating",
        ],
        "failure_modes": [
            "RCR (replication-competent retrovirus) risk",
            "Requires cell division for integration",
            "Insertional mutagenesis (historical concern with MLV)",
        ],
        "release_criteria": {
            "CAR_expression": ">40% CAR+",
            "RCR": "Negative",
        },
        "products_using": ["Yescarta", "Tecartus"],
    },
    "t_cell_activation": {
        "description": "T-cell activation using anti-CD3/CD28 stimulation prior to transduction",
        "methods": [
            "Anti-CD3/CD28 magnetic beads (Dynabeads)",
            "TransAct (soluble anti-CD3/CD28)",
            "Plate-bound anti-CD3 + soluble anti-CD28",
            "Artificial APCs",
        ],
        "critical_parameters": [
            "Bead-to-cell ratio (3:1 typical for Dynabeads)",
            "Activation duration (24-72 hours)",
            "Cytokine cocktail (IL-2, IL-7, IL-15)",
            "Activation marker verification (CD25, CD69)",
        ],
        "failure_modes": [
            "Over-activation leading to exhaustion",
            "Poor activation in lymphopenic patients",
            "Contaminating tumor cells activated",
        ],
    },
    "ex_vivo_expansion": {
        "description": "Expansion of CAR-T cells to therapeutic dose (typically 1e8-1e9 cells)",
        "target_dose": "1e6 to 1e8 CAR+ cells/kg",
        "expansion_fold": "100-1000x typical",
        "duration": "7-14 days",
        "platforms": [
            "G-Rex (gas-permeable static culture)",
            "WAVE/Xuri bioreactor (rocking motion)",
            "CliniMACS Prodigy (automated closed system)",
            "Lonza Cocoon (automated)",
            "Standard culture flasks/bags",
        ],
        "critical_parameters": [
            "Cell density", "Media/feed schedule",
            "Cytokine concentration (IL-2: 50-300 IU/mL; IL-7/IL-15: 5-10 ng/mL)",
            "pH and dissolved O2", "Temperature (37C)",
        ],
        "failure_modes": [
            "Insufficient expansion (<50x)", "T-cell exhaustion from over-expansion",
            "Contamination", "Shift to effector phenotype (loss of Tscm/Tcm)",
        ],
    },
    "leukapheresis": {
        "description": "Collection of patient PBMCs via apheresis as starting material",
        "volume_processed": "2-3 blood volumes typical",
        "target_collection": ">2e9 total nucleated cells",
        "critical_parameters": [
            "Timing relative to prior therapy (washout period)",
            "Anti-coagulant (ACD-A)",
            "Flow rate and processing time",
            "T-cell purity (CD3+ percentage)",
        ],
        "failure_modes": [
            "Low T-cell count (lymphopenic patient)",
            "High circulating tumor cell contamination",
            "Poor T-cell quality (exhausted/senescent)",
            "Apheresis access issues",
        ],
        "enrichment_options": [
            "CD4/CD8 positive selection (CliniMACS)",
            "T-cell enrichment by elutriation",
            "Tumor cell depletion",
        ],
    },
    "cryopreservation": {
        "description": "Cryopreservation of final CAR-T product for storage and shipping",
        "media": "CryoStor CS10 (10% DMSO) or equivalent",
        "protocol": [
            "Controlled-rate freezing (-1C/min to -40C, -10C/min to -80C)",
            "Transfer to liquid nitrogen vapor phase (-150 to -196C)",
            "Chain of custody and temperature monitoring",
        ],
        "critical_parameters": [
            "Post-thaw viability (>70% required)",
            "Post-thaw recovery (% of pre-freeze)",
            "Post-thaw potency (functional assay)",
            "DMSO concentration",
            "Cooling rate",
        ],
        "failure_modes": [
            "Freezing too fast/slow", "Temperature excursion during shipping",
            "Thaw protocol deviation", "Low post-thaw viability",
        ],
    },
    "release_testing": {
        "description": "Quality control testing of final product before release",
        "required_tests": {
            "identity": "CAR expression by flow cytometry (Protein L or anti-idiotype)",
            "purity": "CD3+ percentage; tumor cell absence",
            "viability": ">70% viable cells (trypan blue or flow)",
            "potency": "Cytotoxicity assay or IFN-gamma secretion",
            "sterility": "USP <71> or BacT/ALERT (14-day or rapid)",
            "endotoxin": "<3.5 EU/mL (LAL or recombinant Factor C)",
            "mycoplasma": "PCR or culture-based",
            "VCN": "<5 copies/cell (qPCR for integrated transgene)",
            "RCL_RCR": "Negative for replication-competent virus",
        },
        "turnaround": "Sterility is rate-limiting (14-day incubation)",
        "failure_modes": [
            "Sterility failure", "Low viability (<70%)",
            "Low CAR expression", "Potency failure",
            "Out-of-spec VCN",
        ],
    },
    "point_of_care_manufacturing": {
        "description": "Decentralized manufacturing at or near the treatment site",
        "platforms": [
            "CliniMACS Prodigy (Miltenyi)",
            "Lonza Cocoon",
            "Ori Biotech",
        ],
        "advantages": [
            "Reduced vein-to-vein time (3-5 days vs 3-6 weeks)",
            "Eliminated shipping/cold-chain logistics",
            "Fresh (non-cryopreserved) product possible",
            "Lower manufacturing cost potential",
        ],
        "challenges": [
            "GMP compliance at multiple sites",
            "Operator training and standardization",
            "Regulatory framework for decentralized manufacturing",
            "Scale-out (not scale-up) model",
        ],
    },
    "lymphodepletion": {
        "description": "Pre-infusion conditioning chemotherapy to create immunologic space",
        "standard_regimen": "Fludarabine 30mg/m2 x3 days + Cyclophosphamide 500mg/m2 x3 days",
        "alternative_regimens": [
            "Flu 25/Cy 250 (reduced intensity)",
            "Bendamustine (CD19 products)",
            "Flu/Cy/Alemtuzumab (allogeneic products)",
        ],
        "mechanism": [
            "Depletion of regulatory T-cells",
            "Reduction of competing lymphocytes",
            "Induction of homeostatic cytokines (IL-7, IL-15)",
            "Creation of immunologic space for CAR-T expansion",
        ],
        "critical_parameters": [
            "Timing: typically day -5 to day -3 before infusion",
            "Dose adjustments for renal function",
            "ALC (absolute lymphocyte count) at infusion",
        ],
        "failure_modes": [
            "Inadequate lymphodepletion (poor CAR-T expansion)",
            "Excessive myelosuppression",
            "Infection before CAR-T infusion",
        ],
    },
    "vein_to_vein_time": {
        "description": "Total time from leukapheresis to patient infusion",
        "typical_centralized": "3-6 weeks (commercial products)",
        "typical_academic": "2-4 weeks",
        "typical_poc": "3-7 days (point-of-care)",
        "bottlenecks": [
            "Leukapheresis scheduling",
            "Shipping to manufacturing site",
            "Manufacturing slot availability",
            "Release testing (14-day sterility)",
            "Return shipping",
            "Lymphodepletion scheduling",
        ],
        "clinical_impact": [
            "Disease progression during manufacturing",
            "Need for bridging therapy",
            "Patient dropout/death before infusion",
            "30-40% of patients never receive manufactured product",
        ],
    },
}


# =============================================================================
# 4. PUBLIC API — Context retrieval functions
# =============================================================================

def get_target_context(antigen: str) -> str:
    """Get formatted knowledge context for a target antigen.

    Args:
        antigen: Target antigen name (case-insensitive, e.g. 'CD19', 'BCMA').

    Returns:
        Formatted string with target knowledge, or empty string if not found.
    """
    key = antigen.upper()
    if key not in CART_TARGETS:
        # Try case-insensitive match
        for k in CART_TARGETS:
            if k.upper() == key:
                key = k
                break
        else:
            return ""

    data = CART_TARGETS[key]
    lines = [f"## Target Antigen: {key}"]
    lines.append(f"- **Protein:** {data['protein']}")
    lines.append(f"- **Expression:** {data['expression']}")
    lines.append(f"- **Diseases:** {', '.join(data['diseases'])}")

    if data.get("approved_products"):
        lines.append(f"- **Approved Products:** {'; '.join(data['approved_products'])}")
    if data.get("key_trials"):
        lines.append(f"- **Key Trials:** {', '.join(data['key_trials'])}")
    if data.get("known_resistance"):
        lines.append(f"- **Resistance Mechanisms:** {'; '.join(data['known_resistance'])}")
    if data.get("toxicity_profile"):
        tox = ", ".join(f"{k}: {v}" for k, v in data["toxicity_profile"].items())
        lines.append(f"- **Toxicity Profile:** {tox}")
    if data.get("normal_tissue"):
        lines.append(f"- **Normal Tissue Expression:** {data['normal_tissue']}")

    return "\n".join(lines)


def get_toxicity_context(toxicity: str) -> str:
    """Get formatted knowledge context for a toxicity profile.

    Args:
        toxicity: Toxicity identifier (e.g. 'CRS', 'ICANS', 'HLH_MAS').

    Returns:
        Formatted string with toxicity knowledge, or empty string if not found.
    """
    key = toxicity.upper()
    if key not in CART_TOXICITIES:
        for k in CART_TOXICITIES:
            if k.upper() == key:
                key = k
                break
        else:
            return ""

    data = CART_TOXICITIES[key]
    lines = [f"## Toxicity: {data['full_name']} ({key})"]
    lines.append(f"- **Mechanism:** {data['mechanism']}")
    lines.append(f"- **Incidence:** {data.get('incidence', 'Variable')}")
    lines.append(f"- **Timing:** {data.get('timing', 'Variable')}")

    if data.get("management"):
        lines.append("- **Management:**")
        for m in data["management"]:
            lines.append(f"  - {m}")
    if data.get("biomarkers"):
        lines.append(f"- **Biomarkers:** {', '.join(data['biomarkers'])}")
    if data.get("risk_factors"):
        lines.append(f"- **Risk Factors:** {'; '.join(data['risk_factors'])}")

    return "\n".join(lines)


def get_manufacturing_context(process: str) -> str:
    """Get formatted knowledge context for a manufacturing process.

    Args:
        process: Process identifier (e.g. 'lentiviral_transduction', 'expansion').

    Returns:
        Formatted string with manufacturing knowledge, or empty string.
    """
    key = process.lower().replace(" ", "_")
    if key not in CART_MANUFACTURING:
        for k in CART_MANUFACTURING:
            if key in k.lower():
                key = k
                break
        else:
            return ""

    data = CART_MANUFACTURING[key]
    lines = [f"## Manufacturing: {key.replace('_', ' ').title()}"]
    lines.append(f"- **Description:** {data['description']}")

    for field in ["typical_efficiency", "target_vcn", "target_dose",
                  "expansion_fold", "duration", "standard_regimen"]:
        if field in data:
            label = field.replace("_", " ").title()
            lines.append(f"- **{label}:** {data[field]}")

    if data.get("critical_parameters"):
        lines.append("- **Critical Parameters:**")
        for p in data["critical_parameters"]:
            lines.append(f"  - {p}")
    if data.get("failure_modes"):
        lines.append("- **Failure Modes:**")
        for f in data["failure_modes"]:
            lines.append(f"  - {f}")

    return "\n".join(lines)


def get_all_context_for_query(query: str) -> str:
    """Extract all relevant knowledge context from a query string.

    Scans the query for mentions of target antigens, toxicities, and
    manufacturing terms, returning combined context.

    Args:
        query: User question about CAR-T therapy.

    Returns:
        Combined knowledge context string.
    """
    query_upper = query.upper()
    query_lower = query.lower()
    sections = []

    # Check targets
    for antigen in CART_TARGETS:
        if antigen.upper() in query_upper:
            ctx = get_target_context(antigen)
            if ctx:
                sections.append(ctx)

    # Check toxicities
    tox_aliases = {
        "CRS": ["CRS", "CYTOKINE RELEASE"],
        "ICANS": ["ICANS", "NEUROTOXICITY", "NEUROLOGIC"],
        "B_CELL_APLASIA": ["B-CELL APLASIA", "HYPOGAMMAGLOBULINEMIA", "AGAMMAGLOBULINEMIA"],
        "HLH_MAS": ["HLH", "MAS", "HEMOPHAGOCYTIC"],
        "CYTOPENIAS": ["CYTOPENIA", "NEUTROPENIA", "THROMBOCYTOPENIA"],
        "TLS": ["TUMOR LYSIS"],
        "GVHD": ["GVHD", "GRAFT-VERSUS-HOST"],
        "ON_TARGET_OFF_TUMOR": ["ON-TARGET OFF-TUMOR", "ON TARGET OFF TUMOR"],
    }
    for tox_id, aliases in tox_aliases.items():
        if any(a in query_upper for a in aliases):
            ctx = get_toxicity_context(tox_id)
            if ctx:
                sections.append(ctx)

    # Check manufacturing
    mfg_keywords = {
        "lentiviral_transduction": ["TRANSDUCTION", "LENTIVIRAL", "LENTIVIRUS"],
        "retroviral_transduction": ["RETROVIRAL", "RETROVIRUS", "GAMMA-RETROVIRAL"],
        "t_cell_activation": ["ACTIVATION", "ANTI-CD3", "DYNABEADS"],
        "ex_vivo_expansion": ["EXPANSION", "EX VIVO", "BIOREACTOR", "G-REX"],
        "leukapheresis": ["LEUKAPHERESIS", "APHERESIS"],
        "cryopreservation": ["CRYOPRESERVATION", "CRYO", "THAW"],
        "release_testing": ["RELEASE TESTING", "QC", "POTENCY ASSAY"],
        "point_of_care_manufacturing": ["POINT OF CARE", "POC MANUFACTURING", "DECENTRALIZED"],
        "lymphodepletion": ["LYMPHODEPLETION", "FLUDARABINE", "CYCLOPHOSPHAMIDE"],
        "vein_to_vein_time": ["VEIN-TO-VEIN", "TURNAROUND", "MANUFACTURING TIME"],
    }
    for proc_id, keywords in mfg_keywords.items():
        if any(kw in query_upper for kw in keywords):
            ctx = get_manufacturing_context(proc_id)
            if ctx:
                sections.append(ctx)

    # Check biomarkers
    for key in CART_BIOMARKERS:
        aliases = [key, CART_BIOMARKERS[key]["full_name"].lower()]
        if any(a.lower() in query_lower for a in aliases):
            ctx = get_biomarker_context(key)
            if ctx:
                sections.append(ctx)

    # Check regulatory (product names)
    for key in CART_REGULATORY:
        if key.lower() in query_lower or CART_REGULATORY[key]["generic_name"].lower() in query_lower:
            ctx = get_regulatory_context(key)
            if ctx:
                sections.append(ctx)

    # Check immunogenicity
    immuno_keywords = {
        "murine_scfv_immunogenicity": ["IMMUNOGENICITY", "ADA", "ANTI-DRUG ANTIBOD", "HAMA", "ANTI-MURINE"],
        "humanization_strategies": ["HUMANIZATION", "HUMANIZED", "CDR GRAFTING", "DEIMMUNIZ", "FRAMEWORK SHUFFL"],
        "ada_clinical_impact": ["ADA TITER", "NEUTRALIZING ANTIBOD", "ANTI-CAR ANTIBOD"],
        "hla_restricted_epitopes": ["HLA", "MHC", "T-CELL EPITOPE", "NETCHMII", "NETMHC", "EPIMATRIX"],
        "immunogenicity_testing": ["IMMUNOGENICITY TEST", "ELISPOT", "DC-T CELL", "MAPPS"],
        "allogeneic_hla_considerations": ["ALLOGENEIC HLA", "B2M KNOCKOUT", "HLA-E", "TRAC KNOCKOUT", "OFF-THE-SHELF"],
    }
    for immuno_id, keywords in immuno_keywords.items():
        if any(kw in query_upper for kw in keywords):
            ctx = get_immunogenicity_context(immuno_id)
            if ctx:
                sections.append(ctx)

    if not sections:
        return ""

    return "\n\n".join(sections)


def get_knowledge_stats() -> Dict[str, int]:
    """Return statistics about the CAR-T knowledge graph."""
    return {
        "target_antigens": len(CART_TARGETS),
        "targets_with_approved_products": sum(
            1 for t in CART_TARGETS.values() if t.get("approved_products")
        ),
        "toxicity_profiles": len(CART_TOXICITIES),
        "manufacturing_processes": len(CART_MANUFACTURING),
        "biomarkers": len(CART_BIOMARKERS),
        "regulatory_products": len(CART_REGULATORY),
        "immunogenicity_topics": len(CART_IMMUNOGENICITY),
    }


# =============================================================================
# 4. CART_BIOMARKERS — Biomarker knowledge graph (~15 entries)
# =============================================================================

CART_BIOMARKERS: Dict[str, Dict[str, Any]] = {
    "ferritin": {
        "full_name": "Serum Ferritin",
        "type": "predictive",
        "assay_method": "Immunoassay (ELISA/CLIA)",
        "clinical_cutoff": ">500 mg/L pre-infusion",
        "predictive_value": "Elevated baseline ferritin (>500 mg/L) predicts grade 3+ CRS with PPV 72%",
        "associated_outcome": "CRS severity",
        "evidence_level": "validated",
        "key_references": ["PMID:29385376", "PMID:30409105"],
    },
    "crp": {
        "full_name": "C-Reactive Protein",
        "type": "predictive",
        "assay_method": "Immunoturbidimetry",
        "clinical_cutoff": ">200 mg/L within 72h post-infusion",
        "predictive_value": "Peak CRP >200 mg/L within 72h has 85% sensitivity for grade 3+ CRS",
        "associated_outcome": "CRS severity and timing",
        "evidence_level": "validated",
        "key_references": ["PMID:29385376", "PMID:31533922"],
    },
    "il6": {
        "full_name": "Interleukin-6",
        "type": "pharmacodynamic",
        "assay_method": "ELISA / Luminex multiplex",
        "clinical_cutoff": ">1000 pg/mL",
        "predictive_value": "Peak IL-6 correlates with CRS grade (r=0.82). Target of tocilizumab therapy",
        "associated_outcome": "CRS grade, tocilizumab response",
        "evidence_level": "validated",
        "key_references": ["PMID:29385376", "PMID:30409105"],
    },
    "sil2r": {
        "full_name": "Soluble IL-2 Receptor (sCD25)",
        "type": "pharmacodynamic",
        "assay_method": "ELISA",
        "clinical_cutoff": ">5000 pg/mL",
        "predictive_value": "Elevated sIL-2R correlates with T-cell activation and CRS/HLH risk",
        "associated_outcome": "CRS, HLH/MAS",
        "evidence_level": "validated",
        "key_references": ["PMID:30409105"],
    },
    "car_t_expansion": {
        "full_name": "CAR-T Cell Peak Expansion (Cmax)",
        "type": "pharmacodynamic",
        "assay_method": "qPCR (transgene copies/μg DNA) or flow cytometry",
        "clinical_cutoff": "Peak >50,000 copies/μg DNA (varies by product)",
        "predictive_value": "Higher peak expansion correlates with response (OR 3.2) but also CRS risk",
        "associated_outcome": "Response rate, CRS, durability",
        "evidence_level": "validated",
        "key_references": ["PMID:29385376", "PMID:28687837"],
    },
    "tcm_percentage": {
        "full_name": "Central Memory T-cell Percentage (Tcm%)",
        "type": "predictive",
        "assay_method": "Flow cytometry (CD45RA-/CCR7+/CD62L+)",
        "clinical_cutoff": ">40% Tcm in apheresis product",
        "predictive_value": "Higher Tcm% predicts better expansion, persistence, and response",
        "associated_outcome": "Manufacturing success, clinical response",
        "evidence_level": "emerging",
        "key_references": ["PMID:31040380"],
    },
    "cd4_cd8_ratio": {
        "full_name": "CD4:CD8 T-cell Ratio",
        "type": "predictive",
        "assay_method": "Flow cytometry",
        "clinical_cutoff": "Optimal 1:1 (defined CD4/CD8 composition in product)",
        "predictive_value": "Balanced CD4:CD8 ratio in product associated with improved persistence (lisocabtagene)",
        "associated_outcome": "CAR-T persistence, reduced toxicity",
        "evidence_level": "emerging",
        "key_references": ["PMID:33888460"],
    },
    "ldh": {
        "full_name": "Lactate Dehydrogenase",
        "type": "prognostic",
        "assay_method": "Enzymatic assay (serum)",
        "clinical_cutoff": ">ULN (upper limit of normal) at lymphodepletion",
        "predictive_value": "Elevated LDH (>ULN) pre-lymphodepletion predicts inferior PFS (HR 1.8)",
        "associated_outcome": "PFS, OS, tumor burden",
        "evidence_level": "validated",
        "key_references": ["PMID:28687837", "PMID:31040380"],
    },
    "pd1": {
        "full_name": "Programmed Death-1 (PD-1/CD279)",
        "type": "resistance",
        "assay_method": "Flow cytometry / IHC",
        "clinical_cutoff": ">30% PD-1+ on CAR-T cells",
        "predictive_value": "High PD-1 expression on infused product predicts shorter persistence and inferior response",
        "associated_outcome": "T-cell exhaustion, relapse",
        "evidence_level": "emerging",
        "key_references": ["PMID:31040380"],
    },
    "lag3": {
        "full_name": "Lymphocyte Activation Gene-3 (LAG-3/CD223)",
        "type": "resistance",
        "assay_method": "Flow cytometry",
        "clinical_cutoff": ">20% LAG-3+ on CAR-T cells",
        "predictive_value": "Co-expression with PD-1 and TIM-3 marks terminally exhausted T-cells",
        "associated_outcome": "T-cell exhaustion, reduced efficacy",
        "evidence_level": "emerging",
        "key_references": ["PMID:31040380"],
    },
    "tim3": {
        "full_name": "T-cell Immunoglobulin and Mucin-3 (TIM-3)",
        "type": "resistance",
        "assay_method": "Flow cytometry",
        "clinical_cutoff": ">25% TIM-3+ on CAR-T cells",
        "predictive_value": "TIM-3+ CAR-T cells show reduced cytokine production and cytotoxicity",
        "associated_outcome": "T-cell exhaustion, functional impairment",
        "evidence_level": "emerging",
        "key_references": ["PMID:31040380"],
    },
    "mrd_flow": {
        "full_name": "Minimal Residual Disease by Flow Cytometry",
        "type": "monitoring",
        "assay_method": "Multiparameter flow cytometry (≥8-color)",
        "clinical_cutoff": "<10^-4 (0.01%)",
        "predictive_value": "MRD negativity at day 28 post-infusion predicts PFS (HR 0.3 vs MRD+)",
        "associated_outcome": "PFS, relapse-free survival",
        "evidence_level": "validated",
        "key_references": ["PMID:29385376", "PMID:30409105"],
    },
    "ctdna": {
        "full_name": "Circulating Tumor DNA",
        "type": "monitoring",
        "assay_method": "NGS panel / ddPCR",
        "clinical_cutoff": "2-log reduction from baseline by day 28",
        "predictive_value": "Early ctDNA clearance correlates with durable CR in DLBCL post-CAR-T",
        "associated_outcome": "Response durability, early relapse detection",
        "evidence_level": "emerging",
        "key_references": ["PMID:34516408"],
    },
    "sbcma": {
        "full_name": "Soluble BCMA (sBCMA)",
        "type": "resistance",
        "assay_method": "ELISA",
        "clinical_cutoff": ">40 ng/mL baseline",
        "predictive_value": "High sBCMA acts as decoy, sequestering CAR-T cells and reducing on-target activity",
        "associated_outcome": "BCMA CAR-T response, resistance mechanism",
        "evidence_level": "emerging",
        "key_references": ["PMID:33657416"],
    },
    "ifn_gamma": {
        "full_name": "Interferon-gamma (IFN-γ)",
        "type": "pharmacodynamic",
        "assay_method": "ELISA / Luminex / ELISpot",
        "clinical_cutoff": ">500 pg/mL at day 7",
        "predictive_value": "Peak IFN-γ correlates with CAR-T effector function and antitumor activity",
        "associated_outcome": "Antitumor response, immune activation",
        "evidence_level": "validated",
        "key_references": ["PMID:28687837"],
    },
}


# =============================================================================
# 5. CART_REGULATORY — Regulatory knowledge graph (~6 FDA-approved products)
# =============================================================================

CART_REGULATORY: Dict[str, Dict[str, Any]] = {
    "Kymriah": {
        "generic_name": "tisagenlecleucel",
        "manufacturer": "Novartis",
        "initial_approval": "2017-08-30",
        "initial_indication": "Pediatric/young adult r/r B-ALL",
        "pivotal_trial": "ELIANA (NCT02435849)",
        "designations": ["Breakthrough Therapy (2014)", "RMAT (not applicable, pre-RMAT)"],
        "subsequent_approvals": [
            {"date": "2018-05-01", "indication": "Adult r/r DLBCL (2+ prior lines)", "trial": "JULIET"},
            {"date": "2022-05-27", "indication": "Adult r/r FL (3+ prior lines)", "trial": "ELARA"},
        ],
        "rems": "CAR-T REMS: CRS and neurological toxicity management certification required",
        "post_marketing": ["15-year follow-up study for secondary malignancies", "RCR testing protocol"],
        "ema_approval": "2018-08-23",
    },
    "Yescarta": {
        "generic_name": "axicabtagene ciloleucel",
        "manufacturer": "Kite/Gilead",
        "initial_approval": "2017-10-18",
        "initial_indication": "Adult r/r LBCL (2+ prior lines)",
        "pivotal_trial": "ZUMA-1 (NCT02348216)",
        "designations": ["Breakthrough Therapy (2015)", "Priority Review"],
        "subsequent_approvals": [
            {"date": "2021-03-05", "indication": "Adult r/r LBCL (2L after 1st-line chemoimmunotherapy)", "trial": "ZUMA-7"},
            {"date": "2024-03-15", "indication": "Adult r/r FL (3+ prior lines)", "trial": "ZUMA-5"},
        ],
        "rems": "CAR-T REMS: Shared REMS program with other CAR-T products",
        "post_marketing": ["15-year follow-up study", "Post-marketing observational study"],
        "ema_approval": "2018-08-27",
    },
    "Tecartus": {
        "generic_name": "brexucabtagene autoleucel",
        "manufacturer": "Kite/Gilead",
        "initial_approval": "2020-07-24",
        "initial_indication": "Adult r/r MCL",
        "pivotal_trial": "ZUMA-2 (NCT02601313)",
        "designations": ["Breakthrough Therapy (2018)", "Priority Review", "Orphan Drug"],
        "subsequent_approvals": [
            {"date": "2021-10-01", "indication": "Adult r/r B-ALL", "trial": "ZUMA-3"},
        ],
        "rems": "CAR-T REMS",
        "post_marketing": ["15-year follow-up study"],
        "ema_approval": "2020-12-14",
    },
    "Breyanzi": {
        "generic_name": "lisocabtagene maraleucel",
        "manufacturer": "Bristol Myers Squibb/Juno",
        "initial_approval": "2021-02-05",
        "initial_indication": "Adult r/r LBCL (2+ prior lines)",
        "pivotal_trial": "TRANSCEND (NCT02631044)",
        "designations": ["Breakthrough Therapy (2016)", "RMAT (2018)", "Priority Review"],
        "subsequent_approvals": [
            {"date": "2022-06-24", "indication": "Adult r/r LBCL (2L)", "trial": "TRANSFORM"},
        ],
        "rems": "CAR-T REMS",
        "post_marketing": ["15-year follow-up study", "TRANSCEND WORLD long-term follow-up"],
        "ema_approval": "2022-04-04",
    },
    "Abecma": {
        "generic_name": "idecabtagene vicleucel",
        "manufacturer": "Bristol Myers Squibb/bluebird bio",
        "initial_approval": "2021-03-26",
        "initial_indication": "Adult r/r multiple myeloma (4+ prior lines incl. PI, IMiD, anti-CD38)",
        "pivotal_trial": "KarMMa (NCT03361748)",
        "designations": ["Breakthrough Therapy (2017)", "Priority Review", "Orphan Drug"],
        "subsequent_approvals": [],
        "rems": "CAR-T REMS",
        "post_marketing": ["KarMMa-3 confirmatory trial", "15-year follow-up study"],
        "ema_approval": "2021-08-18",
    },
    "Carvykti": {
        "generic_name": "ciltacabtagene autoleucel",
        "manufacturer": "Janssen/Legend Biotech",
        "initial_approval": "2022-02-28",
        "initial_indication": "Adult r/r multiple myeloma (4+ prior lines incl. PI, IMiD, anti-CD38)",
        "pivotal_trial": "CARTITUDE-1 (NCT03548207)",
        "designations": ["Breakthrough Therapy (2019)", "Priority Review", "Orphan Drug"],
        "subsequent_approvals": [
            {"date": "2024-04-04", "indication": "Adult r/r MM (1-3 prior lines incl. PI and IMiD)", "trial": "CARTITUDE-4"},
        ],
        "rems": "CAR-T REMS",
        "post_marketing": ["CARTITUDE-2 expansion cohorts", "15-year follow-up study"],
        "ema_approval": "2022-05-26",
    },
}


# ═══════════════════════════════════════════════════════════════════════
# COMPARATIVE ANALYSIS — Entity Resolution
# ═══════════════════════════════════════════════════════════════════════

ENTITY_ALIASES: Dict[str, Dict[str, str]] = {
    # FDA-approved products → canonical name + target antigen
    "KYMRIAH": {"type": "product", "canonical": "Kymriah (tisagenlecleucel)", "target": "CD19"},
    "TISAGENLECLEUCEL": {"type": "product", "canonical": "Kymriah (tisagenlecleucel)", "target": "CD19"},
    "YESCARTA": {"type": "product", "canonical": "Yescarta (axicabtagene ciloleucel)", "target": "CD19"},
    "AXICABTAGENE": {"type": "product", "canonical": "Yescarta (axicabtagene ciloleucel)", "target": "CD19"},
    "TECARTUS": {"type": "product", "canonical": "Tecartus (brexucabtagene autoleucel)", "target": "CD19"},
    "BREXUCABTAGENE": {"type": "product", "canonical": "Tecartus (brexucabtagene autoleucel)", "target": "CD19"},
    "BREYANZI": {"type": "product", "canonical": "Breyanzi (lisocabtagene maraleucel)", "target": "CD19"},
    "LISOCABTAGENE": {"type": "product", "canonical": "Breyanzi (lisocabtagene maraleucel)", "target": "CD19"},
    "ABECMA": {"type": "product", "canonical": "Abecma (idecabtagene vicleucel)", "target": "BCMA"},
    "IDECABTAGENE": {"type": "product", "canonical": "Abecma (idecabtagene vicleucel)", "target": "BCMA"},
    "CARVYKTI": {"type": "product", "canonical": "Carvykti (ciltacabtagene autoleucel)", "target": "BCMA"},
    "CILTACABTAGENE": {"type": "product", "canonical": "Carvykti (ciltacabtagene autoleucel)", "target": "BCMA"},
    # Costimulatory domains
    "4-1BB": {"type": "costimulatory", "canonical": "4-1BB (CD137)"},
    "CD137": {"type": "costimulatory", "canonical": "4-1BB (CD137)"},
    "CD28": {"type": "costimulatory", "canonical": "CD28"},
    "OX40": {"type": "costimulatory", "canonical": "OX40 (CD134)"},
    "ICOS": {"type": "costimulatory", "canonical": "ICOS"},
    # Vector types
    "LENTIVIRAL": {"type": "manufacturing", "canonical": "lentiviral_transduction"},
    "RETROVIRAL": {"type": "manufacturing", "canonical": "retroviral_transduction"},
    # Biomarker aliases
    "FERRITIN": {"type": "biomarker", "canonical": "ferritin"},
    "CRP": {"type": "biomarker", "canonical": "crp"},
    "IL-6": {"type": "biomarker", "canonical": "il6"},
    "IL6": {"type": "biomarker", "canonical": "il6"},
    "PD-1": {"type": "biomarker", "canonical": "pd1"},
    "PD1": {"type": "biomarker", "canonical": "pd1"},
    "LAG-3": {"type": "biomarker", "canonical": "lag3"},
    "LAG3": {"type": "biomarker", "canonical": "lag3"},
    "TIM-3": {"type": "biomarker", "canonical": "tim3"},
    "TIM3": {"type": "biomarker", "canonical": "tim3"},
    "MRD": {"type": "biomarker", "canonical": "mrd_flow"},
    "SBCMA": {"type": "biomarker", "canonical": "sbcma"},
    "CTDNA": {"type": "biomarker", "canonical": "ctdna"},
    # Immunogenicity aliases
    "ADA": {"type": "immunogenicity", "canonical": "murine_scfv_immunogenicity"},
    "ANTI-DRUG ANTIBODY": {"type": "immunogenicity", "canonical": "murine_scfv_immunogenicity"},
    "HAMA": {"type": "immunogenicity", "canonical": "murine_scfv_immunogenicity"},
    "HUMANIZATION": {"type": "immunogenicity", "canonical": "humanization_strategies"},
    "DEIMMUNIZATION": {"type": "immunogenicity", "canonical": "humanization_strategies"},
    "HLA": {"type": "immunogenicity", "canonical": "hla_restricted_epitopes"},
    "MHC": {"type": "immunogenicity", "canonical": "hla_restricted_epitopes"},
}


def get_biomarker_context(biomarker: str) -> str:
    """Return formatted knowledge for a biomarker."""
    key = biomarker.lower().replace("-", "").replace(" ", "_")
    data = CART_BIOMARKERS.get(key)
    if not data:
        return ""
    lines = [
        f"Biomarker: {data['full_name']}",
        f"  Type: {data['type']}",
        f"  Assay Method: {data['assay_method']}",
        f"  Clinical Cutoff: {data['clinical_cutoff']}",
        f"  Predictive Value: {data['predictive_value']}",
        f"  Associated Outcome: {data['associated_outcome']}",
        f"  Evidence Level: {data['evidence_level']}",
    ]
    return "\n".join(lines)


def get_regulatory_context(product: str) -> str:
    """Return formatted regulatory knowledge for a product."""
    data = CART_REGULATORY.get(product)
    if not data:
        # Try matching by generic name
        for k, v in CART_REGULATORY.items():
            if v.get("generic_name", "").lower() in product.lower() or product.lower() in k.lower():
                data = v
                break
    if not data:
        return ""
    lines = [
        f"Regulatory Profile: {product}",
        f"  Generic Name: {data['generic_name']}",
        f"  Manufacturer: {data['manufacturer']}",
        f"  Initial FDA Approval: {data['initial_approval']}",
        f"  Initial Indication: {data['initial_indication']}",
        f"  Pivotal Trial: {data['pivotal_trial']}",
        f"  Designations: {', '.join(data['designations'])}",
        f"  REMS: {data['rems']}",
    ]
    if data.get("subsequent_approvals"):
        lines.append("  Subsequent Approvals:")
        for sa in data["subsequent_approvals"]:
            lines.append(f"    - {sa['date']}: {sa['indication']} ({sa['trial']})")
    if data.get("ema_approval"):
        lines.append(f"  EMA Approval: {data['ema_approval']}")
    return "\n".join(lines)


# =============================================================================
# 6. CART_IMMUNOGENICITY — HLA & Immunogenicity knowledge (~6 entries)
# =============================================================================

CART_IMMUNOGENICITY: Dict[str, Dict[str, Any]] = {
    "murine_scfv_immunogenicity": {
        "topic": "Murine scFv Immunogenicity in CAR-T",
        "description": (
            "Murine-derived scFvs (e.g., FMC63 in Kymriah/Yescarta/Breyanzi, "
            "11D5-3 in Abecma) elicit anti-drug antibodies (ADA) in ~3-5% of patients. "
            "ADA can reduce CAR-T persistence, impair re-dosing efficacy, and rarely "
            "cause infusion reactions. ELIANA trial: ~5% ADA against FMC63."
        ),
        "key_constructs": ["FMC63 (murine anti-CD19)", "SJ25C1 (murine anti-CD19)",
                           "11D5-3 (murine anti-BCMA)", "14G2a (murine anti-GD2)"],
        "ada_incidence": "3-8% for murine scFvs, <1% for humanized, <0.5% for fully human",
        "clinical_impact": "Reduced persistence at 12 months; impaired re-dosing response",
        "management": "Monitor ADA titers; consider humanized alternatives for re-dosing",
    },
    "humanization_strategies": {
        "topic": "scFv Humanization Strategies for CAR-T",
        "description": (
            "Humanization reduces immunogenicity by grafting murine CDRs onto human "
            "framework regions. Key approaches: (1) CDR grafting onto closest human "
            "germline (VH3-23, VK1-39 are common acceptors), (2) framework back-mutations "
            "for affinity retention, (3) deimmunization via T-cell epitope removal, "
            "(4) fully human binders from phage/yeast display libraries."
        ),
        "methods": ["CDR grafting", "Framework shuffling", "Back-mutation optimization",
                     "Deimmunization (epitope deletion)", "Fully human library selection"],
        "tools": ["EpiMatrix/EpiVax", "NetMHCIIpan", "IEDB", "AbDesigner", "Rosetta"],
        "tradeoffs": "Humanization may reduce affinity (1.5-5x); back-mutations partially restore",
    },
    "ada_clinical_impact": {
        "topic": "Clinical Impact of Anti-Drug Antibodies on CAR-T",
        "description": (
            "ADA against CAR constructs can: (1) neutralize CAR-T cells (blocking "
            "scFv-antigen interaction), (2) accelerate clearance via Fc-mediated "
            "opsonization, (3) cause infusion reactions on re-dosing, (4) activate "
            "complement-dependent cytotoxicity of CAR-T cells. High-titer ADA "
            "(>1:1000) correlates with 40% reduced persistence at 6 months."
        ),
        "risk_factors": ["Murine scFv origin", "Repeated dosing", "Non-human hinge/spacer",
                         "High transgene immunogenicity", "Intact immune system (less lymphodepletion)"],
        "monitoring": "ELISA (screening) → confirmatory (drug tolerance) → titer → neutralizing assay",
    },
    "hla_restricted_epitopes": {
        "topic": "HLA-Restricted T-Cell Epitopes in CAR Constructs",
        "description": (
            "MHC class II presentation of CAR-derived peptides drives CD4 T-cell help "
            "for ADA production. HLA-DRB1*04:01 and HLA-DRB1*15:01 alleles show "
            "highest risk for FMC63-derived epitope presentation. Computational tools "
            "(NetMHCIIpan, EpiMatrix) predict 15-20 potential epitopes in murine scFvs "
            "vs 2-5 in humanized versions."
        ),
        "high_risk_alleles": ["HLA-DRB1*04:01", "HLA-DRB1*15:01", "HLA-DRB1*07:01"],
        "prediction_tools": ["NetMHCIIpan-4.0", "EpiMatrix", "IEDB MHC-II binding"],
        "epitope_counts": {"murine_FMC63": 18, "humanized_FMC63": 4, "fully_human_CD19": 2},
    },
    "immunogenicity_testing": {
        "topic": "Immunogenicity Testing Paradigm for CAR-T",
        "description": (
            "Three-tier testing: (1) In silico: EpiMatrix/NetMHCIIpan T-cell epitope "
            "prediction, aggregatrix score for epitope clustering. (2) In vitro: DC-T cell "
            "co-culture, ELISpot for IL-4/IFN-γ, MAPPs (MHC-associated peptide proteomics). "
            "(3) Clinical: ADA sampling at baseline, day 28, months 3/6/12; tiered approach "
            "per FDA 2024 guidance."
        ),
        "in_silico": ["EpiMatrix", "NetMHCIIpan", "IEDB", "iTope/TCED"],
        "in_vitro": ["DC-T cell co-culture", "ELISpot (IL-4, IFN-γ)", "MAPPs assay"],
        "clinical": ["Screening ELISA", "Confirmatory assay", "Titer", "Neutralizing antibody"],
        "fda_guidance": "FDA 2024: Immunogenicity Testing of Therapeutic Protein Products",
    },
    "allogeneic_hla_considerations": {
        "topic": "HLA Considerations in Allogeneic CAR-T",
        "description": (
            "Allogeneic (off-the-shelf) CAR-T requires HLA management to prevent "
            "graft-versus-host disease (GvHD) and host-versus-graft rejection. "
            "Strategies: (1) TRAC/TRBC knockout (TALEN or CRISPR) eliminates GvHD, "
            "(2) B2M knockout removes MHC-I to evade host CD8 T-cells, (3) HLA-E "
            "overexpression (B2M-HLA-E fusion) prevents NK cell lysis of MHC-I-negative "
            "cells, (4) CD52 knockout enables alemtuzumab-based lymphodepletion selectivity."
        ),
        "gene_edits": ["TRAC KO (prevents GvHD)", "B2M KO (evades CD8 rejection)",
                        "HLA-E knock-in (NK evasion)", "CD52 KO (alemtuzumab resistance)"],
        "platforms": ["TALEN (Cellectis UCART)", "CRISPR/Cas9 (CRISPR Therapeutics CTX110)",
                      "Base editing (Beam BEAM-201)", "iPSC-derived (Fate FT819)"],
    },
}


def get_immunogenicity_context(topic: str) -> str:
    """Return formatted knowledge for an immunogenicity topic.

    Args:
        topic: Topic key or keyword (e.g. 'humanization', 'ada', 'hla').

    Returns:
        Formatted string with immunogenicity knowledge, or empty string.
    """
    key = topic.lower().replace("-", "_").replace(" ", "_")
    data = CART_IMMUNOGENICITY.get(key)
    if not data:
        # Try partial match
        for k in CART_IMMUNOGENICITY:
            if key in k.lower():
                data = CART_IMMUNOGENICITY[k]
                break
    if not data:
        return ""

    lines = [f"## Immunogenicity: {data['topic']}", f"- {data['description']}"]
    for field in ["ada_incidence", "clinical_impact", "management", "tradeoffs",
                  "fda_guidance"]:
        if field in data:
            label = field.replace("_", " ").title()
            lines.append(f"- **{label}:** {data[field]}")
    return "\n".join(lines)


def resolve_comparison_entity(text: str) -> Optional[Dict[str, str]]:
    """Resolve a raw text string to a known CAR-T entity for comparison.

    Checks target antigens, product aliases, toxicities, and manufacturing
    processes in priority order.

    Returns:
        Dict with 'type', 'canonical', and optionally 'target' keys,
        or None if the entity is not recognized.
    """
    cleaned = text.strip().upper()

    # 1. Target antigens (exact match)
    for key in CART_TARGETS:
        if key.upper() == cleaned:
            return {"type": "target", "canonical": key, "target": key}

    # 2. Product / alias table
    if cleaned in ENTITY_ALIASES:
        return dict(ENTITY_ALIASES[cleaned])

    # 3. Toxicities
    for key in CART_TOXICITIES:
        if key.upper() == cleaned or key.upper().replace("_", " ") == cleaned:
            return {"type": "toxicity", "canonical": key}

    # 4. Manufacturing processes (fuzzy substring)
    for key in CART_MANUFACTURING:
        if cleaned.replace(" ", "_").lower() in key.lower() or key.lower().startswith(cleaned.lower()):
            return {"type": "manufacturing", "canonical": key}

    # 5. Biomarkers
    for key in CART_BIOMARKERS:
        if cleaned.lower() == key or CART_BIOMARKERS[key]["full_name"].upper() == cleaned:
            return {"type": "biomarker", "canonical": key}

    return None


def get_comparison_context(entity_a: Dict[str, str], entity_b: Dict[str, str]) -> str:
    """Build side-by-side knowledge graph context for two entities.

    Reuses existing get_target_context / get_toxicity_context /
    get_manufacturing_context depending on entity type.

    Returns:
        Formatted comparison context string with both entities' data.
    """
    def _get_entity_context(entity: Dict[str, str]) -> str:
        etype = entity["type"]
        canonical = entity["canonical"]
        if etype == "target":
            return get_target_context(canonical)
        elif etype == "product":
            target = entity.get("target", "")
            return get_target_context(target) if target else ""
        elif etype == "toxicity":
            return get_toxicity_context(canonical)
        elif etype == "manufacturing":
            return get_manufacturing_context(canonical)
        elif etype == "biomarker":
            return get_biomarker_context(canonical)
        elif etype == "costimulatory":
            return f"Costimulatory Domain: {canonical}"
        return ""

    sections = []
    ctx_a = _get_entity_context(entity_a)
    ctx_b = _get_entity_context(entity_b)

    if ctx_a:
        sections.append(f"### {entity_a['canonical']}\n{ctx_a}")
    if ctx_b:
        sections.append(f"### {entity_b['canonical']}\n{ctx_b}")

    return "\n\n---\n\n".join(sections)
