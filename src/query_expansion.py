"""CAR-T query expansion maps for multi-collection RAG search.

When a user asks about a CAR-T topic, expand the search to include
related terms, antigens, products, and concepts.  This improves recall
across the 5 CAR-T collections (literature, trials, constructs, assays,
manufacturing).

Pattern from: rag-chat-pipeline/src/rag_engine.py
  - 10 therapeutic-area expansion dictionaries
  - _get_expanded_genes() scans all maps for keyword hits

Author: Adam Jones
Date: February 2026
"""

from typing import Dict, List, Set

from loguru import logger


# ═══════════════════════════════════════════════════════════════════════
# 1. TARGET_ANTIGEN_EXPANSION
#    Maps antigen keywords to related terms (diseases, products, aliases)
# ═══════════════════════════════════════════════════════════════════════

TARGET_ANTIGEN_EXPANSION: Dict[str, List[str]] = {
    # --- B-cell antigens ---
    "cd19": [
        "CD19", "B-ALL", "DLBCL", "B-cell lymphoma", "B-cell leukemia",
        "tisagenlecleucel", "axicabtagene ciloleucel", "lisocabtagene maraleucel",
        "brexucabtagene autoleucel", "Kymriah", "Yescarta", "Breyanzi", "Tecartus",
        "FMC63", "SJ25C1",
    ],
    "cd22": [
        "CD22", "B-ALL", "DLBCL", "hairy cell leukemia",
        "inotuzumab ozogamicin", "moxetumomab pasudotox",
        "CD19/CD22 bispecific", "dual-targeting", "m971",
    ],
    "cd20": [
        "CD20", "DLBCL", "follicular lymphoma", "B-NHL",
        "rituximab", "obinutuzumab", "Leu-16",
    ],

    # --- Myeloma antigens ---
    "bcma": [
        "BCMA", "B-cell maturation antigen", "TNFRSF17",
        "multiple myeloma", "relapsed refractory myeloma",
        "idecabtagene vicleucel", "ciltacabtagene autoleucel",
        "Abecma", "Carvykti", "teclistamab", "APRIL", "BAFF",
        "bispecific T-cell engager",
    ],
    "gprc5d": [
        "GPRC5D", "multiple myeloma", "talquetamab",
        "BCMA/GPRC5D bispecific", "orphan GPCR",
    ],
    "cd38": [
        "CD38", "multiple myeloma", "daratumumab", "isatuximab",
        "Darzalex", "Sarclisa", "NAD+ hydrolase",
    ],

    # --- Lymphoma / Hodgkin ---
    "cd30": [
        "CD30", "TNFRSF8", "Hodgkin lymphoma", "HL",
        "anaplastic large cell lymphoma", "ALCL",
        "brentuximab vedotin", "Adcetris",
    ],

    # --- Myeloid / AML ---
    "cd33": [
        "CD33", "Siglec-3", "AML", "acute myeloid leukemia",
        "myelodysplastic syndrome", "MDS",
        "gemtuzumab ozogamicin", "Mylotarg",
    ],
    "cd123": [
        "CD123", "IL-3R-alpha", "IL3RA",
        "AML", "BPDCN", "blastic plasmacytoid dendritic cell neoplasm",
        "tagraxofusp", "Elzonris",
    ],
    "flt3": [
        "FLT3", "CD135", "FMS-like tyrosine kinase 3",
        "AML", "FLT3-ITD", "midostaurin", "gilteritinib",
    ],
    "cll-1": [
        "CLL-1", "CLEC12A", "AML", "myeloid leukemia",
        "CLL1/CD33 bispecific",
    ],

    # --- T-cell malignancies ---
    "cd7": [
        "CD7", "T-ALL", "T-cell lymphoma",
        "fratricide", "CD7 knockout", "CRISPR-edited",
        "universal CAR-T", "allogeneic",
    ],
    "cd5": [
        "CD5", "T-ALL", "T-cell lymphoma",
        "fratricide resistance", "CD5 knockout",
    ],

    # --- Solid tumor antigens ---
    "her2": [
        "HER2", "ERBB2", "HER2-positive", "breast cancer",
        "gastric cancer", "glioblastoma", "osteosarcoma",
        "trastuzumab", "Herceptin", "4D5",
    ],
    "gpc3": [
        "GPC3", "glypican-3", "hepatocellular carcinoma", "HCC",
        "liver cancer", "pediatric liver tumors",
    ],
    "egfr": [
        "EGFR", "EGFRvIII", "glioblastoma", "GBM",
        "non-small cell lung cancer", "NSCLC",
        "cetuximab", "variant III",
    ],
    "mesothelin": [
        "mesothelin", "MSLN", "mesothelioma",
        "ovarian cancer", "pancreatic cancer",
        "SS1P", "anetumab ravtansine",
    ],
    "claudin18.2": [
        "Claudin 18.2", "CLDN18.2", "gastric cancer",
        "pancreatic cancer", "gastric adenocarcinoma",
        "zolbetuximab", "tight junction",
    ],
    "gd2": [
        "GD2", "disialoganglioside", "neuroblastoma",
        "retinoblastoma", "osteosarcoma", "melanoma",
        "dinutuximab", "Unituxin", "14g2a",
    ],
    "psma": [
        "PSMA", "prostate-specific membrane antigen", "FOLH1",
        "prostate cancer", "mCRPC",
        "metastatic castration-resistant",
    ],
    "ror1": [
        "ROR1", "receptor tyrosine kinase-like orphan receptor 1",
        "CLL", "mantle cell lymphoma", "triple-negative breast cancer",
        "TNBC", "Wnt signaling",
    ],
    "dll3": [
        "DLL3", "delta-like ligand 3", "small cell lung cancer", "SCLC",
        "neuroendocrine tumors", "Notch ligand",
        "tarlatamab", "rovalpituzumab tesirine",
    ],
    "b7-h3": [
        "B7-H3", "CD276", "neuroblastoma", "NSCLC",
        "head and neck cancer", "checkpoint",
    ],
    "muc1": [
        "MUC1", "mucin-1", "breast cancer",
        "pancreatic cancer", "ovarian cancer",
        "Tn-MUC1", "tumor-associated glycoform",
    ],
    "il13ra2": [
        "IL13Ralpha2", "IL-13 receptor alpha-2",
        "glioblastoma", "GBM", "intracranial delivery",
    ],
    "epcam": [
        "EpCAM", "epithelial cell adhesion molecule",
        "colorectal cancer", "gastric cancer",
        "epithelial tumors",
    ],
}


# ═══════════════════════════════════════════════════════════════════════
# 2. DISEASE_EXPANSION
#    Maps disease keywords to related terms, antigens, and therapies
# ═══════════════════════════════════════════════════════════════════════

DISEASE_EXPANSION: Dict[str, List[str]] = {
    # --- Hematologic malignancies ---
    "b-all": [
        "B-ALL", "B-cell acute lymphoblastic leukemia",
        "precursor B-cell ALL", "Philadelphia-positive ALL", "Ph+ ALL",
        "CD19", "CD22", "CD19/CD22 dual",
        "tisagenlecleucel", "Kymriah",
        "blinatumomab", "Blincyto", "MRD",
        "minimal residual disease", "pediatric ALL",
    ],
    "t-all": [
        "T-ALL", "T-cell acute lymphoblastic leukemia",
        "T-lymphoblastic leukemia",
        "CD7", "CD5", "CD1a",
        "fratricide", "nelarabine",
    ],
    "dlbcl": [
        "DLBCL", "diffuse large B-cell lymphoma",
        "relapsed refractory DLBCL", "r/r DLBCL",
        "CD19", "axicabtagene ciloleucel", "lisocabtagene maraleucel",
        "Yescarta", "Breyanzi",
        "GCB subtype", "ABC subtype", "double-hit lymphoma",
        "high-grade B-cell lymphoma", "transformed follicular",
    ],
    "follicular lymphoma": [
        "follicular lymphoma", "FL", "indolent lymphoma",
        "B-NHL", "CD19", "CD20",
        "grade 3B", "transformed FL",
        "axicabtagene ciloleucel",
    ],
    "mantle cell lymphoma": [
        "mantle cell lymphoma", "MCL",
        "CD19", "brexucabtagene autoleucel", "Tecartus",
        "BTK inhibitor", "ibrutinib",
    ],
    "multiple myeloma": [
        "multiple myeloma", "MM", "myeloma",
        "relapsed refractory myeloma", "r/r MM",
        "BCMA", "GPRC5D", "CD38", "FcRH5",
        "idecabtagene vicleucel", "ciltacabtagene autoleucel",
        "Abecma", "Carvykti",
        "APRIL", "plasma cell",
    ],
    "hodgkin": [
        "Hodgkin lymphoma", "HL", "classical Hodgkin",
        "CD30", "Reed-Sternberg",
        "brentuximab vedotin", "nivolumab",
    ],
    "aml": [
        "AML", "acute myeloid leukemia",
        "CD33", "CD123", "CLL-1", "FLT3", "CLEC12A",
        "myeloid antigen", "myeloid leukemia",
        "lineage switch", "mixed phenotype",
    ],
    "cll": [
        "CLL", "chronic lymphocytic leukemia",
        "CD19", "ROR1",
        "ibrutinib", "venetoclax", "Richter transformation",
    ],

    # --- Solid tumors ---
    "neuroblastoma": [
        "neuroblastoma", "GD2", "B7-H3",
        "pediatric solid tumor", "MYCN amplification",
        "dinutuximab", "high-risk neuroblastoma",
    ],
    "glioblastoma": [
        "glioblastoma", "GBM", "glioma",
        "EGFRvIII", "IL13Ralpha2", "HER2", "GD2",
        "intracranial", "blood-brain barrier",
        "tumor microenvironment", "immunosuppressive",
    ],
    "mesothelioma": [
        "mesothelioma", "malignant pleural mesothelioma",
        "mesothelin", "MSLN",
        "asbestos", "pleural",
    ],
    "pancreatic cancer": [
        "pancreatic cancer", "PDAC",
        "mesothelin", "Claudin 18.2", "MUC1", "HER2",
        "pancreatic ductal adenocarcinoma",
        "desmoplastic stroma", "immunosuppressive TME",
    ],
    "liver cancer": [
        "liver cancer", "hepatocellular carcinoma", "HCC",
        "GPC3", "glypican-3", "AFP",
    ],
    "prostate cancer": [
        "prostate cancer", "PSMA", "mCRPC",
        "castration-resistant", "prostate-specific membrane antigen",
    ],
    "ovarian cancer": [
        "ovarian cancer", "mesothelin", "MUC16",
        "HER2", "folate receptor alpha", "FRa",
    ],
}


# ═══════════════════════════════════════════════════════════════════════
# 3. TOXICITY_EXPANSION
#    Maps toxicity / safety keywords to related terms
# ═══════════════════════════════════════════════════════════════════════

TOXICITY_EXPANSION: Dict[str, List[str]] = {
    # --- Cytokine release syndrome ---
    "crs": [
        "CRS", "cytokine release syndrome", "cytokine storm",
        "tocilizumab", "Actemra", "siltuximab",
        "IL-6", "IL-6 receptor", "sIL-6R",
        "ferritin", "CRP", "C-reactive protein",
        "fever", "hypotension", "hypoxia",
        "Lee grading", "ASTCT grading",
        "grade 3 CRS", "grade 4 CRS",
        "vasopressors", "dexamethasone",
    ],
    "cytokine release": [
        "CRS", "cytokine release syndrome", "cytokine storm",
        "tocilizumab", "IL-6", "ferritin",
    ],

    # --- Neurotoxicity / ICANS ---
    "icans": [
        "ICANS", "immune effector cell-associated neurotoxicity syndrome",
        "neurotoxicity", "CRES",
        "ICE score", "encephalopathy",
        "cerebral edema", "aphasia", "tremor", "seizure",
        "dexamethasone", "corticosteroids",
        "endothelial activation", "BBB disruption",
        "blood-brain barrier",
    ],
    "neurotoxicity": [
        "ICANS", "neurotoxicity", "CRES",
        "immune effector cell-associated neurotoxicity syndrome",
        "ICE score", "encephalopathy",
        "cerebral edema", "dexamethasone",
    ],

    # --- B-cell aplasia ---
    "b-cell aplasia": [
        "B-cell aplasia", "hypogammaglobulinemia",
        "immunoglobulin replacement", "IVIG",
        "agammaglobulinemia", "B-cell recovery",
        "on-target off-tumor", "CD19 depletion",
        "infection risk", "humoral immunity",
    ],
    "hypogammaglobulinemia": [
        "hypogammaglobulinemia", "B-cell aplasia",
        "IVIG", "immunoglobulin replacement",
        "infection risk",
    ],

    # --- HLH / MAS ---
    "hlh": [
        "HLH", "hemophagocytic lymphohistiocytosis",
        "macrophage activation syndrome", "MAS",
        "ferritin", "triglycerides", "fibrinogen",
        "soluble IL-2 receptor", "sCD25",
        "NK cell dysfunction", "etoposide", "ruxolitinib",
        "anakinra",
    ],
    "mas": [
        "MAS", "macrophage activation syndrome",
        "HLH", "hemophagocytic lymphohistiocytosis",
        "ferritin", "hyperinflammation",
    ],

    # --- Tumor lysis syndrome ---
    "tumor lysis": [
        "tumor lysis syndrome", "TLS",
        "hyperuricemia", "hyperkalemia", "hyperphosphatemia",
        "hypocalcemia", "rasburicase", "allopurinol",
        "high tumor burden", "renal injury",
    ],
    "tls": [
        "tumor lysis syndrome", "TLS",
        "hyperuricemia", "rasburicase", "high tumor burden",
    ],

    # --- Graft-versus-host disease ---
    "gvhd": [
        "GvHD", "graft-versus-host disease",
        "allogeneic", "donor-derived",
        "TRAC knockout", "TCR disruption",
        "CRISPR", "universal CAR-T",
        "alemtuzumab", "cyclophosphamide",
        "skin", "liver", "GI tract",
    ],
    "graft versus host": [
        "GvHD", "graft-versus-host disease",
        "allogeneic", "TRAC knockout", "TCR disruption",
    ],

    # --- Cytopenias ---
    "cytopenia": [
        "cytopenia", "pancytopenia", "neutropenia",
        "thrombocytopenia", "prolonged cytopenia",
        "bone marrow suppression", "GCSF",
        "hematopoietic recovery", "delayed recovery",
    ],

    # --- On-target off-tumor ---
    "on-target off-tumor": [
        "on-target off-tumor", "B-cell aplasia",
        "normal tissue toxicity", "antigen expression",
        "safety switch", "suicide gene",
        "iCasp9", "inducible caspase 9",
        "truncated EGFR", "CD20 safety switch",
    ],
}


# ═══════════════════════════════════════════════════════════════════════
# 4. MANUFACTURING_EXPANSION
#    Maps manufacturing / CMC terms to related concepts
# ═══════════════════════════════════════════════════════════════════════

MANUFACTURING_EXPANSION: Dict[str, List[str]] = {
    # --- Transduction ---
    "transduction": [
        "transduction efficiency", "lentiviral vector", "LVV",
        "gamma-retroviral", "retroviral",
        "MOI", "multiplicity of infection",
        "VCN", "vector copy number",
        "viral titer", "functional titer",
        "p24", "FACS transduction",
        "RetroNectin", "polybrene",
        "spinoculation", "static transduction",
    ],
    "lentiviral": [
        "lentiviral vector", "LVV", "lentivirus",
        "HIV-1 derived", "self-inactivating", "SIN vector",
        "VSV-G pseudotyping", "third-generation packaging",
        "transduction efficiency", "VCN",
        "insertional mutagenesis",
    ],
    "viral vector": [
        "lentiviral vector", "gamma-retroviral vector",
        "AAV", "adeno-associated virus",
        "viral titer", "transduction",
        "VSV-G", "pseudotyping",
    ],
    "non-viral": [
        "non-viral delivery", "electroporation",
        "transposon", "Sleeping Beauty", "piggyBac",
        "mRNA electroporation", "lipid nanoparticle", "LNP",
        "CRISPR knock-in", "HDR",
        "transient expression",
    ],

    # --- T-cell expansion ---
    "expansion": [
        "T-cell expansion", "ex vivo expansion",
        "anti-CD3/CD28 beads", "Dynabeads",
        "TransAct", "T Cell TransAct",
        "Expander beads",
        "IL-2", "IL-7", "IL-15", "IL-21",
        "fold expansion", "doubling time",
        "population doublings",
        "G-Rex", "WAVE bioreactor", "Prodigy",
        "CliniMACS Prodigy",
        "fed-batch", "perfusion",
    ],
    "bioreactor": [
        "bioreactor", "G-Rex", "WAVE bioreactor",
        "CliniMACS Prodigy", "Cocoon",
        "Lonza Cocoon", "Miltenyi Prodigy",
        "automated manufacturing", "closed system",
        "fed-batch", "rocking motion",
    ],

    # --- Leukapheresis ---
    "leukapheresis": [
        "leukapheresis", "apheresis",
        "starting material", "PBMC",
        "peripheral blood mononuclear cells",
        "T-cell enrichment", "CD4/CD8 selection",
        "Ficoll", "elutriation",
        "Spectra Optia", "lymphodepletion timing",
    ],
    "apheresis": [
        "apheresis", "leukapheresis",
        "starting material", "PBMC", "T-cell enrichment",
    ],

    # --- Cryopreservation ---
    "cryopreservation": [
        "cryopreservation", "cryopreserved product",
        "CryoStor", "CS10", "DMSO",
        "controlled-rate freezer", "CRF",
        "liquid nitrogen", "LN2", "vapor phase",
        "thaw", "post-thaw viability",
        "post-thaw recovery", "cold chain",
        "bedside thaw",
    ],
    "formulation": [
        "formulation", "final product",
        "infusion bag", "cryopreservation medium",
        "excipients", "human serum albumin", "HSA",
        "PlasmaLyte", "dextran",
    ],

    # --- Release testing ---
    "release testing": [
        "release testing", "certificate of analysis", "CoA",
        "lot release", "quality control", "QC",
        "sterility", "endotoxin", "LAL",
        "mycoplasma", "RCL", "replication-competent lentivirus",
        "identity", "potency", "purity",
        "CAR expression", "percent CAR+",
        "viability", "cell count", "total viable cells",
        "VCN release", "in-process testing",
    ],
    "potency": [
        "potency assay", "functional potency",
        "cytotoxicity assay", "cytokine secretion",
        "IFN-gamma", "specific lysis",
        "potency release criterion",
    ],
    "identity": [
        "identity testing", "flow cytometry",
        "CAR detection", "Protein L",
        "anti-idiotype", "CAR+ percentage",
        "CD3+", "CD4/CD8 ratio",
    ],

    # --- Vein-to-vein time ---
    "vein-to-vein": [
        "vein-to-vein time", "turnaround time",
        "manufacturing slot", "scheduling",
        "bridging therapy", "patient waitlist",
        "out-of-spec", "manufacturing failure",
        "point-of-care manufacturing",
    ],
    "point-of-care": [
        "point-of-care", "POC manufacturing",
        "decentralized manufacturing",
        "bedside manufacturing", "automated",
        "CliniMACS Prodigy", "Cocoon",
        "rapid manufacturing",
    ],

    # --- Lymphodepletion ---
    "lymphodepletion": [
        "lymphodepletion", "conditioning regimen",
        "fludarabine", "cyclophosphamide",
        "Flu/Cy", "bendamustine",
        "T-cell depletion", "lymphopenia",
        "homeostatic expansion",
    ],
}


# ═══════════════════════════════════════════════════════════════════════
# 5. MECHANISM_EXPANSION
#    Maps mechanism / biology terms to related concepts
# ═══════════════════════════════════════════════════════════════════════

MECHANISM_EXPANSION: Dict[str, List[str]] = {
    # --- Resistance / Antigen escape ---
    "resistance": [
        "antigen loss", "antigen escape", "antigen-negative relapse",
        "lineage switch", "myeloid switch",
        "trogocytosis", "antigen masking", "epitope masking",
        "CD19-negative", "BCMA-negative",
        "bispecific CAR", "dual-targeting",
        "tandem CAR", "OR-gate",
    ],
    "antigen escape": [
        "antigen escape", "antigen loss", "antigen-negative relapse",
        "CD19-negative", "lineage switch",
        "bispecific", "dual-targeting",
        "alternative splicing", "truncated antigen",
    ],
    "antigen loss": [
        "antigen loss", "antigen escape", "antigen-negative relapse",
        "lineage switch", "bispecific", "dual-targeting",
    ],
    "lineage switch": [
        "lineage switch", "myeloid switch", "phenotypic switch",
        "mixed phenotype acute leukemia", "MPAL",
        "antigen escape", "CD19-negative AML",
    ],
    "trogocytosis": [
        "trogocytosis", "antigen transfer", "antigen stripping",
        "CAR-mediated trogocytosis", "fratricide",
        "antigen density reduction",
    ],

    # --- T-cell exhaustion ---
    "exhaustion": [
        "T-cell exhaustion", "CAR-T exhaustion",
        "PD-1", "PDCD1", "LAG-3", "LAG3",
        "TIM-3", "HAVCR2", "TIGIT",
        "TOX", "NR4A", "transcription factor",
        "tonic signaling", "chronic stimulation",
        "epigenetic remodeling", "chromatin accessibility",
        "terminal differentiation",
        "checkpoint blockade", "PD-1 knockout",
        "anti-PD-1", "ipilimumab",
    ],
    "pd-1": [
        "PD-1", "PDCD1", "programmed death-1",
        "checkpoint inhibitor", "pembrolizumab", "nivolumab",
        "PD-L1", "CD274",
        "T-cell exhaustion", "checkpoint blockade",
    ],
    "tonic signaling": [
        "tonic signaling", "ligand-independent signaling",
        "scFv clustering", "antigen-independent activation",
        "T-cell exhaustion", "premature differentiation",
        "FMC63", "4-1BB amelioration",
    ],

    # --- Persistence ---
    "persistence": [
        "T-cell persistence", "CAR-T persistence",
        "long-term remission", "B-cell aplasia duration",
        "memory T cells", "Tcm", "central memory",
        "stem cell memory", "Tscm",
        "4-1BB", "CD28 costimulation",
        "IL-7", "IL-15", "homeostatic cytokines",
        "qPCR transgene", "flow CAR detection",
    ],
    "memory": [
        "memory T cells", "central memory", "Tcm",
        "effector memory", "Tem",
        "stem cell memory", "Tscm",
        "naive T cells", "Tn",
        "T-cell differentiation",
        "CD62L", "CCR7", "CD45RA", "CD45RO",
        "long-lived persistence",
    ],

    # --- Costimulation ---
    "costimulation": [
        "CD28", "4-1BB", "CD137", "TNFRSF9",
        "OX40", "CD134", "ICOS", "CD278",
        "costimulatory domain", "second generation CAR",
        "third generation CAR",
        "CD28 versus 4-1BB", "signaling kinetics",
        "PI3K", "NF-kB", "TRAF",
    ],
    "4-1bb": [
        "4-1BB", "CD137", "TNFRSF9",
        "costimulatory domain", "TRAF1", "TRAF2",
        "NF-kB signaling", "persistence advantage",
        "oxidative metabolism", "central memory",
        "tisagenlecleucel", "Kymriah",
    ],
    "cd28 costimulation": [
        "CD28", "costimulatory domain",
        "PI3K signaling", "rapid expansion",
        "effector function", "glycolytic metabolism",
        "axicabtagene ciloleucel", "Yescarta",
    ],

    # --- Signaling ---
    "signaling": [
        "CD3-zeta", "CD247", "ITAM",
        "immunoreceptor tyrosine-based activation motif",
        "ZAP-70", "LAT", "SLP-76",
        "phosphorylation", "signal transduction",
        "proximal signaling", "distal signaling",
        "PI3K", "AKT", "mTOR", "NF-kB", "NFAT", "AP-1",
    ],
    "cd3-zeta": [
        "CD3-zeta", "CD247", "ITAM",
        "signal 1", "activation domain",
        "first-generation CAR",
    ],

    # --- Trafficking / Infiltration ---
    "trafficking": [
        "T-cell trafficking", "tumor infiltration",
        "homing", "chemokine receptor",
        "CXCR1", "CXCR2", "CXCR4", "CCR2", "CCR4",
        "IL-8 receptor", "tumor homing",
        "extravasation", "adhesion molecules",
        "integrin", "selectin",
    ],
    "tumor microenvironment": [
        "tumor microenvironment", "TME",
        "immunosuppressive", "regulatory T cells", "Tregs",
        "myeloid-derived suppressor cells", "MDSC",
        "tumor-associated macrophages", "TAM",
        "TGF-beta", "IL-10", "PGE2", "IDO",
        "hypoxia", "HIF-1alpha",
        "extracellular matrix", "fibrotic stroma",
        "checkpoint ligands", "PD-L1",
    ],
    "tme": [
        "tumor microenvironment", "TME",
        "immunosuppressive", "Tregs", "MDSC",
        "TGF-beta", "hypoxia",
    ],

    # --- Cytokine biology ---
    "cytokine": [
        "cytokine", "IFN-gamma", "TNF-alpha", "IL-2",
        "IL-6", "GM-CSF", "IL-10", "IL-17",
        "granzyme B", "perforin",
        "cytolytic activity", "degranulation",
    ],
}


# ═══════════════════════════════════════════════════════════════════════
# 6. CONSTRUCT_EXPANSION
#    Maps CAR design / engineering terms to related concepts
# ═══════════════════════════════════════════════════════════════════════

CONSTRUCT_EXPANSION: Dict[str, List[str]] = {
    # --- scFv / binding domain ---
    "scfv": [
        "scFv", "single-chain variable fragment",
        "VH", "VL", "variable heavy", "variable light",
        "linker", "G4S linker",
        "humanized", "fully human",
        "VHH", "nanobody", "camelid",
        "binding affinity", "Kd",
        "FMC63", "m971",
    ],
    "nanobody": [
        "nanobody", "VHH", "single-domain antibody",
        "camelid", "llama-derived",
        "small format", "biparatopic",
    ],

    # --- Hinge / spacer ---
    "hinge": [
        "hinge", "spacer", "extracellular spacer",
        "IgG1 hinge", "IgG4 hinge",
        "CD8-alpha hinge", "CD28 hinge",
        "spacer length", "immunological synapse distance",
        "Fc receptor binding", "FcgammaR",
    ],

    # --- Transmembrane domain ---
    "transmembrane": [
        "transmembrane domain", "TM domain",
        "CD8-alpha TM", "CD28 TM",
        "membrane anchoring", "dimerization",
        "signal transduction",
    ],

    # --- Armored / 4th gen ---
    "armored": [
        "armored CAR", "fourth generation", "4th generation",
        "TRUCK", "T cells redirected for universal cytokine killing",
        "IL-12 secreting", "IL-15 secreting", "IL-18 secreting",
        "IL-21 armored",
        "dominant-negative TGF-beta receptor", "dnTGFbRII",
        "PD-1 dominant negative", "switch receptor",
        "constitutive cytokine",
    ],
    "4th generation": [
        "fourth generation CAR", "4th generation",
        "armored CAR", "TRUCK",
        "cytokine-secreting CAR",
        "IL-12", "IL-15", "IL-18",
    ],
    "truck": [
        "TRUCK", "T cells redirected for universal cytokine killing",
        "armored CAR", "IL-12 secreting",
        "cytokine payload", "transgenic cytokine",
    ],

    # --- Bispecific / multi-target ---
    "bispecific": [
        "bispecific CAR", "dual-targeting CAR",
        "tandem CAR", "bivalent CAR",
        "OR-gate logic", "AND-gate logic",
        "CD19/CD22", "BCMA/CD38", "BCMA/GPRC5D",
        "antigen escape prevention",
        "loop CAR", "split CAR",
    ],
    "tandem": [
        "tandem CAR", "bispecific", "bivalent",
        "OR-gate", "two binding domains",
        "CD19/CD22 tandem",
    ],
    "logic gate": [
        "logic gate", "AND-gate", "OR-gate",
        "NOT-gate", "synNotch",
        "synthetic biology", "Boolean logic",
        "split CAR", "if-then circuit",
        "on-switch", "safety circuit",
    ],

    # --- Universal / allogeneic ---
    "universal": [
        "universal CAR-T", "allogeneic CAR-T",
        "off-the-shelf", "donor-derived",
        "TRAC knockout", "TCR knockout",
        "beta-2-microglobulin knockout", "B2M knockout",
        "HLA-negative", "MHC class I knockout",
        "CRISPR-Cas9", "TALEN", "zinc finger nuclease",
        "gene editing", "base editing",
        "NK cell evasion", "HLA-E overexpression",
        "CD52 knockout", "alemtuzumab-resistant",
        "UCART19", "ALLO-501",
    ],
    "allogeneic": [
        "allogeneic", "off-the-shelf", "donor-derived",
        "universal CAR-T",
        "TRAC knockout", "B2M knockout", "HLA knockout",
        "GvHD prevention", "gene editing",
        "CRISPR", "TALEN",
    ],
    "gene editing": [
        "gene editing", "CRISPR-Cas9", "CRISPR",
        "TALEN", "zinc finger nuclease", "ZFN",
        "base editing", "prime editing",
        "TRAC knockout", "B2M knockout", "PD-1 knockout",
        "homology-directed repair", "HDR",
        "NHEJ", "guide RNA", "sgRNA",
        "off-target effects", "translocation",
    ],

    # --- Safety switches ---
    "safety switch": [
        "safety switch", "kill switch", "suicide gene",
        "iCasp9", "inducible caspase 9",
        "AP1903", "rimiducid",
        "truncated EGFR", "EGFRt", "cetuximab",
        "CD20 safety switch", "rituximab",
        "HSV-TK", "herpes simplex thymidine kinase",
        "ganciclovir",
    ],
    "icasp9": [
        "iCasp9", "inducible caspase 9",
        "AP1903", "rimiducid", "dimerizer",
        "safety switch", "suicide gene",
    ],

    # --- CAR generations overview ---
    "first generation": [
        "first generation CAR", "1st generation",
        "CD3-zeta only", "no costimulation",
        "limited persistence", "historical",
    ],
    "second generation": [
        "second generation CAR", "2nd generation",
        "CD28 or 4-1BB costimulation",
        "CD3-zeta plus costimulatory",
        "FDA-approved products",
        "tisagenlecleucel", "axicabtagene ciloleucel",
    ],
    "third generation": [
        "third generation CAR", "3rd generation",
        "dual costimulatory domains",
        "CD28 plus 4-1BB", "CD28 plus OX40",
        "enhanced signaling",
    ],

    # --- Next-gen platforms ---
    "synnotch": [
        "synNotch", "synthetic Notch",
        "AND-gate", "logic-gated",
        "priming receptor", "response element",
        "tissue-specific", "conditional activation",
    ],
    "adapter car": [
        "adapter CAR", "switchable CAR",
        "FITC adapter", "leucine zipper",
        "SpyCatcher/SpyTag", "universal adapter",
        "dose-titratable", "modular targeting",
    ],
}


# ═══════════════════════════════════════════════════════════════════════
# 7. SAFETY_EXPANSION
# ═══════════════════════════════════════════════════════════════════════

SAFETY_EXPANSION: Dict[str, List[str]] = {
    "adverse event": [
        "adverse event", "AE", "safety signal", "toxicity", "side effect",
        "post-market", "pharmacovigilance", "FAERS", "label update",
    ],
    "pharmacovigilance": [
        "pharmacovigilance", "post-market surveillance", "FAERS", "safety monitoring",
        "adverse event reporting", "label change", "boxed warning", "REMS",
    ],
    "rems": [
        "REMS", "risk evaluation", "mitigation strategy", "certification",
        "treatment center", "safety protocol", "CRS management", "ICANS management",
    ],
    "secondary malignancy": [
        "secondary malignancy", "T-cell lymphoma", "MDS", "AML",
        "insertional mutagenesis", "oncogenesis", "long-term safety",
        "FDA boxed warning", "15-year follow-up",
    ],
    "long-term safety": [
        "long-term safety", "prolonged cytopenia", "B-cell aplasia",
        "hypogammaglobulinemia", "IVIG", "immunoglobulin replacement",
        "late-onset", "delayed toxicity", "chronic GVHD",
    ],
    "cytopenia": [
        "cytopenia", "neutropenia", "thrombocytopenia", "anemia",
        "pancytopenia", "prolonged cytopenia", "bone marrow suppression",
        "G-CSF", "platelet transfusion",
    ],
    "infection": [
        "infection", "opportunistic infection", "bacterial", "fungal",
        "viral reactivation", "CMV", "HHV-6", "aspergillus",
        "hypogammaglobulinemia", "immunodeficiency",
    ],
    "cardiac": [
        "cardiac toxicity", "cardiomyopathy", "arrhythmia", "troponin",
        "heart failure", "cardiac arrest", "myocarditis",
    ],
}


# ═══════════════════════════════════════════════════════════════════════
# 8. BIOMARKER_EXPANSION
# ═══════════════════════════════════════════════════════════════════════

BIOMARKER_EXPANSION: Dict[str, List[str]] = {
    "biomarker": [
        "biomarker", "predictive marker", "prognostic marker",
        "pharmacodynamic", "response prediction", "outcome prediction",
    ],
    "ferritin": [
        "ferritin", "serum ferritin", "CRS prediction",
        "inflammatory marker", "iron storage", "hyperferritinemia",
    ],
    "crp": [
        "CRP", "C-reactive protein", "inflammatory biomarker",
        "acute phase", "CRS monitoring",
    ],
    "il-6": [
        "IL-6", "interleukin-6", "cytokine storm", "tocilizumab target",
        "CRS biomarker", "inflammatory cytokine",
    ],
    "mrd": [
        "MRD", "minimal residual disease", "flow cytometry MRD",
        "PCR MRD", "measurable residual disease", "MRD negative",
        "deep response", "molecular remission",
    ],
    "car expansion": [
        "CAR-T expansion", "peak expansion", "Cmax", "transgene copies",
        "qPCR", "in vivo proliferation", "pharmacokinetics",
    ],
    "exhaustion marker": [
        "exhaustion", "PD-1", "LAG-3", "TIM-3", "TOX", "NR4A",
        "T-cell dysfunction", "checkpoint", "terminal differentiation",
        "epigenetic exhaustion",
    ],
    "ldh": [
        "LDH", "lactate dehydrogenase", "tumor burden marker",
        "prognostic factor", "metabolic marker",
    ],
    "ctdna": [
        "ctDNA", "circulating tumor DNA", "liquid biopsy",
        "cell-free DNA", "molecular response", "genomic profiling",
    ],
    "sbcma": [
        "sBCMA", "soluble BCMA", "BCMA shedding",
        "gamma-secretase", "decoy antigen", "BCMA resistance",
    ],
    "tcm": [
        "Tcm", "central memory", "CD45RA-", "CCR7+", "CD62L+",
        "T-cell fitness", "memory phenotype", "naive T-cell",
    ],
    "cd4 cd8": [
        "CD4:CD8", "CD4/CD8 ratio", "T-cell composition",
        "defined composition", "product phenotype",
    ],
}


# ═══════════════════════════════════════════════════════════════════════
# 9. REGULATORY_EXPANSION
# ═══════════════════════════════════════════════════════════════════════

REGULATORY_EXPANSION: Dict[str, List[str]] = {
    "fda": [
        "FDA", "Food and Drug Administration", "regulatory",
        "BLA", "biologics license", "approval", "label",
    ],
    "bla": [
        "BLA", "biologics license application", "regulatory submission",
        "NDA", "marketing authorization", "approval pathway",
    ],
    "breakthrough": [
        "breakthrough therapy", "BTD", "expedited program",
        "accelerated development", "FDA designation",
    ],
    "rmat": [
        "RMAT", "regenerative medicine advanced therapy",
        "cell therapy designation", "expedited approval",
    ],
    "accelerated approval": [
        "accelerated approval", "surrogate endpoint",
        "confirmatory trial", "conditional approval", "full approval",
    ],
    "ema": [
        "EMA", "European Medicines Agency", "CHMP",
        "marketing authorization", "conditional approval", "EU approval",
    ],
    "label update": [
        "label update", "prescribing information", "boxed warning",
        "black box warning", "safety communication", "Dear Doctor letter",
    ],
    "post-marketing": [
        "post-marketing", "Phase 4", "post-approval",
        "registry study", "long-term follow-up", "PMR", "PMC",
    ],
}


# ═══════════════════════════════════════════════════════════════════════
# 10. SEQUENCE_EXPANSION
# ═══════════════════════════════════════════════════════════════════════

SEQUENCE_EXPANSION: Dict[str, List[str]] = {
    "scfv": [
        "scFv", "single-chain variable fragment", "antibody fragment",
        "VH", "VL", "linker", "binding domain",
    ],
    "cdr": [
        "CDR", "complementarity determining region", "CDR3",
        "hypervariable region", "antigen binding loop", "paratope",
    ],
    "binding affinity": [
        "binding affinity", "Kd", "dissociation constant",
        "kon", "koff", "SPR", "BLI", "affinity maturation",
    ],
    "humanization": [
        "humanization", "humanized antibody", "CDR grafting",
        "framework", "deimmunization", "anti-drug antibody", "ADA",
        "immunogenicity",
    ],
    "fmc63": [
        "FMC63", "anti-CD19 scFv", "murine origin",
        "tisagenlecleucel binder", "axicabtagene binder",
    ],
    "nanobody": [
        "nanobody", "VHH", "single-domain antibody", "camelid",
        "llama antibody", "sdAb", "LCAR-B38M",
    ],
    "darpin": [
        "DARPin", "designed ankyrin repeat", "non-antibody scaffold",
        "Centyrin", "fibronectin", "alternative binding domain",
    ],
    "bispecific car": [
        "bispecific CAR", "tandem CAR", "dual-targeting",
        "bicistronic", "loop CAR", "CD19/CD22", "OR-gate",
    ],
}


# ═══════════════════════════════════════════════════════════════════════
# 11. REALWORLD_EXPANSION
# ═══════════════════════════════════════════════════════════════════════

REALWORLD_EXPANSION: Dict[str, List[str]] = {
    "real-world": [
        "real-world", "RWE", "real-world evidence", "real-world data",
        "clinical practice", "commercial experience", "post-approval",
    ],
    "cibmtr": [
        "CIBMTR", "Center for International Blood and Marrow Transplant Research",
        "transplant registry", "national registry",
    ],
    "registry": [
        "registry", "CIBMTR", "EBMT", "DESCAR-T", "observational study",
        "national registry", "multi-center registry",
    ],
    "community": [
        "community center", "community practice", "non-academic",
        "community oncology", "access to care", "referral pattern",
    ],
    "academic": [
        "academic center", "academic medical center", "tertiary center",
        "CAR-T center of excellence", "FACT accredited",
    ],
    "elderly": [
        "elderly", "older adults", "geriatric", "age ≥65",
        "age ≥70", "frailty", "comorbidities", "fitness",
    ],
    "bridging therapy": [
        "bridging therapy", "bridging chemotherapy", "bridging radiation",
        "pre-CAR-T", "disease control", "tumor debulking",
    ],
    "disparities": [
        "disparities", "racial", "ethnic", "socioeconomic",
        "access", "underserved", "minority", "equity",
    ],
    "resource utilization": [
        "resource utilization", "ICU admission", "readmission",
        "length of stay", "cost", "healthcare economics", "HCRU",
    ],
    "long-term follow-up": [
        "long-term follow-up", "durability", "late relapse",
        "5-year", "3-year", "sustained remission", "cure",
    ],
}


# ═══════════════════════════════════════════════════════════════════════
# 12. IMMUNOGENICITY_EXPANSION
#     Maps HLA, ADA, and immunogenicity terms to related concepts
# ═══════════════════════════════════════════════════════════════════════

IMMUNOGENICITY_EXPANSION: Dict[str, List[str]] = {
    "hla": [
        "HLA", "human leukocyte antigen", "MHC", "major histocompatibility complex",
        "HLA-DRB1", "HLA-A*02:01", "antigen presentation", "T-cell epitope",
    ],
    "immunogenicity": [
        "immunogenicity", "immunogenic", "anti-drug antibody", "ADA",
        "neutralizing antibody", "binding antibody", "titer",
        "pre-existing immunity", "HAMA",
    ],
    "ada": [
        "anti-drug antibody", "ADA", "neutralizing antibody", "NAb",
        "binding antibody", "immunogenicity testing", "tiered assay",
        "drug tolerance", "ADA incidence",
    ],
    "humanization": [
        "humanization", "humanized", "CDR grafting", "framework selection",
        "back-mutation", "VH3-23", "VK1-39", "deimmunization",
        "fully human", "phage display",
    ],
    "deimmunization": [
        "deimmunization", "deimmunized", "T-cell epitope removal",
        "EpiMatrix", "NetMHCIIpan", "epitope prediction",
        "framework shuffling", "germline humanization",
    ],
    "anti-drug antibody": [
        "anti-drug antibody", "ADA", "HAMA", "anti-murine antibody",
        "immunogenicity", "CAR-T persistence", "neutralizing",
        "infusion reaction", "anaphylaxis",
    ],
    "cdr grafting": [
        "CDR grafting", "complementarity-determining region", "framework region",
        "humanization", "VH germline", "VL germline", "Kabat numbering",
    ],
    "t-cell epitope": [
        "T-cell epitope", "MHC-II", "HLA-DRB1", "CD4 T helper",
        "ELISpot", "IFN-gamma", "DC-T cell assay",
        "in silico prediction", "NetMHCIIpan",
    ],
    "elispot": [
        "ELISpot", "enzyme-linked immunospot", "IFN-gamma spot",
        "T-cell response", "immunogenicity testing", "in vitro assay",
    ],
    "framework shuffling": [
        "framework shuffling", "framework selection", "germline framework",
        "VH3 family", "VK1 family", "CDR loop grafting",
        "thermal stability", "aggregation resistance",
    ],
    "hama": [
        "HAMA", "human anti-mouse antibody", "anti-murine",
        "murine scFv", "FMC63", "pre-existing immunity",
        "cross-reactivity", "clearance",
    ],
    "netmhciipan": [
        "NetMHCIIpan", "MHC-II prediction", "epitope prediction",
        "EpiMatrix", "IEDB", "immunoinformatics", "in silico immunogenicity",
    ],
}


# ═══════════════════════════════════════════════════════════════════════
# ALL EXPANSION MAPS — ordered list for iteration
# ═══════════════════════════════════════════════════════════════════════

ALL_EXPANSION_MAPS: List[tuple] = [
    ("Target Antigen", TARGET_ANTIGEN_EXPANSION),
    ("Disease", DISEASE_EXPANSION),
    ("Toxicity", TOXICITY_EXPANSION),
    ("Manufacturing", MANUFACTURING_EXPANSION),
    ("Mechanism", MECHANISM_EXPANSION),
    ("Construct", CONSTRUCT_EXPANSION),
    ("Safety", SAFETY_EXPANSION),
    ("Biomarker", BIOMARKER_EXPANSION),
    ("Regulatory", REGULATORY_EXPANSION),
    ("Sequence", SEQUENCE_EXPANSION),
    ("RealWorld", REALWORLD_EXPANSION),
    ("Immunogenicity", IMMUNOGENICITY_EXPANSION),
]


# ═══════════════════════════════════════════════════════════════════════
# EXPANSION FUNCTION
# ═══════════════════════════════════════════════════════════════════════


def expand_query(query: str) -> List[str]:
    """Extract expansion terms from a user query.

    Scans the query for keywords matching any expansion map,
    returns a deduplicated list of related terms to broaden
    the search across the 5 CAR-T Milvus collections.

    Args:
        query: Raw user question (e.g., "What causes CRS after
               CD19 CAR-T therapy?")

    Returns:
        Deduplicated list of expansion terms.  Empty list if no
        keywords matched.

    Example::

        >>> expand_query("Why do patients relapse with CD19-negative disease?")
        ['CD19', 'B-ALL', 'DLBCL', ..., 'antigen loss', 'antigen escape', ...]
    """
    query_lower = query.lower()
    matched_terms: Set[str] = set()

    for category, mapping in ALL_EXPANSION_MAPS:
        for keyword, terms in mapping.items():
            if keyword in query_lower:
                matched_terms.update(terms)
                logger.info(
                    f"CAR-T expansion [{category}]: '{keyword}' -> "
                    f"{len(terms)} terms"
                )

    result = sorted(matched_terms)
    if result:
        logger.info(
            f"Query expansion produced {len(result)} unique terms "
            f"from query: {query[:80]}..."
        )
    return result


def expand_query_by_category(query: str) -> Dict[str, List[str]]:
    """Like expand_query but returns terms grouped by expansion category.

    Useful when different collections should weight different
    categories (e.g., manufacturing terms are most relevant to
    the cart_manufacturing collection).

    Args:
        query: Raw user question.

    Returns:
        Dict mapping category name to list of matched terms.
        Only categories with at least one match are included.

    Example::

        >>> expand_query_by_category("transduction efficiency for CD19 CAR")
        {
            'Target Antigen': ['CD19', 'B-ALL', ...],
            'Manufacturing': ['transduction efficiency', 'lentiviral vector', ...],
        }
    """
    query_lower = query.lower()
    categories: Dict[str, Set[str]] = {}

    for category, mapping in ALL_EXPANSION_MAPS:
        for keyword, terms in mapping.items():
            if keyword in query_lower:
                if category not in categories:
                    categories[category] = set()
                categories[category].update(terms)
                logger.debug(
                    f"CAR-T expansion [{category}]: '{keyword}' matched"
                )

    # Convert sets to sorted lists
    return {cat: sorted(terms) for cat, terms in categories.items()}


def get_expansion_stats() -> Dict[str, int]:
    """Return the number of keywords and total terms per expansion map.

    Useful for logging / health checks.

    Returns:
        Dict with category names as keys, keyword counts as values.
    """
    stats: Dict[str, int] = {}
    for category, mapping in ALL_EXPANSION_MAPS:
        total_terms = sum(len(v) for v in mapping.values())
        stats[category] = {
            "keywords": len(mapping),
            "total_terms": total_terms,
        }
    return stats
