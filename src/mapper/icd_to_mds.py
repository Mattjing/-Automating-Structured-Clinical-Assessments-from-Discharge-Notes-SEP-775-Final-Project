"""
ICD-10-CM / ICD-9-CM → MDS 3.0 item mapping tables.

These mappings convert MIMIC-IV structured diagnosis codes, prescription drug
names, and procedure codes into ground-truth MDS item values.  They are used
by ``scripts/generate_labeled_samples.py`` to produce labeled (note → MDS JSON)
pairs for:

* Fine-tuning the Seq2Seq extractor
* Evaluating all extraction approaches (LLM, Seq2Seq, RAG+LLM)

Mapping sources
---------------
* CMS MDS 3.0 RAI Manual v1.18.11 (October 2023) — item definitions
* ICD-10-CM 2024 code ranges — diagnosis groupings
* ICD-9-CM legacy codes — MIMIC-IV uses both ICD-9 and ICD-10

Each mapping entry is a tuple of (pattern_type, pattern, mds_item_id) where
pattern_type is "prefix" (startswith match) or "exact" (full code match).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Section I — Active Diagnoses: ICD code prefix → MDS item
# ---------------------------------------------------------------------------

# (match_type, code_prefix_or_exact, mds_item_id)
ICD10_TO_MDS_I: List[Tuple[str, str, str]] = [
    # Cancer
    ("prefix", "C",    "I0100"),   # all malignant neoplasms C00-C96
    ("prefix", "D0",   "I0100"),   # in situ neoplasms

    # Anemia
    ("prefix", "D50",  "I0200"),   # iron deficiency anemia
    ("prefix", "D51",  "I0200"),   # vitamin B12 deficiency anemia
    ("prefix", "D52",  "I0200"),   # folate deficiency anemia
    ("prefix", "D53",  "I0200"),   # other nutritional anemias
    ("prefix", "D55",  "I0200"),   # hemolytic anemias
    ("prefix", "D56",  "I0200"),   # thalassemia
    ("prefix", "D57",  "I0200"),   # sickle-cell
    ("prefix", "D58",  "I0200"),
    ("prefix", "D59",  "I0200"),
    ("prefix", "D60",  "I0200"),   # aplastic anemia
    ("prefix", "D61",  "I0200"),
    ("prefix", "D62",  "I0200"),
    ("prefix", "D63",  "I0200"),
    ("prefix", "D64",  "I0200"),

    # Atrial fibrillation / dysrhythmias
    ("prefix", "I48",  "I0300"),   # atrial fibrillation and flutter
    ("prefix", "I49",  "I0300"),   # other cardiac arrhythmias
    ("prefix", "I47",  "I0300"),   # paroxysmal tachycardia
    ("prefix", "I44",  "I0300"),   # AV block
    ("prefix", "I45",  "I0300"),   # conduction disorders

    # Coronary artery disease
    ("prefix", "I25",  "I0400"),   # chronic ischemic heart disease
    ("prefix", "I21",  "I0400"),   # acute MI
    ("prefix", "I22",  "I0400"),   # subsequent MI
    ("prefix", "I20",  "I0400"),   # angina pectoris

    # DVT / PE / PTE
    ("prefix", "I26",  "I0500"),   # pulmonary embolism
    ("prefix", "I82",  "I0500"),   # other venous thrombosis (incl DVT)

    # Heart failure
    ("prefix", "I50",  "I0600"),   # heart failure
    ("prefix", "I11.0","I0600"),   # hypertensive heart disease with HF
    ("prefix", "I13.0","I0600"),   # hypertensive heart+kidney with HF
    ("prefix", "I13.2","I0600"),

    # Hypertension
    ("prefix", "I10",  "I0700"),   # essential hypertension
    ("prefix", "I11",  "I0700"),   # hypertensive heart disease
    ("prefix", "I12",  "I0700"),   # hypertensive CKD
    ("prefix", "I13",  "I0700"),   # hypertensive heart + CKD
    ("prefix", "I15",  "I0700"),   # secondary hypertension

    # Orthostatic hypotension
    ("prefix", "I95.1","I0800"),

    # PVD / PAD
    ("prefix", "I73",  "I0900"),   # other peripheral vascular diseases
    ("prefix", "I70",  "I0900"),   # atherosclerosis

    # Cirrhosis
    ("prefix", "K74",  "I1100"),   # fibrosis and cirrhosis of liver
    ("prefix", "K70.3","I1100"),   # alcoholic cirrhosis

    # GERD / ulcer
    ("prefix", "K21",  "I1200"),   # GERD
    ("prefix", "K25",  "I1200"),   # gastric ulcer
    ("prefix", "K26",  "I1200"),   # duodenal ulcer
    ("prefix", "K27",  "I1200"),   # peptic ulcer
    ("prefix", "K22.1","I1200"),   # esophageal ulcer

    # IBD
    ("prefix", "K50",  "I1300"),   # Crohn's disease
    ("prefix", "K51",  "I1300"),   # ulcerative colitis

    # BPH
    ("prefix", "N40",  "I1400"),

    # Renal insufficiency / failure / ESRD
    ("prefix", "N18",  "I1500"),   # CKD
    ("prefix", "N17",  "I1500"),   # AKI
    ("prefix", "N19",  "I1500"),   # unspecified kidney failure
    ("prefix", "Z99.2","I1500"),   # dependence on renal dialysis

    # Neurogenic bladder
    ("prefix", "N31",  "I1550"),

    # Obstructive uropathy
    ("prefix", "N13",  "I1650"),

    # Pneumonia
    ("prefix", "J12",  "I2000"),
    ("prefix", "J13",  "I2000"),
    ("prefix", "J14",  "I2000"),
    ("prefix", "J15",  "I2000"),
    ("prefix", "J16",  "I2000"),
    ("prefix", "J17",  "I2000"),
    ("prefix", "J18",  "I2000"),

    # Septicemia
    ("prefix", "A40",  "I2100"),   # streptococcal sepsis
    ("prefix", "A41",  "I2100"),   # other sepsis
    ("prefix", "R65.2","I2100"),   # severe sepsis

    # Tuberculosis
    ("prefix", "A15",  "I2200"),
    ("prefix", "A16",  "I2200"),
    ("prefix", "A17",  "I2200"),
    ("prefix", "A18",  "I2200"),
    ("prefix", "A19",  "I2200"),

    # UTI
    ("prefix", "N39.0","I2300"),

    # Viral hepatitis
    ("prefix", "B15",  "I2400"),
    ("prefix", "B16",  "I2400"),
    ("prefix", "B17",  "I2400"),
    ("prefix", "B18",  "I2400"),
    ("prefix", "B19",  "I2400"),

    # Diabetes mellitus
    ("prefix", "E10",  "I2900"),
    ("prefix", "E11",  "I2900"),
    ("prefix", "E13",  "I2900"),

    # Hyponatremia
    ("exact",  "E87.1","I3100"),

    # Hyperkalemia
    ("exact",  "E87.5","I3200"),

    # Hyperlipidemia
    ("prefix", "E78",  "I3300"),

    # Thyroid disorder
    ("prefix", "E01",  "I3400"),
    ("prefix", "E02",  "I3400"),
    ("prefix", "E03",  "I3400"),   # hypothyroidism
    ("prefix", "E04",  "I3400"),
    ("prefix", "E05",  "I3400"),   # hyperthyroidism
    ("prefix", "E06",  "I3400"),   # thyroiditis

    # Arthritis
    ("prefix", "M05",  "I3700"),   # RA
    ("prefix", "M06",  "I3700"),   # other RA
    ("prefix", "M15",  "I3700"),   # polyosteoarthritis
    ("prefix", "M16",  "I3700"),   # hip OA
    ("prefix", "M17",  "I3700"),   # knee OA
    ("prefix", "M18",  "I3700"),
    ("prefix", "M19",  "I3700"),   # other OA

    # Osteoporosis
    ("prefix", "M80",  "I3800"),
    ("prefix", "M81",  "I3800"),

    # Hip fracture
    ("prefix", "S72.0","I3900"),   # femoral neck fracture
    ("prefix", "S72.1","I3900"),   # trochanteric fracture
    ("prefix", "S72.2","I3900"),   # subtrochanteric fracture
    ("prefix", "M84.35","I3900"),  # stress fracture femur

    # Other fracture
    ("prefix", "S",    "I4000"),   # NOTE: broad — refined below
    # (S72 already captured above for hip; other S-codes = other fracture)

    # Alzheimer's
    ("prefix", "G30",  "I4200"),
    ("exact",  "F00",  "I4200"),

    # Aphasia
    ("prefix", "R47.0","I4300"),

    # Cerebral palsy
    ("prefix", "G80",  "I4400"),

    # CVA / TIA / stroke
    ("prefix", "I60",  "I4500"),   # subarachnoid hemorrhage
    ("prefix", "I61",  "I4500"),   # intracerebral hemorrhage
    ("prefix", "I62",  "I4500"),   # other nontraumatic intracranial hemorrhage
    ("prefix", "I63",  "I4500"),   # cerebral infarction
    ("prefix", "I65",  "I4500"),   # occlusion of precerebral arteries
    ("prefix", "I66",  "I4500"),   # occlusion of cerebral arteries
    ("prefix", "I67.89","I4500"),
    ("prefix", "G45",  "I4500"),   # TIA

    # Non-Alzheimer's dementia
    ("prefix", "F01",  "I4800"),   # vascular dementia
    ("prefix", "F02",  "I4800"),   # dementia in other diseases
    ("prefix", "F03",  "I4800"),   # unspecified dementia
    ("prefix", "G31.0","I4800"),   # frontotemporal dementia
    ("prefix", "G31.8","I4800"),   # Lewy body

    # Hemiplegia / hemiparesis
    ("prefix", "G81",  "I4900"),

    # Paraplegia
    ("prefix", "G82.2","I5000"),

    # Quadriplegia
    ("prefix", "G82.5","I5100"),

    # Multiple sclerosis
    ("prefix", "G35",  "I5200"),

    # Huntington's
    ("prefix", "G10",  "I5250"),

    # Parkinson's
    ("prefix", "G20",  "I5300"),

    # Seizure disorder / epilepsy
    ("prefix", "G40",  "I5400"),
    ("prefix", "G41",  "I5400"),
    ("prefix", "R56",  "I5400"),

    # TBI
    ("prefix", "S06",  "I5500"),

    # Malnutrition
    ("prefix", "E40",  "I5600"),
    ("prefix", "E41",  "I5600"),
    ("prefix", "E42",  "I5600"),
    ("prefix", "E43",  "I5600"),
    ("prefix", "E44",  "I5600"),
    ("prefix", "E46",  "I5600"),

    # Anxiety
    ("prefix", "F41",  "I5700"),
    ("prefix", "F40",  "I5700"),

    # Depression
    ("prefix", "F32",  "I5800"),   # major depressive episode
    ("prefix", "F33",  "I5800"),   # recurrent depressive disorder
    ("prefix", "F34.1","I5800"),   # dysthymia

    # Bipolar
    ("prefix", "F31",  "I5900"),

    # Psychotic disorder
    ("prefix", "F23",  "I5950"),   # brief psychotic disorder
    ("prefix", "F28",  "I5950"),
    ("prefix", "F29",  "I5950"),

    # Schizophrenia
    ("prefix", "F20",  "I6000"),
    ("prefix", "F25",  "I6000"),   # schizoaffective

    # PTSD
    ("exact",  "F43.1","I6100"),
    ("prefix", "F43.10","I6100"),
    ("prefix", "F43.11","I6100"),
    ("prefix", "F43.12","I6100"),

    # Asthma / COPD / chronic lung
    ("prefix", "J44",  "I6200"),   # COPD
    ("prefix", "J45",  "I6200"),   # asthma
    ("prefix", "J43",  "I6200"),   # emphysema
    ("prefix", "J41",  "I6200"),   # chronic bronchitis
    ("prefix", "J42",  "I6200"),
    ("prefix", "J84",  "I6200"),   # interstitial lung disease

    # Respiratory failure
    ("prefix", "J96",  "I6300"),

    # Cataracts / glaucoma / macular degeneration
    ("prefix", "H25",  "I6500"),   # senile cataract
    ("prefix", "H26",  "I6500"),   # other cataract
    ("prefix", "H40",  "I6500"),   # glaucoma
    ("prefix", "H35.3","I6500"),   # macular degeneration
]


# ICD-9 legacy mappings (MIMIC-IV uses icd_version=9 for older admissions)
ICD9_TO_MDS_I: List[Tuple[str, str, str]] = [
    ("prefix", "410",  "I0400"),   # acute MI
    ("prefix", "414",  "I0400"),   # chronic ischemic heart disease
    ("prefix", "427",  "I0300"),   # cardiac dysrhythmias
    ("prefix", "428",  "I0600"),   # heart failure
    ("prefix", "401",  "I0700"),   # essential hypertension
    ("prefix", "250",  "I2900"),   # diabetes mellitus
    ("prefix", "585",  "I1500"),   # CKD
    ("prefix", "486",  "I2000"),   # pneumonia
    ("prefix", "038",  "I2100"),   # septicemia
    ("prefix", "434",  "I4500"),   # cerebral artery occlusion
    ("prefix", "436",  "I4500"),   # acute CVA
    ("prefix", "496",  "I6200"),   # COPD
    ("prefix", "493",  "I6200"),   # asthma
    ("prefix", "518.81","I6300"),  # respiratory failure
    ("prefix", "571",  "I1100"),   # cirrhosis
    ("prefix", "530.81","I1200"),  # GERD
    ("prefix", "272",  "I3300"),   # hyperlipidemia
    ("prefix", "244",  "I3400"),   # hypothyroidism
    ("prefix", "331.0","I4200"),   # Alzheimer's
    ("prefix", "290",  "I4800"),   # dementias
    ("prefix", "332",  "I5300"),   # Parkinson's
    ("prefix", "345",  "I5400"),   # epilepsy
    ("prefix", "296.2","I5800"),   # major depressive (single)
    ("prefix", "296.3","I5800"),   # major depressive (recurrent)
    ("prefix", "300.0","I5700"),   # anxiety
    ("prefix", "140",  "I0100"),   # malignant neoplasm (140-239)
    ("prefix", "280",  "I0200"),   # iron deficiency anemia
    ("prefix", "281",  "I0200"),   # other anemias
    ("prefix", "415.1","I0500"),   # pulmonary embolism
    ("prefix", "453",  "I0500"),   # DVT
    ("prefix", "599.0","I2300"),   # UTI
]


# ---------------------------------------------------------------------------
# Section N — Medications: drug name keyword → MDS item
# ---------------------------------------------------------------------------

# Maps lowercase drug name substrings to N0415 sub-items
DRUG_TO_MDS_N: Dict[str, str] = {
    # Antipsychotics → N0415A
    "haloperidol": "N0415A", "quetiapine": "N0415A", "olanzapine": "N0415A",
    "risperidone": "N0415A", "aripiprazole": "N0415A", "ziprasidone": "N0415A",
    "clozapine": "N0415A", "paliperidone": "N0415A", "lurasidone": "N0415A",
    "chlorpromazine": "N0415A", "fluphenazine": "N0415A", "perphenazine": "N0415A",

    # Antianxiety → N0415B
    "lorazepam": "N0415B", "alprazolam": "N0415B", "diazepam": "N0415B",
    "clonazepam": "N0415B", "buspirone": "N0415B", "hydroxyzine": "N0415B",
    "midazolam": "N0415B", "oxazepam": "N0415B",

    # Antidepressants → N0415C
    "sertraline": "N0415C", "fluoxetine": "N0415C", "citalopram": "N0415C",
    "escitalopram": "N0415C", "paroxetine": "N0415C", "venlafaxine": "N0415C",
    "duloxetine": "N0415C", "bupropion": "N0415C", "mirtazapine": "N0415C",
    "trazodone": "N0415C", "amitriptyline": "N0415C", "nortriptyline": "N0415C",
    "desvenlafaxine": "N0415C",

    # Hypnotics → N0415D
    "zolpidem": "N0415D", "eszopiclone": "N0415D", "zaleplon": "N0415D",
    "suvorexant": "N0415D", "ramelteon": "N0415D", "lemborexant": "N0415D",
    "temazepam": "N0415D",

    # Anticoagulants → N0415E
    "warfarin": "N0415E", "heparin": "N0415E", "enoxaparin": "N0415E",
    "apixaban": "N0415E", "rivaroxaban": "N0415E", "dabigatran": "N0415E",
    "edoxaban": "N0415E", "fondaparinux": "N0415E", "dalteparin": "N0415E",
    "argatroban": "N0415E", "bivalirudin": "N0415E",

    # Antibiotics → N0415F
    "vancomycin": "N0415F", "piperacillin": "N0415F", "metronidazole": "N0415F",
    "ceftriaxone": "N0415F", "cefazolin": "N0415F", "cefepime": "N0415F",
    "ciprofloxacin": "N0415F", "levofloxacin": "N0415F", "meropenem": "N0415F",
    "amoxicillin": "N0415F", "azithromycin": "N0415F", "doxycycline": "N0415F",
    "trimethoprim": "N0415F", "sulfamethoxazole": "N0415F", "clindamycin": "N0415F",
    "linezolid": "N0415F", "ampicillin": "N0415F", "gentamicin": "N0415F",
    "tobramycin": "N0415F", "nitrofurantoin": "N0415F", "cephalexin": "N0415F",

    # Diuretics → N0415G
    "furosemide": "N0415G", "torsemide": "N0415G", "bumetanide": "N0415G",
    "hydrochlorothiazide": "N0415G", "chlorthalidone": "N0415G",
    "spironolactone": "N0415G", "metolazone": "N0415G", "triamterene": "N0415G",
    "acetazolamide": "N0415G", "indapamide": "N0415G",

    # Opioids → N0415H
    "morphine": "N0415H", "oxycodone": "N0415H", "hydrocodone": "N0415H",
    "fentanyl": "N0415H", "tramadol": "N0415H", "hydromorphone": "N0415H",
    "methadone": "N0415H", "codeine": "N0415H", "buprenorphine": "N0415H",
    "oxymorphone": "N0415H", "tapentadol": "N0415H",

    # Antiplatelets → N0415I
    "aspirin": "N0415I", "clopidogrel": "N0415I", "ticagrelor": "N0415I",
    "prasugrel": "N0415I", "dipyridamole": "N0415I", "ticlopidine": "N0415I",

    # Hypoglycemics → N0415J
    "metformin": "N0415J", "glipizide": "N0415J", "glyburide": "N0415J",
    "glimepiride": "N0415J", "insulin": "N0415J", "sitagliptin": "N0415J",
    "pioglitazone": "N0415J", "empagliflozin": "N0415J", "canagliflozin": "N0415J",
    "dapagliflozin": "N0415J", "liraglutide": "N0415J", "semaglutide": "N0415J",
    "dulaglutide": "N0415J", "saxagliptin": "N0415J", "linagliptin": "N0415J",
    "repaglinide": "N0415J", "acarbose": "N0415J",
}


# ---------------------------------------------------------------------------
# Section O — Procedures: ICD-10-PCS prefix → MDS item
# ---------------------------------------------------------------------------

# These map ICD-10-PCS procedure codes to Section O items.
# MIMIC-IV procedures_icd uses both ICD-9 and ICD-10 procedure codes.
PROCEDURE_KEYWORD_TO_MDS_O: Dict[str, str] = {
    # Keywords found in procedure descriptions → MDS O item
    "chemotherapy":       "O0110A1",
    "radiation":          "O0110B1",
    "oxygen":             "O0110C1",
    "suction":            "O0110D1",
    "tracheostomy":       "O0110E1",
    "mechanical ventil":  "O0110F1",
    "ventilator":         "O0110F1",
    "intubat":            "O0110F1",
    "bipap":              "O0110G2",
    "cpap":               "O0110G3",
    "non-invasive ventil":"O0110G1",
    "transfus":           "O0110I1",
    "hemodialysis":       "O0110J2",
    "peritoneal dialysis":"O0110J3",
    "dialysis":           "O0110J1",
    "picc":               "O0110O4",
    "central line":       "O0110O4",
    "central venous":     "O0110O4",
    "peripheral iv":      "O0110O2",
    "iv access":          "O0110O1",
    "physical therapy":   "O0400C1",
    "occupational therap":"O0400B1",
    "speech therap":      "O0400A1",
    "respiratory therap": "O0400D1",
}


# ---------------------------------------------------------------------------
# Public lookup functions
# ---------------------------------------------------------------------------

def icd_to_mds_items(
    icd_code: str,
    icd_version: int = 10,
) -> Set[str]:
    """
    Return the set of MDS item IDs triggered by an ICD code.

    Parameters
    ----------
    icd_code:
        ICD-10-CM or ICD-9-CM code (with or without dots).
    icd_version:
        9 or 10.

    Returns
    -------
    set of str
        MDS item IDs (e.g. {"I0400", "I0700"}).
    """
    code = icd_code.strip().upper().replace(".", "")
    table = ICD10_TO_MDS_I if icd_version == 10 else ICD9_TO_MDS_I

    matched: Set[str] = set()
    for match_type, pattern, mds_item in table:
        pattern_clean = pattern.replace(".", "").upper()
        if match_type == "prefix" and code.startswith(pattern_clean):
            matched.add(mds_item)
        elif match_type == "exact" and code == pattern_clean:
            matched.add(mds_item)

    # Special handling: S-codes → I4000 (Other Fracture) only if NOT hip (I3900)
    if "I4000" in matched and "I3900" in matched:
        matched.discard("I4000")  # hip fracture is more specific

    return matched


def drug_to_mds_items(drug_name: str) -> Set[str]:
    """
    Return the set of MDS N-section item IDs triggered by a drug name.

    Parameters
    ----------
    drug_name:
        Drug name string (e.g. "Warfarin Sodium 5mg tablet").

    Returns
    -------
    set of str
        MDS item IDs (e.g. {"N0415E"}).
    """
    lower = drug_name.strip().lower()
    matched: Set[str] = set()
    for keyword, mds_item in DRUG_TO_MDS_N.items():
        if keyword in lower:
            matched.add(mds_item)
    # Insulin detection — also triggers N0350A
    if "insulin" in lower:
        matched.add("N0350A")
    return matched


def procedure_to_mds_items(procedure_desc: str) -> Set[str]:
    """
    Return the set of MDS O-section item IDs triggered by a procedure description.

    Parameters
    ----------
    procedure_desc:
        Free-text procedure description.

    Returns
    -------
    set of str
        MDS item IDs (e.g. {"O0110F1"}).
    """
    lower = procedure_desc.strip().lower()
    matched: Set[str] = set()
    for keyword, mds_item in PROCEDURE_KEYWORD_TO_MDS_O.items():
        if keyword in lower:
            matched.add(mds_item)
    return matched
