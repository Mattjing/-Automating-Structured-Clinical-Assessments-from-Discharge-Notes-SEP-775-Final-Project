"""
MDS 3.0 (Minimum Data Set) schema definitions.

This module defines the structure of the MDS 3.0 assessment form used in
nursing homes to track patient conditions.  Each section, item, and response
option is modelled as a plain Python dataclass so that the schema can be
queried, serialised, and used by the extractor and mapper modules without
any external dependencies.

Reference: CMS MDS 3.0 RAI Manual (v1.18.11, October 2023)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence


# ---------------------------------------------------------------------------
# Low-level primitives
# ---------------------------------------------------------------------------


class MDSItemType(str, Enum):
    """Allowed response types for an MDS item."""

    BOOLEAN = "boolean"      # Yes / No
    INTEGER = "integer"      # numeric code or count
    TEXT = "text"            # free-text entry
    DATE = "date"            # date field (YYYY-MM-DD)
    SELECT = "select"        # one of a fixed set of coded values
    MULTI = "multi_select"   # one or more of a fixed set of coded values


@dataclass
class MDSResponseOption:
    """A single selectable response option for a SELECT or MULTI item."""

    code: str
    label: str


@dataclass
class MDSItem:
    """
    A single item (question) on the MDS 3.0 form.

    Attributes
    ----------
    item_id : str
        MDS item identifier (e.g. ``"A0800"``).
    label : str
        Human-readable description of the item.
    item_type : MDSItemType
        The expected response type.
    options : list of MDSResponseOption
        For SELECT/MULTI items: the list of valid coded responses.
    description : str, optional
        Extended clinical description or instructions.
    """

    item_id: str
    label: str
    item_type: MDSItemType
    options: List[MDSResponseOption] = field(default_factory=list)
    description: str = ""

    def option_codes(self) -> List[str]:
        """Return the list of valid option codes."""
        return [o.code for o in self.options]

    def option_labels(self) -> List[str]:
        """Return the list of valid option labels."""
        return [o.label for o in self.options]


@dataclass
class MDSSection:
    """
    A section of the MDS 3.0 form.

    Attributes
    ----------
    section_id : str
        Single-letter section identifier (e.g. ``"A"``, ``"G"``).
    title : str
        Full title of the section.
    items : list of MDSItem
        Ordered list of items in this section.
    """

    section_id: str
    title: str
    items: List[MDSItem] = field(default_factory=list)

    def get_item(self, item_id: str) -> Optional[MDSItem]:
        """Return the item with *item_id*, or ``None`` if not found."""
        for item in self.items:
            if item.item_id == item_id:
                return item
        return None


# ---------------------------------------------------------------------------
# Full MDS 3.0 schema
# ---------------------------------------------------------------------------


class MDSSchema:
    """
    Complete MDS 3.0 schema.

    The schema covers the clinical sections most relevant to information that
    can be extracted from a hospital discharge summary:

    * A  – Identification Information
    * B  – Hearing, Speech, and Vision
    * C  – Cognitive Patterns
    * D  – Mood
    * E  – Behavior
    * G  – Functional Status (Activities of Daily Living)
    * H  – Bladder and Bowel
    * I  – Active Disease Diagnoses
    * J  – Health Conditions
    * K  – Swallowing / Nutritional Status
    * M  – Skin Conditions
    * N  – Medications
    * O  – Special Treatments, Procedures, and Programs
    """

    def __init__(self, section_ids: Optional[Sequence[str]] = None) -> None:
        self._sections: Dict[str, MDSSection] = {}
        self._requested_section_ids = (
            {section_id.upper() for section_id in section_ids}
            if section_ids
            else None
        )
        self._build_schema()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_section(self, section_id: str) -> Optional[MDSSection]:
        """Return a section by its single-letter identifier."""
        return self._sections.get(section_id.upper())

    def get_item(self, item_id: str) -> Optional[MDSItem]:
        """Search all sections and return the item with *item_id*."""
        for section in self._sections.values():
            item = section.get_item(item_id)
            if item is not None:
                return item
        return None

    def all_sections(self) -> List[MDSSection]:
        """Return all sections in alphabetical order."""
        return [self._sections[k] for k in sorted(self._sections)]

    def all_items(self) -> List[MDSItem]:
        """Return a flat list of all items across all sections."""
        items: List[MDSItem] = []
        for section in self.all_sections():
            items.extend(section.items)
        return items

    def section_ids(self) -> List[str]:
        """Return sorted list of section identifiers."""
        return sorted(self._sections.keys())

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the schema to a plain dictionary."""
        return {
            sec_id: {
                "title": sec.title,
                "items": [
                    {
                        "item_id": item.item_id,
                        "label": item.label,
                        "type": item.item_type.value,
                        "options": [
                            {"code": o.code, "label": o.label}
                            for o in item.options
                        ],
                        "description": item.description,
                    }
                    for item in sec.items
                ],
            }
            for sec_id, sec in self._sections.items()
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialise the schema to JSON."""
        return json.dumps(self.to_dict(), indent=indent)

    # ------------------------------------------------------------------
    # Schema construction
    # ------------------------------------------------------------------

    def _add_section(self, section: MDSSection) -> None:
        self._sections[section.section_id] = section

    def _build_schema(self) -> None:
        """Populate the requested MDS 3.0 sections."""
        build_steps: List[tuple[str, Callable[[], None]]] = [
            ("A", self._build_section_a),
            ("B", self._build_section_b),
            ("C", self._build_section_c),
            ("D", self._build_section_d),
            ("E", self._build_section_e),
            ("G", self._build_section_g),
            ("H", self._build_section_h),
            ("I", self._build_section_i),
            ("J", self._build_section_j),
            ("K", self._build_section_k),
            ("M", self._build_section_m),
            ("N", self._build_section_n),
            ("O", self._build_section_o),
        ]

        for section_id, builder in build_steps:
            if self._requested_section_ids and section_id not in self._requested_section_ids:
                continue
            builder()

    def _build_section_a(self) -> None:
        sec = MDSSection("A", "Identification Information")
        sec.items = [
            MDSItem(
                "A0800",
                "Gender",
                MDSItemType.SELECT,
                [
                    MDSResponseOption("1", "Male"),
                    MDSResponseOption("2", "Female"),
                ],
            ),
            MDSItem(
                "A0900",
                "Birth Date",
                MDSItemType.DATE,
                description="MM-DD-YYYY",
            ),
            MDSItem(
                "A1000",
                "Race / Ethnicity",
                MDSItemType.MULTI,
                [
                    MDSResponseOption("A", "American Indian or Alaska Native"),
                    MDSResponseOption("B", "Asian"),
                    MDSResponseOption("C", "Black or African American"),
                    MDSResponseOption("D", "Hispanic or Latino"),
                    MDSResponseOption("E", "Native Hawaiian or Other Pacific Islander"),
                    MDSResponseOption("F", "White"),
                ],
            ),
            MDSItem(
                "A1800",
                "Admitted From",
                MDSItemType.SELECT,
                [
                    MDSResponseOption("01", "Community (private home/apt, board/care, assisted living)"),
                    MDSResponseOption("02", "Another nursing home or swing bed"),
                    MDSResponseOption("03", "Acute hospital"),
                    MDSResponseOption("04", "Psychiatric hospital"),
                    MDSResponseOption("05", "Inpatient rehabilitation facility"),
                    MDSResponseOption("06", "ID/DD facility"),
                    MDSResponseOption("07", "Hospice"),
                    MDSResponseOption("99", "Other"),
                ],
            ),
        ]
        self._add_section(sec)

    def _build_section_b(self) -> None:
        sec = MDSSection("B", "Hearing, Speech, and Vision")
        sec.items = [
            MDSItem(
                "B0200",
                "Hearing",
                MDSItemType.SELECT,
                [
                    MDSResponseOption("0", "Adequate — no difficulty in normal conversation, social interaction, listening to TV"),
                    MDSResponseOption("1", "Minimal difficulty — difficulty in some environments (e.g., when not one-on-one or loud)"),
                    MDSResponseOption("2", "Moderate difficulty — speaker has to increase volume and speak distinctly"),
                    MDSResponseOption("3", "Highly impaired — absence of useful hearing"),
                ],
            ),
            MDSItem(
                "B0300",
                "Hearing Aid",
                MDSItemType.BOOLEAN,
                description="Does the resident use a hearing aid?",
            ),
            MDSItem(
                "B0600",
                "Speech Clarity",
                MDSItemType.SELECT,
                [
                    MDSResponseOption("0", "Clear speech — distinct, intelligible words"),
                    MDSResponseOption("1", "Unclear speech — slurred or difficult to understand"),
                    MDSResponseOption("2", "No speech — absence of spoken words"),
                ],
            ),
            MDSItem(
                "B1000",
                "Vision",
                MDSItemType.SELECT,
                [
                    MDSResponseOption("0", "Adequate — sees fine detail, such as regular print in newspapers/books"),
                    MDSResponseOption("1", "Impaired — sees large print, but not regular print in newspapers/books"),
                    MDSResponseOption("2", "Moderately impaired — limited vision; not able to see newspaper headlines but can identify objects"),
                    MDSResponseOption("3", "Highly impaired — object identification in question; but eyes appear to follow objects"),
                    MDSResponseOption("4", "Severely impaired — no vision or sees only light, colors or shapes; eyes do not appear to follow objects"),
                ],
            ),
            MDSItem(
                "B1200",
                "Corrective Lenses",
                MDSItemType.BOOLEAN,
                description="Does the resident use corrective lenses?",
            ),
        ]
        self._add_section(sec)

    def _build_section_c(self) -> None:
        sec = MDSSection("C", "Cognitive Patterns")
        sec.items = [
            MDSItem(
                "C0500",
                "BIMS Summary Score",
                MDSItemType.INTEGER,
                description=(
                    "Brief Interview for Mental Status (BIMS) total score. "
                    "Range 0-15.  13-15: cognitively intact; 8-12: moderately impaired; "
                    "0-7: severely impaired."
                ),
            ),
            MDSItem(
                "C0700",
                "Short-term Memory OK",
                MDSItemType.BOOLEAN,
                description="Seems or appears to recall after 5 minutes.",
            ),
            MDSItem(
                "C0800",
                "Long-term Memory OK",
                MDSItemType.BOOLEAN,
                description="Seems or appears to recall long past.",
            ),
            MDSItem(
                "C0900",
                "Memory / Recall Ability",
                MDSItemType.MULTI,
                [
                    MDSResponseOption("A", "Current season"),
                    MDSResponseOption("B", "Location of own room"),
                    MDSResponseOption("C", "Staff names and faces"),
                    MDSResponseOption("D", "That he/she is in a nursing home"),
                    MDSResponseOption("Z", "None of the above were recalled"),
                ],
            ),
            MDSItem(
                "C1000",
                "Cognitive Skills for Daily Decision Making",
                MDSItemType.SELECT,
                [
                    MDSResponseOption("0", "Independent — decisions consistent/reasonable"),
                    MDSResponseOption("1", "Modified independence — some difficulty in new situations only"),
                    MDSResponseOption("2", "Moderately impaired — decisions poor; cues/supervision required"),
                    MDSResponseOption("3", "Severely impaired — never/rarely makes decisions"),
                ],
            ),
        ]
        self._add_section(sec)

    def _build_section_d(self) -> None:
        sec = MDSSection("D", "Mood")
        phq9_options = [
            MDSResponseOption("0", "Never or 1 day"),
            MDSResponseOption("1", "2-6 days (several days)"),
            MDSResponseOption("2", "7-11 days (half or more of the days)"),
            MDSResponseOption("3", "12-14 days (nearly every day)"),
        ]
        sec.items = [
            MDSItem("D0200A1", "Little interest or pleasure in doing things", MDSItemType.SELECT, phq9_options),
            MDSItem("D0200B1", "Feeling down, depressed or hopeless", MDSItemType.SELECT, phq9_options),
            MDSItem("D0200C1", "Trouble falling or staying asleep, or sleeping too much", MDSItemType.SELECT, phq9_options),
            MDSItem("D0200D1", "Feeling tired or having little energy", MDSItemType.SELECT, phq9_options),
            MDSItem("D0200E1", "Poor appetite or overeating", MDSItemType.SELECT, phq9_options),
            MDSItem("D0200F1", "Feeling bad about yourself — or that you are a failure or have let yourself or your family down", MDSItemType.SELECT, phq9_options),
            MDSItem("D0200G1", "Trouble concentrating on things", MDSItemType.SELECT, phq9_options),
            MDSItem("D0200H1", "Moving or speaking so slowly that other people could have noticed. Or the opposite — being so fidgety or restless", MDSItemType.SELECT, phq9_options),
            MDSItem("D0200I1", "Thoughts that you would be better off dead", MDSItemType.SELECT, phq9_options),
            MDSItem(
                "D0300",
                "PHQ-9 Total Severity Score",
                MDSItemType.INTEGER,
                description="Sum of D0200A1–D0200I1 scores.  Range 0-27.",
            ),
        ]
        self._add_section(sec)

    def _build_section_e(self) -> None:
        freq_options = [
            MDSResponseOption("0", "Behavior not exhibited in last 7 days"),
            MDSResponseOption("1", "Behavior occurred 1-3 days"),
            MDSResponseOption("2", "Behavior occurred 4-6 days, but less than daily"),
            MDSResponseOption("3", "Behavior occurred daily"),
        ]
        sec = MDSSection("E", "Behavior")
        sec.items = [
            MDSItem("E0200A", "Physical behavioral symptoms directed toward others", MDSItemType.SELECT, freq_options,
                    description="e.g. hitting, kicking, scratching, sexually abusing others"),
            MDSItem("E0200B", "Verbal behavioral symptoms directed toward others", MDSItemType.SELECT, freq_options,
                    description="e.g. threatening others, screaming at others, cursing at others"),
            MDSItem("E0200C", "Other behavioral symptoms not directed toward others", MDSItemType.SELECT, freq_options,
                    description="e.g. making disruptive sounds, noisy breathing, screaming, self-injurious behavior"),
            MDSItem("E0800", "Rejection of Care", MDSItemType.SELECT, freq_options,
                    description="Did the resident reject evaluation or care (e.g. medications, ADL assistance)?"),
            MDSItem("E0900", "Wandering", MDSItemType.SELECT, freq_options,
                    description="Has the resident wandered?"),
        ]
        self._add_section(sec)

    def _build_section_g(self) -> None:
        adl_options = [
            MDSResponseOption("0", "Independent — no help or staff oversight"),
            MDSResponseOption("1", "Supervision — oversight, encouragement, or cueing"),
            MDSResponseOption("2", "Limited assistance — resident highly involved in activity; staff provide guided maneuvering or light touching"),
            MDSResponseOption("3", "Extensive assistance — resident involved in activity, staff provide weight-bearing support"),
            MDSResponseOption("4", "Total dependence — full staff performance, resident does not participate"),
            MDSResponseOption("7", "Activity occurred only once or twice — not enough information"),
            MDSResponseOption("8", "Activity did not occur during the entire 7-day period"),
        ]
        sec = MDSSection("G", "Functional Status")
        sec.items = [
            MDSItem("G0110A1", "Bed Mobility — Self-Performance", MDSItemType.SELECT, adl_options),
            MDSItem("G0110B1", "Transfer — Self-Performance", MDSItemType.SELECT, adl_options),
            MDSItem("G0110C1", "Walk in Room — Self-Performance", MDSItemType.SELECT, adl_options),
            MDSItem("G0110D1", "Walk in Corridor — Self-Performance", MDSItemType.SELECT, adl_options),
            MDSItem("G0110E1", "Locomotion On Unit — Self-Performance", MDSItemType.SELECT, adl_options),
            MDSItem("G0110F1", "Locomotion Off Unit — Self-Performance", MDSItemType.SELECT, adl_options),
            MDSItem("G0110G1", "Dressing — Self-Performance", MDSItemType.SELECT, adl_options),
            MDSItem("G0110H1", "Eating — Self-Performance", MDSItemType.SELECT, adl_options),
            MDSItem("G0110I1", "Toilet Use — Self-Performance", MDSItemType.SELECT, adl_options),
            MDSItem("G0110J1", "Personal Hygiene — Self-Performance", MDSItemType.SELECT, adl_options),
            MDSItem("G0300A", "Balance while sitting on side of bed", MDSItemType.SELECT, [
                MDSResponseOption("0", "Steady at all times"),
                MDSResponseOption("1", "Not steady, but able to stabilize without staff assistance"),
                MDSResponseOption("2", "Not steady, only able to stabilize with staff assistance"),
            ]),
            MDSItem("G0600", "Mobility Devices", MDSItemType.MULTI, [
                MDSResponseOption("A", "Cane/crutch"),
                MDSResponseOption("B", "Walker"),
                MDSResponseOption("C", "Wheelchair (manual)"),
                MDSResponseOption("D", "Wheelchair (electric)"),
                MDSResponseOption("E", "Limb prosthesis"),
                MDSResponseOption("Z", "None of the above"),
            ]),
        ]
        self._add_section(sec)

    def _build_section_h(self) -> None:
        continence_options = [
            MDSResponseOption("0", "Always continent"),
            MDSResponseOption("1", "Occasionally incontinent (less than 7 episodes in 7 days)"),
            MDSResponseOption("2", "Frequently incontinent (7 or more episodes in 7 days, but at least one continent void)"),
            MDSResponseOption("3", "Always incontinent"),
            MDSResponseOption("9", "Not rated — resident had an indwelling catheter or ostomy, or no urine output for the entire 7-day period"),
        ]
        sec = MDSSection("H", "Bladder and Bowel")
        sec.items = [
            MDSItem("H0300", "Urinary Continence", MDSItemType.SELECT, continence_options),
            MDSItem("H0400", "Bowel Continence", MDSItemType.SELECT, [
                MDSResponseOption("0", "Always continent"),
                MDSResponseOption("1", "Occasionally incontinent"),
                MDSResponseOption("2", "Frequently incontinent"),
                MDSResponseOption("3", "Always incontinent"),
                MDSResponseOption("9", "Not rated"),
            ]),
            MDSItem("H0100", "Appliances", MDSItemType.MULTI, [
                MDSResponseOption("A", "Indwelling catheter (including suprapubic catheter and nephrostomy tube)"),
                MDSResponseOption("B", "External catheter"),
                MDSResponseOption("C", "Ostomy (including colostomy, urostomy and ileostomy)"),
                MDSResponseOption("D", "Intermittent catheterization"),
                MDSResponseOption("Z", "None of the above"),
            ]),
        ]
        self._add_section(sec)

    def _build_section_i(self) -> None:
        sec = MDSSection("I", "Active Disease Diagnoses")
        sec.items = [
            MDSItem("I0020", "Primary medical condition category", MDSItemType.SELECT, [
                MDSResponseOption("01", "Stroke"),
                MDSResponseOption("02", "Non-Traumatic Brain Dysfunction"),
                MDSResponseOption("03", "Traumatic Brain Dysfunction"),
                MDSResponseOption("04", "Non-Traumatic Spinal Cord Dysfunction"),
                MDSResponseOption("05", "Traumatic Spinal Cord Dysfunction"),
                MDSResponseOption("06", "Progressive Neurological Conditions"),
                MDSResponseOption("07", "Other Neurological Conditions"),
                MDSResponseOption("08", "Amputation"),
                MDSResponseOption("09", "Hip and Knee Replacement"),
                MDSResponseOption("10", "Fractures and Other Multiple Trauma"),
                MDSResponseOption("11", "Other Orthopedic Conditions"),
                MDSResponseOption("12", "Debility, Cardiorespiratory Conditions"),
                MDSResponseOption("13", "Medically Complex Conditions"),
            ]),
            MDSItem("I0020B", "Primary medical condition ICD code", MDSItemType.TEXT),
            MDSItem("I0100", "Cancer", MDSItemType.BOOLEAN),
            MDSItem("I0200", "Anemia", MDSItemType.BOOLEAN),
            MDSItem("I0300", "Atrial Fibrillation or Other Dysrhythmias", MDSItemType.BOOLEAN),
            MDSItem("I0400", "Coronary Artery Disease (CAD)", MDSItemType.BOOLEAN),
            MDSItem("I0500", "Deep Venous Thrombosis (DVT), Pulmonary Embolus (PE), or Pulmonary Thrombo-Embolism (PTE)", MDSItemType.BOOLEAN),
            MDSItem("I0600", "Heart Failure", MDSItemType.BOOLEAN),
            MDSItem("I0700", "Hypertension", MDSItemType.BOOLEAN),
            MDSItem("I0800", "Orthostatic Hypotension", MDSItemType.BOOLEAN),
            MDSItem("I0900", "Peripheral Vascular Disease (PVD) or Peripheral Arterial Disease (PAD)", MDSItemType.BOOLEAN),
            MDSItem("I1100", "Cirrhosis", MDSItemType.BOOLEAN),
            MDSItem("I1200", "Gastroesophageal Reflux Disease (GERD) or Ulcer", MDSItemType.BOOLEAN),
            MDSItem("I1300", "Ulcerative Colitis, Crohn's Disease, or Inflammatory Bowel Disease", MDSItemType.BOOLEAN),
            MDSItem("I1400", "Benign Prostatic Hyperplasia (BPH)", MDSItemType.BOOLEAN),
            MDSItem("I1500", "Renal Insufficiency, Renal Failure, or End-Stage Renal Disease (ESRD)", MDSItemType.BOOLEAN),
            MDSItem("I1550", "Neurogenic Bladder", MDSItemType.BOOLEAN),
            MDSItem("I1650", "Obstructive Uropathy", MDSItemType.BOOLEAN),
            MDSItem("I1700", "Multidrug-Resistant Organism (MDRO)", MDSItemType.BOOLEAN),
            MDSItem("I2000", "Pneumonia", MDSItemType.BOOLEAN),
            MDSItem("I2100", "Septicemia", MDSItemType.BOOLEAN),
            MDSItem("I2200", "Tuberculosis", MDSItemType.BOOLEAN),
            MDSItem("I2300", "Urinary Tract Infection (UTI) (LAST 30 DAYS)", MDSItemType.BOOLEAN),
            MDSItem("I2400", "Viral Hepatitis", MDSItemType.BOOLEAN),
            MDSItem("I2500", "Wound Infection (other than foot)", MDSItemType.BOOLEAN),
            MDSItem("I2900", "Diabetes Mellitus (DM)", MDSItemType.BOOLEAN),
            MDSItem("I3100", "Hyponatremia", MDSItemType.BOOLEAN),
            MDSItem("I3200", "Hyperkalemia", MDSItemType.BOOLEAN),
            MDSItem("I3300", "Hyperlipidemia", MDSItemType.BOOLEAN),
            MDSItem("I3400", "Thyroid Disorder", MDSItemType.BOOLEAN),
            MDSItem("I3700", "Arthritis", MDSItemType.BOOLEAN),
            MDSItem("I3800", "Osteoporosis", MDSItemType.BOOLEAN),
            MDSItem("I3900", "Hip Fracture", MDSItemType.BOOLEAN),
            MDSItem("I4000", "Other Fracture", MDSItemType.BOOLEAN),
            MDSItem("I4200", "Alzheimer's Disease", MDSItemType.BOOLEAN),
            MDSItem("I4300", "Aphasia", MDSItemType.BOOLEAN),
            MDSItem("I4400", "Cerebral Palsy", MDSItemType.BOOLEAN),
            MDSItem("I4500", "Cerebrovascular Accident (CVA), Transient Ischemic Attack (TIA), or Stroke", MDSItemType.BOOLEAN),
            MDSItem("I4800", "Non-Alzheimer's Dementia", MDSItemType.BOOLEAN),
            MDSItem("I4900", "Hemiplegia or Hemiparesis", MDSItemType.BOOLEAN),
            MDSItem("I5000", "Paraplegia", MDSItemType.BOOLEAN),
            MDSItem("I5100", "Quadriplegia", MDSItemType.BOOLEAN),
            MDSItem("I5200", "Multiple Sclerosis (MS)", MDSItemType.BOOLEAN),
            MDSItem("I5250", "Huntington's Disease", MDSItemType.BOOLEAN),
            MDSItem("I5300", "Parkinson's Disease", MDSItemType.BOOLEAN),
            MDSItem("I5350", "Tourette's Syndrome", MDSItemType.BOOLEAN),
            MDSItem("I5400", "Seizure Disorder or Epilepsy", MDSItemType.BOOLEAN),
            MDSItem("I5500", "Traumatic Brain Injury (TBI)", MDSItemType.BOOLEAN),
            MDSItem("I5600", "Malnutrition (protein or calorie) or at risk for malnutrition", MDSItemType.BOOLEAN),
            MDSItem("I5700", "Anxiety Disorder", MDSItemType.BOOLEAN),
            MDSItem("I5800", "Depression (other than bipolar)", MDSItemType.BOOLEAN),
            MDSItem("I5900", "Bipolar Disorder", MDSItemType.BOOLEAN),
            MDSItem("I5950", "Psychotic Disorder (other than schizophrenia)", MDSItemType.BOOLEAN),
            MDSItem("I6000", "Schizophrenia", MDSItemType.BOOLEAN),
            MDSItem("I6100", "Post Traumatic Stress Disorder (PTSD)", MDSItemType.BOOLEAN),
            MDSItem("I6200", "Asthma, Chronic Obstructive Pulmonary Disease (COPD), or Chronic Lung Disease", MDSItemType.BOOLEAN),
            MDSItem("I6300", "Respiratory Failure", MDSItemType.BOOLEAN),
            MDSItem("I6500", "Cataracts, Glaucoma, or Macular Degeneration", MDSItemType.BOOLEAN),
            MDSItem("I7900", "None of the above active diagnoses within the last 7 days", MDSItemType.BOOLEAN),
            MDSItem("I8000", "Additional active diagnoses", MDSItemType.TEXT,
                    description="Free-text active diagnoses not captured by specific items above."),
        ]
        self._add_section(sec)

    def _build_section_j(self) -> None:
        sec = MDSSection("J", "Health Conditions")
        sec.items = [
            MDSItem("J0300", "Pain Presence", MDSItemType.SELECT, [
                MDSResponseOption("0", "No pain"),
                MDSResponseOption("1", "Yes, pain present"),
                MDSResponseOption("9", "Unable to answer"),
            ]),
            MDSItem("J0400", "Pain Frequency", MDSItemType.SELECT, [
                MDSResponseOption("1", "Almost constantly"),
                MDSResponseOption("2", "Frequently"),
                MDSResponseOption("3", "Occasionally"),
                MDSResponseOption("4", "Rarely"),
            ]),
            MDSItem("J0500A", "Pain Effect on Function — sleep", MDSItemType.BOOLEAN),
            MDSItem("J0500B", "Pain Effect on Function — day activities", MDSItemType.BOOLEAN),
            MDSItem("J0600A", "Pain Intensity (0-10 numeric scale)", MDSItemType.INTEGER,
                    description="0 = no pain, 10 = worst possible pain."),
            MDSItem("J1100", "Shortness of Breath (dyspnea)", MDSItemType.MULTI, [
                MDSResponseOption("A", "Shortness of breath or trouble breathing with exertion"),
                MDSResponseOption("B", "Shortness of breath or trouble breathing when sitting at rest"),
                MDSResponseOption("C", "Shortness of breath or trouble breathing when lying flat"),
                MDSResponseOption("Z", "None of the above"),
            ]),
            MDSItem("J1300", "Current Tobacco Use", MDSItemType.BOOLEAN),
            MDSItem("J1400", "Prognosis — life expectancy of less than 6 months", MDSItemType.BOOLEAN),
            MDSItem("J1550A", "Problem Conditions — fever", MDSItemType.BOOLEAN),
            MDSItem("J1550B", "Problem Conditions — vomiting", MDSItemType.BOOLEAN),
            MDSItem("J1550C", "Problem Conditions — dehydration", MDSItemType.BOOLEAN),
            MDSItem("J1550D", "Problem Conditions — internal bleeding", MDSItemType.BOOLEAN),
            MDSItem("J1700B", "Falls — number of falls since admission/prior assessment with injury", MDSItemType.INTEGER),
            MDSItem("J1800", "Any Falls Since Admission / Prior Assessment", MDSItemType.BOOLEAN),
        ]
        self._add_section(sec)

    def _build_section_k(self) -> None:
        sec = MDSSection("K", "Swallowing / Nutritional Status")
        sec.items = [
            MDSItem("K0100", "Swallowing Disorder", MDSItemType.MULTI, [
                MDSResponseOption("A", "Loss of liquids/solids from mouth when eating or drinking"),
                MDSResponseOption("B", "Holding food in mouth/cheeks or residual food in mouth after meals"),
                MDSResponseOption("C", "Coughing or choking during meals or when swallowing medications"),
                MDSResponseOption("D", "Complaints of difficulty or pain with swallowing"),
                MDSResponseOption("Z", "None of the above"),
            ]),
            MDSItem("K0200A", "Height (in inches)", MDSItemType.INTEGER),
            MDSItem("K0200B", "Weight (in pounds)", MDSItemType.INTEGER),
            MDSItem("K0300", "Weight Loss", MDSItemType.SELECT, [
                MDSResponseOption("0", "No or unknown"),
                MDSResponseOption("1", "Yes, on physician-prescribed weight-loss regimen"),
                MDSResponseOption("2", "Yes, not on physician-prescribed weight-loss regimen"),
            ]),
            MDSItem("K0500", "Nutritional Approaches", MDSItemType.MULTI, [
                MDSResponseOption("A", "Parenteral/IV feeding"),
                MDSResponseOption("B", "Feeding tube — nasogastric or abdominal (PEG)"),
                MDSResponseOption("C", "Mechanically altered diet — require change in texture of food or liquids"),
                MDSResponseOption("D", "Therapeutic diet (e.g. diabetic, low salt, low fat)"),
                MDSResponseOption("Z", "None of the above"),
            ]),
        ]
        self._add_section(sec)

    def _build_section_m(self) -> None:
        sec = MDSSection("M", "Skin Conditions")
        sec.items = [
            MDSItem("M0100A", "Pressure Ulcer Risk — Braden scale or care-plan risk assessment", MDSItemType.BOOLEAN),
            MDSItem("M0210", "Unhealed Pressure Ulcers", MDSItemType.BOOLEAN,
                    description="Does the resident have one or more unhealed pressure ulcers?"),
            MDSItem("M0300A", "Stage 1 Pressure Ulcers", MDSItemType.INTEGER,
                    description="Number of stage 1 pressure ulcers."),
            MDSItem("M0300B1", "Stage 2 Pressure Ulcers — current number", MDSItemType.INTEGER),
            MDSItem("M0300C1", "Stage 3 Pressure Ulcers — current number", MDSItemType.INTEGER),
            MDSItem("M0300D1", "Stage 4 Pressure Ulcers — current number", MDSItemType.INTEGER),
            MDSItem("M0300E1", "Unstageable — non-removable dressing/device", MDSItemType.INTEGER),
            MDSItem("M0300F1", "Unstageable — slough/eschar", MDSItemType.INTEGER),
            MDSItem("M0300G1", "Unstageable — deep tissue injury", MDSItemType.INTEGER),
            MDSItem("M1040", "Other Ulcers, Wounds and Skin Problems", MDSItemType.MULTI, [
                MDSResponseOption("A", "Diabetic foot ulcer(s)"),
                MDSResponseOption("B", "Venous/arterial ulcer(s)"),
                MDSResponseOption("C", "Other open lesion(s) on the foot"),
                MDSResponseOption("D", "Open lesion(s) other than ulcers, rashes, cuts on the body"),
                MDSResponseOption("E", "Surgical wound(s)"),
                MDSResponseOption("F", "Burn(s) (second or third degree)"),
                MDSResponseOption("Z", "None of the above"),
            ]),
            MDSItem("M1200", "Skin and Ulcer Treatments", MDSItemType.MULTI, [
                MDSResponseOption("A", "Pressure reducing device for chair"),
                MDSResponseOption("B", "Pressure reducing device for bed"),
                MDSResponseOption("C", "Turning/repositioning program"),
                MDSResponseOption("D", "Nutrition or hydration intervention to manage skin problems"),
                MDSResponseOption("E", "Ulcer care"),
                MDSResponseOption("F", "Surgical wound care"),
                MDSResponseOption("G", "Application of nonsurgical dressings (with or without topical medications)"),
                MDSResponseOption("H", "Applications of ointments/medications other than to feet"),
                MDSResponseOption("I", "Application of medications to the feet"),
                MDSResponseOption("Z", "None of the above"),
            ]),
        ]
        self._add_section(sec)

    def _build_section_n(self) -> None:
        taking_options = [
            MDSResponseOption("1", "Is taking"),
            MDSResponseOption("2", "Indication noted"),
        ]
        sec = MDSSection("N", "Medications")
        n0415_items = [
            MDSItem("N0300", "Injections", MDSItemType.INTEGER,
                    description="Number of days in last 7 days that any injections were received."),
            MDSItem("N0350A", "Insulin — number of days injected", MDSItemType.INTEGER),
            MDSItem("N0350B", "Insulin — ordered change in insulin dose in last 7 days", MDSItemType.BOOLEAN),
            # N0415 — High-Risk Drug Classes: Use and Indication (check all that apply)
            # Column 1 = Is taking (code "1"); Column 2 = Indication noted (code "2")
            MDSItem("N0415A", "High-Risk Drug — Antipsychotic", MDSItemType.MULTI, taking_options,
                    description="e.g. haloperidol, quetiapine, olanzapine, risperidone"),
            MDSItem("N0415B", "High-Risk Drug — Antianxiety", MDSItemType.MULTI, taking_options,
                    description="e.g. lorazepam, alprazolam, diazepam"),
            MDSItem("N0415C", "High-Risk Drug — Antidepressant", MDSItemType.MULTI, taking_options,
                    description="e.g. sertraline, fluoxetine, citalopram"),
            MDSItem("N0415D", "High-Risk Drug — Hypnotic", MDSItemType.MULTI, taking_options,
                    description="e.g. zolpidem, eszopiclone"),
            MDSItem("N0415E", "High-Risk Drug — Anticoagulant", MDSItemType.MULTI, taking_options,
                    description="e.g. warfarin, heparin, low-molecular weight heparin, apixaban, rivaroxaban"),
            MDSItem("N0415F", "High-Risk Drug — Antibiotic", MDSItemType.MULTI, taking_options,
                    description="e.g. vancomycin, piperacillin, metronidazole"),
            MDSItem("N0415G", "High-Risk Drug — Diuretic", MDSItemType.MULTI, taking_options,
                    description="e.g. furosemide, torsemide, bumetanide"),
            MDSItem("N0415H", "High-Risk Drug — Opioid", MDSItemType.MULTI, taking_options,
                    description="e.g. morphine, oxycodone, hydrocodone, fentanyl, tramadol"),
            MDSItem("N0415I", "High-Risk Drug — Antiplatelet", MDSItemType.MULTI, taking_options,
                    description="e.g. aspirin, clopidogrel, ticagrelor, prasugrel"),
            MDSItem("N0415J", "High-Risk Drug — Hypoglycemic", MDSItemType.MULTI, taking_options,
                    description="Including insulin; e.g. metformin, glipizide, insulin"),
            MDSItem("N0415Z", "High-Risk Drug — None of the above", MDSItemType.BOOLEAN),
            MDSItem("N2001", "Drug Regimen Review — drug regimen reviewed", MDSItemType.BOOLEAN),
        ]

        # N0450 — Antipsychotic Medication Review (separate subgroup from N0415)
        n0450_items = [
            MDSItem("N0450A", "Antipsychotic Medication Review — received since admission/prior OBRA?", MDSItemType.SELECT, [
                MDSResponseOption("0", "No — antipsychotics were not received"),
                MDSResponseOption("1", "Yes — received on a routine basis only"),
                MDSResponseOption("2", "Yes — received on a PRN basis only"),
                MDSResponseOption("3", "Yes — received on a routine and PRN basis"),
            ]),
            MDSItem("N0450B", "Antipsychotic Medication Review — gradual dose reduction (GDR) attempted", MDSItemType.BOOLEAN),
            MDSItem("N0450C", "Antipsychotic Medication Review — date of last attempted GDR", MDSItemType.DATE),
            MDSItem("N0450D", "Antipsychotic Medication Review — physician documented GDR clinically contraindicated", MDSItemType.BOOLEAN),
            MDSItem("N0450E", "Antipsychotic Medication Review — date physician documented GDR clinically contraindicated", MDSItemType.DATE),
            MDSItem("N2003", "Medication Follow-up — contact and action completed by next calendar day", MDSItemType.BOOLEAN),
            MDSItem("N2005", "Medication Intervention — contact/action completed each time issues identified", MDSItemType.BOOLEAN),
        ]

        sec.items = [*n0415_items, *n0450_items]
        self._add_section(sec)

    def _build_section_o(self) -> None:
        sec = MDSSection("O", "Special Treatments, Procedures, and Programs")
        sec.items = [
            # O0110 — Special Treatments, Procedures, and Programs
            # Cancer Treatments
            MDSItem("O0110A1",  "Chemotherapy", MDSItemType.BOOLEAN),
            MDSItem("O0110A2",  "Chemotherapy — IV", MDSItemType.BOOLEAN),
            MDSItem("O0110A3",  "Chemotherapy — Oral", MDSItemType.BOOLEAN),
            MDSItem("O0110A10", "Chemotherapy — Other", MDSItemType.BOOLEAN),
            MDSItem("O0110B1",  "Radiation", MDSItemType.BOOLEAN),
            # Respiratory Treatments
            MDSItem("O0110C1",  "Oxygen therapy", MDSItemType.BOOLEAN),
            MDSItem("O0110C2",  "Oxygen therapy — Continuous", MDSItemType.BOOLEAN),
            MDSItem("O0110C3",  "Oxygen therapy — Intermittent", MDSItemType.BOOLEAN),
            MDSItem("O0110C4",  "Oxygen therapy — High-concentration", MDSItemType.BOOLEAN),
            MDSItem("O0110D1",  "Suctioning", MDSItemType.BOOLEAN),
            MDSItem("O0110D2",  "Suctioning — Scheduled", MDSItemType.BOOLEAN),
            MDSItem("O0110D3",  "Suctioning — As needed", MDSItemType.BOOLEAN),
            MDSItem("O0110E1",  "Tracheostomy care", MDSItemType.BOOLEAN),
            MDSItem("O0110F1",  "Invasive Mechanical Ventilator (ventilator or respirator)", MDSItemType.BOOLEAN),
            MDSItem("O0110G1",  "Non-invasive Mechanical Ventilator", MDSItemType.BOOLEAN),
            MDSItem("O0110G2",  "Non-invasive Mechanical Ventilator — BiPAP", MDSItemType.BOOLEAN),
            MDSItem("O0110G3",  "Non-invasive Mechanical Ventilator — CPAP", MDSItemType.BOOLEAN),
            # Other
            MDSItem("O0110H1",  "IV Medications", MDSItemType.BOOLEAN),
            MDSItem("O0110H2",  "IV Medications — Vasoactive", MDSItemType.BOOLEAN),
            MDSItem("O0110H3",  "IV Medications — Antibiotics", MDSItemType.BOOLEAN),
            MDSItem("O0110H4",  "IV Medications — Anticoagulant", MDSItemType.BOOLEAN),
            MDSItem("O0110H10", "IV Medications — Other", MDSItemType.BOOLEAN),
            MDSItem("O0110I1",  "Transfusions", MDSItemType.BOOLEAN),
            MDSItem("O0110J1",  "Dialysis", MDSItemType.BOOLEAN),
            MDSItem("O0110J2",  "Dialysis — Hemodialysis", MDSItemType.BOOLEAN),
            MDSItem("O0110J3",  "Dialysis — Peritoneal dialysis", MDSItemType.BOOLEAN),
            MDSItem("O0110K1",  "Hospice care", MDSItemType.BOOLEAN),
            MDSItem("O0110M1",  "Isolation or quarantine for active infectious disease", MDSItemType.BOOLEAN),
            MDSItem("O0110O1",  "IV Access", MDSItemType.BOOLEAN),
            MDSItem("O0110O2",  "IV Access — Peripheral", MDSItemType.BOOLEAN),
            MDSItem("O0110O3",  "IV Access — Midline", MDSItemType.BOOLEAN),
            MDSItem("O0110O4",  "IV Access — Central (e.g., PICC, tunneled, port)", MDSItemType.BOOLEAN),
            MDSItem("O0110Z1",  "None of the above", MDSItemType.BOOLEAN),
            MDSItem("O0250A", "Influenza Vaccine — received in this facility?", MDSItemType.SELECT, [
                MDSResponseOption("0", "No"),
                MDSResponseOption("1", "Yes"),
            ]),
            MDSItem("O0250B", "Influenza Vaccine — date received", MDSItemType.DATE),
            MDSItem("O0250C", "Influenza Vaccine — reason not received", MDSItemType.SELECT, [
                MDSResponseOption("1", "Resident not in this facility during influenza vaccination season"),
                MDSResponseOption("2", "Received outside of this facility"),
                MDSResponseOption("3", "Not eligible — medical contraindication"),
                MDSResponseOption("4", "Offered and declined"),
                MDSResponseOption("5", "Not offered"),
                MDSResponseOption("6", "Inability to obtain due to declared shortage"),
                MDSResponseOption("9", "None of the above"),
            ]),
            MDSItem("O0300A", "Pneumococcal Vaccine — up to date?", MDSItemType.SELECT, [
                MDSResponseOption("0", "No"),
                MDSResponseOption("1", "Yes"),
            ]),
            MDSItem("O0300B", "Pneumococcal Vaccine — reason not received", MDSItemType.SELECT, [
                MDSResponseOption("1", "Not eligible — medical contraindication"),
                MDSResponseOption("2", "Offered and declined"),
                MDSResponseOption("3", "Not offered"),
            ]),
            MDSItem("O0400A1", "Speech-Language Pathology and Audiology Services — Individual minutes", MDSItemType.INTEGER),
            MDSItem("O0400A2", "Speech-Language Pathology and Audiology Services — Concurrent minutes", MDSItemType.INTEGER),
            MDSItem("O0400A3", "Speech-Language Pathology and Audiology Services — Group minutes", MDSItemType.INTEGER),
            MDSItem("O0400A3A", "Speech-Language Pathology and Audiology Services — Co-treatment minutes", MDSItemType.INTEGER),
            MDSItem("O0400A4", "Speech-Language Pathology and Audiology Services — Days (at least 15 minutes/day)", MDSItemType.INTEGER),
            MDSItem("O0400A5", "Speech-Language Pathology and Audiology Services — Therapy start date", MDSItemType.DATE),
            MDSItem("O0400A6", "Speech-Language Pathology and Audiology Services — Therapy end date", MDSItemType.DATE),
            MDSItem("O0400B1", "Occupational Therapy — Individual minutes", MDSItemType.INTEGER),
            MDSItem("O0400B2", "Occupational Therapy — Concurrent minutes", MDSItemType.INTEGER),
            MDSItem("O0400B3", "Occupational Therapy — Group minutes", MDSItemType.INTEGER),
            MDSItem("O0400B3A", "Occupational Therapy — Co-treatment minutes", MDSItemType.INTEGER),
            MDSItem("O0400B4", "Occupational Therapy — Days (at least 15 minutes/day)", MDSItemType.INTEGER),
            MDSItem("O0400B5", "Occupational Therapy — Therapy start date", MDSItemType.DATE),
            MDSItem("O0400B6", "Occupational Therapy — Therapy end date", MDSItemType.DATE),
            MDSItem("O0400C1", "Physical Therapy — Individual minutes", MDSItemType.INTEGER),
            MDSItem("O0400C2", "Physical Therapy — Concurrent minutes", MDSItemType.INTEGER),
            MDSItem("O0400C3", "Physical Therapy — Group minutes", MDSItemType.INTEGER),
            MDSItem("O0400C3A", "Physical Therapy — Co-treatment minutes", MDSItemType.INTEGER),
            MDSItem("O0400C4", "Physical Therapy — Days (at least 15 minutes/day)", MDSItemType.INTEGER),
            MDSItem("O0400C5", "Physical Therapy — Therapy start date", MDSItemType.DATE),
            MDSItem("O0400C6", "Physical Therapy — Therapy end date", MDSItemType.DATE),
            MDSItem("O0400D1", "Respiratory Therapy — Total minutes", MDSItemType.INTEGER),
            MDSItem("O0400D2", "Respiratory Therapy — Days (at least 15 minutes/day)", MDSItemType.INTEGER),
            MDSItem("O0400E1", "Psychological Therapy — Total minutes", MDSItemType.INTEGER),
            MDSItem("O0400E2", "Psychological Therapy — Days (at least 15 minutes/day)", MDSItemType.INTEGER),
            MDSItem("O0400F1", "Recreational Therapy — Total minutes", MDSItemType.INTEGER),
            MDSItem("O0400F2", "Recreational Therapy — Days (at least 15 minutes/day)", MDSItemType.INTEGER),
            MDSItem("O0420", "Distinct Calendar Days of Therapy", MDSItemType.INTEGER),
            MDSItem("O0425A1", "Part A Speech-Language Pathology and Audiology Services — Individual minutes", MDSItemType.INTEGER),
            MDSItem("O0425A2", "Part A Speech-Language Pathology and Audiology Services — Concurrent minutes", MDSItemType.INTEGER),
            MDSItem("O0425A3", "Part A Speech-Language Pathology and Audiology Services — Group minutes", MDSItemType.INTEGER),
            MDSItem("O0425A4", "Part A Speech-Language Pathology and Audiology Services — Co-treatment minutes", MDSItemType.INTEGER),
            MDSItem("O0425A5", "Part A Speech-Language Pathology and Audiology Services — Days", MDSItemType.INTEGER),
            MDSItem("O0425B1", "Part A Occupational Therapy — Individual minutes", MDSItemType.INTEGER),
            MDSItem("O0425B2", "Part A Occupational Therapy — Concurrent minutes", MDSItemType.INTEGER),
            MDSItem("O0425B3", "Part A Occupational Therapy — Group minutes", MDSItemType.INTEGER),
            MDSItem("O0425B4", "Part A Occupational Therapy — Co-treatment minutes", MDSItemType.INTEGER),
            MDSItem("O0425B5", "Part A Occupational Therapy — Days", MDSItemType.INTEGER),
            MDSItem("O0425C1", "Part A Physical Therapy — Individual minutes", MDSItemType.INTEGER),
            MDSItem("O0425C2", "Part A Physical Therapy — Concurrent minutes", MDSItemType.INTEGER),
            MDSItem("O0425C3", "Part A Physical Therapy — Group minutes", MDSItemType.INTEGER),
            MDSItem("O0425C4", "Part A Physical Therapy — Co-treatment minutes", MDSItemType.INTEGER),
            MDSItem("O0425C5", "Part A Physical Therapy — Days", MDSItemType.INTEGER),
            MDSItem("O0430", "Distinct Calendar Days of Part A Therapy", MDSItemType.INTEGER),
            MDSItem("O0500A", "Restorative Nursing Programs — Range of motion (passive), number of days", MDSItemType.INTEGER),
            MDSItem("O0500B", "Restorative Nursing Programs — Range of motion (active), number of days", MDSItemType.INTEGER),
            MDSItem("O0500C", "Restorative Nursing Programs — Splint or brace assistance, number of days", MDSItemType.INTEGER),
            MDSItem("O0500D", "Restorative Nursing Programs — Bed mobility, number of days", MDSItemType.INTEGER),
            MDSItem("O0500E", "Restorative Nursing Programs — Transfer, number of days", MDSItemType.INTEGER),
            MDSItem("O0500F", "Restorative Nursing Programs — Walking, number of days", MDSItemType.INTEGER),
            MDSItem("O0500G", "Restorative Nursing Programs — Dressing and/or grooming, number of days", MDSItemType.INTEGER),
            MDSItem("O0500H", "Restorative Nursing Programs — Eating and/or swallowing, number of days", MDSItemType.INTEGER),
            MDSItem("O0500I", "Restorative Nursing Programs — Amputation/prostheses care, number of days", MDSItemType.INTEGER),
            MDSItem("O0500J", "Restorative Nursing Programs — Communication, number of days", MDSItemType.INTEGER),
        ]
        self._add_section(sec)


# ---------------------------------------------------------------------------
# MDS Assessment — holds extracted field values for a single patient note
# ---------------------------------------------------------------------------


@dataclass
class MDSAssessment:
    """
    Holds the MDS 3.0 field values extracted from a single discharge note.

    Attributes
    ----------
    note_id : str
        Identifier of the source discharge note.
    subject_id : str
        Patient identifier.
    hadm_id : str
        Hospital admission identifier.
    fields : dict
        Mapping from MDS item_id to the extracted value.
    confidence : dict
        Optional mapping from item_id to a confidence score (0.0–1.0).
    """

    note_id: str
    subject_id: str
    hadm_id: str
    fields: Dict[str, Any] = field(default_factory=dict)
    confidence: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def set_field(self, item_id: str, value: Any, confidence: float = 1.0) -> None:
        """Set the extracted value for *item_id*."""
        self.fields[item_id] = value
        self.confidence[item_id] = confidence

    def get_field(self, item_id: str) -> Optional[Any]:
        """Return the extracted value for *item_id*, or ``None``."""
        return self.fields.get(item_id)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the assessment to a plain dictionary."""
        return {
            "note_id": self.note_id,
            "subject_id": self.subject_id,
            "hadm_id": self.hadm_id,
            "fields": self.fields,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }

    def to_flat_dict(self) -> Dict[str, Any]:
        """Return a flat dict suitable for a single CSV row."""
        row: Dict[str, Any] = {
            "note_id": self.note_id,
            "subject_id": self.subject_id,
            "hadm_id": self.hadm_id,
        }
        row.update(self.fields)
        return row
