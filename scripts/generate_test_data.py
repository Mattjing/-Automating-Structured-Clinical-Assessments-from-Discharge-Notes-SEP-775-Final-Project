"""
generate_test_data.py
=====================
Generates a labeled synthetic test dataset for evaluating MDS 3.0 extraction.

15 clinically distinct patients are created, covering a broad range of MDS 3.0
Section I (diagnoses), N (medications), and O (treatments) fields.

Output files written to data/test/:
  discharge_test.csv           – unstructured discharge notes in MIMIC-IV format
                                  (note_id, subject_id, hadm_id, note_type,
                                   charttime, storetime, text)
  extraction_ground_truth.json – per-note ground truth in the exact format
                                  that MDSMapper.map() receives as its
                                  `extraction` argument: MDS item IDs are
                                  top-level keys alongside note_id /
                                  subject_id / hadm_id, with a nested
                                  "confidence" dict (all values = 1.0).
                                  Pass note_id / subject_id / hadm_id to
                                  mapper.map() separately and the remaining
                                  keys as the extraction dict.

Run from the project root:
  python scripts/generate_test_data.py
"""

import csv
import json
import os
import textwrap

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "test")

# ---------------------------------------------------------------------------
# Discharge note definitions
# ---------------------------------------------------------------------------
# Each tuple: (note_id, subject_id, hadm_id, sex, service, full_text)
# Notes are written in MIMIC-IV discharge summary style with ___ placeholders.

DISCHARGE_NOTES = [
    # ------------------------------------------------------------------ #
    # 1  Atrial Fibrillation · CHF · HTN · T2DM · COPD                   #
    #    Meds: warfarin (anticoag), furosemide (diuretic), metformin      #
    #    Procedures: peripheral IV                                         #
    # ------------------------------------------------------------------ #
    (
        "TEST0001-DS-01", "90000001", "91000001", "F", "MEDICINE",
        textwrap.dedent("""\
            Name: ___ Unit No: ___
            Admission Date: ___ Discharge Date: ___
            Date of Birth: ___ Sex: F
            Service: MEDICINE
            Allergies: No Known Allergies / Adverse Drug Reactions
            Attending: ___

            Chief Complaint:
            Shortness of breath and lower extremity edema.

            Major Surgical or Invasive Procedure:
            Peripheral intravenous (IV) access placed.

            History of Present Illness:
            ___ year-old female with atrial fibrillation on warfarin, chronic
            systolic heart failure (EF 30-35%), hypertension, type 2 diabetes
            mellitus, and chronic obstructive pulmonary disease (COPD) who
            presents with worsening dyspnea and bilateral lower extremity edema.
            Weight gain of 8 lbs over one week. Patient ran out of furosemide.

            Past Medical History:
            1. Atrial fibrillation, persistent -- on warfarin (anticoagulant)
            2. Systolic heart failure, EF 30-35%
            3. Hypertension
            4. Type 2 diabetes mellitus -- managed with metformin
            5. Chronic obstructive pulmonary disease (COPD) -- albuterol PRN

            Pertinent Results:
            BNP 2840 (elevated). INR 2.4 (therapeutic on warfarin).
            ECG: Atrial fibrillation, rate 88.
            CXR: Cardiomegaly, bilateral pleural effusions.

            Brief Hospital Course:
            Acute decompensated systolic heart failure treated with IV furosemide
            (diuretic); transitioned to oral furosemide at discharge.
            Atrial fibrillation rate-controlled; warfarin continued (anticoagulant).
            Metformin held during admission, restarted at discharge (hypoglycemic).
            COPD stable; albuterol PRN continued.

            Discharge Medications:
            1. warfarin 5 mg PO daily (anticoagulant)
            2. furosemide 80 mg PO daily (diuretic)
            3. metformin 1000 mg PO BID (hypoglycemic)
            4. metoprolol succinate 50 mg PO daily
            5. albuterol inhaler PRN

            Discharge Diagnoses:
            Primary: Acute decompensated systolic heart failure
            Secondary: Atrial fibrillation, Hypertension, Type 2 diabetes mellitus,
            Chronic obstructive pulmonary disease (COPD)
        """),
    ),
    # ------------------------------------------------------------------ #
    # 2  Ischemic Stroke (CVA) · Post-stroke Depression · HTN            #
    #    Meds: aspirin + clopidogrel (antiplatelet), sertraline (antidep) #
    #    Procedures: PT 60 min/day · OT 45 min/day · SLP 30 min/day      #
    # ------------------------------------------------------------------ #
    (
        "TEST0002-DS-01", "90000002", "91000002", "M", "NEUROLOGY",
        textwrap.dedent("""\
            Name: ___ Unit No: ___
            Admission Date: ___ Discharge Date: ___
            Date of Birth: ___ Sex: M
            Service: NEUROLOGY
            Allergies: Penicillin (rash)
            Attending: ___

            Chief Complaint:
            Acute left-sided weakness and expressive aphasia.

            Major Surgical or Invasive Procedure:
            IV tPA administration.

            History of Present Illness:
            ___ year-old male with hypertension presenting with acute right MCA
            ischemic stroke (cerebrovascular accident). NIHSS 12 on arrival.
            CT head: no hemorrhage. CTA: right MCA M1 occlusion. IV tPA given.
            MRI confirmed right MCA territory ischemic infarct.

            Past Medical History:
            1. Hypertension -- lisinopril
            2. Hyperlipidemia -- atorvastatin

            Brief Hospital Course:
            #Ischemic stroke (CVA): Aspirin 325 mg started day 2 post-tPA.
            Clopidogrel 75 mg added for dual antiplatelet therapy. Repeat MRI
            showed no hemorrhagic transformation. Residual left leg weakness
            and expressive aphasia persisted.

            #Post-stroke depression: Psychiatry consulted. Major depressive
            disorder diagnosed. Sertraline 50 mg daily initiated (antidepressant).

            #Rehabilitation:
            Physical therapy (PT): 60 minutes per day for 5 days.
            Occupational therapy (OT): 45 minutes per day for 5 days.
            Speech-language pathology (SLP): 30 minutes per day for 5 days.

            Discharge Medications:
            1. aspirin 81 mg PO daily (antiplatelet)
            2. clopidogrel 75 mg PO daily (antiplatelet)
            3. sertraline 50 mg PO daily (antidepressant -- post-stroke depression)
            4. lisinopril 10 mg PO daily
            5. atorvastatin 40 mg PO daily

            Discharge Diagnoses:
            Primary: Acute ischemic stroke, right MCA territory (cerebrovascular accident)
            Secondary: Hypertension, Post-stroke major depressive disorder,
            Expressive aphasia, Left hemiparesis
        """),
    ),
    # ------------------------------------------------------------------ #
    # 3  Non-small cell Lung Cancer · COPD · HTN                         #
    #    Meds: carboplatin IV (chemo), morphine (opioid), vancomycin (abx)#
    #    Procedures: IV chemo · radiation · continuous O2 · PICC · IV abx#
    # ------------------------------------------------------------------ #
    (
        "TEST0003-DS-01", "90000003", "91000003", "M", "ONCOLOGY",
        textwrap.dedent("""\
            Name: ___ Unit No: ___
            Admission Date: ___ Discharge Date: ___
            Date of Birth: ___ Sex: M
            Service: ONCOLOGY
            Allergies: Sulfa drugs
            Attending: ___

            Chief Complaint:
            Cycle 3 IV chemotherapy and management of MRSA pneumonia with pain.

            Major Surgical or Invasive Procedure:
            PICC line placed for IV access and chemotherapy administration.

            History of Present Illness:
            ___ year-old male with non-small cell lung cancer (NSCLC, stage IIIB
            adenocarcinoma) on concurrent chemoradiation, chronic obstructive
            pulmonary disease (COPD) on home oxygen, and hypertension. Admitted
            for cycle 3 of intravenous (IV) carboplatin/pemetrexed and radiation
            fractions 16-20, and management of MRSA pneumonia and pain escalation.

            Past Medical History:
            1. Non-small cell lung cancer (NSCLC), stage IIIB -- active chemoradiation
            2. Chronic obstructive pulmonary disease (COPD) -- home O2 2L/min continuous
            3. Hypertension -- amlodipine

            Brief Hospital Course:
            #Cancer treatment: Cycle 3 intravenous (IV) carboplatin and pemetrexed
            administered via PICC line. Radiation oncology delivered fractions 16-20.

            #MRSA pneumonia: Treated with intravenous vancomycin (IV antibiotic) for
            7 days via PICC line. Fever and leukocytosis resolved.

            #COPD / Oxygen: Required continuous oxygen therapy at 4 L/min throughout
            admission. Discharged home on continuous supplemental oxygen.

            #Pain: Managed with morphine extended release 30 mg BID and IV morphine
            2 mg PRN (opioid). Oral morphine continued at discharge.

            Discharge Medications:
            1. morphine ER 30 mg PO BID (opioid analgesic)
            2. amlodipine 5 mg PO daily
            3. tiotropium inhaler daily
            4. supplemental oxygen 3 L/min continuous via nasal cannula

            Discharge Diagnoses:
            Primary: Non-small cell lung cancer (NSCLC), stage IIIB, active treatment
            Secondary: MRSA pneumonia (treated), COPD, Hypertension
        """),
    ),
    # ------------------------------------------------------------------ #
    # 4  Alzheimer's Disease · Major Depression · HTN · CKD              #
    #    Meds: sertraline (antidep), lorazepam (antianxiety), furosemide  #
    #    Procedures: none                                                  #
    # ------------------------------------------------------------------ #
    (
        "TEST0004-DS-01", "90000004", "91000004", "F", "MEDICINE",
        textwrap.dedent("""\
            Name: ___ Unit No: ___
            Admission Date: ___ Discharge Date: ___
            Date of Birth: ___ Sex: F
            Service: MEDICINE
            Allergies: Codeine (nausea)
            Attending: ___

            Chief Complaint:
            Acute confusion and behavioral agitation in patient with dementia.

            Major Surgical or Invasive Procedure:
            None.

            History of Present Illness:
            ___ year-old female with Alzheimer's disease (moderate stage), major
            depressive disorder, anxiety, hypertension, and chronic kidney disease
            (CKD stage 3 / renal insufficiency, baseline creatinine 1.8-2.0).
            Presents with increased agitation, confusion, and worsening depressive
            symptoms (tearfulness, withdrawal, anorexia) over 3 days.

            Past Medical History:
            1. Alzheimer's disease, moderate stage -- donepezil
            2. Major depressive disorder -- sertraline
            3. Anxiety disorder -- lorazepam PRN
            4. Hypertension -- lisinopril, furosemide
            5. Chronic kidney disease (CKD) stage 3 / renal insufficiency --
               creatinine baseline 1.8-2.0

            Pertinent Results:
            Cr 1.9 (at baseline). UA: no UTI. CT head: cortical/hippocampal
            atrophy consistent with Alzheimer's disease, no acute process.

            Brief Hospital Course:
            #Alzheimer's disease: Workup negative for reversible delirium causes.
            Lorazepam 0.5 mg PRN for acute agitation (antianxiety). Donepezil
            continued for Alzheimer's disease.

            #Major depressive disorder: Sertraline increased from 50 mg to 100 mg
            daily (antidepressant). Psychiatry consulted and agreed.

            #Hypertension: Furosemide 20 mg PO daily continued (diuretic).

            #Renal insufficiency (CKD): Creatinine stable at 1.9 (baseline).

            Discharge Medications:
            1. donepezil 10 mg PO bedtime
            2. sertraline 100 mg PO daily (antidepressant)
            3. lorazepam 0.5 mg PO PRN agitation (antianxiety)
            4. lisinopril 5 mg PO daily
            5. furosemide 20 mg PO daily (diuretic)

            Discharge Diagnoses:
            Primary: Delirium on Alzheimer's dementia (moderate stage)
            Secondary: Major depressive disorder, Anxiety, Hypertension,
            Chronic kidney disease (renal insufficiency), stage 3
        """),
    ),
    # ------------------------------------------------------------------ #
    # 5  ESRD on Hemodialysis · CAD · HTN · Insulin-Dependent DM         #
    #    Meds: insulin SC x7 days, aspirin (antiplatelet), heparin        #
    #    Procedures: hemodialysis · central line / PICC                   #
    # ------------------------------------------------------------------ #
    (
        "TEST0005-DS-01", "90000005", "91000005", "M", "NEPHROLOGY",
        textwrap.dedent("""\
            Name: ___ Unit No: ___
            Admission Date: ___ Discharge Date: ___
            Date of Birth: ___ Sex: M
            Service: NEPHROLOGY
            Allergies: IV contrast (AKI)
            Attending: ___

            Chief Complaint:
            AV fistula thrombosis and hyperkalemia -- missed dialysis sessions.

            Major Surgical or Invasive Procedure:
            Tunneled dialysis central venous catheter (PICC / right IJ) placed
            for temporary hemodialysis access.

            History of Present Illness:
            ___ year-old male with end-stage renal disease (ESRD) on maintenance
            hemodialysis (3x/week), coronary artery disease (CAD, s/p CABG),
            hypertension, and insulin-dependent diabetes mellitus presenting with
            AV fistula thrombosis and missed dialysis sessions.

            Past Medical History:
            1. End-stage renal disease (ESRD) -- hemodialysis 3x/week via AV fistula
            2. Coronary artery disease (CAD) -- s/p CABG, stable angina
            3. Hypertension
            4. Insulin-dependent diabetes mellitus -- insulin glargine 30 units SQ
               nightly + insulin lispro sliding scale SQ with meals

            Brief Hospital Course:
            #ESRD / AVF thrombosis: Tunneled central venous catheter placed for
            temporary dialysis access. Hemodialysis performed on days 1, 3, 5,
            and 7 of the admission (4 sessions total). Heparin administered IV
            during each hemodialysis session (anticoagulant). IR thrombectomy
            restored AVF function; catheter removed prior to discharge.

            #Coronary artery disease: Aspirin 81 mg daily continued (antiplatelet).
            No acute coronary syndrome.

            #Insulin-dependent diabetes: Subcutaneous insulin injections administered
            every day of the 7-day admission:
            - Insulin glargine 30 units SQ at bedtime nightly (7 days)
            - Insulin lispro SQ with meals per sliding scale (7 days)
            Total of 7 days of insulin injections.

            Discharge Medications:
            1. insulin glargine 30 units SQ nightly (hypoglycemic)
            2. insulin lispro per sliding scale SQ with meals (hypoglycemic)
            3. aspirin 81 mg PO daily (antiplatelet)
            4. amlodipine 10 mg PO daily
            5. sevelamer 800 mg PO TID with meals

            Discharge Diagnoses:
            Primary: ESRD with AV fistula thrombosis
            Secondary: Coronary artery disease (CAD), Hypertension,
            Insulin-dependent diabetes mellitus, Hyperkalemia (resolved)
        """),
    ),
    # ------------------------------------------------------------------ #
    # 6  Bipolar Disorder · PTSD · HTN                                   #
    #    Meds: quetiapine (antipsychotic), sertraline (antidep)           #
    #    Procedures: none                                                  #
    # ------------------------------------------------------------------ #
    (
        "TEST0006-DS-01", "90000006", "91000006", "F", "PSYCHIATRY",
        textwrap.dedent("""\
            Name: ___ Unit No: ___
            Admission Date: ___ Discharge Date: ___
            Date of Birth: ___ Sex: F
            Service: PSYCHIATRY
            Allergies: No Known Allergies
            Attending: ___

            Chief Complaint:
            Manic episode with psychotic features and suicidal ideation.

            Major Surgical or Invasive Procedure:
            None.

            History of Present Illness:
            ___ year-old female with bipolar disorder (type I), post-traumatic
            stress disorder (PTSD), and hypertension presenting with acute manic
            episode with psychotic features and passive suicidal ideation.
            Patient had been non-compliant with quetiapine for 2 weeks prior
            to admission. She endorses racing thoughts, decreased sleep, and
            intrusive flashbacks consistent with PTSD.

            Past Medical History:
            1. Bipolar disorder, type I -- on quetiapine and valproic acid
            2. Post-traumatic stress disorder (PTSD) -- on sertraline
            3. Hypertension -- on lisinopril

            Brief Hospital Course:
            #Bipolar disorder, manic episode with psychosis: Quetiapine restarted
            at 400 mg nightly (antipsychotic, routine scheduled dosing).
            Valproic acid level subtherapeutic; dose increased.

            #PTSD: Sertraline 100 mg PO daily continued (antidepressant).
            Trauma-informed therapy initiated.

            #Hypertension: Lisinopril continued, BP well-controlled.

            Discharge Medications:
            1. quetiapine 400 mg PO nightly (antipsychotic -- bipolar disorder,
               routine scheduled)
            2. valproic acid 750 mg PO BID
            3. sertraline 100 mg PO daily (antidepressant -- PTSD)
            4. lisinopril 10 mg PO daily

            Discharge Diagnoses:
            Primary: Bipolar disorder, type I, manic episode with psychotic features
            Secondary: Post-traumatic stress disorder (PTSD), Hypertension
        """),
    ),
    # ------------------------------------------------------------------ #
    # 7  Pneumonia · DVT/PE · HTN                                        #
    #    Meds: heparin (anticoag), piperacillin-tazo (antibiotic)         #
    #    Procedures: intermittent O2 · IV antibiotics · peripheral IV     #
    # ------------------------------------------------------------------ #
    (
        "TEST0007-DS-01", "90000007", "91000007", "M", "MEDICINE",
        textwrap.dedent("""\
            Name: ___ Unit No: ___
            Admission Date: ___ Discharge Date: ___
            Date of Birth: ___ Sex: M
            Service: MEDICINE
            Allergies: No Known Allergies
            Attending: ___

            Chief Complaint:
            Fever, productive cough, and right leg swelling.

            Major Surgical or Invasive Procedure:
            Peripheral intravenous (IV) access placed.

            History of Present Illness:
            ___ year-old male with hypertension presenting with community-acquired
            pneumonia and deep venous thrombosis (DVT) of the right lower extremity
            with associated pulmonary embolism (PE). CXR: right lower lobe
            consolidation. CTA chest: bilateral pulmonary emboli.
            Doppler: right popliteal DVT.

            Past Medical History:
            1. Hypertension -- amlodipine

            Brief Hospital Course:
            #Pneumonia: Treated with intravenous piperacillin-tazobactam (IV
            antibiotic) for 5 days then transitioned to oral amoxicillin-clavulanate.
            Intermittent supplemental oxygen therapy used as needed (SpO2 maintained
            above 92%). Fever resolved by day 3.

            #DVT / Pulmonary Embolism: Heparin infusion started (intravenous
            anticoagulant). Transitioned to apixaban at discharge. Outpatient
            DVT/PE follow-up arranged.

            Discharge Medications:
            1. apixaban 10 mg PO BID x7 days then 5 mg BID (anticoagulant)
            2. amoxicillin-clavulanate 875-125 mg PO BID x5 days (antibiotic)
            3. amlodipine 5 mg PO daily

            Discharge Diagnoses:
            Primary: Community-acquired pneumonia
            Secondary: Deep venous thrombosis (DVT) with pulmonary embolism (PE),
            Hypertension
        """),
    ),
    # ------------------------------------------------------------------ #
    # 8  Severe COPD Exacerbation · BiPAP · HTN                          #
    #    Meds: azithromycin (antibiotic), furosemide (diuretic)           #
    #    Procedures: BiPAP (non-invasive ventilation) · continuous O2     #
    # ------------------------------------------------------------------ #
    (
        "TEST0008-DS-01", "90000008", "91000008", "M", "MEDICINE",
        textwrap.dedent("""\
            Name: ___ Unit No: ___
            Admission Date: ___ Discharge Date: ___
            Date of Birth: ___ Sex: M
            Service: MEDICINE
            Allergies: No Known Drug Allergies
            Attending: ___

            Chief Complaint:
            Worsening dyspnea and increased sputum production.

            Major Surgical or Invasive Procedure:
            BiPAP (bilevel positive airway pressure) non-invasive mechanical
            ventilation applied intermittently.

            History of Present Illness:
            ___ year-old male with severe COPD (FEV1 38% predicted) and
            hypertension presenting with acute exacerbation of COPD. Increased
            sputum production, worsening dyspnea, and respiratory acidosis
            (pH 7.28, pCO2 62). SpO2 82% on room air.

            Past Medical History:
            1. Chronic obstructive pulmonary disease (COPD), severe -- home O2
               2 L/min, tiotropium, fluticasone/salmeterol
            2. Hypertension -- amlodipine

            Brief Hospital Course:
            #COPD exacerbation: BiPAP non-invasive mechanical ventilation applied
            at 10/5 cm H2O for first 3 days with improvement in respiratory
            acidosis. Patient weaned off BiPAP by day 4. Continuous supplemental
            oxygen therapy required throughout admission (2-3 L/min). Prednisone
            40 mg PO daily x5 days. Azithromycin 500 mg PO daily x5 days
            (antibiotic). Nebulized albuterol and ipratropium Q4H.

            #Fluid status: Mild fluid retention. Furosemide 20 mg PO daily
            (diuretic) added temporarily; discontinued at discharge.

            Discharge Medications:
            1. prednisone taper
            2. azithromycin -- completed
            3. tiotropium inhaler daily
            4. fluticasone/salmeterol inhaler BID
            5. supplemental oxygen 2 L/min continuous

            Discharge Diagnoses:
            Primary: Acute exacerbation of chronic obstructive pulmonary disease (COPD)
            Secondary: Hypertension
        """),
    ),
    # ------------------------------------------------------------------ #
    # 9  Acute CHF · Respiratory Failure · Intubation · Tracheostomy     #
    #    Meds: IV vasopressors · furosemide (diuretic)                    #
    #    Procedures: invasive MV · tracheostomy care · central line       #
    # ------------------------------------------------------------------ #
    (
        "TEST0009-DS-01", "90000009", "91000009", "M", "MICU",
        textwrap.dedent("""\
            Name: ___ Unit No: ___
            Admission Date: ___ Discharge Date: ___
            Date of Birth: ___ Sex: M
            Service: MICU
            Allergies: No Known Allergies
            Attending: ___

            Chief Complaint:
            Acute hypoxic respiratory failure in setting of flash pulmonary edema.

            Major Surgical or Invasive Procedure:
            1. Endotracheal intubation and invasive mechanical ventilation.
            2. Percutaneous tracheostomy placed on hospital day 14.
            3. Central venous catheter placed (right internal jugular).

            History of Present Illness:
            ___ year-old male with hypertension and newly diagnosed systolic
            heart failure (EF 20%) presenting with flash pulmonary edema and
            acute hypoxic respiratory failure requiring emergent intubation.

            Past Medical History:
            1. Hypertension -- prior antihypertensives
            2. Systolic heart failure, newly diagnosed (EF 20%)

            Brief Hospital Course:
            #Acute hypoxic respiratory failure / heart failure: Patient intubated
            emergently and placed on invasive mechanical ventilator. IV furosemide
            (diuretic) administered for aggressive diuresis. Vasopressors (IV
            norepinephrine and dopamine, vasoactive medications) required for
            cardiogenic shock, weaned over 5 days. Patient failed multiple
            extubation attempts; percutaneous tracheostomy performed on day 14.
            Ongoing tracheostomy care provided. Central venous catheter maintained
            throughout ICU stay.

            Discharge Medications:
            1. furosemide 80 mg PO daily (diuretic)
            2. carvedilol 3.125 mg PO BID
            3. lisinopril 5 mg PO daily

            Discharge Diagnoses:
            Primary: Acute hypoxic respiratory failure requiring mechanical ventilation
            Secondary: Acute decompensated systolic heart failure (EF 20%),
            Hypertension, Cardiogenic shock (resolved)
        """),
    ),
    # ------------------------------------------------------------------ #
    # 10  Cirrhosis · GERD · Depression · HTN                            #
    #     Meds: sertraline (antidep), spironolactone (diuretic)           #
    #     Procedures: none                                                 #
    # ------------------------------------------------------------------ #
    (
        "TEST0010-DS-01", "90000010", "91000010", "M", "MEDICINE",
        textwrap.dedent("""\
            Name: ___ Unit No: ___
            Admission Date: ___ Discharge Date: ___
            Date of Birth: ___ Sex: M
            Service: MEDICINE
            Allergies: Percocet
            Attending: ___

            Chief Complaint:
            Worsening abdominal distension and confusion.

            Major Surgical or Invasive Procedure:
            Therapeutic paracentesis.

            History of Present Illness:
            ___ year-old male with HCV cirrhosis complicated by ascites and
            hepatic encephalopathy, GERD (gastroesophageal reflux disease),
            major depressive disorder, and hypertension presenting with
            worsening abdominal distension, increasing confusion, and decreased
            oral intake. Na-restricted diet noncompliance.

            Past Medical History:
            1. HCV cirrhosis with ascites and portal hypertension
            2. GERD (gastroesophageal reflux disease) / peptic ulcer disease
            3. Major depressive disorder -- sertraline
            4. Hypertension -- propranolol

            Brief Hospital Course:
            #Cirrhosis / decompensated ascites: Therapeutic paracentesis performed
            (4 L removed). Spironolactone 100 mg PO daily and furosemide continued
            (diuretics). Rifaximin and lactulose for hepatic encephalopathy.

            #GERD / Ulcer: Pantoprazole 40 mg PO daily continued.

            #Depression: Sertraline 100 mg PO daily continued (antidepressant).

            #Hypertension: Propranolol continued, BP well-controlled.

            Discharge Medications:
            1. spironolactone 100 mg PO daily (diuretic)
            2. furosemide 40 mg PO daily (diuretic)
            3. sertraline 100 mg PO daily (antidepressant)
            4. rifaximin 550 mg PO BID
            5. lactulose 30 mL PO TID
            6. pantoprazole 40 mg PO daily

            Discharge Diagnoses:
            Primary: Decompensated HCV cirrhosis with ascites and hepatic encephalopathy
            Secondary: GERD/Peptic ulcer disease, Major depressive disorder, Hypertension
        """),
    ),
    # ------------------------------------------------------------------ #
    # 11  Rheumatoid Arthritis · DVT · HTN                               #
    #     Meds: rivaroxaban (anticoag), oxycodone (opioid)                #
    #     Procedures: peripheral IV                                        #
    # ------------------------------------------------------------------ #
    (
        "TEST0011-DS-01", "90000011", "91000011", "F", "MEDICINE",
        textwrap.dedent("""\
            Name: ___ Unit No: ___
            Admission Date: ___ Discharge Date: ___
            Date of Birth: ___ Sex: F
            Service: MEDICINE
            Allergies: NSAIDs (GI bleed)
            Attending: ___

            Chief Complaint:
            Left leg pain and swelling; new DVT on imaging.

            Major Surgical or Invasive Procedure:
            Peripheral intravenous (IV) access placed.

            History of Present Illness:
            ___ year-old female with rheumatoid arthritis (RA) on methotrexate
            and prednisone, hypertension, and prior DVT presenting with new
            left lower extremity deep venous thrombosis (DVT). Doppler confirmed
            left femoral and popliteal DVT. No pulmonary embolism on CTA.

            Past Medical History:
            1. Rheumatoid arthritis (RA) -- methotrexate, prednisone
            2. Hypertension -- lisinopril
            3. Chronic pain -- opioid regimen for refractory arthritis pain

            Brief Hospital Course:
            #DVT: Rivaroxaban started (anticoagulant). Hematology and rheumatology
            consulted. Anticoagulation to continue for minimum 3 months.

            #Rheumatoid arthritis: Methotrexate held during acute illness.
            Prednisone continued at maintenance dose for disease control.
            Ongoing chronic pain managed with oxycodone 5 mg PO Q6H PRN (opioid).

            #Hypertension: Lisinopril continued.

            Discharge Medications:
            1. rivaroxaban 15 mg PO BID x21 days then 20 mg daily (anticoagulant)
            2. oxycodone 5 mg PO Q6H PRN pain (opioid)
            3. methotrexate 15 mg PO weekly (to restart in 1 week)
            4. prednisone 5 mg PO daily
            5. lisinopril 10 mg PO daily

            Discharge Diagnoses:
            Primary: Deep venous thrombosis (DVT), left lower extremity
            Secondary: Rheumatoid arthritis, Hypertension
        """),
    ),
    # ------------------------------------------------------------------ #
    # 12  Colorectal Cancer · Oral Chemotherapy · Anemia · Transfusion   #
    #     Meds: capecitabine (oral chemo), opioid, antibiotic             #
    #     Procedures: oral chemo · transfusion · peripheral IV            #
    # ------------------------------------------------------------------ #
    (
        "TEST0012-DS-01", "90000012", "91000012", "M", "ONCOLOGY",
        textwrap.dedent("""\
            Name: ___ Unit No: ___
            Admission Date: ___ Discharge Date: ___
            Date of Birth: ___ Sex: M
            Service: ONCOLOGY
            Allergies: No Known Drug Allergies
            Attending: ___

            Chief Complaint:
            Fatigue, symptomatic anemia, and chemotherapy cycle 4.

            Major Surgical or Invasive Procedure:
            Red blood cell transfusion (1 unit).
            Peripheral intravenous (IV) access placed.

            History of Present Illness:
            ___ year-old male with stage III colorectal cancer (cancer) on
            adjuvant oral capecitabine chemotherapy, and hypertension. Admitted
            for cycle 4 of oral capecitabine, management of symptomatic anemia
            (Hgb 6.8), and neutropenic fever.

            Past Medical History:
            1. Colorectal cancer, stage III -- adjuvant oral capecitabine
               chemotherapy (post-resection)
            2. Anemia -- related to chemotherapy
            3. Hypertension -- lisinopril

            Brief Hospital Course:
            #Cancer / oral chemotherapy: Oral capecitabine cycle 4 administered.
            #Anemia: Hemoglobin 6.8 on admission; symptomatic. One unit packed
            red blood cell (pRBC) transfusion administered.

            #Neutropenic fever: Treated with intravenous cefepime (IV antibiotic)
            empirically until cultures finalized (no growth). Transitioned to
            oral antibiotics at discharge.

            #Pain: Oxycodone 5 mg PO Q6H PRN administered for abdominal pain
            related to colorectal cancer (opioid analgesic).

            Discharge Medications:
            1. capecitabine 1250 mg/m2 PO BID days 1-14 of cycle
               (oral chemotherapy)
            2. oxycodone 5 mg PO Q6H PRN (opioid)
            3. ciprofloxacin 500 mg PO BID x3 days (antibiotic)
            4. lisinopril 10 mg PO daily

            Discharge Diagnoses:
            Primary: Colorectal cancer, stage III, on adjuvant oral chemotherapy
            Secondary: Symptomatic anemia (transfused), Neutropenic fever (resolved),
            Hypertension
        """),
    ),
    # ------------------------------------------------------------------ #
    # 13  DKA · Insulin-Dependent DM · HTN                               #
    #     Meds: insulin drip then SC (7 days, dose change), IV antibiotic #
    #     Procedures: peripheral IV                                        #
    # ------------------------------------------------------------------ #
    (
        "TEST0013-DS-01", "90000013", "91000013", "F", "MEDICINE",
        textwrap.dedent("""\
            Name: ___ Unit No: ___
            Admission Date: ___ Discharge Date: ___
            Date of Birth: ___ Sex: F
            Service: MEDICINE
            Allergies: No Known Drug Allergies
            Attending: ___

            Chief Complaint:
            Hyperglycemia, nausea, vomiting -- diabetic ketoacidosis.

            Major Surgical or Invasive Procedure:
            Peripheral intravenous (IV) access placed.

            History of Present Illness:
            ___ year-old female with type 1 diabetes mellitus (insulin-dependent)
            and hypertension presenting with diabetic ketoacidosis (DKA) with
            blood glucose 680, pH 7.18, anion gap 28. Additionally found to
            have a diabetic foot infection requiring IV antibiotics.

            Past Medical History:
            1. Type 1 diabetes mellitus (insulin-dependent) -- insulin glargine
               and insulin lispro sliding scale
            2. Hypertension -- metoprolol

            Brief Hospital Course:
            #DKA / insulin-dependent diabetes: Patient placed on continuous IV
            insulin drip (regular insulin) for the first 24 hours with anion gap
            closure. Insulin dose significantly increased from prior regimen
            (insulin dose change). Transitioned to subcutaneous insulin injections
            on day 2 and continued for the remaining 6 days of the 7-day admission.
            Total: 7 days of insulin injections during hospitalization.
            Endocrinology adjusted insulin glargine from 20 to 35 units nightly
            and lispro correction doses (dose change from prior insulin regimen).

            #Diabetic foot infection: Treated with intravenous vancomycin
            (IV antibiotic) for 7 days. Wound care daily.

            Discharge Medications:
            1. insulin glargine 35 units SQ nightly (insulin -- hypoglycemic,
               dose increased from prior 20 units)
            2. insulin lispro per sliding scale SQ with meals (hypoglycemic)
            3. metoprolol 25 mg PO BID

            Discharge Diagnoses:
            Primary: Diabetic ketoacidosis (DKA)
            Secondary: Diabetic foot infection (treated), Hypertension,
            Insulin-dependent type 1 diabetes mellitus
        """),
    ),
    # ------------------------------------------------------------------ #
    # 14  MRSA Bacteremia · Isolation · COPD · HTN                       #
    #     Meds: vancomycin IV (antibiotic)                                #
    #     Procedures: isolation/quarantine · IV antibiotics · PICC       #
    # ------------------------------------------------------------------ #
    (
        "TEST0014-DS-01", "90000014", "91000014", "M", "MEDICINE",
        textwrap.dedent("""\
            Name: ___ Unit No: ___
            Admission Date: ___ Discharge Date: ___
            Date of Birth: ___ Sex: M
            Service: MEDICINE
            Allergies: No Known Drug Allergies
            Attending: ___

            Chief Complaint:
            Fever, chills, and hypotension -- sepsis.

            Major Surgical or Invasive Procedure:
            PICC line placed for long-term IV antibiotic access.

            History of Present Illness:
            ___ year-old male with chronic obstructive pulmonary disease (COPD)
            and hypertension presenting with fever, rigors, and hypotension.
            Blood cultures: 2/2 bottles MRSA. Patient placed in contact isolation
            and droplet precautions (isolation/quarantine) per infection control.

            Past Medical History:
            1. COPD -- tiotropium, albuterol PRN
            2. Hypertension -- amlodipine
            3. Prior MRSA skin and soft tissue infection (SSTI)

            Brief Hospital Course:
            #MRSA bacteremia: Placed immediately in contact isolation
            (isolation/quarantine). Intravenous vancomycin started as IV antibiotic
            therapy via PICC line; 6-week course planned. Echocardiogram negative
            for endocarditis. ID consulted.

            #COPD: Continued maintenance inhalers. No acute exacerbation.

            #Hypertension: Amlodipine continued.

            Discharge Medications:
            1. vancomycin IV (via PICC) -- 6-week outpatient course (IV antibiotic)
            2. tiotropium inhaler daily
            3. albuterol inhaler PRN
            4. amlodipine 5 mg PO daily

            Discharge Diagnoses:
            Primary: MRSA bacteremia
            Secondary: COPD, Hypertension
            Infection control: Contact and droplet isolation/quarantine precautions
            maintained throughout admission.
        """),
    ),
    # ------------------------------------------------------------------ #
    # 15  Terminal Lung Cancer · Hospice Care · Opioid · Antianxiety     #
    #     Meds: morphine (opioid), lorazepam (antianxiety),              #
    #           haloperidol (antipsychotic for terminal agitation)         #
    #     Procedures: hospice care                                         #
    # ------------------------------------------------------------------ #
    (
        "TEST0015-DS-01", "90000015", "91000015", "F", "MEDICINE",
        textwrap.dedent("""\
            Name: ___ Unit No: ___
            Admission Date: ___ Discharge Date: ___
            Date of Birth: ___ Sex: F
            Service: MEDICINE
            Allergies: No Known Allergies
            Attending: ___

            Chief Complaint:
            Goals of care discussion and transition to inpatient hospice.

            Major Surgical or Invasive Procedure:
            None (comfort measures only).

            History of Present Illness:
            ___ year-old female with stage IV non-small cell lung cancer (cancer,
            metastatic to liver and bone) and major depressive disorder, declining
            function, and refractory dyspnea. Patient and family elected to transition
            to comfort-focused hospice care after extensive goals-of-care discussion.
            No further disease-directed therapy.

            Past Medical History:
            1. Non-small cell lung cancer (NSCLC), stage IV -- metastatic, no
               further curative treatment
            2. Major depressive disorder -- on sertraline
            3. Chronic pain -- prior opioid regimen

            Brief Hospital Course:
            Patient enrolled in inpatient hospice care program. All disease-directed
            interventions discontinued. Comfort medications titrated:
            - Morphine IV/SQ PRN for dyspnea and pain (opioid analgesic)
            - Lorazepam 1 mg SQ Q4H PRN for anxiety and dyspnea (antianxiety)
            - Haloperidol 1 mg SQ Q6H PRN for terminal agitation and delirium
              (antipsychotic)
            Patient remained comfortable. Family present throughout.

            Discharge Medications (comfort/hospice):
            1. morphine 5 mg SQ/IV Q2H PRN pain/dyspnea (opioid)
            2. lorazepam 1 mg SQ Q4H PRN anxiety/dyspnea (antianxiety)
            3. haloperidol 1 mg SQ Q6H PRN agitation (antipsychotic)
            4. glycopyrrolate PRN secretions

            Discharge Diagnoses:
            Primary: Non-small cell lung cancer (NSCLC), stage IV -- hospice care
            Secondary: Major depressive disorder

            Discharge Disposition: Inpatient hospice / comfort care only
        """),
    ),
]

# ---------------------------------------------------------------------------
# Ground-truth extraction dicts (mapper input format)
# ---------------------------------------------------------------------------
# Format mirrors the dict returned by LLMExtractor.extract():
#   - Top-level keys are MDS item IDs (plus note_id / subject_id / hadm_id)
#   - "confidence" is a nested dict with float values (1.0 = ground truth)
#   - False values are included for the most commonly confused/tested fields
#     to provide TN signal for evaluation

EXTRACTION_GROUND_TRUTH = [
    # ---- Patient 1: AF + CHF + HTN + T2DM + COPD ----------------------
    {
        "note_id": "TEST0001-DS-01",
        "subject_id": "90000001",
        "hadm_id": "91000001",
        # Section I – Active Disease Diagnoses
        "I0100": False,  # Cancer – absent
        "I0200": False,  # Anemia – absent
        "I0300": True,   # Atrial Fibrillation
        "I0400": False,  # CAD – absent
        "I0500": False,  # DVT/PE – absent
        "I0600": True,   # Heart Failure
        "I0700": True,   # Hypertension
        "I1100": False,  # Cirrhosis – absent
        "I1200": False,  # GERD/Ulcer – absent
        "I1500": False,  # Renal Insufficiency – absent
        "I2000": False,  # Pneumonia – absent
        "I2900": True,   # Diabetes Mellitus (Type 2)
        "I3700": False,  # Arthritis – absent
        "I4200": False,  # Alzheimer's – absent
        "I4500": False,  # Stroke – absent
        "I5800": False,  # Depression – absent
        "I5900": False,  # Bipolar Disorder – absent
        "I6100": False,  # PTSD – absent
        "I6200": True,   # COPD
        # Section N – Medications
        "N0415A": ["2"],  # Antipsychotic – not taking (indication not noted)
        "N0415B": ["2"],  # Antianxiety – not taking
        "N0415C": ["2"],  # Antidepressant – not taking
        "N0415D": ["2"],  # Hypnotic – not taking
        "N0415E": ["1"],  # Anticoagulant (warfarin) – Is taking
        "N0415F": ["2"],  # Antibiotic – not taking
        "N0415G": ["1"],  # Diuretic (furosemide) – Is taking
        "N0415H": ["2"],  # Opioid – not taking
        "N0415I": ["2"],  # Antiplatelet – not taking
        "N0415J": ["1"],  # Hypoglycemic (metformin) – Is taking
        "N0415Z": False,
        # Section O – Special Treatments
        "O0110A1": False,  # IV Chemotherapy – absent
        "O0110B1": False,  # Radiation – absent
        "O0110C1": False,  # Continuous O2 – absent
        "O0110F1": False,  # Invasive MV – absent
        "O0110G2": False,  # BiPAP – absent
        "O0110H1": False,  # IV Vasoactive – absent
        "O0110H2": False,  # IV Antibiotics – absent
        "O0110I1": False,  # Transfusion – absent
        "O0110J1": False,  # Hemodialysis – absent
        "O0110K1": False,  # Hospice – absent
        "O0110M1": False,  # Isolation – absent
        "O0110O1": True,   # Peripheral IV access
        "O0110O3": False,  # Central/PICC – absent
        "confidence": {
            "I0300": 1.0, "I0600": 1.0, "I0700": 1.0, "I2900": 1.0, "I6200": 1.0,
            "I0100": 1.0, "I0200": 1.0, "I0400": 1.0, "I0500": 1.0,
            "I1100": 1.0, "I1200": 1.0, "I1500": 1.0, "I2000": 1.0,
            "I3700": 1.0, "I4200": 1.0, "I4500": 1.0, "I5800": 1.0,
            "I5900": 1.0, "I6100": 1.0,
            "N0415E": 1.0, "N0415G": 1.0, "N0415J": 1.0,
            "O0110O1": 1.0,
        },
    },
    # ---- Patient 2: Stroke + Post-stroke Depression + HTN -------------
    {
        "note_id": "TEST0002-DS-01",
        "subject_id": "90000002",
        "hadm_id": "91000002",
        "I0020":  "1",   # Primary condition: Stroke (MDS Select code 1)
        "I0100": False,
        "I0200": False,
        "I0300": False,  # AF – absent
        "I0400": False,
        "I0500": False,
        "I0600": False,  # Heart Failure – absent
        "I0700": True,   # Hypertension
        "I1100": False,
        "I1200": False,
        "I1500": False,
        "I2000": False,
        "I2900": False,  # Diabetes – absent
        "I3700": False,
        "I4200": False,  # Alzheimer's – absent
        "I4500": True,   # CVA / Stroke
        "I5800": True,   # Depression (post-stroke)
        "I5900": False,
        "I6100": False,
        "I6200": False,  # COPD – absent
        "N0415A": ["2"],
        "N0415B": ["2"],
        "N0415C": ["1"],  # Antidepressant (sertraline) – Is taking
        "N0415D": ["2"],
        "N0415E": ["2"],
        "N0415F": ["2"],
        "N0415G": ["2"],
        "N0415H": ["2"],
        "N0415I": ["1"],  # Antiplatelet (aspirin + clopidogrel)
        "N0415J": ["2"],
        "N0415Z": False,
        "O0110A1": False,
        "O0110B1": False,
        "O0110J1": False,
        "O0110K1": False,
        "O0110O1": False,
        "O0400A1": 60,    # PT: 60 min/day
        "O0400B1": 45,    # OT: 45 min/day
        "O0400C1": 30,    # SLP: 30 min/day
        "confidence": {
            "I0020": 1.0, "I0700": 1.0, "I4500": 1.0, "I5800": 1.0,
            "I0300": 1.0, "I0600": 1.0, "I2900": 1.0, "I4200": 1.0, "I6200": 1.0,
            "N0415C": 1.0, "N0415I": 1.0,
            "O0400A1": 1.0, "O0400B1": 1.0, "O0400C1": 1.0,
        },
    },
    # ---- Patient 3: NSCLC + COPD + HTN --------------------------------
    {
        "note_id": "TEST0003-DS-01",
        "subject_id": "90000003",
        "hadm_id": "91000003",
        "I0100": True,   # Cancer (NSCLC)
        "I0200": False,
        "I0300": False,
        "I0400": False,
        "I0500": False,
        "I0600": False,
        "I0700": True,   # Hypertension
        "I1100": False,
        "I1200": False,
        "I1500": False,
        "I2000": True,   # Pneumonia (MRSA pneumonia)
        "I2900": False,
        "I3700": False,
        "I4200": False,
        "I4500": False,
        "I5800": False,
        "I5900": False,
        "I6100": False,
        "I6200": True,   # COPD
        "N0415A": ["2"],
        "N0415B": ["2"],
        "N0415C": ["2"],
        "N0415D": ["2"],
        "N0415E": ["2"],
        "N0415F": ["1"],  # Antibiotic (vancomycin IV)
        "N0415G": ["2"],
        "N0415H": ["1"],  # Opioid (morphine)
        "N0415I": ["2"],
        "N0415J": ["2"],
        "N0415Z": False,
        "O0110A1": True,  # IV Chemotherapy (carboplatin)
        "O0110A3": False, # Oral chemo – absent
        "O0110B1": True,  # Radiation therapy
        "O0110C1": True,  # Continuous O2 therapy
        "O0110F1": False,
        "O0110G2": False,
        "O0110H2": True,  # IV Antibiotics (vancomycin via PICC)
        "O0110I1": False,
        "O0110J1": False,
        "O0110K1": False,
        "O0110M1": False,
        "O0110O1": False,
        "O0110O3": True,  # PICC line
        "confidence": {
            "I0100": 1.0, "I0700": 1.0, "I2000": 1.0, "I6200": 1.0,
            "I0300": 1.0, "I0600": 1.0, "I2900": 1.0, "I4500": 1.0,
            "N0415F": 1.0, "N0415H": 1.0,
            "O0110A1": 1.0, "O0110B1": 1.0, "O0110C1": 1.0,
            "O0110H2": 1.0, "O0110O3": 1.0,
        },
    },
    # ---- Patient 4: Alzheimer's + Depression + HTN + CKD --------------
    {
        "note_id": "TEST0004-DS-01",
        "subject_id": "90000004",
        "hadm_id": "91000004",
        "I0100": False,
        "I0200": False,
        "I0300": False,
        "I0400": False,
        "I0500": False,
        "I0600": False,
        "I0700": True,   # Hypertension
        "I1100": False,
        "I1200": False,
        "I1500": True,   # Renal Insufficiency (CKD stage 3)
        "I2000": False,
        "I2900": False,
        "I3700": False,
        "I4200": True,   # Alzheimer's Disease
        "I4500": False,
        "I5800": True,   # Depression (major depressive disorder)
        "I5900": False,
        "I6100": False,
        "I6200": False,
        "N0415A": ["2"],
        "N0415B": ["1"],  # Antianxiety (lorazepam)
        "N0415C": ["1"],  # Antidepressant (sertraline)
        "N0415D": ["2"],
        "N0415E": ["2"],
        "N0415F": ["2"],
        "N0415G": ["1"],  # Diuretic (furosemide)
        "N0415H": ["2"],
        "N0415I": ["2"],
        "N0415J": ["2"],
        "N0415Z": False,
        "O0110A1": False,
        "O0110B1": False,
        "O0110C1": False,
        "O0110J1": False,
        "O0110K1": False,
        "O0110M1": False,
        "O0110O1": False,
        "O0110O3": False,
        "confidence": {
            "I0700": 1.0, "I1500": 1.0, "I4200": 1.0, "I5800": 1.0,
            "I0100": 1.0, "I0300": 1.0, "I0600": 1.0, "I2900": 1.0,
            "I4500": 1.0, "I6200": 1.0,
            "N0415B": 1.0, "N0415C": 1.0, "N0415G": 1.0,
        },
    },
    # ---- Patient 5: ESRD + CAD + HTN + Insulin DM ---------------------
    {
        "note_id": "TEST0005-DS-01",
        "subject_id": "90000005",
        "hadm_id": "91000005",
        "I0100": False,
        "I0200": False,
        "I0300": False,
        "I0400": True,   # CAD
        "I0500": False,
        "I0600": False,
        "I0700": True,   # Hypertension
        "I1100": False,
        "I1200": False,
        "I1500": True,   # Renal Insufficiency / ESRD
        "I2000": False,
        "I2900": True,   # Diabetes Mellitus (insulin-dependent)
        "I3700": False,
        "I4200": False,
        "I4500": False,
        "I5800": False,
        "I5900": False,
        "I6100": False,
        "I6200": False,
        "N0300":  7,      # Total injections past 7 days
        "N0350A": 7,      # Insulin injections past 7 days
        "N0350B": False,  # Insulin dose change – not in this admission
        "N0415A": ["2"],
        "N0415B": ["2"],
        "N0415C": ["2"],
        "N0415D": ["2"],
        "N0415E": ["1"],  # Anticoagulant (heparin during dialysis)
        "N0415F": ["2"],
        "N0415G": ["2"],
        "N0415H": ["2"],
        "N0415I": ["1"],  # Antiplatelet (aspirin)
        "N0415J": ["1"],  # Hypoglycemic (insulin)
        "N0415Z": False,
        "O0110A1": False,
        "O0110B1": False,
        "O0110C1": False,
        "O0110F1": False,
        "O0110J1": True,  # Hemodialysis
        "O0110K1": False,
        "O0110M1": False,
        "O0110O1": False,
        "O0110O3": True,  # Central line / PICC (tunneled dialysis catheter)
        "confidence": {
            "I0400": 1.0, "I0700": 1.0, "I1500": 1.0, "I2900": 1.0,
            "I0100": 1.0, "I0300": 1.0, "I0600": 1.0, "I4200": 1.0,
            "I4500": 1.0, "I5800": 1.0, "I6200": 1.0,
            "N0300": 1.0, "N0350A": 1.0,
            "N0415E": 1.0, "N0415I": 1.0, "N0415J": 1.0,
            "O0110J1": 1.0, "O0110O3": 1.0,
        },
    },
    # ---- Patient 6: Bipolar Disorder + PTSD + HTN ---------------------
    {
        "note_id": "TEST0006-DS-01",
        "subject_id": "90000006",
        "hadm_id": "91000006",
        "I0100": False,
        "I0300": False,
        "I0400": False,
        "I0600": False,
        "I0700": True,   # Hypertension
        "I1100": False,
        "I1500": False,
        "I2000": False,
        "I2900": False,
        "I4200": False,
        "I4500": False,
        "I5800": False,  # Depression – not primary; bipolar is primary mood dx
        "I5900": True,   # Bipolar Disorder
        "I6100": True,   # PTSD
        "I6200": False,
        "N0415A": ["1"],  # Antipsychotic (quetiapine) – Is taking, routine
        "N0415B": ["2"],
        "N0415C": ["1"],  # Antidepressant (sertraline) – Is taking
        "N0415D": ["2"],
        "N0415E": ["2"],
        "N0415F": ["2"],
        "N0415G": ["2"],
        "N0415H": ["2"],
        "N0415I": ["2"],
        "N0415J": ["2"],
        "N0415Z": False,
        "N0450A": "1",   # Antipsychotic use: routine only (no PRN)
        "O0110A1": False,
        "O0110J1": False,
        "O0110K1": False,
        "O0110M1": False,
        "O0110O1": False,
        "confidence": {
            "I0700": 1.0, "I5900": 1.0, "I6100": 1.0,
            "I0300": 1.0, "I0600": 1.0, "I2900": 1.0, "I4200": 1.0,
            "I4500": 1.0, "I5800": 1.0, "I6200": 1.0,
            "N0415A": 1.0, "N0415C": 1.0, "N0450A": 1.0,
        },
    },
    # ---- Patient 7: Pneumonia + DVT/PE + HTN --------------------------
    {
        "note_id": "TEST0007-DS-01",
        "subject_id": "90000007",
        "hadm_id": "91000007",
        "I0100": False,
        "I0300": False,
        "I0400": False,
        "I0500": True,   # DVT / Pulmonary Embolism
        "I0600": False,
        "I0700": True,   # Hypertension
        "I1100": False,
        "I1500": False,
        "I2000": True,   # Pneumonia
        "I2900": False,
        "I4200": False,
        "I4500": False,
        "I5800": False,
        "I5900": False,
        "I6100": False,
        "I6200": False,
        "N0415A": ["2"],
        "N0415B": ["2"],
        "N0415C": ["2"],
        "N0415D": ["2"],
        "N0415E": ["1"],  # Anticoagulant (heparin -> apixaban)
        "N0415F": ["1"],  # Antibiotic (piperacillin-tazobactam IV then PO)
        "N0415G": ["2"],
        "N0415H": ["2"],
        "N0415I": ["2"],
        "N0415J": ["2"],
        "N0415Z": False,
        "O0110A1": False,
        "O0110B1": False,
        "O0110C2": True,  # Intermittent O2 therapy (as needed)
        "O0110F1": False,
        "O0110H2": True,  # IV Antibiotics (pip-tazo)
        "O0110I1": False,
        "O0110J1": False,
        "O0110K1": False,
        "O0110M1": False,
        "O0110O1": True,  # Peripheral IV access
        "O0110O3": False,
        "confidence": {
            "I0500": 1.0, "I0700": 1.0, "I2000": 1.0,
            "I0100": 1.0, "I0300": 1.0, "I0600": 1.0, "I2900": 1.0,
            "I4200": 1.0, "I4500": 1.0, "I5800": 1.0, "I6200": 1.0,
            "N0415E": 1.0, "N0415F": 1.0,
            "O0110C2": 1.0, "O0110H2": 1.0, "O0110O1": 1.0,
        },
    },
    # ---- Patient 8: Severe COPD + BiPAP + HTN -------------------------
    {
        "note_id": "TEST0008-DS-01",
        "subject_id": "90000008",
        "hadm_id": "91000008",
        "I0100": False,
        "I0300": False,
        "I0400": False,
        "I0500": False,
        "I0600": False,
        "I0700": True,   # Hypertension
        "I1100": False,
        "I1500": False,
        "I2000": False,
        "I2900": False,
        "I4200": False,
        "I4500": False,
        "I5800": False,
        "I5900": False,
        "I6100": False,
        "I6200": True,   # COPD
        "N0415A": ["2"],
        "N0415B": ["2"],
        "N0415C": ["2"],
        "N0415D": ["2"],
        "N0415E": ["2"],
        "N0415F": ["1"],  # Antibiotic (azithromycin)
        "N0415G": ["1"],  # Diuretic (furosemide, temporary)
        "N0415H": ["2"],
        "N0415I": ["2"],
        "N0415J": ["2"],
        "N0415Z": False,
        "O0110A1": False,
        "O0110C1": True,  # Continuous O2 therapy
        "O0110F1": False,
        "O0110G2": True,  # BiPAP (non-invasive mechanical ventilation)
        "O0110H2": False,
        "O0110I1": False,
        "O0110J1": False,
        "O0110K1": False,
        "O0110M1": False,
        "O0110O1": False,
        "O0110O3": False,
        "confidence": {
            "I0700": 1.0, "I6200": 1.0,
            "I0100": 1.0, "I0300": 1.0, "I0600": 1.0, "I2900": 1.0,
            "I4200": 1.0, "I4500": 1.0, "I5800": 1.0,
            "N0415F": 1.0, "N0415G": 1.0,
            "O0110C1": 1.0, "O0110G2": 1.0,
        },
    },
    # ---- Patient 9: CHF + Respiratory Failure + Intubation + Trach ---
    {
        "note_id": "TEST0009-DS-01",
        "subject_id": "90000009",
        "hadm_id": "91000009",
        "I0100": False,
        "I0300": False,
        "I0400": False,
        "I0500": False,
        "I0600": True,   # Heart Failure
        "I0700": True,   # Hypertension
        "I1100": False,
        "I1500": False,
        "I2000": False,
        "I2900": False,
        "I4200": False,
        "I4500": False,
        "I5800": False,
        "I5900": False,
        "I6100": False,
        "I6200": False,
        "N0415A": ["2"],
        "N0415B": ["2"],
        "N0415C": ["2"],
        "N0415D": ["2"],
        "N0415E": ["2"],
        "N0415F": ["2"],
        "N0415G": ["1"],  # Diuretic (furosemide IV for aggressive diuresis)
        "N0415H": ["2"],
        "N0415I": ["2"],
        "N0415J": ["2"],
        "N0415Z": False,
        "O0110A1": False,
        "O0110B1": False,
        "O0110C1": False,
        "O0110E1": True,  # Tracheostomy care
        "O0110F1": True,  # Invasive mechanical ventilator
        "O0110G2": False,
        "O0110H1": True,  # IV Vasoactive medications (vasopressors)
        "O0110H2": False,
        "O0110I1": False,
        "O0110J1": False,
        "O0110K1": False,
        "O0110M1": False,
        "O0110O1": False,
        "O0110O3": True,  # Central venous catheter
        "confidence": {
            "I0600": 1.0, "I0700": 1.0,
            "I0100": 1.0, "I0300": 1.0, "I2900": 1.0, "I4200": 1.0,
            "I4500": 1.0, "I5800": 1.0, "I6200": 1.0,
            "N0415G": 1.0,
            "O0110E1": 1.0, "O0110F1": 1.0, "O0110H1": 1.0, "O0110O3": 1.0,
        },
    },
    # ---- Patient 10: Cirrhosis + GERD + Depression + HTN --------------
    {
        "note_id": "TEST0010-DS-01",
        "subject_id": "90000010",
        "hadm_id": "91000010",
        "I0100": False,
        "I0300": False,
        "I0400": False,
        "I0500": False,
        "I0600": False,
        "I0700": True,   # Hypertension
        "I1100": True,   # Cirrhosis (HCV cirrhosis)
        "I1200": True,   # GERD / Ulcer
        "I1500": False,
        "I2000": False,
        "I2900": False,
        "I3700": False,
        "I4200": False,
        "I4500": False,
        "I5800": True,   # Depression
        "I5900": False,
        "I6100": False,
        "I6200": False,
        "N0415A": ["2"],
        "N0415B": ["2"],
        "N0415C": ["1"],  # Antidepressant (sertraline)
        "N0415D": ["2"],
        "N0415E": ["2"],
        "N0415F": ["2"],
        "N0415G": ["1"],  # Diuretic (spironolactone + furosemide)
        "N0415H": ["2"],
        "N0415I": ["2"],
        "N0415J": ["2"],
        "N0415Z": False,
        "O0110A1": False,
        "O0110B1": False,
        "O0110C1": False,
        "O0110J1": False,
        "O0110K1": False,
        "O0110M1": False,
        "O0110O1": False,
        "O0110O3": False,
        "confidence": {
            "I0700": 1.0, "I1100": 1.0, "I1200": 1.0, "I5800": 1.0,
            "I0100": 1.0, "I0300": 1.0, "I0600": 1.0, "I2900": 1.0,
            "I4200": 1.0, "I4500": 1.0, "I6200": 1.0,
            "N0415C": 1.0, "N0415G": 1.0,
        },
    },
    # ---- Patient 11: Rheumatoid Arthritis + DVT + HTN -----------------
    {
        "note_id": "TEST0011-DS-01",
        "subject_id": "90000011",
        "hadm_id": "91000011",
        "I0100": False,
        "I0200": False,
        "I0300": False,
        "I0400": False,
        "I0500": True,   # DVT
        "I0600": False,
        "I0700": True,   # Hypertension
        "I1100": False,
        "I1200": False,
        "I1500": False,
        "I2000": False,
        "I2900": False,
        "I3700": True,   # Arthritis (rheumatoid arthritis)
        "I4200": False,
        "I4500": False,
        "I5800": False,
        "I5900": False,
        "I6100": False,
        "I6200": False,
        "N0415A": ["2"],
        "N0415B": ["2"],
        "N0415C": ["2"],
        "N0415D": ["2"],
        "N0415E": ["1"],  # Anticoagulant (rivaroxaban)
        "N0415F": ["2"],
        "N0415G": ["2"],
        "N0415H": ["1"],  # Opioid (oxycodone for arthritis pain)
        "N0415I": ["2"],
        "N0415J": ["2"],
        "N0415Z": False,
        "O0110A1": False,
        "O0110B1": False,
        "O0110C1": False,
        "O0110J1": False,
        "O0110K1": False,
        "O0110M1": False,
        "O0110O1": True,  # Peripheral IV access
        "O0110O3": False,
        "confidence": {
            "I0500": 1.0, "I0700": 1.0, "I3700": 1.0,
            "I0100": 1.0, "I0300": 1.0, "I0600": 1.0, "I2900": 1.0,
            "I4200": 1.0, "I4500": 1.0, "I5800": 1.0, "I6200": 1.0,
            "N0415E": 1.0, "N0415H": 1.0,
            "O0110O1": 1.0,
        },
    },
    # ---- Patient 12: Colorectal Cancer + Oral Chemo + Anemia ----------
    {
        "note_id": "TEST0012-DS-01",
        "subject_id": "90000012",
        "hadm_id": "91000012",
        "I0100": True,   # Cancer (colorectal)
        "I0200": True,   # Anemia (chemotherapy-induced)
        "I0300": False,
        "I0400": False,
        "I0500": False,
        "I0600": False,
        "I0700": True,   # Hypertension
        "I1100": False,
        "I1200": False,
        "I1500": False,
        "I2000": False,
        "I2900": False,
        "I3700": False,
        "I4200": False,
        "I4500": False,
        "I5800": False,
        "I5900": False,
        "I6100": False,
        "I6200": False,
        "N0415A": ["2"],
        "N0415B": ["2"],
        "N0415C": ["2"],
        "N0415D": ["2"],
        "N0415E": ["2"],
        "N0415F": ["1"],  # Antibiotic (cefepime IV -> ciprofloxacin PO)
        "N0415G": ["2"],
        "N0415H": ["1"],  # Opioid (oxycodone)
        "N0415I": ["2"],
        "N0415J": ["2"],
        "N0415Z": False,
        "O0110A1": False,   # IV chemo – absent (oral only)
        "O0110A3": True,    # Oral/other chemotherapy (capecitabine)
        "O0110B1": False,
        "O0110C1": False,
        "O0110F1": False,
        "O0110H2": True,    # IV Antibiotics (cefepime during neutropenic fever)
        "O0110I1": True,    # Transfusion (1 unit pRBC)
        "O0110J1": False,
        "O0110K1": False,
        "O0110M1": False,
        "O0110O1": True,    # Peripheral IV access
        "O0110O3": False,
        "confidence": {
            "I0100": 1.0, "I0200": 1.0, "I0700": 1.0,
            "I0300": 1.0, "I0600": 1.0, "I2900": 1.0, "I4200": 1.0,
            "I4500": 1.0, "I5800": 1.0, "I6200": 1.0,
            "N0415F": 1.0, "N0415H": 1.0,
            "O0110A3": 1.0, "O0110H2": 1.0, "O0110I1": 1.0, "O0110O1": 1.0,
        },
    },
    # ---- Patient 13: DKA + Insulin-Dependent DM + Dose Change ---------
    {
        "note_id": "TEST0013-DS-01",
        "subject_id": "90000013",
        "hadm_id": "91000013",
        "I0100": False,
        "I0300": False,
        "I0400": False,
        "I0500": False,
        "I0600": False,
        "I0700": True,   # Hypertension
        "I1100": False,
        "I1500": False,
        "I2000": False,
        "I2900": True,   # Diabetes Mellitus (type 1, insulin-dependent)
        "I3700": False,
        "I4200": False,
        "I4500": False,
        "I5800": False,
        "I5900": False,
        "I6100": False,
        "I6200": False,
        "N0300":  7,      # Total injections past 7 days (insulin every day)
        "N0350A": 7,      # Insulin injections past 7 days
        "N0350B": True,   # Insulin dose change (glargine increased 20→35 units)
        "N0415A": ["2"],
        "N0415B": ["2"],
        "N0415C": ["2"],
        "N0415D": ["2"],
        "N0415E": ["2"],
        "N0415F": ["1"],  # Antibiotic (vancomycin IV for foot infection)
        "N0415G": ["2"],
        "N0415H": ["2"],
        "N0415I": ["2"],
        "N0415J": ["1"],  # Hypoglycemic (insulin)
        "N0415Z": False,
        "O0110A1": False,
        "O0110B1": False,
        "O0110C1": False,
        "O0110F1": False,
        "O0110J1": False,
        "O0110K1": False,
        "O0110M1": False,
        "O0110O1": True,  # Peripheral IV access
        "O0110O3": False,
        "confidence": {
            "I0700": 1.0, "I2900": 1.0,
            "I0100": 1.0, "I0300": 1.0, "I0600": 1.0, "I4200": 1.0,
            "I4500": 1.0, "I5800": 1.0, "I6200": 1.0,
            "N0300": 1.0, "N0350A": 1.0, "N0350B": 1.0,
            "N0415F": 1.0, "N0415J": 1.0,
            "O0110O1": 1.0,
        },
    },
    # ---- Patient 14: MRSA Bacteremia + Isolation + COPD + HTN ---------
    {
        "note_id": "TEST0014-DS-01",
        "subject_id": "90000014",
        "hadm_id": "91000014",
        "I0100": False,
        "I0300": False,
        "I0400": False,
        "I0500": False,
        "I0600": False,
        "I0700": True,   # Hypertension
        "I1100": False,
        "I1500": False,
        "I2000": True,   # Pneumonia (MRSA bacteremia / infection)
        "I2900": False,
        "I3700": False,
        "I4200": False,
        "I4500": False,
        "I5800": False,
        "I5900": False,
        "I6100": False,
        "I6200": True,   # COPD
        "N0415A": ["2"],
        "N0415B": ["2"],
        "N0415C": ["2"],
        "N0415D": ["2"],
        "N0415E": ["2"],
        "N0415F": ["1"],  # Antibiotic (vancomycin IV via PICC)
        "N0415G": ["2"],
        "N0415H": ["2"],
        "N0415I": ["2"],
        "N0415J": ["2"],
        "N0415Z": False,
        "O0110A1": False,
        "O0110B1": False,
        "O0110C1": False,
        "O0110F1": False,
        "O0110H2": True,  # IV Antibiotics (vancomycin via PICC)
        "O0110I1": False,
        "O0110J1": False,
        "O0110K1": False,
        "O0110M1": True,  # Isolation / quarantine (contact + droplet)
        "O0110O1": False,
        "O0110O3": True,  # PICC line for long-term IV antibiotics
        "confidence": {
            "I0700": 1.0, "I2000": 1.0, "I6200": 1.0,
            "I0100": 1.0, "I0300": 1.0, "I0600": 1.0, "I2900": 1.0,
            "I4200": 1.0, "I4500": 1.0, "I5800": 1.0,
            "N0415F": 1.0,
            "O0110H2": 1.0, "O0110M1": 1.0, "O0110O3": 1.0,
        },
    },
    # ---- Patient 15: Terminal Cancer + Hospice + Opioid + Antianxiety -
    {
        "note_id": "TEST0015-DS-01",
        "subject_id": "90000015",
        "hadm_id": "91000015",
        "I0100": True,   # Cancer (NSCLC, stage IV)
        "I0200": False,
        "I0300": False,
        "I0400": False,
        "I0500": False,
        "I0600": False,
        "I0700": False,
        "I1100": False,
        "I1500": False,
        "I2000": False,
        "I2900": False,
        "I3700": False,
        "I4200": False,
        "I4500": False,
        "I5800": True,   # Depression (major depressive disorder)
        "I5900": False,
        "I6100": False,
        "I6200": False,
        "N0415A": ["1"],  # Antipsychotic (haloperidol for terminal agitation)
        "N0415B": ["1"],  # Antianxiety (lorazepam)
        "N0415C": ["2"],
        "N0415D": ["2"],
        "N0415E": ["2"],
        "N0415F": ["2"],
        "N0415G": ["2"],
        "N0415H": ["1"],  # Opioid (morphine)
        "N0415I": ["2"],
        "N0415J": ["2"],
        "N0415Z": False,
        "O0110A1": False,  # No active IV chemo (hospice)
        "O0110B1": False,
        "O0110C1": False,
        "O0110F1": False,
        "O0110H2": False,
        "O0110I1": False,
        "O0110J1": False,
        "O0110K1": True,  # Hospice care
        "O0110M1": False,
        "O0110O1": False,
        "O0110O3": False,
        "confidence": {
            "I0100": 1.0, "I5800": 1.0,
            "I0300": 1.0, "I0600": 1.0, "I2900": 1.0, "I4200": 1.0,
            "I4500": 1.0, "I6200": 1.0,
            "N0415A": 1.0, "N0415B": 1.0, "N0415H": 1.0,
            "O0110K1": 1.0,
        },
    },
]


# ---------------------------------------------------------------------------
# Write helpers
# ---------------------------------------------------------------------------


def write_discharge_csv(path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(
            ["note_id", "subject_id", "hadm_id", "note_type",
             "charttime", "storetime", "text"]
        )
        for note_id, subject_id, hadm_id, sex, service, text in DISCHARGE_NOTES:
            writer.writerow([
                note_id, subject_id, hadm_id,
                "DS",
                "2024-01-01 00:00:00",
                "2024-01-01 12:00:00",
                text,
            ])
    print(f"  Wrote {path}  ({len(DISCHARGE_NOTES)} notes)")


def write_extraction_ground_truth(path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(EXTRACTION_GROUND_TRUTH, f, indent=2)
    print(f"  Wrote {path}  ({len(EXTRACTION_GROUND_TRUTH)} records)")


# ---------------------------------------------------------------------------
# Quick self-check
# ---------------------------------------------------------------------------

def _check_alignment() -> None:
    note_ids_csv = {n[0] for n in DISCHARGE_NOTES}
    note_ids_gt  = {r["note_id"] for r in EXTRACTION_GROUND_TRUTH}
    missing_gt   = note_ids_csv - note_ids_gt
    missing_csv  = note_ids_gt  - note_ids_csv
    if missing_gt:
        print(f"  WARNING: no ground truth for notes: {sorted(missing_gt)}")
    if missing_csv:
        print(f"  WARNING: no discharge note for GT records: {sorted(missing_csv)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Writing test data to: {os.path.abspath(OUT_DIR)}\n")

    write_discharge_csv(
        os.path.join(OUT_DIR, "discharge_test.csv")
    )
    write_extraction_ground_truth(
        os.path.join(OUT_DIR, "extraction_ground_truth.json")
    )
    _check_alignment()

    print(f"\nDone. {len(DISCHARGE_NOTES)} synthetic patients.\n")
    print(f"{'note_id':<22} {'service':<12}  {'+ fields':>8}  {'- fields':>8}")
    print("-" * 58)
    for note_id, _, _, _, service, _ in DISCHARGE_NOTES:
        gt = next(r for r in EXTRACTION_GROUND_TRUTH if r["note_id"] == note_id)
        fields = {k: v for k, v in gt.items()
                  if k not in {"note_id", "subject_id", "hadm_id", "confidence"}}
        pos = sum(1 for v in fields.values()
                  if v not in (False, None, 0, [], ["2"]))
        neg = sum(1 for v in fields.values()
                  if v in (False, 0) or v == ["2"])
        print(f"{note_id:<22} {service:<12}  {pos:>8}  {neg:>8}")


if __name__ == "__main__":
    main()
