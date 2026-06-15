import type { Lens, PatientCase } from "./types";

export const lenses: Lens[] = [
  {
    id: "medicine",
    label: "Medicine",
    question:
      "What is wrong with this person, how serious is it, and what should happen now?",
    example:
      "A cough becomes a bedside decision when fever, low oxygen, kidney disease, and confusion change risk.",
    output:
      "Diagnosis, prognosis, care setting, treatment plan, and communication.",
  },
  {
    id: "biology",
    label: "Biology",
    question: "How does the body work?",
    example:
      "Inflammation in the lung air spaces explains why oxygen exchange can fail.",
    output: "Mechanism of normal or abnormal function.",
  },
  {
    id: "public-health",
    label: "Public health",
    question: "How do we improve health across a population?",
    example:
      "Vaccination, air quality, outbreak detection, and prevention reduce respiratory infections.",
    output:
      "Policy, prevention, surveillance, and population-level allocation.",
  },
  {
    id: "clinical-research",
    label: "Clinical research",
    question: "How do we know whether an intervention works?",
    example:
      "A trial estimates whether a drug lowers mortality or complications across many patients.",
    output: "Evidence strength, effect estimates, and applicability limits.",
  },
  {
    id: "healthcare",
    label: "Healthcare delivery",
    question: "How is care organized and delivered?",
    example:
      "Triage, staffing, bed availability, cost, and follow-up shape what can actually happen.",
    output: "Workflow, operations, access, financing, and quality design.",
  },
];

export const patientCases: PatientCase[] = [
  {
    id: "chest-pain",
    title: "Chest Pain Under Time Pressure",
    chiefComplaint: "Chest pain",
    setting: "Emergency department triage",
    opening:
      "A 62-year-old man arrives with chest pain that started while walking uphill. He looks anxious but is speaking in full sentences.",
    patientVoice:
      "It feels like pressure in the center of my chest. It started less than an hour ago and now my left arm feels strange.",
    coreQuestion:
      "Is this a common benign symptom, a dangerous cardiac pattern, or another time-sensitive diagnosis?",
    thinRepresentation: "Patient has chest pain.",
    modelRepresentation:
      "A 62-year-old man with diabetes, hypertension, and long smoking history presents with 45 minutes of exertional central chest pressure radiating to the left arm, associated with nausea and sweating.",
    evidence: [
      {
        id: "age-sex",
        label: "62-year-old man",
        detail:
          "Age and sex shift the prior probability of cardiovascular disease.",
        kind: "context",
        stage: "arrival",
        representationRole: "patient",
        representationPhrase: "62-year-old man",
        riskWeight: 10,
      },
      {
        id: "central-pressure",
        label: "Central chest pressure",
        detail:
          "Subjective symptom; pressure-like quality fits a cardiac illness script better than a sharp surface pain.",
        kind: "symptom",
        stage: "arrival",
        representationRole: "syndrome",
        representationPhrase: "central chest pressure",
        riskWeight: 14,
      },
      {
        id: "time-onset",
        label: "Started 45 minutes ago",
        detail:
          "Acute timing makes the encounter a now-decision, not only a diagnostic puzzle.",
        kind: "context",
        stage: "history",
        representationRole: "timing",
        representationPhrase: "45 minutes of acute symptoms",
        riskWeight: 10,
      },
      {
        id: "exertional",
        label: "Began while walking uphill",
        detail: "Exertional trigger increases concern for myocardial ischemia.",
        kind: "context",
        stage: "history",
        representationRole: "danger",
        representationPhrase: "exertional onset",
        riskWeight: 14,
      },
      {
        id: "radiation",
        label: "Radiates to the left arm",
        detail:
          "A symptom pattern that increases fit with acute coronary syndrome.",
        kind: "symptom",
        stage: "history",
        representationRole: "danger",
        representationPhrase: "radiating to the left arm",
        riskWeight: 15,
      },
      {
        id: "nausea-sweat",
        label: "Nausea and sweating",
        detail: "Associated symptoms that strengthen a cardiac pattern.",
        kind: "symptom",
        stage: "history",
        representationRole: "danger",
        representationPhrase: "with nausea and sweating",
        riskWeight: 10,
      },
      {
        id: "diabetes",
        label: "Type 2 diabetes",
        detail:
          "Comorbidity and cardiovascular risk factor; it also changes prognosis.",
        kind: "comorbidity",
        stage: "context",
        representationRole: "risk",
        representationPhrase: "diabetes",
        riskWeight: 9,
      },
      {
        id: "hypertension",
        label: "Hypertension",
        detail: "Chronic condition that increases cardiovascular risk.",
        kind: "comorbidity",
        stage: "context",
        representationRole: "risk",
        representationPhrase: "hypertension",
        riskWeight: 8,
      },
      {
        id: "smoking",
        label: "Smoked for 30 years",
        detail: "Exposure history shifts pre-test probability.",
        kind: "context",
        stage: "context",
        representationRole: "risk",
        representationPhrase: "long smoking history",
        riskWeight: 8,
      },
      {
        id: "ecg",
        label: "ECG: ischemic changes",
        detail:
          "A test result that sharply raises the need for urgent cardiac management.",
        kind: "test",
        stage: "tests",
        riskWeight: 20,
      },
      {
        id: "troponin",
        label: "Troponin elevated",
        detail:
          "Useful in context; not interpreted as an isolated truth machine.",
        kind: "test",
        stage: "tests",
        riskWeight: 16,
      },
    ],
    diagnoses: [
      {
        id: "acs",
        name: "Acute coronary syndrome",
        category: "Cardiac",
        likelihood: 90,
        danger: 96,
        urgency: 98,
        why: "The story strongly fits ischemia and delay has high cost.",
        discriminators: [
          "ECG",
          "troponin",
          "vital signs",
          "response to initial management",
        ],
      },
      {
        id: "dissection",
        name: "Aortic dissection",
        category: "Vascular",
        likelihood: 22,
        danger: 99,
        urgency: 95,
        why: "Less likely here, but catastrophic enough to keep in view if the story changes.",
        discriminators: [
          "tearing pain",
          "pulse deficit",
          "neurologic signs",
          "imaging",
        ],
      },
      {
        id: "pe",
        name: "Pulmonary embolism",
        category: "Pulmonary",
        likelihood: 34,
        danger: 88,
        urgency: 84,
        why: "Can cause chest pain and shortness of breath; needs more context.",
        discriminators: [
          "oxygen saturation",
          "leg swelling",
          "risk factors",
          "CT pulmonary angiography",
        ],
      },
      {
        id: "reflux",
        name: "Reflux or esophageal spasm",
        category: "Gastrointestinal",
        likelihood: 28,
        danger: 18,
        urgency: 16,
        why: "Possible, but a benign explanation should not end the reasoning before danger is handled.",
        discriminators: [
          "meal relation",
          "burning quality",
          "normal cardiac evaluation",
        ],
      },
      {
        id: "strain",
        name: "Muscle strain",
        category: "Musculoskeletal",
        likelihood: 18,
        danger: 10,
        urgency: 10,
        why: "Common, but the exertional pressure pattern and risk factors do not fit well.",
        discriminators: ["reproducible tenderness", "movement-related pain"],
      },
    ],
    planOptions: [
      {
        id: "ecg-now",
        label: "Get ECG now",
        type: "test",
        value: 22,
        feedback:
          "High value because it can identify a time-sensitive cardiac pattern.",
      },
      {
        id: "troponin",
        label: "Order troponin",
        type: "test",
        value: 14,
        feedback:
          "Useful, but interpreted with the story and ECG rather than by itself.",
      },
      {
        id: "monitor",
        label: "Continuous monitoring",
        type: "setting",
        value: 16,
        feedback:
          "Appropriate because the patient may deteriorate or need rapid intervention.",
      },
      {
        id: "reassure",
        label: "Reassure and discharge",
        type: "setting",
        value: -28,
        feedback:
          "Unsafe: it accepts a benign explanation before dangerous causes are handled.",
      },
      {
        id: "antacid-only",
        label: "Treat as reflux only",
        type: "treat",
        value: -20,
        feedback:
          "Premature closure. The symptom pattern and risk factors demand cardiac evaluation.",
      },
      {
        id: "aspirin-selected",
        label: "Consider aspirin if no contraindication",
        type: "treat",
        value: 12,
        feedback:
          "Reasonable in a suspected acute coronary syndrome pathway, with clinical safeguards.",
      },
    ],
    idealPlanIds: ["ecg-now", "troponin", "monitor", "aspirin-selected"],
    unsafePlanIds: ["reassure", "antacid-only"],
    debrief: [
      "Most likely and most dangerous must be considered together.",
      "The same chest pain symptom has different meaning in a high-risk exertional story.",
      "Tests update probability; they do not replace the clinical representation.",
    ],
  },
  {
    id: "febrile-confusion",
    title: "Confusion, Cough, and Instability",
    chiefComplaint: "Confusion",
    setting: "Hospital arrival",
    opening:
      "A 74-year-old woman is brought in by her daughter because she has been confused for one day after several days of cough and poor appetite.",
    patientVoice:
      "Her daughter says: She is not herself today. She has been coughing and barely eating. She usually knows where she is.",
    coreQuestion:
      "Is this just a respiratory infection, or has the patient crossed a severity threshold that changes the care setting?",
    thinRepresentation: "Patient has a cough.",
    modelRepresentation:
      "A 74-year-old woman with diabetes and chronic kidney disease presents with acute confusion, cough, fever, tachycardia, hypotension, tachypnea, and hypoxemia.",
    evidence: [
      {
        id: "age",
        label: "74-year-old woman",
        detail:
          "Age changes risk, presentation, and tolerance of acute illness.",
        kind: "context",
        stage: "arrival",
        representationRole: "patient",
        representationPhrase: "74-year-old woman",
        riskWeight: 8,
      },
      {
        id: "confusion",
        label: "Confused for one day",
        detail:
          "A symptom and observed mental-status change; in older adults this can be a danger signal.",
        kind: "symptom",
        stage: "arrival",
        representationRole: "syndrome",
        representationPhrase: "acute confusion",
        riskWeight: 16,
      },
      {
        id: "cough",
        label: "Cough for three days",
        detail:
          "Respiratory symptom that points toward infection but is nonspecific alone.",
        kind: "symptom",
        stage: "history",
        representationRole: "syndrome",
        representationPhrase: "three days of cough",
        riskWeight: 6,
      },
      {
        id: "poor-appetite",
        label: "Poor appetite",
        detail:
          "Nonspecific symptom that matters when paired with frailty, infection, or dehydration.",
        kind: "symptom",
        stage: "history",
        representationRole: "context",
        representationPhrase: "poor appetite",
        riskWeight: 4,
      },
      {
        id: "diabetes",
        label: "Diabetes",
        detail:
          "Comorbidity that increases infection risk and affects recovery.",
        kind: "comorbidity",
        stage: "context",
        representationRole: "risk",
        representationPhrase: "diabetes",
        riskWeight: 8,
      },
      {
        id: "kidney",
        label: "Chronic kidney disease",
        detail:
          "Changes drug dosing, test interpretation, and physiologic reserve.",
        kind: "comorbidity",
        stage: "context",
        representationRole: "risk",
        representationPhrase: "chronic kidney disease",
        riskWeight: 9,
      },
      {
        id: "temp",
        label: "Temperature 38.7 C",
        detail: "Measured fever, a sign of inflammation or infection.",
        kind: "sign",
        stage: "exam",
        representationRole: "danger",
        representationPhrase: "fever",
        riskWeight: 10,
      },
      {
        id: "hr",
        label: "Heart rate 112",
        detail:
          "Tachycardia can reflect stress, fever, shock, pain, or arrhythmia.",
        kind: "sign",
        stage: "exam",
        representationRole: "danger",
        representationPhrase: "tachycardia",
        riskWeight: 10,
      },
      {
        id: "bp",
        label: "Blood pressure 92/58",
        detail: "Hypotension is an immediate severity signal.",
        kind: "sign",
        stage: "exam",
        representationRole: "danger",
        representationPhrase: "hypotension",
        riskWeight: 18,
      },
      {
        id: "rr",
        label: "Respiratory rate 26",
        detail:
          "Often underappreciated; fast breathing can be an early marker of deterioration.",
        kind: "sign",
        stage: "exam",
        representationRole: "danger",
        representationPhrase: "tachypnea",
        riskWeight: 15,
      },
      {
        id: "o2",
        label: "Oxygen saturation 89%",
        detail:
          "Measured gas-exchange problem and a strong care-setting signal.",
        kind: "sign",
        stage: "exam",
        representationRole: "danger",
        representationPhrase: "hypoxemia",
        riskWeight: 18,
      },
      {
        id: "xray",
        label: "Chest X-ray: right lower lobe opacity",
        detail: "Supports pneumonia but does not by itself determine severity.",
        kind: "test",
        stage: "tests",
        riskWeight: 10,
      },
      {
        id: "lactate",
        label: "Lactate elevated",
        detail: "Suggests impaired perfusion or severe systemic stress.",
        kind: "test",
        stage: "tests",
        riskWeight: 16,
      },
    ],
    diagnoses: [
      {
        id: "pneumonia-sepsis",
        name: "Pneumonia with sepsis",
        category: "Infectious",
        likelihood: 88,
        danger: 97,
        urgency: 98,
        why: "Respiratory symptoms plus fever, hypoxemia, hypotension, tachypnea, and confusion form a dangerous syndrome.",
        discriminators: [
          "chest X-ray",
          "oxygen requirement",
          "blood pressure trend",
          "lactate",
        ],
      },
      {
        id: "uti-delirium",
        name: "Urinary infection with delirium",
        category: "Infectious",
        likelihood: 42,
        danger: 76,
        urgency: 74,
        why: "Older adults can present with delirium, but respiratory signs still need explanation.",
        discriminators: ["urinalysis", "culture", "respiratory findings"],
      },
      {
        id: "heart-failure",
        name: "Heart failure exacerbation",
        category: "Cardiopulmonary",
        likelihood: 36,
        danger: 84,
        urgency: 78,
        why: "Can cause hypoxemia and fast breathing; fever and infection signs may coexist.",
        discriminators: ["lung exam", "chest X-ray", "BNP", "fluid status"],
      },
      {
        id: "pe",
        name: "Pulmonary embolism",
        category: "Pulmonary",
        likelihood: 28,
        danger: 89,
        urgency: 84,
        why: "Hypoxemia and tachycardia can fit, though cough and fever point elsewhere.",
        discriminators: [
          "risk factors",
          "leg symptoms",
          "CT pulmonary angiography",
        ],
      },
      {
        id: "viral",
        name: "Viral respiratory infection",
        category: "Infectious",
        likelihood: 45,
        danger: 48,
        urgency: 42,
        why: "Possible start of the story, but instability makes mild outpatient framing unsafe.",
        discriminators: ["severity markers", "oxygen need", "viral testing"],
      },
    ],
    planOptions: [
      {
        id: "oxygen",
        label: "Give oxygen and monitor saturation",
        type: "treat",
        value: 18,
        feedback:
          "Appropriate because low oxygen is a direct instability signal.",
      },
      {
        id: "cultures-antibiotics",
        label: "Cultures and prompt antibiotics",
        type: "treat",
        value: 20,
        feedback:
          "Reasonable when severe infection is likely and delay increases harm.",
      },
      {
        id: "iv-fluids",
        label: "Assess perfusion and consider IV fluids",
        type: "treat",
        value: 12,
        feedback:
          "The low blood pressure means perfusion must be assessed and supported carefully.",
      },
      {
        id: "admit",
        label: "Admit for close monitoring",
        type: "setting",
        value: 18,
        feedback:
          "The care setting changes because confusion, hypoxemia, and hypotension are danger signals.",
      },
      {
        id: "home-cough",
        label: "Send home with cough medicine",
        type: "setting",
        value: -32,
        feedback: "Unsafe: this ignores instability and comorbid risk.",
      },
      {
        id: "wait-for-certainty",
        label: "Wait for every test before acting",
        type: "test",
        value: -18,
        feedback:
          "Medicine often acts before certainty when the risk of delay is high.",
      },
    ],
    idealPlanIds: ["oxygen", "cultures-antibiotics", "iv-fluids", "admit"],
    unsafePlanIds: ["home-cough", "wait-for-certainty"],
    debrief: [
      "This is more than a cough because vital signs and mental status reveal severity.",
      "A disease label is not enough; the care setting depends on risk and trajectory.",
      "Comorbidities and physiologic reserve change what the same infection means.",
    ],
  },
];
