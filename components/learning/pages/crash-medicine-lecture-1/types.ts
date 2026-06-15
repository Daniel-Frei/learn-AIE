export type EvidenceKind =
  | "symptom"
  | "sign"
  | "context"
  | "comorbidity"
  | "test"
  | "value";

export type EvidenceStage =
  | "arrival"
  | "history"
  | "exam"
  | "tests"
  | "context";

export type EvidenceItem = {
  id: string;
  label: string;
  detail: string;
  kind: EvidenceKind;
  stage: EvidenceStage;
  representationRole?:
    | "patient"
    | "timing"
    | "syndrome"
    | "risk"
    | "danger"
    | "context";
  representationPhrase?: string;
  riskWeight: number;
};

export type Diagnosis = {
  id: string;
  name: string;
  category: string;
  likelihood: number;
  danger: number;
  urgency: number;
  why: string;
  discriminators: string[];
};

export type PlanOption = {
  id: string;
  label: string;
  type: "test" | "treat" | "setting";
  value: number;
  feedback: string;
};

export type PatientCase = {
  id: string;
  title: string;
  chiefComplaint: string;
  opening: string;
  setting: string;
  patientVoice: string;
  coreQuestion: string;
  evidence: EvidenceItem[];
  diagnoses: Diagnosis[];
  planOptions: PlanOption[];
  idealPlanIds: string[];
  unsafePlanIds: string[];
  modelRepresentation: string;
  thinRepresentation: string;
  debrief: string[];
};

export type Lens = {
  id: string;
  label: string;
  question: string;
  example: string;
  output: string;
};
