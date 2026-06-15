import type { Diagnosis, EvidenceItem, PatientCase, PlanOption } from "./types";

type RepresentationRole = NonNullable<EvidenceItem["representationRole"]>;
type RepresentableEvidence = EvidenceItem & {
  representationPhrase: string;
  representationRole: RepresentationRole;
};

function isRepresentable(item: EvidenceItem): item is RepresentableEvidence {
  return Boolean(item.representationPhrase && item.representationRole);
}

export function byPriority(diagnoses: Diagnosis[]) {
  return [...diagnoses].sort((a, b) => priorityScore(b) - priorityScore(a));
}

export function priorityScore(diagnosis: Diagnosis) {
  return Math.round(
    diagnosis.likelihood * 0.35 +
      diagnosis.danger * 0.35 +
      diagnosis.urgency * 0.3,
  );
}

export function revealedRisk(evidence: EvidenceItem[], revealedIds: string[]) {
  const revealed = evidence.filter((item) => revealedIds.includes(item.id));
  return Math.min(
    100,
    revealed.reduce((total, item) => total + item.riskWeight, 0),
  );
}

export function makeProblemRepresentation(
  patientCase: PatientCase,
  selectedEvidenceIds: string[],
  useModel = false,
) {
  if (useModel) return patientCase.modelRepresentation;

  const selected = patientCase.evidence.filter(
    (item): item is RepresentableEvidence =>
      selectedEvidenceIds.includes(item.id) && isRepresentable(item),
  );

  if (selected.length === 0) return patientCase.thinRepresentation;

  const orderedRoles = [
    "patient",
    "risk",
    "timing",
    "syndrome",
    "danger",
    "context",
  ] as const;
  const phrases = orderedRoles.flatMap((role) =>
    selected
      .filter((item) => item.representationRole === role)
      .map((item) => item.representationPhrase),
  );

  return `Current representation: ${phrases.join("; ")}.`;
}

export function representationCompleteness(
  patientCase: PatientCase,
  selectedEvidenceIds: string[],
) {
  const neededRoles = new Set(
    patientCase.evidence
      .filter(
        (
          item,
        ): item is EvidenceItem & { representationRole: RepresentationRole } =>
          Boolean(item.representationRole && item.riskWeight >= 8),
      )
      .map((item) => item.representationRole),
  );
  const coveredRoles = new Set(
    patientCase.evidence
      .filter(
        (
          item,
        ): item is EvidenceItem & { representationRole: RepresentationRole } =>
          Boolean(
            selectedEvidenceIds.includes(item.id) && item.representationRole,
          ),
      )
      .map((item) => item.representationRole),
  );
  const missing = [...neededRoles].filter((role) => !coveredRoles.has(role));
  const score =
    neededRoles.size === 0
      ? 1
      : (neededRoles.size - missing.length) / neededRoles.size;

  return { score, missing };
}

export function planScore(caseData: PatientCase, selectedPlanIds: string[]) {
  const selected = caseData.planOptions.filter((option) =>
    selectedPlanIds.includes(option.id),
  );
  const raw = selected.reduce((total, option) => total + option.value, 0);
  const idealCovered = caseData.idealPlanIds.filter((id) =>
    selectedPlanIds.includes(id),
  ).length;
  const unsafeChosen = caseData.unsafePlanIds.filter((id) =>
    selectedPlanIds.includes(id),
  ).length;
  const bounded = Math.max(
    0,
    Math.min(100, 25 + raw + idealCovered * 4 - unsafeChosen * 20),
  );

  return Math.round(bounded);
}

export function planFeedback(option: PlanOption, selected: boolean) {
  return selected ? option.feedback : "";
}
