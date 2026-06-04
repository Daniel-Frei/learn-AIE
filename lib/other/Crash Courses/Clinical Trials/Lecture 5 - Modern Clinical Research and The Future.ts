import { Question } from "../../../quiz";

export const ClinicalTrialsLecture5Questions: Question[] = [
  {
    id: "clinical-trials-l5-q01",
    chapter: 5,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe the central direction of modern clinical research?",
    options: [
      {
        text: "Clinical research is expanding beyond traditional pharmaceuticals to include devices, diagnostics, software, digital therapeutics, and AI-based tools.",
        isCorrect: true,
      },
      {
        text: "Real-world evidence, decentralized methods, adaptive designs, and AI tools are becoming more important parts of evidence generation.",
        isCorrect: true,
      },
      {
        text: "The core question remains whether an intervention truly benefits patients in humans.",
        isCorrect: true,
      },
      {
        text: "New tools change data sources and workflows, but they do not remove the need for trustworthy safety and benefit-risk evidence.",
        isCorrect: true,
      },
    ],
    explanation:
      "Modern clinical research is becoming more digital, data-driven, and AI-assisted, but the basic epistemic problem has not disappeared. New tools can improve evidence generation, but they still need to answer whether interventions help or harm real patients.",
  },
  {
    id: "clinical-trials-l5-q02",
    chapter: 5,
    difficulty: "easy",
    prompt:
      "Which statements correctly compare evidence generation for drugs and medical devices?",
    options: [
      {
        text: "Drugs often fit staged development with defined dose, route, schedule, and randomized controlled trials.",
        isCorrect: true,
      },
      {
        text: "Devices may depend on design iterations, operator skill, learning curves, workflow, and technical performance.",
        isCorrect: true,
      },
      {
        text: "Device evidence is simpler because blinding and placebo control are easier for implants and surgical robots than for pills.",
        isCorrect: false,
      },
      {
        text: "Drug and device evidence ask identical questions because both products enter the body and require regulatory review.",
        isCorrect: false,
      },
    ],
    explanation:
      "Drug trials often evaluate a relatively stable intervention at a defined dose and schedule. Devices can evolve during development, may require user training, may be hard to blind, and may need evidence that technical performance translates into patient benefit.",
  },
  {
    id: "clinical-trials-l5-q03",
    chapter: 5,
    difficulty: "easy",
    prompt:
      "A diagnostic test is being evaluated before routine clinical use. Which question most directly addresses clinical utility?",
    options: [
      {
        text: "Does using the test improve clinical decisions or patient outcomes?",
        isCorrect: true,
      },
      {
        text: "Does the assay measure the intended analyte with repeatable laboratory precision?",
        isCorrect: false,
      },
      {
        text: "Does the test identify disease status in a curated validation dataset?",
        isCorrect: false,
      },
      {
        text: "Does the manufacturer provide an updated software interface for displaying results?",
        isCorrect: false,
      },
    ],
    explanation:
      "Clinical utility asks whether using a diagnostic changes management or improves outcomes. Analytical validity asks whether the test measures what it claims to measure, while clinical validity asks whether the test identifies or predicts the clinical condition of interest.",
  },
  {
    id: "clinical-trials-l5-q04",
    chapter: 5,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe diagnostic-test metrics and interpretation?",
    options: [
      {
        text: "Sensitivity is the proportion of people with the disease who are correctly identified as positive.",
        isCorrect: true,
      },
      {
        text: "Specificity is the proportion of people without the disease who are correctly classified as negative.",
        isCorrect: true,
      },
      {
        text: "Positive predictive value depends on how common the disease is in the tested population.",
        isCorrect: true,
      },
      {
        text: "A high-sensitivity diagnostic automatically has strong clinical utility even when results do not change treatment decisions.",
        isCorrect: false,
      },
    ],
    explanation:
      "Sensitivity, specificity, and predictive values answer different diagnostic questions. A test can identify disease accurately and still have limited utility if the result does not affect treatment, prevention, or patient outcomes.",
  },
  {
    id: "clinical-trials-l5-q05",
    chapter: 5,
    difficulty: "easy",
    prompt:
      "Which evidence questions are especially important for Software as a Medical Device (SaMD) and AI-based medical products?",
    options: [
      {
        text: "Whether the evaluated version remains representative after software updates, model retraining, workflow changes, or data-pipeline changes.",
        isCorrect: true,
      },
      {
        text: "Whether performance generalizes across patient subgroups, devices, clinical settings, countries, and input conditions.",
        isCorrect: true,
      },
      {
        text: "Whether regulatory review can focus on cybersecurity and manufacturing documentation while treating post-deployment clinical performance as unchanged after updates.",
        isCorrect: false,
      },
      {
        text: "Whether model updates remove responsibility from clinicians, hospitals, developers, and sponsors when errors occur.",
        isCorrect: false,
      },
    ],
    explanation:
      "Software can change faster than traditional medical products, so the evaluated version and future versions may not be identical. AI-based products also require attention to generalization, subgroup performance, clinical impact, update governance, and accountability.",
  },
  {
    id: "clinical-trials-l5-q06",
    chapter: 5,
    difficulty: "easy",
    prompt:
      "Which statement best distinguishes real-world data from real-world evidence?",
    options: [
      {
        text: "Real-world data are routine-care or everyday-life data sources, while real-world evidence is clinical evidence generated by analyzing those data for a question.",
        isCorrect: true,
      },
      {
        text: "Real-world evidence is the raw data extracted from electronic systems before cleaning, and real-world data are the published conclusions.",
        isCorrect: false,
      },
      {
        text: "Real-world data come from randomized trials, while real-world evidence comes from animal and laboratory experiments.",
        isCorrect: false,
      },
      {
        text: "Real-world evidence is a registry type, while real-world data are limited to insurance claims.",
        isCorrect: false,
      },
    ],
    explanation:
      "Real-World Data (RWD) are the raw material, such as records, claims, registries, wearables, or apps. Real-World Evidence (RWE) is produced when those data are analyzed with a clinical question and study design in mind.",
  },
  {
    id: "clinical-trials-l5-q07",
    chapter: 5,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe major sources of real-world data?",
    options: [
      {
        text: "Electronic Health Records (EHRs) can contain diagnoses, medications, procedures, labs, imaging reports, notes, vital signs, and hospitalizations.",
        isCorrect: true,
      },
      {
        text: "Claims data can capture diagnoses, procedures, prescriptions, utilization, and reimbursement-linked treatment patterns.",
        isCorrect: true,
      },
      {
        text: "Disease registries can collect structured long-term data for specific conditions, procedures, or patient groups.",
        isCorrect: true,
      },
      {
        text: "Claims data usually contain detailed symptom severity, imaging interpretation, and patient-reported outcomes for every visit.",
        isCorrect: false,
      },
    ],
    explanation:
      "EHRs, claims, and registries each have strengths and weaknesses. EHRs may be clinically rich but messy, claims are large and useful for utilization but often lack clinical detail, and registries can provide structured disease-specific follow-up.",
  },
  {
    id: "clinical-trials-l5-q08",
    chapter: 5,
    difficulty: "easy",
    prompt:
      "Which uses of real-world evidence are common in clinical development and post-market research?",
    options: [
      {
        text: "Safety monitoring for rare or long-term adverse events after broader use.",
        isCorrect: true,
      },
      {
        text: "Comparative effectiveness studies in routine clinical practice.",
        isCorrect: true,
      },
      {
        text: "Post-market commitments, selected regulatory submissions, external-control contexts, or label-expansion support.",
        isCorrect: true,
      },
      {
        text: "Treatment-pattern studies that examine adherence, access, utilization, and real-world patient populations.",
        isCorrect: true,
      },
    ],
    explanation:
      "RWE can complement trial evidence by showing what happens in broader and longer-term routine-care settings. It is especially useful for safety surveillance, utilization, comparative effectiveness, rare-disease contexts, and post-approval evidence needs.",
  },
  {
    id: "clinical-trials-l5-q09",
    chapter: 5,
    difficulty: "easy",
    prompt:
      "Patients receiving Drug A in routine-care data have better outcomes than patients receiving Drug B. Which interpretations are appropriate?",
    options: [
      {
        text: "Drug A may be better, but the comparison may also reflect differences in patient severity, physician choice, hospital quality, access, or earlier diagnosis.",
        isCorrect: true,
      },
      {
        text: "Confounding by indication is a concern because the reason a patient receives a treatment can be related to prognosis.",
        isCorrect: true,
      },
      {
        text: "The observational comparison is automatically equivalent to a randomized treatment comparison because the dataset is large, routine-care based, and drawn from many hospitals.",
        isCorrect: false,
      },
      {
        text: "Causal inference concerns are resolved once an electronic health record contains diagnosis and medication fields.",
        isCorrect: false,
      },
    ],
    explanation:
      "Real-world comparisons are vulnerable to confounding because treatment choices are not randomized. Methods such as matching, adjustment, target-trial emulation, and sensitivity analysis can help, but unmeasured confounding may remain.",
  },
  {
    id: "clinical-trials-l5-q10",
    chapter: 5,
    difficulty: "easy",
    prompt:
      "Which statement best describes the future relationship between randomized trials and real-world evidence?",
    options: [
      {
        text: "The future will combine randomized trials for high internal validity with real-world evidence for scale, representativeness, long-term follow-up, and routine-care context.",
        isCorrect: true,
      },
      {
        text: "Real-world evidence will replace randomized trials for causal efficacy questions once datasets include enough patients, long follow-up, and advanced prediction models.",
        isCorrect: false,
      },
      {
        text: "Randomized trials will replace registries, claims, and EHR studies after approval because post-market questions are already settled.",
        isCorrect: false,
      },
      {
        text: "Randomized trials and real-world evidence answer the same question with the same bias structure and the same operating costs.",
        isCorrect: false,
      },
    ],
    explanation:
      "Randomized trials and RWE provide different strengths. Trials control bias and support causal inference, while RWE can expand evidence into broader populations, longer follow-up, safety surveillance, and routine-care performance.",
  },
  {
    id: "clinical-trials-l5-q11",
    chapter: 5,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe decentralized and hybrid clinical trials?",
    options: [
      {
        text: "Decentralized Clinical Trials (DCTs) move trial activities closer to the patient instead of requiring every activity at a central site.",
        isCorrect: true,
      },
      {
        text: "Most future models are likely to be hybrid, combining traditional sites with selected remote components.",
        isCorrect: true,
      },
      {
        text: "Remote components can reduce travel burden and may help access for patients far from major research centers.",
        isCorrect: true,
      },
      {
        text: "Fully decentralized models are the default for procedures such as biopsies, complex infusions, and specialized imaging.",
        isCorrect: false,
      },
    ],
    explanation:
      "Decentralization is about moving appropriate activities closer to patients, not removing sites from every trial. Hybrid models preserve site-based oversight for complex procedures while using remote tools for consent, follow-up, questionnaires, home measurements, and selected safety checks.",
  },
  {
    id: "clinical-trials-l5-q12",
    chapter: 5,
    difficulty: "easy",
    prompt:
      "Which components can appear in decentralized or hybrid trial operations?",
    options: [
      {
        text: "Telemedicine visits and remote physician assessments.",
        isCorrect: true,
      },
      {
        text: "Home nursing for blood collection, vital signs, drug administration, or safety checks.",
        isCorrect: true,
      },
      {
        text: "Electronic consent with digital review, comprehension support, and electronic signature.",
        isCorrect: true,
      },
      {
        text: "Wearables, smartphone apps, remote patient-reported outcomes, local laboratories, and remote monitoring.",
        isCorrect: true,
      },
    ],
    explanation:
      "Hybrid trials can use several technologies and services to make participation easier and collect richer data. These components still need quality control, privacy protections, participant support, and integration with the trial's data and safety workflows.",
  },
  {
    id: "clinical-trials-l5-q13",
    chapter: 5,
    difficulty: "easy",
    prompt: "Which statement best describes a digital endpoint?",
    options: [
      {
        text: "An outcome measured using digital tools, such as connected home blood-pressure devices, wearables, home spirometry, or smartphone-based symptom reporting.",
        isCorrect: true,
      },
      {
        text: "A traditional clinic endpoint that becomes digital once the final study report, source documents, and monitoring notes are stored in an electronic Trial Master File.",
        isCorrect: false,
      },
      {
        text: "An administrative measure of how quickly sites activate after electronic consent forms are approved.",
        isCorrect: false,
      },
      {
        text: "A statistical rule that changes randomization ratios after an interim analysis.",
        isCorrect: false,
      },
    ],
    explanation:
      "Digital endpoints are clinical outcomes captured through digital tools, often outside the clinic or at higher frequency than traditional assessments. They can add richer information, but they require validation, reliable devices, data-quality plans, and regulatory acceptance.",
  },
  {
    id: "clinical-trials-l5-q14",
    chapter: 5,
    difficulty: "easy",
    prompt:
      "A remote trial relies on smartphone apps, wearables, and home measurements. Which challenge is most specific to this model?",
    options: [
      {
        text: "Device reliability, patient compliance, privacy, digital access, data volume, missing data, and endpoint validation must be managed carefully.",
        isCorrect: true,
      },
      {
        text: "Participants need support for charging, wearing, using, and troubleshooting devices so measurements remain interpretable.",
        isCorrect: true,
      },
      {
        text: "Large streams of sensor and app data require plans for artifacts, missingness, feature selection, storage, and review.",
        isCorrect: true,
      },
      {
        text: "Regulators may ask whether a digital endpoint is validated and clinically meaningful for the decision being made.",
        isCorrect: true,
      },
    ],
    explanation:
      "Digital and remote methods solve some access problems while creating new operational and evidence problems. The trial still needs reliable devices, participant adherence, privacy protections, representative access, validated endpoints, and interpretable data handling.",
  },
  {
    id: "clinical-trials-l5-q15",
    chapter: 5,
    difficulty: "medium",
    prompt: "Which statements correctly describe adaptive trial designs?",
    options: [
      {
        text: "They allow planned modifications based on accumulating trial data.",
        isCorrect: true,
      },
      {
        text: "Adaptations can include sample-size re-estimation, dose selection, dropping ineffective arms, stopping for futility, or modifying randomization ratios.",
        isCorrect: true,
      },
      {
        text: "The adaptation rules should be pre-specified with statistical control to avoid bias.",
        isCorrect: true,
      },
      {
        text: "Adaptive designs let teams choose new primary endpoints after seeing which results are most favorable.",
        isCorrect: false,
      },
    ],
    explanation:
      "Adaptive designs try to learn more efficiently during a trial, but they are not free improvisation. The adaptation rules must be planned in advance, aligned with regulators when needed, and implemented with data quality and operational discipline.",
  },
  {
    id: "clinical-trials-l5-q16",
    chapter: 5,
    difficulty: "medium",
    prompt:
      "Which examples are valid adaptive-design features when planned in advance?",
    options: [
      {
        text: "Dropping a low-dose arm through an informal decision after unblinded review that was not described in the protocol.",
        isCorrect: false,
      },
      {
        text: "Increasing sample size under a pre-planned rule when the event rate is lower than expected.",
        isCorrect: true,
      },
      {
        text: "Stopping early for futility or overwhelming efficacy under pre-defined criteria.",
        isCorrect: true,
      },
      {
        text: "Changing the statistical analysis plan after unblinded data reveal an unplanned subgroup benefit.",
        isCorrect: false,
      },
    ],
    explanation:
      "Adaptive features can save time, resources, and patient exposure when they are pre-specified. Informal changes after unblinded review and post hoc analysis-plan changes can introduce bias and damage credibility.",
  },
  {
    id: "clinical-trials-l5-q17",
    chapter: 5,
    difficulty: "medium",
    prompt:
      "Which statements correctly compare platform, basket, and umbrella trials?",
    options: [
      {
        text: "A platform trial uses shared infrastructure to evaluate multiple interventions, with arms entering or leaving over time.",
        isCorrect: true,
      },
      {
        text: "A basket trial tests one therapy across multiple diseases or tumor types that share a molecular feature.",
        isCorrect: true,
      },
      {
        text: "An umbrella trial tests several therapies within one disease, often assigning patients by biomarker profile.",
        isCorrect: true,
      },
      {
        text: "A platform trial is defined by testing one dose in healthy volunteers before any patient trial begins.",
        isCorrect: false,
      },
    ],
    explanation:
      "These designs respond to the need for more efficient evidence generation, especially when many therapies or small molecular subgroups are involved. Platform trials share infrastructure, basket trials follow a shared biological feature across diseases, and umbrella trials personalize treatment within one disease.",
  },
  {
    id: "clinical-trials-l5-q18",
    chapter: 5,
    difficulty: "medium",
    prompt:
      "Why do platform trials matter during a public-health emergency with several candidate therapies?",
    options: [
      {
        text: "They can evaluate multiple interventions within a shared framework, allowing new arms to enter and ineffective arms to leave more efficiently than separate trials.",
        isCorrect: true,
      },
      {
        text: "They remove the need for control groups because multiple active arms are compared in the same protocol.",
        isCorrect: false,
      },
      {
        text: "They avoid governance, monitoring, and regulatory coordination because the master protocol handles every operational issue automatically.",
        isCorrect: false,
      },
      {
        text: "They are mainly used to validate diagnostic sensitivity and specificity across unrelated laboratory assays.",
        isCorrect: false,
      },
    ],
    explanation:
      "Platform trials can accelerate learning when several interventions need evaluation, as seen during COVID-19. Their efficiency comes with complexity: governance, statistical planning, monitoring, operations, and regulatory coordination must be strong.",
  },
  {
    id: "clinical-trials-l5-q19",
    chapter: 5,
    difficulty: "medium",
    prompt:
      "Which statements correctly connect precision medicine to innovative trial designs?",
    options: [
      {
        text: "Molecular subgroups can be small, making traditional large single-population trials harder to run.",
        isCorrect: true,
      },
      {
        text: "Basket trials can make research feasible when one mutation-targeted therapy is relevant across several tumor types.",
        isCorrect: true,
      },
      {
        text: "Umbrella trials can assign patients within one disease to different therapies based on molecular profiles.",
        isCorrect: true,
      },
      {
        text: "Precision medicine reduces the need for biomarker validation because treatment assignment is individualized.",
        isCorrect: false,
      },
    ],
    explanation:
      "Precision medicine divides diseases into biologically defined groups, which can make conventional designs less practical. Basket and umbrella trials help organize evidence generation around molecular features, but biomarkers still need validation and clinical meaning.",
  },
  {
    id: "clinical-trials-l5-q20",
    chapter: 5,
    difficulty: "medium",
    prompt:
      "Which clinical-research tasks are realistic AI assistance targets?",
    options: [
      {
        text: "Patient recruitment and EHR-based candidate identification.",
        isCorrect: true,
      },
      {
        text: "Eligibility screening support against complex inclusion and exclusion criteria.",
        isCorrect: true,
      },
      {
        text: "Autonomous protocol approval, final safety judgment, and regulatory accountability without qualified human review.",
        isCorrect: false,
      },
      {
        text: "Replacing human evidence by declaring efficacy from literature patterns without observing treated patients.",
        isCorrect: false,
      },
    ],
    explanation:
      "AI can help with data-heavy and document-heavy workflows, especially recruitment, screening, writing, safety triage, anomaly detection, feasibility, and evidence synthesis. It does not replace the need to observe outcomes in humans or the responsibility of qualified clinical teams.",
  },
  {
    id: "clinical-trials-l5-q21",
    chapter: 5,
    difficulty: "medium",
    prompt:
      "Why is AI-assisted patient recruitment considered especially promising?",
    options: [
      {
        text: "Recruitment is a major bottleneck, and eligibility information often exists across EHRs, registries, pathology reports, imaging systems, and clinical notes.",
        isCorrect: true,
      },
      {
        text: "AI and natural language processing can flag potentially eligible patients for human review and prioritization.",
        isCorrect: true,
      },
      {
        text: "AI recruitment tools establish final eligibility without coordinator review because inclusion and exclusion criteria are deterministic.",
        isCorrect: false,
      },
      {
        text: "AI recruitment removes patient consent requirements because screening is performed from existing health records.",
        isCorrect: false,
      },
    ],
    explanation:
      "Recruitment is expensive and slow, and relevant eligibility information is often scattered across structured and unstructured records. AI can help find candidates, but human review, privacy protections, consent processes, and clinical judgment remain necessary.",
  },
  {
    id: "clinical-trials-l5-q22",
    chapter: 5,
    difficulty: "medium",
    prompt:
      "Which AI limitation is most important when analyzing observational healthcare data for treatment effects?",
    options: [
      {
        text: "AI can find patterns, but complex models do not remove confounding or turn observational associations into causal treatment effects.",
        isCorrect: true,
      },
      {
        text: "AI systems cannot process electronic health records, claims, registries, or clinical notes at useful scale.",
        isCorrect: false,
      },
      {
        text: "AI models are unsuitable for prediction tasks because clinical records contain both structured and unstructured data.",
        isCorrect: false,
      },
      {
        text: "AI is limited mainly by the absence of medical-writing tasks in regulated clinical development.",
        isCorrect: false,
      },
    ],
    explanation:
      "Machine learning can model patterns, but causal inference still requires design, assumptions, and careful interpretation. A model trained on observational data can reproduce selection effects, confounding by indication, and healthcare-system biases.",
  },
  {
    id: "clinical-trials-l5-q23",
    chapter: 5,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe accountability when AI is used in clinical research?",
    options: [
      {
        text: "A sponsor or qualified team remains responsible for protocol content, eligibility decisions, safety summaries, and regulated submissions.",
        isCorrect: true,
      },
      {
        text: "Clinical documents produced with AI must still be accurate, traceable, compliant, and expert-reviewed.",
        isCorrect: true,
      },
      {
        text: "AI systems used in regulated workflows need validation, privacy protection, auditability, and quality oversight.",
        isCorrect: true,
      },
      {
        text: "A model output transfers responsibility for scientific validity and patient safety away from sponsors, clinicians, or CROs.",
        isCorrect: false,
      },
    ],
    explanation:
      "AI can assist, but accountability cannot be delegated to a model. Regulated clinical research needs traceable decisions, validated systems, qualified review, data protection, and clear responsibility for patient safety and evidence quality.",
  },
  {
    id: "clinical-trials-l5-q24",
    chapter: 5,
    difficulty: "medium",
    prompt:
      "Which statements describe AI's likely role in Contract Research Organizations (CROs) and pharma development teams?",
    options: [
      {
        text: "AI may improve patient matching, feasibility planning, monitoring focus, data review, safety case processing, medical writing, and evidence synthesis.",
        isCorrect: true,
      },
      {
        text: "AI adoption in regulated clinical research will need validation, audit trails, privacy controls, quality systems, and client trust.",
        isCorrect: true,
      },
      {
        text: "AI is best viewed as a copilot for clinical development teams rather than a replacement for clinical researchers.",
        isCorrect: true,
      },
      {
        text: "AI will likely change CRO work by augmenting recruitment, monitoring, data, regulatory, writing, statistics, safety, and vendor workflows.",
        isCorrect: true,
      },
    ],
    explanation:
      "AI is likely to reshape CRO and pharma workflows by making repetitive and data-heavy tasks more efficient. It can support many functions, but regulated teams still need validation, oversight, quality systems, clinical judgment, and human accountability.",
  },
  {
    id: "clinical-trials-l5-q25",
    chapter: 5,
    difficulty: "medium",
    prompt:
      "In a rare inflammatory disease program, scientists identify an inflammatory pathway, animal models improve when the pathway is blocked, and Drug X engages the target. Which risks remain before human benefit can be claimed?",
    options: [
      {
        text: "The pathway may be associated with disease but not causal in humans.",
        isCorrect: true,
      },
      {
        text: "Animal-model benefit may not translate into clinical benefit for patients.",
        isCorrect: true,
      },
      {
        text: "Biological plausibility is sufficient to conclude the drug will improve patient outcomes.",
        isCorrect: false,
      },
      {
        text: "Target engagement replaces the need to evaluate safety, dose, and patient-relevant endpoints.",
        isCorrect: false,
      },
    ],
    explanation:
      "Discovery and preclinical evidence establish plausibility, not clinical efficacy. Human testing is needed because mechanisms, animal models, target engagement, and biomarkers can fail to translate into patient benefit or acceptable safety.",
  },
  {
    id: "clinical-trials-l5-q26",
    chapter: 5,
    difficulty: "medium",
    prompt:
      "For an immune-modulating drug entering Phase I, which main question is most appropriate?",
    options: [
      {
        text: "Can humans receive the drug safely enough to continue development, and what dose range should be studied?",
        isCorrect: true,
      },
      {
        text: "Does the drug prove sustained remission benefit across all intended patient subgroups?",
        isCorrect: false,
      },
      {
        text: "Does routine-care use after launch match trial efficacy in older patients with comorbidities?",
        isCorrect: false,
      },
      {
        text: "Can claims data alone determine the final regulatory indication before any patient dosing?",
        isCorrect: false,
      },
    ],
    explanation:
      "Phase I focuses on safety, tolerability, pharmacokinetics, and dose range, not definitive efficacy. For riskier interventions, developers must also decide whether healthy volunteers or patient volunteers are more appropriate.",
  },
  {
    id: "clinical-trials-l5-q27",
    chapter: 5,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe Phase II planning for Drug X in a rare inflammatory disease?",
    options: [
      {
        text: "The trial should define a population with confirmed diagnosis, active disease, and clinically relevant inclusion criteria.",
        isCorrect: true,
      },
      {
        text: "The comparator might be placebo plus standard care or an active control, depending on ethics and standard treatment.",
        isCorrect: true,
      },
      {
        text: "Outcomes might include disease activity, steroid sparing, flare reduction, biomarkers, patient-reported fatigue, and safety.",
        isCorrect: true,
      },
      {
        text: "Randomization becomes unnecessary in rare diseases once registries estimate disease prevalence.",
        isCorrect: false,
      },
    ],
    explanation:
      "Phase II tests whether the drug appears effective in patients and helps choose dose and endpoints for later development. Rare disease constraints can justify modern supports such as registries, decentralized elements, and AI screening, but they do not erase the need for bias control.",
  },
  {
    id: "clinical-trials-l5-q28",
    chapter: 5,
    difficulty: "hard",
    prompt:
      "Which modern approaches could help a rare-disease Phase II or Phase III program without changing the core trial logic?",
    options: [
      {
        text: "Registries and real-world data can estimate prevalence, support natural-history understanding, and help identify experienced sites.",
        isCorrect: true,
      },
      {
        text: "AI-assisted EHR screening can flag potential participants for human review.",
        isCorrect: true,
      },
      {
        text: "Hybrid components can reduce travel burden and improve retention for selected assessments.",
        isCorrect: true,
      },
      {
        text: "Adaptive design can help select doses or adjust sample size under pre-specified rules.",
        isCorrect: true,
      },
    ],
    explanation:
      "Modern tools can make rare-disease development more feasible, but they support rather than replace the core logic of clinical evidence. Developers still need a defined population, comparator, meaningful outcomes, safety monitoring, bias control, and interpretable analysis.",
  },
  {
    id: "clinical-trials-l5-q29",
    chapter: 5,
    difficulty: "hard",
    prompt:
      "A Phase III trial in a rare inflammatory disease reports sustained remission benefit, a clinically meaningful absolute effect, reasonably precise confidence interval, improved quality of life, lower steroid use, and more mild infections without a major unexpected signal. Which interpretations are appropriate?",
    options: [
      {
        text: "The result supports benefit, but the final judgment still requires benefit-risk interpretation and regulatory review.",
        isCorrect: true,
      },
      {
        text: "Safety, endpoint validity, intended population, missing data, subgroup consistency, and follow-up duration remain relevant.",
        isCorrect: true,
      },
      {
        text: "The improved efficacy endpoint makes safety review secondary and no longer central to approval decisions.",
        isCorrect: false,
      },
      {
        text: "Quality-of-life improvement replaces the need to assess whether the primary endpoint was clinically meaningful.",
        isCorrect: false,
      },
    ],
    explanation:
      "A strong Phase III result is not interpreted from one number alone. Regulators and developers assess efficacy, safety, endpoint meaning, precision, missing data, population fit, follow-up, and whether the overall benefit-risk profile supports use.",
  },
  {
    id: "clinical-trials-l5-q30",
    chapter: 5,
    difficulty: "hard",
    prompt:
      "After approval, Drug X enters broad clinical use. Which question belongs most directly to Phase IV and real-world evidence?",
    options: [
      {
        text: "Are rare adverse events, long-term outcomes, adherence patterns, routine-care effectiveness, and performance in broader populations emerging after launch?",
        isCorrect: true,
      },
      {
        text: "Can animal toxicology alone determine whether first-in-human dosing should begin?",
        isCorrect: false,
      },
      {
        text: "Should the Phase I dose-escalation design include healthy volunteers or patient volunteers?",
        isCorrect: false,
      },
      {
        text: "Which preclinical pathway should be selected before any candidate drug has been synthesized?",
        isCorrect: false,
      },
    ],
    explanation:
      "Approval is a transition, not the end of evidence generation. Phase IV and real-world evidence examine broader use, rare safety signals, long-term effectiveness, adherence, label use, comorbid patients, and outcomes that pre-approval trials may not fully capture.",
  },
  {
    id: "clinical-trials-l5-q31",
    chapter: 5,
    difficulty: "hard",
    prompt:
      "As Head of Clinical Development for Drug X, which risks should be considered before committing to Phase III?",
    options: [
      {
        text: "Scientific risk that the pathway is not causal or does not translate from animal models to humans.",
        isCorrect: true,
      },
      {
        text: "Endpoint and statistical risk that the chosen outcome is not meaningful, accepted, powered, or precisely estimated.",
        isCorrect: true,
      },
      {
        text: "Operational and data-quality risks can be set aside because rare-disease trials usually run at specialized sites.",
        isCorrect: false,
      },
      {
        text: "The absence of commercial and access risk once Phase II suggests a biologically plausible treatment effect.",
        isCorrect: false,
      },
    ],
    explanation:
      "Clinical development decisions combine science, statistics, operations, regulation, patient needs, and market access. Even a promising Phase II signal can fail if the biology is wrong, endpoints are weak, power is inadequate, recruitment fails, data quality is poor, regulators disagree, or payers and clinicians find the benefit insufficient.",
  },
  {
    id: "clinical-trials-l5-q32",
    chapter: 5,
    difficulty: "hard",
    prompt:
      "Which materials and judgments are part of regulatory review for a new therapy?",
    options: [
      {
        text: "Preclinical evidence, manufacturing information, Phase I safety, Phase II dose and efficacy data, and Phase III confirmatory evidence.",
        isCorrect: true,
      },
      {
        text: "Clinical study reports, statistical analyses, safety summaries, proposed labeling, and benefit-risk assessment.",
        isCorrect: true,
      },
      {
        text: "Questions about endpoint meaning, population fit, missing data, safety monitoring, manufacturing reliability, and post-marketing commitments.",
        isCorrect: true,
      },
      {
        text: "A scientific and clinical judgment about whether the evidence supports making the product available to patients.",
        isCorrect: true,
      },
    ],
    explanation:
      "Regulatory review is more than checking that trials were completed. Regulators evaluate quality, safety, efficacy, benefit-risk, manufacturing, endpoint validity, data integrity, labeling, and whether additional evidence is needed after approval.",
  },
  {
    id: "clinical-trials-l5-q33",
    chapter: 5,
    difficulty: "hard",
    prompt:
      "Which statement best captures why medical uncertainty persists despite digital tools, real-world data, and AI?",
    options: [
      {
        text: "Biology, patient heterogeneity, bias, confounding, endpoint meaning, safety, and real-world implementation still require evidence in humans.",
        isCorrect: true,
      },
      {
        text: "Digital tools mainly solve the problem by converting all clinical outcomes into continuous measurements.",
        isCorrect: false,
      },
      {
        text: "AI models remove the uncertainty created by disease natural history, placebo effects, and confounding.",
        isCorrect: false,
      },
      {
        text: "Software updates, remote data collection, and registries make benefit-risk assessment unnecessary after launch.",
        isCorrect: false,
      },
    ],
    explanation:
      "Digital tools can expand what is measured and how efficiently trials run, but they do not eliminate uncertainty about causality, safety, clinical meaning, or patient benefit. The central question remains whether an intervention helps or harms humans under credible evidence standards.",
  },
  {
    id: "clinical-trials-l5-q34",
    chapter: 5,
    difficulty: "hard",
    prompt:
      "A diagnostic AI model performs well in a curated image dataset but worse in rural clinics using different cameras and lighting. Which evidence gap does this most directly reveal?",
    options: [
      {
        text: "Generalizability and clinical-setting performance, not merely model accuracy in the original validation dataset.",
        isCorrect: true,
      },
      {
        text: "The need to test performance across equipment, lighting, workflow, patient mix, and care settings.",
        isCorrect: true,
      },
      {
        text: "The possibility that an apparently strong model can perform differently once deployed outside the curated dataset.",
        isCorrect: true,
      },
      {
        text: "The importance of evaluating whether AI use improves decisions or outcomes in the settings where it will be used.",
        isCorrect: true,
      },
    ],
    explanation:
      "AI performance must generalize across real patients, settings, devices, workflows, and subgroups. A curated dataset can overstate performance if deployment conditions differ from the evaluated data, and clinical utility still depends on whether use improves decisions or outcomes.",
  },
  {
    id: "clinical-trials-l5-q35",
    chapter: 5,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe equity considerations in decentralized or digital trials?",
    options: [
      {
        text: "Remote participation can improve access for patients who live far from research centers or cannot travel frequently.",
        isCorrect: true,
      },
      {
        text: "Digital requirements can exclude patients without smartphones, stable internet, digital literacy, or comfort with remote technology.",
        isCorrect: true,
      },
      {
        text: "Equity planning should consider device support, training, language, accessibility, reimbursement, and alternative participation paths.",
        isCorrect: true,
      },
      {
        text: "Moving visits online makes a trial representative because geographic distance is the main barrier to participation.",
        isCorrect: false,
      },
    ],
    explanation:
      "Decentralization can widen access for some patients and narrow it for others if digital access and support are not planned. Equity depends on design details, including technology access, patient support, language, disability, reimbursement, and alternative ways to participate.",
  },
  {
    id: "clinical-trials-l5-q36",
    chapter: 5,
    difficulty: "hard",
    prompt:
      "Which course-level questions organize the full clinical-trials evidence framework?",
    options: [
      {
        text: "Is the intervention biologically plausible?",
        isCorrect: true,
      },
      {
        text: "Does the intervention actually help patients?",
        isCorrect: true,
      },
      {
        text: "Can the evidence be trusted?",
        isCorrect: true,
      },
      {
        text: "Can the evidence be generated efficiently and ethically?",
        isCorrect: true,
      },
    ],
    explanation:
      "The full course can be organized around plausibility, patient benefit, evidence trustworthiness, and efficient ethical evidence generation. These questions connect discovery, trial design, statistics, operations, regulation, real-world evidence, digital tools, and AI.",
  },
  {
    id: "clinical-trials-l5-q37",
    chapter: 5,
    difficulty: "hard",
    prompt:
      "A health-tech founder argues that a diagnostic app needs evidence only for model accuracy. Which response is strongest?",
    options: [
      {
        text: "Accuracy is necessary, but evidence should also address generalization, clinical utility, workflow impact, false positives and negatives, safety, privacy, updates, and accountability.",
        isCorrect: true,
      },
      {
        text: "Diagnostic apps follow the same staged Phase I to Phase III drug framework because software is regulated like a fixed-dose molecule.",
        isCorrect: false,
      },
      {
        text: "Clinical utility is part of evidence generation because a diagnostic app can change decisions even when it does not administer treatment.",
        isCorrect: true,
      },
      {
        text: "Privacy and update governance become operational details after launch rather than part of the evidence discussion.",
        isCorrect: false,
      },
    ],
    explanation:
      "Diagnostic software and AI tools require more than a headline accuracy number. Good evidence asks whether performance generalizes, whether use changes decisions or outcomes, what harms arise from errors, how updates are governed, and who is accountable.",
  },
  {
    id: "clinical-trials-l5-q38",
    chapter: 5,
    difficulty: "hard",
    prompt:
      "Which use of real-world evidence is most defensible as a complement to a randomized trial rather than a replacement for causal efficacy evidence?",
    options: [
      {
        text: "Using registries after approval to monitor long-term outcomes and rare adverse events in patient groups excluded from pivotal trials.",
        isCorrect: true,
      },
      {
        text: "Using untreated claims data to conclude a drug works because treated patients have fewer hospitalizations without addressing confounding by indication.",
        isCorrect: false,
      },
      {
        text: "Using an EHR extract to replace endpoint adjudication because routine-care codes are available at large scale.",
        isCorrect: false,
      },
      {
        text: "Using wearable data volume as proof that a digital endpoint is clinically meaningful without validation.",
        isCorrect: false,
      },
    ],
    explanation:
      "RWE is especially valuable for questions that trials may not fully answer, such as rare harms, long-term outcomes, broader populations, and routine-care use. It becomes weaker when used to make causal efficacy claims without adequate design, validation, or confounding control.",
  },
  {
    id: "clinical-trials-l5-q39",
    chapter: 5,
    difficulty: "hard",
    prompt:
      "Which statements correctly distinguish AI support from AI replacement in clinical development?",
    options: [
      {
        text: "AI can draft, search, summarize, triage, flag anomalies, and predict feasibility, while humans remain accountable for decisions.",
        isCorrect: true,
      },
      {
        text: "AI can support risk-based monitoring and data review by identifying unusual site behavior, missingness, duplicate records, or outliers.",
        isCorrect: true,
      },
      {
        text: "AI can help synthesize prior evidence, but systematic review methods still require transparent criteria and qualified judgment.",
        isCorrect: true,
      },
      {
        text: "AI replacement is appropriate when a model can generate a regulatory narrative that sounds internally consistent.",
        isCorrect: false,
      },
    ],
    explanation:
      "AI is strongest as a tool that assists clinical teams with search, drafting, triage, monitoring, and pattern detection. Regulated decisions still need human accountability, validation, auditability, clinical judgment, and transparent methods.",
  },
  {
    id: "clinical-trials-l5-q40",
    chapter: 5,
    difficulty: "hard",
    prompt:
      "Which statements synthesize the role of clinical trials across the whole course?",
    options: [
      {
        text: "Clinical trials are structured systems for generating trustworthy evidence about interventions in humans.",
        isCorrect: true,
      },
      {
        text: "They turn medical uncertainty into evidence by combining design, bias control, statistics, operations, ethics, regulation, and patient participation.",
        isCorrect: true,
      },
      {
        text: "They depend on real people accepting risk and burden so medicine can learn reliably.",
        isCorrect: true,
      },
      {
        text: "Their future will be more digital and AI-assisted, but their mission remains determining as reliably and ethically as possible whether interventions help or harm humans.",
        isCorrect: true,
      },
    ],
    explanation:
      "The unifying idea is that clinical trials convert medical uncertainty into trustworthy evidence. Digital tools, RWE, adaptive designs, and AI may change how research is conducted, but the mission remains ethical and reliable learning about human benefit and harm.",
  },
];
