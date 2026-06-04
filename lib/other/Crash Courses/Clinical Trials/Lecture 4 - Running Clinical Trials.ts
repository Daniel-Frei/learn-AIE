import { Question } from "../../../quiz";

export const ClinicalTrialsLecture4Questions: Question[] = [
  {
    id: "clinical-trials-l4-q01",
    chapter: 4,
    difficulty: "easy",
    prompt:
      "Which functions does a clinical trial protocol serve during trial execution?",
    options: [
      {
        text: "It defines the trial question, population, treatment plan, outcomes, safety monitoring, data collection, and analysis approach.",
        isCorrect: true,
      },
      {
        text: "It standardizes how different sites recruit, treat, assess, and document participants.",
        isCorrect: true,
      },
      {
        text: "It gives site staff an operational plan for visits, assessments, and follow-up.",
        isCorrect: true,
      },
      {
        text: "It helps make the resulting data credible, comparable, and interpretable.",
        isCorrect: true,
      },
    ],
    explanation:
      "The protocol is the trial's master plan because it turns a treatment idea into standardized operations across sites. Without that shared plan, sites could enroll different patients, measure outcomes differently, and produce data that are difficult to interpret.",
  },
  {
    id: "clinical-trials-l4-q02",
    chapter: 4,
    difficulty: "easy",
    prompt:
      "A protocol says the primary objective is to test whether Drug X reduces cardiovascular mortality in heart-failure patients. Which statements correctly connect objectives and endpoints?",
    options: [
      {
        text: "The primary objective identifies the main question the trial is designed to answer.",
        isCorrect: true,
      },
      {
        text: "Endpoints translate objectives into specific measurable outcomes.",
        isCorrect: true,
      },
      {
        text: "Secondary objectives replace the primary objective once recruitment begins slowly.",
        isCorrect: false,
      },
      {
        text: "Objectives are operational notes for site startup rather than drivers of sample size and interpretation.",
        isCorrect: false,
      },
    ],
    explanation:
      "Objectives define what the trial is trying to learn, and endpoints define how those objectives will be measured. The primary objective is central to sample size, statistical testing, and interpretation, while secondary objectives add supporting questions without replacing the main one.",
  },
  {
    id: "clinical-trials-l4-q03",
    chapter: 4,
    difficulty: "easy",
    prompt:
      "Which endpoint definition is strongest for a pain trial that aims to measure symptom improvement?",
    options: [
      {
        text: "Change from baseline in a validated pain score at Week 12.",
        isCorrect: true,
      },
      {
        text: "Patient feels better during treatment, as judged at any visit.",
        isCorrect: false,
      },
      {
        text: "Investigator believes pain improved by the end of follow-up.",
        isCorrect: false,
      },
      {
        text: "Any improvement in comfort, mobility, mood, or medication use reported after dosing.",
        isCorrect: false,
      },
    ],
    explanation:
      "A strong endpoint specifies the measurement tool, baseline comparison, and timing. Vague phrases such as feeling better or investigator impression leave too much room for inconsistent measurement across sites.",
  },
  {
    id: "clinical-trials-l4-q04",
    chapter: 4,
    difficulty: "medium",
    prompt:
      "A protocol for moderate Alzheimer's disease excludes many patients for kidney disease, liver disease, prior drug exposure, interacting medications, and lack of a caregiver. Which operational consequences are likely?",
    options: [
      {
        text: "Screen failure rates may rise because many evaluated patients will not qualify.",
        isCorrect: true,
      },
      {
        text: "Recruitment may slow because the eligible population is smaller than the diagnosed population.",
        isCorrect: true,
      },
      {
        text: "The criteria may improve interpretability or safety while reducing feasibility.",
        isCorrect: true,
      },
      {
        text: "Strict criteria remove the need to evaluate recruitment assumptions during feasibility planning.",
        isCorrect: false,
      },
    ],
    explanation:
      "Eligibility criteria shape safety, interpretability, recruitment, and generalizability. A scientifically clean protocol with many exclusions may be hard to recruit because many patients who look relevant clinically fail screening.",
  },
  {
    id: "clinical-trials-l4-q05",
    chapter: 4,
    difficulty: "medium",
    prompt:
      "Which protocol content helps a study coordinator know exactly what to do at each participant visit?",
    options: [
      {
        text: "The schedule of assessments showing visits, labs, imaging, questionnaires, safety checks, and follow-up timing.",
        isCorrect: true,
      },
      {
        text: "The study procedures section describing how assessments and treatment administration are performed.",
        isCorrect: true,
      },
      {
        text: "The published press release announcing that the investigational product entered Phase III.",
        isCorrect: false,
      },
      {
        text: "The pooled estimate from a prior meta-analysis, used as the visit-by-visit operations checklist.",
        isCorrect: false,
      },
    ],
    explanation:
      "Site staff need operationally precise instructions, not just the scientific rationale. The schedule of assessments and study procedures tell coordinators what must happen at screening, baseline, treatment visits, follow-up, and end-of-study visits.",
  },
  {
    id: "clinical-trials-l4-q06",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "Researchers inspect early unblinded outcome data and then choose the most favorable endpoint analysis as the primary analysis. Which protocol-design principle was violated?",
    options: [
      {
        text: "Analyses should be pre-specified before results are known to protect credibility and reduce outcome-driven interpretation.",
        isCorrect: true,
      },
      {
        text: "Eligibility criteria should be broad enough to maximize recruitment even when the study question becomes less precise.",
        isCorrect: false,
      },
      {
        text: "Study procedures should be placed in the schedule of assessments instead of in the statistical analysis plan.",
        isCorrect: false,
      },
      {
        text: "Protocol amendments should be avoided when site training, ethics review, and database changes create extra work.",
        isCorrect: false,
      },
    ],
    explanation:
      "Pre-specification matters because choosing analyses after seeing data can bias the interpretation toward the most favorable result. The statistical analysis plan and protocol should define the main analysis, populations, missing-data handling, and multiplicity approach before outcomes are known.",
  },
  {
    id: "clinical-trials-l4-q07",
    chapter: 4,
    difficulty: "easy",
    prompt:
      "A sponsor proposes a protocol amendment after sites report that recruitment criteria are too strict and a lab schedule is impractical. Which statements correctly describe amendments?",
    options: [
      {
        text: "They are formal protocol changes that can address feasibility, safety, regulatory, dose, or endpoint-definition problems.",
        isCorrect: true,
      },
      {
        text: "They may require ethics approval, regulatory notification, consent updates, retraining, and system changes.",
        isCorrect: true,
      },
      {
        text: "They can slow a trial even when the change is scientifically or operationally justified.",
        isCorrect: true,
      },
      {
        text: "They convert prior protocol deviations into compliant trial conduct without further review.",
        isCorrect: false,
      },
    ],
    explanation:
      "Amendments are common because real-world execution exposes problems in the original plan. They can be necessary, but they are disruptive because sites, ethics committees, regulators, vendors, and data systems may need updates.",
  },
  {
    id: "clinical-trials-l4-q08",
    chapter: 4,
    difficulty: "easy",
    prompt:
      "Which conditions must usually line up before a patient can be recruited and enrolled in a clinical trial?",
    options: [
      {
        text: "The patient has the right diagnosis and disease severity.",
        isCorrect: true,
      },
      {
        text: "The patient meets inclusion criteria and avoids exclusion criteria.",
        isCorrect: true,
      },
      {
        text: "The patient is willing and able to participate after informed consent.",
        isCorrect: true,
      },
      {
        text: "The physician and site have the infrastructure to identify, screen, and follow the patient.",
        isCorrect: true,
      },
    ],
    explanation:
      "Recruitment is difficult because eligibility, willingness, physician awareness, site infrastructure, and patient practicality all have to align. A diagnosed patient is not automatically an enrolled participant.",
  },
  {
    id: "clinical-trials-l4-q09",
    chapter: 4,
    difficulty: "medium",
    prompt:
      "A trial needs 500 enrolled patients, and feasibility work suggests that 1 in 5 screened patients will qualify and proceed. Which statements are correct?",
    options: [
      {
        text: "The sites should expect to screen about 2,500 patients to enroll 500.",
        isCorrect: true,
      },
      {
        text: "A high screen-failure rate increases site workload, cost, and timeline pressure.",
        isCorrect: true,
      },
      {
        text: "Screened and enrolled mean the same thing once a patient signs any study document.",
        isCorrect: false,
      },
      {
        text: "Screen failure affects recruitment reporting but not operational planning or budget.",
        isCorrect: false,
      },
    ],
    explanation:
      "If one in five screened patients enrolls, the screening target is five times the enrollment target. Screen failures matter operationally because screening consumes staff time, patient effort, tests, scheduling, and money.",
  },
  {
    id: "clinical-trials-l4-q10",
    chapter: 4,
    difficulty: "easy",
    prompt:
      "Which site-selection problem describes a site that completes startup, contracts, training, activation, and initiation but enrolls no patients?",
    options: [
      {
        text: "A non-enrolling site.",
        isCorrect: true,
      },
      {
        text: "A database-lock site.",
        isCorrect: false,
      },
      {
        text: "A pharmacovigilance site.",
        isCorrect: false,
      },
      {
        text: "A statistical-programming site.",
        isCorrect: false,
      },
    ],
    explanation:
      "A non-enrolling site creates cost and delay because the sponsor or Contract Research Organization (CRO) invested in startup but receives no enrolled participants. Site selection tries to reduce this risk by evaluating patient access, past performance, activation speed, competing trials, and investigator engagement.",
  },
  {
    id: "clinical-trials-l4-q11",
    chapter: 4,
    difficulty: "medium",
    prompt:
      "Which behaviors would indicate strong investigator and coordinator engagement at a trial site?",
    options: [
      {
        text: "The investigator actively identifies patients, reminds colleagues, and resolves clinical questions.",
        isCorrect: true,
      },
      {
        text: "The coordinator maintains screening logs, schedules visits, documents consent, enters data, and answers queries.",
        isCorrect: true,
      },
      {
        text: "The site communicates barriers early when eligibility, visit burden, or staffing affects recruitment.",
        isCorrect: true,
      },
      {
        text: "The investigator delegates trial conduct and therefore has no ongoing oversight responsibility.",
        isCorrect: false,
      },
    ],
    explanation:
      "Investigators and coordinators are central to recruitment and data quality. Delegation is normal, but the investigator remains responsible for oversight, while coordinators often handle the practical tasks that keep the trial running.",
  },
  {
    id: "clinical-trials-l4-q12",
    chapter: 4,
    difficulty: "easy",
    prompt:
      "Which factors can cause enrolled participants to drop out or stop providing useful trial data?",
    options: [
      {
        text: "Side effects or lack of perceived benefit.",
        isCorrect: true,
      },
      {
        text: "Travel burden, difficult scheduling, or relocation.",
        isCorrect: true,
      },
      {
        text: "Disease progression, loss of motivation, or dissatisfaction with procedures.",
        isCorrect: true,
      },
      {
        text: "Poor communication from the research team.",
        isCorrect: true,
      },
    ],
    explanation:
      "Retention matters because enrolling a patient is not enough; the trial needs follow-up data. Dropout can undermine validity, especially when dropout differs by group or removes important endpoint information.",
  },
  {
    id: "clinical-trials-l4-q13",
    chapter: 4,
    difficulty: "medium",
    prompt:
      "A trial requiring weekly visits, repeated biopsies, long questionnaires, and complex travel is recruiting slowly. Which statements correctly diagnose the problem?",
    options: [
      {
        text: "Patient burden can reduce willingness to participate and remain in the trial.",
        isCorrect: true,
      },
      {
        text: "Making participation practical and acceptable is part of recruitment strategy, not a separate issue from recruitment.",
        isCorrect: true,
      },
      {
        text: "Visit burden mainly affects sponsor paperwork and has little effect on patient decisions.",
        isCorrect: false,
      },
      {
        text: "High-burden procedures improve recruitment because they signal a scientifically serious protocol to patients.",
        isCorrect: false,
      },
    ],
    explanation:
      "Recruitment depends on whether participation fits into a patient's life and feels worthwhile. Visit frequency, travel, invasive procedures, reimbursement, communication, and trust can determine whether eligible patients enroll and stay.",
  },
  {
    id: "clinical-trials-l4-q14",
    chapter: 4,
    difficulty: "easy",
    prompt:
      "A nurse measures a participant's blood pressure at a study visit, and the value is first recorded in a chart or source worksheet. What is that original record called?",
    options: [
      {
        text: "Source data.",
        isCorrect: true,
      },
      {
        text: "Database lock.",
        isCorrect: false,
      },
      {
        text: "A pooled estimate.",
        isCorrect: false,
      },
      {
        text: "A site activation metric.",
        isCorrect: false,
      },
    ],
    explanation:
      "Source data are the original records of clinical observations and study activities. They matter because monitors and regulators need to verify that database entries reflect what actually happened at the site.",
  },
  {
    id: "clinical-trials-l4-q15",
    chapter: 4,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe Case Report Forms (CRFs) and Electronic Data Capture (EDC) systems?",
    options: [
      {
        text: "CRFs structure the protocol-required data that sites capture for each participant.",
        isCorrect: true,
      },
      {
        text: "Modern Electronic Data Capture (EDC) systems support permissions, validation rules, audit trails, queries, and data export.",
        isCorrect: true,
      },
      {
        text: "EDC systems are informal spreadsheets maintained separately by each site for local convenience.",
        isCorrect: false,
      },
      {
        text: "CRF entries should remain traceable to source documents so monitors can compare database values with original records.",
        isCorrect: true,
      },
    ],
    explanation:
      "CRFs and eCRFs organize the study data required by the protocol, while EDC systems provide controlled electronic collection and review. They do not eliminate source documentation; entered data still need to be traceable and verifiable against original records.",
  },
  {
    id: "clinical-trials-l4-q16",
    chapter: 4,
    difficulty: "easy",
    prompt:
      "Which steps can occur between last patient last visit and final statistical analysis?",
    options: [
      {
        text: "Data cleaning and resolution of open queries.",
        isCorrect: true,
      },
      {
        text: "Medical review, coding, and reconciliation across data sources.",
        isCorrect: true,
      },
      {
        text: "Database preparation and database lock.",
        isCorrect: true,
      },
      {
        text: "Final analysis according to the statistical analysis plan after the database is locked.",
        isCorrect: true,
      },
    ],
    explanation:
      "A study is not ready for analysis simply because the final visit occurred. Data must be cleaned, coded, reconciled, reviewed, and locked before the final analysis proceeds according to the statistical analysis plan.",
  },
  {
    id: "clinical-trials-l4-q17",
    chapter: 4,
    difficulty: "medium",
    prompt:
      "Which statements correctly compare traditional source data verification with risk-based monitoring?",
    options: [
      {
        text: "Source Data Verification (SDV) checks database entries against original source records.",
        isCorrect: true,
      },
      {
        text: "Risk-based monitoring focuses resources on critical data, critical processes, and higher-risk sites.",
        isCorrect: true,
      },
      {
        text: "Risk-based monitoring means patient safety data are reviewed after final analysis rather than during conduct.",
        isCorrect: false,
      },
      {
        text: "SDV is a statistical method for estimating treatment effects from censored time-to-event data.",
        isCorrect: false,
      },
    ],
    explanation:
      "SDV is a data-quality activity that compares entered values to source records. Risk-based monitoring changes the allocation of monitoring effort so teams focus on the data and processes most important to participant safety and trial credibility.",
  },
  {
    id: "clinical-trials-l4-q18",
    chapter: 4,
    difficulty: "medium",
    prompt:
      'One site records an adverse event as "heart attack," another writes "myocardial infarction," and another writes "MI." Which data-management process addresses this issue?',
    options: [
      {
        text: "Medical coding maps variant site terms to standardized dictionary terms for consistent analysis.",
        isCorrect: true,
      },
      {
        text: "Database lock changes the endpoint definition to match the most frequent site wording.",
        isCorrect: false,
      },
      {
        text: "Site activation confirms that each local term should be analyzed as a separate outcome category.",
        isCorrect: false,
      },
      {
        text: "Randomization balances the wording differences so no coding step is needed for analysis datasets.",
        isCorrect: false,
      },
    ],
    explanation:
      "Coding uses standardized dictionaries, such as MedDRA for adverse events and WHO Drug for medications, to make site-entered terms analyzable. Without coding, equivalent clinical concepts could be split across inconsistent local wording.",
  },
  {
    id: "clinical-trials-l4-q19",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "Which tasks should be substantially complete before database lock in a regulated clinical trial?",
    options: [
      {
        text: "Required data entry and query resolution.",
        isCorrect: true,
      },
      {
        text: "Coding, medical review, reconciliation, and quality checks.",
        isCorrect: true,
      },
      {
        text: "Controlled confirmation that the clinical database is final for analysis.",
        isCorrect: true,
      },
      {
        text: "Exploratory switching of the primary analysis to whichever endpoint looks strongest.",
        isCorrect: false,
      },
    ],
    explanation:
      "Database lock is the point where the clinical database is treated as final for analysis. It should follow data entry, query resolution, coding, reconciliation, medical review, and quality checks, not outcome-driven changes to the planned analysis.",
  },
  {
    id: "clinical-trials-l4-q20",
    chapter: 4,
    difficulty: "easy",
    prompt: "Which core goals are central to Good Clinical Practice (GCP)?",
    options: [
      {
        text: "Protecting the rights, safety, and wellbeing of trial participants.",
        isCorrect: true,
      },
      {
        text: "Protecting the credibility and integrity of trial data.",
        isCorrect: true,
      },
      {
        text: "Ensuring that human research risks are reasonable in relation to potential benefits and knowledge gained.",
        isCorrect: true,
      },
      {
        text: "Ensuring that documented trial conduct can be inspected and reconstructed.",
        isCorrect: true,
      },
    ],
    explanation:
      "Good Clinical Practice is the ethical and quality framework for clinical trials. Its participant-protection and data-integrity goals are linked because exposing people to trial burden without producing reliable knowledge is itself an ethical problem.",
  },
  {
    id: "clinical-trials-l4-q21",
    chapter: 4,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe informed consent under Good Clinical Practice?",
    options: [
      {
        text: "Participants should understand the purpose, procedures, risks, benefits, alternatives, confidentiality, compensation, and right to withdraw.",
        isCorrect: true,
      },
      {
        text: "Consent is a process of voluntary understanding, not just a signed form.",
        isCorrect: true,
      },
      {
        text: "Consent is unnecessary when a trial has ethics committee approval and a favorable benefit-risk rationale.",
        isCorrect: false,
      },
      {
        text: "Vulnerable populations make consent documentation less important because family or clinicians can infer preferences.",
        isCorrect: false,
      },
    ],
    explanation:
      "Informed consent is meaningful only when participants receive adequate information and voluntarily agree. A signature alone is not enough if the participant does not understand the trial, and vulnerable populations require special care rather than weaker consent practices.",
  },
  {
    id: "clinical-trials-l4-q22",
    chapter: 4,
    difficulty: "easy",
    prompt:
      "A participant is enrolled despite failing a key eligibility criterion. What is this best classified as?",
    options: [
      {
        text: "A protocol deviation that may affect safety, data integrity, and regulatory acceptability.",
        isCorrect: true,
      },
      {
        text: "A data query that is resolved by deleting the eligibility criterion from the database.",
        isCorrect: false,
      },
      {
        text: "A site startup delay that ends once the participant receives treatment.",
        isCorrect: false,
      },
      {
        text: "A meta-analysis heterogeneity issue caused by pooling incompatible studies.",
        isCorrect: false,
      },
    ],
    explanation:
      "A protocol deviation is a departure from the approved protocol, and enrolling an ineligible participant can be major. Major deviations can threaten participant safety, trial interpretation, and whether regulators accept the data.",
  },
  {
    id: "clinical-trials-l4-q23",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe ALCOA as a data-integrity framework?",
    options: [
      {
        text: "Attributable means the record identifies who created or changed the data.",
        isCorrect: true,
      },
      {
        text: "Contemporaneous means data are recorded at the time of the observation or activity.",
        isCorrect: true,
      },
      {
        text: "Original and accurate mean records should come from original sources and correctly reflect what happened.",
        isCorrect: true,
      },
      {
        text: "ALCOA replaces audit trails because readable records do not need change history.",
        isCorrect: false,
      },
    ],
    explanation:
      "Attributable, Legible, Contemporaneous, Original, Accurate (ALCOA) is a practical way to think about trustworthy clinical data. It complements audit trails and documentation; it does not remove the need to trace who changed data, when, and why.",
  },
  {
    id: "clinical-trials-l4-q24",
    chapter: 4,
    difficulty: "easy",
    prompt:
      "Which items belong in the Trial Master File or electronic Trial Master File for a regulated clinical trial?",
    options: [
      {
        text: "Protocol, investigator brochure, informed consent forms, and ethics approvals.",
        isCorrect: true,
      },
      {
        text: "Regulatory correspondence, delegation logs, and training records.",
        isCorrect: true,
      },
      {
        text: "Monitoring visit reports, safety reports, lab certifications, and contracts.",
        isCorrect: true,
      },
      {
        text: "Essential documents that allow reconstruction of how the trial was conducted.",
        isCorrect: true,
      },
    ],
    explanation:
      "The Trial Master File (TMF) is the collection of essential documents showing how the trial was conducted. Regulators can inspect it to evaluate participant protection, protocol compliance, monitoring, safety reporting, and data integrity.",
  },
  {
    id: "clinical-trials-l4-q25",
    chapter: 4,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe investigator and sponsor responsibility?",
    options: [
      {
        text: "The investigator remains responsible for site conduct even when tasks are delegated to trained staff.",
        isCorrect: true,
      },
      {
        text: "The sponsor retains ultimate accountability for trial quality and compliance even when tasks are delegated to a CRO.",
        isCorrect: true,
      },
      {
        text: "Documented delegation transfers site oversight responsibility from the investigator to the study coordinator.",
        isCorrect: false,
      },
      {
        text: "A CRO contract transfers regulatory accountability away from the sponsor for delegated trial functions.",
        isCorrect: false,
      },
    ],
    explanation:
      "Delegation is common in clinical research, but accountability does not disappear. Investigators must oversee site conduct, and sponsors must oversee CROs and vendors because regulators still hold the sponsor responsible for overall trial quality.",
  },
  {
    id: "clinical-trials-l4-q26",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "Which statement best describes the sponsor-CRO relationship in a clinical trial?",
    options: [
      {
        text: "The sponsor owns the study and may delegate operational activities to a Contract Research Organization (CRO), while retaining ultimate responsibility.",
        isCorrect: true,
      },
      {
        text: "The CRO owns the study once full-service outsourcing begins, while the sponsor becomes a scientific advisor.",
        isCorrect: false,
      },
      {
        text: "The sponsor owns regulatory submissions, while the CRO owns patient safety and data integrity after site activation.",
        isCorrect: false,
      },
      {
        text: "The CRO becomes accountable for trial design, and the sponsor becomes accountable only for investigational product supply.",
        isCorrect: false,
      },
    ],
    explanation:
      "A CRO can execute many delegated tasks, but the sponsor remains accountable for the study. This is why sponsor oversight, clear contracts, transfer-of-obligation documents, governance, metrics, audits, and escalation pathways matter.",
  },
  {
    id: "clinical-trials-l4-q27",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "Which statements correctly match CRO operational functions to their responsibilities?",
    options: [
      {
        text: "Project management coordinates timelines, budgets, vendors, milestones, risks, and communication.",
        isCorrect: true,
      },
      {
        text: "Clinical operations manages site selection, startup, activation, monitoring, recruitment tracking, and site performance.",
        isCorrect: true,
      },
      {
        text: "A Clinical Research Associate (CRA) monitors sites for protocol compliance, data quality, safety reporting, and investigational product accountability.",
        isCorrect: true,
      },
      {
        text: "Regulatory affairs is the group that creates analysis datasets, tables, figures, and listings for final study outputs.",
        isCorrect: false,
      },
    ],
    explanation:
      "Project management, clinical operations, and CRAs are central to trial execution. Analysis datasets and tables are usually produced by statistical programming, while regulatory affairs handles submissions, authority communication, and country-specific requirements.",
  },
  {
    id: "clinical-trials-l4-q28",
    chapter: 4,
    difficulty: "easy",
    prompt:
      "Which tasks commonly occur before a site can start enrolling patients?",
    options: [
      {
        text: "Contracts and budgets are negotiated.",
        isCorrect: true,
      },
      {
        text: "Ethics approval and required regulatory documents are obtained.",
        isCorrect: true,
      },
      {
        text: "Site staff are trained and systems such as EDC are available.",
        isCorrect: true,
      },
      {
        text: "Trial materials, lab kits, investigational product, and the site initiation visit are completed as needed.",
        isCorrect: true,
      },
    ],
    explanation:
      "Site startup is often a bottleneck because many dependencies must be ready before enrollment begins. A scientifically attractive site may still activate late if contracts, ethics approval, training, systems, or supply logistics lag.",
  },
  {
    id: "clinical-trials-l4-q29",
    chapter: 4,
    difficulty: "medium",
    prompt:
      "Which statements correctly distinguish common CRO or trial-team functions?",
    options: [
      {
        text: "Data management builds databases, designs CRFs, manages queries, handles coding and reconciliation, and prepares for database lock.",
        isCorrect: true,
      },
      {
        text: "Biostatistics designs analyses, calculates sample size, supports interim analyses, and analyzes final data.",
        isCorrect: true,
      },
      {
        text: "Medical writing dispenses investigational product at sites and documents temperature excursions in pharmacy logs.",
        isCorrect: false,
      },
      {
        text: "Quality assurance writes the primary endpoint analysis code and produces final statistical tables.",
        isCorrect: false,
      },
    ],
    explanation:
      "Data management and biostatistics have distinct but connected responsibilities. Medical writing produces protocols, study reports, and regulatory documents, while quality assurance focuses on audits, compliance systems, corrective actions, and inspection readiness.",
  },
  {
    id: "clinical-trials-l4-q30",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "Which statement best distinguishes full-service outsourcing from a functional-service provider model?",
    options: [
      {
        text: "Full-service outsourcing delegates many trial functions to one CRO, while a functional-service provider model delegates specific functions and leaves more coordination with the sponsor.",
        isCorrect: true,
      },
      {
        text: "Full-service outsourcing is used for scientific design, while functional-service providers are used after database lock for publication writing.",
        isCorrect: false,
      },
      {
        text: "Functional-service providers own the study, while full-service CROs support the sponsor without receiving delegated tasks.",
        isCorrect: false,
      },
      {
        text: "Full-service outsourcing removes the sponsor's need for oversight, while functional outsourcing preserves regulatory accountability.",
        isCorrect: false,
      },
    ],
    explanation:
      "Full-service outsourcing can simplify execution for sponsors with limited infrastructure, but it still requires oversight. Functional outsourcing can provide specialized support for areas such as monitoring or data management, but the sponsor must coordinate the remaining pieces.",
  },
  {
    id: "clinical-trials-l4-q31",
    chapter: 4,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe vendors in modern clinical trial operations?",
    options: [
      {
        text: "Central labs, imaging vendors, eCOA vendors, randomization systems, drug depots, translation vendors, and wearable vendors can support specialized trial functions.",
        isCorrect: true,
      },
      {
        text: "Vendors add capability but also create dependencies that require coordination and oversight.",
        isCorrect: true,
      },
      {
        text: "Vendor problems can affect endpoint data, treatment assignment, drug supply, patient visits, or regulatory documents.",
        isCorrect: true,
      },
      {
        text: "Vendor management ends once a contract is signed because vendors operate outside the trial quality system.",
        isCorrect: false,
      },
    ],
    explanation:
      "Trials often depend on networks of specialized vendors. Those vendors can add needed expertise and infrastructure, but delays, inconsistent processes, device failures, or supply issues can undermine trial conduct and data quality.",
  },
  {
    id: "clinical-trials-l4-q32",
    chapter: 4,
    difficulty: "easy",
    prompt:
      "Which activities are part of investigational product management in a drug trial?",
    options: [
      {
        text: "Manufacturing, labeling, packaging, shipping, and storage.",
        isCorrect: true,
      },
      {
        text: "Temperature control, drug supply, dispensing, and returns.",
        isCorrect: true,
      },
      {
        text: "Accountability for what was received, dispensed, returned, lost, damaged, or destroyed.",
        isCorrect: true,
      },
      {
        text: "Packaging controls that preserve blinding when the trial is blinded.",
        isCorrect: true,
      },
    ],
    explanation:
      "Investigational product management is a practical trial-quality issue. Cold-chain failures, stockouts, incorrect kits, wrong dispensing, or poor accountability can create missed doses, safety issues, deviations, or threats to blinding.",
  },
  {
    id: "clinical-trials-l4-q33",
    chapter: 4,
    difficulty: "medium",
    prompt:
      "Which statements correctly distinguish scientific failure from operational failure?",
    options: [
      {
        text: "Scientific failure means the intervention lacks sufficient benefit or has unacceptable toxicity despite a well-run trial.",
        isCorrect: true,
      },
      {
        text: "Operational failure means execution problems prevent the trial from adequately answering the question.",
        isCorrect: true,
      },
      {
        text: "Operational failure is limited to recruitment delays and excludes data quality, consent, vendor, or monitoring problems.",
        isCorrect: false,
      },
      {
        text: "Scientific failure is diagnosed when database lock occurs later than the original project timeline.",
        isCorrect: false,
      },
    ],
    explanation:
      "A treatment can fail because the biology is wrong, but a trial can also fail because execution undermines the evidence. Recruitment, endpoint measurement, data quality, protocol deviations, consent, safety reporting, and delays can all make a trial unable to answer its question.",
  },
  {
    id: "clinical-trials-l4-q34",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "A cardiovascular trial assumed a 10% control event rate, but improved standard care lowers the actual event rate to 5%. What is the main implication for the trial?",
    options: [
      {
        text: "The trial may become underpowered because fewer events occur than planned, leading to wider intervals and inconclusive results.",
        isCorrect: true,
      },
      {
        text: "The trial gains power because lower event rates make each event more informative for treatment comparison.",
        isCorrect: false,
      },
      {
        text: "The issue is a protocol-deviation problem because enrolled patients failed eligibility criteria after standard care improved.",
        isCorrect: false,
      },
      {
        text: "The endpoint becomes clinically meaningless because event rates below the planning assumption invalidate mortality outcomes.",
        isCorrect: false,
      },
    ],
    explanation:
      "Power depends on the amount of information, and event-driven trials need enough events. If the actual event rate is much lower than expected, the trial may have less power, wider confidence intervals, and a higher risk of an inconclusive result.",
  },
  {
    id: "clinical-trials-l4-q35",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "Which examples correctly match common failure categories in clinical trials?",
    options: [
      {
        text: "An endpoint is too rare or not clinically meaningful: poor endpoint selection.",
        isCorrect: true,
      },
      {
        text: "Unexpected adverse events lead to a hold or termination: safety issue.",
        isCorrect: true,
      },
      {
        text: "Missing primary endpoint data and inconsistent assessments undermine interpretation: data quality problem.",
        isCorrect: true,
      },
      {
        text: "Sites enroll faster than expected and queries close quickly: operational delay.",
        isCorrect: false,
      },
    ],
    explanation:
      "Trial failures can come from endpoint choice, safety, data quality, and many other operational categories. Fast enrollment and rapid query closure are positive operational signals, not examples of delay.",
  },
  {
    id: "clinical-trials-l4-q36",
    chapter: 4,
    difficulty: "hard",
    prompt: "Which problems can excessive protocol complexity create?",
    options: [
      {
        text: "Lower recruitment because patients decline burdensome visits or procedures.",
        isCorrect: true,
      },
      {
        text: "Lower retention because participants struggle with travel, schedules, or assessments.",
        isCorrect: true,
      },
      {
        text: "More protocol deviations because sites have too many procedures, windows, or criteria to execute consistently.",
        isCorrect: true,
      },
      {
        text: "Higher cost and data-quality pressure because more assessments and vendors must be coordinated.",
        isCorrect: true,
      },
    ],
    explanation:
      "Protocol complexity is a major operational risk because it affects patients, sites, vendors, timelines, costs, and data. A scientifically ambitious protocol can become unrealistic if the visit schedule, procedures, endpoints, and criteria exceed what sites and participants can sustain.",
  },
  {
    id: "clinical-trials-l4-q37",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "A trial's EDC build is delayed, lab kits arrive late, and contracts in several countries take months longer than planned. Which statements correctly describe the operational risk?",
    options: [
      {
        text: "Delays in one dependency can cascade into screening, data entry, recruitment, database lock, reporting, and regulatory submission.",
        isCorrect: true,
      },
      {
        text: "Global trials are vulnerable to country-specific differences in contracts, ethics approval, privacy rules, language, and regulation.",
        isCorrect: true,
      },
      {
        text: "Startup delays usually improve data quality because sites have more time before first patient first visit.",
        isCorrect: false,
      },
      {
        text: "Operational delays matter for project timelines but not for patient access or sponsor development decisions.",
        isCorrect: false,
      },
    ],
    explanation:
      "Trial operations are interconnected, so a delay in systems, supplies, contracts, approvals, or vendors can move many downstream milestones. In pharma, months of delay can affect cost, patient access, regulatory timing, and company decisions.",
  },
  {
    id: "clinical-trials-l4-q38",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "A monitor identifies that several sites misunderstand an eligibility criterion, but escalation is slow and the clarification does not reach every site. What failure mode does this best illustrate?",
    options: [
      {
        text: "Poor communication across sponsor, CRO, sites, and monitors.",
        isCorrect: true,
      },
      {
        text: "Publication bias caused by selective reporting of positive trials.",
        isCorrect: false,
      },
      {
        text: "Odds-ratio exaggeration because the endpoint is common.",
        isCorrect: false,
      },
      {
        text: "A pharmacokinetic dose-proportionality problem caused by drug metabolism.",
        isCorrect: false,
      },
    ],
    explanation:
      "The problem is not statistical interpretation or drug mechanism; it is a communication and escalation failure. Trial teams need structured communication so protocol clarifications, recurring site issues, vendor changes, and safety concerns reach the right people quickly.",
  },
  {
    id: "clinical-trials-l4-q39",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "Which operational metrics can help a trial team detect execution problems early?",
    options: [
      {
        text: "Site activation counts, activation timelines, screening counts, screen-failure rate, and enrollment rate.",
        isCorrect: true,
      },
      {
        text: "Dropout rate, data-entry timeliness, open queries, query aging, and missing endpoint data.",
        isCorrect: true,
      },
      {
        text: "Monitoring visit completion, protocol-deviation rate, and serious adverse event reporting timeliness.",
        isCorrect: true,
      },
      {
        text: "The final p-value, which is the earliest indicator that operational execution is on track.",
        isCorrect: false,
      },
    ],
    explanation:
      "Key performance indicators are operational early-warning signals. They help teams decide whether to add sites, retrain staff, revise recruitment strategy, resolve data backlogs, or escalate site performance issues before the scientific question is compromised.",
  },
  {
    id: "clinical-trials-l4-q40",
    chapter: 4,
    difficulty: "hard",
    prompt:
      "Which sequence and roles correctly describe the operational lifecycle of a clinical trial?",
    options: [
      {
        text: "Protocol development is followed by startup, recruitment, conduct, monitoring, data cleaning, database lock, analysis, and reporting.",
        isCorrect: true,
      },
      {
        text: "Clinical scientists, project managers, study managers, CRAs, investigators, coordinators, data managers, statisticians, programmers, medical monitors, safety teams, regulatory teams, writers, quality teams, ethics committees, and regulators each contribute different parts.",
        isCorrect: true,
      },
      {
        text: "Last patient last visit is followed by query resolution, coding, reconciliation, database lock, statistical analysis, and clinical study reporting.",
        isCorrect: true,
      },
      {
        text: "Reliable evidence requires scientific design, ethical conduct, operational feasibility, clean data, safety oversight, and planned analysis to work together.",
        isCorrect: true,
      },
    ],
    explanation:
      "Clinical trials are distributed operational systems, not just scientific ideas. The lifecycle runs from protocol and startup through recruitment, conduct, monitoring, cleaning, lock, analysis, and reporting, with many specialized roles needed to produce credible and ethical evidence.",
  },
];
