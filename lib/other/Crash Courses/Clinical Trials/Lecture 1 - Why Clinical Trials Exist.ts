import { Question } from "../../../quiz";

export const ClinicalTrialsLecture1Questions: Question[] = [
  {
    id: "clinical-trials-l1-q01",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "A clinician sees patients improve after a plausible new treatment and wants to know whether the treatment caused the improvement. Which reasons explain why a clinical trial is needed?",
    options: [
      {
        text: "Patient outcomes can improve for reasons unrelated to a treatment.",
        isCorrect: true,
      },
      {
        text: "Expert judgment and biological plausibility can be wrong.",
        isCorrect: true,
      },
      {
        text: "Clinical experience alone can confuse coincidence with causation.",
        isCorrect: true,
      },
      {
        text: "Controlled evidence helps separate real treatment effects from noise.",
        isCorrect: true,
      },
    ],
    explanation:
      "Clinical trials exist because medicine is full of variation, bias, and uncertainty. A patient may improve because of natural recovery, placebo effects, regression to the mean, or other factors, so controlled evidence is needed before concluding that a treatment caused the improvement.",
  },
  {
    id: "clinical-trials-l1-q02",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "A doctor gives a new pill to 100 people with headaches, and 80 feel better after three days. Which conclusions are justified from that observation alone?",
    options: [
      {
        text: "The pill caused the improvement in most patients.",
        isCorrect: false,
      },
      {
        text: "Some headaches may have resolved naturally.",
        isCorrect: true,
      },
      {
        text: "The observed improvement is enough to conclude the pill outperformed placebo.",
        isCorrect: false,
      },
      {
        text: "A comparison group would be needed to estimate the pill's effect.",
        isCorrect: true,
      },
    ],
    explanation:
      "Improvement after treatment does not prove improvement because of treatment. Headaches often improve on their own, and without a control group there is no way to know how many patients would have improved without the pill.",
  },
  {
    id: "clinical-trials-l1-q03",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statements correctly distinguish correlation from causation in clinical research?",
    options: [
      {
        text: "Correlation means two variables move together or are associated.",
        isCorrect: true,
      },
      {
        text: "Causation means one factor directly contributes to a change in another.",
        isCorrect: true,
      },
      {
        text: "A correlation can be created by a third factor that affects both variables.",
        isCorrect: true,
      },
      {
        text: "A correlation by itself is not enough to prove a treatment works.",
        isCorrect: true,
      },
    ],
    explanation:
      "Correlation is an association, while causation is a claim about what produced an effect. Clinical research is difficult because correlated variables may be linked through confounding, selection, or disease severity rather than a true treatment effect.",
  },
  {
    id: "clinical-trials-l1-q04",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "In an observational study, sicker patients are more likely to receive Drug A and also more likely to die. Which statements correctly describe the confounding problem?",
    options: [
      {
        text: "Confounding occurs when another factor helps explain an apparent exposure-outcome relationship.",
        isCorrect: true,
      },
      {
        text: "Sicker patients receiving a medication can make the medication look harmful even if illness severity is the real driver.",
        isCorrect: true,
      },
      {
        text: "Randomization is one way to reduce confounding in treatment comparisons.",
        isCorrect: true,
      },
      {
        text: "Confounding is a major threat in observational medical studies.",
        isCorrect: true,
      },
    ],
    explanation:
      "A confounder is a factor that is related to both the exposure and the outcome, creating a misleading association. Observational studies are especially vulnerable because people who receive an exposure or treatment often differ from those who do not.",
  },
  {
    id: "clinical-trials-l1-q05",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe placebo effects in clinical research?",
    options: [
      {
        text: "Expectations and treatment context can influence symptoms.",
        isCorrect: true,
      },
      {
        text: "Placebo responses are especially relevant for subjective outcomes such as pain.",
        isCorrect: true,
      },
      {
        text: "A placebo response proves the patient's symptoms were fake rather than experienced.",
        isCorrect: false,
      },
      {
        text: "Placebo effects make single-arm improvement a direct estimate of drug effect.",
        isCorrect: false,
      },
    ],
    explanation:
      "Placebo effects can involve real perceived or physiological changes driven by expectation, care context, and behavior. They do not mean patients are dishonest, and they make comparison groups more important rather than unnecessary.",
  },
  {
    id: "clinical-trials-l1-q06",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "In a headache study, many untreated patients would improve within three days. Which statement best captures the natural history issue?",
    options: [
      {
        text: "Some conditions improve, worsen, or fluctuate without any intervention.",
        isCorrect: true,
      },
      {
        text: "Untreated patients provide no useful information once treated patients improve.",
        isCorrect: false,
      },
      {
        text: "Natural history makes treated improvement easier to interpret without a comparator.",
        isCorrect: false,
      },
      {
        text: "Natural history is relevant only when the disease is fatal and never fluctuates.",
        isCorrect: false,
      },
    ],
    explanation:
      "Natural history refers to how a disease behaves without the tested intervention. Because many conditions fluctuate or resolve over time, researchers need controls to avoid crediting a treatment for changes that would have happened anyway.",
  },
  {
    id: "clinical-trials-l1-q07",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "A pain trial enrolls patients when their pain scores are unusually high. Which statements correctly describe regression to the mean in that setting?",
    options: [
      {
        text: "People often seek care or enroll in studies when symptoms are unusually severe.",
        isCorrect: true,
      },
      {
        text: "Extreme measurements often become less extreme on later measurement.",
        isCorrect: true,
      },
      {
        text: "Regression to the mean can create the appearance of treatment benefit.",
        isCorrect: true,
      },
      {
        text: "Regression to the mean can occur without fraud or deliberate manipulation.",
        isCorrect: true,
      },
    ],
    explanation:
      "Regression to the mean is a statistical and clinical pattern in which unusually high or low values tend to move closer to typical values over time. If patients enter a study at a symptom peak, some improvement may occur even without an effective treatment.",
  },
  {
    id: "clinical-trials-l1-q08",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "A trial mainly enrolls healthier, highly motivated volunteers who differ from the broader patient population. Which statements correctly describe selection bias?",
    options: [
      {
        text: "Selection bias can occur when study groups differ before treatment is assessed.",
        isCorrect: true,
      },
      {
        text: "Trial volunteers may be healthier or more motivated than the broader patient population.",
        isCorrect: true,
      },
      {
        text: "Selection bias can affect how well results generalize.",
        isCorrect: true,
      },
      {
        text: "Selection bias concerns who enters or remains in comparison groups rather than the biological treatment mechanism itself.",
        isCorrect: true,
      },
    ],
    explanation:
      "Selection bias happens when the people included in a group are systematically different from the people being compared against or from the target population. Those differences can distort estimated treatment effects or limit generalizability.",
  },
  {
    id: "clinical-trials-l1-q09",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "An unblinded clinician assesses borderline outcomes after learning which patients received the experimental drug. Which statements correctly describe observer bias?",
    options: [
      {
        text: "Knowledge of treatment assignment can influence how clinicians assess outcomes.",
        isCorrect: true,
      },
      {
        text: "Observer bias can occur even when researchers are acting honestly.",
        isCorrect: true,
      },
      {
        text: "Blinding can reduce observer bias.",
        isCorrect: true,
      },
      {
        text: "Observer bias can affect human judgments even when laboratory machines are not the issue.",
        isCorrect: true,
      },
    ],
    explanation:
      "Observer bias is not necessarily intentional; expectations can shape how people interpret symptoms, scans, or borderline outcomes. Blinding helps because it reduces the chance that knowledge of assignment changes assessment behavior.",
  },
  {
    id: "clinical-trials-l1-q10",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "A reviewer can find several positive trials but suspects negative or inconclusive studies may be missing from the literature. Which statements correctly describe publication bias?",
    options: [
      {
        text: "Studies with positive findings may be more likely to appear in the literature.",
        isCorrect: true,
      },
      {
        text: "Publication bias can make treatments appear more effective than they are.",
        isCorrect: true,
      },
      {
        text: "Publication bias is one reason systematic reviewers look for unpublished or missing studies.",
        isCorrect: true,
      },
      {
        text: "Publication bias can remain even when published studies went through peer review.",
        isCorrect: true,
      },
    ],
    explanation:
      "If positive studies are easier to find than negative or inconclusive ones, the visible literature can exaggerate benefits. Evidence synthesis must therefore consider the possibility that some unfavorable results are missing.",
  },
  {
    id: "clinical-trials-l1-q11",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which evidence types generally sit lower in the evidence hierarchy for causal claims?",
    options: [
      { text: "Expert opinion.", isCorrect: true },
      { text: "Case reports.", isCorrect: true },
      { text: "Case series without a control group.", isCorrect: true },
      {
        text: "High-quality systematic reviews of randomized trials.",
        isCorrect: false,
      },
    ],
    explanation:
      "Expert opinion, case reports, and uncontrolled case series can be useful for generating hypotheses or detecting unusual events, but they are weak for proving causation. Systematic reviews of high-quality randomized evidence are usually stronger because they synthesize controlled comparisons.",
  },
  {
    id: "clinical-trials-l1-q12",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "What is the central purpose of a control group in a clinical trial?",
    options: [
      {
        text: "To estimate what would have happened without the tested intervention.",
        isCorrect: true,
      },
      {
        text: "To make the experimental arm larger whenever early responses look promising.",
        isCorrect: false,
      },
      {
        text: "To stop natural recovery, placebo effects, and background care from occurring.",
        isCorrect: false,
      },
      {
        text: "To estimate safety while leaving efficacy to clinician judgment.",
        isCorrect: false,
      },
    ],
    explanation:
      "A control group gives researchers a comparison for natural recovery, placebo effects, background care, and other influences. It cannot remove all uncertainty by itself, but it makes the causal question much more answerable.",
  },
  {
    id: "clinical-trials-l1-q13",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statement best states the fundamental counterfactual problem in treatment research?",
    options: [
      {
        text: "The same patient cannot be observed both receiving and not receiving the treatment at the same time.",
        isCorrect: true,
      },
      {
        text: "The untreated outcome for a treated patient can be directly recovered from the same patient's treated follow-up.",
        isCorrect: false,
      },
      {
        text: "The treated outcome is uninterpretable whenever patients differ at baseline.",
        isCorrect: false,
      },
      {
        text: "Randomization reveals both possible outcomes for each individual participant.",
        isCorrect: false,
      },
    ],
    explanation:
      "The causal question asks what would have happened to the same person under a different treatment condition. Because that counterfactual world cannot be directly observed, trials use comparison groups and randomization as practical substitutes.",
  },
  {
    id: "clinical-trials-l1-q14",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which stakeholders commonly participate in the clinical trial ecosystem?",
    options: [
      {
        text: "Sponsors such as pharmaceutical, biotech, device, or academic organizations.",
        isCorrect: true,
      },
      {
        text: "Investigators and clinical sites that recruit and care for participants.",
        isCorrect: true,
      },
      {
        text: "Regulators and ethics oversight bodies that review evidence and protect patients.",
        isCorrect: true,
      },
      {
        text: "Patients whose outcomes, safety, and consent are central to the study.",
        isCorrect: true,
      },
    ],
    explanation:
      "Clinical trials are coordinated systems rather than isolated scientific exercises. Sponsors, investigators, sites, regulators, ethics bodies, Contract Research Organizations, and patients each affect whether the study is ethical, feasible, and interpretable.",
  },
  {
    id: "clinical-trials-l1-q15",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which claims about historical medical practice are supported by the logic of evidence-based medicine?",
    options: [
      {
        text: "Long-standing use of a treatment is not proof that it helps patients.",
        isCorrect: true,
      },
      {
        text: "A plausible theory is sufficient evidence that a treatment benefits patients.",
        isCorrect: false,
      },
      {
        text: "Bloodletting illustrates how tradition can persist despite harm.",
        isCorrect: true,
      },
      {
        text: "Historical authority is more reliable than controlled comparison.",
        isCorrect: false,
      },
    ],
    explanation:
      "Evidence-based medicine grew partly from recognizing that authority, tradition, and plausible mechanisms can mislead. Historical examples such as bloodletting show why good intentions and accepted practice need empirical testing rather than assuming plausible theories guarantee benefit.",
  },
  {
    id: "clinical-trials-l1-q16",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Observational studies suggested hormone replacement therapy reduced cardiovascular risk, but later randomized evidence showed increased risks. Which lessons does that example teach?",
    options: [
      {
        text: "Observational associations are immune to differences between people who choose or receive treatment and those who do not.",
        isCorrect: false,
      },
      {
        text: "Randomized trials can overturn confident conclusions based on nonrandomized evidence.",
        isCorrect: true,
      },
      {
        text: "Biological plausibility and favorable observational findings are not enough for causal certainty.",
        isCorrect: true,
      },
      {
        text: "A treatment must be beneficial whenever healthier people are more likely to use it.",
        isCorrect: false,
      },
    ],
    explanation:
      "The hormone replacement therapy example is a classic warning about confounding and selection in observational research. People receiving a therapy may differ in income, health behavior, baseline risk, and healthcare access, so observational associations are not immune to bias and randomized evidence can change the conclusion.",
  },
  {
    id: "clinical-trials-l1-q17",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements correctly compare observational studies with randomized controlled trials?",
    options: [
      {
        text: "Observational studies are inherently slower and less practical for rare outcomes.",
        isCorrect: false,
      },
      {
        text: "Randomized controlled trials are generally stronger for causal inference about interventions.",
        isCorrect: true,
      },
      {
        text: "Observational studies are invalid by design and lack evidentiary value.",
        isCorrect: false,
      },
      {
        text: "Randomization helps balance unknown as well as known confounders.",
        isCorrect: true,
      },
    ],
    explanation:
      "Observational research is often valuable, especially when randomization is infeasible, unethical, or too slow, and it can sometimes be practical for rare outcomes. However, randomized controlled trials are usually stronger for intervention causality because assignment by chance reduces systematic group differences.",
  },
  {
    id: "clinical-trials-l1-q18",
    chapter: 1,
    difficulty: "medium",
    prompt: "Which statements correctly describe case reports and case series?",
    options: [
      {
        text: "They can highlight rare diseases, unexpected events, or new hypotheses.",
        isCorrect: true,
      },
      {
        text: "They usually include randomized comparison groups.",
        isCorrect: false,
      },
      {
        text: "They are weak evidence for proving that a treatment caused improvement.",
        isCorrect: true,
      },
      {
        text: "They eliminate placebo effects and natural recovery by design.",
        isCorrect: false,
      },
    ],
    explanation:
      "Case reports and case series can be clinically important because they notice unusual patterns. Their weakness is that they usually lack randomized comparison groups, so they cannot reliably distinguish treatment effects from background disease course, expectation, selection, or coincidence.",
  },
  {
    id: "clinical-trials-l1-q19",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe systematic reviews and meta-analyses?",
    options: [
      {
        text: "They synthesize evidence from a single selected study.",
        isCorrect: false,
      },
      {
        text: "Their conclusions depend on the quality and completeness of the underlying studies.",
        isCorrect: true,
      },
      {
        text: "They can be weakened by publication bias or heterogeneity.",
        isCorrect: true,
      },
      {
        text: "They make poor individual studies flawless simply by combining them.",
        isCorrect: false,
      },
    ],
    explanation:
      "Evidence synthesis can be powerful because it summarizes evidence across multiple studies, not just one study. It does not magically remove flaws; biased, missing, or highly inconsistent studies can still lead to unreliable pooled conclusions.",
  },
  {
    id: "clinical-trials-l1-q20",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statement best explains why randomization emerged as a major tool in clinical trials?",
    options: [
      {
        text: "It makes treatment assignment less dependent on patient prognosis, physician preference, or hidden risk factors.",
        isCorrect: true,
      },
      {
        text: "It forces measured baseline variables to match exactly in each group.",
        isCorrect: false,
      },
      {
        text: "It prevents missing data and protocol deviations by design.",
        isCorrect: false,
      },
      {
        text: "It removes the need for a prespecified outcome.",
        isCorrect: false,
      },
    ],
    explanation:
      "Randomization is powerful because it breaks the link between treatment assignment and many patient characteristics. It does not guarantee perfect balance or solve every trial problem, but it reduces systematic bias in ways physician-selected treatment cannot.",
  },
  {
    id: "clinical-trials-l1-q21",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "A study finds that people who take vitamins have lower mortality. Later analysis shows vitamin users exercise more, smoke less, and have higher income. Which interpretations are appropriate?",
    options: [
      {
        text: "The observed association is insulated from healthier behaviors or socioeconomic differences.",
        isCorrect: false,
      },
      {
        text: "Vitamin use is proven to be the direct cause of lower mortality.",
        isCorrect: false,
      },
      {
        text: "Differences in baseline risk can explain an apparent benefit.",
        isCorrect: true,
      },
      {
        text: "The example illustrates why observational associations need cautious interpretation.",
        isCorrect: true,
      },
    ],
    explanation:
      "The vitamin example illustrates how a treatment or exposure can be correlated with many other favorable characteristics. If those characteristics reduce mortality, the vitamin association may be confounded rather than representing a causal vitamin effect.",
  },
  {
    id: "clinical-trials-l1-q22",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which roles are typically associated with a clinical trial sponsor?",
    options: [
      {
        text: "Funding or overseeing the trial.",
        isCorrect: true,
      },
      {
        text: "Avoiding any involvement in study protocol responsibilities.",
        isCorrect: false,
      },
      {
        text: "Interacting with regulators about the evidence package.",
        isCorrect: true,
      },
      {
        text: "Personally providing routine care at study sites as the sponsor's main role.",
        isCorrect: false,
      },
    ],
    explanation:
      "The sponsor is the organization responsible for initiating, funding, and overseeing the trial, including broad responsibility for the study plan. Site investigators and clinical staff provide direct patient care and collect data, while the sponsor coordinates the broader study and regulatory strategy.",
  },
  {
    id: "clinical-trials-l1-q23",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "A sponsor outsources monitoring, data management, and project-management work while retaining responsibility for the trial. Which statements correctly describe the Contract Research Organization (CRO) role?",
    options: [
      {
        text: "They can conduct trial activities on behalf of sponsors.",
        isCorrect: true,
      },
      {
        text: "They can replace regulators by deciding whether the therapy should be approved for marketing.",
        isCorrect: false,
      },
      {
        text: "They are used because sponsors often outsource substantial trial operations.",
        isCorrect: true,
      },
      {
        text: "They remove the sponsor's ethical and regulatory obligations once operations are outsourced.",
        isCorrect: false,
      },
    ],
    explanation:
      "A Contract Research Organization is an operational partner that helps run parts of a study, and sponsors may outsource substantial trial operations to them. They do not approve therapies for marketing or remove the ethical and regulatory obligations that protect patients and preserve data integrity.",
  },
  {
    id: "clinical-trials-l1-q24",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "At a clinical site, an investigator team interacts directly with participants during a trial. Which responsibilities are commonly associated with that site role?",
    options: [
      {
        text: "Recruiting eligible participants.",
        isCorrect: true,
      },
      {
        text: "Obtaining informed consent from participants.",
        isCorrect: true,
      },
      {
        text: "Collecting study data according to the protocol.",
        isCorrect: true,
      },
      {
        text: "Making the final regulatory approval decision after the site's participants complete follow-up.",
        isCorrect: false,
      },
    ],
    explanation:
      "Investigators and sites are where the trial meets patients, so recruitment, consent, care, and data collection are central responsibilities. Marketing approval is a regulatory decision made by agencies, not by individual clinical sites.",
  },
  {
    id: "clinical-trials-l1-q25",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statement correctly describes the role of regulators such as the U.S. Food and Drug Administration (FDA), European Medicines Agency (EMA), or Swissmedic?",
    options: [
      {
        text: "They evaluate evidence submitted for therapies or medical products.",
        isCorrect: true,
      },
      {
        text: "They evaluate only commercial demand and leave patient safety to sponsors.",
        isCorrect: false,
      },
      {
        text: "They mainly provide trial operations staff while CROs decide evidence requirements.",
        isCorrect: false,
      },
      {
        text: "They approve protocols ethically at each site but do not evaluate marketing applications.",
        isCorrect: false,
      },
    ],
    explanation:
      "Regulators review whether evidence supports approval and ongoing use of medical products, and patient safety is central to those decisions. Contract Research Organizations help operationalize trials, but they do not serve the same public oversight role as regulatory agencies.",
  },
  {
    id: "clinical-trials-l1-q26",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Before a human-subject trial starts, an independent body reviews risk, consent, and participant protection. Which statement correctly describes ethics committees or Institutional Review Boards (IRBs)?",
    options: [
      {
        text: "They provide independent ethical review of human-subject research.",
        isCorrect: true,
      },
      {
        text: "They replace scientific review by guaranteeing that the trial hypothesis will be positive.",
        isCorrect: false,
      },
      {
        text: "They operate mainly as sponsor project managers for recruitment and data cleaning.",
        isCorrect: false,
      },
      {
        text: "They approve market access once the trial has produced enough favorable data.",
        isCorrect: false,
      },
    ],
    explanation:
      "Ethics committees and Institutional Review Boards focus on independent ethical review and participant protection. They do not write marketing copy, guarantee commercial success, or determine scientific success; a well-protected trial can still produce negative or inconclusive results.",
  },
  {
    id: "clinical-trials-l1-q27",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe patients' role in modern clinical research?",
    options: [
      {
        text: "Patients are central stakeholders rather than merely data sources.",
        isCorrect: true,
      },
      {
        text: "Patient-centered outcomes can include quality of life and daily functioning.",
        isCorrect: true,
      },
      {
        text: "Patient-reported outcomes can capture symptoms that clinicians cannot directly observe.",
        isCorrect: true,
      },
      {
        text: "Patients' consent can be bypassed when a sponsor wants faster evidence.",
        isCorrect: false,
      },
    ],
    explanation:
      "Modern clinical research increasingly treats patients as stakeholders whose safety, consent, experience, and outcomes matter. Patient-reported outcomes are especially useful when the lived experience of disease is a key part of benefit or harm.",
  },
  {
    id: "clinical-trials-l1-q28",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "A physician reports that 90% of patients improved after receiving a new treatment. Which missing information would most directly help judge whether the treatment caused the improvement?",
    options: [
      {
        text: "How similar patients did without the new treatment.",
        isCorrect: true,
      },
      {
        text: "Whether patients were randomly assigned to treatment or comparison groups.",
        isCorrect: true,
      },
      {
        text: "Whether outcome assessors knew who received the treatment.",
        isCorrect: true,
      },
      {
        text: "Whether improvement could reflect natural recovery, placebo effects, or regression to the mean.",
        isCorrect: true,
      },
    ],
    explanation:
      "A high improvement rate is not enough to establish causation because the counterfactual is unknown. Control groups, randomization, blinding, and attention to alternative explanations all help determine whether the treatment caused the observed improvement.",
  },
  {
    id: "clinical-trials-l1-q29",
    chapter: 1,
    difficulty: "hard",
    prompt:
      'Which trial feature most directly addresses the counterfactual question, "What would have happened to these patients without the tested intervention?"',
    options: [
      { text: "A control or comparator group.", isCorrect: true },
      {
        text: "A longer follow-up period with no comparison arm.",
        isCorrect: false,
      },
      {
        text: "A mechanistic biomarker measured only in treated patients.",
        isCorrect: false,
      },
      {
        text: "A larger treated case series without untreated or usual-care patients.",
        isCorrect: false,
      },
    ],
    explanation:
      "The counterfactual question is approximated by comparing treated participants with a relevant control or comparator group. Branding, marketing, and investigator enthusiasm do not show what would have happened without the intervention.",
  },
  {
    id: "clinical-trials-l1-q30",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "A randomized controlled trial is poorly conducted, while an observational study is carefully designed and highly relevant. Which statement is the best interpretation?",
    options: [
      {
        text: "The evidence hierarchy is useful, but study quality and context still matter.",
        isCorrect: true,
      },
      {
        text: "A randomized trial is more informative than an observational study regardless of design quality or question.",
        isCorrect: false,
      },
      {
        text: "Observational designs lack value for medical evidence.",
        isCorrect: false,
      },
      {
        text: "Randomization removes the need for valid measurement and execution.",
        isCorrect: false,
      },
    ],
    explanation:
      "The evidence hierarchy is a guide, not a mechanical rule that ignores quality. A poorly executed randomized trial can be less useful than a rigorous observational study for some questions, although randomization remains especially valuable for causal inference.",
  },
  {
    id: "clinical-trials-l1-q31",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which factors could make an uncontrolled treatment series look successful even if the treatment has no active benefit?",
    options: [
      { text: "Natural recovery.", isCorrect: true },
      { text: "Placebo effects.", isCorrect: true },
      { text: "Regression to the mean.", isCorrect: true },
      {
        text: "A comparison group showing equal improvement without treatment.",
        isCorrect: false,
      },
    ],
    explanation:
      "Natural recovery, placebo effects, and regression to the mean can all create improvement that is not caused by the active treatment. A comparison group with equal improvement would undermine the claim of active benefit rather than make the uncontrolled series look more convincing.",
  },
  {
    id: "clinical-trials-l1-q32",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "A nonrandomized study shows patients receiving Drug A have higher mortality than untreated patients. Which explanations remain plausible without more design details?",
    options: [
      {
        text: "Drug A could be harmful.",
        isCorrect: true,
      },
      {
        text: "Sicker patients may have been more likely to receive Drug A.",
        isCorrect: true,
      },
      {
        text: "The groups may differ in unmeasured risk factors.",
        isCorrect: true,
      },
      {
        text: "The association establishes Drug A as the cause of deaths in the treated group.",
        isCorrect: false,
      },
    ],
    explanation:
      "A harmful drug effect is possible, but confounding by indication is also possible when sicker patients are more likely to receive treatment. Without randomization or careful adjustment, the observed association cannot be treated as definitive causation.",
  },
  {
    id: "clinical-trials-l1-q33",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements correctly explain why blinding becomes relevant even after randomization?",
    options: [
      {
        text: "Randomization addresses assignment bias, while blinding addresses expectation and assessment bias.",
        isCorrect: true,
      },
      {
        text: "Knowledge of assignment can affect patient reports, clinician behavior, or outcome assessment.",
        isCorrect: true,
      },
      {
        text: "Blinding has little value whenever outcomes are partly subjective.",
        isCorrect: false,
      },
      {
        text: "Randomization alone does not prevent every post-assignment source of bias.",
        isCorrect: true,
      },
    ],
    explanation:
      "Randomization helps create comparable groups at assignment, but it does not prevent expectations from influencing behavior or assessment afterward. Blinding is especially important when outcomes involve judgment, symptoms, or clinical discretion.",
  },
  {
    id: "clinical-trials-l1-q34",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "A treatment is biologically plausible and several clinicians report impressive patient recoveries. Which statements best reflect an evidence-based response?",
    options: [
      {
        text: "The observations are hypothesis-generating rather than definitive proof.",
        isCorrect: true,
      },
      {
        text: "A comparison against appropriate controls is needed to estimate causal effect.",
        isCorrect: true,
      },
      {
        text: "The mechanism and anecdotes eliminate the possibility of placebo effects or natural recovery.",
        isCorrect: false,
      },
      {
        text: "Bias and random variation remain possible explanations.",
        isCorrect: true,
      },
    ],
    explanation:
      "Biological plausibility and clinical anecdotes can justify further testing, but they do not settle the causal question. Evidence-based reasoning asks what happened in comparable patients without the intervention and whether bias, variation, or natural disease course can explain the pattern.",
  },
  {
    id: "clinical-trials-l1-q35",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statement best explains why positive published literature can overstate a treatment's benefit?",
    options: [
      {
        text: "Positive studies may be easier to publish and find than negative or inconclusive studies.",
        isCorrect: true,
      },
      {
        text: "Peer review guarantees that the visible published studies represent all completed studies.",
        isCorrect: false,
      },
      {
        text: "A meta-analysis of only published studies automatically corrects for missing negative studies.",
        isCorrect: false,
      },
      {
        text: "The direction of results affects interpretation only after a treatment is approved.",
        isCorrect: false,
      },
    ],
    explanation:
      "Publication bias means the accessible evidence may be a selected subset of all completed evidence. If favorable findings are overrepresented, readers may overestimate benefit and underestimate uncertainty or harm.",
  },
  {
    id: "clinical-trials-l1-q36",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements correctly connect evidence hierarchy to causal confidence?",
    options: [
      {
        text: "Controlled comparison generally increases confidence compared with uncontrolled observation.",
        isCorrect: true,
      },
      {
        text: "Randomization generally increases confidence by reducing systematic baseline differences.",
        isCorrect: true,
      },
      {
        text: "Combining multiple high-quality studies can increase confidence in a conclusion.",
        isCorrect: true,
      },
      {
        text: "A high evidence-hierarchy position makes a conclusion true across patients.",
        isCorrect: false,
      },
    ],
    explanation:
      "The hierarchy reflects design features that reduce common threats to validity, such as absent controls, confounding, and small sample uncertainty. It still does not guarantee universal truth because study quality, applicability, bias, and chance must be considered.",
  },
  {
    id: "clinical-trials-l1-q37",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which questions would a critical reader ask before trusting a claim that an intervention works?",
    options: [
      {
        text: "Was there an appropriate comparator?",
        isCorrect: true,
      },
      {
        text: "Were patients assigned in a way that reduced confounding?",
        isCorrect: true,
      },
      {
        text: "Were outcomes assessed in a way that reduced expectation or observer bias?",
        isCorrect: true,
      },
      {
        text: "Did some patients improve after receiving the intervention?",
        isCorrect: true,
      },
    ],
    explanation:
      "All four questions are relevant, but the last one is only a starting observation rather than sufficient proof. A critical reader connects improvement to design features that help rule out natural recovery, placebo effects, confounding, and biased assessment.",
  },
  {
    id: "clinical-trials-l1-q38",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe why humans are poor at judging treatment effects from ordinary observation?",
    options: [
      {
        text: "Humans are good at noticing patterns, including patterns produced by chance.",
        isCorrect: true,
      },
      {
        text: "Clinicians may remember dramatic successes more than ordinary failures.",
        isCorrect: true,
      },
      {
        text: "Without a counterfactual comparison, improvement is easy to misattribute.",
        isCorrect: true,
      },
      {
        text: "Training as a clinician removes cognitive bias.",
        isCorrect: false,
      },
    ],
    explanation:
      "Clinical expertise is valuable, but human judgment still has pattern-detection and memory biases. Trial methods exist because ordinary observation does not reveal the untreated counterfactual and can easily assign causality to the wrong factor.",
  },
  {
    id: "clinical-trials-l1-q39",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "A small uncontrolled case series reports dramatic improvement in a rare condition. Which interpretation is most defensible?",
    options: [
      {
        text: "The finding may be important enough to motivate further study, but it does not establish causal efficacy.",
        isCorrect: true,
      },
      {
        text: "The finding is sufficient to approve the treatment broadly across patients.",
        isCorrect: false,
      },
      {
        text: "The absence of a control group strengthens causal inference.",
        isCorrect: false,
      },
      {
        text: "Rare conditions are insulated from selection bias and natural history.",
        isCorrect: false,
      },
    ],
    explanation:
      "Case series are often useful signals, especially in rare diseases or unexpected responses. They still lack the comparison needed to estimate what would have happened without treatment, so they are usually a starting point for stronger evidence.",
  },
  {
    id: "clinical-trials-l1-q40",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements correctly summarize the logic that connects clinical trials, stakeholders, and patient protection?",
    options: [
      {
        text: "Trials seek reliable evidence while protecting people who participate.",
        isCorrect: true,
      },
      {
        text: "Sponsors, sites, Contract Research Organizations, regulators, ethics bodies, and patients each shape trial quality.",
        isCorrect: true,
      },
      {
        text: "Patient consent and safety oversight are part of trustworthy clinical research.",
        isCorrect: true,
      },
      {
        text: "Operational coordination is a secondary concern once a scientific hypothesis exists.",
        isCorrect: false,
      },
    ],
    explanation:
      "Clinical research is both a scientific and operational system. Trustworthy trials require a strong causal design, ethical oversight, patient-centered conduct, and coordinated work across organizations that fund, run, monitor, regulate, and participate in the research.",
  },
  {
    id: "clinical-trials-l1-q41",
    chapter: 1,
    difficulty: "medium",
    type: "assertion-reason",
    prompt:
      "Assertion: Improvement after receiving a treatment is not, by itself, enough to show that the treatment caused the improvement.\n\nReason: Natural recovery, placebo effects, regression to the mean, and random variation can all produce improvement without a specific treatment effect.",
    options: [
      {
        text: "Assertion is true, Reason is false.",
        isCorrect: false,
      },
      {
        text: "Assertion is false, Reason is true.",
        isCorrect: false,
      },
      {
        text: "Both are false.",
        isCorrect: false,
      },
      {
        text: "Both are true, and the Reason is the correct explanation of the Assertion.",
        isCorrect: true,
      },
      {
        text: "Both are true, but the Reason is NOT the correct explanation of the Assertion.",
        isCorrect: false,
      },
    ],
    explanation:
      "The assertion is true because before-after improvement alone does not answer the causal question. The reason is also true and directly explains the assertion: several non-treatment mechanisms can make patients improve, so a trial needs comparison and design features that separate those mechanisms from a real treatment effect.",
  },
  {
    id: "clinical-trials-l1-q42",
    chapter: 1,
    difficulty: "hard",
    type: "assertion-reason",
    prompt:
      "Assertion: A randomized controlled trial is always more informative than any observational study, regardless of execution quality or research question.\n\nReason: Randomization helps reduce confounding by balancing known and unknown baseline factors between treatment groups.",
    options: [
      {
        text: "Assertion is true, Reason is false.",
        isCorrect: false,
      },
      {
        text: "Assertion is false, Reason is true.",
        isCorrect: true,
      },
      {
        text: "Both are false.",
        isCorrect: false,
      },
      {
        text: "Both are true, and the Reason is the correct explanation of the Assertion.",
        isCorrect: false,
      },
      {
        text: "Both are true, but the Reason is NOT the correct explanation of the Assertion.",
        isCorrect: false,
      },
    ],
    explanation:
      "The assertion is false because the evidence hierarchy is useful but not mechanical; poor execution, irrelevant populations, invalid measurement, or the wrong research question can make a randomized trial less informative than a strong observational study. The reason is true because randomization is valuable precisely because it reduces systematic baseline differences, including confounders researchers did not measure.",
  },
  {
    id: "clinical-trials-l1-q43",
    chapter: 1,
    difficulty: "medium",
    type: "assertion-reason",
    prompt:
      "Assertion: A high-quality systematic review can be strong evidence, but it does not automatically eliminate publication bias.\n\nReason: Systematic reviews and meta-analyses combine information across multiple studies.",
    options: [
      {
        text: "Assertion is true, Reason is false.",
        isCorrect: false,
      },
      {
        text: "Assertion is false, Reason is true.",
        isCorrect: false,
      },
      {
        text: "Both are false.",
        isCorrect: false,
      },
      {
        text: "Both are true, and the Reason is the correct explanation of the Assertion.",
        isCorrect: false,
      },
      {
        text: "Both are true, but the Reason is NOT the correct explanation of the Assertion.",
        isCorrect: true,
      },
    ],
    explanation:
      "The assertion is true because a review can only synthesize the evidence it identifies, and the visible evidence may still overrepresent favorable studies. The reason is also true, but it does not explain why publication bias remains; combining studies can increase precision, while missing negative or inconclusive studies can still distort the overall estimate.",
  },
  {
    id: "clinical-trials-l1-q44",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "A historically accepted treatment had a plausible theory, long clinical tradition, and many anecdotes of improvement, but later evidence showed it often harmed patients. Which lessons follow for evidence-based medicine?",
    options: [
      {
        text: "Biological plausibility and clinical familiarity can support a hypothesis, but they do not establish net benefit in patients.",
        isCorrect: true,
      },
      {
        text: "Anecdotes become stronger causal evidence when the underlying theory has been accepted for many years.",
        isCorrect: false,
      },
      {
        text: "A method can appear to help when clinicians observe natural recovery, transient symptom change, or selective memorable successes.",
        isCorrect: true,
      },
      {
        text: "Long use in practice makes formal comparison unnecessary unless the treatment is new.",
        isCorrect: false,
      },
    ],
    explanation:
      "Historical examples such as bloodletting show why plausibility, authority, and memorable clinical impressions can mislead. The core lesson is not that mechanistic thinking is worthless; it is that a treatment claim still needs controlled evidence about benefits and harms in humans.",
  },
  {
    id: "clinical-trials-l1-q45",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe the counterfactual problem behind treatment evaluation?",
    options: [
      {
        text: "The ideal comparison asks what would have happened to the same patient at the same time under the alternative treatment condition.",
        isCorrect: true,
      },
      {
        text: "Ordinary clinical observation misses the counterfactual because the clinician sees the treated path, not the untreated path for that same patient.",
        isCorrect: true,
      },
      {
        text: "Randomized comparisons try to approximate the missing counterfactual by comparing groups that are similar except for assignment.",
        isCorrect: true,
      },
      {
        text: "The counterfactual problem is solved by following treated patients long enough after therapy starts.",
        isCorrect: false,
      },
    ],
    explanation:
      "The causal question asks what would have happened under a different treatment choice for the same person, which cannot be directly observed. Randomization and control groups do not reveal that exact alternate history, but they create a credible group-level approximation.",
  },
  {
    id: "clinical-trials-l1-q46",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "A clinic starts offering a new fatigue treatment to patients who seek care when symptoms are at their worst. Three months later, most report feeling better. Which mechanisms could create that pattern without a specific treatment effect?",
    options: [
      {
        text: "Symptoms may fluctuate downward after patients enter care at an unusually severe point.",
        isCorrect: true,
      },
      {
        text: "Some patients may improve because the condition naturally waxes and wanes over time.",
        isCorrect: true,
      },
      {
        text: "Expectations, attention, and the care context may change subjective symptom reports.",
        isCorrect: true,
      },
      {
        text: "Patients who return for follow-up may differ from those who stop attending the clinic.",
        isCorrect: true,
      },
    ],
    explanation:
      "This scenario combines regression to the mean, natural history, placebo/context effects, and follow-up selection. A high improvement rate after treatment is therefore compatible with several noncausal explanations unless the study includes a credible comparison.",
  },
  {
    id: "clinical-trials-l1-q47",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Observational evidence suggested that hormone users had lower cardiovascular risk, but randomized trials later showed a less favorable benefit-risk picture. Which interpretations fit that contrast?",
    options: [
      {
        text: "Hormone users may have differed from nonusers in socioeconomic status, health behaviors, healthcare access, or baseline risk.",
        isCorrect: true,
      },
      {
        text: "The randomized trials were irrelevant because observational studies had already established the causal cardiovascular benefit.",
        isCorrect: false,
      },
      {
        text: "The example illustrates how a plausible mechanism can coexist with confounding in human outcome data.",
        isCorrect: true,
      },
      {
        text: "Confounding is limited to measured variables, so unmeasured behavior differences could not have affected the earlier association.",
        isCorrect: false,
      },
    ],
    explanation:
      "The hormone-therapy example is a classic warning about healthy-user and related confounding. A biological rationale may make an association tempting, but randomized evidence can reveal that the patient groups in observational data were not comparable in causal terms.",
  },
  {
    id: "clinical-trials-l1-q48",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "A treatment area has many small published studies, most with favorable findings, while trial registries show several completed studies with no available results. Which concerns are justified?",
    options: [
      {
        text: "The published literature may overestimate benefit if positive studies are easier to find than neutral or unfavorable studies.",
        isCorrect: true,
      },
      {
        text: "A meta-analysis of the published studies can increase precision around a biased evidence base.",
        isCorrect: true,
      },
      {
        text: "Searching registries and protocols can help detect missing evidence that ordinary database searches might miss.",
        isCorrect: true,
      },
      {
        text: "Publication bias matters mainly for case reports and does not affect randomized evidence syntheses.",
        isCorrect: false,
      },
    ],
    explanation:
      "Publication bias means the accessible literature may be a selected subset of the evidence. Meta-analysis can make the pooled estimate look precise, but it cannot fully repair evidence that is missing or selectively reported.",
  },
  {
    id: "clinical-trials-l1-q49",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "A sponsor funds a multicenter trial, hires a Contract Research Organization (CRO), and submits the results to regulators after independent ethics review. Which role descriptions are accurate?",
    options: [
      {
        text: "The sponsor is responsible for the trial and its evidence package even when work is outsourced.",
        isCorrect: true,
      },
      {
        text: "The CRO may run operational activities such as monitoring, data management, statistics support, or project management.",
        isCorrect: true,
      },
      {
        text: "Regulators evaluate whether the submitted evidence supports approval and whether safety has been adequately addressed.",
        isCorrect: true,
      },
      {
        text: "Ethics committees or Institutional Review Boards review participant protection, consent, and risk before and during human-subject research.",
        isCorrect: true,
      },
    ],
    explanation:
      "Modern clinical trials are multi-organization systems, but accountability and oversight remain structured. Sponsors, CROs, regulators, and ethics bodies have different responsibilities that together support credible evidence and participant protection.",
  },
  {
    id: "clinical-trials-l1-q50",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "A single patient experiences an unexpected severe reaction shortly after receiving a new biologic. What is the strongest use of that case report?",
    options: [
      {
        text: "It can generate a safety signal that deserves follow-up, especially if the event is rare or biologically plausible.",
        isCorrect: true,
      },
      {
        text: "It establishes the event rate because the outcome was severe and temporally close to treatment.",
        isCorrect: false,
      },
      {
        text: "It ranks above randomized evidence for estimating average benefit because individual detail is richer.",
        isCorrect: false,
      },
      {
        text: "It removes the need for comparison because rare adverse events are self-explanatory.",
        isCorrect: false,
      },
    ],
    explanation:
      "Case reports are valuable for noticing unusual events and generating hypotheses, particularly for rare harms. They usually cannot estimate incidence or prove causality because they lack a denominator, a comparison group, and control for alternative explanations.",
  },
  {
    id: "clinical-trials-l1-q51",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "A trial's primary outcome is a clinician-rated symptom score with borderline cases. Which design choices directly address observer bias?",
    options: [
      {
        text: "Keep outcome assessors unaware of treatment assignment when feasible.",
        isCorrect: true,
      },
      {
        text: "Use a pre-specified, standardized scoring rubric for assessments.",
        isCorrect: true,
      },
      {
        text: "Train assessors so similar patient presentations are scored consistently across sites.",
        isCorrect: true,
      },
      {
        text: "Let assessors review expected treatment mechanisms before rating each participant's outcome.",
        isCorrect: false,
      },
    ],
    explanation:
      "Observer bias is a major concern when outcomes involve judgment, especially near decision thresholds. Blinded assessment, standard definitions, and training reduce the chance that expectations about the treatment influence how outcomes are recorded.",
  },
  {
    id: "clinical-trials-l1-q52",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe the evidence hierarchy without treating it as a mechanical rule?",
    options: [
      {
        text: "Expert opinion and case reports can be useful for ideas, rare events, or early signals.",
        isCorrect: true,
      },
      {
        text: "Observational studies can be informative for rare outcomes, long-term harms, feasibility, and real-world patterns.",
        isCorrect: true,
      },
      {
        text: "Randomized controlled trials are especially strong for causal treatment comparisons when they are well designed and executed.",
        isCorrect: true,
      },
      {
        text: "Systematic reviews and meta-analyses depend on the quality, completeness, and similarity of the included studies.",
        isCorrect: true,
      },
    ],
    explanation:
      "The hierarchy is a guide to typical causal strength, not a substitute for judgment. Each evidence type has useful roles and limitations, and the value of any study still depends on execution, relevance, completeness, and the question being asked.",
  },
  {
    id: "clinical-trials-l1-q53",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "People who take vitamin supplements have lower mortality in a large observational dataset. Later work shows supplement users exercise more, smoke less, and have higher incomes. Which conclusions are justified?",
    options: [
      {
        text: "The supplement-mortality association may reflect differences between users and nonusers rather than a supplement effect.",
        isCorrect: true,
      },
      {
        text: "Adjustment for exercise, smoking, and income proves the supplement effect if the adjusted estimate remains favorable.",
        isCorrect: false,
      },
      {
        text: "The example illustrates why health behavior and access to care can create misleading treatment associations.",
        isCorrect: true,
      },
      {
        text: "The association is immune to confounding because mortality is an objective outcome.",
        isCorrect: false,
      },
    ],
    explanation:
      "Objective outcomes can still be confounded when exposure groups differ at baseline. Measured adjustment may help, but residual and unmeasured confounding can remain, so the observational association should not be treated as definitive causal evidence.",
  },
  {
    id: "clinical-trials-l1-q54",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "A researcher wants to know whether Drug X caused one patient's recovery. Which statements correctly connect the individual causal question to trial design?",
    options: [
      {
        text: "The exact untreated outcome for that same patient is unavailable once the patient receives Drug X.",
        isCorrect: true,
      },
      {
        text: "A randomized control group estimates what would happen among comparable patients who did not receive Drug X.",
        isCorrect: true,
      },
      {
        text: "The trial answer is probabilistic and group-based rather than a direct view of the individual's alternate history.",
        isCorrect: true,
      },
      {
        text: "If the patient's recovery is dramatic, the counterfactual question no longer matters.",
        isCorrect: false,
      },
    ],
    explanation:
      "Clinical trials address the unobservable individual counterfactual by creating comparison groups. Even strong trial evidence usually estimates average causal effects and uncertainty, so a dramatic single-patient recovery still needs context.",
  },
  {
    id: "clinical-trials-l1-q55",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which responsibilities reflect the ethical side of trustworthy clinical research?",
    options: [
      {
        text: "Participants should receive understandable information about risks, benefits, alternatives, and voluntary participation.",
        isCorrect: true,
      },
      {
        text: "Independent review should examine whether risks are justified and participant protections are adequate.",
        isCorrect: true,
      },
      {
        text: "Safety monitoring should continue because learning from humans creates obligations during the study, not just at approval.",
        isCorrect: true,
      },
      {
        text: "Patient burden and patient-centered outcomes matter because participants are partners in generating evidence, not merely data sources.",
        isCorrect: true,
      },
    ],
    explanation:
      "Ethics is not separate from evidence quality; people accept risk and burden so medicine can learn. Informed consent, independent oversight, safety monitoring, and patient-centered outcomes protect participants and improve the relevance of the research.",
  },
  {
    id: "clinical-trials-l1-q56",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "A systematic review finds a pooled benefit, but most included studies are small, unblinded, and funded by product sponsors. Which interpretations are most defensible?",
    options: [
      {
        text: "The pooled estimate should be interpreted alongside risk of bias, study size, blinding, funding, and consistency.",
        isCorrect: true,
      },
      {
        text: "Combining studies removes the need to evaluate the credibility of the individual studies.",
        isCorrect: false,
      },
      {
        text: "If small studies are systematically favorable, the pooled result may be less trustworthy than its numerical precision suggests.",
        isCorrect: true,
      },
      {
        text: "Sponsorship automatically invalidates each study regardless of methods or transparency.",
        isCorrect: false,
      },
    ],
    explanation:
      "Evidence synthesis is strongest when the underlying studies are credible and complete. Small, biased, or selectively reported studies can produce a precise-looking pooled estimate that still overstates benefit, while sponsorship is a risk factor to evaluate rather than an automatic disqualifier.",
  },
  {
    id: "clinical-trials-l1-q57",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which examples reflect a patient-centered view of clinical research?",
    options: [
      {
        text: "Selecting outcomes that capture how patients feel, function, or survive.",
        isCorrect: true,
      },
      {
        text: "Considering visit burden, travel, time, and risk when designing participation.",
        isCorrect: true,
      },
      {
        text: "Including patient-reported outcomes when symptoms or quality of life are central to the condition.",
        isCorrect: true,
      },
      {
        text: "Treating participants as passive sources of data after consent is signed.",
        isCorrect: false,
      },
    ],
    explanation:
      "Modern clinical research increasingly treats patients as people whose outcomes, burden, preferences, and safety matter throughout the trial. Patient-reported outcomes and practical participation burden can be central to whether evidence is meaningful.",
  },
  {
    id: "clinical-trials-l1-q58",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "A promising treatment idea is moving toward a first controlled trial. Which activities connect the scientific hypothesis to credible human evidence?",
    options: [
      {
        text: "Define the population, intervention, comparator, outcomes, and time frame before evaluating results.",
        isCorrect: true,
      },
      {
        text: "Choose design features such as randomization, blinding, and control groups to reduce bias where feasible.",
        isCorrect: true,
      },
      {
        text: "Collect safety data and obtain ethics review because patients face real risk during evidence generation.",
        isCorrect: true,
      },
      {
        text: "Interpret results in context rather than treating a favorable before-after pattern as a completed causal answer.",
        isCorrect: true,
      },
    ],
    explanation:
      "A trial converts a plausible idea into interpretable evidence by specifying the question and controlling major sources of bias. Ethical review, safety collection, and contextual interpretation are part of the same evidence system because human participants are involved.",
  },
  {
    id: "clinical-trials-l1-q59",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "When interpreting real-world clinical evidence, which uses of observational studies fit their strengths while respecting their causal limitations?",
    options: [
      {
        text: "Studying rare harms or long-term outcomes that would be difficult to capture in a pre-approval randomized trial.",
        isCorrect: true,
      },
      {
        text: "Estimating routine-care patterns, adherence, and outcomes in broader patient populations.",
        isCorrect: true,
      },
      {
        text: "Replacing randomized evidence for any treatment benefit claim whenever the observational sample is large.",
        isCorrect: false,
      },
      {
        text: "Avoiding concerns about confounding because real-world clinicians choose treatments for practical reasons.",
        isCorrect: false,
      },
    ],
    explanation:
      "Observational evidence can be extremely useful for safety, long-term outcomes, and real-world practice patterns. Its major limitation is that treatment choice is not assigned by chance, so confounding and selection remain central threats when making causal claims.",
  },
  {
    id: "clinical-trials-l1-q60",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "A clinician reports that a new treatment helped 18 of 20 patients in routine practice. Which next step most directly tests whether the treatment caused the improvement?",
    options: [
      {
        text: "Run a study with a comparable control group and assignment process that reduces baseline differences between groups.",
        isCorrect: true,
      },
      {
        text: "Collect more testimonials from patients who improved so the pattern becomes clearer.",
        isCorrect: false,
      },
      {
        text: "Ask expert clinicians whether the mechanism sounds plausible enough to explain the recoveries.",
        isCorrect: false,
      },
      {
        text: "Compare patients' symptoms after treatment with their symptoms at their worst pre-treatment visit.",
        isCorrect: false,
      },
    ],
    explanation:
      "The decisive missing piece is a credible comparison, ideally with assignment that reduces systematic baseline differences. Testimonials, mechanism, and before-after comparisons can suggest hypotheses, but they do not separate treatment effect from natural recovery, regression to the mean, placebo effects, or selection.",
  },
];
