import { Question } from "../../../quiz";

export const ClinicalTrialsLecture1Questions: Question[] = [
  {
    id: "clinical-trials-l1-q01",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statements correctly explain why clinical trials are needed in medicine?",
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
        text: "The observed improvement proves the pill is better than placebo.",
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
      "Which statements correctly distinguish correlation from causation?",
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
    prompt: "Which statements correctly describe confounding?",
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
    prompt: "Which statements correctly describe placebo effects?",
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
        text: "A placebo effect means a patient must be lying about improvement.",
        isCorrect: false,
      },
      {
        text: "Placebo effects make comparison groups unnecessary.",
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
    prompt: "Which statement best captures natural history of disease?",
    options: [
      {
        text: "Some conditions improve, worsen, or fluctuate without any intervention.",
        isCorrect: true,
      },
      {
        text: "Every untreated illness steadily worsens at the same rate.",
        isCorrect: false,
      },
      {
        text: "Natural history removes the need for comparison groups.",
        isCorrect: false,
      },
      {
        text: "Natural history proves that symptomatic improvement is caused by treatment.",
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
    prompt: "Which statements correctly describe regression to the mean?",
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
    prompt: "Which statements correctly describe selection bias?",
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
    prompt: "Which statements correctly describe observer bias?",
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
    prompt: "Which statements correctly describe publication bias?",
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
        text: "To make every participant receive the experimental treatment.",
        isCorrect: false,
      },
      {
        text: "To guarantee that no patient can improve naturally.",
        isCorrect: false,
      },
      {
        text: "To replace the need for outcome measurement.",
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
      "Which statement best states the fundamental problem of causal inference in treatment research?",
    options: [
      {
        text: "The same patient cannot be observed both receiving and not receiving the treatment at the same time.",
        isCorrect: true,
      },
      {
        text: "Researchers cannot measure any outcomes after treatment.",
        isCorrect: false,
      },
      {
        text: "Treatments only work when no comparison group exists.",
        isCorrect: false,
      },
      {
        text: "Causality is solved by collecting one anecdote.",
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
      "Which stakeholders commonly participate in the modern clinical research ecosystem?",
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
        text: "A plausible theory guarantees that a treatment benefits patients.",
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
        text: "Observational studies are always slower and less practical for rare outcomes.",
        isCorrect: false,
      },
      {
        text: "Randomized controlled trials are generally stronger for causal inference about interventions.",
        isCorrect: true,
      },
      {
        text: "Observational studies are automatically invalid and never useful.",
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
        text: "They synthesize evidence from only one study.",
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
        text: "It guarantees every measured baseline variable is exactly identical in every group.",
        isCorrect: false,
      },
      {
        text: "It prevents all missing data and protocol deviations.",
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
        text: "The observed association cannot be confounded by healthier behaviors or socioeconomic differences.",
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
        text: "Personally providing all routine care at every study site.",
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
      "Which statements correctly describe Contract Research Organizations (CROs)?",
    options: [
      {
        text: "They can conduct trial activities on behalf of sponsors.",
        isCorrect: true,
      },
      {
        text: "They approve therapies for marketing on behalf of public regulators.",
        isCorrect: false,
      },
      {
        text: "They are used because sponsors often outsource substantial trial operations.",
        isCorrect: true,
      },
      {
        text: "They replace the need for patient consent and ethical oversight.",
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
      "Which responsibilities are commonly associated with investigators and clinical sites?",
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
        text: "Serving as the only authority that approves therapies for marketing.",
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
      "Which statements correctly describe regulators such as the U.S. Food and Drug Administration (FDA), European Medicines Agency (EMA), or Swissmedic?",
    options: [
      {
        text: "They evaluate evidence submitted for therapies or medical products.",
        isCorrect: true,
      },
      {
        text: "They ignore patient safety when making approval decisions.",
        isCorrect: false,
      },
      {
        text: "They cannot influence what evidence is required before marketing.",
        isCorrect: false,
      },
      {
        text: "They are the same organizations as Contract Research Organizations.",
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
      "Which statements correctly describe ethics committees or Institutional Review Boards (IRBs)?",
    options: [
      {
        text: "They provide independent ethical review of human-subject research.",
        isCorrect: true,
      },
      {
        text: "They write sponsor marketing copy for approved products.",
        isCorrect: false,
      },
      {
        text: "They guarantee that a medical product will be commercially successful.",
        isCorrect: false,
      },
      {
        text: "They exist to guarantee that every trial result is positive.",
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
        text: "Patients' consent is irrelevant when a sponsor wants faster evidence.",
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
      { text: "A longer brand name for the treatment.", isCorrect: false },
      { text: "A larger marketing budget.", isCorrect: false },
      { text: "A more enthusiastic investigator biography.", isCorrect: false },
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
        text: "Every randomized trial is automatically more informative than every observational study.",
        isCorrect: false,
      },
      {
        text: "Observational designs never contribute useful evidence.",
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
        text: "The association proves Drug A caused every death in the treated group.",
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
        text: "Blinding is unnecessary whenever outcomes are even partly subjective.",
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
        text: "Every published study is designed with no bias.",
        isCorrect: false,
      },
      {
        text: "Meta-analysis prevents missing studies from mattering.",
        isCorrect: false,
      },
      {
        text: "Publication decisions are unrelated to the direction of results.",
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
        text: "Being high in the hierarchy guarantees that a conclusion is true in every patient.",
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
        text: "Training as a clinician removes all cognitive bias.",
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
        text: "The finding proves the treatment should be approved for all patients.",
        isCorrect: false,
      },
      {
        text: "The absence of a control group strengthens causal inference.",
        isCorrect: false,
      },
      {
        text: "Rare conditions cannot be affected by selection bias or natural history.",
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
        text: "Operational coordination is irrelevant once a scientific hypothesis exists.",
        isCorrect: false,
      },
    ],
    explanation:
      "Clinical research is both a scientific and operational system. Trustworthy trials require a strong causal design, ethical oversight, patient-centered conduct, and coordinated work across organizations that fund, run, monitor, regulate, and participate in the research.",
  },
];
