import { Question } from "../../../quiz";

export const BiologyChemistryForLifeScienceLecture5Questions: Question[] = [
  {
    id: "bio-chem-life-l5-q01",
    chapter: 5,
    difficulty: "easy",
    prompt:
      "Which statements correctly explain why clinical trials are necessary?",
    options: [
      {
        text: "Humans are noisy and heterogeneous biological systems.",
        isCorrect: true,
      },
      { text: "Observation alone can be misleading.", isCorrect: true },
      {
        text: "Promising mechanisms can fail in real patients.",
        isCorrect: true,
      },
      {
        text: "Controlled evidence is needed to know whether an intervention works.",
        isCorrect: true,
      },
    ],
    explanation:
      "Patients improve, worsen, and vary for many reasons unrelated to an intervention. Clinical trials are structured attempts to reduce uncertainty and distinguish true effects from misleading observations.",
  },
  {
    id: "bio-chem-life-l5-q02",
    chapter: 5,
    difficulty: "easy",
    prompt: "Which statements correctly describe placebo effects?",
    options: [
      {
        text: "Placebo effects can produce real physiological or behavioral changes.",
        isCorrect: true,
      },
      {
        text: "Patient expectations can influence outcomes such as pain.",
        isCorrect: true,
      },
      {
        text: "Improvement after treatment does not prove the treatment caused the improvement.",
        isCorrect: true,
      },
      {
        text: "Placebo effects mean patients are always lying about symptoms.",
        isCorrect: false,
      },
    ],
    explanation:
      "Placebo effects are real changes related to expectation, context, and behavior. They make uncontrolled observation difficult because improvement after treatment may not be caused by the treatment's active mechanism.",
  },
  {
    id: "bio-chem-life-l5-q03",
    chapter: 5,
    difficulty: "easy",
    prompt: "Which statements correctly describe confounding?",
    options: [
      {
        text: "A hidden factor can create a misleading association.",
        isCorrect: true,
      },
      {
        text: "Vitamin users living longer might reflect exercise, diet, income, or other differences rather than vitamins themselves.",
        isCorrect: true,
      },
      {
        text: "Confounding proves every association is causal.",
        isCorrect: false,
      },
      {
        text: "Confounding is irrelevant to observational medical studies.",
        isCorrect: false,
      },
    ],
    explanation:
      "Confounding occurs when another factor explains an apparent relationship between exposure and outcome. It is one reason correlation does not automatically imply causation.",
  },
  {
    id: "bio-chem-life-l5-q04",
    chapter: 5,
    difficulty: "easy",
    prompt: "Which statement best describes regression to the mean?",
    options: [
      {
        text: "Extreme symptoms or measurements often naturally move closer to typical levels later, even without effective treatment.",
        isCorrect: true,
      },
      {
        text: "Regression to the mean proves every treatment works.",
        isCorrect: false,
      },
      {
        text: "Regression to the mean is the same thing as randomization.",
        isCorrect: false,
      },
      {
        text: "Regression to the mean only affects computer hardware, not medicine.",
        isCorrect: false,
      },
    ],
    explanation:
      "Patients often seek care or enroll when symptoms are unusually severe, and later measurements may improve naturally. Without a control group, that natural movement can be mistaken for treatment efficacy.",
  },
  {
    id: "bio-chem-life-l5-q05",
    chapter: 5,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe common sources of bias in research?",
    options: [
      {
        text: "Selection bias can occur when participants differ systematically.",
        isCorrect: true,
      },
      {
        text: "Observer bias can occur when researchers influence or interpret measurements differently.",
        isCorrect: true,
      },
      {
        text: "Publication bias can occur when positive results are more likely to appear in the literature.",
        isCorrect: true,
      },
      {
        text: "Systematic error can distort evidence even when sample size is large.",
        isCorrect: true,
      },
    ],
    explanation:
      "Medical research must fight both randomness and systematic error. Bias can enter through who is studied, how outcomes are measured, and which results become visible.",
  },
  {
    id: "bio-chem-life-l5-q06",
    chapter: 5,
    difficulty: "easy",
    prompt: "Which statements correctly describe randomization?",
    options: [
      {
        text: "Randomization assigns participants to groups by chance.",
        isCorrect: true,
      },
      {
        text: "Randomization helps balance known and unknown factors across groups.",
        isCorrect: true,
      },
      { text: "Randomization supports causal inference.", isCorrect: true },
      {
        text: "Randomization means the researchers choose the healthiest patients for the treatment group.",
        isCorrect: false,
      },
    ],
    explanation:
      "Randomization makes groups more comparable by reducing systematic differences at baseline. It is powerful because it can balance factors researchers know about and factors they do not know about.",
  },
  {
    id: "bio-chem-life-l5-q07",
    chapter: 5,
    difficulty: "easy",
    prompt: "Which statements correctly describe control groups?",
    options: [
      {
        text: "A control group answers the question 'compared to what?'",
        isCorrect: true,
      },
      {
        text: "Controls can include placebo, standard of care, alternative treatment, or no treatment.",
        isCorrect: true,
      },
      {
        text: "Control groups are unnecessary because every patient changes only because of treatment.",
        isCorrect: false,
      },
      {
        text: "Control groups prevent any comparison from being made.",
        isCorrect: false,
      },
    ],
    explanation:
      "Patients change over time for many reasons, so a comparison group is essential. Control groups help estimate what would have happened without the tested intervention or with another relevant option.",
  },
  {
    id: "bio-chem-life-l5-q08",
    chapter: 5,
    difficulty: "easy",
    prompt: "Which statement best describes blinding?",
    options: [
      {
        text: "Blinding keeps participants, researchers, or analysts unaware of group assignments to reduce expectation and measurement bias.",
        isCorrect: true,
      },
      {
        text: "Blinding means removing all outcome measurements.",
        isCorrect: false,
      },
      { text: "Blinding makes randomization impossible.", isCorrect: false },
      { text: "Blinding is only a type of drug metabolism.", isCorrect: false },
    ],
    explanation:
      "Blinding reduces bias from expectations and behavior changes. It is compatible with randomization and measurement, and it is a trial-design feature rather than a pharmacokinetic process.",
  },
  {
    id: "bio-chem-life-l5-q09",
    chapter: 5,
    difficulty: "easy",
    prompt: "Which statements correctly describe clinical endpoints?",
    options: [
      {
        text: "An endpoint is an outcome used to evaluate a treatment.",
        isCorrect: true,
      },
      {
        text: "Survival, pain reduction, blood pressure, and disease progression can be endpoints.",
        isCorrect: true,
      },
      {
        text: "A surrogate endpoint may not guarantee improved patient outcomes.",
        isCorrect: true,
      },
      {
        text: "A clinical endpoint directly measures something meaningful for the patient, such as survival or symptoms.",
        isCorrect: true,
      },
    ],
    explanation:
      "Endpoints define what the trial is trying to measure. Surrogates such as biomarker changes can be useful, but they do not always translate into outcomes patients care about.",
  },
  {
    id: "bio-chem-life-l5-q10",
    chapter: 5,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe inclusion and exclusion criteria?",
    options: [
      { text: "They define who can enter a trial.", isCorrect: true },
      {
        text: "They can include age, disease severity, comorbidities, or prior treatment rules.",
        isCorrect: true,
      },
      {
        text: "More restrictive criteria can improve control but reduce generalizability.",
        isCorrect: true,
      },
      {
        text: "They are unrelated to trial validity or interpretation.",
        isCorrect: false,
      },
    ],
    explanation:
      "Eligibility criteria shape the study population. They affect how trustworthy the causal conclusion is and how well results generalize to real-world patients.",
  },
  {
    id: "bio-chem-life-l5-q11",
    chapter: 5,
    difficulty: "easy",
    prompt:
      "Which statements correctly distinguish internal and external validity?",
    options: [
      {
        text: "Internal validity asks whether the causal conclusion can be trusted.",
        isCorrect: true,
      },
      {
        text: "External validity asks whether results generalize to real-world patients.",
        isCorrect: true,
      },
      {
        text: "A perfectly controlled study can never have limited generalizability.",
        isCorrect: false,
      },
      {
        text: "Internal validity and external validity are always identical.",
        isCorrect: false,
      },
    ],
    explanation:
      "Internal validity concerns causal trustworthiness, while external validity concerns generalization. A highly controlled trial can be internally strong but less representative of broader clinical practice.",
  },
  {
    id: "bio-chem-life-l5-q12",
    chapter: 5,
    difficulty: "easy",
    prompt: "Which statement best describes Phase I clinical trials?",
    options: [
      {
        text: "Phase I primarily asks whether humans can tolerate the intervention safely.",
        isCorrect: true,
      },
      {
        text: "Phase I is usually the definitive proof of broad real-world effectiveness.",
        isCorrect: false,
      },
      {
        text: "Phase I always includes millions of participants.",
        isCorrect: false,
      },
      {
        text: "Phase I occurs only after the drug is already approved for routine use.",
        isCorrect: false,
      },
    ],
    explanation:
      "Phase I trials are early human studies focused mainly on safety, tolerability, and dosing. They are not usually large definitive effectiveness trials or post-approval monitoring.",
  },
  {
    id: "bio-chem-life-l5-q13",
    chapter: 5,
    difficulty: "easy",
    prompt: "Which statements correctly describe clinical trial phases?",
    options: [
      { text: "Phase I emphasizes safety and tolerability.", isCorrect: true },
      {
        text: "Phase II looks for early evidence of efficacy.",
        isCorrect: true,
      },
      {
        text: "Phase III aims to provide stronger evidence in larger populations.",
        isCorrect: true,
      },
      {
        text: "Phase IV monitors what happens after approval in real-world use.",
        isCorrect: true,
      },
    ],
    explanation:
      "The phases move from early safety through efficacy testing, larger confirmatory studies, and post-approval monitoring. The exact details vary, but the basic progression reflects increasing evidence and exposure.",
  },
  {
    id: "bio-chem-life-l5-q14",
    chapter: 5,
    difficulty: "medium",
    prompt: "Which statements correctly describe effect size?",
    options: [
      {
        text: "Effect size asks how much a treatment helps or harms.",
        isCorrect: true,
      },
      {
        text: "A mortality reduction from 10% to 5% is larger than one from 10% to 9%.",
        isCorrect: true,
      },
      {
        text: "Statistical significance alone does not tell whether the effect is large enough to matter clinically.",
        isCorrect: true,
      },
      {
        text: "Effect size is irrelevant if a p-value is below 0.05.",
        isCorrect: false,
      },
    ],
    explanation:
      "Effect size concerns magnitude, not just whether an estimate is unlikely under a null model. Large studies can make tiny effects statistically significant, so clinical meaning still matters.",
  },
  {
    id: "bio-chem-life-l5-q15",
    chapter: 5,
    difficulty: "medium",
    prompt:
      "Which statements correctly distinguish statistical and clinical significance?",
    options: [
      {
        text: "Statistical significance can occur for very small effects in huge datasets.",
        isCorrect: true,
      },
      {
        text: "Clinical significance asks whether the effect matters for patients or decisions.",
        isCorrect: true,
      },
      {
        text: "A statistically significant 0.5 mmHg blood-pressure change is automatically clinically meaningful.",
        isCorrect: false,
      },
      {
        text: "Statistical significance automatically proves a treatment is worth using.",
        isCorrect: false,
      },
    ],
    explanation:
      "Statistical significance and clinical importance answer different questions. A tiny effect can be statistically reliable but too small to matter for real patient care.",
  },
  {
    id: "bio-chem-life-l5-q16",
    chapter: 5,
    difficulty: "medium",
    prompt: "Which statement best describes survival analysis?",
    options: [
      {
        text: "Survival analysis studies time until an event such as death, relapse, or disease progression.",
        isCorrect: true,
      },
      {
        text: "Survival analysis only counts the number of hospital buildings.",
        isCorrect: false,
      },
      {
        text: "Survival analysis assumes no participant ever leaves a study.",
        isCorrect: false,
      },
      {
        text: "Survival analysis is unrelated to time-to-event outcomes.",
        isCorrect: false,
      },
    ],
    explanation:
      "Survival analysis handles time-to-event questions, which are common in medicine. It also deals with complications such as censoring when participants leave or do not experience the event during follow-up.",
  },
  {
    id: "bio-chem-life-l5-q17",
    chapter: 5,
    difficulty: "medium",
    prompt: "Which statements correctly describe hazard ratios and censoring?",
    options: [
      {
        text: "A hazard ratio compares event rates over time between groups.",
        isCorrect: true,
      },
      {
        text: "A hazard ratio of 0.8 can be interpreted roughly as a lower event rate in the treatment group.",
        isCorrect: true,
      },
      {
        text: "Censoring can occur when a participant leaves the study or has not had the event by study end.",
        isCorrect: true,
      },
      {
        text: "Time-to-event data require methods that can handle incomplete follow-up.",
        isCorrect: true,
      },
    ],
    explanation:
      "Hazard ratios summarize relative event rates over time, and censoring captures incomplete event observation. These features make survival analysis different from simple before-after comparisons.",
  },
  {
    id: "bio-chem-life-l5-q18",
    chapter: 5,
    difficulty: "medium",
    prompt: "Which statements correctly describe sensitivity and specificity?",
    options: [
      {
        text: "Sensitivity measures how well a test detects disease when disease is present.",
        isCorrect: true,
      },
      {
        text: "Specificity measures how well a test avoids false positives when disease is absent.",
        isCorrect: true,
      },
      {
        text: "Cancer screening can involve tradeoffs between missed disease and overdiagnosis.",
        isCorrect: true,
      },
      {
        text: "Sensitivity and specificity are always both 100% for any useful test.",
        isCorrect: false,
      },
    ],
    explanation:
      "Diagnostic tests involve tradeoffs between detecting true disease and avoiding false positives. Useful tests are rarely perfect, so sensitivity and specificity must be interpreted together.",
  },
  {
    id: "bio-chem-life-l5-q19",
    chapter: 5,
    difficulty: "medium",
    prompt: "Which statements correctly describe ROC curves?",
    options: [
      {
        text: "Receiver Operating Characteristic (ROC) curves show a tradeoff between sensitivity and specificity-related false-positive rates.",
        isCorrect: true,
      },
      {
        text: "Changing a diagnostic threshold can change sensitivity and specificity.",
        isCorrect: true,
      },
      {
        text: "ROC curves prove every test is clinically useful.",
        isCorrect: false,
      },
      {
        text: "ROC curves are unrelated to diagnostic decision thresholds.",
        isCorrect: false,
      },
    ],
    explanation:
      "ROC curves summarize how test behavior changes as a decision threshold changes. They are useful for evaluation, but they do not by themselves prove a test is clinically appropriate.",
  },
  {
    id: "bio-chem-life-l5-q20",
    chapter: 5,
    difficulty: "medium",
    prompt:
      "Which statement best describes the translational research pipeline?",
    options: [
      {
        text: "A biomedical idea can move from discovery to preclinical testing, clinical trials, approval, and clinical practice.",
        isCorrect: true,
      },
      {
        text: "A discovery becomes a treatment without testing safety or efficacy.",
        isCorrect: false,
      },
      {
        text: "Preclinical research is identical to Phase IV monitoring.",
        isCorrect: false,
      },
      {
        text: "Clinical trials occur before any target or mechanism is considered.",
        isCorrect: false,
      },
    ],
    explanation:
      "Translational research tries to move scientific ideas into patient care through staged evidence generation. Each stage tests different aspects such as mechanism, toxicity, efficacy, and real-world use.",
  },
  {
    id: "bio-chem-life-l5-q21",
    chapter: 5,
    difficulty: "medium",
    prompt: "Which statements correctly describe preclinical research?",
    options: [
      {
        text: "It can include cell, organoid, and animal studies.",
        isCorrect: true,
      },
      {
        text: "It can test mechanism, toxicity, and feasibility.",
        isCorrect: true,
      },
      {
        text: "Animal models often fail to predict humans perfectly.",
        isCorrect: true,
      },
      {
        text: "Preclinical success does not guarantee clinical success.",
        isCorrect: true,
      },
    ],
    explanation:
      "Preclinical work is essential but limited. Biological systems differ across models and humans, so many promising interventions still fail later.",
  },
  {
    id: "bio-chem-life-l5-q22",
    chapter: 5,
    difficulty: "medium",
    prompt: "Which statements correctly explain why many drug candidates fail?",
    options: [
      { text: "The biology may be misunderstood.", isCorrect: true },
      { text: "The target may be wrong.", isCorrect: true },
      {
        text: "Toxicity or insufficient efficacy can stop development.",
        isCorrect: true,
      },
      {
        text: "Failure is impossible once a mechanism sounds plausible.",
        isCorrect: false,
      },
    ],
    explanation:
      "Drug development is risky because biological systems are complex and models are imperfect. A plausible mechanism is not enough if the intervention is unsafe, ineffective, or aimed at the wrong target.",
  },
  {
    id: "bio-chem-life-l5-q23",
    chapter: 5,
    difficulty: "medium",
    prompt: "Which statements correctly describe the reproducibility crisis?",
    options: [
      { text: "Some published findings fail to replicate.", isCorrect: true },
      { text: "Small samples can overestimate effects.", isCorrect: true },
      {
        text: "Evidence quality does not matter for biomedical progress.",
        isCorrect: false,
      },
      {
        text: "Reproducibility concerns prove all research is false.",
        isCorrect: false,
      },
    ],
    explanation:
      "The reproducibility crisis highlights that not every published result is equally reliable. Evidence quality matters, but reproducibility problems do not imply that all research is worthless.",
  },
  {
    id: "bio-chem-life-l5-q24",
    chapter: 5,
    difficulty: "medium",
    prompt: "Which statement best describes foundation models for biology?",
    options: [
      {
        text: "They aim to learn useful representations of biological systems such as proteins, genomes, or multimodal data.",
        isCorrect: true,
      },
      {
        text: "They eliminate the need for biological measurements.",
        isCorrect: false,
      },
      {
        text: "They prove clinical evidence is unnecessary.",
        isCorrect: false,
      },
      {
        text: "They can only process handwritten paper charts.",
        isCorrect: false,
      },
    ],
    explanation:
      "Foundation models for biology learn representations from large biological datasets. They can accelerate research, but they still depend on reliable measurements and downstream validation.",
  },
  {
    id: "bio-chem-life-l5-q25",
    chapter: 5,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe protein models and AI in biology?",
    options: [
      { text: "AI can support protein structure prediction.", isCorrect: true },
      {
        text: "AI can support protein design and enzyme engineering.",
        isCorrect: true,
      },
      {
        text: "Computational protein design can accelerate biological engineering.",
        isCorrect: true,
      },
      {
        text: "Protein models connect molecular sequence, structure, and function.",
        isCorrect: true,
      },
    ],
    explanation:
      "AI protein models connect sequence, structure, and function to support prediction and design. They can accelerate tasks such as structure prediction, protein design, and enzyme engineering.",
  },
  {
    id: "bio-chem-life-l5-q26",
    chapter: 5,
    difficulty: "medium",
    prompt: "Which statements correctly describe multimodal medicine?",
    options: [
      {
        text: "Medicine produces genomics, imaging, labs, notes, and wearable data.",
        isCorrect: true,
      },
      {
        text: "AI can help integrate different medical data types.",
        isCorrect: true,
      },
      {
        text: "Multimodal data can support prediction and stratification.",
        isCorrect: true,
      },
      {
        text: "Only one data type is ever relevant to clinical reasoning.",
        isCorrect: false,
      },
    ],
    explanation:
      "Modern medicine produces many data streams that each capture part of patient state. AI can help combine these modalities, though integration must still be validated clinically.",
  },
  {
    id: "bio-chem-life-l5-q27",
    chapter: 5,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe digital pathology and clinical agents?",
    options: [
      {
        text: "Digital pathology can use computer vision on histology slides.",
        isCorrect: true,
      },
      {
        text: "Clinical agents may help with documentation, patient interaction, protocol design, or trial management.",
        isCorrect: true,
      },
      {
        text: "Digital pathology can never support disease detection or classification.",
        isCorrect: false,
      },
      {
        text: "Clinical agents remove the need for clinical oversight.",
        isCorrect: false,
      },
    ],
    explanation:
      "Digital pathology applies computational vision to tissue images, and clinical agents may support parts of clinical workflow. These tools can assist but still require oversight, validation, and careful integration.",
  },
  {
    id: "bio-chem-life-l5-q28",
    chapter: 5,
    difficulty: "hard",
    prompt:
      "Which statement best explains the phrase 'controlled uncertainty reduction' in medicine?",
    options: [
      {
        text: "Medical evidence is generated by designing studies that reduce bias, confounding, randomness, and uncertainty about causal effects.",
        isCorrect: true,
      },
      {
        text: "Medicine becomes certain whenever a mechanism sounds plausible.",
        isCorrect: false,
      },
      {
        text: "Clinical trials are designed to increase confounding.",
        isCorrect: false,
      },
      {
        text: "Uncertainty reduction means ignoring patient outcomes.",
        isCorrect: false,
      },
    ],
    explanation:
      "Clinical research uses design and statistics to make uncertain claims more trustworthy. The goal is not perfect certainty but better decisions based on controlled evidence.",
  },
  {
    id: "bio-chem-life-l5-q29",
    chapter: 5,
    difficulty: "hard",
    prompt:
      "Which statements correctly explain why observational studies can mislead?",
    options: [
      {
        text: "Patients who choose a treatment can differ from those who do not.",
        isCorrect: true,
      },
      {
        text: "Symptoms can improve naturally after unusually bad periods.",
        isCorrect: true,
      },
      {
        text: "Researchers may measure or publish results selectively.",
        isCorrect: true,
      },
      {
        text: "Unmeasured variables can create false associations.",
        isCorrect: true,
      },
    ],
    explanation:
      "Observational evidence can be valuable, but it is vulnerable to confounding, regression to the mean, selection effects, and bias. Trial design tries to control these threats when causal claims matter.",
  },
  {
    id: "bio-chem-life-l5-q30",
    chapter: 5,
    difficulty: "hard",
    prompt:
      "Which statements correctly connect trial design to causal inference?",
    options: [
      { text: "Randomization helps make groups comparable.", isCorrect: true },
      {
        text: "Control groups estimate what happens without the tested intervention or under an alternative.",
        isCorrect: true,
      },
      {
        text: "Blinding reduces expectation and measurement bias.",
        isCorrect: true,
      },
      {
        text: "Endpoints do not define what counts as evidence of benefit or harm.",
        isCorrect: false,
      },
    ],
    explanation:
      "Randomization, controls, blinding, and endpoints are all tools for making causal claims more credible. Endpoints define what counts as measured evidence, so denying that role is incorrect.",
  },
  {
    id: "bio-chem-life-l5-q31",
    chapter: 5,
    difficulty: "hard",
    prompt: "Which statements correctly describe surrogate endpoints?",
    options: [
      {
        text: "They can be easier or faster to measure than clinical outcomes.",
        isCorrect: true,
      },
      { text: "They can be useful when validated well.", isCorrect: true },
      {
        text: "They automatically guarantee better patient survival or quality of life.",
        isCorrect: false,
      },
      {
        text: "They are always superior to clinical endpoints.",
        isCorrect: false,
      },
    ],
    explanation:
      "Surrogate endpoints can speed research, but they must be interpreted carefully. A biomarker improvement does not automatically mean patients live longer or feel better.",
  },
  {
    id: "bio-chem-life-l5-q32",
    chapter: 5,
    difficulty: "hard",
    prompt:
      "Which statement best explains why internal and external validity can trade off?",
    options: [
      {
        text: "Restrictive trial conditions can improve causal control while making the study population less like real-world patients.",
        isCorrect: true,
      },
      {
        text: "A study can only have external validity if it has no comparison group.",
        isCorrect: false,
      },
      {
        text: "Internal validity means a trial must ignore causal inference.",
        isCorrect: false,
      },
      {
        text: "External validity means the results apply only to the exact enrolled patients and nobody else.",
        isCorrect: false,
      },
    ],
    explanation:
      "Tight control can reduce confounding and improve internal validity, but it may exclude patients seen in practice. Broad inclusion can improve generalizability but may make causal interpretation harder.",
  },
  {
    id: "bio-chem-life-l5-q33",
    chapter: 5,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe statistical reasoning in medicine?",
    options: [
      {
        text: "Medicine often requires decision-making under uncertainty.",
        isCorrect: true,
      },
      {
        text: "False positives and false negatives can both be costly.",
        isCorrect: true,
      },
      {
        text: "Large datasets can make small differences statistically significant.",
        isCorrect: true,
      },
      {
        text: "Risk-benefit interpretation matters alongside p-values.",
        isCorrect: true,
      },
    ],
    explanation:
      "Medical statistics is about uncertainty, risk, and decisions, not just formulas. The same p-value can have different practical meaning depending on effect size, harms, alternatives, and patient context.",
  },
  {
    id: "bio-chem-life-l5-q34",
    chapter: 5,
    difficulty: "hard",
    prompt:
      "Which statements correctly identify evidence-generation bottlenecks in medicine?",
    options: [
      { text: "Reliable measurement can be a bottleneck.", isCorrect: true },
      {
        text: "Data quality and clinical workflows can be bottlenecks.",
        isCorrect: true,
      },
      {
        text: "Patient recruitment and trial execution can be bottlenecks.",
        isCorrect: true,
      },
      {
        text: "A better model alone automatically solves regulation, evidence, and workflow problems.",
        isCorrect: false,
      },
    ],
    explanation:
      "AI models can help, but biomedical progress often depends on measurement, workflow, recruitment, regulation, and evidence generation. Better models do not automatically fix weak data or poor trial execution.",
  },
  {
    id: "bio-chem-life-l5-q35",
    chapter: 5,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe digital twins and simulation in medicine?",
    options: [
      {
        text: "A digital twin aims to represent a patient or biological system computationally.",
        isCorrect: true,
      },
      {
        text: "Simulation may help with treatment prediction or trial planning.",
        isCorrect: true,
      },
      {
        text: "Digital twins do not require validation against real evidence.",
        isCorrect: false,
      },
      {
        text: "A simulation automatically proves clinical efficacy.",
        isCorrect: false,
      },
    ],
    explanation:
      "Digital twins and simulations can support reasoning and planning, but they are models rather than proof. They must be validated against reliable data and clinical outcomes.",
  },
  {
    id: "bio-chem-life-l5-q36",
    chapter: 5,
    difficulty: "hard",
    prompt:
      "Which statement best explains why AI is often not the primary bottleneck in medicine?",
    options: [
      {
        text: "Model quality matters, but reliable data, biological understanding, workflows, regulation, and evidence generation often limit impact.",
        isCorrect: true,
      },
      {
        text: "AI has no possible role in biomedical research.",
        isCorrect: false,
      },
      {
        text: "Clinical evidence becomes unnecessary once an AI model is large enough.",
        isCorrect: false,
      },
      {
        text: "Biology is already perfectly measured and understood.",
        isCorrect: false,
      },
    ],
    explanation:
      "AI can accelerate biomedical work, but it depends on the surrounding evidence system. Poor measurements, weak workflows, missing validation, and regulatory constraints can block impact even when models are strong.",
  },
  {
    id: "bio-chem-life-l5-q37",
    chapter: 5,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe synthetic biology in the biomedical research landscape?",
    options: [
      {
        text: "Synthetic biology involves programming or engineering biological systems.",
        isCorrect: true,
      },
      {
        text: "It reflects convergence between biology and software-like design thinking.",
        isCorrect: true,
      },
      {
        text: "It can interact with AI-assisted design and biological measurement.",
        isCorrect: true,
      },
      {
        text: "It still requires validation because biological systems remain uncertain and context-dependent.",
        isCorrect: true,
      },
    ],
    explanation:
      "Synthetic biology treats biological systems as engineerable platforms, but biology remains uncertain and context-dependent. Engineering biology therefore still requires measurement, validation, and iteration.",
  },
  {
    id: "bio-chem-life-l5-q38",
    chapter: 5,
    difficulty: "hard",
    prompt:
      "Which statements correctly identify misconceptions about clinical evidence?",
    options: [
      {
        text: "A plausible mechanism does not prove clinical benefit.",
        isCorrect: true,
      },
      {
        text: "An observed improvement does not automatically prove causation.",
        isCorrect: true,
      },
      {
        text: "A statistically significant result is not automatically clinically important.",
        isCorrect: true,
      },
      {
        text: "A positive early study always guarantees approval and real-world benefit.",
        isCorrect: false,
      },
    ],
    explanation:
      "Clinical evidence requires careful inference from controlled data. Mechanisms, observations, statistical significance, and early studies can all be useful but insufficient by themselves.",
  },
  {
    id: "bio-chem-life-l5-q39",
    chapter: 5,
    difficulty: "hard",
    prompt: "Which statements correctly connect AI to evidence generation?",
    options: [
      {
        text: "AI can help design protocols or manage trials.",
        isCorrect: true,
      },
      {
        text: "AI can support patient stratification or recruitment.",
        isCorrect: true,
      },
      {
        text: "AI-generated hypotheses do not need empirical validation.",
        isCorrect: false,
      },
      {
        text: "AI eliminates the need for causal inference.",
        isCorrect: false,
      },
    ],
    explanation:
      "AI can support many parts of evidence generation, but hypotheses and predictions still need validation. Causal inference remains necessary when deciding whether an intervention works.",
  },
  {
    id: "bio-chem-life-l5-q40",
    chapter: 5,
    difficulty: "hard",
    prompt:
      "Which statement best summarizes clinical trials and modern biomedical research?",
    options: [
      {
        text: "Medicine advances by reducing uncertainty through rigorous evidence, careful statistics, translational testing, and validated use of new tools such as AI.",
        isCorrect: true,
      },
      {
        text: "Observation alone always proves treatment efficacy.",
        isCorrect: false,
      },
      {
        text: "Clinical trials exist only to replace biology with paperwork.",
        isCorrect: false,
      },
      {
        text: "AI makes patient outcomes, regulation, and evidence quality irrelevant.",
        isCorrect: false,
      },
    ],
    explanation:
      "Biomedical progress requires both biological understanding and trustworthy evidence. Clinical trials, statistics, translational research, and AI tools all matter, but patient outcomes and evidence quality remain central.",
  },
];

export const BiologyChemistryLifeScienceL5Questions =
  BiologyChemistryForLifeScienceLecture5Questions;
