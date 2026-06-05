import { Question } from "../../../quiz";

export const ClinicalTrialsLecture2Questions: Question[] = [
  {
    id: "clinical-trials-l2-q01",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "Which components belong in a clear clinical trial question using the PICO(T) structure?",
    options: [
      { text: "Population: who is being studied.", isCorrect: true },
      { text: "Intervention: what is being tested.", isCorrect: true },
      {
        text: "Comparator: what the intervention is compared against.",
        isCorrect: true,
      },
      {
        text: "Outcome and time: what is measured and over what period.",
        isCorrect: true,
      },
    ],
    explanation:
      "A trial question must define who is studied, what is tested, what it is compared with, what outcome matters, and when it is measured. Without these components, a treatment claim is too vague to evaluate.",
  },
  {
    id: "clinical-trials-l2-q02",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "A trial enrolls adults with Type 2 Diabetes. Which PICO(T) component does that describe?",
    options: [
      { text: "Population.", isCorrect: true },
      { text: "Comparator.", isCorrect: false },
      { text: "Outcome.", isCorrect: false },
      { text: "Time.", isCorrect: false },
    ],
    explanation:
      "The population defines the patients or participants included in the study. Comparator, outcome, and time answer different design questions about the alternative, measurement, and follow-up period.",
  },
  {
    id: "clinical-trials-l2-q03",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe the comparator's role in a clinical trial?",
    options: [
      {
        text: "It may be placebo, standard of care, no treatment, or an active competing treatment.",
        isCorrect: true,
      },
      {
        text: "It defines what the tested intervention is judged against.",
        isCorrect: true,
      },
      {
        text: "It can be replaced by pre-treatment symptoms whenever the study observes patients for long enough.",
        isCorrect: false,
      },
      {
        text: "It helps turn a vague claim into a comparison that can be interpreted.",
        isCorrect: true,
      },
    ],
    explanation:
      "A treatment effect is meaningful only relative to something else, such as placebo, usual care, or another therapy. Without a comparator, researchers cannot tell whether observed outcomes are better than what would otherwise happen.",
  },
  {
    id: "clinical-trials-l2-q04",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe randomization's role in reducing assignment bias and confounding?",
    options: [
      {
        text: "Randomization assigns treatment by chance rather than physician preference.",
        isCorrect: true,
      },
      {
        text: "Randomization aims to make groups comparable at baseline.",
        isCorrect: true,
      },
      {
        text: "Randomization helps reduce confounding.",
        isCorrect: true,
      },
      {
        text: "Randomization by itself makes trial results correct even when follow-up, measurement, or analysis are biased.",
        isCorrect: false,
      },
    ],
    explanation:
      "Randomization is a core design feature because it reduces systematic differences between groups. It does not guarantee a correct result by itself, since measurement, follow-up, sample size, adherence, and analysis still matter.",
  },
  {
    id: "clinical-trials-l2-q05",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "Which statements correctly distinguish known and unknown confounders in randomized trials?",
    options: [
      {
        text: "Known confounders include measured factors such as age, sex, or disease severity.",
        isCorrect: true,
      },
      {
        text: "Unknown confounders include unmeasured factors such as genetics or lifestyle patterns not captured in the data.",
        isCorrect: true,
      },
      {
        text: "Randomization can help balance both known and unknown confounders.",
        isCorrect: true,
      },
      {
        text: "Known confounders are the full set of factors that can distort interpretation in a trial or observational comparison.",
        isCorrect: false,
      },
    ],
    explanation:
      "Known confounders are factors researchers can name and often measure, while unknown confounders are hidden or unmeasured. Random assignment is valuable because it can reduce systematic imbalance even for factors researchers did not anticipate.",
  },
  {
    id: "clinical-trials-l2-q06",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "Which randomization method is most like repeatedly flipping a coin for each participant?",
    options: [
      { text: "Simple randomization.", isCorrect: true },
      { text: "Block randomization.", isCorrect: false },
      { text: "Stratified randomization.", isCorrect: false },
      { text: "Outcome randomization.", isCorrect: false },
    ],
    explanation:
      "Simple randomization assigns each participant by a chance process similar to independent coin flips. It is easy to understand, but small studies can end up with imbalanced group sizes just by chance.",
  },
  {
    id: "clinical-trials-l2-q07",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "A small trial wants to avoid long enrollment runs where one arm gets many more participants than the other. Which statements correctly describe block randomization?",
    options: [
      {
        text: "It helps keep treatment groups balanced as enrollment proceeds.",
        isCorrect: true,
      },
      {
        text: "It can be especially useful in smaller studies.",
        isCorrect: true,
      },
      {
        text: "It balances enrollment counts but also guarantees equal baseline risk within every block.",
        isCorrect: false,
      },
      {
        text: "It is mainly a method for hiding outcomes from assessors after follow-up.",
        isCorrect: false,
      },
    ],
    explanation:
      "Block randomization is used to avoid long runs that leave one arm overrepresented during enrollment. It helps with allocation balance, but it does not make patients biologically identical or replace the rest of trial design.",
  },
  {
    id: "clinical-trials-l2-q08",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "A trial worries that disease stage could be uneven across treatment arms if assignment is purely simple randomization. Which statements correctly describe stratified randomization?",
    options: [
      {
        text: "It randomizes separately within important baseline subgroups.",
        isCorrect: true,
      },
      {
        text: "It can help balance characteristics such as disease stage or sex.",
        isCorrect: true,
      },
      {
        text: "It is useful when imbalance in a key variable would threaten interpretation.",
        isCorrect: true,
      },
      {
        text: "It restricts enrollment to a narrow patient type as its defining purpose rather than balancing factors within enrolled participants.",
        isCorrect: false,
      },
    ],
    explanation:
      "Stratified randomization protects balance for selected variables that are especially important to the outcome. It does not mean excluding everyone outside one subgroup; it means randomizing within defined strata.",
  },
  {
    id: "clinical-trials-l2-q09",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe blinding in trials where expectations could affect patient reports or clinician assessments?",
    options: [
      {
        text: "Blinding hides treatment assignment from one or more parties.",
        isCorrect: true,
      },
      {
        text: "Blinding can reduce bias caused by expectations.",
        isCorrect: true,
      },
      {
        text: "Blinding is especially useful when outcomes involve judgment or patient reporting.",
        isCorrect: true,
      },
      {
        text: "Blinding assigns participants by chance, making randomization unnecessary.",
        isCorrect: false,
      },
    ],
    explanation:
      "Blinding reduces the chance that knowledge of treatment assignment changes patient behavior, clinician behavior, or outcome assessment. It complements randomization but addresses a different source of bias.",
  },
  {
    id: "clinical-trials-l2-q10",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "Which statements correctly distinguish open-label, single-blind, and double-blind trial designs?",
    options: [
      {
        text: "In an open-label trial, treatment assignments are known.",
        isCorrect: true,
      },
      {
        text: "In a single-blind trial, patients commonly do not know their assignment.",
        isCorrect: true,
      },
      {
        text: "In a double-blind trial, patients and investigators commonly do not know assignments.",
        isCorrect: true,
      },
      {
        text: "Open-label designs remove expectation bias by showing assignments openly.",
        isCorrect: false,
      },
    ],
    explanation:
      "The labels describe who knows the treatment assignment. Open-label trials can be necessary or practical, but they are more vulnerable to expectation, reporting, and treatment-behavior biases.",
  },
  {
    id: "clinical-trials-l2-q11",
    chapter: 2,
    difficulty: "easy",
    prompt: "Which statements correctly describe endpoints?",
    options: [
      {
        text: "An endpoint is an outcome used to evaluate treatment success.",
        isCorrect: true,
      },
      {
        text: "Endpoint choice strongly affects whether a trial answers a clinically meaningful question.",
        isCorrect: true,
      },
      {
        text: "Death, stroke, hospitalization, pain scores, and biomarkers can all be endpoints depending on the trial.",
        isCorrect: true,
      },
      {
        text: "Endpoint selection is a technical detail once treatment assignment is randomized, even when outcomes differ in clinical meaning.",
        isCorrect: false,
      },
    ],
    explanation:
      "The endpoint defines what success means in the study. A randomized trial can still be unhelpful if it measures an outcome that does not matter or fails to capture the intended benefit.",
  },
  {
    id: "clinical-trials-l2-q12",
    chapter: 2,
    difficulty: "easy",
    prompt: "Which endpoint is a direct clinical endpoint?",
    options: [
      { text: "Stroke occurrence.", isCorrect: true },
      {
        text: "A small change in a laboratory marker with uncertain patient relevance.",
        isCorrect: false,
      },
      { text: "An exploratory gene-expression signature.", isCorrect: false },
      {
        text: "A screening score that has not been validated against patient outcomes.",
        isCorrect: false,
      },
    ],
    explanation:
      "A clinical endpoint directly reflects something patients or clinicians care about, such as death, stroke, heart attack, or hospitalization. Biomarkers and exploratory measures can be useful, but they are not automatically direct clinical outcomes.",
  },
  {
    id: "clinical-trials-l2-q13",
    chapter: 2,
    difficulty: "easy",
    prompt: "What is a patient-reported outcome?",
    options: [
      {
        text: "Information about symptoms, function, or quality of life reported directly by the patient.",
        isCorrect: true,
      },
      {
        text: "A clinician's interpretation of symptoms recorded without asking the patient directly.",
        isCorrect: false,
      },
      {
        text: "A biomarker used as a surrogate because it is easier to measure than symptoms.",
        isCorrect: false,
      },
      {
        text: "A site-level recruitment metric used to track whether enrollment is on schedule.",
        isCorrect: false,
      },
    ],
    explanation:
      "Patient-reported outcomes capture information directly from the patient's experience, such as pain, fatigue, daily functioning, or quality of life. They are especially important when the treatment goal is symptom relief or improved lived experience.",
  },
  {
    id: "clinical-trials-l2-q14",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe Phase I-IV clinical trial phases?",
    options: [
      {
        text: "Phase I focuses mainly on safety and tolerability.",
        isCorrect: true,
      },
      {
        text: "Phase II looks for early efficacy and dose information.",
        isCorrect: true,
      },
      {
        text: "Phase III provides larger confirmatory evidence.",
        isCorrect: true,
      },
      {
        text: "Phase IV studies happen after approval and can examine real-world safety or effectiveness.",
        isCorrect: true,
      },
    ],
    explanation:
      "Clinical development phases answer different questions as uncertainty decreases and investment increases. Early phases emphasize safety and signals, while later phases test whether evidence is strong enough for broad use and continued monitoring.",
  },
  {
    id: "clinical-trials-l2-q15",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which design questions are part of defining the population in a clinical trial?",
    options: [
      {
        text: "What disease severity or stage is eligible.",
        isCorrect: true,
      },
      {
        text: "Whether comorbidities or prior treatments are allowed.",
        isCorrect: true,
      },
      {
        text: "Whether the outcome will be measured by a laboratory machine or questionnaire.",
        isCorrect: false,
      },
      {
        text: "Which regulatory agency will review the final submission.",
        isCorrect: false,
      },
    ],
    explanation:
      "Population criteria define which patients the evidence applies to and who may participate. Outcome measurement and regulatory review are important, but they answer different trial-design questions.",
  },
  {
    id: "clinical-trials-l2-q16",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "A back-pain trial defines adults with chronic back pain, a new drug, standard therapy, pain reduction at 12 weeks, and follow-up length. Which parts are now specified?",
    options: [
      { text: "Population.", isCorrect: true },
      { text: "Intervention.", isCorrect: true },
      { text: "Comparator.", isCorrect: true },
      { text: "Outcome and time.", isCorrect: true },
    ],
    explanation:
      "This example specifies the major elements needed to evaluate a clinical claim: who is included, what is tested, what it is compared against, what is measured, and when. Clear specification makes the trial interpretable and prevents vague claims from standing in for evidence.",
  },
  {
    id: "clinical-trials-l2-q17",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "If physicians choose which patients receive a new therapy, which problems can result?",
    options: [
      {
        text: "Younger or healthier patients may be preferentially treated.",
        isCorrect: true,
      },
      {
        text: "More severe patients may be preferentially treated.",
        isCorrect: true,
      },
      {
        text: "Treatment groups can become systematically different before outcomes are measured.",
        isCorrect: true,
      },
      {
        text: "Confounding disappears because physicians know the patients.",
        isCorrect: false,
      },
    ],
    explanation:
      "Physician-selected assignment can reflect prognosis, disease severity, access, or clinical judgment. Those differences can confound the comparison because outcomes may differ even if the treatment itself has no effect.",
  },
  {
    id: "clinical-trials-l2-q18",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statements correctly explain why randomization does not guarantee identical groups?",
    options: [
      {
        text: "Chance imbalance can still occur, especially in small studies.",
        isCorrect: true,
      },
      {
        text: "Randomization makes systematic assignment bias less likely rather than mathematically impossible.",
        isCorrect: true,
      },
      {
        text: "Baseline checks are still useful for understanding trial groups.",
        isCorrect: true,
      },
      {
        text: "Randomization means patients have matched biology across groups.",
        isCorrect: false,
      },
    ],
    explanation:
      "Randomization uses chance to make group differences less systematic, but chance can still produce imbalance. This is why sample size, allocation methods, and baseline descriptions remain important.",
  },
  {
    id: "clinical-trials-l2-q19",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which scenarios are examples of bias that blinding can help reduce?",
    options: [
      {
        text: "Patients who know they received a new pain drug report lower pain because they expect improvement.",
        isCorrect: true,
      },
      {
        text: "Clinicians who know assignment evaluate borderline outcomes more favorably for the experimental group.",
        isCorrect: true,
      },
      {
        text: "A random number generator creates treatment assignments by chance.",
        isCorrect: false,
      },
      {
        text: "A protocol defines inclusion criteria before enrollment starts.",
        isCorrect: false,
      },
    ],
    explanation:
      "Blinding mainly addresses expectation, reporting, assessment, and behavior differences after assignment. Random number generation and prespecified eligibility are important, but they are not examples of bias caused by knowing the treatment assignment.",
  },
  {
    id: "clinical-trials-l2-q20",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which interventions may be difficult or impossible to blind completely?",
    options: [
      { text: "Surgery.", isCorrect: true },
      { text: "Rehabilitation programs.", isCorrect: true },
      { text: "Lifestyle interventions.", isCorrect: true },
      { text: "Some behavioral interventions.", isCorrect: true },
    ],
    explanation:
      "Some interventions are visible or experiential, making full blinding unrealistic. Researchers then need other bias-reduction methods, such as objective endpoints, blinded adjudication, standardized procedures, or careful comparator selection.",
  },
  {
    id: "clinical-trials-l2-q21",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "A cardiovascular drug lowers a biomarker quickly, but the trial also tracks stroke and hospitalization. Which statements correctly compare clinical endpoints and biomarkers?",
    options: [
      {
        text: "Clinical endpoints directly reflect outcomes patients care about, such as death or hospitalization.",
        isCorrect: true,
      },
      {
        text: "Biomarkers are objective biological measurements that may be available sooner.",
        isCorrect: true,
      },
      {
        text: "A biomarker improvement is sufficient evidence that patients live longer or feel better in the intended population.",
        isCorrect: false,
      },
      {
        text: "Clinical endpoints are less relevant whenever a biomarker is objective and easier to measure.",
        isCorrect: false,
      },
    ],
    explanation:
      "Clinical endpoints are usually more directly meaningful, while biomarkers can be faster or easier to measure. A biomarker must be interpreted carefully because improving a measurement does not always translate into improved patient outcomes.",
  },
  {
    id: "clinical-trials-l2-q22",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "A trial uses blood pressure as an earlier substitute for long-term stroke outcomes. Which statements correctly describe surrogate endpoints?",
    options: [
      {
        text: "A surrogate endpoint is a biomarker or intermediate outcome used as a substitute for a clinical outcome.",
        isCorrect: true,
      },
      {
        text: "Surrogates can speed research when the true clinical outcome takes a long time to observe.",
        isCorrect: true,
      },
      {
        text: "Surrogates are safer and more meaningful than clinical endpoints by default.",
        isCorrect: false,
      },
      {
        text: "Using a surrogate removes the need for a scientific rationale.",
        isCorrect: false,
      },
    ],
    explanation:
      "Surrogates can be useful when they validly predict clinical benefit or allow earlier evidence. They are risky when the surrogate changes but the patient-centered outcome does not improve, or even worsens.",
  },
  {
    id: "clinical-trials-l2-q23",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "What is the key lesson from anti-arrhythmic drugs that improved electrocardiogram measurements but later increased mortality?",
    options: [
      {
        text: "Improving a surrogate measure does not guarantee improved patient survival or wellbeing.",
        isCorrect: true,
      },
      {
        text: "A surrogate is acceptable whenever it improves, even if it has not been validated against patient outcomes.",
        isCorrect: false,
      },
      {
        text: "A biomarker should outrank survival when it is more directly tied to the drug's mechanism.",
        isCorrect: false,
      },
      {
        text: "Mechanistic plausibility removes the need to monitor hard outcomes once the surrogate improves.",
        isCorrect: false,
      },
    ],
    explanation:
      "The anti-arrhythmic example shows that a treatment can improve a biological or technical measurement while harming patients. It does not mean surrogates are never useful, but it proves they need validation and caution.",
  },
  {
    id: "clinical-trials-l2-q24",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statements correctly connect clinical trial phases to risk reduction and investment?",
    options: [
      {
        text: "Early development begins with high uncertainty and limited human evidence.",
        isCorrect: true,
      },
      {
        text: "Later phases typically involve more participants and higher cost.",
        isCorrect: true,
      },
      {
        text: "Evidence accumulates across phases before broad approval decisions.",
        isCorrect: true,
      },
      {
        text: "Each phase answers a different main question.",
        isCorrect: true,
      },
    ],
    explanation:
      "Drug development is structured to reduce uncertainty before large investments and broad patient exposure. Phase I, II, III, and IV studies do not simply repeat the same question; they move from safety and signal detection toward confirmation and real-world monitoring.",
  },
  {
    id: "clinical-trials-l2-q25",
    chapter: 2,
    difficulty: "medium",
    prompt: "Which statements correctly describe Phase I trials?",
    options: [
      {
        text: "They commonly focus on safety, tolerability, dose, and side effects.",
        isCorrect: true,
      },
      {
        text: "They are often smaller than Phase III trials.",
        isCorrect: true,
      },
      {
        text: "They usually provide definitive proof of patient benefit because exposure is tightly monitored.",
        isCorrect: false,
      },
      {
        text: "They are primarily real-world effectiveness studies after broad approval.",
        isCorrect: false,
      },
    ],
    explanation:
      "Phase I studies are early human studies that ask whether the intervention can be given safely and at what dose. They are not usually designed to provide definitive efficacy or post-market real-world evidence.",
  },
  {
    id: "clinical-trials-l2-q26",
    chapter: 2,
    difficulty: "medium",
    prompt: "Which statements correctly describe Phase II trials?",
    options: [
      {
        text: "They often look for early evidence that the treatment works.",
        isCorrect: true,
      },
      {
        text: "They can help choose doses for later confirmatory testing.",
        isCorrect: true,
      },
      {
        text: "They replace the need for any Phase III evidence.",
        isCorrect: false,
      },
      {
        text: "They are designed primarily to monitor rare post-marketing events in millions of users.",
        isCorrect: false,
      },
    ],
    explanation:
      "Phase II studies bridge early safety work and large confirmatory trials by testing signals of efficacy and dose selection. They usually do not provide the same level of confirmation or rare-event detection as later development.",
  },
  {
    id: "clinical-trials-l2-q27",
    chapter: 2,
    difficulty: "medium",
    prompt: "Which statement best describes Phase III trials?",
    options: [
      {
        text: "They are larger confirmatory studies asking whether the treatment truly works and has clinically meaningful benefit.",
        isCorrect: true,
      },
      {
        text: "They mainly ask whether the drug can be tolerated in the first few exposed humans.",
        isCorrect: false,
      },
      {
        text: "They mainly monitor rare events after routine use has already begun.",
        isCorrect: false,
      },
      {
        text: "They focus on mechanism only and usually avoid clinically meaningful comparators.",
        isCorrect: false,
      },
    ],
    explanation:
      "Phase III trials are usually the major confirmatory studies used to support approval decisions. They are larger and more expensive because they need enough evidence to show benefit, safety, and clinical relevance.",
  },
  {
    id: "clinical-trials-l2-q28",
    chapter: 2,
    difficulty: "hard",
    prompt: "Which statements correctly describe Phase IV studies?",
    options: [
      {
        text: "They occur after approval.",
        isCorrect: true,
      },
      {
        text: "They can study long-term safety.",
        isCorrect: true,
      },
      {
        text: "They can detect rare adverse events that earlier trials may miss.",
        isCorrect: true,
      },
      {
        text: "They can examine effectiveness in real-world practice.",
        isCorrect: true,
      },
    ],
    explanation:
      "Phase IV studies extend learning after a product enters broader use. Earlier trials may be too small, too short, or too selective to reveal long-term safety, rare harms, or routine-practice effectiveness.",
  },
  {
    id: "clinical-trials-l2-q29",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe internal validity in clinical trials?",
    options: [
      {
        text: "Internal validity asks whether the observed result was actually caused by the intervention.",
        isCorrect: true,
      },
      {
        text: "Threats include confounding, bias, poor randomization, and protocol deviations.",
        isCorrect: true,
      },
      {
        text: "Internal validity is the same as broad generalizability to real-world patients.",
        isCorrect: false,
      },
      {
        text: "Internal validity matters less when a study has many participants.",
        isCorrect: false,
      },
    ],
    explanation:
      "Internal validity concerns whether the study result is true for the people and conditions studied. A large trial can still have poor internal validity if assignment, measurement, adherence, or analysis are biased.",
  },
  {
    id: "clinical-trials-l2-q30",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "A rigorous trial works in a narrow academic setting, but clinicians wonder whether the result applies in routine practice. Which statements correctly describe external validity?",
    options: [
      {
        text: "External validity asks whether results generalize beyond the study setting.",
        isCorrect: true,
      },
      {
        text: "Strict eligibility criteria can limit external validity.",
        isCorrect: true,
      },
      {
        text: "External validity follows directly from high internal validity.",
        isCorrect: false,
      },
      {
        text: "External validity means randomization was concealed from investigators.",
        isCorrect: false,
      },
    ],
    explanation:
      "External validity is about applicability to real-world patients, settings, adherence patterns, and clinical practice. A rigorous trial can answer the causal question well in a narrow population while still leaving uncertainty about broader use.",
  },
  {
    id: "clinical-trials-l2-q31",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "A trial includes only patients aged 40-60 with few comorbidities and excellent adherence, but the real-world population is older with multiple chronic conditions. Which concerns or interpretations are appropriate?",
    options: [
      {
        text: "The trial may have limited external validity.",
        isCorrect: true,
      },
      {
        text: "Real-world patients may have different risks, benefits, or adherence patterns.",
        isCorrect: true,
      },
      {
        text: "Additional evidence may be needed in broader populations.",
        isCorrect: true,
      },
      {
        text: "Strict eligibility can make a study less representative of routine practice.",
        isCorrect: true,
      },
    ],
    explanation:
      "The concern is whether findings in a narrow, idealized population apply to older, more complex real-world patients. This is a generalizability problem that may require broader evidence, not automatic proof of harm or failed randomization.",
  },
  {
    id: "clinical-trials-l2-q32",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe the tradeoff between strict eligibility criteria and generalizability?",
    options: [
      {
        text: "Strict criteria can reduce heterogeneity and strengthen internal validity.",
        isCorrect: true,
      },
      {
        text: "Strict criteria can exclude patients who would receive the treatment in routine practice.",
        isCorrect: true,
      },
      {
        text: "Broader criteria may improve real-world relevance but add variability.",
        isCorrect: true,
      },
      {
        text: "There is no possible tradeoff between internal and external validity.",
        isCorrect: false,
      },
    ],
    explanation:
      "Trialists often control variation to make causal inference cleaner, but that can make the study less representative. Broader inclusion may improve applicability while making the signal harder to detect or interpret.",
  },
  {
    id: "clinical-trials-l2-q33",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "Which statement best distinguishes explanatory and pragmatic trials?",
    options: [
      {
        text: "Explanatory trials ask whether a treatment can work under ideal conditions, while pragmatic trials ask whether it works in routine practice.",
        isCorrect: true,
      },
      {
        text: "Explanatory trials focus on routine-practice workflow, while pragmatic trials focus on ideal adherence and narrow eligibility.",
        isCorrect: false,
      },
      {
        text: "Pragmatic trials sacrifice all causal inference because real-world practice is variable.",
        isCorrect: false,
      },
      {
        text: "The distinction mainly describes whether the sponsor or regulator names the trial.",
        isCorrect: false,
      },
    ],
    explanation:
      "Explanatory trials prioritize clean causal inference under controlled conditions. Pragmatic trials prioritize performance in real clinical settings, where patients, clinicians, adherence, and workflows may be more variable.",
  },
  {
    id: "clinical-trials-l2-q34",
    chapter: 2,
    difficulty: "hard",
    prompt: "Which design choices would usually strengthen internal validity?",
    options: [
      {
        text: "Random assignment with allocation processes that reduce selection bias.",
        isCorrect: true,
      },
      {
        text: "Blinded outcome assessment when feasible.",
        isCorrect: true,
      },
      {
        text: "Letting investigators change endpoints after seeing results.",
        isCorrect: false,
      },
      {
        text: "Using vague eligibility rules that differ by site.",
        isCorrect: false,
      },
    ],
    explanation:
      "Randomization and blinded assessment reduce major threats to causal interpretation. Changing endpoints after seeing data and using inconsistent eligibility rules can introduce bias and make the result less trustworthy.",
  },
  {
    id: "clinical-trials-l2-q35",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "Which endpoint hierarchy is generally most defensible when patient benefit is the goal?",
    options: [
      {
        text: "Direct clinical outcomes first, validated surrogates next, exploratory biomarkers with the most caution.",
        isCorrect: true,
      },
      {
        text: "Exploratory biomarkers first when they move quickly, direct clinical outcomes last when they are slower.",
        isCorrect: false,
      },
      {
        text: "Unvalidated surrogates can be treated like direct clinical outcomes when the mechanism is plausible.",
        isCorrect: false,
      },
      {
        text: "Endpoint hierarchy should be based mainly on measurement convenience rather than patient relevance.",
        isCorrect: false,
      },
    ],
    explanation:
      "Direct clinical outcomes are closest to patient benefit, while validated surrogates can be acceptable when they reliably predict that benefit. Exploratory biomarkers are useful for learning but require caution before being treated as proof of clinical value.",
  },
  {
    id: "clinical-trials-l2-q36",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "A trial uses a surrogate endpoint because the clinical outcome would take years to observe. Which questions should a critical reviewer ask?",
    options: [
      {
        text: "Has the surrogate been validated against the clinical outcome in this disease area?",
        isCorrect: true,
      },
      {
        text: "Could the treatment improve the surrogate while failing to improve patient outcomes?",
        isCorrect: true,
      },
      {
        text: "Are safety outcomes being monitored even if the surrogate improves?",
        isCorrect: true,
      },
      {
        text: "Is the surrogate being interpreted with appropriate uncertainty?",
        isCorrect: true,
      },
    ],
    explanation:
      "Surrogates can make trials feasible, but they require careful validation and interpretation. A reviewer should ask whether the surrogate truly predicts patient benefit and whether harms or disconnects between biomarker and clinical outcome are being missed.",
  },
  {
    id: "clinical-trials-l2-q37",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "Which statements correctly evaluate a trial design that is randomized, double-blind, has a relevant comparator, and measures a patient-important outcome?",
    options: [
      {
        text: "Randomization helps reduce confounding.",
        isCorrect: true,
      },
      {
        text: "Double blinding helps reduce expectation and assessment bias.",
        isCorrect: true,
      },
      {
        text: "A relevant comparator makes the treatment effect interpretable.",
        isCorrect: true,
      },
      {
        text: "A patient-important outcome improves clinical relevance.",
        isCorrect: true,
      },
    ],
    explanation:
      "These are all favorable design features because they address different weaknesses in treatment evaluation. They do not guarantee perfection, but they make the evidence more trustworthy and clinically meaningful.",
  },
  {
    id: "clinical-trials-l2-q38",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "A trial is open-label because the intervention is a surgery. Which mitigation is most directly aimed at reducing assessment bias?",
    options: [
      {
        text: "Use blinded adjudicators or assessors for outcomes when possible.",
        isCorrect: true,
      },
      {
        text: "Use the same surgeons who delivered treatment to adjudicate subjective outcomes.",
        isCorrect: false,
      },
      {
        text: "Prefer an unvalidated biomarker solely because it is less subjective.",
        isCorrect: false,
      },
      {
        text: "Remove the comparator group so participants cannot compare expectations.",
        isCorrect: false,
      },
    ],
    explanation:
      "When participants or treating clinicians cannot be blinded, blinded outcome assessment can still reduce biased interpretation. Difficulty blinding an intervention is not a reason to abandon meaningful endpoints or comparison groups.",
  },
  {
    id: "clinical-trials-l2-q39",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "A Phase II trial shows a promising biomarker improvement but has limited patient-outcome data. What is the most defensible next interpretation?",
    options: [
      {
        text: "The finding supports further testing but does not by itself prove clinical benefit.",
        isCorrect: true,
      },
      {
        text: "The treatment should be assumed to improve survival across patients.",
        isCorrect: false,
      },
      {
        text: "Phase III confirmation can be skipped because biomarkers translate into outcomes.",
        isCorrect: false,
      },
      {
        text: "Safety monitoring can stop once a biomarker moves in the desired direction.",
        isCorrect: false,
      },
    ],
    explanation:
      "A Phase II biomarker signal can be valuable for deciding whether to continue development and which dose to test. It is not the same as definitive evidence that patients live longer, feel better, or avoid important clinical events.",
  },
  {
    id: "clinical-trials-l2-q40",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "Which statements correctly summarize how trial design determines whether evidence is trustworthy?",
    options: [
      {
        text: "PICO(T) clarifies exactly what question the trial asks.",
        isCorrect: true,
      },
      {
        text: "Randomization and blinding reduce major threats to internal validity.",
        isCorrect: true,
      },
      {
        text: "Endpoint choice determines whether the result is meaningful to patients, regulators, or clinicians.",
        isCorrect: true,
      },
      {
        text: "Eligibility criteria and trial setting influence how well the result generalizes.",
        isCorrect: true,
      },
    ],
    explanation:
      "Trustworthy evidence depends on asking a precise question, reducing bias, measuring the right outcomes, and understanding the target population. Good design links causal inference with practical relevance rather than treating a trial as a generic data-collection exercise.",
  },
  {
    id: "clinical-trials-l2-q41",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "A sponsor says a new therapy reduces chronic back pain. Which missing design details are most needed to turn that claim into an interpretable PICO(T) question?",
    options: [
      {
        text: "The patient group, including disease duration, severity, prior treatments, and relevant comorbidities.",
        isCorrect: true,
      },
      {
        text: "The comparator, outcome instrument, clinically relevant time point, and safety follow-up.",
        isCorrect: true,
      },
      {
        text: "The commercial forecast for the therapy's first year after launch.",
        isCorrect: false,
      },
      {
        text: "The number of clinicians who believe the mechanism is plausible before the trial starts.",
        isCorrect: false,
      },
    ],
    explanation:
      "PICO(T) makes a treatment claim testable by specifying who is studied, what is tested, what it is compared with, what is measured, and when. Commercial interest and expert enthusiasm may influence development, but they do not define the clinical comparison needed for evidence.",
  },
  {
    id: "clinical-trials-l2-q42",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "A 48-patient trial enrolls slowly over six months and disease stage strongly predicts outcome. Which randomization choices are defensible?",
    options: [
      {
        text: "Block randomization can reduce long temporary imbalances while enrollment is still underway.",
        isCorrect: true,
      },
      {
        text: "Stratified randomization can balance treatment groups within important disease-stage categories.",
        isCorrect: true,
      },
      {
        text: "Simple randomization may still be valid, but it has a higher chance of imbalance in a small trial.",
        isCorrect: true,
      },
      {
        text: "Physician-selected assignment is preferable because physicians can intentionally balance prognosis using clinical judgment.",
        isCorrect: false,
      },
    ],
    explanation:
      "Small trials are more vulnerable to chance imbalances, especially when enrollment is gradual or a prognostic factor is crucial. Block and stratified methods preserve random assignment while addressing different balance problems that simple randomization may leave to chance.",
  },
  {
    id: "clinical-trials-l2-q43",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "Which design practices help preserve the value of randomization after the assignment sequence has been generated?",
    options: [
      {
        text: "Conceal the next assignment so enrollment decisions cannot be influenced by anticipated treatment group.",
        isCorrect: true,
      },
      {
        text: "Use clear eligibility criteria before assignment so post-randomization exclusions do not selectively reshape groups.",
        isCorrect: true,
      },
      {
        text: "Track baseline characteristics to detect meaningful imbalances that can occur by chance.",
        isCorrect: true,
      },
      {
        text: "Analyze and report protocol deviations because biased follow-up or adherence can weaken causal interpretation.",
        isCorrect: true,
      },
    ],
    explanation:
      "Randomization is not just a random-number list; its value depends on protecting the assignment process and preserving comparable groups through conduct and analysis. Allocation concealment, pre-specified criteria, baseline checks, and deviation reporting keep the causal comparison interpretable.",
  },
  {
    id: "clinical-trials-l2-q44",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "In a diabetes trial, HbA1c improves at 12 months, but cardiovascular events and serious adverse events are still being followed. Which interpretation is most appropriate?",
    options: [
      {
        text: "HbA1c is useful evidence about glycemic control, but patient-important benefit and safety still require outcome and harm assessment.",
        isCorrect: true,
      },
      {
        text: "HbA1c improvement settles the full benefit-risk question because biomarkers are more objective than clinical outcomes.",
        isCorrect: false,
      },
      {
        text: "Cardiovascular outcomes are irrelevant once a diabetes drug changes a diabetes biomarker.",
        isCorrect: false,
      },
      {
        text: "Safety follow-up is mainly a Phase IV concern and should not affect Phase II or Phase III interpretation.",
        isCorrect: false,
      },
    ],
    explanation:
      "HbA1c can be an important biomarker, but it is not the entire clinical question. Trial design must connect biomarker effects with safety, patient outcomes, and the specific claim the study is meant to support.",
  },
  {
    id: "clinical-trials-l2-q45",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "A double-blind drug trial uses matching placebo and concealed randomization codes, with emergency unblinding available for safety. Which statements are correct?",
    options: [
      {
        text: "Matching appearance and concealed codes reduce expectation and assessment differences between groups.",
        isCorrect: true,
      },
      {
        text: "Emergency unblinding can be compatible with blinding when it is reserved for participant safety and documented.",
        isCorrect: true,
      },
      {
        text: "Double blinding eliminates the need for objective endpoint definitions and safety monitoring.",
        isCorrect: false,
      },
      {
        text: "A trial remains fully blinded even if investigators routinely check assignments before rating outcomes.",
        isCorrect: false,
      },
    ],
    explanation:
      "Blinding is meant to prevent expectations and behavior from differing by assignment, but safety exceptions can be built into the protocol. It does not replace endpoint clarity, safety oversight, or disciplined conduct.",
  },
  {
    id: "clinical-trials-l2-q46",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "A rehabilitation trial cannot blind participants or therapists. Which strategies can still improve interpretability?",
    options: [
      {
        text: "Use blinded adjudicators for outcomes that require judgment.",
        isCorrect: true,
      },
      {
        text: "Choose objective or validated endpoints where they match the clinical question.",
        isCorrect: true,
      },
      {
        text: "Standardize co-interventions and follow-up procedures across groups.",
        isCorrect: true,
      },
      {
        text: "Drop the comparator group so expectation differences cannot be observed.",
        isCorrect: false,
      },
    ],
    explanation:
      "Some interventions are inherently difficult to blind, but design can still reduce bias. Blinded assessment, validated outcomes, standardized care, and a relevant comparator preserve more interpretability than abandoning the comparison.",
  },
  {
    id: "clinical-trials-l2-q47",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe endpoint types used in clinical trials?",
    options: [
      {
        text: "Clinical endpoints directly reflect events or states that matter to patients, such as death, stroke, hospitalization, symptoms, or functioning.",
        isCorrect: true,
      },
      {
        text: "Patient-reported outcomes capture information directly from patients, such as pain, fatigue, or quality of life.",
        isCorrect: true,
      },
      {
        text: "Biomarkers are objective biological measurements such as blood pressure, tumor size, viral load, or laboratory values.",
        isCorrect: true,
      },
      {
        text: "Surrogate endpoints use a biomarker or intermediate measure as a substitute for a clinical outcome.",
        isCorrect: true,
      },
    ],
    explanation:
      "Endpoint selection determines what a trial is actually testing. Clinical outcomes, patient-reported outcomes, biomarkers, and surrogate endpoints each answer different questions and carry different strengths and limitations.",
  },
  {
    id: "clinical-trials-l2-q48",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "A pain trial relies on a patient-reported outcome collected through weekly questionnaires. Which design choices strengthen that endpoint?",
    options: [
      {
        text: "Use a validated pain instrument with defined scoring and timing.",
        isCorrect: true,
      },
      {
        text: "Specify the analysis of change from baseline before looking at treatment results.",
        isCorrect: true,
      },
      {
        text: "Let each site choose a different pain scale if local clinicians prefer it.",
        isCorrect: false,
      },
      {
        text: "Treat any favorable comment recorded during follow-up as equivalent to the primary endpoint.",
        isCorrect: false,
      },
    ],
    explanation:
      "Patient-reported outcomes can be highly relevant, but they need standardized instruments, timing, scoring, and analysis. Otherwise subjective outcomes become vulnerable to inconsistent measurement and selective interpretation.",
  },
  {
    id: "clinical-trials-l2-q49",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "Which statements correctly evaluate surrogate endpoints across different disease settings?",
    options: [
      {
        text: "A surrogate is more credible when prior evidence shows that changing it reliably predicts meaningful patient outcomes in that context.",
        isCorrect: true,
      },
      {
        text: "A surrogate can fail if the intervention improves the marker while worsening mortality, symptoms, or serious harms.",
        isCorrect: true,
      },
      {
        text: "Validation depends on the disease, intervention mechanism, endpoint history, and outcome being substituted.",
        isCorrect: true,
      },
      {
        text: "A biomarker that is easy to measure has the same evidentiary strength across diseases once it changes in the desired direction.",
        isCorrect: false,
      },
    ],
    explanation:
      "Surrogates are not uniformly good or bad. Their credibility depends on whether the marker is a validated substitute for patient benefit in the relevant disease and intervention context, not merely on convenience or objectivity.",
  },
  {
    id: "clinical-trials-l2-q50",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "Certain anti-arrhythmic drugs improved electrocardiogram rhythm measures but were later found to increase mortality. Which lessons does this example support?",
    options: [
      {
        text: "Improving an intermediate physiological marker can be misleading when the marker is not a reliable substitute for survival or serious outcomes.",
        isCorrect: true,
      },
      {
        text: "A treatment can move the intended biological signal while still producing net harm.",
        isCorrect: true,
      },
      {
        text: "Patient-important outcomes and safety can overturn confidence based on a plausible surrogate.",
        isCorrect: true,
      },
      {
        text: "Surrogate endpoint trials should be interpreted through evidence history and biological mechanism, not measurement convenience alone.",
        isCorrect: true,
      },
    ],
    explanation:
      "The anti-arrhythmic example is a strong warning that a biomarker can move in the desired direction while patients do worse. Surrogates need validation against outcomes that matter, and benefit-risk interpretation must include harms.",
  },
  {
    id: "clinical-trials-l2-q51",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "A first-in-human study of a potentially toxic oncology drug enrolls patients with advanced cancer rather than healthy volunteers. Which phase-related interpretation is strongest?",
    options: [
      {
        text: "The study can still be Phase I because the central goals are safety, tolerability, dose range, and early pharmacology.",
        isCorrect: true,
      },
      {
        text: "Any study in patients is Phase II because Phase I always uses healthy volunteers.",
        isCorrect: false,
      },
      {
        text: "The study is Phase III if the disease is severe enough to justify risk.",
        isCorrect: false,
      },
      {
        text: "Phase labels depend mainly on whether a CRO or academic site conducts the trial.",
        isCorrect: false,
      },
    ],
    explanation:
      "Phase labels are tied to development questions, not a rigid participant rule. Many Phase I oncology studies enroll patients because exposing healthy volunteers to cytotoxic or high-risk agents would be inappropriate.",
  },
  {
    id: "clinical-trials-l2-q52",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "A Phase II program tests three doses and shows the highest dose has more toxicity while the middle dose has similar biomarker response and better tolerability. Which planning conclusions are reasonable?",
    options: [
      {
        text: "Dose selection should weigh early efficacy signals together with safety and tolerability.",
        isCorrect: true,
      },
      {
        text: "The middle dose may be a better Phase III candidate if it preserves likely benefit with less harm.",
        isCorrect: true,
      },
      {
        text: "The highest dose should move forward by default because Phase III exists to find safety problems later.",
        isCorrect: false,
      },
      {
        text: "Phase II dose decisions are unrelated to endpoint choice in later confirmatory trials.",
        isCorrect: false,
      },
    ],
    explanation:
      "Phase II often reduces uncertainty about whether a drug appears to work and which dose should be tested. A Phase III dose should not be chosen from efficacy signal alone if tolerability or safety would undermine benefit-risk.",
  },
  {
    id: "clinical-trials-l2-q53",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "A Phase III trial is being designed after a promising Phase II signal. Which choices align with the confirmatory purpose of Phase III?",
    options: [
      {
        text: "Use a population and comparator that support the intended clinical or regulatory claim.",
        isCorrect: true,
      },
      {
        text: "Choose a primary endpoint whose magnitude can be judged for patient or clinical importance.",
        isCorrect: true,
      },
      {
        text: "Plan sample size and analysis before results are known to test the main question credibly.",
        isCorrect: true,
      },
      {
        text: "Rely on the Phase II biomarker result as the primary proof of clinical benefit.",
        isCorrect: false,
      },
    ],
    explanation:
      "Phase III is usually where development seeks confirmatory evidence that can support practice, labeling, or approval decisions. That requires a relevant population, comparator, endpoint, sample size, and pre-specified analysis rather than recycling a preliminary signal as proof.",
  },
  {
    id: "clinical-trials-l2-q54",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which study questions commonly belong in Phase IV or post-approval research?",
    options: [
      {
        text: "Whether rare adverse events emerge when many more patients receive the product.",
        isCorrect: true,
      },
      {
        text: "How effectiveness and adherence look in routine clinical practice.",
        isCorrect: true,
      },
      {
        text: "Whether long-term outcomes differ from those observed in pre-approval trials.",
        isCorrect: true,
      },
      {
        text: "How the product performs in broader populations that were excluded or underrepresented earlier.",
        isCorrect: true,
      },
    ],
    explanation:
      "Approval does not end evidence generation. Phase IV and post-market studies often address long-term safety, rare harms, comparative effectiveness, adherence, and performance in broader routine-care populations.",
  },
  {
    id: "clinical-trials-l2-q55",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "A trial enrolls highly adherent patients aged 40-60 with few comorbidities, uses intensive reminders, and shows benefit. Clinicians treat many older patients with multiple conditions and inconsistent adherence. What is the most precise concern?",
    options: [
      {
        text: "External validity is limited because the trial conditions and participants differ from routine practice.",
        isCorrect: true,
      },
      {
        text: "Internal validity is absent because narrow eligibility criteria make randomization impossible.",
        isCorrect: false,
      },
      {
        text: "The result is statistically uninterpretable because pragmatic patients were not enrolled.",
        isCorrect: false,
      },
      {
        text: "The trial proves the treatment fails in older routine-care populations.",
        isCorrect: false,
      },
    ],
    explanation:
      "The trial may have strong internal validity for the enrolled population while still raising generalizability concerns. External validity asks whether the result applies to different patients, settings, adherence patterns, and implementation conditions.",
  },
  {
    id: "clinical-trials-l2-q56",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "A smartphone intervention is first tested with extensive training, frequent reminders, and motivated users, then later tested across routine clinics with minimal support. Which comparisons are correct?",
    options: [
      {
        text: "The first study is more explanatory because it asks whether the intervention can work under favorable conditions.",
        isCorrect: true,
      },
      {
        text: "The later study is more pragmatic because it asks whether the intervention works in ordinary implementation.",
        isCorrect: true,
      },
      {
        text: "The pragmatic study is automatically less useful because it accepts more real-world variation.",
        isCorrect: false,
      },
      {
        text: "The explanatory study fully answers adherence and effectiveness questions for older or less tech-comfortable routine-care patients.",
        isCorrect: false,
      },
    ],
    explanation:
      "Explanatory and pragmatic trials answer different questions. Explanatory designs help establish whether an intervention has an effect under controlled conditions, while pragmatic designs test performance under routine constraints and broader patient behavior.",
  },
  {
    id: "clinical-trials-l2-q57",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "Which factors commonly belong in clinical trial eligibility criteria?",
    options: [
      {
        text: "Diagnosis, disease severity, age range, and prior treatment history.",
        isCorrect: true,
      },
      {
        text: "Comorbidities, interacting medications, pregnancy status, or safety-related exclusions.",
        isCorrect: true,
      },
      {
        text: "Ability to complete required visits, procedures, follow-up, and informed consent.",
        isCorrect: true,
      },
      {
        text: "Preference for patients who are expected to respond favorably after investigators know treatment assignment.",
        isCorrect: false,
      },
    ],
    explanation:
      "Eligibility criteria define who can participate and protect safety, interpretability, and feasibility. They should be specified before assignment rather than used after treatment assignment to select patients who make the result look favorable.",
  },
  {
    id: "clinical-trials-l2-q58",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe how comparator choice affects trial interpretation?",
    options: [
      {
        text: "A placebo comparator can help isolate specific treatment effect when placebo control is ethical and feasible.",
        isCorrect: true,
      },
      {
        text: "An active comparator can show whether a new treatment improves on existing standard care.",
        isCorrect: true,
      },
      {
        text: "No-treatment or usual-care comparators may be appropriate for some pragmatic or behavioral questions.",
        isCorrect: true,
      },
      {
        text: "The comparator determines what the estimated treatment effect means in practice.",
        isCorrect: true,
      },
    ],
    explanation:
      "A treatment effect is always relative to the alternative tested. Placebo, active treatment, usual care, and no-treatment comparators answer different questions and support different clinical interpretations.",
  },
  {
    id: "clinical-trials-l2-q59",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "A cardiovascular drug lowers LDL cholesterol substantially, but the trial is too short to observe heart attacks, stroke, hospitalization, or survival. Which endpoint critique is strongest?",
    options: [
      {
        text: "The study shows an effect on a biomarker, but the patient-outcome claim depends on whether LDL change is a validated surrogate for this intervention and context.",
        isCorrect: true,
      },
      {
        text: "The trial proves cardiovascular event reduction because LDL is part of the causal pathway.",
        isCorrect: false,
      },
      {
        text: "The endpoint is unusable because biomarkers never support clinical development decisions.",
        isCorrect: false,
      },
      {
        text: "Short follow-up improves certainty because it reduces the number of competing clinical events.",
        isCorrect: false,
      },
    ],
    explanation:
      "A biomarker endpoint may be valuable, especially when supported by strong validation history, but it is not automatically equivalent to clinical benefit. The reviewer should ask whether the surrogate is credible for the disease, intervention, and claim being made.",
  },
  {
    id: "clinical-trials-l2-q60",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "A trial has a clear PICO(T) question, randomization, double blinding, a validated patient-reported primary endpoint, and broad eligibility criteria. Which remaining issue would most directly threaten interpretation?",
    options: [
      {
        text: "High missing primary-endpoint data that differs between treatment groups.",
        isCorrect: true,
      },
      {
        text: "The use of broad eligibility criteria, which always weakens every causal comparison.",
        isCorrect: false,
      },
      {
        text: "The presence of a patient-reported outcome, which cannot be meaningful if it is subjective.",
        isCorrect: false,
      },
      {
        text: "The fact that randomization balances unknown confounders probabilistically rather than by guaranteeing identical groups.",
        isCorrect: false,
      },
    ],
    explanation:
      "Good design features can still be undermined by biased missing outcome data. Broad eligibility and patient-reported endpoints are not flaws by themselves, and probabilistic balance is an expected feature of randomization rather than a direct defect.",
  },
];
