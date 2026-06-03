import { Question } from "../../../quiz";

type QuestionDifficulty = "easy" | "medium" | "hard";
type OptionSeed = readonly [text: string, isCorrect: boolean];

function makeQuestion(
  number: number,
  difficulty: QuestionDifficulty,
  prompt: string,
  optionSeeds: readonly OptionSeed[],
  explanation: string,
): Question {
  if (optionSeeds.length !== 4) {
    throw new Error(`Lecture 4 question ${number} must have four options.`);
  }

  return {
    id: `bio-chem-life-l4-q${String(number).padStart(2, "0")}`,
    chapter: 4,
    difficulty,
    prompt,
    options: optionSeeds.map(([text, isCorrect]) => ({ text, isCorrect })),
    explanation,
  };
}

export const BiologyChemistryForLifeScienceLecture4Questions: Question[] = [
  makeQuestion(
    1,
    "easy",
    "Which statements correctly describe physiology as multi-scale coordination?",
    [
      [
        "Cells, tissues, organs, and organ systems interact to produce organism-level function.",
        true,
      ],
      ["Emergent functions can appear only when many cells coordinate.", true],
      [
        "Organ systems communicate through signals, transport, and feedback.",
        true,
      ],
      [
        "No organ system operates fully independently of signals, transport, and feedback from other systems.",
        true,
      ],
    ],
    "Physiology is the study of coordinated function across scales. A heart cell, neuron, immune cell, or endocrine cell does not explain the whole organism alone; interactions among systems create the regulated state we call health.",
  ),
  makeQuestion(
    2,
    "easy",
    "Which statement best describes homeostasis?",
    [
      [
        "Maintaining internal variables within functional ranges despite changing conditions.",
        true,
      ],
      [
        "Holding molecular concentrations at a fixed value despite changing needs.",
        false,
      ],
      [
        "Suppressing feedback loops so internal variables drift with external conditions.",
        false,
      ],
      [
        "A regulatory state where every internal variable is held at one exact value.",
        false,
      ],
    ],
    "Homeostasis is active regulation, not frozen sameness. Variables such as glucose, temperature, blood pressure, pH, and electrolytes fluctuate, but feedback systems keep them within ranges compatible with life.",
  ),
  makeQuestion(
    3,
    "easy",
    "Which components are usually needed for a homeostatic control loop?",
    [
      ["A sensor or measurement of system state.", true],
      ["A comparison against a useful range or set point.", true],
      ["An effector response that changes the variable.", true],
      [
        "Feedback that helps prevent dangerous deviation from a functional range.",
        true,
      ],
    ],
    "Homeostatic loops need information and action. They detect state, compare it to a useful range, respond through effectors, and use feedback to reduce dangerous deviation.",
  ),
  makeQuestion(
    4,
    "easy",
    "Which statements correctly distinguish nervous and endocrine signaling?",
    [
      ["Nervous signaling is often fast and targeted.", true],
      [
        "Endocrine signaling uses hormones that can act over longer distances.",
        true,
      ],
      ["Both systems can participate in physiological regulation.", true],
      [
        "Endocrine signals such as hormones can be carried through blood.",
        true,
      ],
    ],
    "The nervous and endocrine systems coordinate physiology on different timescales and spatial ranges. They also interact, which is why stress, metabolism, sleep, reproduction, and cardiovascular function cannot be understood as isolated systems.",
  ),
  makeQuestion(
    5,
    "easy",
    "Which statement best describes infection as a disease category?",
    [
      [
        "Disease caused by interaction between a host and a pathogen such as a bacterium, virus, fungus, or parasite.",
        true,
      ],
      [
        "A disease category defined by inherited chromosome number rather than host-pathogen interaction.",
        false,
      ],
      ["A drug receptor becoming activated by an agonist.", false],
      [
        "A state where immune response is absent despite pathogen exposure.",
        false,
      ],
    ],
    "Infectious disease depends on both pathogen biology and host response. Symptoms can result from pathogen damage, immune inflammation, tissue injury, or disrupted physiology.",
  ),
  makeQuestion(
    6,
    "easy",
    "Which statements correctly describe cancer at a high level?",
    [
      ["Cancer involves dysregulated cell growth and survival.", true],
      [
        "Cancer can involve evolution of cell populations inside tissues.",
        true,
      ],
      [
        "Cancer cells may evade signals that normally limit division or trigger death.",
        true,
      ],
      [
        "Cancer can differ by tissue, mutation, regulation, immune context, and treatment response.",
        true,
      ],
    ],
    "Cancer is a family of diseases involving altered growth, survival, invasion, and evolution. The systems view matters because tumor behavior depends on mutations, regulation, tissue context, immune pressure, and therapy selection.",
  ),
  makeQuestion(
    7,
    "easy",
    "Which statements correctly describe receptors in pharmacology?",
    [
      ["A receptor is often a protein that receives a signal.", true],
      ["Some drugs work by activating or blocking receptors.", true],
      [
        "Receptors are restricted to single tissues, so shared pathway effects are unlikely.",
        false,
      ],
      [
        "A receptor is any drug molecule that circulates until it finds a protein target.",
        false,
      ],
    ],
    "Receptors connect molecular binding to physiological response. Because receptors and pathways can appear in multiple tissues, the option claiming they never do so is the misconception behind many side-effect misunderstandings.",
  ),
  makeQuestion(
    8,
    "easy",
    "Which statements correctly distinguish agonists and antagonists?",
    [
      ["An agonist activates a receptor or pathway.", true],
      [
        "An antagonist blocks activation or prevents a signal's usual effect.",
        true,
      ],
      [
        "Antagonists have therapeutic uses, while agonists are laboratory artifacts rather than therapeutic drugs.",
        false,
      ],
      [
        "An antagonist works by deleting the receptor gene from treated tissues.",
        false,
      ],
    ],
    "Agonists and antagonists are common pharmacology concepts, and both can be therapeutically useful. Their effects depend on dose, receptor distribution, downstream signaling, and patient context.",
  ),
  makeQuestion(
    9,
    "easy",
    "Which statement best describes pharmacokinetics?",
    [
      [
        "What the body does to the drug, including absorption, distribution, metabolism, and excretion.",
        true,
      ],
      [
        "What the drug does to the body at its target and downstream systems.",
        false,
      ],
      [
        "How strongly the drug activates or blocks its target once it arrives.",
        false,
      ],
      [
        "The patient's reported improvement caused by expectation rather than drug exposure.",
        false,
      ],
    ],
    "Pharmacokinetics, often abbreviated PK, describes drug exposure over time. It asks how the body absorbs, distributes, metabolizes, and eliminates the drug, which strongly affects dose and safety.",
  ),
  makeQuestion(
    10,
    "easy",
    "Which statements correctly describe biomarkers?",
    [
      ["A biomarker is a measurable indicator of a biological state.", true],
      [
        "Biomarkers can support diagnosis, prognosis, monitoring, or treatment selection.",
        true,
      ],
      [
        "A biomarker requires validation before it can guide important decisions.",
        true,
      ],
      [
        "A biomarker can be associated with a disease without being its direct cause.",
        true,
      ],
    ],
    "Biomarkers convert biology into measurement, but measurement is not automatically meaning. A biomarker can be useful only when its relationship to disease state, treatment response, or patient outcome is understood and validated.",
  ),
  makeQuestion(
    11,
    "medium",
    "Which statements correctly explain negative feedback in blood glucose regulation?",
    [
      ["Rising blood glucose can trigger insulin release.", true],
      [
        "Insulin can promote glucose uptake and storage, pushing glucose back toward range.",
        true,
      ],
      [
        "Failure of this regulation is isolated from diseases such as diabetes.",
        false,
      ],
      [
        "Negative feedback means glucose must increase without limit after insulin release.",
        false,
      ],
    ],
    "Blood glucose regulation is a classic homeostatic example. The system senses a variable, responds through hormones and tissues, and counteracts deviation; diabetes shows why the option denying disease relevance is wrong.",
  ),
  makeQuestion(
    12,
    "medium",
    "Which interpretation best explains why symptoms during infection can persist even when pathogen load falls?",
    [
      [
        "Inflammation and tissue repair can continue contributing to symptoms after direct pathogen activity decreases.",
        true,
      ],
      [
        "Symptoms come from direct pathogen contact with pain receptors rather than host inflammatory pathways.",
        false,
      ],
      [
        "A falling pathogen load indicates symptoms are independent of immune activation.",
        false,
      ],
      ["Infection symptoms are unrelated to host physiology.", false],
    ],
    "Symptoms often reflect host-pathogen interaction, not pathogen action alone. Immune activation, cytokines, tissue damage, fever, repair processes, and altered physiology can continue after pathogen burden begins to decline.",
  ),
  makeQuestion(
    13,
    "medium",
    "Which statements correctly describe dose-response relationships?",
    [
      [
        "Effect often increases with dose until targets or downstream systems saturate.",
        true,
      ],
      ["Toxicity can increase with exposure as well as desired effects.", true],
      [
        "Increasing dose keeps improving benefit after target saturation without adding toxicity.",
        false,
      ],
      [
        "Dose-response curves are unrelated to receptor binding or system response.",
        false,
      ],
    ],
    "Dose-response reasoning connects molecular exposure to biological effect. More drug can increase benefit up to a point, but saturation, toxicity, compensatory pathways, and patient variation limit simple more-is-better thinking.",
  ),
  makeQuestion(
    14,
    "medium",
    "Which statements correctly describe a therapeutic window?",
    [
      ["It is a range where expected benefit outweighs expected harm.", true],
      [
        "A narrow therapeutic window can require monitoring and careful dosing.",
        true,
      ],
      ["It is independent of drug exposure and patient vulnerability.", false],
      [
        "Doses below the window are expected to be maximally toxic rather than subtherapeutic.",
        false,
      ],
    ],
    "The therapeutic window is a practical decision concept, not a magic boundary. Benefit and harm depend on exposure and patient factors, so denying those dependencies is unsafe reasoning.",
  ),
  makeQuestion(
    15,
    "medium",
    "Which statements correctly describe pharmacodynamics?",
    [
      ["Pharmacodynamics asks what the drug does to the body.", true],
      [
        "It includes target engagement and downstream biological effects.",
        true,
      ],
      [
        "It treats desired effects as biological while classifying adverse effects outside drug action.",
        false,
      ],
      [
        "It is limited to measuring how fast the liver eliminates a drug.",
        false,
      ],
    ],
    "Pharmacodynamics, often abbreviated PD, describes effect rather than exposure alone. Desired effects and adverse effects both arise from drug action on biological systems.",
  ),
  makeQuestion(
    16,
    "medium",
    "Which statement best describes side effects in systems terms?",
    [
      [
        "They can occur because targets, pathways, and compensatory responses are shared across tissues and functions.",
        true,
      ],
      [
        "They show that target binding observed in vitro was biologically meaningless.",
        false,
      ],
      [
        "They are inconsistent with a drug that has a plausible target mechanism.",
        false,
      ],
      [
        "They arise from genome-wide gene replacement rather than shared targets or pathways.",
        false,
      ],
    ],
    "Side effects are often expected consequences of intervening in connected systems. A target may be present in multiple tissues, and changing one pathway can alter feedback loops, compensatory mechanisms, or unrelated functions.",
  ),
  makeQuestion(
    17,
    "medium",
    "Which statements correctly describe autoimmune disease?",
    [
      ["It involves immune responses directed against self tissues.", true],
      ["It can reflect failure of immune tolerance or regulation.", true],
      [
        "It is mechanistically equivalent to bacterial infection rather than misdirected immune recognition.",
        false,
      ],
      ["It means the immune system is absent rather than misdirected.", false],
    ],
    "Autoimmune disease is not simply weak immunity or infection. It reflects misdirected immune recognition and regulation, which can cause tissue damage even without an external pathogen as the main driver.",
  ),
  makeQuestion(
    18,
    "medium",
    "Which statements correctly describe metabolic disease as disrupted regulation?",
    [
      [
        "Energy storage, hormone signaling, inflammation, genetics, and environment can interact.",
        true,
      ],
      [
        "Diabetes can involve failure to regulate blood glucose effectively.",
        true,
      ],
      [
        "Metabolic disease is best explained by one isolated molecular defect without regulatory or environmental context.",
        false,
      ],
      [
        "Metabolic disease is a pathogen-driven category treated by antibiotics.",
        false,
      ],
    ],
    "Metabolic diseases are systems diseases. They often involve endocrine signaling, tissues such as liver, muscle, fat, and pancreas, inflammatory state, genetic risk, behavior, environment, and time.",
  ),
  makeQuestion(
    19,
    "medium",
    "Which statements correctly describe precision medicine?",
    [
      [
        "It aims to match interventions to biologically meaningful patient subgroups.",
        true,
      ],
      [
        "It avoids biomarkers, genomics, imaging, and clinical features when assigning therapy.",
        false,
      ],
      [
        "It treats benefit and harm as unrelated to patient subgroup biology.",
        false,
      ],
      [
        "It recommends the same intervention for patients regardless of disease subtype.",
        false,
      ],
    ],
    "Precision medicine is about better stratification and decision-making. It can use measurements to identify patients more likely to benefit, be harmed, or require different monitoring, so the negative options are wrong.",
  ),
  makeQuestion(
    20,
    "medium",
    "Which statement best distinguishes a predictive biomarker from a merely associated marker?",
    [
      [
        "A predictive biomarker helps identify who is more likely to respond to a particular intervention.",
        true,
      ],
      [
        "A predictive biomarker is any molecule whose average value differs between two groups.",
        false,
      ],
      [
        "A predictive biomarker has to be the primary disease cause rather than a treatment-response marker.",
        false,
      ],
      [
        "A predictive biomarker is a drug pump that excretes medicine from the kidney.",
        false,
      ],
    ],
    "Association alone is not enough for treatment selection. A predictive biomarker should add information about response to a specific intervention, which requires validation against treatment outcomes.",
  ),
  makeQuestion(
    21,
    "hard",
    "A drug has strong target binding but weak clinical effect. Which explanations are plausible?",
    [
      [
        "The target may not be causally important in the disease context.",
        true,
      ],
      [
        "Drug exposure at the relevant tissue may be insufficient despite binding in vitro.",
        true,
      ],
      ["Compensatory pathways may reduce downstream effect.", true],
      [
        "Strong binding in an assay is sufficient evidence for large patient benefit.",
        false,
      ],
    ],
    "Target binding is only one layer of pharmacology. Tissue exposure, pathway relevance, disease heterogeneity, compensatory biology, timing, and endpoint choice all affect whether a molecular effect becomes a patient benefit.",
  ),
  makeQuestion(
    22,
    "hard",
    "Which statements correctly compare internal and external validity?",
    [
      [
        "Internal validity asks whether the causal conclusion is trustworthy in the studied setting.",
        true,
      ],
      [
        "External validity asks whether the conclusion generalizes to other patients or contexts.",
        true,
      ],
      [
        "Highly restrictive criteria can improve control while reducing generalizability.",
        true,
      ],
      ["External validity means the trial had no control group.", false],
    ],
    "Internal and external validity trade off in many biomedical studies. Although this course does not center trial design, students need the distinction to interpret whether evidence is trustworthy and applicable.",
  ),
  makeQuestion(
    23,
    "hard",
    "Which statements correctly explain why organ systems cannot be treated as independent modules?",
    [
      ["The cardiovascular system transports hormones and immune cells.", true],
      [
        "The nervous system regulates breathing, heart rate, appetite, and endocrine outputs.",
        true,
      ],
      [
        "The endocrine system changes metabolism, reproduction, stress, and growth across tissues.",
        true,
      ],
      [
        "Immune inflammation can alter metabolism, vascular function, and neural signaling.",
        true,
      ],
    ],
    "Organ systems are coupled through transport, signals, feedback, and shared resources. This coupling is why disease in one system often affects others and why therapies can produce broad consequences.",
  ),
  makeQuestion(
    24,
    "hard",
    "A biomarker improves after treatment, but patients do not feel better or live longer. Which interpretation is best?",
    [
      [
        "The biomarker may not be a valid surrogate for the patient-relevant outcome.",
        true,
      ],
      [
        "A biomarker improvement is sufficient evidence of clinical effectiveness.",
        false,
      ],
      [
        "The biomarker improvement makes controlled outcome testing redundant for future claims.",
        false,
      ],
      [
        "The result means biomarker evidence should replace patient-centered outcomes in other settings.",
        false,
      ],
    ],
    "Biomarkers can be useful but must be linked to meaningful outcomes for the decision at hand. Changing a marker is not automatically the same as improving symptoms, function, survival, or safety.",
  ),
  makeQuestion(
    25,
    "hard",
    "Which statements correctly identify why patients respond differently to the same drug?",
    [
      [
        "They can differ in pharmacokinetics such as metabolism or excretion.",
        true,
      ],
      [
        "They can differ in pharmacodynamics such as target expression or pathway state.",
        true,
      ],
      [
        "They can differ in disease subtype, genetics, age, organ function, and co-medications.",
        true,
      ],
      [
        "Patient variation disappears when the active molecule has a consistent chemical structure.",
        false,
      ],
    ],
    "A chemically identical drug can have different exposure and effects in different patients. Variation in biology, disease mechanism, organ function, genetics, microbiome, adherence, and other medicines affects outcomes.",
  ),
  makeQuestion(
    26,
    "hard",
    "Which statements correctly connect homeostasis to pharmacology?",
    [
      [
        "A drug effect can trigger compensatory feedback that changes the net response.",
        true,
      ],
      [
        "Chronic treatment can lead to adaptation or tolerance in some systems.",
        true,
      ],
      [
        "A therapy can restore a variable toward range or push another variable out of range.",
        true,
      ],
      [
        "Homeostatic feedback prevents drugs from changing physiological variables.",
        false,
      ],
    ],
    "Drugs act inside regulated systems. Feedback and compensation can reduce, amplify, or redirect effects, which is why dose, timing, patient state, and monitoring matter clinically.",
  ),
  makeQuestion(
    27,
    "hard",
    "Which statements correctly describe neurodegeneration as a disease category?",
    [
      [
        "It involves progressive dysfunction or loss of neurons or neural networks.",
        true,
      ],
      ["It may develop over years or decades before obvious symptoms.", true],
      [
        "It can involve protein aggregation, inflammation, genetics, metabolism, or environmental factors depending on disease.",
        true,
      ],
      [
        "It is best defined as a rapid bacterial infection of neural tissue rather than progressive dysfunction.",
        false,
      ],
    ],
    "Neurodegenerative diseases are typically chronic and multi-factorial. They illustrate why medicine must reason across molecules, cells, circuits, aging, immune state, genetics, and clinical symptoms.",
  ),
  makeQuestion(
    28,
    "hard",
    "Which statements correctly synthesize PK and PD for a dose decision?",
    [
      ["PK influences how much drug reaches tissues over time.", true],
      [
        "PD influences what effect a given exposure has on targets and physiology.",
        true,
      ],
      ["Both PK and PD can differ across patients.", true],
      [
        "A dose decision can require balancing efficacy, toxicity, timing, and monitoring.",
        true,
      ],
    ],
    "PK and PD must be integrated for real dosing decisions. Exposure without effect is not enough, and effect without exposure information is hard to manage safely.",
  ),
  makeQuestion(
    29,
    "hard",
    "Which statements correctly describe disease as a systems problem?",
    [
      [
        "A disease can involve feedback failure rather than one isolated defect.",
        true,
      ],
      [
        "A disease can have different dominant mechanisms in different patients.",
        true,
      ],
      [
        "Treating one pathway can reveal compensation or side effects elsewhere.",
        true,
      ],
      [
        "A systems view treats molecular mechanisms as background details outside disease explanation.",
        false,
      ],
    ],
    "Systems thinking does not replace molecular biology; it connects molecular mechanisms to cells, tissues, organs, patients, and environments. That connection is essential for understanding complex disease and treatment response.",
  ),
  makeQuestion(
    30,
    "hard",
    "Which statements correctly connect Lecture 4 to modern AI and precision medicine?",
    [
      [
        "AI can help integrate multimodal measurements such as imaging, genomics, labs, and notes.",
        true,
      ],
      [
        "Predictions still need biological interpretation and validation.",
        true,
      ],
      [
        "Patient stratification can be useful when disease mechanisms or treatment responses differ.",
        true,
      ],
      [
        "AI removes the need to understand physiology, pharmacology, biomarkers, or evidence.",
        false,
      ],
    ],
    "AI can support prediction and integration, but it does not make physiology irrelevant. Useful medical AI must map measurements to biological states, clinical decisions, patient outcomes, and evidence.",
  ),
  makeQuestion(
    31,
    "easy",
    "Which statement best describes physiology?",
    [
      [
        "Physiology studies coordinated function across cells, tissues, organs, organ systems, and the organism.",
        true,
      ],
      [
        "Physiology is mainly the naming of DNA bases in inherited sequence information.",
        false,
      ],
      [
        "Physiology is the process of manufacturing proteins in bacterial plasmids.",
        false,
      ],
      [
        "Physiology is the chemical synthesis of small-molecule drugs outside the body.",
        false,
      ],
    ],
    "Physiology focuses on how body systems function and coordinate across scales. Cells, tissues, organs, organ systems, and the whole organism interact to maintain useful internal states.",
  ),
  makeQuestion(
    32,
    "easy",
    "Which statement best describes homeostasis?",
    [
      [
        "Homeostasis means maintaining internal stability despite external change.",
        true,
      ],
      [
        "Homeostasis means driving every body variable upward until it reaches its maximum.",
        false,
      ],
      [
        "Homeostasis means replacing feedback loops with fixed, unregulated outputs.",
        false,
      ],
      [
        "Homeostasis means disease progression without any sensing or response.",
        false,
      ],
    ],
    "Homeostasis keeps variables such as glucose, temperature, blood pressure, pH, oxygen, electrolytes, and fluid balance in functional ranges. It depends on sensing, comparison, response, and feedback.",
  ),
  makeQuestion(
    33,
    "easy",
    "Which statement best defines an agonist?",
    [
      ["An agonist activates a receptor or pathway.", true],
      [
        "An agonist blocks a receptor from responding to its usual signal.",
        false,
      ],
      [
        "An agonist describes how the body absorbs, distributes, metabolizes, and excretes a drug.",
        false,
      ],
      ["An agonist is a measurable indicator of biological state.", false],
    ],
    "An agonist activates a receptor or pathway. By contrast, an antagonist blocks activation or prevents a signal from producing its usual effect. This distinction is basic for understanding receptor pharmacology.",
  ),
  makeQuestion(
    34,
    "easy",
    "Which statement best defines pharmacokinetics?",
    [
      ["Pharmacokinetics asks what the body does to the drug.", true],
      ["Pharmacokinetics asks what the drug does to target physiology.", false],
      ["Pharmacokinetics is the maximum effect a drug can achieve.", false],
      ["Pharmacokinetics is the immune system attacking self tissues.", false],
    ],
    "Pharmacokinetics (PK) describes drug exposure over time through absorption, distribution, metabolism, and excretion. Pharmacodynamics (PD) describes what the drug does to the body.",
  ),
  makeQuestion(
    35,
    "easy",
    "Which statement best defines a biomarker?",
    [
      ["A biomarker is a measurable indicator of biological state.", true],
      ["A biomarker is any treatment that changes a disease mechanism.", false],
      [
        "A biomarker is useful for every clinical decision as soon as it can be measured.",
        false,
      ],
      [
        "A biomarker is the same thing as the disease cause it is associated with.",
        false,
      ],
    ],
    "A biomarker is something measurable that indicates a biological state. Examples include blood glucose, blood pressure, cholesterol, tumor mutations, protein expression, inflammatory markers, and imaging findings.",
  ),
  makeQuestion(
    36,
    "easy",
    "Which statements correctly describe nervous and endocrine signaling?",
    [
      [
        "The nervous system supports fast information processing and control.",
        true,
      ],
      [
        "The endocrine system uses hormones for longer-distance regulation.",
        true,
      ],
      [
        "The nervous system mainly communicates by circulating hormones over days.",
        false,
      ],
      [
        "The endocrine system is limited to reflex arcs across synapses.",
        false,
      ],
    ],
    "The nervous system commonly uses electrical signals, neurotransmitters, reflexes, sensory input, and motor output. The endocrine system regulates over longer distances through hormones such as insulin, cortisol, thyroid hormone, and sex hormones.",
  ),
  makeQuestion(
    37,
    "easy",
    "Which statements correctly describe the cardiovascular system?",
    [
      ["It transports oxygen and carbon dioxide.", true],
      [
        "It transports nutrients, hormones, immune cells, heat, and waste.",
        true,
      ],
      [
        "It mainly works as a closed loop that keeps blood inside the heart and away from tissues.",
        false,
      ],
      [
        "It mainly coordinates long-distance regulation by releasing hormones instead of transporting blood.",
        false,
      ],
    ],
    "The cardiovascular system is a transport network. Coordinated cardiac tissue pumps blood, and the system supports oxygen delivery, waste removal, hormone movement, immune-cell movement, heat distribution, and nutrient transport.",
  ),
  makeQuestion(
    38,
    "easy",
    "Which statements correctly describe a homeostatic loop?",
    [
      ["It needs a sensor.", true],
      ["It needs a set point or useful range plus comparison.", true],
      ["It works by sensing deviation but preventing any response.", false],
      [
        "It works best when feedback amplifies every deviation without limit.",
        false,
      ],
    ],
    "A homeostatic loop needs sensing, a useful range or set point, comparison, an effector response, and feedback. These pieces let the body counter deviations and keep variables in functional ranges.",
  ),
  makeQuestion(
    39,
    "easy",
    "Which statements correctly distinguish negative and positive feedback?",
    [
      [
        "Negative feedback counteracts deviation and stabilizes a system.",
        true,
      ],
      [
        "Positive feedback amplifies change or drives a process toward completion.",
        true,
      ],
      [
        "Negative feedback is the same as uncontrolled amplification of clotting or contractions.",
        false,
      ],
      [
        "Positive feedback mainly returns a variable toward its useful range.",
        false,
      ],
    ],
    "Negative feedback stabilizes systems such as blood glucose regulation or temperature control. Positive feedback amplifies processes such as blood clotting, childbirth contractions, and some cell-fate decisions.",
  ),
  makeQuestion(
    40,
    "easy",
    "Which statements correctly describe disease categories?",
    [
      [
        "Infection involves pathogens such as bacteria, viruses, fungi, or parasites.",
        true,
      ],
      [
        "Autoimmune disease involves the immune system attacking self tissues.",
        true,
      ],
      [
        "Metabolic disease is mainly disrupted immune recognition of self tissue.",
        false,
      ],
      [
        "Neurodegeneration is mainly short-term blood glucose feedback after a meal.",
        false,
      ],
    ],
    "Disease categories can be understood as recurring regulatory failure modes. Infection, cancer, metabolic disease, autoimmune disease, neurodegeneration, and genetic disease involve different mechanisms and often overlap with host regulation.",
  ),
  makeQuestion(
    41,
    "easy",
    "Which statements correctly describe infection and symptoms?",
    [
      ["Pathogens can cause disease through direct damage.", true],
      ["Host immune responses can contribute to symptoms.", true],
      [
        "Inflammation can be protective while also causing tissue effects.",
        true,
      ],
      [
        "Symptoms prove that pathogen damage is the only active mechanism.",
        false,
      ],
    ],
    "Infection is shaped by pathogen activity and the host response. Symptoms can come from direct pathogen damage, immune activation, inflammation, tissue repair, and disrupted physiology.",
  ),
  makeQuestion(
    42,
    "easy",
    "Which statements correctly describe cancer as a disease category?",
    [
      ["Cancer can involve growth advantage.", true],
      ["Cancer can involve evasion of death and tissue constraints.", true],
      ["Cancer can evolve inside tissues.", true],
      [
        "Cancer is mainly normal programmed cell death removing risky cells at the right time.",
        false,
      ],
    ],
    "Cancer is a family of evolutionary diseases in which cells gain behaviors that disrupt tissue regulation. Growth, survival, control escape, and ongoing variation can shape disease progression.",
  ),
  makeQuestion(
    43,
    "easy",
    "Which statements correctly describe dose-response ideas?",
    [
      ["Drug effect can increase with dose and then plateau.", true],
      ["Potency describes how much drug is needed for an effect.", true],
      ["Efficacy describes the maximum effect achievable.", true],
      [
        "Toxicity is the same as beneficial target activation at low exposure.",
        false,
      ],
    ],
    "Dose-response reasoning separates amount, effect, and harm. Potency is about the dose needed for an effect, efficacy is about maximum effect, and toxicity describes harmful effects that often rise with exposure.",
  ),
  makeQuestion(
    44,
    "easy",
    "Which statements correctly describe pharmacodynamics?",
    [
      ["Pharmacodynamics asks what the drug does to the body.", true],
      [
        "Pharmacodynamics includes effects such as lowering blood pressure or reducing inflammation.",
        true,
      ],
      [
        "Pharmacodynamics can involve killing bacteria or inhibiting an enzyme.",
        true,
      ],
      [
        "Pharmacodynamics is the absorption, distribution, metabolism, and excretion of the drug.",
        false,
      ],
    ],
    "Pharmacodynamics (PD) describes drug effects on targets, pathways, and physiology. Pharmacokinetics (PK) describes drug exposure processes such as absorption, distribution, metabolism, and excretion.",
  ),
  makeQuestion(
    45,
    "easy",
    "Which statements correctly describe patient variation?",
    [
      [
        "Patients can differ in genetics, age, sex, organ function, immune state, and microbiome.",
        true,
      ],
      [
        "Patients can differ in lifestyle, environment, previous treatments, and medication interactions.",
        true,
      ],
      ["Disease subtype can affect treatment response.", true],
      [
        "Patient variation is best treated as a measurement error with no biological source.",
        false,
      ],
    ],
    "Patients are biologically and clinically heterogeneous. Differences in genetics, physiology, disease subtype, immune state, organ function, exposures, and medications can change risk, treatment response, and side effects.",
  ),
  makeQuestion(
    46,
    "medium",
    "Which statements correctly describe physiology as multi-scale coordination?",
    [
      [
        "Single cells can have behaviors that combine into tissue function.",
        true,
      ],
      [
        "Organs coordinate with other organs through nervous, endocrine, immune, and transport systems.",
        true,
      ],
      ["Organ-system interactions can create organism-level regulation.", true],
      ["Emergent properties can appear when many parts interact.", true],
    ],
    "The body is a distributed regulatory network rather than a simple centrally controlled machine. Many physiological functions emerge from interactions among cells, tissues, organs, organ systems, and organism-level context.",
  ),
  makeQuestion(
    47,
    "medium",
    "Which statements correctly describe the therapeutic window?",
    [
      ["It is the dose or exposure range where benefit outweighs harm.", true],
      ["Narrow-window drugs require careful monitoring.", true],
      ["The useful window depends on both desired effect and toxicity.", true],
      ["Patient variation can shift where benefit and harm occur.", true],
    ],
    "The therapeutic window is about balancing benefit and harm. It depends on exposure, pharmacodynamics, toxicity, patient factors, and monitoring, especially for drugs with narrow margins.",
  ),
  makeQuestion(
    48,
    "medium",
    "Which statements correctly describe ADME?",
    [
      ["Absorption describes drug entry into the body or bloodstream.", true],
      ["Distribution describes movement into tissues and compartments.", true],
      ["Metabolism describes chemical modification of a drug.", true],
      ["Excretion describes removal from the body.", true],
    ],
    "ADME is the core pharmacokinetic framework: absorption, distribution, metabolism, and excretion. Together these processes determine how much drug reaches relevant tissues and how exposure changes over time.",
  ),
  makeQuestion(
    49,
    "medium",
    "Which statements correctly describe biomarkers and precision medicine?",
    [
      [
        "Biomarkers can support diagnosis, prognosis, monitoring, treatment selection, and safety tracking.",
        true,
      ],
      ["Biomarkers need validation for the decision they support.", true],
      [
        "Companion diagnostics can identify likely responders or patients at risk of harm.",
        true,
      ],
      [
        "Precision medicine matches intervention to biologically meaningful patient subgroups.",
        true,
      ],
    ],
    "Biomarkers are useful when the measurement is reliable and connected to a decision. Precision medicine does not mean a unique treatment for each person; it means using meaningful biological stratification to improve decisions.",
  ),
  makeQuestion(
    50,
    "medium",
    "Which statements correctly describe why side effects are expected in pharmacology?",
    [
      ["Targets can appear in multiple tissues.", true],
      [
        "Pathways can connect to downstream systems beyond the intended effect.",
        true,
      ],
      [
        "Changing one receptor, enzyme, channel, or transporter can shift broader physiology.",
        true,
      ],
      ["Dose, exposure, patient variation, and context can affect harm.", true],
    ],
    "Drugs perturb biological systems rather than isolated components. Side effects are common because targets, pathways, and patients are interconnected across tissues and scales.",
  ),
];

export const BiologyChemistryLifeScienceL4Questions =
  BiologyChemistryForLifeScienceLecture4Questions;
