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
    throw new Error(`Lecture 5 question ${number} must have four options.`);
  }

  return {
    id: `bio-chem-life-l5-q${String(number).padStart(2, "0")}`,
    chapter: 5,
    difficulty,
    prompt,
    options: optionSeeds.map(([text, isCorrect]) => ({ text, isCorrect })),
    explanation,
  };
}

export const BiologyChemistryForLifeScienceLecture5Questions: Question[] = [
  makeQuestion(
    1,
    "easy",
    "Which statements correctly describe bacteria and viruses at a high level?",
    [
      [
        "Bacteria are living cells with DNA, ribosomes, metabolism, and membranes.",
        true,
      ],
      [
        "Viruses are genetic material packaged in protein and sometimes lipid envelopes.",
        true,
      ],
      ["Viruses require host-cell machinery to reproduce.", true],
      [
        "Their biological differences are why treatment and prevention strategies can differ.",
        true,
      ],
    ],
    "Bacteria and viruses can both cause infectious disease, but they are biologically different. The distinction matters because antibiotics, antivirals, vaccines, immune responses, and diagnostic strategies target different mechanisms.",
  ),
  makeQuestion(
    2,
    "easy",
    "Which statement best explains why antibiotics usually do not treat viral infections?",
    [
      [
        "Antibiotics often target bacterial cellular processes that viruses do not have.",
        true,
      ],
      ["Viruses are too large for medicines to reach.", false],
      [
        "Viral particles are defined by carbohydrate shells rather than nucleic acids and proteins.",
        false,
      ],
      [
        "Antibiotics treat viruses by targeting host chromosomes rather than bacterial structures.",
        false,
      ],
    ],
    "Many antibiotics exploit bacterial-specific structures or processes, such as cell walls, bacterial ribosomes, or bacterial metabolism. Viruses lack those cellular targets and instead depend on host-cell machinery plus virus-specific steps.",
  ),
  makeQuestion(
    3,
    "easy",
    "Which steps can be part of a viral life cycle?",
    [
      ["Binding to a host-cell receptor.", true],
      ["Entering a host cell and releasing viral genetic material.", true],
      [
        "Maintaining independent ATP-producing metabolism outside any host cell.",
        false,
      ],
      [
        "Producing ATP independently in viral mitochondria while outside any cell.",
        false,
      ],
    ],
    "Viruses are dependent on cells. They can bind, enter, and release genetic material, but they do not maintain independent ATP-producing cellular metabolism outside hosts.",
  ),
  makeQuestion(
    4,
    "easy",
    "Which statements correctly describe vaccines?",
    [
      [
        "Vaccines train immune recognition in a safer context than dangerous infection.",
        true,
      ],
      [
        "Vaccines can present antigens or instructions for making antigens.",
        true,
      ],
      ["Vaccines rely on adaptive immune memory.", true],
      [
        "Vaccines can reduce severe disease risk by preparing faster immune responses.",
        true,
      ],
    ],
    "Vaccines are a practical application of immune learning. They expose the immune system to a target or target instructions so later encounters can be met faster and more effectively.",
  ),
  makeQuestion(
    5,
    "easy",
    "Which statement best describes antimicrobial or antiviral resistance?",
    [
      [
        "A population-level change where variants that survive treatment become more common.",
        true,
      ],
      [
        "A conscious choice by each microbe to ignore a drug after reading its label.",
        false,
      ],
      [
        "A predictable immediate outcome produced uniformly by treatment in each patient.",
        false,
      ],
      [
        "A process that occurs when genetic and phenotypic variation are absent from the population.",
        false,
      ],
    ],
    "Resistance is evolution under selection pressure. Variants with survival advantages can expand when treatment changes the environment, and gene transfer or new mutations can accelerate the process.",
  ),
  makeQuestion(
    6,
    "easy",
    "Which statements correctly describe recombinant protein production?",
    [
      [
        "A gene can be inserted into cells so they produce a desired protein.",
        true,
      ],
      [
        "The produced protein can be used as medicine without purification or quality control.",
        false,
      ],
      [
        "Recombinant production uses cellular machinery for transcription and translation.",
        true,
      ],
      [
        "The process requires finished human organs to grow inside bacteria.",
        false,
      ],
    ],
    "Recombinant protein production uses cells as manufacturing systems. The inserted information is expressed through molecular biology, but the final medicine still requires purification, testing, formulation, and safety controls.",
  ),
  makeQuestion(
    7,
    "easy",
    "Which statements correctly distinguish therapeutic modalities?",
    [
      [
        "Small molecules often bind proteins and can sometimes enter cells.",
        true,
      ],
      [
        "Biologics such as antibodies are usually large molecules made using living systems.",
        true,
      ],
      [
        "mRNA medicines replace treated-cell chromosomes with durable RNA copies.",
        false,
      ],
      [
        "Therapeutic modalities share the same mechanism, dose route, and safety profile.",
        false,
      ],
    ],
    "Therapeutic modality matters because delivery, target access, durability, manufacturing, immune response, and safety differ. Messenger RNA medicines are temporary instruction systems, not permanent replacement of all chromosomes.",
  ),
  makeQuestion(
    8,
    "easy",
    "Which statements correctly describe diagnostics and biomarkers?",
    [
      [
        "Diagnostics classify or measure biological states for decisions.",
        true,
      ],
      [
        "Biomarkers lose usefulness when they support more than one decision type.",
        false,
      ],
      ["A measurement must be validated for the decision it supports.", true],
      [
        "A biomarker association is sufficient to show that changing the marker improves patient outcomes.",
        false,
      ],
    ],
    "Diagnostics and biomarkers translate biology into measurements, but measurements require interpretation. A marker can serve different purposes, and association, causation, prediction, and treatment benefit are different claims.",
  ),
  makeQuestion(
    9,
    "easy",
    "Which statement best describes why model systems are used in biomedical research?",
    [
      [
        "They simplify biological systems so mechanisms, toxicity, or feasibility can be studied before or alongside human evidence.",
        true,
      ],
      [
        "They replicate human disease with full fidelity across mechanisms, symptoms, and treatment response.",
        false,
      ],
      ["They remove the need for any later validation.", false],
      [
        "They are useful mainly as abstract measurements without cells, molecules, or biological readouts.",
        false,
      ],
    ],
    "Model systems are useful because they make biology experimentally tractable. Their limitation is also their strength: simplification helps isolate mechanisms but can omit the complexity that determines human outcomes.",
  ),
  makeQuestion(
    10,
    "easy",
    "Which statements correctly describe the role of clinical evidence in this course?",
    [
      [
        "Clinical evidence tests whether a biological intervention benefits patients in context.",
        true,
      ],
      [
        "Controls, randomization, blinding, and endpoints help reduce misleading conclusions.",
        true,
      ],
      [
        "Clinical trials are important but are a validation layer rather than the core topic of the biology crash course.",
        true,
      ],
      [
        "Evidence connects mechanism and measurement to patient-relevant claims.",
        true,
      ],
    ],
    "Clinical evidence matters because humans are heterogeneous, noisy systems and mechanisms can fail in translation. In this course, trial concepts help students interpret claims without turning the final lecture into a clinical-trials course.",
  ),
  makeQuestion(
    11,
    "medium",
    "A patient has fever, fatigue, and inflammation during infection. Which interpretations are correct?",
    [
      [
        "Some symptoms can come from immune response rather than direct pathogen damage alone.",
        true,
      ],
      [
        "Host physiology stops shaping symptoms once pathogen replication begins.",
        false,
      ],
      [
        "The pathogen and host response together shape disease presentation.",
        true,
      ],
      [
        "Symptoms indicate immune response is absent rather than activated or misdirected.",
        false,
      ],
    ],
    "Infectious disease is a host-pathogen system. Fever and inflammation can be protective in some contexts but harmful in others, so denying host contribution to symptoms is incorrect.",
  ),
  makeQuestion(
    12,
    "medium",
    "Which interpretation best explains why a viral spike protein can be useful for both infection and vaccination?",
    [
      [
        "The same molecular surface can mediate receptor binding and also serve as an immune-recognition target.",
        true,
      ],
      [
        "A spike protein is a carbohydrate storage polymer unrelated to host cells.",
        false,
      ],
      [
        "A spike protein prevents antibodies from binding any virus in principle.",
        false,
      ],
      [
        "A spike protein is useful as an immune target because it edits host DNA.",
        false,
      ],
    ],
    "A viral surface protein can be part of the infection mechanism and an antigen for immune training. This is a recurring biomedical pattern: understanding molecular function can suggest diagnostic, vaccine, or therapeutic targets.",
  ),
  makeQuestion(
    13,
    "medium",
    "Which statements correctly describe antivirals?",
    [
      [
        "They can block viral entry, genome replication, assembly, release, or viral enzymes.",
        true,
      ],
      ["They exploit steps in the viral life cycle.", true],
      ["They are identical to broad bacterial cell-wall antibiotics.", false],
      ["They work by giving viruses their own independent ribosomes.", false],
    ],
    "Antivirals target viral-specific vulnerabilities or virus-dependent host processes. They differ from antibiotics because viruses are not cells with bacterial cell walls, bacterial ribosomes, or independent metabolism.",
  ),
  makeQuestion(
    14,
    "medium",
    "Which statements correctly connect resistance to treatment pressure?",
    [
      [
        "Treatment can kill susceptible variants while resistant variants survive.",
        true,
      ],
      [
        "Incomplete suppression can give surviving variants opportunities to expand.",
        true,
      ],
      [
        "Resistance is disconnected from mutation, selection, and gene transfer.",
        false,
      ],
      [
        "Resistance is restricted to microbes rather than evolving tumor populations.",
        false,
      ],
    ],
    "Selection pressure applies to any varying population that can survive and expand, including bacteria, viruses, and tumor cells. Mutation, selection, and gene transfer can all contribute depending on the organism and context.",
  ),
  makeQuestion(
    15,
    "medium",
    "Which statements correctly compare small molecules and antibodies?",
    [
      [
        "Small molecules are often chemically manufactured and may access intracellular targets.",
        true,
      ],
      [
        "Antibodies are large biologics that often bind extracellular or cell-surface targets.",
        true,
      ],
      [
        "Antibodies are small oral chemicals with tissue access like intracellular small molecules.",
        false,
      ],
      [
        "Small molecules and antibodies have identical manufacturing, dosing, and tissue-access constraints.",
        false,
      ],
    ],
    "Small molecules and antibodies occupy different regions of therapeutic design space. Their strengths and limits follow from size, chemistry, target location, manufacturing, delivery, half-life, and immune interactions.",
  ),
  makeQuestion(
    16,
    "medium",
    "Which statement best distinguishes mRNA medicine from gene editing?",
    [
      [
        "mRNA usually gives temporary protein-production instructions, while gene editing aims to change DNA more durably.",
        true,
      ],
      [
        "mRNA medicine works by disabling ribosomes rather than providing translated instructions.",
        false,
      ],
      [
        "Gene editing is temporary because edited DNA is discarded after a translation event.",
        false,
      ],
      [
        "mRNA and gene editing are identical because both are ordinary antibiotics.",
        false,
      ],
    ],
    "Both approaches intervene in biological information flow, but at different layers. mRNA is translated and degraded, while genome editing changes DNA sequence or regulation and therefore raises different durability and safety questions.",
  ),
  makeQuestion(
    17,
    "medium",
    "Which statements correctly describe cell therapy?",
    [
      [
        "It can involve giving selected, expanded, or engineered cells to a patient.",
        true,
      ],
      [
        "CAR T-cell therapy is an ordinary antibiotic that blocks bacterial cell-wall synthesis.",
        false,
      ],
      [
        "Cell therapies have no manufacturing, safety, persistence, or patient-selection challenges.",
        false,
      ],
      [
        "Cell therapy means prescribing a noncellular small molecule as the therapeutic platform.",
        false,
      ],
    ],
    "Cell therapy treats cells themselves as the intervention or delivery platform. CAR T-cell therapy is an engineered immune-cell approach, and living-cell therapies create real manufacturing, safety, persistence, and selection challenges.",
  ),
  makeQuestion(
    18,
    "medium",
    "Which statements correctly describe translational limits of preclinical models?",
    [
      [
        "A cell culture may omit immune, vascular, endocrine, or tissue context.",
        true,
      ],
      ["An animal model can differ from human disease biology.", true],
      [
        "A biomarker response in a model may not match patient-relevant benefit.",
        true,
      ],
      [
        "A successful model result is sufficient to infer matched safety and efficacy in humans.",
        false,
      ],
    ],
    "Preclinical models are essential but incomplete. Translation fails when simplified systems miss the mechanisms, exposures, heterogeneity, toxicity, or outcomes that matter in patients.",
  ),
  makeQuestion(
    19,
    "medium",
    "Which statements correctly describe endpoints in biomedical evidence?",
    [
      ["An endpoint is the outcome used to evaluate an intervention.", true],
      [
        "A clinical endpoint can reflect how patients feel, function, or survive.",
        true,
      ],
      [
        "A surrogate endpoint can be useful but may not guarantee patient benefit.",
        true,
      ],
      [
        "Endpoint choice is a minor detail because a measured change is sufficient for treatment value.",
        false,
      ],
    ],
    "Endpoint choice determines what claim the evidence supports. A biomarker or surrogate can be valuable, but students must distinguish it from outcomes that directly matter to patients.",
  ),
  makeQuestion(
    20,
    "medium",
    "Which statement best describes AI's role in modern biomedicine?",
    [
      [
        "AI can accelerate discovery and interpretation, but results still need biological grounding, reliable measurement, workflow fit, and evidence.",
        true,
      ],
      [
        "AI removes the need for experiments, diagnostics, clinical validation, or mechanistic understanding.",
        false,
      ],
      [
        "AI predictions establish causation from correlation in observational datasets.",
        false,
      ],
      [
        "AI requires error-free, fully measured biological datasets before it can contribute.",
        false,
      ],
    ],
    "AI is useful but not magic. Biomedical AI depends on data quality, labels, biology, causal interpretation, validation, and human workflows, especially when predictions affect patients.",
  ),
  makeQuestion(
    21,
    "hard",
    "A vaccine induces antibodies against a viral protein, but a new viral variant partly escapes protection. Which explanations are plausible?",
    [
      [
        "Mutations may alter the antigenic surface recognized by antibodies.",
        true,
      ],
      [
        "Immune memory may still reduce severe disease even if infection protection drops.",
        true,
      ],
      [
        "Selection can favor variants that spread despite existing immunity.",
        true,
      ],
      [
        "Partial escape indicates the original vaccine failed to induce adaptive immune memory.",
        false,
      ],
    ],
    "Immune escape is evolutionary and quantitative. Variants can reduce recognition without eliminating all memory or protection, which is why vaccine effectiveness can differ by endpoint, variant, and population.",
  ),
  makeQuestion(
    22,
    "hard",
    "Which statements correctly describe why recombinant production of a human protein can still be technically difficult?",
    [
      [
        "The protein may require correct folding, modification, or assembly.",
        true,
      ],
      ["The production cell type can affect processing and impurities.", true],
      [
        "Purification and quality control must confirm identity, activity, and safety.",
        true,
      ],
      [
        "A DNA sequence by itself is sufficient for matched safety and activity across production systems.",
        false,
      ],
    ],
    "A gene sequence is necessary but not always sufficient for a therapeutic protein. Expression level, folding, post-translational modification, host-cell impurities, purification, formulation, and stability all affect the final product.",
  ),
  makeQuestion(
    23,
    "hard",
    "Which statements correctly identify why a diagnostic test can be accurate in one setting but less useful in another?",
    [
      [
        "Disease prevalence can change the meaning of positive and negative results.",
        true,
      ],
      ["Patient population and sampling method can affect performance.", true],
      [
        "The clinical decision may require different sensitivity-specificity tradeoffs.",
        true,
      ],
      [
        "Context, consequence, and prevalence can affect how useful a test is.",
        true,
      ],
    ],
    "Diagnostics are decision tools, not context-free truth machines. Accuracy metrics, prevalence, population, sample quality, false-positive costs, false-negative costs, and downstream actions determine usefulness.",
  ),
  makeQuestion(
    24,
    "hard",
    "A drug improves a disease biomarker in an animal model but fails to improve symptoms in patients. Which interpretation is best?",
    [
      [
        "The model or biomarker may not capture the patient-relevant disease mechanism or outcome.",
        true,
      ],
      [
        "Animal biomarker improvement is sufficient evidence that patient symptoms will improve.",
        false,
      ],
      [
        "The failure shows animal models are unsuitable for studying mechanisms.",
        false,
      ],
      [
        "The result indicates patient symptoms are disconnected from biological mechanisms.",
        false,
      ],
    ],
    "A translational failure does not mean models are useless; it means the evidence chain was incomplete for the patient claim. Mechanism, model validity, exposure, biomarker relevance, and endpoint selection all need scrutiny.",
  ),
  makeQuestion(
    25,
    "hard",
    "Which statements correctly describe why clinical trials use controls and randomization?",
    [
      ["Controls answer compared to what.", true],
      [
        "Randomization helps balance known and unknown factors in expectation.",
        true,
      ],
      [
        "Both help reduce misleading conclusions from natural history, bias, or confounding.",
        true,
      ],
      [
        "They are used because biology has no mechanisms worth studying.",
        false,
      ],
    ],
    "Controls and randomization are not replacements for mechanism; they are safeguards when testing effects in noisy humans. They help distinguish intervention effects from background change, bias, confounding, and expectation effects.",
  ),
  makeQuestion(
    26,
    "hard",
    "Which statements correctly connect biotechnology to the central dogma?",
    [
      ["DNA constructs can be transcribed into RNA.", true],
      ["RNA instructions can be translated into proteins.", true],
      [
        "Protein products can become medicines, antigens, enzymes, or research tools.",
        true,
      ],
      [
        "Central dogma concepts help explain recombinant proteins, mRNA vaccines, and gene therapies.",
        true,
      ],
    ],
    "Biotechnology is not separate from core biology. It works by redirecting or modifying DNA, RNA, protein production, regulation, and cellular behavior for useful biomedical purposes.",
  ),
  makeQuestion(
    27,
    "hard",
    "Which statements correctly identify AI failure modes in biomedical work?",
    [
      [
        "Biased or incomplete training data can produce misleading predictions.",
        true,
      ],
      ["Noisy labels can teach the model the wrong target.", true],
      [
        "A model can learn correlations that do not transfer or cause the outcome.",
        true,
      ],
      [
        "High benchmark accuracy is sufficient to show clinical usefulness after workflow changes.",
        false,
      ],
    ],
    "Biomedical AI must be evaluated in the setting where it will be used. Dataset bias, label quality, distribution shift, causal mismatch, workflow fit, and patient-impact evidence all matter.",
  ),
  makeQuestion(
    28,
    "hard",
    "Which statements correctly synthesize infection, immunity, and evolution?",
    [
      [
        "Pathogens interact with host-cell molecules and immune defenses.",
        true,
      ],
      ["Immune pressure can select for variants with escape advantages.", true],
      ["Vaccines and treatments change the selective environment.", true],
      [
        "Host damage can result from pathogen activity, immune response, or both.",
        true,
      ],
    ],
    "Infectious disease is a systems topic. It connects molecular binding, cellular takeover, immune recognition, inflammation, population variation, treatment, and evolutionary selection.",
  ),
  makeQuestion(
    29,
    "hard",
    "Which statements correctly describe why evidence generation remains a bottleneck even with strong AI models?",
    [
      [
        "Measurements may be noisy, biased, or weakly linked to outcomes.",
        true,
      ],
      [
        "Clinical workflows and incentives can determine whether a tool is usable.",
        true,
      ],
      ["Safety and efficacy claims still require validation.", true],
      [
        "A more complex model removes uncertainty about patient benefit by increasing parameter count.",
        false,
      ],
    ],
    "AI can improve prediction or discovery, but the bottleneck often lies in measurement, outcome definition, deployment, regulation, safety, and evidence. Better models do not automatically solve poor data or weak clinical validation.",
  ),
  makeQuestion(
    30,
    "hard",
    "Which statements correctly describe the revised role of clinical trials in this biology and chemistry course?",
    [
      [
        "Clinical trials are important for testing whether interventions help patients.",
        true,
      ],
      [
        "Trial concepts help students distinguish mechanism, association, and patient benefit.",
        true,
      ],
      [
        "Detailed trial operations belong in a dedicated clinical-trials course rather than dominating this crash course.",
        true,
      ],
      [
        "Treating clinical trials as context keeps the final lecture focused on biology, biotechnology, diagnostics, and evidence.",
        true,
      ],
    ],
    "Clinical trials remain highly relevant, but they are not the central content of a biology and chemistry crash course. The revised capstone uses trial ideas to interpret biomedical claims while keeping the core on mechanisms, measurements, tools, and translation.",
  ),
  makeQuestion(
    31,
    "easy",
    "Which statement best describes bacteria?",
    [
      [
        "Bacteria are living cells with membranes, metabolism, DNA, ribosomes, and regulated gene expression.",
        true,
      ],
      [
        "Bacteria are protein shells that reproduce by using host-cell ribosomes.",
        false,
      ],
      [
        "Bacteria are antibodies that bind antigens during adaptive immunity.",
        false,
      ],
      ["Bacteria are clinical endpoints used to judge patient benefit.", false],
    ],
    "Bacteria are cells, which is why antibiotics can target bacterial cell walls, ribosomes, DNA replication, or metabolic pathways. Many bacteria are harmless or beneficial, while some cause disease.",
  ),
  makeQuestion(
    32,
    "easy",
    "Which statement best describes viruses?",
    [
      [
        "Viruses are genetic material packaged in protein and sometimes lipid envelopes, and they use host-cell machinery.",
        true,
      ],
      [
        "Viruses are free-living cells with their own ribosomes and complete metabolism.",
        false,
      ],
      [
        "Viruses are small-molecule drugs that enter cells and bind proteins.",
        false,
      ],
      ["Viruses are biomarkers that measure drug exposure over time.", false],
    ],
    "Viruses are not cells and do not reproduce independently. They must enter host cells and use host machinery to replicate genomes, produce proteins, assemble particles, and spread.",
  ),
  makeQuestion(
    33,
    "easy",
    "Which statement best describes a vaccine?",
    [
      [
        "A vaccine trains adaptive immunity to recognize a pathogen or pathogen component before dangerous infection.",
        true,
      ],
      [
        "A vaccine directly kills bacteria by blocking their cell wall synthesis.",
        false,
      ],
      [
        "A vaccine is a diagnostic test that classifies disease state from a sample.",
        false,
      ],
      [
        "A vaccine is a pharmacokinetic process that removes drug from the body.",
        false,
      ],
    ],
    "Vaccines prepare adaptive immunity by presenting a pathogen-related target or instructions for making one. The goal is faster and stronger recognition later, often involving memory B and T cells.",
  ),
  makeQuestion(
    34,
    "easy",
    "Which statement best describes antibiotic resistance?",
    [
      [
        "Resistance evolves when variation affects survival under antibiotic treatment pressure.",
        true,
      ],
      [
        "Resistance means antibiotics convert viral particles into harmless bacteria.",
        false,
      ],
      ["Resistance is the same as a control group in a clinical trial.", false],
      [
        "Resistance occurs because vaccines remove immune memory from B and T cells.",
        false,
      ],
    ],
    "Treatment changes the selective environment. Bacteria with resistance traits can survive and spread under antibiotic pressure, especially when resistance genes move through plasmids or other mechanisms.",
  ),
  makeQuestion(
    35,
    "easy",
    "Which statement best defines a diagnostic?",
    [
      [
        "A diagnostic classifies a biological state in a way that can support a decision.",
        true,
      ],
      [
        "A diagnostic is a protein medicine produced using living systems.",
        false,
      ],
      ["A diagnostic is a drug receptor that activates a pathway.", false],
      [
        "A diagnostic is the randomization step that balances trial groups.",
        false,
      ],
    ],
    "Diagnostics turn biological information into measurements or classifications. Examples include pathogen tests, genetic tests, blood chemistry, imaging, pathology, and protein biomarkers.",
  ),
  makeQuestion(
    36,
    "easy",
    "Which statements correctly describe pathogens?",
    [
      ["Pathogens can include bacteria.", true],
      ["Pathogens can include viruses.", true],
      [
        "Pathogens are mainly clinical-trial endpoints rather than biological agents.",
        false,
      ],
      ["Pathogens are immune memory cells that prevent reinfection.", false],
    ],
    "Pathogens include bacteria, viruses, fungi, and parasites. They differ in structure, replication, treatment options, and immune responses, so identifying the pathogen type matters for diagnosis and therapy.",
  ),
  makeQuestion(
    37,
    "easy",
    "Which statements correctly describe the basic viral life cycle?",
    [
      ["A virus can bind a host receptor and enter a cell.", true],
      [
        "A virus can release genetic material, replicate, assemble particles, and exit.",
        true,
      ],
      [
        "A virus normally divides by mitosis as a free-living eukaryotic cell.",
        false,
      ],
      [
        "A virus normally makes ATP through its own mitochondria before infection.",
        false,
      ],
    ],
    "Viruses must use host cells to reproduce. A simplified pattern is receptor binding, entry, genome release, replication and protein production, particle assembly, and exit.",
  ),
  makeQuestion(
    38,
    "easy",
    "Which statements correctly distinguish antibiotics and antivirals?",
    [
      [
        "Antibiotics can target bacterial cell wall synthesis, ribosomes, DNA replication, or metabolism.",
        true,
      ],
      [
        "Antivirals can target viral entry, genome replication, viral enzymes, assembly, or release.",
        true,
      ],
      [
        "Antibiotics target viruses well because viruses have bacterial ribosomes.",
        false,
      ],
      ["Antivirals are mainly drugs that build bacterial cell walls.", false],
    ],
    "Antibiotics and antivirals target different biology. Antibiotics work against bacterial processes, while antivirals target steps in a viral life cycle.",
  ),
  makeQuestion(
    39,
    "easy",
    "Which statements correctly describe biologics?",
    [
      [
        "Biologics include antibodies, proteins, and other large molecules produced using living systems.",
        true,
      ],
      ["Biologics can be highly specific for extracellular targets.", true],
      [
        "Biologics are usually tiny chemically manufactured compounds designed for oral delivery.",
        false,
      ],
      [
        "Biologics are trial design features such as blinding and endpoint selection.",
        false,
      ],
    ],
    "Biologics are large therapeutic molecules often produced with living systems. They can be powerful and specific, but manufacturing is complex and injection or infusion is often required.",
  ),
  makeQuestion(
    40,
    "easy",
    "Which statements correctly describe model systems in biomedical research?",
    [
      [
        "Cell cultures, organoids, animal models, and computational models are simplified representations.",
        true,
      ],
      [
        "Model systems can be useful for studying mechanisms before human testing.",
        true,
      ],
      [
        "A model system is a validated proof of patient benefit by definition.",
        false,
      ],
      [
        "A model system removes disease heterogeneity from human medicine.",
        false,
      ],
    ],
    "Model systems are useful because they make biology easier to study under controlled conditions. They are limited because they simplify human biology, dosing, immune context, tissue context, heterogeneity, and clinical outcomes.",
  ),
  makeQuestion(
    41,
    "easy",
    "Which statements correctly describe immune memory?",
    [
      ["Some B cells can become memory cells.", true],
      ["Some T cells can become memory cells.", true],
      [
        "Memory can support faster and stronger responses after later exposure.",
        true,
      ],
      ["Immune memory is the same as a drug's metabolism by the liver.", false],
    ],
    "Adaptive immunity can learn and remember molecular targets. Memory B and T cells are part of why vaccines can prepare the immune system before dangerous exposure.",
  ),
  makeQuestion(
    42,
    "easy",
    "Which statements correctly describe recombinant DNA and protein medicines?",
    [
      ["Scientists can insert a gene into cells.", true],
      ["Cells can express the inserted gene as a protein.", true],
      [
        "Insulin can be produced by cells carrying an inserted insulin gene.",
        true,
      ],
      [
        "Recombinant DNA works by removing translation from the production cell.",
        false,
      ],
    ],
    "Recombinant DNA uses cellular information flow for production. A gene can be inserted into bacteria or other production cells, expressed as a protein, and then purified as a medicine.",
  ),
  makeQuestion(
    43,
    "easy",
    "Which statements correctly describe small-molecule drugs?",
    [
      ["They are chemically manufactured compounds.", true],
      ["They can sometimes enter cells and bind intracellular proteins.", true],
      ["Oral delivery is sometimes possible.", true],
      ["They are living cells engineered to recognize cancer targets.", false],
    ],
    "Small molecules are chemically manufactured and can sometimes reach intracellular targets. Their strengths include scalable manufacturing and possible oral delivery, while limits include off-target effects, difficult targets, and toxicity.",
  ),
  makeQuestion(
    44,
    "easy",
    "Which statements correctly describe gene therapy and genome editing?",
    [
      [
        "They aim to add, replace, silence, or modify genetic information.",
        true,
      ],
      ["Delivery to the right cells is a key challenge.", true],
      [
        "Durability, immune response, off-target effects, reversibility, and ethics matter.",
        true,
      ],
      ["They are mainly antibiotics that block bacterial ribosomes.", false],
    ],
    "Gene therapy and genome editing act at the genetic-information layer. Their promise depends on getting the right change into the right cells with acceptable safety, durability, and ethical justification.",
  ),
  makeQuestion(
    45,
    "easy",
    "Which statements correctly describe core clinical evidence concepts?",
    [
      ["A control group asks what the intervention is compared against.", true],
      [
        "Randomization helps balance known and unknown factors in expectation.",
        true,
      ],
      ["Blinding reduces expectation and measurement bias.", true],
      [
        "An endpoint is the chemical structure of a small-molecule drug.",
        false,
      ],
    ],
    "Clinical evidence asks whether biomedical claims hold up in humans. Control groups, randomization, blinding, and endpoints help separate plausible mechanisms or associations from trustworthy causal conclusions about patient outcomes.",
  ),
  makeQuestion(
    46,
    "medium",
    "Which statements correctly describe why antibiotics do not treat viral infections?",
    [
      ["Viruses lack bacterial cell walls.", true],
      ["Viruses lack bacterial ribosomes.", true],
      ["Viruses use host-cell machinery for replication.", true],
      [
        "Antibiotic targets are usually bacterial processes rather than viral life-cycle steps.",
        true,
      ],
    ],
    "Antibiotics target bacterial biology, not viral replication strategies. Viruses are not cells, so antiviral drugs need to target viral entry, viral enzymes, genome replication, assembly, release, or host-virus interactions.",
  ),
  makeQuestion(
    47,
    "medium",
    "Which statements correctly describe therapeutic modalities?",
    [
      [
        "mRNA medicines deliver temporary instructions for cells to make a protein.",
        true,
      ],
      [
        "Cell therapy modifies or selects cells and gives them to a patient.",
        true,
      ],
      [
        "CAR T-cell therapy engineers immune cells to recognize cancer targets.",
        true,
      ],
      [
        "Biologics, small molecules, vaccines, mRNA, gene therapy, and cell therapy differ in delivery, target type, durability, and risks.",
        true,
      ],
    ],
    "Therapeutic modalities intervene at different biological layers. Comparing them requires thinking about molecule size, production, delivery, target location, durability, immune response, safety, and evidence.",
  ),
  makeQuestion(
    48,
    "medium",
    "Which statements correctly describe biomarker types?",
    [
      ["A diagnostic biomarker can help identify disease.", true],
      ["A prognostic biomarker can help predict likely disease course.", true],
      ["A predictive biomarker can help predict treatment response.", true],
      [
        "A pharmacodynamic biomarker can show biological effect of a drug.",
        true,
      ],
    ],
    "Biomarkers can support different decisions depending on what they measure and predict. The same measurement is not automatically useful for diagnosis, prognosis, treatment selection, pharmacodynamic tracking, and safety; its use needs validation.",
  ),
  makeQuestion(
    49,
    "medium",
    "Which statements correctly explain why preclinical results can fail to translate to humans?",
    [
      [
        "The model may omit immune, endocrine, vascular, or tissue context.",
        true,
      ],
      ["Dosing and exposure may differ between the model and humans.", true],
      ["Animal biology may differ from human biology.", true],
      [
        "The measured biomarker may differ from the clinical outcome that matters.",
        true,
      ],
    ],
    "A treatment can work in a dish or animal model and still fail in patients. Human disease includes organism-scale context, heterogeneity, dosing realities, toxicity, and outcomes that simplified models may not capture.",
  ),
  makeQuestion(
    50,
    "medium",
    "Which statements correctly describe useful and risky AI roles in biomedicine?",
    [
      [
        "AI can help with protein structure, genomics, image analysis, drug discovery, and patient stratification.",
        true,
      ],
      [
        "AI can fail when data are biased, labels are noisy, measurements are unreliable, or populations shift.",
        true,
      ],
      [
        "AI predictions require biological interpretation and evidence before clinical reliance.",
        true,
      ],
      [
        "AI is most useful when paired with mechanism, measurement quality, workflow fit, safety, and validation.",
        true,
      ],
    ],
    "AI can accelerate biomedical work, but it does not bypass biology or evidence. Common bottlenecks include biological understanding, reliable measurement, data quality, workflow integration, regulation, safety, and validation.",
  ),
];

export const BiologyChemistryLifeScienceL5Questions =
  BiologyChemistryForLifeScienceLecture5Questions;
