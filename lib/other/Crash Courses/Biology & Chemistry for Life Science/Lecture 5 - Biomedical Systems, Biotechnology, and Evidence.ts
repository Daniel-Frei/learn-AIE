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
      [
        "Antibiotics usually block viral receptor binding more specifically than antiviral drugs do.",
        false,
      ],
      [
        "Viruses normally carry bacterial-type ribosomes, but antibiotics cannot slow them enough during infection.",
        false,
      ],
      [
        "Antibiotics mainly train adaptive immune memory rather than interrupting microbial replication.",
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
        "Completing genome replication and particle assembly without entering or using a cell.",
        false,
      ],
      [
        "Synthesizing viral proteins on ribosomes packaged inside the viral particle while outside cells.",
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
        "A short-term stress response that appears uniformly in every exposed microbe and is not inherited.",
        false,
      ],
      [
        "A treatment effect where susceptible variants become more common because the drug favors them.",
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
        "Gene insertion alone makes the final protein medicine clinically usable without purification or testing.",
        false,
      ],
      [
        "Recombinant production uses cellular machinery for transcription and translation.",
        true,
      ],
      [
        "The same production host always gives every human protein identical folding, modification, and impurities.",
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
        "They are chosen because simplified systems reproduce all human dosing, toxicity, and heterogeneity.",
        false,
      ],
      [
        "They replace human evidence once a plausible molecular mechanism has been observed.",
        false,
      ],
      [
        "They are useful only when they exclude living cells, molecules, tissues, or biological readouts.",
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
        "It is useful for infection only when it stays hidden inside the viral genome and away from host receptors.",
        false,
      ],
      [
        "It is useful for vaccination because antibodies recognize the host receptor instead of the viral surface.",
        false,
      ],
      [
        "It becomes a vaccine target only after losing the binding shape involved in infection.",
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
      [
        "They mostly block bacterial peptidoglycan synthesis and rely on that to stop viral entry later.",
        false,
      ],
      [
        "They work by giving infected cells a general toxicity signal without targeting viral replication steps.",
        false,
      ],
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
        "mRNA medicine usually aims to alter DNA sequence in every treated cell for durable inheritance.",
        false,
      ],
      [
        "Gene editing mainly supplies a transient RNA template that ribosomes translate and then degrade.",
        false,
      ],
      [
        "mRNA and gene editing have the same durability and off-target profile because both use nucleic acids.",
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
        "CAR T-cell therapy gives soluble antibodies without transferring living immune cells.",
        false,
      ],
      [
        "Cell therapies avoid persistence and safety questions because infused cells stop interacting after delivery.",
        false,
      ],
      [
        "Cell therapy means delivering messenger RNA instructions while never transferring cells.",
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
        "The model result should override patient symptoms because biomarkers are closer to mechanism.",
        false,
      ],
      [
        "The failure proves the drug target is irrelevant in every model and disease setting.",
        false,
      ],
      [
        "The patient endpoint must be wrong because animal biomarkers cannot mislead translation.",
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
    "Which statement best describes a pathogen in biomedical systems?",
    [
      [
        "A biological agent that can cause disease through interaction with a host, such as a bacterium, virus, fungus, or parasite.",
        true,
      ],
      [
        "Any microbe carried by a person, including harmless commensals, regardless of disease potential.",
        false,
      ],
      [
        "Any immune molecule that recognizes an antigen, whether or not disease occurs.",
        false,
      ],
      [
        "A treatment pressure that makes susceptible variants expand faster than resistant variants.",
        false,
      ],
    ],
    "A pathogen is defined by its disease-causing relationship with a host, not merely by being biological or microscopic. Harmless microbes, immune molecules, and treatment pressures can all matter in biomedicine, but they are not themselves pathogens by that definition.",
  ),
  makeQuestion(
    32,
    "easy",
    "Which statement best describes a retrovirus?",
    [
      [
        "An RNA virus that can copy RNA into DNA before integration into a host genome.",
        true,
      ],
      [
        "A virus that reproduces independently because RNA gives it complete metabolism.",
        false,
      ],
      [
        "A bacterium that stores antibiotic-resistance genes on plasmids.",
        false,
      ],
      [
        "A vaccine vector that cannot enter host cells or express genetic information.",
        false,
      ],
    ],
    "Some viruses use RNA and some use DNA, and retroviruses are a special RNA-virus case because reverse transcription creates DNA that can integrate into the host genome. Plasmids, vaccine vectors, and ordinary viral replication are related topics, but they do not define a retrovirus.",
  ),
  makeQuestion(
    33,
    "easy",
    "Which statement best describes what vaccine platforms have in common?",
    [
      [
        "They expose the immune system to an antigen or antigen-making instructions so adaptive memory can form before dangerous exposure.",
        true,
      ],
      [
        "They work only by directly killing pathogens after infection begins.",
        false,
      ],
      [
        "They are useful only when they avoid B-cell and T-cell memory formation.",
        false,
      ],
      [
        "They replace diagnostics because trained immunity proves a person is uninfected.",
        false,
      ],
    ],
    "Vaccine platforms differ, but the shared idea is safer immune training before a dangerous infection. Vaccines do not classify current infection, and their value depends heavily on adaptive immune recognition and memory rather than direct drug-like killing after disease starts.",
  ),
  makeQuestion(
    34,
    "easy",
    "Which statement best describes B cells, antibodies, and immune memory?",
    [
      [
        "B cells can produce antibodies, and some B and T cells become memory cells for faster later responses.",
        true,
      ],
      [
        "Antibodies are enzymes that copy viral genomes so immune cells can read them.",
        false,
      ],
      [
        "Immune memory means the original pathogen remains actively replicating as a reminder.",
        false,
      ],
      [
        "Memory T cells mainly function by secreting free antibodies with the same role as B cells.",
        false,
      ],
    ],
    "Adaptive immunity includes antibody-producing B-cell responses and memory cells that support faster later responses. Antibodies bind antigens; they do not copy viral genomes, and immune memory does not require persistent active infection.",
  ),
  makeQuestion(
    35,
    "easy",
    "Which statement best describes when a diagnostic is useful?",
    [
      [
        "It is accurate enough, in the relevant context, for the decision it is meant to support.",
        true,
      ],
      [
        "Any measurable biomarker is automatically a diagnostic for every disease in which it changes.",
        false,
      ],
      [
        "Its usefulness is independent of sampling method, patient population, and false-result consequences.",
        false,
      ],
      [
        "Disease prevalence can change disease frequency but not the interpretation of test results.",
        false,
      ],
    ],
    "Diagnostics translate biological information into classifications or measurements for decisions. A test can be technically impressive but clinically weak if the wrong population, sample, threshold, prevalence, or decision context changes the meaning of its results.",
  ),
  makeQuestion(
    36,
    "easy",
    "Which statements correctly compare major pathogen types?",
    [
      [
        "Pathogen type affects replication strategy, immune response, and treatment choice.",
        true,
      ],
      [
        "Bacterial antigens and viral proteins can both matter for immune recognition or diagnosis.",
        true,
      ],
      [
        "All pathogens are living cells with bacterial ribosomes and bacterial metabolism.",
        false,
      ],
      [
        "All symptoms during infection come directly from pathogen damage rather than host response.",
        false,
      ],
    ],
    "Bacteria, viruses, fungi, and parasites differ in structure, replication, treatment options, and immune interactions. Symptoms can arise from pathogen damage, immune response, inflammation, tissue repair, or disrupted physiology, so pathogen identity is only part of the disease picture.",
  ),
  makeQuestion(
    37,
    "easy",
    "Which statements correctly describe the basic viral life cycle?",
    [
      ["Receptor binding helps determine which cells a virus can enter.", true],
      [
        "After entry, viral genetic material must be copied and viral proteins produced to assemble particles.",
        true,
      ],
      [
        "A virion usually completes protein synthesis before contact with a host cell.",
        false,
      ],
      [
        "Viral exit is normally when the virus becomes a free-living cell with its own ribosomes.",
        false,
      ],
    ],
    "A simplified viral pattern is receptor binding, entry, genome release, replication and protein production, assembly, and exit. Viruses do not become self-sufficient cells; they rely on host-cell machinery to make the components needed for new particles.",
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
        "A drug's antimicrobial label determines its target better than the pathogen's biology does.",
        false,
      ],
      [
        "The same bacterial ribosome target explains most antiviral and antibiotic activity.",
        false,
      ],
    ],
    "Antibiotics and antivirals target different biological vulnerabilities. The same symptom pattern can come from different pathogens, so treatment reasoning starts with the pathogen's structure and life cycle rather than a generic antimicrobial category.",
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
      [
        "Their size and specificity often favor extracellular or cell-surface targets.",
        true,
      ],
      [
        "They are defined by chemical synthesis into tiny oral molecules with no protein structure.",
        false,
      ],
      [
        "Their manufacturing is usually simpler than small molecules because living systems add no variability.",
        false,
      ],
    ],
    "Biologics are large therapeutic molecules such as antibodies and proteins, often made with living systems. Their specificity can be powerful, but size, delivery route, immune reactions, and manufacturing variability create constraints that differ from small molecules.",
  ),
  makeQuestion(
    40,
    "easy",
    "Which statements correctly describe model systems in biomedical research?",
    [
      [
        "Cell cultures, organoids, animal models, human observational data, and computational models are simplified representations.",
        true,
      ],
      [
        "Model systems can isolate mechanisms or feasibility before a patient claim is tested.",
        true,
      ],
      [
        "A model result proves patient benefit as long as it changes a biological readout.",
        false,
      ],
      [
        "A good model is defined by excluding dosing, toxicity, heterogeneity, and tissue context from the question.",
        false,
      ],
    ],
    "Model systems are useful because they make biological questions tractable under controlled conditions. Their limits come from simplification: dosing, exposure, tissue context, immune context, heterogeneity, toxicity, and patient outcomes may not match the model readout.",
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
      [
        "Immune memory means antibodies permanently remove the need for innate or current immune responses.",
        false,
      ],
    ],
    "Adaptive immunity can learn and remember molecular targets, which is why vaccines can prepare the immune system before dangerous exposure. Memory improves later responses, but current immune activation, innate defense, and infection context still matter.",
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
        "Purification is optional because the correct DNA sequence guarantees a final medicine's identity, folding, and safety.",
        false,
      ],
    ],
    "Recombinant DNA uses cellular information flow for production. A gene can be inserted into bacteria or other production cells, expressed as a protein, and then the product still needs purification and quality control before it can be used as medicine.",
  ),
  makeQuestion(
    43,
    "easy",
    "Which statements correctly describe small-molecule drugs?",
    [
      ["They are chemically manufactured compounds.", true],
      ["They can sometimes enter cells and bind intracellular proteins.", true],
      ["Oral delivery is sometimes possible.", true],
      [
        "Being small by itself determines target selectivity and safety, so target biology is secondary.",
        false,
      ],
    ],
    "Small molecules are chemically manufactured and can sometimes reach intracellular targets, with oral delivery sometimes possible. Their value still depends on target fit, exposure, selectivity, metabolism, and toxicity; small size alone does not guarantee safety.",
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
      [
        "They avoid genetic information and work chiefly by blocking extracellular receptors.",
        false,
      ],
    ],
    "Gene therapy and genome editing act at the genetic-information layer by adding, replacing, silencing, or modifying genetic information. Their promise depends on getting the right change into the right cells with acceptable safety, durability, reversibility, and ethical justification.",
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
        "An endpoint should be chosen after results are known so it can match the strongest observed change.",
        false,
      ],
    ],
    "Clinical evidence asks whether biomedical claims hold up in humans. Control groups, randomization, blinding, and pre-specified endpoints help separate plausible mechanisms or associations from trustworthy conclusions about patient outcomes.",
  ),
  makeQuestion(
    46,
    "medium",
    "Which statements correctly describe why antibiotics do not treat viral infections?",
    [
      [
        "Antibiotic targets such as peptidoglycan synthesis are absent from viruses.",
        true,
      ],
      [
        "Bacterial ribosome differences can be drug targets, but viruses use host translation machinery.",
        true,
      ],
      [
        "Viral infections usually require prevention, immune response, supportive care, or antivirals aimed at viral steps rather than bacterial cell processes.",
        true,
      ],
      [
        "Knowing whether illness is bacterial or viral changes treatment reasoning and resistance stewardship.",
        true,
      ],
    ],
    "Antibiotics target bacterial biology, not viral replication strategies. Because viruses use host-cell machinery and virus-specific life-cycle steps, treatment reasoning depends on diagnosis, prevention, immune response, supportive care, and antiviral mechanisms when available.",
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
        "Cell therapy can involve selecting, expanding, or engineering cells before giving them to a patient.",
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
    "Therapeutic modalities intervene at different biological layers. Comparing them requires thinking about molecule size, production system, delivery route, target location, durability, immune response, safety, patient selection, and evidence.",
  ),
  makeQuestion(
    48,
    "medium",
    "Which statements correctly describe biomarker types?",
    [
      [
        "Diagnostic and prognostic biomarkers support different claims: identifying disease versus estimating likely course.",
        true,
      ],
      [
        "A predictive biomarker can help predict response to a particular treatment.",
        true,
      ],
      [
        "A pharmacodynamic biomarker can show biological effect of a drug.",
        true,
      ],
      ["A safety biomarker can signal potential harm.", true],
    ],
    "Biomarkers are measurable indicators of biological state, but their meaning depends on the decision they support. Diagnostic, prognostic, predictive, pharmacodynamic, and safety uses are different claims and each needs validation in context.",
  ),
  makeQuestion(
    49,
    "medium",
    "Which statements correctly explain why preclinical results can fail to translate to humans?",
    [
      [
        "A model may omit immune, endocrine, vascular, or tissue context.",
        true,
      ],
      ["Dosing and exposure may differ between the model and humans.", true],
      [
        "Animal biology and patient disease heterogeneity may differ from the study system.",
        true,
      ],
      [
        "Toxicity and patient-relevant endpoints may appear only at organism or clinical scale.",
        true,
      ],
    ],
    "A treatment can work in a dish or animal model and still fail in patients. Human disease includes organism-scale context, exposure, heterogeneity, toxicity, and outcomes that simplified systems may not capture.",
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
        "AI can fail when data are biased, labels are noisy, measurements are unreliable, correlations are noncausal, or populations shift.",
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
    "AI can accelerate biomedical work, but it does not bypass biology or evidence. Common bottlenecks include biological understanding, reliable measurement, data quality, workflow integration, regulation, safety, trust, and validation.",
  ),
];

export const BiologyChemistryLifeScienceL5Questions =
  BiologyChemistryForLifeScienceLecture5Questions;
