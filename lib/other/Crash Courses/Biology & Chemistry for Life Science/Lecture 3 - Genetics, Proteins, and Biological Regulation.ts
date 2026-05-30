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
    throw new Error(`Lecture 3 question ${number} must have four options.`);
  }

  return {
    id: `bio-chem-life-l3-q${String(number).padStart(2, "0")}`,
    chapter: 3,
    difficulty,
    prompt,
    options: optionSeeds.map(([text, isCorrect]) => ({ text, isCorrect })),
    explanation,
  };
}

export const BiologyChemistryForLifeScienceLecture3Questions: Question[] = [
  makeQuestion(
    1,
    "easy",
    "Which statements correctly describe DNA as an information molecule?",
    [
      ["DNA stores information in nucleotide sequence.", true],
      ["Complementary base pairing helps DNA be copied and repaired.", true],
      ["DNA is chemically stable enough to support inheritance.", true],
      [
        "DNA influences cellular work through RNA, proteins, and regulated information use.",
        true,
      ],
    ],
    "DNA is central because it stores inheritable sequence information. Most cellular execution is performed by proteins and RNAs whose production and regulation depend on how DNA information is accessed and used.",
  ),
  makeQuestion(
    2,
    "easy",
    "Which statement best describes a gene?",
    [
      [
        "A DNA sequence that can be used to produce a functional product, often a protein or functional RNA.",
        true,
      ],
      ["A finished protein that has already folded in the cytoplasm.", false],
      [
        "A lipid bilayer that controls transport across the cell membrane.",
        false,
      ],
      ["A clinical-trial endpoint used to measure patient survival.", false],
    ],
    "A gene is a usable information segment within DNA, not the final protein or the whole biological outcome. Genes influence traits through regulated expression, RNA, protein function, environment, and interaction with other systems.",
  ),
  makeQuestion(
    3,
    "easy",
    "Which pairings in the central dogma are correct?",
    [
      ["Transcription: DNA information is copied into RNA.", true],
      [
        "Translation: ribosomes use messenger RNA to build an amino acid chain.",
        true,
      ],
      ["Codon: a lipid bilayer that stores ATP in mitochondria.", false],
      [
        "Replication: proteins are converted directly into DNA sequence by ribosomes.",
        false,
      ],
    ],
    "The central dogma is DNA -> RNA -> protein, with replication copying DNA. Codons are three-nucleotide information units, not lipid bilayers, and later questions use that vocabulary directly.",
  ),
  makeQuestion(
    4,
    "easy",
    "Which statements correctly describe RNA?",
    [
      [
        "Messenger RNA can carry protein-coding instructions from DNA to ribosomes.",
        true,
      ],
      [
        "RNA can be shorter-lived and more context-dependent than genomic DNA.",
        true,
      ],
      [
        "Some RNAs have regulatory or functional roles beyond carrying protein instructions.",
        true,
      ],
      ["Some viruses use RNA as their genetic material.", true],
    ],
    "RNA is a versatile working layer in biology. Messenger RNA is especially important for protein production and mRNA medicines, but other RNAs can regulate, catalyze, process, or support cellular information flow.",
  ),
  makeQuestion(
    5,
    "easy",
    "Which statement best corrects the phrase DNA is a blueprint?",
    [
      [
        "DNA is better understood as an inherited information library that cells regulate and interpret in context.",
        true,
      ],
      [
        "DNA is irrelevant because proteins appear without sequence information.",
        false,
      ],
      [
        "DNA contains one deterministic switch for every complex trait with no environmental effects.",
        false,
      ],
      [
        "DNA is a clinical drug dose table rather than a biological information molecule.",
        false,
      ],
    ],
    "The blueprint metaphor hides regulation, context, environment, and networks. DNA provides information, but cell behavior depends on which genes are expressed, how products function, and what signals and environments are present.",
  ),
  makeQuestion(
    6,
    "easy",
    "Which statements correctly describe gene expression?",
    [
      [
        "Gene expression refers to how much or when a gene product is made.",
        true,
      ],
      [
        "Cells can change gene expression without changing the underlying DNA sequence.",
        true,
      ],
      [
        "Different cell types must express exactly the same genes at exactly the same levels.",
        false,
      ],
      [
        "Gene expression means every gene is active at the same level in every cell.",
        false,
      ],
    ],
    "Gene expression is regulated use of genetic information. It explains why different cells can behave differently even when their DNA sequence is mostly the same, because expression levels and active programs differ.",
  ),
  makeQuestion(
    7,
    "easy",
    "Which statements correctly describe transcription factors?",
    [
      ["They are often proteins that help regulate gene transcription.", true],
      ["They can help turn genes on or off in particular contexts.", true],
      [
        "They cannot be influenced by signals or other regulatory proteins.",
        false,
      ],
      [
        "They are the same thing as carbohydrates used only for short-term energy storage.",
        false,
      ],
    ],
    "Transcription factors are part of the control logic of gene expression. They can connect signals and cell state to DNA regulatory regions, so the claim that they cannot be influenced by regulatory context is wrong.",
  ),
  makeQuestion(
    8,
    "easy",
    "Which statements correctly describe mutation and evolution?",
    [
      ["A mutation is a change in DNA sequence.", true],
      [
        "Evolution can occur when inherited variation affects survival or reproduction.",
        true,
      ],
      ["Genetic drift can change variant frequencies by chance.", true],
      ["Selection can act in pathogens, cancers, and populations.", true],
    ],
    "Evolution is not only a historical topic. Mutation, selection, drift, and inherited variation are active in medicine through infectious disease, cancer, resistance, and inherited risk.",
  ),
  makeQuestion(
    9,
    "easy",
    "Which statement best describes epigenetic regulation?",
    [
      [
        "Regulation of access to genetic information without changing the underlying DNA sequence.",
        true,
      ],
      ["Replacement of every nucleotide with an amino acid.", false],
      [
        "A process that makes all genes impossible to transcribe forever.",
        false,
      ],
      ["The same process as bacterial cell-wall synthesis.", false],
    ],
    "Epigenetic regulation affects how genetic information is packaged, accessed, and inherited across cell states. It should not be treated as mystical rewriting of the genome or as a synonym for mutation.",
  ),
  makeQuestion(
    10,
    "easy",
    "Which statements correctly describe biotechnology tools at a high level?",
    [
      ["Sequencing reads DNA order.", true],
      ["CRISPR-based tools can target DNA using guide sequences.", true],
      [
        "mRNA medicines can deliver temporary instructions for protein production.",
        true,
      ],
      [
        "Gene therapy can aim to add, replace, silence, or edit genetic information.",
        true,
      ],
    ],
    "Modern biotechnology is powerful because biological information is readable, writable, editable, and expressible. Each tool has limits, but each depends on the core DNA, RNA, protein, and regulation concepts in this lecture.",
  ),
  makeQuestion(
    11,
    "medium",
    "A neuron and a liver cell contain nearly the same genome but perform very different functions. Which explanations are correct?",
    [
      ["They express different sets of genes.", true],
      ["They maintain different regulatory states and protein networks.", true],
      [
        "Their differentiation history cannot affect which programs are accessible.",
        false,
      ],
      [
        "Their difference requires every gene sequence to be replaced in one of the cell types.",
        false,
      ],
    ],
    "Cell identity depends strongly on regulation rather than wholesale genome replacement. Differentiated cells use different transcription factors, chromatin states, signaling contexts, and protein networks to produce different behavior.",
  ),
  makeQuestion(
    12,
    "medium",
    "Which interpretation best explains why codons contain three nucleotides?",
    [
      [
        "Four bases read two at a time give only 16 combinations, while three at a time gives 64, enough to encode 20 amino acids plus stop signals.",
        true,
      ],
      [
        "Three nucleotides are required because proteins contain exactly three amino acids.",
        false,
      ],
      [
        "Codons contain three nucleotides because DNA has three total bases.",
        false,
      ],
      ["A codon is three proteins arranged in a lipid membrane.", false],
    ],
    "The triplet code is an information-capacity solution. With four possible bases, three-position codons provide enough combinations to encode the amino acid alphabet and stop signals with redundancy.",
  ),
  makeQuestion(
    13,
    "medium",
    "Which statements correctly connect DNA sequence to protein function?",
    [
      [
        "DNA can be transcribed into RNA that is translated into amino acid sequence.",
        true,
      ],
      [
        "Amino acid sequence can influence folding, binding, and catalysis.",
        true,
      ],
      [
        "DNA affects traits only through direct electrical signals sent from chromosomes to organs.",
        false,
      ],
      [
        "Every DNA sequence is automatically translated into a functional protein in every cell.",
        false,
      ],
    ],
    "DNA can influence protein function through the central dogma, but only if the sequence is expressed and processed in the relevant context. Expression, RNA handling, translation, folding, modification, and cellular environment all matter.",
  ),
  makeQuestion(
    14,
    "medium",
    "Which statements correctly describe plasmids in bacteria and biotechnology?",
    [
      ["Plasmids are small DNA molecules that can carry genes.", true],
      ["Plasmids cannot carry antibiotic resistance genes.", false],
      [
        "Engineered plasmids can be used to introduce genes into bacteria for cloning or expression.",
        true,
      ],
      ["Plasmids are the same as human red blood cells.", false],
    ],
    "Plasmids matter medically and technologically because they can carry genes and can be engineered as vectors. Some plasmids do carry antibiotic resistance genes, so saying they cannot do so is incorrect.",
  ),
  makeQuestion(
    15,
    "medium",
    "Which statements correctly explain recombinant insulin production?",
    [
      ["A human insulin gene can be inserted into a DNA vector.", true],
      [
        "Production cells can express the inserted instructions and make insulin protein.",
        true,
      ],
      [
        "The protein can be used as medicine without purification or quality checks.",
        false,
      ],
      [
        "The process works because bacteria naturally contain finished human pancreas tissue.",
        false,
      ],
    ],
    "Recombinant insulin is a clear example of using biological information and cellular machinery. Cells read inserted genetic instructions and produce a protein, but medicine production still requires purification and quality control.",
  ),
  makeQuestion(
    16,
    "medium",
    "Which statement best distinguishes mutation from epigenetic regulation?",
    [
      [
        "Mutation changes DNA sequence, while epigenetic regulation changes access or usage of DNA without changing the sequence.",
        true,
      ],
      [
        "Mutation means messenger RNA is translated, while epigenetic regulation means proteins are hydrolyzed.",
        false,
      ],
      [
        "Mutation and epigenetic regulation are both names for ATP production in mitochondria.",
        false,
      ],
      [
        "Epigenetic regulation always deletes chromosomes, while mutation never affects sequence.",
        false,
      ],
    ],
    "Mutation and epigenetic regulation can both affect biology, but through different mechanisms. One changes the stored sequence, while the other changes how information is packaged, accessed, or maintained in cell states.",
  ),
  makeQuestion(
    17,
    "medium",
    "Which statements correctly describe antibiotic resistance as evolution?",
    [
      [
        "Bacterial populations can contain variants with different susceptibility.",
        true,
      ],
      ["Antibiotic exposure can select for resistant variants.", true],
      [
        "Resistance genes cannot spread through plasmids or horizontal gene transfer.",
        false,
      ],
      [
        "Resistance appears because individual bacteria intentionally decide to become immune after seeing a drug.",
        false,
      ],
    ],
    "Resistance is population-level evolution under selection pressure. Gene transfer can spread resistance, so the option denying plasmids and horizontal transfer is the misconception.",
  ),
  makeQuestion(
    18,
    "medium",
    "Which statements correctly describe mRNA medicines?",
    [
      [
        "They can provide temporary instructions for cells to make a protein.",
        true,
      ],
      ["They do not use the cell's translation machinery.", false],
      [
        "They cannot be used in vaccine strategies that encode an antigen.",
        false,
      ],
      [
        "They must permanently replace the entire genome in every treated cell to work.",
        false,
      ],
    ],
    "Messenger RNA approaches use a temporary information molecule, not permanent replacement of all genomic DNA. They rely on translation machinery, and some vaccine strategies use mRNA to encode an antigen.",
  ),
  makeQuestion(
    19,
    "medium",
    "Which statements correctly describe genetic drift?",
    [
      ["It is random change in variant frequencies.", true],
      ["It is especially important in small populations.", true],
      ["It can change populations without improving adaptation.", true],
      ["It is identical to deliberate genome editing with CRISPR.", false],
    ],
    "Genetic drift reminds students that evolution is not always directed improvement. Chance events can change variant frequencies, especially when populations are small or pass through bottlenecks.",
  ),
  makeQuestion(
    20,
    "medium",
    "Which statement best describes gene therapy?",
    [
      [
        "A treatment strategy that aims to add, replace, silence, or edit genetic information to alter disease-relevant biology.",
        true,
      ],
      ["Any drug that blocks a receptor for ten minutes.", false],
      ["A diagnostic test that only measures blood pressure.", false],
      ["The process by which ribosomes digest carbohydrates.", false],
    ],
    "Gene therapy intervenes at the genetic-information layer, but it still must solve delivery, durability, safety, tissue targeting, and patient-selection problems. It is not a synonym for all drugs or all diagnostics.",
  ),
  makeQuestion(
    21,
    "hard",
    "A DNA variant is associated with higher disease risk, but most carriers never develop the disease. Which interpretations are reasonable?",
    [
      [
        "The variant may increase probability rather than deterministically cause disease.",
        true,
      ],
      ["Environment, regulation, other genes, and time may modify risk.", true],
      [
        "The association could require further validation in different populations.",
        true,
      ],
      [
        "The variant proves that every carrier must have identical symptoms at birth.",
        false,
      ],
    ],
    "Genetic risk is often probabilistic. Penetrance, modifier genes, environment, measurement, population structure, and disease definition can all affect whether an associated variant predicts actual disease.",
  ),
  makeQuestion(
    22,
    "hard",
    "A CRISPR edit cuts the intended DNA site, but treated cells show mixed outcomes. Which explanations are plausible?",
    [
      [
        "Different repair pathways can produce different final sequence changes.",
        true,
      ],
      ["Editing and delivery may not occur in every target cell.", true],
      ["Off-target or unintended effects may need to be checked.", true],
      [
        "A guide sequence guarantees a perfectly identical outcome in every cell and patient.",
        false,
      ],
    ],
    "Genome editing involves targeting plus cellular repair, delivery, and biological context. A successful cut is not the same as a uniform therapeutic result, which is why editing outcomes and safety need careful validation.",
  ),
  makeQuestion(
    23,
    "hard",
    "Which statements correctly synthesize gene regulation with signaling?",
    [
      ["External signals can change transcription factor activity.", true],
      ["Transcription factors can alter gene expression programs.", true],
      ["Gene-expression changes can alter future signaling responses.", true],
      [
        "Regulatory networks can include feedback loops across proteins, RNA, and DNA accessibility.",
        true,
      ],
    ],
    "Regulation is networked and dynamic. Signals can change gene expression, gene-expression programs can change receptor levels or pathway components, and feedback can stabilize, amplify, or switch cell states.",
  ),
  makeQuestion(
    24,
    "hard",
    "A tumor shrinks during targeted therapy and then later regrows with resistant cells. Which interpretation is best?",
    [
      [
        "Therapy can impose selection pressure that favors resistant tumor subclones.",
        true,
      ],
      [
        "The initial shrinkage proves evolution cannot occur inside a tumor.",
        false,
      ],
      [
        "Resistance means no genetic or regulatory diversity existed before treatment.",
        false,
      ],
      [
        "Regrowth proves the drug never interacted with a biological target.",
        false,
      ],
    ],
    "Tumors can contain diverse cell populations. A treatment may kill sensitive cells while resistant cells survive, expand, or acquire additional adaptations, making cancer evolution clinically important.",
  ),
  makeQuestion(
    25,
    "hard",
    "Which statements correctly identify limitations of one-gene one-trait reasoning?",
    [
      ["Many traits depend on multiple genes and environmental context.", true],
      [
        "A gene product can participate in networks with other molecules.",
        true,
      ],
      [
        "One gene always maps to one visible trait with no regulation or interaction.",
        false,
      ],
      [
        "Changing expression level can matter even when protein sequence is unchanged.",
        true,
      ],
    ],
    "Simple Mendelian examples are useful, but many biomedical traits are networked, regulated, and context-dependent. Expression level, timing, cell type, environment, and interacting pathways can matter as much as coding sequence.",
  ),
  makeQuestion(
    26,
    "hard",
    "Which statements correctly describe why sequencing is powerful but not sufficient by itself?",
    [
      [
        "Sequencing can identify variants, pathogens, or gene-expression-related information depending on method.",
        true,
      ],
      [
        "Interpreting a sequence often requires functional, clinical, or population context.",
        true,
      ],
      [
        "A sequence difference does not automatically prove disease causation.",
        true,
      ],
      [
        "Sequencing alone directly measures every protein activity, metabolite, and clinical outcome without additional data.",
        false,
      ],
    ],
    "Sequencing is foundational, but raw sequence is not the same as mechanism or clinical meaning. Interpretation requires annotation, validation, biological context, and often other measurements.",
  ),
  makeQuestion(
    27,
    "hard",
    "Which statements correctly explain why protein-coding gene count does not fully explain organism complexity?",
    [
      ["Regulatory networks can reuse genes in different contexts.", true],
      [
        "Alternative RNA processing and protein regulation can diversify outcomes.",
        true,
      ],
      ["Interactions among cells and tissues create emergent behavior.", true],
      [
        "Organism complexity is determined only by the number of protein-coding genes and nothing else.",
        false,
      ],
    ],
    "Complexity comes from regulation, networks, interactions, timing, and context, not simply a larger list of protein-coding genes. This is why gene count alone is a weak measure of biological sophistication.",
  ),
  makeQuestion(
    28,
    "hard",
    "Which statements correctly compare mRNA therapy and genome editing?",
    [
      ["mRNA therapy usually provides temporary instructions.", true],
      [
        "Genome editing aims to change DNA sequence or regulation more durably.",
        true,
      ],
      ["Both require delivery to relevant cells.", true],
      [
        "Both need safety and efficacy evidence in the biological context where they are used.",
        true,
      ],
    ],
    "mRNA and genome editing both intervene in biological information flow, but at different layers and with different durability. Delivery, immune response, dose, expression level, target tissue, and evidence all matter.",
  ),
  makeQuestion(
    29,
    "hard",
    "Which statements correctly describe horizontal gene transfer in bacteria?",
    [
      [
        "Genes can move between bacteria outside parent-to-offspring inheritance.",
        true,
      ],
      ["Plasmids can be one vehicle for transferable traits.", true],
      ["Horizontal transfer can spread antibiotic resistance.", true],
      [
        "Horizontal gene transfer is the same as human meiosis producing eggs and sperm.",
        false,
      ],
    ],
    "Horizontal gene transfer is one reason bacterial evolution can be rapid and clinically important. It differs from vertical inheritance and can move traits such as resistance across bacterial lineages.",
  ),
  makeQuestion(
    30,
    "hard",
    "Which statements correctly connect Lecture 3 to modern medicine and biotechnology?",
    [
      ["Genetic information can be read through sequencing.", true],
      [
        "Gene products can be manufactured through recombinant expression.",
        true,
      ],
      [
        "Gene regulation helps explain cell identity, disease state, and therapy response.",
        true,
      ],
      [
        "Evolution helps explain pathogen, tumor, and resistance dynamics.",
        true,
      ],
    ],
    "Genetics is not just heredity vocabulary. It provides the information layer for biotechnology, diagnostics, cancer, infection, pharmacology, evolution, and AI models of biological systems.",
  ),
];

export const BiologyChemistryLifeScienceL3Questions =
  BiologyChemistryForLifeScienceLecture3Questions;
