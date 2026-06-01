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
        "DNA has little practical role once proteins are present in the cell.",
        false,
      ],
      [
        "DNA contains a direct switch for each complex trait that works independently of environment.",
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
        "Gene expression means a cell runs a uniform expression program across its genome.",
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
        "They operate separately from signals and other regulatory proteins.",
        false,
      ],
      [
        "They are carbohydrate storage molecules rather than transcription regulators.",
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
      ["Replacement of nucleotide sequences with amino acid chains.", false],
      [
        "A process that locks genes into a permanently inaccessible state across the genome.",
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
        "Their differentiation history has little influence on which programs are accessible.",
        false,
      ],
      [
        "Their difference requires wholesale replacement of one cell type's gene sequences.",
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
        "DNA affects traits through direct electrical signals sent from chromosomes to organs.",
        false,
      ],
      [
        "A DNA sequence becomes a functional protein regardless of expression state, codon context, or cell type.",
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
      [
        "Plasmids are mainly inert DNA loops unrelated to transferable resistance traits.",
        false,
      ],
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
        "Epigenetic regulation deletes chromosomes, while mutation leaves sequence unchanged.",
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
        "Resistance genes spread vertically but are disconnected from plasmids and horizontal gene transfer.",
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
        "They are unsuitable for vaccine strategies that encode an antigen.",
        false,
      ],
      [
        "They work by replacing the treated cell's genome with a permanent RNA-derived chromosome.",
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
      [
        "A diagnostic workflow that reports blood pressure without altering disease biology.",
        false,
      ],
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
        "The variant implies identical symptoms at birth for carriers across environments.",
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
        "A guide sequence is sufficient to produce uniform repair outcomes across cells and patients.",
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
        "The initial shrinkage rules out resistant subclones as a later clinical problem.",
        false,
      ],
      [
        "Resistance means no genetic or regulatory diversity existed before treatment.",
        false,
      ],
      [
        "Regrowth indicates the drug had no biologically relevant target interaction.",
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
        "One gene maps directly to one visible trait without regulation or interaction.",
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
        "Sequencing alone directly measures protein activity, metabolite levels, and clinical outcome.",
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
        "Organism complexity is determined mainly by protein-coding gene count rather than regulation or interaction.",
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
  makeQuestion(
    31,
    "easy",
    "Which statement best describes DNA as an information molecule?",
    [
      ["DNA stores information in nucleotide sequence.", true],
      ["DNA stores information mainly as folded enzyme shapes.", false],
      [
        "DNA stores information by changing into ATP during cell division.",
        false,
      ],
      [
        "DNA stores information as a membrane gradient across mitochondria.",
        false,
      ],
    ],
    "DNA stores inherited biological information in the order of its nucleotide bases. The chemical structure is stable enough to preserve information, while the sequence can be copied, read, and regulated.",
  ),
  makeQuestion(
    32,
    "easy",
    "Which statement best describes complementary base pairing in DNA?",
    [
      ["A pairs with T, and G pairs with C.", true],
      ["A pairs with G, and T pairs with C.", false],
      ["A pairs with C, and G pairs with T.", false],
      [
        "Bases pair according to amino acid size rather than nucleotide chemistry.",
        false,
      ],
    ],
    "Complementary base pairing gives DNA a predictable copying rule. Adenine pairs with thymine, and guanine pairs with cytosine, supporting replication, repair, inheritance, and molecular recognition.",
  ),
  makeQuestion(
    33,
    "easy",
    "Which statement best defines a gene?",
    [
      [
        "A gene is a DNA sequence that can be used to produce a functional product.",
        true,
      ],
      ["A gene is a complete organism's set of chromosomes.", false],
      ["A gene is a protein machine that reads messenger RNA.", false],
      [
        "A gene is a random mutation that improves survival in a population.",
        false,
      ],
    ],
    "A gene is a DNA sequence that can produce a functional product, often a protein or functional RNA. Genes influence traits through regulated expression, product function, cellular context, environment, and interactions with other genes.",
  ),
  makeQuestion(
    34,
    "easy",
    "Which statement best describes transcription?",
    [
      ["Transcription copies a DNA region into RNA.", true],
      [
        "Transcription builds a protein by reading codons at a ribosome.",
        false,
      ],
      ["Transcription separates copied chromosomes during mitosis.", false],
      [
        "Transcription inserts a plasmid into bacteria for recombinant production.",
        false,
      ],
    ],
    "Transcription is the process of making RNA from a DNA template. RNA polymerase reads DNA and produces RNA, including messenger RNA that can carry protein-coding instructions.",
  ),
  makeQuestion(
    35,
    "easy",
    "Which statement best describes translation?",
    [
      ["Translation builds a protein from a messenger RNA sequence.", true],
      ["Translation copies DNA into a shorter-lived RNA working copy.", false],
      [
        "Translation changes chromatin packing without changing DNA sequence.",
        false,
      ],
      [
        "Translation randomly changes variant frequencies in a small population.",
        false,
      ],
    ],
    "Translation is the process in which ribosomes read messenger RNA and build an amino acid chain. It connects nucleotide instructions to protein sequence.",
  ),
  makeQuestion(
    36,
    "easy",
    "Which statements correctly describe chromosomes and genomes?",
    [
      [
        "Chromosomes contain DNA organized with genes and regulatory regions.",
        true,
      ],
      [
        "A genome is the full genetic information of an organism or cell.",
        true,
      ],
      ["A chromosome is the three-nucleotide unit read by ribosomes.", false],
      ["A genome is the folded active site of an enzyme.", false],
    ],
    "The hierarchy runs from base pairs to genes and regulatory regions to chromosomes to the genome. A genome refers to the full genetic information, while chromosomes are large DNA structures containing many genetic elements.",
  ),
  makeQuestion(
    37,
    "easy",
    "Which statements correctly describe the central dogma pattern?",
    [
      ["DNA can be transcribed into RNA.", true],
      ["RNA can be translated into protein.", true],
      [
        "Protein sequence is usually copied back into DNA during ordinary gene expression.",
        false,
      ],
      ["The central dogma says cells use DNA as a membrane lipid.", false],
    ],
    "The central pattern students need is DNA to RNA to protein. Biology contains additional regulatory flows, but this core pathway explains how stored sequence information can become working molecules.",
  ),
  makeQuestion(
    38,
    "easy",
    "Which statements correctly describe codons?",
    [
      ["A codon is a three-nucleotide unit read during translation.", true],
      [
        "Three-base codons provide enough combinations for amino acids and stop signals.",
        true,
      ],
      [
        "A codon is a complete chromosome containing thousands of genes.",
        false,
      ],
      [
        "A codon is a protein that turns genes on and off by binding DNA.",
        false,
      ],
    ],
    "Ribosomes read messenger RNA in codons, and each codon has three nucleotides. Three bases allow 64 combinations, enough to encode amino acids plus stop signals.",
  ),
  makeQuestion(
    39,
    "easy",
    "Which statements correctly describe gene expression?",
    [
      [
        "Gene expression means how much a gene product is made in a given context.",
        true,
      ],
      ["Cells regulate which instructions are active.", true],
      [
        "Gene expression means each cell uses the same genes at the same level.",
        false,
      ],
      [
        "Gene expression is the random loss of chromosomes during normal differentiation.",
        false,
      ],
    ],
    "Gene expression is about which gene products are made and how much of them are made. Different cell types and conditions use different expression programs even when the underlying DNA is very similar.",
  ),
  makeQuestion(
    40,
    "easy",
    "Which statements correctly describe transcription factors?",
    [
      [
        "Transcription factors are proteins that help turn genes on or off.",
        true,
      ],
      ["Transcription factors can bind DNA or regulatory complexes.", true],
      [
        "Transcription factors are the three-base units that encode amino acids.",
        false,
      ],
      [
        "Transcription factors are small circular DNA molecules used mainly in bacteria.",
        false,
      ],
    ],
    "Transcription factors are regulatory proteins that influence gene use. They help cells decide which gene programs to run in particular contexts by connecting DNA regulatory regions, signals, and cellular state.",
  ),
  makeQuestion(
    41,
    "easy",
    "Which statements correctly describe epigenetic regulation?",
    [
      [
        "It can change access to genetic information without changing the DNA sequence.",
        true,
      ],
      ["DNA methylation and histone modification are examples.", true],
      ["Chromatin packing can affect which genes are easier to use.", true],
      [
        "Epigenetic regulation is the same as replacing the base sequence with a new mutation.",
        false,
      ],
    ],
    "Epigenetic regulation is about accessibility and cell state rather than changing the underlying DNA sequence. DNA methylation, histone modification, and chromatin packing can influence gene use.",
  ),
  makeQuestion(
    42,
    "easy",
    "Which statements correctly describe mutation?",
    [
      ["A mutation is a change in DNA sequence.", true],
      [
        "Replication errors, radiation, chemicals, viruses, and mobile genetic elements can cause mutations.",
        true,
      ],
      [
        "A mutation can be neutral, harmful, or beneficial depending on context.",
        true,
      ],
      [
        "A mutation is the normal process of translating RNA into protein.",
        false,
      ],
    ],
    "Mutation means a DNA sequence change. Its effect depends on location, context, environment, and selection pressure, so mutations are not automatically beneficial or harmful in every situation.",
  ),
  makeQuestion(
    43,
    "easy",
    "Which statements correctly describe natural selection?",
    [
      ["Selection requires variation.", true],
      ["Selection requires inheritance.", true],
      ["Selection involves differential survival or reproduction.", true],
      [
        "Selection is the same as random variant-frequency change with no fitness difference.",
        false,
      ],
    ],
    "Natural selection changes variant frequencies when inherited differences affect survival or reproduction. Genetic drift is the random change in variant frequencies, especially important in small populations.",
  ),
  makeQuestion(
    44,
    "easy",
    "Which statements correctly describe plasmids and recombinant DNA?",
    [
      ["Plasmids are small DNA molecules found in many bacteria.", true],
      [
        "Plasmids can carry useful genes, including antibiotic resistance genes.",
        true,
      ],
      [
        "Inserted genes can be carried into bacteria for copying or expression.",
        true,
      ],
      [
        "Plasmids are ribosomes that assemble amino acids into insulin protein.",
        false,
      ],
    ],
    "Plasmids are useful because they can carry DNA into bacteria. Recombinant DNA methods can insert a gene, such as a human insulin gene, so cells copy or express the encoded product.",
  ),
  makeQuestion(
    45,
    "easy",
    "Which statements correctly describe mRNA technology?",
    [
      ["mRNA medicines deliver temporary instructions.", true],
      ["Cells can translate the delivered mRNA into an encoded protein.", true],
      ["mRNA can be used in vaccines and other therapeutic strategies.", true],
      [
        "mRNA medicines work by permanently replacing the patient's genome in each target cell.",
        false,
      ],
    ],
    "Messenger RNA medicines operate at the working-copy layer of biological information. They give cells temporary instructions to make a protein without directly rewriting the patient's DNA sequence.",
  ),
  makeQuestion(
    46,
    "medium",
    "Which statements correctly explain why DNA is a library rather than a simple blueprint?",
    [
      ["Cells regulate which parts of DNA are read.", true],
      [
        "The same DNA can support different cell behaviors through different expression programs.",
        true,
      ],
      [
        "Traits emerge through gene products, regulation, environment, and interactions.",
        true,
      ],
      ["Cells copy, repair, package, and access DNA dynamically.", true],
    ],
    "DNA contains inherited information, but cells do not execute the entire genome in a fixed way. Regulation, context, cell type, environment, and interactions determine which information is used and what effects it has.",
  ),
  makeQuestion(
    47,
    "medium",
    "Which statements correctly describe why neurons and liver cells can behave differently despite sharing DNA?",
    [
      ["They can run different gene expression programs.", true],
      ["They can maintain different regulatory states.", true],
      ["They can respond to different signals and cellular contexts.", true],
      [
        "They can produce different sets of proteins from shared genetic information.",
        true,
      ],
    ],
    "Most cells in a body share nearly the same DNA, but they do not use it in the same way. Cell identity depends heavily on regulated gene expression, signaling, chromatin state, and feedback.",
  ),
  makeQuestion(
    48,
    "medium",
    "Which statements correctly describe regulatory networks?",
    [
      ["Genes can help produce proteins that regulate other genes.", true],
      ["Signals can change gene expression.", true],
      ["RNA molecules can affect translation or stability.", true],
      [
        "Regulatory networks connect genes, proteins, signals, RNA, and feedback.",
        true,
      ],
    ],
    "Biological regulation is networked rather than a simple one-gene one-trait chain. Genes, proteins, RNAs, signals, and feedback can influence each other over time.",
  ),
  makeQuestion(
    49,
    "medium",
    "Which statements correctly connect evolution to medicine?",
    [
      ["Antibiotic resistance can evolve under treatment pressure.", true],
      ["Cancer progression can involve evolving cell populations.", true],
      ["Immune escape can occur when variants avoid recognition.", true],
      [
        "Pathogens can adapt in ways that change transmission, disease, or treatment response.",
        true,
      ],
    ],
    "Evolution is active in modern medicine, not just ancient history. Pathogens, tumors, and resistant populations can change as selection pressures favor variants that survive and spread.",
  ),
  makeQuestion(
    50,
    "medium",
    "Which statements correctly describe biotechnology tools built on biological information flow?",
    [
      ["Sequencing reads nucleotide order.", true],
      ["CRISPR-based tools use sequence-guided targeting.", true],
      [
        "Gene therapy aims to add, replace, silence, or edit genetic information.",
        true,
      ],
      [
        "Synthetic biology treats cells as programmable platforms while still respecting biological noise and regulation.",
        true,
      ],
    ],
    "Biotechnology works because scientists can read, copy, write, edit, and express biological information. These tools are powerful, but delivery, safety, regulation, context, and evidence remain central.",
  ),
];

export const BiologyChemistryLifeScienceL3Questions =
  BiologyChemistryForLifeScienceLecture3Questions;
