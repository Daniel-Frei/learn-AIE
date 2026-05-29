import { Question } from "../../../quiz";

export const BiologyChemistryForLifeScienceLecture3Questions: Question[] = [
  {
    id: "bio-chem-life-l3-q01",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe DNA as biological information storage?",
    options: [
      {
        text: "DNA stores information in sequences of bases.",
        isCorrect: true,
      },
      {
        text: "The four DNA bases are adenine, thymine, guanine, and cytosine.",
        isCorrect: true,
      },
      {
        text: "DNA can be copied because base pairing provides a template-like mechanism.",
        isCorrect: true,
      },
      {
        text: "DNA primarily solves the information-storage problem for cells.",
        isCorrect: true,
      },
    ],
    explanation:
      "DNA stores biological information in ordered base sequences. Complementary base pairing makes replication, repair, and inheritance possible because each strand can guide reconstruction of the other.",
  },
  {
    id: "bio-chem-life-l3-q02",
    chapter: 3,
    difficulty: "easy",
    prompt: "Which statements correctly describe complementary base pairing?",
    options: [
      { text: "A pairs with T in DNA.", isCorrect: true },
      { text: "G pairs with C in DNA.", isCorrect: true },
      {
        text: "Complementary pairing helps DNA act as a copyable information molecule.",
        isCorrect: true,
      },
      {
        text: "Complementary pairing means every base pairs with every other base equally.",
        isCorrect: false,
      },
    ],
    explanation:
      "Specific base pairing is what lets DNA strands carry recoverable information. If every base paired equally with every other base, the template-copying logic would break down.",
  },
  {
    id: "bio-chem-life-l3-q03",
    chapter: 3,
    difficulty: "easy",
    prompt: "Which statements correctly describe genes?",
    options: [
      {
        text: "A gene is a DNA sequence that can be used to produce a functional product.",
        isCorrect: true,
      },
      {
        text: "A gene can encode a protein or functional RNA.",
        isCorrect: true,
      },
      {
        text: "A gene directly and deterministically causes one trait in every context.",
        isCorrect: false,
      },
      {
        text: "Genes function outside larger cellular systems and regulation.",
        isCorrect: false,
      },
    ],
    explanation:
      "Genes are functional information units, but their effects depend on context and regulation. Treating a gene as a one-trait blueprint ignores networks, expression levels, environment, and protein function.",
  },
  {
    id: "bio-chem-life-l3-q04",
    chapter: 3,
    difficulty: "easy",
    prompt: "Which statement best describes a genome?",
    options: [
      {
        text: "A genome is the full genetic information set of an organism or cell.",
        isCorrect: true,
      },
      {
        text: "A genome is a single protein produced by a ribosome.",
        isCorrect: false,
      },
      {
        text: "A genome is only the blood glucose level of an organism.",
        isCorrect: false,
      },
      { text: "A genome is the same thing as one codon.", isCorrect: false },
    ],
    explanation:
      "A genome is the complete set of genetic information, organized across chromosomes or similar genetic structures. It is not a protein, a metabolic measurement, or a single three-base codon.",
  },
  {
    id: "bio-chem-life-l3-q05",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe why DNA is not a simple blueprint?",
    options: [
      {
        text: "DNA contains potential instructions that cells selectively access.",
        isCorrect: true,
      },
      {
        text: "Gene execution is regulated and context-dependent.",
        isCorrect: true,
      },
      {
        text: "Cellular behavior depends on networks of genes, proteins, signals, and environment.",
        isCorrect: true,
      },
      {
        text: "The same genome can support different cell types through different regulation.",
        isCorrect: true,
      },
    ],
    explanation:
      "A blueprint metaphor is too rigid because DNA is not executed uniformly in every cell. Cells regulate which instructions are active, and biological behavior emerges from dynamic networks rather than raw sequence alone.",
  },
  {
    id: "bio-chem-life-l3-q06",
    chapter: 3,
    difficulty: "easy",
    prompt: "Which statements correctly describe the central dogma?",
    options: [
      {
        text: "A common information flow is DNA to RNA to protein.",
        isCorrect: true,
      },
      { text: "Transcription produces RNA from DNA.", isCorrect: true },
      { text: "Translation produces protein from RNA.", isCorrect: true },
      {
        text: "The central dogma means proteins are copied directly into DNA for every cell function.",
        isCorrect: false,
      },
    ],
    explanation:
      "The central dogma summarizes a core biological information flow: stored DNA information is transcribed into RNA and translated into protein. It does not mean proteins are simply copied back into DNA to perform every function.",
  },
  {
    id: "bio-chem-life-l3-q07",
    chapter: 3,
    difficulty: "easy",
    prompt: "Which statements correctly describe RNA?",
    options: [
      {
        text: "RNA can act as a working copy of genetic information.",
        isCorrect: true,
      },
      {
        text: "Messenger RNA can carry instructions to ribosomes.",
        isCorrect: true,
      },
      {
        text: "RNA is always the permanent genome storage molecule in human cells.",
        isCorrect: false,
      },
      {
        text: "RNA cannot participate in biotechnology or therapeutics.",
        isCorrect: false,
      },
    ],
    explanation:
      "RNA often acts as a temporary or working information molecule, with messenger RNA carrying instructions for protein production. RNA is not the main permanent genome store in human cells, and RNA technologies are important in modern medicine.",
  },
  {
    id: "bio-chem-life-l3-q08",
    chapter: 3,
    difficulty: "easy",
    prompt: "Which statement best describes RNA polymerase?",
    options: [
      {
        text: "RNA polymerase reads DNA and produces RNA during transcription.",
        isCorrect: true,
      },
      {
        text: "RNA polymerase is an antibody that neutralizes viruses.",
        isCorrect: false,
      },
      { text: "RNA polymerase is a lipid bilayer.", isCorrect: false },
      {
        text: "RNA polymerase translates codons into amino acids inside the ribosome.",
        isCorrect: false,
      },
    ],
    explanation:
      "RNA polymerase is the enzyme that transcribes DNA into RNA. Ribosomes handle translation, while antibodies and lipid bilayers serve very different biological roles.",
  },
  {
    id: "bio-chem-life-l3-q09",
    chapter: 3,
    difficulty: "easy",
    prompt: "Which statements correctly describe translation?",
    options: [
      { text: "Ribosomes read RNA and build proteins.", isCorrect: true },
      {
        text: "RNA is read in groups of three bases called codons.",
        isCorrect: true,
      },
      {
        text: "Codons specify amino acids or translation signals.",
        isCorrect: true,
      },
      {
        text: "Translation connects genetic instructions to protein execution.",
        isCorrect: true,
      },
    ],
    explanation:
      "Translation turns RNA information into amino-acid sequences that become proteins. Codons provide the mapping from sequence information to the protein-building process.",
  },
  {
    id: "bio-chem-life-l3-q10",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "Which statements correctly explain why codons use triplets conceptually?",
    options: [
      {
        text: "Four bases read two at a time would give only \\(4^2=16\\) combinations.",
        isCorrect: true,
      },
      {
        text: "Four bases read three at a time give \\(4^3=64\\) combinations.",
        isCorrect: true,
      },
      {
        text: "Triplets provide enough combinations to encode the standard amino acids plus signals.",
        isCorrect: true,
      },
      {
        text: "Triplets are used because a single base has infinite possible meanings.",
        isCorrect: false,
      },
    ],
    explanation:
      "The triplet code is enough to represent the amino acid vocabulary and translation signals. A two-base code would be too small, while a single base has only four possibilities, not infinite meanings.",
  },
  {
    id: "bio-chem-life-l3-q11",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe proteins as biological executors?",
    options: [
      {
        text: "Proteins perform much of the active work inside cells.",
        isCorrect: true,
      },
      {
        text: "Genes influence traits partly through the proteins they help produce.",
        isCorrect: true,
      },
      {
        text: "Proteins cannot act as enzymes, receptors, transporters, or structural components.",
        isCorrect: false,
      },
      { text: "Proteins are irrelevant once DNA exists.", isCorrect: false },
    ],
    explanation:
      "Proteins perform many functions, including catalysis, signaling, transport, and structure. Genes influence traits through regulated protein production, so proteins are not irrelevant to biological behavior.",
  },
  {
    id: "bio-chem-life-l3-q12",
    chapter: 3,
    difficulty: "easy",
    prompt: "Which statement best summarizes DNA, RNA, and protein roles?",
    options: [
      {
        text: "DNA stores information, RNA communicates working instructions, and proteins execute many cellular functions.",
        isCorrect: true,
      },
      {
        text: "DNA executes every function directly without RNA or proteins.",
        isCorrect: false,
      },
      { text: "RNA is only a membrane lipid.", isCorrect: false },
      {
        text: "Proteins are only passive copies of chromosomes.",
        isCorrect: false,
      },
    ],
    explanation:
      "A useful mental model is storage, communication, and execution. DNA stores, RNA carries or regulates information, and proteins perform many active molecular functions.",
  },
  {
    id: "bio-chem-life-l3-q13",
    chapter: 3,
    difficulty: "easy",
    prompt: "Which statements correctly describe gene expression?",
    options: [
      {
        text: "Gene expression means how much or whether a gene is used.",
        isCorrect: true,
      },
      {
        text: "Cells can activate some genes and suppress others.",
        isCorrect: true,
      },
      {
        text: "Regulated expression helps different cell types behave differently.",
        isCorrect: true,
      },
      {
        text: "Not every gene is active all the time in every cell.",
        isCorrect: true,
      },
    ],
    explanation:
      "Gene expression is about which genetic instructions are actually being used and at what level. This regulated access is crucial for cell identity, development, and adaptation.",
  },
  {
    id: "bio-chem-life-l3-q14",
    chapter: 3,
    difficulty: "medium",
    prompt: "Which statements correctly describe transcription factors?",
    options: [
      {
        text: "They are proteins that help regulate gene activity.",
        isCorrect: true,
      },
      {
        text: "They can influence when genes turn on or off.",
        isCorrect: true,
      },
      {
        text: "They can act like cellular workflow controllers for gene expression.",
        isCorrect: true,
      },
      {
        text: "They are random DNA damage events rather than regulatory molecules.",
        isCorrect: false,
      },
    ],
    explanation:
      "Transcription factors are regulatory proteins that influence gene expression. They are not random damage events; they are part of the control machinery that determines which instructions get executed.",
  },
  {
    id: "bio-chem-life-l3-q15",
    chapter: 3,
    difficulty: "medium",
    prompt: "Which statements correctly describe gene regulatory networks?",
    options: [
      {
        text: "Genes, proteins, and signals can regulate one another.",
        isCorrect: true,
      },
      {
        text: "Individual genes rarely act in complete isolation.",
        isCorrect: true,
      },
      {
        text: "Regulatory networks are irrelevant to cell identity.",
        isCorrect: false,
      },
      {
        text: "Biology is best understood as one gene acting alone with no interactions.",
        isCorrect: false,
      },
    ],
    explanation:
      "Biological behavior often emerges from networks of interacting genes, proteins, and signals. Cell identity depends strongly on these networks, so single-gene isolation is usually an oversimplified model.",
  },
  {
    id: "bio-chem-life-l3-q16",
    chapter: 3,
    difficulty: "medium",
    prompt: "Which statement best describes epigenetics?",
    options: [
      {
        text: "Epigenetics involves chemical or structural changes that influence access to DNA without changing the underlying sequence.",
        isCorrect: true,
      },
      {
        text: "Epigenetics always means the DNA base sequence has been rewritten.",
        isCorrect: false,
      },
      { text: "Epigenetics proves genes do not exist.", isCorrect: false },
      {
        text: "Epigenetics prevents regulation from affecting cells.",
        isCorrect: false,
      },
    ],
    explanation:
      "Epigenetic mechanisms affect whether parts of the genome are accessible or active. They influence expression without necessarily changing the DNA letters themselves.",
  },
  {
    id: "bio-chem-life-l3-q17",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements correctly explain how different cell types can share the same genome?",
    options: [
      { text: "Different cells can express different genes.", isCorrect: true },
      {
        text: "Different regulatory programs can run in different cells.",
        isCorrect: true,
      },
      {
        text: "Transcription factors and epigenetic state can shape cell identity.",
        isCorrect: true,
      },
      {
        text: "A neuron and a liver cell can behave differently despite nearly identical DNA.",
        isCorrect: true,
      },
    ],
    explanation:
      "Different cell types usually contain the same genetic library but use different parts of it. Cell identity comes from regulated execution of information, not from every cell needing a completely different genome.",
  },
  {
    id: "bio-chem-life-l3-q18",
    chapter: 3,
    difficulty: "medium",
    prompt: "Which statements correctly describe mutation?",
    options: [
      { text: "A mutation is a change in DNA sequence.", isCorrect: true },
      {
        text: "Mutations can arise from replication errors, radiation, chemicals, or viruses.",
        isCorrect: true,
      },
      {
        text: "Most mutations are neutral or harmful rather than strongly beneficial.",
        isCorrect: true,
      },
      {
        text: "Every mutation is immediately beneficial and adaptive.",
        isCorrect: false,
      },
    ],
    explanation:
      "Mutation creates genetic variation by changing DNA sequence. Some mutations matter strongly, but many are neutral or harmful, and only a minority are beneficial in a given environment.",
  },
  {
    id: "bio-chem-life-l3-q19",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe natural selection and evolution?",
    options: [
      {
        text: "Variation, inheritance, and selection can produce evolution.",
        isCorrect: true,
      },
      {
        text: "Variants that improve survival or reproduction can spread.",
        isCorrect: true,
      },
      {
        text: "Evolution requires all changes to be intentional.",
        isCorrect: false,
      },
      { text: "Selection has no relevance to medicine.", isCorrect: false },
    ],
    explanation:
      "Evolution occurs when heritable variation affects survival or reproduction across generations. It does not require intention, and it remains medically relevant in cancer, viruses, and antibiotic resistance.",
  },
  {
    id: "bio-chem-life-l3-q20",
    chapter: 3,
    difficulty: "medium",
    prompt: "Which statement best describes genetic drift?",
    options: [
      {
        text: "Genetic drift is random change in variant frequencies, especially important in small populations.",
        isCorrect: true,
      },
      {
        text: "Genetic drift means every evolutionary change is adaptive.",
        isCorrect: false,
      },
      {
        text: "Genetic drift is transcription from DNA to RNA.",
        isCorrect: false,
      },
      { text: "Genetic drift is a ribosome reading codons.", isCorrect: false },
    ],
    explanation:
      "Genetic drift is random fluctuation in genetic variants, not adaptive selection or gene expression. It matters most when populations are small enough for chance to strongly influence outcomes.",
  },
  {
    id: "bio-chem-life-l3-q21",
    chapter: 3,
    difficulty: "medium",
    prompt: "Which statements correctly connect evolution to medicine?",
    options: [
      {
        text: "Antibiotic resistance can evolve under drug pressure.",
        isCorrect: true,
      },
      { text: "Viruses can evolve in populations over time.", isCorrect: true },
      { text: "Cancer can involve evolution inside tissues.", isCorrect: true },
      {
        text: "Pathogen and tumor evolution can affect treatment success.",
        isCorrect: true,
      },
    ],
    explanation:
      "Evolution is not just historical; it actively shapes medical problems. Drug pressure, immune pressure, and cellular competition can select resistant microbes, viral variants, or cancer cell populations.",
  },
  {
    id: "bio-chem-life-l3-q22",
    chapter: 3,
    difficulty: "medium",
    prompt: "Which statements correctly describe DNA sequencing and genomics?",
    options: [
      { text: "Sequencing reads DNA information.", isCorrect: true },
      {
        text: "Falling sequencing costs enabled large-scale genomics.",
        isCorrect: true,
      },
      {
        text: "Genomics can support precision medicine and population studies.",
        isCorrect: true,
      },
      {
        text: "Sequencing requires destroying all biological information before reading it.",
        isCorrect: false,
      },
    ],
    explanation:
      "DNA sequencing lets researchers read genetic information at scale. This supports genomics, disease studies, ancestry and population analysis, and increasingly precision medicine.",
  },
  {
    id: "bio-chem-life-l3-q23",
    chapter: 3,
    difficulty: "medium",
    prompt: "Which statements correctly describe CRISPR and genome editing?",
    options: [
      {
        text: "CRISPR can be thought of as a programmable DNA editing tool.",
        isCorrect: true,
      },
      {
        text: "Genome editing can target specific DNA sequences.",
        isCorrect: true,
      },
      {
        text: "CRISPR is perfect and never has off-target or delivery challenges.",
        isCorrect: false,
      },
      {
        text: "Genome editing cannot affect biological function.",
        isCorrect: false,
      },
    ],
    explanation:
      "CRISPR made genome editing more programmable, but it is not magic or perfect. Targeting, delivery, specificity, and biological consequences still matter.",
  },
  {
    id: "bio-chem-life-l3-q24",
    chapter: 3,
    difficulty: "medium",
    prompt: "Which statement best describes mRNA technology?",
    options: [
      {
        text: "mRNA technology can deliver instructions so cells make a protein.",
        isCorrect: true,
      },
      {
        text: "mRNA technology works by delivering a finished chromosome to every cell.",
        isCorrect: false,
      },
      {
        text: "mRNA technology is unrelated to the DNA to RNA to protein information flow.",
        isCorrect: false,
      },
      {
        text: "mRNA technology means proteins are never produced.",
        isCorrect: false,
      },
    ],
    explanation:
      "mRNA technologies exploit the cell's translation machinery by delivering instructions rather than a finished protein in every case. This directly relies on the central-dogma idea that RNA can guide protein production.",
  },
  {
    id: "bio-chem-life-l3-q25",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe gene therapy and synthetic biology?",
    options: [
      {
        text: "Gene therapy can aim to replace, supplement, or modify defective genetic function.",
        isCorrect: true,
      },
      {
        text: "Synthetic biology treats cells and molecular systems as engineerable platforms.",
        isCorrect: true,
      },
      {
        text: "Both fields depend on understanding biological information flow.",
        isCorrect: true,
      },
      {
        text: "Both fields reflect the ability to read, write, or edit biological information.",
        isCorrect: true,
      },
    ],
    explanation:
      "Gene therapy and synthetic biology became possible because biology can increasingly be read, edited, and programmed. They depend on understanding how DNA, RNA, proteins, and regulation connect to function.",
  },
  {
    id: "bio-chem-life-l3-q26",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Which statements correctly explain why gene count alone does not explain organism complexity?",
    options: [
      {
        text: "Regulation affects when and where genes are used.",
        isCorrect: true,
      },
      {
        text: "Interactions among genes and proteins create networks.",
        isCorrect: true,
      },
      {
        text: "The same gene can participate in different contexts.",
        isCorrect: true,
      },
      {
        text: "More complex organisms must always have millions more protein-coding genes.",
        isCorrect: false,
      },
    ],
    explanation:
      "Complexity comes from regulation, interaction, context, and network behavior, not just raw gene count. A relatively modest number of genes can support many cell types and behaviors through controlled expression.",
  },
  {
    id: "bio-chem-life-l3-q27",
    chapter: 3,
    difficulty: "medium",
    prompt: "Which statements correctly describe the software analogy for DNA?",
    options: [
      {
        text: "DNA can resemble source code or stored procedures as an analogy.",
        isCorrect: true,
      },
      {
        text: "The analogy is limited because biological execution is noisy and distributed.",
        isCorrect: true,
      },
      {
        text: "The analogy means cells run deterministic code with no regulation or context.",
        isCorrect: false,
      },
      {
        text: "The analogy means evolution never modifies biological instructions.",
        isCorrect: false,
      },
    ],
    explanation:
      "The software analogy helps technically minded learners think about stored instructions, but biology is messier than ordinary software. Execution is noisy, context-dependent, distributed, and continuously shaped by evolution.",
  },
  {
    id: "bio-chem-life-l3-q28",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Which statement best explains why regulation can matter more than raw DNA sequence for cell behavior?",
    options: [
      {
        text: "Cells with similar DNA can behave differently because they execute different gene-expression programs.",
        isCorrect: true,
      },
      {
        text: "Raw DNA sequence has no informational role in biology.",
        isCorrect: false,
      },
      {
        text: "Regulation means every gene is active equally in every cell.",
        isCorrect: false,
      },
      {
        text: "Cell behavior is independent of proteins and signals.",
        isCorrect: false,
      },
    ],
    explanation:
      "DNA provides the library, but regulation controls which instructions are used and when. Raw sequence matters, but cellular behavior depends on expression programs, proteins, signals, and environment.",
  },
  {
    id: "bio-chem-life-l3-q29",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Which statements correctly explain why proteins connect genes to traits?",
    options: [
      { text: "Genes can influence which proteins are made.", isCorrect: true },
      {
        text: "Proteins perform molecular functions that affect cell behavior.",
        isCorrect: true,
      },
      {
        text: "Protein amount, location, structure, and regulation all matter.",
        isCorrect: true,
      },
      {
        text: "Traits often emerge from protein networks rather than one isolated molecule.",
        isCorrect: true,
      },
    ],
    explanation:
      "Genes influence traits through the proteins and functional RNAs they help produce, but traits emerge through systems. Protein structure, abundance, location, and interactions all affect biological outcomes.",
  },
  {
    id: "bio-chem-life-l3-q30",
    chapter: 3,
    difficulty: "hard",
    prompt: "Which statements correctly connect epigenetics to cell identity?",
    options: [
      {
        text: "Epigenetic state can affect which DNA regions are accessible.",
        isCorrect: true,
      },
      { text: "Accessibility can influence gene expression.", isCorrect: true },
      {
        text: "Different cell types can maintain different regulatory states.",
        isCorrect: true,
      },
      {
        text: "Epigenetics always changes A into T or G into C in the DNA sequence.",
        isCorrect: false,
      },
    ],
    explanation:
      "Epigenetic mechanisms can help maintain different expression patterns without rewriting the DNA letters. That makes them important for cell identity, development, and regulation.",
  },
  {
    id: "bio-chem-life-l3-q31",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe mutation and selection in cancer?",
    options: [
      {
        text: "Cancer cells can acquire genetic or regulatory changes.",
        isCorrect: true,
      },
      {
        text: "Cell variants with growth or survival advantages can expand.",
        isCorrect: true,
      },
      { text: "Therapy cannot create selection pressure.", isCorrect: false },
      {
        text: "Cancer evolution stops once a tumor begins growing.",
        isCorrect: false,
      },
    ],
    explanation:
      "Cancer can evolve as cell populations acquire changes and compete inside tissues. Therapy can create selection pressure, and tumor evolution can continue during progression and treatment.",
  },
  {
    id: "bio-chem-life-l3-q32",
    chapter: 3,
    difficulty: "hard",
    prompt: "Which statement best explains antibiotic resistance evolution?",
    options: [
      {
        text: "Antibiotics can select for microbes carrying variants that survive treatment.",
        isCorrect: true,
      },
      {
        text: "Antibiotics make every microbe equally sensitive forever.",
        isCorrect: false,
      },
      {
        text: "Resistance evolution requires microbes to plan future drug exposure.",
        isCorrect: false,
      },
      {
        text: "Resistance is impossible when genetic variation exists.",
        isCorrect: false,
      },
    ],
    explanation:
      "Antibiotic resistance evolves when heritable variation affects survival under drug pressure. Microbes do not need to plan; selection changes population composition over time.",
  },
  {
    id: "bio-chem-life-l3-q33",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe biotechnology as programming biology?",
    options: [
      {
        text: "Sequencing helps read biological information.",
        isCorrect: true,
      },
      {
        text: "Genome editing helps write or modify biological information.",
        isCorrect: true,
      },
      {
        text: "mRNA technology can deliver executable instructions to cells.",
        isCorrect: true,
      },
      {
        text: "Synthetic biology designs biological systems using engineering principles.",
        isCorrect: true,
      },
    ],
    explanation:
      "Modern biotechnology increasingly treats biological information as something that can be read, edited, delivered, and engineered. This is why the language of programming is useful, even though biological systems remain noisy and regulated.",
  },
  {
    id: "bio-chem-life-l3-q34",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Which statements correctly identify misconceptions about genetics?",
    options: [
      {
        text: "Genes are not deterministic blueprints acting alone.",
        isCorrect: true,
      },
      {
        text: "Regulation is central to biological behavior.",
        isCorrect: true,
      },
      {
        text: "DNA sequence matters, but context matters too.",
        isCorrect: true,
      },
      {
        text: "One gene always maps cleanly to one complex trait in every environment.",
        isCorrect: false,
      },
    ],
    explanation:
      "Genes matter, but they operate in regulatory and environmental contexts. The one-gene, one-trait mental model is often misleading for complex traits and diseases.",
  },
  {
    id: "bio-chem-life-l3-q35",
    chapter: 3,
    difficulty: "hard",
    prompt: "Which statements correctly connect genetics to AI for biology?",
    options: [
      {
        text: "Sequence data can be represented and modeled computationally.",
        isCorrect: true,
      },
      {
        text: "Protein sequence to structure to function is a learnable mapping problem.",
        isCorrect: true,
      },
      {
        text: "Gene regulation cannot create context-dependent outputs from similar inputs.",
        isCorrect: false,
      },
      {
        text: "AI for biology can ignore biological regulation entirely.",
        isCorrect: false,
      },
    ],
    explanation:
      "AI for biology often models sequences, structures, regulatory context, and molecular function. Gene regulation can create context-dependent outputs, so denying regulation would make biological modeling weaker.",
  },
  {
    id: "bio-chem-life-l3-q36",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Which statement best explains why mRNA therapies differ from many traditional drugs?",
    options: [
      {
        text: "They can deliver instructions that cells translate into a protein rather than only delivering a small molecule that binds a target.",
        isCorrect: true,
      },
      { text: "They require cells to stop using ribosomes.", isCorrect: false },
      {
        text: "They have no relationship to genetic information flow.",
        isCorrect: false,
      },
      {
        text: "They work only by deleting every gene in the genome.",
        isCorrect: false,
      },
    ],
    explanation:
      "mRNA therapies use the cell's translation system to produce a protein from delivered instructions. They rely on ribosomes and the information-flow logic from RNA to protein.",
  },
  {
    id: "bio-chem-life-l3-q37",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe why central dogma is not the whole story?",
    options: [
      { text: "Regulation controls when information flows.", isCorrect: true },
      { text: "Epigenetics can affect access to DNA.", isCorrect: true },
      {
        text: "Proteins and signals can feed back into gene expression.",
        isCorrect: true,
      },
      {
        text: "Evolution can change the information over time.",
        isCorrect: true,
      },
    ],
    explanation:
      "DNA to RNA to protein is a crucial core pathway, but cells regulate that pathway extensively. Feedback, epigenetic accessibility, signaling, and evolution all shape what information is used and what it becomes.",
  },
  {
    id: "bio-chem-life-l3-q38",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe why modern biology became programmable?",
    options: [
      {
        text: "Scientists learned to read DNA with sequencing.",
        isCorrect: true,
      },
      {
        text: "Scientists learned to edit DNA with tools such as CRISPR.",
        isCorrect: true,
      },
      {
        text: "Scientists learned to deliver instructions with technologies such as mRNA.",
        isCorrect: true,
      },
      {
        text: "Biology became programmable because regulation stopped existing.",
        isCorrect: false,
      },
    ],
    explanation:
      "Biology became more programmable because information could be read, modified, and delivered more precisely. Regulation still exists and remains one of the main reasons biological programming is hard.",
  },
  {
    id: "bio-chem-life-l3-q39",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Which statements correctly connect genetic variation to disease risk?",
    options: [
      { text: "Some variants can increase disease risk.", isCorrect: true },
      { text: "Some variants can affect drug response.", isCorrect: true },
      {
        text: "Disease risk cannot depend on interactions among genes, environment, and time.",
        isCorrect: false,
      },
      {
        text: "All diseases are purely genetic and independent of environment.",
        isCorrect: false,
      },
    ],
    explanation:
      "Genetic variation can influence disease risk and treatment response, but most disease risk is contextual. Genes, environment, and time can interact, so pure genetic determinism is usually misleading.",
  },
  {
    id: "bio-chem-life-l3-q40",
    chapter: 3,
    difficulty: "hard",
    prompt: "Which statement best summarizes biological regulation?",
    options: [
      {
        text: "Cells use regulated information flow from DNA, RNA, proteins, signals, and networks to produce context-dependent behavior.",
        isCorrect: true,
      },
      {
        text: "Cells execute every gene equally at all times.",
        isCorrect: false,
      },
      {
        text: "DNA sequence alone directly determines every outcome without regulation.",
        isCorrect: false,
      },
      {
        text: "Evolution, mutation, and biotechnology are unrelated to biological information.",
        isCorrect: false,
      },
    ],
    explanation:
      "Biological behavior emerges from controlled expression and execution of information. DNA matters, but RNA, proteins, regulatory networks, signals, evolution, and biotechnology all shape how that information is used.",
  },
];

export const BiologyChemistryLifeScienceL3Questions =
  BiologyChemistryForLifeScienceLecture3Questions;
