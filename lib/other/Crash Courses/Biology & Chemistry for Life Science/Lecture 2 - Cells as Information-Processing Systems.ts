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
    throw new Error(`Lecture 2 question ${number} must have four options.`);
  }

  return {
    id: `bio-chem-life-l2-q${String(number).padStart(2, "0")}`,
    chapter: 2,
    difficulty,
    prompt,
    options: optionSeeds.map(([text, isCorrect]) => ({ text, isCorrect })),
    explanation,
  };
}

export const BiologyChemistryForLifeScienceLecture2Questions: Question[] = [
  makeQuestion(
    1,
    "easy",
    "Which statements explain why cells need boundaries and compartments?",
    [
      ["Compartments help keep useful molecules near each other.", true],
      ["Boundaries let cells regulate what enters and leaves.", true],
      ["Compartments allow different chemical environments to coexist.", true],
      [
        "Boundaries help separate the cell interior from the external environment.",
        true,
      ],
    ],
    "Cells need local organization because reactions depend on concentration, location, and compatibility. A membrane is not just packaging; it creates controlled chemistry and lets the cell maintain internal conditions different from the outside.",
  ),
  makeQuestion(
    2,
    "easy",
    "Which statement best distinguishes prokaryotic from eukaryotic cells at this course level?",
    [
      [
        "Prokaryotes generally lack a nucleus, while eukaryotes are more compartmentalized and include human cells.",
        true,
      ],
      [
        "Prokaryotes have no DNA, while eukaryotes always have plasmids as their main chromosomes.",
        false,
      ],
      ["Prokaryotes are viruses, while eukaryotes are bacteria.", false],
      [
        "Prokaryotes cannot regulate genes, while eukaryotes regulate every gene identically in every tissue.",
        false,
      ],
    ],
    "The high-yield difference is compartmental organization, especially the nucleus in eukaryotic cells. Prokaryotes still have DNA, ribosomes, metabolism, and gene regulation, and many bacteria are medically important because they are living cells with targetable machinery.",
  ),
  makeQuestion(
    3,
    "easy",
    "Which organelle-function pairings are correct?",
    [
      ["Ribosome: translates messenger RNA into protein.", true],
      ["Mitochondrion: helps generate ATP using membrane gradients.", true],
      [
        "Golgi apparatus: stores the entire nuclear genome as its primary job.",
        false,
      ],
      [
        "Nucleus: burns glucose directly as its main role in ATP production.",
        false,
      ],
    ],
    "Organelles divide cellular labor without acting as isolated modules. Ribosomes and mitochondria support protein production and energy flow, while the Golgi is better understood as a processing and sorting system than as the whole-genome storage site.",
  ),
  makeQuestion(
    4,
    "easy",
    "Which statements correctly describe the cell membrane?",
    [
      ["It contains a lipid bilayer with embedded proteins.", true],
      [
        "It is selectively permeable rather than equally open to all substances.",
        true,
      ],
      ["It helps maintain gradients and cellular identity.", true],
      ["It provides surfaces for signaling and transport proteins.", true],
    ],
    "The membrane is an active interface, not a passive wall. Its lipid structure, proteins, and selective permeability let cells separate inside from outside while still communicating and exchanging selected materials.",
  ),
  makeQuestion(
    5,
    "easy",
    "Which statement best describes diffusion?",
    [
      [
        "Net movement from higher concentration toward lower concentration due to random molecular motion.",
        true,
      ],
      [
        "Movement against a concentration gradient by direct ATP use in every case.",
        false,
      ],
      [
        "Water movement only, with no relevance to gases or small solutes.",
        false,
      ],
      ["A change in DNA sequence that creates a new cell type.", false],
    ],
    "Diffusion is passive net movement down a concentration gradient. It is important for gases, small molecules, and many cellular processes, but it is different from active transport and from genetic change.",
  ),
  makeQuestion(
    6,
    "easy",
    "Which statements correctly describe osmosis?",
    [
      [
        "Osmosis is water movement across a selectively permeable membrane.",
        true,
      ],
      [
        "Water tends to move toward the side with higher effective solute concentration.",
        true,
      ],
      ["Osmosis is unrelated to cell volume or clinical fluid balance.", false],
      ["Osmosis means ions are pumped against their gradients by ATP.", false],
    ],
    "Osmosis is specifically about water movement shaped by solute concentration and membrane permeability. It is clinically important, so the option denying any link to cell volume or fluid balance is wrong.",
  ),
  makeQuestion(
    7,
    "easy",
    "Which statements correctly distinguish channels and pumps?",
    [
      [
        "Channels allow selected substances to move down existing gradients.",
        true,
      ],
      ["Pumps can use energy to move substances against gradients.", true],
      [
        "Channels and pumps are identical because both move every molecule with no selectivity.",
        false,
      ],
      [
        "Channels and pumps are identical because neither can be selective.",
        false,
      ],
    ],
    "Channels and pumps both help membranes control traffic, but they do different work. Channels provide selective pathways down gradients, while pumps build or maintain gradients by coupling transport to energy use.",
  ),
  makeQuestion(
    8,
    "easy",
    "Which statements correctly describe ligands and receptors?",
    [
      ["A ligand is a signaling molecule that can bind a target.", true],
      [
        "A receptor is often a protein that changes behavior when it detects a ligand.",
        true,
      ],
      [
        "Binding depends on molecular properties such as shape, charge, and chemistry.",
        true,
      ],
      [
        "Receptor signaling can change metabolism, movement, secretion, survival, or gene expression.",
        true,
      ],
    ],
    "Ligand-receptor signaling is a central way cells process information. The same molecular logic from Lecture 1 applies: recognition depends on chemical compatibility, and binding can trigger downstream cellular changes.",
  ),
  makeQuestion(
    9,
    "easy",
    "Which statement best describes negative feedback?",
    [
      [
        "A response that counteracts deviation and helps stabilize a variable.",
        true,
      ],
      ["A response that necessarily makes a signal grow forever.", false],
      ["A process that only happens in DNA replication.", false],
      [
        "A signaling failure caused by every receptor being permanently inactive.",
        false,
      ],
    ],
    "Negative feedback is stabilizing control. It is common in cells and physiology because living systems need to keep variables such as glucose, pH, temperature, and signaling activity within functional ranges.",
  ),
  makeQuestion(
    10,
    "easy",
    "Which statements correctly compare innate and adaptive immunity?",
    [
      ["Innate immunity is fast and broad.", true],
      ["Adaptive immunity is more specific and can form memory.", true],
      ["Adaptive immunity excludes B cells, antibodies, and T cells.", false],
      [
        "Innate immunity works only after years of training by vaccines.",
        false,
      ],
    ],
    "Innate and adaptive immunity are layered defenses. Innate responses act quickly against broad patterns, while adaptive responses involve B cells, antibodies, and T cells that can learn specific targets.",
  ),
  makeQuestion(
    11,
    "medium",
    "A cell maintains high calcium outside the cytoplasm and very low calcium inside. Which interpretations are correct?",
    [
      [
        "The calcium gradient can act as stored information or energy for signaling.",
        true,
      ],
      [
        "Opening calcium channels can rapidly change intracellular state.",
        true,
      ],
      [
        "Maintaining the gradient usually requires transport systems and energy over time.",
        true,
      ],
      [
        "A gradient means calcium concentration is identical on both sides of the membrane.",
        false,
      ],
    ],
    "Ion gradients are powerful because separation creates potential for rapid change. Calcium is especially important as a signal: opening channels can convert a stored gradient into a burst of intracellular information.",
  ),
  makeQuestion(
    12,
    "medium",
    "Which explanation best describes signal amplification in cells?",
    [
      [
        "One activated receptor or enzyme can trigger many downstream molecules, producing a large response from a small signal.",
        true,
      ],
      [
        "Amplification means every signal molecule must directly enter the nucleus to become DNA.",
        false,
      ],
      [
        "Amplification means the cell deletes negative feedback permanently.",
        false,
      ],
      [
        "Amplification means receptors bind ligands without specificity.",
        false,
      ],
    ],
    "Amplification lets cells respond strongly to low concentrations of signals. It also creates a need for regulation, because amplified pathways can overshoot or cause harm if feedback and shutoff mechanisms fail.",
  ),
  makeQuestion(
    13,
    "medium",
    "Which statements correctly describe the nucleus in a eukaryotic cell?",
    [
      ["It stores genomic DNA.", true],
      [
        "It directly performs every transport, folding, and ATP-generating function in the cell by itself.",
        false,
      ],
      [
        "It is the site where every protein is fully folded and secreted.",
        false,
      ],
      [
        "It is a simple master controller that directly executes all cellular work without proteins or cytoplasm.",
        false,
      ],
    ],
    "The nucleus stores genomic DNA, but cellular behavior is distributed across many structures and molecular networks. Proteins, RNA, cytoplasm, organelles, membranes, and signals all participate in execution and regulation.",
  ),
  makeQuestion(
    14,
    "medium",
    "Which statements correctly connect the endoplasmic reticulum and Golgi apparatus to protein handling?",
    [
      [
        "The endoplasmic reticulum helps synthesize, fold, and process many proteins.",
        true,
      ],
      ["The Golgi helps modify, sort, and ship molecules.", true],
      [
        "This pathway is important for many secreted and membrane proteins.",
        true,
      ],
      ["The Golgi translates codons into amino acids as its main job.", false],
    ],
    "The endoplasmic reticulum and Golgi are part of a processing and routing system for many proteins and lipids. Ribosomes perform translation, while the endoplasmic reticulum and Golgi handle folding, modification, quality control, and trafficking.",
  ),
  makeQuestion(
    15,
    "medium",
    "Which statements correctly describe mitochondria?",
    [
      ["They help generate ATP from nutrient-derived energy.", true],
      ["They use membrane gradients as part of ATP production.", true],
      [
        "They have evolutionary links to ancient bacteria-like organisms.",
        true,
      ],
      [
        "They are the main storage site of nuclear chromosomes in human cells.",
        false,
      ],
    ],
    "Mitochondria connect energy metabolism to membrane gradients. Their bacterial ancestry helps explain why they have their own DNA, but most human genomic DNA is stored in the nucleus.",
  ),
  makeQuestion(
    16,
    "medium",
    "Which statement best describes apoptosis?",
    [
      [
        "Programmed cell death that removes damaged, risky, or unnecessary cells in a controlled way.",
        true,
      ],
      [
        "Uncontrolled cell bursting that always spreads infection to every neighboring cell.",
        false,
      ],
      ["A process that converts a bacterium into a virus.", false],
      [
        "A membrane transport process that pumps sodium against a gradient.",
        false,
      ],
    ],
    "Apoptosis is regulated cell death and is often protective. It helps development, tissue maintenance, immune function, and cancer prevention by eliminating cells without the same inflammatory consequences as uncontrolled rupture.",
  ),
  makeQuestion(
    17,
    "medium",
    "Which statements correctly describe cancer as cellular evolution?",
    [
      [
        "Cancer cells can acquire growth advantages over neighboring cells.",
        true,
      ],
      [
        "Cancer progression can involve selection among diverse cell populations.",
        true,
      ],
      [
        "Cancer requires every cell in the body to have exactly the same mutation at the same time.",
        false,
      ],
      [
        "Cancer is best understood as ordinary apoptosis working too efficiently.",
        false,
      ],
    ],
    "Cancer is a family of diseases where cells gain traits that let them grow, survive, invade, or resist control. Selection can act within tissues, so tumors can evolve under immune pressure, nutrient limits, and treatment.",
  ),
  makeQuestion(
    18,
    "medium",
    "Which statements correctly connect cell signaling to gene expression?",
    [
      [
        "External signals can activate pathways that change transcription factor activity.",
        true,
      ],
      [
        "Signals can alter which genes are expressed without changing DNA sequence.",
        true,
      ],
      [
        "Cells can use signaling to adjust metabolism, division, differentiation, or survival.",
        true,
      ],
      [
        "Signaling cascades can include phosphorylation and other protein modifications.",
        true,
      ],
    ],
    "Cell signaling links the outside environment to internal programs. A signal can change protein activity quickly and can also alter gene expression patterns, allowing cells to adapt without rewriting their genome.",
  ),
  makeQuestion(
    19,
    "medium",
    "Which statements correctly describe immune recognition?",
    [
      ["Antibodies bind specific molecular targets called antigens.", true],
      [
        "T cells can help inspect cells for signs of infection or abnormality.",
        true,
      ],
      [
        "Immune recognition depends on molecular specificity, not visual appearance.",
        true,
      ],
      [
        "The immune system recognizes threats only by measuring body temperature.",
        false,
      ],
    ],
    "Immune recognition is molecular and cellular. Antibodies, receptors, antigen presentation, and inflammatory signals help the immune system distinguish many kinds of danger, but recognition is imperfect and can misfire.",
  ),
  makeQuestion(
    20,
    "medium",
    "Which statement best explains why selective permeability matters?",
    [
      [
        "It lets cells maintain internal conditions and gradients while still exchanging selected substances and signals.",
        true,
      ],
      [
        "It means nothing can ever cross the membrane under any circumstance.",
        false,
      ],
      ["It means every molecule crosses membranes at the same rate.", false],
      ["It means membranes store genetic information in codons.", false],
    ],
    "Selective permeability is the basis of cellular control. Cells are open enough to exchange matter and information, but closed enough to maintain concentrations, electrical differences, compartments, and identity.",
  ),
  makeQuestion(
    21,
    "hard",
    "A toxin blocks an ATP-dependent sodium-potassium pump. Which consequences are plausible?",
    [
      [
        "Ion gradients may gradually dissipate because active maintenance is impaired.",
        true,
      ],
      [
        "Membrane potential and downstream processes such as neuron signaling can be disrupted.",
        true,
      ],
      ["The pump blockade directly changes every codon in the genome.", false],
      [
        "Diffusion will automatically rebuild the original gradients without energy input.",
        false,
      ],
    ],
    "Pumps maintain gradients by using energy to move ions against their preferred direction. If active pumping stops, passive leaks and transport can erode gradients, disrupting electrical signaling, osmotic balance, and cellular physiology.",
  ),
  makeQuestion(
    22,
    "hard",
    "A hormone binds a receptor on liver cells but not on red blood cells. Which explanations are plausible?",
    [
      [
        "The receptor may be expressed in liver cells but absent or low in red blood cells.",
        true,
      ],
      [
        "The cell types may have different downstream signaling machinery.",
        true,
      ],
      [
        "Cell-type-specific gene expression can create different responses to the same circulating signal.",
        true,
      ],
      [
        "The hormone must have a different DNA sequence inside each cell it contacts.",
        false,
      ],
    ],
    "Multicellular signaling depends on cell-type-specific receptor expression and downstream context. A hormone can circulate widely while producing strong effects only in cells that have the relevant receptor and response machinery.",
  ),
  makeQuestion(
    23,
    "hard",
    "Which statements correctly identify why a cell is more like a regulated network than a simple bag of parts?",
    [
      ["Molecules are localized into compartments and pathways.", true],
      ["Signals can be amplified, damped, integrated, and fed back.", true],
      ["Energy and material allocation change with cell state.", true],
      [
        "Gene expression, protein activity, transport, and metabolism influence one another.",
        true,
      ],
    ],
    "Cells integrate chemistry, information, energy, and structure. Treating cellular components as isolated parts misses the feedback, localization, timing, and state-dependence that make living systems adaptive.",
  ),
  makeQuestion(
    24,
    "hard",
    "A growth factor causes one cell type to divide but another to differentiate. Which interpretation is best?",
    [
      [
        "The same signal can produce different outcomes depending on receptor levels, intracellular state, and regulatory context.",
        true,
      ],
      [
        "The growth factor must be two unrelated molecules with identical names.",
        false,
      ],
      [
        "The signal cannot be real because one ligand must always produce one identical response in every cell.",
        false,
      ],
      [
        "The cells must have no membranes because membranes prevent signaling.",
        false,
      ],
    ],
    "Cellular responses depend on context, not just the identity of the external signal. Receptors, pathway wiring, chromatin state, transcription factors, and existing cell identity can change how the same ligand is interpreted.",
  ),
  makeQuestion(
    25,
    "hard",
    "Which statements correctly connect apoptosis and cancer prevention?",
    [
      ["Apoptosis can remove cells with severe DNA damage.", true],
      ["Evading apoptosis can give abnormal cells a survival advantage.", true],
      [
        "Apoptosis is harmful in every context because tissues should never remove cells.",
        false,
      ],
      [
        "Cancer prevention depends only on killing pathogens, not on regulating cell survival.",
        false,
      ],
    ],
    "Controlled removal of dangerous cells is part of tissue-level quality control. If cells accumulate mutations and also evade death, they can persist long enough to acquire additional changes that support tumor evolution.",
  ),
  makeQuestion(
    26,
    "hard",
    "Which statements correctly describe how vaccines connect to adaptive immunity?",
    [
      [
        "Vaccines present antigens or antigen instructions in a safer context than dangerous infection.",
        true,
      ],
      ["Vaccines can support memory B-cell and T-cell responses.", true],
      [
        "Vaccines depend on the immune system's ability to recognize molecular targets.",
        true,
      ],
      [
        "Vaccines work by making antibiotics effective against every virus.",
        false,
      ],
    ],
    "Vaccines use adaptive immune learning. They do not turn antibiotics into antiviral drugs; instead, they train recognition so later exposure can be met faster and more effectively.",
  ),
  makeQuestion(
    27,
    "hard",
    "Which statements correctly describe a failure mode in cell signaling?",
    [
      ["A receptor mutation could make a pathway active without ligand.", true],
      [
        "A downstream kinase could be overactive even when receptor binding is normal.",
        true,
      ],
      [
        "Loss of negative feedback could make a response too strong or too persistent.",
        true,
      ],
      [
        "A signaling failure must mean ligands no longer contain any atoms.",
        false,
      ],
    ],
    "Signaling pathways can fail at receptors, downstream proteins, feedback loops, degradation systems, or gene-expression outputs. These failures are common in cancer, endocrine disease, immune disease, and pharmacology.",
  ),
  makeQuestion(
    28,
    "hard",
    "Which statements correctly synthesize membranes, gradients, and signaling?",
    [
      ["Membranes create separations that make gradients possible.", true],
      [
        "Gradients can be rapidly converted into signals by opening channels.",
        true,
      ],
      [
        "Receptors and transporters embedded in membranes can control responses to the environment.",
        true,
      ],
      [
        "Energy-consuming pumps help maintain gradients that passive leaks would otherwise erode.",
        true,
      ],
    ],
    "Membranes are central to information and energy processing. They separate, sense, and transport, allowing cells to store potential energy in gradients and convert external or internal signals into action.",
  ),
  makeQuestion(
    29,
    "hard",
    "Which statements correctly identify why immune responses can both protect and harm?",
    [
      [
        "Inflammation can help fight infection but also damage tissue if excessive or chronic.",
        true,
      ],
      [
        "Adaptive recognition can target pathogens but can also misdirect against self tissues.",
        true,
      ],
      [
        "Immune response cannot affect symptoms because only pathogens cause symptoms.",
        false,
      ],
      [
        "A stronger immune response is always clinically better in every disease.",
        false,
      ],
    ],
    "Immune systems trade off defense and damage. Infections, autoimmunity, allergies, sepsis, and immunotherapy all show that immune activation must be specific, proportionate, and regulated.",
  ),
  makeQuestion(
    30,
    "hard",
    "Which statements correctly connect Lecture 2 to later disease and pharmacology topics?",
    [
      [
        "Drugs can target receptors, channels, pumps, enzymes, or signaling proteins.",
        true,
      ],
      [
        "Disease can emerge from failed gradients, signaling, division control, or immune regulation.",
        true,
      ],
      [
        "Biomarkers can measure cellular or tissue states shaped by these systems.",
        true,
      ],
      [
        "Therapies often perturb networks rather than fixing one isolated component with no downstream effects.",
        true,
      ],
    ],
    "The cell concepts in this lecture are direct foundations for medicine. Pharmacology, cancer, immunity, endocrine disease, neurobiology, diagnostics, and AI-for-health applications all rely on understanding cellular organization and signaling.",
  ),
];

export const BiologyChemistryLifeScienceL2Questions =
  BiologyChemistryForLifeScienceLecture2Questions;
