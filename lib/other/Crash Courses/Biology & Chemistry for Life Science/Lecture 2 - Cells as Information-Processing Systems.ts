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
        "Prokaryotic DNA is stored in a membrane-bound nucleus, while eukaryotic chromosomes usually float freely in cytoplasm.",
        false,
      ],
      [
        "Prokaryotes are less compartmentalized because they lack DNA, ribosomes, and regulated metabolism.",
        false,
      ],
      [
        "Prokaryotic gene regulation is absent, while eukaryotic tissues use identical expression programs.",
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
      ["Golgi apparatus: modifies, sorts, and ships nuclear DNA.", false],
      ["Nucleus: translates messenger RNA into proteins for export.", false],
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
    "A dissolved molecule is more concentrated outside a cell than inside, and it can cross through an open pathway without ATP. Which statement best describes the expected net movement?",
    [
      [
        "Net movement from higher concentration toward lower concentration due to random molecular motion.",
        true,
      ],
      [
        "Net movement against the concentration gradient, because crossing a membrane reverses random molecular motion.",
        false,
      ],
      [
        "No net movement unless an ATP-powered pump pushes each molecule across the pathway.",
        false,
      ],
      [
        "Equal movement from low to high and high to low concentration until the gradient becomes steeper.",
        false,
      ],
    ],
    "Diffusion is passive net movement down a concentration gradient, driven by random molecular motion. Membranes and channels can shape what crosses, but diffusion itself does not require ATP and does not build a steeper gradient.",
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
        "Channels mainly build gradients by spending ATP, while pumps mainly let solutes drift down existing gradients.",
        false,
      ],
      [
        "Channels and pumps are interchangeable as long as both are embedded in the membrane.",
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
    "Blood glucose rises after a meal, insulin helps lower it, and insulin release falls as glucose returns toward range. Which statement best describes that control pattern?",
    [
      [
        "A response that counteracts deviation and helps stabilize a variable.",
        true,
      ],
      [
        "A response that amplifies deviation until the process commits to completion.",
        false,
      ],
      [
        "A response that detects deviation but has no effector to change the variable.",
        false,
      ],
      [
        "A pathway that is inactive whenever the original signal is no longer present.",
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
        "Innate immunity develops antigen-specific memory after repeated vaccination.",
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
        "Amplification means each upstream molecule activates fewer downstream molecules than it receives.",
        false,
      ],
      [
        "Amplification means feedback is absent, so the pathway cannot be shut down.",
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
    "Which statement about the nucleus avoids the common command-center misconception?",
    [
      [
        "It stores genomic DNA and helps regulate access to that information.",
        true,
      ],
      [
        "It directly performs most protein folding, membrane transport, and ATP production once DNA is stored there.",
        false,
      ],
      [
        "It is passive DNA storage with little role in controlling which information can be used.",
        false,
      ],
      [
        "It replaces cytoplasm and organelles as the execution layer for most cellular work.",
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
        "Uncontrolled cell rupture that releases contents and tends to provoke inflammation.",
        false,
      ],
      [
        "A cell-cycle checkpoint that pauses division while a damaged cell remains alive indefinitely.",
        false,
      ],
      [
        "A differentiation program that specializes a cell without removing it from the tissue.",
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
        "Cancer requires a body-wide synchronized mutation before a tumor can evolve.",
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
        "The immune system identifies threats mainly by sensing body temperature changes.",
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
        "It means the membrane behaves as a sealed wall with no regulated transport.",
        false,
      ],
      [
        "It means membrane passage is governed by molecular size rather than charge, polarity, or transport proteins.",
        false,
      ],
      [
        "It means membrane passage is fixed by the bilayer alone and cannot be changed by channels or transporters.",
        false,
      ],
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
      [
        "The pump blockade should immediately strengthen the sodium and potassium gradients it normally maintains.",
        false,
      ],
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
        "The signal should force identical gene-expression outputs whenever receptor binding occurs.",
        false,
      ],
      [
        "Different outcomes imply one response cannot be biological because growth factors only cause division.",
        false,
      ],
      [
        "Once both cell types detect the ligand, downstream regulatory context no longer matters.",
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
        "Apoptosis is harmful in development because tissue maintenance should preserve damaged cells.",
        false,
      ],
      [
        "Cancer prevention is mainly a pathogen-killing problem rather than a cell-survival regulation problem.",
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
        "Vaccines mainly work by killing pathogens directly at injection rather than priming recognition.",
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
        "A signaling failure must mean the ligand has lost its molecular structure entirely.",
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
        "Immune responses affect pathogen clearance but are separate from symptom generation.",
        false,
      ],
      [
        "A stronger immune response is clinically preferable across inflammatory, autoimmune, and infectious disease contexts.",
        false,
      ],
    ],
    "Immune systems trade off defense and damage. Infections, autoimmunity, allergies, sepsis, and immunotherapy all show that immune activation must be specific, proportionate, and regulated.",
  ),
  makeQuestion(
    30,
    "hard",
    "A later pharmacology unit asks why cell biology matters for disease and treatment. Which connections from this cell-systems model are valid?",
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
  makeQuestion(
    31,
    "easy",
    "Which statement best describes why cells need compartments?",
    [
      [
        "Compartments help keep molecules concentrated, protected, and chemically controlled.",
        true,
      ],
      [
        "Compartments mainly exist to prevent molecules from taking part in reactions.",
        false,
      ],
      [
        "Compartments remove the need for membranes, transport, and regulation.",
        false,
      ],
      [
        "Compartments turn a cell into a uniform mixture with no local differences.",
        false,
      ],
    ],
    "Compartments let cells organize chemistry in space. They help maintain local concentrations, protect information, and separate reactions that would interfere with each other.",
  ),
  makeQuestion(
    32,
    "easy",
    "Which statement best distinguishes prokaryotic and eukaryotic cells?",
    [
      [
        "Prokaryotes lack a nucleus, while eukaryotic cells are more compartmentalized.",
        true,
      ],
      [
        "Prokaryotes have mitochondria, while eukaryotic cells lack membrane-bound organelles.",
        false,
      ],
      [
        "Prokaryotes and eukaryotes are distinguished mainly by whether their membranes contain phospholipids.",
        false,
      ],
      [
        "Prokaryotes use DNA, while eukaryotic cells store information in proteins instead.",
        false,
      ],
    ],
    "Prokaryotes such as bacteria are usually smaller and lack a nucleus. Eukaryotic cells such as human cells are larger and contain more internal compartments and organelles.",
  ),
  makeQuestion(
    33,
    "easy",
    "Which statement best describes the cell membrane?",
    [
      [
        "It is a lipid bilayer with embedded proteins that acts as a boundary and transport controller.",
        true,
      ],
      [
        "It is a freely open lipid sheet that lets charged particles cross as easily as oxygen.",
        false,
      ],
      [
        "It is only a passive wrapper and does not participate in signaling or transport.",
        false,
      ],
      [
        "It is a protein-only wall whose main job is to copy genetic information.",
        false,
      ],
    ],
    "The membrane is built from a lipid bilayer with embedded proteins. It separates the cell from its surroundings and supports transport, communication, and signaling.",
  ),
  makeQuestion(
    34,
    "easy",
    "Which statement best describes ribosomes?",
    [
      ["Ribosomes translate messenger RNA into amino acid chains.", true],
      ["Ribosomes store DNA inside a membrane-bound nucleus.", false],
      ["Ribosomes modify and ship secreted proteins after folding.", false],
      [
        "Ribosomes maintain ion gradients by pumping sodium and potassium.",
        false,
      ],
    ],
    "Ribosomes are the cellular machines that translate messenger RNA instructions into protein chains. They connect nucleotide information to amino acid sequence.",
  ),
  makeQuestion(
    35,
    "easy",
    "Which statement best describes apoptosis?",
    [
      [
        "Apoptosis is programmed cell death that removes damaged, risky, or unneeded cells.",
        true,
      ],
      [
        "Apoptosis is uncontrolled cell bursting that spreads inflammation through tissue.",
        false,
      ],
      ["Apoptosis is the copying and separation of DNA during mitosis.", false],
      [
        "Apoptosis is the movement of water toward higher solute concentration.",
        false,
      ],
    ],
    "Apoptosis is a regulated form of cell death. It helps remove damaged or unnecessary cells in a controlled way, reducing the risk of uncontrolled inflammation or harmful survival.",
  ),
  makeQuestion(
    36,
    "easy",
    "Which statements correctly describe major eukaryotic organelles?",
    [
      [
        "The nucleus stores DNA and regulates access to genetic information.",
        true,
      ],
      ["Mitochondria convert energy from nutrients into ATP.", true],
      [
        "The Golgi apparatus is the main structure that reads codons during translation.",
        false,
      ],
      [
        "The cytoskeleton is mainly a passive fluid with no role in movement or transport.",
        false,
      ],
    ],
    "The nucleus stores genetic information, and mitochondria help produce ATP from nutrient energy. The Golgi modifies, sorts, and ships molecules, while the cytoskeleton supports structure, movement, and intracellular transport.",
  ),
  makeQuestion(
    37,
    "easy",
    "Which statements correctly describe the endoplasmic reticulum and Golgi apparatus?",
    [
      [
        "The endoplasmic reticulum helps synthesize, fold, and process proteins and lipids.",
        true,
      ],
      ["The Golgi modifies, sorts, and ships molecules.", true],
      [
        "The endoplasmic reticulum is the main organelle that stores chromosomes as a genome.",
        false,
      ],
      [
        "The Golgi is the membrane channel that lets sodium diffuse down its gradient.",
        false,
      ],
    ],
    "The endoplasmic reticulum and Golgi form an important processing and trafficking pathway. They are especially relevant for secreted proteins, membrane proteins, lipids, and many therapeutic targets.",
  ),
  makeQuestion(
    38,
    "easy",
    "Which statements correctly distinguish diffusion and active transport?",
    [
      [
        "Diffusion moves molecules from higher concentration toward lower concentration.",
        true,
      ],
      [
        "Active transport can use energy to move substances against a gradient.",
        true,
      ],
      [
        "Diffusion requires ATP-driven pumps for each molecule that moves down a gradient.",
        false,
      ],
      [
        "Active transport is the passive spreading of molecules without energy input.",
        false,
      ],
    ],
    "Diffusion follows an existing concentration gradient and does not require direct energy input. Active transport uses energy, often ATP, to build or maintain gradients by moving substances against their favored direction.",
  ),
  makeQuestion(
    39,
    "easy",
    "Which statements correctly describe ligands and receptors?",
    [
      ["A ligand is a signaling molecule.", true],
      [
        "A receptor is usually a protein that detects a ligand and changes activity or shape.",
        true,
      ],
      [
        "A ligand is the lipid tail that makes the membrane hydrophobic.",
        false,
      ],
      [
        "A receptor is mainly a stored energy gradient across a membrane.",
        false,
      ],
    ],
    "Ligands and receptors are basic pieces of cell signaling. A ligand binds a receptor, and the receptor response can start internal changes in protein activity, gene expression, movement, secretion, growth, or survival.",
  ),
  makeQuestion(
    40,
    "easy",
    "Which statements correctly distinguish innate and adaptive immunity?",
    [
      ["Innate immunity is fast and broad.", true],
      ["Adaptive immunity is more specific and can form memory.", true],
      [
        "Innate immunity is based mainly on antibodies made after target learning.",
        false,
      ],
      ["Adaptive immunity lacks B cells, antibodies, and T cells.", false],
    ],
    "Innate immunity responds quickly to common danger patterns with broad defenses. Adaptive immunity is slower at first but more specific, involving B cells, antibodies, T cells, and memory.",
  ),
  makeQuestion(
    41,
    "easy",
    "Which statements correctly describe selective permeability of membranes?",
    [
      [
        "Small nonpolar molecules cross lipid bilayers more easily than charged particles.",
        true,
      ],
      ["Charged particles often require channels or transporters.", true],
      ["Large polar molecules often need transport systems or vesicles.", true],
      [
        "Selective permeability means the membrane functions like an open hole with no filtering.",
        false,
      ],
    ],
    "A lipid bilayer separates environments and filters movement based on size, polarity, and charge. Proteins such as channels and transporters help substances cross when the lipid core is unfavorable.",
  ),
  makeQuestion(
    42,
    "easy",
    "Which statements correctly describe osmosis?",
    [
      [
        "Osmosis is water movement across a selectively permeable membrane.",
        true,
      ],
      [
        "Water tends to move toward higher effective solute concentration.",
        true,
      ],
      [
        "Osmosis matters for cell swelling, shrinking, dehydration, and intravenous fluids.",
        true,
      ],
      [
        "Osmosis is the movement of DNA from the nucleus to the cytoplasm.",
        false,
      ],
    ],
    "Osmosis describes water movement across membranes in response to solute differences. It matters clinically because water balance affects cell volume, kidney function, dehydration, and fluid therapy.",
  ),
  makeQuestion(
    43,
    "easy",
    "Which statements correctly describe ion gradients?",
    [
      ["Cells maintain different ion concentrations across membranes.", true],
      [
        "Gradients store usable energy through separated charge and concentration.",
        true,
      ],
      [
        "Ion gradients help explain neuron firing, muscle contraction, and mitochondrial ATP production.",
        true,
      ],
      [
        "Ion gradients are identical ion concentrations on both sides of a membrane.",
        false,
      ],
    ],
    "Ion gradients are differences in concentration and charge across a membrane. Those differences store energy and can drive electrical signaling, transport, secretion, and ATP production.",
  ),
  makeQuestion(
    44,
    "easy",
    "Which statements correctly describe signal transduction?",
    [
      ["A ligand can bind a receptor.", true],
      ["A receptor can change conformation or activity.", true],
      [
        "Intracellular proteins can be activated or inhibited downstream.",
        true,
      ],
      [
        "Signal transduction is the direct copying of DNA before cell division.",
        false,
      ],
    ],
    "Signal transduction converts an external or local signal into internal cellular change. The chain can involve receptor changes, protein cascades, metabolism, gene expression, movement, secretion, growth, or survival.",
  ),
  makeQuestion(
    45,
    "easy",
    "Which statements correctly describe cancer as a cellular process?",
    [
      ["Cancer cells can gain growth advantages.", true],
      ["Cancer cells can evade death or ignore tissue constraints.", true],
      ["Cancer can evolve inside the body as cell populations change.", true],
      [
        "Cancer is best understood as a state where cell-cycle checkpoints and death controls work too strongly.",
        false,
      ],
    ],
    "Cancer involves dysregulated cell behavior, including growth, survival, control escape, and evolution within tissues. This framing explains why cancer is not a single static condition but a family of evolving diseases.",
  ),
  makeQuestion(
    46,
    "medium",
    "Which statements correctly describe the problems a cell solves?",
    [
      [
        "It keeps useful molecules concentrated through compartmentalization.",
        true,
      ],
      [
        "It processes information from nutrients, damage, pathogens, neighboring cells, hormones, and physical conditions.",
        true,
      ],
      [
        "It allocates energy, building blocks, enzymes, membrane area, and repair capacity.",
        true,
      ],
      [
        "It coordinates division, differentiation, or death when appropriate.",
        true,
      ],
    ],
    "A cell is an organized system that controls chemistry, information, resources, and fate decisions. This view is more useful than treating the cell as a bag of molecules.",
  ),
  makeQuestion(
    47,
    "medium",
    "Which statements correctly describe amplification and feedback in signaling?",
    [
      ["Amplification lets a small signal produce a large response.", true],
      ["Negative feedback can stabilize a system.", true],
      ["Positive feedback can amplify change or commit a process.", true],
      [
        "Feedback can shape metabolism, gene expression, movement, secretion, growth, or survival.",
        true,
      ],
    ],
    "Cell signaling is not just detection; it includes response control. Amplification increases signal impact, negative feedback stabilizes systems, and positive feedback can drive decisive transitions.",
  ),
  makeQuestion(
    48,
    "medium",
    "Which statements correctly describe cell division and differentiation?",
    [
      [
        "Mitosis creates two daughter cells after DNA is copied and separated.",
        true,
      ],
      [
        "Checkpoints help decide whether conditions are suitable for division.",
        true,
      ],
      ["Stem cells can produce specialized cell types.", true],
      [
        "Differentiation depends on regulated gene expression rather than different DNA in each cell type.",
        true,
      ],
    ],
    "Cells must balance growth, repair, specialization, and removal. Division requires accurate copying and separation, while differentiation depends on regulatory programs that make cells use shared genetic information differently.",
  ),
  makeQuestion(
    49,
    "medium",
    "Which statements correctly describe antibodies and T cells?",
    [
      ["Antibodies are proteins that bind specific molecular targets.", true],
      [
        "Antibodies can neutralize pathogens or mark targets for destruction.",
        true,
      ],
      ["T cells can inspect cells and coordinate immune responses.", true],
      ["Some T cells can destroy infected or abnormal cells.", true],
    ],
    "Antibodies and T cells are central pieces of adaptive immunity. Antibodies bind targets outside or on cells, while T cells inspect cellular state, coordinate responses, and can destroy infected or abnormal cells.",
  ),
  makeQuestion(
    50,
    "medium",
    "Which statements correctly connect cell biology to medicine?",
    [
      [
        "Membranes and transport help explain drug entry, ion channels, and gradients.",
        true,
      ],
      [
        "Signaling helps explain hormones, growth factors, cytokines, and therapeutic targets.",
        true,
      ],
      [
        "Apoptosis and cell-cycle control help explain cancer and tissue maintenance.",
        true,
      ],
      [
        "Innate and adaptive immunity help explain infection, vaccines, autoimmunity, and immunotherapy.",
        true,
      ],
    ],
    "Cell biology provides the basic mechanisms behind many medical topics. Transport, signaling, growth control, death control, and immunity connect molecular events to disease and treatment.",
  ),
];

export const BiologyChemistryLifeScienceL2Questions =
  BiologyChemistryForLifeScienceLecture2Questions;
