import { Question } from "../../../quiz";

type PrepDifficulty = "easy" | "medium" | "hard";
type OptionSeed = readonly [text: string, isCorrect: boolean];

function makeQuestion(
  number: number,
  difficulty: PrepDifficulty,
  prompt: string,
  optionSeeds: readonly OptionSeed[],
  explanation: string,
): Question {
  if (optionSeeds.length !== 4) {
    throw new Error(`Lecture 0 question ${number} must have four options.`);
  }

  return {
    id: `bio-chem-life-l0-q${String(number).padStart(2, "0")}`,
    chapter: 0,
    difficulty,
    prompt,
    options: optionSeeds.map(([text, isCorrect]) => ({ text, isCorrect })),
    explanation,
  };
}

export const BiologyChemistryForLifeScienceLecture0PreparationQuestions: Question[] =
  [
    makeQuestion(
      1,
      "easy",
      "Which statement best describes an atom?",
      [
        [
          "A tiny unit of matter that can make up elements and molecules.",
          true,
        ],
        ["A living compartment that stores genes and makes proteins.", false],
        ["A disease category caused by disrupted regulation.", false],
        [
          "A measurement used to decide whether a treatment helped patients.",
          false,
        ],
      ],
      "An atom is a basic unit of matter, not a living cell or a medical outcome. Understanding atoms makes later ideas such as elements, molecules, bonds, water, DNA, and proteins less mysterious.",
    ),
    makeQuestion(
      2,
      "easy",
      "Which statement best describes an element?",
      [
        ["A substance defined by atoms with the same number of protons.", true],
        ["A chain of amino acids folded into a working shape.", false],
        ["A membrane protein that detects a signaling molecule.", false],
        ["A patient subgroup selected by a diagnostic test.", false],
      ],
      "An element is defined by the number of protons in its atoms. Carbon, hydrogen, oxygen, nitrogen, phosphorus, and sulfur are examples that matter heavily in living systems.",
    ),
    makeQuestion(
      3,
      "easy",
      "Which statement best describes a molecule?",
      [
        ["A group of atoms held together by chemical bonds.", true],
        ["A tissue made from many specialized organs.", false],
        ["A clinical trial group used for comparison.", false],
        ["A random change in DNA sequence.", false],
      ],
      "A molecule is made when atoms are bonded together. Water, glucose, oxygen gas, DNA, and proteins are very different molecules, but they share the basic idea of bonded atoms.",
    ),
    makeQuestion(
      4,
      "easy",
      "Which statement best describes a chemical bond?",
      [
        ["An interaction that holds atoms or molecular parts together.", true],
        ["A program that decides whether a patient enters a trial.", false],
        ["A type of immune memory cell formed after vaccination.", false],
        ["A whole organ system that transports blood through the body.", false],
      ],
      "Chemical bonds and interactions explain why molecules have stable shapes and behaviors. The course later uses this idea for water, proteins, membranes, DNA, drug binding, and enzymes.",
    ),
    makeQuestion(
      5,
      "easy",
      "Which statement best describes a cell?",
      [
        [
          "A bounded living system that uses molecules, energy, and information.",
          true,
        ],
        ["A single chemical bond between carbon and hydrogen.", false],
        ["A drug exposure range where benefit outweighs harm.", false],
        ["A three-nucleotide unit read during protein synthesis.", false],
      ],
      "A cell is the basic living unit emphasized across the crash course. It is bounded by a membrane, contains organized molecules, uses energy, processes information, and responds to its environment.",
    ),
    makeQuestion(
      6,
      "easy",
      "Which statement best describes DNA?",
      [
        [
          "A molecule that stores genetic information in nucleotide sequence.",
          true,
        ],
        ["A lipid layer that separates a cell from its environment.", false],
        ["A protein catalyst that lowers activation energy.", false],
        ["A hormone that lowers blood glucose after a meal.", false],
      ],
      "Deoxyribonucleic acid (DNA) stores inherited information in the order of its bases. Cells copy, read, regulate, and protect DNA rather than using it as a membrane, enzyme, or hormone.",
    ),
    makeQuestion(
      7,
      "easy",
      "Which statement best describes a protein?",
      [
        [
          "A molecule made from amino acids that can fold and perform cellular work.",
          true,
        ],
        ["A genetic storage polymer made from A, T, G, and C bases.", false],
        ["A small circular DNA carrier common in bacteria.", false],
        [
          "A water movement process across a selectively permeable membrane.",
          false,
        ],
      ],
      "Proteins are built from amino acids and can fold into shapes that do work. They can act as enzymes, receptors, transporters, structural materials, antibodies, signals, and motors.",
    ),
    makeQuestion(
      8,
      "easy",
      "Which statement best describes a biological membrane?",
      [
        [
          "A boundary made largely from lipids and proteins that separates environments.",
          true,
        ],
        ["A chromosome region that codes for a functional product.", false],
        ["A random change in the frequency of a genetic variant.", false],
        ["A blinded comparison used to reduce measurement bias.", false],
      ],
      "Membranes let cells and organelles separate inside from outside. Their lipid and protein structure supports selective transport, signaling, gradients, and compartmentalized chemistry.",
    ),
    makeQuestion(
      9,
      "easy",
      "Which statement best describes an enzyme?",
      [
        ["A biological catalyst that helps reactions happen faster.", true],
        ["A molecule that stores inherited sequence information.", false],
        ["A disease caused by immune attack on self tissues.", false],
        ["A data shift that makes an AI model less reliable.", false],
      ],
      "An enzyme is a catalyst, meaning it speeds a reaction without being used up as the reaction's raw material. Many enzymes are proteins, and their shape and chemistry help make them specific.",
    ),
    makeQuestion(
      10,
      "easy",
      "Which statement best describes a gene?",
      [
        ["A DNA sequence that can be used to make a functional product.", true],
        ["A lipid molecule that stores long-term energy.", false],
        ["A drug that blocks a receptor signal.", false],
        ["A water molecule moving toward higher solute concentration.", false],
      ],
      "A gene is a usable stretch of DNA, often connected to making a protein or functional RNA. Genes influence traits through expression, regulation, environment, and interactions with other genes.",
    ),
    makeQuestion(
      11,
      "easy",
      "Which statement best describes a receptor?",
      [
        [
          "A molecule, often a protein, that detects a signal and changes activity.",
          true,
        ],
        ["A small sugar used as a central fuel source.", false],
        ["A random outcome used instead of a control group.", false],
        ["A bacterial chromosome that lacks genetic information.", false],
      ],
      "Receptors are central to cell signaling and pharmacology. They bind ligands or detect conditions, then help convert that information into cellular responses.",
    ),
    makeQuestion(
      12,
      "easy",
      "Which statement best describes a pathogen?",
      [
        [
          "A biological agent that can cause disease, such as a bacterium or virus.",
          true,
        ],
        ["A useful range of drug exposure between benefit and harm.", false],
        ["A protein that transports oxygen in red blood cells.", false],
        ["A chemical bond formed by shared electrons.", false],
      ],
      "A pathogen is an organism or agent that can cause disease. Bacteria, viruses, fungi, and parasites differ in structure and treatment options, so the type of pathogen matters.",
    ),
    makeQuestion(
      13,
      "easy",
      "Which statement best describes a drug in pharmacology?",
      [
        [
          "A substance used to change a biological process for diagnosis, prevention, or treatment.",
          true,
        ],
        ["A section of DNA that is read by RNA polymerase.", false],
        ["A molecule that defines an element by proton number.", false],
        ["A cell structure that translates messenger RNA.", false],
      ],
      "A drug perturbs biology, often by binding a protein or changing a pathway. The crash course later connects this simple idea to receptors, enzymes, channels, transporters, dose, toxicity, and patient variation.",
    ),
    makeQuestion(
      14,
      "easy",
      "Which statement best describes a biomarker?",
      [
        ["A measurable indicator of biological state.", true],
        [
          "A membrane that blocks charged particles from crossing freely.",
          false,
        ],
        ["A folded amino acid chain that catalyzes reactions.", false],
        [
          "A pathogen protein that a vaccine presents to the immune system.",
          false,
        ],
      ],
      "A biomarker is a measurement that says something about biology. Blood glucose, blood pressure, tumor mutations, protein levels, inflammation markers, and imaging findings can be biomarkers when used for a decision.",
    ),
    makeQuestion(
      15,
      "easy",
      "Which statement best describes clinical evidence?",
      [
        [
          "Information used to judge whether a medical claim holds up in patients.",
          true,
        ],
        ["The process of atoms sharing electrons in a stable bond.", false],
        ["The conversion of DNA sequence into messenger RNA.", false],
        ["The folding of a protein into an active site.", false],
      ],
      "Clinical evidence connects biological claims to patient outcomes. Mechanisms and lab results can be promising, but medical claims still need evidence that they help real people in the intended setting.",
    ),
    makeQuestion(
      16,
      "easy",
      "Which statements correctly describe parts of atoms?",
      [
        ["Protons help define the element.", true],
        ["Electrons strongly influence bonding and charge.", true],
        ["Neutrons are the main particles that form chemical bonds.", false],
        ["Electrons add most of the atom's mass in ordinary biology.", false],
      ],
      "Atoms contain protons, neutrons, and electrons. Protons define the element, neutrons affect isotopes and mass, and electrons are most directly involved in bonding, charge, and reactivity.",
    ),
    makeQuestion(
      17,
      "easy",
      "Which statements correctly connect common biological elements to roles?",
      [
        ["Carbon can form backbones for many organic molecules.", true],
        [
          "Phosphorus appears in DNA/RNA backbones, ATP, and phospholipids.",
          true,
        ],
        [
          "Nitrogen is mainly the element that makes lipid tails hydrophobic.",
          false,
        ],
        ["Sulfur is the central element that defines water's polarity.", false],
      ],
      "The CHNOPS elements are common in biology because their chemistry supports living systems. Carbon supports molecular diversity, phosphorus appears in information and energy molecules, nitrogen appears in amino acids and nucleic acids, and sulfur appears in some amino acids and redox chemistry.",
    ),
    makeQuestion(
      18,
      "easy",
      "Which statements correctly describe water for beginners?",
      [
        [
          "Water is polar because oxygen pulls shared electrons more strongly than hydrogen.",
          true,
        ],
        ["Water can dissolve many charged or polar substances.", true],
        ["Water behaves like a nonpolar oil inside cells.", false],
        [
          "Water's properties are separate from protein folding and membranes.",
          false,
        ],
      ],
      "Water is not just background fluid. Its polarity and hydrogen bonding help explain solubility, hydrophobic effects, membranes, protein folding, temperature buffering, and cellular organization.",
    ),
    makeQuestion(
      19,
      "easy",
      "Which statements correctly describe monomers and polymers?",
      [
        ["A monomer is a small building block.", true],
        [
          "A polymer is a larger chain made from repeated building blocks.",
          true,
        ],
        ["A monomer is a whole organ system made from tissues.", false],
        ["A polymer is a clinical comparison group in a trial.", false],
      ],
      "The course uses monomer and polymer language for macromolecules. Amino acids build proteins, nucleotides build DNA and RNA, and sugars can build carbohydrate polymers.",
    ),
    makeQuestion(
      20,
      "easy",
      "Which statements correctly describe hydrolysis and dehydration reactions?",
      [
        ["Hydrolysis uses water to break bonds.", true],
        [
          "Dehydration reactions remove water while joining building blocks.",
          true,
        ],
        [
          "Hydrolysis is the main process that stores DNA in chromosomes.",
          false,
        ],
        [
          "Dehydration reactions are the same as immune memory after vaccination.",
          false,
        ],
      ],
      "Hydrolysis and dehydration are basic reaction patterns for biological molecules. Hydrolysis helps digestion break polymers apart, while dehydration reactions help cells build larger molecules.",
    ),
    makeQuestion(
      21,
      "easy",
      "Which statements correctly describe carbohydrates and lipids?",
      [
        [
          "Carbohydrates include sugars and can help with energy, storage, structure, and recognition.",
          true,
        ],
        [
          "Lipids are often hydrophobic and can help form membranes or store energy.",
          true,
        ],
        [
          "Carbohydrates are the main molecules that store inherited DNA sequence.",
          false,
        ],
        [
          "Lipids are built as amino acid chains that fold into enzymes.",
          false,
        ],
      ],
      "Carbohydrates and lipids are two major biological molecule classes. Sugars and carbohydrate polymers support fuel, storage, structure, and recognition, while lipids help with membranes, energy storage, hormones, and signaling.",
    ),
    makeQuestion(
      22,
      "easy",
      "Which statements correctly describe proteins and nucleic acids?",
      [
        ["Proteins are built from amino acids.", true],
        ["DNA and RNA are nucleic acids built from nucleotides.", true],
        ["Proteins are usually made from repeating glucose monomers.", false],
        ["Nucleic acids are mainly hydrophobic membrane fats.", false],
      ],
      "Proteins and nucleic acids are major macromolecule classes with different building blocks. Proteins use amino acids for work such as catalysis and signaling, while nucleic acids use nucleotides for information storage and transfer.",
    ),
    makeQuestion(
      23,
      "easy",
      "Which statements correctly describe DNA, RNA, transcription, and translation?",
      [
        ["Transcription copies DNA information into RNA.", true],
        ["Translation uses messenger RNA to build a protein chain.", true],
        ["Translation means copying a protein sequence back into DNA.", false],
        ["Transcription means breaking food polymers into monomers.", false],
      ],
      "A useful starting model is DNA to RNA to protein. Transcription makes RNA from DNA, and translation uses messenger RNA instructions to build an amino acid chain.",
    ),
    makeQuestion(
      24,
      "easy",
      "Which statements correctly distinguish prokaryotic and eukaryotic cells?",
      [
        ["Bacteria are examples of prokaryotes.", true],
        ["Human cells are eukaryotic cells.", true],
        ["Prokaryotes are large human cells with a nucleus.", false],
        ["Eukaryotic cells lack internal compartments by definition.", false],
      ],
      "Prokaryotes such as bacteria are usually smaller and lack a nucleus. Eukaryotic cells such as human cells are larger and more compartmentalized, with organelles such as a nucleus and mitochondria.",
    ),
    makeQuestion(
      25,
      "easy",
      "Which statements correctly describe the nucleus and ribosomes?",
      [
        ["The nucleus stores DNA in eukaryotic cells.", true],
        ["Ribosomes translate messenger RNA into protein chains.", true],
        [
          "The nucleus is the main site where proteins are assembled from codons.",
          false,
        ],
        ["Ribosomes are lipid bilayers that keep ions separated.", false],
      ],
      "The nucleus and ribosomes belong to different parts of information flow. The nucleus stores and regulates DNA access, while ribosomes read messenger RNA to build proteins.",
    ),
    makeQuestion(
      26,
      "easy",
      "Which statements correctly describe passive and active transport?",
      [
        ["Passive transport follows an existing gradient.", true],
        [
          "Active transport can use energy to move substances against a gradient.",
          true,
        ],
        ["Passive transport builds gradients with ATP-driven pumps.", false],
        [
          "Active transport is the same as water randomly crossing a membrane without proteins.",
          false,
        ],
      ],
      "Cells control movement across membranes. Passive transport uses existing differences, while active transport spends energy to build or maintain differences such as ion gradients.",
    ),
    makeQuestion(
      27,
      "easy",
      "Which statements correctly describe ligands and receptors?",
      [
        ["A ligand is a signaling molecule.", true],
        ["A receptor detects a ligand and changes activity or shape.", true],
        ["A ligand is the genetic code used to make amino acids.", false],
        ["A receptor is a random change in DNA sequence.", false],
      ],
      "Ligands and receptors are core signaling vocabulary. A ligand binds or is detected, and a receptor helps convert that information into a cellular response.",
    ),
    makeQuestion(
      28,
      "easy",
      "Which statements correctly describe feedback?",
      [
        ["Negative feedback helps stabilize a system.", true],
        [
          "Positive feedback amplifies a process or drives it toward completion.",
          true,
        ],
        [
          "Negative feedback means a disease has no regulation involved.",
          false,
        ],
        [
          "Positive feedback means a biomarker has been validated for treatment selection.",
          false,
        ],
      ],
      "Feedback is a basic control idea used throughout biology and physiology. Negative feedback helps maintain useful ranges, while positive feedback can intensify processes such as clotting or childbirth contractions.",
    ),
    makeQuestion(
      29,
      "easy",
      "Which statements correctly describe broad disease categories?",
      [
        [
          "Infection involves pathogens such as bacteria, viruses, fungi, or parasites.",
          true,
        ],
        ["Autoimmune disease involves immune attack on self tissues.", true],
        ["Cancer is best defined as ordinary regulated cell death.", false],
        [
          "Metabolic disease is the direct copying of DNA before division.",
          false,
        ],
      ],
      "Disease categories are broad patterns of disrupted biology. Infection, autoimmunity, cancer, metabolic disease, genetic disease, and neurodegeneration have different mechanisms, although real patients can have overlapping problems.",
    ),
    makeQuestion(
      30,
      "easy",
      "Which statements correctly distinguish vaccines and antibiotics?",
      [
        ["Vaccines train adaptive immunity to recognize a target.", true],
        ["Antibiotics target bacterial biology.", true],
        [
          "Vaccines are drugs that directly block bacterial ribosomes after infection.",
          false,
        ],
        ["Antibiotics treat viruses by targeting viral cell walls.", false],
      ],
      "Vaccines and antibiotics use different strategies. Vaccines prepare immune memory, while antibiotics target bacterial structures or processes rather than viral life cycles.",
    ),
    makeQuestion(
      31,
      "easy",
      "Which statements correctly describe pH and charge?",
      [
        ["pH is related to hydrogen ion concentration.", true],
        ["Molecules can change charge when they gain or lose protons.", true],
        ["Charge can affect binding, folding, and enzyme activity.", true],
        ["pH is a count of carbon atoms in an organic molecule.", false],
      ],
      "pH is a simple but important idea for biological chemistry. Changes in protonation can change charge, and charge can affect molecular shape, binding, enzyme function, blood chemistry, and cellular compartments.",
    ),
    makeQuestion(
      32,
      "easy",
      "Which statements correctly describe weak molecular interactions?",
      [
        [
          "Hydrogen bonds can help water properties, DNA base pairing, and protein structure.",
          true,
        ],
        [
          "Ionic interactions involve attraction between opposite charges.",
          true,
        ],
        [
          "Hydrophobic effects help organize membranes and protein interiors.",
          true,
        ],
        [
          "Weak interactions are unrelated to molecular shape or binding.",
          false,
        ],
      ],
      "Weak interactions are central to biological organization because they can be reversible yet specific. Many weak interactions together can shape water behavior, folding, recognition, membranes, and drug binding.",
    ),
    makeQuestion(
      33,
      "easy",
      "Which statements correctly connect molecule classes to examples?",
      [
        ["Glucose is a carbohydrate used as an important fuel.", true],
        ["Phospholipids are lipids that help form membranes.", true],
        ["DNA and RNA are nucleic acids.", true],
        [
          "Antibodies are carbohydrate storage polymers rather than proteins.",
          false,
        ],
      ],
      "Examples make macromolecule categories easier to remember. Glucose is a sugar, phospholipids are membrane lipids, DNA and RNA are nucleic acids, and antibodies are proteins.",
    ),
    makeQuestion(
      34,
      "easy",
      "Which statements correctly describe amino acid side chains?",
      [
        ["Side chains can differ in charge.", true],
        ["Side chains can differ in polarity and hydrophobicity.", true],
        ["Side chains help proteins fold and interact.", true],
        ["Side chains are the sugar-phosphate backbone of DNA.", false],
      ],
      "Proteins are made from amino acids, and side chains give amino acids different chemical personalities. Those differences help determine folding, binding, catalysis, transport, signaling, and structural roles.",
    ),
    makeQuestion(
      35,
      "easy",
      "Which statements correctly describe energy in cells?",
      [
        [
          "Adenosine triphosphate (ATP) is a common cellular energy currency.",
          true,
        ],
        [
          "Catabolism breaks molecules down and can release usable energy.",
          true,
        ],
        ["Anabolism builds molecules and usually consumes energy.", true],
        ["Metabolism is a storage library made from chromosomes.", false],
      ],
      "Cells need energy to build, repair, move, transport, signal, and replicate. ATP, catabolism, anabolism, and gradients are basic terms for understanding how cells maintain organization.",
    ),
    makeQuestion(
      36,
      "easy",
      "Which statements correctly describe cell organization?",
      [
        ["Membranes separate environments.", true],
        ["Mitochondria help convert nutrient energy into ATP.", true],
        [
          "The cytoskeleton helps with shape, movement, and internal transport.",
          true,
        ],
        [
          "The Golgi apparatus is the inherited DNA sequence of the organism.",
          false,
        ],
      ],
      "Cells are organized systems with specialized structures. Membranes, mitochondria, cytoskeleton, endoplasmic reticulum, Golgi apparatus, ribosomes, and the nucleus support different parts of cellular function.",
    ),
    makeQuestion(
      37,
      "easy",
      "Which statements correctly describe cell signaling?",
      [
        [
          "Cells can detect hormones, nutrients, damage, pathogens, and neighboring cells.",
          true,
        ],
        ["Signal transduction converts a signal into internal change.", true],
        [
          "Signaling can affect gene expression, movement, secretion, growth, or survival.",
          true,
        ],
        ["Cell signaling is the same as a molecule's atomic mass.", false],
      ],
      "Cells use signaling to sense and respond. A signal can be detected by a receptor, passed through intracellular proteins, amplified or regulated by feedback, and converted into behavior.",
    ),
    makeQuestion(
      38,
      "easy",
      "Which statements correctly describe division, differentiation, apoptosis, and cancer?",
      [
        [
          "Mitosis produces daughter cells after DNA is copied and separated.",
          true,
        ],
        ["Differentiation lets cells become specialized.", true],
        ["Apoptosis is programmed cell death.", true],
        [
          "Cancer is ordinary stable homeostasis with no change in growth control.",
          false,
        ],
      ],
      "The course later connects growth, specialization, death, and cancer. Cells need regulated division and regulated death, while cancer can involve disrupted growth control, survival, tissue constraints, and evolution.",
    ),
    makeQuestion(
      39,
      "easy",
      "Which statements correctly describe immunity?",
      [
        ["Innate immunity is fast and broad.", true],
        ["Adaptive immunity is more specific and can form memory.", true],
        ["Antibodies are proteins that bind specific targets.", true],
        ["T cells are the lipid tails that make membranes hydrophobic.", false],
      ],
      "The immune system is a distributed recognition and response system. Innate immunity provides broad early defense, while adaptive immunity uses B cells, antibodies, T cells, and memory.",
    ),
    makeQuestion(
      40,
      "easy",
      "Which statements correctly describe biological information flow?",
      [
        ["DNA stores information in nucleotide sequence.", true],
        ["RNA can act as a working copy or regulatory molecule.", true],
        ["Proteins perform much cellular work.", true],
        ["The central pattern is lipid to oxygen gas to chromosome.", false],
      ],
      "A helpful starting model is DNA, RNA, and protein connected by information flow. DNA stores information, RNA helps carry or regulate it, and proteins do many jobs in the cell.",
    ),
    makeQuestion(
      41,
      "easy",
      "Which statements correctly describe mutation and evolution?",
      [
        ["A mutation is a change in DNA sequence.", true],
        [
          "Natural selection changes variant frequencies when inherited differences affect survival or reproduction.",
          true,
        ],
        ["Genetic drift is random change in variant frequencies.", true],
        [
          "Evolution is the same as a drug being absorbed into the bloodstream.",
          false,
        ],
      ],
      "Mutation creates or changes variation, and evolution changes variant frequencies over time. These ideas matter for antibiotic resistance, viral evolution, cancer progression, immune escape, and inherited disease risk.",
    ),
    makeQuestion(
      42,
      "easy",
      "Which statements correctly describe basic biotechnology ideas?",
      [
        ["Sequencing reads nucleotide order.", true],
        ["Plasmids can carry inserted genes into bacteria.", true],
        ["Messenger RNA medicines deliver temporary instructions.", true],
        ["CRISPR-based tools are mainly blood-pressure sensors.", false],
      ],
      "Biotechnology uses biological information systems. Scientists can read DNA, move genes with plasmids, deliver messenger RNA instructions, and use sequence-guided tools such as CRISPR.",
    ),
    makeQuestion(
      43,
      "easy",
      "Which statements correctly describe basic pharmacology terms?",
      [
        ["An agonist activates a receptor or pathway.", true],
        [
          "An antagonist blocks activation or prevents a usual signal effect.",
          true,
        ],
        ["Pharmacokinetics asks what the body does to the drug.", true],
        ["Efficacy means the color of a pill or capsule.", false],
      ],
      "Pharmacology vocabulary helps connect drugs to biology. Agonists and antagonists describe receptor effects, pharmacokinetics describes exposure, and efficacy describes the maximum effect a drug can produce.",
    ),
    makeQuestion(
      44,
      "easy",
      "Which statements correctly describe patient variation and precision medicine?",
      [
        [
          "Patients can differ in genetics, age, organ function, immune state, and prior treatments.",
          true,
        ],
        [
          "A companion diagnostic can help identify likely responders or patients at risk of harm.",
          true,
        ],
        [
          "Precision medicine uses biologically meaningful subgroups to improve decisions.",
          true,
        ],
        [
          "Patient variation is best ignored when studying drug response.",
          false,
        ],
      ],
      "Medicine is probabilistic because patients differ in many ways. Biomarkers, diagnostics, and precision medicine try to connect those differences to better diagnosis, monitoring, treatment selection, and safety.",
    ),
    makeQuestion(
      45,
      "easy",
      "Which statements correctly describe models, evidence, and AI?",
      [
        [
          "Cell cultures and animal models are simplified representations of biology.",
          true,
        ],
        [
          "Control groups and randomization can help test clinical claims.",
          true,
        ],
        [
          "AI predictions still need biological interpretation and validation.",
          true,
        ],
        [
          "A high score on one dataset is the same as proven patient benefit.",
          false,
        ],
      ],
      "Biomedical claims move from mechanism to measurement to evidence. Models and AI can help, but simplified systems, biased data, noisy labels, workflow fit, and patient outcome evidence still matter.",
    ),
    makeQuestion(
      46,
      "medium",
      "Which statements are useful ways to connect chemistry to biology?",
      [
        ["Atoms can form molecules.", true],
        [
          "Molecules can interact through shape, charge, polarity, and bonding.",
          true,
        ],
        ["Molecular interactions can affect cells.", true],
        ["Cell behavior can affect tissues, organs, and patients.", true],
      ],
      "The course moves across scales. A learner does not need advanced chemistry first, but they do need the idea that molecular structure and interactions can create larger biological effects.",
    ),
    makeQuestion(
      47,
      "medium",
      "Which statements are useful starting points for water, lipids, and proteins?",
      [
        ["Water's polarity helps explain solubility.", true],
        ["Hydrophobic effects help explain lipid bilayers.", true],
        [
          "Hydrophobic and polar amino acid side chains help explain protein folding.",
          true,
        ],
        [
          "Weak interactions can still produce specific binding when many act together.",
          true,
        ],
      ],
      "Water, lipids, and proteins are connected by chemistry. Polarity, hydrophobicity, and weak interactions help explain membranes, folded proteins, molecular recognition, enzymes, and drug binding.",
    ),
    makeQuestion(
      48,
      "medium",
      "Which statements correctly connect building blocks to large biological molecules?",
      [
        ["Amino acids build proteins.", true],
        ["Nucleotides build DNA and RNA.", true],
        ["Monosaccharides can build carbohydrate polymers.", true],
        [
          "Fatty acids and related hydrophobic units help build many lipid structures.",
          true,
        ],
      ],
      "Large biological molecules are often built from smaller recurring units. Knowing the building blocks helps students understand digestion, biosynthesis, membranes, genetic information, and protein function.",
    ),
    makeQuestion(
      49,
      "medium",
      "Which statements are useful starting points for cells as systems?",
      [
        ["Cells use membranes and compartments to organize chemistry.", true],
        ["Cells control transport across boundaries.", true],
        ["Cells sense signals and change internal state.", true],
        [
          "Cells allocate energy and materials for growth, repair, or response.",
          true,
        ],
      ],
      "A cell is more than a container of molecules. It is a bounded, energy-using, information-processing system that organizes reactions, senses conditions, controls transport, and manages resources.",
    ),
    makeQuestion(
      50,
      "medium",
      "Which statements are useful starting points for genes and regulation?",
      [
        ["Genes influence traits through products and regulation.", true],
        [
          "Different cell types can share DNA but use different gene expression programs.",
          true,
        ],
        ["Transcription factors help regulate gene use.", true],
        [
          "Epigenetic regulation can change access to DNA without changing the base sequence.",
          true,
        ],
      ],
      "DNA is better viewed as an information library than as a simple deterministic blueprint. Cells regulate which instructions are used, how strongly they are used, and how gene products interact with context.",
    ),
    makeQuestion(
      51,
      "medium",
      "Which statements are useful starting points for evolution in medicine?",
      [
        ["Antibiotic resistance can evolve under treatment pressure.", true],
        [
          "Viruses can evolve variants that change recognition or treatment response.",
          true,
        ],
        ["Tumor cell populations can evolve inside the body.", true],
        [
          "Selection, drift, inheritance, and variation help explain changing populations.",
          true,
        ],
      ],
      "Evolution is not just background theory. It helps explain resistance, pathogen adaptation, cancer progression, immune escape, inherited disease risk, and why treatment can change the population it acts on.",
    ),
    makeQuestion(
      52,
      "medium",
      "Which statements are useful starting points for drugs and dosing?",
      [
        [
          "A drug can act by binding a receptor, enzyme, channel, transporter, or immune mediator.",
          true,
        ],
        ["Dose can affect benefit and toxicity.", true],
        ["Pharmacokinetics affects exposure over time.", true],
        ["Pharmacodynamics connects exposure to biological effect.", true],
      ],
      "Drugs perturb biological systems rather than acting in isolation. Dose, exposure, target binding, downstream physiology, toxicity, and patient variation together shape treatment response.",
    ),
    makeQuestion(
      53,
      "medium",
      "Which statements are useful starting points for infection?",
      [
        ["Bacteria are cells with their own biological machinery.", true],
        ["Viruses use host-cell machinery to reproduce.", true],
        [
          "Symptoms can come from pathogen damage, immune response, inflammation, or disrupted physiology.",
          true,
        ],
        [
          "Treatment choice depends on pathogen biology and host context.",
          true,
        ],
      ],
      "Infection brings together molecules, cells, immunity, evolution, and treatment. The basic distinction between bacteria and viruses prepares students for antibiotics, antivirals, vaccines, and resistance.",
    ),
    makeQuestion(
      54,
      "medium",
      "Which statements are useful starting points for measurement and evidence?",
      [
        [
          "A diagnostic needs enough accuracy for the decision it supports.",
          true,
        ],
        [
          "A biomarker can be associated with disease without causing it.",
          true,
        ],
        ["A clinical endpoint is an outcome used to judge effect.", true],
        [
          "Patient benefit needs evidence beyond a plausible biological story.",
          true,
        ],
      ],
      "Modern biomedicine depends on measurement, but measurements need interpretation. A useful biomarker, diagnostic, endpoint, or AI prediction must be connected to the decision and outcome that matter.",
    ),
    makeQuestion(
      55,
      "medium",
      "Which statements are useful starting points for AI in biomedicine?",
      [
        [
          "AI can help with images, genomics, proteins, drug discovery, and patient stratification.",
          true,
        ],
        ["AI systems can fail when data are biased or labels are noisy.", true],
        ["Population shift can reduce model performance.", true],
        [
          "Biology, measurement, workflow, safety, and evidence remain important.",
          true,
        ],
      ],
      "AI can accelerate parts of biology and medicine, but it does not remove the need for biological reasoning. Data quality, validation, workflow, safety, and evidence determine whether a system is trustworthy.",
    ),
    makeQuestion(
      56,
      "hard",
      "Which statements describe graph and dose-response ideas that help with the course?",
      [
        [
          "A curve can show how one measured variable changes as another variable changes.",
          true,
        ],
        [
          "A dose-response curve can rise and then plateau when targets or downstream systems saturate.",
          true,
        ],
        ["Potency is about how much drug is needed for an effect.", true],
        [
          "Toxicity is about harmful effects that can increase with exposure.",
          true,
        ],
      ],
      "Students do not need advanced math, but they should be comfortable with simple curves and comparisons. Dose-response reasoning appears in pharmacology, safety, therapeutic windows, biomarkers, and clinical evidence.",
    ),
    makeQuestion(
      57,
      "hard",
      "Which statements describe structure-function reasoning in simple terms?",
      [
        ["A molecule's shape can affect what it binds.", true],
        ["Charge and polarity can affect how molecules interact.", true],
        ["Protein sequence can affect folding.", true],
        ["Cellular context can change the effect of a molecule.", true],
      ],
      "Structure-function reasoning means that what something is made of and how it is shaped affect what it can do. The course uses this for proteins, enzymes, receptors, antibodies, drugs, membranes, and nucleic acids.",
    ),
    makeQuestion(
      58,
      "hard",
      "Which statements describe regulation in simple terms?",
      [
        [
          "Regulation means changing activity, amount, location, or timing of biological processes.",
          true,
        ],
        [
          "Feedback can make a response stronger or help bring a variable back toward a useful range.",
          true,
        ],
        [
          "Cells regulate gene expression, protein activity, transport, division, and death.",
          true,
        ],
        [
          "Disease can arise when regulation fails across molecules, cells, tissues, or organs.",
          true,
        ],
      ],
      "Regulation is one of the main ideas of the crash course. It lets learners connect molecular switches, feedback loops, cell decisions, physiology, disease, drug effects, and patient variation.",
    ),
    makeQuestion(
      59,
      "hard",
      "Which statements describe therapeutic modalities at a beginner level?",
      [
        [
          "Small molecules are chemically manufactured compounds that can bind targets.",
          true,
        ],
        [
          "Biologics include large molecules such as antibodies or therapeutic proteins.",
          true,
        ],
        [
          "Messenger RNA medicines deliver temporary instructions for cells to make a protein.",
          true,
        ],
        [
          "Cell therapy can involve modifying or selecting cells and giving them to a patient.",
          true,
        ],
      ],
      "Modern medicine uses several kinds of interventions. The names are less important than the basic differences in what is delivered, where it acts, how long it lasts, and what evidence is needed.",
    ),
    makeQuestion(
      60,
      "hard",
      "Which statements describe the preparation mindset for the crash course?",
      [
        [
          "Start with simple definitions before trying to synthesize mechanisms.",
          true,
        ],
        [
          "Connect molecules, cells, genes, physiology, disease, treatments, and evidence as layers.",
          true,
        ],
        [
          "Treat uncertainty and patient variation as normal parts of biology and medicine.",
          true,
        ],
        ["Use basic vocabulary to make later lectures easier to follow.", true],
      ],
      "Lecture 0 should make the later crash course feel approachable rather than surprising. Mastering the basics here gives students a vocabulary bridge into chemistry, cells, genetics, physiology, biotechnology, evidence, and AI.",
    ),
  ];

export const BiologyChemistryLifeScienceL0Questions =
  BiologyChemistryForLifeScienceLecture0PreparationQuestions;
