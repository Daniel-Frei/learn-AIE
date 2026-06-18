import { Question } from "../../../quiz";

type PrepDifficulty = "easy" | "medium" | "hard";
type OptionSeed = readonly [text: string, isCorrect: boolean];

function makeQuestion(
  number: number,
  difficulty: PrepDifficulty,
  prompt: string,
  optionSeeds: readonly OptionSeed[],
  explanation: string,
  id = `bio-chem-life-l0-q${String(number).padStart(2, "0")}`,
): Question {
  if (optionSeeds.length !== 4) {
    throw new Error(`Lecture 0 question ${number} must have four options.`);
  }

  return {
    id,
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
        ["A molecule made by bonding several atoms together.", false],
        [
          "A type of matter defined by atoms with the same proton number.",
          false,
        ],
        ["A charged version of a molecule dissolved in water.", false],
      ],
      "An atom is a basic unit of matter, while an element is a type of atom and a molecule is a bonded group of atoms. Keeping these early distinctions clear makes later ideas such as bonds, water, DNA, and proteins less mysterious.",
    ),
    makeQuestion(
      2,
      "easy",
      "Which statement best describes an element?",
      [
        ["A substance defined by atoms with the same number of protons.", true],
        ["A group of atoms held together in a particular arrangement.", false],
        ["A version of a molecule with uneven charge distribution.", false],
        ["A mixture of several molecules dissolved in water.", false],
      ],
      "An element is defined by proton number, so carbon atoms are carbon because of their protons. Molecules, polar substances, and mixtures are different ideas built from or involving elements.",
    ),
    makeQuestion(
      3,
      "easy",
      "Which statement best describes a molecule?",
      [
        ["A group of atoms held together by chemical bonds.", true],
        ["One atom type defined by its proton number.", false],
        ["A single unbonded atom with no connection to other atoms.", false],
        [
          "A loose mixture of substances with no bonded atom arrangement.",
          false,
        ],
      ],
      "A molecule is made when atoms are chemically bonded into a particular arrangement. It is different from an element, an isolated atom, or a loose mixture because the atoms are connected as one molecular unit.",
    ),
    makeQuestion(
      4,
      "easy",
      "Which statement best describes a chemical bond?",
      [
        ["An interaction that holds atoms or molecular parts together.", true],
        [
          "A whole molecule rather than the interaction holding its atoms together.",
          false,
        ],
        [
          "A property of the nucleus that determines which element an atom is.",
          false,
        ],
        [
          "A brief collision between molecules that does not hold atoms together.",
          false,
        ],
      ],
      "A chemical bond is an interaction that helps hold atoms or molecular parts together. Bonds are not whole molecules, proton-number definitions, or brief collisions by themselves, but they help explain why molecules have shapes, stability, and biological behavior.",
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
        ["A loose bag of molecules with no boundary or regulation.", false],
        ["A living structure defined by having a nucleus inside it.", false],
        [
          "A nonliving molecule that stores one kind of biological information.",
          false,
        ],
      ],
      "A cell is the basic living unit emphasized across the crash course. Cells have boundaries and regulation, some cells lack a nucleus, and DNA is an information molecule inside cells rather than a complete living cell.",
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
        ["A protein chain built by reading messenger RNA.", false],
        ["A lipid bilayer that controls what enters and leaves a cell.", false],
        [
          "A molecule used mainly as a short-term cellular energy currency.",
          false,
        ],
      ],
      "Deoxyribonucleic acid (DNA) stores inherited information in the order of its bases. Proteins, membranes, and ATP are different molecule types with different jobs, so distinguishing them early prevents later confusion.",
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
        [
          "A genetic storage polymer whose sequence uses A, T, G, and C bases.",
          false,
        ],
        [
          "A small circular DNA molecule that can carry genes in bacteria.",
          false,
        ],
        [
          "A lipid molecule that forms the hydrophobic part of a membrane.",
          false,
        ],
      ],
      "Proteins are built from amino acids and can fold into shapes that do work. DNA and plasmids are nucleic acids, while membrane lipids are a different molecule class with different building blocks and roles.",
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
        ["A solid wall that lets no molecules or signals cross.", false],
        [
          "A layer made mostly from DNA strands wrapped around proteins.",
          false,
        ],
        [
          "A boundary made from water molecules rather than lipids and proteins.",
          false,
        ],
      ],
      "Membranes let cells and organelles separate inside from outside. They are selective, not sealed walls, and their lipid and protein structure supports transport, signaling, gradients, and compartmentalized chemistry.",
    ),
    makeQuestion(
      9,
      "easy",
      "Which statement best describes an enzyme?",
      [
        ["A biological catalyst that helps reactions happen faster.", true],
        [
          "A molecule that is consumed as the main raw material of the reaction it speeds.",
          false,
        ],
        [
          "A catalyst that changes which reaction is energetically possible overall.",
          false,
        ],
        [
          "A folded molecule whose shape is unrelated to what it can bind.",
          false,
        ],
      ],
      "An enzyme is a catalyst, meaning it speeds a reaction without being used up as the reaction's raw material. Enzymes lower activation energy and depend on shape and chemistry, but they do not rewrite the overall energy balance of a reaction.",
    ),
    makeQuestion(
      10,
      "easy",
      "Which statement best describes a gene?",
      [
        ["A DNA sequence that can be used to make a functional product.", true],
        [
          "A visible trait that appears without regulation or environment.",
          false,
        ],
        [
          "A protein product rather than the DNA instruction used to make it.",
          false,
        ],
        [
          "A chromosome that contains no regulatory regions or neighboring genes.",
          false,
        ],
      ],
      "A gene is a usable stretch of DNA, often connected to making a protein or functional RNA. It is not the same as a whole trait or the protein product, because expression, regulation, environment, and other genes shape the outcome.",
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
        ["The signaling molecule that binds and is detected.", false],
        ["A passive lipid tail with no role in signal detection.", false],
        [
          "A molecule that responds to many unrelated signals with the same activity change.",
          false,
        ],
      ],
      "Receptors are central to cell signaling and pharmacology. A ligand is the signal being detected, while a receptor is the molecule that binds or senses it and helps convert that event into a cellular response.",
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
        [
          "Any microbe present in the body, including harmless gut bacteria.",
          false,
        ],
        ["A host immune cell that recognizes infected cells.", false],
        [
          "A beneficial organism used to make a medicine in biotechnology.",
          false,
        ],
      ],
      "A pathogen is an organism or agent that can cause disease. Not every microbe is a pathogen, and host immune cells or production organisms are different biological categories with different roles.",
    ),
    makeQuestion(
      13,
      "easy",
      "A patient receives a molecule that binds an enzyme and reduces activity in a disease-relevant pathway. Which description best classifies it as a drug?",
      [
        [
          "An intervention intended to perturb a biological process for diagnosis, prevention, or treatment.",
          true,
        ],
        [
          "A biomarker, because target binding measures disease state but does not alter the pathway.",
          false,
        ],
        [
          "A molecule that must be made by the patient's own cells to count as treatment.",
          false,
        ],
        [
          "A diagnostic measurement, because any dose-dependent biological effect rules out drug action.",
          false,
        ],
      ],
      "A drug is an intervention that changes biology, often through a target such as an enzyme, receptor, transporter, or pathway component. Biomarkers and diagnostics measure or classify biological state, while a drug is judged by its intended perturbation and its benefit-risk evidence.",
    ),
    makeQuestion(
      14,
      "easy",
      "A blood protein level tracks inflammation and may help monitor disease, but changing the protein has not been shown to improve outcomes. Which description best fits the term biomarker?",
      [
        [
          "A measurable indicator of biological state that still needs validation for the decision it supports.",
          true,
        ],
        [
          "A treatment target whose measurement automatically proves that lowering it will help patients.",
          false,
        ],
        [
          "A direct disease cause whenever its value differs between healthy and sick groups.",
          false,
        ],
        [
          "A diagnostic or monitoring tool that is useful for every disease decision once it can be measured.",
          false,
        ],
      ],
      "A biomarker is a measurement that says something about biological state, but the measurement has to be validated for a specific use. Association with disease does not automatically make the marker causal, therapeutic, or useful for every clinical decision.",
    ),
    makeQuestion(
      15,
      "easy",
      "A therapy has a plausible mechanism and improves a marker in a model system. Which statement best describes the clinical evidence still needed?",
      [
        [
          "Patient-level information that tests whether the claim holds up in the intended clinical setting.",
          true,
        ],
        [
          "A mechanism-only explanation, because a plausible pathway is enough once the target is known.",
          false,
        ],
        [
          "A biomarker change treated as patient benefit without showing that the marker supports that decision.",
          false,
        ],
        [
          "A simplified model result treated as sufficient because models remove patient heterogeneity.",
          false,
        ],
      ],
      "Clinical evidence connects biological claims to patient outcomes in the setting where the claim will be used. Mechanisms, biomarkers, and model systems can be promising, but they do not by themselves show benefit, safety, or decision usefulness for real patients.",
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
          "Nitrogen is the main element that makes lipid tails hydrophobic.",
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
        [
          "A monomer is the finished long chain rather than a building block.",
          false,
        ],
        ["A polymer is a single small building block before joining.", false],
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
        ["Hydrolysis joins building blocks by removing water.", false],
        ["Dehydration reactions break polymers by adding water.", false],
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
          "Carbohydrates are usually hydrophobic molecules that form membrane bilayers.",
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
        [
          "Transcription means building an amino acid chain at a ribosome.",
          false,
        ],
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
        ["Ribosomes store DNA in eukaryotic cells.", false],
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
          "Active transport means drifting down a gradient without energy use.",
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
        ["A ligand is the same structure as the receptor it binds.", false],
        [
          "A receptor is the downstream response rather than the detector.",
          false,
        ],
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
        ["Negative feedback means a response keeps amplifying itself.", false],
        [
          "Positive feedback means a response cancels the original change and restores baseline.",
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
          "Metabolic disease is mainly infection by a virus or bacterium.",
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
          "Weak interactions are too weak to influence molecular shape or binding.",
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
        [
          "Metabolism means energy use but not molecule breakdown or building.",
          false,
        ],
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
          "The Golgi apparatus is the compartment where DNA is stored and read directly.",
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
        [
          "Cell signaling is passive movement with no receptor or internal response.",
          false,
        ],
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
          "Cancer is normal differentiation that simply makes cells specialized.",
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
        [
          "T cells are antibodies that float freely and bind targets directly.",
          false,
        ],
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
        [
          "The central pattern is DNA becoming protein directly without an RNA step.",
          false,
        ],
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
          "Evolution means an individual organism deliberately changes its DNA to adapt.",
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
        [
          "CRISPR-based tools edit DNA by choosing targets without sequence matching.",
          false,
        ],
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
        [
          "Efficacy means how quickly the body removes a drug over time.",
          false,
        ],
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
          "A high score on one dataset is enough to choose patient treatment.",
          false,
        ],
      ],
      "Biomedical claims move from mechanism to measurement to evidence. Models and AI can help, but simplified systems, biased data, noisy labels, workflow fit, and patient outcome evidence still matter.",
    ),
    makeQuestion(
      46,
      "medium",
      "Which statements usefully connect chemistry to biology without overclaiming?",
      [
        ["Atoms can form molecules.", true],
        [
          "Molecules can interact through shape, charge, polarity, and bonding.",
          true,
        ],
        ["Molecular interactions can affect cells.", true],
        [
          "Molecular structure alone fully predicts patient outcomes without considering cells, tissues, or evidence.",
          false,
        ],
      ],
      "The course moves across scales from atoms and molecules toward cells and patients. Chemistry helps explain biology, but molecular structure by itself is not enough to predict clinical outcomes without cellular context, tissue-level effects, and evidence.",
      "bio-chem-life-l0-q61",
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
          "Hydrophobic effects make membranes independent of water's chemistry.",
          false,
        ],
      ],
      "Water, lipids, and proteins are connected by chemistry. Polarity, hydrophobicity, and weak interactions help explain membranes, folded proteins, molecular recognition, enzymes, and drug binding; hydrophobic organization depends on water rather than being separate from it.",
      "bio-chem-life-l0-q62",
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
          "The same monomer type builds proteins, nucleic acids, and carbohydrates, so building-block identity is not important.",
          false,
        ],
      ],
      "Large biological molecules are often built from smaller recurring units, but the identity of those units matters. Amino acids, nucleotides, sugars, and lipid components support different structures and functions, so confusing their building blocks creates downstream errors.",
      "bio-chem-life-l0-q63",
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
          "A membrane boundary makes transport and internal chemistry chemically unregulated.",
          false,
        ],
      ],
      "A cell is more than a container of molecules. It is a bounded, energy-using, information-processing system that organizes reactions, senses conditions, controls transport, and manages resources; membranes create regulated boundaries rather than removing the need for regulation.",
      "bio-chem-life-l0-q64",
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
          "Epigenetic regulation changes gene use only by rewriting the DNA base sequence.",
          false,
        ],
      ],
      "DNA is better viewed as an information library than as a simple deterministic blueprint. Cells regulate which instructions are used, how strongly they are used, and how gene products interact with context; epigenetic mechanisms can alter DNA access without changing the base sequence.",
      "bio-chem-life-l0-q65",
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
          "Medical evolution requires organisms or cells to deliberately change their DNA in response to treatment.",
          false,
        ],
      ],
      "Evolution is not just background theory. It helps explain resistance, pathogen adaptation, cancer progression, immune escape, inherited disease risk, and why treatment can change the population it acts on, but the change is population-level rather than deliberate individual adaptation.",
      "bio-chem-life-l0-q66",
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
        [
          "Pharmacodynamics describes what the body does to the drug before it reaches a target.",
          false,
        ],
      ],
      "Drugs perturb biological systems rather than acting in isolation. Dose, exposure, target binding, downstream physiology, toxicity, and patient variation together shape treatment response; pharmacokinetics tracks exposure over time, while pharmacodynamics connects exposure to biological effect.",
      "bio-chem-life-l0-q67",
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
          "Because viruses use host-cell machinery, antibiotics that target bacterial ribosomes are broadly effective against viral infections.",
          false,
        ],
      ],
      "Infection brings together molecules, cells, immunity, evolution, and treatment. The basic distinction between bacteria and viruses prepares students for antibiotics, antivirals, vaccines, and resistance; bacterial targets do not automatically apply to viruses that depend on host-cell machinery.",
      "bio-chem-life-l0-q68",
    ),
    makeQuestion(
      54,
      "medium",
      "A team wants to decide whether a new blood test should guide treatment choice. Which checks belong in that reasoning?",
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
          "A disease association is enough to use the test for treatment selection without outcome evidence.",
          false,
        ],
      ],
      "Modern biomedicine depends on measurement, but measurements need interpretation for a specific decision. A useful biomarker, diagnostic, endpoint, or prediction must connect accuracy, mechanism, clinical context, and patient-relevant outcomes rather than treating association alone as clinical usefulness.",
      "bio-chem-life-l0-q69",
    ),
    makeQuestion(
      55,
      "medium",
      "An AI model predicts disease risk from images collected at one hospital. Which cautions should a learner keep in view before trusting it for patient care?",
      [
        [
          "The model could learn site-specific image artifacts instead of disease biology.",
          true,
        ],
        ["AI systems can fail when data are biased or labels are noisy.", true],
        ["Population shift can reduce model performance.", true],
        [
          "A high score on the original hospital's data proves the model is safe across patient populations.",
          false,
        ],
      ],
      "AI can accelerate parts of biology and medicine, but a strong-looking prediction is not enough for patient care. Data quality, site artifacts, population shift, label reliability, biological interpretation, workflow fit, safety, and evidence determine whether the model is trustworthy.",
      "bio-chem-life-l0-q70",
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
          "A dose-response plateau means additional exposure cannot add toxicity.",
          false,
        ],
      ],
      "Students do not need advanced math, but they should be comfortable with simple curves and comparisons. Dose-response reasoning appears in pharmacology, safety, therapeutic windows, biomarkers, and clinical evidence; a plateau in desired effect does not prove extra exposure is harmless.",
      "bio-chem-life-l0-q71",
    ),
    makeQuestion(
      57,
      "hard",
      "Which statements describe structure-function reasoning in simple terms?",
      [
        ["A molecule's shape can affect what it binds.", true],
        ["Charge and polarity can affect how molecules interact.", true],
        ["Protein sequence can affect folding.", true],
        [
          "Once a protein sequence is known, charge, shape, binding partners, and cellular context become irrelevant.",
          false,
        ],
      ],
      "Structure-function reasoning means that what something is made of and how it is shaped affect what it can do. The course uses this for proteins, enzymes, receptors, antibodies, drugs, membranes, and nucleic acids, while cellular context and interaction chemistry still matter.",
      "bio-chem-life-l0-q72",
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
          "Positive and negative feedback mainly describe molecule shape, not control of activity over time.",
          false,
        ],
      ],
      "Regulation is one of the main ideas of the crash course. It lets learners connect molecular switches, feedback loops, cell decisions, physiology, disease, drug effects, and patient variation; feedback is about control of processes, not merely the physical shape of a molecule.",
      "bio-chem-life-l0-q73",
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
          "Therapeutic modalities differ mainly in brand name rather than in what is delivered or where it acts.",
          false,
        ],
      ],
      "Modern medicine uses several kinds of interventions. The names are less important than the basic differences in what is delivered, where it acts, how long it lasts, and what evidence is needed; modality choice is not just branding.",
      "bio-chem-life-l0-q74",
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
