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
      "Which statements are useful starting assumptions for studying biology and chemistry for medicine?",
      [
        ["Cells are made from atoms and molecules.", true],
        ["Living systems use energy to maintain organization.", true],
        ["Molecular information can affect cell behavior.", true],
        [
          "Visible disease can have molecular, cellular, and physiological causes.",
          true,
        ],
      ],
      "The course repeatedly connects molecules to cells, tissues, organisms, and medicine. These assumptions help learners treat biology as mechanism and regulation rather than as isolated vocabulary.",
    ),
    makeQuestion(
      2,
      "easy",
      "Which statements correctly distinguish atoms, elements, and molecules?",
      [
        ["An element is defined by one kind of atom.", true],
        ["A molecule contains atoms held together by chemical bonds.", true],
        [
          "A molecule is defined as a loose mixture of unrelated element samples.",
          false,
        ],
        [
          "An element is defined by how many different molecules are mixed together.",
          false,
        ],
      ],
      "Atoms are the basic units, elements are atom types, and molecules are bonded atom groups. Keeping these terms separate makes later ideas such as carbon compounds, water, salts, proteins, and DNA easier to understand.",
    ),
    makeQuestion(
      3,
      "easy",
      "Which statements correctly describe ions and polarity?",
      [
        ["An ion has a net electrical charge.", true],
        [
          "A polar molecule has perfectly even charge distribution everywhere.",
          false,
        ],
        [
          "Charge and polarity mainly determine molecular mass rather than interactions with water.",
          false,
        ],
        [
          "Net charge is the defining property that makes a molecule a lipid bilayer.",
          false,
        ],
      ],
      "An ion is defined by net charge. Polarity and solubility are related ideas that students will use soon, but the incorrect options here reverse their definitions or confuse charged molecules with membranes.",
    ),
    makeQuestion(
      4,
      "easy",
      "Which statement best describes a covalent bond?",
      [
        ["Atoms share electrons to form a relatively stable bond.", true],
        [
          "Water moves across a membrane toward higher solute concentration.",
          false,
        ],
        ["A receptor detects a hormone and triggers signaling.", false],
        ["A ribosome reads messenger RNA to build protein.", false],
      ],
      "A covalent bond is a chemical bond based on shared electrons. The other choices describe osmosis, signaling, and translation, which are important later but are not definitions of covalent bonding.",
    ),
    makeQuestion(
      5,
      "easy",
      "Which statements correctly describe water as a biological solvent?",
      [
        ["Water is polar.", true],
        ["Water often dissolves charged or polar substances well.", true],
        [
          "Water dissolves nonpolar oils more readily than ionic salts because oils are larger.",
          false,
        ],
        ["Water's polarity is unrelated to biological organization.", false],
      ],
      "Water's polarity helps it interact with ions and polar groups. Nonpolar molecules interact poorly with water, which helps explain membranes, protein folding, and why cells need organized compartments.",
    ),
    makeQuestion(
      6,
      "easy",
      "Which statement best describes pH?",
      [
        ["A measure related to hydrogen ion concentration.", true],
        ["A count of amino acids in a protein.", false],
        ["The number of genes in a chromosome.", false],
        ["The rate at which a virus enters a cell.", false],
      ],
      "pH is basic acid-base vocabulary needed for enzymes, blood chemistry, lysosomes, and protein charge. It is not a protein length, gene count, or viral entry rate.",
    ),
    makeQuestion(
      7,
      "easy",
      "Which statements correctly describe organic molecules in biology?",
      [
        ["They contain carbon in their structure.", true],
        [
          "Carbon can form four covalent bonds, supporting complex structures.",
          true,
        ],
        [
          "Proteins, carbohydrates, lipids, and nucleic acids are unrelated to carbon-based biological chemistry.",
          false,
        ],
        [
          "Organic molecule means the same thing as food marketed without synthetic additives.",
          false,
        ],
      ],
      "In chemistry, organic molecules are carbon-containing compounds, not grocery-label categories. The major biological molecule classes rely heavily on carbon chemistry, so saying they are unrelated to it is backwards.",
    ),
    makeQuestion(
      8,
      "easy",
      "Which statements correctly describe monomers and polymers?",
      [
        ["A monomer is a smaller building block.", true],
        ["A polymer is built from repeated building blocks.", true],
        [
          "A polymer is best understood as a tissue-level structure rather than a chain of building blocks.",
          false,
        ],
        ["A monomer is the same thing as a clinical endpoint.", false],
      ],
      "Macromolecules such as proteins, nucleic acids, and many carbohydrates are built from smaller units. This vocabulary helps students understand digestion, biosynthesis, DNA, RNA, and proteins.",
    ),
    makeQuestion(
      9,
      "easy",
      "Which statements correctly describe hydrolysis and dehydration reactions?",
      [
        ["Hydrolysis uses water to break bonds.", true],
        [
          "Dehydration reactions join building blocks while releasing water.",
          true,
        ],
        ["Hydrolysis is the process of reading codons at a ribosome.", false],
        [
          "Dehydration reactions describe cell dehydration rather than bond-forming chemistry.",
          false,
        ],
      ],
      "Hydrolysis and dehydration reactions are basic ways to break and build biological molecules. They prepare students to understand digestion, polymer assembly, and molecular recycling.",
    ),
    makeQuestion(
      10,
      "easy",
      "Which statement best describes carbohydrates in introductory biology?",
      [
        [
          "They often serve as fuel, energy storage, structure, or cell-recognition molecules.",
          true,
        ],
        [
          "They are membrane receptors whose main role is neurotransmitter binding.",
          false,
        ],
        [
          "They are the primary inherited-information polymers in chromosomes.",
          false,
        ],
        ["They are antibiotics that block bacterial ribosomes.", false],
      ],
      "Carbohydrates include simple sugars and larger polysaccharides. They are important for energy and structure, but they are not the same as receptors, DNA, RNA, or antibiotics.",
    ),
    makeQuestion(
      11,
      "easy",
      "Which statements correctly describe lipids and phospholipids?",
      [
        ["Many lipids are partly or mostly hydrophobic.", true],
        [
          "Phospholipids can form bilayers with hydrophilic heads and hydrophobic tails.",
          true,
        ],
        [
          "Lipids are unrelated to membranes, energy storage, and signaling.",
          false,
        ],
        ["Lipids are nucleotide polymers that encode codons.", false],
      ],
      "Lipids are essential for boundaries and signaling because of their hydrophobic or amphipathic chemistry. They are not nucleotides, and the claim that they never support membranes or signaling reverses important introductory biology.",
    ),
    makeQuestion(
      12,
      "easy",
      "Which statement best describes an amino acid?",
      [
        [
          "A building block of proteins with a shared backbone and a variable side chain.",
          true,
        ],
        ["A glucose polymer used for short-term energy storage.", false],
        ["A viral envelope used to enter a host cell.", false],
        [
          "A clinical-trial design feature that prevents selection bias.",
          false,
        ],
      ],
      "Amino acids are the basic units of proteins. The variable side chain is crucial because it gives each amino acid different chemical properties that affect protein folding and function.",
    ),
    makeQuestion(
      13,
      "easy",
      "Which statements correctly describe proteins at a beginner level?",
      [
        [
          "Proteins are chains of amino acids that fold into functional structures.",
          true,
        ],
        ["Protein function can depend on shape and chemical properties.", true],
        [
          "Proteins can act as enzymes, signals, receptors, transporters, antibodies, or structural materials.",
          true,
        ],
        [
          "Proteins are distinct from DNA chromosomes even though genes can encode protein sequences.",
          true,
        ],
      ],
      "Proteins perform much of the active work in cells, while DNA stores sequence information. Understanding amino acids and folding prepares students for enzymes, receptors, antibodies, drug targets, disease mechanisms, and biotechnology.",
    ),
    makeQuestion(
      14,
      "easy",
      "Which statements correctly describe enzymes?",
      [
        ["Enzymes are catalysts.", true],
        ["Enzymes can speed reactions by lowering activation energy.", true],
        [
          "Enzymes work by editing the genetic sequence of each substrate molecule.",
          false,
        ],
        [
          "Enzymes serve as stoichiometric fuel that is used up during catalysis.",
          false,
        ],
      ],
      "Enzymes make biological reaction rates compatible with life by catalysis. They do not act by rewriting substrate genomes, and they are not simply burned as fuel in each reaction cycle.",
    ),
    makeQuestion(
      15,
      "easy",
      "Which statements correctly describe ATP and metabolism?",
      [
        ["ATP is a short-term usable energy currency in cells.", true],
        [
          "Metabolism includes chemical pathways that build and break molecules.",
          true,
        ],
        ["ATP is the same thing as a chromosome.", false],
        [
          "Metabolism means cellular reaction networks become chemically inactive.",
          false,
        ],
      ],
      "ATP and metabolism are needed because cells constantly do chemical work. They build, break, transport, signal, repair, and maintain order rather than remaining static.",
    ),
    makeQuestion(
      16,
      "easy",
      "Which statements correctly describe cells and membranes?",
      [
        ["Cells are bounded systems.", true],
        ["Membranes help regulate what enters and leaves.", true],
        ["Membranes help cells maintain internal conditions.", true],
        ["Membranes can contain proteins for transport or signaling.", true],
      ],
      "Membranes let cells organize chemistry and information. They are selective boundaries with embedded proteins, not inert bags, and they make gradients and signaling possible.",
    ),
    makeQuestion(
      17,
      "easy",
      "Which organelle-function pairings are correct?",
      [
        ["Nucleus: stores genomic DNA in eukaryotic cells.", true],
        ["Ribosome: builds proteins from messenger RNA instructions.", true],
        [
          "Mitochondrion: stores the entire nuclear genome as its primary role.",
          false,
        ],
        [
          "Golgi apparatus: stores the entire hereditary genome as its primary role.",
          false,
        ],
      ],
      "These pairings prepare students for cellular organization without requiring memorization of every organelle detail. Mitochondria help with ATP production, but whole-genome storage in eukaryotic cells is primarily associated with the nucleus.",
    ),
    makeQuestion(
      18,
      "easy",
      "Which statements correctly describe gradients, channels, and pumps?",
      [
        [
          "A gradient is a difference across space, such as concentration or charge.",
          true,
        ],
        ["Channels often allow movement down gradients.", true],
        ["Pumps can use energy to move substances against gradients.", true],
        ["Gradients can store usable energy for cells.", true],
      ],
      "Gradients are central to membranes, neurons, mitochondria, muscles, and physiology. Channels and pumps help cells use and maintain these differences, which is why gradients are treated as both chemistry and control mechanisms.",
    ),
    makeQuestion(
      19,
      "easy",
      "Which statements correctly describe receptors and ligands?",
      [
        [
          "A ligand is a molecule that can bind a target such as a receptor.",
          true,
        ],
        ["A receptor can detect a signal and trigger a response.", true],
        ["Receptor binding can depend on molecular shape and chemistry.", true],
        [
          "A receptor is a carbohydrate storage polymer rather than a signal-detecting molecule.",
          false,
        ],
      ],
      "Receptors and ligands are basic signaling vocabulary. They connect chemistry to cell behavior and later help explain hormones, neurotransmitters, immune recognition, and drug action.",
    ),
    makeQuestion(
      20,
      "easy",
      "Which statement best describes negative feedback?",
      [
        [
          "A response that tends to counteract a change and stabilize a system.",
          true,
        ],
        [
          "A response that reinforces deviation instead of stabilizing a variable.",
          false,
        ],
        ["A chemical bond formed by sharing electrons.", false],
        [
          "A DNA sequence used as a codon rather than a regulatory control loop.",
          false,
        ],
      ],
      "Negative feedback is a control concept used throughout physiology. It helps students reason about homeostasis, blood glucose, temperature, hormones, and drug compensation.",
    ),
    makeQuestion(
      21,
      "medium",
      "Which statements correctly describe the DNA -> RNA -> protein flow?",
      [
        ["DNA can be transcribed into RNA.", true],
        [
          "Messenger RNA can be translated by ribosomes into amino acid chains.",
          true,
        ],
        ["Proteins can perform many cellular functions.", true],
        [
          "The flow is a core way genetic information can influence protein function.",
          true,
        ],
      ],
      "The central dogma is a foundation for later genetics and biotechnology. It explains how stored DNA sequence can influence cellular function through RNA and proteins.",
    ),
    makeQuestion(
      22,
      "medium",
      "Which statements correctly distinguish genes, chromosomes, and genomes?",
      [
        [
          "A gene is a DNA sequence that can produce a functional product.",
          true,
        ],
        [
          "A chromosome is a large DNA-containing structure with many genetic regions.",
          true,
        ],
        [
          "A genome is the complete genetic information of an organism or cell.",
          true,
        ],
        [
          "A gene can encode a protein such as an antibody chain when expressed in the right cells.",
          true,
        ],
      ],
      "These terms form the basic hierarchy of genetic information. Genes are usable segments, chromosomes are larger structures, and the genome refers to the complete set of genetic information.",
    ),
    makeQuestion(
      23,
      "medium",
      "Which statement best describes a codon?",
      [
        [
          "A three-nucleotide unit in RNA or DNA sequence that corresponds to an amino acid or stop signal during translation.",
          true,
        ],
        ["A lipid bilayer surrounding a bacterial cell.", false],
        ["A hormone that lowers blood glucose.", false],
        ["A clinical endpoint that measures survival.", false],
      ],
      "Codons connect nucleotide sequence to amino acid sequence. Students do not need to memorize the codon table, but they need to understand how triplets carry protein-building information.",
    ),
    makeQuestion(
      24,
      "medium",
      "Which statements correctly describe gene expression and cell identity?",
      [
        [
          "Gene expression means how much or when a gene product is made.",
          true,
        ],
        [
          "Different cell types can share nearly the same DNA but express different genes.",
          true,
        ],
        [
          "Signals can change gene expression without changing DNA sequence.",
          true,
        ],
        [
          "Gene expression can change during development, immune responses, or disease.",
          true,
        ],
      ],
      "Gene expression explains why cell types can behave differently without each having a completely different genome. It also connects signaling, development, disease, and drug response.",
    ),
    makeQuestion(
      25,
      "medium",
      "Which statements correctly describe mutation and evolution?",
      [
        ["A mutation is a change in DNA sequence.", true],
        [
          "Natural selection can change which variants become more common.",
          true,
        ],
        ["Genetic drift can change variant frequencies by chance.", true],
        [
          "Evolution can matter in infection, cancer, and drug resistance.",
          true,
        ],
      ],
      "Evolution is a practical medical idea, not just ancient history. Variation, inheritance, selection, and chance help explain pathogens, tumors, resistance, and inherited risk.",
    ),
    makeQuestion(
      26,
      "medium",
      "Which statements correctly compare innate and adaptive immunity?",
      [
        ["Innate immunity is fast and broad.", true],
        ["Adaptive immunity is more specific and can form memory.", true],
        ["Adaptive immunity includes B cells and T cells.", true],
        [
          "Innate and adaptive immunity can interact during immune responses.",
          true,
        ],
      ],
      "Innate and adaptive immunity are complementary. The distinction prepares students for vaccines, inflammation, autoimmune disease, infection, and immunotherapy.",
    ),
    makeQuestion(
      27,
      "medium",
      "Which statements correctly describe antibodies, T cells, and vaccines?",
      [
        ["Antibodies can bind specific molecular targets.", true],
        [
          "T cells can help coordinate immune responses or kill infected or abnormal cells.",
          true,
        ],
        [
          "Vaccines can train immune recognition before dangerous infection.",
          true,
        ],
        ["Vaccines and antibodies depend on molecular recognition.", true],
      ],
      "Adaptive immunity uses molecular recognition and memory. Vaccines exploit these features by training recognition, while antibodies and T cells help connect target recognition to immune action.",
    ),
    makeQuestion(
      28,
      "medium",
      "Which statements correctly distinguish bacteria and viruses?",
      [
        ["Bacteria are cells with their own ribosomes and metabolism.", true],
        ["Viruses require host cells to reproduce.", true],
        [
          "Antibiotics commonly target bacterial processes rather than viral life cycles.",
          true,
        ],
        ["Viruses are the same thing as eukaryotic cells with nuclei.", false],
      ],
      "Bacteria and viruses can both cause disease, but their biology differs sharply. This distinction is required for understanding antibiotics, antivirals, vaccines, and resistance.",
    ),
    makeQuestion(
      29,
      "medium",
      "Which statements correctly describe homeostasis?",
      [
        ["It maintains internal variables within functional ranges.", true],
        ["It often uses sensors, comparison, effectors, and feedback.", true],
        [
          "It can apply to blood glucose, temperature, pH, and blood pressure.",
          true,
        ],
        [
          "It keeps variables fixed at one exact value despite changing conditions.",
          false,
        ],
      ],
      "Homeostasis is regulated stability, not absolute sameness. The body adjusts variables continuously to keep them within ranges compatible with life, so change and stability can coexist in a healthy system.",
    ),
    makeQuestion(
      30,
      "medium",
      "Which statements correctly describe broad disease categories?",
      [
        ["Infection involves host-pathogen interaction.", true],
        ["Cancer involves dysregulated cell growth and evolution.", true],
        [
          "Autoimmune disease involves immune response against self tissues.",
          true,
        ],
        [
          "Metabolic disease is best explained as a viral receptor problem rather than disrupted regulation.",
          false,
        ],
      ],
      "Disease categories are useful when they point to mechanisms. Infection, cancer, autoimmune disease, metabolic disease, genetic disease, and neurodegeneration involve different but sometimes interacting failures.",
    ),
    makeQuestion(
      31,
      "medium",
      "Which statements correctly describe drugs, receptors, agonists, and antagonists?",
      [
        ["Many drugs work by binding proteins or changing pathways.", true],
        ["An agonist activates a receptor or pathway.", true],
        [
          "An antagonist blocks activation or prevents a signal's usual effect.",
          true,
        ],
        [
          "Drugs work by replacing body-wide genetic information as their common mechanism.",
          false,
        ],
      ],
      "Introductory pharmacology builds on receptors, ligands, proteins, and signaling. Agonists and antagonists are basic ways drugs can perturb biological systems.",
    ),
    makeQuestion(
      32,
      "medium",
      "Which statement best describes a therapeutic window?",
      [
        [
          "A dose or exposure range where expected benefit outweighs expected harm.",
          true,
        ],
        ["The exact number of chromosomes in a human cell.", false],
        [
          "A receptor whose binding behavior defines the safe dose range.",
          false,
        ],
        ["A rule that increasing dose steadily improves safety.", false],
      ],
      "A therapeutic window helps students reason about benefit and toxicity together. It prepares them for dose-response curves, patient variation, monitoring, and side effects.",
    ),
    makeQuestion(
      33,
      "medium",
      "Which statements correctly distinguish pharmacokinetics and pharmacodynamics?",
      [
        ["Pharmacokinetics asks what the body does to the drug.", true],
        ["Pharmacodynamics asks what the drug does to the body.", true],
        [
          "ADME stands for absorption, distribution, metabolism, and excretion.",
          true,
        ],
        ["Pharmacodynamics is the same thing as DNA replication.", false],
      ],
      "PK and PD are core vocabulary for understanding medicines. PK describes exposure over time, while PD describes biological effect at targets, pathways, tissues, and patients.",
    ),
    makeQuestion(
      34,
      "medium",
      "Which statements correctly describe biomarkers?",
      [
        ["A biomarker is a measurable indicator of biological state.", true],
        [
          "A biomarker can support diagnosis, prognosis, monitoring, or treatment selection.",
          true,
        ],
        [
          "A biomarker measurement is sufficient by itself to establish disease causation.",
          false,
        ],
        [
          "A biomarker is a transplanted tissue used as the measurement itself.",
          false,
        ],
      ],
      "Biomarkers are measurements, not automatic proof of mechanism. They become useful when validated for the decision they support, such as diagnosis, monitoring, prognosis, or treatment selection.",
    ),
    makeQuestion(
      35,
      "medium",
      "Which statements correctly describe basic biomedical evidence reasoning?",
      [
        ["A control group helps answer compared to what.", true],
        ["Randomization can reduce confounding in expectation.", true],
        ["An endpoint is the outcome used to judge an intervention.", true],
        [
          "A plausible mechanism is sufficient by itself to establish patient benefit.",
          false,
        ],
      ],
      "Students need enough evidence vocabulary to interpret claims without making clinical trials the core course topic. Controls, randomization, and endpoints help separate mechanism, association, and patient outcomes.",
    ),
    makeQuestion(
      36,
      "medium",
      "Which statements correctly describe model systems in biomedical research?",
      [
        [
          "Cell culture can help study mechanisms under controlled conditions.",
          true,
        ],
        [
          "Animal models can reveal organism-level effects but may not match humans perfectly.",
          true,
        ],
        [
          "Organoids can model some tissue-like features while remaining simplified.",
          true,
        ],
        [
          "A model system is expected to reproduce human outcomes with full clinical fidelity.",
          false,
        ],
      ],
      "Model systems are valuable because they simplify complex biology enough to study it. Their limitations matter because simplification can omit immune, vascular, endocrine, behavioral, or disease context.",
    ),
    makeQuestion(
      37,
      "medium",
      "Which statements correctly describe biotechnology at a beginner level?",
      [
        ["Sequencing reads genetic information.", true],
        [
          "Plasmids can carry genes in bacteria and biotechnology systems.",
          true,
        ],
        ["CRISPR-based tools can target genetic sequences.", true],
        [
          "mRNA technologies can deliver temporary protein-making instructions.",
          true,
        ],
      ],
      "Biotechnology depends on the ability to read, move, express, or modify biological information. These tools build directly on DNA, RNA, proteins, cells, and regulation.",
    ),
    makeQuestion(
      38,
      "medium",
      "Which statement best describes careful use of AI in biomedical data?",
      [
        [
          "AI outputs should be validated with appropriate data and interpreted in biological or clinical context.",
          true,
        ],
        [
          "AI predictions eliminate the need for measurements, controls, or validation.",
          false,
        ],
        ["AI establishes causation from correlation in a dataset.", false],
        ["AI is unaffected by biased or incomplete data.", false],
      ],
      "AI can help find patterns, but biomedical use still depends on data quality, biological interpretation, workflow fit, and evidence. Prediction is not automatically causation or patient benefit.",
    ),
    makeQuestion(
      39,
      "medium",
      "Which statements correctly describe precision medicine?",
      [
        [
          "It uses biological or clinical differences to guide treatment decisions.",
          true,
        ],
        ["It can use biomarkers, genomics, imaging, or disease subtype.", true],
        [
          "It aims to identify groups more likely to benefit or be harmed.",
          true,
        ],
        [
          "It means each patient receives a unique custom molecule rather than subgroup-guided care.",
          false,
        ],
      ],
      "Precision medicine is about meaningful stratification, not a guarantee of one custom drug per person. It connects measurement, biology, and treatment choice.",
    ),
    makeQuestion(
      40,
      "medium",
      "Which statements correctly describe systems thinking in biology?",
      [
        [
          "Molecules, cells, tissues, organs, and environments can influence one another.",
          true,
        ],
        ["Disease can emerge from failed regulation across scales.", true],
        ["Systems thinking means molecular mechanisms do not matter.", false],
        ["A systems view requires ignoring feedback loops.", false],
      ],
      "Systems thinking connects mechanisms across scales. It does not replace molecular biology; it explains how molecular changes can propagate through cells, tissues, organisms, and patients.",
    ),
    makeQuestion(
      41,
      "hard",
      "An enzyme works well near neutral pH but poorly in a very acidic compartment. Which explanations are plausible?",
      [
        ["pH can change charge states of amino acid side chains.", true],
        ["pH can disrupt folding or active-site chemistry.", true],
        [
          "pH directly changes chromosome number as the main way it affects enzymes.",
          false,
        ],
        [
          "pH mainly matters in laboratory glassware rather than living systems.",
          false,
        ],
      ],
      "This is still prerequisite-level reasoning: pH can change molecular charge and protein behavior. Students do not need advanced enzyme kinetics to see why a different chemical environment can alter activity.",
    ),
    makeQuestion(
      42,
      "hard",
      "A red blood cell placed in a solution with much lower solute concentration swells. Which concept best explains the swelling?",
      [
        [
          "Osmosis moves water into the cell because the outside solution is effectively more dilute.",
          true,
        ],
        ["Translation turns water into amino acids inside the cell.", false],
        ["CRISPR edits the cell membrane instantly.", false],
        ["Antibodies become ATP pumps.", false],
      ],
      "Osmosis links water movement to solute concentration across a membrane. The scenario prepares students for membrane transport and physiology without requiring advanced kidney or fluid-balance details.",
    ),
    makeQuestion(
      43,
      "hard",
      "A protein mutation replaces a charged amino acid near a binding site with a hydrophobic amino acid. Which outcomes are plausible?",
      [
        [
          "Binding could weaken if the charge helped recognize the ligand.",
          true,
        ],
        ["Folding or local structure could change.", true],
        ["The protein automatically becomes DNA.", false],
        [
          "The mutation should have no functional effect unless the entire gene is removed.",
          false,
        ],
      ],
      "This question checks whether students understand what an amino acid is and why side chains matter. A single substitution can matter if it changes charge, hydrophobicity, folding, or binding chemistry.",
    ),
    makeQuestion(
      44,
      "hard",
      "The same hormone circulates through the body but affects some tissues more than others. Which explanations are plausible?",
      [
        ["Some tissues may express more of the relevant receptor.", true],
        [
          "Downstream signaling machinery is identical across cell types regardless of gene-expression state.",
          false,
        ],
        [
          "Gene-expression state is separate from cellular response to a hormone.",
          false,
        ],
        [
          "The hormone must contain different DNA sequences for each tissue.",
          false,
        ],
      ],
      "Hormone specificity can depend on receptor expression and cellular context. The false options reverse that idea by denying cell-type differences or by confusing a circulating hormone with DNA carried for each tissue.",
    ),
    makeQuestion(
      45,
      "hard",
      "Which statements correctly explain antibiotic resistance using basic evolution?",
      [
        [
          "A bacterial population can contain variants with different drug susceptibility.",
          true,
        ],
        ["Treatment can select for variants that survive.", true],
        [
          "Surviving resistant bacteria can reproduce or pass resistance genes.",
          true,
        ],
        [
          "Resistance can spread through plasmids or other gene-transfer mechanisms.",
          true,
        ],
      ],
      "Resistance follows from variation, selection, inheritance, and sometimes gene transfer. This is high-quality prerequisite reasoning because it connects genetics, evolution, microbes, and medicine.",
    ),
    makeQuestion(
      46,
      "hard",
      "A biomarker is higher in people with a disease than in healthy controls. Which conclusion is best supported by that fact alone?",
      [
        [
          "The biomarker is associated with the disease in the compared groups.",
          true,
        ],
        ["The biomarker must be the root cause of the disease.", false],
        ["Lowering the biomarker must improve patient survival.", false],
        ["The biomarker must be perfectly sensitive and specific.", false],
      ],
      "Association is not the same as causation, prediction, or treatment value. This distinction prepares students to interpret diagnostics, biomarkers, surrogate endpoints, and AI predictions carefully.",
    ),
    makeQuestion(
      47,
      "hard",
      "A drug lowers a laboratory marker but patients do not feel better or live longer. Which interpretations are reasonable?",
      [
        [
          "The marker may not be a valid surrogate for the patient-relevant outcome.",
          true,
        ],
        [
          "The drug may affect biology without producing meaningful clinical benefit.",
          true,
        ],
        [
          "A marker change is sufficient evidence of clinical success by itself.",
          false,
        ],
        [
          "Laboratory markers are disconnected from diagnosis, monitoring, and treatment decisions.",
          false,
        ],
      ],
      "Markers can be useful, but only when their relationship to meaningful outcomes is validated. This prepares students for the revised Lecture 5 treatment of evidence as context rather than as the course core.",
    ),
    makeQuestion(
      48,
      "hard",
      "Which statements correctly describe an mRNA vaccine at a basic level?",
      [
        ["It can deliver instructions for cells to make an antigen.", true],
        ["The antigen can train immune recognition.", true],
        [
          "The mRNA is intended as a temporary instruction rather than a full permanent genome replacement.",
          true,
        ],
        [
          "It works because ribosomes translate lipids into chromosomes.",
          false,
        ],
      ],
      "mRNA vaccines depend on central dogma vocabulary and immune recognition. Students should understand that mRNA can instruct protein production without needing to permanently replace the genome.",
    ),
    makeQuestion(
      49,
      "hard",
      "Which statements correctly describe recombinant insulin production?",
      [
        [
          "A human insulin gene can be inserted into a production system.",
          true,
        ],
        [
          "Cells can use the inserted information to make insulin protein.",
          true,
        ],
        ["The insulin must be purified and quality-controlled.", true],
        [
          "The method depends on DNA, RNA, protein production, and biotechnology principles.",
          true,
        ],
      ],
      "This example connects many prerequisites: genes, plasmids or vectors, expression, proteins, purification, and medicine. It is a useful bridge from molecular biology to therapeutics.",
    ),
    makeQuestion(
      50,
      "hard",
      "Which statements correctly compare mRNA medicine, gene therapy, and CRISPR at a basic level?",
      [
        ["mRNA medicine usually provides temporary instructions.", true],
        [
          "Gene therapy can aim to add, replace, silence, or edit genetic information.",
          true,
        ],
        [
          "CRISPR-based tools are associated with targeted genome editing.",
          true,
        ],
        [
          "These technologies are ordinary bacterial cell-wall antibiotics.",
          false,
        ],
      ],
      "These technologies all relate to biological information but at different layers. The comparison prepares students for Lecture 3 and Lecture 5 without expecting advanced delivery or regulatory details.",
    ),
    makeQuestion(
      51,
      "hard",
      "Two patients receive the same oral drug dose, but one has much higher blood levels. Which explanations are plausible?",
      [
        ["Absorption could differ.", true],
        ["Metabolism or excretion could differ.", true],
        ["Drug interactions or organ function could differ.", true],
        [
          "The tablets must contain different genes because exposure is determined by the dose label.",
          false,
        ],
      ],
      "This is prerequisite PK reasoning. The same dose can produce different exposure because the body handles the drug differently across patients through absorption, metabolism, excretion, interactions, and organ function.",
    ),
    makeQuestion(
      52,
      "hard",
      "Which statement best describes type 2 diabetes as a systems problem?",
      [
        [
          "It can involve disrupted glucose regulation across hormones, tissues, metabolism, genetics, environment, and time.",
          true,
        ],
        [
          "It is best explained as a single-cell viral entry event rather than disrupted regulation.",
          false,
        ],
        ["It is identical to a single covalent bond forming in water.", false],
        [
          "It indicates that regulated physiology is absent from organisms with metabolic disease.",
          false,
        ],
      ],
      "The point is not to teach diabetes in depth, but to prepare systems reasoning. Disease can emerge from interactions across scales rather than from one isolated molecule.",
    ),
    makeQuestion(
      53,
      "hard",
      "Which statement best distinguishes autoimmune disease from ordinary immune defense against a pathogen?",
      [
        [
          "Autoimmune disease involves immune responses against self tissues, while pathogen defense targets non-self or danger-associated threats.",
          true,
        ],
        [
          "Autoimmune disease means the immune system is completely absent.",
          false,
        ],
        [
          "Autoimmune disease is the same thing as antibiotic resistance.",
          false,
        ],
        [
          "Autoimmune disease results from protein-folding failure in water rather than immune recognition of self.",
          false,
        ],
      ],
      "Autoimmunity is misdirected or poorly regulated immunity, not simply no immunity. This distinction prepares students for immunity, disease categories, biologic drugs, and side effects.",
    ),
    makeQuestion(
      54,
      "hard",
      "Which statements correctly describe why viruses depend on host cells?",
      [
        [
          "Viruses lack the full cellular machinery needed for independent reproduction.",
          true,
        ],
        [
          "Viruses can use host ribosomes or other host machinery during infection.",
          true,
        ],
        [
          "Viral replication can damage cells or disrupt normal function.",
          true,
        ],
        [
          "Viruses maintain mitochondria-like energy metabolism outside cells.",
          false,
        ],
      ],
      "Viruses are not free-living cells. Their dependence on host machinery explains both pathogenesis and why antiviral strategies often target entry, replication, viral enzymes, or host-virus interactions.",
    ),
    makeQuestion(
      55,
      "hard",
      "A drug blocks an ion pump that maintains a sodium gradient. Which consequences are plausible?",
      [
        ["The gradient may dissipate over time.", true],
        ["Processes depending on that gradient can be disrupted.", true],
        [
          "The drug may affect multiple tissues if the pump is widely used.",
          true,
        ],
        ["Blocking the pump directly edits genomic codons.", false],
      ],
      "This scenario links gradients, pumps, drugs, and side effects. It prepares students to reason from a molecular target to cell physiology and tissue-level consequences.",
    ),
    makeQuestion(
      56,
      "hard",
      "Which statements correctly explain why controls and randomization are useful in human studies?",
      [
        [
          "Patients can improve or worsen for reasons unrelated to treatment.",
          true,
        ],
        [
          "A control group helps separate treatment effects from background change.",
          true,
        ],
        [
          "Randomization helps balance known and unknown factors in expectation.",
          true,
        ],
        [
          "They reduce the risk of being misled by confounding or selection bias.",
          true,
        ],
      ],
      "Clinical evidence remains contextual in this course, but students need basic reasoning about trustworthy claims. Human outcomes are noisy, and controls plus randomization help distinguish causation from misleading observation.",
    ),
    makeQuestion(
      57,
      "hard",
      "An AI model predicts disease from images, but the images came from one hospital with a special scanner. Which concerns are reasonable?",
      [
        [
          "The model may have learned scanner or site artifacts instead of disease biology.",
          true,
        ],
        ["Performance may drop in other hospitals or populations.", true],
        [
          "High accuracy on the original data establishes a causal disease mechanism.",
          false,
        ],
        [
          "Validation becomes redundant when the model has many parameters.",
          false,
        ],
      ],
      "AI questions in this course should test evidence and biology, not hype. Data source, labels, distribution shift, and validation all matter before a prediction can be trusted in biomedical practice.",
    ),
    makeQuestion(
      58,
      "hard",
      "Which statements correctly connect cancer, apoptosis, and evolution?",
      [
        ["Failure to undergo apoptosis can help abnormal cells survive.", true],
        [
          "Tumor cell populations can evolve under immune or treatment pressure.",
          true,
        ],
        [
          "Cancer can involve disrupted regulation of division and survival.",
          true,
        ],
        [
          "Cancer is best described as normal tissue turnover rather than dysregulated growth and survival.",
          false,
        ],
      ],
      "This prepares students for cancer as dysregulated cellular evolution. It connects cell division, cell death, mutation, selection, immune pressure, and therapy resistance.",
    ),
    makeQuestion(
      59,
      "hard",
      "Which statements correctly describe why a treatment can work in a model system but fail in humans?",
      [
        [
          "The model may omit immune, vascular, endocrine, or tissue context.",
          true,
        ],
        ["Human disease may be more heterogeneous than the model.", true],
        [
          "Drug exposure or toxicity can differ between model and patient.",
          true,
        ],
        [
          "The measured model outcome may not match a patient-relevant endpoint.",
          true,
        ],
      ],
      "Model systems are essential, but translation is hard. The purpose of this question is to prepare learners for evidence reasoning without making clinical research the dominant subject.",
    ),
    makeQuestion(
      60,
      "hard",
      "Which statement best summarizes the prerequisite mindset for Lectures 1-5?",
      [
        [
          "Reason from molecules to cells, regulation, disease, treatment, measurement, and evidence while keeping uncertainty in view.",
          true,
        ],
        ["Memorize terms as isolated facts and ignore mechanisms.", false],
        [
          "Assume a plausible story is sufficient evidence for a biological claim.",
          false,
        ],
        [
          "Treat clinical-trial operations as the core of biology and chemistry.",
          false,
        ],
      ],
      "Lecture 0 prepares students for the course by building a bridge from basic vocabulary to mechanistic reasoning. The later lectures require understanding chemistry, cells, genes, physiology, biotechnology, and evidence as connected layers.",
    ),
  ];

export const BiologyChemistryLifeScienceL0Questions =
  BiologyChemistryForLifeScienceLecture0PreparationQuestions;
