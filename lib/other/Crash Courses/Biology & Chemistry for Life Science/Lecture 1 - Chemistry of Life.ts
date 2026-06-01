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
    throw new Error(`Lecture 1 question ${number} must have four options.`);
  }

  return {
    id: `bio-chem-life-l1-q${String(number).padStart(2, "0")}`,
    chapter: 1,
    difficulty,
    prompt,
    options: optionSeeds.map(([text, isCorrect]) => ({ text, isCorrect })),
    explanation,
  };
}

export const BiologyChemistryForLifeScienceLecture1Questions: Question[] = [
  makeQuestion(
    1,
    "easy",
    "Which statements give a mechanistic reason chemistry is foundational for biology?",
    [
      [
        "Proteins, membranes, DNA, and drugs all depend on molecular interactions.",
        true,
      ],
      ["Cells can be understood partly as organized reaction networks.", true],
      [
        "Biological specificity depends on shape, charge, polarity, and dynamics.",
        true,
      ],
      [
        "Living systems are exempt from chemical and physical constraints once they evolve regulation.",
        false,
      ],
    ],
    "Biology is implemented through molecules that bind, fold, react, and exchange energy. Regulation and evolution make the systems complex, but they do not remove the chemical constraints that determine what molecules can do.",
  ),
  makeQuestion(
    2,
    "easy",
    "A student says carbon is central to life because it is simply the most reactive element. Which correction is most accurate?",
    [
      [
        "Carbon is useful because it can form four stable covalent bonds, supporting chains, branches, rings, and isomers.",
        true,
      ],
      [
        "Carbon is useful because its bonding is limited to carbon-carbon networks, excluding heteroatoms from most biomolecules.",
        false,
      ],
      [
        "Carbon is useful because elemental carbon by itself serves as the main solvent for polar biomolecules.",
        false,
      ],
      [
        "Carbon is useful because it stores genetic information without needing other elements.",
        false,
      ],
    ],
    "Carbon's importance comes from bonding flexibility, not from being generically the most reactive element. Its four covalent bonds let organic molecules form diverse backbones that can include hydrogen, oxygen, nitrogen, phosphorus, sulfur, and other atoms.",
  ),
  makeQuestion(
    3,
    "easy",
    "Which pairings correctly connect a CHNOPS element to a high-yield biological role?",
    [
      [
        "Phosphorus: DNA/RNA backbones, ATP, phospholipids, and phosphorylation.",
        true,
      ],
      [
        "Sulfur: some amino acids, protein structure, and redox chemistry.",
        true,
      ],
      [
        "Oxygen: genetic code storage through base pairing by itself, without carbon or nitrogen.",
        false,
      ],
      ["Nitrogen: hydrophobic lipid tails as its main role in cells.", false],
    ],
    "Phosphorus and sulfur have specific roles that recur throughout biology, especially in information molecules, energy transfer, membranes, and protein chemistry. Oxygen and nitrogen are also essential, but the incorrect pairings assign them roles that belong to larger molecular systems or to different atoms.",
  ),
  makeQuestion(
    4,
    "easy",
    "Which statements correctly distinguish major biological interactions?",
    [
      ["Covalent bonds often create stable molecular skeletons.", true],
      ["Ionic interactions involve attraction between opposite charges.", true],
      [
        "Hydrogen bonds use partial charges and can strongly matter when many act together.",
        true,
      ],
      [
        "Hydrophobic effects help nonpolar groups cluster away from water.",
        true,
      ],
    ],
    "These interaction types explain different layers of biological structure. Stable covalent skeletons define molecules, while ionic interactions, hydrogen bonds, and hydrophobic effects often control folding, binding, membranes, and regulation.",
  ),
  makeQuestion(
    5,
    "easy",
    "Which explanation best connects water polarity to solubility?",
    [
      [
        "Water's partial charges can stabilize ions and polar groups, so many charged or polar substances dissolve well in water.",
        true,
      ],
      [
        "Water dissolves nonpolar oils best because hydrogen bonds form most strongly with hydrocarbon chains.",
        false,
      ],
      [
        "Water polarity mainly changes freezing behavior, with little relevance to solvation inside cells.",
        false,
      ],
      [
        "Water solubility is predicted mainly by molecular mass rather than charge or polarity.",
        false,
      ],
    ],
    "Water is polar, so it interacts favorably with charged and polar groups. Nonpolar hydrocarbon-rich molecules interact poorly with water, which helps explain why lipids form membranes instead of simply dissolving like salts or sugars.",
  ),
  makeQuestion(
    6,
    "easy",
    "Which statements correctly describe the hydrophobic effect in biology?",
    [
      ["It helps drive phospholipid bilayer formation in water.", true],
      [
        "It often helps bury nonpolar amino acid side chains inside folded proteins.",
        true,
      ],
      [
        "It means nonpolar molecules form strong covalent bonds with water.",
        false,
      ],
      [
        "Polar molecules cross membranes at the same rate as small nonpolar gases through the lipid core.",
        false,
      ],
    ],
    "The hydrophobic effect is a major organizing force because water interacts better with itself and polar groups than with nonpolar surfaces. It does not mean water covalently reacts with nonpolar molecules, and it does not eliminate membrane transport mechanisms for polar substances.",
  ),
  makeQuestion(
    7,
    "easy",
    "Which statements correctly connect monomers, polymers, hydrolysis, and dehydration reactions?",
    [
      ["A polymer is built from repeated smaller building blocks.", true],
      ["Hydrolysis uses water to break bonds in larger molecules.", true],
      [
        "Dehydration reactions join building blocks while releasing water.",
        true,
      ],
      [
        "Hydrolysis is the same as translating messenger RNA into protein.",
        false,
      ],
    ],
    "Macromolecule chemistry depends on building and breaking bonds between smaller units. Hydrolysis and dehydration reactions are chemical processes, while translation is an information-to-protein process that uses amino acids but is not itself the definition of hydrolysis.",
  ),
  makeQuestion(
    8,
    "easy",
    "Which statements correctly describe the four major classes of biological macromolecules?",
    [
      [
        "Carbohydrates can provide fuel, storage, structure, or cell-recognition roles.",
        true,
      ],
      [
        "Lipids support membranes, energy storage, and some signaling molecules.",
        true,
      ],
      [
        "Proteins perform many active cellular jobs such as catalysis, transport, signaling, and structure.",
        true,
      ],
      [
        "Nucleic acids store or transfer biological sequence information.",
        true,
      ],
    ],
    "The macromolecule classes are not just labels; they map to recurring biological functions. Each class has characteristic chemistry and building blocks that make it suited for particular roles in cells and organisms.",
  ),
  makeQuestion(
    9,
    "easy",
    "Which statement most accurately describes an amino acid?",
    [
      [
        "An amino acid is a protein building block with shared backbone features and a side chain that gives it distinct properties.",
        true,
      ],
      ["An amino acid is a nucleotide that stores one DNA base.", false],
      [
        "An amino acid has a hydrophobic tail and hydrophilic head arranged like a phospholipid.",
        false,
      ],
      [
        "An amino acid is a monosaccharide building block used mainly for glycogen storage.",
        false,
      ],
    ],
    "Amino acids are the monomers of proteins. Their shared backbone lets cells link them into chains, while side chains create differences in charge, polarity, hydrophobicity, size, and reactivity that affect folding and function.",
  ),
  makeQuestion(
    10,
    "easy",
    "Which statements correctly describe ATP in cellular metabolism?",
    [
      [
        "ATP can couple energy-releasing processes to energy-requiring cellular work.",
        true,
      ],
      [
        "ATP is regenerated continuously because cells keep consuming it.",
        true,
      ],
      [
        "ATP is the long-term hereditary molecule that stores chromosome sequence.",
        false,
      ],
      [
        "ATP replaces enzyme catalysis by directly forcing thermodynamically unfavorable reactions to completion.",
        false,
      ],
    ],
    "ATP is a short-term usable energy currency, not the molecule of heredity. Cells use ATP to drive transport, biosynthesis, movement, and regulation, but enzymes are still needed to make reaction rates compatible with life.",
  ),
  makeQuestion(
    11,
    "medium",
    "A folded protein loses function after a single amino acid substitution. Which explanations are plausible?",
    [
      [
        "The substitution could disrupt local charge or hydrogen bonding near an active site.",
        true,
      ],
      [
        "The substitution could alter hydrophobic packing and destabilize the folded structure.",
        true,
      ],
      [
        "The substitution could change a binding surface used to recognize another molecule.",
        true,
      ],
      [
        "A single amino acid substitution matters mainly when it removes the whole gene from the chromosome.",
        false,
      ],
    ],
    "Protein function can depend on local chemistry and global folding. A single substitution can matter if it changes an active site, a binding interface, folding stability, or regulation, even though many substitutions are neutral or mild.",
  ),
  makeQuestion(
    12,
    "medium",
    "Which interpretation best explains why phospholipids self-assemble into bilayers in water?",
    [
      [
        "Hydrophilic heads interact with water while hydrophobic tails are shielded from water in the bilayer interior.",
        true,
      ],
      [
        "Phospholipids form bilayers because their head groups and tails have the same polarity in water.",
        false,
      ],
      [
        "Bilayers require ribosomes to assemble lipid molecules into a membrane sequence.",
        false,
      ],
      [
        "Bilayers form because covalent bonds join adjacent lipid tails into a continuous sheet.",
        false,
      ],
    ],
    "Phospholipids are amphipathic, meaning they contain both water-interacting and water-avoiding regions. Bilayers can self-assemble from these chemical properties, though cells also regulate membrane composition and organization.",
  ),
  makeQuestion(
    13,
    "medium",
    "Which statements correctly distinguish enzyme catalysis from changes in reaction thermodynamics?",
    [
      ["Enzymes lower activation energy and speed reactions.", true],
      [
        "Enzymes do not make an energetically unfavorable overall reaction favorable by catalysis alone.",
        true,
      ],
      [
        "Enzymes serve as stoichiometric fuel that is used up during catalysis.",
        false,
      ],
      ["Enzymes work by changing the genetic code of their substrates.", false],
    ],
    "Enzymes alter reaction rates by stabilizing transition states or productive molecular arrangements. They do not by themselves change the overall free-energy difference of a reaction, and they are not consumed like fuel in each catalytic cycle.",
  ),
  makeQuestion(
    14,
    "medium",
    "Which statements correctly connect water's properties to biological function?",
    [
      ["Hydrogen bonding contributes to cohesion and surface tension.", true],
      ["Water's high heat capacity helps buffer temperature changes.", true],
      [
        "Water's polarity helps ions and polar molecules participate in reactions.",
        true,
      ],
      ["Ice floating can protect liquid-water environments underneath.", true],
    ],
    "Water's unusual properties are consequences of polarity and hydrogen bonding. Not every property is equally important in every medical context, but together they help explain why water is such a powerful medium for life.",
  ),
  makeQuestion(
    15,
    "medium",
    "Which statements correctly describe pH as a biological variable?",
    [
      ["pH reflects hydrogen ion concentration.", true],
      ["Changing pH can alter charge states of amino acid side chains.", true],
      ["Enzyme function can depend strongly on pH.", true],
      [
        "pH mainly describes purified lab water and has little relevance to cells or blood.",
        false,
      ],
    ],
    "Many biological molecules contain groups that can gain or lose protons. Those charge changes can alter binding, folding, catalysis, and physiology, which is why blood pH and compartment pH are tightly regulated.",
  ),
  makeQuestion(
    16,
    "medium",
    "A molecule has the same chemical formula as another molecule but a different atom arrangement. Which term best applies?",
    [
      ["Isomer.", true],
      ["Polymer.", false],
      ["Ribosome.", false],
      ["Gradient.", false],
    ],
    "Isomers have the same formula but different arrangements, which can lead to different shapes and biological effects. This matters because molecular function depends on structure, not only on which atoms are present.",
  ),
  makeQuestion(
    17,
    "medium",
    "Which statements correctly compare carbohydrates and lipids?",
    [
      [
        "Carbohydrates are often more polar and can serve as readily mobilized fuels or structural polymers.",
        true,
      ],
      [
        "Lipids are often hydrophobic or amphipathic and are important in membranes and energy storage.",
        true,
      ],
      ["Carbohydrates are genetic polymers translated by ribosomes.", false],
      [
        "Lipids are catalytic polymers that lower activation energy through active sites.",
        false,
      ],
    ],
    "Carbohydrates and lipids differ in chemistry and typical biological roles. Some carbohydrates are structural or recognition molecules, while lipids often form membranes, store energy, or act as signals because of their hydrophobic chemistry.",
  ),
  makeQuestion(
    18,
    "medium",
    "Which statements correctly describe biological specificity?",
    [
      [
        "Specificity can arise from complementary shape between molecules.",
        true,
      ],
      ["Specificity can depend on charge and polarity patterns.", true],
      ["Specificity can involve many weak interactions acting together.", true],
      ["Specificity can change when a protein changes conformation.", true],
    ],
    "Specificity is rarely one simple lock-and-key feature. Shape, charge, polarity, hydrophobic surfaces, flexibility, and conformational change can all contribute to whether two biological molecules bind or react productively.",
  ),
  makeQuestion(
    19,
    "medium",
    "Which statements correctly describe why life is non-equilibrium chemistry?",
    [
      [
        "Cells constantly consume energy to build, repair, transport, and regulate.",
        true,
      ],
      [
        "Cells maintain gradients and ordered structures that would dissipate without energy input.",
        true,
      ],
      [
        "Cells export entropy to their environment while maintaining local order.",
        true,
      ],
      [
        "Cells become metabolically functional by letting reactions settle into permanent chemical equilibrium.",
        false,
      ],
    ],
    "Living systems maintain organization through continuous flux of matter and energy. Equilibrium would erase many gradients and dynamic processes that cells need for transport, signaling, metabolism, and survival.",
  ),
  makeQuestion(
    20,
    "medium",
    "Which statement best describes the relationship between a protein sequence and protein function?",
    [
      [
        "Sequence influences folding and chemical surfaces, which influence dynamics, binding, catalysis, and regulation.",
        true,
      ],
      [
        "Sequence is important for DNA, while protein activity is mostly independent of amino acid order.",
        false,
      ],
      [
        "Any amino acid sequence will fold into the same structure if ATP is present.",
        false,
      ],
      [
        "Protein function is determined mainly by total molecular mass rather than side-chain chemistry.",
        false,
      ],
    ],
    "A protein's amino acid sequence constrains the structures and conformations it can adopt. Function emerges from the resulting surfaces, active sites, binding interfaces, and regulation in a cellular context.",
  ),
  makeQuestion(
    21,
    "hard",
    "A membrane protein stops functioning when charged residues lining its pore are replaced by hydrophobic residues. Which explanations are most plausible?",
    [
      [
        "The pore may no longer stabilize polar or charged molecules passing through it.",
        true,
      ],
      [
        "The change could alter folding or localization in the lipid bilayer.",
        true,
      ],
      [
        "The protein must now become a carbohydrate because it contains hydrophobic residues.",
        false,
      ],
      [
        "The change suggests membrane-protein function is unrelated to covalent structure.",
        false,
      ],
    ],
    "Channels and transporters depend on both membrane compatibility and a suitable internal path for the transported substance. Replacing charged residues with hydrophobic ones can disrupt selectivity, conductance, folding, or placement without changing the molecule into a different macromolecule class.",
  ),
  makeQuestion(
    22,
    "hard",
    "A drug binds tightly to a purified enzyme in water but fails in cells. Which explanations are scientifically reasonable?",
    [
      [
        "The drug may not cross the cell membrane or reach the enzyme at sufficient concentration.",
        true,
      ],
      [
        "The enzyme's cellular partners, pH, modifications, or compartment could change binding or activity.",
        true,
      ],
      [
        "The purified assay may omit competing molecules or regulatory states present in cells.",
        true,
      ],
      [
        "Tight binding in purified water is sufficient evidence that the drug will work in cells.",
        false,
      ],
    ],
    "Biological function depends on context as well as molecular affinity. Cellular access, compartmentalization, competing interactions, regulation, pH, and protein state can all explain why a purified biochemical result does not automatically transfer to cells.",
  ),
  makeQuestion(
    23,
    "hard",
    "Which statements correctly connect food macromolecules to the molecules a cell builds for itself?",
    [
      ["Digestion can hydrolyze polymers into smaller building blocks.", true],
      [
        "Cells can reuse absorbed building blocks to synthesize their own macromolecules.",
        true,
      ],
      [
        "Dehydration reactions can help assemble new polymers from monomers.",
        true,
      ],
      [
        "The same broad building-block logic applies across many organisms because of shared evolutionary chemistry.",
        true,
      ],
    ],
    "Food is not simply burned as intact pasta, meat, or fat. Digestion and metabolism break molecules into usable pieces, and cells rebuild molecules according to their own needs using shared biochemical strategies.",
  ),
  makeQuestion(
    24,
    "hard",
    "A protein has a strong binding pocket for a molecule but no catalytic residues near the bound chemical group. Which interpretation is best?",
    [
      [
        "Binding alone does not prove the protein is an enzyme for that molecule.",
        true,
      ],
      [
        "Any molecule that binds a protein must be chemically converted by that protein.",
        false,
      ],
      [
        "The binding protein is better classified as DNA because specificity is a property of nucleic acids.",
        false,
      ],
      [
        "The binding pocket by itself shows the coupled reaction has a favorable overall free-energy change.",
        false,
      ],
    ],
    "Specific binding and catalysis are related but distinct. A protein can bind a ligand for signaling, transport, inhibition, or regulation without catalyzing a chemical transformation of that ligand.",
  ),
  makeQuestion(
    25,
    "hard",
    "Which statements correctly describe gradients as biological energy stores?",
    [
      [
        "A concentration gradient can drive passive movement down the gradient.",
        true,
      ],
      [
        "An electrochemical gradient can combine concentration difference and charge separation.",
        true,
      ],
      [
        "A gradient means molecules are evenly distributed and no work can be extracted.",
        false,
      ],
      [
        "A gradient is built by rewriting DNA sequence before each transport event.",
        false,
      ],
    ],
    "Gradients represent separations that can drive movement or work. Cells build and maintain gradients using pumps, metabolism, and membranes, not by changing DNA sequence every time a transport process is needed.",
  ),
  makeQuestion(
    26,
    "hard",
    "A mutation changes a hydrophobic amino acid in a protein core to a charged amino acid. Which consequences are plausible?",
    [
      [
        "The folded structure could become less stable because a buried charge is unfavorable.",
        true,
      ],
      [
        "The protein might misfold or be degraded by quality-control systems.",
        true,
      ],
      [
        "The effect could be small if the residue is not actually buried or the structure compensates.",
        true,
      ],
      [
        "The mutation should improve function because charged residues generally substitute for hydrophobic residues without changing folding or binding.",
        false,
      ],
    ],
    "The impact of a substitution depends on structural context. A charged residue in a hydrophobic core can be disruptive, but proteins can sometimes compensate or the residue may not be buried in the relevant state.",
  ),
  makeQuestion(
    27,
    "hard",
    "Which statements correctly describe why weak interactions are central to life?",
    [
      ["Weak interactions allow reversible binding and regulation.", true],
      ["Many weak interactions together can create high specificity.", true],
      [
        "Weak interactions support dynamic protein conformations and molecular recognition.",
        true,
      ],
      [
        "Weak interactions can be tuned by pH, concentration, charge state, and environment.",
        true,
      ],
    ],
    "Life depends on interactions that are strong enough to organize molecules but flexible enough to reverse and regulate. Weak interactions are especially useful because cells can tune them through local chemistry, concentration, modification, and compartment conditions.",
  ),
  makeQuestion(
    28,
    "hard",
    "A protein unfolds when temperature rises above a narrow range. Which explanation is most accurate?",
    [
      [
        "Higher temperature can disrupt the balance of weak interactions that stabilize the folded state.",
        true,
      ],
      [
        "Temperature changes rewrite the amino acid sequence into a different genetic code.",
        false,
      ],
      [
        "Unfolding indicates that the covalent peptide backbone has been converted into separate amino-acid monomers.",
        false,
      ],
      [
        "Protein folding requires removing water from the biological environment.",
        false,
      ],
    ],
    "Protein folding depends on a balance of many weak interactions and the surrounding solvent. Heat can disrupt that balance without necessarily breaking every covalent bond in the backbone or changing the underlying sequence.",
  ),
  makeQuestion(
    29,
    "hard",
    "Which statements correctly identify limits of the phrase structure determines function?",
    [
      [
        "Protein function can depend on dynamics and conformational changes, not only one static shape.",
        true,
      ],
      [
        "Cellular context, binding partners, and modifications can change what a structure does.",
        true,
      ],
      [
        "The phrase treats sequence, environment, and regulation as minor details outside molecular function.",
        false,
      ],
      [
        "The phrase applies to proteins but excludes membranes, nucleic acids, and small molecules from structure-function reasoning.",
        false,
      ],
    ],
    "Structure-function reasoning is central, but structure should not be treated as one frozen image. Dynamics, modifications, partners, concentration, compartment, and environment all affect what a molecule can do.",
  ),
  makeQuestion(
    30,
    "hard",
    "Which statements correctly synthesize Lecture 1 into a model useful for later medicine and biotechnology?",
    [
      [
        "Drug effects can often be traced to molecular binding and downstream system response.",
        true,
      ],
      [
        "Disease mechanisms can involve altered protein folding, enzyme activity, transport, or signaling.",
        true,
      ],
      [
        "Biotechnology often exploits the fact that biological molecules are modular and chemically constrained.",
        true,
      ],
      [
        "AI protein tools are useful partly because sequence, structure, and function are connected.",
        true,
      ],
    ],
    "The chemistry of life is not isolated background material. It provides the mechanistic basis for pharmacology, disease, biotechnology, diagnostics, and computational biology, where molecular structure and interaction determine larger-scale outcomes.",
  ),
  makeQuestion(
    31,
    "easy",
    "Which statement best describes what protons do in atoms?",
    [
      ["They help define which element the atom is.", true],
      [
        "They mainly control isotope mass without affecting element identity.",
        false,
      ],
      [
        "They form the electron cloud that determines most bonding behavior.",
        false,
      ],
      ["They are temporary particles created during enzyme reactions.", false],
    ],
    "Protons are central to element identity because the number of protons defines the atomic number. Neutrons affect isotopes and mass, while electrons are most directly involved in bonding, charge, and reactivity.",
  ),
  makeQuestion(
    32,
    "easy",
    "Which statement best describes the role of electrons in biological chemistry?",
    [
      ["They strongly influence bonding, charge, and reactivity.", true],
      ["They define the element more directly than the nucleus does.", false],
      ["They add most of the atom's mass in biological molecules.", false],
      [
        "They are found between protons and neutrons inside the nucleus.",
        false,
      ],
    ],
    "Electrons occupy regions outside the nucleus and determine how atoms interact chemically. Their arrangement affects covalent bonding, ionic charge, polarity, and many reactions that matter in cells.",
  ),
  makeQuestion(
    33,
    "easy",
    "Which statement best explains why carbon supports molecular diversity in life?",
    [
      [
        "Carbon can form four covalent bonds, enabling chains, branches, rings, and isomers.",
        true,
      ],
      [
        "Carbon mainly acts as the solvent that surrounds proteins and nucleic acids.",
        false,
      ],
      [
        "Carbon prevents organic molecules from interacting with nitrogen or oxygen.",
        false,
      ],
      [
        "Carbon forms biological molecules by avoiding stable covalent bonds.",
        false,
      ],
    ],
    "Carbon's four covalent bonds allow stable but flexible organic frameworks. This makes it possible to build many shapes and arrangements, including chains, branches, rings, and isomers.",
  ),
  makeQuestion(
    34,
    "easy",
    "Which statement best defines a monomer?",
    [
      [
        "A small building block that can be joined into a larger molecule.",
        true,
      ],
      ["A membrane compartment that stores ATP for later reactions.", false],
      ["A folded protein region that binds a specific ligand.", false],
      ["A charged atom that forms when salt dissolves in water.", false],
    ],
    "A monomer is a smaller molecular building block used to construct larger molecules. For example, amino acids are monomers for proteins, monosaccharides are monomers for many carbohydrates, and nucleotides are monomers for DNA and RNA.",
  ),
  makeQuestion(
    35,
    "easy",
    "Which statement best describes ATP in cells?",
    [
      [
        "ATP is an energy currency that helps couple energy-releasing processes to cellular work.",
        true,
      ],
      [
        "ATP is the genetic polymer that stores inherited information in chromosomes.",
        false,
      ],
      [
        "ATP is a membrane lipid that forms the hydrophobic core of bilayers.",
        false,
      ],
      ["ATP is a structural protein that transports oxygen in blood.", false],
    ],
    "Adenosine triphosphate (ATP) is used to connect energy-releasing reactions to energy-requiring work. It is not DNA, a membrane lipid, or a transport protein, although ATP-dependent processes support many of those systems.",
  ),
  makeQuestion(
    36,
    "easy",
    "Which statements correctly connect CHNOPS elements to biological roles?",
    [
      ["Nitrogen appears in amino acids, proteins, and nucleic acids.", true],
      [
        "Phosphorus appears in DNA/RNA backbones, ATP, phospholipids, and phosphorylation.",
        true,
      ],
      [
        "Sulfur is mainly the element that gives DNA bases their sugar-phosphate backbone.",
        false,
      ],
      [
        "Oxygen is mainly avoided in metabolism and polar biological groups.",
        false,
      ],
    ],
    "CHNOPS summarizes carbon, hydrogen, nitrogen, oxygen, phosphorus, and sulfur because each has common biological roles. Nitrogen is central in amino acids and nucleic acids, while phosphorus is central in nucleic acid backbones, ATP, phospholipids, and phosphorylation.",
  ),
  makeQuestion(
    37,
    "easy",
    "Which statements correctly distinguish covalent bonds from weaker biological interactions?",
    [
      ["Covalent bonds involve atoms sharing electrons.", true],
      ["Covalent bonds often form stable molecular backbones.", true],
      [
        "Hydrogen bonds are the main covalent skeleton of DNA, proteins, carbohydrates, and lipids.",
        false,
      ],
      [
        "The hydrophobic effect is a covalent bond between nonpolar groups and water.",
        false,
      ],
    ],
    "Covalent bonds involve shared electrons and often create stable molecular skeletons. Hydrogen bonds, ionic interactions, and hydrophobic effects are usually weaker interactions that help determine shape, binding, folding, and organization.",
  ),
  makeQuestion(
    38,
    "easy",
    "Which statements correctly describe why water is polar?",
    [
      ["Oxygen pulls shared electrons more strongly than hydrogen.", true],
      ["Water has partial charges that support hydrogen bonding.", true],
      [
        "Water is polar because hydrogen and oxygen share electrons equally.",
        false,
      ],
      [
        "Water is polar because it lacks any charged or partially charged regions.",
        false,
      ],
    ],
    "Water is polar because oxygen attracts shared electrons more strongly than hydrogen, creating partial charges. Those partial charges allow water molecules to hydrogen bond and shape solubility, protein folding, and cellular organization.",
  ),
  makeQuestion(
    39,
    "easy",
    "Which statements correctly describe hydrolysis and dehydration reactions?",
    [
      ["Hydrolysis uses water to break bonds.", true],
      [
        "Dehydration reactions remove water while joining building blocks.",
        true,
      ],
      ["Hydrolysis joins monomers by removing water between them.", false],
      [
        "Dehydration reactions digest food polymers by adding water to bonds.",
        false,
      ],
    ],
    "Hydrolysis and dehydration reactions are opposite patterns in polymer chemistry. Hydrolysis helps break polymers into smaller pieces, while dehydration reactions help cells join building blocks into larger molecules.",
  ),
  makeQuestion(
    40,
    "easy",
    "Which statements correctly describe catabolism and anabolism?",
    [
      ["Catabolism breaks molecules down and can release usable energy.", true],
      ["Anabolism builds molecules and consumes energy.", true],
      [
        "Catabolism builds proteins, nucleic acids, and membranes from small precursors.",
        false,
      ],
      ["Anabolism is the direct digestion of polymers into monomers.", false],
    ],
    "Catabolism and anabolism describe two broad directions of metabolism. Catabolic pathways break molecules down, while anabolic pathways build larger molecules and require energy input.",
  ),
  makeQuestion(
    41,
    "easy",
    "Which statements correctly describe the four major biological macromolecule classes?",
    [
      [
        "Carbohydrates include sugars and can support energy storage, structure, and recognition.",
        true,
      ],
      [
        "Proteins are amino acid polymers involved in catalysis, transport, signaling, movement, structure, and immunity.",
        true,
      ],
      [
        "Nucleic acids are nucleotide polymers that store and transfer genetic information.",
        true,
      ],
      [
        "Lipids are nucleotide chains that encode inherited sequence information.",
        false,
      ],
    ],
    "Carbohydrates, lipids, proteins, and nucleic acids have different building blocks and roles. Lipids are mostly hydrophobic molecules involved in membranes, energy storage, hormones, and signaling, not nucleotide polymers.",
  ),
  makeQuestion(
    42,
    "easy",
    "Which statements correctly describe lipids and phospholipids?",
    [
      ["Many lipids are mostly hydrophobic.", true],
      ["Phospholipids have hydrophilic heads and hydrophobic tails.", true],
      ["Phospholipids can self-assemble into bilayers.", true],
      [
        "Lipids are usually made from amino acid monomers linked into folded chains.",
        false,
      ],
    ],
    "Lipids include fatty acids, phospholipids, steroid hormones, and energy-storage molecules. Phospholipids are amphipathic, so their head and tail chemistry helps them self-assemble into bilayers in water.",
  ),
  makeQuestion(
    43,
    "easy",
    "Which statements correctly describe protein side chains?",
    [
      ["Side chains can differ in charge.", true],
      ["Side chains can differ in polarity and hydrophobicity.", true],
      ["Side chains help drive folding and molecular interactions.", true],
      [
        "Side chains are the repeating sugar-phosphate units of proteins.",
        false,
      ],
    ],
    "Proteins are built from amino acids, and amino acid side chains vary in charge, polarity, hydrophobicity, size, and reactivity. Those differences shape folding, binding, catalysis, and specificity.",
  ),
  makeQuestion(
    44,
    "easy",
    "Which statements correctly describe enzymes?",
    [
      ["Enzymes are biological catalysts.", true],
      ["Enzymes speed reactions by lowering activation energy.", true],
      [
        "Enzymes can be specific because of shape, charge, polarity, and local chemistry.",
        true,
      ],
      [
        "Enzymes speed reactions by changing the overall energy balance into a different reaction.",
        false,
      ],
    ],
    "Enzymes help reactions proceed fast enough for cells by lowering activation energy. They do not change the overall energy balance of the reaction, and their specificity depends on molecular complementarity and dynamics.",
  ),
  makeQuestion(
    45,
    "easy",
    "Which statements correctly describe pH in biological systems?",
    [
      ["pH reflects hydrogen ion concentration.", true],
      [
        "Changing protonation can change charge, shape, and binding behavior.",
        true,
      ],
      [
        "pH matters for enzyme function, protein folding, blood chemistry, and cellular compartments.",
        true,
      ],
      [
        "pH is a direct count of how many carbon atoms are present in a molecule.",
        false,
      ],
    ],
    "pH measures hydrogen ion concentration, and biological groups can gain or lose protons. That can alter charge, folding, binding, and enzyme activity, which is why pH matters across cells and tissues.",
  ),
  makeQuestion(
    46,
    "medium",
    "Which statements correctly describe how weak interactions support biological specificity?",
    [
      [
        "Hydrogen bonds can help shape water behavior, DNA base pairing, and protein structure.",
        true,
      ],
      [
        "Ionic interactions can help charged regions of molecules attract each other.",
        true,
      ],
      [
        "Hydrophobic effects can help organize membranes and protein cores.",
        true,
      ],
      [
        "Many weak interactions together can produce strong, specific binding.",
        true,
      ],
    ],
    "Weak interactions are individually reversible, but many of them acting together can create strong specificity. This balance lets biological systems bind, fold, organize, and regulate without turning every interaction into a permanent covalent connection.",
  ),
  makeQuestion(
    47,
    "medium",
    "Which statements correctly describe why nonpolar substances behave differently from salts or sugars in water?",
    [
      [
        "Water can surround and stabilize many polar or charged substances.",
        true,
      ],
      [
        "Nonpolar substances interact poorly with water compared with water-water interactions.",
        true,
      ],
      ["Nonpolar groups often cluster away from water.", true],
      [
        "This behavior helps explain lipid bilayers and hydrophobic protein interiors.",
        true,
      ],
    ],
    "Water's polarity makes it a good environment for many charged and polar substances. Nonpolar groups tend to cluster because water-water interactions are more favorable than water-nonpolar interactions, creating the hydrophobic effect.",
  ),
  makeQuestion(
    48,
    "medium",
    "Which statements correctly connect monomers, polymers, and digestion?",
    [
      ["Cells can build polymers from smaller building blocks.", true],
      ["Food polymers can be broken into smaller pieces by hydrolysis.", true],
      [
        "Macromolecules are large biological molecules built from or organized around smaller chemical units.",
        true,
      ],
      [
        "Dehydration reactions and hydrolysis help connect chemistry to food, cells, and biosynthesis.",
        true,
      ],
    ],
    "Polymer chemistry explains both construction and breakdown. Cells build larger molecules from smaller units, while digestion often uses hydrolysis to break food polymers into usable pieces.",
  ),
  makeQuestion(
    49,
    "medium",
    "Which statements correctly describe proteins as sequence-to-function molecules?",
    [
      ["Amino acid sequence influences folding.", true],
      ["Folded structure influences binding surfaces and active sites.", true],
      [
        "Protein dynamics and conformational change can matter for function.",
        true,
      ],
      ["Cellular context and regulation can affect what a protein does.", true],
    ],
    "Protein function is often summarized as sequence to structure to dynamics to function. That chain is useful, but the environment, binding partners, modifications, and cellular context also influence what the protein does.",
  ),
  makeQuestion(
    50,
    "medium",
    "Which statements correctly describe why cells are not at equilibrium?",
    [
      [
        "Cells require constant energy flow to build, repair, transport, signal, move, and replicate.",
        true,
      ],
      ["Cells maintain organization by using energy.", true],
      ["Gradients can store usable energy across membranes.", true],
      ["Metabolism helps cells couple energy supply to cellular work.", true],
    ],
    "Living cells maintain organized states that require ongoing energy flow. ATP, metabolism, and gradients help cells do work and keep internal organization despite the tendency of systems to move toward equilibrium.",
  ),
];

export const BiologyChemistryLifeScienceL1Questions =
  BiologyChemistryForLifeScienceLecture1Questions;
