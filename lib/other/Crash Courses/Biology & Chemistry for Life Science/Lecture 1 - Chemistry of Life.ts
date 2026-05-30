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
        "Carbon is useful because it cannot form covalent bonds with hydrogen, oxygen, or nitrogen.",
        false,
      ],
      [
        "Carbon is useful because it dissolves all biological molecules in water by itself.",
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
        "Water polarity matters only for ice, not for chemical reactions inside cells.",
        false,
      ],
      [
        "Water dissolves molecules only according to molecular mass, not charge or polarity.",
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
        "It means polar molecules cannot ever cross membranes by any route.",
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
        "An amino acid is always a lipid with a hydrophobic tail and hydrophilic head.",
        false,
      ],
      [
        "An amino acid is a sugar monomer used only for short-term energy.",
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
        "ATP makes enzymes unnecessary by forcing every reaction to happen instantly.",
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
        "A single amino acid substitution cannot matter unless the protein's entire gene is deleted.",
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
        "Phospholipids form bilayers because all parts of the molecule are equally charged.",
        false,
      ],
      [
        "Bilayers require a ribosome to place every lipid into position one at a time.",
        false,
      ],
      [
        "Bilayers form because covalent bonds link every lipid tail to every neighboring lipid tail.",
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
        "Enzymes are consumed as the required fuel for every reaction they catalyze.",
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
        "pH matters only in pure water and has no effect in cells or blood.",
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
      ["Carbohydrates are always genetic polymers read by ribosomes.", false],
      ["Lipids are always enzymes that lower activation energy.", false],
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
        "Cells become alive only when all reactions reach permanent chemical equilibrium.",
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
        "Sequence matters only for DNA, while proteins work independently of their amino acid order.",
        false,
      ],
      [
        "Any amino acid sequence will fold into the same structure if ATP is present.",
        false,
      ],
      [
        "Protein function is determined only by total molecular mass, not by side-chain chemistry.",
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
        "The change proves covalent bonds are irrelevant to membrane proteins.",
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
        "Tight binding in purified water proves the drug must work in every biological context.",
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
        "The protein must be DNA because only DNA can bind molecules specifically.",
        false,
      ],
      [
        "The binding pocket proves the reaction is thermodynamically favorable.",
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
      ["A gradient can only be built by changing DNA sequence.", false],
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
        "The mutation must improve function because charged residues are always better than hydrophobic residues.",
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
        "Unfolding proves covalent bonds in the backbone cannot exist in proteins.",
        false,
      ],
      ["Proteins only fold when no water is present.", false],
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
        "The phrase means sequence, environment, and regulation never matter.",
        false,
      ],
      [
        "The phrase applies only to proteins and never to membranes, nucleic acids, or small molecules.",
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
];

export const BiologyChemistryLifeScienceL1Questions =
  BiologyChemistryForLifeScienceLecture1Questions;
