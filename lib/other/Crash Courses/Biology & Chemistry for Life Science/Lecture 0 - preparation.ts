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
      "Which statements describe useful starting assumptions for studying living systems?",
      [
        ["Cells are made from atoms and molecules.", true],
        [
          "Living systems use energy to maintain organization and do work.",
          true,
        ],
        [
          "Information stored in molecules can affect cell structure and behavior.",
          true,
        ],
        [
          "Visible traits and diseases often have explanations at molecular or cellular levels.",
          true,
        ],
      ],
      "Life science connects molecules, cells, tissues, and whole organisms. These starting assumptions help students link vocabulary such as gene, protein, energy, and disease to mechanisms rather than memorizing isolated words.",
    ),
    makeQuestion(
      2,
      "easy",
      "Which statements correctly distinguish atoms, elements, and molecules?",
      [
        ["An element is a substance defined by one kind of atom.", true],
        [
          "A molecule contains two or more atoms held together by chemical bonds.",
          true,
        ],
        [
          "Carbon dioxide is a molecule because it contains bonded atoms.",
          true,
        ],
        [
          "An element is defined by the number of different molecules in a sample.",
          false,
        ],
      ],
      "Elements are defined by atom identity, while molecules are built from bonded atoms. Confusing elements with mixtures or samples makes later chemistry terms harder to use correctly.",
    ),
    makeQuestion(
      3,
      "easy",
      "Which statements correctly describe subatomic particles and isotopes?",
      [
        ["Protons help determine which element an atom is.", true],
        [
          "Isotopes of the same element have different numbers of neutrons.",
          true,
        ],
        [
          "Electrons account for almost all of an atom's mass in basic chemistry calculations.",
          false,
        ],
        [
          "Changing the number of protons leaves the element identity unchanged.",
          false,
        ],
      ],
      "The number of protons identifies the element, while neutrons can vary among isotopes. Electrons strongly affect bonding, but they do not contribute most of the atom's mass.",
    ),
    makeQuestion(
      4,
      "easy",
      "Which statement best describes a covalent bond?",
      [
        ["Atoms share pairs of electrons.", true],
        [
          "Oppositely charged ions attract after one atom transfers electrons to another.",
          false,
        ],
        [
          "A weak attraction forms between a partially positive hydrogen and a partially negative atom nearby.",
          false,
        ],
        [
          "Nonpolar molecules cluster away from water without sharing electrons with each other.",
          false,
        ],
      ],
      "A covalent bond forms when atoms share electrons. The other statements describe ionic bonding, hydrogen bonding, or hydrophobic interactions, which are different kinds of chemical interactions.",
    ),
    makeQuestion(
      5,
      "easy",
      "Which statements correctly describe ionic and polar interactions?",
      [
        [
          "An ion has a net electrical charge because it has gained or lost electrons.",
          true,
        ],
        [
          "Ionic bonds or attractions can form between oppositely charged ions.",
          true,
        ],
        ["A polar bond has an uneven distribution of electron density.", true],
        [
          "Charged and polar groups often affect how a molecule behaves in water.",
          true,
        ],
      ],
      "Ions and polar bonds are central to solubility, protein structure, membranes, and signaling. Charge and uneven electron sharing help explain why some molecules interact strongly with water or with charged parts of proteins.",
    ),
    makeQuestion(
      6,
      "easy",
      "Which statements correctly describe hydrogen bonds and water?",
      [
        ["Water molecules can form hydrogen bonds with each other.", true],
        [
          "Hydrogen bonds are important for the behavior of DNA and proteins.",
          true,
        ],
        [
          "Many weak hydrogen bonds together can strongly influence molecular shape.",
          true,
        ],
        [
          "Hydrogen bonds require complete transfer of electrons from hydrogen to oxygen.",
          false,
        ],
      ],
      "Hydrogen bonds are weaker than covalent bonds but very important when many occur together. They are based on partial charges, not complete electron transfer.",
    ),
    makeQuestion(
      7,
      "easy",
      "Which statements correctly connect polarity and solubility?",
      [
        ["Polar or charged substances often dissolve well in water.", true],
        [
          "Nonpolar substances often dissolve better in nonpolar environments than in water.",
          true,
        ],
        [
          "A molecule's solubility is determined only by its molecular mass.",
          false,
        ],
        [
          "Water dissolves all lipids easily because water and lipids have the same polarity.",
          false,
        ],
      ],
      "Water is polar, so polar and charged substances often interact with it well. Lipids are often partly or mostly nonpolar, which helps explain why membranes form rather than simply dissolving into the watery cell environment.",
    ),
    makeQuestion(
      8,
      "easy",
      "Which statement best describes pH?",
      [
        [
          "pH is a measure related to hydrogen ion concentration in a solution.",
          true,
        ],
        [
          "pH is higher in more acidic solutions under the same conditions.",
          false,
        ],
        [
          "pH directly tells you the concentration of every dissolved ion.",
          false,
        ],
        [
          "pH is unrelated to enzyme activity because enzymes work equally well under all acid-base conditions.",
          false,
        ],
      ],
      "pH is a chemistry measure connected to acidity and hydrogen ion concentration. It matters in biology because enzyme activity, protein shape, and cellular conditions can depend strongly on pH.",
    ),
    makeQuestion(
      9,
      "easy",
      "Which statements correctly describe acids, bases, buffers, and biological stability?",
      [
        ["Acids increase hydrogen ion availability in solution.", true],
        [
          "Bases reduce hydrogen ion availability or accept hydrogen ions.",
          true,
        ],
        [
          "Buffers help resist large pH changes when acid or base is added.",
          true,
        ],
        [
          "Stable pH can matter because many proteins work best over a limited pH range.",
          true,
        ],
      ],
      "Acid-base chemistry is not just laboratory vocabulary; it affects living systems. Buffers help organisms maintain conditions in which proteins and reactions can function.",
    ),
    makeQuestion(
      10,
      "easy",
      "Which statements correctly describe carbon and organic molecules?",
      [
        ["Carbon can form chains, rings, and branched structures.", true],
        ["Carbon commonly forms covalent bonds in biological molecules.", true],
        [
          "Carbon skeletons can support many different functional groups.",
          true,
        ],
        [
          "Carbon is biologically important because it cannot bond to other carbon atoms.",
          false,
        ],
      ],
      "Carbon's bonding flexibility supports the diversity of biological molecules. If carbon could not bond to itself, many large organic structures would be impossible.",
    ),
    makeQuestion(
      11,
      "easy",
      "Which statements correctly describe functional groups and macromolecules?",
      [
        [
          "Functional groups can change a molecule's charge, polarity, or reactivity.",
          true,
        ],
        [
          "Macromolecules such as proteins and nucleic acids are built from smaller subunits.",
          true,
        ],
        ["A functional group is the full set of chromosomes in a cell.", false],
        [
          "Macromolecules are defined by being unable to interact with water or ions.",
          false,
        ],
      ],
      "Functional groups help explain why molecules with similar carbon skeletons can behave differently. Macromolecules are large biological molecules, often assembled from repeated smaller building blocks.",
    ),
    makeQuestion(
      12,
      "easy",
      "Which statement best describes carbohydrates?",
      [
        [
          "Carbohydrates include sugars and polymers that can store energy or provide structure.",
          true,
        ],
        ["Carbohydrates are genetic polymers built from nucleotides.", false],
        [
          "Carbohydrates are made only from amino acids joined by peptide bonds.",
          false,
        ],
        [
          "Carbohydrates are the same class of molecule as steroid hormones and fats.",
          false,
        ],
      ],
      "Carbohydrates include simple sugars, starch, glycogen, and cellulose. Nucleic acids, proteins, and many lipids use different building blocks and have different roles.",
    ),
    makeQuestion(
      13,
      "easy",
      "Which statements correctly describe lipids?",
      [
        [
          "Many lipids contain large nonpolar regions that do not mix well with water.",
          true,
        ],
        [
          "Phospholipids help form the basic structure of cell membranes.",
          true,
        ],
        ["Some lipids store energy or act as signaling molecules.", true],
        ["Steroid hormones are lipid-related molecules.", true],
      ],
      "Lipids are a diverse group rather than one single structure. Their nonpolar character helps explain membranes, energy storage, and some forms of signaling.",
    ),
    makeQuestion(
      14,
      "easy",
      "Which statements correctly describe proteins and amino acids?",
      [
        ["Proteins are polymers made from amino acids.", true],
        [
          "Amino-acid side chains help determine how a protein folds and functions.",
          true,
        ],
        [
          "Proteins can act as enzymes, receptors, transporters, or structural components.",
          true,
        ],
        [
          "Proteins store hereditary information using base pairing between nucleotides.",
          false,
        ],
      ],
      "Proteins are built from amino acids and perform many jobs in cells. Hereditary base-pair information is mainly associated with nucleic acids such as Deoxyribonucleic acid (DNA), not proteins.",
    ),
    makeQuestion(
      15,
      "easy",
      "Which statements correctly connect protein shape and denaturation?",
      [
        [
          "A protein's three-dimensional shape can affect which molecules it binds.",
          true,
        ],
        [
          "Heat or pH changes can denature some proteins by disrupting their shape.",
          true,
        ],
        [
          "Denaturation always improves enzyme activity by making the active site more flexible.",
          false,
        ],
        [
          "Protein shape is determined only by the number of chromosomes in the cell.",
          false,
        ],
      ],
      "Protein function often depends on shape, especially at binding sites and active sites. Denaturation usually reduces or destroys function because the original structure is disrupted.",
    ),
    makeQuestion(
      16,
      "easy",
      "Which statement best describes an enzyme?",
      [
        [
          "An enzyme is a biological catalyst that speeds a reaction without being consumed as a reactant.",
          true,
        ],
        [
          "An enzyme raises activation energy so reactions occur only at high temperature.",
          false,
        ],
        [
          "An enzyme is consumed stoichiometrically each time product forms.",
          false,
        ],
        [
          "An enzyme speeds a reaction mainly by changing the final equilibrium ratio.",
          false,
        ],
      ],
      "Enzymes lower activation energy and make reactions proceed faster under biological conditions. Membranes, DNA sequences, and clinical measurements are different concepts that may interact with enzymes but do not define them.",
    ),
    makeQuestion(
      17,
      "easy",
      "Which statements correctly describe nucleic acids?",
      [
        ["DNA and Ribonucleic acid (RNA) are nucleic acids.", true],
        ["Nucleic acids are built from nucleotide subunits.", true],
        [
          "Nucleotide sequence can store or transmit biological information.",
          true,
        ],
        [
          "Base pairing helps explain how nucleic-acid information can be copied or used.",
          true,
        ],
      ],
      "Nucleic acids are information-rich molecules. Their nucleotide sequences and base-pairing properties are central to inheritance, gene expression, and biotechnology.",
    ),
    makeQuestion(
      18,
      "easy",
      "Which statements correctly compare DNA and RNA?",
      [
        ["DNA commonly stores long-term genetic information.", true],
        [
          "Messenger RNA can carry information from DNA toward protein synthesis.",
          true,
        ],
        ["RNA can have roles in translation and gene regulation.", true],
        [
          "DNA and RNA are both built from amino acids rather than nucleotides.",
          false,
        ],
      ],
      "DNA and RNA are nucleic acids, so they are built from nucleotides. RNA is especially important for moving and using genetic information, while DNA is usually the long-term information store.",
    ),
    makeQuestion(
      19,
      "easy",
      "Which statements correctly describe Adenosine triphosphate (ATP)?",
      [
        ["ATP is a common energy carrier in cells.", true],
        [
          "ATP can help couple energy-releasing processes to energy-requiring work.",
          true,
        ],
        [
          "ATP is the hereditary polymer that stores chromosomes in the nucleus.",
          false,
        ],
        [
          "ATP is a membrane channel that moves sodium ions across lipid bilayers.",
          false,
        ],
      ],
      "ATP is often used to power chemical work, transport, and movement. It is not DNA and it is not itself a transport protein, though ATP can provide energy for some transport processes.",
    ),
    makeQuestion(
      20,
      "easy",
      "Which statement best describes metabolism?",
      [
        [
          "Metabolism is the network of chemical reactions that transforms matter and energy in living systems.",
          true,
        ],
        [
          "Metabolism includes only reactions that release energy, not reactions that build molecules.",
          false,
        ],
        [
          "Metabolism is limited to digestion in the stomach and does not occur inside cells.",
          false,
        ],
        [
          "Metabolism refers only to movement of molecules across membranes.",
          false,
        ],
      ],
      "Metabolism includes building molecules, breaking molecules down, and transferring energy. Osmosis, randomization, and transcription are specific concepts, not the whole metabolic network.",
    ),
    makeQuestion(
      21,
      "easy",
      "Which statements correctly describe cells and cell theory?",
      [
        ["Cells are basic units of life.", true],
        ["All known living organisms are made of one or more cells.", true],
        ["Cells arise from pre-existing cells through cell division.", true],
        [
          "A cell can maintain an internal environment different from its surroundings.",
          true,
        ],
      ],
      "Cell theory provides a foundation for biology before moving into organelles or molecular details. Cells are organized units that can grow, divide, regulate conditions, and interact with their environment.",
    ),
    makeQuestion(
      22,
      "easy",
      "Which statements correctly compare prokaryotic and eukaryotic cells?",
      [
        [
          "Bacteria are prokaryotes and do not have a membrane-bound nucleus.",
          true,
        ],
        ["Human cells are eukaryotic and usually contain a nucleus.", true],
        ["Eukaryotic cells can contain membrane-bound organelles.", true],
        [
          "Prokaryotic cells lack DNA and therefore cannot pass information to offspring.",
          false,
        ],
      ],
      "Prokaryotes and eukaryotes differ in internal organization, especially the nucleus and organelles. Prokaryotes still have genetic material and can reproduce.",
    ),
    makeQuestion(
      23,
      "easy",
      "Which statements correctly describe the basic structure of cell membranes?",
      [
        [
          "Phospholipid bilayers have hydrophilic surfaces and hydrophobic interiors.",
          true,
        ],
        [
          "Membrane proteins can help with transport, signaling, or cell recognition.",
          true,
        ],
        ["Membranes are rigid sheets made only of DNA and cellulose.", false],
        [
          "A membrane's selective permeability means every molecule crosses at the same rate.",
          false,
        ],
      ],
      "Membranes are built largely from lipids and proteins, giving them selective permeability. The hydrophobic interior makes it easier for some substances to cross than others.",
    ),
    makeQuestion(
      24,
      "easy",
      "Which statement best describes diffusion?",
      [
        [
          "Particles tend to spread from higher concentration toward lower concentration.",
          true,
        ],
        [
          "Particles diffuse from lower concentration to higher concentration when no energy is supplied.",
          false,
        ],
        [
          "Diffusion requires a membrane transport protein in every case.",
          false,
        ],
        [
          "Diffusion stops because molecules stop moving once a concentration gradient exists.",
          false,
        ],
      ],
      "Diffusion is movement down a concentration gradient caused by random molecular motion. It does not require direct ATP spending for each particle moving down the gradient.",
    ),
    makeQuestion(
      25,
      "easy",
      "Which statements correctly describe osmosis and tonicity?",
      [
        [
          "Osmosis is the movement of water across a selectively permeable membrane.",
          true,
        ],
        [
          "Solute concentration differences can affect the direction of net water movement.",
          true,
        ],
        ["A hypotonic environment can cause some cells to gain water.", true],
        ["A hypertonic environment can cause some cells to lose water.", true],
      ],
      "Osmosis depends on water movement and solute concentration differences. Tonicity vocabulary helps explain why cells swell or shrink in different solutions.",
    ),
    makeQuestion(
      26,
      "easy",
      "Which statements correctly describe active transport?",
      [
        [
          "Active transport can move substances against a concentration gradient.",
          true,
        ],
        ["Active transport often uses energy directly or indirectly.", true],
        ["Protein pumps are one way cells perform active transport.", true],
        [
          "Active transport is the same as simple diffusion through the lipid bilayer.",
          false,
        ],
      ],
      "Active transport differs from passive movement because it can move substances uphill against a gradient. Protein pumps and energy coupling make this possible.",
    ),
    makeQuestion(
      27,
      "easy",
      "Which statements correctly match organelles with common functions?",
      [
        ["The nucleus stores most DNA in many eukaryotic cells.", true],
        [
          "Mitochondria are involved in ATP production in many eukaryotic cells.",
          true,
        ],
        [
          "Ribosomes package and modify proteins for secretion in the same way the Golgi apparatus does.",
          false,
        ],
        [
          "Chloroplasts are the main site of cellular respiration in animal cells.",
          false,
        ],
      ],
      "Organelles divide cellular work into specialized locations. Ribosomes build proteins, the Golgi apparatus modifies and sorts many proteins, and chloroplasts are photosynthetic organelles found in plants and algae rather than animal cells.",
    ),
    makeQuestion(
      28,
      "medium",
      "Which statement best describes ribosomes?",
      [
        [
          "Ribosomes are molecular machines that build proteins from RNA instructions.",
          true,
        ],
        ["Ribosomes copy DNA into RNA during transcription.", false],
        ["Ribosomes synthesize membrane lipids from fatty acids.", false],
        [
          "Ribosomes chemically digest proteins into amino acids for energy.",
          false,
        ],
      ],
      "Ribosomes carry out translation by linking amino acids according to RNA information. DNA storage, lipid storage, and ion pumping are different cellular functions.",
    ),
    makeQuestion(
      29,
      "medium",
      "Which statements correctly describe mitochondria and cellular respiration?",
      [
        ["Mitochondria help convert energy from nutrients into ATP.", true],
        [
          "Cellular respiration can use oxygen as a final electron acceptor in many cells.",
          true,
        ],
        [
          "Carbon dioxide can be produced during aerobic breakdown of glucose.",
          true,
        ],
        [
          "ATP production is connected to gradients across mitochondrial membranes.",
          true,
        ],
      ],
      "Mitochondria connect food molecules, oxygen use, electron transfer, gradients, and ATP production. These ideas support later topics such as metabolism, physiology, and disease.",
    ),
    makeQuestion(
      30,
      "medium",
      "Which statements correctly describe photosynthesis?",
      [
        [
          "Photosynthesis converts light energy into chemical energy in plants, algae, and some bacteria.",
          true,
        ],
        [
          "Carbon dioxide can be used to build sugars during photosynthesis.",
          true,
        ],
        ["Photosynthesis is linked to chloroplasts in plant cells.", true],
        [
          "Photosynthesis is the same pathway as aerobic cellular respiration in animal mitochondria.",
          false,
        ],
      ],
      "Photosynthesis and cellular respiration are connected in ecosystems but are not the same pathway. Photosynthesis stores energy in chemical bonds, while respiration releases usable energy from molecules such as glucose.",
    ),
    makeQuestion(
      31,
      "medium",
      "Which statements correctly describe the cell cycle and mitosis?",
      [
        [
          "DNA is copied before a typical mitotic division produces daughter cells.",
          true,
        ],
        [
          "Mitosis helps eukaryotic cells distribute copied chromosomes to daughter cells.",
          true,
        ],
        [
          "Mitosis halves the chromosome number to make gametes in animals.",
          false,
        ],
        ["The cell cycle has no checkpoints or regulatory controls.", false],
      ],
      "Mitosis supports growth, repair, and asexual cell division by distributing copied chromosomes. Meiosis, not mitosis, is the division process that produces gametes with reduced chromosome number in animals.",
    ),
    makeQuestion(
      32,
      "medium",
      "Which statement best describes meiosis?",
      [
        [
          "Meiosis produces gametes with half the usual chromosome number in sexually reproducing organisms.",
          true,
        ],
        [
          "Meiosis produces two genetically identical diploid daughter cells for tissue repair.",
          false,
        ],
        [
          "Meiosis occurs in all body cells whenever damaged proteins need replacement.",
          false,
        ],
        [
          "Meiosis maintains chromosome number by skipping chromosome separation.",
          false,
        ],
      ],
      "Meiosis is tied to sexual reproduction and chromosome-number reduction. It should not be confused with protein synthesis, membrane transport, or bacterial binary fission.",
    ),
    makeQuestion(
      33,
      "medium",
      "Which statements correctly describe genes, genomes, and chromosomes?",
      [
        ["A gene is a DNA sequence with biological information.", true],
        [
          "A genome is the full set of genetic information in an organism or cell.",
          true,
        ],
        ["A chromosome is an organized DNA-containing structure.", true],
        [
          "Genes can influence traits by affecting RNA and protein production.",
          true,
        ],
      ],
      "These genetics terms describe different levels of biological information organization. They prepare students for gene expression, inheritance, mutations, and biotechnology.",
    ),
    makeQuestion(
      34,
      "medium",
      "Which statements correctly describe gene expression?",
      [
        [
          "Gene expression includes using genetic information to make functional products such as RNA or protein.",
          true,
        ],
        ["Transcription makes RNA from a DNA template.", true],
        ["Translation builds a protein from messenger RNA instructions.", true],
        [
          "Gene expression means every gene in every cell is active at the same level.",
          false,
        ],
      ],
      "Gene expression is regulated, so different cells can use different genes at different times. Transcription and translation are core vocabulary for understanding how DNA information affects cell function.",
    ),
    makeQuestion(
      35,
      "medium",
      "Which statements correctly describe codons and the genetic code?",
      [
        ["A codon is a three-nucleotide sequence in messenger RNA.", true],
        ["Many codons specify amino acids during translation.", true],
        ["Codons are the lipid tails that make membranes hydrophobic.", false],
        [
          "The genetic code means each protein can be made without RNA or ribosomes.",
          false,
        ],
      ],
      "Codons connect nucleotide sequences to amino-acid sequences. They are part of translation, not membrane structure, and ribosomes are needed to read messenger RNA into protein.",
    ),
    makeQuestion(
      36,
      "medium",
      "Which statement best describes a mutation?",
      [
        ["A mutation is a change in genetic sequence.", true],
        [
          "A mutation is any change in protein shape caused by heat, even when DNA sequence is unchanged.",
          false,
        ],
        [
          "A mutation is a planned change in gene expression that never alters sequence.",
          false,
        ],
        [
          "A mutation must remove an entire chromosome to count as genetic change.",
          false,
        ],
      ],
      "A mutation is a change in DNA or another genetic sequence. It may affect a protein, regulation, or no obvious trait at all depending on where it occurs and what changes.",
    ),
    makeQuestion(
      37,
      "medium",
      "Which statements correctly describe genotype, phenotype, and alleles?",
      [
        [
          "A genotype is an organism's genetic makeup at one or more loci.",
          true,
        ],
        ["A phenotype is an observable trait or characteristic.", true],
        ["Alleles are different versions of a gene.", true],
        [
          "Phenotypes can be influenced by genes, environment, and their interaction.",
          true,
        ],
      ],
      "Genotype and phenotype are connected but not identical. Alleles can influence traits, and environmental context can also shape what is observed.",
    ),
    makeQuestion(
      38,
      "medium",
      "Which statements correctly describe basic Mendelian inheritance?",
      [
        [
          "Diploid organisms can carry two alleles for a gene, one from each parent.",
          true,
        ],
        ["Gametes carry one allele for each gene after meiosis.", true],
        ["Punnett squares can organize possible allele combinations.", true],
        [
          "Mendelian inheritance requires acquired traits from exercise to be copied directly into DNA.",
          false,
        ],
      ],
      "Mendelian inheritance tracks allele transmission across generations. Acquired body changes are not automatically written into the DNA sequence passed to offspring.",
    ),
    makeQuestion(
      39,
      "medium",
      "Which statements correctly describe dominant, recessive, and carrier vocabulary?",
      [
        [
          "A dominant allele can affect phenotype when one copy is present.",
          true,
        ],
        [
          "A recessive phenotype often appears only when two recessive alleles are present.",
          true,
        ],
        [
          "A carrier for a recessive condition usually has two copies of the recessive allele.",
          false,
        ],
        [
          "Dominant alleles are always more common in the population than recessive alleles.",
          false,
        ],
      ],
      "Dominance describes phenotype relationships, not how common or beneficial an allele is. A carrier for a recessive condition usually has one recessive allele and one allele that masks it.",
    ),
    makeQuestion(
      40,
      "medium",
      "Which statement best describes natural selection?",
      [
        [
          "Heritable traits that improve reproductive success can become more common in a population over generations.",
          true,
        ],
        [
          "Individuals evolve new traits because they need them during their lifetime.",
          false,
        ],
        [
          "Natural selection changes all individuals in a population in exactly the same way.",
          false,
        ],
        ["Natural selection requires every mutation to be harmful.", false],
      ],
      "Natural selection acts on heritable variation and changes populations over generations. It is not a conscious process and does not guarantee that every individual changes or that every mutation is harmful.",
    ),
    makeQuestion(
      41,
      "medium",
      "Which statements correctly describe evolution in populations?",
      [
        [
          "Evolution can be described as changes in heritable traits or allele frequencies over generations.",
          true,
        ],
        ["Mutation can introduce new genetic variation.", true],
        [
          "Selection, genetic drift, and gene flow can change populations.",
          true,
        ],
        [
          "Evolutionary reasoning helps explain antibiotic resistance and viral variants.",
          true,
        ],
      ],
      "Evolution is a population-level process based on heritable variation. It is useful in medicine because pathogens and cancer cell populations can change under selection pressures.",
    ),
    makeQuestion(
      42,
      "medium",
      "Which statements correctly compare bacteria and viruses?",
      [
        [
          "Bacteria are cells that can often reproduce independently under suitable conditions.",
          true,
        ],
        [
          "Viruses require host-cell machinery to make more virus particles.",
          true,
        ],
        [
          "Antibiotics generally target bacterial processes rather than viral replication.",
          true,
        ],
        ["Viruses and bacteria are both eukaryotic cells with nuclei.", false],
      ],
      "Bacteria and viruses differ in structure and replication. This difference matters because treatments that affect bacterial cell processes may not work against viruses.",
    ),
    makeQuestion(
      43,
      "medium",
      "Which statements correctly describe immune responses?",
      [
        ["Innate immunity provides rapid, general defense mechanisms.", true],
        [
          "Adaptive immunity can produce highly specific responses and memory.",
          true,
        ],
        [
          "Adaptive immunity has no connection to antibodies or lymphocytes.",
          false,
        ],
        [
          "The immune system responds only to viruses and cannot respond to bacteria.",
          false,
        ],
      ],
      "Innate and adaptive immunity work together to defend the body. Antibodies and lymphocytes are major parts of adaptive immunity, and immune responses can target many kinds of threats.",
    ),
    makeQuestion(
      44,
      "medium",
      "Which statement best describes an antibody?",
      [
        ["An antibody is a protein that can bind a specific antigen.", true],
        [
          "An antibody is an immune cell that engulfs bacteria directly.",
          false,
        ],
        ["An antibody is the antigen displayed by a pathogen.", false],
        ["An antibody is a receptor found only inside mitochondria.", false],
      ],
      "Antibodies are proteins used by the immune system to recognize specific molecular targets. They are not genetic code, energy storage lipids, or clinical trial design methods.",
    ),
    makeQuestion(
      45,
      "medium",
      "Which statements correctly describe nervous and endocrine signaling?",
      [
        [
          "Neurons can transmit signals rapidly using electrical and chemical processes.",
          true,
        ],
        [
          "Hormones are chemical signals that can travel through the blood.",
          true,
        ],
        [
          "Target cells must have appropriate receptors to respond to many signals.",
          true,
        ],
        [
          "Both nervous and endocrine signaling help coordinate body functions.",
          true,
        ],
      ],
      "Body systems coordinate through signals, receptors, and responses. Neural signals are often rapid and localized, while hormones can travel through circulation to affect target tissues.",
    ),
    makeQuestion(
      46,
      "medium",
      "Which statements correctly describe feedback and homeostasis?",
      [
        [
          "Homeostasis means maintaining internal conditions within a useful range.",
          true,
        ],
        [
          "Negative feedback often reduces deviations from a set point or range.",
          true,
        ],
        [
          "Positive feedback can amplify a process in some biological situations.",
          true,
        ],
        [
          "Feedback regulation means internal conditions never change at all.",
          false,
        ],
      ],
      "Homeostasis is regulated stability, not complete absence of change. Feedback loops help biological systems respond to disturbances and maintain useful operating ranges.",
    ),
    makeQuestion(
      47,
      "medium",
      "Which statements correctly describe blood glucose regulation?",
      [
        [
          "Insulin helps many cells take up glucose or store energy after blood glucose rises.",
          true,
        ],
        ["Glucagon helps raise blood glucose when it becomes too low.", true],
        [
          "Insulin and glucagon are digestive enzymes that cut DNA into nucleotides.",
          false,
        ],
        [
          "Blood glucose is regulated only by random diffusion with no hormonal control.",
          false,
        ],
      ],
      "Insulin and glucagon are hormones involved in glucose homeostasis. They are useful examples of signaling, feedback, metabolism, and disease relevance.",
    ),
    makeQuestion(
      48,
      "medium",
      "Which statement best describes a receptor-ligand interaction?",
      [
        [
          "A ligand binds to a receptor and can change receptor activity or cell response.",
          true,
        ],
        ["A receptor copies the ligand's genetic sequence into RNA.", false],
        ["A ligand must be a whole cell rather than a molecule.", false],
        [
          "A receptor-ligand interaction can occur only when both partners are DNA.",
          false,
        ],
      ],
      "Receptors and ligands are central to cell signaling and drug action. They should not be confused with transcription, inheritance, or polymer chemistry.",
    ),
    makeQuestion(
      49,
      "medium",
      "Which statements correctly describe drugs and biological targets?",
      [
        [
          "Many drugs act by binding proteins such as receptors, enzymes, channels, or transporters.",
          true,
        ],
        [
          "A drug's effect can depend on dose, target tissue, and patient biology.",
          true,
        ],
        ["A treatment can have intended effects and unintended effects.", true],
        [
          "Understanding molecular targets can help explain why a medicine works.",
          true,
        ],
      ],
      "Drug action is based on interactions with biological systems, not just memorizing drug names. Target, dose, exposure, and context all help explain benefits and risks.",
    ),
    makeQuestion(
      50,
      "medium",
      "Which statements correctly describe agonists and antagonists?",
      [
        [
          "An agonist activates a receptor or mimics the effect of a signal.",
          true,
        ],
        ["An antagonist blocks or reduces receptor activation.", true],
        [
          "These terms are useful for describing some drug effects on signaling pathways.",
          true,
        ],
        [
          "An antagonist is defined as any molecule that increases DNA mutation rate.",
          false,
        ],
      ],
      "Agonists and antagonists describe effects on receptor activity. Mutation rate is a separate genetics concept and is not what defines antagonist action.",
    ),
    makeQuestion(
      51,
      "medium",
      "Which statements correctly describe dose-response relationships?",
      [
        [
          "A dose-response curve can show how an effect changes as dose changes.",
          true,
        ],
        [
          "Higher dose can increase benefits, side effects, or both depending on the drug.",
          true,
        ],
        [
          "Dose-response curves are used only for inherited traits and not for medicines.",
          false,
        ],
        ["Dose has no relevance once a molecule can bind a receptor.", false],
      ],
      "Dose-response thinking helps connect chemistry, physiology, and pharmacology. Binding is important, but the amount of drug and the size of the response also matter.",
    ),
    makeQuestion(
      52,
      "medium",
      "Which statement best describes a side effect?",
      [
        [
          "A side effect is an unintended effect of a treatment or intervention.",
          true,
        ],
        [
          "A side effect is the main beneficial effect for which a treatment is prescribed.",
          false,
        ],
        [
          "A side effect is any outcome measured in a study, whether intended or unintended.",
          false,
        ],
        [
          "A side effect can occur only when a drug misses every biological target.",
          false,
        ],
      ],
      "Side effects are unintended outcomes, which may be mild, serious, predictable, or rare. They are not the same as trial endpoints, DNA replication, or transport proteins.",
    ),
    makeQuestion(
      53,
      "medium",
      "Which statements correctly describe pharmacokinetics?",
      [
        [
          "Pharmacokinetics describes what the body does to a drug over time.",
          true,
        ],
        ["Absorption affects how a drug enters the body.", true],
        ["Metabolism can chemically modify a drug, often in the liver.", true],
        ["Excretion removes drug or drug products from the body.", true],
      ],
      "Pharmacokinetics is often summarized as absorption, distribution, metabolism, and excretion. These processes shape drug exposure and help explain timing, dosing, and variability between patients.",
    ),
    makeQuestion(
      54,
      "medium",
      "Which statements correctly describe biomarkers and diagnosis?",
      [
        [
          "A biomarker is a measurable biological sign related to a process, condition, or response.",
          true,
        ],
        ["Blood glucose can be a biomarker relevant to diabetes.", true],
        [
          "Some biomarkers help monitor whether a treatment is having an effect.",
          true,
        ],
        [
          "A biomarker must always be a gene sequence and can never be a protein or physiological measurement.",
          false,
        ],
      ],
      "Biomarkers can be molecules, physiological measurements, imaging findings, or genetic variants. They are useful for diagnosis, monitoring, risk assessment, and treatment decisions.",
    ),
    makeQuestion(
      55,
      "hard",
      "Which statements correctly describe broad disease categories?",
      [
        [
          "Infectious diseases involve pathogens such as bacteria, viruses, fungi, or parasites.",
          true,
        ],
        [
          "Autoimmune diseases involve immune responses against the body's own components.",
          true,
        ],
        [
          "Metabolic disease is defined by the absence of any chemical reactions in cells.",
          false,
        ],
        ["All diseases are inherited single-gene disorders.", false],
      ],
      "Diseases can be infectious, autoimmune, metabolic, genetic, degenerative, malignant, or multifactorial. No single category explains all disease.",
    ),
    makeQuestion(
      56,
      "hard",
      "Which statement best describes cancer at a basic level?",
      [
        [
          "Cancer involves abnormal cell growth and division caused by disrupted regulation.",
          true,
        ],
        [
          "Cancer is controlled cell division that stops normally after tissue repair.",
          false,
        ],
        [
          "Cancer requires all cells in the body to divide at the same rate.",
          false,
        ],
        [
          "Cancer is defined by infection alone, regardless of cell growth regulation.",
          false,
        ],
      ],
      "Cancer is tied to failures in controls over growth, division, survival, and tissue behavior. Infections, meiosis, and mutation-free cell division are different concepts.",
    ),
    makeQuestion(
      57,
      "hard",
      "Which statements correctly describe inflammation and immunity?",
      [
        [
          "Inflammation can increase blood flow and immune activity in affected tissue.",
          true,
        ],
        ["Inflammation can help fight infection or repair damage.", true],
        ["Excessive or chronic inflammation can contribute to disease.", true],
        [
          "Immune signaling molecules can coordinate responses between cells.",
          true,
        ],
      ],
      "Inflammation can be protective, but it can also cause harm when poorly regulated. This makes it a useful bridge between cell signaling, physiology, and disease.",
    ),
    makeQuestion(
      58,
      "hard",
      "Which statements correctly describe vaccines?",
      [
        [
          "Vaccines train the immune system to recognize a pathogen or pathogen component.",
          true,
        ],
        [
          "Vaccination can create immune memory before a dangerous exposure occurs.",
          true,
        ],
        [
          "Some vaccines use proteins, weakened pathogens, inactivated pathogens, or genetic instructions.",
          true,
        ],
        [
          "Vaccines work by directly killing every pathogen already present in the body within seconds.",
          false,
        ],
      ],
      "Vaccines prepare immune responses rather than acting like instant pathogen-killing chemicals. Different vaccine designs present antigens or instructions in different ways.",
    ),
    makeQuestion(
      59,
      "hard",
      "Which statements correctly compare antibiotics and antivirals?",
      [
        ["Antibiotics often target bacterial structures or processes.", true],
        [
          "Antivirals often target steps in viral entry, replication, or release.",
          true,
        ],
        [
          "Antibiotics usually treat viral infections by blocking viral capsid assembly.",
          false,
        ],
        [
          "Antivirals are defined by killing only human cells and never affecting viruses.",
          false,
        ],
      ],
      "Antibiotics and antivirals differ because bacteria and viruses differ. A treatment must match the biology of the pathogen and the infection.",
    ),
    makeQuestion(
      60,
      "hard",
      "Which statement best describes an independent variable in an experiment?",
      [
        [
          "The independent variable is the factor the investigator changes or compares to test its effect.",
          true,
        ],
        [
          "The independent variable is the outcome measured after the experiment.",
          false,
        ],
        [
          "The independent variable is any uncontrolled difference between two groups.",
          false,
        ],
        [
          "The independent variable is the random error caused by measuring instruments.",
          false,
        ],
      ],
      "The independent variable is the input, treatment, or condition being tested. The measured outcome is the dependent variable, while uncontrolled differences and measurement error are separate design issues.",
    ),
    makeQuestion(
      61,
      "hard",
      "Which statements correctly describe controls and reproducibility in experiments?",
      [
        [
          "A control group helps provide a comparison for the experimental group.",
          true,
        ],
        [
          "A negative control can show what happens when the expected active factor is absent.",
          true,
        ],
        [
          "Reproducibility is stronger when independent investigators can obtain similar results under comparable conditions.",
          true,
        ],
        [
          "Controls help identify whether a result might be due to the tested factor or some other part of the setup.",
          true,
        ],
      ],
      "Controls make experimental interpretation more precise by separating the tested factor from background effects. Reproducibility matters because a result is more trustworthy when it does not depend on one fragile setup.",
    ),
    makeQuestion(
      62,
      "hard",
      "Which statements correctly describe sample size and variability?",
      [
        [
          "Larger sample sizes can reduce the influence of random variation when the design is sound.",
          true,
        ],
        [
          "Biological measurements often vary between individuals even without a treatment effect.",
          true,
        ],
        [
          "High variability can make it harder to detect a real difference.",
          true,
        ],
        [
          "A large sample size removes the need for a comparison group or clear endpoint.",
          false,
        ],
      ],
      "Sample size helps with random noise, but it does not fix every design problem. A large poorly controlled study can still be misleading.",
    ),
    makeQuestion(
      63,
      "hard",
      "Which statements correctly describe randomization, blinding, and placebo controls?",
      [
        [
          "Randomization helps distribute known and unknown participant differences across groups.",
          true,
        ],
        [
          "Blinding can reduce bias from expectations by participants or investigators.",
          true,
        ],
        [
          "A placebo control guarantees that the active treatment has no biological effect.",
          false,
        ],
        [
          "Randomization means participants choose the group they believe will help them most.",
          false,
        ],
      ],
      "Randomization and blinding strengthen causal interpretation by reducing systematic differences and expectation effects. A placebo is a comparison tool, not proof about the active treatment by itself.",
    ),
    makeQuestion(
      64,
      "hard",
      "Which statement best describes a clinical trial endpoint?",
      [
        [
          "An endpoint is a pre-defined outcome used to evaluate an intervention.",
          true,
        ],
        [
          "An endpoint is any result a researcher notices after looking at the data, whether or not it was planned.",
          false,
        ],
        [
          "An endpoint is a baseline variable controlled to make groups comparable.",
          false,
        ],
        [
          "An endpoint is a subgroup created after randomization to improve the result.",
          false,
        ],
      ],
      "Endpoints should be defined clearly so the study has an interpretable target. Choosing outcomes only after seeing data can inflate misleading findings.",
    ),
    makeQuestion(
      65,
      "hard",
      "Which statements correctly distinguish correlation, causation, and confounding?",
      [
        ["Correlation means two variables tend to vary together.", true],
        [
          "Causation means one factor contributes to producing an effect.",
          true,
        ],
        ["A confounder is related to both the exposure and the outcome.", true],
        [
          "Randomized experiments can help reduce confounding when they are designed and executed well.",
          true,
        ],
      ],
      "Associations are common in biomedical data, but not every association is causal. Confounding is one reason observational patterns can point in the wrong direction.",
    ),
    makeQuestion(
      66,
      "hard",
      "Which statements correctly compare statistical and clinical significance?",
      [
        [
          "Statistical significance depends on a specified analysis and chance model.",
          true,
        ],
        [
          "Clinical significance asks whether the effect is meaningful for health or decisions.",
          true,
        ],
        [
          "A very small effect can be statistically significant in a large study.",
          true,
        ],
        [
          "Statistical significance proves the study had no bias or confounding.",
          false,
        ],
      ],
      "Statistical significance and practical importance are related but different. Bias, confounding, and measurement choices still matter even when a p-value looks impressive.",
    ),
    makeQuestion(
      67,
      "hard",
      "Which statements correctly describe sensitivity and specificity?",
      [
        [
          "Sensitivity measures how well a test identifies people who truly have a condition.",
          true,
        ],
        [
          "Specificity measures how well a test identifies people who truly do not have a condition.",
          true,
        ],
        [
          "A highly sensitive test necessarily has perfect specificity in every population.",
          false,
        ],
        [
          "Sensitivity and specificity are the same as treatment efficacy and side-effect rate.",
          false,
        ],
      ],
      "Sensitivity and specificity describe diagnostic test performance from different angles. They do not automatically move together and should not be confused with treatment outcomes.",
    ),
    makeQuestion(
      68,
      "hard",
      "A graph shows enzyme activity rising with temperature up to 37 degrees C and then falling at higher temperatures. Which interpretation is best supported?",
      [
        [
          "The enzyme has an activity optimum near 37 degrees C under the tested conditions.",
          true,
        ],
        [
          "The enzyme's DNA sequence must be changing at each temperature point.",
          false,
        ],
        [
          "The enzyme is equally active at all temperatures because enzymes are catalysts.",
          false,
        ],
        ["Temperature cannot affect protein shape or reaction rate.", false],
      ],
      "The graph supports an optimum under the tested conditions, not a claim that DNA sequence changes with temperature. Enzymes are catalysts, but their activity can still depend on temperature and protein structure.",
    ),
    makeQuestion(
      69,
      "hard",
      "Which statements correctly describe concentration, moles, and solutions?",
      [
        [
          "A mole is a counting unit for particles such as atoms or molecules.",
          true,
        ],
        ["Molarity describes moles of solute per liter of solution.", true],
        [
          "Concentration affects how often dissolved molecules may encounter each other.",
          true,
        ],
        [
          "Changing concentration can affect diffusion, reaction rates, and dose.",
          true,
        ],
      ],
      "Concentration vocabulary connects basic chemistry to biology and medicine. Cells, enzymes, gradients, and drug dosing all depend on how much substance is present in a given volume.",
    ),
    makeQuestion(
      70,
      "hard",
      "Which statements correctly describe dilutions?",
      [
        [
          "Dilution lowers solute concentration by adding solvent or increasing total volume.",
          true,
        ],
        [
          "If solute amount stays the same and volume increases, concentration decreases.",
          true,
        ],
        ["Serial dilutions can create a range of known concentrations.", true],
        [
          "Diluting a solution increases concentration because the solute becomes more chemically active.",
          false,
        ],
      ],
      "Dilution changes concentration by changing the ratio of solute to total volume. This is basic but important for lab assays, dose preparation, and interpreting concentration-response data.",
    ),
    makeQuestion(
      71,
      "hard",
      "Which statements correctly describe reaction rates and equilibrium?",
      [
        [
          "Reaction rate describes how quickly reactants are converted to products.",
          true,
        ],
        [
          "At dynamic equilibrium, forward and reverse reactions occur at equal rates.",
          true,
        ],
        ["Equilibrium means every molecule has stopped moving.", false],
        [
          "A catalyst changes the final equilibrium ratio by being consumed in the reaction.",
          false,
        ],
      ],
      "Equilibrium is dynamic, so molecules still move and reactions can still occur. Catalysts speed approach to equilibrium but do not get consumed as ordinary reactants.",
    ),
    makeQuestion(
      72,
      "hard",
      "Which statement best describes activation energy?",
      [
        [
          "Activation energy is the energy barrier that must be overcome for a reaction to proceed.",
          true,
        ],
        [
          "Activation energy is the total heat released after products form.",
          false,
        ],
        [
          "Activation energy is the final concentration of product at equilibrium.",
          false,
        ],
        [
          "Activation energy is the volume of solvent added during a dilution.",
          false,
        ],
      ],
      "Activation energy explains why many reactions need a push to proceed at useful rates. Enzymes help by lowering this barrier rather than by changing DNA amount or dilution volume.",
    ),
    makeQuestion(
      73,
      "hard",
      "Which statements correctly describe oxidation and reduction?",
      [
        [
          "Oxidation involves loss of electrons or increased oxidation state.",
          true,
        ],
        [
          "Reduction involves gain of electrons or decreased oxidation state.",
          true,
        ],
        ["Redox reactions are important in cellular respiration.", true],
        ["Electron carriers can transfer electrons between reactions.", true],
      ],
      "Redox vocabulary helps explain energy transfer in metabolism. Cellular respiration depends on controlled electron movement, which is why electron carriers matter.",
    ),
    makeQuestion(
      74,
      "hard",
      "Which statements correctly describe dehydration synthesis and hydrolysis?",
      [
        [
          "Dehydration synthesis can join monomers while releasing water.",
          true,
        ],
        ["Hydrolysis can break polymers by adding water.", true],
        [
          "These reactions are relevant to building and breaking biological macromolecules.",
          true,
        ],
        [
          "Hydrolysis always joins amino acids into longer chains without using water.",
          false,
        ],
      ],
      "Dehydration and hydrolysis are opposite patterns often used for polymers. They help students understand how cells assemble and break down carbohydrates, proteins, and nucleic acids.",
    ),
    makeQuestion(
      75,
      "hard",
      "Which statements correctly describe ions and membrane potentials?",
      [
        [
          "Unequal ion distributions across a membrane can create an electrical potential.",
          true,
        ],
        [
          "Ion channels can allow specific ions to move across membranes.",
          true,
        ],
        [
          "Membrane potentials require chromosomes to move across the plasma membrane.",
          false,
        ],
        [
          "Sodium and potassium ions are irrelevant to nerve and muscle cell signaling.",
          false,
        ],
      ],
      "Membrane potentials depend on ion gradients and selective permeability. They are especially important for excitable cells such as neurons and muscle cells.",
    ),
    makeQuestion(
      76,
      "hard",
      "A red blood cell is placed in pure water and swells. Which explanation is best?",
      [
        [
          "Water enters the cell by osmosis because the outside solution is hypotonic relative to the cytoplasm.",
          true,
        ],
        [
          "The cell swells because water leaves the cell faster than it enters.",
          false,
        ],
        [
          "The cell swells because the membrane becomes completely impermeable to water.",
          false,
        ],
        [
          "The cell swells because pure water has a higher solute concentration than cytoplasm.",
          false,
        ],
      ],
      "Pure water has very low solute concentration compared with the inside of a red blood cell. Net water entry by osmosis can cause swelling and, if extreme, lysis.",
    ),
    makeQuestion(
      77,
      "hard",
      "Which statements correctly describe cell communication pathways?",
      [
        ["A signal can be received by a receptor.", true],
        [
          "Signal transduction can convert receptor activation into intracellular changes.",
          true,
        ],
        [
          "A cellular response might involve enzyme activity, gene expression, secretion, or movement.",
          true,
        ],
        [
          "Feedback can adjust the strength or duration of a signaling response.",
          true,
        ],
      ],
      "Cell communication often involves reception, transduction, and response. This vocabulary prepares students for hormones, neurotransmitters, immune signals, and drug effects.",
    ),
    makeQuestion(
      78,
      "hard",
      "Which statements correctly describe transcription factors and epigenetic regulation?",
      [
        [
          "Transcription factors are proteins that can affect transcription of genes.",
          true,
        ],
        [
          "Epigenetic changes can influence gene expression without changing the DNA sequence.",
          true,
        ],
        [
          "Cell type differences can partly reflect different patterns of gene regulation.",
          true,
        ],
        [
          "Epigenetic regulation means every nucleotide in DNA is replaced by an amino acid.",
          false,
        ],
      ],
      "Gene regulation helps explain how cells with the same genome can behave differently. Epigenetic regulation changes how information is used, not the basic DNA sequence itself.",
    ),
    makeQuestion(
      79,
      "hard",
      "Which statements correctly describe gene therapy, messenger RNA medicines, and CRISPR at a basic level?",
      [
        [
          "Gene therapy aims to treat disease by adding, replacing, or modifying genetic information.",
          true,
        ],
        [
          "Messenger RNA medicines can provide temporary instructions for cells to make a protein.",
          true,
        ],
        [
          "CRISPR-based treatments are best described as ordinary antibiotics that block bacterial cell wall synthesis.",
          false,
        ],
        [
          "Gene therapy and messenger RNA medicines work without interacting with cells or molecules.",
          false,
        ],
      ],
      "Modern biotechnology builds on basic ideas about DNA, RNA, proteins, and cells. CRISPR is associated with targeted genome editing, while messenger RNA approaches use RNA instructions rather than changing every cell permanently.",
    ),
    makeQuestion(
      80,
      "hard",
      "Which statement best describes a careful use of AI in biomedical data?",
      [
        [
          "AI model outputs should be validated with appropriate data and interpreted using biological or clinical context.",
          true,
        ],
        [
          "AI predictions remove the need for controls, measurements, or independent validation.",
          false,
        ],
        [
          "AI models prove causation whenever they find a correlation in patient data.",
          false,
        ],
        [
          "AI systems are unaffected by biased, incomplete, or poorly measured training data.",
          false,
        ],
      ],
      "AI can help find patterns, but biomedical interpretation still depends on data quality and evidence. Validation, bias checks, and domain knowledge remain necessary.",
    ),
    makeQuestion(
      81,
      "hard",
      "A student reads that a biomarker is higher in patients with a disease than in healthy controls. Which interpretation is best supported by that statement alone?",
      [
        [
          "The biomarker is associated with the disease in the compared groups.",
          true,
        ],
        ["The biomarker must be the cause of the disease.", false],
        [
          "Changing the biomarker with a drug must improve patient outcomes.",
          false,
        ],
        [
          "The biomarker must be perfectly sensitive and perfectly specific.",
          false,
        ],
      ],
      "An observed difference supports an association, but it does not by itself prove causation or treatment value. Biomarkers also need validation before they can be treated as reliable diagnostic or clinical-decision tools.",
    ),
  ];

export const BiologyChemistryLifeScienceL0Questions =
  BiologyChemistryForLifeScienceLecture0PreparationQuestions;
