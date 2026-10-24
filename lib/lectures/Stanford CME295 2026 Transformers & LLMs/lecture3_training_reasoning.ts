import type { Question } from "../../quiz";
import { referenceQuestion } from "../../questionReferences";
import { stanfordCME295Lecture4TrainingQuestions } from "../Stanford CME295 Transformers & LLMs/lecture4_training";
import { stanfordCME295Lecture5PreferenceTuningQuestions } from "../Stanford CME295 Transformers & LLMs/lecture5_preference_tuning";
import { stanfordCME295Lecture6ReasoningQuestions } from "../Stanford CME295 Transformers & LLMs/lecture6_reasoning";

export const stanfordCME2952026Lecture3Questions: Question[] = [
  // Canonical reuse; source membership does not change the original chapter or ID.
  referenceQuestion(
    stanfordCME295Lecture4TrainingQuestions,
    "cme295-lect4-q109",
  ),
  // 2026-specific item; stable identity retained.
  {
    id: "cme295-2026-lect3-q02",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "A decoder-only model is pretrained on a large corpus of ordinary text and code. Which statements describe its core objective?",
    options: [
      {
        text: "Predict each next token from its preceding tokens.",
        isCorrect: true,
      },
      {
        text: "Predict each token using the true future tokens as unrestricted context.",
        isCorrect: false,
      },
      {
        text: "Fit a human preference score attached to every corpus token.",
        isCorrect: false,
      },
      {
        text: "Obtain prediction targets from the corpus itself.",
        isCorrect: true,
      },
    ],
    explanation:
      "Autoregressive pretraining learns from the next tokens already present in raw sequences, so it does not require preference annotations. Causal masking excludes future-token information; otherwise a model could reconstruct labels from information unavailable at generation time.",
  },
  // 2026-specific item; stable identity retained.
  {
    id: "cme295-2026-lect3-q03",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Why can teacher-forced causal training process many target positions in parallel even though sampling a completion is sequential?",
    options: [
      {
        text: "The training mask permits each position to use its true next token as input.",
        isCorrect: false,
      },
      {
        text: "During sampling, the next prefix depends on the token just generated.",
        isCorrect: true,
      },
      {
        text: "The true prefix tokens for every training position are already known.",
        isCorrect: true,
      },
      {
        text: "Teacher forcing removes attention's dependency on prefix tokens by averaging independent token losses.",
        isCorrect: false,
      },
    ],
    explanation:
      "A shifted training sequence supplies the actual prefix tokens at every position, allowing a masked Transformer pass to evaluate many next-token losses in parallel. Sampling must construct those prefixes as it proceeds. Parallel loss evaluation does not permit the answer token as its own input and does not remove the hidden state's dependence on earlier prefix tokens.",
  },
  // 2026-specific item; stable identity retained.
  {
    id: "cme295-2026-lect3-q04",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "A two-token target receives probabilities \\(1/2\\) and \\(1/4\\) at its respective causal prefixes. What is its mean cross-entropy in nats, \\(L=-\\frac{1}{2}\\sum_t\\ln p_t\\)?",
    options: [
      {
        text: "\\(\\ln 2\\)",
        isCorrect: false,
      },
      {
        text: "\\(-\\frac{3}{2}\\ln 2\\)",
        isCorrect: false,
      },
      {
        text: "\\(3\\ln 2\\)",
        isCorrect: false,
      },
      {
        text: "\\(\\frac{3}{2}\\ln 2\\)",
        isCorrect: true,
      },
    ],
    explanation:
      "The two negative log probabilities are ln 2 and ln 4, whose sum is 3 ln 2. Averaging over two target tokens gives 1.5 ln 2; omitting the average doubles the result, and a negative value would use the wrong sign for negative log likelihood.",
  },
  // Replaces retired ID cme295-2026-lect3-q05; new reasoning task.
  {
    id: "cme295-2026-lect3-q071",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "A corpus has 200 tokens and 400 bytes. Mean token cross-entropy is \\(\\ln2\\) nats. Using \\(\\mathrm{BPB}=L_{\\mathrm{CE}}(T/B)/\\ln2\\), select all true statements.",
    options: [
      {
        text: "Its total negative log likelihood is 400 bits.",
        isCorrect: false,
      },
      {
        text: "Its bits-per-byte score is 2.",
        isCorrect: false,
      },
      {
        text: "Its bits-per-byte score is 0.5.",
        isCorrect: true,
      },
      {
        text: "Its total negative log likelihood is 200 bits.",
        isCorrect: true,
      },
    ],
    explanation:
      "The mean loss is ln 2 nats per token, which is one bit per token after dividing by ln 2. There are 200 tokens, so the total negative log likelihood is 200 bits; dividing by 400 bytes gives 0.5 bits per byte. The two-bits-per-byte result reverses tokens per byte, while counting 400 bits confuses the byte count with the number of one-bit token losses.",
  },
  // 2026-specific item; stable identity retained.
  {
    id: "cme295-2026-lect3-q06",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "Which sources can supply token sequences for a broad pretraining mixture?",
    options: [
      {
        text: "Mathematical or scientific documents.",
        isCorrect: true,
      },
      {
        text: "Web pages and reference articles.",
        isCorrect: true,
      },
      {
        text: "Books and other long-form writing.",
        isCorrect: true,
      },
      {
        text: "Source-code repositories and programming discussions.",
        isCorrect: true,
      },
    ],
    explanation:
      "The next-token objective applies to many kinds of language and code, so a corpus can combine web, code, mathematical, scientific, and long-form sources. Their proportions and quality matter because the training distribution influences which patterns and capabilities receive learning signal.",
  },
  // Replaces retired ID cme295-2026-lect3-q07; new reasoning task.
  {
    id: "cme295-2026-lect3-q051",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "A pretraining corpus contains two million distinct token positions and is processed for three complete epochs with no augmentation. Select all true statements about this data exposure.",
    options: [
      {
        text: "Training processes six million token positions in total.",
        isCorrect: true,
      },
      {
        text: "The corpus now contains six million distinct token positions.",
        isCorrect: false,
      },
      {
        text: "Each original token position is presented three times.",
        isCorrect: true,
      },
      {
        text: "The later epochs reuse text rather than supplying newly collected text.",
        isCorrect: true,
      },
    ],
    explanation:
      "Three passes over two million positions produce six million processed positions but still only two million distinct source positions. Each position is encountered three times. Repetition can affect learning, yet repeated exposure is not new unique data; confusing those counts can overstate the diversity supporting a training run.",
  },
  // Replaces retired ID cme295-2026-lect3-q08; new reasoning task.
  {
    id: "cme295-2026-lect3-q052",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Before deduplication, a training mixture has eight million tokens from source A and two million from source B. Deduplication leaves four million from A and two million from B. Select all true statements about the remaining corpus and restoring its original sampling proportions.",
    options: [
      {
        text: "Sampling A at twice the per-token rate of B restores an 80% A / 20% B mixture.",
        isCorrect: true,
      },
      {
        text: "Without reweighting, B's share rises from 20% to one third.",
        isCorrect: true,
      },
      {
        text: "Equal per-token sampling after deduplication gives A a two-thirds share.",
        isCorrect: true,
      },
      {
        text: "Six million tokens remain in the deduplicated corpus.",
        isCorrect: true,
      },
    ],
    explanation:
      "The remaining counts sum to six million, with A:B = 4:2 instead of the original 8:2. B therefore rises from 2/10 to 2/6 of the unweighted corpus. Giving each A token twice the sampling weight yields weighted counts 8:2 and restores the original mixture. Deduplication can change domain proportions even when its direct aim is removing repeated content.",
  },
  // Replaces retired ID cme295-2026-lect3-q09; new reasoning task.
  {
    id: "cme295-2026-lect3-q053",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Two data mixtures are compared with lower validation loss preferred. A small proxy model gets loss 2.0 on mixture A and 2.1 on B. At the intended model scale, matched-budget runs get 2.4 on A and 2.2 on B. Which interpretation of this comparison is correct?",
    options: [
      {
        text: "The proxy favors B by 0.1, and the larger model favors B by 0.2; the ranking transferred.",
        isCorrect: false,
      },
      {
        text: "The proxy favors A by 0.1, but the larger model favors B by 0.4; this pair's ranking did not transfer.",
        isCorrect: false,
      },
      {
        text: "The proxy favors A by 0.1, and the larger model favors A by 0.2; the ranking transferred.",
        isCorrect: false,
      },
      {
        text: "The proxy favors A by 0.1, but the larger model favors B by 0.2; this pair's ranking did not transfer.",
        isCorrect: true,
      },
    ],
    explanation:
      "Lower loss selects A at the proxy scale because 2.0 < 2.1, but B at the larger scale because 2.2 < 2.4. The differences are 0.1 and 0.2 within their respective comparisons. Proxy experiments can reduce search cost, but their usefulness depends on ranking transfer; losses across different scales should not be subtracted as though they were the mixture effect.",
  },
  // Replaces retired ID cme295-2026-lect3-q10; new reasoning task.
  {
    id: "cme295-2026-lect3-q054",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "A model's context curriculum increases sequence length from 2,048 to 32,768 tokens. Assume dense attention and unchanged head count. Select all true consequences and requirements of this extension.",
    options: [
      {
        text: "The number of token positions per sequence increases by a factor of 16.",
        isCorrect: true,
      },
      {
        text: "A positional formula being computable at 32,768 positions does not establish learned use of information at those distances.",
        isCorrect: true,
      },
      {
        text: "Training examples with meaningful long-distance dependencies can test capabilities that merely padding short examples would not exercise.",
        isCorrect: true,
      },
      {
        text: "A full attention score matrix has 256 times as many entries per head.",
        isCorrect: true,
      },
    ],
    explanation:
      "The length ratio is 32,768/2,048 = 16, and the dense score-count ratio is 16² = 256. Extending positional computations makes those positions representable, but does not teach the model to use distant evidence by itself. A context curriculum needs informative long examples and evaluation of actual long-distance use, while also accounting for the higher cost of each dense-attention sequence.",
  },
  // Replaces retired ID cme295-2026-lect3-q11; new reasoning task.
  {
    id: "cme295-2026-lect3-q055",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "Which training example directly matches a base language model's ordinary next-token pretraining objective?",
    options: [
      {
        text: "Sample several answers to an instruction and increase the probability of the ones with higher test rewards.",
        isCorrect: false,
      },
      {
        text: "Use a text prefix as input and its actual following token as the prediction target.",
        isCorrect: true,
      },
      {
        text: "Sample a student continuation and match a teacher's next-token distribution at its visited prefixes.",
        isCorrect: false,
      },
      {
        text: "Use an instruction as input and a curated assistant response as the response-only target.",
        isCorrect: false,
      },
    ],
    explanation:
      "Ordinary pretraining learns to predict the next token in text, so it does not require that the prefix be a user instruction. Curated response targets describe supervised fine-tuning, sampled reward-based updates describe reinforcement learning, and teacher distributions at student prefixes describe on-policy distillation. These stages can follow pretraining but provide different training signals.",
  },
  // Canonical reuse; source membership does not change the original chapter or ID.
  referenceQuestion(
    stanfordCME295Lecture4TrainingQuestions,
    "cme295-lect4-q149",
  ),
  // Replaces retired ID cme295-2026-lect3-q13; new reasoning task.
  {
    id: "cme295-2026-lect3-q056",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "A supervised fine-tuning example has three prompt tokens followed by two assistant-response tokens. After the usual prediction shift, list the loss weights in target-token order, one for each of these five tokens. Which mask implements response-only loss?",
    options: [
      {
        text: "[0, 0, 0, 1, 1]",
        isCorrect: true,
      },
      {
        text: "[1, 1, 1, 1, 1]",
        isCorrect: false,
      },
      {
        text: "[1, 1, 1, 0, 0]",
        isCorrect: false,
      },
      {
        text: "[0, 0, 1, 1, 1]",
        isCorrect: false,
      },
    ],
    explanation:
      "Response-only loss assigns zero weight to targets that belong to the prompt and unit weight to the two assistant targets. Masking prompt targets does not remove the prompt from the model's input context. Including the last prompt token is an off-by-one target-boundary error; including every target instead changes the objective to full-sequence loss.",
  },
  // Replaces retired ID cme295-2026-lect3-q14; new reasoning task.
  {
    id: "cme295-2026-lect3-q057",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "A team wants to isolate whether adding representative customer-support dialogues to supervised fine-tuning improves support-task performance. Which experiment best isolates that data-mixture change?",
    options: [
      {
        text: "Compare matched runs but evaluate the old mixture on archived support tasks and the new mixture on newly collected support tasks.",
        isCorrect: false,
      },
      {
        text: "Compare two runs from the same checkpoint with matched training-token budgets and hyperparameters, changing the share of support dialogues, then evaluate both on the same held-out support set.",
        isCorrect: true,
      },
      {
        text: "Compare equal-size runs with the new mixture and twice as many training tokens, then evaluate both on the same held-out support set.",
        isCorrect: false,
      },
      {
        text: "Compare the current model with a larger checkpoint trained on the new mixture, then evaluate both on the same held-out support set.",
        isCorrect: false,
      },
    ],
    explanation:
      "Keeping the checkpoint, budget, optimization setup, and evaluation set fixed makes the dialogue share the intended treatment. Changing model size or training-token count adds a competing explanation for a performance difference. Changing the evaluation set also changes what is measured, so it cannot isolate the mixture effect without further controls.",
  },
  // 2026-specific item; stable identity retained.
  {
    id: "cme295-2026-lect3-q15",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Low-Rank Adaptation (LoRA) uses \\(W=W_0+BA\\), with frozen \\(W_0\\). Which interpretations are correct?",
    options: [
      {
        text: "The adapter factors can have far fewer parameters than a full matrix update.",
        isCorrect: true,
      },
      {
        text: "The learned change is the product BA.",
        isCorrect: true,
      },
      {
        text: "LoRA requires the original pretrained matrix W0 to be low rank.",
        isCorrect: false,
      },
      {
        text: "The rank of BA is bounded by the adapter rank.",
        isCorrect: true,
      },
    ],
    explanation:
      "LoRA constrains the update, not the already trained base matrix. The two smaller factors define an additive low-rank correction; W0 can remain full rank and frozen while gradients update the adapter parameters.",
  },
  // 2026-specific item; stable identity retained.
  {
    id: "cme295-2026-lect3-q16",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "A full matrix has shape \\(4096\\times4096\\). A LoRA update uses rank \\(r=8\\), with \\(B\\in\\mathbb{R}^{4096\\times8}\\) and \\(A\\in\\mathbb{R}^{8\\times4096}\\). How many trainable adapter parameters are in \\(B\\) and \\(A\\) together?",
    options: [
      {
        text: "16,777,216 parameters.",
        isCorrect: false,
      },
      {
        text: "8 parameters.",
        isCorrect: false,
      },
      {
        text: "65,536 parameters.",
        isCorrect: true,
      },
      {
        text: "32,768 parameters.",
        isCorrect: false,
      },
    ],
    explanation:
      "The adapter count is \\(4096\\times8 + 8\\times4096 = 65{,}536\\). The full matrix has \\(4096^2=16{,}777{,}216\\) entries, which shows why low-rank adaptation can be much cheaper. The 32,768 result counts just one adapter factor, while rank 8 is a latent width rather than the total number of learned scalars.",
  },
  // Canonical reuse; source membership does not change the original chapter or ID.
  referenceQuestion(
    stanfordCME295Lecture4TrainingQuestions,
    "cme295-lect4-q168",
  ),
  // Replaces retired ID cme295-2026-lect3-q18; new reasoning task.
  {
    id: "cme295-2026-lect3-q058",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Two low-rank adaptation (LoRA) configurations start from the same frozen Transformer. A adds adapters only to attention projections; B adds adapters to both attention and feed-forward projections, with the same rank at shared locations. Select all true statements.",
    options: [
      {
        text: "A can still change a feed-forward layer's output by changing the hidden input that reaches it.",
        isCorrect: true,
      },
      {
        text: "A leaves the feed-forward weight matrices unchanged.",
        isCorrect: true,
      },
      {
        text: "B introduces more trainable adapter matrices than A.",
        isCorrect: true,
      },
      {
        text: "Neither configuration requires updating the original frozen matrices to train its adapters.",
        isCorrect: true,
      },
    ],
    explanation:
      "Adapter placement controls which weight transformations can be changed directly. Adding feed-forward adapters increases trainable locations, while frozen feed-forward matrices in A still process potentially changed attention outputs. Freezing weights does not freeze activations; both configurations learn their added low-rank matrices while retaining the original matrices.",
  },
  // 2026-specific item; stable identity retained.
  {
    id: "cme295-2026-lect3-q19",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "Which statements correctly distinguish pretraining, supervised fine-tuning, and preference tuning?",
    options: [
      {
        text: "Supervised fine-tuning demonstrates desired input-output behavior.",
        isCorrect: true,
      },
      {
        text: "Preference tuning uses comparisons or rewards to favor certain responses.",
        isCorrect: true,
      },
      {
        text: "Pretraining learns broad token patterns from large corpora.",
        isCorrect: true,
      },
      {
        text: "Supervised fine-tuning and preference tuning use identical labels and objectives.",
        isCorrect: false,
      },
    ],
    explanation:
      "These stages use different forms of training signal even though they update related model weights. Next-token corpus learning, demonstrated target responses, and preferences over candidate behavior serve different roles; a preferred response pair is not simply the same label as a single supervised continuation.",
  },
  // 2026-specific item; stable identity retained.
  {
    id: "cme295-2026-lect3-q20",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "Which annotations each directly express a single pairwise winner/loser comparison between two responses to one shared prompt?",
    options: [
      {
        text: '"Choose B over C."',
        isCorrect: true,
      },
      {
        text: '"The ordering is B, then A, then C."',
        isCorrect: false,
      },
      {
        text: '"Response A scores 8 out of 10."',
        isCorrect: false,
      },
      {
        text: '"Response A is better than response B."',
        isCorrect: true,
      },
    ],
    explanation:
      "A pairwise label compares two candidates under the same prompt. An individual numeric score is pointwise and a complete ordering of several candidates is listwise; these formats can be related but are not identical annotations.",
  },
  // Canonical reuse; source membership does not change the original chapter or ID.
  referenceQuestion(
    stanfordCME295Lecture5PreferenceTuningQuestions,
    "cme295-lect5-q05",
  ),
  // Replaces retired ID cme295-2026-lect3-q22; new reasoning task.
  {
    id: "cme295-2026-lect3-q059",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Three feedback datasets are available: A uses responses sampled from the current student and labels from an AI judge; B uses responses from an older student with a different output distribution and human labels; C uses responses from the current student and human labels. Select all true classifications.",
    options: [
      {
        text: "AI labeling makes A off-policy even though the current student generated its responses.",
        isCorrect: false,
      },
      {
        text: "A is on-policy with respect to the current student's sampling distribution and uses proxy preference labels.",
        isCorrect: true,
      },
      {
        text: "B has human labels but is not sampled on-policy from the current student.",
        isCorrect: true,
      },
      {
        text: "Human labeling makes B on-policy regardless of which policy generated its responses.",
        isCorrect: false,
      },
    ],
    explanation:
      "On-policy describes the policy that generated the training trajectories, whereas human versus AI feedback describes the label source. Current-student samples make A and C on-policy in that sense; archived older-policy samples do not acquire that property from human labels. AI judges can supply proxy labels for on-policy samples, so these two classifications should be tracked separately.",
  },
  // Canonical reuse; source membership does not change the original chapter or ID.
  referenceQuestion(
    stanfordCME295Lecture5PreferenceTuningQuestions,
    "cme295-lect5-q09",
  ),
  // Canonical reuse; source membership does not change the original chapter or ID.
  referenceQuestion(
    stanfordCME295Lecture5PreferenceTuningQuestions,
    "cme295-lect5-q10",
  ),
  // Canonical reuse; source membership does not change the original chapter or ID.
  referenceQuestion(
    stanfordCME295Lecture5PreferenceTuningQuestions,
    "cme295-lect5-q11",
  ),
  // 2026-specific item; stable identity retained.
  {
    id: "cme295-2026-lect3-q26",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "A Bradley-Terry reward model assigns \\(r_w=2\\) and \\(r_l=0.5\\). What is the modeled probability \\(\\sigma(r_w-r_l)\\), approximately?",
    options: [
      {
        text: "1.500",
        isCorrect: false,
      },
      {
        text: "0.500",
        isCorrect: false,
      },
      {
        text: "0.818",
        isCorrect: true,
      },
      {
        text: "0.182",
        isCorrect: false,
      },
    ],
    explanation:
      "The reward difference is 1.5, and the sigmoid gives 1/(1+exp(-1.5)), approximately 0.818. Reversing the difference gives 0.182, equal rewards would give 0.5, and the raw reward gap is not itself a probability.",
  },
  // Canonical reuse; source membership does not change the original chapter or ID.
  referenceQuestion(
    stanfordCME295Lecture5PreferenceTuningQuestions,
    "cme295-lect5-q42",
  ),
  // Canonical reuse; source membership does not change the original chapter or ID.
  referenceQuestion(
    stanfordCME295Lecture5PreferenceTuningQuestions,
    "cme295-lect5-q14",
  ),
  // Canonical reuse; source membership does not change the original chapter or ID.
  referenceQuestion(
    stanfordCME295Lecture5PreferenceTuningQuestions,
    "cme295-lect5-q25",
  ),
  // 2026-specific item; stable identity retained.
  {
    id: "cme295-2026-lect3-q30",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "A Direct Preference Optimization (DPO) pair has these sequence log probabilities:\n\n| Response | Trainable policy | Frozen reference |\n| --- | ---: | ---: |\n| Chosen | -5 | -6 |\n| Rejected | -7 | -6.5 |\n\nWith \\(\\beta=0.1\\), what is the preference logit \\(\\beta[(\\log\\pi_\\theta(y_w)-\\log\\pi_{\\rm ref}(y_w))-(\\log\\pi_\\theta(y_l)-\\log\\pi_{\\rm ref}(y_l))]\\)?",
    options: [
      {
        text: "\\(0.20\\)",
        isCorrect: false,
      },
      {
        text: "\\(0.15\\)",
        isCorrect: true,
      },
      {
        text: "\\(-0.15\\)",
        isCorrect: false,
      },
      {
        text: "\\(0.05\\)",
        isCorrect: false,
      },
    ],
    explanation:
      "The chosen policy-minus-reference log probability is 1, while the rejected difference is -0.5. Subtracting them gives 1.5 and multiplying by beta gives 0.15; using the policy gap alone or reversing chosen and rejected changes the objective.",
  },
  // 2026-specific item; stable identity retained.
  {
    id: "cme295-2026-lect3-q31",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "Which tasks require combining supplied information rather than retrieving a stated fact?",
    options: [
      {
        text: "Copy an answer from an already completed answer field.",
        isCorrect: false,
      },
      {
        text: "Repeat a course code explicitly given in the prompt.",
        isCorrect: false,
      },
      {
        text: "Infer a bug from code and failing test traces.",
        isCorrect: true,
      },
      {
        text: "Compute an age from a birth year and an explicitly stated current year.",
        isCorrect: true,
      },
    ],
    explanation:
      "Arithmetic and debugging require transformations of the supplied information, unlike copying an explicitly supplied fact. This distinction motivates reasoning evaluations: success should depend on solving a problem rather than merely recognizing or repeating a stored or visible answer.",
  },
  // 2026-specific item; stable identity retained.
  {
    id: "cme295-2026-lect3-q32",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "Which mechanisms can make intermediate reasoning tokens useful when solving a difficult problem?",
    options: [
      {
        text: "Supply newly verified external facts without any observation or tool call.",
        isCorrect: false,
      },
      {
        text: "Provide additional autoregressive computation before the final answer.",
        isCorrect: true,
      },
      {
        text: "Decompose the task into simpler intermediate operations.",
        isCorrect: true,
      },
      {
        text: "Make earlier intermediate results available as context for later steps.",
        isCorrect: true,
      },
    ],
    explanation:
      "An intermediate trace gives the model more computation and a written state on which subsequent token predictions can condition. It can help organize a solution, but it does not create verified external information; intermediate statements may themselves be mistaken.",
  },
  // Canonical reuse; source membership does not change the original chapter or ID.
  referenceQuestion(
    stanfordCME295Lecture6ReasoningQuestions,
    "cme295-lect6-q04",
  ),
  // 2026-specific item; stable identity retained.
  {
    id: "cme295-2026-lect3-q34",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "Which setups provide verifiable outcome signals for reasoning training or evaluation?",
    options: [
      {
        text: "Check whether a proposed patch passes compilation and regression tests.",
        isCorrect: true,
      },
      {
        text: "Record whether an output used a confident writing style.",
        isCorrect: false,
      },
      {
        text: "Compare a final numerical answer with a known solution.",
        isCorrect: true,
      },
      {
        text: "Run a generated program against unit tests.",
        isCorrect: true,
      },
    ],
    explanation:
      "Execution and known-answer checks can provide objective signals about whether a task succeeded. Confidence is a style feature rather than a correctness check; even concrete verifiers are limited by the coverage of their tests or answer criteria.",
  },
  // Canonical reuse; source membership does not change the original chapter or ID.
  referenceQuestion(
    stanfordCME295Lecture6ReasoningQuestions,
    "cme295-lect6-q11",
  ),
  // Canonical reuse; source membership does not change the original chapter or ID.
  referenceQuestion(
    stanfordCME295Lecture6ReasoningQuestions,
    "cme295-lect6-q12",
  ),
  // Replaces retired ID cme295-2026-lect3-q37; new reasoning task.
  {
    id: "cme295-2026-lect3-q060",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "A Group Relative Policy Optimization (GRPO) training group contains four completions for the same prompt with rewards [0.8, 0.2, 0.5, 0.5]. Compare each completion with the group's mean reward before any clipping or reference penalty. Select all true statements.",
    options: [
      {
        text: "The mean reward is 0.5.",
        isCorrect: true,
      },
      {
        text: "The reward-0.8 completion is above the group baseline.",
        isCorrect: true,
      },
      {
        text: "Both reward-0.5 completions equal the group baseline.",
        isCorrect: true,
      },
      {
        text: "The reward-0.2 completion is below the group baseline.",
        isCorrect: true,
      },
    ],
    explanation:
      "The rewards sum to 2.0, so dividing by four gives a mean of 0.5. The differences from that baseline are +0.3, −0.3, 0, and 0. Group-relative credit compares sibling completions for one prompt; a reward's sign relative to zero is not the same as its sign relative to this baseline.",
  },
  // Replaces retired ID cme295-2026-lect3-q38; new reasoning task.
  {
    id: "cme295-2026-lect3-q061",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "Reasoning mode A solves a task with probability 0.4 and costs one compute unit per attempt. Mode B solves it with probability 0.6 and costs three units per attempt. Assume independent tasks and a budget of twelve units, with no answer selection or verification step. Select all true comparisons.",
    options: [
      {
        text: "A yields 0.4 expected correct answers per compute unit, compared with 0.2 for B.",
        isCorrect: true,
      },
      {
        text: "B is more accurate per attempt but gives fewer expected correct answers under this total budget.",
        isCorrect: true,
      },
      {
        text: "The expected count of correct answers is 4.8 for A and 2.4 for B.",
        isCorrect: true,
      },
      {
        text: "The budget supports twelve A attempts or four B attempts.",
        isCorrect: true,
      },
    ],
    explanation:
      "Expected correct counts are attempts times success probability: 12 × 0.4 = 4.8 and 4 × 0.6 = 2.4. B spends more compute on each attempt and improves per-attempt accuracy, yet its accuracy gain is smaller than its threefold cost. This comparison concerns total expected correct answers across tasks, not the probability that at least one sampled answer is correct on a single task.",
  },
  // Replaces retired ID cme295-2026-lect3-q39; new reasoning task.
  {
    id: "cme295-2026-lect3-q062",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "To collect on-policy distillation trajectories for the current student, which policy should generate the continuation tokens that determine the prefixes where the teacher is queried?",
    options: [
      {
        text: "The fixed teacher policy.",
        isCorrect: false,
      },
      {
        text: "The current student policy.",
        isCorrect: true,
      },
      {
        text: "An archived version of the student policy.",
        isCorrect: false,
      },
      {
        text: "A human demonstration policy represented by stored reference continuations.",
        isCorrect: false,
      },
    ],
    explanation:
      "The on-policy setup visits prefixes produced by the student that is being trained, then queries the teacher at those states. Teacher-generated continuations or fixed reference traces instead determine a different prefix distribution. An archived student is also a different policy when its output probabilities differ from the current student.",
  },
  // Replaces retired ID cme295-2026-lect3-q40; new reasoning task.
  {
    id: "cme295-2026-lect3-q063",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "As a toy illustration of dense teacher feedback, score three sampled tokens by \\(-\\ln p_T(\\text{token}\\mid\\text{visited prefix})\\). Their teacher probabilities are \\((0.9,0.1,0.8)\\); the final completion receives a single outcome reward of 1. Select all true statements about these signals.",
    options: [
      {
        text: "Local teacher feedback can distinguish these positions even though they share the same final outcome.",
        isCorrect: true,
      },
      {
        text: "The terminal reward alone supplies no separate numerical evaluation of the three individual tokens.",
        isCorrect: true,
      },
      {
        text: "The token penalties are approximately \\((0.105,2.303,0.223)\\).",
        isCorrect: true,
      },
      {
        text: "The second sampled token receives the largest local teacher penalty.",
        isCorrect: true,
      },
    ],
    explanation:
      "Taking negative natural logs gives approximately 0.105, 2.303, and 0.223, making the teacher's low-probability second token the largest local penalty. A single successful terminal reward evaluates the completion as a whole and does not identify which intermediate token the teacher disliked. This toy sampled-token score illustrates feedback granularity; it is not the full distributional distillation objective.",
  },
  // Replaces retired ID cme295-2026-lect3-q41; new reasoning task.
  {
    id: "cme295-2026-lect3-q064",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "A self-distillation teacher and student start from identical weights; the teacher is frozen while the student is trained. Initially, at the same student prefix, the model assigns probability 0.3 to a useful next token without a hint and 0.8 with a hint. Teacher queries include the hint; student inputs do not. Select all true statements.",
    options: [
      {
        text: "The student can be trained toward the hint-conditioned teacher target without receiving the hint as an input.",
        isCorrect: true,
      },
      {
        text: "The teacher probability 0.8 alone demonstrates that its weights were updated during that query.",
        isCorrect: false,
      },
      {
        text: "For a fixed student policy, student-generated continuations determine the visited-prefix distribution even when teacher queries receive extra context.",
        isCorrect: true,
      },
      {
        text: "Different conditioning information can produce different target probabilities even when the starting weights match.",
        isCorrect: true,
      },
    ],
    explanation:
      "A model conditions on its inputs as well as its weights, so adding a hint can change a probability from 0.3 to 0.8 without any parameter update. Distillation can use that more informed distribution as a target for the unhinted student. The generator of the continuations determines which prefixes are visited; extra information supplied only to the teacher does not by itself change that sampling policy.",
  },
  // 2026-specific item; stable identity retained.
  {
    id: "cme295-2026-lect3-q42",
    chapter: 3,
    difficulty: "easy",
    prompt:
      "A system has specialized teachers for mathematics and software engineering. Which choices fit multi-teacher on-policy distillation?",
    options: [
      {
        text: "Treat the teacher identity as a replacement for token-level supervision.",
        isCorrect: false,
      },
      {
        text: "Transfer different teachers' capabilities into one student.",
        isCorrect: true,
      },
      {
        text: "Use domain-appropriate teacher guidance on student-generated trajectories.",
        isCorrect: true,
      },
      {
        text: "Route the prompt to a teacher and deploy that teacher instead of training a student.",
        isCorrect: false,
      },
    ],
    explanation:
      "Multi-teacher distillation uses specialized supervision to integrate capabilities into a shared learner. Teacher routing at serving time is a different system design, and selecting a teacher does not by itself supply the distribution-matching signal used to update the student.",
  },
  // Replaces retired ID cme295-2026-lect3-q43; new reasoning task.
  {
    id: "cme295-2026-lect3-q065",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "At the student's visited prefix solve 2+2, the teacher's next-token distribution over [4, 5] is [0.9, 0.1]. At a reference trace's prefix solve 2+3, its distribution is [0.2, 0.8]. Both prefixes have the same token count. Select all true statements for on-policy distillation of that student state.",
    options: [
      {
        text: "Averaging the two distributions to [0.55, 0.45] is required to align equal-length prefixes.",
        isCorrect: false,
      },
      {
        text: "The relevant teacher distribution is [0.2, 0.8], because reference traces determine the target independently of student context.",
        isCorrect: false,
      },
      {
        text: "Matching prefix lengths does not make the reference-prefix distribution a target for the same conditional state.",
        isCorrect: true,
      },
      {
        text: "The relevant teacher distribution is [0.9, 0.1], obtained at the student's actual prefix.",
        isCorrect: true,
      },
    ],
    explanation:
      "Next-token probabilities are conditional on the prefix's content, not merely its length or time index. The student visited solve 2+2, so the teacher should be queried on that context and supplies [0.9, 0.1]. A reference trace for a different arithmetic state provides a different conditional; substituting or averaging it changes the supervision rather than aligning the visited state.",
  },
  // 2026-specific item; stable identity retained.
  {
    id: "cme295-2026-lect3-q44",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Two documents, including boundary tokens, have lengths 5 and 3. A concatenate-and-chunk pretraining pipeline uses blocks of length 4. How can it package their combined 8-token stream?",
    options: [
      {
        text: "Three full blocks, because every document needs a dedicated padded block.",
        isCorrect: false,
      },
      {
        text: "Two full blocks, with the first document potentially split across a boundary.",
        isCorrect: true,
      },
      {
        text: "Two full blocks, with each document necessarily remaining intact.",
        isCorrect: false,
      },
      {
        text: "One full block and four tokens discarded because the first document is too long.",
        isCorrect: false,
      },
    ],
    explanation:
      "Concatenating produces a continuous eight-token stream that can be cut into two four-token blocks. Chunk boundaries need not match document boundaries, so the five-token document can be split; document-preserving packing is a different choice and may use padding.",
  },
  // 2026-specific item; stable identity retained.
  {
    id: "cme295-2026-lect3-q45",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Complete training examples, including their boundary tokens, have lengths 6, 5, and 3. They must remain intact and fit in blocks of capacity 8. What is the minimum block count and total padding?",
    options: [
      {
        text: "Three blocks and ten padding tokens.",
        isCorrect: false,
      },
      {
        text: "One block and six padding tokens.",
        isCorrect: false,
      },
      {
        text: "Two blocks and two padding tokens.",
        isCorrect: true,
      },
      {
        text: "Two blocks and zero padding tokens.",
        isCorrect: false,
      },
    ],
    explanation:
      "The 5-token and 3-token examples fill one block, while the 6-token example leaves two unused positions in another block. Fourteen content tokens require at least two 8-token blocks; preserving complete examples is feasible here without a third block or splitting a document.",
  },
  // Replaces retired ID cme295-2026-lect3-q46; new reasoning task.
  {
    id: "cme295-2026-lect3-q066",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "A generator emits a rare construction with probability \\(10^{-4}\\). A synthetic corpus contains 1,000 independent samples, and a simplified next generator assigns zero probability to constructions absent from that corpus. Select all true statements about this toy recursive-training risk.",
    options: [
      {
        text: "If it is absent, the simplified fitted generator assigns it zero probability.",
        isCorrect: true,
      },
      {
        text: "Sampling only from that zero-probability fitted generator will not recover the missing construction.",
        isCorrect: true,
      },
      {
        text: "The chance of seeing the rare construction at least once is approximately \\(0.905\\).",
        isCorrect: false,
      },
      {
        text: "The chance that the rare construction is absent is \\((1-10^{-4})^{1000}\\approx0.905\\).",
        isCorrect: true,
      },
    ],
    explanation:
      "The no-occurrence probability is about 0.905, leaving only about 0.095 chance of at least one occurrence in this corpus. Under the explicitly simplified fitting rule, absence becomes zero support and later samples from that generator alone cannot restore it. Real training need not follow this rule exactly, but the example isolates how finite synthetic sampling can lose rare modes.",
  },
  // Replaces retired ID cme295-2026-lect3-q47; new reasoning task.
  {
    id: "cme295-2026-lect3-q067",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "A human corpus contains 100 examples, 20 of them from a rare category. A synthetic corpus contains 100 examples, 10 rare. Sampling is uniform within each corpus. Select all true mixture calculations.",
    options: [
      {
        text: "Pooling the two corpora equally gives a 15% rare-category rate.",
        isCorrect: true,
      },
      {
        text: "Sampling 80% of examples from the human corpus and 20% from the synthetic corpus gives an 18% rare-category rate.",
        isCorrect: true,
      },
      {
        text: "The 18% mixture uses a human-to-synthetic sampling ratio of 4:1.",
        isCorrect: true,
      },
      {
        text: "Equal corpus weights halve the human corpus's share from 100% to 50% without halving its within-corpus rare-category rate.",
        isCorrect: true,
      },
    ],
    explanation:
      "Equal pooling gives (20 + 10)/200 = 15% rare examples. With human weight w, the mixture rate is 0.20w + 0.10(1−w), so reaching 18% requires w = 0.8 and sampling odds 4:1. Mixture weights change how often a corpus is chosen; they do not change the 20% category rate conditional on choosing a human example.",
  },
  // Replaces retired ID cme295-2026-lect3-q48; new reasoning task.
  {
    id: "cme295-2026-lect3-q068",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "After math fine-tuning, a model's loss on a fixed held-out coding set rises from 0.8 to 1.2, while its loss on a fixed math set falls from 1.1 to 0.6. Lower loss is better. Select all conclusions supported by these measurements.",
    options: [
      {
        text: "Performance on the measured math set improved.",
        isCorrect: true,
      },
      {
        text: "A lower average of the two losses would rule out forgetting on either individual skill.",
        isCorrect: false,
      },
      {
        text: "Performance on the measured coding set regressed.",
        isCorrect: true,
      },
      {
        text: "The coding result is evidence of forgetting or interference in a previously measured skill.",
        isCorrect: true,
      },
    ],
    explanation:
      "The coding loss increase is a regression on that unchanged evaluation set, while the math loss decrease is an improvement. Their average falls from 0.95 to 0.90, yet that aggregate hides the coding regression. Skill-specific evaluations therefore matter when assessing whether later training preserves earlier abilities; a better average does not imply every capability was retained.",
  },
  // Replaces retired ID cme295-2026-lect3-q49; new reasoning task.
  {
    id: "cme295-2026-lect3-q069",
    chapter: 3,
    difficulty: "hard",
    prompt:
      "A later training stage combines a new-task loss with a retention loss from a frozen earlier checkpoint: \\(L=L_{\\rm new}+\\lambda L_{\\rm retain}\\). At one scalar parameter, the gradients are \\(\\partial L_{\\rm new}/\\partial\\theta=-2\\) and \\(\\partial L_{\\rm retain}/\\partial\\theta=1\\). Select all true statements.",
    options: [
      {
        text: "With \\(\\lambda=3\\), the combined gradient is \\(1\\).",
        isCorrect: true,
      },
      {
        text: "The earlier checkpoint can supply a fixed retention target while the current student is updated.",
        isCorrect: true,
      },
      {
        text: "With \\(\\lambda=0.5\\), the combined gradient is \\(-1.5\\).",
        isCorrect: true,
      },
      {
        text: "A gradient-descent step moves this parameter in opposite directions in those two cases.",
        isCorrect: true,
      },
    ],
    explanation:
      "The gradients add as −2 + λ, giving −1.5 and +1 for the two retention weights. Gradient descent subtracts the gradient, so one setting increases the parameter and the other decreases it. Keeping the earlier checkpoint fixed preserves a target for retention while the student learns; the example shows how preservation pressure can compete with new-task improvement.",
  },
  // Replaces retired ID cme295-2026-lect3-q50; new reasoning task.
  {
    id: "cme295-2026-lect3-q070",
    chapter: 3,
    difficulty: "medium",
    prompt:
      "Consider four training batches: P predicts each of five ordinary text tokens; S has three prompt tokens and two response tokens with response-only loss; R samples four completions and obtains one terminal scalar reward per completion; D queries teacher distributions at four student-visited prefixes. Select all correct signal counts.",
    options: [
      {
        text: "R has four terminal scalar reward observations.",
        isCorrect: true,
      },
      {
        text: "P has five next-token targets.",
        isCorrect: true,
      },
      {
        text: "S has two response-token loss targets.",
        isCorrect: true,
      },
      {
        text: "D has four conditional teacher distributions.",
        isCorrect: true,
      },
    ],
    explanation:
      "Each batch's signal must be counted at the level specified by its objective. Full-text prediction uses five token targets, response masking leaves two supervised targets, and four sampled completions supply four terminal rewards. Querying a teacher at four visited prefixes supplies four distributions, whose granularity differs from a single scalar completion outcome.",
  },
];
