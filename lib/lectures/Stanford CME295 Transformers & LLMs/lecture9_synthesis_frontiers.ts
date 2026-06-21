import { Question } from "../../quiz";

type Lecture9Difficulty = "easy" | "medium" | "hard";
type OptionSeed = readonly [text: string, isCorrect: boolean];
type AssertionReasonChoice = 1 | 2 | 3 | 4 | 5;

const assertionReasonOptionTexts = [
  "Assertion is true, Reason is false.",
  "Assertion is false, Reason is true.",
  "Both are false.",
  "Both are true, and the Reason is the correct explanation of the Assertion.",
  "Both are true, but the Reason is NOT the correct explanation of the Assertion.",
] as const;

function makeQuestion(
  id: string,
  difficulty: Lecture9Difficulty,
  prompt: string,
  optionSeeds: readonly OptionSeed[],
  explanation: string,
): Question {
  if (optionSeeds.length !== 4) {
    throw new Error(`CME295 Lecture 9 question ${id} needs 4 options.`);
  }

  return {
    id,
    chapter: 9,
    difficulty,
    prompt,
    options: optionSeeds.map(([text, isCorrect]) => ({ text, isCorrect })),
    explanation,
  };
}

function makeAssertionReasonQuestion(
  id: string,
  difficulty: Lecture9Difficulty,
  prompt: string,
  correctChoice: AssertionReasonChoice,
  explanation: string,
): Question {
  return {
    id,
    chapter: 9,
    difficulty,
    type: "assertion-reason",
    prompt,
    options: assertionReasonOptionTexts.map((text, index) => ({
      text,
      isCorrect: index + 1 === correctChoice,
    })),
    explanation,
  };
}

export const stanfordCME295Lecture9SynthesisFrontiersQuestions: Question[] = [
  makeQuestion(
    "cme295-lect9-q01",
    "easy",
    "A language model pipeline begins by converting raw text into tokens. Which statements correctly describe subword tokenization and why it became a common default?",
    [
      [
        "It can split rare words into reusable pieces while keeping frequent words closer to whole units.",
        true,
      ],
      [
        "It removes the need to learn vector representations for tokens.",
        false,
      ],
      [
        "It can reuse roots, prefixes, suffixes, or other repeated pieces across related words.",
        true,
      ],
      [
        "Its segmentation choices affect vocabulary size, sequence length, and how unknown words are handled.",
        true,
      ],
    ],
    "Subword tokenization is a compromise between word-level and character-level representations. The model still needs embeddings for the resulting token IDs, and the chosen segmentation changes both the computation length and how well rare or novel words are represented.",
  ),
  makeQuestion(
    "cme295-lect9-q02",
    "easy",
    "Which comparisons between older text representations and transformer self-attention are accurate?",
    [
      [
        "Word2vec-style embeddings are learned from proxy prediction tasks but assign the same stored vector to a word across contexts.",
        true,
      ],
      [
        "A Recurrent Neural Network (RNN) processes tokens sequentially and carries information through a hidden state.",
        true,
      ],
      [
        "Word2vec is context-aware in the same way a transformer layer is because each sentence recomputes a new word vector.",
        false,
      ],
      [
        "An RNN creates a direct content-based attention link between every pair of tokens in one layer.",
        false,
      ],
    ],
    "Static embeddings and recurrent sequence models were important steps toward modern language modeling, but they have different limitations. Transformers made contextual token interactions much more direct by computing attention weights between token representations instead of relying on a single recurrent state to carry distant information.",
  ),
  makeQuestion(
    "cme295-lect9-q03",
    "hard",
    "A self-attention head receives query, key, and value matrices \\(Q\\), \\(K\\), and \\(V\\), with key dimension \\(d_k\\). Which expression matches scaled dot-product attention?",
    [
      [
        "\\(\\operatorname{softmax}(QK^T / \\sqrt{d_k})V\\), because scaled query-key scores weight the value vectors.",
        true,
      ],
      [
        "\\(\\operatorname{softmax}(QV^T / \\sqrt{d_k})K\\), because query-value scores should weight key vectors.",
        false,
      ],
      [
        "\\(K\\operatorname{softmax}(Q^TV / d_k)\\), because keys should be multiplied after the probability matrix.",
        false,
      ],
      [
        "\\(\\operatorname{softmax}(VK^T / \\sqrt{d_k})Q\\), because value-key scores choose query vectors.",
        false,
      ],
    ],
    "Scaled dot-product attention scores how strongly each query should attend to each key, normalizes those scores with a softmax, and then forms a weighted combination of values. The distractor formulas swap the roles of keys, queries, and values, which changes the mechanism rather than merely rewriting the same equation.",
  ),
  makeAssertionReasonQuestion(
    "cme295-lect9-q04",
    "medium",
    "Assertion: Self-attention mitigates the long-range dependency problem that made simple recurrent sequence processing difficult.\n\nReason: Query-key scores let a token place weight directly on distant token representations rather than relying only on a hidden state passed through many sequential steps.",
    4,
    "The assertion is true because attention gives distant positions a direct content-based route to influence one another. The reason is also true and explains the assertion: the direct query-key-value computation is the mechanism that avoids forcing all long-range information through a recurrent chain.",
  ),
  makeQuestion(
    "cme295-lect9-q05",
    "easy",
    "In an encoder-decoder transformer used for translation, which components or data flows are part of the standard architecture?",
    [
      [
        "The encoder builds contextual representations of the source-language tokens.",
        true,
      ],
      [
        "The decoder uses masked self-attention over the target prefix produced so far.",
        true,
      ],
      [
        "Decoder cross-attention can use source representations as keys and values while decoding target tokens.",
        true,
      ],
      [
        "Generated target tokens are fed back into the decoder until an end-of-sequence condition is reached.",
        true,
      ],
    ],
    "The encoder-decoder transformer separates source understanding from target generation while connecting them through cross-attention. Masking prevents the decoder from looking at future target tokens during training or generation, while autoregressive feedback lets the model build the translation one token at a time.",
  ),
  makeQuestion(
    "cme295-lect9-q06",
    "medium",
    "Rotary Position Embedding (RoPE) is used in many modern transformer models. Which statements capture its role?",
    [
      [
        "It rotates query and key components so attention scores can encode relative position information.",
        true,
      ],
      [
        "It puts positional structure inside the self-attention computation rather than only adding a separate absolute vector to token embeddings.",
        true,
      ],
      [
        "It makes the attention interaction depend on relative distance between positions.",
        true,
      ],
      [
        "It is exactly the original absolute learned position embedding table from the first transformer paper.",
        false,
      ],
    ],
    "RoPE changes the query and key representations before their dot product, which makes relative position visible to attention. Absolute learned position embeddings instead add a position-specific vector to token embeddings, so they encode a different kind of positional signal.",
  ),
  makeQuestion(
    "cme295-lect9-q07",
    "medium",
    "Grouped-Query Attention (GQA) and related head-sharing designs are mainly used for which purposes?",
    [
      [
        "They keep multiple query heads while sharing key and value projections across groups of heads.",
        true,
      ],
      [
        "They reduce key-value cache size and memory bandwidth pressure during inference.",
        true,
      ],
      [
        "They force every query head to have separate key and value matrices exactly as in ordinary multi-head attention.",
        false,
      ],
      [
        "They change the tokenizer so fewer subword tokens need to be embedded.",
        false,
      ],
    ],
    "GQA is an attention-architecture change, not a tokenization change. It preserves much of the flexibility of multiple query heads while reducing how many key and value projections must be stored and read during generation.",
  ),
  makeQuestion(
    "cme295-lect9-q08",
    "easy",
    "Which model-family distinctions are accurate?",
    [
      [
        "Encoder-only models such as BERT are well suited to contextual embeddings and classification-style tasks.",
        true,
      ],
      [
        "Decoder-only models such as GPT-style language models are naturally aligned with autoregressive text generation.",
        true,
      ],
      [
        "Encoder-decoder models such as T5 support text-in/text-out tasks by encoding an input and decoding an output.",
        true,
      ],
      [
        "Encoder-only models must include decoder cross-attention in order to produce a `[CLS]` embedding.",
        false,
      ],
    ],
    "The three transformer families reuse the same core building blocks in different ways. BERT-style encoders produce bidirectional contextual representations, GPT-style decoders generate from a prefix, and encoder-decoder systems combine source encoding with target decoding.",
  ),
  makeQuestion(
    "cme295-lect9-q09",
    "medium",
    "Which statements describe sparse mixture-of-experts designs in large language models?",
    [
      [
        "Experts are commonly implemented as separate feed-forward networks that replace or augment the usual feed-forward sublayer inside transformer blocks.",
        true,
      ],
      [
        "A router or gate selects a small subset of experts for each token representation.",
        true,
      ],
      [
        "Sparse activation can increase total parameter count without using every parameter for every token.",
        true,
      ],
      [
        "Expert placement, load balancing, and routing behavior matter for efficient training and serving.",
        true,
      ],
    ],
    "Sparse MoE models separate total capacity from active compute by routing each token through selected experts. That design creates new engineering and learning issues: routers can collapse, experts can become imbalanced, and hardware placement affects whether sparse activation actually helps.",
  ),
  makeAssertionReasonQuestion(
    "cme295-lect9-q10",
    "medium",
    "Assertion: Sparse mixture-of-experts can let a model contain many total parameters without activating every parameter on every token.\n\nReason: In many transformer mixture-of-experts designs, the experts replace or augment feed-forward sublayers.",
    5,
    "The assertion is true because sparse routing activates a subset of experts per token. The reason is also true, but it names where experts often sit in the transformer block; it does not by itself explain sparsity, which comes from the router selecting only some experts.",
  ),
  makeQuestion(
    "cme295-lect9-q11",
    "easy",
    "Which statements correctly describe next-token decoding and temperature?",
    [
      [
        "Greedy decoding chooses the highest-probability next token at each step.",
        true,
      ],
      [
        "Sampling draws from the model's next-token distribution and can produce varied completions.",
        true,
      ],
      [
        "Lower temperature sharpens the distribution and tends to make generation more deterministic.",
        true,
      ],
      [
        "Higher temperature directly verifies whether the generated content is factually correct.",
        false,
      ],
    ],
    "Temperature changes the shape of the sampling distribution, so it affects diversity and determinism. It is not an evaluation method; a more random answer can still be wrong, and a deterministic answer can still be unsupported.",
  ),
  makeQuestion(
    "cme295-lect9-q12",
    "hard",
    "A compute-optimal training rule of thumb says a model should be trained on about 20 tokens per parameter. For a 100 billion parameter model, which token count follows?",
    [
      ["\\(100\\text{B} \\times 20 = 2\\text{T}\\) tokens.", true],
      ["\\(100\\text{B} / 20 = 5\\text{B}\\) tokens.", false],
      ["\\(100\\text{B} + 20\\text{B} = 120\\text{B}\\) tokens.", false],
      ["\\(100\\text{B} \\times 200 = 20\\text{T}\\) tokens.", false],
    ],
    "The rule multiplies parameter count by a token-per-parameter factor, so 100 billion parameters maps to about 2 trillion training tokens. The other calculations divide, add, or use the wrong multiplier, which would change the intended model-data balance.",
  ),
  makeQuestion(
    "cme295-lect9-q13",
    "hard",
    "Which statements about FlashAttention and GPU memory hierarchy are correct?",
    [
      [
        "FlashAttention is an exact attention method rather than an approximation to different attention values.",
        true,
      ],
      [
        "FlashAttention improves speed by reducing reads and writes to high-bandwidth memory (HBM) through tiling into smaller fast memory.",
        true,
      ],
      [
        "FlashAttention can recompute intermediate quantities during backpropagation when recomputation is cheaper than storing and rereading them.",
        true,
      ],
      [
        "FlashAttention would lose its memory advantage if it materialized the full attention matrix in HBM for every layer and head.",
        false,
      ],
    ],
    "The central idea is IO awareness: memory movement can dominate runtime even when the mathematical operation is unchanged. FlashAttention avoids writing the full attention matrix to slower memory and can trade extra arithmetic for less memory traffic.",
  ),
  makeQuestion(
    "cme295-lect9-q14",
    "medium",
    "A training system needs to use multiple accelerators. Which distinctions between data parallelism and model parallelism are correct?",
    [
      [
        "Data parallelism splits batches across devices that each hold a model replica and later synchronize updates.",
        true,
      ],
      [
        "Model parallelism splits parts of the model or a forward pass across devices when one device is not enough or throughput needs improvement.",
        true,
      ],
      [
        "Data parallelism removes the need to aggregate gradients or equivalent update information across workers.",
        false,
      ],
      [
        "Model parallelism is a tokenizer optimization that happens before token embeddings are looked up.",
        false,
      ],
    ],
    "Data parallelism and model parallelism attack different bottlenecks. Splitting examples across replicas still requires coordination, while splitting the model helps when activations, parameters, or computations cannot be handled efficiently on one device.",
  ),
  makeQuestion(
    "cme295-lect9-q15",
    "easy",
    "Which stages belong in the common modern language-model training pipeline?",
    [
      [
        "Pretraining teaches broad language and code structure through next-token prediction on large corpora.",
        true,
      ],
      [
        "Supervised fine-tuning teaches the model to respond to desired input-output task formats.",
        true,
      ],
      [
        "Preference tuning uses comparative feedback so the model learns which acceptable-looking outputs are preferred.",
        true,
      ],
      [
        "Reference-model or base-model constraints can help prevent preference optimization from drifting too far.",
        true,
      ],
    ],
    "Pretraining, SFT, and preference tuning solve different problems. A pretrained model can autocomplete text, SFT makes it follow tasks, and preference tuning injects comparative signals about helpfulness, safety, tone, and other human-valued dimensions.",
  ),
  makeQuestion(
    "cme295-lect9-q16",
    "medium",
    "Which statements about pairwise preference data and reward modeling are correct?",
    [
      [
        "Pairwise labels can train a reward model by comparing a preferred response with a less preferred response.",
        true,
      ],
      [
        "A Bradley-Terry-style formulation can turn a difference between reward scores into a preference probability.",
        true,
      ],
      [
        "A reward model trained on comparisons can later score one candidate completion at a time.",
        true,
      ],
      [
        "Reward modeling cannot use human comparisons unless every prompt has one exact ground-truth answer string.",
        false,
      ],
    ],
    "Preference tuning often works with comparisons because it can be easier to choose the better of two outputs than to write a perfect answer. The reward model learns a scoring function from those comparisons and can then be used to guide or rank new completions.",
  ),
  makeQuestion(
    "cme295-lect9-q17",
    "hard",
    "In reinforcement learning from human feedback (RLHF), which statements describe the policy-optimization stage?",
    [
      [
        "The language model samples a completion, and a reward model scores that completion for the prompt.",
        true,
      ],
      [
        "The optimization objective usually balances higher reward against staying close to a reference or base model.",
        true,
      ],
      [
        "The reward model must be retrained after every generated token before the policy can continue decoding.",
        false,
      ],
      [
        "The KL-style constraint is meant to push the policy as far away from the SFT model as possible.",
        false,
      ],
    ],
    "RLHF treats the language model as a policy over next tokens or completions and uses the reward model as a learned preference signal. The reference constraint matters because learned rewards are imperfect; without regularization, the policy can exploit reward-model blind spots.",
  ),
  makeAssertionReasonQuestion(
    "cme295-lect9-q18",
    "medium",
    "Assertion: A policy optimized only against an imperfect reward model can learn outputs that score well but do not match what humans actually want.\n\nReason: A reward model is an approximation of preference data and can contain exploitable blind spots.",
    4,
    "The assertion is true because reward hacking is possible when the optimized signal is not the real human objective. The reason is also true and explains the assertion: blind spots in the reward model create a target the policy can exploit without improving genuine usefulness or safety.",
  ),
  makeQuestion(
    "cme295-lect9-q19",
    "hard",
    "Which comparisons among PPO-style RLHF, Best-of-N, and Direct Preference Optimization (DPO) are accurate?",
    [
      [
        "DPO uses preference pairs and a current-policy versus reference-policy log-probability contrast instead of training a separate reward model and running a PPO loop.",
        true,
      ],
      [
        "Best-of-N avoids policy-gradient training but shifts extra cost and latency to inference by generating and scoring multiple candidates.",
        true,
      ],
      [
        "PPO-style RLHF can be strong when tuned well, but it requires more moving parts such as policy, value, reward, and reference models.",
        true,
      ],
      [
        "DPO is the same algorithm as PPO because it requires environment rollouts and a learned value model for every update.",
        false,
      ],
    ],
    "These methods use preference information in different places. PPO-style RLHF performs an RL update against reward-model feedback, Best-of-N chooses among candidates at inference time, and DPO turns the preference objective into a supervised-style update involving a reference policy.",
  ),
  makeQuestion(
    "cme295-lect9-q20",
    "easy",
    "Which statements describe reasoning models and test-time scaling?",
    [
      [
        "Chain-of-thought prompting or training encourages intermediate steps before a final answer.",
        true,
      ],
      [
        "Test-time scaling spends more inference tokens, attempts, or thinking budget to improve solution quality.",
        true,
      ],
      [
        "Verifiable rewards are especially useful for domains such as math and code where answers can be checked.",
        true,
      ],
      [
        "Reasoning traces can be hidden or summarized even when the model internally uses longer intermediate computation.",
        true,
      ],
    ],
    "Reasoning models are not just larger autocomplete systems; they spend computation on intermediate problem-solving behavior. The tradeoff is that more thinking, samples, or verification can improve difficult tasks while increasing latency and cost.",
  ),
  makeQuestion(
    "cme295-lect9-q21",
    "hard",
    "A model has an independent \\(p=0.4\\) chance of solving a problem on each sampled attempt. Which expression gives \\(Pass@3\\), the probability that at least one of three attempts succeeds?",
    [
      ["\\(1-(1-0.4)^3 = 1-0.6^3 = 0.784\\).", true],
      ["\\(0.4^3 = 0.064\\).", false],
      ["\\(3 \\times 0.4 = 1.2\\).", false],
      ["\\((1-0.4)^3 = 0.216\\).", false],
    ],
    "Pass@k asks whether at least one sampled attempt succeeds, so the easiest computation is the complement of all attempts failing. Multiplying the success probabilities gives the stricter probability that every attempt succeeds, and summing probabilities without a cap is not a valid probability calculation here.",
  ),
  makeQuestion(
    "cme295-lect9-q22",
    "hard",
    "Which statements about Group Relative Policy Optimization (GRPO) are correct?",
    [
      [
        "It samples a group of completions for the same prompt and compares their rewards within that group.",
        true,
      ],
      [
        "It can avoid a separate learned value model by using normalized group-relative rewards as an advantage-like signal.",
        true,
      ],
      [
        "It can still include policy-ratio clipping and a KL-style pressure toward a reference model.",
        true,
      ],
      [
        "It requires a human to label every generated token before any reward can be computed.",
        false,
      ],
    ],
    "GRPO keeps the broad policy-optimization idea while changing how the advantage signal is estimated. Group-relative reward normalization is useful when several completions for the same prompt can be scored by verifiable or learned rewards.",
  ),
  makeQuestion(
    "cme295-lect9-q23",
    "medium",
    "Which statements fit the reasoning-model training and distillation patterns used in recent systems?",
    [
      [
        "A reasoning RL stage can use verifiable answer rewards and formatting rewards even when human-written reasoning traces are scarce.",
        true,
      ],
      [
        "Reasoning distillation can train smaller models on traces or outputs generated by a stronger reasoning teacher.",
        true,
      ],
      [
        "A distilled student must have the same number of parameters as the teacher for distillation to count.",
        false,
      ],
      [
        "A reasoning model is useful only when its full hidden chain is exposed verbatim to the user.",
        false,
      ],
    ],
    "Reasoning systems often combine verifiable rewards, formatting constraints, curated data, and teacher-generated traces. Distillation is valuable precisely because a smaller student can learn useful behavior from a larger teacher without matching its full size or exposing every internal step.",
  ),
  makeAssertionReasonQuestion(
    "cme295-lect9-q24",
    "hard",
    "Assertion: GRPO requires training a separate value model in order to estimate advantages.\n\nReason: GRPO can use rewards normalized across multiple completions sampled for the same prompt as a relative advantage signal.",
    2,
    "The assertion is false because avoiding a separate value model is one of GRPO's key simplifications relative to PPO-style setups. The reason is true: group-relative reward normalization supplies the comparison signal that would otherwise require a value-model baseline.",
  ),
  makeQuestion(
    "cme295-lect9-q25",
    "easy",
    "Which motivations correctly explain retrieval-augmented generation (RAG)?",
    [
      [
        "Model weights have a knowledge cutoff and are difficult to edit for changing facts.",
        true,
      ],
      [
        "Long prompts can be expensive or distracting if irrelevant context is included.",
        true,
      ],
      [
        "Retrieval can fetch a smaller set of relevant external documents or chunks for the model to use.",
        true,
      ],
      [
        "The final answer can be generated from a prompt augmented with retrieved evidence.",
        true,
      ],
    ],
    "RAG works around fixed model knowledge by moving relevant external information into the prompt context. It does not magically update weights, but it can make answers more current, grounded, and efficient when retrieval quality is good.",
  ),
  makeQuestion(
    "cme295-lect9-q26",
    "medium",
    "A team is building a retrieval system for a policy-answering assistant. Which design choices match the RAG pipeline?",
    [
      [
        "Choose chunk size and overlap carefully because they affect what evidence can be retrieved together.",
        true,
      ],
      [
        "Use semantic retrieval, keyword retrieval such as BM25, or a hybrid of both depending on the query/document mismatch.",
        true,
      ],
      [
        "Use a cross-encoder reranker on a smaller candidate set when better top-ranked ordering is worth extra cost.",
        true,
      ],
      [
        "Skip retrieval evaluation because reranking automatically proves the generated answer is faithful.",
        false,
      ],
    ],
    "RAG quality depends on both the knowledge-base construction and the retrieval/reranking pipeline. Reranking can improve ordering, but it still needs relevance judgments and answer-level checks because a good-looking answer can misuse or omit evidence.",
  ),
  makeQuestion(
    "cme295-lect9-q27",
    "hard",
    "Which retrieval-evaluation statements are accurate?",
    [
      [
        "Reciprocal Rank at k gives more credit when the first relevant result appears earlier in the ranked list.",
        true,
      ],
      [
        "Normalized Discounted Cumulative Gain at k can handle graded relevance and discounts useful results that appear lower in the list.",
        true,
      ],
      [
        "Retrieval metrics directly measure whether the final generated prose is coherent, safe, and well written.",
        false,
      ],
      [
        "Retrieval quality is identical to binary answer correctness, so the retrieved context does not need separate inspection.",
        false,
      ],
    ],
    "Retrieval metrics isolate whether the system found relevant evidence and ranked it usefully. Generation quality is a later stage, so a system can retrieve good evidence and still write a bad answer or retrieve weak evidence and hallucinate a confident answer.",
  ),
  makeQuestion(
    "cme295-lect9-q28",
    "easy",
    "Which steps are part of a typical tool-calling flow?",
    [
      [
        "Expose tool descriptions and argument schemas so the model knows what calls are available.",
        true,
      ],
      [
        "Predict a tool name and structured arguments from the user request.",
        true,
      ],
      ["Execute the backend function or API outside the model.", true],
      [
        "Use the returned tool output when synthesizing the final natural-language response.",
        true,
      ],
    ],
    "Tool calling separates model prediction from external execution. The model chooses a structured call, the system executes it, and the final response should be grounded in the observed tool result rather than invented data.",
  ),
  makeQuestion(
    "cme295-lect9-q29",
    "hard",
    "An assistant selects the correct account-balance tool but passes the wrong `account_id` before the backend runs. Which diagnosis is most precise?",
    [
      ["This is a tool prediction error involving wrong arguments.", true],
      [
        "This is a tool call execution error caused by a backend returning malformed JSON.",
        false,
      ],
      [
        "This is a response generation error caused by ignoring a correct tool result.",
        false,
      ],
      [
        "This is benchmark contamination because the account API appeared in pretraining data.",
        false,
      ],
    ],
    "The failure happens while predicting the call, before the backend has a chance to execute correctly or incorrectly. Separating wrong tool names, wrong arguments, backend failures, and final synthesis failures makes agent debugging actionable.",
  ),
  makeQuestion(
    "cme295-lect9-q30",
    "medium",
    "Which statements correctly connect MCP, ReAct-style loops, and agent systems?",
    [
      [
        "Model Context Protocol (MCP) standardizes how hosts, clients, and servers expose tools and resources.",
        true,
      ],
      [
        "A ReAct-style loop alternates reasoning, actions, observations, and further planning.",
        true,
      ],
      [
        "Agent systems often combine retrieval, tool calling, state, and repeated steps to pursue a user goal.",
        true,
      ],
      [
        "Agent-to-agent communication requires every participating agent to share the same neural-network weights.",
        false,
      ],
    ],
    "MCP is about standardizing access boundaries, while ReAct is a control pattern for using actions and observations during a task. Multi-agent coordination can involve protocols and capability descriptions without requiring the agents to be identical models.",
  ),
  makeAssertionReasonQuestion(
    "cme295-lect9-q31",
    "hard",
    "Assertion: Multi-step agent workflows can become less reliable as the number of required successful steps grows.\n\nReason: If each conditional step has success probability below 1, the end-to-end success probability multiplies across the required steps.",
    4,
    "The assertion is true because an agent task can fail at planning, tool selection, tool execution, observation handling, or final synthesis. The reason is also true and explains the assertion: even high per-step reliability can compound into a noticeably lower end-to-end success rate.",
  ),
  makeQuestion(
    "cme295-lect9-q32",
    "easy",
    "An evaluation plan separates model-output quality from serving-system performance. Which measurements are useful but belong to different layers of that plan?",
    [
      ["Instruction following and answer relevance.", true],
      ["Factual support or claim faithfulness.", true],
      ["Endpoint latency and uptime.", true],
      ["Cost per request or per generated token.", true],
    ],
    "Output quality asks whether the answer itself is good: relevant, coherent, factual, safe, and aligned with the task. Latency, uptime, and cost are also important, but they measure whether the system can be served under operational constraints.",
  ),
  makeQuestion(
    "cme295-lect9-q33",
    "medium",
    "Which practices make human ratings more useful for evaluating open-ended language-model answers?",
    [
      ["Give raters a clear rubric for the dimension being judged.", true],
      [
        "Track inter-rater agreement rather than trusting raw labels blindly.",
        true,
      ],
      [
        "Use calibration or alignment sessions when raters disagree too much.",
        true,
      ],
      [
        "Budget for the fact that human review is slower and more expensive than most automated scoring.",
        true,
      ],
    ],
    "Human ratings are often the closest practical signal for open-ended quality, but they are not automatically consistent or cheap. Rubrics, agreement tracking, calibration, and realistic cost planning help turn subjective judgments into a more reliable evaluation process.",
  ),
  makeQuestion(
    "cme295-lect9-q34",
    "medium",
    "Which statements about reference-based text metrics are correct?",
    [
      [
        "METEOR combines precision- and recall-like matching with an ordering or fragmentation penalty.",
        true,
      ],
      [
        "BLEU emphasizes n-gram precision and uses a brevity penalty to discourage overly short outputs.",
        true,
      ],
      [
        "ROUGE is commonly associated with summarization and recall-oriented overlap variants.",
        true,
      ],
      [
        "These metrics perfectly handle every paraphrase that preserves meaning but shares few words with the reference.",
        false,
      ],
    ],
    "Reference metrics are useful when surface overlap with fixed references is meaningful and repeatable. They are weaker for semantic equivalence, style variation, and open-ended quality, so they often need to be complemented by human or judge-based evaluation.",
  ),
  makeQuestion(
    "cme295-lect9-q35",
    "medium",
    "Which design choices strengthen an LLM-as-a-Judge evaluation?",
    [
      [
        "Provide the judge with the prompt, the model response, and explicit criteria.",
        true,
      ],
      [
        "Ask for structured outputs such as rationale and score fields when results feed an automated pipeline.",
        true,
      ],
      [
        "Use low temperature or other reproducibility controls when stable scores matter.",
        true,
      ],
      [
        "Mitigate known biases such as position bias, verbosity bias, and self-enhancement bias.",
        true,
      ],
    ],
    "LLM-as-a-Judge is a protocol, not just a model call. Criteria, structure, reproducibility controls, bias checks, and calibration against humans make the resulting scores easier to interpret and less likely to reflect artifacts of the judge prompt.",
  ),
  makeQuestion(
    "cme295-lect9-q36",
    "medium",
    "A generated answer is fluent and polite but contains several unsupported factual claims. Which evaluation approach most directly targets that failure?",
    [
      [
        "Decompose the answer into atomic claims and check each claim against trusted evidence or ground truth.",
        true,
      ],
      ["Increase the temperature so the answer becomes more diverse.", false],
      [
        "Score only endpoint uptime because the text already sounds coherent.",
        false,
      ],
      [
        "Use only unigram overlap with a reference and ignore whether individual claims are supported.",
        false,
      ],
    ],
    "Fluency and factuality are separate dimensions. Claim decomposition targets factuality by asking whether each assertion is supported, contradicted, or unverifiable instead of rewarding prose that merely sounds confident.",
  ),
  makeQuestion(
    "cme295-lect9-q37",
    "hard",
    "Which benchmark-interpretation statements are sound?",
    [
      [
        "Scores on knowledge, math, coding, safety, and agent benchmarks describe capability slices rather than complete product readiness.",
        true,
      ],
      [
        "Data contamination and Goodhart's law can make leaderboard gains less informative than they appear.",
        true,
      ],
      [
        "The highest benchmark score should determine deployment even when latency, cost, safety, and task fit point elsewhere.",
        false,
      ],
      [
        "A public benchmark score proves the model did not memorize or indirectly access any benchmark content.",
        false,
      ],
    ],
    "Benchmarks are useful tools, but each one measures a limited construct under a particular protocol. Deployment decisions require contamination checks, slice analysis, and product constraints such as latency, cost, safety, and reliability.",
  ),
  makeQuestion(
    "cme295-lect9-q38",
    "hard",
    "Which statements correctly distinguish \\(Pass@k\\) and \\(Pass^k\\) for repeated model attempts?",
    [
      [
        "\\(Pass@k\\) asks whether at least one of the \\(k\\) attempts succeeds.",
        true,
      ],
      [
        "\\(Pass^k\\) asks whether all \\(k\\) attempts succeed, which is stricter for repeated agent use.",
        true,
      ],
      [
        "\\(Pass@k\\) and \\(Pass^k\\) are identical whenever \\(k>1\\).",
        false,
      ],
      [
        "If each attempt succeeds independently with probability below 1, \\(Pass^k\\) increases as more required attempts are added.",
        false,
      ],
    ],
    "Pass@k is an at-least-one-success metric, so it can improve when more attempts are sampled. Pass^k is an all-attempts-success reliability metric, so it becomes harder as more required attempts are added unless success is certain.",
  ),
  makeAssertionReasonQuestion(
    "cme295-lect9-q39",
    "hard",
    "Assertion: A high MMLU score guarantees reliable performance in multi-step tool workflows.\n\nReason: MMLU-style evaluation uses constrained answer choices across many knowledge tasks.",
    2,
    "The assertion is false because broad knowledge accuracy does not guarantee correct tool selection, state changes, safety behavior, or repeated workflow reliability. The reason is true: constrained knowledge benchmarks are valuable, but their format and target capability are narrower than agentic workflow execution.",
  ),
  makeQuestion(
    "cme295-lect9-q40",
    "medium",
    "A user-facing LLM answer can depend on many course concepts at once. Which links in that chain are realistic?",
    [
      [
        "Tokenization, embeddings, attention, and decoder architecture shape how the model processes the prompt.",
        true,
      ],
      [
        "Pretraining, supervised fine-tuning, and preference tuning shape what behavior the model has learned.",
        true,
      ],
      [
        "Retrieval and tools can add external information or actions that fixed weights alone do not contain.",
        true,
      ],
      [
        "Evaluation and monitoring help decide whether the resulting answer and system are good enough for the use case.",
        true,
      ],
    ],
    "A modern LLM product is a stack rather than a single trick. Architecture, training, post-training, retrieval, tools, decoding, and evaluation each contribute different failure modes and improvement levers.",
  ),
  makeQuestion(
    "cme295-lect9-q41",
    "medium",
    "Which concept-role pairs are correctly matched?",
    [
      [
        "Tokenization: defines the discrete sequence units that receive embeddings.",
        true,
      ],
      [
        "Query-key-value attention: computes contextual mixing through content-based weights over value vectors.",
        true,
      ],
      [
        "LoRA or QLoRA: adapts a model through parameter-efficient fine-tuning, often with frozen base weights.",
        true,
      ],
      [
        "Key-value caching: stores prior queries instead of prior keys and values during autoregressive decoding.",
        false,
      ],
    ],
    "The first three pairs match the role each concept plays in the model stack. KV caching stores keys and values from previous tokens so later decoding steps do not recompute them; prior queries are not the reusable objects needed for future attention.",
  ),
  makeQuestion(
    "cme295-lect9-q42",
    "hard",
    "A deployed assistant is accurate enough but too slow and expensive for a high-volume workflow. Which interventions are aligned with the systems ideas covered by modern LLM practice?",
    [
      [
        "Consider inference optimizations such as KV caching, GQA-style cache reduction, PagedAttention, speculative decoding, or a smaller model.",
        true,
      ],
      [
        "Compare models on a quality/cost/latency frontier rather than selecting by quality score alone.",
        true,
      ],
      [
        "Retrain the tokenizer as the primary fix even if generation and memory movement dominate the measured cost.",
        false,
      ],
      [
        "Replace latency measurement with human preference ratings because user-facing quality is the only metric that matters.",
        false,
      ],
    ],
    "Serving constraints require measuring the bottleneck and choosing a model or optimization that attacks it. Human quality ratings still matter, but a high-volume product also needs latency, throughput, memory, and cost metrics.",
  ),
  makeQuestion(
    "cme295-lect9-q43",
    "easy",
    "Why can transformers be reused beyond text in domains such as vision?",
    [
      [
        "Self-attention is a general mechanism for relating tokens or token-like units through learned representations.",
        true,
      ],
      [
        "Transformers have weaker built-in modality-specific assumptions than architectures designed only around local convolution.",
        true,
      ],
      [
        "Transformer layers can process images without any representation of patches, positions, or visual features.",
        false,
      ],
      [
        "The query-key-value mechanism is defined only for natural-language words and cannot be applied to vectors from other modalities.",
        false,
      ],
    ],
    "Attention operates on vectors, so the key question is how to turn a modality into useful vectors with appropriate positional structure. Images still need representation choices such as patches and 2D position information; the transformer block does not remove the need for input design.",
  ),
  makeQuestion(
    "cme295-lect9-q44",
    "easy",
    "Which steps belong in a Vision Transformer (ViT)-style image classifier?",
    [
      ["Split the image into fixed-size patches.", true],
      ["Project each patch into a vector embedding.", true],
      ["Add position information and often a class-style token.", true],
      [
        "Process the sequence with a transformer encoder and classify from an encoded representation.",
        true,
      ],
    ],
    "ViT treats image patches as token-like units and then applies transformer encoder machinery. Position information matters because the model needs to know where patches came from in the image, and the classifier reads from an encoded representation rather than from raw pixels directly.",
  ),
  makeQuestion(
    "cme295-lect9-q45",
    "hard",
    "An image is \\(224 \\times 224\\) pixels and a Vision Transformer uses non-overlapping \\(16 \\times 16\\) patches. If one class token is prepended, how many sequence tokens enter the transformer?",
    [
      ["\\((224/16) \\times (224/16) + 1 = 14 \\times 14 + 1 = 197\\).", true],
      ["\\((224+224)/16 + 1 = 29\\).", false],
      ["\\(224 \\times 224 / 16 + 1 = 3137\\).", false],
      ["\\(16 \\times 16 + 1 = 257\\).", false],
    ],
    "A non-overlapping patch grid has 14 patches along height and 14 patches along width, giving 196 image patches. The class token adds one extra sequence element, so the transformer receives 197 tokens.",
  ),
  makeQuestion(
    "cme295-lect9-q46",
    "medium",
    "Which statements distinguish common vision-language model design patterns?",
    [
      [
        "One pattern feeds visual tokens or projected visual features into a decoder-only language-model-style architecture.",
        true,
      ],
      [
        "Another pattern uses cross-attention so a decoder can condition on visual encoder representations.",
        true,
      ],
      [
        "Visual instruction tuning helps the system turn image-conditioned representations into useful answers to user requests.",
        true,
      ],
      [
        "A system that sees image features cannot generate text because text generation requires a text-only input.",
        false,
      ],
    ],
    "Vision-language models reuse language decoders while adding a path for visual information. The visual path can be fused as tokens or through cross-attention, and instruction tuning teaches the system how to answer multimodal requests rather than merely encode images.",
  ),
  makeQuestion(
    "cme295-lect9-q47",
    "medium",
    "Which distinctions among multimodal tasks are accurate?",
    [
      [
        "Image understanding can mean classifying or describing visual content from image-derived representations.",
        true,
      ],
      [
        "Image generation creates visual data, and modern diffusion image systems may use transformer blocks internally.",
        true,
      ],
      [
        "Visual question answering is identical to image generation because both tasks must output pixels.",
        false,
      ],
      [
        "Using a transformer in an image-generation system means query, key, value, and position ideas no longer matter.",
        false,
      ],
    ],
    "Understanding, generation, and vision-language answering use overlapping building blocks for different objectives. A model can answer in text about an image without generating pixels, and image-generation architectures can still reuse transformer attention concepts.",
  ),
  makeAssertionReasonQuestion(
    "cme295-lect9-q48",
    "medium",
    "Assertion: Autoregressive language-model inference is serial in a way that creates a generation-time latency bottleneck.\n\nReason: Teacher-forced training can evaluate many target positions in parallel.",
    5,
    "The assertion is true because each generated token becomes part of the prefix used to predict the next token. The reason is also true, but it is a contrast with training rather than the causal explanation; the serial dependency during generation is what creates the latency bottleneck.",
  ),
  makeQuestion(
    "cme295-lect9-q49",
    "easy",
    "Which statements capture the high-level intuition of diffusion image generation?",
    [
      ["Noise is easy to sample as a starting point.", true],
      ["A forward process can add noise to real data.", true],
      [
        "A learned reverse process can denoise step by step toward realistic data.",
        true,
      ],
      [
        "The overall goal is to learn a transformation from a simple noise distribution toward the desired data distribution.",
        true,
      ],
    ],
    "Diffusion models frame generation as reversing a corruption process. The useful intuition is not that noise itself is meaningful, but that the model learns how to transform a simple starting distribution into samples that resemble the training data.",
  ),
  makeQuestion(
    "cme295-lect9-q50",
    "medium",
    "How does a masked diffusion language model adapt the diffusion idea to discrete text?",
    [
      [
        "The forward process can mask tokens with some probability instead of adding continuous Gaussian noise to words.",
        true,
      ],
      ["The reverse process learns to unmask or reconstruct tokens.", true],
      [
        "Several token positions may be filled across fewer forward passes than strict left-to-right generation.",
        true,
      ],
      [
        "It works by adding pixel-level image noise directly to character strings.",
        false,
      ],
    ],
    "Text is discrete, so masking is a more natural corruption operation than Gaussian image noise over pixels. The model learns a denoising-style unmasking process, which can reduce the strict one-new-token-per-step bottleneck for some generation setups.",
  ),
  makeQuestion(
    "cme295-lect9-q51",
    "hard",
    "Which statements fairly describe diffusion-style LLMs compared with autoregressive language models?",
    [
      [
        "They may be better suited for some tasks because generation need not follow a strictly left-to-right one-token-per-pass pattern.",
        true,
      ],
      [
        "Current work still has to adapt techniques developed for autoregressive models and close quality or performance gaps.",
        true,
      ],
      [
        "They automatically solve factuality, controllability, and evaluation for every language task.",
        false,
      ],
      [
        "They remove the need for transformer or attention-style architectures in text modeling.",
        false,
      ],
    ],
    "Diffusion-style LLMs are interesting because they attack the generation bottleneck from a different modeling direction. That does not make them a universal replacement; quality, tooling, evaluation, and integration with existing autoregressive techniques remain active issues.",
  ),
  makeQuestion(
    "cme295-lect9-q52",
    "medium",
    "Which examples show cross-pollination between modalities?",
    [
      [
        "Diffusion-style training ideas moving from image generation toward text generation.",
        true,
      ],
      ["Transformers being used inside image-generation architectures.", true],
      [
        "Visual-token representations being explored as compact carriers of text or document information.",
        true,
      ],
      [
        "RoPE-like position ideas being adapted to two-dimensional or multimodal layouts.",
        true,
      ],
    ],
    "Modern model research often transfers ideas across modalities rather than keeping text, images, and multimodal systems separate. The details must change, such as moving from one-dimensional token positions to two-dimensional image grids, but the same architectural vocabulary can reappear.",
  ),
  makeQuestion(
    "cme295-lect9-q53",
    "hard",
    "What changes when a one-dimensional positional idea such as RoPE is adapted to images or multimodal inputs?",
    [
      [
        "The position scheme must represent two-dimensional image-grid relationships rather than only left-to-right token distance.",
        true,
      ],
      [
        "Text and image tokens may need compatible placement so relative-position computations remain meaningful in a shared architecture.",
        true,
      ],
      [
        "The adaptation reflects that useful representation design is still part of the model, not just a preprocessing detail.",
        true,
      ],
      [
        "Text tokenizers become irrelevant because visual patches always preserve every semantic detail at lower cost.",
        false,
      ],
    ],
    "Moving RoPE into visual or multimodal settings changes the geometry the model has to encode. Image patches live on a grid, and multimodal systems must decide how text and visual positions interact; that is a representation-design problem, not a reason to ignore tokenization or information loss.",
  ),
  makeQuestion(
    "cme295-lect9-q54",
    "hard",
    "In an idealized comparison, an autoregressive model emits one new token per forward pass, while a masked diffusion model fills 8 masked positions per forward pass. For a 64-token output, which pass-count comparison follows from those assumptions?",
    [
      [
        "Autoregressive generation needs \\(64\\) passes, while the masked diffusion setup needs \\(64/8 = 8\\) passes.",
        true,
      ],
      [
        "Autoregressive generation needs \\(8\\) passes, while the masked diffusion setup needs \\(64\\) passes.",
        false,
      ],
      [
        "Both methods need \\(64 \\times 8 = 512\\) passes because every token must be revisited eight times.",
        false,
      ],
      [
        "Both methods need exactly one pass because training-time parallelism and inference-time generation are identical.",
        false,
      ],
    ],
    "The calculation follows the simplified assumptions in the prompt: one token per autoregressive pass versus eight filled positions per diffusion pass. Real systems have more details, but the arithmetic shows why non-left-to-right generation is attractive for latency.",
  ),
  makeQuestion(
    "cme295-lect9-q55",
    "medium",
    "Which design axes remain active areas of foundational transformer research?",
    [
      [
        "Optimizer choices, such as AdamW-style methods versus newer alternatives.",
        true,
      ],
      [
        "Normalization choices, including where normalization appears and which normalization variant is used.",
        true,
      ],
      ["Attention-head sharing patterns such as MHA, MQA, and GQA.", true],
      [
        "Activation functions, MoE choices, layer counts, head counts, and feed-forward dimensions.",
        true,
      ],
    ],
    "The current transformer stack is not a settled endpoint. Papers continue to vary basic components because stability, quality, memory use, inference speed, data regime, and hardware constraints all interact with these design choices.",
  ),
  makeAssertionReasonQuestion(
    "cme295-lect9-q56",
    "hard",
    "Assertion: Training heavily on low-diversity LLM-generated text can degrade what future models learn from the data distribution.\n\nReason: Generated text can narrow or shift the distribution relative to richer human-generated data, creating model-collapse risk.",
    4,
    "The assertion is true because training data quality and diversity affect what patterns the next model can learn. The reason is also true and explains the assertion: if generated data recursively replaces diverse human data, the training distribution can lose coverage and become less informative.",
  ),
  makeQuestion(
    "cme295-lect9-q57",
    "medium",
    "Which responses address the data-quality problem created by increasingly synthetic web text?",
    [
      [
        "Curate data more deliberately instead of treating every scraped token as equally valuable.",
        true,
      ],
      [
        "Use higher-quality intermediate or mid-training corpora when a broad pretrained model needs domain or quality shaping before final fine-tuning.",
        true,
      ],
      [
        "Assume more synthetic text is always better because it is cheaper to generate.",
        false,
      ],
      [
        "Treat data as irrelevant once model architecture and optimizer are chosen.",
        false,
      ],
    ],
    "Data remains a central ingredient, especially when the web contains more generated and repetitive content. Curation and mid-training are ways to improve the signal the model sees, while blindly adding cheap synthetic text can amplify distribution shift or diversity loss.",
  ),
  makeQuestion(
    "cme295-lect9-q58",
    "hard",
    "Which statements connect cost frontiers and hardware specialization to future LLM deployment?",
    [
      [
        "A quality/cost Pareto frontier helps choose a model based on use-case constraints rather than headline quality alone.",
        true,
      ],
      [
        "Attention-specific hardware ideas are motivated partly by key-value reads, writes, and memory movement that current GPUs handle indirectly.",
        true,
      ],
      [
        "Small language models are useless for deployment because a larger model is better on every cost, latency, and quality dimension.",
        false,
      ],
      [
        "A promising analog-attention hardware result makes software inference optimizations such as FlashAttention irrelevant.",
        false,
      ],
    ],
    "Serving economics push model selection toward tradeoffs, not a single leaderboard number. Hardware specialization and software optimization both respond to the same pressure: attention and KV movement are costly, so latency and energy can matter as much as raw benchmark quality.",
  ),
  makeQuestion(
    "cme295-lect9-q59",
    "medium",
    "Which open problems remain for current LLM systems?",
    [
      [
        "Fixed weights mean RAG and tools work around stale knowledge more than they provide true continuous learning.",
        true,
      ],
      [
        "Hallucination is tied to next-token prediction not being the same objective as mapping every statement to verified facts.",
        true,
      ],
      [
        "Personalization, interpretability, and safety remain hard deployment questions.",
        true,
      ],
      [
        "Browser or OS-level agents raise security issues such as prompt injection and data exfiltration.",
        true,
      ],
    ],
    "RAG, tools, and agent wrappers are useful, but they do not erase core limitations of the modeling objective or deployment environment. Current systems still need reliability, security, personalization boundaries, interpretability, and safety work before they can handle broad autonomous responsibility.",
  ),
  makeQuestion(
    "cme295-lect9-q61",
    "medium",
    "A product team is choosing between a small language model and a much larger model for a high-volume autocomplete feature. Which evaluation questions are most relevant?",
    [
      [
        "Whether the smaller model is good enough at the target task once latency and serving cost are included.",
        true,
      ],
      [
        "Whether the larger model's quality gain justifies its additional inference cost and energy use.",
        true,
      ],
      [
        "Whether the largest model should be chosen automatically because benchmark quality is the only deployment constraint.",
        false,
      ],
      [
        "Whether latency and cost can be ignored once a model is on the quality/cost Pareto frontier.",
        false,
      ],
    ],
    "A quality/cost frontier is useful because deployment is not decided by capability alone. For a high-volume feature, a smaller model can be the better engineering choice if it meets the quality bar with lower latency, lower serving cost, and lower energy use.",
  ),
];
