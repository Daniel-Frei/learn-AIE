import { Question } from "../../quiz";

type Lecture3Difficulty = "easy" | "medium" | "hard";
type OptionSeed = readonly [text: string, isCorrect: boolean];

function makeQuestion(
  id: string,
  difficulty: Lecture3Difficulty,
  prompt: string,
  optionSeeds: readonly OptionSeed[],
  explanation: string,
): Question {
  if (optionSeeds.length !== 4) {
    throw new Error(`CME295 Lecture 3 question ${id} needs 4 options.`);
  }

  return {
    id,
    chapter: 3,
    difficulty,
    prompt,
    options: optionSeeds.map(([text, isCorrect]) => ({ text, isCorrect })),
    explanation,
  };
}

export const stanfordCME295Lecture3LLMsQuestions: Question[] = [
  makeQuestion(
    "cme295-lect3-q101",
    "easy",
    "A model receives the token sequence `[BOS] A teddy bear` and predicts a distribution for the next token. Which statement best identifies the language-modeling task?",
    [
      [
        "Estimate the probability of the next token conditioned on the preceding tokens.",
        true,
      ],
      [
        "Encode the sequence into a single fixed document vector for classification.",
        false,
      ],
      [
        "Translate the sequence by attending to a separate encoder output.",
        false,
      ],
      [
        "Assign each previous token to a hand-written syntactic category.",
        false,
      ],
    ],
    "A language model assigns probabilities to token sequences by modeling the next-token distribution given the tokens already present. The other choices describe representation learning, encoder-decoder translation, or manual linguistic annotation rather than autoregressive language modeling.",
  ),
  makeQuestion(
    "cme295-lect3-q102",
    "easy",
    "Which statements correctly distinguish current large language models from BERT-style encoder-only models?",
    [
      [
        "Current large language models are generally decoder-only text-to-text generators.",
        true,
      ],
      [
        "BERT-style models are useful for contextual embeddings and classification-style tasks.",
        true,
      ],
      [
        "BERT-style models use the decoder stack and cross-attention as their core architecture.",
        false,
      ],
      [
        "Encoder-only models match the current text-generating definition of a large language model.",
        false,
      ],
    ],
    "The lecture uses the current convention that large language models are large text-to-text language models, usually decoder-only. BERT is important, but it is encoder-only and is mainly used to produce contextual representations for tasks such as classification rather than to generate text autoregressively.",
  ),
  makeQuestion(
    "cme295-lect3-q103",
    "easy",
    "Which statements describe the scale dimensions that make a language model 'large'?",
    [
      ["Parameter count reaches the billion-parameter range or beyond.", true],
      [
        "Pretraining data is measured in hundreds of billions or trillions of tokens.",
        true,
      ],
      ["Training and serving require substantial accelerator compute.", true],
      [
        "The model becomes large because it stores one human-written rule for each vocabulary item.",
        false,
      ],
    ],
    "Large language models are large along model-size, data-size, and compute dimensions. A rule table is not the mechanism described here; the model learns neural parameters from a very large corpus and then uses those parameters for next-token prediction.",
  ),
  makeQuestion(
    "cme295-lect3-q104",
    "easy",
    "Which statements correctly characterize the decoder-only Transformer backbone used by modern text-generating LLMs?",
    [
      [
        "Masked self-attention prevents a position from using future tokens during next-token prediction.",
        true,
      ],
      ["The feed-forward network remains part of each decoder block.", true],
      ["Residual and normalization layers remain part of the stack.", true],
      [
        "Cross-attention to an encoder is removed because there is no encoder stream.",
        true,
      ],
    ],
    "Decoder-only models keep the causal self-attention and feed-forward machinery needed for autoregressive generation. Cross-attention belongs to encoder-decoder models, so it is removed when the architecture keeps only the decoder side.",
  ),
  makeQuestion(
    "cme295-lect3-q105",
    "medium",
    "A dense feed-forward decoder block and a sparse Mixture-of-Experts (MoE) block have the same input token representation. Which statement best describes the MoE substitution?",
    [
      [
        "The feed-forward sublayer is replaced by several feed-forward experts plus a learned router that chooses expert contributions.",
        true,
      ],
      [
        "The masked self-attention sublayer is replaced by a tokenizer that chooses longer subwords.",
        false,
      ],
      [
        "The residual connections are replaced by a beam-search decoder during training.",
        false,
      ],
      [
        "The output softmax is replaced by a fixed table of expert names.",
        false,
      ],
    ],
    "MoE layers usually replace the expensive feed-forward sublayer with multiple expert feed-forward networks and a router. Attention, tokenization, residual connections, and the vocabulary softmax are different parts of the system and are not the MoE substitution described in the lecture.",
  ),
  makeQuestion(
    "cme295-lect3-q106",
    "medium",
    "Which statements correctly compare dense and sparse MoE computation?",
    [
      [
        "Dense MoE forms a weighted combination using contributions from the expert set.",
        true,
      ],
      [
        "Sparse MoE restricts computation to selected top-k experts for a token.",
        true,
      ],
      [
        "Sparse MoE routes tokens by selecting a separate encoder-decoder architecture for each prompt.",
        false,
      ],
      [
        "Dense MoE reduces compute by skipping the router and running a single expert chosen before training.",
        false,
      ],
    ],
    "Dense MoE can weight many expert outputs, while sparse MoE limits active computation to top-k selected experts. The router chooses experts inside a layer, not whole model architectures, and dense MoE does not get its cost savings by using a preselected single expert.",
  ),
  makeQuestion(
    "cme295-lect3-q107",
    "medium",
    "For an MoE layer with gate values \\(G(x)_i\\) and expert outputs \\(E_i(x)\\), which statements about \\(\\hat{y}=\\sum_i G(x)_iE_i(x)\\) are correct?",
    [
      [
        "The gate values determine how much each expert contributes to the layer output.",
        true,
      ],
      [
        "The experts and gate can be trained jointly through the model objective.",
        true,
      ],
      [
        "A sparse top-k version sets nonselected experts' contribution to zero for that token.",
        true,
      ],
      [
        "The formula describes a post-hoc majority vote over completed text responses.",
        false,
      ],
    ],
    "The MoE formula is a layer computation: a learned gate weights expert subnetworks and the weighted outputs are combined. It is not a voting procedure over final completions; voting appears later in self-consistency prompting and operates at the response level.",
  ),
  makeQuestion(
    "cme295-lect3-q108",
    "hard",
    "Which statements correctly explain why the feed-forward network is the natural target for MoE layers inside a decoder block?",
    [
      [
        "The feed-forward network often projects through a larger hidden dimension \\(d_{ff}\\), making it parameter-heavy.",
        true,
      ],
      [
        "Replacing the feed-forward network with experts lets the model increase total capacity while controlling active parameters.",
        true,
      ],
      [
        "Attention heads and MoE experts are separate design axes, so expert routing is not tied to one expert per attention head.",
        true,
      ],
      [
        "The feed-forward network is the location where the model stores the KV cache for previous tokens.",
        false,
      ],
    ],
    "The feed-forward sublayer is a large part of the decoder block's parameter and compute budget, so it is the usual location for expert substitution. KV cache storage is associated with attention keys and values during inference, not with making the feed-forward network into a set of experts.",
  ),
  makeQuestion(
    "cme295-lect3-q109",
    "medium",
    "Which statement best describes routing collapse in an MoE model?",
    [
      [
        "A small subset of experts receives most tokens, leaving other experts underused.",
        true,
      ],
      [
        "The model switches from sparse routing to beam search during generation.",
        false,
      ],
      ["The tokenizer merges rare words into expert identifiers.", false],
      [
        "The KV cache stores key vectors in non-contiguous memory blocks.",
        false,
      ],
    ],
    "Routing collapse is a training failure mode where the router repeatedly sends tokens to the same experts, reducing the value of having many experts. Beam search, tokenization, and KV-cache memory layout are separate topics and do not describe the expert-utilization failure.",
  ),
  makeQuestion(
    "cme295-lect3-q110",
    "hard",
    "Which statements correctly describe the auxiliary load-balancing quantities used to reduce MoE routing collapse?",
    [
      [
        "\\(f_i\\) measures the fraction of tokens routed to expert \\(i\\).",
        true,
      ],
      [
        "\\(P_i\\) measures the average routing probability assigned to expert \\(i\\).",
        true,
      ],
      [
        "The added loss term is unrelated to whether expert usage becomes more uniform.",
        false,
      ],
      [
        "The auxiliary term replaces next-token prediction as the main training objective.",
        false,
      ],
    ],
    "The auxiliary objective supplements the language-modeling loss by discouraging severe expert imbalance, so it is directly related to expert-usage uniformity. It uses routing statistics such as the fraction of tokens and average routing probability for each expert, but it does not replace the main task of predicting language-model targets.",
  ),
  makeQuestion(
    "cme295-lect3-q111",
    "medium",
    "Which statements about token-level MoE routing are correct?",
    [
      [
        "Different tokens in the same sequence can be routed to different experts.",
        true,
      ],
      [
        "The router uses a token representation as input when making a routing decision.",
        true,
      ],
      [
        "Routing decisions are fixed once for the whole prompt before the first decoder layer runs.",
        false,
      ],
      [
        "A single prompt-level expert is chosen before tokenization and reused for the whole network.",
        false,
      ],
    ],
    "The lecture emphasizes routing at the token level: each token representation can be sent to an expert selected by the router. This is finer-grained than choosing one expert for an entire prompt before the model computes token representations, and routing can vary across positions and layers.",
  ),
  makeQuestion(
    "cme295-lect3-q112",
    "medium",
    "Which statements correctly describe why sparse MoE can support very large total parameter counts?",
    [
      ["More experts increase total stored parameters.", true],
      [
        "Top-k routing keeps active parameters for a token much smaller than total parameters.",
        true,
      ],
      [
        "Increasing expert count can increase model capacity without proportional forward-pass compute.",
        true,
      ],
      [
        "The active-parameter budget can stay controlled even as the stored expert pool grows.",
        true,
      ],
    ],
    "Sparse MoE separates total parameters from active parameters: many experts can exist, but a token uses a selected subset. This is why MoE can raise stored capacity while keeping the per-token compute tied to top-k routing rather than to the full expert pool.",
  ),
  makeQuestion(
    "cme295-lect3-q113",
    "hard",
    "A sparse MoE layer has 8 experts and uses top-2 routing for each token. Which statement best describes one forward pass for a single token at that layer?",
    [
      [
        "The router scores the experts and the token is processed by the two selected experts for that layer.",
        true,
      ],
      [
        "The token is processed by eight experts and the top two are chosen after the output text is complete.",
        false,
      ],
      [
        "Two attention heads are disabled and the remaining heads act as feed-forward experts.",
        false,
      ],
      [
        "Two vocabulary tokens are sampled before the expert outputs are computed.",
        false,
      ],
    ],
    "Top-k routing is a layer-internal compute decision: the router chooses expert subnetworks for a token representation before that layer's output is produced. It is not a post-generation selection, an attention-head pruning rule, or a decoding step over vocabulary tokens.",
  ),
  makeQuestion(
    "cme295-lect3-q114",
    "hard",
    "Which statements correctly connect FLOPs to dense and sparse LLM design?",
    [
      [
        "FLOPs count floating point additions, multiplications, and related arithmetic work.",
        true,
      ],
      [
        "Sparse MoE can lower active forward-pass FLOPs relative to running every expert.",
        true,
      ],
      [
        "FLOPs measure wall-clock latency directly, independent of hardware and memory movement.",
        false,
      ],
      [
        "A model with fewer FLOPs per token necessarily stores fewer total parameters.",
        false,
      ],
    ],
    "FLOPs are a useful hardware-agnostic count of arithmetic work, but they are not the same as wall-clock latency because memory movement and hardware throughput matter. Sparse MoE can reduce active compute for a token while storing many inactive expert parameters, so total capacity and per-token compute must be reasoned about separately.",
  ),
  makeQuestion(
    "cme295-lect3-q115",
    "hard",
    "Which statements about noisy gating and auxiliary losses in MoE training are correct?",
    [
      [
        "Noisy gating perturbs router scores so less-used experts have chances to receive tokens during training.",
        true,
      ],
      [
        "Auxiliary losses can penalize imbalanced expert usage at the batch level.",
        true,
      ],
      [
        "Both techniques eliminate the need to train the router jointly with the experts.",
        false,
      ],
      [
        "Both techniques make expert routing independent of the learned token representations.",
        false,
      ],
    ],
    "Noisy gating and load-balancing losses are tools for keeping expert usage healthy while the router is being learned. They do not remove joint training or make routing independent of representations; the router still uses learned signals, but the training setup discourages collapse.",
  ),
  makeQuestion(
    "cme295-lect3-q116",
    "medium",
    "Which statements correctly distinguish expert capacity from active computation in MoE LLMs?",
    [
      [
        "Capacity refers to the larger pool of stored expert parameters available to the model.",
        true,
      ],
      [
        "Active computation refers to the parameters used for a particular token's forward pass.",
        true,
      ],
      [
        "A trillion-parameter sparse MoE can use far fewer than a trillion parameters for one token.",
        true,
      ],
      [
        "The stored expert pool and the active expert subset answer different scaling questions.",
        true,
      ],
    ],
    "MoE makes it possible to store a large pool of expert parameters while activating a subset for each token. The important distinction is total capacity versus active parameters, because a model can be large in storage while keeping each routed token's computation much smaller.",
  ),
  makeQuestion(
    "cme295-lect3-q117",
    "easy",
    "During autoregressive generation, what happens after the model samples or selects one next token?",
    [
      [
        "The new token is appended to the sequence and the model predicts the following token from the extended prefix.",
        true,
      ],
      [
        "The entire training set is updated before the next token is considered.",
        false,
      ],
      [
        "The decoder removes causal masking because the next token is already known.",
        false,
      ],
      [
        "The model switches from probabilities to hand-written grammar rules for the rest of the response.",
        false,
      ],
    ],
    "Autoregressive generation repeats next-token prediction: each chosen token becomes part of the prefix for the following step. The model weights are not updated during ordinary inference, and causal masking remains part of the decoder-only computation.",
  ),
  makeQuestion(
    "cme295-lect3-q118",
    "hard",
    "A decoder outputs logits `[4.0, 2.0, 1.0, -1.0]` for four candidate tokens. Which statements about greedy decoding are correct?",
    [
      [
        "Greedy decoding chooses the token with logit `4.0` at this step.",
        true,
      ],
      ["The choice is deterministic for fixed logits.", true],
      [
        "Greedy decoding keeps several candidate continuations before choosing a final sequence.",
        false,
      ],
      [
        "Greedy decoding samples among the four tokens according to the softmax probabilities.",
        false,
      ],
    ],
    "Greedy decoding takes the highest-scoring token at each step, so fixed logits produce a fixed next token. It keeps one path rather than a beam of candidates, and it is not the same as sampling from the softmax distribution.",
  ),
  makeQuestion(
    "cme295-lect3-q119",
    "medium",
    "Which statements correctly describe beam search in sequence generation?",
    [
      [
        "It keeps several high-scoring partial sequences instead of committing to one token path.",
        true,
      ],
      [
        "It is useful when high-likelihood structured outputs are preferred.",
        true,
      ],
      [
        "It is designed to maximize diversity by sampling tail tokens from the full vocabulary.",
        false,
      ],
      [
        "It draws the next token from the full vocabulary distribution without ranking candidate paths.",
        false,
      ],
    ],
    "Beam search tracks multiple candidate sequences to approximate high-probability completions. It is not broad stochastic sampling from the full vocabulary, and that is why it can be less suitable for creative open-ended generation than sampling methods.",
  ),
  makeQuestion(
    "cme295-lect3-q120",
    "medium",
    "Which statements correctly compare greedy decoding and beam search?",
    [
      [
        "Greedy decoding keeps one path, while beam search keeps multiple candidate paths.",
        true,
      ],
      [
        "Beam search requires more computation than greedy decoding because it expands and scores more candidates.",
        true,
      ],
      [
        "Both methods can be deterministic when model outputs and tie-breaking are fixed.",
        true,
      ],
      [
        "Beam search broadens high-probability search without introducing token-sampling randomness.",
        true,
      ],
    ],
    "Greedy and beam search are likelihood-oriented decoding strategies rather than randomness-oriented creativity controls. Beam search broadens search over high-probability paths, but sampling methods are the tools used when controlled randomness and diversity are desired.",
  ),
  makeQuestion(
    "cme295-lect3-q121",
    "easy",
    "A model produces token probabilities where `bear` has the highest probability at the current step. Which statement best describes top-k sampling with `k = 4`?",
    [
      [
        "It restricts the candidate set to the four most probable tokens and samples within that set.",
        true,
      ],
      [
        "It deterministically emits `bear` because `bear` has the highest probability.",
        false,
      ],
      [
        "It samples from the smallest probability-mass set whose total exceeds a threshold `p`.",
        false,
      ],
      [
        "It changes the model weights so the four tokens become more likely in future prompts.",
        false,
      ],
    ],
    "Top-k sampling truncates the vocabulary to the k highest-probability candidates and then samples from the truncated distribution. Deterministically taking the top token is greedy decoding, while cumulative probability thresholding is top-p sampling.",
  ),
  makeQuestion(
    "cme295-lect3-q122",
    "medium",
    "Which statements correctly describe top-p, or nucleus, sampling?",
    [
      [
        "It forms the smallest high-probability token set whose cumulative probability reaches the chosen threshold.",
        true,
      ],
      [
        "The number of tokens in the candidate set can change from one decoding step to another.",
        true,
      ],
      [
        "It is the same procedure as choosing a fixed number `k` of tokens at each step.",
        false,
      ],
      [
        "It scores whole completed sequences rather than next-token candidates.",
        false,
      ],
    ],
    "Top-p sampling is adaptive: the candidate set size depends on how probability mass is distributed at the current step. A fixed candidate count is top-k sampling, and sequence-level scoring belongs to search methods such as beam search rather than nucleus sampling.",
  ),
  makeQuestion(
    "cme295-lect3-q123",
    "medium",
    "A next-token distribution is `A: 0.50`, `B: 0.25`, `C: 0.15`, `D: 0.06`, `E: 0.04`. With top-p sampling at `p = 0.90`, which statements are correct?",
    [
      [
        "The nucleus contains `A`, `B`, and `C` because their cumulative probability is `0.90`.",
        true,
      ],
      [
        "`D` and `E` are excluded before renormalizing the remaining probabilities.",
        true,
      ],
      [
        "The candidate count is three for this step, even though `p` is not a count.",
        true,
      ],
      [
        "The method must include `D` because `D` is needed to make the cumulative probability greater than `0.90`.",
        false,
      ],
    ],
    "Top-p uses the smallest set reaching the threshold, and here `0.50 + 0.25 + 0.15 = 0.90`. The remaining candidates are removed for this sampling step, and the included probabilities are renormalized before drawing a token.",
  ),
  makeQuestion(
    "cme295-lect3-q124",
    "medium",
    "Which statements correctly describe temperature scaling before softmax?",
    [
      [
        "Lower temperature makes high-logit tokens take a larger share of probability mass.",
        true,
      ],
      [
        "Higher temperature flattens the distribution and increases diversity pressure.",
        true,
      ],
      ["Temperature rescales logits before probabilities are computed.", true],
      ["Temperature leaves the tokenizer and vocabulary unchanged.", true],
    ],
    "Temperature is a decoding-time transformation of logits before softmax. It affects sharpness and diversity of token probabilities, while the tokenizer and vocabulary remain the same objects used by the trained model.",
  ),
  makeQuestion(
    "cme295-lect3-q125",
    "easy",
    "Which statement best explains why sampling can produce different completions for the same prompt?",
    [
      [
        "Randomness enters when the decoder draws a token from the probability distribution.",
        true,
      ],
      [
        "The Transformer weights are retrained after each sampled token.",
        false,
      ],
      [
        "The context window randomly changes size during the forward pass.",
        false,
      ],
      [
        "The softmax layer stops producing probabilities after the first token.",
        false,
      ],
    ],
    "Sampling introduces randomness at token selection, so repeated generations can diverge even with the same prompt and model. Ordinary inference does not retrain weights, randomly resize the context window, or abandon next-token probability computation.",
  ),
  makeQuestion(
    "cme295-lect3-q126",
    "medium",
    "Which statements correctly describe logits and softmax in next-token prediction?",
    [
      ["Logits are unnormalized scores for vocabulary tokens.", true],
      [
        "Softmax converts logits into nonnegative probabilities that sum to one.",
        true,
      ],
      [
        "Temperature changes token strings before the tokenizer creates model inputs.",
        false,
      ],
      [
        "Softmax chooses the final token by itself without any decoding policy.",
        false,
      ],
    ],
    "The model produces logits, and softmax turns those scores into a probability distribution. Temperature rescales logits rather than token strings, and a separate decoding policy such as greedy selection, top-k sampling, top-p sampling, or beam search determines how the distribution is used.",
  ),
  makeQuestion(
    "cme295-lect3-q127",
    "easy",
    "A developer wants structured JSON output from an LLM. Which statement best describes guided decoding?",
    [
      [
        "Constrain the next-token choices during generation so invalid JSON continuations are filtered out.",
        true,
      ],
      [
        "Ask for JSON, parse the final response, and retry from scratch after each invalid output.",
        false,
      ],
      [
        "Fine-tune the model until the vocabulary contains no tokens outside JSON syntax.",
        false,
      ],
      [
        "Run beam search and accept the highest-probability sequence even if it violates the schema.",
        false,
      ],
    ],
    "Guided decoding constrains token choices at inference time so the generator follows a formal output structure such as JSON. Retrying after failure is a weaker prompt-and-validate loop, while fine-tuning or beam search alone does not enforce valid next-token choices.",
  ),
  makeQuestion(
    "cme295-lect3-q128",
    "easy",
    "Which statements correctly identify valid-token filtering in guided decoding?",
    [
      [
        "At the start of a JSON object, an opening brace can be allowed while unrelated word tokens are blocked.",
        true,
      ],
      [
        "After a property name in JSON, a colon can be allowed while a random noun can be blocked.",
        true,
      ],
      [
        "When several valid next tokens remain, a normal token-selection strategy can choose among them.",
        true,
      ],
      [
        "No model retraining is required merely because invalid next-token paths are filtered.",
        true,
      ],
    ],
    "Guided decoding works by filtering invalid next tokens according to a grammar, schema, or state machine, while still using the model distribution over valid choices. Invalid paths are constrained during inference rather than removed from the model weights through repeated retraining.",
  ),
  makeQuestion(
    "cme295-lect3-q129",
    "medium",
    "Which statement best separates top-k sampling from guided decoding?",
    [
      [
        "Top-k limits candidates by probability rank, while guided decoding limits candidates by structural validity.",
        true,
      ],
      [
        "Top-k uses a grammar and guided decoding uses a fixed number of most probable tokens.",
        false,
      ],
      [
        "Top-k operates during training and guided decoding operates during tokenization.",
        false,
      ],
      [
        "Top-k produces JSON validity proofs and guided decoding optimizes sequence likelihood.",
        false,
      ],
    ],
    "Both methods restrict candidate next tokens, but they use different criteria. Top-k uses probability rank from the model distribution, while guided decoding uses an external validity constraint such as a schema or grammar.",
  ),
  makeQuestion(
    "cme295-lect3-q130",
    "medium",
    "Which statements about inference-time nondeterminism are correct?",
    [
      ["Sampling creates nondeterminism through random token draws.", true],
      [
        "Numerical and hardware effects can create small differences even for otherwise fixed inference settings.",
        true,
      ],
      [
        "Greedy decoding samples from the probability distribution after ranking the tokens.",
        false,
      ],
      [
        "Decoder-only Transformers require stochastic internal layers at inference to produce text.",
        false,
      ],
    ],
    "Nondeterminism can come from sampling and from numerical implementation details, but the Transformer computation itself need not contain stochastic layers during inference. Greedy decoding can be deterministic under fixed conditions because it chooses the top token rather than drawing a random one.",
  ),
  makeQuestion(
    "cme295-lect3-q131",
    "hard",
    "A generation system uses beam search for a creative writing prompt and gets repetitive, safe-sounding outputs. Which explanation best matches the decoding choice?",
    [
      [
        "Beam search favors high-likelihood paths and can reduce diversity, so sampling methods are often a better fit for open-ended generation.",
        true,
      ],
      [
        "Beam search samples many low-probability tokens, so the model becomes too random for creative writing.",
        false,
      ],
      [
        "Beam search disables the language model's probability distribution and uses hand-written templates.",
        false,
      ],
      [
        "Beam search routes each token to a different MoE expert, which prevents narrative variation.",
        false,
      ],
    ],
    "Beam search is designed to keep high-scoring candidate sequences, so it can converge on bland or repetitive high-probability text. Creative generation usually benefits from controlled sampling rather than a search procedure that primarily rewards likelihood.",
  ),
  makeQuestion(
    "cme295-lect3-q132",
    "medium",
    "Which statements correctly compare decoding controls?",
    [
      [
        "Greedy decoding chooses the highest-probability token at the current step.",
        true,
      ],
      [
        "Top-k sampling samples after removing tokens outside the top-k set.",
        true,
      ],
      [
        "Top-p sampling samples after removing tokens outside a cumulative-probability nucleus.",
        true,
      ],
      [
        "Guided decoding filters invalid continuations before the next token is chosen.",
        true,
      ],
    ],
    "Greedy, top-k, top-p, and guided decoding are different ways to use or constrain next-token probabilities. Guided decoding filters invalid continuations before token selection rather than waiting to repair invalid text after it has been generated.",
  ),
  makeQuestion(
    "cme295-lect3-q133",
    "easy",
    "Which statement best describes context length, also called context window or window size?",
    [
      [
        "It is the number of tokens the model can process as the conditioning context for a generation step.",
        true,
      ],
      [
        "It is the number of parameters activated by a sparse MoE router.",
        false,
      ],
      [
        "It is the number of examples needed before zero-shot learning becomes few-shot learning.",
        false,
      ],
      ["It is the number of candidate paths kept by beam search.", false],
    ],
    "Context length is a token-capacity concept: it bounds how much prompt and generated text can be available to the model at a step. Expert activation count, number of examples, and beam width are different quantities with different effects.",
  ),
  makeQuestion(
    "cme295-lect3-q134",
    "hard",
    "Which statements about long context and context rot are correct?",
    [
      ["Longer context can increase compute and memory pressure.", true],
      [
        "Relevant information can become harder for the model to use as distracting tokens accumulate.",
        true,
      ],
      [
        "The answer being present in the context does not ensure the model retrieves and uses it well.",
        true,
      ],
      [
        "Adding tokens to the context improves retrieval quality by construction.",
        false,
      ],
    ],
    "Long context increases the amount of information available, but it also raises attention and memory costs and can make retrieval less reliable. Context rot refers to degradation in effective use of relevant information, not to a guarantee that more tokens improve performance.",
  ),
  makeQuestion(
    "cme295-lect3-q135",
    "easy",
    "Which statements correctly describe a prompt's main parts in the lecture's prompt-structure example?",
    [
      ["Context gives background information the model should use.", true],
      ["Instructions state the task to perform.", true],
      ["Input supplies the concrete user case or content to transform.", true],
      [
        "Constraints are unrelated to format, audience, or response requirements.",
        false,
      ],
    ],
    "Prompts can include context, instructions, input, and constraints. Constraints are precisely where format, audience, style, or other response requirements can be stated, so saying they are unrelated to those requirements reverses the prompt-structure idea.",
  ),
  makeQuestion(
    "cme295-lect3-q136",
    "medium",
    "Which statements correctly distinguish zero-shot and few-shot prompting?",
    [
      [
        "Zero-shot prompting gives the task instruction without example input-output pairs.",
        true,
      ],
      [
        "Few-shot prompting includes examples in the prompt to shape the model's behavior.",
        true,
      ],
      [
        "Few-shot prompting consumes context tokens and can increase cost or latency.",
        true,
      ],
      [
        "Both zero-shot and few-shot prompting condition a fixed model through context.",
        true,
      ],
    ],
    "Zero-shot and few-shot prompting both steer a fixed model through context rather than weight updates. Few-shot examples can improve task alignment but use context budget and inference compute, while zero-shot prompting relies on the instruction and the model's existing capabilities.",
  ),
  makeQuestion(
    "cme295-lect3-q137",
    "easy",
    "Which statement best describes in-context learning?",
    [
      [
        "The model changes its behavior based on information in the prompt while its weights remain fixed.",
        true,
      ],
      [
        "The optimizer performs gradient descent on the examples inside the prompt.",
        false,
      ],
      [
        "The router permanently assigns a prompt to a single MoE expert.",
        false,
      ],
      [
        "The KV cache stores examples from previous users and reuses them as training data.",
        false,
      ],
    ],
    "In-context learning is behavioral adaptation through the prompt context, not gradient-based training on the fly. It should not be confused with MoE routing or KV-cache reuse, which operate at different layers of the inference system.",
  ),
  makeQuestion(
    "cme295-lect3-q138",
    "medium",
    "Which statements about few-shot prompting tradeoffs are correct?",
    [
      ["Examples can clarify the desired input-output pattern.", true],
      ["Examples consume part of the available context window.", true],
      [
        "Few-shot examples reduce input-token count because demonstrations are stored outside the prompt.",
        false,
      ],
      [
        "Examples remove the need to evaluate performance on benchmark or task data.",
        false,
      ],
    ],
    "Few-shot examples are useful because they demonstrate the target behavior, but they are not free. They use context budget and add tokens to process, and they still require evaluation because prompt examples do not prove generalization.",
  ),
  makeQuestion(
    "cme295-lect3-q139",
    "medium",
    "Which statements correctly describe chain-of-thought prompting?",
    [
      [
        "It asks the model to produce intermediate reasoning before the final answer.",
        true,
      ],
      ["It can improve performance on reasoning-heavy tasks.", true],
      [
        "It increases generated token count compared with a terse answer.",
        true,
      ],
      [
        "It enforces logical correctness by verifying each reasoning step against ground truth.",
        false,
      ],
    ],
    "Chain-of-thought prompting encourages explicit intermediate reasoning and can improve reasoning task performance. It also costs more tokens and does not by itself verify that the reasoning is correct or grounded in truth.",
  ),
  makeQuestion(
    "cme295-lect3-q140",
    "hard",
    "Which statements correctly describe chain-of-thought interpretability limits?",
    [
      [
        "The visible reasoning can help debug where a response went wrong.",
        true,
      ],
      [
        "The reasoning trace can reveal that the model used a wrong date or premise from the prompt.",
        true,
      ],
      ["The extra tokens create a latency and cost tradeoff.", true],
      ["A fluent reasoning trace still needs answer-level evaluation.", true],
    ],
    "Reasoning traces can be useful for inspection and root-cause analysis, but they are not a proof of correctness. A model can produce plausible-looking reasoning with an incorrect premise or invalid step, so chain-of-thought should be evaluated by outcomes as well as by trace readability.",
  ),
  makeQuestion(
    "cme295-lect3-q141",
    "easy",
    "Which statement best describes self-consistency prompting?",
    [
      [
        "Generate several reasoning paths, parse their final answers, and aggregate the answers, often by majority vote.",
        true,
      ],
      [
        "Force the model to use the same hidden attention pattern for every sample.",
        false,
      ],
      [
        "Keep one greedy completion and ask a second model to rewrite it as JSON.",
        false,
      ],
      [
        "Average the MoE router probabilities across experts to pick a final token.",
        false,
      ],
    ],
    "Self-consistency operates at the level of multiple sampled responses: it compares final answers after several reasoning paths. It is distinct from attention, guided decoding, and MoE routing, which operate inside a single generation process.",
  ),
  makeQuestion(
    "cme295-lect3-q142",
    "medium",
    "Which statements about self-consistency tradeoffs are correct?",
    [
      [
        "Sampling multiple reasoning paths can make the final answer more robust.",
        true,
      ],
      ["Parsing or extracting the final answer is needed before voting.", true],
      [
        "The sampled paths must be generated serially because each path depends on the previous path's hidden state.",
        false,
      ],
      [
        "Self-consistency reduces total compute because it samples fewer tokens than a single direct answer.",
        false,
      ],
    ],
    "Self-consistency trades more generation work for robustness. The branches can be independent and parallelizable, but total compute and token usage generally increase because several completions are produced.",
  ),
  makeQuestion(
    "cme295-lect3-q143",
    "easy",
    "Which statement best explains the goal of KV caching during autoregressive inference?",
    [
      [
        "Store previous keys and values so later tokens can reuse them instead of recomputing them.",
        true,
      ],
      [
        "Store previous query vectors because future tokens attend through old queries.",
        false,
      ],
      [
        "Store final text responses so the model can skip tokenization next time.",
        false,
      ],
      [
        "Store MoE expert weights on the CPU so routing collapse disappears.",
        false,
      ],
    ],
    "During generation, the current token needs keys and values from previous tokens for attention. KV caching saves those key and value tensors, while previous queries are not the useful cached objects for computing the current token's attention over the prefix.",
  ),
  makeQuestion(
    "cme295-lect3-q144",
    "medium",
    "Which statements correctly describe KV caching?",
    [
      [
        "It avoids recomputing key and value projections for earlier tokens at each new decoding step.",
        true,
      ],
      [
        "It is most relevant during autoregressive inference rather than teacher-forced training over a whole sequence.",
        true,
      ],
      [
        "Its memory footprint is constant with respect to sequence length once the first token is cached.",
        false,
      ],
      [
        "It changes the target model's next-token distribution by approximating attention scores.",
        false,
      ],
    ],
    "KV caching is an exact reuse strategy for autoregressive inference: it stores tensors that would otherwise be recomputed. The tradeoff is memory, and that memory grows with generated or prompt sequence length rather than staying constant.",
  ),
  makeQuestion(
    "cme295-lect3-q145",
    "easy",
    "Which statements correctly explain why previous query vectors are not the main cached object in KV caching?",
    [
      [
        "The current token forms a new query that attends to previous keys and values.",
        true,
      ],
      ["Previous keys are needed because the current query scores them.", true],
      [
        "Previous values are needed because attention weights combine them into the current representation.",
        true,
      ],
      [
        "Previous queries are not the main cached object for computing the current token's attention over the prefix.",
        true,
      ],
    ],
    "In causal self-attention for a new token, the new representation supplies the query, while earlier tokens supply keys and values. This is why the cache focuses on K and V tensors rather than Q tensors from earlier positions.",
  ),
  makeQuestion(
    "cme295-lect3-q146",
    "medium",
    "Which statements correctly compare Multi-Head Attention (MHA), Grouped Query Attention (GQA), and Multi-Query Attention (MQA) for KV-cache size?",
    [
      [
        "MHA keeps separate key/value projections for each attention head.",
        true,
      ],
      ["GQA shares key/value projections across groups of query heads.", true],
      [
        "MQA is the extreme case where many query heads share one key/value set.",
        true,
      ],
      [
        "GQA can reduce KV-cache memory compared with full MHA by storing fewer distinct key/value heads.",
        true,
      ],
    ],
    "GQA and MQA reduce the number of key/value heads relative to full MHA, which reduces KV-cache storage and bandwidth. They do this by sharing K/V representations across query heads rather than duplicating separate K/V tensors for each head.",
  ),
  makeQuestion(
    "cme295-lect3-q147",
    "medium",
    "Which statements correctly describe why KV cache memory can limit serving throughput?",
    [
      [
        "Each active request may need cached keys and values for each generated or prompt token.",
        true,
      ],
      [
        "Longer contexts increase the amount of cache memory held per request.",
        true,
      ],
      [
        "KV cache memory disappears after the prompt is tokenized, before generation starts.",
        false,
      ],
      [
        "KV caching removes the need to store model weights during inference.",
        false,
      ],
    ],
    "KV cache memory grows with request length and concurrency, so it can become a bottleneck even when the model weights are already loaded. Caching saves computation, but the cache persists during generation and does not eliminate the memory required for model parameters.",
  ),
  makeQuestion(
    "cme295-lect3-q148",
    "hard",
    "Which statements correctly describe PagedAttention-style memory management?",
    [
      [
        "It stores KV cache blocks in smaller chunks rather than reserving one maximum-length contiguous region per request.",
        true,
      ],
      [
        "It uses a mapping from token positions to physical cache blocks.",
        true,
      ],
      [
        "It reduces wasted reserved memory when requests stop before the maximum context length.",
        true,
      ],
      [
        "It can reduce fragmentation by letting logical cache positions map to non-contiguous physical blocks.",
        true,
      ],
    ],
    "PagedAttention is a memory-management technique for KV cache storage, inspired by paging. It reduces fragmentation and waste by allocating fixed-size blocks and mapping logical positions to physical memory rather than reserving a single maximum-length block per request.",
  ),
  makeQuestion(
    "cme295-lect3-q149",
    "medium",
    "Which statement best distinguishes internal fragmentation from external fragmentation in the KV-cache serving example?",
    [
      [
        "Internal fragmentation is reserved-but-unused space inside an allocation, while external fragmentation is unusable gaps between allocations.",
        true,
      ],
      [
        "Internal fragmentation is the router choosing one expert, while external fragmentation is top-p choosing several tokens.",
        false,
      ],
      [
        "Internal fragmentation is caused by tokenization, while external fragmentation is caused by temperature scaling.",
        false,
      ],
      [
        "Internal fragmentation stores queries, while external fragmentation stores values.",
        false,
      ],
    ],
    "The memory-management issue is about how cache space is allocated. Internal fragmentation wastes space inside reserved regions, while external fragmentation leaves scattered gaps between regions that are hard to use efficiently.",
  ),
  makeQuestion(
    "cme295-lect3-q150",
    "hard",
    "Which statements correctly describe multi-latent attention as a KV-cache reduction idea?",
    [
      [
        "It stores a compressed latent representation instead of separate full key and value vectors for each head.",
        true,
      ],
      [
        "It factorizes projection work through a lower-dimensional intermediate representation.",
        true,
      ],
      [
        "It can share the compression representation across keys, values, or heads before decompression.",
        true,
      ],
      [
        "It keeps causal decoder behavior while changing the representation stored in the cache.",
        true,
      ],
    ],
    "Multi-latent attention reduces what must be stored by caching compact latent representations and later expanding them for attention use. It saves memory through representation design while causal decoder behavior remains part of generation.",
  ),
  makeQuestion(
    "cme295-lect3-q151",
    "medium",
    "Which statements correctly describe the difference between GQA and latent attention for cache efficiency?",
    [
      [
        "GQA reduces the number of key/value heads by sharing K/V projections across query-head groups.",
        true,
      ],
      [
        "Latent attention reduces the dimensional payload stored for keys and values through compression.",
        true,
      ],
      [
        "Both are decoding policies that rank candidate vocabulary tokens by probability.",
        false,
      ],
      [
        "GQA and latent attention are two names for the same top-k expert routing rule.",
        false,
      ],
    ],
    "GQA and latent attention both attack KV-cache cost, but one reduces the number of K/V heads while the other compresses the representation stored for K/V. Neither is a vocabulary decoding policy or an MoE routing rule.",
  ),
  makeQuestion(
    "cme295-lect3-q152",
    "medium",
    "Which statements about exact inference optimizations are correct?",
    [
      ["KV caching avoids redundant key/value computations.", true],
      [
        "PagedAttention improves memory allocation for cached keys and values.",
        true,
      ],
      [
        "GQA and MQA reduce duplicated K/V storage across attention heads.",
        true,
      ],
      [
        "Exact optimizations aim to improve latency or memory use without changing the target model semantics.",
        true,
      ],
    ],
    "Exact optimizations preserve the model computation or intended distribution while making it faster or more memory-efficient. KV reuse, cache paging, and key/value sharing attack redundant work or memory pressure rather than deliberately changing the target model semantics.",
  ),
  makeQuestion(
    "cme295-lect3-q153",
    "medium",
    "Which statement best explains why inference can be memory-bound in large decoder models?",
    [
      [
        "Moving model weights and KV-cache tensors through memory can dominate time compared with the arithmetic available on accelerators.",
        true,
      ],
      [
        "The tokenizer must read the whole internet before each generated token.",
        false,
      ],
      [
        "The loss function must backpropagate through the full training corpus at inference time.",
        false,
      ],
      [
        "Beam search disables accelerator arithmetic and runs every operation on the CPU.",
        false,
      ],
    ],
    "Large-model inference often spends much of its time moving weights and cache tensors rather than saturating arithmetic units. The other choices confuse inference with data collection, training backpropagation, or an inaccurate hardware story.",
  ),
  makeQuestion(
    "cme295-lect3-q154",
    "hard",
    "Which statements correctly classify the lecture's inference optimization families?",
    [
      ["Avoiding redundant work includes KV caching.", true],
      ["Memory management includes PagedAttention.", true],
      [
        "Reformulating attention storage includes GQA, MQA, and latent attention.",
        true,
      ],
      [
        "Approximate token-prediction methods include speculative decoding and multi-token prediction.",
        false,
      ],
    ],
    "The first three items are exact or near-exact efficiency themes around attention computation and memory. Speculative decoding can preserve the target distribution under its accept/reject rule, while multi-token prediction changes the training/inference setup; the important point is to separate cache reuse, memory layout, attention representation, and token-level acceleration.",
  ),
  makeQuestion(
    "cme295-lect3-q155",
    "easy",
    "Which statement best describes speculative decoding?",
    [
      [
        "A smaller draft model proposes tokens, and a larger target model validates them so generation can advance faster.",
        true,
      ],
      [
        "A larger model writes a prompt, and a smaller model grades it after deployment.",
        false,
      ],
      [
        "The target model skips probability computation and accepts every draft token by default.",
        false,
      ],
      [
        "Several MoE experts vote on the final answer after the response is finished.",
        false,
      ],
    ],
    "Speculative decoding uses a fast draft model to propose candidate tokens and the target model to check them. The target model still computes probabilities; the method is not prompt grading or a post-response expert vote.",
  ),
  makeQuestion(
    "cme295-lect3-q156",
    "hard",
    "Which statements about speculative decoding's accept/reject step are correct?",
    [
      [
        "If the target model assigns at least as much probability to the drafted token as the draft model did, the token can be accepted.",
        true,
      ],
      [
        "If the target probability is lower, acceptance can occur with probability based on the ratio of target to draft probability.",
        true,
      ],
      [
        "After a rejection, generation resumes from the rejected position with an adjusted distribution.",
        true,
      ],
      [
        "The accept/reject rule is designed to preserve the target model's output distribution.",
        true,
      ],
    ],
    "The accept/reject rule is designed so the accepted tokens match the target model distribution rather than merely imitating the draft model. Rejections trigger resampling from an adjusted distribution, which is why the method can be fast while preserving the target distribution in the formal algorithm.",
  ),
  makeQuestion(
    "cme295-lect3-q157",
    "medium",
    "Which statements correctly explain why speculative decoding can be faster?",
    [
      ["The draft model proposes several tokens cheaply.", true],
      [
        "The target model can evaluate the drafted tokens in a single forward pass.",
        true,
      ],
      [
        "When several draft tokens are accepted, the system advances multiple positions after one target-model pass.",
        true,
      ],
      ["The method is faster because the target model is never called.", false],
    ],
    "Speculative decoding still uses the target model; its speedup comes from batching validation of draft tokens and accepting several positions at once when the draft is good. The draft model is useful because it is smaller or cheaper, not because it replaces the target model completely.",
  ),
  makeQuestion(
    "cme295-lect3-q158",
    "medium",
    "Which statements correctly describe the roles of draft and target models in speculative decoding?",
    [
      ["The draft model proposes candidate continuations.", true],
      [
        "The target model supplies the probability distributions used for validation.",
        true,
      ],
      [
        "The draft model should be cheaper to run than the target model for the method to help.",
        true,
      ],
      [
        "The target model is trained from scratch during each speculative decoding request.",
        false,
      ],
    ],
    "The draft model is a cheap proposer and the target model is the distribution authority. No per-request training occurs; speculative decoding is an inference-time method.",
  ),
  makeQuestion(
    "cme295-lect3-q159",
    "medium",
    "Which statements correctly describe multi-token prediction (MTP)?",
    [
      [
        "The model is trained to predict multiple future tokens from a representation.",
        true,
      ],
      [
        "Extra prediction heads can act like an embedded draft mechanism at inference time.",
        true,
      ],
      [
        "MTP changes the training objective compared with ordinary next-token-only prediction.",
        true,
      ],
      ["MTP is identical to increasing top-k during sampling.", false],
    ],
    "MTP changes the model and objective so multiple future-token predictions are available. Increasing top-k is only a decoding distribution truncation choice and does not add multi-token heads or change the training objective.",
  ),
  makeQuestion(
    "cme295-lect3-q160",
    "hard",
    "Which statements correctly compare speculative decoding and multi-token prediction?",
    [
      [
        "Speculative decoding commonly uses a separate smaller draft model.",
        true,
      ],
      ["MTP can embed draft-like heads inside the same model.", true],
      [
        "Speculative decoding's formal accept/reject scheme can preserve the target distribution.",
        true,
      ],
      [
        "MTP changes the objective by asking the model to predict multiple future tokens.",
        true,
      ],
    ],
    "Speculative decoding and MTP both try to move through output tokens faster, but they do it differently. MTP changes the model objective by predicting multiple future tokens, whereas sampler thresholds such as top-k and top-p are separate decoding controls.",
  ),
  makeQuestion(
    "cme295-lect3-q161",
    "medium",
    "Which statement best explains why speculative decoding includes the token after the draft block in the target-model pass?",
    [
      [
        "Evaluating the drafted block also yields a distribution for the next position, which can be used if the draft tokens are accepted.",
        true,
      ],
      [
        "The extra token is needed to update the target model's weights before accepting the block.",
        false,
      ],
      [
        "The extra token lets the router rebalance MoE experts across the training batch.",
        false,
      ],
      ["The extra token stores the JSON grammar for guided decoding.", false],
    ],
    "A target-model pass over the drafted sequence can produce probability distributions for each drafted token and the following position. That extra distribution is useful when the system accepts the draft tokens and needs to continue generation.",
  ),
  makeQuestion(
    "cme295-lect3-q162",
    "medium",
    "Which statements correctly connect inference acceleration to output quality?",
    [
      [
        "Exact reuse methods such as KV caching aim to preserve the same model computation.",
        true,
      ],
      [
        "Speculative decoding's validation step aims to preserve the target model distribution.",
        true,
      ],
      [
        "Approximate methods must be evaluated for quality as well as speed.",
        true,
      ],
      [
        "Any speedup that lowers latency automatically improves answer correctness.",
        false,
      ],
    ],
    "Inference optimization is not just about lower latency; the quality and distribution of outputs still matter. Some methods are exact, some include validation to preserve a target distribution, and more approximate methods require empirical evaluation.",
  ),
  makeQuestion(
    "cme295-lect3-q163",
    "hard",
    "A serving system has short prompts, long generations, and repeated recomputation of old attention keys and values. Which optimization most directly addresses the repeated computation?",
    [
      ["KV caching.", true],
      ["Auxiliary MoE load balancing.", false],
      ["Few-shot prompting.", false],
      ["Temperature scaling.", false],
    ],
    "The repeated computation is specifically the old key and value projections needed by later decoding steps, so KV caching is the direct fix. MoE balancing, prompting examples, and temperature affect different parts of modeling or generation behavior.",
  ),
  makeQuestion(
    "cme295-lect3-q164",
    "hard",
    "A serving system reserves memory for the maximum context length for each request, but most requests finish early. Which statements identify the most relevant problem and remedy?",
    [
      [
        "The problem includes internal fragmentation from reserved but unused KV-cache slots.",
        true,
      ],
      ["PagedAttention-style block allocation is a relevant remedy.", true],
      [
        "A logical-to-physical block mapping can let the cache use non-contiguous memory.",
        true,
      ],
      [
        "Temperature scaling is the relevant remedy because it makes responses shorter by definition.",
        false,
      ],
    ],
    "The described waste is a KV-cache memory-allocation problem, not a probability-sharpness problem. PagedAttention attacks the waste by allocating and mapping smaller blocks instead of reserving one large contiguous maximum-length region.",
  ),
  makeQuestion(
    "cme295-lect3-q165",
    "medium",
    "Which statement best describes multi-query attention (MQA) as the endpoint of grouped key/value sharing?",
    [
      ["Many query heads share a single set of key and value heads.", true],
      [
        "Each query head has its own independent key and value projection, as in full MHA.",
        false,
      ],
      [
        "Each MoE expert receives a separate copy of the vocabulary softmax.",
        false,
      ],
      [
        "The model predicts multiple future tokens with separate output heads.",
        false,
      ],
    ],
    "MQA is the high-sharing endpoint of the MHA-GQA-MQA family: query heads remain multiple, but key/value heads are shared heavily. It should not be confused with MoE expert routing or multi-token prediction heads.",
  ),
  makeQuestion(
    "cme295-lect3-q166",
    "hard",
    "Which statements correctly reason about cache compression in latent attention?",
    [
      [
        "A lower-dimensional latent can be stored instead of full per-head key/value vectors.",
        true,
      ],
      [
        "Separate decompression matrices can recover key-like and value-like representations when needed.",
        true,
      ],
      [
        "Sharing the compressed representation across keys and values reduces duplicate cache payload.",
        true,
      ],
      [
        "Compression means attention no longer needs token representations from the prefix.",
        false,
      ],
    ],
    "Latent attention compresses what is stored, then reconstructs representations needed for attention. It does not remove the need for prefix information; it changes how that information is represented and cached.",
  ),
  makeQuestion(
    "cme295-lect3-q167",
    "medium",
    "Which statements correctly describe why very low-probability tokens are often filtered during sampling?",
    [
      [
        "They can create incoherent continuations if sampled despite tiny probability.",
        true,
      ],
      ["Top-k filtering removes candidates outside a fixed rank cutoff.", true],
      [
        "Top-p filtering removes candidates outside a probability-mass nucleus.",
        true,
      ],
      [
        "Filtering low-probability tokens turns sampling into beam search.",
        false,
      ],
    ],
    "Top-k and top-p sampling keep randomness while removing tail candidates that are unlikely to be useful. Beam search is a different search procedure over high-scoring paths, not just tail filtering followed by sampling.",
  ),
  makeQuestion(
    "cme295-lect3-q168",
    "medium",
    "Which statements correctly describe when guided decoding is useful?",
    [
      ["Generating machine-readable formats such as JSON.", true],
      [
        "Conforming to a grammar or schema while still using the model's probabilities.",
        true,
      ],
      [
        "Preventing invalid next-token choices during the generation process.",
        true,
      ],
      [
        "Pairing structural validity with semantic evaluation when factual correctness matters.",
        true,
      ],
    ],
    "Guided decoding is valuable for structural constraints, especially machine-readable output. It does not prove the semantic content is correct, so structurally valid responses still need semantic evaluation when the values or reasoning matter.",
  ),
  makeQuestion(
    "cme295-lect3-q169",
    "easy",
    "Which statement best identifies the main tradeoff of chain-of-thought prompting?",
    [
      [
        "It can improve reasoning behavior but uses more output tokens and increases cost or latency.",
        true,
      ],
      [
        "It reduces token usage by hiding intermediate reasoning inside the KV cache.",
        false,
      ],
      [
        "It replaces next-token prediction with supervised fine-tuning during inference.",
        false,
      ],
      [
        "It prevents arithmetic errors because every step is externally verified.",
        false,
      ],
    ],
    "Chain-of-thought prompting asks the model to produce intermediate reasoning, which can help on reasoning tasks. The cost is more generated text, and the reasoning is not automatically externally verified.",
  ),
  makeQuestion(
    "cme295-lect3-q170",
    "hard",
    "Which statements correctly describe how self-consistency uses parallel sampled paths?",
    [
      [
        "Several completions can be sampled independently from the same prompt.",
        true,
      ],
      [
        "The final answers can be extracted from those completions and compared.",
        true,
      ],
      [
        "Parallel sampling can keep latency closer to the slowest branch than to the sum of branch latencies.",
        true,
      ],
      [
        "Each branch must be appended to the context of the next branch before voting.",
        false,
      ],
    ],
    "Self-consistency samples multiple paths that do not depend on one another, then aggregates their final answers. Because the branches are independent, they can be run in parallel rather than chained into one another's contexts.",
  ),
  makeQuestion(
    "cme295-lect3-q171",
    "hard",
    "A team adds few-shot examples, chain-of-thought instructions, and self-consistency voting to a math prompt. Which statements correctly predict system-level effects?",
    [
      ["Few-shot examples increase input context length.", true],
      ["Chain-of-thought increases expected output length.", true],
      ["Self-consistency increases the number of sampled completions.", true],
      [
        "These changes increase total token work even though the model weights remain fixed.",
        true,
      ],
    ],
    "Prompting techniques can improve task behavior, but they often increase token processing and generation work. Prompt text is not free: examples add input tokens, reasoning adds output tokens, and self-consistency multiplies the number of completions.",
  ),
  makeQuestion(
    "cme295-lect3-q172",
    "medium",
    "Which statements correctly connect context length to prompting strategies?",
    [
      [
        "Few-shot prompting uses part of the context window for demonstrations.",
        true,
      ],
      [
        "Long instructions and constraints compete with source material for context budget.",
        true,
      ],
      [
        "Self-consistency usually creates separate sampled contexts rather than one ever-growing shared context.",
        true,
      ],
      [
        "A larger context window removes the need to design concise prompts.",
        false,
      ],
    ],
    "Context budget must be managed even when the model supports long windows, because cost, latency, and context-rot effects still matter. Few-shot examples and long instructions consume tokens, while self-consistency usually samples separate branches from the same prompt.",
  ),
  makeQuestion(
    "cme295-lect3-q173",
    "easy",
    "Which statement best describes the relationship between prompt constraints and guided decoding constraints?",
    [
      [
        "Prompt constraints are text instructions to the model, while guided decoding enforces allowed next-token choices during generation.",
        true,
      ],
      [
        "Prompt constraints and guided decoding both require retraining the model before each response.",
        false,
      ],
      [
        "Prompt constraints operate on model weights, while guided decoding operates on the training dataset.",
        false,
      ],
      [
        "Prompt constraints are used for JSON, while guided decoding is used for MoE load balancing.",
        false,
      ],
    ],
    "Prompt constraints ask the model to follow instructions, but they do not mechanically prevent invalid continuations. Guided decoding adds an inference-time token filter, which is why it is stronger for structural validity such as JSON.",
  ),
  makeQuestion(
    "cme295-lect3-q174",
    "medium",
    "Which statements correctly describe the output layer's role in generation and acceleration?",
    [
      [
        "The output layer produces token scores that decoding strategies convert into chosen tokens.",
        true,
      ],
      [
        "Speculative decoding tries to advance through output tokens faster by validating draft proposals.",
        true,
      ],
      [
        "Multi-token prediction modifies the model so multiple future tokens can be proposed from one representation.",
        true,
      ],
      [
        "KV-cache paging changes the vocabulary logits so fewer tokens need probabilities.",
        false,
      ],
    ],
    "Decoding strategies and token-prediction accelerators operate near the output-token process. PagedAttention is a memory-layout method for attention caches, not a way to reduce the vocabulary distribution directly.",
  ),
  makeQuestion(
    "cme295-lect3-q175",
    "medium",
    "Which statements correctly connect MoE and inference efficiency?",
    [
      [
        "Sparse MoE can reduce active feed-forward computation per token.",
        true,
      ],
      [
        "The model must still store or access the expert parameters needed for routing and execution.",
        true,
      ],
      ["Load balancing matters because unused experts waste capacity.", true],
      [
        "Sparse MoE is the same mechanism as KV caching because both skip previous tokens.",
        false,
      ],
    ],
    "Sparse MoE is a model-architecture efficiency idea for feed-forward computation, while KV caching is an inference reuse idea for attention tensors. Both can affect serving cost, but they operate on different parts of the model.",
  ),
  makeQuestion(
    "cme295-lect3-q176",
    "hard",
    "Which statements correctly compare active parameters, total parameters, and FLOPs in an MoE model?",
    [
      [
        "Total parameters include experts that are inactive for a particular token.",
        true,
      ],
      [
        "Active parameters are the subset used in that token's forward pass.",
        true,
      ],
      [
        "Per-token FLOPs depend more directly on active computation than on inactive stored experts.",
        true,
      ],
      [
        "A higher total parameter count can coexist with controlled per-token FLOPs when routing is sparse.",
        true,
      ],
    ],
    "MoE design decouples total parameter count from per-token active computation. A model can store many experts while routing a token through a subset, so total parameters alone do not determine the forward-pass FLOPs for that token.",
  ),
  makeQuestion(
    "cme295-lect3-q177",
    "medium",
    "Which statement best diagnoses a model that has many experts but sends nearly every token to expert 3?",
    [
      [
        "Routing collapse is occurring, so auxiliary balancing or noisy gating should be considered.",
        true,
      ],
      ["Top-p sampling is using a threshold that is too low.", false],
      ["KV caching is storing queries instead of values.", false],
      [
        "Self-consistency is sampling too many independent reasoning paths.",
        false,
      ],
    ],
    "The symptom is expert-utilization imbalance inside an MoE layer, which is routing collapse. Sampling thresholds, cache contents, and self-consistency branch counts are different mechanisms with different failure modes.",
  ),
  makeQuestion(
    "cme295-lect3-q178",
    "medium",
    "Which statements correctly identify lecture concepts that preserve model weights during inference?",
    [
      [
        "Prompting changes behavior through input context rather than weight updates.",
        true,
      ],
      [
        "Guided decoding constrains token choices without retraining the model.",
        true,
      ],
      ["KV caching reuses stored tensors without modifying parameters.", true],
      [
        "Speculative decoding validates draft tokens by fine-tuning the target model online.",
        false,
      ],
    ],
    "Most inference-time controls discussed here operate without changing model weights. Speculative decoding uses draft and target model probabilities during generation; it does not fine-tune the target model for each request.",
  ),
  makeQuestion(
    "cme295-lect3-q179",
    "easy",
    "Which statement best summarizes the lecture's view of LLM system design?",
    [
      [
        "Modern LLM systems combine scale, sparse capacity, decoding controls, prompting, and inference optimizations to manage quality and cost.",
        true,
      ],
      [
        "The main design rule is to maximize parameter count without considering active compute or memory.",
        false,
      ],
      [
        "The main design rule is to use greedy decoding for every open-ended generation task.",
        false,
      ],
      [
        "The main design rule is to replace prompting with retraining for every user request.",
        false,
      ],
    ],
    "The lecture connects several layers of design: model scale, MoE capacity, decoding behavior, prompt conditioning, and serving optimizations. Treating any one knob as the whole design misses the tradeoffs among quality, latency, memory, and compute.",
  ),
  makeQuestion(
    "cme295-lect3-q180",
    "easy",
    "Which statements belong together as examples of inference-time techniques rather than pretraining-data changes?",
    [
      [
        "Temperature, top-k, and top-p adjust token selection during generation.",
        true,
      ],
      ["Guided decoding filters valid next tokens during generation.", true],
      [
        "KV caching and PagedAttention improve reuse and memory handling during serving.",
        true,
      ],
      [
        "Few-shot prompting and self-consistency condition or aggregate inference-time responses.",
        true,
      ],
    ],
    "These techniques operate while using a trained model: they control decoding, constrain output structure, reuse inference tensors, manage cache memory, or change how prompts and samples are used. They are distinct from changing the pretraining corpus or retraining the model from scratch.",
  ),
];
