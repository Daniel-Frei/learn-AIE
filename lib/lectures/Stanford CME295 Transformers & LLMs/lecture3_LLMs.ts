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
    throw new Error("CME295 Lecture 3 question " + id + " needs 4 options.");
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
  // LLM overview (6): working definition, scale, and decoder-only behavior.
  makeQuestion(
    "cme295-lect3-q181",
    "easy",
    "Which properties are part of the working definition of a modern large language model (LLM) used here?",
    [
      [
        "It assigns probabilities to token sequences and generates autoregressively with a decoder-only Transformer.",
        true,
      ],
      [
        "Its scale reflects parameters, training tokens, and the compute used to train or serve it.",
        true,
      ],
      [
        "It must contain an encoder that turns every input into one fixed vector before generation.",
        false,
      ],
      [
        "It must use cross-attention to a separate encoder stream for every output token.",
        false,
      ],
    ],
    "The working convention combines a language-modeling objective with a large decoder-only Transformer and substantial model, data, and compute scale. Encoder-only representation models and encoder-decoder models remain useful Transformer families, but neither of the last two properties is required by this lecture's narrower LLM definition.",
  ),
  makeQuestion(
    "cme295-lect3-q182",
    "easy",
    "A decoder-only language model assigns probabilities one token at a time. Which expression gives the probability of a sequence \\(w_1,\\ldots,w_T\\) after a beginning-of-sequence token?",
    [
      ["\\(\\prod_{t=1}^{T} P(w_t\\mid w_{<t})\\)", true],
      ["\\(\\sum_{t=1}^{T} P(w_t\\mid w_{<t})\\)", false],
      ["\\(\\prod_{t=1}^{T} P(w_{t-1}\\mid w_t)\\)", false],
      ["\\(P(w_T\\mid w_1)\\) regardless of the intervening prefix", false],
    ],
    "The chain rule factorizes a sequence probability into the product of next-token probabilities conditioned on the complete earlier prefix. Adding the conditionals does not produce a sequence probability, reversing the conditionals changes the modeled direction, and conditioning only on the first token discards most of the autoregressive context.",
  ),
  makeQuestion(
    "cme295-lect3-q183",
    "easy",
    "A Transformer is being converted from an encoder-decoder translation model into a decoder-only text generator. Which architectural consequences follow?",
    [
      [
        "Causal self-attention remains so a position cannot use future target tokens.",
        true,
      ],
      [
        "The decoder feed-forward sublayers, residual paths, and normalization remain part of the stack.",
        true,
      ],
      [
        "Cross-attention to encoder outputs is removed because there is no separate encoder stream.",
        true,
      ],
      [
        "An output projection still converts decoder hidden states into vocabulary-token scores.",
        true,
      ],
    ],
    "A decoder-only model keeps causal self-attention, feed-forward and residual machinery, normalization, and a vocabulary projection, while removing cross-attention that expected a separate encoder representation. The causal mask still prevents a generated position from reading future target tokens unavailable at inference time.",
  ),
  makeQuestion(
    "cme295-lect3-q184",
    "medium",
    "A team compares three pretrained Transformers: Model A is encoder-only and produces contextual embeddings, Model B is encoder-decoder and maps text to text, and Model C is decoder-only and generates from a causal prefix. Which classifications are accurate under the working convention?",
    [
      [
        "Model C is the clearest match for the modern text-generating LLM category.",
        true,
      ],
      [
        "Model A resembles the BERT family and is naturally suited to representation or classification tasks.",
        true,
      ],
      ["Model B must be decoder-only because its output is text.", false],
      [
        "Model A becomes autoregressive merely by adding a classification head to its pooled embedding.",
        false,
      ],
    ],
    "Output modality alone does not identify the Transformer family: an encoder-decoder model can also produce text, while the working LLM convention emphasizes a causal decoder-only backbone. Adding a classifier to an encoder-only model changes the downstream task head but does not turn bidirectional encoding into autoregressive next-token generation.",
  ),
  makeQuestion(
    "cme295-lect3-q185",
    "medium",
    "A sparse model is advertised as a very large language model. Which measurements are needed to interpret that claim responsibly?",
    [
      [
        "Total stored parameter count, because it measures the model's full capacity pool.",
        true,
      ],
      [
        "Active parameters per token, because sparse routing may use only part of that pool.",
        true,
      ],
      [
        "Pretraining-token scale, because model size alone does not describe how much data shaped the weights.",
        true,
      ],
      [
        "Training and serving compute, because equal parameter counts can have different operational costs.",
        true,
      ],
    ],
    "The word large compresses several distinct dimensions: stored capacity, active computation, data scale, and hardware work. Sparse Mixture-of-Experts models make the distinction especially important because a huge expert pool can coexist with a much smaller active path for each token.",
  ),
  makeQuestion(
    "cme295-lect3-q186",
    "hard",
    "For the sequence [BOS] A teddy [EOS], a model gives \\(P(A\\mid[BOS])=0.5\\), \\(P(teddy\\mid[BOS],A)=0.2\\), and \\(P([EOS]\\mid[BOS],A,teddy)=0.4\\). What sequence probability does the autoregressive factorization assign?",
    [
      ["\\(0.5\\times0.2\\times0.4=0.04\\)", true],
      ["\\(0.5+0.2+0.4=1.10\\)", false],
      ["\\((0.5+0.2+0.4)/3\\approx0.367\\)", false],
      ["\\(0.4/0.2=2.0\\)", false],
    ],
    "The joint probability of an autoregressive sequence is the product of the conditional next-token probabilities, so the result is \\(0.04\\). The sum and average do not represent the probability of all three choices occurring in sequence, while the ratio compares two conditionals without applying the chain rule.",
  ),

  // Mixture of Experts (9): routing, active capacity, and collapse.
  makeQuestion(
    "cme295-lect3-q187",
    "easy",
    "Where does a Mixture-of-Experts (MoE) module usually enter a decoder block, and what new component chooses its computation path?",
    [
      [
        "Several expert feed-forward networks replace or augment the ordinary feed-forward sublayer.",
        true,
      ],
      [
        "A learned router scores experts from the current token representation.",
        true,
      ],
      [
        "The tokenizer selects one expert before it has produced token representations.",
        false,
      ],
      [
        "Beam search assigns complete output sequences to attention heads inside the block.",
        false,
      ],
    ],
    "Modern Transformer MoE layers normally target the parameter-heavy feed-forward network and add a learned gate or router for token-level expert selection. Tokenization and beam search occur at different system layers, so they cannot perform the representation-dependent routing described by the MoE computation.",
  ),
  makeQuestion(
    "cme295-lect3-q188",
    "easy",
    "Which comparisons between dense and sparse Mixture-of-Experts computation are correct?",
    [
      ["A dense MoE can combine contributions from the full expert set.", true],
      [
        "A sparse MoE uses top-k routing so only selected experts process a token.",
        true,
      ],
      [
        "Both forms use learned gate values to determine expert contributions.",
        true,
      ],
      [
        "Sparse routing means one expert is chosen once for the entire prompt and for every layer.",
        false,
      ],
    ],
    "Dense and sparse MoE differ in how many expert outputs remain active in the weighted combination, not in whether routing is learned. Sparse routing is typically token- and layer-dependent, so different tokens and different layers may use different expert subsets rather than sharing one prompt-level choice.",
  ),
  makeQuestion(
    "cme295-lect3-q189",
    "medium",
    "A dense MoE has scalar expert outputs \\(E_1(x)=4\\), \\(E_2(x)=1\\), and \\(E_3(x)=-2\\), with gate weights \\((0.7,0.2,0.1)\\). What is the combined output \\(\\sum_i G(x)_iE_i(x)\\)?",
    [
      ["\\(2.8\\)", true],
      ["\\(3.0\\)", false],
      ["\\(1.0\\)", false],
      ["\\(0.7\\)", false],
    ],
    "The weighted sum is \\(0.7(4)+0.2(1)+0.1(-2)=2.8\\). The nearby distractors result from ignoring the negative expert, averaging the raw expert outputs without the gate, or reporting a gate weight instead of the gated expert combination.",
  ),
  makeQuestion(
    "cme295-lect3-q190",
    "medium",
    "An MoE layer has eight 2-billion-parameter experts and routes each token to the top two experts. Ignore shared non-expert parameters. Which statements follow for one token at this layer?",
    [
      ["The stored expert pool contains 16 billion parameters.", true],
      ["The active expert path uses 4 billion parameters.", true],
      ["Six experts do not execute for that token.", true],
      [
        "Adding experts can raise total capacity without making per-token expert compute grow in direct proportion.",
        true,
      ],
    ],
    "Eight experts of 2 billion parameters produce a 16-billion-parameter pool, while top-two routing activates 4 billion expert parameters for this token. The other six experts remain available for other tokens but do not execute here, which is the capacity-versus-active-compute separation that motivates sparse MoE designs.",
  ),
  makeQuestion(
    "cme295-lect3-q191",
    "medium",
    "In one decoder layer, the tokens 'The', 'bear', and 'reads' are routed to experts 2, 5, and 2. Which inferences are justified?",
    [
      [
        "The routing decision is fine-grained enough for tokens in one sequence to take different expert paths.",
        true,
      ],
      ["Expert 2 is active for two token computations in this layer.", true],
      [
        "The complete prompt was permanently assigned to expert 2 before tokenization.",
        false,
      ],
      [
        "Expert 2 in every later decoder layer must be the same network and receive the same tokens.",
        false,
      ],
    ],
    "The trace demonstrates token-level routing within one specific layer and shows two routed token computations for expert 2 there. Expert identities are local to their layer, and later routers receive different hidden representations, so neither a fixed prompt assignment nor identical downstream routing follows.",
  ),
  makeQuestion(
    "cme295-lect3-q192",
    "hard",
    "A four-expert router sends 80% of a mini-batch to one expert. Which facts make the auxiliary load-balancing objective relevant?",
    [
      [
        "\\(f_i\\) records the fraction of batch tokens actually routed to expert \\(i\\).",
        true,
      ],
      [
        "\\(P_i\\) records the average routing probability assigned to expert \\(i\\).",
        true,
      ],
      [
        "Penalizing concentrated routing statistics can push the learned router toward broader expert use.",
        true,
      ],
      [
        "The auxiliary term replaces next-token prediction, so language modeling is no longer optimized.",
        false,
      ],
    ],
    "The balancing term is computed from routing frequency and routing-probability statistics over a training batch, making it sensitive to the described concentration. It supplements the main language-modeling loss; removing next-token learning would defeat the purpose of training a capable MoE language model.",
  ),
  makeQuestion(
    "cme295-lect3-q193",
    "hard",
    "A router repeatedly sends nearly every token to expert 1 while seven experts receive almost no training signal. Which analyses are appropriate?",
    [
      [
        "This is routing collapse because expert capacity exists but is not being used.",
        true,
      ],
      [
        "A load-balancing loss can add gradient pressure against the concentrated routing pattern.",
        true,
      ],
      [
        "Noisy gating can occasionally expose less-used experts to tokens while routing is learned.",
        true,
      ],
      [
        "Balanced utilization alone would not prove that each expert learned a clean human-interpretable specialty.",
        true,
      ],
    ],
    "The symptom is routing collapse, and both auxiliary balancing and noisy gating are plausible training remedies discussed for increasing expert participation. Utilization is only an operational signal: even a well-balanced router does not by itself establish that experts correspond to neat semantic categories or that routing quality is optimal.",
  ),
  makeQuestion(
    "cme295-lect3-q194",
    "hard",
    "For four experts, suppose the routing frequencies equal the average routing probabilities, so \\(f=P\\). Ignoring the common constants in the auxiliary loss, which candidate has the smallest value of \\(\\sum_i f_iP_i\\)?",
    [
      ["\\((0.25,0.25,0.25,0.25)\\)", true],
      ["\\((0.40,0.30,0.20,0.10)\\)", false],
      ["\\((0.50,0.20,0.20,0.10)\\)", false],
      ["\\((1.00,0.00,0.00,0.00)\\)", false],
    ],
    "When \\(f=P\\), the relevant sum is the sum of squared shares. The uniform candidate gives \\(4(0.25^2)=0.25\\), which is lower than \\(0.30\\), \\(0.34\\), and \\(1.00\\) for the increasingly concentrated alternatives, illustrating why the term favors balanced use.",
  ),
  makeQuestion(
    "cme295-lect3-q195",
    "hard",
    "A visualization colors each token by the expert selected in decoder layer 0, and the sentence contains several colors. Which conclusions are supported?",
    [
      [
        "The shown sentence did not route every token to one expert in that layer.",
        true,
      ],
      [
        "The plot is a layer-specific routing trace rather than a map of the entire model.",
        true,
      ],
      [
        "Each color proves that its expert has one stable, human-nameable linguistic function across all inputs.",
        false,
      ],
      [
        "One varied sentence proves that expert usage is balanced over the complete training distribution.",
        false,
      ],
    ],
    "Multiple colors rule out complete single-expert routing for this particular sentence and layer, and the layer label limits the trace's scope. Semantic specialization and population-level balance require broader analysis; neither follows from a single qualitative visualization.",
  ),

  // Response generation (18): autoregression, search, sampling, and constraints.
  makeQuestion(
    "cme295-lect3-q196",
    "easy",
    "A decoder has generated the prefix 'A teddy bear'. Which statements describe the next autoregressive step?",
    [
      [
        "The complete prefix is used to produce scores for the next vocabulary token.",
        true,
      ],
      [
        "A decoding rule converts the next-token distribution into a chosen token.",
        true,
      ],
      [
        "The chosen token is appended before the following prediction step.",
        true,
      ],
      [
        "The model must update its trained weights after appending each token.",
        false,
      ],
    ],
    "Autoregressive inference repeatedly scores a next token from the available prefix, chooses a token through a decoding policy, and appends it to extend the conditioning context. Ordinary generation keeps the trained parameters fixed; changing weights after every token would be online training rather than decoding.",
  ),
  makeQuestion(
    "cme295-lect3-q197",
    "medium",
    "At the first step, token A has probability 0.6 and token B has probability 0.4. The best continuation after A has conditional probability 0.5, while the best continuation after B has probability 0.9. Which statement is correct for the best two-token path?",
    [
      [
        "The B path is better because \\(0.4\\times0.9=0.36\\), exceeding the A path's \\(0.6\\times0.5=0.30\\).",
        true,
      ],
      [
        "The A path is necessarily better because greedy decoding chose the larger first-step probability.",
        false,
      ],
      [
        "The paths tie because their first-step probabilities sum to one.",
        false,
      ],
      [
        "The B path has probability \\(0.4+0.9=1.3\\) because sequence scores add raw probabilities.",
        false,
      ],
    ],
    "A path probability multiplies conditional token probabilities, so the locally weaker first token can still lead to the stronger complete path. Greedy decoding cannot recover that alternative after committing to A, which illustrates why a locally optimal token does not guarantee a globally better sequence.",
  ),
  makeQuestion(
    "cme295-lect3-q198",
    "easy",
    "Which comparisons among greedy decoding, beam search, and sampling are accurate?",
    [
      [
        "Greedy decoding commits to the highest-probability token at each step.",
        true,
      ],
      ["Beam search retains several high-scoring partial sequences.", true],
      [
        "Sampling draws from an allowed probability distribution and can produce different outputs.",
        true,
      ],
      [
        "Beam search spends additional computation to explore more paths than greedy decoding.",
        true,
      ],
    ],
    "The three methods differ in how they turn next-token probabilities into a continuation: one local maximum, several scored paths, or a random draw. Beam search broadens likelihood-oriented search but costs more, while sampling is the mechanism among the three that directly introduces controlled output diversity.",
  ),
  makeQuestion(
    "cme295-lect3-q199",
    "hard",
    "A width-2 beam expands two prefixes. The joint scores of the four new paths are \\(P(Ax)=0.24\\), \\(P(Ay)=0.36\\), \\(P(Bx)=0.36\\), and \\(P(By)=0.04\\). Which paths remain after pruning, assuming ties are allowed?",
    [
      ["\\(Ay\\)", true],
      ["\\(Bx\\)", true],
      ["\\(Ax\\)", false],
      ["\\(By\\)", false],
    ],
    "A width-2 beam keeps the two highest-scoring partial sequences across all expansions, which are \\(Ay\\) and \\(Bx\\), both at \\(0.36\\). It does not reserve one survivor per parent prefix, so \\(Ax\\) is pruned despite descending from A, and \\(By\\) is far below the cutoff.",
  ),
  makeQuestion(
    "cme295-lect3-q200",
    "medium",
    "A speech-recognition system prefers a highly likely completed transcript and can afford extra decoding work. Which observations support beam search over greedy decoding?",
    [
      [
        "Keeping several partial transcripts delays commitment to one early token.",
        true,
      ],
      [
        "Sequence-level scores can rescue a path whose first token was not locally best.",
        true,
      ],
      [
        "The beam width trades additional compute and memory for broader search.",
        true,
      ],
      [
        "Beam search is preferable because it samples more tail tokens and therefore maximizes creativity.",
        false,
      ],
    ],
    "Beam search is useful when exploring several high-likelihood candidates matters more than generating diverse creative continuations. Its extra state and expansions cost resources, and it remains likelihood-oriented rather than deliberately drawing low-probability tail tokens.",
  ),
  makeQuestion(
    "cme295-lect3-q201",
    "easy",
    "A next-token distribution is A: 0.40, B: 0.30, C: 0.20, and D: 0.10. With top-k sampling at \\(k=2\\), which candidate set is sampled after filtering?",
    [
      ["A and B", true],
      ["A, B, and C", false],
      ["C and D", false],
      ["A only", false],
    ],
    "Top-k filtering uses probability rank, so the two highest-probability tokens A and B remain. The probabilities of the retained tokens are then renormalized for sampling; cumulative mass is not the rule for deciding the set size in top-k.",
  ),
  makeQuestion(
    "cme295-lect3-q202",
    "hard",
    "A distribution is A: 0.50, B: 0.30, C: 0.10, and D: 0.10. After top-k filtering with \\(k=2\\), which renormalized probabilities are correct?",
    [
      ["\\(P'(A)=0.50/(0.50+0.30)=0.625\\)", true],
      ["\\(P'(B)=0.30/(0.50+0.30)=0.375\\)", true],
      ["\\(P'(C)=0.10/(0.50+0.30)=0.125\\)", false],
      ["\\(P'(D)=0.10/(0.50+0.30)=0.125\\)", false],
    ],
    "Only A and B survive the rank cutoff, so their original mass of \\(0.80\\) is rescaled to one, giving \\(0.625\\) and \\(0.375\\). C and D are excluded before sampling and therefore receive zero probability, not a renormalized share.",
  ),
  makeQuestion(
    "cme295-lect3-q203",
    "medium",
    "A sorted distribution is A: 0.45, B: 0.30, C: 0.15, and D: 0.10. For top-p sampling with \\(p=0.80\\), which resulting nucleus and renormalized distribution are correct?",
    [
      [
        "Keep A, B, C and use probabilities \\((0.50,0.333\\ldots,0.166\\ldots)\\).",
        true,
      ],
      ["Keep A, B and use probabilities \\((0.60,0.40)\\).", false],
      [
        "Keep A, B, C and use the unchanged probabilities \\((0.45,0.30,0.15)\\).",
        false,
      ],
      [
        "Keep A, B, C, D and use the original distribution because 0.80 is not a token count.",
        false,
      ],
    ],
    "A and B reach only 0.75, so C is required; those three tokens reach 0.90 and D is excluded. Dividing their probabilities by 0.90 gives \\((0.50,0.333\\ldots,0.166\\ldots)\\); leaving the mass at 0.90 would not define a normalized sampling distribution.",
  ),
  makeQuestion(
    "cme295-lect3-q204",
    "hard",
    "Two decoding steps both use top-p with \\(p=0.80\\). Step 1 has sorted probabilities \\((0.75,0.15,0.06,0.04)\\); step 2 has \\((0.30,0.25,0.20,0.15,0.10)\\). Which conclusions follow?",
    [
      ["Step 1 needs its first two tokens to reach at least 0.80.", true],
      ["Step 2 needs its first four tokens to reach at least 0.80.", true],
      [
        "The candidate count changes because top-p adapts to the shape of each distribution.",
        true,
      ],
      [
        "The more diffuse second distribution retains more candidates at this threshold.",
        true,
      ],
    ],
    "The first distribution reaches \\(0.90\\) after two tokens, while the second reaches only \\(0.75\\) after three and therefore needs a fourth token to reach \\(0.90\\). This is the defining adaptive behavior of nucleus sampling: the same mass threshold can create different set sizes as confidence changes.",
  ),

  makeQuestion(
    "cme295-lect3-q205",
    "easy",
    "What is the direct effect of lowering temperature below 1 before applying softmax to fixed logits?",
    [
      [
        "Probability mass becomes more concentrated on the higher-logit tokens.",
        true,
      ],
      ["The candidate set is truncated to a fixed number of tokens.", false],
      [
        "A grammar removes tokens that would make the output structurally invalid.",
        false,
      ],
      [
        "The model's learned weights are updated to prefer shorter responses.",
        false,
      ],
    ],
    "Temperature divides the logits before softmax, so a value below one enlarges logit differences and sharpens the resulting distribution. Fixed-rank truncation belongs to top-k, structural filtering belongs to guided decoding, and no parameter update occurs during this inference-time rescaling.",
  ),
  makeQuestion(
    "cme295-lect3-q206",
    "medium",
    "Two tokens have logits 2 and 0. Their probability odds after temperature scaling are \\(\\exp((2-0)/T)\\). Which comparisons are correct?",
    [
      ["At \\(T=1\\), the odds are \\(e^2\\).", true],
      ["At \\(T=2\\), the odds are \\(e^1\\).", true],
      [
        "Raising \\(T\\) from 1 to 2 increases the odds in favor of the higher-logit token.",
        false,
      ],
      [
        "At \\(T=0.5\\), the odds become \\(e^1\\) because lower temperature flattens the distribution.",
        false,
      ],
    ],
    "Dividing the logit gap by temperature gives gaps of 2 and 1 at temperatures 1 and 2, respectively. Higher temperature reduces the odds ratio and flattens the distribution, while \\(T=0.5\\) would produce a gap of 4 and odds of \\(e^4\\), making the distribution sharper.",
  ),
  makeQuestion(
    "cme295-lect3-q207",
    "hard",
    "At temperature \\(T=1\\), three token logits are \\((\\ln 4,\\ln 2,0)\\). Which softmax probability vector is correct?",
    [
      ["\\((4/7,2/7,1/7)\\)", true],
      ["\\((4/6,2/6,0)\\)", false],
      ["\\((\\ln4/(\\ln4+\\ln2),\\ln2/(\\ln4+\\ln2),0)\\)", false],
      ["\\((1/2,1/3,1/6)\\)", false],
    ],
    "Exponentiating the logits yields unnormalized weights \\((4,2,1)\\), whose sum is 7, so the normalized probabilities are \\((4/7,2/7,1/7)\\). Softmax depends on exponentiated scores and their common normalizer, not on treating the logarithms themselves as linearly proportional probabilities.",
  ),
  makeQuestion(
    "cme295-lect3-q208",
    "easy",
    "A service must emit JSON that matches a known schema. Which statements describe guided decoding?",
    [
      [
        "A state derived from the schema determines which next tokens can still form a valid output.",
        true,
      ],
      [
        "Structurally invalid tokens can be masked before the next token is selected.",
        true,
      ],
      [
        "The model's probabilities can still rank or sample among the remaining valid tokens.",
        true,
      ],
      [
        "The constraint can be applied at inference time without retraining the language model.",
        true,
      ],
    ],
    "Guided decoding intersects the model's next-token possibilities with those allowed by a grammar or schema state, then applies a selection rule within that valid set. It is stronger than merely requesting JSON in prose because invalid structural continuations are removed during generation, yet it does not require changing the model weights.",
  ),
  makeQuestion(
    "cme295-lect3-q209",
    "medium",
    'A JSON generator has already emitted an opening brace followed by the property name "age". The schema now requires the key-value separator. Which next-token action best matches guided decoding?',
    [
      [
        "Allow a colon token and mask tokens that cannot begin the required separator.",
        true,
      ],
      [
        "Keep every vocabulary token and repair the structure only after the response is complete.",
        false,
      ],
      [
        "Select the globally highest-probability token even if it closes the object before supplying a value.",
        false,
      ],
      [
        "Change top-k to the vocabulary size so the grammar has more choices.",
        false,
      ],
    ],
    "The parser state identifies the colon as the next required structural element, so guided decoding masks incompatible continuations before selection. Post-hoc repair and unconstrained maximum-probability decoding can still produce invalid structure, while increasing top-k does nothing to encode the JSON grammar.",
  ),
  makeQuestion(
    "cme295-lect3-q210",
    "hard",
    "A guided decoder returns valid JSON with fields for a patient's age and medication, but the medication is factually wrong. Which conclusions are warranted?",
    [
      [
        "The decoder successfully enforced syntax but did not guarantee semantic correctness.",
        true,
      ],
      [
        "A separate factual, retrieval, or domain validation step is still needed.",
        true,
      ],
      [
        "The valid braces and field types prove that the underlying claim came from reliable evidence.",
        false,
      ],
      [
        "Replacing the grammar with greedy decoding would verify the medication value.",
        false,
      ],
    ],
    "A grammar can guarantee that a continuation belongs to a structural language such as a JSON schema, but it cannot establish that the generated values are true or safe. Semantic validation needs evidence-aware checks, and changing the unconstrained token-selection rule does not create that evidence.",
  ),
  makeQuestion(
    "cme295-lect3-q211",
    "medium",
    "A decoder applies both a JSON grammar and top-p sampling at one step. Which statements correctly describe their interaction?",
    [
      [
        "The grammar can remove a high-probability token if that token would violate the current JSON state.",
        true,
      ],
      [
        "Top-p can then form a probability-mass nucleus among candidates that remain valid.",
        true,
      ],
      [
        "The surviving candidates must be renormalized before a probability draw.",
        true,
      ],
      [
        "The grammar and top-p are equivalent because both keep a fixed number of highest-ranked tokens.",
        false,
      ],
    ],
    "Grammar validity and nucleus probability are independent filters: one is structural, while the other adapts to probability mass. After their constraints are applied, the remaining weights need normalization for sampling; neither procedure implies a fixed candidate count.",
  ),
  makeQuestion(
    "cme295-lect3-q212",
    "easy",
    "Which observations correctly separate probability shaping from nondeterministic token choice?",
    [
      [
        "Sampling introduces randomness by drawing from the allowed token distribution.",
        true,
      ],
      [
        "Greedy decoding is deterministic for fixed logits and fixed tie-breaking.",
        true,
      ],
      [
        "Hardware or numerical effects can still perturb logits in some nominally fixed inference pipelines.",
        true,
      ],
      [
        "Temperature reshapes probabilities, but randomness appears only when the subsequent selection rule samples.",
        true,
      ],
    ],
    "Temperature changes the distribution but does not itself specify whether to draw randomly or take an argmax. Greedy selection is deterministic under fixed numerical results and tie behavior, whereas sampling is explicitly random; real hardware kernels can add a separate source of small numerical nondeterminism.",
  ),
  makeQuestion(
    "cme295-lect3-q213",
    "hard",
    "A creative-writing API must return schema-valid JSON, offer diverse story ideas, and reject outputs whose cited facts are unsupported. Which design best addresses all three requirements?",
    [
      [
        "Use guided decoding for the schema, probability sampling for diversity, and a separate evidence check for factual claims.",
        true,
      ],
      [
        "Use greedy decoding alone because the highest-probability token guarantees valid JSON and factual support.",
        false,
      ],
      [
        "Use beam search alone because retaining several paths proves that the final fields are grounded.",
        false,
      ],
      [
        "Use a very low temperature alone because a sharp distribution enforces the schema and validates citations.",
        false,
      ],
    ],
    "The requirements live at three different layers: grammatical constraints handle structure, stochastic decoding creates diversity, and evidence validation handles truth. No single likelihood-oriented decoding knob supplies all three guarantees, so collapsing the design to greedy, beam search, or temperature leaves important requirements unmet.",
  ),

  // Prompting strategies (5): context budgeting and in-context behavior.
  makeQuestion(
    "cme295-lect3-q214",
    "easy",
    "Which statements correctly interpret context length, context size, or window size?",
    [
      [
        "They refer to the token budget the model can condition on for a generation step.",
        true,
      ],
      [
        "Prompt demonstrations and the user's input consume part of that token budget.",
        true,
      ],
      [
        "Longer windows can increase attention compute and KV-cache memory even when the model weights are unchanged.",
        true,
      ],
      [
        "Context rot means adding more tokens can make relevant material harder to use.",
        true,
      ],
    ],
    "The three terms describe the available conditioning token budget, which must hold instructions, examples, source material, and other prompt content. More capacity has real compute and cache costs and does not guarantee perfect utilization; distracting or very long inputs can produce context-rot effects.",
  ),
  makeQuestion(
    "cme295-lect3-q215",
    "easy",
    "A prompt says: 'My teddy bear had a long day. Write a bedtime story. Location: a forest library. Use fewer than 200 words.' Which role does each part play?",
    [
      ["'My teddy bear had a long day' supplies background context.", true],
      ["'Write a bedtime story' supplies the instruction.", true],
      ["'Location: a forest library' supplies the concrete input.", true],
      ["'Use fewer than 200 words' supplies a constraint.", true],
    ],
    "The prompt separates background, requested action, instance-specific content, and an output boundary. Making those roles explicit helps diagnose failures: changing the location is different from changing the task, and a length limit is a constraint rather than additional story context.",
  ),
  makeQuestion(
    "cme295-lect3-q216",
    "medium",
    "A classification prompt moves from zero-shot instructions to six labeled input-output demonstrations. Which consequences are plausible?",
    [
      [
        "The demonstrations can make the intended mapping clearer through in-context learning.",
        true,
      ],
      [
        "The examples consume input tokens and therefore increase context use and cost.",
        true,
      ],
      [
        "Preparing representative examples requires effort and can introduce example-selection bias.",
        true,
      ],
      [
        "The model permanently updates its parameters from the demonstrations before answering.",
        false,
      ],
    ],
    "Few-shot prompting conditions the existing model on examples, often improving task specification at the price of prompt construction, tokens, latency, and possible bias from unrepresentative demonstrations. The examples affect the current forward pass through context; they do not trigger gradient descent or persist as a weight update.",
  ),
  makeQuestion(
    "cme295-lect3-q217",
    "medium",
    "A model follows a new label format after seeing three examples in its prompt, then loses that behavior when the examples are removed. Which explanation fits?",
    [
      [
        "The behavior was induced by in-context learning while the examples were present.",
        true,
      ],
      ["The trained weights can remain unchanged across both requests.", true],
      [
        "The first request necessarily fine-tuned the model and the second request rolled back that checkpoint.",
        false,
      ],
      [
        "The KV cache must have stored the examples as permanent training data shared with later users.",
        false,
      ],
    ],
    "In-context learning is temporary conditioning through the current token sequence, so removing the examples can remove the induced behavior without any parameter change. A request-local KV cache supports computation for that request and is not evidence of fine-tuning or cross-user training-data storage.",
  ),
  makeQuestion(
    "cme295-lect3-q218",
    "hard",
    "A self-consistency run samples five independent reasoning paths with final answers A, A, B, A, and B. Which statements correctly describe the aggregation and cost?",
    [
      ["Majority vote returns A with three of five votes.", true],
      [
        "The system must extract a comparable final answer from each reasoning trace before voting.",
        true,
      ],
      [
        "The five branches can run in parallel because one branch need not condition on another.",
        true,
      ],
      [
        "Compared with one direct answer, the method removes generation cost because only the majority trace is retained.",
        false,
      ],
    ],
    "Self-consistency aggregates independently sampled solutions, so A wins the final-answer vote and the branches can be parallelized after sharing the same prompt. Parallel execution can reduce wall-clock latency relative to serial execution, but all five completions still consume compute and generated tokens even if only one aggregate answer is returned.",
  ),

  // Inference optimizations (22): reuse, memory layout, attention sharing, and token prediction.
  makeQuestion(
    "cme295-lect3-q219",
    "easy",
    "Which mappings match the inference-efficiency categories in the optimization overview?",
    [
      ["Avoid redundant computation: key-value (KV) caching.", true],
      ["Manage allocated memory: PagedAttention.", true],
      [
        "Reformulate token generation while preserving the target distribution: speculative decoding.",
        true,
      ],
      [
        "Architectural or representation approximations include GQA, latent attention, and Multi-Token Prediction.",
        true,
      ],
    ],
    "KV caching, PagedAttention, and speculative decoding attack repeated projections, wasteful cache allocation, and serial target-model token generation. GQA, latent attention, and Multi-Token Prediction instead alter architecture, stored representation, or prediction heads, so they belong to the approximation side of the overview rather than simple cache reuse.",
  ),
  makeQuestion(
    "cme295-lect3-q220",
    "medium",
    "A serving team records four bottlenecks: recomputing old attention tensors, reserving unused cache space, storing too many key/value heads, and taking one large-model pass per output token. Which remedies target those bottlenecks in the same order?",
    [
      ["KV caching targets recomputation of old keys and values.", true],
      ["PagedAttention targets waste from cache allocation.", true],
      [
        "Grouped Query Attention (GQA) targets duplicated key/value-head storage.",
        true,
      ],
      [
        "Speculative decoding targets serial target-model token generation.",
        true,
      ],
    ],
    "Each technique corresponds to a different resource mechanism, so a useful diagnosis must match the symptom to the layer it changes. The methods can coexist: reuse does not solve allocation, head sharing does not validate draft tokens, and faster token proposals do not by themselves shrink the KV cache.",
  ),
  makeQuestion(
    "cme295-lect3-q221",
    "easy",
    "During generation of token \\(t+1\\), what does ordinary KV caching primarily avoid?",
    [
      [
        "Recomputing the key and value projections already produced for prefix tokens \\(1,\\ldots,t\\).",
        true,
      ],
      ["Computing a new query for token \\(t+1\\).", false],
      ["Reading any representation of the prefix during attention.", false],
      ["Producing vocabulary logits for the newly generated position.", false],
    ],
    "The current token still needs a fresh query, attention over prefix state, and an output distribution. KV caching saves work by reusing the earlier tokens' key and value projections rather than rebuilding those same tensors at every decoding step.",
  ),
  makeQuestion(
    "cme295-lect3-q222",
    "easy",
    "Why are previous keys and values cached while previous queries are not the main reusable state for the next token?",
    [
      ["The new token creates its own query to score the prefix keys.", true],
      [
        "Those attention scores weight the prefix values to form the new representation.",
        true,
      ],
      [
        "Previous queries were used to build previous outputs but are not needed to score the new query against the prefix.",
        true,
      ],
      [
        "Previous queries are discarded because a causal decoder never attends to earlier positions.",
        false,
      ],
    ],
    "For the new position, attention uses a new query together with the stored keys and values of earlier positions. Causality prevents attention to the future, not the past, so the reason for omitting old queries is their role in the formula rather than an inability to revisit the prefix.",
  ),
  makeQuestion(
    "cme295-lect3-q223",
    "medium",
    "Why is KV caching especially useful in autoregressive inference but not the same kind of win during teacher-forced training over a full sequence?",
    [
      [
        "Inference revisits an expanding prefix once per generated token, creating repeated key/value work without a cache.",
        true,
      ],
      [
        "Teacher-forced training can compute representations for all sequence positions in parallel within a forward pass.",
        true,
      ],
      [
        "Training never computes keys or values, so there is nothing to cache.",
        false,
      ],
      [
        "Inference uses bidirectional attention, whereas training alone uses causal attention.",
        false,
      ],
    ],
    "Serial decoding repeatedly extends the same prefix, which makes reuse across steps valuable. Teacher forcing still computes keys and values and still uses causal masking for a decoder-only model, but it processes the known target positions together rather than issuing one expanding-prefix pass per generated token.",
  ),
  makeQuestion(
    "cme295-lect3-q224",
    "hard",
    "A model has 32 layers, 8 key/value heads per layer, head dimension 128, and a cached sequence length of 2048. Storing both K and V in bfloat16 uses how much memory for one request, ignoring metadata?",
    [
      [
        "\\(32\\times2048\\times8\\times128\\times2\\) elements times 2 bytes, which is 256 MiB.",
        true,
      ],
      ["\\(2048\\times128\\) elements times 2 bytes, which is 0.5 MiB.", false],
      [
        "\\(32\\times2048\\times8\\times128\\) elements times 2 bytes, which is 128 MiB because K and V share storage.",
        false,
      ],
      [
        "\\(32\\times8\\times128\\times2\\) elements times 2 bytes, which is 128 KiB because sequence length does not affect the cache.",
        false,
      ],
    ],
    "The cache stores a key and a value for every layer, cached position, key/value head, and head coordinate. That is 134,217,728 bfloat16 elements or 268,435,456 bytes, equal to 256 MiB; omitting layers, sequence positions, or the separate K/V payload produces the smaller distractors.",
  ),
  makeQuestion(
    "cme295-lect3-q225",
    "medium",
    "A request has a cached prefix of length \\(L\\) and generates one new token. Which work still occurs with an ordinary KV cache?",
    [
      ["The new token's query, key, and value projections are computed.", true],
      [
        "The new query is compared with the \\(L\\) cached keys plus the new key.",
        true,
      ],
      [
        "The resulting attention weights combine the cached values plus the new value.",
        true,
      ],
      [
        "The key and value projections for all \\(L\\) old tokens are recomputed before attention.",
        false,
      ],
    ],
    "Caching removes repeated prefix projections, but it does not make attention independent of the prefix: the current query must still interact with all relevant cached keys and values. The cache therefore changes redundant projection work and memory use, not the causal dependency of the new token on earlier tokens.",
  ),
  makeQuestion(
    "cme295-lect3-q226",
    "hard",
    "A server holds 24 concurrent requests whose KV caches each occupy 0.5 GiB. If every context length doubles and cache size scales linearly with length, which capacity estimates are correct?",
    [
      ["The original batch holds 12 GiB of KV-cache data.", true],
      [
        "After doubling context length, the batch holds about 24 GiB of KV-cache data.",
        true,
      ],
      [
        "Doubling context length leaves cache memory unchanged because model weights are fixed.",
        false,
      ],
      [
        "The new batch holds 48 GiB because both request count and context length doubled.",
        false,
      ],
    ],
    "Twenty-four requests times 0.5 GiB gives 12 GiB, and doubling only the per-request context doubles that cache total to roughly 24 GiB. The model weights may be unchanged, but request-local KV state scales with cached positions; the request count did not change in this scenario.",
  ),
  makeQuestion(
    "cme295-lect3-q227",
    "easy",
    "For a decoder with \\(h\\) query heads and \\(G\\) key/value groups, which head-count relationships are correct?",
    [
      [
        "Multi-Head Attention (MHA) uses \\(G=h\\), giving each query head its own key/value head.",
        true,
      ],
      [
        "Grouped Query Attention (GQA) uses \\(1<G<h\\), sharing each key/value head across a query group.",
        true,
      ],
      [
        "Multi-Query Attention (MQA) uses \\(G=1\\), sharing one key/value set across all query heads.",
        true,
      ],
      [
        "GQA reduces cache memory by deleting query heads until only \\(G\\) queries remain.",
        false,
      ],
    ],
    "MHA, GQA, and MQA form a sharing spectrum for keys and values while retaining multiple query heads. The memory reduction comes from fewer distinct K/V projections and cached heads, not from collapsing the query-head count to the number of groups.",
  ),
  makeQuestion(
    "cme295-lect3-q228",
    "medium",
    "A model has 32 query heads and uses GQA with 8 key/value heads. Which combined statement is correct?",
    [
      [
        "Each K/V head is shared by four query heads, and the K/V cache is one quarter of the comparable full-MHA payload.",
        true,
      ],
      [
        "Each K/V head is shared by eight query heads, and the cache is one eighth of the MHA payload.",
        false,
      ],
      [
        "The model keeps eight query heads and creates 32 independent K/V heads.",
        false,
      ],
      [
        "The query and K/V head counts both remain 32, so only the attention softmax changes.",
        false,
      ],
    ],
    "Dividing 32 query heads into 8 groups gives four queries per shared K/V head, reducing the distinct cached K/V payload by a factor of four relative to MHA. Query diversity remains represented by 32 query heads; grouping changes which key/value projections they share.",
  ),
  makeQuestion(
    "cme295-lect3-q229",
    "hard",
    "A full-MHA KV cache occupies 16 GiB with 32 key/value heads. Holding all other dimensions fixed, which estimates are correct?",
    [
      ["GQA with 8 K/V heads would use about 4 GiB.", true],
      ["MQA with 1 K/V head would use about 0.5 GiB.", true],
      [
        "The MQA cache would be one eighth the size of the 8-head GQA cache.",
        true,
      ],
      [
        "GQA with 8 K/V heads would still use 16 GiB because the query-head count remains 32.",
        false,
      ],
    ],
    "K/V cache size scales with the number of distinct key/value heads, so 8 of 32 heads gives \\(16/4=4\\) GiB and 1 of 32 gives \\(16/32=0.5\\) GiB. The MQA result is one eighth of the 8-head GQA result; retaining query heads does not restore the removed K/V payload.",
  ),
  makeQuestion(
    "cme295-lect3-q230",
    "hard",
    "A model designer moves from MHA to GQA while keeping 32 query heads but sharing 8 key/value heads. Which tradeoffs should be considered?",
    [
      [
        "KV-cache memory and K/V memory bandwidth can fall because fewer distinct heads are stored and read.",
        true,
      ],
      [
        "The architectural sharing may affect quality, so the model should be trained or evaluated with that design.",
        true,
      ],
      [
        "The change is different from ordinary KV caching: it reduces what is stored per token rather than merely reusing old projections.",
        true,
      ],
      [
        "The query heads can remain distinct even though their groups share keys and values.",
        true,
      ],
    ],
    "GQA is an architectural representation tradeoff that reduces K/V multiplicity while preserving multiple query heads, so it can improve memory efficiency but still needs quality evaluation. Ordinary KV caching and GQA are complementary: caching avoids recomputation across steps, whereas grouping shrinks each step's stored K/V state.",
  ),

  makeQuestion(
    "cme295-lect3-q231",
    "easy",
    "Which statements describe PagedAttention-style KV-cache management?",
    [
      [
        "A request's logical token positions can map to fixed-size physical cache blocks.",
        true,
      ],
      [
        "The physical blocks for one request need not be contiguous in device memory.",
        true,
      ],
      [
        "Allocating blocks as a sequence grows can reduce large reserved-but-unused regions.",
        true,
      ],
      [
        "Paging compresses each key and value vector into a lower-dimensional latent.",
        false,
      ],
    ],
    "PagedAttention changes allocation and address mapping: logical cache positions are backed by smaller physical blocks that can be placed non-contiguously. It does not change the numerical representation inside each key/value entry; lower-dimensional storage is the separate latent-attention idea.",
  ),
  makeQuestion(
    "cme295-lect3-q232",
    "medium",
    "A cache allocator uses blocks that hold 16 tokens. Two requests currently need 33 and 49 cached positions. Which allocation facts are correct?",
    [
      ["The first request needs 3 blocks and the second needs 4 blocks.", true],
      ["Together they reserve capacity for 112 token positions.", true],
      [
        "Together they leave 30 positions unused inside their final blocks.",
        true,
      ],
      [
        "Each request needs exactly 2 blocks because only complete blocks count.",
        false,
      ],
    ],
    "Ceiling division gives \\(\\lceil33/16\\rceil=3\\) and \\(\\lceil49/16\\rceil=4\\), for seven blocks or 112 positions. The requests use 82 positions, leaving 30 positions of internal block slack; partial final blocks still require physical allocation.",
  ),
  makeQuestion(
    "cme295-lect3-q233",
    "hard",
    "An MHA layer has 32 heads of dimension 128, so full K and V storage uses \\(2\\times32\\times128=8192\\) scalar values per token. A latent-attention design stores one 512-value shared latent instead. What is the payload ratio for this simplified comparison?",
    [
      [
        "\\(512/8192=1/16\\), so the stored payload is sixteen times smaller.",
        true,
      ],
      [
        "\\(512/4096=1/8\\), because K and V should be counted only once in the baseline.",
        false,
      ],
      [
        "\\(8192/512=16\\), so the latent payload is sixteen times larger.",
        false,
      ],
      [
        "\\(512/32=16\\), so compression depends only on the head count.",
        false,
      ],
    ],
    "The stated baseline includes both keys and values, so its 8192 scalars must be compared with the 512 stored latent scalars. Decompression can later produce key-like and value-like representations, but those reconstructed tensors do not change the simplified cache payload ratio given in the prompt.",
  ),
  makeQuestion(
    "cme295-lect3-q234",
    "medium",
    "A server already uses exact K/V vectors but wastes space by reserving a maximum-length contiguous region per request. A model team instead wants to reduce the dimensional payload stored for every token. Which technique matches each problem?",
    [
      [
        "PagedAttention matches the allocation and fragmentation problem.",
        true,
      ],
      [
        "Latent attention matches the per-token representation-size problem.",
        true,
      ],
      [
        "PagedAttention matches both problems because paging automatically changes vector dimension.",
        false,
      ],
      [
        "Latent attention matches only fragmentation because its latent is stored in non-contiguous blocks by definition.",
        false,
      ],
    ],
    "Paging changes where and when cache blocks are allocated while keeping their contained representation conceptually intact. Latent attention changes what representation is stored by compressing key/value information, so the methods address orthogonal memory costs and may be combined.",
  ),
  makeQuestion(
    "cme295-lect3-q235",
    "hard",
    "A serving design combines GQA, a compressed latent cache, and PagedAttention. Which description of the combined design is accurate?",
    [
      [
        "GQA reduces the number of distinct key/value head groups represented.",
        true,
      ],
      [
        "Latent attention reduces the dimensional payload stored for attention state.",
        true,
      ],
      [
        "PagedAttention maps the resulting logical cache into physical blocks to limit allocation waste.",
        true,
      ],
      [
        "Because the methods act on different mechanisms, their benefits and quality tradeoffs must be measured together rather than assumed to be interchangeable.",
        true,
      ],
    ],
    "The methods address head multiplicity, representation dimension, and physical allocation, respectively, so combining them is conceptually coherent. Their effects are not automatically multiplicative and architectural changes may affect quality, making end-to-end memory, latency, and model evaluation necessary.",
  ),
  makeQuestion(
    "cme295-lect3-q236",
    "easy",
    "Which statements correctly assign roles in speculative decoding?",
    [
      ["A cheaper draft model proposes several candidate tokens.", true],
      [
        "The target model evaluates those proposals and remains the authority for the desired output distribution.",
        true,
      ],
      [
        "Accepted draft tokens can advance generation by several positions after one target-model pass.",
        true,
      ],
      [
        "The target model is fine-tuned on the draft continuation during every request.",
        false,
      ],
    ],
    "Speculative decoding is an inference algorithm: the draft supplies cheap proposals and the target supplies validation probabilities. Its speedup comes when multiple proposals are accepted together, and no per-request training or weight update is part of that validation loop.",
  ),
  makeQuestion(
    "cme295-lect3-q237",
    "hard",
    "For a drafted token, let draft probability be \\(P\\) and target probability be \\(Q\\). Which acceptance calculations are correct?",
    [
      [
        "If \\(P=0.6\\) and \\(Q=0.3\\), the token is accepted with probability \\(Q/P=0.5\\).",
        true,
      ],
      [
        "If \\(P=0.4\\) and \\(Q=0.7\\), the token is accepted with probability 1.",
        true,
      ],
      [
        "Whenever \\(Q<P\\), the token must be rejected with probability 1.",
        false,
      ],
      [
        "Acceptance depends only on whether the complete draft sequence has a larger joint score than the target's greedy sequence.",
        false,
      ],
    ],
    "The token-level acceptance probability is \\(\\min(1,Q/P)\\), giving 0.5 in the first case and certain acceptance in the second. A lower target probability does not force rejection; the randomized ratio rule is what allows the overall sampler to preserve the target distribution.",
  ),
  makeQuestion(
    "cme295-lect3-q238",
    "hard",
    "At a rejected position, the draft distribution is \\(P=(0.5,0.3,0.2)\\) for tokens A, B, C and the target distribution is \\(Q=(0.4,0.4,0.2)\\). The correction samples from the normalized positive residual \\([Q-P]_+\\). Which conclusions follow?",
    [
      [
        "Only token B has positive residual mass, so the correction selects B.",
        true,
      ],
      [
        "Using the residual correction is part of preserving the target model's distribution after rejection.",
        true,
      ],
      [
        "Token A is most likely because it had the highest probability under the draft model.",
        false,
      ],
      [
        "Token C receives half the correction mass because its draft and target probabilities are equal.",
        false,
      ],
    ],
    "The residuals are \\((-0.1,0.1,0)\\), whose positive part normalizes entirely onto B. Draft preference alone cannot determine the replacement after rejection, and an equal target and draft probability contributes zero corrective mass rather than a positive share.",
  ),
  makeQuestion(
    "cme295-lect3-q239",
    "easy",
    "Which statements describe Multi-Token Prediction (MTP)?",
    [
      [
        "Training adds prediction heads for several future token offsets.",
        true,
      ],
      [
        "The training objective supervises more than only the immediate next token.",
        true,
      ],
      [
        "The extra heads can provide draft-like future-token proposals within the same model.",
        true,
      ],
      [
        "MTP is another name for sampling from the top k vocabulary tokens.",
        false,
      ],
    ],
    "MTP changes the model and its training targets so one representation supports predictions for several future positions. Top-k is only an inference-time truncation of one next-token distribution and neither adds future-offset heads nor changes the training objective.",
  ),
  makeQuestion(
    "cme295-lect3-q240",
    "medium",
    "Which comparisons between speculative decoding and Multi-Token Prediction are correct?",
    [
      [
        "Speculative decoding commonly pairs a separate cheap draft model with a target model.",
        true,
      ],
      [
        "MTP builds multiple future-token prediction heads into one trained model.",
        true,
      ],
      [
        "Both approaches try to propose or validate more than one future token per expensive serial step.",
        true,
      ],
      [
        "Speculative decoding changes the inference algorithm, while MTP also changes the model's training objective.",
        true,
      ],
    ],
    "The two methods share a goal of advancing token generation in larger chunks but create proposals differently. Speculative decoding can wrap existing draft and target models with an acceptance sampler, whereas MTP trains built-in future-offset heads and therefore requires an architectural and objective change.",
  ),
];
