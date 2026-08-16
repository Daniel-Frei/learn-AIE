import type { Difficulty, Question } from "../../quiz";

type AnswerOption = readonly [text: string, isCorrect: boolean];

const assertionReasonOptions = (correctIndex: number): Question["options"] => [
  {
    text: "Assertion is true, Reason is false.",
    isCorrect: correctIndex === 0,
  },
  {
    text: "Assertion is false, Reason is true.",
    isCorrect: correctIndex === 1,
  },
  { text: "Both are false.", isCorrect: correctIndex === 2 },
  {
    text: "Both are true, and the Reason is the correct explanation of the Assertion.",
    isCorrect: correctIndex === 3,
  },
  {
    text: "Both are true, but the Reason is NOT the correct explanation of the Assertion.",
    isCorrect: correctIndex === 4,
  },
];

function makeQuestion(
  id: string,
  chapter: number,
  difficulty: Difficulty,
  prompt: string,
  options: readonly AnswerOption[],
  explanation: string,
): Question {
  return {
    id,
    chapter,
    difficulty,
    prompt,
    options: options.map(([text, isCorrect]) => ({ text, isCorrect })),
    explanation,
  };
}

function makeAssertionReasonQuestion(
  id: string,
  chapter: number,
  difficulty: Difficulty,
  assertion: string,
  reason: string,
  correctIndex: number,
  explanation: string,
): Question {
  return {
    id,
    chapter,
    difficulty,
    type: "assertion-reason",
    prompt: `Assertion: ${assertion}\n\nReason: ${reason}`,
    options: assertionReasonOptions(correctIndex),
    explanation,
  };
}

export const diffusionGemmaTechnicalReportQuestions: Question[] = [
  makeQuestion(
    "cme296-diffusiongemma-q01",
    1,
    "easy",
    "Which operating point is the defining goal of DiffusionGemma's text-diffusion mode?",
    [
      [
        "Generate a 256-token canvas through parallel iterative refinement, reducing the number of model forward passes needed for low-concurrency decoding.",
        true,
      ],
      [
        "Draft eight tokens autoregressively and accept only the prefix verified by a larger frozen target model.",
        false,
      ],
      [
        "Encode the whole response once into an unrestricted continuous latent vector, then recover each position independently by projecting the final vector to its nearest vocabulary embedding.",
        false,
      ],
      [
        "Increase high-batch throughput by replacing mixture-of-experts routing with a dense single-token decoder.",
        false,
      ],
    ],
    "DiffusionGemma targets low-latency text generation by refining many token positions in parallel inside a 256-token canvas. It is a standalone diffusion generator rather than an eight-token draft-and-verify system, and it operates on discrete tokens instead of decoding a single continuous response vector. The released model also retains its mixture-of-experts backbone rather than replacing it with a dense decoder.",
  ),
  makeQuestion(
    "cme296-diffusiongemma-q02",
    1,
    "medium",
    "Why can parallel canvas denoising outperform token-by-token autoregressive decoding for a single request on a modern GPU?",
    [
      [
        "It uses fewer weight and key-value cache transfers by producing many tokens per model forward pass.",
        true,
      ],
      [
        "It trades additional arithmetic within each step for better utilization of otherwise idle accelerator compute.",
        true,
      ],
      [
        "It makes every diffusion step cheaper than a single-token autoregressive step in both FLOPs and memory traffic.",
        false,
      ],
      [
        "It removes the need to evaluate a vocabulary distribution while denoising the canvas.",
        false,
      ],
    ],
    "Single-request autoregressive decoding is often memory-bound because every token requires another transfer of model weights and context state. DiffusionGemma spends more computation per step, but a step can advance many token positions, shifting work toward the GPU's underused compute capacity. Its steps are not universally cheaper, and full-canvas vocabulary sampling remains a measurable cost.",
  ),
  makeQuestion(
    "cme296-diffusiongemma-q03",
    1,
    "hard",
    "Which statements accurately characterize the released DiffusionGemma checkpoint and its conversion from Gemma 4?",
    [
      [
        "It is warm-started from the post-trained Gemma 4 26B A4B mixture-of-experts checkpoint rather than pretrained as a diffusion model from scratch.",
        true,
      ],
      [
        "Its two-stage conversion uses supervised diffusion fine-tuning followed by joint sampler distillation and reinforcement learning.",
        true,
      ],
      [
        "The conversion uses less than 10% of the starting autoregressive model's total training-token budget.",
        true,
      ],
      [
        "Its activated parameter count rises to the full 25.2 billion parameters on every token because diffusion disables sparse expert routing.",
        false,
      ],
    ],
    "The model reuses Gemma 4 26B A4B and converts it with supervised fine-tuning followed by the joint sampler-distillation and reinforcement-learning stage, avoiding native diffusion pretraining. The report states that this conversion consumes under one tenth of the original autoregressive training-token budget. Sparse mixture-of-experts routing remains: total parameters are about 25.2B, while roughly 3.85B are activated, excluding the vision encoder.",
  ),
  makeQuestion(
    "cme296-diffusiongemma-q04",
    1,
    "easy",
    "Which reported properties belong to DiffusionGemma's main text-diffusion configuration?",
    [
      ["The generation canvas contains 256 token positions.", true],
      ["The sampler permits at most 48 denoising steps per canvas.", true],
      [
        "Adaptive stopping reduces the observed average to about 12 denoising steps across the reported downstream evaluations.",
        true,
      ],
      [
        "The checkpoint retains thinking mode, multimodal inputs, long-context support, and a usable autoregressive decoding mode.",
        true,
      ],
    ],
    "The recommended setup uses 256-token canvases, a maximum budget of 48 denoising steps, and adaptive stopping that averages roughly 12 steps on the evaluated workloads. Because the conversion preserves the Gemma 4 backbone, the checkpoint also retains thinking, multimodal, long-context, and autoregressive capabilities. These properties jointly define the paper's speed-oriented but flexible operating point.",
  ),
  makeQuestion(
    "cme296-diffusiongemma-q05",
    1,
    "medium",
    "A serving team compares DiffusionGemma with an autoregressive model using speculative decoding. Which conclusions are supported by the reported tokens-per-forward figures?",
    [
      [
        "DiffusionGemma's average of about 20 tokens per forward is several times the roughly 3-6 tokens per forward cited for strong speculative decoding.",
        true,
      ],
      [
        "A higher tokens-per-forward value can offset a slower individual diffusion forward pass when computing end-to-end token throughput.",
        true,
      ],
      [
        "Tokens per forward alone proves superior throughput at every context length, batch size, hardware type, and sampling implementation.",
        false,
      ],
      [
        "The comparison means speculative decoding verifies no more than one proposed token in any target-model call.",
        false,
      ],
    ],
    "The paper contrasts roughly 20 tokens per forward for DiffusionGemma with about 3-6 for contemporary speculative decoding, explaining how fewer calls can overcome heavier diffusion steps. Tokens per forward is only one factor: wall-clock throughput also depends on the cost of each forward pass, context, hardware, batching, and kernels. Speculative verification may accept several draft tokens, so it is not restricted to one accepted token per call.",
  ),
  makeQuestion(
    "cme296-diffusiongemma-q06",
    1,
    "hard",
    "Which consequences follow from using a block-autoregressive strategy rather than one unbounded diffusion canvas?",
    [
      [
        "A completed canvas can be appended to the key-value cache directly from decoder logits without an additional causal encoder pass.",
        false,
      ],
      [
        "Bidirectional revision is available among positions inside the active canvas but not for tokens committed in earlier canvases.",
        true,
      ],
      [
        "Open-ended responses can extend beyond the fixed 256-position diffusion canvas.",
        true,
      ],
      [
        "The causal encoder can update the growing history without recomputing every earlier token's keys and values.",
        true,
      ],
    ],
    "Block-autoregressive generation turns a fixed-size diffusion model into an open-ended generator by committing one clean canvas at a time. The clean block needs an additional causal encoder pass before its keys and values can be appended for the next canvas. The tradeoff is a revision boundary: current-canvas tokens can change together, but previously committed canvases are frozen.",
  ),
  makeQuestion(
    "cme296-diffusiongemma-q07",
    2,
    "easy",
    "Which distinctions motivate native discrete diffusion for language tokens?",
    [
      [
        "Categorical corruption operates directly on vocabulary states, avoiding a final nearest-token projection from an unrestricted embedding space.",
        true,
      ],
      [
        "Uniform or masking transitions can be expressed as probabilistic moves between discrete token states.",
        true,
      ],
      [
        "Continuous embedding diffusion can drift through regions that contain no valid token representation.",
        true,
      ],
      [
        "Hard rounding from continuous latents can disrupt likelihood guarantees and map degenerate embeddings to unrelated tokens.",
        true,
      ],
    ],
    "Native discrete diffusion keeps both corruption and denoising on categorical vocabulary states, so it does not need to round an arbitrary continuous vector to a token. Continuous text diffusion can encounter empty regions of embedding space, and a hard projection may produce unrelated vocabulary items while breaking theoretical bounds. Uniform and absorbing-mask transitions are examples of well-defined discrete corruption mechanisms.",
  ),
  makeQuestion(
    "cme296-diffusiongemma-q08",
    2,
    "medium",
    "At one canvas position, the forward transition is \\(P(X_t=v\\mid X_0=x_0)=\\kappa_t\\delta(v,x_0)+(1-\\kappa_t)/V\\). If \\(\\kappa_t=0.7\\), which probability assignment follows?",
    [
      [
        "\\(P(X_t=x_0)=0.7+0.3/V\\), while each particular non-original token has probability \\(0.3/V\\).",
        true,
      ],
      [
        "\\(P(X_t=x_0)=0.3+0.7/V\\), while each particular non-original token has probability \\(0.7/V\\).",
        false,
      ],
      [
        "\\(P(X_t=x_0)=0.7\\), while each non-original token has probability \\(0.3/(V-1)\\).",
        false,
      ],
      [
        "\\(P(X_t=x_0)=1/V\\), and every non-original token also has probability \\(1/V\\) for any \\(\\kappa_t\\).",
        false,
      ],
    ],
    "The uniform component can redraw the original token, so its probability includes both the 0.7 point-mass contribution and its 0.3/V share of uniform noise. Every particular alternative receives only 0.3/V. Dividing 0.3 across the other \\(V-1\\) tokens would describe a different corruption rule that explicitly forbids the uniform draw from returning the original token.",
  ),
  makeQuestion(
    "cme296-diffusiongemma-q09",
    2,
    "hard",
    "A clean canvas has \\(C\\) positions and vocabulary size \\(V\\). Which statements follow from the factorized forward transition used for training?",
    [
      [
        "Conditioned on the clean canvas, the corruption decision is independent across token coordinates.",
        true,
      ],
      [
        "At \\(t=1\\), each coordinate is uniformly distributed over \\(V\\), so the joint source distribution is uniform over \\(V^C\\).",
        true,
      ],
      [
        "The state \\(X_t^i\\) must be sampled left to right because coordinate \\(i\\) is conditioned on the already corrupted value \\(X_t^{i-1}\\).",
        false,
      ],
      [
        "The closed-form transition requires simulating every earlier noise level before a training example at time \\(t\\) can be obtained.",
        false,
      ],
    ],
    "The forward path factorizes across coordinates, and its endpoint is an independent uniform draw at every one of the \\(C\\) positions. This makes the joint endpoint uniform over the categorical canvas state space and removes any left-to-right dependency from corruption. Because the transition is available in closed form, a training example can be noised directly at a sampled time rather than simulating all earlier noise levels.",
  ),
  makeQuestion(
    "cme296-diffusiongemma-q10",
    2,
    "easy",
    "What information must the learned reverse process approximate to move a corrupted canvas toward clean text?",
    [
      [
        "The posterior distribution of each original clean token conditioned on the current noisy canvas.",
        true,
      ],
      [
        "A transition mapping that combines the current state with the estimated clean-token posterior for a backward-time update.",
        true,
      ],
      [
        "Categorical probabilities that can shift mass from uniform noise toward modes of the data distribution.",
        true,
      ],
      [
        "A deterministic left-to-right ordering that fixes the first canvas token before any other position is evaluated.",
        false,
      ],
    ],
    "Discrete flow matching reverses corruption by estimating the distribution of clean tokens given the noisy canvas and feeding that estimate to a backward transition rule. Repeated categorical updates move probability mass away from uniform noise toward coherent data modes. The positions are processed in parallel within the canvas, so the reverse process does not require a causal order that permanently fixes the first token.",
  ),
  makeAssertionReasonQuestion(
    "cme296-diffusiongemma-q11",
    2,
    "medium",
    "DiffusionGemma's adaptive stopping rule requires both low mean predictive entropy and identical deterministic predictions on two consecutive denoising steps.",
    "The sampler halts as soon as either the entropy condition or the stability condition is satisfied.",
    0,
    "The assertion is true: the default stopping rule is a conjunction of confidence and sequence stability, evaluated after the first step. The reason is false because satisfying only one condition is insufficient; low entropy can accompany a confidently wrong or repetitive state, while unchanged argmax tokens can still have substantial uncertainty. Requiring both conditions reduces premature termination.",
  ),
  makeQuestion(
    "cme296-diffusiongemma-q12",
    2,
    "hard",
    "Why can independently sampled coordinate updates create local inconsistencies during one reverse step?",
    [
      [
        "Every coordinate conditions on the current noisy canvas, but it cannot condition on the simultaneous sampled choices made at the other coordinates in that same step.",
        true,
      ],
      [
        "Two positions may therefore make individually plausible updates that are grammatically or semantically incompatible when combined.",
        true,
      ],
      [
        "Self-conditioning can feed the previous predicted token distributions into the next step, improving coordination over successive refinements.",
        true,
      ],
      [
        "Entropy-bounded acceptance can retain lower-uncertainty positions while renoising less trusted positions for further exploration.",
        true,
      ],
    ],
    "The reverse transition is coordinate-factorized at an individual step: all positions see the old canvas, not the other fresh samples being drawn alongside them. That can create mismatched local choices even when each marginal looks reasonable. Self-conditioning and entropy-bounded resampling do not remove the factorization, but they give later steps information and exploration that can repair such conflicts.",
  ),
  makeQuestion(
    "cme296-diffusiongemma-q13",
    3,
    "easy",
    "Which attention arrangement distinguishes DiffusionGemma's generation pipeline from a conventional encoder-decoder model such as T5?",
    [
      [
        "It causally encodes sequence history into a cache and bidirectionally decodes the active diffusion canvas while cross-attending to that cache.",
        true,
      ],
      [
        "It bidirectionally re-encodes the entire growing history after every denoising step and causally decodes one canvas token.",
        false,
      ],
      [
        "It uses bidirectional attention over both committed history and the active canvas, then recomputes all earlier keys and values whenever another clean canvas is appended.",
        false,
      ],
      [
        "It removes cross-attention and conditions canvas tokens only through a pooled prompt embedding.",
        false,
      ],
    ],
    "DiffusionGemma inverts the familiar pattern: the context/history encoder is causal, while the active canvas decoder is bidirectional and cross-attends to the cached history. This makes newly committed canvases appendable without revisiting all earlier keys and values. A pooled prompt without cross-attention would not provide the token-level context conditioning described by the architecture.",
  ),
  makeQuestion(
    "cme296-diffusiongemma-q14",
    3,
    "medium",
    "During one decoder forward pass, which inputs jointly determine the clean-token logits for the active canvas?",
    [
      [
        "The current noisy categorical canvas and a continuous self-conditioning signal derived from earlier predictions.",
        true,
      ],
      [
        "The causally encoded key-value cache containing the prompt and committed prior canvases.",
        true,
      ],
      [
        "A separately trained verifier model that accepts or rejects the canvas prefix.",
        false,
      ],
      [
        "Ground-truth future canvases copied into the cache during inference.",
        false,
      ],
    ],
    "The shared transformer decoder receives the current noisy tokens, the self-conditioning vectors, and the key-value cache for available context. Bidirectional attention operates within the active canvas, while cross-attention brings in prompt and history information. No external verifier or unavailable ground-truth future canvas is part of ordinary inference.",
  ),
  makeQuestion(
    "cme296-diffusiongemma-q15",
    3,
    "hard",
    "Which operations form one non-final denoising iteration after the decoder produces logits \\(L_t\\)?",
    [
      [
        "Apply the time-dependent temperature before softmax to obtain estimated clean-token probabilities.",
        true,
      ],
      [
        "Multiply those probabilities by the token embedding matrix and pass the result through a feedforward block to form the next self-conditioning signal.",
        true,
      ],
      [
        "Use the sampler's transition rule to produce the next noisy canvas state from the current state and estimated clean-token distribution.",
        true,
      ],
      [
        "Append the still-noisy canvas to the causal key-value cache before the next denoising iteration.",
        false,
      ],
    ],
    "A denoising iteration turns logits into tempered clean-token probabilities, updates the continuous self-conditioning representation, and samples a refined canvas through the chosen transition. The key-value cache is not updated with intermediate noisy states. Only a completed clean canvas is causally encoded and appended between blocks.",
  ),
  makeQuestion(
    "cme296-diffusiongemma-q16",
    3,
    "easy",
    "Which behaviors are enabled by multinomial diffusion within the active canvas?",
    [
      [
        "Any vocabulary token can transition to another token during refinement rather than only changing a special mask state.",
        true,
      ],
      [
        "A token accepted in an early denoising step can still be revised before the canvas is committed.",
        true,
      ],
      [
        "Early answer words and later reasoning words can co-evolve under bidirectional attention inside one canvas.",
        true,
      ],
      [
        "Committed tokens from previous canvases remain fixed even though current-canvas tokens can change.",
        true,
      ],
    ],
    "Multinomial diffusion allows transitions among ordinary vocabulary tokens, so provisional choices remain editable throughout refinement. Together with bidirectional attention, this supports within-canvas correction and coordination between early and later text. The ability stops at the block boundary: a committed previous canvas becomes causal history and is not reopened.",
  ),
  makeQuestion(
    "cme296-diffusiongemma-q17",
    3,
    "medium",
    "What roles does self-conditioning play in the denoising architecture?",
    [
      [
        "It feeds a continuous representation of the model's prior clean-token predictions into the following denoising step.",
        true,
      ],
      [
        "It gives the next step information about the previous belief state beyond the sampled categorical canvas alone.",
        true,
      ],
      [
        "It permanently writes high-confidence token predictions into the causal history cache after every step.",
        false,
      ],
      [
        "It replaces the token embedding matrix with a second vocabulary whose entries are denoising times.",
        false,
      ],
    ],
    "Self-conditioning projects the previous probability-weighted token embeddings through a feedforward network and supplies the resulting continuous vectors to the next decoder pass. This preserves information about the full predictive distribution even when the sampled categorical canvas is noisy. It neither commits intermediate tokens to history nor substitutes a time-token vocabulary for the embedding matrix.",
  ),
  makeQuestion(
    "cme296-diffusiongemma-q18",
    3,
    "hard",
    "A response needs 600 valid output tokens with canvas size 256. What follows from block-autoregressive generation?",
    [
      [
        "At least three canvases are needed because two full canvases cover only 512 valid token positions.",
        true,
      ],
      [
        "The first two completed canvases are causally encoded into history before the third canvas is generated.",
        true,
      ],
      [
        "The final canvas can be only partially valid if an end-of-sequence marker occurs before position 256.",
        true,
      ],
      [
        "Bidirectional attention in the third canvas can retroactively edit tokens in the first two canvases if their entropy becomes high.",
        false,
      ],
    ],
    "A 600-token response spans three canvases because each block holds at most 256 valid tokens, and the final block may terminate early. Completed blocks are encoded and appended to the cache so later blocks can condition on them. They are nevertheless frozen history, so uncertainty in the active block cannot reopen or edit the earlier 512 tokens.",
  ),
  makeQuestion(
    "cme296-diffusiongemma-q19",
    3,
    "easy",
    "Which settings define the recommended entropy-bounded sampler?",
    [
      ["Maximum denoising budget \\(N=48\\).", true],
      ["Token-selection entropy budget \\(b=0.1\\).", true],
      ["Adaptive-stopping mean-entropy threshold \\(e_{stop}=0.005\\).", true],
      ["A linear temperature schedule from 0.8 to 0.4.", true],
    ],
    "The default algorithm combines a 48-step maximum, an entropy budget of 0.1 for accepted token positions, a very low 0.005 mean-entropy stopping threshold, and linear temperature annealing from 0.8 to 0.4. These values govern different parts of sampling and should not be conflated. Adaptive stopping commonly ends a canvas far earlier than the maximum budget.",
  ),
  makeQuestion(
    "cme296-diffusiongemma-q20",
    3,
    "medium",
    "After sorting canvas positions by increasing predictive entropy, how does entropy-bounded refinement choose which positions to carry forward?",
    [
      [
        "It accepts the largest low-entropy prefix whose accumulated entropy remains within the budget and uniformly renoises the remaining positions.",
        true,
      ],
      [
        "It accepts the highest-entropy suffix so the sampler preserves the positions with the largest uncertainty.",
        false,
      ],
      [
        "It freezes every position whose argmax token matches the preceding step, regardless of the entropy budget.",
        false,
      ],
      [
        "It discards low-entropy positions and resamples them from the model while copying uncertain positions unchanged.",
        false,
      ],
    ],
    "The sampler ranks positions from most to least confident, measured by low to high entropy, and retains a prefix constrained by the cumulative entropy budget. Positions outside that accepted set are returned to uniform noise so later passes can explore alternatives rather than inherit uncertain samples. Stability of the full argmax sequence is used for stopping, not as a per-position replacement for the entropy budget.",
  ),
  makeQuestion(
    "cme296-diffusiongemma-q21",
    4,
    "hard",
    "Why does the sampler combine temperature annealing with entropy-bounded token acceptance?",
    [
      [
        "The higher early temperature supports broader exploration while the lower late temperature sharpens predictions as structure becomes clearer.",
        true,
      ],
      [
        "Entropy-bounded acceptance preserves sufficiently confident sampled positions while forcing uncertain positions back toward a uniform prior for another attempt.",
        true,
      ],
      [
        "Annealing raises temperature from 0.4 near the noisy source to 0.8 near the clean endpoint so late steps become more diverse.",
        false,
      ],
      [
        "The entropy budget decides when the whole canvas stops, while mean entropy and sequence stability decide how many individual tokens are accepted.",
        false,
      ],
    ],
    "The temperature schedule begins at 0.8 in the highly noised state and falls toward 0.4, changing the distribution from exploratory to sharper as denoising progresses. The token-level entropy budget then limits which sampled positions are trusted and which are renoised. Whole-canvas termination is handled separately by mean entropy plus consecutive argmax stability.",
  ),
  makeQuestion(
    "cme296-diffusiongemma-q22",
    4,
    "easy",
    "Which statements correctly distinguish the sampler's maximum step budget from its adaptive stopping behavior?",
    [
      [
        "The maximum budget is a hard upper limit even when uncertainty remains high.",
        true,
      ],
      [
        "Adaptive stopping can return a canvas earlier when confidence is high and the deterministic prediction has stabilized.",
        true,
      ],
      [
        "Changing the maximum budget exposes an explicit latency-quality control even if adaptive stopping is available.",
        true,
      ],
      [
        "Adaptive stopping guarantees the same number of denoising steps for every prompt in a benchmark.",
        false,
      ],
    ],
    "The maximum \\(N\\) caps work, whereas adaptive stopping makes actual work depend on the model's confidence and stability for each canvas. Consequently, easier or more structured tasks can stop earlier, and harder tasks can consume more of the budget. The maximum remains a deployment control because a lower cap can reduce latency at the risk of less precise generation.",
  ),
  makeAssertionReasonQuestion(
    "cme296-diffusiongemma-q23",
    4,
    "medium",
    "The supervised-fine-tuned checkpoint already maintained strong quality when restricted to the ultra-low-latency few-step sampler.",
    "Its quality collapsed in the few-step regime, motivating a training stage that compressed high-quality trajectories while improving reward.",
    1,
    "The assertion is false because supervised fine-tuning produced strong samples mainly when given many denoising steps; aggressively limiting steps caused severe degradation and repetitive failures. The reason is true and identifies the central motivation for sampler distillation and reinforcement learning. That joint stage was designed to preserve or improve quality while moving generation into the few-step regime.",
  ),
  makeQuestion(
    "cme296-diffusiongemma-q24",
    4,
    "hard",
    "Why must effective denoising steps be weighted by the number of valid tokens in each canvas?",
    [
      [
        "The final canvas is often partially filled because an end-of-sequence token appears before position 256.",
        true,
      ],
      [
        "A simple unweighted average could let a short final canvas influence the metric as much as a full canvas.",
        true,
      ],
      [
        "Token weighting represents the average denoising-step exposure of a valid generated token.",
        true,
      ],
      [
        "Without adaptive stopping, the token-weighted value reduces to the fixed step budget \\(N\\).",
        true,
      ],
    ],
    "Effective denoising steps is defined as \\(\\sum_k N_k C_k / \\sum_k C_k\\), so a canvas contributes in proportion to its valid output length. This avoids an artificially low value when the last, short canvas happens to terminate with fewer steps. If every canvas always executes \\(N\\) steps, the weighted numerator becomes \\(N\\) times total tokens and the metric equals \\(N\\).",
  ),
  makeQuestion(
    "cme296-diffusiongemma-q25",
    4,
    "easy",
    "A generation produces 480 valid tokens across two canvases and executes 18 total denoising steps. Accounting for one between-canvas cache-update pass, what is its tokens per forward (TPF)?",
    [
      ["\\(480/(18+2-1) \\approx 25.3\\) TPF.", true],
      ["\\(480/18 \\approx 26.7\\) TPF.", false],
      ["\\((480+2-1)/18 \\approx 26.7\\) TPF.", false],
      ["\\(18/(480+2-1) \\approx 0.038\\) TPF.", false],
    ],
    "The denominator includes 18 denoising forwards plus \\(K-1=1\\) extra forward to causally encode and append the first completed canvas before the second. Therefore \\(\\mathrm{TPF}=480/19\\), which is about 25.3. Omitting the between-canvas encoding pass slightly overstates efficiency, while reversing the ratio no longer measures tokens per model forward.",
  ),
  makeQuestion(
    "cme296-diffusiongemma-q26",
    4,
    "medium",
    "Which interpretations of the reported inference-efficiency metrics are correct?",
    [
      [
        "Total denoising steps is a primary latency driver because every denoising step invokes the model on a canvas.",
        true,
      ],
      [
        "Tokens per forward includes between-canvas encoding overhead, so it is not simply canvas length divided by effective denoising steps.",
        true,
      ],
      [
        "Total tokens counts all 256 positions in every allocated canvas even after an end-of-sequence token.",
        false,
      ],
      [
        "Effective denoising steps is the unweighted median of the per-canvas step counts.",
        false,
      ],
    ],
    "Total denoising steps directly counts expensive iterative model calls, while TPF normalizes useful tokens by those calls plus the clean-canvas encoding passes between blocks. Total tokens includes only valid positions, ending at the first end-of-sequence marker in a partial canvas. Effective denoising steps is a token-weighted mean, not a median or an equal-weight canvas average.",
  ),
  makeQuestion(
    "cme296-diffusiongemma-q27",
    4,
    "hard",
    "Two canvases produce \\(C_1=256\\) and \\(C_2=64\\) valid tokens using \\(N_1=12\\) and \\(N_2=6\\) denoising steps. Which calculations are correct?",
    [
      ["Total valid tokens are \\(256+64=320\\).", true],
      ["Total denoising steps are \\(12+6=18\\).", true],
      [
        "Effective denoising steps are \\((12\\cdot256+6\\cdot64)/320=10.8\\).",
        true,
      ],
      [
        "The unweighted calculation \\((12+6)/2=9\\) is the reported effective denoising-steps metric.",
        false,
      ],
    ],
    "The two canvases yield 320 valid tokens and consume 18 denoising calls. Token weighting gives the full first canvas four times the influence of the 64-token final canvas, producing \\(3456/320=10.8\\) effective steps. The simple mean of 9 would understate the work experienced by most generated tokens.",
  ),
  makeQuestion(
    "cme296-diffusiongemma-q28",
    4,
    "easy",
    "Which observations support the claim that adaptive computation responds to task structure?",
    [
      [
        "Structured code tasks tend to stop in fewer steps than more open-ended natural-language tasks in the reported distributions.",
        true,
      ],
      [
        "LiveCodeBench problems require more steps than easier HumanEval problems despite both being code benchmarks.",
        true,
      ],
      [
        "A sequential binary-generation rule took seven denoising steps, whereas a parallel rule over a static input took four.",
        true,
      ],
      [
        "Highly constrained JSON extraction converged in two steps because much of the output structure was determined by the schema and input.",
        true,
      ],
    ],
    "Adaptive stopping varies with both domain and instance structure rather than assigning a fixed effort to each token. Harder code, output-dependent sequential rules, and open-ended text leave uncertainty unresolved longer, while static local transformations and rigid schemas permit rapid parallel convergence. The examples illustrate an empirical tendency, not a guarantee for every prompt from a domain.",
  ),
  makeQuestion(
    "cme296-diffusiongemma-q29",
    5,
    "medium",
    "What is optimized during the initial supervised fine-tuning stage?",
    [
      [
        "Cross-entropy between predicted clean tokens and the ground-truth canvas, conditioned on a sampled noisy canvas, context cache, and self-conditioning state.",
        true,
      ],
      [
        "Bidirectional denoising within a 256-token block while context and earlier clean canvases remain available through the causal encoder cache.",
        true,
      ],
      [
        "A reward-only objective that directly minimizes the number of adaptive denoising steps without clean targets.",
        false,
      ],
      [
        "A masked-only corruption process in which ordinary vocabulary tokens never replace one another.",
        false,
      ],
    ],
    "Supervised fine-tuning samples a noise level, corrupts the target canvas with multinomial noise, and trains the model to reconstruct every clean token by cross-entropy. The block-diagonal mask supports bidirectional attention within each denoising block while preventing information leakage from other target blocks. Reward maximization and trajectory compression belong to the later joint online stage.",
  ),
  makeQuestion(
    "cme296-diffusiongemma-q30",
    5,
    "hard",
    "Why was extended supervised fine-tuning especially important for thinking-mode behavior?",
    [
      [
        "Basic non-thinking denoising became useful after a moderate amount of adaptation, but coherent internal reasoning improved on a longer training timescale.",
        true,
      ],
      [
        "Early thinking traces were prone to stuttering and cyclic degeneration even after ordinary denoising had begun to work.",
        true,
      ],
      [
        "Thinking-mode performance began lower but followed a steeper log-linear improvement trend during extended fine-tuning.",
        true,
      ],
      [
        "The extra training was used to replace bidirectional canvas attention with a causal thought-only decoder.",
        false,
      ],
    ],
    "The training curves separate learning to denoise ordinary responses from learning to sustain long coherent reasoning traces. Thinking mode started from a weaker point and suffered repetitive pathologies, yet it continued improving steeply with more supervised updates. The architecture remained a bidirectional diffusion decoder; extended training changed competence rather than replacing its attention pattern.",
  ),
  makeQuestion(
    "cme296-diffusiongemma-q31",
    5,
    "easy",
    "Which functions are combined in sampler distillation and reinforcement learning (SD·RL)?",
    [
      [
        "An online teacher samples high-quality denoising trajectories using a generous step budget.",
        true,
      ],
      [
        "Reward optimization raises capabilities such as helpfulness, reasoning, coding, and instruction following.",
        true,
      ],
      [
        "Sampler distillation compresses the teacher's strong behavior into fewer denoising steps.",
        true,
      ],
      [
        "A single joint training stage advances generation quality and inference efficiency together.",
        true,
      ],
    ],
    "SD·RL deliberately combines two goals that are often handled in separate phases. The online teacher supplies improving high-step trajectories, the reward term pushes their quality upward, and distillation transfers that behavior toward a short sampler. The result is one training process that shifts both axes of the quality-speed Pareto frontier.",
  ),
  makeQuestion(
    "cme296-diffusiongemma-q32",
    5,
    "medium",
    "What creates the implicit curriculum during SD·RL training?",
    [
      [
        "As reward training and distillation lower predictive entropy, adaptive stopping ends teacher trajectories earlier, so later updates increasingly train on shorter paths.",
        true,
      ],
      [
        "A fixed external scheduler halves the maximum denoising budget whenever the reward moving average plateaus, independently of the teacher's predictive entropy or stopping behavior.",
        false,
      ],
      [
        "The curriculum sorts examples by prompt length and removes long examples from later training batches.",
        false,
      ],
      [
        "The teacher is replaced by a separate autoregressive model once its average reward stops increasing.",
        false,
      ],
    ],
    "The curriculum emerges from the interaction between falling predictive entropy and the same adaptive-stopping rule used during sampling. Early uncertain teachers need long trajectories; as confidence grows, they satisfy the stopping conditions sooner and expose the learner to increasingly compressed paths. This also explains why training can keep improving speed after the average reward has largely plateaued.",
  ),
  makeQuestion(
    "cme296-diffusiongemma-q33",
    5,
    "hard",
    "Why could the supervised checkpoint appear to use very few effective steps under a restrictive 48-step sampler while producing poor answers?",
    [
      [
        "Repetitive token loops could become low-entropy states and trigger adaptive stopping prematurely.",
        true,
      ],
      [
        "A low effective-step count can therefore indicate confident degeneration rather than successful compression.",
        true,
      ],
      [
        "The metric automatically removes repeated tokens from total-token accounting, making any loop look fast.",
        false,
      ],
      [
        "The supervised checkpoint was evaluated with a separate verifier that stopped once it detected an incorrect answer.",
        false,
      ],
    ],
    "The stopping heuristic observes confidence and argmax stability, not semantic correctness. A repetitive loop can collapse entropy and stop changing, satisfying both checks even though the generation is unusable. SD·RL reduced these degeneracies, showing why efficiency metrics must be interpreted alongside output quality rather than treated as self-validating.",
  ),
  makeQuestion(
    "cme296-diffusiongemma-q34",
    5,
    "easy",
    "Which effects of SD·RL contribute to the final model's low end-to-end latency?",
    [
      [
        "Tokens per forward rose from about 5 for the supervised frontier to nearly 20 in the joint-trained configuration.",
        true,
      ],
      [
        "The final generations became nearly two times shorter than those of the supervised checkpoint.",
        true,
      ],
      [
        "Fewer effective steps and fewer total generated tokens compound to reduce total model forwards.",
        true,
      ],
      [
        "The optimization doubled response length so that parallel canvas positions would never be left unused.",
        false,
      ],
    ],
    "The joint stage improves step efficiency and also induces concise outputs, so it attacks both factors in total generation work. The paper reports a rise from roughly 5 to nearly 20 TPF and almost a halving of generation length relative to the supervised checkpoint. Brevity acts as a speed multiplier, although it can also limit gains that might come from longer reasoning traces.",
  ),
  makeQuestion(
    "cme296-diffusiongemma-q35",
    5,
    "medium",
    "Which evidence indicates that SD·RL specializes the model for a few-step operating regime?",
    [
      [
        "After joint training, performance improves quickly as the step budget grows to about 48 and then shows diminishing returns.",
        true,
      ],
      [
        "The supervised checkpoint continues benefiting from larger budgets up to roughly 192 steps.",
        true,
      ],
      [
        "Joint training explicitly reduces predictive entropy, making high-quality commitments possible earlier in the reverse process.",
        true,
      ],
      [
        "The reported quality-speed frontier improves in both benchmark score and tokens per forward after joint training.",
        true,
      ],
    ],
    "The step-sweep experiment removes adaptive stopping and temperature annealing, yet still shows that the final checkpoint reaches its useful plateau much earlier than the supervised one. This is consistent with the joint objective compressing probability mass and high-quality behavior into early denoising steps. The accompanying frontier gains show that this is more than merely stopping early at unchanged quality.",
  ),
  makeAssertionReasonQuestion(
    "cme296-diffusiongemma-q36",
    5,
    "hard",
    "A 256-token DiffusionGemma forward pass is approximately 256 times slower than a single-token Gemma 4 autoregressive forward pass on the measured H100 setup.",
    "Full-canvas sampling has negligible cost because it avoids both a vocabulary softmax and a self-conditioning matrix multiplication.",
    2,
    "Both statements are false. The measured diffusion step processes 256 times as many token positions but is only about 3.2 times slower, reflecting far better parallel hardware utilization. Sampling is not negligible: it includes a full-canvas softmax over a 262k-token vocabulary and a self-conditioning embedding multiplication, taking substantially longer than autoregressive single-token sampling.",
  ),
  makeQuestion(
    "cme296-diffusiongemma-q37",
    6,
    "easy",
    "Which operation explains much of the measured mixture-of-experts slowdown for a diffusion canvas?",
    [
      [
        "The 256 canvas tokens collectively activate about 84 unique experts per layer on PG-19, requiring more expert-weight movement than the eight experts used by one autoregressive token.",
        true,
      ],
      [
        "Every one of the 256 canvas positions evaluates all 128 routed experts densely, eliminating sparse activation and making expert arithmetic rather than expert-weight transfer the dominant measured bottleneck.",
        false,
      ],
      [
        "The vision encoder is rerun once per canvas token even for text-only prompts.",
        false,
      ],
      [
        "Causal cache encoding sends every expert's weights from the CPU to the GPU after each denoising step.",
        false,
      ],
    ],
    "Sparse routing remains active, but many tokens in a canvas collectively touch many more unique experts than a single autoregressive token. At batch size one, moving those expert weights from high-bandwidth memory is a major bottleneck, producing the reported MoE kernel slowdown. This is GPU memory traffic within the model, not repeated vision encoding or CPU loading of every expert.",
  ),
  makeQuestion(
    "cme296-diffusiongemma-q38",
    6,
    "medium",
    "Using \\(\\mathrm{TPS}=\\mathrm{TPF}/t_{fwd}\\), which statements match a measured \\(\\mathrm{TPF}=19.74\\) and \\(t_{fwd}=13.56\\) ms?",
    [
      [
        "Converting 13.56 ms to 0.01356 s gives \\(19.74/0.01356 \\approx 1456\\) tokens per second.",
        true,
      ],
      [
        "The estimate is about 7.1 times the reported 204 TPS of the single-token Gemma 4 autoregressive baseline on the same setup.",
        true,
      ],
      [
        "Using \\(19.74/13.56\\approx1.46\\) directly gives the correctly unit-converted throughput in tokens per second.",
        false,
      ],
      [
        "The relation is \\(\\mathrm{TPS}=\\mathrm{TPF}/(t_{fwd}+t_{prefill})\\), so it measures complete request latency from the first input token.",
        false,
      ],
    ],
    "Throughput divides useful tokens advanced per model call by seconds per call, so the millisecond measurement must be divided by 1000 before use. The resulting 1456 TPS is the reported single-request decoding estimate and is about 7.1 times 204 TPS. It excludes prefill and therefore should not be interpreted as full request latency for arbitrary prompt lengths.",
  ),
  makeQuestion(
    "cme296-diffusiongemma-q39",
    6,
    "hard",
    "Which implementation choices reduce DiffusionGemma's per-step overhead without changing its learned text distribution?",
    [
      [
        "FlashAttention-4 accelerates bidirectional attention over the 256-token canvas.",
        true,
      ],
      [
        "Compiled PyTorch primitives implement the full-canvas sampler while preserving extensibility.",
        true,
      ],
      [
        "GPU-side asynchronous scheduling and a per-sequence causal-attention flag avoid extra CPU-GPU synchronization between denoising and cache-update passes.",
        true,
      ],
      [
        "Replacing the entropy-bounded sampler with greedy autoregressive decoding is required to use optimized kernels.",
        false,
      ],
    ],
    "The systems work targets the wall-clock cost of an unchanged diffusion procedure: optimized attention, compiled sampling operations, and GPU-resident scheduling reduce kernel or synchronization overhead. Different requests may need denoising or causal cache-update attention within the same batch, so a per-sequence flag is important. Switching to greedy autoregression would change the generation algorithm rather than merely optimize it.",
  ),
  makeQuestion(
    "cme296-diffusiongemma-q40",
    6,
    "easy",
    "Which statements describe the reported batch-size tradeoff between DiffusionGemma and Gemma 4 with multi-token prediction?",
    [
      [
        "DiffusionGemma leads in both per-user and total throughput in the low-concurrency regime.",
        true,
      ],
      [
        "The autoregressive model begins to gain a total-throughput advantage at around 32 concurrent requests.",
        true,
      ],
      [
        "DiffusionGemma's greater arithmetic per generated token becomes less favorable as batching fills available compute.",
        true,
      ],
      [
        "The high-batch measurements were not yet backed by sampling and kernel choices targeted specifically at large batches.",
        true,
      ],
    ],
    "At low concurrency, autoregressive serving leaves compute underused and remains dominated by repeated memory movement, so DiffusionGemma's parallel work wins on latency and aggregate throughput. As the batch grows, autoregressive work uses the hardware more fully and the diffusion model's higher compute cost becomes important around 32 users. The paper also cautions that its large-batch implementation was not specifically optimized, so this crossover is not a universal architectural constant.",
  ),
  makeQuestion(
    "cme296-diffusiongemma-q41",
    7,
    "medium",
    "Which comparisons are needed to interpret the model-quality results fairly?",
    [
      [
        "DiffusionGemma is evaluated in text-diffusion and retained autoregressive modes, with thinking and no-thinking variants separated.",
        true,
      ],
      [
        "Hardware and precision differ across some external baselines, so raw tokens-per-second figures are accompanied by measurement details.",
        true,
      ],
      [
        "Every compared model completed every benchmark, making missing result entries equivalent to zero scores.",
        false,
      ],
      [
        "Prefill time is included in all reported output-speed values, making them full request-latency comparisons.",
        false,
      ],
    ],
    "The evaluation separates generation algorithm and thinking configuration because each changes quality, output length, and speed. It also identifies devices and numerical precision: for example, DiffusionGemma uses one H100 in FP8, while some diffusion baselines use different accelerators or bfloat16. Missing benchmark coverage is not scored as zero, and output TPS excludes prefill rather than measuring the entire request.",
  ),
  makeQuestion(
    "cme296-diffusiongemma-q42",
    7,
    "hard",
    "Which conclusions are supported by the main quality-speed comparison?",
    [
      [
        "Text-diffusion mode advances the open-weight diffusion quality frontier while raising tokens per forward by roughly an order of magnitude over the reported open diffusion baselines.",
        true,
      ],
      [
        "Relative to Gemma 4 with multi-token prediction, text-diffusion mode sacrifices some benchmark performance but delivers close to five times the output throughput.",
        true,
      ],
      [
        "Loading the converted weights in autoregressive mode recovers part of the capability gap, though it gives up the main diffusion speed advantage.",
        true,
      ],
      [
        "Text-diffusion mode exceeds the starting Gemma 4 checkpoint on every measured capability area while also decoding faster.",
        false,
      ],
    ],
    "The reported contribution is a new Pareto operating point, not universal dominance over the autoregressive initializer. DiffusionGemma is much faster and compares strongly with other diffusion models, but its text-diffusion mode loses some absolute capability relative to Gemma 4. Its retained autoregressive mode closes part of that quality gap and demonstrates a possible latency-aware routing choice.",
  ),
  makeQuestion(
    "cme296-diffusiongemma-q43",
    7,
    "easy",
    "Which values match the seven-benchmark aggregate reported for the main thinking configurations?",
    [
      [
        "DiffusionGemma text diffusion averages about 1,479 output tokens per second.",
        true,
      ],
      [
        "Gemma 4 with multi-token prediction averages about 303 output tokens per second.",
        true,
      ],
      [
        "DiffusionGemma text diffusion averages about 19.74 tokens per forward.",
        true,
      ],
      [
        "The comparable Gemma 4 multi-token-prediction measurement averages about 1.40 tokens per forward.",
        true,
      ],
    ],
    "The aggregate rows combine AIME 2026, GPQA Diamond, LiveCodeBench-v6, MGSM, HumanEval, LBPP, and Natural2Code, where full measurements are available. They report 1,479 TPS and 19.74 TPF for DiffusionGemma's thinking text-diffusion mode versus 303 TPS and 1.40 TPF for Gemma 4 with multi-token prediction. These are averages over that common benchmark subset, not guarantees for every task.",
  ),
  makeQuestion(
    "cme296-diffusiongemma-q44",
    7,
    "medium",
    "A capability-area chart omits a model's coding bar. What is the correct interpretation?",
    [
      [
        "The model lacked complete results for at least one benchmark assigned to that area, so no area mean was shown.",
        true,
      ],
      [
        "The model's mean coding score was exactly zero after normalization to a 0-100 scale.",
        false,
      ],
      [
        "The model generated no valid code tokens, so its tokens-per-forward denominator was undefined.",
        false,
      ],
      [
        "The model was excluded because its output speed was measured on more than one accelerator.",
        false,
      ],
    ],
    "The area visualization reports an unweighted mean only when a model has completed every constituent benchmark in that group. An absent bar is therefore missing coverage, not an inferred zero capability score or a token-generation failure. Hardware differences are disclosed elsewhere but are not the rule that determines whether an area bar appears.",
  ),
  makeQuestion(
    "cme296-diffusiongemma-q45",
    7,
    "hard",
    "Which findings caution against treating thinking mode as an unconditional improvement?",
    [
      [
        "Thinking usually generates many more total tokens and model forwards, so even similar TPS can lead to much longer end-to-end responses.",
        true,
      ],
      [
        "On MMMU-Pro, a multimodal thought-tag formatting failure helped push the thinking score below the no-thinking score.",
        true,
      ],
      [
        "No-thinking mode always uses more effective denoising steps because it lacks an adaptive stopping condition.",
        false,
      ],
      [
        "Thinking disables block-autoregressive caching, forcing the whole response into one 256-token canvas.",
        false,
      ],
    ],
    "Thinking can improve reasoning quality, but it often lengthens traces substantially, increasing total forwards and wall-clock generation time even if per-step throughput remains high. The report also identifies a concrete multimodal formatting bug: omitted closing thought tags hurt thinking-mode evaluation on MMMU-Pro. Both modes use the same canvas and adaptive-stopping machinery.",
  ),
  makeQuestion(
    "cme296-diffusiongemma-q46",
    7,
    "easy",
    "Which caveats apply to the reported Mercury 2 speed estimate?",
    [
      [
        "The estimate is inferred from public API metadata with a non-negative least-squares model because direct forward-pass metrics are unavailable.",
        true,
      ],
      [
        "Per-benchmark output-token coefficients are fit separately before speeds are averaged, allowing task-dependent generation rates.",
        true,
      ],
      [
        "Corrupted 50,000-token thinking loops can bias some aggregate speed estimates upward even though they fail to return valid answers.",
        true,
      ],
      [
        "The estimate measures Mercury 2 on the same local H100, FP8 kernels, and batch-size-one implementation used for DiffusionGemma.",
        false,
      ],
    ],
    "Mercury 2 is closed source, so the authors regress wall-clock time on fresh prefill tokens, output tokens, and a constant overhead using API observations. They fit tasks separately and disclose pathological long thinking loops that can make token generation appear fast despite unusable responses. This is a black-box service estimate, not a controlled same-device H100 benchmark.",
  ),
  makeQuestion(
    "cme296-diffusiongemma-q47",
    8,
    "medium",
    "Which components belong to the released downstream supervised fine-tuning recipe?",
    [
      [
        "A causal encoder loss predicts the next token across the prompt and all clean canvases.",
        true,
      ],
      [
        "A decoder loss chooses one canvas, corrupts it, and reconstructs its clean tokens using the cache for the prompt and prior canvases.",
        true,
      ],
      [
        "Half of the training examples receive a self-conditioning state from a previous forward pass, while the other half use a zero state.",
        true,
      ],
      [
        "The final objective sums encoder and diffusion-decoder cross-entropy losses.",
        true,
      ],
    ],
    "The open-source recipe trains both modes of the shared checkpoint rather than adapting only the diffusion decoder. The encoder sees the clean sequence causally, while the decoder learns one sampled noisy canvas conditioned on the appropriate earlier history; self-conditioning is stochastically present. Adding the two losses preserves next-token behavior while specializing denoising.",
  ),
  makeQuestion(
    "cme296-diffusiongemma-q48",
    8,
    "hard",
    "Which results illustrate how domain adaptation changes both quality and denoising behavior?",
    [
      [
        "A rank-8 LoRA Sudoku model exceeds 80% puzzle accuracy after the unadapted model scores 0%.",
        true,
      ],
      [
        "Sudoku adaptation lowers effective denoising steps from about 40.65 to 10.72 as the model becomes more confident on the task.",
        true,
      ],
      [
        "PubMedQA rank-4 LoRA mainly improves explanation BLEU, from about 10.76 to 20.67, while accuracy rises only slightly.",
        true,
      ],
      [
        "PubMedQA adaptation increases effective denoising steps, showing that specialization can spend more compute when generating richer domain explanations.",
        true,
      ],
    ],
    "Sudoku is a strongly constrained task where adaptation both solves the format and lowers uncertainty, producing a large accuracy gain with far fewer effective steps. PubMedQA begins with reasonable categorical accuracy, and adaptation primarily improves the long explanation text, as reflected by the BLEU gain. Its effective steps rise rather than fall, so downstream adaptation does not have one universal effect on computation.",
  ),
  makeAssertionReasonQuestion(
    "cme296-diffusiongemma-q49",
    8,
    "easy",
    "The open-source downstream objective is the sum of a causal encoder loss and a diffusion decoder loss.",
    "The shared checkpoint is trained both to predict clean sequences autoregressively and to reconstruct a sampled noisy canvas conditioned on earlier clean context.",
    3,
    "Both statements are true, and the reason explains the two terms in the objective. Causal next-token prediction supplies the encoder component, while noisy-canvas reconstruction supplies the decoder component. Training both behaviors is consistent with a checkpoint that remains usable in autoregressive mode while gaining domain-specific diffusion behavior.",
  ),
  makeQuestion(
    "cme296-diffusiongemma-q50",
    8,
    "medium",
    "What makes the provided Low-Rank Adaptation (LoRA) recipe parameter-efficient without restricting it to one transformer subsystem?",
    [
      [
        "Adapters are attached to linear operations across attention, multilayer-perceptron gates, expert routers, and the self-conditioning feedforward block.",
        true,
      ],
      [
        "For Sudoku, rank 8 updates about 8 million parameters and can be trained on two 80GB A100 GPUs.",
        true,
      ],
      [
        "Parameter efficiency comes from updating only the 550M-parameter vision encoder while freezing every text layer.",
        false,
      ],
      [
        "LoRA eliminates the need to train the encoder objective because adapters can affect only bidirectional decoder attention.",
        false,
      ],
    ],
    "The adapters span all major linear transformations, including routing and self-conditioning, so their low-rank parameterization is broad in placement but small in trainable count. The Sudoku recipe reports roughly 8M updated parameters with a two-A100 minimum. It is not a vision-only method, and the paired encoder and decoder losses remain part of the recipe.",
  ),
  makeQuestion(
    "cme296-diffusiongemma-q51",
    8,
    "hard",
    "Which prompt-formatting practices match the released chat template?",
    [
      [
        "Begin the conversation with the beginning-of-sequence token and delimit messages with explicit turn and role tokens.",
        true,
      ],
      [
        "In a multi-turn exchange, retain the previous assistant's final response but omit its hidden thought content from the next context.",
        true,
      ],
      [
        "Represent tool schemas, tool calls, and tool responses with their dedicated control-token delimiters.",
        true,
      ],
      [
        "Insert raw image bytes into the ordinary text-token sequence instead of using image placeholder tokens replaced by visual features.",
        false,
      ],
    ],
    "The template makes roles, turns, channels, tools, and media explicit so the model can distinguish their functions. Previous hidden reasoning is intentionally excluded from future turns, while the final assistant answer remains conversational history. Images enter through placeholder tokens that are replaced by visual features during the forward pass, not as raw bytes masquerading as text tokens.",
  ),
  makeQuestion(
    "cme296-diffusiongemma-q52",
    9,
    "easy",
    "Which observations demonstrate within-canvas bidirectional correction rather than ordinary post-hoc autoregressive correction?",
    [
      [
        "On the arithmetic example, an early answer evolves from -1 to -15 to -25 while later reasoning tokens are also being refined.",
        true,
      ],
      [
        "On the frog puzzle, the initial high-confidence 'Yes' changes to 'No' before the canvas is committed.",
        true,
      ],
      [
        "The final answer and its supporting explanation can influence one another through bidirectional attention during repeated denoising.",
        true,
      ],
      [
        "The correction occurs by editing earlier positions during the reverse process rather than appending a later textual retraction.",
        true,
      ],
    ],
    "Both qualitative traces show an initially plausible but wrong opening token being replaced before the response is finalized. Because all active-canvas positions are repeatedly recomputed with bidirectional attention, emerging reasoning can alter the answer token and the updated answer can shape the rest of the response. An autoregressive model can append a correction, but it cannot rewrite its already emitted prefix.",
  ),
  makeQuestion(
    "cme296-diffusiongemma-q53",
    9,
    "medium",
    "Two binary tasks use the same local truth table. Why did the static-input transformation converge in fewer denoising steps than generation where each output depends on the two preceding outputs?",
    [
      [
        "Static input makes each output position locally computable in parallel, whereas output-to-output dependence propagates information sequentially through the canvas.",
        true,
      ],
      [
        "The static task used a smaller vocabulary, while the sequential task switched to the full 262k-token vocabulary.",
        false,
      ],
      [
        "Adaptive stopping was disabled for the sequential task but enabled for the static transformation.",
        false,
      ],
      [
        "The static transformation was copied from the key-value cache without running the diffusion decoder.",
        false,
      ],
    ],
    "The difference is dependency structure, not the logical rule itself. Every static transformation position can inspect its supplied neighboring input digits immediately, so bidirectional processing resolves them together in four steps. When later outputs depend on earlier generated outputs, information must propagate across denoising iterations, producing a roughly left-to-right resolution that took seven steps.",
  ),
  makeQuestion(
    "cme296-diffusiongemma-q54",
    9,
    "hard",
    "Why are rigid JSON extraction and localized code editing favorable cases for parallel text diffusion?",
    [
      [
        "Schemas or source code constrain many output positions before generation, allowing high-confidence structure to appear across the canvas early.",
        true,
      ],
      [
        "Only a small subset of code positions may require a semantic change while most tokens can align with the provided input in parallel.",
        true,
      ],
      [
        "Autoregressive models can skip all predictable tokens at zero cost once a JSON schema is provided.",
        false,
      ],
      [
        "Diffusion guarantees syntactically valid constrained output after one step without domain fine-tuning or stopping checks.",
        false,
      ],
    ],
    "Strong output priors reduce uncertainty at many positions simultaneously: JSON punctuation and fields come from the schema, while code editing often copies most of the input and changes a localized defect. An autoregressive decoder still emits those predictable positions one by one, whereas diffusion can settle them together. The examples converged in two or three steps, but they do not establish a one-step correctness guarantee.",
  ),
  makeQuestion(
    "cme296-diffusiongemma-q55",
    9,
    "easy",
    "Which practical advantages arise specifically from iterative parallel generation?",
    [
      [
        "Earlier and later token positions can be revised together while an active canvas is still denoising.",
        true,
      ],
      [
        "Adaptive stopping can spend fewer steps on predictable structured outputs and more steps on difficult dependencies.",
        true,
      ],
      [
        "Copied or schema-determined regions can converge concurrently instead of incurring one mandatory model call per output token.",
        true,
      ],
      [
        "Previously committed canvases remain editable for the entire conversation, providing unlimited global backtracking.",
        false,
      ],
    ],
    "Parallel iterative refinement provides three linked benefits: local bidirectional correction, task-dependent computation, and rapid convergence when many positions are predictable from structure or input. These benefits apply within the active block and across its denoising trajectory. The block-autoregressive boundary still freezes earlier canvases, so the mechanism is not unlimited whole-response backtracking.",
  ),
  makeQuestion(
    "cme296-diffusiongemma-q56",
    9,
    "medium",
    "Which deployment strategies are enabled by retaining autoregressive capability in the same final weights?",
    [
      [
        "Route latency-sensitive requests to text diffusion and capability-sensitive requests to autoregressive decoding.",
        true,
      ],
      [
        "Explore hybrid systems in which diffusion proposes blocks and autoregressive computation verifies or completes them.",
        true,
      ],
      [
        "Use the same checkpoint in the original causal Gemma 4 architecture without training a second set of model weights.",
        true,
      ],
      [
        "Choose generation mode per workload rather than treating the speed-quality tradeoff as one immutable checkpoint setting.",
        true,
      ],
    ],
    "The shared transformer architecture and retained causal behavior make generation mode a serving choice rather than a separate model family. A system can route requests according to latency and task complexity or investigate diffusion drafting with autoregressive verification. The paper presents these as promising directions; it does not claim that an optimal router or hybrid sampler is already solved.",
  ),
  makeQuestion(
    "cme296-diffusiongemma-q57",
    10,
    "hard",
    "Which explanation best accounts for the remaining quality gap between text-diffusion mode and the Gemma 4 autoregressive initializer?",
    [
      [
        "The conversion warm-starts from autoregressive weights, uses limited supervised adaptation, inherits autoregressive design choices, and tunes the joint stage for few-step latency rather than maximum asymptotic quality.",
        true,
      ],
      [
        "Warm-starting preserves the same knowledge but diffusion activates fewer routed experts per token, so the quality loss is attributed entirely to a smaller effective parameter count during denoising.",
        false,
      ],
      [
        "The main loss comes from rounding continuous denoising vectors to their nearest token embeddings, while the adaptation duration and speed-oriented objective have little effect on final quality.",
        false,
      ],
      [
        "The apparent gap is primarily an evaluation-accounting artifact: malformed or missing diffusion benchmark rows are counted as zeros, whereas the autoregressive initializer omits those rows from its averages.",
        false,
      ],
    ],
    "Several practical compromises contribute: an autoregressive warm start instead of native diffusion pretraining, comparatively short supervised adaptation, inherited architecture and data choices, and a joint objective tuned for few-step speed. The sparse expert count is not reduced, and this model uses native discrete tokens rather than final continuous-vector rounding. Benchmark rows are reported directly rather than asymmetrically converted to zeros for diffusion.",
  ),
  makeQuestion(
    "cme296-diffusiongemma-q58",
    10,
    "easy",
    "Which limitations directly qualify the headline speed result?",
    [
      [
        "Highly concise outputs reduce total work but can forgo capability gains associated with longer reasoning traces.",
        true,
      ],
      [
        "At sufficiently high batch sizes, the model's greater per-token computation allows autoregressive serving to regain a throughput advantage.",
        true,
      ],
      [
        "The 1,500 TPS result includes prompt prefill, network transfer, and arbitrary high-concurrency production traffic.",
        false,
      ],
      [
        "Adaptive stopping removes every repetitive loop, so rare stuttering cannot affect production generations.",
        false,
      ],
    ],
    "The headline throughput is a low-batch decoding measurement, not a complete production-service latency guarantee. Concision helps speed but limits long-trace reasoning, and high concurrency changes hardware utilization enough for autoregressive models to catch up. Rare stuttering remains possible despite SD·RL, so adaptive stopping and training mitigate rather than eliminate degeneration.",
  ),
  makeQuestion(
    "cme296-diffusiongemma-q59",
    10,
    "medium",
    "Which known issues should an evaluator or application developer monitor?",
    [
      [
        "Occasional localized token repetition can still appear in the aggressively compressed few-step regime.",
        true,
      ],
      [
        "Multimodal thinking responses may omit a closing thought tag even when their underlying reasoning is correct.",
        true,
      ],
      [
        "Text-diffusion mode generally remains below the original Gemma 4 checkpoint in absolute benchmark capability despite its speed advantage.",
        true,
      ],
      [
        "Previously committed canvases are routinely rewritten after later blocks finish, making cached history nondeterministic.",
        false,
      ],
    ],
    "The experimental checkpoint still exhibits rare stuttering, a specific multimodal thought-channel formatting bug, and a quality tradeoff relative to its autoregressive source model. These issues can affect user experience or benchmark parsing even when average results are strong. Committed canvases are deliberately frozen and cached, so later rewriting of history is not one of the listed failure modes.",
  ),
  makeAssertionReasonQuestion(
    "cme296-diffusiongemma-q60",
    10,
    "hard",
    "DiffusionGemma's primary text-diffusion mode is much faster than Gemma 4 with multi-token prediction but scores lower on several capability measures.",
    "The converted checkpoint retains a usable autoregressive generation mode that recovers part of the capability gap.",
    4,
    "Both statements are true, but the retained autoregressive mode does not cause the text-diffusion speed-quality tradeoff. The tradeoff arises from parallel few-step diffusion conversion and optimization choices, whereas autoregressive retention provides an alternative operating mode in the same weights. That alternative motivates dynamic routing or hybrid decoding rather than explaining why diffusion mode is faster and somewhat less capable.",
  ),
];
