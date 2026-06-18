import { Question } from "../../quiz";

type Lecture4Difficulty = "easy" | "medium" | "hard";
type OptionSeed = readonly [text: string, isCorrect: boolean];

function makeQuestion(
  id: string,
  difficulty: Lecture4Difficulty,
  prompt: string,
  optionSeeds: readonly OptionSeed[],
  explanation: string,
): Question {
  if (optionSeeds.length !== 4) {
    throw new Error(`CME295 Lecture 4 question ${id} needs 4 options.`);
  }

  return {
    id,
    chapter: 4,
    difficulty,
    prompt,
    options: optionSeeds.map(([text, isCorrect]) => ({ text, isCorrect })),
    explanation,
  };
}

export const stanfordCME295Lecture4TrainingQuestions: Question[] = [
  makeQuestion(
    "cme295-lect4-q101",
    "easy",
    "A decoder-only language model is trained on the sequence `[BOS] A teddy bear is reading`. Which statements describe the pretraining objective?",
    [
      [
        "The model predicts each next token from the tokens that come before it.",
        true,
      ],
      [
        "The target can be ordinary text or code, because both are token sequences.",
        true,
      ],
      [
        "The model receives a human preference score for every generated answer during this stage.",
        false,
      ],
      [
        "The model is trained mainly on curated user-assistant conversations at this stage.",
        false,
      ],
    ],
    "Pretraining for the models described here is autoregressive next-token prediction over very large token corpora. Human preference scores and user-assistant dialogs belong to later post-training stages, not to the base pretraining objective.",
  ),
  makeQuestion(
    "cme295-lect4-q102",
    "easy",
    "Which data sources fit the broad mixture used for large-scale language-model pretraining?",
    [
      ["Web text such as Common Crawl and Wikipedia-like pages.", true],
      ["Source code from repositories or programming forums.", true],
      [
        "Multilingual text that helps the model learn non-English patterns.",
        true,
      ],
      [
        "Small hand-labeled classification tables with one label per document.",
        false,
      ],
    ],
    "Large language-model pretraining uses broad text and code mixtures so the model learns many token patterns before task-specific tuning. Small labeled classification tables may be useful for supervised tasks, but they do not match the scale or form of the pretraining corpus.",
  ),
  makeQuestion(
    "cme295-lect4-q103",
    "easy",
    "A paper reports that a training run used \\(10^{25}\\) FLOPs, while a GPU vendor reports 1,000 TFLOP/s throughput. Which statements distinguish the two quantities?",
    [
      ["FLOPs count total floating-point operations performed.", true],
      ["FLOP/s describes how quickly hardware can execute operations.", true],
      [
        "The two quantities are interchangeable because both measure the number of model parameters.",
        false,
      ],
      [
        "FLOP/s is the loss value minimized during next-token prediction.",
        false,
      ],
    ],
    "FLOPs measure total arithmetic work, while FLOP/s measures hardware speed. Confusing the two hides the difference between how much work a training run requires and how quickly a device can perform that work.",
  ),
  makeQuestion(
    "cme295-lect4-q104",
    "medium",
    "Use the rough scaling heuristic \\(C \\propto N \\times D\\), where \\(N\\) is parameters and \\(D\\) is training tokens. If both \\(N\\) and \\(D\\) double, what happens to the rough compute estimate?",
    [
      ["It increases by about 4x.", true],
      ["It increases by about 2x because a single factor changes.", false],
      [
        "It stays constant because larger models are more sample efficient.",
        false,
      ],
      [
        "It decreases because more data reduces the need for parameters.",
        false,
      ],
    ],
    "The simplified compute model multiplies the parameter count by the number of training tokens, so doubling both factors gives \\(2N \\times 2D = 4ND\\). Sample efficiency matters for loss, but it does not make the arithmetic work vanish.",
  ),
  makeQuestion(
    "cme295-lect4-q105",
    "medium",
    "A compute-optimal rule of thumb says a model should see about 20 training tokens per parameter. Roughly how many tokens would that suggest for a 70B-parameter model?",
    [
      ["About 1.4T tokens.", true],
      ["About 70B tokens.", false],
      ["About 3.5B tokens.", false],
      [
        "About 20T tokens for every billion parameters, or 1,400T tokens total.",
        false,
      ],
    ],
    "The calculation is \\(70\\text{B} \\times 20 = 1{,}400\\text{B}\\), which is 1.4 trillion tokens. This is a rule of thumb for balancing parameters and data under a compute budget, not a universal law for every architecture or data mixture.",
  ),
  makeQuestion(
    "cme295-lect4-q106",
    "hard",
    "GPT-3-style pretraining used about 175B parameters and 300B tokens. Which statements follow from comparing this to a 20-token-per-parameter compute-optimal target?",
    [
      ["The run used about 1.7 training tokens per parameter.", true],
      [
        "A 20-token-per-parameter target would be about 3.5T tokens for 175B parameters.",
        true,
      ],
      [
        "The 300B-token run was data-heavy relative to the 20-token-per-parameter rule.",
        false,
      ],
      [
        "The ratio proves that the model had too few parameters for its data.",
        false,
      ],
    ],
    "The token-per-parameter ratio is approximately \\(300/175 \\approx 1.7\\), far below 20. That comparison motivates the undertrained-model interpretation, but it is still an empirical scaling argument rather than a proof that only one design would work.",
  ),
  makeQuestion(
    "cme295-lect4-q107",
    "medium",
    "Two teams process the same number of tokens. Team A trains a larger model than Team B. Which statements correctly describe the sample-efficiency claim from scaling-law studies?",
    [
      [
        "The larger model can reach lower loss after seeing the same number of tokens.",
        true,
      ],
      [
        "The larger model can still be wasteful if the fixed compute budget would have been better spent on more data.",
        true,
      ],
      [
        "The larger model automatically removes the need to tune data mixture quality.",
        false,
      ],
      [
        "The larger model is sample efficient because it performs fewer floating-point operations per token.",
        false,
      ],
    ],
    "Sample efficiency means larger models can make better use of a given amount of data, not that they are cheaper per token. Compute-optimal training still asks whether the chosen parameter count and token count are balanced for the available budget.",
  ),
  makeQuestion(
    "cme295-lect4-q108",
    "easy",
    "Which issues follow from storing model knowledge in weights learned from a fixed pretraining corpus?",
    [
      [
        "The model has an intrinsic knowledge cutoff tied to the data it was trained on.",
        true,
      ],
      [
        "Editing one fact in the weights can have side effects on other behavior.",
        true,
      ],
      ["The model can memorize or reproduce pieces of training data.", true],
      [
        "The model automatically knows any event that happens after deployment.",
        false,
      ],
    ],
    "A base model only learns from data that was available during training, so fresh facts need retrieval, retraining, or another update mechanism. Because knowledge is distributed across parameters, targeted edits and memorization control are difficult engineering problems.",
  ),
  makeQuestion(
    "cme295-lect4-q109",
    "easy",
    "Which statements correctly describe transfer learning in the LLM training pipeline?",
    [
      [
        "A general pretrained model is reused instead of starting every task from random weights.",
        true,
      ],
      [
        "Later tuning can adapt the model toward spam detection, sentiment extraction, assistant behavior, or another task.",
        true,
      ],
      [
        "Transfer learning means pretraining is skipped because the final task data is enough.",
        false,
      ],
      [
        "Transfer learning requires each downstream task to use a separate tokenizer and architecture.",
        false,
      ],
    ],
    "Transfer learning is the staged reuse of a broadly trained model for later tasks. The point is to avoid learning general language structure from scratch for every use case, while still allowing task-specific tuning.",
  ),
  makeQuestion(
    "cme295-lect4-q110",
    "medium",
    "A team has a fixed accelerator budget. Which design choices are part of the compute-optimal scaling tradeoff?",
    [
      ["How many model parameters to train.", true],
      ["How many training tokens to process.", true],
      ["Whether the architecture and hardware change the FLOP constant.", true],
      [
        "Whether to ignore data quality because the parameter count alone determines loss.",
        false,
      ],
    ],
    "Compute-optimal scaling is a tradeoff among model size, data size, and implementation details under a fixed budget. Data quality still matters because more tokens are not equally useful if they are duplicated, noisy, or poorly matched to the target domain.",
  ),
  makeQuestion(
    "cme295-lect4-q111",
    "medium",
    "A corpus contains 2 million documents, each averaging 750 tokens after tokenization. What is the approximate corpus size in tokens?",
    [
      ["1.5 billion tokens.", true],
      ["1.5 million tokens.", false],
      ["750 million tokens.", false],
      ["2.75 trillion tokens.", false],
    ],
    "The token count is \\(2{,}000{,}000 \\times 750 = 1{,}500{,}000{,}000\\), or 1.5 billion tokens. LLM dataset scale is usually discussed in tokens because token count directly affects training compute and optimization steps.",
  ),
  makeQuestion(
    "cme295-lect4-q112",
    "easy",
    "A pretrained base model responds to a user question by continuing with encyclopedia-style facts rather than giving practical advice. Which statements best explain the behavior?",
    [
      [
        "The base model has learned likely text continuations, not necessarily assistant-style helpfulness.",
        true,
      ],
      [
        "Instruction tuning can shift the same base model toward direct answers to user requests.",
        true,
      ],
      [
        "The behavior proves that pretraining failed to learn language structure.",
        false,
      ],
      [
        "The behavior is caused by FlashAttention changing the attention output.",
        false,
      ],
    ],
    "A base model can understand many token patterns but still behave like a continuation engine. Supervised fine-tuning and instruction tuning teach the model the response style and task framing expected from an assistant.",
  ),
  makeQuestion(
    "cme295-lect4-q113",
    "easy",
    "During a large-model training step, which objects may need GPU memory?",
    [
      ["Model parameters.", true],
      ["Forward-pass activations.", true],
      ["Gradients from the backward pass.", true],
      ["Optimizer state such as Adam moments.", true],
    ],
    "Training requires much more memory than just the final model weights. Activations, gradients, and optimizer state can dominate the memory budget, especially with long context lengths, large batches, and adaptive optimizers.",
  ),
  makeQuestion(
    "cme295-lect4-q114",
    "medium",
    "A team doubles the microbatch size while keeping sequence length and model architecture fixed. Which training-memory effects are most directly expected?",
    [
      ["Activation memory generally increases.", true],
      [
        "The parameter memory for a dense model generally stays the same.",
        true,
      ],
      ["The self-attention matrix for each sequence becomes shorter.", false],
      [
        "The Adam optimizer state disappears because more examples are processed together.",
        false,
      ],
    ],
    "More examples in a microbatch require more intermediate activations to be stored for backpropagation. The model parameters and optimizer state are properties of the model and optimizer, not of the number of examples in one microbatch.",
  ),
  makeQuestion(
    "cme295-lect4-q115",
    "medium",
    "For one attention head with sequence length \\(L=2{,}048\\), how many entries are in the dense score matrix \\(QK^\\top\\) before masking or softmax?",
    [
      ["\\(2{,}048^2 = 4{,}194{,}304\\) entries.", true],
      ["\\(2\\times 2{,}048 = 4{,}096\\) entries.", false],
      ["\\(2{,}048\\log_2 2{,}048 = 22{,}528\\) entries.", false],
      ["\\(2{,}048\\) entries because each token only scores itself.", false],
    ],
    "Dense self-attention compares each query position with each key position, producing an \\(L\\times L\\) score matrix for each head. This quadratic shape is why long context length pressures activation memory and motivates IO-aware attention kernels.",
  ),
  makeQuestion(
    "cme295-lect4-q116",
    "hard",
    "A model has 10B parameters and an optimizer stores two FP32 Adam moment values per parameter. Ignoring gradients and the parameters themselves, how much memory do the two moment buffers require?",
    [
      [
        "About 80 GB, because \\(10\\text{B}\\times 2\\times 4\\) bytes = 80B bytes.",
        true,
      ],
      [
        "About 20 GB, because \\(10\\text{B}\\times 2\\) bytes = 20B bytes.",
        false,
      ],
      [
        "About 10 GB, because \\(10\\text{B}\\times 1\\) byte = 10B bytes.",
        false,
      ],
      [
        "About 160 GB, because \\(10\\text{B}\\times 4\\times 4\\) bytes = 160B bytes.",
        false,
      ],
    ],
    "Each FP32 value is 4 bytes, and two moment buffers store two such values per parameter. This simple calculation shows why optimizer state is a major memory object during training, even before counting parameters, gradients, and activations.",
  ),
  makeQuestion(
    "cme295-lect4-q117",
    "medium",
    "Which statements correctly describe ordinary data parallelism?",
    [
      ["Each device processes a different slice of the batch.", true],
      [
        "Each device typically holds a full copy of the model parameters.",
        true,
      ],
      [
        "Each device owns a different contiguous range of transformer layers.",
        false,
      ],
      [
        "It removes the need to synchronize gradients before a shared update.",
        false,
      ],
    ],
    "Data parallelism splits data while replicating the model, so the devices must communicate gradient information to make a consistent update. Layer ownership is a model-parallel or pipeline-parallel idea, not ordinary data parallelism.",
  ),
  makeQuestion(
    "cme295-lect4-q118",
    "medium",
    "A 90B-parameter dense model cannot fit a full copy of its weights on one GPU. Why is plain data parallelism alone insufficient?",
    [
      [
        "Plain data parallelism still requires each device to hold the full model copy.",
        true,
      ],
      [
        "It can split examples across devices while still replicating parameter memory.",
        true,
      ],
      [
        "The system needs sharding, model parallelism, offload, or related memory-saving work to fit the state.",
        true,
      ],
      [
        "Plain data parallelism is primarily a decoding-time sampling method.",
        false,
      ],
    ],
    "The memory blocker is the full model replica on each device. Splitting the batch helps activation memory per device, but it does not solve the problem of a model whose parameters and optimizer state cannot fit on one GPU.",
  ),
  makeQuestion(
    "cme295-lect4-q119",
    "medium",
    "Which statements correctly match ZeRO stages to what gets sharded?",
    [
      ["ZeRO-1 shards optimizer state.", true],
      ["ZeRO-2 shards optimizer state and gradients.", true],
      ["ZeRO-3 shards optimizer state, gradients, and parameters.", true],
      [
        "ZeRO stages work by turning every transformer layer into a separate tokenizer.",
        false,
      ],
    ],
    "Zero Redundancy Optimization reduces duplicated training state across devices. The stages progressively shard optimizer state, gradients, and then parameters, but they do not change tokenization or the mathematical role of transformer layers.",
  ),
  makeQuestion(
    "cme295-lect4-q120",
    "hard",
    "A replicated optimizer state takes 96 GB per GPU. If ZeRO-style sharding splits only that optimizer state evenly across 8 GPUs, what is the per-GPU optimizer-state footprint?",
    [
      ["12 GB per GPU.", true],
      ["8 GB per GPU.", false],
      ["96 GB per GPU, because \\(96/8=96\\) for optimizer state.", false],
      ["768 GB per GPU, because each device stores every shard.", false],
    ],
    "Even sharding divides the replicated object by the number of devices, so \\(96/8=12\\) GB per GPU. Real systems also pay communication and scheduling costs to gather needed shards at the right time.",
  ),
  makeQuestion(
    "cme295-lect4-q121",
    "medium",
    "Which methods are forms of model parallelism rather than ordinary data parallelism?",
    [
      ["Tensor parallelism for splitting large matrix operations.", true],
      ["Pipeline parallelism for assigning layer ranges to devices.", true],
      [
        "Expert parallelism for placing MoE experts on different devices.",
        true,
      ],
      [
        "Randomly shuffling the examples before making a single-device batch.",
        false,
      ],
    ],
    "Model parallelism partitions model computation, while data parallelism partitions examples. Tensor, pipeline, and expert parallelism split different parts of the model computation across accelerators.",
  ),
  makeQuestion(
    "cme295-lect4-q122",
    "medium",
    "A transformer block contains a very large matrix multiplication and an MoE feed-forward layer. Which statements match parallelism choices to the bottleneck?",
    [
      [
        "Tensor parallelism can split large matrix multiplications across devices.",
        true,
      ],
      [
        "Expert parallelism can place different experts on different devices.",
        true,
      ],
      [
        "Pipeline parallelism is the method that samples top-p tokens at inference time.",
        false,
      ],
      [
        "Data parallelism makes a single matrix multiplication smaller by slicing its columns.",
        false,
      ],
    ],
    "Tensor parallelism is about splitting tensor operations, while expert parallelism is about distributing expert modules. Top-p sampling is a decoding method, and ordinary data parallelism does not slice one matrix multiplication.",
  ),
  makeQuestion(
    "cme295-lect4-q123",
    "hard",
    "A 24-layer decoder is placed on 4 pipeline stages with equal layer counts. How many layers does each stage own in the simplest balanced split?",
    [
      ["6 layers per stage.", true],
      ["4 layers per stage.", false],
      ["24 layers per stage.", false],
      ["96 layers per stage.", false],
    ],
    "The simplest equal split assigns \\(24/4=6\\) layers to each stage. Pipeline parallelism can reduce per-device memory for layers, but it introduces scheduling issues such as pipeline bubbles.",
  ),
  makeQuestion(
    "cme295-lect4-q124",
    "medium",
    "Which statements correctly describe expert parallelism for a sparse MoE layer?",
    [
      ["Different experts can be stored on different devices.", true],
      [
        "Token routing can create communication and load-balancing challenges.",
        true,
      ],
      [
        "Most tokens run the full expert set before routing scores are applied.",
        false,
      ],
      [
        "Expert parallelism replaces the next-token objective with majority voting.",
        false,
      ],
    ],
    "Expert parallelism uses the sparsity of MoE layers to distribute expert modules, but routed tokens may need to travel to the device that owns a selected expert. The language-model objective remains next-token prediction unless another training objective is explicitly added.",
  ),
  makeQuestion(
    "cme295-lect4-q125",
    "hard",
    "A training system uses data parallelism plus ZeRO plus tensor parallelism. Which tradeoffs should the team expect?",
    [
      ["Lower per-device memory pressure than naive replication.", true],
      ["More device-to-device communication than a single-GPU run.", true],
      ["More complex scheduling and state movement.", true],
      ["No need to compute gradients.", false],
    ],
    "Parallelism methods help a large run fit and scale, but they introduce coordination costs. None of these methods removes the need for backpropagation; they reorganize where the training work and state live.",
  ),
  makeQuestion(
    "cme295-lect4-q126",
    "easy",
    "Which statements correctly describe the forward, backward, and update phases of training?",
    [
      ["The forward pass computes activations and loss.", true],
      ["The backward pass computes gradients.", true],
      [
        "The optimizer update changes parameters using gradient information.",
        true,
      ],
      [
        "The update phase is where the tokenizer decides the vocabulary size for the first time.",
        false,
      ],
    ],
    "The training loop computes predictions and loss, propagates gradient information backward, and then updates parameters with an optimizer. Tokenizer design is a preprocessing/model-design issue, not the optimizer update step.",
  ),
  makeQuestion(
    "cme295-lect4-q127",
    "medium",
    "Why are forward-pass activations usually stored during training?",
    [
      [
        "They are needed to compute gradients efficiently during the backward pass.",
        true,
      ],
      [
        "They connect each layer output to the loss that is differentiated later.",
        true,
      ],
      [
        "They are stored because the model will reuse them as a KV cache during all future inference requests.",
        false,
      ],
      [
        "They replace the trainable parameters after the first optimizer step.",
        false,
      ],
    ],
    "Backpropagation needs information from the forward computation to calculate gradients with respect to parameters. KV caching is an inference optimization for autoregressive decoding, not the reason training stores activations.",
  ),
  makeQuestion(
    "cme295-lect4-q128",
    "medium",
    "A team increases both context length and batch size. Which statements explain why memory pressure rises?",
    [
      [
        "More tokens per sequence create more attention and activation values.",
        true,
      ],
      [
        "More sequences per batch require storing more per-example activations.",
        true,
      ],
      [
        "Longer context can make attention-related memory grow quadratically in sequence length.",
        true,
      ],
      [
        "Increasing context length reduces the number of layers in the model.",
        false,
      ],
    ],
    "Training memory grows with both how many examples are processed and how many token positions each example contains. Self-attention is especially sensitive to sequence length because the dense attention score shape is tied to pairs of positions.",
  ),
  makeQuestion(
    "cme295-lect4-q129",
    "medium",
    "An H100-class GPU has memory on the order of tens of GB, but a training run needs hundreds of GB of parameters, optimizer state, gradients, and activations. Which conclusion is most appropriate?",
    [
      [
        "The training system must distribute state, computation, or both across devices.",
        true,
      ],
      [
        "The model must be converted into a recurrent neural network before training.",
        false,
      ],
      [
        "The optimizer state can be ignored because it is not used for updates.",
        false,
      ],
      [
        "The context length should not affect memory because attention is constant-size.",
        false,
      ],
    ],
    "The memory mismatch motivates distributed training and memory-saving optimizations. Changing the architecture family is not the direct conclusion, and optimizer state plus long-context activations are real parts of the memory budget.",
  ),
  makeQuestion(
    "cme295-lect4-q130",
    "hard",
    "A model stores BF16 parameters using 2 bytes per parameter and two FP32 Adam moments using 8 bytes total per parameter. Which statements are correct for optimizer state versus parameter memory?",
    [
      [
        "The Adam moments can take about 4x as much memory as the BF16 parameters.",
        true,
      ],
      ["For 30B parameters, BF16 parameter storage is about 60 GB.", true],
      [
        "The Adam moments require no memory because they are recomputed from the current batch only.",
        false,
      ],
      [
        "Using BF16 parameters automatically makes all optimizer state BF16 as well.",
        false,
      ],
    ],
    "Two FP32 moments use \\(2\\times 4=8\\) bytes per parameter, which is four times the 2 bytes used by BF16 parameter storage. Optimizer implementation choices vary, but adaptive optimizer state is a central memory cost during large-model training.",
  ),
  makeQuestion(
    "cme295-lect4-q131",
    "hard",
    "Let \\(Q\\in\\mathbb{R}^{L\\times d}\\), \\(K\\in\\mathbb{R}^{L\\times d}\\), and \\(V\\in\\mathbb{R}^{L\\times d_v}\\). What is the shape of the standard attention output \\(\\mathrm{softmax}(QK^\\top/\\sqrt d)V\\)?",
    [
      ["\\(L\\times d_v\\).", true],
      ["\\(d\\times d_v\\).", false],
      ["\\(L\\times L\\).", false],
      ["\\(d_v\\times L\\).", false],
    ],
    "\\(QK^\\top\\) has shape \\(L\\times L\\), and multiplying the normalized scores by \\(V\\) returns one output vector per query position. The output therefore has the sequence dimension \\(L\\) and the value dimension \\(d_v\\).",
  ),
  makeQuestion(
    "cme295-lect4-q132",
    "medium",
    "In a straightforward attention implementation, which intermediate objects are commonly written to and read from high-bandwidth memory (HBM)?",
    [
      ["The attention score matrix \\(S=QK^\\top/\\sqrt d\\).", true],
      ["The post-softmax probability matrix \\(P\\).", true],
      ["A separately trained reward model for every attention head.", false],
      ["A permanent natural-language explanation for each token pair.", false],
    ],
    "A standard implementation can materialize large score and probability matrices, creating expensive HBM traffic. Reward models and textual explanations are unrelated to the low-level attention kernel.",
  ),
  makeQuestion(
    "cme295-lect4-q133",
    "medium",
    "Which statements correctly distinguish FlashAttention from approximate sparse attention?",
    [
      [
        "FlashAttention computes the same mathematical attention result as standard attention, up to normal numerical behavior.",
        true,
      ],
      ["Its speedup comes from IO-aware tiling and reduced HBM traffic.", true],
      [
        "It speeds up attention by dropping low-probability keys before softmax.",
        false,
      ],
      [
        "It replaces the value matrix \\(V\\) with a learned reward model.",
        false,
      ],
    ],
    "FlashAttention is exact attention reorganized around GPU memory hierarchy. It is not the same idea as sparsifying attention patterns or changing the probability distribution by pruning keys.",
  ),
  makeQuestion(
    "cme295-lect4-q134",
    "hard",
    "Which statements describe the tiling strategy behind FlashAttention?",
    [
      [
        "Blocks of \\(Q\\), \\(K\\), and \\(V\\) are loaded into fast SRAM.",
        true,
      ],
      ["Partial attention outputs are computed block by block.", true],
      [
        "The implementation avoids materializing the full probability matrix in HBM.",
        true,
      ],
      [
        "The implementation stores a full copy of every layer on every GPU expert.",
        false,
      ],
    ],
    "The key systems idea is to keep small blocks near compute and avoid repeated slow memory movement for giant intermediate matrices. Storing full model layers on every expert is not part of the FlashAttention algorithm.",
  ),
  makeQuestion(
    "cme295-lect4-q135",
    "medium",
    "Which statements about recomputation in FlashAttention-style backward passes are correct?",
    [
      ["Some forward intermediates can be discarded rather than stored.", true],
      ["Needed values can be recomputed during the backward pass.", true],
      [
        "The tradeoff can reduce memory traffic enough to improve runtime.",
        true,
      ],
      [
        "Recomputation still uses the needed inputs to recover exact intermediate quantities.",
        true,
      ],
    ],
    "The recomputation strategy uses the original inputs to regenerate needed quantities instead of storing every intermediate. It trades arithmetic for lower memory movement and storage while preserving the mathematical computation needed for gradients.",
  ),
  makeQuestion(
    "cme295-lect4-q136",
    "hard",
    "A kernel does 15% more arithmetic but cuts slow HBM traffic enough that wall-clock time falls. Which explanation matches the FlashAttention lesson?",
    [
      [
        "Runtime can be limited by memory IO, so more FLOPs can still be faster when memory traffic falls.",
        true,
      ],
      [
        "Arithmetic count determines runtime independently of memory hierarchy.",
        false,
      ],
      [
        "The kernel must be approximate because exact algorithms cannot recompute values.",
        false,
      ],
      ["The speedup must come from changing the learned model weights.", false],
    ],
    "GPU runtime is not determined only by total arithmetic; memory movement can dominate. FlashAttention exploits this by reducing HBM reads and writes while preserving the exact attention computation.",
  ),
  makeQuestion(
    "cme295-lect4-q137",
    "hard",
    "For sequence length \\(L=4{,}096\\), how many entries would a full attention probability matrix \\(P\\) contain for one head?",
    [
      ["\\(4{,}096^2 = 16{,}777{,}216\\) entries.", true],
      ["\\(4{,}096\\) entries.", false],
      ["\\(4{,}096\\times 12 = 49{,}152\\) entries.", false],
      ["\\(\\sqrt{4{,}096}=64\\) entries.", false],
    ],
    "A dense attention probability matrix compares every query position with every key position, so it has \\(L^2\\) entries. FlashAttention avoids storing that full matrix in HBM, which becomes increasingly valuable as context length grows.",
  ),
  makeQuestion(
    "cme295-lect4-q138",
    "medium",
    "Which statements correctly compare HBM and SRAM on a GPU?",
    [
      [
        "HBM is much larger and is the main GPU memory visible in hardware specifications.",
        true,
      ],
      [
        "SRAM is smaller but closer to compute units and faster to access.",
        true,
      ],
      [
        "Attention kernels can improve speed by reducing unnecessary HBM movement.",
        true,
      ],
      [
        "HBM and SRAM have identical size and latency, so tiling cannot matter.",
        false,
      ],
    ],
    "The memory hierarchy is central to IO-aware attention. HBM provides capacity, while SRAM provides fast on-chip storage for small tiles that can be processed near the compute units.",
  ),
  makeQuestion(
    "cme295-lect4-q139",
    "medium",
    "Which statements correctly compare FP16, FP32, FP64, and BF16-style floating-point formats?",
    [
      [
        "FP32 uses more bits than FP16 and can represent more mantissa precision.",
        true,
      ],
      [
        "BF16 keeps an exponent width like FP32 but uses fewer mantissa bits.",
        true,
      ],
      [
        "FP64 and FP16 use the same number of bits and therefore the same memory.",
        false,
      ],
      [
        "The sign bit alone determines the numerical precision of a floating-point format.",
        false,
      ],
    ],
    "Floating-point formats allocate bits among sign, exponent, and mantissa fields. Lower-precision formats save memory and can increase throughput, but the exact exponent and mantissa allocation affects numerical behavior.",
  ),
  makeQuestion(
    "cme295-lect4-q140",
    "medium",
    "A tensor has 1 billion scalar values. Ignoring overhead, how much memory is saved by storing it in FP16 instead of FP32?",
    [
      [
        "About 2 GB, because FP32 uses 4 bytes and FP16 uses 2 bytes per value.",
        true,
      ],
      ["About 1 MB, because the format name only changes metadata.", false],
      ["No memory is saved because both formats use 32 bits.", false],
      ["About 16 GB, because FP16 uses 16 bytes per value.", false],
    ],
    "FP32 uses 4 bytes per scalar and FP16 uses 2 bytes per scalar, so 1 billion values drop from about 4 GB to about 2 GB. The exact storage can include overhead, but the basic precision-memory tradeoff is the point.",
  ),
  makeQuestion(
    "cme295-lect4-q141",
    "medium",
    "Which statements correctly describe mixed precision training?",
    [
      [
        "Forward activations may be computed or stored in lower precision.",
        true,
      ],
      [
        "Backward gradient computations may use lower precision where safe.",
        true,
      ],
      [
        "Sensitive master weights or updates can be kept in higher precision.",
        true,
      ],
      ["Mixed precision means tensors are rounded to one bit.", false],
    ],
    "Mixed precision is selective: it uses lower precision where it buys speed or memory without unacceptable loss, while preserving higher precision where accumulation error would be harmful. It is not a blanket conversion of all training values to the lowest possible precision.",
  ),
  makeQuestion(
    "cme295-lect4-q142",
    "medium",
    "Which statements explain why lower precision can speed up training on modern accelerators?",
    [
      ["Lower-precision values can reduce memory bandwidth pressure.", true],
      [
        "Hardware often has higher throughput for lower-precision matrix operations.",
        true,
      ],
      [
        "The speedup comes from representation and hardware efficiency, not from changing the loss objective.",
        true,
      ],
      [
        "Precision choices still need validation because overly coarse formats can hurt stability.",
        true,
      ],
    ],
    "The practical benefits are memory and throughput, not magic quality improvements or removal of optimization. Numerical choices still need care because too little precision can destabilize or degrade training.",
  ),
  makeQuestion(
    "cme295-lect4-q143",
    "medium",
    "Which statements correctly describe quantization in the training or fine-tuning context?",
    [
      ["It represents weights or other values with fewer bits.", true],
      [
        "It can reduce memory footprint and sometimes improve throughput.",
        true,
      ],
      ["It keeps quality unchanged across model and layer choices.", false],
      [
        "It changes a decoder-only transformer into an encoder-only model.",
        false,
      ],
    ],
    "Quantization changes numeric representation, not the high-level transformer architecture. It can be very useful for memory and speed, but quality depends on the format, the layer, and whether computations need dequantized or higher-precision paths.",
  ),
  makeQuestion(
    "cme295-lect4-q144",
    "hard",
    "Which statements explain why mixed precision often keeps important weights or updates in higher precision?",
    [
      ["Repeated low-precision updates can accumulate rounding error.", true],
      [
        "Small update directions can be lost if the format is too coarse.",
        true,
      ],
      [
        "Higher precision can preserve small accumulated changes that low precision may round away.",
        true,
      ],
      [
        "Keeping some state in higher precision is a numerical-stability choice, not a tokenization choice.",
        true,
      ],
    ],
    "The concern is numerical stability in the training dynamics: small updates and accumulated state can be damaged by overly coarse formats. Master weights or sensitive update state are often preserved at higher precision even when activations or gradients use lower precision.",
  ),
  makeQuestion(
    "cme295-lect4-q145",
    "hard",
    "If context length doubles from \\(L\\) to \\(2L\\), how does the number of entries in a dense attention score matrix change?",
    [
      ["It increases by 4x, from \\(L^2\\) to \\((2L)^2=4L^2\\).", true],
      ["It increases by 2x, from \\(L^2\\) to \\(2L^2\\).", false],
      [
        "It stays constant at \\(L^2\\) because the model has the same number of layers.",
        false,
      ],
      [
        "It decreases to \\(L^2/2\\) because masking removes future positions.",
        false,
      ],
    ],
    "A dense attention matrix contains one score for each query-key position pair, so its size is quadratic in sequence length. Causal masking changes which scores are usable, but a straightforward dense representation still scales with the number of token pairs.",
  ),
  makeQuestion(
    "cme295-lect4-q146",
    "hard",
    "A training run doubles both sequence length and batch size. Which statements follow for activation pressure in a decoder-only transformer?",
    [
      [
        "The number of token positions in the batch doubles from the batch-size change alone.",
        true,
      ],
      [
        "Attention-related per-sequence score storage can grow by about 4x from the sequence-length change alone.",
        true,
      ],
      [
        "The two changes are harmless because activations depend only on vocabulary size.",
        false,
      ],
      [
        "The model skips gradient computation because there are more tokens per step.",
        false,
      ],
    ],
    "Batch size affects how many examples produce activations, and sequence length affects how many token positions and attention pairs exist inside each example. Both changes can increase memory pressure even if parameter count is unchanged.",
  ),
  makeQuestion(
    "cme295-lect4-q147",
    "medium",
    "A colleague claims FlashAttention speeds up training by approximating softmax with a cheaper heuristic. Which response is most accurate?",
    [
      [
        "FlashAttention is designed to compute exact attention while reorganizing memory access.",
        true,
      ],
      [
        "The claim is right because FlashAttention drops half the keys before softmax.",
        false,
      ],
      [
        "The claim is right because FlashAttention replaces attention with a recurrent layer.",
        false,
      ],
      [
        "The claim is right because FlashAttention changes the training labels.",
        false,
      ],
    ],
    "The central distinction is exactness: FlashAttention changes how the computation is scheduled on GPU memory, not the mathematical attention formula. Approximating attention, changing architectures, or changing labels would be different interventions.",
  ),
  makeQuestion(
    "cme295-lect4-q148",
    "easy",
    "Which statements correctly distinguish training-time and inference-time GPU requirements?",
    [
      [
        "Training stores gradients and optimizer state, while inference usually does not.",
        true,
      ],
      [
        "Autoregressive inference can store KV cache to avoid recomputing previous keys and values.",
        true,
      ],
      [
        "Training stores forward activations so the backward pass can compute gradients.",
        true,
      ],
      ["Inference runs without a backward pass or optimizer update.", true],
    ],
    "Training and inference stress hardware differently. Training must support forward, backward, and optimizer updates, while inference focuses on fast generation and often trades memory for speed through KV caching.",
  ),
  makeQuestion(
    "cme295-lect4-q149",
    "easy",
    "Which statements correctly describe supervised fine-tuning (SFT)?",
    [
      [
        "It trains from pretrained weights rather than from random initialization.",
        true,
      ],
      [
        "It uses input-output examples that demonstrate desired behavior.",
        true,
      ],
      [
        "It can use a much smaller supervised dataset than the pretraining corpus.",
        true,
      ],
      [
        "It adapts behavior after pretraining and before later alignment stages.",
        true,
      ],
    ],
    "SFT adapts a pretrained model using supervised examples, often to make it follow tasks or instructions. It is still neural training, not a switch to rules or a repeat of broad pretraining without target outputs.",
  ),
  makeQuestion(
    "cme295-lect4-q150",
    "medium",
    "An instruction-tuning example is `[BOS] Write a short story. Sure, here is a story...`. Where is the supervised loss typically applied?",
    [
      ["On the target response tokens conditioned on the instruction.", true],
      ["Only on the instruction tokens before the response begins.", false],
      [
        "Only on a final scalar user vote after the whole response is generated.",
        false,
      ],
      [
        "Only on the GPU memory addresses touched by the attention kernel.",
        false,
      ],
    ],
    "In SFT, the instruction is context and the desired response is the supervised continuation to predict. User votes and memory addresses belong to other topics: preference evaluation and low-level kernel implementation.",
  ),
  makeQuestion(
    "cme295-lect4-q151",
    "easy",
    "Which examples fit instruction-tuning data for making a model more assistant-like?",
    [
      ["A user instruction paired with a helpful answer.", true],
      ["A math or reasoning prompt paired with a worked response.", true],
      [
        "A safety-related prompt paired with an appropriate refusal or redirection.",
        true,
      ],
      ["A raw dump of shuffled tokens with no input-output structure.", false],
    ],
    "Instruction tuning teaches a model how to answer requests, solve prompted tasks, and handle safety-sensitive situations. Raw shuffled text does not demonstrate the input-output behavior expected from an assistant.",
  ),
  makeQuestion(
    "cme295-lect4-q152",
    "easy",
    "Which statements correctly compare pretraining scale with SFT scale?",
    [
      [
        "Pretraining commonly uses hundreds of billions to trillions of tokens.",
        true,
      ],
      ["SFT can use thousands to millions of high-quality examples.", true],
      ["SFT uses more examples than pretraining uses tokens.", false],
      [
        "Pretraining mainly uses assistant-dialog examples with hand-written answers.",
        false,
      ],
    ],
    "Pretraining is broad and enormous, while SFT is narrower and more curated. The smaller SFT scale works because it starts from a base model that already learned general language and code patterns.",
  ),
  makeQuestion(
    "cme295-lect4-q153",
    "medium",
    "A team instruction-tunes mostly on textbook-style answers, then deploys the model for terse customer-support chats full of abbreviations. Which risk is most directly illustrated?",
    [
      [
        "Prompt-distribution mismatch between SFT data and deployment use.",
        true,
      ],
      ["The Chinchilla token-per-parameter rule being exactly 1:1.", false],
      ["FlashAttention becoming approximate on short prompts.", false],
      ["ZeRO sharding replacing the loss function.", false],
    ],
    "The problem is that SFT behavior generalizes best when the training prompt distribution resembles the prompts the model will face. Scaling ratios, attention kernels, and optimizer-state sharding do not explain this particular deployment mismatch.",
  ),
  makeQuestion(
    "cme295-lect4-q154",
    "medium",
    "A model is SFT-trained on one prompt-response pair and later receives the exact same prompt with nonzero sampling temperature. Which statements are reasonable?",
    [
      [
        "The response may have a similar style without reproducing the exact wording.",
        true,
      ],
      [
        "Higher temperature can make lower-probability continuations more likely.",
        true,
      ],
      ["The model will output the training response word for word.", false],
      [
        "Temperature changes the stored SFT dataset before inference begins.",
        false,
      ],
    ],
    "SFT changes the model distribution, but inference sampling still draws from a probability distribution. Nonzero temperature can create variation, while exact memorized reproduction is a risk to analyze rather than a guarantee.",
  ),
  makeQuestion(
    "cme295-lect4-q155",
    "easy",
    "Which benchmark-to-capability matches are appropriate?",
    [
      ["MMLU: broad multitask knowledge and reasoning.", true],
      ["GSM8K: grade-school math word problems.", true],
      ["HumanEval: code generation tasks.", true],
      ["HBM bandwidth: a direct score for factual answer quality.", false],
    ],
    "MMLU, GSM8K, and HumanEval are evaluation benchmarks for different model capabilities. HBM bandwidth is a hardware characteristic; it can affect speed, but it is not a direct factuality or reasoning benchmark.",
  ),
  makeQuestion(
    "cme295-lect4-q156",
    "hard",
    "Why can training on the benchmark task confound model comparison even if the exact test examples are not included?",
    [
      [
        "A model exposed to the task format may learn strategies specific to that evaluation.",
        true,
      ],
      [
        "Two models are harder to compare if one saw auxiliary data from the benchmark domain and the other did not.",
        true,
      ],
      [
        "A sudden benchmark jump can reflect task exposure rather than a general capability jump.",
        true,
      ],
      [
        "Task exposure matters less than the inference hardware kernel used for the benchmark.",
        false,
      ],
    ],
    "Benchmark validity depends on more than avoiding exact test-set leakage. Training on the same task distribution can change what a score means, especially when comparing models with different data mixtures.",
  ),
  makeQuestion(
    "cme295-lect4-q157",
    "medium",
    "Which statements correctly describe arena-style pairwise preference evaluation?",
    [
      ["Users compare two model responses to the same prompt.", true],
      ["The system aggregates pairwise preferences into a ranking.", true],
      [
        "The resulting score can reflect user taste as well as model quality.",
        true,
      ],
      ["The preferred answer is factually correct by definition.", false],
    ],
    "Arena-style evaluation puts a number on user preference, which is useful but not the same as ground-truth factuality. A preferred answer can be more confident, stylish, or agreeable while still being wrong or unsafe.",
  ),
  makeQuestion(
    "cme295-lect4-q158",
    "medium",
    "Which limitations can affect user-preference leaderboards?",
    [
      ["Cold-start or unequal exposure for newly added models.", true],
      ["Users may not know which response is factual.", true],
      [
        "Preferences may penalize safety refusals that are intended behavior.",
        true,
      ],
      [
        "The voting process can be vulnerable to manipulation or model-identification tricks.",
        true,
      ],
    ],
    "Preference leaderboards are useful but brittle: exposure, manipulation, factuality blind spots, population mismatch, and safety preferences can all affect results. They should be interpreted alongside task benchmarks and expert evaluation, not as a complete measure of model quality.",
  ),
  makeQuestion(
    "cme295-lect4-q159",
    "easy",
    "Which statements correctly describe alignment stages in the LLM lifecycle?",
    [
      [
        "Fine-tuning can make a pretrained model better suited to specific tasks.",
        true,
      ],
      ["Preference tuning can further shape behavior after SFT.", true],
      [
        "Instruction tuning is one supervised fine-tuning route toward alignment.",
        true,
      ],
      [
        "Alignment removes harmful behavior through a closed-form mathematical formula.",
        false,
      ],
    ],
    "Alignment refers to post-pretraining work that makes the model behave more like the intended system, including fine-tuning and preference-based methods. It improves behavior but does not provide a proof that all failures are gone.",
  ),
  makeQuestion(
    "cme295-lect4-q160",
    "medium",
    "What is mid-training in the model lifecycle described here?",
    [
      [
        "A stage after broad pretraining that continues the pretraining objective on data closer to target tasks.",
        true,
      ],
      [
        "A stage before broad pretraining that uses pairwise preference votes instead of token prediction.",
        false,
      ],
      [
        "A replacement for tokenization that runs inside FlashAttention.",
        false,
      ],
      ["A preference-voting website for ranking deployed chatbots.", false],
    ],
    "Mid-training keeps the language-modeling style objective but shifts the data toward domains or tasks the builders care about. It is not a benchmark leaderboard, tokenizer mechanism, or attention kernel.",
  ),
  makeQuestion(
    "cme295-lect4-q161",
    "medium",
    "Which statements describe high-quality SFT data curation?",
    [
      [
        "Human-written examples can be used when expert judgment is needed.",
        true,
      ],
      [
        "Synthetic examples can supplement human data when quality is controlled.",
        true,
      ],
      ["The prompt distribution should resemble the intended use cases.", true],
      [
        "Reusing curated datasets can amortize the cost of high-quality human or synthetic data collection.",
        true,
      ],
    ],
    "SFT data quality is about useful demonstrations, coverage, and fit to the target prompt distribution. Curated datasets are expensive to build, so reuse can be valuable as long as the examples still cover the target behaviors and prompt distribution.",
  ),
  makeQuestion(
    "cme295-lect4-q162",
    "hard",
    "A product team must choose a model for coding help, factual tutoring, and safe customer support. Which evaluation signals are relevant?",
    [
      ["Code benchmarks such as HumanEval-style tasks.", true],
      [
        "Factual and reasoning benchmarks, interpreted with data-mixture caveats.",
        true,
      ],
      ["Expert review of factuality and safety behavior.", true],
      [
        "User-preference tests that capture perceived helpfulness and style.",
        true,
      ],
    ],
    "No single score captures all the product needs in this scenario. A sensible evaluation combines task benchmarks, expert factuality and safety checks, and user preference measurements while accounting for how the models were trained.",
  ),
  makeQuestion(
    "cme295-lect4-q163",
    "medium",
    "A model refuses to answer a harmful request, and many casual users downvote it because it did not comply. Which statements are correct?",
    [
      [
        "Preference votes can conflict with a product's intended safety policy.",
        true,
      ],
      [
        "A refusal can be a desired behavior even if some users dislike it.",
        true,
      ],
      ["The downvotes prove the model is less aligned in every sense.", false],
      ["The downvotes prove that SFT cannot teach assistant behavior.", false],
    ],
    "User preference is not the same as the product's complete objective. Safety, factuality, and policy adherence may intentionally trade off against immediate user satisfaction in some prompts.",
  ),
  makeQuestion(
    "cme295-lect4-q164",
    "easy",
    "A raw pretrained model answers `Can I put my teddy bear in the washer?` with material facts about teddy bears. An instruction-tuned model says to check the label and prefer hand washing. What changed most directly?",
    [
      [
        "The model behavior was tuned toward answering the user's instruction helpfully.",
        true,
      ],
      [
        "The attention formula was replaced by a washing-machine simulator.",
        false,
      ],
      [
        "The model became smaller because all optimizer state was deleted at inference time.",
        false,
      ],
      [
        "The model learned the answer through FlashAttention approximating softmax.",
        false,
      ],
    ],
    "Instruction tuning teaches the model how to respond to user requests, not only how to continue plausible text. The teddy-bear example illustrates behavior tuning, not a change to attention math or GPU memory management.",
  ),
  makeQuestion(
    "cme295-lect4-q165",
    "medium",
    "Which statements correctly interpret the LoRA update \\(W = W_0 + BA\\)?",
    [
      ["\\(W_0\\) is the frozen pretrained weight matrix.", true],
      ["\\(B\\) and \\(A\\) are trainable low-rank adapter factors.", true],
      [
        "The product \\(BA\\) acts as the learned update to the base matrix.",
        true,
      ],
      [
        "\\(BA\\) is the beam-search table \\(B_{\\mathrm{beam}}A_{\\mathrm{answer}}\\) over final generated words.",
        false,
      ],
    ],
    "LoRA freezes the base weights and learns a low-rank update through smaller matrices. The formula describes a parameter update inside the model, not a decoding-time list of completed tokens.",
  ),
  makeQuestion(
    "cme295-lect4-q166",
    "hard",
    "A full matrix has shape \\(4096\\times4096\\). A LoRA update uses rank \\(r=8\\), with \\(B\\in\\mathbb{R}^{4096\\times8}\\) and \\(A\\in\\mathbb{R}^{8\\times4096}\\). How many trainable adapter parameters are in \\(B\\) and \\(A\\) together?",
    [
      ["65,536 parameters.", true],
      ["16,777,216 parameters.", false],
      ["32,768 parameters.", false],
      ["8 parameters.", false],
    ],
    "The adapter count is \\(4096\\times8 + 8\\times4096 = 65{,}536\\). The full matrix has \\(4096^2=16{,}777{,}216\\) entries, which shows why low-rank adaptation can be much cheaper.",
  ),
  makeQuestion(
    "cme295-lect4-q167",
    "hard",
    "Using the same \\(4096\\times4096\\) matrix and rank \\(r=8\\) LoRA update, approximately what fraction of full-matrix parameters are trainable in the adapter?",
    [
      [
        "About 0.39%, because \\(65{,}536 / 16{,}777{,}216 \\approx 0.0039\\).",
        true,
      ],
      [
        "About 39%, because \\(8/2048 \\approx 0.0039\\) is treated as 39%.",
        false,
      ],
      ["About 8%, because \\(r=8\\) directly gives the percentage.", false],
      [
        "About 100%, because LoRA directly updates the entries of \\(W_0\\) during adapter training.",
        false,
      ],
    ],
    "The adapter has 65,536 trainable parameters compared with 16,777,216 entries in the full matrix, so the trainable fraction is under one percent. LoRA changes the effective matrix through a low-rank update while leaving \\(W_0\\) frozen during adapter training.",
  ),
  makeQuestion(
    "cme295-lect4-q168",
    "medium",
    "Why is swapping LoRA adapters useful for task specialization?",
    [
      [
        "Different adapter matrices can encode different task-specific updates.",
        true,
      ],
      [
        "The same frozen base model can be paired with a spam adapter, sentiment adapter, or translation adapter.",
        true,
      ],
      [
        "Adapters are smaller to store and move than a full copy of the base model.",
        true,
      ],
      [
        "Adapter swapping can support several tasks without retraining the full base weights each time.",
        true,
      ],
    ],
    "LoRA separates the large reusable base model from small task-specific updates. This makes it practical to train or store specialized behavior without duplicating every pretrained parameter for every task.",
  ),
  makeQuestion(
    "cme295-lect4-q169",
    "medium",
    "Which statements describe where LoRA adapters may be applied in transformer models?",
    [
      [
        "The original LoRA paper emphasized attention projection matrices.",
        true,
      ],
      [
        "Modern guidance often applies LoRA to both attention and feed-forward blocks, with feed-forward blocks especially important.",
        true,
      ],
      [
        "LoRA is applied to model weight matrices rather than to raw token strings.",
        true,
      ],
      [
        "LoRA works by replacing every transformer layer with a benchmark evaluator.",
        false,
      ],
    ],
    "LoRA is applied to model weight matrices, not to raw strings or benchmark code. Later empirical guidance expanded beyond the original attention-only emphasis and highlights feed-forward blocks as important adaptation locations.",
  ),
  makeQuestion(
    "cme295-lect4-q170",
    "hard",
    "Which empirical training-dynamics differences are associated with LoRA fine-tuning?",
    [
      ["LoRA often uses a higher learning rate than full fine-tuning.", true],
      [
        "LoRA can perform worse with very large batch sizes compared with full fine-tuning.",
        true,
      ],
      [
        "The low-rank parameterization can change the optimization dynamics.",
        true,
      ],
      ["LoRA removes the need to choose learning rate or batch size.", false],
    ],
    "LoRA is not just full fine-tuning with fewer parameters; the low-rank factors change how optimization behaves. Learning rate, batch size, rank, and adapter placement remain real design choices.",
  ),
  makeQuestion(
    "cme295-lect4-q171",
    "medium",
    "In QLoRA-style fine-tuning, which statements correctly describe how quantized base weights and LoRA adapters reduce VRAM pressure?",
    [
      [
        "The frozen base weights can be stored in a quantized format to save memory.",
        true,
      ],
      [
        "The trainable LoRA adapter path can still use higher-precision computation.",
        true,
      ],
      [
        "QLoRA targets fine-tuning memory pressure rather than changing the high-level task objective.",
        true,
      ],
      [
        "QLoRA keeps the base model available while reducing the memory needed to store it.",
        true,
      ],
    ],
    "QLoRA combines quantized frozen base weights with trainable low-rank adapters. The goal is to make fine-tuning feasible under tighter VRAM budgets while preserving the base model as the foundation for the adapter update.",
  ),
  makeQuestion(
    "cme295-lect4-q172",
    "hard",
    "Which statements correctly describe NormalFloat 4 (NF4) quantization?",
    [
      [
        "It uses 4-bit values designed around normally distributed weights.",
        true,
      ],
      [
        "It places quantization levels by normal-distribution quantiles rather than uniform fixed-width buckets.",
        true,
      ],
      [
        "It is useful because neural weights often concentrate near the center of the distribution.",
        true,
      ],
      [
        "It is a weight quantization method rather than tokenizer compression.",
        true,
      ],
    ],
    "NF4 allocates its limited codes according to a normal-distribution assumption, which uses 4-bit capacity more effectively for typical weight distributions. It is a weight quantization method, not FP64 storage or tokenizer compression.",
  ),
  makeQuestion(
    "cme295-lect4-q173",
    "hard",
    "Which statements describe the double-quantization idea in QLoRA-style methods?",
    [
      ["First, the model weights are quantized.", true],
      [
        "Second, the quantization constants used to map values are themselves quantized.",
        true,
      ],
      [
        "The second quantization can save additional memory beyond single quantization.",
        true,
      ],
      [
        "It is a representation trick, not a second full pretraining run.",
        true,
      ],
    ],
    "Double quantization compresses both the weights and the metadata or constants needed for quantization. It is not a second full pretraining run; it is an additional memory-saving representation trick.",
  ),
  makeQuestion(
    "cme295-lect4-q174",
    "medium",
    "A QLoRA implementation reports about 16x VRAM savings for fine-tuning and an additional 6% savings from double quantization. Which statements correctly interpret those numbers?",
    [
      [
        "A 16x saving would reduce a 160 GB footprint to roughly 10 GB for the part being compared.",
        true,
      ],
      ["The extra 6% is smaller than the main 16x quantization effect.", true],
      ["The savings make smaller-GPU fine-tuning more feasible.", true],
      [
        "The numbers describe memory savings rather than a direct factuality improvement.",
        true,
      ],
    ],
    "The 16x figure is a memory-footprint comparison, not an answer-quality multiplier: for example, the compared part of a 160 GB footprint would become about 10 GB. The additional 6% from double quantization is a smaller incremental saving on top of the main quantization effect, and it can still matter for fitting fine-tuning into a smaller GPU budget while quality remains a separate model, data, training, and evaluation question.",
  ),
  makeQuestion(
    "cme295-lect4-q175",
    "hard",
    "Which statements correctly describe the LoRA rank \\(r\\) tradeoff?",
    [
      [
        "Increasing \\(r\\) increases the number of trainable adapter parameters.",
        true,
      ],
      [
        "A larger \\(r\\) can represent a richer update than a very small \\(r\\).",
        true,
      ],
      [
        "The best rank \\(r\\) is a design choice that can depend on task, model, and resources.",
        true,
      ],
      [
        "Setting \\(r=0\\) would remove the useful trainable update rather than maximize expressiveness.",
        true,
      ],
    ],
    "Rank controls the capacity and cost of the low-rank update. A zero-rank update would add no useful trainable matrix product, while higher ranks trade more parameters for more adaptation capacity.",
  ),
  makeQuestion(
    "cme295-lect4-q176",
    "hard",
    "A full fine-tune would train a \\(2048\\times2048\\) matrix. A LoRA adapter with \\(r=4\\) trains \\(2048\\times4 + 4\\times2048\\) parameters. Which statements are correct?",
    [
      ["The full matrix has 4,194,304 entries.", true],
      ["The LoRA adapter has 16,384 trainable entries.", true],
      [
        "The adapter trains less than \\(1\\%\\) as many parameters as the full matrix.",
        true,
      ],
      [
        "The adapter is cheaper even though it has two factors, because \\(r=4\\) is small relative to 2048.",
        true,
      ],
    ],
    "The full matrix has \\(2048^2=4{,}194{,}304\\) entries, while the adapter has \\(2\\times2048\\times4=16{,}384\\). Two small factors are still far cheaper than one huge dense matrix in this example.",
  ),
  makeQuestion(
    "cme295-lect4-q177",
    "medium",
    "Which statements distinguish LoRA from mixture-of-experts routing?",
    [
      [
        "LoRA adds trainable low-rank updates to selected weight matrices.",
        true,
      ],
      [
        "MoE routing chooses expert subnetworks for token representations during a forward pass.",
        true,
      ],
      [
        "Both techniques can affect feed-forward layers, but they solve different problems.",
        true,
      ],
      ["LoRA does not perform token-level expert routing.", true],
    ],
    "LoRA is a parameter-efficient fine-tuning method, while MoE routing is a conditional-computation architecture. They can both touch feed-forward parts of a transformer, but LoRA does not perform token-level expert routing.",
  ),
  makeQuestion(
    "cme295-lect4-q178",
    "easy",
    "Which statements correctly compare LoRA with other parameter-efficient tuning methods mentioned alongside it?",
    [
      [
        "Prefix tuning and adapters are alternative parameter-efficient methods.",
        true,
      ],
      [
        "LoRA is widely used because it can train a small number of additional parameters.",
        true,
      ],
      [
        "These methods are motivated by the cost of full supervised fine-tuning.",
        true,
      ],
      [
        "These methods adapt an existing pretrained model rather than rebuilding the pretraining dataset from scratch.",
        true,
      ],
    ],
    "Parameter-efficient tuning methods are alternatives to updating every base-model weight. They are motivated by memory and compute constraints during adaptation, not by a requirement to repeat broad pretraining.",
  ),
  makeQuestion(
    "cme295-lect4-q179",
    "hard",
    "Which statements correctly describe computation and storage in QLoRA-style fine-tuning?",
    [
      [
        "Frozen base weights may be stored quantized to reduce VRAM usage.",
        true,
      ],
      [
        "Trainable adapter matrices can be kept in a higher-precision format such as BF16.",
        true,
      ],
      [
        "Quantized base weights can be dequantized or used through kernels that support the needed computation path.",
        true,
      ],
      [
        "The frozen base weights remain necessary; the adapter modifies rather than replaces the full model.",
        true,
      ],
    ],
    "QLoRA keeps the base model, but stores it compactly and trains small adapters. The adapter alone is not the full model; it modifies the behavior of the frozen base through a low-rank update path.",
  ),
  makeQuestion(
    "cme295-lect4-q180",
    "hard",
    "A team wants to build a useful assistant from scratch-like resources and then adapt it cheaply to several tasks. Which lifecycle statements are correct?",
    [
      [
        "Pretraining teaches broad next-token prediction from large text and code mixtures.",
        true,
      ],
      [
        "Training optimizations make the large run fit and finish by managing memory, precision, and parallelism.",
        true,
      ],
      [
        "SFT and later preference tuning shape the model toward desired assistant behavior.",
        true,
      ],
      [
        "LoRA or QLoRA can cheaply adapt the pretrained model without full-weight fine-tuning for every task.",
        true,
      ],
    ],
    "The LLM lifecycle is staged: broad pretraining builds a base model, systems optimizations make the run practical, and post-training changes behavior. Parameter-efficient methods then make task adaptation cheaper than full fine-tuning for every use case.",
  ),
];
