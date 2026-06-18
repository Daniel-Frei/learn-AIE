import { Question } from "../../quiz";

export const stanfordCME295Lecture2Questions: Question[] = [
  // ============================================================
  // Lecture 2 – Transformer-Based Models & Tricks
  // Coverage: position embeddings, RoPE, attention variants,
  // normalization, sparse attention, model families, BERT
  // ============================================================

  // Q1-Q20: MATH-FOCUSED REPLACEMENTS
  // ============================================================

  {
    id: "cme295-lect2-q01",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "For a sinusoidal positional-encoding pair at frequency \\(\\omega_i\\), define \\(p_m^{(i)}=(\\sin(\\omega_i m), \\cos(\\omega_i m))\\). Which expression equals the dot product \\(p_m^{(i)} \\cdot p_n^{(i)}\\)?",
    options: [
      {
        text: "\\(\\cos(\\omega_i(m-n))\\)",
        isCorrect: true,
      },
      {
        text: "\\(\\sin(\\omega_i(m-n))\\)",
        isCorrect: false,
      },
      {
        text: "\\(\\cos(\\omega_i(m+n))\\)",
        isCorrect: false,
      },
      {
        text: "\\(\\omega_i mn\\)",
        isCorrect: false,
      },
    ],
    explanation:
      "The dot product is \\(\\sin(\\omega_i m)\\sin(\\omega_i n)+\\cos(\\omega_i m)\\cos(\\omega_i n)\\), which equals \\(\\cos(\\omega_i(m-n))\\) by the cosine difference identity. The other expressions either use sine instead of the cosine identity, depend on the sum rather than the offset, or replace the trigonometric similarity with an unrelated product.",
  },

  {
    id: "cme295-lect2-q02",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "The sinusoidal construction can write \\(\\omega_i=10000^{-2i/d_{\\text{model}}}\\). If \\(d_{\\text{model}}=4\\) and \\(i=1\\), what is \\(\\omega_i\\)?",
    options: [
      {
        text: "\\(10000^{-1/2}=\\frac{1}{100}\\)",
        isCorrect: true,
      },
      {
        text: "\\(10000^{-1}=\\frac{1}{10000}\\)",
        isCorrect: false,
      },
      {
        text: "\\(10000^{-1/4}=\\frac{1}{10}\\)",
        isCorrect: false,
      },
      {
        text: "\\(10000^{1/2}=100\\)",
        isCorrect: false,
      },
    ],
    explanation:
      "Substituting \\(d_{\\text{model}}=4\\) and \\(i=1\\) gives the exponent \\(-2/4=-1/2\\), so \\(10000^{-1/2}=1/\\sqrt{10000}=1/100\\). The other choices come from using the wrong exponent, dropping the minus sign, or confusing the fourth root with the square root.",
  },

  {
    id: "cme295-lect2-q03",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "For one sinusoidal 2D pair, why does comparing a position with itself produce the largest possible pairwise dot-product contribution?",
    options: [
      {
        text: "The offset is zero, so the contribution becomes \\(\\cos(0)=1\\).",
        isCorrect: true,
      },
      {
        text: "The offset is zero, so the contribution becomes \\(\\sin(0)=1\\).",
        isCorrect: false,
      },
      {
        text: "The frequency \\(\\omega_i\\) becomes zero whenever the two positions match.",
        isCorrect: false,
      },
      {
        text: "The attention softmax forces every diagonal query-key score to equal one.",
        isCorrect: false,
      },
    ],
    explanation:
      "The relevant identity reduces the pair contribution to \\(\\cos(\\omega_i(m-n))\\), and when \\(m=n\\), the argument is zero. \\(\\sin(0)\\) is zero rather than one, \\(\\omega_i\\) is fixed by dimension rather than by equality of positions, and the softmax does not impose a fixed diagonal value.",
  },

  {
    id: "cme295-lect2-q04",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "In scaled dot-product attention, a logit uses \\(q \\cdot k / \\sqrt{d_k}\\) before the softmax. If \\(q \\cdot k=16\\), \\(d_k=64\\), and no positional bias is added, what is the scaled logit?",
    options: [
      {
        text: "\\(2\\)",
        isCorrect: true,
      },
      {
        text: "\\(16\\)",
        isCorrect: false,
      },
      {
        text: "\\(128\\)",
        isCorrect: false,
      },
      {
        text: "\\(\\frac{1}{2}\\)",
        isCorrect: false,
      },
    ],
    explanation:
      "The scaling denominator is \\(\\sqrt{64}=8\\), so the logit is \\(16/8=2\\). The other values come from forgetting the square root, multiplying by the scale, or dividing by the wrong quantity.",
  },

  {
    id: "cme295-lect2-q05",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "A T5-style relative position bias can be written as \\(\\ell_{mn}=q_m k_n^T/\\sqrt{d_k}+b_{h,\\operatorname{bucket}(n-m)}\\) before the softmax. Which statements are correct?",
    options: [
      {
        text: "The bias changes attention logits before normalization, so the softmax still produces weights that sum to one.",
        isCorrect: true,
      },
      {
        text: "Two token pairs in the same distance bucket and attention head receive the same learned bias term.",
        isCorrect: true,
      },
      {
        text: "The bias is multiplied into the value vectors after the softmax has already computed attention weights.",
        isCorrect: false,
      },
      {
        text: "The bias is a learned absolute position vector that is added once to the input token embedding.",
        isCorrect: false,
      },
    ],
    explanation:
      "The T5 mechanism modifies the score that enters the softmax, so the resulting attention weights are still normalized probabilities. It is a relative, bucketized, head-specific logit bias, not a post-softmax value multiplier or an absolute input embedding.",
  },

  {
    id: "cme295-lect2-q06",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Suppose an Attention with Linear Biases (ALiBi) head uses bias \\(-a|m-n|\\) with \\(a=0.25\\). Which calculations are correct?",
    options: [
      {
        text: "A token pair at distance \\(8\\) receives bias \\(-2.0\\).",
        isCorrect: true,
      },
      {
        text: "A token pair at distance \\(2\\) receives bias \\(-0.5\\), so the distance-8 pair is penalized \\(1.5\\) logits more.",
        isCorrect: true,
      },
      {
        text: "A token pair at distance \\(8\\) receives bias \\(+2.0\\), increasing its logit relative to nearby tokens.",
        isCorrect: false,
      },
      {
        text: "The penalty is the same for distances \\(2\\) and \\(8\\) because ALiBi ignores the magnitude of the offset.",
        isCorrect: false,
      },
    ],
    explanation:
      "With the stated slope, distance \\(8\\) gives \\(-0.25\\times 8=-2.0\\), and distance \\(2\\) gives \\(-0.5\\). The sign matters because ALiBi is used as a distance penalty, and its magnitude changes linearly with relative distance.",
  },

  {
    id: "cme295-lect2-q07",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "Rotary Position Embeddings (RoPE) use 2D rotation blocks. If \\(R(\\pi/2)=\\begin{bmatrix}0&-1\\\\1&0\\end{bmatrix}\\) and \\(x=(2,1)\\), which statements are correct?",
    options: [
      {
        text: "\\(R(\\pi/2)x=(-1,2)\\).",
        isCorrect: true,
      },
      {
        text: "The vector norm remains \\(\\sqrt{5}\\) after rotation.",
        isCorrect: true,
      },
      {
        text: "\\(R(\\pi/2)x=(1,-2)\\).",
        isCorrect: false,
      },
      {
        text: "The vector norm becomes \\(3\\) because rotation adds the coordinates.",
        isCorrect: false,
      },
    ],
    explanation:
      "Multiplying by the rotation matrix gives \\((0\\cdot2-1\\cdot1, 1\\cdot2+0\\cdot1)=(-1,2)\\). Rotations preserve Euclidean norm, so the length stays \\(\\sqrt{2^2+1^2}=\\sqrt{5}\\); the other vector has the wrong rotation direction, and summing coordinates is not how norms transform.",
  },

  {
    id: "cme295-lect2-q08",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Layer normalization over \\(x=(1,3)\\) uses \\(\\mu=\\frac{1}{d}\\sum_j x_j\\) and variance \\(\\frac{1}{d}\\sum_j(x_j-\\mu)^2\\). With \\(\\epsilon=0\\), which intermediate quantities are correct?",
    options: [
      {
        text: "The mean is \\(2\\).",
        isCorrect: true,
      },
      {
        text: "The variance is \\(1\\).",
        isCorrect: true,
      },
      {
        text: "The centered vector is \\((1,-1)\\).",
        isCorrect: false,
      },
      {
        text: "With \\(\\gamma=(2,2)\\) and \\(\\beta=(0,0)\\), the final output is \\((-1,1)\\).",
        isCorrect: false,
      },
    ],
    explanation:
      "The average of \\(1\\) and \\(3\\) is \\(2\\), and the squared deviations are \\(1\\) and \\(1\\), so the variance is \\(1\\). The centered vector is \\((-1,1)\\), and multiplying the normalized vector by \\(\\gamma=(2,2)\\) would give \\((-2,2)\\), not \\((-1,1)\\).",
  },

  {
    id: "cme295-lect2-q09",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "RMSNorm over \\(x=(3,4)\\) with \\(\\gamma=(1,1)\\) and \\(\\epsilon=0\\) divides by \\(\\sqrt{\\frac{1}{d}\\sum_j x_j^2}\\). Which statements are correct?",
    options: [
      {
        text: "The root mean square is \\(\\frac{5}{\\sqrt{2}}\\).",
        isCorrect: true,
      },
      {
        text: "The normalized output is \\((\\frac{3\\sqrt{2}}{5}, \\frac{4\\sqrt{2}}{5})\\).",
        isCorrect: true,
      },
      {
        text: "The squared norm of the normalized output is \\(2\\).",
        isCorrect: true,
      },
      {
        text: "RMSNorm first subtracts the mean, producing \\((-0.5,0.5)\\) before scaling.",
        isCorrect: false,
      },
    ],
    explanation:
      "The RMS is \\(\\sqrt{(9+16)/2}=5/\\sqrt{2}\\), so each coordinate is multiplied by \\(\\sqrt{2}/5\\). RMSNorm does not do LayerNorm's mean subtraction step, and after dividing by the RMS the average squared component is one, giving squared norm \\(d=2\\).",
  },

  {
    id: "cme295-lect2-q10",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "Assume full attention over length \\(n\\) uses \\(n^2\\) query-key score computations, while sliding-window attention uses roughly \\(n w\\) when each token attends to \\(w\\) keys. For \\(n=4096\\) and \\(w=128\\), which statements are correct?",
    options: [
      {
        text: "Full attention uses \\(4096^2=16{,}777{,}216\\) score computations.",
        isCorrect: true,
      },
      {
        text: "Sliding-window attention uses about \\(4096\\times128=524{,}288\\) score computations.",
        isCorrect: true,
      },
      {
        text: "The full-attention count is \\(32\\) times the sliding-window count under these assumptions.",
        isCorrect: true,
      },
      {
        text: "Sliding-window attention uses \\(32\\) times more score computations than full attention in this setup.",
        isCorrect: false,
      },
    ],
    explanation:
      "\\(4096^2\\) is about 16.8 million, while \\(4096\\times128\\) is about 0.52 million. Dividing the two counts gives \\(4096/128=32\\), so the stated windowed pattern is cheaper, not more expensive.",
  },

  // ============================================================
  // Q11-Q20: MATH-FOCUSED REPLACEMENTS CONTINUED
  // ============================================================

  {
    id: "cme295-lect2-q11",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "For an autoregressive key-value cache, approximate stored elements as \\(2nh_{kv}d_{head}\\), counting both keys and values. With \\(n=2048\\), \\(d_{head}=128\\), fp16 storage at 2 bytes per element, and 32 query heads, which statements are correct?",
    options: [
      {
        text: "Multi-head attention with \\(h_{kv}=32\\) uses \\(33{,}554{,}432\\) bytes, or 32 MiB.",
        isCorrect: true,
      },
      {
        text: "Multi-query attention with \\(h_{kv}=1\\) uses 1 MiB under the same assumptions.",
        isCorrect: true,
      },
      {
        text: "Grouped-query attention with \\(h_{kv}=8\\) uses 8 MiB under the same assumptions.",
        isCorrect: true,
      },
      {
        text: "Multi-query attention uses the same cache size as full multi-head attention because the query head count remains 32.",
        isCorrect: false,
      },
    ],
    explanation:
      "The cache size scales with the number of key/value heads, not merely with the number of query heads. Plugging in \\(h_{kv}=32,8,1\\) gives 32 MiB, 8 MiB, and 1 MiB respectively, which is why MQA and GQA reduce decoding memory.",
  },

  {
    id: "cme295-lect2-q12",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "In BERT-style Masked Language Modeling (MLM), 15% of input tokens are selected for prediction; of those selected tokens, 80% become [MASK], 10% become random tokens, and 10% are left unchanged. For 1000 input tokens, which statements are correct?",
    options: [
      {
        text: "150 tokens are selected for the MLM prediction objective.",
        isCorrect: true,
      },
      {
        text: "120 tokens are replaced by [MASK].",
        isCorrect: true,
      },
      {
        text: "30 selected tokens are either random replacements or left unchanged.",
        isCorrect: true,
      },
      {
        text: "All 1000 tokens directly contribute to the MLM loss because the model is bidirectional.",
        isCorrect: false,
      },
    ],
    explanation:
      "Fifteen percent of 1000 is 150; 80% of 150 is 120, and the two 10% branches contribute 15 tokens each. Bidirectionality gives contextual information, but the MLM prediction loss is applied to the selected subset, not every token.",
  },

  {
    id: "cme295-lect2-q13",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "For a single RoPE 2D block, let \\(q=k=(1,0)\\), \\(\\theta=\\pi/3\\), and positions \\(m=2\\), \\(n=5\\). Which calculations are consistent with the rotated query-key dot product?",
    options: [
      {
        text: "The relative angle is \\((n-m)\\theta=\\pi\\).",
        isCorrect: true,
      },
      {
        text: "The resulting dot product is \\(\\cos(\\pi)=-1\\).",
        isCorrect: true,
      },
      {
        text: "Using \\((m-n)\\theta=-\\pi\\) gives the same cosine value.",
        isCorrect: true,
      },
      {
        text: "The score depends on the position offset here rather than on the absolute indices alone.",
        isCorrect: true,
      },
    ],
    explanation:
      "RoPE's rotation algebra makes the dot product depend on an angle difference, so the relevant offset is \\(5-2=3\\) blocks of \\(\\pi/3\\). Because cosine is even, \\(\\cos(\\pi)\\) and \\(\\cos(-\\pi)\\) agree, illustrating why relative distance enters the score.",
  },

  {
    id: "cme295-lect2-q14",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "The BERT model-size table gives BERT-Tiny \\((L=2,H=128,A=2)\\), BERT-Small \\((L=4,H=512,A=8)\\), BERT-Base \\((L=12,H=768,A=12)\\), and BERT-Large \\((L=24,H=1024,A=16)\\). Which calculations are correct?",
    options: [
      {
        text: "BERT-Base has per-head hidden size \\(768/12=64\\).",
        isCorrect: true,
      },
      {
        text: "BERT-Large has per-head hidden size \\(1024/16=64\\).",
        isCorrect: true,
      },
      {
        text: "BERT-Small has per-head hidden size \\(512/8=64\\).",
        isCorrect: true,
      },
      {
        text: "BERT-Base has \\(12/2=6\\) times as many layers as BERT-Tiny.",
        isCorrect: true,
      },
    ],
    explanation:
      "The table's \\(H/A\\) ratio is 64 for the listed Small, Base, and Large configurations, matching the common per-head dimension. The layer comparison is also direct: 12 layers in Base divided by 2 layers in Tiny gives a factor of 6.",
  },

  {
    id: "cme295-lect2-q15",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "Next Sentence Prediction (NSP) samples sentence pairs so that 50% are consecutive and 50% are not. In a balanced batch of 512 sentence pairs, which statements are correct?",
    options: [
      {
        text: "The expected number of consecutive pairs is 256.",
        isCorrect: true,
      },
      {
        text: "The expected number of non-consecutive pairs is 256.",
        isCorrect: true,
      },
      {
        text: "The positive-class prior is \\(0.5\\).",
        isCorrect: true,
      },
      {
        text: "A classifier that always predicts consecutive would have 50% accuracy on this balanced construction.",
        isCorrect: true,
      },
    ],
    explanation:
      "Half of 512 is 256, so the intended construction has equal positive and negative examples in expectation. Because the labels are balanced, a constant classifier that always chooses one class gets only half correct, which is why the task is a binary classification proxy rather than a generation objective.",
  },

  {
    id: "cme295-lect2-q16",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "In distillation, a student distribution \\(s\\) can be trained against a teacher distribution \\(t\\). If \\(t=(0,1,0)\\) and \\(s=(0.2,0.6,0.2)\\), which statements are correct?",
    options: [
      {
        text: "The hard-label cross-entropy is \\(-\\log(0.6)\\).",
        isCorrect: true,
      },
      {
        text: "For this one-hot teacher, \\(D_{KL}(t\\|s)=-\\log(0.6)\\) because the teacher entropy is zero.",
        isCorrect: true,
      },
      {
        text: "For a soft teacher such as \\(t=(0.7,0.2,0.1)\\), the KL term is \\(\\sum_i t_i\\log(t_i/s_i)\\).",
        isCorrect: true,
      },
      {
        text: "Matching the full teacher distribution can transfer more information than matching only the argmax class.",
        isCorrect: true,
      },
    ],
    explanation:
      "With a one-hot teacher on the second class, cross-entropy reduces to the negative log probability the student assigns to that class. Soft distillation generalizes this by comparing full probability distributions with KL divergence, preserving information about non-argmax classes that hard labels discard.",
  },

  {
    id: "cme295-lect2-q17",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "A sinusoidal positional embedding has \\(d=6\\), so it contains three sine/cosine frequency pairs. If two positions are identical, which statements are correct?",
    options: [
      {
        text: "Each 2D pair contributes \\(\\cos(0)=1\\) to the dot product.",
        isCorrect: true,
      },
      {
        text: "The unnormalized dot product between the two positional embeddings is \\(3\\).",
        isCorrect: true,
      },
      {
        text: "If each embedding has norm \\(\\sqrt{3}\\), their cosine similarity is \\(3/(\\sqrt{3}\\sqrt{3})=1\\).",
        isCorrect: true,
      },
      {
        text: "For non-identical positions, each pair contributes a term of the form \\(\\cos(\\omega_i(m-n))\\).",
        isCorrect: true,
      },
    ],
    explanation:
      "Each sine/cosine pair has unit norm and contributes one when the positions match, so three pairs give a dot product of three. Normalizing by the two norms gives cosine similarity one, while nonzero offsets introduce the frequency-dependent cosine terms that make relative distance matter.",
  },

  {
    id: "cme295-lect2-q18",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "A decoder has 32 query heads. Full multi-head attention uses 32 key/value heads, grouped-query attention (GQA) uses 8 key/value heads, and multi-query attention (MQA) uses 1 key/value head. Which cache-ratio statements are correct?",
    options: [
      {
        text: "GQA uses \\(8/32=1/4\\) as many key/value cache entries as full multi-head attention.",
        isCorrect: true,
      },
      {
        text: "MQA uses \\(1/32\\) as many key/value cache entries as full multi-head attention.",
        isCorrect: true,
      },
      {
        text: "GQA with 8 key/value heads uses 8 times as many key/value cache entries as MQA.",
        isCorrect: true,
      },
      {
        text: "The query head count can remain 32 even when key/value heads are shared.",
        isCorrect: true,
      },
    ],
    explanation:
      "The memory reduction tracks how many key/value heads are stored, not how many query projections are used. GQA sits between full MHA and MQA: it is cheaper than storing 32 K/V heads, but it stores more K/V states than the single shared set used by MQA.",
  },

  {
    id: "cme295-lect2-q19",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "A sliding-window attention layer lets each token directly attend up to 2 positions left and 2 positions right, plus itself. Ignoring sequence boundaries, which receptive-field calculations are correct after stacking local-attention layers?",
    options: [
      {
        text: "After one layer, a token has direct access to positions at offsets from \\(-2\\) to \\(+2\\).",
        isCorrect: true,
      },
      {
        text: "After two layers, information can travel as far as offset \\(4\\) through intermediate tokens.",
        isCorrect: true,
      },
      {
        text: "After three layers, the effective offset reach is up to \\(6\\) on each side.",
        isCorrect: true,
      },
      {
        text: "After three layers, the effective receptive field can include \\(13\\) positions counting the center token.",
        isCorrect: true,
      },
    ],
    explanation:
      "Local attention does not give global access in one layer, but stacked layers let information hop through neighboring tokens. With radius 2, each layer can expand reach by roughly 2 positions per side, so three layers reach offsets \\(-6\\) through \\(+6\\), or 13 positions including the center.",
  },

  {
    id: "cme295-lect2-q20",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "For an activation vector \\(x\\in\\mathbb{R}^d\\), compare LayerNorm and RMSNorm. Which formula statements are correct?",
    options: [
      {
        text: "LayerNorm uses \\(\\mu=\\frac{1}{d}\\sum_{j=1}^d x_j\\) as the per-token feature mean.",
        isCorrect: true,
      },
      {
        text: "LayerNorm divides by \\(\\sqrt{\\frac{1}{d}\\sum_{j=1}^d(x_j-\\mu)^2+\\epsilon}\\).",
        isCorrect: true,
      },
      {
        text: "RMSNorm divides by \\(\\sqrt{\\frac{1}{d}\\sum_{j=1}^d x_j^2+\\epsilon}\\) without subtracting the mean.",
        isCorrect: true,
      },
      {
        text: "Both methods can include a learned multiplicative scale \\(\\gamma\\).",
        isCorrect: true,
      },
    ],
    explanation:
      "LayerNorm centers each token's feature vector and rescales it by its feature standard deviation, then typically applies learned scale and shift. RMSNorm keeps the rescaling idea but uses the root mean square directly, which omits mean subtraction and commonly omits the learned bias.",
  },

  {
    id: "cme295-lect2-q21",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe sparse attention mechanisms such as Longformer?",
    options: [
      {
        text: "They reduce quadratic complexity by restricting attention patterns.",
        isCorrect: true,
      },
      {
        text: "They combine local and global attention mechanisms.",
        isCorrect: true,
      },
      {
        text: "They enable longer context lengths at lower cost.",
        isCorrect: true,
      },
      {
        text: "They trade off full connectivity for efficiency.",
        isCorrect: true,
      },
    ],
    explanation:
      "Sparse attention limits interactions to improve scalability while preserving performance. Core ideas: they reduce quadratic complexity by restricting attention patterns; they combine local and global attention mechanisms; they enable longer context lengths at lower cost; they trade off full connectivity for efficiency.",
  },

  {
    id: "cme295-lect2-q22",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statements correctly relate sliding window attention to convolutional receptive fields?",
    options: [
      {
        text: "Local attention defines which tokens influence a given token.",
        isCorrect: true,
      },
      {
        text: "Stacking layers expands the effective receptive field.",
        isCorrect: true,
      },
      {
        text: "Information propagates across layers beyond the local window.",
        isCorrect: true,
      },
      {
        text: "The analogy mirrors how CNNs aggregate spatial context.",
        isCorrect: true,
      },
    ],
    explanation:
      "Sliding window attention behaves similarly to receptive fields in convolutional networks. Core ideas: Local attention defines which tokens influence a given token; Stacking layers expands the effective receptive field; Information propagates across layers beyond the local window; The analogy mirrors how CNNs aggregate spatial context.",
  },

  {
    id: "cme295-lect2-q23",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe motivations for sharing key/value projections across heads?",
    options: [
      { text: "It reduces memory usage of the KV cache.", isCorrect: true },
      {
        text: "Keys and values are reused during autoregressive decoding.",
        isCorrect: true,
      },
      {
        text: "Queries benefit from retaining head-specific diversity.",
        isCorrect: true,
      },
      { text: "Sharing improves inference efficiency.", isCorrect: true },
    ],
    explanation:
      "Sharing K/V projections lowers memory and compute costs while keeping expressive queries. Core ideas: it reduces memory usage of the KV cache; Keys and values are reused during autoregressive decoding; Queries benefit from retaining head-specific diversity; Sharing improves inference efficiency.",
  },

  {
    id: "cme295-lect2-q24",
    chapter: 2,
    difficulty: "easy",
    prompt: "Which statements correctly describe Multi-Query Attention (MQA)?",
    options: [
      {
        text: "All heads share the same key and value projections.",
        isCorrect: true,
      },
      {
        text: "Each head still has its own query projection.",
        isCorrect: true,
      },
      {
        text: "It minimizes memory usage for key/value caches.",
        isCorrect: true,
      },
      { text: "It is an extreme case of projection sharing.", isCorrect: true },
    ],
    explanation:
      "MQA aggressively shares K/V projections to reduce memory while keeping query diversity. Core ideas: All heads share the same key and value projections; Each head still has its own query projection; it minimizes memory usage for key/value caches; it is an extreme case of projection sharing.",
  },

  {
    id: "cme295-lect2-q25",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe Grouped-Query Attention (GQA)?",
    options: [
      {
        text: "It groups heads to share key/value projections.",
        isCorrect: true,
      },
      { text: "It interpolates between MHA and MQA.", isCorrect: true },
      { text: "It balances efficiency and expressivity.", isCorrect: true },
      {
        text: "It is widely used in modern decoder-only LLMs.",
        isCorrect: true,
      },
    ],
    explanation:
      "GQA offers a compromise between full multi-head attention and full K/V sharing. Core ideas: it groups heads to share key/value projections; it interpolates between MHA and MQA; it balances efficiency and expressivity; it is widely used in modern decoder-only LLMs.",
  },

  // ============================================================
  // Q26–Q50: EXACTLY 3 TRUE
  // ============================================================

  {
    id: "cme295-lect2-q26",
    chapter: 2,
    difficulty: "medium",
    prompt: "Which statements about decoder-only Transformers are correct?",
    options: [
      { text: "They remove the encoder entirely.", isCorrect: true },
      { text: "They rely on masked self-attention.", isCorrect: true },
      { text: "They are well suited for text generation.", isCorrect: true },
      {
        text: "They require decoder states to cross-attend to a separately encoded source sequence.",
        isCorrect: false,
      },
    ],
    explanation:
      "Decoder-only models use masked self-attention and do not include cross-attention. Core ideas: they remove the encoder entirely; they rely on masked self-attention; they are well suited for text generation. Common misconceptions: they require decoder states to cross-attend to a separately encoded source sequence.",
  },

  {
    id: "cme295-lect2-q27",
    chapter: 2,
    difficulty: "easy",
    prompt: "Which statements about T5-style models are correct?",
    options: [
      { text: "They use an encoder–decoder architecture.", isCorrect: true },
      { text: "They frame all tasks as text-to-text.", isCorrect: true },
      {
        text: "They use span corruption as a pretraining objective.",
        isCorrect: true,
      },
      {
        text: "They rely exclusively on next-token prediction with a causal decoder objective.",
        isCorrect: false,
      },
    ],
    explanation:
      "T5 reformulates tasks into text-to-text using span corruption rather than pure next-token prediction. Core ideas: they use an encoder–decoder architecture; they frame all tasks as text-to-text; they use span corruption as a pretraining objective. Common misconceptions: they rely exclusively on next-token prediction with a causal decoder objective.",
  },

  {
    id: "cme295-lect2-q28",
    chapter: 2,
    difficulty: "medium",
    prompt: "Which statements about BERT’s bidirectionality are correct?",
    options: [
      {
        text: "Each token can attend to tokens on both sides.",
        isCorrect: true,
      },
      {
        text: "Bidirectionality is enabled by unmasked self-attention.",
        isCorrect: true,
      },
      { text: "BERT lacks causal masking.", isCorrect: true },
      {
        text: "Bidirectionality enables left-to-right autoregressive generation without causal masking.",
        isCorrect: false,
      },
    ],
    explanation:
      "BERT is bidirectional but not suitable for autoregressive generation. Core ideas: Each token can attend to tokens on both sides; Bidirectionality is enabled by unmasked self-attention; BERT lacks causal masking. Common misconceptions: Bidirectionality enables left-to-right autoregressive generation without causal masking.",
  },

  {
    id: "cme295-lect2-q29",
    chapter: 2,
    difficulty: "medium",
    prompt: "Which statements about the [CLS] token in BERT are correct?",
    options: [
      {
        text: "It aggregates information from the entire sequence.",
        isCorrect: true,
      },
      { text: "It is used as input to classification heads.", isCorrect: true },
      {
        text: "It participates in self-attention like any other token.",
        isCorrect: true,
      },
      {
        text: "It is ignored during pretraining and added only when a downstream classifier is attached.",
        isCorrect: false,
      },
    ],
    explanation:
      "The CLS token is fully integrated into attention and used for downstream classification. Core ideas: it aggregates information from the entire sequence; it is used as input to classification heads; it participates in self-attention like any other token. Common misconceptions: it is ignored during pretraining and added only when a downstream classifier is attached.",
  },

  {
    id: "cme295-lect2-q30",
    chapter: 2,
    difficulty: "medium",
    prompt: "Which statements about segment embeddings in BERT are correct?",
    options: [
      {
        text: "They distinguish between sentence A and sentence B.",
        isCorrect: true,
      },
      {
        text: "They are added to token and position embeddings.",
        isCorrect: true,
      },
      { text: "They are learned parameters.", isCorrect: true },
      {
        text: "They encode word order within each sentence rather than marking sentence-pair membership.",
        isCorrect: false,
      },
    ],
    explanation:
      "Segment embeddings identify sentence membership, not token order. Core ideas: they distinguish between sentence A and sentence B; they are added to token and position embeddings; they are learned parameters. Common misconceptions: they encode word order within each sentence rather than marking sentence-pair membership.",
  },

  // ============================================================
  // Q31–Q50 (3 TRUE) – continues seamlessly
  // ============================================================

  {
    id: "cme295-lect2-q31",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "Which statements about Masked Language Modeling (MLM) are correct?",
    options: [
      {
        text: "Only a subset of tokens contribute to the MLM loss.",
        isCorrect: true,
      },
      {
        text: "Some masked tokens are replaced with random words.",
        isCorrect: true,
      },
      { text: "Some selected tokens are left unchanged.", isCorrect: true },
      {
        text: "The full sequence is replaced by mask tokens before the encoder sees it.",
        isCorrect: false,
      },
    ],
    explanation:
      "MLM uses a mixture of masking, random replacement, and unchanged tokens. Core ideas: Only a subset of tokens contribute to the MLM loss; Some masked tokens are replaced with random words; Some selected tokens are left unchanged. Common misconceptions: The full sequence is replaced by mask tokens before the encoder sees it.",
  },

  {
    id: "cme295-lect2-q32",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statements about Next Sentence Prediction (NSP) are correct?",
    options: [
      { text: "It is a binary classification task.", isCorrect: true },
      { text: "It uses sentence pairs as input.", isCorrect: true },
      { text: "It was later shown to be unnecessary.", isCorrect: true },
      {
        text: "It directly improves text generation quality.",
        isCorrect: false,
      },
    ],
    explanation:
      "NSP was later removed in models like RoBERTa with little performance loss. Core ideas: it is a binary classification task; it uses sentence pairs as input; it was later shown to be unnecessary. Common misconceptions: it directly improves text generation quality.",
  },

  {
    id: "cme295-lect2-q33",
    chapter: 2,
    difficulty: "medium",
    prompt: "Which statements about RoBERTa are correct?",
    options: [
      { text: "It removes the NSP objective.", isCorrect: true },
      { text: "It uses dynamic masking across epochs.", isCorrect: true },
      {
        text: "It increases pretraining data size substantially.",
        isCorrect: true,
      },
      {
        text: "It introduces a new attention mechanism rather than changing pretraining data and masking.",
        isCorrect: false,
      },
    ],
    explanation:
      "RoBERTa improves performance through training strategy changes, not architectural ones. Core ideas: it removes the NSP objective; it uses dynamic masking across epochs; it increases pretraining data size substantially. Common misconceptions: it introduces a new attention mechanism rather than changing pretraining data and masking.",
  },

  {
    id: "cme295-lect2-q34",
    chapter: 2,
    difficulty: "easy",
    prompt: "Which statements about DistilBERT are correct?",
    options: [
      { text: "It uses knowledge distillation.", isCorrect: true },
      { text: "It has fewer layers than BERT-base.", isCorrect: true },
      { text: "It retains most of BERT’s performance.", isCorrect: true },
      {
        text: "It increases BERT model size to improve accuracy instead of compressing the teacher model.",
        isCorrect: false,
      },
    ],
    explanation:
      "DistilBERT trades depth for efficiency while preserving performance. Core ideas: it uses knowledge distillation; it has fewer layers than BERT-base; it retains most of BERT’s performance. Common misconceptions: it increases BERT model size to improve accuracy instead of compressing the teacher model.",
  },

  {
    id: "cme295-lect2-q35",
    chapter: 2,
    difficulty: "medium",
    prompt: "Which statements about knowledge distillation are correct?",
    options: [
      {
        text: "It trains a student model to match a teacher’s output distribution.",
        isCorrect: true,
      },
      { text: "It often uses KL divergence as a loss.", isCorrect: true },
      {
        text: "It leverages soft targets rather than hard labels.",
        isCorrect: true,
      },
      {
        text: "It requires task-specific labeled downstream data rather than learning from a teacher distribution.",
        isCorrect: false,
      },
    ],
    explanation:
      "Distillation transfers knowledge via output distributions, not labeled data. Core ideas: it trains a student model to match a teacher’s output distribution; it often uses KL divergence as a loss; it leverages soft targets rather than hard labels. Common misconceptions: it requires task-specific labeled downstream data rather than learning from a teacher distribution.",
  },

  // ============================================================
  // Q51–Q60: EXACTLY 2 TRUE
  // ============================================================

  {
    id: "cme295-lect2-q51",
    chapter: 2,
    difficulty: "medium",
    prompt: "Which statements about encoder-only models are correct?",
    options: [
      {
        text: "They are well suited for classification tasks.",
        isCorrect: true,
      },
      {
        text: "They are not designed to generate text autoregressively.",
        isCorrect: true,
      },
      { text: "They rely on bidirectional self-attention.", isCorrect: true },
      {
        text: "They include masked causal attention for left-to-right generation inside every layer.",
        isCorrect: false,
      },
    ],
    explanation:
      "Encoder-only models are bidirectional and optimized for representation learning. Core ideas: they are well suited for classification tasks; they are not designed to generate text autoregressively; they rely on bidirectional self-attention. Common misconceptions: they include masked causal attention for left-to-right generation inside every layer.",
  },

  {
    id: "cme295-lect2-q52",
    chapter: 2,
    difficulty: "easy",
    prompt: "Which statements about BERT tokenization are correct?",
    options: [
      { text: "It uses WordPiece tokenization.", isCorrect: true },
      {
        text: "It operates as a byte-level tokenizer with a fixed 256-symbol vocabulary.",
        isCorrect: false,
      },
      { text: "Its vocabulary size is around 30k.", isCorrect: true },
      {
        text: "It allows subword splitting for rare words.",
        isCorrect: true,
      },
    ],
    explanation:
      "BERT uses WordPiece with a moderately sized subword vocabulary. Core ideas: it uses WordPiece tokenization; Its vocabulary size is around 30k; it allows subword splitting for rare words. Common misconceptions: it operates as a byte-level tokenizer with a fixed 256-symbol vocabulary.",
  },

  {
    id: "cme295-lect2-q53",
    chapter: 2,
    difficulty: "medium",
    prompt: "Which statements about BERT fine-tuning are correct?",
    options: [
      {
        text: "A task-specific head is added on top of pretrained embeddings.",
        isCorrect: true,
      },
      {
        text: "The entire model can be updated during fine-tuning.",
        isCorrect: true,
      },
      {
        text: "Fine-tuning can require relatively little labeled data.",
        isCorrect: true,
      },
      {
        text: "It replaces the pretrained encoder objective with a new model trained from scratch.",
        isCorrect: false,
      },
    ],
    explanation:
      "Fine-tuning adapts pretrained representations to downstream tasks. Core ideas: A task-specific head is added on top of pretrained embeddings; The entire model can be updated during fine-tuning; Fine-tuning can require relatively little labeled data. Common misconceptions: it replaces the pretrained encoder objective with a new model trained from scratch.",
  },

  {
    id: "cme295-lect2-q54",
    chapter: 2,
    difficulty: "hard",
    prompt: "Which statements about attention complexity are correct?",
    options: [
      { text: "Full self-attention has O(n²) complexity.", isCorrect: true },
      {
        text: "Sparse attention reduces worst-case complexity.",
        isCorrect: true,
      },
      {
        text: "Local attention preserves global connectivity in a single layer.",
        isCorrect: false,
      },
      {
        text: "Stacking local-attention layers can expand the effective receptive field beyond a single window.",
        isCorrect: true,
      },
    ],
    explanation:
      "Full attention forms pairwise query-key scores, so its cost grows quadratically with sequence length. Sparse or sliding-window attention can reduce the per-layer cost, and stacked local layers can pass information farther over depth, but one local layer still does not provide direct global connectivity.",
  },

  {
    id: "cme295-lect2-q55",
    chapter: 2,
    difficulty: "medium",
    prompt: "Which statements about RMSNorm vs LayerNorm are correct?",
    options: [
      { text: "RMSNorm removes mean subtraction.", isCorrect: true },
      {
        text: "LayerNorm subtracts the feature mean before scaling by a feature standard deviation.",
        isCorrect: true,
      },
      {
        text: "RMSNorm keeps performance comparable in practice.",
        isCorrect: true,
      },
      {
        text: "LayerNorm is avoided in Transformers because residual branches already normalize activations.",
        isCorrect: false,
      },
    ],
    explanation:
      "LayerNorm centers each token's feature vector and rescales it, while RMSNorm keeps the rescaling idea but skips mean subtraction. RMSNorm can be lighter while retaining comparable behavior in modern transformers, but LayerNorm is not made unnecessary by residual connections.",
  },

  {
    id: "cme295-lect2-q56",
    chapter: 2,
    difficulty: "medium",
    prompt: "Which statements about GQA vs MQA are correct?",
    options: [
      { text: "GQA shares K/V projections within groups.", isCorrect: true },
      { text: "MQA shares K/V projections across all heads.", isCorrect: true },
      {
        text: "Both GQA and MQA reduce key/value cache memory relative to full multi-head attention.",
        isCorrect: true,
      },
      { text: "GQA eliminates query diversity.", isCorrect: false },
    ],
    explanation:
      "Both methods reduce key/value redundancy while preserving multiple query heads. MQA is the most aggressive sharing pattern, while GQA shares within groups; neither is meant to eliminate query diversity.",
  },

  {
    id: "cme295-lect2-q57",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "Which statements about positional encoding extrapolation are correct?",
    options: [
      {
        text: "Sinusoidal encodings extrapolate naturally to longer sequences.",
        isCorrect: true,
      },
      {
        text: "Learned absolute embeddings extrapolate automatically.",
        isCorrect: false,
      },
      { text: "RoPE supports length extrapolation.", isCorrect: true },
      {
        text: "Attention with Linear Biases (ALiBi) is designed to extrapolate through a deterministic distance bias.",
        isCorrect: true,
      },
    ],
    explanation:
      "Formula-based and relative-position methods are less tied to a fixed learned table of absolute positions. Learned absolute embeddings do not automatically provide vectors beyond their trained range, while sinusoidal encodings, RoPE, and ALiBi are all motivated partly by better behavior outside the training context length.",
  },

  {
    id: "cme295-lect2-q58",
    chapter: 2,
    difficulty: "medium",
    prompt: "Which statements about decoder-only models are correct?",
    options: [
      {
        text: "They are easier to scale with next-token prediction.",
        isCorrect: true,
      },
      {
        text: "They use causal masking so each generated position cannot attend to future tokens.",
        isCorrect: true,
      },
      { text: "They dominate modern LLM architectures.", isCorrect: true },
      {
        text: "They remove cross-attention unless a separate encoder-decoder variant is intentionally built.",
        isCorrect: true,
      },
    ],
    explanation:
      "Decoder-only models align naturally with next-token prediction and generation, which made them attractive for scaling modern large language models. They rely on causal self-attention rather than BERT-style bidirectional attention, and the plain decoder-only family does not need cross-attention to a separate encoder.",
  },

  {
    id: "cme295-lect2-q59",
    chapter: 2,
    difficulty: "easy",
    prompt: "Which statements about span corruption in T5 are correct?",
    options: [
      { text: "It masks contiguous spans of tokens.", isCorrect: true },
      {
        text: "It uses sentinel tokens to mark masked spans.",
        isCorrect: true,
      },
      { text: "It predicts tokens independently of order.", isCorrect: false },
      { text: "It replaces MLM entirely in BERT.", isCorrect: false },
    ],
    explanation:
      "Span corruption replaces contiguous spans and predicts them sequentially. Core ideas: it masks contiguous spans of tokens; it uses sentinel tokens to mark masked spans. Common misconceptions: it predicts tokens independently of order; it replaces MLM entirely in BERT.",
  },

  {
    id: "cme295-lect2-q60",
    chapter: 2,
    difficulty: "medium",
    prompt: "Which statements about model families are correct?",
    options: [
      {
        text: "Encoder–decoder models are well suited for translation.",
        isCorrect: true,
      },
      {
        text: "Encoder-only models excel at representation learning.",
        isCorrect: true,
      },
      {
        text: "Decoder-only models require cross-attention.",
        isCorrect: false,
      },
      {
        text: "All Transformer variants use identical objectives.",
        isCorrect: false,
      },
    ],
    explanation:
      "Different Transformer families specialize in different task regimes. Core ideas: Encoder–decoder models are well suited for translation; Encoder-only models excel at representation learning. Common misconceptions: Decoder-only models require cross-attention; All Transformer variants use identical objectives.",
  },

  // ============================================================
  // Q61–Q75: EXACTLY 2 TRUE
  // ============================================================

  {
    id: "cme295-lect2-q61",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "Which statements about causal (masked) self-attention are correct?",
    options: [
      {
        text: "It prevents a token from attending to future tokens.",
        isCorrect: true,
      },
      {
        text: "It is typically used in decoder-only language models for generation.",
        isCorrect: true,
      },
      {
        text: "It makes the model bidirectional during training.",
        isCorrect: false,
      },
      { text: "It requires a separate encoder to work.", isCorrect: false },
    ],
    explanation:
      "Causal masking enforces an autoregressive constraint: each position can only use information from earlier (and itself) positions. Core ideas: it prevents a token from attending to future tokens; it is typically used in decoder-only language models for generation. Common misconceptions: it makes the model bidirectional during training; it requires a separate encoder to work.",
  },

  {
    id: "cme295-lect2-q62",
    chapter: 2,
    difficulty: "medium",
    prompt: "Which statements about the key–value (KV) cache are correct?",
    options: [
      {
        text: "It stores previously computed key and value tensors to speed up decoding.",
        isCorrect: true,
      },
      {
        text: "Its memory cost grows with sequence length and the number of key/value heads.",
        isCorrect: true,
      },
      { text: "It is only useful for encoder-only models.", isCorrect: false },
      {
        text: "It removes the need to compute attention scores.",
        isCorrect: false,
      },
    ],
    explanation:
      "During autoregressive decoding, past keys/values are reused at every step; caching avoids recomputing them and reduces latency. Core ideas: it stores previously computed key and value tensors to speed up decoding; Its memory cost grows with sequence length and the number of key/value heads. Common misconceptions: it is only useful for encoder-only models; it removes the need to compute attention scores.",
  },

  {
    id: "cme295-lect2-q63",
    chapter: 2,
    difficulty: "hard",
    prompt: "Which statements about attention complexity are correct?",
    options: [
      {
        text: "Full self-attention over length n has time complexity proportional to n².",
        isCorrect: true,
      },
      {
        text: "Sliding window attention with window size w reduces attention computation to roughly n·w.",
        isCorrect: true,
      },
      {
        text: "Sliding window attention makes each token globally connected within a single layer.",
        isCorrect: false,
      },
      {
        text: "Sparse attention always preserves identical outputs to full attention.",
        isCorrect: false,
      },
    ],
    explanation:
      "Restricting attention patterns reduces computation, but exact equivalence to full attention is not guaranteed, and global connectivity typically requires stacking layers. Core ideas: Full self-attention over length n has time complexity proportional to n²; Sliding window attention with window size w reduces attention computation to roughly n·w. Common misconceptions: Sliding window attention makes each token globally connected within a single layer; Sparse attention always preserves identical outputs to full attention.",
  },

  {
    id: "cme295-lect2-q64",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statements about rotation matrices used in Rotary Position Embeddings (RoPE) are correct?",
    options: [
      {
        text: "A 2D rotation matrix preserves vector length (norm).",
        isCorrect: true,
      },
      {
        text: "A 2D rotation matrix is built from sine and cosine of an angle.",
        isCorrect: true,
      },
      {
        text: "A rotation matrix changes dot products in a way that removes all positional information.",
        isCorrect: false,
      },
      {
        text: "RoPE requires learning a separate embedding vector for each absolute position index.",
        isCorrect: false,
      },
    ],
    explanation:
      "RoPE uses deterministic rotations (sine/cosine structure) to inject position into query–key interactions while preserving norms. Core ideas: A 2D rotation matrix preserves vector length (norm); A 2D rotation matrix is built from sine and cosine of an angle. Common misconceptions: A rotation matrix changes dot products in a way that removes all positional information; RoPE requires learning a separate embedding vector for each absolute position index.",
  },

  {
    id: "cme295-lect2-q65",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "Which statements about padding tokens in transformer training are correct?",
    options: [
      {
        text: "Padding helps batch sequences to a fixed length for efficient matrix operations.",
        isCorrect: true,
      },
      {
        text: "Padding tokens are often masked so they do not affect attention or loss.",
        isCorrect: true,
      },
      {
        text: "Padding is required at inference for all models and all deployments.",
        isCorrect: false,
      },
      {
        text: "Padding tokens are used to enforce causal masking.",
        isCorrect: false,
      },
    ],
    explanation:
      "Padding is mainly a batching convenience; models typically mask pads to avoid learning from artificial tokens. Core ideas: Padding helps batch sequences to a fixed length for efficient matrix operations; Padding tokens are often masked so they do not affect attention or loss. Common misconceptions: Padding is required at inference for all models and all deployments; Padding tokens are used to enforce causal masking.",
  },

  {
    id: "cme295-lect2-q66",
    chapter: 2,
    difficulty: "medium",
    prompt: "Which statements about knowledge distillation are correct?",
    options: [
      {
        text: "Distillation often trains the student to match the teacher’s probability distribution, not just the argmax label.",
        isCorrect: true,
      },
      {
        text: "KL divergence is a common loss for matching teacher and student distributions.",
        isCorrect: true,
      },
      {
        text: "Distillation always increases the number of layers in the student.",
        isCorrect: false,
      },
      {
        text: "Distillation requires a decoder-only architecture to work.",
        isCorrect: false,
      },
    ],
    explanation:
      "Distillation is an architecture-agnostic training strategy: compress a model by learning from a stronger teacher’s outputs. Core ideas: Distillation often trains the student to match the teacher’s probability distribution, not just the argmax label; KL divergence is a common loss for matching teacher and student distributions. Common misconceptions: Distillation always increases the number of layers in the student; Distillation requires a decoder-only architecture to work.",
  },

  {
    id: "cme295-lect2-q67",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "In transformer-style sequence models, which statements correctly compare Layer Normalization (LayerNorm) with Batch Normalization (BatchNorm)?",
    options: [
      {
        text: "LayerNorm normalizes across feature dimensions within a single token’s activation vector.",
        isCorrect: true,
      },
      {
        text: "BatchNorm normalizes using statistics aggregated across the batch dimension.",
        isCorrect: true,
      },
      {
        text: "BatchNorm behaves identically at training time and inference time without any special handling.",
        isCorrect: false,
      },
      {
        text: "LayerNorm requires large batch sizes to work well in transformers.",
        isCorrect: false,
      },
    ],
    explanation:
      "LayerNorm is batch-size independent and stable for sequence models; BatchNorm depends on batch statistics and can introduce train–test behavior differences. Core ideas: LayerNorm normalizes across feature dimensions within a single token’s activation vector; BatchNorm normalizes using statistics aggregated across the batch dimension. Common misconceptions: BatchNorm behaves identically at training time and inference time without any special handling; LayerNorm requires large batch sizes to work well in transformers.",
  },

  {
    id: "cme295-lect2-q68",
    chapter: 2,
    difficulty: "medium",
    prompt: "Which statements about RMSNorm are correct?",
    options: [
      {
        text: "It normalizes by the root mean square of the activation components.",
        isCorrect: true,
      },
      {
        text: "It typically uses a learned scaling parameter (gamma) but no learned bias (beta).",
        isCorrect: true,
      },
      {
        text: "It subtracts the mean of the activation vector as a required step.",
        isCorrect: false,
      },
      {
        text: "It cannot be used with residual connections.",
        isCorrect: false,
      },
    ],
    explanation:
      "RMSNorm is a streamlined alternative to LayerNorm: it rescales using RMS and often omits mean-centering and bias. Core ideas: it normalizes by the root mean square of the activation components; it typically uses a learned scaling parameter (gamma) but no learned bias (beta). Common misconceptions: it subtracts the mean of the activation vector as a required step; it cannot be used with residual connections.",
  },

  {
    id: "cme295-lect2-q69",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "Which statements about the [SEP] token in BERT-style inputs are correct?",
    options: [
      {
        text: "It is used to separate sentence A and sentence B in paired inputs.",
        isCorrect: true,
      },
      {
        text: "It helps mark boundaries that are useful for tasks like next sentence prediction.",
        isCorrect: true,
      },
      {
        text: "It is the only token used for classification.",
        isCorrect: false,
      },
      { text: "It forces the model to use causal masking.", isCorrect: false },
    ],
    explanation:
      "[SEP] is a structural delimiter; classification is typically done via the [CLS] representation, not [SEP]. Core ideas: it is used to separate sentence A and sentence B in paired inputs; it helps mark boundaries that are useful for tasks like next sentence prediction. Common misconceptions: it is the only token used for classification; it forces the model to use causal masking.",
  },

  {
    id: "cme295-lect2-q70",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statements about the Masked Language Modeling (MLM) corruption strategy in BERT are correct?",
    options: [
      {
        text: "A portion of selected tokens are replaced by a special [MASK] token.",
        isCorrect: true,
      },
      {
        text: "Some selected tokens are replaced by a random vocabulary token.",
        isCorrect: true,
      },
      {
        text: "Every token in the sequence is selected for masking.",
        isCorrect: false,
      },
      {
        text: "Selected tokens are always replaced; leaving them unchanged is not allowed.",
        isCorrect: false,
      },
    ],
    explanation:
      "BERT-style MLM typically mixes: mask replacement, random replacement, and leaving some selected tokens unchanged. Core ideas: A portion of selected tokens are replaced by a special [MASK] token; Some selected tokens are replaced by a random vocabulary token. Common misconceptions: Every token in the sequence is selected for masking; Selected tokens are always replaced; leaving them unchanged is not allowed.",
  },

  {
    id: "cme295-lect2-q71",
    chapter: 2,
    difficulty: "hard",
    prompt: "Which statements about RoPE and relative position are correct?",
    options: [
      {
        text: "After applying RoPE, the query–key dot product can be expressed to depend on relative offsets (m − n) rather than absolute positions alone.",
        isCorrect: true,
      },
      {
        text: "RoPE can be implemented by applying independent 2D rotations to pairs of embedding dimensions.",
        isCorrect: true,
      },
      {
        text: "RoPE encodes position by adding a learned vector to each token embedding at the input.",
        isCorrect: false,
      },
      {
        text: "RoPE requires bucketizing distances into discrete bins, as in T5 relative bias.",
        isCorrect: false,
      },
    ],
    explanation:
      "RoPE is a rotation-based relative method; T5 relative bias is a learned bias lookup over bucketized distances. Core ideas: After applying RoPE, the query–key dot product can be expressed to depend on relative offsets (m − n) rather than absolute positions alone; RoPE can be implemented by applying independent 2D rotations to pairs of embedding dimensions. Common misconceptions: RoPE encodes position by adding a learned vector to each token embedding at the input; RoPE requires bucketizing distances into discrete bins, as in T5 relative bias.",
  },

  {
    id: "cme295-lect2-q72",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statements about sliding window attention across multiple layers are correct?",
    options: [
      {
        text: "Even with local windows, information can propagate across long distances by stacking layers.",
        isCorrect: true,
      },
      {
        text: "The effective receptive field can expand as depth increases.",
        isCorrect: true,
      },
      {
        text: "A single local-attention layer gives every token direct access to every other token.",
        isCorrect: false,
      },
      {
        text: "Stacking local-attention layers reduces context mixing compared to a single layer.",
        isCorrect: false,
      },
    ],
    explanation:
      "Local attention limits direct connections per layer, but deeper stacks allow signals to travel farther—similar to receptive field growth. Core ideas: Even with local windows, information can propagate across long distances by stacking layers; The effective receptive field can expand as depth increases. Common misconceptions: A single local-attention layer gives every token direct access to every other token; Stacking local-attention layers reduces context mixing compared to a single layer.",
  },

  {
    id: "cme295-lect2-q73",
    chapter: 2,
    difficulty: "easy",
    prompt: "Which statements about tokenizer vocabulary sizes are correct?",
    options: [
      {
        text: "Subword tokenizers commonly have vocabularies on the order of tens of thousands of tokens.",
        isCorrect: true,
      },
      {
        text: "A byte-level vocabulary can be as small as 256 symbols (2^8).",
        isCorrect: true,
      },
      {
        text: "All modern tokenizers must be strictly word-level (no subwords).",
        isCorrect: false,
      },
      {
        text: "A smaller vocabulary always guarantees better model accuracy.",
        isCorrect: false,
      },
    ],
    explanation:
      "Vocabulary size trades off sequence length, representation granularity, and model capacity; byte-level approaches drastically reduce vocab size. Core ideas: Subword tokenizers commonly have vocabularies on the order of tens of thousands of tokens; A byte-level vocabulary can be as small as 256 symbols (2^8). Common misconceptions: All modern tokenizers must be strictly word-level (no subwords); A smaller vocabulary always guarantees better model accuracy.",
  },

  {
    id: "cme295-lect2-q74",
    chapter: 2,
    difficulty: "medium",
    prompt: "Which statements about T5-style span corruption are correct?",
    options: [
      {
        text: "It masks contiguous spans (one or more tokens) rather than masking isolated tokens only.",
        isCorrect: true,
      },
      {
        text: "Sentinel tokens act as placeholders that indicate where spans were removed.",
        isCorrect: true,
      },
      {
        text: "It requires causal masking in the encoder to work.",
        isCorrect: false,
      },
      {
        text: "The decoder output is restricted to a single token per sentinel span.",
        isCorrect: false,
      },
    ],
    explanation:
      "Span corruption replaces spans with sentinel markers; the decoder reconstructs the missing spans as sequences. Core ideas: it masks contiguous spans (one or more tokens) rather than masking isolated tokens only; Sentinel tokens act as placeholders that indicate where spans were removed. Common misconceptions: it requires causal masking in the encoder to work; The decoder output is restricted to a single token per sentinel span.",
  },

  {
    id: "cme295-lect2-q75",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "Which statements about pre-norm vs post-norm transformers are correct?",
    options: [
      {
        text: "Pre-norm applies normalization before the attention or feed-forward sublayer.",
        isCorrect: true,
      },
      {
        text: "Pre-norm is often preferred for stability when training deeper transformers.",
        isCorrect: true,
      },
      {
        text: "Post-norm places normalization before every residual addition.",
        isCorrect: false,
      },
      {
        text: "Post-norm eliminates the need for residual connections.",
        isCorrect: false,
      },
    ],
    explanation:
      "Pre-norm improves gradient flow in deep stacks; post-norm normalizes after residual addition and can be less stable at large depth. Core ideas: Pre-norm applies normalization before the attention or feed-forward sublayer; Pre-norm is often preferred for stability when training deeper transformers. Common misconceptions: Post-norm places normalization before every residual addition; Post-norm eliminates the need for residual connections.",
  },

  // ============================================================
  // Q76–Q100: EXACTLY 1 TRUE
  // ============================================================

  {
    id: "cme295-lect2-q76",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "Which statements correctly explain why a transformer needs an explicit position signal?",
    options: [
      {
        text: "Self-attention compares tokens as a set of query-key-value vectors rather than processing them one step at a time.",
        isCorrect: true,
      },
      {
        text: "Without a position signal, the attention computation has no built-in way to distinguish the same tokens in a different order.",
        isCorrect: true,
      },
      {
        text: "Adding or injecting position information lets the model make similarity depend partly on order or distance.",
        isCorrect: true,
      },
      {
        text: "The position signal can be supplied at the input embedding level or directly inside attention logits or query-key interactions.",
        isCorrect: true,
      },
    ],
    explanation:
      "Self-attention itself is not a recurrent scan, so order must enter through an explicit mechanism. The original transformer added position encodings to inputs, while later approaches such as T5 bias, ALiBi, and RoPE move positional effects closer to the attention score calculation.",
  },

  {
    id: "cme295-lect2-q77",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "A transformer was trained with learned absolute position embeddings for positions 1 through 512 and is then asked to process 1024-token inputs. Which statements correctly diagnose the position-embedding issue?",
    options: [
      {
        text: "The learned embedding table directly supplies vectors only for the positions it was trained or parameterized to cover.",
        isCorrect: true,
      },
      {
        text: "A formula-based sinusoidal encoding can be evaluated for positions beyond those seen during training.",
        isCorrect: true,
      },
      {
        text: "Relative-position methods avoid relying only on a separate learned vector for each absolute index.",
        isCorrect: true,
      },
      {
        text: "A learned absolute table may also absorb training-set-specific positional regularities rather than a clean distance rule.",
        isCorrect: true,
      },
    ],
    explanation:
      "Learned absolute embeddings are simple and trainable, but they tie position handling to a finite table and the training distribution. The lecture contrasts that with hard-coded sinusoidal functions and relative or attention-level methods, which are motivated partly by extrapolation and distance-aware attention.",
  },

  {
    id: "cme295-lect2-q78",
    chapter: 2,
    difficulty: "hard",
    prompt: "Which statement about sinusoidal positional encodings is correct?",
    options: [
      {
        text: "They use multiple frequencies so some dimensions vary quickly with position while others vary slowly.",
        isCorrect: true,
      },
      {
        text: "They require learning one trainable embedding vector for every possible position.",
        isCorrect: false,
      },
      {
        text: "They encode position by adding a learned bias term inside the attention softmax.",
        isCorrect: false,
      },
      {
        text: "They prevent extrapolation to longer sequences by construction.",
        isCorrect: false,
      },
    ],
    explanation:
      "Sinusoidal encodings use sine/cosine waves at different frequencies across dimensions to represent positions and can extrapolate. Core idea: they use multiple frequencies so some dimensions vary quickly with position while others vary slowly. Common misconceptions: they require learning one trainable embedding vector for every possible position; they encode position by adding a learned bias term inside the attention softmax; they prevent extrapolation to longer sequences by construction.",
  },

  {
    id: "cme295-lect2-q79",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statements correctly distinguish T5-style relative position bias from Rotary Position Embeddings (RoPE)?",
    options: [
      {
        text: "T5-style bias adds learned distance-dependent terms to attention logits before the softmax.",
        isCorrect: true,
      },
      {
        text: "T5-style bias can bucket relative offsets so several distances share a learned bias value.",
        isCorrect: true,
      },
      {
        text: "RoPE rotates query and key vectors, making their dot product depend on relative angle differences.",
        isCorrect: true,
      },
      {
        text: "Both methods act closer to the attention score computation than merely adding an absolute position vector to token embeddings.",
        isCorrect: true,
      },
    ],
    explanation:
      "T5 and RoPE both move positional information into the attention mechanism, but they do it differently. T5 adds learned relative logit biases, while RoPE changes the query-key geometry through deterministic rotations.",
  },

  {
    id: "cme295-lect2-q80",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "Which statements correctly characterize Attention with Linear Biases (ALiBi)?",
    options: [
      {
        text: "It uses a deterministic linear bias based on relative position distance.",
        isCorrect: true,
      },
      {
        text: "It is added to attention logits before the softmax rather than to value vectors after attention.",
        isCorrect: true,
      },
      {
        text: "It avoids learning a separate absolute embedding vector for every position index.",
        isCorrect: true,
      },
      {
        text: "It was proposed as a way to train on shorter sequences while testing on longer ones.",
        isCorrect: true,
      },
    ],
    explanation:
      "ALiBi keeps the attention softmax but changes the logits with a deterministic distance-based term. Its main contrast with learned absolute embeddings is that it uses a simple relative bias instead of a finite learned position table.",
  },

  {
    id: "cme295-lect2-q81",
    chapter: 2,
    difficulty: "hard",
    prompt: "Which statement about RoPE is correct?",
    options: [
      {
        text: "It injects position by rotating query and key vectors so attention depends on relative offsets.",
        isCorrect: true,
      },
      {
        text: "It injects position by learning an embedding per absolute position and adding it only to values.",
        isCorrect: false,
      },
      {
        text: "It works by masking future tokens to enforce causality.",
        isCorrect: false,
      },
      {
        text: "It encodes relative distance only by discretizing offsets into buckets.",
        isCorrect: false,
      },
    ],
    explanation:
      "RoPE is a rotation-based positional method that encodes relative position into the dot products used by attention. Core idea: it injects position by rotating query and key vectors so attention depends on relative offsets. Common misconceptions: it injects position by learning an embedding per absolute position and adding it only to values; it works by masking future tokens to enforce causality; it encodes relative distance only by discretizing offsets into buckets.",
  },

  {
    id: "cme295-lect2-q82",
    chapter: 2,
    difficulty: "medium",
    prompt: "Which statement about Grouped-Query Attention (GQA) is correct?",
    options: [
      {
        text: "It shares key/value projections within groups of heads to reduce KV cache size.",
        isCorrect: true,
      },
      {
        text: "It removes multi-head attention and replaces it with a single head.",
        isCorrect: false,
      },
      {
        text: "It shares query projections across all heads while keeping distinct keys/values.",
        isCorrect: false,
      },
      {
        text: "It requires an encoder and decoder to share attention weights.",
        isCorrect: false,
      },
    ],
    explanation:
      "GQA reduces key/value redundancy (memory) by grouping heads to share K/V while keeping multiple query heads. Core idea: it shares key/value projections within groups of heads to reduce KV cache size. Common misconceptions: it removes multi-head attention and replaces it with a single head; it shares query projections across all heads while keeping distinct keys/values; it requires an encoder and decoder to share attention weights.",
  },

  {
    id: "cme295-lect2-q83",
    chapter: 2,
    difficulty: "medium",
    prompt: "Which statement about Multi-Query Attention (MQA) is correct?",
    options: [
      {
        text: "It is the extreme case where all heads share the same key and value projections.",
        isCorrect: true,
      },
      {
        text: "It increases the KV cache memory compared to standard multi-head attention.",
        isCorrect: false,
      },
      {
        text: "It forces all heads to share the same query projection as well.",
        isCorrect: false,
      },
      {
        text: "It can only be used in encoder-only transformers.",
        isCorrect: false,
      },
    ],
    explanation:
      "MQA minimizes K/V memory by sharing them across all heads, while queries typically remain per-head. Core idea: it is the extreme case where all heads share the same key and value projections. Common misconceptions: it increases the KV cache memory compared to standard multi-head attention; it forces all heads to share the same query projection as well; it can only be used in encoder-only transformers.",
  },

  {
    id: "cme295-lect2-q84",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe BERT-style input processing before the encoder stack?",
    options: [
      {
        text: "WordPiece tokenization represents text with learned subword units rather than only whole words.",
        isCorrect: true,
      },
      {
        text: "A [CLS] token is placed at the beginning so its contextualized representation can support sequence-level classification.",
        isCorrect: true,
      },
      {
        text: "[SEP] tokens mark boundaries between paired segments and the end of the input.",
        isCorrect: true,
      },
      {
        text: "Token, position, and segment embeddings are added before the encoder processes the sequence.",
        isCorrect: true,
      },
    ],
    explanation:
      "BERT's input pipeline combines tokenization and several embedding signals before bidirectional self-attention. WordPiece handles subwords, [CLS] and [SEP] provide structural tokens, and segment embeddings help mark sentence-pair membership.",
  },

  {
    id: "cme295-lect2-q85",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe the self-supervised signals used in original BERT pretraining?",
    options: [
      {
        text: "Next Sentence Prediction (NSP) is a binary classification task that predicts whether sentence B follows sentence A in the corpus.",
        isCorrect: true,
      },
      {
        text: "Masked Language Modeling (MLM) predicts selected original tokens from bidirectional context.",
        isCorrect: true,
      },
      {
        text: "Both MLM labels and NSP labels can be constructed from raw text without manual task labels.",
        isCorrect: true,
      },
      {
        text: "The NSP classification signal is read from a sequence-level representation rather than from a causal decoder.",
        isCorrect: true,
      },
    ],
    explanation:
      "Original BERT combines MLM with NSP to learn contextual encoder representations from unlabeled corpora. MLM uses masked-token prediction, while NSP is a sentence-pair classification task rather than autoregressive next-sentence generation.",
  },

  {
    id: "cme295-lect2-q86",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "Which statement about RoBERTa’s changes relative to BERT is correct?",
    options: [
      {
        text: "It removes NSP and uses training strategy improvements like dynamic masking and more data.",
        isCorrect: true,
      },
      {
        text: "It replaces self-attention with convolutional layers for efficiency.",
        isCorrect: false,
      },
      {
        text: "It requires an encoder–decoder architecture to work.",
        isCorrect: false,
      },
      {
        text: "It enforces causal masking to enable text generation.",
        isCorrect: false,
      },
    ],
    explanation:
      "RoBERTa’s gains primarily come from better pretraining procedures (data, masking, schedule), not a fundamentally new architecture. Core idea: it removes NSP and uses training strategy improvements like dynamic masking and more data. Common misconceptions: it replaces self-attention with convolutional layers for efficiency; it requires an encoder–decoder architecture to work; it enforces causal masking to enable text generation.",
  },

  {
    id: "cme295-lect2-q87",
    chapter: 2,
    difficulty: "easy",
    prompt: "Which statements about DistilBERT are correct?",
    options: [
      {
        text: "It compresses BERT using knowledge distillation to keep performance while reducing size and latency.",
        isCorrect: true,
      },
      {
        text: "It trains a smaller student model to mimic information from a larger teacher model.",
        isCorrect: true,
      },
      {
        text: "Its motivation is lower inference cost while retaining much of BERT's downstream usefulness.",
        isCorrect: true,
      },
      {
        text: "It is an efficiency-oriented BERT-family variant rather than a replacement of self-attention with recurrence.",
        isCorrect: true,
      },
    ],
    explanation:
      "DistilBERT is presented as a compression strategy: a smaller student learns from a larger BERT-like teacher. The point is efficiency and latency reduction while preserving much of the representational value, not changing the model family to recurrence.",
  },

  {
    id: "cme295-lect2-q88",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "A teacher assigns probabilities \\(t=(0.7,0.2,0.1)\\), while a student assigns \\(s=(0.6,0.3,0.1)\\). Which statement best describes the KL-divergence term used in distillation?",
    options: [
      {
        text: "It penalizes the student when its full probability distribution differs from the teacher's soft distribution, not only when the top class changes.",
        isCorrect: true,
      },
      {
        text: "It reduces to ordinary hard-label cross-entropy only because the teacher assigns nonzero probability to three classes.",
        isCorrect: false,
      },
      {
        text: "It should be computed directly on the unnormalized logits, because applying softmax would discard the teacher's dark knowledge.",
        isCorrect: false,
      },
      {
        text: "It is symmetric, so \\(D_{KL}(t\\|s)\\) and \\(D_{KL}(s\\|t)\\) impose identical gradients on the student.",
        isCorrect: false,
      },
    ],
    explanation:
      "Distillation can use the teacher's whole soft distribution, so KL divergence penalizes probability mass that shifts between non-top classes as well as the top class. It is computed on normalized distributions, is not just MSE on logits, and is not symmetric in the teacher-student direction.",
  },

  {
    id: "cme295-lect2-q89",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statement about the masked language modeling (MLM) intuition is correct?",
    options: [
      {
        text: "Predicting a masked token encourages using both left and right context, yielding bidirectional representations.",
        isCorrect: true,
      },
      {
        text: "MLM forces the model to use only left context to predict the next token.",
        isCorrect: false,
      },
      {
        text: "MLM is a supervised task requiring human labels for each masked token.",
        isCorrect: false,
      },
      {
        text: "MLM makes the attention matrix strictly lower-triangular.",
        isCorrect: false,
      },
    ],
    explanation:
      "MLM is self-supervised and leverages both sides of context because the encoder attention is not causal. Core idea: Predicting a masked token encourages using both left and right context, yielding bidirectional representations. Common misconceptions: MLM forces the model to use only left context to predict the next token; MLM is a supervised task requiring human labels for each masked token; MLM makes the attention matrix strictly lower-triangular.",
  },

  {
    id: "cme295-lect2-q90",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "During encoder-decoder training, the target prefix is \\([\\text{<bos>}, y_1, y_2]\\) and the model is learning to predict \\([y_1,y_2,y_3]\\). Which statement correctly describes teacher forcing?",
    options: [
      {
        text: "The decoder conditions on the ground-truth previous target tokens during training, even though inference must condition on tokens it has generated.",
        isCorrect: true,
      },
      {
        text: "The decoder conditions on its own sampled token at every training step so the training distribution exactly matches inference.",
        isCorrect: false,
      },
      {
        text: "The encoder receives the target prefix, while the decoder receives only the source sequence and predicts all targets independently.",
        isCorrect: false,
      },
      {
        text: "The method applies only to encoder-only masked language models, because decoders never observe previous target tokens during training.",
        isCorrect: false,
      },
    ],
    explanation:
      "Teacher forcing trains the decoder with the true previous target tokens as context, which makes next-token supervision stable and parallelizable across a known target sequence. At inference the model no longer has those ground-truth prefixes, so it must condition on its own generated tokens instead.",
  },

  {
    id: "cme295-lect2-q91",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "A translation model reads a full source sentence and then generates the target sentence left-to-right. Which statement correctly matches the encoder-decoder Transformer pattern?",
    options: [
      {
        text: "The decoder uses causal self-attention over generated target prefixes and cross-attention to the encoder's source representations.",
        isCorrect: true,
      },
      {
        text: "The encoder uses causal masking over the source sentence, while the decoder uses bidirectional attention over the future target tokens.",
        isCorrect: false,
      },
      {
        text: "The decoder replaces cross-attention with segment embeddings, so source and target tokens are processed as one BERT-style pair.",
        isCorrect: false,
      },
      {
        text: "The architecture is encoder-only during training and becomes decoder-only at inference by dropping the source representations.",
        isCorrect: false,
      },
    ],
    explanation:
      "In the classic encoder-decoder Transformer, the encoder builds source representations and the decoder generates target tokens causally while cross-attending to those source states. Segment embeddings do not replace cross-attention, and the source encoder is typically allowed bidirectional self-attention.",
  },

  {
    id: "cme295-lect2-q92",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "Two inputs contain the same token embeddings but in different orders. Which statement best explains why later positional methods move position effects into the attention score computation?",
    options: [
      {
        text: "Attention logits decide which token pairs are treated as similar, so making those logits depend on relative position directly changes the routing of information.",
        isCorrect: true,
      },
      {
        text: "Adding absolute positional vectors to inputs guarantees every later attention layer depends only on relative distance, regardless of content.",
        isCorrect: false,
      },
      {
        text: "Position should affect only value vectors, because query-key scores should remain completely permutation-invariant.",
        isCorrect: false,
      },
      {
        text: "Once a model uses multiple heads, self-attention has an inherent token order even without any positional signal.",
        isCorrect: false,
      },
    ],
    explanation:
      "Attention logits route information by determining which token pairs receive high weight. Methods such as relative bias and RoPE therefore make order or distance affect the actual query-key comparison, whereas absolute input embeddings alone do not guarantee purely relative-distance attention.",
  },

  {
    id: "cme295-lect2-q93",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statement about the relationship between local attention and global context is correct?",
    options: [
      {
        text: "Local attention can still yield long-range influence after multiple layers because intermediate tokens relay information.",
        isCorrect: true,
      },
      {
        text: "Local attention prevents any information from traveling beyond the window even with many layers.",
        isCorrect: false,
      },
      {
        text: "Local attention makes complexity worse than full attention for long sequences.",
        isCorrect: false,
      },
      {
        text: "Local attention is identical to adding sinusoidal positional encodings.",
        isCorrect: false,
      },
    ],
    explanation:
      "Stacking layers lets information hop across windows, gradually expanding the effective context. Core idea: Local attention can still yield long-range influence after multiple layers because intermediate tokens relay information. Common misconceptions: Local attention prevents any information from traveling beyond the window even with many layers; Local attention makes complexity worse than full attention for long sequences; Local attention is identical to adding sinusoidal positional encodings.",
  },

  {
    id: "cme295-lect2-q94",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "A BERT input is formed as `[CLS] sentence A [SEP] sentence B [SEP]`. Which statement correctly identifies the role of segment embeddings?",
    options: [
      {
        text: "They add a learned sentence-A/sentence-B signal so the encoder can distinguish the two segments in paired-input tasks.",
        isCorrect: true,
      },
      {
        text: "They replace absolute position embeddings by assigning a distinct vector to each token index in the sequence.",
        isCorrect: false,
      },
      {
        text: "They mark which tokens were selected for MLM prediction, while [MASK] embeddings carry the sentence-pair signal.",
        isCorrect: false,
      },
      {
        text: "They provide the WordPiece subword identity, while token embeddings only indicate segment membership.",
        isCorrect: false,
      },
    ],
    explanation:
      "BERT adds segment embeddings alongside token and position embeddings so the encoder can tell which tokens belong to sentence A versus sentence B. They are not token identities, MLM markers, or absolute position indexes; those roles are handled by separate input components.",
  },

  {
    id: "cme295-lect2-q95",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "Which statement about why keys/values are prime targets for sharing (MQA/GQA) is correct?",
    options: [
      {
        text: "During autoregressive decoding, past keys/values are reused at every step, so reducing their head count shrinks the KV cache significantly.",
        isCorrect: true,
      },
      {
        text: "Queries are reused across decoding steps in exactly the same way as keys/values, so sharing queries yields the biggest memory savings.",
        isCorrect: false,
      },
      {
        text: "Sharing keys/values forces attention to become local rather than global.",
        isCorrect: false,
      },
      {
        text: "Sharing keys/values removes the need for output projection after concatenating heads.",
        isCorrect: false,
      },
    ],
    explanation:
      "Caching makes K/V memory the bottleneck at long contexts; sharing K/V reduces cache size while keeping multi-head query diversity. Core idea: During autoregressive decoding, past keys/values are reused at every step, so reducing their head count shrinks the KV cache significantly. Common misconceptions: Queries are reused across decoding steps in exactly the same way as keys/values, so sharing queries yields the biggest memory savings; Sharing keys/values forces attention to become local rather than global; Sharing keys/values removes the need for output projection after concatenating heads.",
  },

  {
    id: "cme295-lect2-q96",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statement best explains why decoder-only Transformers are a natural fit for large-scale language generation?",
    options: [
      {
        text: "Their causal objective trains the same conditional next-token distribution that is used when generating text autoregressively.",
        isCorrect: true,
      },
      {
        text: "They use bidirectional masking during pretraining and then switch to causal masking only for fine-tuning.",
        isCorrect: false,
      },
      {
        text: "They avoid the attention softmax at long context by replacing key-value caches with masked language modeling.",
        isCorrect: false,
      },
      {
        text: "They depend on encoder cross-attention to a separately encoded source sequence for ordinary open-ended text generation.",
        isCorrect: false,
      },
    ],
    explanation:
      "Decoder-only LMs use the same causal next-token factorization during training and generation, which makes the objective simple and well aligned with open-ended text generation. That does not make them bidirectional encoders, remove attention costs, or require an encoder source sequence.",
  },

  {
    id: "cme295-lect2-q97",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "Which statement about “cased” vs “uncased” checkpoints is correct?",
    options: [
      {
        text: "A cased model preserves capitalization distinctions (e.g., 'US' vs 'us'), while an uncased model typically lowercases inputs.",
        isCorrect: true,
      },
      {
        text: "A cased model always has a byte-level tokenizer, while an uncased model always uses WordPiece.",
        isCorrect: false,
      },
      {
        text: "Casing changes the transformer architecture (encoder vs decoder).",
        isCorrect: false,
      },
      {
        text: "Uncased models cannot be fine-tuned on classification tasks.",
        isCorrect: false,
      },
    ],
    explanation:
      "Casing is a data/tokenization choice, not an architectural constraint; it can matter depending on downstream tasks. Core idea: A cased model preserves capitalization distinctions (e.g., 'US' vs 'us'), while an uncased model typically lowercases inputs. Common misconceptions: A cased model always has a byte-level tokenizer, while an uncased model always uses WordPiece; Casing changes the transformer architecture (encoder vs decoder); Uncased models cannot be fine-tuned on classification tasks.",
  },

  {
    id: "cme295-lect2-q98",
    chapter: 2,
    difficulty: "hard",
    prompt:
      "A deep Transformer block repeatedly adds residual streams and applies attention/MLP sublayers. Which statement best explains why normalization helps optimization?",
    options: [
      {
        text: "It keeps each token's hidden-state scale more controlled across layers, which makes gradient-based training more stable.",
        isCorrect: true,
      },
      {
        text: "It makes residual connections unnecessary, because normalized sublayers no longer need an identity path for gradient flow.",
        isCorrect: false,
      },
      {
        text: "It turns attention into a convex operation, so all training runs converge to the same global optimum.",
        isCorrect: false,
      },
      {
        text: "It lowers the asymptotic cost of full self-attention from \\(O(n^2)\\) to \\(O(n)\\) by rescaling activations.",
        isCorrect: false,
      },
    ],
    explanation:
      "Normalization controls hidden-state scale before or after sublayers, which helps deep residual Transformer stacks train stably. It complements residual paths and nonlinear MLPs; it does not change the asymptotic self-attention cost or turn the optimization problem into a convex one.",
  },

  {
    id: "cme295-lect2-q99",
    chapter: 2,
    difficulty: "medium",
    prompt:
      "Which statement best explains why BERT's MLM and NSP objectives were self-supervised rather than manually labeled tasks?",
    options: [
      {
        text: "The masked-token labels and sentence-pair labels can be created from raw text by hiding tokens and sampling adjacent or non-adjacent sentence pairs.",
        isCorrect: true,
      },
      {
        text: "Human annotators must label each masked token because the original token is removed before the training example is stored.",
        isCorrect: false,
      },
      {
        text: "They are self-supervised because BERT uses causal masking, so the label for every position is the next token to the right.",
        isCorrect: false,
      },
      {
        text: "They avoid labels by training only on whether WordPiece tokens are cased or uncased, not on token identity or sentence relationships.",
        isCorrect: false,
      },
    ],
    explanation:
      "MLM labels are the original tokens hidden from the input, and NSP labels can be generated by choosing real adjacent sentence pairs or sampled mismatches from the corpus. Those labels come from raw text construction rather than human annotation, and they differ from causal next-token generation.",
  },

  {
    id: "cme295-lect2-q100",
    chapter: 2,
    difficulty: "easy",
    prompt:
      "A model uses fixed sinusoidal positional encodings and is evaluated at a position index larger than any used during training. Which statement is correct?",
    options: [
      {
        text: "The encoding vector can still be computed from the same sine/cosine formulas, although task performance may still depend on whether the model learned to use longer-range patterns.",
        isCorrect: true,
      },
      {
        text: "The model must append newly learned position vectors before inference, because sinusoidal encodings are finite lookup tables.",
        isCorrect: false,
      },
      {
        text: "Extrapolation works only by bucketizing all unseen distances into the same learned relative-position bin.",
        isCorrect: false,
      },
      {
        text: "The formula guarantees identical behavior at longer lengths, so no degradation can occur from the changed context distribution.",
        isCorrect: false,
      },
    ],
    explanation:
      "Sinusoidal encodings are analytic functions of the position index, so there is no finite learned table that blocks computing a vector for a larger index. That only solves the representation lookup problem; the model may still generalize imperfectly if longer contexts differ from its training distribution.",
  },
];
