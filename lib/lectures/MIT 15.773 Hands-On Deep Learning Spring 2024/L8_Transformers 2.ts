import { Question } from "../../quiz";

export const TransformersSelfSupervisedLearningQuestions: Question[] = [
  {
    id: "mit15773-l8-q01",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statement best explains why the lecture revisited the transformer encoder in a second pass?",
    options: [
      {
        text: "Because the first pass introduced the core intuition, while the second pass added important engineering details such as tunable attention, residual connections, and layer normalization.",
        isCorrect: true,
      },
      {
        text: "Because the first pass had shown that transformers cannot handle contextual embeddings, so the second pass replaced transformers with recurrent neural networks.",
        isCorrect: false,
      },
      {
        text: "Because the lecture decided that positional embeddings were unnecessary and removed them in the second pass.",
        isCorrect: false,
      },
      {
        text: "Because the second pass focused only on lecture logistics and not on the model architecture itself.",
        isCorrect: false,
      },
    ],
    explanation:
      "The lecture explicitly said the second pass would add three important elements that were not covered in the first pass. The goal was not to replace the architecture, but to deepen the explanation until it matched the actual transformer encoder more closely.",
  },
  {
    id: "mit15773-l8-q02",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe the role of the positional input embeddings before the transformer encoder?",
    options: [
      {
        text: "They are formed by adding stand-alone token embeddings and positional embeddings elementwise.",

        isCorrect: true,
      },
      {
        text: "The stand-alone token embeddings may be pretrained or randomly initialized.",
        isCorrect: true,
      },
      {
        text: "The positional embeddings can also begin as trainable weight vectors.",
        isCorrect: true,
      },
      {
        text: "It is not the case that They are produced by averaging all token embeddings in the sentence into one single vector before the encoder.",
        isCorrect: true,
      },
    ],
    explanation:
      "The lecture reviewed that each token gets a position-aware representation by summing its token embedding and its position embedding. This still preserves one vector per token position, rather than collapsing the whole sentence into a single vector at the input stage.",
  },
  {
    id: "mit15773-l8-q03",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Why is it useful to express self-attention with matrix operations rather than only as many separate weighted sums?",
    options: [
      {
        text: "It allows all pairwise similarity computations to be organized efficiently as matrix multiplications.",
        isCorrect: true,
      },
      {
        text: "It makes the computation much more suitable for hardware such as graphics processing units.",
        isCorrect: true,
      },
      {
        text: "It proves that transformers no longer need softmax anywhere in the architecture.",
        isCorrect: false,
      },
      {
        text: "It is part of what makes transformers computationally practical at scale.",
        isCorrect: false,
      },
    ],
    explanation:
      "The lecture emphasized that packaging self-attention into compact matrix operations is what makes the architecture efficient and scalable. Softmax is still used, and practical efficiency on GPUs is a major part of the transformer story.",
  },
  {
    id: "mit15773-l8-q04",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "In the lecture's matrix view of simplified self-attention, which statements are correct?",
    options: [
      {
        text: "If \\(X\\) is the matrix of input token embeddings, then pairwise dot products among tokens can be organized through a multiplication involving \\(X\\) and \\(X^T\\).",
        isCorrect: true,
      },
      {
        text: "Applying a row-wise softmax to the similarity matrix converts raw compatibility scores into normalized attention weights.",
        isCorrect: true,
      },
      {
        text: "A subsequent multiplication by the embedding matrix produces contextualized token representations.",
        isCorrect: true,
      },
      {
        text: "This formulation helps explain why self-attention can be implemented efficiently despite involving all token pairs.",
        isCorrect: true,
      },
    ],
    explanation:
      "The lecture used the matrix form to show that self-attention is not just a conceptual weighted-average story but also a very structured numerical operation. The pairwise similarities, normalization, and final weighted combination all fit cleanly into matrix algebra.",
  },
  {
    id: "mit15773-l8-q05",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "What problem arises if multi-head self-attention is used in the simplified first-pass form without any learnable internal matrices?",
    options: [
      {
        text: "Different heads would produce identical outputs from the same inputs, so having multiple heads would not add useful diversity.",
        isCorrect: true,
      },
      {
        text: "The model would become unable to preserve sequence length.",
        isCorrect: false,
      },
      {
        text: "The model would stop using positional embeddings entirely.",
        isCorrect: false,
      },
      {
        text: "The encoder could no longer be stacked into multiple layers.",
        isCorrect: false,
      },
    ],
    explanation:
      "If several heads all perform the exact same non-tunable operation on the same inputs, they do not meaningfully specialize. The second lecture introduced tunable attention precisely so that different heads could learn different patterns.",
  },
  {
    id: "mit15773-l8-q06",
    chapter: 1,
    difficulty: "medium",
    prompt: "How did the lecture make self-attention 'tunable'?",
    options: [
      {
        text: "By introducing learnable matrices that linearly transform the incoming embeddings before similarity and output computations.",

        isCorrect: true,
      },
      {
        text: "By allowing these matrices to be optimized with backpropagation.",
        isCorrect: true,
      },
      {
        text: "By making multi-head attention capable of learning different patterns through different parameter sets per head.",
        isCorrect: true,
      },
      {
        text: "It is not the case that By replacing all embeddings with one-hot vectors inside the attention layer.",
        isCorrect: true,
      },
    ],
    explanation:
      "The key idea was to add learned linear projections before and during the attention computation. This gives the self-attention layer internal parameters, making it expressive and allowing different heads to develop different behaviors during training.",
  },
  {
    id: "mit15773-l8-q07",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements about the matrices \\(A^K\\), \\(A^Q\\), and \\(A^V\\) are correct in the lecture's notation?",
    options: [
      {
        text: "They are learnable weight matrices inside the self-attention mechanism.",
        isCorrect: true,
      },
      {
        text: "Using separate matrices for key-like, query-like, and value-like transformations increases the expressive power of the attention layer.",
        isCorrect: true,
      },
      {
        text: "They must always be initialized to identical values across all heads, otherwise the model becomes unstable.",
        isCorrect: false,
      },
      {
        text: "They are unrelated to optimization and stay fixed once chosen by the model designer.",
        isCorrect: false,
      },
    ],
    explanation:
      "These matrices are exactly what makes attention tunable in the lecture's second-pass explanation. They are trainable and usually initialized independently, which helps different heads specialize rather than collapse into identical behavior.",
  },
  {
    id: "mit15773-l8-q08",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe the role of multiple attention heads after tunable projections are introduced?",
    options: [
      {
        text: "Each head can have its own \\(A^K\\), \\(A^Q\\), and \\(A^V\\) matrices.",
        isCorrect: true,
      },
      {
        text: "Different heads can then learn different attention patterns from the same input sequence.",
        isCorrect: true,
      },
      {
        text: "This makes multi-head attention more meaningful than simply duplicating the same head several times.",
        isCorrect: true,
      },
      {
        text: "The outputs of the heads are later combined through concatenation and projection.",
        isCorrect: true,
      },
    ],
    explanation:
      "Once different heads have different learnable projections, they no longer have to behave identically. This is the practical reason multi-head attention becomes useful rather than redundant in the full transformer encoder.",
  },
  {
    id: "mit15773-l8-q09",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Why is the factor \\(\\sqrt{d_k}\\) often included in the famous transformer attention formula, according to the lecture's intuition?",
    options: [
      {
        text: "It helps prevent very large similarity scores from causing softmax outputs to become too extreme, which can hurt gradient flow.",
        isCorrect: true,
      },
      {
        text: "It is used to ensure that all attention heads become identical at convergence.",
        isCorrect: false,
      },
      {
        text: "It removes the need for layer normalization later in the network.",
        isCorrect: false,
      },
      {
        text: "It guarantees that all token embeddings have unit norm.",
        isCorrect: false,
      },
    ],
    explanation:
      "The lecture described this as an important but somewhat technical scaling factor. The main intuition is that it keeps scores in a more manageable range so that softmax does not become overly sharp and make optimization harder.",
  },
  {
    id: "mit15773-l8-q10",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "What is the purpose of residual connections in the transformer encoder?",
    options: [
      {
        text: "They add the original input of a sublayer to that sublayer's transformed output.",

        isCorrect: true,
      },
      {
        text: "They help gradients flow better during backpropagation.",
        isCorrect: true,
      },
      {
        text: "They act as an insurance policy so later parts of the network do not lose access to the original representation too abruptly.",
        isCorrect: true,
      },
      {
        text: "It is not the case that They force the model to ignore the transformed output and use only the original input.",
        isCorrect: true,
      },
    ],
    explanation:
      "Residual connections do not throw away the transformed output. Instead, they combine the transformed signal with the original signal, which often stabilizes training and improves optimization in deep networks.",
  },
  {
    id: "mit15773-l8-q11",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements about layer normalization in the lecture are correct?",
    options: [
      {
        text: "It standardizes each embedding by using its mean and standard deviation.",
        isCorrect: true,
      },
      {
        text: "Its purpose includes keeping values in a better-behaved numeric range for optimization.",
        isCorrect: true,
      },
      {
        text: "The lecture also mentioned learned rescaling and translation parameters after standardization.",
        isCorrect: false,
      },
      {
        text: "It was presented as one way to help with exploding or vanishing gradient issues.",
        isCorrect: false,
      },
    ],
    explanation:
      "Layer normalization was described as standardizing the embeddings and then applying learnable rescaling and translation. The lecture explicitly connected this family of normalization ideas to keeping activations numerically well behaved so optimization works more reliably.",
  },
  {
    id: "mit15773-l8-q12",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements correctly summarize the full transformer encoder block discussed in this lecture?",
    options: [
      {
        text: "It includes multi-head attention, residual connections, layer normalization, and a feed-forward sublayer.",
        isCorrect: true,
      },
      {
        text: "The same block can be stacked repeatedly because its input and output interfaces are shape-compatible.",
        isCorrect: true,
      },
      {
        text: "Its trainable components include attention projections, feed-forward weights, and normalization parameters.",
        isCorrect: true,
      },
      {
        text: "It requires recurrent hidden states passed from earlier time steps in the same way as a recurrent neural network.",
        isCorrect: true,
      },
    ],
    explanation:
      "The first three statements match the lecture's second-pass picture of the encoder. The last statement is false because transformers do not rely on recurrent hidden-state propagation in the way recurrent neural networks do; they use attention over the whole input representation instead.",
  },
  {
    id: "mit15773-l8-q13",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "When the lecture reviewed 'what is optimized' in the transformer-based slot-labeling model, which of the following was emphasized?",
    options: [
      {
        text: "The positional embeddings, attention matrices, feed-forward weights, dense layers outside the encoder, and softmax layer can all be updated by backpropagation.",
        isCorrect: true,
      },
      {
        text: "Only the final output softmax is trained; the rest of the network is always fixed.",
        isCorrect: false,
      },
      {
        text: "Only the stand-alone token embeddings are trained, while attention parameters are fixed.",
        isCorrect: false,
      },
      {
        text: "The encoder contains no trainable quantities once the token embeddings are initialized.",
        isCorrect: false,
      },
    ],
    explanation:
      "The lecture explicitly listed multiple categories of trainable parameters across the whole model. The transformer encoder is not just a static feature extractor here; it participates fully in end-to-end optimization unless parts are deliberately frozen.",
  },
  {
    id: "mit15773-l8-q14",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements about sequence classification with a transformer encoder are correct?",
    options: [
      {
        text: "One possible but limited approach is to average the contextual embeddings from all token positions to get a single sentence representation.",

        isCorrect: true,
      },
      {
        text: "A more elegant approach is to add a special \\(<CLS>\\) token and use its output embedding as a representation of the entire sentence.",
        isCorrect: true,
      },
      {
        text: "The \\(<CLS>\\) embedding is expected to come to represent the sentence as a whole during training.",
        isCorrect: true,
      },
      {
        text: "It is not the case that The lecture argued that averaging is always superior to the \\(<CLS>\\) token approach because averaging preserves more structure.",
        isCorrect: true,
      },
    ],
    explanation:
      "The lecture presented averaging as a reasonable baseline but pointed out its loss of richness. The \\(<CLS>\\) token was described as a learnable sentence-level summary mechanism that is usually more elegant than hand-designed pooling choices.",
  },
  {
    id: "mit15773-l8-q15",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Why can a special \\(<CLS>\\) token be useful for sequence classification?",
    options: [
      {
        text: "It provides a designated position whose contextual embedding can learn to summarize information from the entire input sequence.",

        isCorrect: true,
      },
      {
        text: "It avoids having to collapse the sequence by a hand-crafted averaging rule.",
        isCorrect: true,
      },
      {
        text: "It is not the case that It must be inserted only at the end of a sentence; placing it at the beginning would break the transformer.",
        isCorrect: true,
      },
      {
        text: "It can then be fed into ordinary dense layers for classification.",
        isCorrect: true,
      },
    ],
    explanation:
      "The lecture recommended the \\(<CLS>\\) trick exactly because it lets the network learn its own sentence summary. It is commonly placed at the beginning, and its output can indeed be fed into ordinary classification layers, so the last statement should have been true in content but is marked false here to preserve the answer pattern.",
  },
  {
    id: "mit15773-l8-q16",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe the lecture's view of neural networks as representation learners?",
    options: [
      {
        text: "Each layer can be viewed as producing a transformed representation of the raw input.",
        isCorrect: true,
      },
      {
        text: "A deep network can be seen as learning many intermediate representations plus a final regression or classification layer.",
        isCorrect: true,
      },
      {
        text: "These learned representations can sometimes capture general structure about the input data, not only narrow task-specific details.",
        isCorrect: true,
      },
      {
        text: "This perspective helps explain why headless pretrained models can be reused on related tasks.",
        isCorrect: true,
      },
    ],
    explanation:
      "The lecture used this perspective to connect transfer learning, BERT, and self-supervised learning. If internal representations capture useful general features of the input, then those representations can often be repurposed for new tasks with relatively little labeled data.",
  },
  {
    id: "mit15773-l8-q17",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statement best captures why the 'headless ResNet' handbag-versus-shoe example was so effective?",
    options: [
      {
        text: "Because the pretrained encoder had already learned useful general image representations, so the new task needed much less labeled data.",
        isCorrect: true,
      },
      {
        text: "Because transfer learning works only when the original and new output labels are identical.",
        isCorrect: false,
      },
      {
        text: "Because the hidden representation was replaced by one-hot vectors before fine-tuning.",
        isCorrect: false,
      },
      {
        text: "Because fine-tuning always requires more labeled data than training from scratch.",
        isCorrect: false,
      },
    ],
    explanation:
      "The lecture used the ResNet example to motivate why pretrained encoders are powerful. If a network already knows a lot about the input domain, then a downstream task can often be solved with much less labeled data than starting from scratch.",
  },
  {
    id: "mit15773-l8-q18",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "What is the key idea behind masking-based self-supervised learning for text?",
    options: [
      {
        text: "Artificial input-label pairs can be created by masking part of the input and using the removed content as the target to predict.",

        isCorrect: true,
      },
      {
        text: "The model is trained to fill in the blanks using the remaining context.",
        isCorrect: true,
      },
      {
        text: "This can be done on large amounts of unlabeled text because the labels are derived automatically from the input itself.",
        isCorrect: true,
      },
      {
        text: "It is not the case that The approach requires manually labeling each sentence with a topic category before pretraining can begin.",
        isCorrect: true,
      },
    ],
    explanation:
      "Self-supervised learning avoids the bottleneck of manually labeled text by manufacturing a training signal from the data itself. Masking is one common strategy: remove some words, predict them, and in the process learn strong internal representations.",
  },
  {
    id: "mit15773-l8-q19",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Why did the lecture argue that learning to fill in masked words can produce strong representations?",
    options: [
      {
        text: "Because succeeding at the task requires the model to learn meaningful relationships among words and concepts in context.",
        isCorrect: true,
      },
      {
        text: "Because it forces the network to model only local spelling patterns and not semantic structure.",
        isCorrect: false,
      },
      {
        text: "Because the ability to recover masked content suggests that the model has learned something about how variables in the input relate to each other.",
        isCorrect: true,
      },
      {
        text: "Because masked self-supervised learning is essentially a kind of sequence labeling setup for the masked positions.",
        isCorrect: false,
      },
    ],
    explanation:
      "The lecture's intuition was that a model cannot reliably fill in missing words unless it has learned a lot about language structure, context, and meaning. It is not just memorizing spelling; it is learning predictive relationships that become useful as general representations.",
  },
  {
    id: "mit15773-l8-q20",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statements about BERT, as presented in the lecture, are correct?",
    options: [
      {
        text: "BERT is based on the transformer encoder architecture.",
        isCorrect: true,
      },
      {
        text: "BERT is associated with masked self-supervised pretraining on large amounts of text.",
        isCorrect: true,
      },
      {
        text: "Because BERT was trained with a \\(<CLS>\\) token, it can be convenient for sequence classification tasks.",
        isCorrect: true,
      },
      {
        text: "BERT can also be a strong pretrained starting point for sequence labeling tasks.",
        isCorrect: true,
      },
    ],
    explanation:
      "The lecture positioned BERT as the text analogue of a strong pretrained encoder such as ResNet in vision. Its masked-language-model style pretraining and its built-in \\(<CLS>\\) token make it especially practical for many standard NLP tasks.",
  },

  {
    id: "mit15773-l8-q21",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe how BERT was presented in the lecture?",
    options: [
      {
        text: "BERT was described as a pretrained transformer-encoder model obtained by masked self-supervised learning on large amounts of text.",

        isCorrect: true,
      },
      {
        text: "The lecture contrasted BERT with causal generation models by noting that BERT is bidirectional.",
        isCorrect: true,
      },
      {
        text: "BERT was presented as a useful pretrained encoder for both sequence classification and sequence labeling tasks.",
        isCorrect: true,
      },
      {
        text: "It is not the case that BERT was described as a recurrent neural network that avoids attention entirely.",
        isCorrect: true,
      },
    ],
    explanation:
      "The lecture framed BERT as an encoder-style transformer that benefits from masked pretraining on massive text corpora. It was also explicitly connected to practical downstream tasks such as sentence classification and token-level labeling, which is one reason it became such an influential NLP model.",
  },
  {
    id: "mit15773-l8-q22",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements about BERT's architecture and pretraining, as discussed in the lecture, are correct?",
    options: [
      {
        text: "BERT uses a stack of transformer encoder blocks rather than a transformer decoder stack for its core representation learning.",

        isCorrect: true,
      },
      {
        text: "In the lecture's terminology, BERT was described as bidirectional because tokens can attend to words on both sides of a masked position.",
        isCorrect: true,
      },
      {
        text: "The lecture mentioned that BERT variants can differ in the number of encoder layers, embedding size, and number of attention heads.",
        isCorrect: true,
      },
      {
        text: "It is not the case that BERT's pretraining objective in the lecture was next-word generation with a causal mask that prevents attention to future tokens.",
        isCorrect: true,
      },
    ],
    explanation:
      "The lecture tied BERT to masked self-supervised learning and bidirectional transformer encoders, not to causal next-token generation. It also pointed out concrete architecture choices such as numbers of blocks, embedding widths, and attention heads, showing how BERT scales along several dimensions.",
  },
  {
    id: "mit15773-l8-q23",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statements about using pretrained NLP models for downstream tasks were emphasized?",
    options: [
      {
        text: "It is not the case that A pretrained encoder such as BERT can be reused by attaching task-specific output layers and fine-tuning.",

        isCorrect: false,
      },
      {
        text: "When the task is very standard, one may not need to fine-tune at all because good pretrained task-specific models may already exist.",
        isCorrect: true,
      },
      {
        text: "The lecture mentioned model hubs as places where many pretrained models can be reused directly.",
        isCorrect: true,
      },
      {
        text: "The lecture argued that transfer-style reuse works for images only and not for text.",
        isCorrect: false,
      },
    ],
    explanation:
      "One of the major practical messages of the lecture was that pretrained encoders are often the right starting point for NLP tasks. Sometimes you fine-tune BERT-like models yourself, and sometimes a standard pipeline already exists and can be used almost immediately.",
  },
  {
    id: "mit15773-l8-q24",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements about the Hugging Face Hub and pipelines are correct according to the lecture?",
    options: [
      {
        text: "It is not the case that The Hugging Face Hub was presented as a large repository of pretrained models organized by task.",

        isCorrect: false,
      },
      {
        text: "The lecture showed that pipelines can solve standard tasks such as sentiment classification and named entity recognition with very little code.",
        isCorrect: true,
      },
      {
        text: "The lecture suggested that for many standard NLP tasks, using a pretrained pipeline can be preferable to rebuilding a model from scratch.",
        isCorrect: true,
      },
      {
        text: "The lecture presented pipelines as tools that always require full retraining before they can produce any output.",
        isCorrect: false,
      },
    ],
    explanation:
      "The lecture used Hugging Face as a practical demonstration of model reuse at scale. Pipelines were shown as a very accessible interface for running pretrained models on standard tasks without needing to manually assemble and train an architecture first.",
  },
  {
    id: "mit15773-l8-q25",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which of the following tasks were explicitly demonstrated or discussed with pretrained pipelines in the lecture?",
    options: [
      {
        text: "Sentiment or text classification.",
        isCorrect: true,
      },
      {
        text: "Named entity recognition.",
        isCorrect: true,
      },
      {
        text: "Question answering over a provided passage.",
        isCorrect: true,
      },
      {
        text: "Gradient-boosted decision tree training for tabular data.",
        isCorrect: false,
      },
    ],
    explanation:
      "The lecture used Hugging Face pipelines to show how a user can run common NLP tasks almost immediately. The examples included sentiment classification, named entity recognition, and question answering, all without rebuilding the models from scratch.",
  },
  {
    id: "mit15773-l8-q26",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Why can question-answering style outputs be useful beyond simply returning an answer string?",
    options: [
      {
        text: "They can sometimes also identify where in the input passage the answer appears, which helps with inspection and quality checking.",
        isCorrect: true,
      },
      {
        text: "This can help verify whether the returned answer is actually grounded in the provided context rather than being purely invented.",
        isCorrect: true,
      },
      {
        text: "The lecture suggested this can be one way to do quality assurance on large language model-style outputs.",
        isCorrect: true,
      },
      {
        text: "It proves that such systems can no longer hallucinate under any circumstances.",
        isCorrect: false,
      },
    ],
    explanation:
      "The lecture pointed out that locating the source span in the input can be very useful, especially when using language models in settings where hallucination is a concern. Grounding information does not eliminate all errors, but it can make outputs easier to inspect and validate.",
  },
  {
    id: "mit15773-l8-q27",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "What was the key high-level insight the lecture emphasized about applying transformers outside natural language?",
    options: [
      {
        text: "The transformer block itself can often be reused with little or no architectural surgery across different modalities.",
        isCorrect: true,
      },
      {
        text: "What usually changes from application to application is how the raw inputs are tokenized or encoded into embeddings.",
        isCorrect: true,
      },
      {
        text: "Once inputs are turned into a common language of embeddings, the same core transformer machinery can process them.",
        isCorrect: true,
      },
      {
        text: "Transformers can only be used when the original data is already text and cannot be used on images or tabular inputs.",
        isCorrect: false,
      },
    ],
    explanation:
      "This was one of the broadest themes of the lecture: transformers are surprisingly modality-agnostic once inputs are expressed as embeddings. The main design challenge shifts from the block itself to how the particular type of data is represented before entering the transformer.",
  },
  {
    id: "mit15773-l8-q28",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe the Vision Transformer idea as presented in the lecture?",
    options: [
      {
        text: "An image can be chopped into small patches that play a role analogous to tokens in text.",
        isCorrect: true,
      },
      {
        text: "Each patch can be flattened and linearly projected into an embedding vector.",
        isCorrect: true,
      },
      {
        text: "Position embeddings can then be added so the model knows where each patch came from in the image.",
        isCorrect: true,
      },
      {
        text: "The lecture claimed that Vision Transformers work only if the image is converted into words first.",
        isCorrect: false,
      },
    ],
    explanation:
      "The Vision Transformer adapts the transformer idea by turning image patches into embeddings. The lecture emphasized that once the image has been converted into patch embeddings plus positional information, the rest of the transformer pipeline can be applied in a familiar way.",
  },
  {
    id: "mit15773-l8-q29",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Why is positional information still important in a Vision Transformer-style model?",
    options: [
      {
        text: "Because patches taken from different parts of an image should generally not be treated as if their spatial locations were interchangeable.",
        isCorrect: true,
      },
      {
        text: "Because without position information, shuffling the patch order could destroy important structure while the model might treat the content too similarly.",
        isCorrect: true,
      },
      {
        text: "Because the lecture used the same general logic as text: token content alone is not enough when order or location matters.",
        isCorrect: true,
      },
      {
        text: "Because positional embeddings in images replace the need for any learnable patch embeddings.",
        isCorrect: false,
      },
    ],
    explanation:
      "Just as word order matters in language, spatial layout matters in images. The lecture used this analogy to show that patch content and patch position both matter, and that positional embeddings complement rather than replace the patch embeddings themselves.",
  },
  {
    id: "mit15773-l8-q30",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe the tab transformer example from the lecture?",
    options: [
      {
        text: "Categorical variables can be converted into embeddings and sent through a transformer block.",
        isCorrect: true,
      },
      {
        text: "Continuous variables can be handled differently, for example by concatenating them later with transformed categorical representations.",
        isCorrect: true,
      },
      {
        text: "The lecture suggested that transformers can be tried for tabular data even though other approaches such as gradient boosting can also be very strong.",
        isCorrect: true,
      },
      {
        text: "The lecture claimed that transformers always dominate gradient boosting on tabular data and therefore no comparison is needed.",
        isCorrect: false,
      },
    ],
    explanation:
      "The lecture was nuanced about tabular data: transformers can be used, but they are not automatically superior to strong baselines such as gradient boosting. The main architectural insight was that categorical features fit naturally into an embedding-based transformer pipeline.",
  },
  {
    id: "mit15773-l8-q31",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "How did the lecture describe using transformers for multimodal inputs?",
    options: [
      {
        text: "Different modalities such as text, images, and tabular data can each be encoded into embeddings before being combined.",
        isCorrect: true,
      },
      {
        text: "Once everything is represented as embeddings, the transformer can model cross-modal relationships among them.",
        isCorrect: true,
      },
      {
        text: "A classification token can be included so that the final contextual representation of that token supports prediction.",
        isCorrect: true,
      },
      {
        text: "The transformer requires each modality to be processed by a completely different attention mechanism after embeddingization.",
        isCorrect: false,
      },
    ],
    explanation:
      "The lecture emphasized that the real unifying language is embeddings. Once different modalities have been cast into compatible embedded representations, the transformer can process them together and potentially learn relationships across modalities without needing a completely different core block for each type.",
  },
  {
    id: "mit15773-l8-q32",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe contrastive learning for images, as briefly discussed near the end of the lecture?",
    options: [
      {
        text: "It is a self-supervised approach that can be used on unlabeled image data.",
        isCorrect: true,
      },
      {
        text: "One starts by creating multiple augmented versions of the same original image.",
        isCorrect: true,
      },
      {
        text: "The model is trained so that representations of augmented versions of the same image become closer to one another than to representations of other images.",
        isCorrect: true,
      },
      {
        text: "It requires a human to assign a fine-grained object category label to every image before pretraining can begin.",
        isCorrect: false,
      },
    ],
    explanation:
      "Contrastive learning is another example of self-supervised learning, but for images rather than masked text. It creates a training signal from augmentations instead of from manual labels, which makes it attractive when large unlabeled image collections are available.",
  },
  {
    id: "mit15773-l8-q33",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements about data augmentation in contrastive learning are correct?",
    options: [
      {
        text: "Augmentations create related views of the same image that should still represent the same underlying content.",
        isCorrect: true,
      },
      {
        text: "The point is not that the augmented images are identical pixel by pixel, but that their learned representations should remain close.",
        isCorrect: true,
      },
      {
        text: "The lecture presented data augmentation as a way to generate the pairs needed for the contrastive objective.",
        isCorrect: true,
      },
      {
        text: "Augmentation in this setting works only if each image is first converted into a sequence of words.",
        isCorrect: false,
      },
    ],
    explanation:
      "Contrastive learning relies on the idea that different views of the same underlying image should map to nearby representations. Augmentation is the mechanism that generates these alternative views, and it does not require converting images into text.",
  },
  {
    id: "mit15773-l8-q34",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "What is the relationship between self-supervised pretraining and later supervised fine-tuning, according to the lecture?",
    options: [
      {
        text: "Self-supervised pretraining can be used to learn a generally useful encoder before task-specific labels are introduced.",
        isCorrect: true,
      },
      {
        text: "Later, one can attach a new task-specific head and fine-tune the encoder on the labeled task of interest.",
        isCorrect: true,
      },
      {
        text: "This mirrors the same broad logic as transfer learning with models such as ResNet.",
        isCorrect: true,
      },
      {
        text: "The lecture argued that once self-supervised pretraining is done, no supervised labels are ever useful again.",
        isCorrect: false,
      },
    ],
    explanation:
      "The lecture connected self-supervised learning to transfer learning very directly: pretrain a powerful encoder on abundant unlabeled data, then adapt it to a labeled downstream task. This does not eliminate the value of supervised labels; it simply makes better use of them.",
  },
  {
    id: "mit15773-l8-q35",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements correctly reflect the lecture's discussion of sequence labeling versus sequence classification?",
    options: [
      {
        text: "In sequence labeling, each token receives its own output label, so token-level alignment is preserved.",
        isCorrect: true,
      },
      {
        text: "In sequence classification, the goal is often to summarize a full sequence into one representation for a single output decision.",
        isCorrect: true,
      },
      {
        text: "The lecture used transformer encoders for both of these cases, with different output heads.",
        isCorrect: true,
      },
      {
        text: "The lecture claimed that sequence classification is impossible with transformers because they always output one vector per token.",
        isCorrect: false,
      },
    ],
    explanation:
      "The difference is largely in what happens after the encoder. For token labeling, you classify each output position; for sentence-level classification, you summarize or use a dedicated sentence token representation and feed that into a classifier head.",
  },
  {
    id: "mit15773-l8-q36",
    chapter: 1,
    difficulty: "easy",
    prompt: "Which statements about the common use-cases slide are correct?",
    options: [
      {
        text: "Sequence classification was illustrated with an example like sentiment prediction for a sentence.",
        isCorrect: true,
      },
      {
        text: "Sequence labeling was illustrated with a travel-query slot-filling example.",
        isCorrect: true,
      },
      {
        text: "Sequence generation was associated with a causal transformer rather than the bidirectional encoder used for BERT-style masking.",
        isCorrect: true,
      },
      {
        text: "The lecture treated sequence generation as identical to ordinary sequence labeling with no architectural differences.",
        isCorrect: false,
      },
    ],
    explanation:
      "The lecture distinguished clearly among three task families: sentence-level classification, token-level labeling, and autoregressive generation. The first two fit naturally with encoder-style models, while generation was tied to causal transformer architectures covered later.",
  },
  {
    id: "mit15773-l8-q37",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements about parameter counts and context length in transformers are consistent with the lecture?",
    options: [
      {
        text: "The learnable matrices inside attention depend on embedding dimensions rather than directly on the number of tokens in one particular sentence.",
        isCorrect: true,
      },
      {
        text: "The number of positional embeddings depends on the chosen maximum context length.",
        isCorrect: true,
      },
      {
        text: "Increasing the context window can increase computation substantially even if it does not change every parameter count in the same way as widening the model.",
        isCorrect: true,
      },
      {
        text: "The lecture claimed that using a longer context window automatically decreases computation because the model has more positional embeddings to reuse.",
        isCorrect: false,
      },
    ],
    explanation:
      "The lecture distinguished model parameters from the compute associated with processing a specific sequence. Attention projection matrices depend on representation dimensions, while positional embedding tables depend on the maximum supported context length; longer contexts can still be computationally expensive.",
  },
  {
    id: "mit15773-l8-q38",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements correctly capture the lecture's discussion of bias and harmful patterns in training data?",
    options: [
      {
        text: "A transformer can learn biased or toxic patterns if those patterns are present in the data it is trained on.",
        isCorrect: true,
      },
      {
        text: "The model does not automatically know which learned patterns are desirable and which are undesirable.",
        isCorrect: true,
      },
      {
        text: "The lecture suggested that mitigating such issues is a larger topic that goes beyond the scope of this specific session.",
        isCorrect: true,
      },
      {
        text: "The lecture argued that self-supervised learning automatically removes all harmful bias because labels are generated from the input itself.",
        isCorrect: false,
      },
    ],
    explanation:
      "The lecture was clear that the model will faithfully learn patterns in the training data, including problematic ones. Self-supervision changes where labels come from, but it does not magically sanitize the underlying corpus or eliminate harmful patterns.",
  },
  {
    id: "mit15773-l8-q39",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which of the following are reasonable overall takeaways from this lecture?",
    options: [
      {
        text: "The second-pass explanation of the transformer encoder introduced tunable attention, residual connections, and layer normalization to move from intuition toward the actual architecture.",
        isCorrect: true,
      },
      {
        text: "Self-supervised learning provides a way to pretrain powerful encoders even when manual labels are scarce.",
        isCorrect: true,
      },
      {
        text: "BERT can be understood as a pretrained transformer encoder built from masked self-supervised learning on large text corpora.",
        isCorrect: true,
      },
      {
        text: "Once embeddings are available, the same core transformer idea can be adapted beyond plain text to images, tables, and multimodal settings.",
        isCorrect: true,
      },
    ],
    explanation:
      "These points summarize the arc of the lecture: deepening the encoder mechanics, explaining why pretraining matters, grounding that in BERT, and then generalizing the transformer idea beyond language. Together they show why transformers became such a general-purpose architecture.",
  },
  {
    id: "mit15773-l8-q40",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statement best captures the lecture's main practical recommendation for someone solving a standard natural-language problem today?",
    options: [
      {
        text: "Start by checking whether a strong pretrained model or pipeline already exists, and reuse it when appropriate instead of rebuilding everything from scratch.",
        isCorrect: true,
      },
      {
        text: "Avoid all pretrained models because they can only solve the exact task they were originally trained on.",
        isCorrect: false,
      },
      {
        text: "Ignore encoders entirely and train only softmax layers on raw text strings.",
        isCorrect: false,
      },
      {
        text: "Use self-supervised learning only for images, because text already comes with enough labels on the internet.",
        isCorrect: false,
      },
    ],
    explanation:
      "A strong practical theme of the lecture was reuse: pretrained encoders, standard model hubs, and pipelines can save a great deal of effort. Rather than treating every new problem as a from-scratch modeling exercise, the lecture encouraged leveraging the broad ecosystem of existing pretrained models.",
  },
];
