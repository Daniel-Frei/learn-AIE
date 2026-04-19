import { Question } from "../../quiz";

export const TransformersSelfSupervisedLearningQuestions: Question[] = [
  {
    id: "mit15773-l8-q01",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Why can it be useful to introduce a transformer encoder first with a simplified intuition and then with a more detailed formulation?",
    options: [
      {
        text: "Because an initial simplified view can help build intuition before introducing architectural details like residual connections and normalization.",
        isCorrect: true,
      },
      {
        text: "Because transformers cannot be understood unless they are replaced by recurrent neural networks.",
        isCorrect: false,
      },
      {
        text: "Because positional embeddings are unnecessary and can be removed in more detailed formulations.",
        isCorrect: false,
      },
      {
        text: "Because the architecture only works when described without mathematical details.",
        isCorrect: false,
      },
    ],
    explanation:
      "A simplified view helps build intuition about how self-attention works before introducing additional components like residual connections and layer normalization. The incorrect options either misunderstand the role of transformers or suggest removing essential components such as positional embeddings, which are required for handling order.",
  },
  {
    id: "mit15773-l8-q02",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe positional input embeddings in transformers?",
    options: [
      {
        text: "They are formed by adding token embeddings and positional embeddings elementwise.",
        isCorrect: true,
      },
      {
        text: "Token embeddings can be pretrained or randomly initialized.",
        isCorrect: true,
      },
      {
        text: "Positional embeddings must always be fixed rather than trainable vectors.",
        isCorrect: false,
      },
      {
        text: "They are created by averaging all token embeddings into a single vector before entering the encoder.",
        isCorrect: false,
      },
    ],
    explanation:
      "Each token receives its own embedding that combines token and positional information, preserving sequence structure. Token embeddings can be pretrained or learned from scratch, and positional embeddings can also be trainable. Averaging all token embeddings into one vector would destroy token-level information and is not how transformers operate.",
  },
  {
    id: "mit15773-l8-q03",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Why is it useful to express self-attention using matrix operations?",
    options: [
      {
        text: "It allows efficient computation of all pairwise similarities using matrix multiplication.",
        isCorrect: true,
      },
      {
        text: "It enables efficient parallel computation on hardware such as GPUs.",
        isCorrect: true,
      },
      {
        text: "It removes the need for the softmax function in attention.",
        isCorrect: false,
      },
      {
        text: "It guarantees that attention cost grows only linearly with sequence length.",
        isCorrect: false,
      },
    ],
    explanation:
      "Matrix operations allow computing all pairwise dot products in one step, which is critical for efficiency. This makes GPU acceleration practical and is part of why transformers scale well in practice. Softmax is still needed, and standard self-attention does not scale linearly with sequence length.",
  },
  {
    id: "mit15773-l8-q04",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "In a matrix view of simplified self-attention, which statements are correct?",
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
      "If several heads all perform the exact same non-tunable operation on the same inputs, they do not meaningfully specialize. Learnable projections are what allow different heads to develop different behaviors from the same input sequence.",
  },
  {
    id: "mit15773-l8-q06",
    chapter: 1,
    difficulty: "medium",
    prompt: "How is self-attention made learnable in transformer models?",
    options: [
      {
        text: "By introducing learnable linear projection matrices applied to the input embeddings.",
        isCorrect: true,
      },
      {
        text: "By optimizing these matrices using backpropagation.",
        isCorrect: true,
      },
      {
        text: "By forcing all attention heads to learn the same transformation.",
        isCorrect: false,
      },
      {
        text: "By replacing embeddings with fixed one-hot vectors inside the attention layer.",
        isCorrect: false,
      },
    ],
    explanation:
      "Learnable projection matrices for queries, keys, and values introduce parameters into the attention mechanism, making it trainable. These matrices are optimized with backpropagation. Different heads are useful precisely because they can learn different transformations, not the same one.",
  },
  {
    id: "mit15773-l8-q07",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements about the matrices \\(A^K\\), \\(A^Q\\), and \\(A^V\\) are correct?",
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
      "Why is the factor \\(\\sqrt{d_k}\\) often included in the transformer attention formula?",
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
      "What are the purposes of residual connections in deep neural networks such as transformers?",
    options: [
      {
        text: "They add the input of a layer to its output.",
        isCorrect: true,
      },
      {
        text: "They help gradients flow more effectively during training.",
        isCorrect: true,
      },
      {
        text: "They help preserve access to earlier representations.",
        isCorrect: true,
      },
      {
        text: "They combine original and transformed information rather than discarding the transformed output.",
        isCorrect: true,
      },
    ],
    explanation:
      "Residual connections add a layer's input back to its transformed output. This helps gradient flow, preserves access to earlier information, and lets the network retain original representations while also benefiting from learned transformations.",
  },
  {
    id: "mit15773-l8-q11",
    chapter: 1,
    difficulty: "medium",
    prompt: "Which statements about layer normalization are correct?",
    options: [
      {
        text: "It standardizes embeddings using their mean and standard deviation.",
        isCorrect: true,
      },
      {
        text: "It helps keep activations in a numerically stable range.",
        isCorrect: true,
      },
      {
        text: "It includes learnable scaling and shifting parameters after normalization.",
        isCorrect: true,
      },
      {
        text: "It helps mitigate issues such as exploding or vanishing gradients.",
        isCorrect: true,
      },
    ],
    explanation:
      "Layer normalization standardizes activations and then applies learnable scaling and shifting. This helps maintain stable numerical ranges and supports more effective gradient flow during training.",
  },
  {
    id: "mit15773-l8-q12",
    chapter: 1,
    difficulty: "hard",
    prompt: "Which statements correctly describe a transformer encoder block?",
    options: [
      {
        text: "It includes multi-head attention, residual connections, layer normalization, and a feed-forward sublayer.",
        isCorrect: true,
      },
      {
        text: "It can be stacked multiple times because input and output dimensions match.",
        isCorrect: true,
      },
      {
        text: "Its trainable parameters are limited to the final output layer outside the encoder.",
        isCorrect: false,
      },
      {
        text: "It relies on recurrent hidden states passed across time steps.",
        isCorrect: false,
      },
    ],
    explanation:
      "A transformer encoder block contains multi-head attention, residual connections, layer normalization, and a feed-forward layer, and its shape compatibility makes stacking possible. The encoder itself has many trainable parameters inside it, and it does not rely on recurrent hidden states like a Recurrent Neural Network (RNN).",
  },
  {
    id: "mit15773-l8-q13",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which parts of a transformer-based slot-labeling model can be updated by backpropagation?",
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
      "Which statements about sequence classification with transformer encoders are correct?",
    options: [
      {
        text: "A simple approach is to average contextual embeddings across tokens.",
        isCorrect: true,
      },
      {
        text: "A common approach is to use a special \\(<CLS>\\) token representation.",
        isCorrect: true,
      },
      {
        text: "The \\(<CLS>\\) embedding can learn to represent the entire sequence.",
        isCorrect: true,
      },
      {
        text: "Averaging contextual embeddings can lose information compared with a learned sequence representation.",
        isCorrect: true,
      },
    ],
    explanation:
      "Averaging token embeddings is a simple baseline, but it can lose some of the richer structure present in the sequence. A special \\(<CLS>\\) token provides a learned sequence-level representation, which is one reason it is widely used for sequence classification.",
  },
  {
    id: "mit15773-l8-q15",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Why is a special \\(<CLS>\\) token useful for sequence classification?",
    options: [
      {
        text: "It provides a dedicated representation that can aggregate information from the entire sequence.",
        isCorrect: true,
      },
      {
        text: "It avoids the need for manually designed pooling methods such as averaging.",
        isCorrect: true,
      },
      {
        text: "It must be placed at the beginning of the sequence; placing it elsewhere would fundamentally break the model.",
        isCorrect: false,
      },
      {
        text: "It cannot be used with standard dense layers for classification.",
        isCorrect: false,
      },
    ],
    explanation:
      "The \\(<CLS>\\) token gives the model a dedicated representation for sequence-level prediction and avoids relying on ad hoc pooling choices. It can be fed into ordinary dense classification layers. Putting it at the beginning is common and convenient, but other positions do not fundamentally break the model.",
  },
  {
    id: "mit15773-l8-q16",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe neural networks as representation learners?",
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
      "Which statement best explains why a headless pretrained convolutional network can be effective for a new image-classification task such as handbag-versus-shoe classification?",
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
      "What is the key idea behind masking-based self-supervised learning?",
    options: [
      {
        text: "Parts of the input are hidden and used as targets for prediction.",
        isCorrect: true,
      },
      {
        text: "The model learns to reconstruct missing information from context.",
        isCorrect: true,
      },
      {
        text: "It allows training on large unlabeled datasets.",
        isCorrect: true,
      },
      {
        text: "It creates training targets without requiring manual labels for each example.",
        isCorrect: true,
      },
    ],
    explanation:
      "Masking-based self-supervised learning removes part of the input and asks the model to predict it from context. This produces training targets automatically, which makes it possible to train on large unlabeled corpora without manually labeling each example.",
  },
  {
    id: "mit15773-l8-q19",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Why can learning to fill in masked words produce strong representations?",
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
    prompt: "Which statements about BERT are correct?",
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
    prompt: "Which statements correctly describe BERT?",
    options: [
      {
        text: "It is a pretrained transformer encoder trained using masked self-supervised learning.",
        isCorrect: true,
      },
      {
        text: "It uses bidirectional context when processing text.",
        isCorrect: true,
      },
      {
        text: "It can be used for sequence classification but not sequence labeling tasks.",
        isCorrect: false,
      },
      {
        text: "It is based on recurrent neural networks without attention.",
        isCorrect: false,
      },
    ],
    explanation:
      "BERT is a pretrained transformer encoder built with masked self-supervised learning and bidirectional attention. It can be used for both sequence classification and sequence labeling, and it is not based on recurrent architectures.",
  },
  {
    id: "mit15773-l8-q22",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements about BERT's architecture and pretraining are correct?",
    options: [
      {
        text: "BERT uses a stack of transformer encoder blocks rather than a transformer decoder stack for its core representation learning.",
        isCorrect: true,
      },
      {
        text: "BERT is bidirectional in the sense that tokens can attend to words on both sides of a masked position.",
        isCorrect: true,
      },
      {
        text: "BERT variants can differ in the number of encoder layers, embedding size, and number of attention heads.",
        isCorrect: true,
      },
      {
        text: "BERT uses masked-token prediction rather than causal next-word prediction as its core pretraining objective.",
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
      "Which statements about using pretrained NLP models for downstream tasks are correct?",
    options: [
      {
        text: "A pretrained encoder such as BERT cannot be reused by attaching task-specific output layers and fine-tuning.",
        isCorrect: false,
      },
      {
        text: "When the task is very standard, one may not need to fine-tune at all because strong pretrained task-specific models may already exist.",
        isCorrect: true,
      },
      {
        text: "Model hubs can make many pretrained NLP models directly reusable.",
        isCorrect: true,
      },
      {
        text: "Transfer-style reuse works for images only and not for text.",
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
      "Which statements about the Hugging Face Hub and pipelines are correct?",
    options: [
      {
        text: "The Hugging Face Hub contains only a small number of pretrained models and is not organized by task.",
        isCorrect: false,
      },
      {
        text: "Pipelines can solve standard tasks such as sentiment classification and named entity recognition with very little code.",
        isCorrect: true,
      },
      {
        text: "For many standard NLP tasks, using a pretrained pipeline can be preferable to rebuilding a model from scratch.",
        isCorrect: true,
      },
      {
        text: "Pipelines always require full retraining before they can produce any output.",
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
      "Which of the following tasks are common examples of what pretrained NLP pipelines can do out of the box?",
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
        text: "Providing an answer span can be useful for quality assurance when evaluating language-model-style outputs.",
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
      "What is the key high-level insight about applying transformers outside natural language?",
    options: [
      {
        text: "The transformer block itself usually has to be redesigned substantially for each new modality.",
        isCorrect: false,
      },
      {
        text: "What usually changes from application to application is the internal attention formula rather than the input encoding.",
        isCorrect: false,
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
      "The central idea is that transformers are relatively modality-agnostic once the inputs are represented as embeddings. What usually changes across applications is how the raw inputs are encoded into embeddings, not the fundamental transformer block itself.",
  },
  {
    id: "mit15773-l8-q28",
    chapter: 1,
    difficulty: "medium",
    prompt: "Which statements correctly describe the Vision Transformer idea?",
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
        text: "Because patch content alone is not enough when spatial order or location matters.",
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
      "Which statements correctly describe a transformer-based approach to tabular data?",
    options: [
      {
        text: "Categorical variables can be converted into embeddings and sent through a transformer block.",
        isCorrect: true,
      },
      {
        text: "Continuous variables must also be converted into token embeddings and cannot be handled separately later.",
        isCorrect: false,
      },
      {
        text: "Transformers remove the need to compare against strong tabular baselines such as gradient boosting.",
        isCorrect: false,
      },
      {
        text: "Transformers always dominate gradient boosting on tabular data and therefore no comparison is needed.",
        isCorrect: false,
      },
    ],
    explanation:
      "Categorical variables fit naturally into an embedding-based transformer pipeline. Continuous variables can be handled differently, such as by concatenating them later. Strong baselines like gradient boosting still matter for tabular problems, so comparison remains important.",
  },
  {
    id: "mit15773-l8-q31",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe how transformers can be used for multimodal inputs?",
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
      "Which statements correctly describe contrastive learning for images?",
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
        text: "Data augmentation can be used to generate the positive pairs needed for the contrastive objective.",
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
      "What is the relationship between self-supervised pretraining and later supervised fine-tuning?",
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
      "Which statements correctly distinguish sequence labeling from sequence classification?",
    options: [
      {
        text: "In sequence labeling, each token receives its own output label, so token-level alignment is preserved.",
        isCorrect: true,
      },
      {
        text: "In sequence classification, the model must output one label for every token before making a sequence-level decision.",
        isCorrect: false,
      },
      {
        text: "Sequence labeling and sequence classification generally use the same type of output head.",
        isCorrect: false,
      },
      {
        text: "Sequence classification is impossible with transformers because they always output one vector per token.",
        isCorrect: false,
      },
    ],
    explanation:
      "Sequence labeling keeps token-level alignment by predicting a label for each token. Sequence classification instead summarizes the sequence into a representation used for a single decision. These tasks typically use different output heads, and transformers can handle both.",
  },
  {
    id: "mit15773-l8-q36",
    chapter: 1,
    difficulty: "easy",
    prompt: "Which statements about common transformer use cases are correct?",
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
      "Which statements about parameter counts and context length in transformers are correct?",
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
      "Which statements correctly describe bias and harmful patterns in training data?",
    options: [
      {
        text: "A transformer can learn biased or toxic patterns if those patterns are present in the data it is trained on.",
        isCorrect: true,
      },
      {
        text: "A model automatically separates desirable from undesirable patterns as long as it is trained on enough data.",
        isCorrect: false,
      },
      {
        text: "Bias and toxicity are fully solved once labels are generated from the input itself.",
        isCorrect: false,
      },
      {
        text: "Self-supervised learning automatically removes all harmful bias because labels are generated from the input itself.",
        isCorrect: false,
      },
    ],
    explanation:
      "A model can absorb harmful patterns that are present in its training data. Generating labels from the input does not automatically solve bias or toxicity problems, because the underlying data can still contain undesirable patterns.",
  },
  {
    id: "mit15773-l8-q39",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which of the following are reasonable overall takeaways about transformers, self-supervised learning, and BERT?",
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
      "Which statement best captures a practical recommendation for solving a standard natural-language problem today?",
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
