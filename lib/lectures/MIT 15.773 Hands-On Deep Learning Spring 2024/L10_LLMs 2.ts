import { Question } from "../../quiz";

export const L10LLMs2Questions: Question[] = [
  {
    id: "mit15773-l10-q01",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe the output side of a causal large language model such as GPT-3?",
    options: [
      {
        text: "The contextual embeddings can flow into a Dense layer with linear activations before the Softmax.",
        isCorrect: true,
      },
      {
        text: "The vector produced right before the Softmax must have length equal to the vocabulary size.",
        isCorrect: true,
      },
      {
        text: "A linear projection to vocabulary-sized logits is the key requirement before applying the Softmax.",
        isCorrect: true,
      },
      {
        text: "The final hidden representation is typically mapped through a learned linear projection so the model produces one logit per vocabulary item.",
        isCorrect: true,
      },
    ],
    explanation:
      "A causal language model must produce one score for each token in the vocabulary before applying Softmax. That is why a linear projection from the hidden state to vocabulary-sized logits is required, and all four statements describe that same basic output idea in compatible ways.",
  },
  {
    id: "mit15773-l10-q02",
    chapter: 1,
    difficulty: "easy",
    prompt: "Which statements correctly compare GPT, GPT-2, and GPT-3?",
    options: [
      {
        text: "They were all trained in a similar next-word-prediction fashion.",
        isCorrect: true,
      },
      {
        text: "GPT-3 differed mainly by scale, including both more data and a much larger neural network.",
        isCorrect: true,
      },
      {
        text: "GPT-3 showed a more dramatic jump in behavior than one might expect from a merely incremental improvement.",
        isCorrect: true,
      },
      {
        text: "GPT, GPT-2, and GPT-3 all stayed within the same general autoregressive transformer family rather than switching to a ChatGPT-style RLHF recipe at the GPT-2 stage.",
        isCorrect: true,
      },
    ],
    explanation:
      "GPT, GPT-2, and GPT-3 all shared the same core idea of autoregressive next-token prediction with transformer-based models. The big difference was scale, and GPT-3’s larger size and training data led to behavior that felt much more capable than a small step forward.",
  },
  {
    id: "mit15773-l10-q03",
    chapter: 1,
    difficulty: "medium",
    prompt: "Why was GPT-3 seen as much more impressive than GPT or GPT-2?",
    options: [
      {
        text: "It could generate much more coherent continuations from a starting prompt.",
        isCorrect: true,
      },
      {
        text: "It sometimes produced continuations that resembled a target writing style surprisingly well.",
        isCorrect: true,
      },
      {
        text: "Its improvement was often described as a qualitative jump rather than only a small quantitative gain.",
        isCorrect: true,
      },
      {
        text: "It showed much stronger in-context and few-shot behavior than earlier GPT models, which made it feel more generally useful.",
        isCorrect: true,
      },
    ],
    explanation:
      "GPT-3 impressed people because it did not just get a little better at text generation. Its outputs were often more coherent, more adaptable to prompts, and more stylistically convincing, so the improvement felt like a real jump in capability.",
  },
  {
    id: "mit15773-l10-q04",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe why GPT-3 often produced unhelpful answers before instruction tuning?",
    options: [
      {
        text: "It was trained to predict the next word, not explicitly to follow user instructions.",
        isCorrect: true,
      },
      {
        text: "It often treated prompts as if they were the beginning of patterns or lists that should be continued.",
        isCorrect: true,
      },
      {
        text: "It could give irrelevant, offensive, or factually incorrect output.",
        isCorrect: true,
      },
      {
        text: "Its base objective rewarded plausible continuation, not reliable helpfulness, honesty, or safe behavior toward a user.",
        isCorrect: true,
      },
    ],
    explanation:
      "Before instruction tuning, GPT-3 was mainly a powerful pattern completer. That means it often continued text in a plausible way, but it was not directly optimized to be helpful, truthful, or safe when responding to a user request.",
  },
  {
    id: "mit15773-l10-q05",
    chapter: 1,
    difficulty: "medium",
    prompt: "Which statements correctly describe Step 1 of instruction tuning?",
    options: [
      {
        text: "Humans wrote high-quality answers to thousands of instructions.",
        isCorrect: true,
      },
      {
        text: "These instruction-answer pairs were then used for supervised fine-tuning of GPT-3.",
        isCorrect: true,
      },
      {
        text: "The training still used the same basic shifted next-word-prediction setup used in causal language modeling.",
        isCorrect: true,
      },
      {
        text: "This step required labeled answers rather than being entirely unsupervised.",
        isCorrect: true,
      },
    ],
    explanation:
      "Instruction tuning starts by collecting human-written examples of good instruction following and then further training the model on those examples. The mechanics still rely on next-token prediction over carefully constructed instruction-answer sequences.",
  },
  {
    id: "mit15773-l10-q06",
    chapter: 1,
    difficulty: "medium",
    prompt: "Which statements about supervised fine-tuning (SFT) are correct?",
    options: [
      {
        text: "SFT improved GPT-3's ability to follow instructions.",
        isCorrect: true,
      },
      {
        text: "SFT used human-written question-answer pairs as training data.",
        isCorrect: true,
      },
      {
        text: "Doing much more SFT is limited partly by the cost of writing many high-quality answers.",
        isCorrect: true,
      },
      {
        text: "SFT was an important alignment step even though later preference-based stages could still improve the model further.",
        isCorrect: true,
      },
    ],
    explanation:
      "SFT helps a base model follow instructions by training it on human-written examples of good responses. It works well, but creating many high-quality examples is expensive, which is one reason later preference-based methods are also useful.",
  },
  {
    id: "mit15773-l10-q07",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "What is easier than writing many high-quality answers from scratch?",
    options: [
      {
        text: "Ranking answers written by somebody else.",
        isCorrect: true,
      },
      {
        text: "Generating multiple candidate answers with GPT-3 and asking humans to compare them.",
        isCorrect: true,
      },
      {
        text: "Using stochastic sampling so GPT-3 can produce several different answers to the same instruction.",
        isCorrect: true,
      },
      {
        text: "Collecting pairwise preference judgments is often easier and cheaper than asking humans to author ideal answers from scratch.",
        isCorrect: true,
      },
    ],
    explanation:
      "A central idea in preference-based alignment is that ranking answers is often easier than writing perfect answers yourself. Once a model can generate multiple candidates, humans can compare them relatively quickly, which makes data collection more scalable.",
  },
  {
    id: "mit15773-l10-q08",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe a reward model used in reinforcement learning from human feedback?",
    options: [
      {
        text: "It takes an instruction and an answer and outputs a single numerical score indicating answer quality for that instruction.",
        isCorrect: true,
      },
      {
        text: "During training, one copy can evaluate a preferred answer while another copy evaluates a less preferred answer for the same instruction.",
        isCorrect: true,
      },
      {
        text: "The loss is designed so that the preferred answer should receive a higher score than the other answer.",
        isCorrect: true,
      },
      {
        text: "The reward model is trained on preference comparisons so that preferred answers get higher scores than less preferred ones.",
        isCorrect: true,
      },
    ],
    explanation:
      "A reward model is a learned judge for instruction-answer pairs. It is trained from human preference data so that better answers receive higher scalar scores, and those scores can later guide reinforcement learning on the language model.",
  },
  {
    id: "mit15773-l10-q09",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Consider the reward-model loss \\(L = -\\log(\\sigma(r_P - r_O))\\), where \\(r_P\\) is the score for the preferred answer and \\(r_O\\) is the score for the other answer. Which statements are correct?",
    options: [
      {
        text: "If \\(r_P - r_O\\) becomes larger, \\(\\sigma(r_P - r_O)\\) increases and the loss decreases.",
        isCorrect: true,
      },
      {
        text: "The loss encourages the preferred answer to receive a higher rating than the non-preferred answer.",
        isCorrect: true,
      },
      {
        text: "If \\(r_P = r_O\\), then \\(\\sigma(r_P - r_O) = \\sigma(0) = 0.5\\).",
        isCorrect: true,
      },
      {
        text: "The loss is minimized when \\(r_P\\) is much larger than \\(r_O\\).",
        isCorrect: true,
      },
    ],
    explanation:
      "This loss depends on the difference between the preferred score and the other score. It gets smaller when the preferred answer is scored higher, and if both scores are equal then the sigmoid sees zero and outputs 0.5.",
  },
  {
    id: "mit15773-l10-q10",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe the role of the reward model after it has been trained?",
    options: [
      {
        text: "It can provide a numerical rating for any instruction-answer pair.",
        isCorrect: true,
      },
      {
        text: "It can be viewed as a learned approximation to how humans rank responses.",
        isCorrect: true,
      },
      {
        text: "Its score can be used as a signal to nudge GPT-3 using reinforcement learning.",
        isCorrect: true,
      },
      {
        text: "It serves as a scoring function for training the chatbot rather than as the model that directly generates user-facing answers.",
        isCorrect: true,
      },
    ],
    explanation:
      "After training, the reward model acts like a learned evaluator of response quality. It does not replace the language model as the chatbot, but it provides the training signal that helps push the chatbot toward responses people prefer.",
  },
  {
    id: "mit15773-l10-q11",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe reinforcement learning from human feedback (RLHF)?",
    options: [
      {
        text: "The term refers to using human preference data to train a reward model and then using reinforcement learning to improve the language model.",
        isCorrect: true,
      },
      {
        text: "Proximal Policy Optimization can be used as the reinforcement learning algorithm in this stage.",
        isCorrect: true,
      },
      {
        text: "The rating from the reward model is used to nudge the language model in a preferred direction.",
        isCorrect: true,
      },
      {
        text: "RLHF is an alignment pipeline that uses learned reward signals rather than manually editing weights for each individual bad answer.",
        isCorrect: true,
      },
    ],
    explanation:
      "RLHF combines human preference data, a trained reward model, and a reinforcement learning algorithm such as PPO. The key idea is to move the language model toward answers humans prefer by optimizing against a learned reward signal, not by hand-editing weights example by example.",
  },
  {
    id: "mit15773-l10-q12",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe the path from GPT-3 to GPT-3.5/InstructGPT and then to ChatGPT?",
    options: [
      {
        text: "GPT-3.5/InstructGPT was built by adapting GPT-3 to follow instructions better.",
        isCorrect: true,
      },
      {
        text: "ChatGPT followed a similar playbook but used conversations as training data rather than isolated instruction-answer pairs.",
        isCorrect: true,
      },
      {
        text: "Because ChatGPT was trained on conversations, it can handle follow-up questions more naturally.",
        isCorrect: true,
      },
      {
        text: "ChatGPT remained part of the same general GPT-style transformer lineage, but it was adapted more strongly for multi-turn dialogue.",
        isCorrect: true,
      },
    ],
    explanation:
      "InstructGPT improved GPT-3 by making it follow instructions better, and ChatGPT extended that idea to multi-turn conversations. The underlying family remained GPT-style transformers, but the training data and alignment process made the model far more conversational.",
  },
  {
    id: "mit15773-l10-q13",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements about adaptation strategies from the lecture's 'ladder of LLM adaptation' are correct?",
    options: [
      {
        text: "Zero-shot prompting means giving a clear instruction without examples.",
        isCorrect: true,
      },
      {
        text: "Few-shot prompting means including a few examples in the prompt to show the desired pattern.",
        isCorrect: true,
      },
      {
        text: "Retrieval-augmented generation adds relevant chunks of external text to the prompt.",
        isCorrect: true,
      },
      {
        text: "Fine-tuning differs from the other three because it actually changes the model weights.",
        isCorrect: true,
      },
    ],
    explanation:
      "The lecture organized LLM adaptation as a ladder from least invasive to most invasive approaches. Prompting methods leave the model unchanged, retrieval augments the prompt with context, and fine-tuning explicitly updates the model parameters.",
  },
  {
    id: "mit15773-l10-q14",
    chapter: 1,
    difficulty: "easy",
    prompt: "Which statements correctly describe zero-shot prompting?",
    options: [
      {
        text: "It can work surprisingly well on some tasks by simply instructing the LLM clearly.",
        isCorrect: true,
      },
      {
        text: "A product-review defect-detection task is an example of something that can work in zero-shot form.",
        isCorrect: true,
      },
      {
        text: "Prompt design can matter a lot for zero-shot performance.",
        isCorrect: true,
      },
      {
        text: "Zero-shot prompting uses the pretrained model directly at inference time without additional task-specific gradient updates.",
        isCorrect: true,
      },
    ],
    explanation:
      "Zero-shot prompting means you give the model a task description but no worked examples. It can still work well because the pretrained model already knows a lot, but the wording of the prompt often matters a great deal.",
  },
  {
    id: "mit15773-l10-q15",
    chapter: 1,
    difficulty: "medium",
    prompt: "Which statements about prompt engineering are correct?",
    options: [
      {
        text: "Prompt engineering always means changing the model weights through gradient-based training.",
        isCorrect: false,
      },
      {
        text: "Breaking a task into explicit intermediate steps can improve reliability on some problems.",
        isCorrect: true,
      },
      {
        text: "First listing the words in a sentence before identifying the fifth word is an example of a useful prompt-engineering trick.",
        isCorrect: true,
      },
      {
        text: "Prompt engineering is unnecessary once a model has undergone instruction tuning or RLHF.",
        isCorrect: false,
      },
    ],
    explanation:
      "Prompt engineering means designing the input text so the model is more likely to behave in the way you want, not changing the model weights. It still matters even for instruction-tuned models, and step-by-step structure can make some tasks much more reliable.",
  },
  {
    id: "mit15773-l10-q16",
    chapter: 1,
    difficulty: "medium",
    prompt: "Which statements correctly describe few-shot prompting?",
    options: [
      {
        text: "It requires updating the model weights on a small labeled dataset before inference.",
        isCorrect: false,
      },
      {
        text: "A grammar-correction pattern with Poor English and Good English examples is a typical few-shot setup.",
        isCorrect: true,
      },
      {
        text: "It is an example of in-context learning because the model infers the task from examples in the prompt without changing its weights.",
        isCorrect: true,
      },
      {
        text: "Few-shot prompting is identical to fine-tuning because both methods update the underlying model parameters using stochastic gradient descent.",
        isCorrect: false,
      },
    ],
    explanation:
      "Few-shot prompting teaches the model through examples placed inside the prompt itself. That is different from fine-tuning, because the model weights stay fixed and the model learns the task pattern only from context at inference time.",
  },
  {
    id: "mit15773-l10-q17",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe retrieval-augmented generation (RAG)?",
    options: [
      {
        text: "RAG mainly works by permanently changing model weights to bake in new private knowledge.",
        isCorrect: false,
      },
      {
        text: "Because of the context window, we usually cannot pack an entire large database into the prompt.",
        isCorrect: true,
      },
      {
        text: "A common strategy is to retrieve the most relevant chunks from an external corpus and include only those chunks in the prompt.",
        isCorrect: true,
      },
      {
        text: "RAG necessarily fine-tunes the LLM weights every time a new question is asked.",
        isCorrect: false,
      },
    ],
    explanation:
      "RAG works by retrieving useful external information and placing it into the prompt, not by changing the model weights for each query. It is helpful because context windows are limited, so the system must select only the most relevant pieces of outside information.",
  },
  {
    id: "mit15773-l10-q18",
    chapter: 1,
    difficulty: "easy",
    prompt: "Which statements about the context window are correct?",
    options: [
      {
        text: "Only the input prompt counts toward the context window, while generated output does not.",
        isCorrect: false,
      },
      {
        text: "In a conversation, the running conversation history is typically part of the prompt for later turns.",
        isCorrect: true,
      },
      {
        text: "A limited context window is one reason we cannot simply include an entire large document repository in every prompt.",
        isCorrect: true,
      },
      {
        text: "The context window matters only during training and has no effect on inference or chatbot use.",
        isCorrect: false,
      },
    ],
    explanation:
      "The context window limits the total amount of text the model can handle in one interaction, including both the input and the generated output. That is why chat history and retrieved documents use up space, and why the limit matters during real chatbot use, not just during training.",
  },
  {
    id: "mit15773-l10-q19",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe a typical retrieval pipeline for RAG?",
    options: [
      {
        text: "A typical RAG system relies only on exact keyword overlap and does not embed chunks into vectors.",
        isCorrect: false,
      },
      {
        text: "A new user question can also be embedded, and cosine similarity can then be used to retrieve related chunks.",
        isCorrect: true,
      },
      {
        text: "The most relevant retrieved chunks are packed into the prompt along with the question.",
        isCorrect: true,
      },
      {
        text: "This retrieval method requires every chunk to be manually labeled with categories before retrieval can work.",
        isCorrect: false,
      },
    ],
    explanation:
      "A standard RAG pipeline usually embeds both document chunks and the user query into vector representations. Retrieval then compares those vectors, often with cosine similarity, and the best chunks are added to the prompt without requiring manual labels on every chunk.",
  },
  {
    id: "mit15773-l10-q20",
    chapter: 1,
    difficulty: "hard",
    prompt: "Which statements correctly describe fine-tuning and LoRA?",
    options: [
      {
        text: "Standard fine-tuning leaves the pretrained weights frozen and changes only the prompt text at inference time.",
        isCorrect: false,
      },
      {
        text: "For very large models, naive fine-tuning can be expensive in both memory and compute because parameters, gradients, and optimizer state all consume memory.",
        isCorrect: true,
      },
      {
        text: "LoRA keeps the base model frozen and learns small low-rank update matrices instead of updating the full attention matrices directly.",
        isCorrect: true,
      },
      {
        text: "LoRA works by increasing the rank of every attention matrix as much as possible so that more memory is used during optimization.",
        isCorrect: false,
      },
    ],
    explanation:
      "Standard fine-tuning changes trainable model weights using gradient-based learning, so it is not the same as prompt-only adaptation. LoRA is attractive because it freezes the base model and learns a compact low-rank update, which can greatly reduce training cost compared with full fine-tuning.",
  },
  {
    id: "mit15773-l10-q21",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statements about using retrieval-augmented generation in practice are correct?",
    options: [
      {
        text: "RAG is a rare niche method and is not one of the main practical business uses of LLMs.",
        isCorrect: false,
      },
      {
        text: "RAG only works well when the source material is perfectly clean and consistently formatted.",
        isCorrect: false,
      },
      {
        text: "It is often safer when the prompt explicitly tells the model to answer truthfully and admit uncertainty.",
        isCorrect: true,
      },
      {
        text: "It guarantees perfect factual accuracy once relevant chunks are retrieved.",
        isCorrect: false,
      },
    ],
    explanation:
      "RAG is one of the most common practical uses of LLMs because many companies want answers grounded in their own documents. It can still work with messy data, but it does not guarantee perfect truth, so good prompting and human oversight still matter.",
  },
  {
    id: "mit15773-l10-q22",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe why fine-tuning can be valuable even when zero-shot, few-shot, and retrieval-augmented prompting exist?",
    options: [
      {
        text: "Fine-tuning cannot meaningfully change style or behavior and only affects factual recall.",
        isCorrect: false,
      },
      {
        text: "Fine-tuning is useless for helping outputs match the language or style of a specific domain.",
        isCorrect: false,
      },
      {
        text: "Fine-tuning can be useful when you want the model to internalize a recurring input-output pattern rather than re-explaining the task every time.",
        isCorrect: true,
      },
      {
        text: "Fine-tuning is only useful when the base model knows absolutely nothing about the task domain.",
        isCorrect: false,
      },
    ],
    explanation:
      "Fine-tuning can change how the model behaves, including style, tone, and recurring task patterns, so it can be valuable even when prompting and RAG are available. It is not limited to cases where the base model knows nothing; often it is used to specialize an already capable model for a narrower job.",
  },
  {
    id: "mit15773-l10-q23",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe the product-review example used to motivate fine-tuning?",
    options: [
      {
        text: "A base model already guaranteed realistic customer-review style, so there was no style mismatch to fix.",
        isCorrect: false,
      },
      {
        text: "Providing product-review training examples could not improve how realistic the generated reviews sounded.",
        isCorrect: false,
      },
      {
        text: "After fine-tuning, the model could generate both positive and negative reviews that felt more authentic.",
        isCorrect: true,
      },
      {
        text: "This example shows that prompting alone is always sufficient for stylistic adaptation.",
        isCorrect: false,
      },
    ],
    explanation:
      "The product-review example showed that a base model may know the topic but still produce the wrong style, such as polished marketing language instead of genuine customer feedback. Fine-tuning on realistic review examples helped the model produce outputs that sounded more like real reviews, which is exactly why the third statement is correct.",
  },
  {
    id: "mit15773-l10-q24",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe standard supervised fine-tuning of a causal LLM?",
    options: [
      {
        text: "Supervised fine-tuning of a causal LLM abandons autoregressive next-token prediction and turns the model into a pure classifier.",
        isCorrect: false,
      },
      {
        text: "In standard causal fine-tuning, the model predicts all output tokens simultaneously rather than using a shifted next-token setup.",
        isCorrect: false,
      },
      {
        text: "It updates trainable weights in the model by backpropagation and gradient-based optimization.",
        isCorrect: true,
      },
      {
        text: "It requires replacing the transformer with a recurrent neural network before domain-specific examples can be used.",
        isCorrect: false,
      },
    ],
    explanation:
      "Standard supervised fine-tuning of a causal LLM still keeps the autoregressive next-token prediction setup. What changes is the data and the learned weights: the model is further trained with backpropagation on task-specific examples, and there is no need to replace the transformer architecture.",
  },
  {
    id: "mit15773-l10-q25",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe the computational challenge of naively fine-tuning a very large model such as Llama-2-70B?",
    options: [
      {
        text: "A huge parameter count has little effect on memory because only activations matter during fine-tuning.",
        isCorrect: false,
      },
      {
        text: "Gradients and optimizer state add almost no extra memory beyond storing the parameters themselves.",
        isCorrect: false,
      },
      {
        text: "Even if the fine-tuning dataset is much smaller than the original pretraining corpus, memory can still be a bottleneck.",
        isCorrect: true,
      },
      {
        text: "Because fine-tuning uses fewer examples than pretraining, memory is automatically negligible even for very large models.",
        isCorrect: false,
      },
    ],
    explanation:
      "The model itself can already be so large that memory becomes a serious problem before you even think about dataset size. On top of the parameters, gradients and optimizer state can add a lot more memory, so a smaller fine-tuning dataset does not make the hardware problem disappear.",
  },
  {
    id: "mit15773-l10-q26",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe rough memory accounting for very large-model fine-tuning?",
    options: [
      {
        text: "Memory use during fine-tuning is basically just the model parameters, while gradients and optimizer state contribute very little.",
        isCorrect: false,
      },
      {
        text: "Using \\(\\text{fp16}\\) storage means roughly 16 bytes per parameter for the model weights themselves.",
        isCorrect: false,
      },
      {
        text: "Optimizer state can require memory on the same order as, or larger than, the memory used for the parameters themselves.",
        isCorrect: true,
      },
      {
        text: "Optimizer state is irrelevant for modern training and can be ignored without approximation tricks.",
        isCorrect: false,
      },
    ],
    explanation:
      "Rough memory accounting for large-model training includes more than just the stored weights. Parameters, gradients, activations, and optimizer state can all matter, and fp16 weight storage is about 2 bytes per parameter rather than 16, which is why the second statement is false.",
  },
  {
    id: "mit15773-l10-q27",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe why fine-tuning datasets can be much smaller than pretraining datasets?",
    options: [
      {
        text: "The base model has already absorbed a great deal of general linguistic and world knowledge during pretraining.",
        isCorrect: true,
      },
      {
        text: "Fine-tuning is often meant only to adapt behavior for a narrower task or domain, not to teach language from scratch.",
        isCorrect: true,
      },
      {
        text: "A dataset with tens of thousands of instruction-answer pairs can be tiny compared with trillions of pretraining tokens yet still be useful for adaptation.",
        isCorrect: true,
      },
      {
        text: "Fine-tuning datasets must be larger than pretraining datasets because they require more specialized supervision.",
        isCorrect: false,
      },
    ],
    explanation:
      "Fine-tuning benefits from standing on top of a model that is already broadly capable. Because the goal is often adaptation rather than full language acquisition, a comparatively small amount of supervised or task-specific data can meaningfully change behavior.",
  },
  {
    id: "mit15773-l10-q28",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe gradient checkpointing in the lecture's discussion?",
    options: [
      {
        text: "It was presented as a way to reduce memory used for gradient-related computations.",
        isCorrect: true,
      },
      {
        text: "It trades extra computation time for reduced memory usage.",
        isCorrect: true,
      },
      {
        text: "It was described as an older trick that is useful but beyond the scope of the lecture's detailed treatment.",
        isCorrect: true,
      },
      {
        text: "It changes the training objective from supervised learning to reinforcement learning.",
        isCorrect: false,
      },
    ],
    explanation:
      "Gradient checkpointing is a systems trick, not a change in the learning objective. The lecture used it to illustrate a broader theme: with large models, practical engineering tricks can dramatically reduce memory pressure even if they make training slower.",
  },
  {
    id: "mit15773-l10-q29",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe the main idea behind parameter-efficient fine-tuning in this lecture?",
    options: [
      {
        text: "Instead of updating all model parameters, we try to update only a much smaller subset or auxiliary parameterization.",
        isCorrect: true,
      },
      {
        text: "This can drastically reduce optimizer-state memory compared with naive full fine-tuning.",
        isCorrect: true,
      },
      {
        text: "The lecture focused on modifying only matrices inside the causal self-attention blocks while freezing much of the rest.",
        isCorrect: true,
      },
      {
        text: "Parameter-efficient fine-tuning means avoiding gradient descent entirely and using only manual prompt engineering.",
        isCorrect: false,
      },
    ],
    explanation:
      "Parameter-efficient fine-tuning keeps the benefits of gradient-based adaptation while reducing the number of trainable degrees of freedom. The lecture used this to motivate why we do not always need to update every parameter in a giant model to get useful task-specific changes.",
  },
  {
    id: "mit15773-l10-q30",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe LoRA (Low-Rank Adaptation) as presented in the lecture?",
    options: [
      {
        text: "LoRA represents an update to a large weight matrix using a product of two much smaller matrices.",
        isCorrect: true,
      },
      {
        text: "The motivation is that fine-tuning may require only a relatively simple change to the original weights.",
        isCorrect: true,
      },
      {
        text: "In the lecture, this was explained using a change matrix \\(\\Delta A^K\\) added to an original attention matrix.",
        isCorrect: true,
      },
      {
        text: "LoRA requires retraining the entire base model from random initialization so that low-rank factors can emerge.",
        isCorrect: false,
      },
    ],
    explanation:
      "LoRA assumes that the necessary adjustment to a pretrained weight matrix can often be expressed in a low-rank form. Instead of storing and optimizing a huge dense update, we optimize a compact factorization that approximates the needed change.",
  },
  {
    id: "mit15773-l10-q31",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Suppose a full attention update matrix \\(\\Delta A\\) is approximated as the product of two skinny matrices. Which statements are consistent with the lecture's explanation?",
    options: [
      {
        text: "The number of trainable parameters can become far smaller than the number in the original full update matrix.",
        isCorrect: true,
      },
      {
        text: "This is why LoRA can reduce optimizer-state memory dramatically.",
        isCorrect: true,
      },
      {
        text: "The original large pretrained matrix can remain frozen while the low-rank update factors are learned.",
        isCorrect: true,
      },
      {
        text: "The whole point of LoRA is to increase the number of trainable parameters so the model becomes more expressive than the base model.",
        isCorrect: false,
      },
    ],
    explanation:
      "The lecture's numerical example was intended to show just how dramatic the parameter reduction can be. The frozen base matrix still carries most of the original model capacity, while the learned low-rank factors provide a compact, trainable correction.",
  },
  {
    id: "mit15773-l10-q32",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe the LoRA optimization procedure discussed in the lecture?",
    options: [
      {
        text: "Base model parameters are frozen.",
        isCorrect: true,
      },
      {
        text: "Low-rank update components such as those corresponding to \\(\\Delta A^K\\), \\(\\Delta A^Q\\), and \\(\\Delta A^V\\) are initialized and then optimized.",
        isCorrect: true,
      },
      {
        text: "Stochastic gradient descent or similar optimizers can still be used to train the low-rank factors.",
        isCorrect: true,
      },
      {
        text: "LoRA requires deleting the original attention matrices from the model before training begins.",
        isCorrect: false,
      },
    ],
    explanation:
      "LoRA does not remove the pretrained model; it augments it with a compact trainable update while leaving the original weights intact. Training still proceeds through ordinary gradient-based optimization, but now over a much smaller set of parameters.",
  },
  {
    id: "mit15773-l10-q33",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe the lecture's comparison between small and large causal LLMs for fine-tuning?",
    options: [
      {
        text: "Small models such as GPT-2-like systems are much easier to fine-tune directly than very large models.",
        isCorrect: true,
      },
      {
        text: "Very large models raise practical issues of RAM and GPU requirements even if the task dataset is modest.",
        isCorrect: true,
      },
      {
        text: "The Llama 2 family was presented as a widely used set of open models for fine-tuning.",
        isCorrect: true,
      },
      {
        text: "The lecture argued that once a model is open source, full fine-tuning no longer poses any meaningful compute challenge.",
        isCorrect: false,
      },
    ],
    explanation:
      "Open access to model weights makes fine-tuning possible, but not automatically cheap. The lecture stressed that size still matters enormously: a 70B-parameter model is a very different engineering problem from a 7B-parameter model.",
  },
  {
    id: "mit15773-l10-q34",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe the relationship between fine-tuning and prompting methods?",
    options: [
      {
        text: "Zero-shot and few-shot prompting adapt behavior through the prompt while leaving weights fixed.",
        isCorrect: true,
      },
      {
        text: "RAG adapts behavior by adding retrieved context to the prompt rather than changing the model itself.",
        isCorrect: true,
      },
      {
        text: "Fine-tuning is different because it changes model weights to encode task-specific behavior more directly.",
        isCorrect: true,
      },
      {
        text: "Prompting and fine-tuning are identical in the lecture because both require exactly the same compute and update the same parameters.",
        isCorrect: false,
      },
    ],
    explanation:
      "The lecture treated prompting, retrieval, and fine-tuning as distinct levels on an adaptation ladder. The key dividing line is whether the model is treated as a fixed black box or whether gradient-based training actually changes its internal parameters.",
  },
  {
    id: "mit15773-l10-q35",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements correctly reflect the lecture's caution about enterprise use of LLMs and RAG?",
    options: [
      {
        text: "Even when RAG retrieves relevant material, small prompt changes can substantially alter the model's behavior.",
        isCorrect: true,
      },
      {
        text: "Without careful controls or human oversight, inaccurate answers can still create business risk.",
        isCorrect: true,
      },
      {
        text: "The lecture used Air Canada's chatbot as an example of why wrong answers in production can matter legally.",
        isCorrect: true,
      },
      {
        text: "Because RAG uses external documents, hallucinations become impossible and human review becomes unnecessary.",
        isCorrect: false,
      },
    ],
    explanation:
      "The lecture made it clear that retrieval helps but does not solve all trust and QA problems. Production deployment still requires care because prompts, retrieved evidence, and model behavior can interact in ways that produce harmful mistakes.",
  },
  {
    id: "mit15773-l10-q36",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe why helper LLMs can reduce human effort in instruction tuning pipelines?",
    options: [
      {
        text: "They can help generate candidate instruction-following responses.",
        isCorrect: true,
      },
      {
        text: "They can assist with ranking or judging answers, reducing direct human effort in some stages.",
        isCorrect: true,
      },
      {
        text: "The lecture noted that researchers have automated parts of the earlier human-intensive steps using helper LLMs.",
        isCorrect: true,
      },
      {
        text: "Once helper LLMs are introduced, reward models become unnecessary because no preference modeling is needed anymore.",
        isCorrect: false,
      },
    ],
    explanation:
      "Helper LLMs can reduce labeling burden, but they do not magically eliminate the need for structured preference learning. The lecture's point was that some previously human-heavy stages can now be partially automated, making the pipeline more scalable.",
  },
  {
    id: "mit15773-l10-q37",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe the lecture's use of the phrase 'more became different' when discussing GPT-3?",
    options: [
      {
        text: "It referred to the idea that scaling model size and data produced qualitatively new behavior rather than only a small quantitative improvement.",
        isCorrect: true,
      },
      {
        text: "It was used to explain why GPT-3's stronger continuation ability seemed surprising relative to GPT and GPT-2.",
        isCorrect: true,
      },
      {
        text: "It suggests that emergent capabilities may not be obvious from looking only at smaller versions of the same architecture.",
        isCorrect: true,
      },
      {
        text: "It meant that GPT-3 abandoned next-word prediction and switched to a completely different training objective.",
        isCorrect: false,
      },
    ],
    explanation:
      "The lecture used the phrase to emphasize that simply scaling an existing setup can sometimes produce behavior that feels qualitatively different. GPT-3 still used the same core autoregressive framework, but its behavior crossed a threshold that earlier models had not reached.",
  },
  {
    id: "mit15773-l10-q38",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe how conversations differ from one-shot instructions in the lecture's presentation of ChatGPT?",
    options: [
      {
        text: "A conversation dataset includes multiple turns rather than isolated instruction-response pairs.",
        isCorrect: true,
      },
      {
        text: "Training on conversations helps the model handle follow-up requests such as changing the tone or style of a previous answer.",
        isCorrect: true,
      },
      {
        text: "The lecture presented this conversational training as a key reason ChatGPT supports multi-turn interaction better than plain InstructGPT.",
        isCorrect: true,
      },
      {
        text: "Conversation training means the model no longer relies on autoregressive token prediction.",
        isCorrect: false,
      },
    ],
    explanation:
      "The transition from isolated instruction-following to multi-turn dialogue required training data that reflects conversational structure. However, the underlying model still remains autoregressive; what changes is the kind of examples it is fine-tuned on.",
  },
  {
    id: "mit15773-l10-q39",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements correctly compare the effort-benefit tradeoff of different LLM adaptation strategies as discussed in the lecture?",
    options: [
      {
        text: "Zero-shot prompting is usually the lowest-effort strategy because it only requires a carefully written instruction.",
        isCorrect: true,
      },
      {
        text: "Few-shot prompting can improve behavior by showing examples without changing the model weights.",
        isCorrect: true,
      },
      {
        text: "RAG often provides a strong middle ground when proprietary context matters but full weight updates are unnecessary or expensive.",
        isCorrect: true,
      },
      {
        text: "The lecture argued that fine-tuning should always be the first adaptation method attempted because it is the simplest and cheapest option.",
        isCorrect: false,
      },
    ],
    explanation:
      "The adaptation ladder in the lecture was meant to encourage progressive escalation. Start with cheaper, less invasive approaches when they are sufficient, and move toward retrieval or fine-tuning only when the task requires more specialized adaptation.",
  },
  {
    id: "mit15773-l10-q40",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements best capture the lecture's overall message about adapting large language models?",
    options: [
      {
        text: "A strong base LLM is often only the starting point; practical usefulness frequently requires additional adaptation.",
        isCorrect: true,
      },
      {
        text: "Instruction tuning, RLHF, prompting, RAG, and fine-tuning are all ways of shaping model behavior for actual use cases.",
        isCorrect: true,
      },
      {
        text: "Parameter-efficient methods such as LoRA make it possible to adapt large models with far fewer trainable parameters than naive full fine-tuning.",
        isCorrect: true,
      },
      {
        text: "Once a model is large enough, adaptation techniques stop mattering because scale alone solves instruction following, enterprise grounding, and domain style automatically.",
        isCorrect: false,
      },
    ],
    explanation:
      "The lecture's central theme was that scale gives you a powerful base, but not a finished product for every task. Real deployment usually requires some combination of prompt design, retrieval, supervision, preference learning, or efficient fine-tuning to make the model reliably useful.",
  },
];
