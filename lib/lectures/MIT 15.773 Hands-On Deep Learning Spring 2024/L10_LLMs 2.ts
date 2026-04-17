import { Question } from "../../quiz";

export const L10LLMs2Questions: Question[] = [
  {
    id: "mit15773-l10-q01",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe the lecture's clarification about the output side of a causal large language model such as GPT-3?",
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
        text: "This setup differs from the earlier simplified drawing that showed a ReLU after the contextual embeddings.",
        isCorrect: true,
      },
      {
        text: "It is not the case that The final hidden vector before the Softmax can be a single scalar because the Softmax will expand it to the vocabulary size automatically.",
        isCorrect: true,
      },
    ],
    explanation:
      "The lecture explicitly corrected the earlier simplification and noted that causal LLMs like GPT-3 use a linear Dense projection into vocabulary-sized logits before Softmax. That vocabulary-sized output is necessary because the model must assign a probability to every possible next token.",
  },
  {
    id: "mit15773-l10-q02",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statements correctly compare GPT, GPT-2, and GPT-3 as presented in the lecture?",
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
        text: "The lecture suggested that GPT-3 showed a kind of qualitative jump in behavior relative to GPT and GPT-2.",
        isCorrect: true,
      },
      {
        text: "It is not the case that GPT-2 was presented as instruction tuned with reinforcement learning from human feedback in the same way as ChatGPT.",
        isCorrect: true,
      },
    ],
    explanation:
      "The lecture emphasized that GPT, GPT-2, and GPT-3 were trained with the same basic autoregressive objective, but GPT-3 was much larger and trained on more data. The important point was that scale alone seemed to produce new capabilities that smaller predecessors did not reliably show.",
  },
  {
    id: "mit15773-l10-q03",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Why was GPT-3 described as more impressive than GPT or GPT-2 in the lecture?",
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
        text: "The lecture described its improvement as an unexpected emergent change rather than a small incremental gain.",
        isCorrect: true,
      },
      {
        text: "It is not the case that It was the first language model ever to use the transformer architecture.",
        isCorrect: true,
      },
    ],
    explanation:
      "GPT-3's striking behavior was not just that it got a bit better, but that it could continue prompts in a far more coherent and convincing way than earlier GPT models. The lecture presented this as one of those cases where simply scaling the model led to behavior researchers did not fully anticipate.",
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
        text: "The lecture showed that GPT-3 could give irrelevant, offensive, or factually incorrect output.",
        isCorrect: true,
      },
      {
        text: "It is not the case that Its main problem was that it had never been trained on any text from the internet.",
        isCorrect: true,
      },
    ],
    explanation:
      "The lecture's key point was that GPT-3 was only optimized for next-token prediction, so it had no direct training objective for helpful instruction following. As a result, it often completed prompts in plausible but unhelpful ways, such as continuing a list rather than obeying a request.",
  },
  {
    id: "mit15773-l10-q05",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe Step 1 of instruction tuning as presented in the lecture?",
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
        text: "It is not the case that This step required no labeled answers because it was entirely unsupervised.",
        isCorrect: true,
      },
    ],
    explanation:
      "In Step 1, humans explicitly created good answers, which gave the model direct supervision for instruction following. The lecture stressed that the mechanics were still based on next-token prediction, but now over carefully curated instruction-answer pairs rather than generic internet text.",
  },
  {
    id: "mit15773-l10-q06",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements about supervised fine-tuning (SFT) are correct in this lecture?",
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
        text: "The lecture noted that doing much more SFT was limited partly by the cost of writing many high-quality answers.",
        isCorrect: true,
      },
      {
        text: "It is not the case that SFT eliminated the need for any later ranking or preference data.",
        isCorrect: true,
      },
    ],
    explanation:
      "Supervised fine-tuning helped a lot, but it was expensive because humans had to write strong answers at scale. That cost motivated the next step of the pipeline, where ranking answers written by the model was treated as easier than writing all answers from scratch.",
  },
  {
    id: "mit15773-l10-q07",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "According to the lecture, what is easier than writing many high-quality answers from scratch?",
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
        text: "It is not the case that Skipping humans entirely and assuming the longest answer is always the best one.",
        isCorrect: true,
      },
    ],
    explanation:
      "The lecture framed answer ranking as a much cheaper and easier source of supervision than writing answers manually. Once GPT-3 can generate multiple candidate responses, people can provide valuable preference information just by ordering them from better to worse.",
  },
  {
    id: "mit15773-l10-q08",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe the reward model used in the lecture's presentation of reinforcement learning from human feedback?",
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
        text: "It is not the case that The reward model is trained to predict the next token in the answer sequence exactly like a standard language model.",
        isCorrect: true,
      },
    ],
    explanation:
      "The reward model is not just another next-token predictor. Its job is to learn human preferences over complete instruction-answer pairs, assigning higher scores to better responses so that later reinforcement learning can nudge the base model in the right direction.",
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
        text: "It is not the case that The loss is minimized when \\(r_P\\) is much smaller than \\(r_O\\).",
        isCorrect: true,
      },
    ],
    explanation:
      "This loss depends on the score difference, not on the absolute values alone. It gets smaller when the preferred answer is scored more highly than the other answer, which is exactly the ranking behavior the model is meant to learn.",
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
        text: "It is not the case that It directly replaces the language model and becomes the chatbot shown to end users.",
        isCorrect: true,
      },
    ],
    explanation:
      "The reward model acts as a learned judge, not as the final chatbot itself. Once trained, it supplies a preference-based scalar signal that reinforcement learning can use to shape the response behavior of the main language model.",
  },
  {
    id: "mit15773-l10-q11",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe reinforcement learning from human feedback (RLHF) in the lecture?",
    options: [
      {
        text: "The term refers to using human preference data to train a reward model and then using reinforcement learning to improve the language model.",

        isCorrect: true,
      },
      {
        text: "The lecture described Proximal Policy Optimization as the reinforcement learning algorithm used in this stage.",
        isCorrect: true,
      },
      {
        text: "The rating from the reward model is used to nudge GPT-3 in a preferred direction.",
        isCorrect: true,
      },
      {
        text: "It is not the case that RLHF means directly editing the model weights by hand for each bad answer the chatbot produces.",
        isCorrect: true,
      },
    ],
    explanation:
      "RLHF combines two ingredients: human preference information and reinforcement learning. Rather than manually editing the model, the system uses reward-model scores to produce learning signals that shift the language model toward responses humans tend to prefer.",
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
        text: "It is not the case that ChatGPT was described as a completely different non-transformer architecture unrelated to GPT-3.5.",
        isCorrect: true,
      },
    ],
    explanation:
      "The lecture presented ChatGPT as an extension of the same general strategy used to build InstructGPT. The important change was that the supervision became conversational, so the model learned not only to answer one instruction, but to sustain multi-turn interaction.",
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
    prompt:
      "Which statements correctly describe zero-shot prompting in the lecture?",
    options: [
      {
        text: "It can work surprisingly well on some tasks by simply instructing the LLM clearly.",

        isCorrect: true,
      },
      {
        text: "The lecture illustrated it with a product-review defect-detection example.",
        isCorrect: true,
      },
      {
        text: "Prompt design can matter a lot for zero-shot performance.",
        isCorrect: true,
      },
      {
        text: "It is not the case that Zero-shot prompting requires further gradient-based training of the base model before any answer can be produced.",
        isCorrect: true,
      },
    ],
    explanation:
      "Zero-shot prompting uses the model as-is and relies on the prompt itself to specify the task. The lecture's defect-detection example showed that a capable model can often do useful work immediately, though prompt wording still matters.",
  },
  {
    id: "mit15773-l10-q15",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements about prompt engineering are correct in the lecture's framing?",
    options: [
      {
        text: "It is not the case that Prompt engineering means carefully designing the prompt so the LLM is more likely to produce the desired behavior.",

        isCorrect: false,
      },
      {
        text: "Breaking a task into explicit intermediate steps can improve reliability on some problems.",
        isCorrect: true,
      },
      {
        text: "The lecture gave the example of first listing the words in a sentence before identifying the fifth word.",
        isCorrect: true,
      },
      {
        text: "Prompt engineering is unnecessary once a model has undergone instruction tuning or RLHF.",
        isCorrect: false,
      },
    ],
    explanation:
      "Even instruction-tuned models still benefit from carefully structured prompts. The lecture used simple examples to show that making the desired reasoning steps explicit can help the model avoid obvious mistakes.",
  },
  {
    id: "mit15773-l10-q16",
    chapter: 1,
    difficulty: "medium",
    prompt: "Which statements correctly describe few-shot prompting?",
    options: [
      {
        text: "It is not the case that It provides a few worked examples inside the prompt itself.",

        isCorrect: false,
      },
      {
        text: "The lecture illustrated it with a grammar-correction pattern using Poor English and Good English examples.",
        isCorrect: true,
      },
      {
        text: "It is an example of in-context learning because the model learns what to do from examples in the prompt without changing its weights.",
        isCorrect: true,
      },
      {
        text: "Few-shot prompting is identical to fine-tuning because both methods update the underlying model parameters using stochastic gradient descent.",
        isCorrect: false,
      },
    ],
    explanation:
      "Few-shot prompting teaches the model through examples placed directly in the prompt rather than through weight updates. It is part of in-context learning, where the model infers the task pattern from the current context alone.",
  },
  {
    id: "mit15773-l10-q17",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe retrieval-augmented generation (RAG) as presented in the lecture?",
    options: [
      {
        text: "It is not the case that RAG is useful when we want the LLM to leverage proprietary or custom information relevant to a specific question.",

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
      "RAG solves a context-selection problem, not a weight-update problem. The model remains unchanged while an external retrieval system selects relevant information to fit inside the context window for the current query.",
  },
  {
    id: "mit15773-l10-q18",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statements about the context window are correct according to the lecture?",
    options: [
      {
        text: "It is not the case that For an LLM, the prompt plus output must fit within a predefined limit called the context window.",

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
      "The context window constrains how much text the model can consider at once, both in training-style setups and in real interactive use. Conversation history consumes part of this budget, which is why long chats or large attached documents need careful handling.",
  },
  {
    id: "mit15773-l10-q19",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe the retrieval pipeline used in the lecture's RAG example?",
    options: [
      {
        text: "It is not the case that Text chunks from documents can be embedded into contextual vectors using a pretrained embedding model.",

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
        text: "The lecture's method required every chunk to be manually labeled with categories before retrieval could work.",
        isCorrect: false,
      },
    ],
    explanation:
      "The core RAG flow is embed, compare, retrieve, and then prompt. The lecture's Olympics example showed that this can work even with fairly messy source text, as long as the relevant chunks can be embedded and ranked by similarity to the question.",
  },
  {
    id: "mit15773-l10-q20",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe fine-tuning and LoRA in the lecture?",
    options: [
      {
        text: "It is not the case that Standard fine-tuning updates model weights using domain-specific input-output examples.",

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
      "The lecture's central memory-saving idea was to avoid updating the entire massive model during fine-tuning. LoRA does this by representing updates with small low-rank components, which dramatically reduces the number of trainable parameters and makes fine-tuning much more practical.",
  },

  {
    id: "mit15773-l10-q21",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statements about using retrieval-augmented generation in practice were emphasized in the lecture?",
    options: [
      {
        text: "It is not the case that It is one of the most prevalent business applications of large language models.",

        isCorrect: false,
      },
      {
        text: "It is not the case that It can work reasonably well even when source text is messy or imperfectly formatted.",
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
      "The lecture presented RAG as highly practical and already widely used in business settings. At the same time, it was clear that RAG is not magic: retrieval helps a lot, but prompt wording, chunk selection, and model limitations still matter, so errors can still happen.",
  },
  {
    id: "mit15773-l10-q22",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe why fine-tuning can be valuable even when zero-shot, few-shot, and retrieval-augmented prompting exist?",
    options: [
      {
        text: "It is not the case that Fine-tuning can alter the style and behavior of the model for a narrow use case rather than only changing the prompt.",

        isCorrect: false,
      },
      {
        text: "It is not the case that Fine-tuning can help a model produce outputs that better match a specific domain, such as realistic customer reviews.",
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
      "Prompting methods leave the model itself unchanged, while fine-tuning changes its internal behavior. This can matter when you want stable, domain-specific behavior repeatedly, such as outputs that sound like authentic reviews rather than marketing copy.",
  },
  {
    id: "mit15773-l10-q23",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe the product-review example used to motivate fine-tuning?",
    options: [
      {
        text: "It is not the case that A base model could generate text that sounded more like marketing copy than like a real customer review.",

        isCorrect: false,
      },
      {
        text: "It is not the case that Fine-tuning with instruction, product description, and product review examples improved the realism of generated reviews.",
        isCorrect: false,
      },
      {
        text: "The lecture showed that after fine-tuning, the model could generate both positive and negative reviews that felt more authentic.",
        isCorrect: true,
      },
      {
        text: "The example was used to argue that prompting alone is always sufficient for stylistic adaptation.",
        isCorrect: false,
      },
    ],
    explanation:
      "The point of the review example was that base LLM behavior can be directionally correct yet stylistically wrong for a business use case. Fine-tuning with a few hundred well-chosen examples can push the model toward outputs that better match the desired genre and tone.",
  },
  {
    id: "mit15773-l10-q24",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "In the lecture's framing, which statements correctly describe standard supervised fine-tuning of a causal LLM?",
    options: [
      {
        text: "It is not the case that It can be viewed as continuing next-token training on domain-specific input-output examples.",

        isCorrect: false,
      },
      {
        text: "It is not the case that The desired output sequence is shifted relative to the input so that prediction remains autoregressive.",
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
      "The lecture emphasized continuity: fine-tuning is not a totally different learning paradigm from pretraining, but rather a further stage of gradient-based learning on new data. The causal next-token structure remains, but the training data is now aligned to the target use case.",
  },
  {
    id: "mit15773-l10-q25",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe the computational challenge of naively fine-tuning a very large model such as Llama-2-70B?",
    options: [
      {
        text: "It is not the case that The raw parameter count alone already implies a very large memory footprint.",

        isCorrect: false,
      },
      {
        text: "It is not the case that Gradients and optimizer state substantially increase memory needs beyond just storing the parameters.",
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
      "A crucial point from the lecture was that fine-tuning is easier than training from scratch mainly because of the smaller dataset, not because the model suddenly becomes small. The huge model still has to live in memory, and gradients plus optimizer state can make the naive footprint enormous.",
  },
  {
    id: "mit15773-l10-q26",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe the lecture's rough memory accounting for very large-model fine-tuning?",
    options: [
      {
        text: "It is not the case that Memory use includes model parameters, gradient-related storage, and optimizer state.",

        isCorrect: false,
      },
      {
        text: "It is not the case that Using \\(\\text{fp16}\\) storage means roughly 2 bytes per parameter for the model weights themselves.",
        isCorrect: false,
      },
      {
        text: "Optimizer state can require memory on the same order as, or larger than, the memory used for the parameters themselves.",
        isCorrect: true,
      },
      {
        text: "The lecture claimed that optimizer state is irrelevant for modern training and can be ignored without approximation tricks.",
        isCorrect: false,
      },
    ],
    explanation:
      "The lecture's message was that the parameter tensor is only part of the story. Training also requires memory for gradients and optimizer statistics, and those additional quantities can make the total naive footprint far larger than the parameter storage alone.",
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
