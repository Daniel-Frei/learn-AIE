import { Question } from "../../quiz";

type Lecture7Difficulty = "easy" | "medium" | "hard";
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
  difficulty: Lecture7Difficulty,
  prompt: string,
  optionSeeds: readonly OptionSeed[],
  explanation: string,
): Question {
  if (optionSeeds.length !== 4) {
    throw new Error(`CME295 Lecture 7 question ${id} needs 4 options.`);
  }

  return {
    id,
    chapter: 7,
    difficulty,
    prompt,
    options: optionSeeds.map(([text, isCorrect]) => ({ text, isCorrect })),
    explanation,
  };
}

function makeAssertionReasonQuestion(
  id: string,
  difficulty: Lecture7Difficulty,
  prompt: string,
  correctChoice: AssertionReasonChoice,
  explanation: string,
): Question {
  return {
    id,
    chapter: 7,
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

export const stanfordCME295Lecture7RagToolsAgentsQuestions: Question[] = [
  makeQuestion(
    "cme295-lect7-q01",
    "easy",
    "A team is moving from a standalone language model to a system that can use outside context and actions. Which limitations motivate adding retrieval, tools, or agent loops?",
    [
      [
        "The model's factual knowledge is constrained by the data available before its pretraining cutoff.",
        true,
      ],
      [
        "A large context window can still be expensive and can distract the model with irrelevant material.",
        true,
      ],
      [
        "A plain text-only model cannot directly execute external actions such as database writes or API calls.",
        true,
      ],
      [
        "A larger context window by itself gives the model permission to perform external writes and API calls.",
        false,
      ],
    ],
    "System augmentation is motivated by static knowledge, limited and costly context, distraction from irrelevant tokens, and inability to act on the outside world. Bigger prompts help only up to a point; they do not solve retrieval relevance, tool correctness, action safety, or evaluation by themselves.",
  ),
  makeQuestion(
    "cme295-lect7-q02",
    "easy",
    'A product request says, "Answer questions about today\'s internal policy changes and, when approved, update the ticket tracker." Which architectural distinctions are correct?',
    [
      [
        "Retrieval-augmented generation can fetch the relevant policy snippets without changing model weights.",
        true,
      ],
      [
        "Tool calling can expose a ticket-update function that a backend executes after the model predicts the function and arguments.",
        true,
      ],
      [
        "Retrieval-augmented generation directly trains the base model on today's policy before each answer.",
        false,
      ],
      [
        "A single retrieved document is already an agent because retrieval always includes autonomous planning and repeated actions.",
        false,
      ],
    ],
    "RAG changes the inference context by adding retrieved information, while tool calling gives the system structured ways to compute, fetch, or act. Neither mechanism automatically implies weight updates, and an agentic loop adds goal-directed iteration rather than just one retrieval result.",
  ),
  makeQuestion(
    "cme295-lect7-q03",
    "easy",
    "Which sequence best matches the basic retrieval-augmented generation pipeline for answering a user question?",
    [
      [
        "Retrieve relevant chunks, augment the prompt with those chunks, then generate the response.",
        true,
      ],
      [
        "Generate a draft from model memory first, retrieve only if the draft is uncertain, then ignore retrieved evidence during the final response.",
        false,
      ],
      [
        "Fine-tune the model on the new documents for every query, then answer without adding retrieved evidence to the prompt.",
        false,
      ],
      [
        "Call all available tools first, concatenate their outputs, then treat the longest tool response as the retrieved context.",
        false,
      ],
    ],
    "The core RAG pattern is retrieve, augment, and generate. Retrieval supplies task-relevant context, prompt augmentation exposes it to the model, and generation uses that context to produce the answer.",
  ),
  makeQuestion(
    "cme295-lect7-q04",
    "easy",
    "A team is preparing documents for a retrieval knowledge base. Which setup choices belong to the knowledge-base construction stage?",
    [
      ["Collect documents that may be useful for future questions.", true],
      [
        "Divide documents into chunks whose sizes preserve enough local context.",
        true,
      ],
      [
        "Compute and store embeddings for chunks so candidate retrieval can compare them to a query.",
        true,
      ],
      [
        "Choose chunk overlap only after generation, because overlap has no effect on the indexed knowledge base.",
        false,
      ],
    ],
    "A RAG knowledge base is not just a pile of raw documents. It needs curated inputs, chunking decisions, embeddings, and storage so later retrieval can find useful evidence rather than arbitrary text spans. Overlap is an indexing-time chunking choice; choosing it only after generation would not repair boundary-loss problems in the stored chunks.",
  ),
  makeQuestion(
    "cme295-lect7-q05",
    "medium",
    "A policy manual has long sections with definitions followed by exceptions. The RAG builder proposes very small non-overlapping chunks because they are cheap to embed. Which concerns should change that design?",
    [
      [
        "Tiny chunks can lose the context needed to interpret exceptions or pronouns.",
        true,
      ],
      [
        "Some overlap can preserve information that crosses an arbitrary chunk boundary.",
        true,
      ],
      [
        "Smaller chunks always improve retrieval because each embedding becomes more specific in every domain.",
        false,
      ],
      [
        "Chunk size should be chosen without regard to document structure because token count is the only meaningful signal.",
        false,
      ],
    ],
    "Chunk size and overlap are hyperparameters with tradeoffs. Very small chunks may be context-poor, very large chunks may produce embeddings that blur several topics, and structured sources such as policies, code, JSON, or markdown often need chunking that respects their organization.",
  ),
  makeQuestion(
    "cme295-lect7-q06",
    "medium",
    "A retrieval system first narrows one million chunks to 100 candidates and then applies a slower cross-encoder to the 100. Which statements correctly explain this two-stage design?",
    [
      [
        "The first stage should emphasize recall so plausible candidates are not filtered out too early.",
        true,
      ],
      [
        "The second stage can spend more computation because it scores a much smaller candidate set.",
        true,
      ],
      [
        "The reranking stage tries to put the most relevant chunks near the top of the final list.",
        true,
      ],
      [
        "The first stage should optimize final precision by running the most expensive ranker over the entire corpus.",
        false,
      ],
    ],
    "Candidate retrieval is a broad, cheaper pass intended to avoid missing relevant material. Reranking is a narrower, more expensive pass that improves ordering after the candidate set is small enough to score in detail.",
  ),
  makeQuestion(
    "cme295-lect7-q07",
    "easy",
    "In embedding-based semantic retrieval, what is the most direct role of cosine similarity or a similar vector-comparison score?",
    [
      [
        "It compares the query embedding with chunk embeddings to rank likely relevant chunks.",
        true,
      ],
      [
        "It supports the fast candidate-retrieval stage before any slower cross-encoder reranking is applied.",
        true,
      ],
      [
        "It measures exact keyword overlap in the same way as BM25, so token identity matters more than vector direction.",
        false,
      ],
      [
        "It is the final cross-encoder score produced after jointly encoding the query and each candidate chunk.",
        false,
      ],
    ],
    "Cosine similarity compares vector directions for separately embedded queries and chunks, which makes it useful for fast candidate retrieval. It is not BM25-style lexical matching, and it is not the same as a later cross-encoder that jointly reads a query and a candidate.",
  ),
  makeQuestion(
    "cme295-lect7-q08",
    "medium",
    'A user asks, "Where is Cuddly?" The relevant note says "Cuddly\'s GPS tag is in the downstairs lab," while many semantically related notes mention hugs, softness, or nearby locations. Which retrieval claims are correct?',
    [
      [
        "BM25-style keyword matching can help when exact names or identifiers matter.",
        true,
      ],
      [
        "Semantic search can help when the query and document use different but related wording.",
        true,
      ],
      [
        "A hybrid method can combine exact-term strength with embedding-based semantic matching.",
        true,
      ],
      [
        "A hybrid method should discard exact-name matches whenever semantic search returns many soft-neighbor results.",
        false,
      ],
    ],
    "The useful contrast is between lexical matching for exact names and semantic matching for meaning. A hybrid approach can combine both signals; discarding exact-name evidence would be risky when the user's query turns on an identifier such as a name, code, or tag.",
  ),
  makeAssertionReasonQuestion(
    "cme295-lect7-q09",
    "medium",
    "Assertion: Query embeddings and document-chunk embeddings can be mismatched because a short question often does not look like the answer-bearing passage it should retrieve.\n\nReason: In a bi-encoder retriever, the query and chunk are encoded independently, so missing answer-like terms in a terse query can move its vector away from the relevant passage.",
    4,
    "The assertion is true because a terse query and a source passage can differ in length, detail, and wording even when they are semantically connected. The reason gives the causal mechanism: independent query/chunk encoding can make the relevant passage less similar in vector space when the query omits answer-like terms.",
  ),
  makeQuestion(
    "cme295-lect7-q10",
    "hard",
    "A dense retriever misses a passage because the user asks a terse question, while the relevant chunk is written as a detailed answer paragraph. Which interventions specifically target this query-document embedding mismatch?",
    [
      [
        "Generate a hypothetical answer-like document from the query and embed that expanded text for retrieval.",
        true,
      ],
      [
        "Rewrite the query with additional terms or paraphrases that make the target evidence easier to match.",
        true,
      ],
      [
        "Increase the final answer temperature while leaving the same failed candidate-retrieval representation unchanged.",
        false,
      ],
      [
        "Tune the response-style prompt so the generator cites retrieved evidence more politely after candidates are selected.",
        false,
      ],
    ],
    "The problem is at candidate retrieval time: the query vector is not close enough to the answer-bearing chunk vector. Query expansion or hypothetical-document retrieval changes the retrieval representation, whereas generation temperature or response-style instructions operate after the candidate has already been missed.",
  ),
  makeQuestion(
    "cme295-lect7-q11",
    "medium",
    'A chunk reads only "It expires after 30 days." The full document is a procurement policy, and the sentence refers to vendor security reviews. Which contextual-retrieval choices are appropriate?',
    [
      [
        "Attach a succinct document-aware context to the chunk before embedding or indexing it.",
        true,
      ],
      [
        'Use the surrounding document to clarify what "it" refers to and which policy topic the chunk belongs to.',
        true,
      ],
      [
        "Keep the added context short enough that it improves search without turning every chunk into a full document.",
        true,
      ],
      [
        "Append the entire procurement manual to every chunk context so each chunk embedding contains the whole document.",
        false,
      ],
    ],
    "Contextual retrieval tries to repair chunk ambiguity by adding a concise description grounded in the whole document. The added context should clarify the chunk for search; appending the whole manual to every chunk would erase the purpose of chunking and create repeated, expensive, poorly focused inputs.",
  ),
  makeQuestion(
    "cme295-lect7-q12",
    "medium",
    "Why can prompt caching be relevant when contextual retrieval uses a model to create document-aware context for many chunks from the same source document?",
    [
      [
        "The whole document or repeated prefix can be reused across many chunk-context calls, reducing repeated input-token cost.",
        true,
      ],
      [
        "It is especially relevant when the same document context is reused while generating contexts for many chunks.",
        true,
      ],
      [
        "Prompt caching makes each contextualized chunk rank higher because cached input tokens receive extra retrieval weight.",
        false,
      ],
      [
        "Prompt caching makes all repeated document-prefix tokens free, so only the first chunk-context call needs input-token accounting.",
        false,
      ],
    ],
    "Contextualizing each chunk may repeatedly send the same full document or long prefix to an LLM. Prompt caching is a cost and latency optimization for repeated prompt material; it does not directly change retrieval weights, guarantee ranking quality, or make all repeated input free.",
  ),
  makeQuestion(
    "cme295-lect7-q13",
    "hard",
    "A cross-encoder reranker scores each candidate by jointly reading the user query and one candidate chunk. Which consequences follow from using it after candidate retrieval?",
    [
      [
        "It can model query-chunk interactions more directly than comparing separately computed embeddings.",
        true,
      ],
      [
        "It is usually applied after a broad retrieval stage because joint scoring is more expensive.",
        true,
      ],
      [
        "It replaces candidate retrieval because joint query-chunk scoring over the full corpus is usually cheaper than embedding search.",
        false,
      ],
      [
        "It can be evaluated independently from final answer generation using ranking labels for retrieved chunks.",
        true,
      ],
    ],
    "A cross-encoder reranker spends extra computation to judge relevance from the query and candidate together. That makes it useful after recall-oriented candidate retrieval and lets teams evaluate retrieval ranking separately from downstream answer generation; it is not usually the cheap first pass over an entire corpus.",
  ),
  makeQuestion(
    "cme295-lect7-q14",
    "hard",
    "A search result list contains one highly relevant chunk at rank 4 and several mildly relevant chunks at ranks 1-3. Which metric interpretations are correct?",
    [
      [
        "Reciprocal rank focuses on how early the first relevant result appears, so the first relevant item at rank 4 gives RR = 1/4.",
        true,
      ],
      [
        "NDCG@k can reward graded relevance and discounts relevance more when it appears lower in the top-k ranking.",
        true,
      ],
      [
        "Recall@k is always identical to reciprocal rank because both ignore all documents after the first relevant result.",
        false,
      ],
      [
        "Precision@k becomes 1 whenever the best document appears anywhere in the top-k results.",
        false,
      ],
    ],
    "RR is dominated by the first relevant result's rank, while NDCG uses ranked and often graded relevance across the top-k list. Precision and recall answer different count-based questions and do not collapse to the reciprocal-rank definition.",
  ),
  makeQuestion(
    "cme295-lect7-q15",
    "hard",
    "A retrieval evaluation labels the top five chunks as follows: rank 1 irrelevant, rank 2 irrelevant, rank 3 relevant, rank 4 relevant, rank 5 irrelevant. What is RR@5?",
    [
      ["1/3, because the first relevant chunk is at rank 3.", true],
      ["2/5, because two of the five chunks are relevant.", false],
      ["1/5, because the metric always divides by k.", false],
      [
        "2, because two relevant chunks appear in the evaluated top five.",
        false,
      ],
    ],
    "Reciprocal rank uses the rank of the first relevant result, not the number of relevant results. Since the first relevant chunk appears at rank 3, RR@5 is 1/3; the two relevant chunks would matter for other metrics such as recall or precision.",
  ),
  makeAssertionReasonQuestion(
    "cme295-lect7-q16",
    "hard",
    "Assertion: A system can have high Recall@100 but still give poor final answers.\n\nReason: NDCG@k discounts lower-ranked relevant documents more than higher-ranked relevant documents.",
    5,
    "Both statements are true, but the reason does not explain why high Recall@100 can coexist with bad answers. High Recall@100 means relevant material survived the broad retrieval stage; it does not guarantee final ranking, prompt construction, or generation quality.",
  ),
  makeQuestion(
    "cme295-lect7-q17",
    "hard",
    "A RAG answer cites no relevant evidence. Logs show that the correct chunk was retrieved at candidate rank 72, reranked to rank 3, but then omitted from the final prompt because only two chunks were included. Which diagnoses are supported?",
    [
      [
        "Candidate retrieval did not completely fail because the correct chunk reached the candidate set.",
        true,
      ],
      [
        "Reranking improved the chunk enough that the prompt-assembly cutoff became the immediate failure point.",
        true,
      ],
      [
        "Increasing final prompt inclusion from two to at least three chunks might address this observed failure.",
        true,
      ],
      [
        "Increasing the candidate-retrieval top-k is the direct fix here because the correct chunk failed to enter the candidate set.",
        false,
      ],
    ],
    "The trace separates stages: the relevant chunk was found, then reranked into a useful position, then lost at prompt assembly. Increasing candidate top-k would address a different failure; this trace points to the prompt-inclusion cutoff or assembly policy as the immediate boundary condition.",
  ),
  makeQuestion(
    "cme295-lect7-q18",
    "medium",
    "Why is RAG often preferred over repeatedly fine-tuning a base model to inject fresh facts for many downstream applications?",
    [
      [
        "Changing model weights for fresh facts can cause regressions or maintenance work across downstream fine-tuned variants.",
        true,
      ],
      [
        "RAG can update the information available at inference time by changing the indexed knowledge base and retrieved context.",
        true,
      ],
      [
        "For many specialized downstream variants, continued training is usually the lowest-maintenance way to update every fresh fact.",
        false,
      ],
      [
        "RAG eliminates the need to evaluate whether the retrieved evidence is relevant or whether the answer used it correctly.",
        false,
      ],
    ],
    "The practical argument is about maintenance, risk, and freshness, not an impossibility theorem. RAG can expose newer information through retrieval while leaving weights untouched, but the retrieval and generation stages still need evaluation.",
  ),
  makeQuestion(
    "cme295-lect7-q19",
    "easy",
    'A tool-enabled assistant answers "Find the nearest service center" using a backend function. Which stages belong to the basic tool-calling flow?',
    [
      [
        "The model predicts which function should be called and what arguments should be supplied.",
        true,
      ],
      ["The backend executes the selected function outside the model.", true],
      [
        "The tool result is returned in a form the model can use to synthesize the final response.",
        true,
      ],
      [
        "The model may need a function schema or description so it can choose and use the tool correctly.",
        true,
      ],
    ],
    "Tool calling separates model-mediated tool prediction from backend execution and response synthesis. The language model does not become the backend API; it chooses and formats calls, then interprets returned data for the user.",
  ),
  makeQuestion(
    "cme295-lect7-q20",
    "easy",
    "A function is exposed to a model as `find_teddy_bear(location: tuple[float, float]) -> TeddyBearInfo`. What is the most important reason to include a clear description of the argument and return value?",
    [
      [
        "The model needs to infer the correct call and interpret the returned object during response synthesis.",
        true,
      ],
      [
        "Argument descriptions are mainly for the backend type checker; the model can infer coordinate order from the function name alone.",
        false,
      ],
      [
        "The return type is irrelevant because the model should ignore tool outputs once the backend succeeds.",
        false,
      ],
      [
        "A clear description automatically enforces authorization, so permission checks are no longer needed.",
        false,
      ],
    ],
    "The schema and description are interface information for the model. They help the model map user intent to function arguments and later understand the output, but they do not replace backend execution, permissions, or safety controls.",
  ),
  makeAssertionReasonQuestion(
    "cme295-lect7-q21",
    "medium",
    "Assertion: In tool calling, the language model should usually predict the function name and arguments, while the application backend performs the actual function execution.\n\nReason: The language model's token generation loop is the same thing as the external API runtime.",
    1,
    "The assertion is true because the model is used to choose a structured call and arguments, while the application executes code, database queries, or network calls. The reason is false: token generation is not the same execution environment as an API runtime.",
  ),
  makeQuestion(
    "cme295-lect7-q22",
    "medium",
    "A team wants a model to use a new weather lookup tool reliably. Which approaches match the tool-use training and prompting strategies?",
    [
      [
        "Create supervised examples where conversation history maps to the desired tool prediction.",
        true,
      ],
      [
        "Create examples where tool outputs are followed by desired response generation.",
        true,
      ],
      [
        "Give the model the function API plus a detailed explanation of how to use it when training data is not available.",
        true,
      ],
      [
        "Use only final-answer examples and omit the desired tool-call predictions, even when the failure is wrong argument selection.",
        false,
      ],
    ],
    "Tool use can be taught with supervised pairs for tool prediction and response generation, or prompted with clear schemas and descriptions. Final-answer examples alone may not teach the model which function and arguments to emit, so interface-level examples matter when argument selection is the failure mode.",
  ),
  makeQuestion(
    "cme295-lect7-q23",
    "easy",
    "Which requests are naturally suited to tool calling rather than answering only from pretrained text memory?",
    [
      ["Look up current weather, stock prices, or database records.", true],
      ["Run a calculator or Python function for a precise computation.", true],
      [
        "Summarize a general concept that is already fully contained in the user prompt.",
        false,
      ],
      [
        "Invent a result for a live external API without checking that API.",
        false,
      ],
    ],
    "Tools are especially useful for current information, structured data, computation, code execution, and external actions. If the answer is already contained in the prompt, a tool may be unnecessary; if live data matters, inventing without a call is the wrong failure mode.",
  ),
  makeQuestion(
    "cme295-lect7-q24",
    "medium",
    "An assistant receives hundreds of possible function APIs in its context for every request. Which challenges can this create?",
    [
      [
        "The context window can fill with tool descriptions that are irrelevant to the current request.",
        true,
      ],
      [
        "The model may pick the wrong tool when many similar or conflicting APIs are present.",
        true,
      ],
      [
        "Latency and cost can rise because the model must process more tool metadata.",
        true,
      ],
      [
        "Tool definition work can scale poorly if every model integration uses bespoke formats.",
        true,
      ],
    ],
    "Adding tools makes systems more capable but also creates selection, context, latency, and integration problems. Those problems motivate tool routing and standardized protocols rather than simply placing every possible API in every prompt.",
  ),
  makeQuestion(
    "cme295-lect7-q25",
    "easy",
    "What is the main purpose of a tool selector or router placed before the final tool-using model call?",
    [
      [
        "Choose a small subset of likely useful tools so the main call sees less irrelevant tool metadata.",
        true,
      ],
      [
        "Reduce latency and improve performance by focusing the final model call on likely relevant tools.",
        true,
      ],
      [
        "Execute the selected functions itself so the backend no longer needs to run external code.",
        false,
      ],
      [
        "Compress the selected tools by deleting their argument schemas before the final model call.",
        false,
      ],
    ],
    "A selector narrows the tool set for the current query, improving focus and reducing context burden. That narrower context can reduce latency and improve performance, but it does not replace backend execution, user-facing synthesis, or model/tool interface documentation.",
  ),
  makeQuestion(
    "cme295-lect7-q26",
    "hard",
    "A tool-selection pipeline has two stages: first a router sees a query plus a short list of all tool names, then the main model sees only the selected full APIs. Which statements are correct?",
    [
      [
        "The router stage reduces context pressure before the main model call.",
        true,
      ],
      [
        "The main model still needs enough schema detail for the selected tools to predict arguments correctly.",
        true,
      ],
      [
        "The router could be implemented with retrieval-like matching, but it can also be an LLM-based classification or selection step.",
        true,
      ],
      [
        "After routing, the main model can infer every selected tool's argument schema from its name alone.",
        false,
      ],
    ],
    "The selector is a scalability device, not a complete interface. It can use retrieval, classification, or another LLM call to reduce the set of tools, while the final model still needs schemas and the system still needs policy and execution controls.",
  ),
  makeAssertionReasonQuestion(
    "cme295-lect7-q27",
    "medium",
    "Assertion: Giving every available tool to the model is always better than routing because more tools strictly increase performance.\n\nReason: Too many tool descriptions can consume context, increase latency, and distract the model with irrelevant or similar APIs.",
    2,
    "The assertion is false because more tools can lower performance and scalability when they overload the context or create selection ambiguity. The reason is true: context pressure, latency, and distraction are exactly why tool selectors and standardized interfaces are useful.",
  ),
  makeQuestion(
    "cme295-lect7-q28",
    "easy",
    "Which goals match Model Context Protocol (MCP) in a tool-using LLM ecosystem?",
    [
      [
        "Expose tools, prompts, and resources to LLM applications through a standard protocol.",
        true,
      ],
      [
        "Reduce duplicated bespoke integrations between many LLM hosts and many tool providers.",
        true,
      ],
      [
        "Centralize provider-specific authorization and execution policy inside the MCP client so servers only contain natural-language docs.",
        false,
      ],
      [
        "Make the MCP client the provider-owned component that directly implements every tool.",
        false,
      ],
    ],
    "MCP is a standardization layer for connecting hosts, clients, servers, tools, prompts, and resources. Standardization reduces repeated integration work, but it does not by itself solve safety, correctness, authorization, or backend implementation, and it does not move provider execution policy into the host-side client.",
  ),
  makeQuestion(
    "cme295-lect7-q29",
    "medium",
    "A book provider exposes an MCP server to a desktop LLM app. Which mapping is consistent with MCP architecture vocabulary?",
    [
      ["The desktop LLM app can act as the MCP host.", true],
      [
        "The host-side connection to a server is represented by an MCP client.",
        true,
      ],
      [
        "The provider's MCP server can expose tools such as finding or recommending books.",
        true,
      ],
      [
        "The provider's MCP server, not the host-side MCP client, is the component that owns the book-tool implementation.",
        true,
      ],
    ],
    "In MCP vocabulary, a host application connects through clients to MCP servers that expose tools, prompts, and resources. A book provider is a natural server owner because it knows and operates the tool implementations and data resources it wants to expose.",
  ),
  makeQuestion(
    "cme295-lect7-q30",
    "easy",
    "A provider wants to expose `find_title` and `recommend_taste` tools to a desktop LLM app through MCP. Which component most directly serves those provider-owned tool implementations?",
    [
      ["The MCP server.", true],
      ["The MCP host application.", false],
      ["The host-side MCP client connection.", false],
      ["The provider's resource data alone, without a server.", false],
    ],
    "The MCP server is the provider-side component that exposes tools, prompts, and resources. The host and client participate in the connection, and resources may supply data, but the server is what serves the provider's tool implementations.",
  ),
  makeAssertionReasonQuestion(
    "cme295-lect7-q31",
    "easy",
    "Assertion: An agent is more than a single tool call because it can pursue a goal through repeated observation, planning, action, and stopping decisions.\n\nReason: Later actions in an agentic workflow can depend on observations from earlier actions, which creates a control loop that a single backend operation does not have.",
    4,
    "Both statements are true, and the reason explains the distinction. A single tool call may be one action inside an agent, but the observation-dependent loop lets the workflow update state, choose the next operation, and decide whether more work is needed.",
  ),
  makeQuestion(
    "cme295-lect7-q32",
    "medium",
    'A ReAct-style workflow receives the input "The room is too cold; please fix it." Which stage descriptions are accurate?',
    [
      [
        "Observe can translate the user complaint into current knowns and unknowns, such as the room temperature being unknown.",
        true,
      ],
      [
        "Plan can identify the next subtask, such as checking the current temperature before changing settings.",
        true,
      ],
      [
        "Act can call a tool such as `get_current_room_temperature()` or `increase_temperature(value=5)`.",
        true,
      ],
      [
        "Output should be produced immediately after the first action even if the new observation shows the goal is not yet satisfied.",
        false,
      ],
    ],
    "ReAct decomposes a goal into observations, plans, actions, and eventual output. The exact terminology can vary across papers, but the core idea is iterative reasoning and acting rather than one immediate answer with no state updates.",
  ),
  makeQuestion(
    "cme295-lect7-q33",
    "medium",
    "A thermostat agent observes a room at 65 F and a target around 70 F. Which steps fit a reasonable agent loop?",
    [
      [
        "Plan to increase the temperature by about 5 F because the observed state is below the target.",
        true,
      ],
      [
        "Call an action tool such as `increase_temperature(value=5)` and then observe whether the state changed.",
        true,
      ],
      [
        "Ignore the tool result because any successful API call proves the user goal was achieved.",
        false,
      ],
      [
        "Increase by 5 F again even after observing the thermostat has reached 70 F, because fixed action counts are more important than goal state.",
        false,
      ],
    ],
    "The observed 65 F state supplies a concrete reason for a 5 F adjustment. An agent should interpret tool outputs and stop when the goal is achieved, rather than treating any API response as sufficient or looping without a termination condition.",
  ),
  makeQuestion(
    "cme295-lect7-q34",
    "hard",
    'An agent has already called a tool, received "thermostat set to 70 F," and the user\'s goal was to warm the room to about 70 F. What is the best next control decision?',
    [
      [
        "Exit the loop and summarize the completed action unless another unmet requirement is present.",
        true,
      ],
      [
        "Plan another increase to create a safety margin even though the observed state already matches the user's target.",
        false,
      ],
      [
        "Ignore the tool observation and ask the model to infer the room state from its pretrained knowledge.",
        false,
      ],
      [
        "Restart retrieval over the policy documents because a successful actuator result cannot be part of the agent state.",
        false,
      ],
    ],
    "Agentic control should check whether the goal has been reached. When the observation satisfies the target, continuing to act can create overshoot or wasted cost, while ignoring the observation defeats the purpose of the loop.",
  ),
  makeQuestion(
    "cme295-lect7-q35",
    "hard",
    "Several specialized home agents handle thermostat control, occupancy, air quality, and energy management. Which A2A-style design concerns are relevant?",
    [
      [
        "Agents need a standard way to advertise skills so other agents can discover what they can do.",
        true,
      ],
      [
        "An agent card can expose metadata such as name, URL, version, and available skills.",
        true,
      ],
      [
        "Execution interfaces need conventions for status updates and cancellation, not only final text answers.",
        true,
      ],
      [
        "Standardized agent communication can reduce bespoke wiring among independently built agents.",
        true,
      ],
    ],
    "A2A-style protocols address communication among independently built agents. Skills, agent metadata, execution, status, and cancellation are part of making agent collaboration interoperable rather than a collection of one-off private APIs.",
  ),
  makeAssertionReasonQuestion(
    "cme295-lect7-q36",
    "hard",
    "Assertion: Agent2Agent (A2A) standardization replaces the need for tools, execution logic, and cancellation handling inside each agent.\n\nReason: An AgentCard by itself executes every advertised skill and resolves every request without an AgentExecutor or implementation-specific logic.",
    3,
    "Both statements are false. A2A-style metadata can advertise an agent and its skills, but the agent still needs execution logic, status handling, cancellation behavior, and underlying tools or APIs to do real work.",
  ),
  makeQuestion(
    "cme295-lect7-q37",
    "hard",
    "A tool-using email assistant can read private files and send messages. Which risks become more serious once tool calls and agent loops can take external actions?",
    [
      [
        "A malicious instruction could try to exfiltrate private data through an outward-facing tool.",
        true,
      ],
      [
        "A wrong argument prediction could send an action to the wrong recipient or target.",
        true,
      ],
      [
        "A loop can waste money or cause repeated external effects if it lacks budget and stopping controls.",
        true,
      ],
      [
        "Action-taking eliminates hallucination because every tool call returns a real backend response.",
        false,
      ],
    ],
    "External actions increase the blast radius of model mistakes and malicious instructions. Tool outputs are useful observations, but they do not eliminate hallucination, argument errors, permission problems, or runaway loops.",
  ),
  makeQuestion(
    "cme295-lect7-q38",
    "hard",
    "Which controls are plausible parts of a safety strategy for tool-using agents?",
    [
      [
        "Training-time data or reinforcement learning that covers harmlessness and safe tool behavior.",
        true,
      ],
      [
        "Inference-time safeguards such as safety classifiers, permission checks, scopes, or approval gates.",
        true,
      ],
      [
        "Benchmarks that test agent and tool-use hazards across realistic scenarios.",
        true,
      ],
      [
        "Observability over intermediate steps so failures can be debugged and audited.",
        true,
      ],
    ],
    "Safety is layered. Training can shape behavior, inference safeguards can block dangerous outputs or actions, benchmarks can reveal failure modes, and observability makes agent decisions inspectable when something goes wrong.",
  ),
  makeAssertionReasonQuestion(
    "cme295-lect7-q39",
    "hard",
    "Assertion: Debugging tool-using agents often requires inspecting intermediate reasoning, tool predictions, arguments, observations, and stopping decisions.\n\nReason: The final answer alone always reveals which retrieval chunk, tool schema, backend call, or safety decision caused the failure.",
    1,
    "The assertion is true because failures can occur at retrieval, routing, argument prediction, execution, observation interpretation, or stopping. The reason is false: the final answer often hides the failing stage, so logs and intermediate traces are needed.",
  ),
  makeAssertionReasonQuestion(
    "cme295-lect7-q40",
    "medium",
    "Assertion: A sensible way to build a new tool-using agent is to start with a small, correct use case and a strong model before optimizing latency or cost.\n\nReason: Tool selection can use a router to choose a subset of APIs before the final model call.",
    5,
    "Both statements are true, but the reason does not explain the build-order recommendation. Starting small and using a strong model first help establish correctness and feasibility, while tool routing is a separate scalability technique for reducing context pressure.",
  ),
  makeQuestion(
    "cme295-lect7-q41",
    "hard",
    "A retrieval index chunks a 10,000-token document with chunk size 500 and overlap 100. The stride is therefore 400 tokens, and the final chunk may be shorter before padding. How many chunks are needed to cover the document?",
    [
      [
        "25 chunks, because \\(\\lceil (10000 - 500) / 400 \\rceil + 1 = 25\\).",
        true,
      ],
      [
        "20 chunks, because \\(10000 / 500 = 20\\) ignores overlap and boundary coverage.",
        false,
      ],
      [
        "24 chunks, because \\(\\lfloor (10000 - 500) / 400 \\rfloor + 1 = 24\\) drops the final partial span.",
        false,
      ],
      [
        "26 chunks, because \\(\\lceil 10000 / 400 \\rceil + 1 = 26\\) adds an extra chunk after already covering the end.",
        false,
      ],
    ],
    "With overlapping windows, the starting positions advance by the stride, not by the full chunk size. The formula needs one initial chunk plus enough additional starts to cover from token 500 through token 10,000, which gives 25 chunks.",
  ),
  makeQuestion(
    "cme295-lect7-q42",
    "hard",
    "A reranker returns three chunks with graded relevance scores \\([2, 0, 1]\\). Using \\(DCG@3 = \\sum_i rel_i / \\log_2(i+1)\\) and \\(NDCG@3 = DCG@3 / IDCG@3\\), what is the closest NDCG@3?",
    [
      [
        "\\(2.5 / (2 + 1 / \\log_2 3) \\approx 0.95\\), because the ideal order is \\([2,1,0]\\).",
        true,
      ],
      [
        "\\((2 + 1 / \\log_2 3) / 2.5 \\approx 1.05\\), because the observed ranking is used as the ideal order.",
        false,
      ],
      [
        "\\((2 + 0 + 1) / (2 + 1 + 0) = 1.00\\), because NDCG ignores rank discounting.",
        false,
      ],
      [
        "\\((2 / \\log_2 3 + 1 / \\log_2 4) / (2 + 1 / \\log_2 3) \\approx 0.61\\), because rank 1 is discounted by \\(\\log_2 3\\).",
        false,
      ],
    ],
    "The observed DCG is \\(2/\\log_2 2 + 0/\\log_2 3 + 1/\\log_2 4 = 2.5\\). The ideal order places relevance 2 first and relevance 1 second, so the normalized score is about 0.95 rather than a raw relevance sum.",
  ),
  makeQuestion(
    "cme295-lect7-q43",
    "hard",
    "A retrieval benchmark has 6 relevant chunks in the corpus. The top five retrieved chunks have binary relevance \\([0, 1, 0, 1, 1]\\). Which metric triple is correct for the top five?",
    [
      [
        "Precision@5 = \\(3/5\\), Recall@5 = \\(3/6\\), and RR@5 = \\(1/2\\).",
        true,
      ],
      [
        "Precision@5 = \\(3/6\\), Recall@5 = \\(3/5\\), and RR@5 = \\(1/3\\).",
        false,
      ],
      [
        "Precision@5 = \\(1/2\\), Recall@5 = \\(3/5\\), and RR@5 = \\(3/5\\).",
        false,
      ],
      [
        "Precision@5 = \\(2/5\\), Recall@5 = \\(2/6\\), and RR@5 = \\(1/2\\).",
        false,
      ],
    ],
    "Three of the five retrieved chunks are relevant, so Precision@5 is 3/5. Those three cover half of the six relevant chunks in the corpus, and the first relevant chunk appears at rank 2, so reciprocal rank is 1/2.",
  ),
  makeQuestion(
    "cme295-lect7-q44",
    "hard",
    "A semantic retriever embeds a query as \\(q=(1,2,2)\\). Which candidate has the highest cosine similarity to the query?",
    [
      [
        "\\(a=(2,4,4)\\), because it points in the same direction as \\(q\\) and has cosine similarity 1.",
        true,
      ],
      [
        "\\(b=(2,0,0)\\), because its dot product is positive and its first coordinate is largest.",
        false,
      ],
      [
        "\\(c=(0,2,2)\\), because it shares two coordinates but omits the query's first component.",
        false,
      ],
      [
        "\\(d=(1,1,0)\\), because its Euclidean length is shortest and therefore dominates cosine similarity.",
        false,
      ],
    ],
    "Cosine similarity depends on vector direction after normalizing by lengths. Candidate \\(a\\) is exactly \\(2q\\), so its cosine similarity is 1; the other candidates are plausible neighbors but point in less aligned directions.",
  ),
  makeQuestion(
    "cme295-lect7-q45",
    "hard",
    "A cross-encoder takes 25 ms to score one query-chunk pair. What is the computational reason for applying it to 100 retrieved candidates instead of directly to 1,000,000 indexed chunks?",
    [
      [
        "Scoring 100 candidates takes about 2.5 seconds, while scoring 1,000,000 chunks takes about 6.94 hours.",
        true,
      ],
      [
        "Scoring 100 candidates takes about 250 seconds, while scoring 1,000,000 chunks takes about 25 minutes.",
        false,
      ],
      [
        "Scoring 100 candidates and 1,000,000 chunks costs the same because the cross-encoder shares one attention pass.",
        false,
      ],
      [
        "Scoring 1,000,000 chunks takes about 25 seconds, so candidate retrieval is mainly a security feature.",
        false,
      ],
    ],
    "A cross-encoder is powerful because it jointly reads the query and each candidate, but that cost is paid per pair. Candidate retrieval makes reranking tractable by reducing the scoring set by a factor of 10,000 in this example.",
  ),
  makeQuestion(
    "cme295-lect7-q46",
    "hard",
    "A prompt has 3,200 available tokens for retrieved chunks after reserving space for instructions, the user query, and the answer. Each included chunk costs 380 content tokens plus a 20-token separator. Which prompt-assembly statements are correct?",
    [
      [
        "The system can include at most \\(\\lfloor 3200 / 400 \\rfloor = 8\\) chunks.",
        true,
      ],
      [
        "Including 9 chunks would consume \\(9 \\times 400 = 3600\\) tokens, exceeding the available chunk budget by 400.",
        true,
      ],
      [
        "The system can include 9 chunks if it budgets each chunk as \\(3200/9 \\approx 356\\) tokens and treats the 20-token separators as free metadata.",
        false,
      ],
      [
        "The optimal top-k is \\(k=8\\) for every query because retrieval metrics are computed before prompt assembly.",
        false,
      ],
    ],
    "Retrieval quality and prompt assembly are coupled by the context budget. Even if the ninth chunk is relevant, it cannot be included under this budget unless chunk size, separators, reserved space, or top-k policy changes.",
  ),
  makeQuestion(
    "cme295-lect7-q47",
    "hard",
    "On a 100-query diagnostic set, semantic retrieval finds a relevant chunk for 72 queries, BM25 finds one for 55 queries, and both methods succeed on 40 of the same queries. Which hybrid-retrieval calculations are correct?",
    [
      ["The union succeeds on \\(72 + 55 - 40 = 87\\) queries.", true],
      [
        "Adding BM25 to semantic retrieval contributes \\(87 - 72 = 15\\) additional successes on this set.",
        true,
      ],
      [
        "The union succeeds on \\(72 + 55 = 127\\) queries because overlap should be counted twice.",
        false,
      ],
      [
        "BM25 contributes \\(0\\) value because its 55 successes are assumed to be contained inside the semantic retriever's 72 successes.",
        false,
      ],
    ],
    "Hybrid retrieval is often justified by complementary errors, not by either method being universally superior. The overlap must be subtracted once, and the remaining 15 BM25-only successes are exactly the kind of exact-term benefit a hybrid system can preserve.",
  ),
  makeQuestion(
    "cme295-lect7-q48",
    "hard",
    "A vector index stores 40,000 chunk embeddings with dimension 1,536. Assume float32 storage uses 4 bytes per number and ignore index overhead. Which storage estimates are correct?",
    [
      [
        "The raw embedding matrix uses \\(40000 \\times 1536 \\times 4 = 245{,}760{,}000\\) bytes, about 245.8 MB in decimal units.",
        true,
      ],
      [
        "Using float16 instead would roughly halve the raw vector storage to about 122.9 MB before index overhead.",
        true,
      ],
      [
        "The raw embedding matrix uses \\(40000 \\times 1536 \\times 32\\) bytes, about 2.0 GB, because each float32 value is mistaken for 32 bytes.",
        false,
      ],
      [
        "Changing chunk overlap from \\(100\\) to \\(0\\) has no possible effect on storage because the embedding dimension remains \\(1536\\).",
        false,
      ],
    ],
    "Embedding dimension controls bytes per chunk, while chunking choices control how many chunks exist. Float precision changes bytes per value, and overlap can increase chunk count even when the vector dimension stays fixed.",
  ),
  makeQuestion(
    "cme295-lect7-q49",
    "hard",
    "A contextual-retrieval job creates context for 200 chunks from the same 18,000-token document. Each call also includes 120 unique chunk tokens. If prompt caching makes the repeated document prefix bill at 20% of normal input-token cost, which effective-token calculations are correct?",
    [
      [
        "Without caching, the job sends \\(200 \\times (18000 + 120) = 3{,}624{,}000\\) input tokens.",
        true,
      ],
      [
        "With the 20% cached-prefix rate, the effective input-token bill is \\((200 \\times 120) + 0.2 \\times (200 \\times 18000) = 744{,}000\\).",
        true,
      ],
      [
        "With caching, the effective bill is \\(200 \\times 120 = 24{,}000\\) tokens because the whole-document prefix becomes completely free.",
        false,
      ],
      [
        "Prompt caching changes the \\(200\\) chunk embeddings directly, so the repeated-prefix input bill can be approximated as \\(200 \\times 0\\).",
        false,
      ],
    ],
    "Prompt caching is a cost model for repeated prompt material, not a retrieval model or an embedding transform. The unique chunk text is still paid normally, while the repeated document prefix is discounted rather than removed entirely in this setup.",
  ),
  makeQuestion(
    "cme295-lect7-q50",
    "hard",
    "A tool router selects 8 tools for a request. Four selected tools are actually relevant, and there are 5 relevant tools in the full tool catalog. The catalog has 900 tools, each full schema averages 140 tokens, and each short router entry averages 12 tokens. Which statements are correct?",
    [
      [
        "Router precision is \\(4/8 = 0.50\\), and router recall is \\(4/5 = 0.80\\).",
        true,
      ],
      [
        "The final call sees about \\((900 \\times 12) + (8 \\times 140) = 11{,}920\\) tool-description tokens instead of \\(900 \\times 140 = 126{,}000\\).",
        true,
      ],
      [
        "Router recall is \\(8/900\\) because recall divides selected tools by all available tools.",
        false,
      ],
      [
        "The final call must still include \\(900 \\times 140 = 126{,}000\\) full-schema tokens because routing only changes tool order.",
        false,
      ],
    ],
    "Tool selection can be evaluated like a retrieval problem: precision asks how many selected tools were useful, and recall asks how many needed tools survived. The context calculation shows why routing can improve latency and focus even when the router is not perfect.",
  ),
  makeQuestion(
    "cme295-lect7-q51",
    "hard",
    "A RAG pipeline has three conditional probabilities: candidate retrieval includes the needed chunk with probability 0.92; given inclusion, reranking places it in the top 4 with probability 0.85; given top-4 placement, prompt assembly includes it with probability 0.95. Which reliability calculations are correct?",
    [
      [
        "The probability that the needed chunk reaches the prompt is \\(0.92 \\times 0.85 \\times 0.95 \\approx 0.743\\).",
        true,
      ],
      [
        "If a top-2 prompt cutoff gives conditional inclusion 0.70 instead of 0.95, the prompt evidence probability drops to \\(0.92 \\times 0.85 \\times 0.70 \\approx 0.547\\).",
        true,
      ],
      [
        "Moving from the top-2 cutoff to the top-4 cutoff improves this evidence probability by about 19.6 percentage points.",
        true,
      ],
      [
        "The final evidence probability is \\(0.92\\) because candidate recall already lets the downstream factors be replaced by \\(1\\).",
        false,
      ],
    ],
    "Pipeline reliability compounds across stages. A strong candidate retriever can still lose evidence through reranking or prompt assembly, so stage-specific logs and probabilities are more informative than a single retrieval-recall number.",
  ),
  makeQuestion(
    "cme295-lect7-q52",
    "hard",
    "An agent must perform four independent steps correctly: infer location with probability 0.98, call the right weather tool with probability 0.94, pass an approval check with probability 0.90, and synthesize the result faithfully with probability 0.97. Which statements follow?",
    [
      [
        "The end-to-end success probability is \\(0.98 \\times 0.94 \\times 0.90 \\times 0.97 \\approx 0.804\\).",
        true,
      ],
      [
        "The failure probability is about \\(1 - 0.804 = 0.196\\), even though each individual step is fairly reliable.",
        true,
      ],
      [
        "Improving the approval step from 0.90 to 0.99 raises success to about \\(0.98 \\times 0.94 \\times 0.99 \\times 0.97 \\approx 0.894\\).",
        true,
      ],
      [
        "The end-to-end success probability is the average \\((0.98 + 0.94 + 0.90 + 0.97)/4 \\approx 0.948\\).",
        false,
      ],
    ],
    "Agent workflows multiply reliability across required steps, so small independent failure rates can accumulate into a meaningful end-to-end failure rate. Averaging step accuracies hides that compounding risk and overstates workflow reliability.",
  ),
  makeQuestion(
    "cme295-lect7-q53",
    "hard",
    "A safety classifier reviews 10,000 proposed tool actions. Two percent are malicious. The classifier blocks 95% of malicious actions and falsely blocks 3% of benign actions. Which confusion-matrix statements are correct?",
    [
      ["It blocks \\(0.95 \\times 200 = 190\\) malicious actions.", true],
      ["It misses \\(0.05 \\times 200 = 10\\) malicious actions.", true],
      [
        "It falsely blocks \\(0.03 \\times 9800 = 294\\) benign actions, so only \\(190/(190+294) \\approx 39\\%\\) of blocked actions are truly malicious.",
        true,
      ],
      [
        "Because the true-positive rate is \\(0.95\\), the precision of blocked actions must be at least \\(0.95\\) as well.",
        false,
      ],
    ],
    "Low base rates matter in safety operations. Even a classifier with high malicious-action recall can generate many benign blocks when malicious actions are rare, so teams need both safety metrics and operational review processes.",
  ),
  makeQuestion(
    "cme295-lect7-q54",
    "hard",
    "A home system has 5 specialized agents. If every ordered pair of agents needs a bespoke directed integration, there are \\(n(n-1)\\) directed links. A coordinator-style protocol needs two directed links per non-coordinator agent. Which communication-scaling statements are correct?",
    [
      [
        "Full bespoke directed integration requires \\(5 \\times 4 = 20\\) links.",
        true,
      ],
      [
        "A coordinator-style setup requires \\(2 \\times (5-1) = 8\\) directed links.",
        true,
      ],
      [
        "Moving from 20 to 8 links is a 60% reduction in directed integration surfaces.",
        true,
      ],
      [
        "A standard protocol reduces the link count to \\(8\\), so it removes the need for any agent to describe skills, status, or cancellation behavior.",
        false,
      ],
    ],
    "The scaling problem is a reason to standardize agent communication, but standardization still needs concrete metadata and execution behavior. Agent cards, skills, status, and cancellation are part of making the reduced integration surface usable.",
  ),
  makeQuestion(
    "cme295-lect7-q55",
    "hard",
    "An agent has a 9,000-token budget for a task and reserves 1,200 tokens for the final answer and audit summary. Each observe-plan-act iteration consumes about 1,800 tokens. Which budget statements are correct?",
    [
      [
        "The agent can afford \\(\\lfloor (9000 - 1200) / 1800 \\rfloor = 4\\) full iterations before the reserved final response.",
        true,
      ],
      [
        "Allowing 5 full iterations would require \\((5 \\times 1800) + 1200 = 10{,}200\\) tokens, exceeding the budget.",
        true,
      ],
      [
        "A stop condition after the fourth iteration can be a cost-control mechanism, not only a reasoning decision.",
        true,
      ],
      [
        "Token budgets are irrelevant to agent loops because each tool call resets the \\(9000\\)-token context for free.",
        false,
      ],
    ],
    "Agent loops are limited by both correctness and resource budgets. A system that can act repeatedly needs explicit stopping and budget controls, especially when observations, plans, tool schemas, and audit traces all consume context.",
  ),
  makeQuestion(
    "cme295-lect7-q56",
    "hard",
    "A RAG system indexes a 12,000-token manual with chunk size 600 and overlap 100. It later includes the top 6 chunks in a prompt. Which calculations are correct?",
    [
      ["The stride is \\(600 - 100 = 500\\) tokens.", true],
      [
        "The chunk count is \\(\\lceil (12000 - 600) / 500 \\rceil + 1 = 24\\).",
        true,
      ],
      [
        "The indexed chunk text totals about \\(24 \\times 600 = 14{,}400\\) chunk-token positions before embedding.",
        true,
      ],
      [
        "Putting the top 6 chunks into the prompt uses about \\(6 \\times 600 = 3{,}600\\) retrieved-context tokens before separators or metadata.",
        true,
      ],
    ],
    "This calculation connects offline indexing choices to online prompt cost. Overlap increases indexed token positions beyond the original document length, and top-k prompt inclusion turns chunk size directly into context-window pressure.",
  ),
  makeQuestion(
    "cme295-lect7-q57",
    "hard",
    "A query has four relevant chunks in the corpus. A system returns top-5 binary relevance \\([1,0,1,0,0]\\). Which metric statements are correct?",
    [
      ["Precision@5 is \\(2/5 = 0.40\\).", true],
      ["Recall@5 is \\(2/4 = 0.50\\).", true],
      ["RR@5 is 1 because the first returned chunk is relevant.", true],
      [
        "A reranker that swaps the rank-3 relevant chunk to rank 2 would improve rank-sensitive metrics while leaving Precision@5 and Recall@5 unchanged.",
        true,
      ],
    ],
    "Top-k metrics expose different failure modes. Precision and recall count relevant items in the set, while reciprocal rank and NDCG care about where relevant items appear inside the ranked list.",
  ),
  makeQuestion(
    "cme295-lect7-q58",
    "hard",
    "A calendar tool schema is being written for an assistant that may schedule real meetings. Which schema and control details would reduce argument and action errors?",
    [
      [
        "Require structured fields such as `start_time_iso`, `duration_minutes`, `attendee_emails`, and `timezone` instead of one free-form string.",
        true,
      ],
      [
        'Use enums or constrained values such as `visibility_enum = "private"` and `meeting_type_id` when the backend supports only a fixed set.',
        true,
      ],
      [
        "Return a preview object with fields such as `requires_approval = true` and `action_id` before committing the action when human approval is required.",
        true,
      ],
      [
        "Include stable `event_id` values in tool outputs so later `update_event(event_id)` or `cancel_event(event_id)` calls can target the correct meeting.",
        true,
      ],
    ],
    "Tool schemas are part of the reliability surface. Clear argument types, constrained values, previews, approvals, and stable IDs help the model use tools correctly and give the backend enough structure to enforce safety.",
  ),
  makeQuestion(
    "cme295-lect7-q59",
    "hard",
    "A contextual retrieval pipeline spends 0.8 seconds to generate context for each chunk and 0.04 seconds to embed each contextualized chunk. It processes 500 chunks offline, then answers 10,000 online queries. Which engineering conclusions are correct?",
    [
      [
        "The offline contextualization-and-embedding job takes about \\(500 \\times (0.8 + 0.04) = 420\\) seconds before indexing overhead.",
        true,
      ],
      [
        "The \\(420\\)-second offline cost can be worthwhile if it improves retrieval quality for \\(10{,}000\\) later online queries.",
        true,
      ],
      [
        "If the context generation were done online for every query and every chunk, it would add \\(10000 \\times 500 \\times 0.8\\) seconds of generation work.",
        true,
      ],
      [
        "The pipeline should still evaluate metrics such as \\(Recall@k\\) or \\(NDCG@k\\) because faster embedding does not prove that contextualized chunks rank correctly.",
        true,
      ],
    ],
    "Contextual retrieval is often an offline indexing investment intended to improve online retrieval. The arithmetic shows why doing the same generation work inside every query path would be prohibitive, and why quality metrics are still needed after optimization.",
  ),
  makeQuestion(
    "cme295-lect7-q60",
    "hard",
    "A deployed email agent handles 50,000 send actions per month. A severe mis-send costs about USD 2,000 in expected damage. Without extra safeguards, the mis-send probability is estimated at 0.08%; with approval gates and scoped tools, it falls to 0.01% while adding USD 3,000 per month in review cost. Which risk calculations are correct?",
    [
      [
        "Expected monthly loss without safeguards is \\(50000 \\times 0.0008 \\times 2000 = \\text{USD }80{,}000\\).",
        true,
      ],
      [
        "Expected monthly loss with safeguards is \\(50000 \\times 0.0001 \\times 2000 = \\text{USD }10{,}000\\), before review cost.",
        true,
      ],
      [
        "Including review cost, the safeguarded expected monthly cost is about \\(10000 + 3000 = \\text{USD }13{,}000\\), far below USD 80,000.",
        true,
      ],
      [
        "The expected avoided loss is about \\(80000 - 13000 = \\text{USD }67{,}000\\) per month, enough that the approval gate can be justified economically before considering trust and compliance benefits.",
        true,
      ],
    ],
    "External-action safety can be analyzed with expected loss, not only qualitative concern. The calculation does not prove the exact probabilities are correct, but it shows how scopes, approvals, and review costs can be compared against the cost of preventable harmful actions.",
  ),
];
