import { Question } from "../../../quiz";

export const AtomMemBuildingSimpleAndEffectiveMemorySystemQuestions: Question[] =
  [
    {
      id: "ai-agents-atommem-q01",
      chapter: 1,
      difficulty: "easy",
      prompt:
        "Which design pressures motivate an AtomMem-style long-term memory system for Large Language Model (LLM) agents?",
      options: [
        {
          text: "Fixed context windows make it difficult to accumulate and reuse information across multi-session interactions.",
          isCorrect: true,
        },
        {
          text: "Raw conversation storage keeps detail but floods retrieval with redundant low-utility context.",
          isCorrect: true,
        },
        {
          text: "Condensed summaries preserve every fine-grained detail while eliminating the need for later retrieval.",
          isCorrect: false,
        },
        {
          text: "Unconstrained memory rewrites guarantee stability because hallucinated edits are automatically removed during later retrieval.",
          isCorrect: false,
        },
      ],
      explanation:
        "AtomMem is motivated by the tension between keeping enough evidence and keeping retrieval efficient. Raw logs preserve detail but add noise, aggressive compression can discard evidence, and unconstrained LLM rewrites can destabilize rather than guarantee memory quality.",
    },
    {
      id: "ai-agents-atommem-q02",
      chapter: 1,
      difficulty: "easy",
      prompt:
        "Which statements correctly describe atomic facts as the base representation in AtomMem?",
      options: [
        {
          text: "They are self-contained memory units extracted from noisy dialogue streams.",
          isCorrect: true,
        },
        {
          text: "They resolve context-dependent references such as pronouns or relative dates before storage.",
          isCorrect: true,
        },
        {
          text: "They require keeping the entire original session transcript attached to every retrieved fact.",
          isCorrect: false,
        },
        {
          text: "They are raw dialogue turns stored verbatim so later components can infer missing context from the original transcript.",
          isCorrect: false,
        },
      ],
      explanation:
        "Atomic facts are designed as compact, standalone facts, not as raw transcripts with full session history attached. Resolving references and retaining high-value content makes each unit easier to retrieve and combine later without depending on nearby turns.",
    },
    {
      id: "ai-agents-atommem-q03",
      chapter: 1,
      difficulty: "medium",
      type: "assertion-reason",
      prompt:
        "Assertion: AtomMem trains a supervised fact extractor instead of relying only on heuristics or zero-shot prompting.\n\nReason: Extracting useful atomic facts requires denoising and lightweight reasoning such as coreference resolution and temporal anchoring.",
      options: [
        { text: "Assertion is true, Reason is false.", isCorrect: false },
        { text: "Assertion is false, Reason is true.", isCorrect: false },
        { text: "Both are false.", isCorrect: false },
        {
          text: "Both are true, and the Reason is the correct explanation of the Assertion.",
          isCorrect: true,
        },
        {
          text: "Both are true, but the Reason is NOT the correct explanation of the Assertion.",
          isCorrect: false,
        },
      ],
      explanation:
        "Both statements are true, and the reason explains the design choice. The extractor has to filter noise and rewrite context-dependent dialogue into standalone facts, which is hard to do reliably with simple rules or an unconstrained prompt alone.",
    },
    {
      id: "ai-agents-atommem-q04",
      chapter: 1,
      difficulty: "easy",
      prompt:
        "Which fields belong to the structured atomic fact representation used by AtomMem?",
      options: [
        {
          text: "A fact identifier, self-contained text, and dense semantic embedding.",
          isCorrect: true,
        },
        {
          text: "Participants, topical keywords, and temporal information.",
          isCorrect: true,
        },
        {
          text: "Associated event identifiers that link the fact to higher-level event memory.",
          isCorrect: true,
        },
        {
          text: "A permanent profile decision that cannot later be revised or assigned historical validity.",
          isCorrect: false,
        },
      ],
      explanation:
        "The structured fact stores both readable content and metadata for retrieval and organization. Stable user profiles are a separate layer, and profile history is explicitly handled outside the base fact object.",
    },
    {
      id: "ai-agents-atommem-q05",
      chapter: 1,
      difficulty: "medium",
      prompt:
        "A newly extracted fact overlaps with earlier memory but also contains a small novel detail. Which verification behavior matches AtomMem?",
      options: [
        {
          text: "Filter existing memory by shared metadata, retrieve close candidates, and store only the residual novel content while updating conflicting prior facts when needed.",
          isCorrect: true,
        },
        {
          text: "Append the full new text without checking related entries because duplicate facts strengthen later similarity search.",
          isCorrect: false,
        },
        {
          text: "Discard the new fact whenever any semantically similar memory already exists, even if the new fact contains a changed value or time.",
          isCorrect: false,
        },
        {
          text: "Rewrite the entire memory store so all facts share one current summary before the fact can be retrieved.",
          isCorrect: false,
        },
      ],
      explanation:
        "AtomMem verifies new facts against candidate memories before storage. The goal is to reduce redundancy while preserving novel information and explicit updates, rather than blindly appending, blindly discarding, or globally rewriting the store.",
    },
    {
      id: "ai-agents-atommem-q06",
      chapter: 1,
      difficulty: "hard",
      prompt:
        "AtomMem ranks fact candidates with a hybrid score that combines semantic embedding similarity and keyword Jaccard overlap. Which interpretations of that scoring choice are correct?",
      options: [
        {
          text: "The semantic term helps match paraphrased facts whose wording differs from the query or new fact.",
          isCorrect: true,
        },
        {
          text: "The keyword Jaccard term adds symbolic overlap so topical metadata can influence candidate ranking.",
          isCorrect: true,
        },
        {
          text: "The weighted combination lets the system tune the relative influence of dense semantics and explicit keywords.",
          isCorrect: true,
        },
        {
          text: "Metadata filtering remains useful before hybrid ranking because participant and time constraints can remove irrelevant comparisons early.",
          isCorrect: true,
        },
      ],
      explanation:
        "The hybrid score mixes dense semantic similarity with explicit keyword overlap, so it can use both paraphrase-level and symbolic signals. Metadata filtering still matters because it narrows the search space and keeps comparisons contextually relevant before ranking.",
    },
    {
      id: "ai-agents-atommem-q07",
      chapter: 1,
      difficulty: "medium",
      prompt:
        "Which statements correctly describe event memory construction in AtomMem?",
      options: [
        {
          text: "Events aggregate related atomic facts into coherent narrative blocks.",
          isCorrect: true,
        },
        {
          text: "A verified new fact can be absorbed into existing events or paired with standalone facts to create new events.",
          isCorrect: true,
        },
        {
          text: "Event metadata is intentionally limited to a summary string, so participants, keywords, and temporal span are not tracked.",
          isCorrect: false,
        },
        {
          text: "Event memory replaces atomic facts, so individual fact identifiers are discarded after event creation.",
          isCorrect: false,
        },
      ],
      explanation:
        "Events provide episodic continuity over precise atomic details. They keep links to constituent fact identifiers and add summaries plus metadata such as participants, keywords, and temporal span, so event memory organizes facts without deleting the base evidence units.",
    },
    {
      id: "ai-agents-atommem-q08",
      chapter: 1,
      difficulty: "medium",
      prompt:
        "A user first prefers quiet hotels, then later says conference travel has made downtown hotels more useful. Which temporal-profile behavior best fits AtomMem?",
      options: [
        {
          text: "Queue candidate long-term preference facts during the session, then update the profile in batch while preserving the earlier profile version with validity metadata.",
          isCorrect: true,
        },
        {
          text: "Treat both statements as unrelated events because user profiles cannot contain historical versions.",
          isCorrect: false,
        },
        {
          text: "Overwrite the old preference immediately and erase its supporting facts so retrieval cannot expose stale information.",
          isCorrect: false,
        },
        {
          text: "Store both statements only as raw dialogue because stable attributes should not be embedded or keyworded.",
          isCorrect: false,
        },
      ],
      explanation:
        "AtomMem builds temporal profile memory for stable but changing user attributes. It can update the current profile while saving the prior state in history and retaining supporting fact identifiers for traceability.",
    },
    {
      id: "ai-agents-atommem-q09",
      chapter: 1,
      difficulty: "easy",
      prompt:
        "Which edge types are used when AtomMem activates its memory graph over atomic facts?",
      options: [
        {
          text: "Entity edges based on shared keywords, with weights adjusted for specificity and query relevance.",
          isCorrect: true,
        },
        {
          text: "Event edges between facts that belong to the same event.",
          isCorrect: true,
        },
        {
          text: "Temporal edges between facts that share any calendar date, even when they come from unrelated sessions.",
          isCorrect: false,
        },
        {
          text: "Gradient edges that connect facts according to changes in the base LLM's hidden weights.",
          isCorrect: false,
        },
      ],
      explanation:
        "AtomMem's graph is built from interpretable relations among stored facts: shared entities or keywords, shared events, and local dialogue continuity. Calendar date alone is not the temporal-edge rule, and the graph does not require inspecting base-model weight updates.",
    },
    {
      id: "ai-agents-atommem-q10",
      chapter: 1,
      difficulty: "medium",
      prompt:
        "Why does AtomMem weight keyword-based entity edges with query-aware inverse-document-frequency style terms?",
      options: [
        {
          text: "To boost keywords that are relevant to the current query.",
          isCorrect: true,
        },
        {
          text: "To penalize frequent conversational terms that would otherwise create noisy bridges between unrelated facts.",
          isCorrect: true,
        },
        {
          text: "To make topical overlap more useful than raw keyword sharing alone.",
          isCorrect: true,
        },
        {
          text: "To keep keyword edges useful alongside event and temporal edges instead of letting generic keyword overlap dominate the graph.",
          isCorrect: true,
        },
      ],
      explanation:
        "Simple keyword overlap can overconnect memory through common terms. AtomMem adjusts entity-edge weights so query-relevant and specific terms matter more, while frequent low-information terms have less influence; the other edge channels still remain part of graph recall.",
    },
    {
      id: "ai-agents-atommem-q11",
      chapter: 1,
      difficulty: "medium",
      prompt:
        "What problem is addressed by penalizing event-edge weights for large events?",
      options: [
        {
          text: "Large events can create overly broad connections, so size penalties reduce the chance that event membership alone dominates recall.",
          isCorrect: true,
        },
        {
          text: "Large events are impossible to summarize, so their facts must be removed from the graph.",
          isCorrect: false,
        },
        {
          text: "Event edges are meant to encode local turn distance, so event size is a proxy for dialogue position.",
          isCorrect: false,
        },
        {
          text: "The penalty converts every event edge into a pure keyword edge before random-walk propagation.",
          isCorrect: false,
        },
      ],
      explanation:
        "Event edges help recover episodic context even when keywords differ, but broad events can connect too many facts. Penalizing large events keeps event co-membership useful without letting large narrative blocks swamp more precise evidence.",
    },
    {
      id: "ai-agents-atommem-q12",
      chapter: 1,
      difficulty: "easy",
      prompt: "Which statements correctly describe AtomMem's temporal edges?",
      options: [
        {
          text: "They connect facts from nearby dialogue turns inside the same session.",
          isCorrect: true,
        },
        {
          text: "Their weights increase as turn distance grows because far-apart turns are treated as stronger continuity evidence.",
          isCorrect: false,
        },
        {
          text: "They are bounded by a maximum turn window.",
          isCorrect: true,
        },
        {
          text: "They connect all facts with the same calendar date regardless of dialogue position or session.",
          isCorrect: false,
        },
      ],
      explanation:
        "Temporal edges preserve short-range conversational continuity rather than broad calendar similarity. The edge weight decays with turn distance, and the window constraint lets nearby turns help recall while limiting unrelated long-distance links.",
    },
    {
      id: "ai-agents-atommem-q13",
      chapter: 1,
      difficulty: "easy",
      prompt:
        "Which pieces of information are extracted during AtomMem's query intent analysis?",
      options: [
        {
          text: "A flag indicating whether profile memory is needed.",
          isCorrect: true,
        },
        {
          text: "Participants involved in the query.",
          isCorrect: true,
        },
        {
          text: "Core intent keywords and any relevant time range.",
          isCorrect: true,
        },
        {
          text: "The final answer text, generated before any memory retrieval occurs.",
          isCorrect: false,
        },
      ],
      explanation:
        "Intent analysis converts the raw user query into retrieval controls such as profile need, participants, keywords, and time constraints. The final answer is generated later after facts and optional profiles have been retrieved.",
    },
    {
      id: "ai-agents-atommem-q14",
      chapter: 1,
      difficulty: "medium",
      prompt:
        "Which stages belong to AtomMem's hierarchical hybrid retrieval process?",
      options: [
        {
          text: "Primary recall filters and ranks facts directly against the parsed query.",
          isCorrect: true,
        },
        {
          text: "Compensatory recall uses the event layer to recover facts that direct fact retrieval may miss.",
          isCorrect: true,
        },
        {
          text: "Associative recall expands from seed facts through the memory graph and ranks activated facts.",
          isCorrect: true,
        },
        {
          text: "Profile augmentation runs only by deleting all episodic facts and answering from stable attributes alone.",
          isCorrect: false,
        },
      ],
      explanation:
        "The retrieval pipeline starts with direct fact recall, adds event-mediated compensatory facts, then propagates through graph structure. Profile augmentation is optional and additive; it does not replace the episodic fact context.",
    },
    {
      id: "ai-agents-atommem-q15",
      chapter: 1,
      difficulty: "hard",
      prompt:
        "During compensatory recall, candidate facts from matched events are ranked by a fusion of event relevance and fact-level relevance. Which failure modes correspond to poor fusion weights?",
      options: [
        {
          text: "Too little event weight makes compensatory recall resemble ordinary fact-level semantic search and underuse broader episodic context.",
          isCorrect: true,
        },
        {
          text: "Too much event weight can retrieve facts from a relevant event that lack direct alignment with the specific query.",
          isCorrect: true,
        },
        {
          text: "A balanced event/fact fusion is itself a poor-weight failure mode because it prevents either signal from contributing.",
          isCorrect: false,
        },
        {
          text: "Any nonzero event weight prevents direct primary recall from contributing seed facts to graph retrieval.",
          isCorrect: false,
        },
      ],
      explanation:
        "Compensatory recall is meant to add missing episode context without losing query precision. Poor weights either collapse the stage into ordinary semantic search or drift toward loosely related event members; a balanced fusion is the mitigation, not the failure.",
    },
    {
      id: "ai-agents-atommem-q16",
      chapter: 1,
      difficulty: "hard",
      type: "assertion-reason",
      prompt:
        "Assertion: AtomMem's graph retrieval uses Random Walk with Restart so activation can spread to indirectly related facts while staying anchored to the query seeds.\n\nReason: The restart distribution is personalized from primary and compensatory seed scores, and the restart probability controls how strongly the walk returns to those seeds.",
      options: [
        { text: "Assertion is true, Reason is false.", isCorrect: false },
        { text: "Assertion is false, Reason is true.", isCorrect: false },
        { text: "Both are false.", isCorrect: false },
        {
          text: "Both are true, and the Reason is the correct explanation of the Assertion.",
          isCorrect: true,
        },
        {
          text: "Both are true, but the Reason is NOT the correct explanation of the Assertion.",
          isCorrect: false,
        },
      ],
      explanation:
        "Both statements are true, and the reason explains the anchoring mechanism. Personalized restarts let the graph surface nearby associated evidence while reducing semantic drift away from high-confidence seed facts.",
    },
    {
      id: "ai-agents-atommem-q17",
      chapter: 1,
      difficulty: "medium",
      prompt:
        "When should AtomMem retrieve temporal profile versions rather than only the current profile value?",
      options: [
        {
          text: "When the query includes a time constraint and the answer should reflect the user state valid at that time.",
          isCorrect: true,
        },
        {
          text: "When a stable profile attribute changed and the prior state remains relevant to a historical question.",
          isCorrect: true,
        },
        {
          text: "When every retrieved profile lacks supporting fact identifiers, making historical validity the substitute for provenance.",
          isCorrect: false,
        },
        {
          text: "When every episodic fact must be converted into a profile before response generation can begin.",
          isCorrect: false,
        },
      ],
      explanation:
        "Temporal profile memory keeps current and historical versions so the agent can answer time-sensitive questions about the user's state. Supporting fact identifiers provide provenance; historical validity is not a replacement for traceability.",
    },
    {
      id: "ai-agents-atommem-q18",
      chapter: 1,
      difficulty: "easy",
      prompt:
        "Which context sources can AtomMem combine when generating a memory-aware response?",
      options: [
        {
          text: "Retrieved episodic facts selected through hierarchical and graph-based recall.",
          isCorrect: true,
        },
        {
          text: "Profile statements when query intent indicates that stable user attributes are needed.",
          isCorrect: true,
        },
        {
          text: "Event summaries are omitted by design because graph recall alone is expected to preserve all broader episode context.",
          isCorrect: false,
        },
        {
          text: "Unverified newly generated facts that bypass extraction, verification, and memory storage.",
          isCorrect: false,
        },
      ],
      explanation:
        "Response generation is grounded in retrieved facts and optional profile context, with event-level organization available where it helps shape evidence. AtomMem's design emphasizes verified memory units rather than injecting unverified generated content as if it were stored evidence.",
    },
    {
      id: "ai-agents-atommem-q19",
      chapter: 1,
      difficulty: "easy",
      prompt:
        "Which evaluation choices were used to compare AtomMem with other long-term memory systems?",
      options: [
        {
          text: "LoCoMo was used to test long conversational memory across categories such as single-hop, multi-hop, temporal, and open-domain questions.",
          isCorrect: true,
        },
        {
          text: "LongMemEval was used as an independent supplementary benchmark for interactive memory capabilities.",
          isCorrect: true,
        },
        {
          text: "Only token-level F1 was reported, so semantic correctness, lexical overlap, and memory cost were left unmeasured.",
          isCorrect: false,
        },
        {
          text: "Every baseline was allowed to use a different backbone model so each system could be evaluated in its preferred configuration.",
          isCorrect: false,
        },
      ],
      explanation:
        "The evaluation combines several answer-quality views with cost-oriented token accounting rather than relying only on F1. Baselines were re-implemented under a uniform GPT-4o-mini backbone to make architectural comparisons less confounded by different generation models.",
    },
    {
      id: "ai-agents-atommem-q20",
      chapter: 1,
      difficulty: "medium",
      prompt:
        "Which experimental details help isolate the contribution of AtomMem's memory architecture?",
      options: [
        {
          text: "Baselines such as MemoryBank, A-MEM, Mem0, MemoryOS, LightMem, and LoCoMo were compared under the same GPT-4o-mini backbone.",
          isCorrect: true,
        },
        {
          text: "The same all-minilm-L6-v2 embedding model was used for AtomMem and the baseline systems.",
          isCorrect: true,
        },
        {
          text: "Retrieval capacity was standardized at top-k 10 for comparative baselines and AtomMem.",
          isCorrect: true,
        },
        {
          text: "The fact extractor was trained with supervised fine-tuning on a constructed extraction dataset rather than left as a zero-shot prompt.",
          isCorrect: true,
        },
      ],
      explanation:
        "Shared backbone, embedding model, and retrieval capacity reduce evaluation confounds. AtomMem's fact executor was also trained with supervised fine-tuning, so extraction quality is part of the proposed architecture rather than an incidental zero-shot behavior.",
    },
    {
      id: "ai-agents-atommem-q21",
      chapter: 1,
      difficulty: "hard",
      prompt:
        "Which interpretations are supported by AtomMem's main LoCoMo results?",
      options: [
        {
          text: "AtomMem's strongest gains appear in tasks requiring long-context integration, especially multi-hop and temporal reasoning.",
          isCorrect: true,
        },
        {
          text: "AtomMem raised the open-domain judge score beyond the reported Mem0 score while using substantially fewer total API tokens than Mem0.",
          isCorrect: true,
        },
        {
          text: "AtomMem-Flat showed that compact atomic facts alone can beat raw-history LoCoMo retrieval on several metrics at very low token cost.",
          isCorrect: true,
        },
        {
          text: "The results show that raw dialogue history is sufficient when paired with a stronger final response generator.",
          isCorrect: false,
        },
      ],
      explanation:
        "The results point to representational and retrieval benefits, not simply final-answer fluency. AtomMem outperformed strong baselines in reasoning-heavy categories, while AtomMem-Flat demonstrated that atomic fact quality itself is a major source of improvement.",
    },
    {
      id: "ai-agents-atommem-q22",
      chapter: 1,
      difficulty: "medium",
      prompt: "What does the AtomMem-Flat variant demonstrate?",
      options: [
        {
          text: "Retrieving atomic facts without the full hierarchy can already outperform raw dialogue-history memory on LoCoMo.",
          isCorrect: true,
        },
        {
          text: "A high-quality base representation can matter more than adding complex orchestration on top of noisy memory units.",
          isCorrect: true,
        },
        {
          text: "The full modular system can still improve on the flat variant by adding events, profiles, and graph recall.",
          isCorrect: true,
        },
        {
          text: "The flat variant uses atomic facts rather than every original turn, helping it keep token use low.",
          isCorrect: true,
        },
      ],
      explanation:
        "AtomMem-Flat validates atomic fact extraction as a strong representation by itself. It does not eliminate the value of the full architecture, because the complete system still improves performance through events, profiles, and graph-based associative recall while the flat variant keeps token cost low.",
    },
    {
      id: "ai-agents-atommem-q23",
      chapter: 1,
      difficulty: "hard",
      type: "assertion-reason",
      prompt:
        "Assertion: Removing profile memory should have no effect on single-hop performance because single-hop questions never depend on stable user attributes.\n\nReason: AtomMem's profile layer tracks persistent user attributes and historical versions derived from accumulated evidence.",
      options: [
        { text: "Assertion is true, Reason is false.", isCorrect: false },
        { text: "Assertion is false, Reason is true.", isCorrect: true },
        { text: "Both are false.", isCorrect: false },
        {
          text: "Both are true, and the Reason is the correct explanation of the Assertion.",
          isCorrect: false,
        },
        {
          text: "Both are true, but the Reason is NOT the correct explanation of the Assertion.",
          isCorrect: false,
        },
      ],
      explanation:
        "The assertion is false because the ablation without profile memory reduced single-hop performance, showing that some apparently simple questions still benefit from stable user-state modeling. The reason is true because the profile layer stores persistent attributes with evidence and history.",
    },
    {
      id: "ai-agents-atommem-q24",
      chapter: 1,
      difficulty: "medium",
      prompt:
        "Which conclusion follows from AtomMem's retrieval-capacity analysis over the number of retrieved atomic facts?",
      options: [
        {
          text: "Increasing retrieved facts from a very small value improves evidence coverage for reasoning-heavy tasks.",
          isCorrect: true,
        },
        {
          text: "Retrieving too many facts can add irrelevant noise and reduce answer quality despite higher recall.",
          isCorrect: true,
        },
        {
          text: "The selected main setting balances evidence coverage, token use, and latency rather than maximizing retrieval count.",
          isCorrect: true,
        },
        {
          text: "The reported main setting retrieves a moderate number of final facts rather than every reachable fact.",
          isCorrect: true,
        },
      ],
      explanation:
        "The capacity analysis shows a non-monotonic tradeoff. More retrieved facts help until noise begins to hurt reasoning, so AtomMem uses a moderate final fact count instead of simply maximizing context size.",
    },
    {
      id: "ai-agents-atommem-q25",
      chapter: 1,
      difficulty: "easy",
      prompt: "Which limitations are acknowledged for AtomMem-style systems?",
      options: [
        {
          text: "Several stages rely on the underlying LLM, making performance sensitive to generation stability.",
          isCorrect: true,
        },
        {
          text: "The current framework processes textual interactions rather than full multimodal conversations.",
          isCorrect: true,
        },
        {
          text: "Token efficiency remains an area for further optimization.",
          isCorrect: true,
        },
        {
          text: "The memory graph requires direct access to private model weights, preventing deployment with API-based models.",
          isCorrect: false,
        },
      ],
      explanation:
        "The stated limitations concern LLM dependence, text-only processing, and additional token-efficiency opportunities. The graph operates over stored facts and metadata, so it does not inherently require direct access to the base model's private parameters.",
    },
    {
      id: "ai-agents-atommem-q26",
      chapter: 1,
      difficulty: "medium",
      prompt:
        "Which steps were used to construct the supervised fact-extraction dataset?",
      options: [
        {
          text: "A teacher model performed preliminary extraction using guidelines for filtering noise and resolving contextual dependencies.",
          isCorrect: true,
        },
        {
          text: "Human annotators refined the teacher outputs to check accuracy, remove residual noise, split long facts, and fix unresolved references.",
          isCorrect: true,
        },
        {
          text: "The final dataset paired instructions and dialogue contexts with target atomic-fact outputs.",
          isCorrect: true,
        },
        {
          text: "The dataset was constructed to avoid overlap with LoCoMo test entities, events, and themes.",
          isCorrect: true,
        },
      ],
      explanation:
        "The dataset uses teacher-model extraction followed by human-guided refinement. It is intended to train robust extraction while avoiding benchmark leakage, not to memorize evaluation answers or entities.",
    },
    {
      id: "ai-agents-atommem-q27",
      chapter: 1,
      difficulty: "easy",
      prompt:
        "Which instructions belong to the atomic fact extraction prompt design?",
      options: [
        {
          text: "Ignore greetings, fillers, acknowledgments, and other low-value conversation content.",
          isCorrect: true,
        },
        {
          text: "Resolve pronouns and vague temporal markers into explicit entities and dates when context permits.",
          isCorrect: true,
        },
        {
          text: "Rewrite first-person statements as attributed third-person facts and output structured JSON.",
          isCorrect: true,
        },
        {
          text: "Merge all distinct facts from a message into one long sentence so later storage has fewer identifiers.",
          isCorrect: false,
        },
      ],
      explanation:
        "The extraction prompt asks for standalone, objective, high-value facts. Keeping distinct facts separate is important because later verification, event linking, and retrieval operate at the atomic-fact level.",
    },
    {
      id: "ai-agents-atommem-q28",
      chapter: 1,
      difficulty: "hard",
      prompt:
        "A verified new fact has no direct event id, but retrieved context contains one standalone fact about the same coherent episode. What can AtomMem's event construction algorithm do?",
      options: [
        {
          text: "Place that standalone fact into the candidate target set for LLM-based semantic routing.",
          isCorrect: true,
        },
        {
          text: "Create a new event pairing the new fact with the matched standalone fact if the routing decision selects it.",
          isCorrect: true,
        },
        {
          text: "Update event attributes such as member facts, participants, temporal span, summary, and keywords when an event absorbs the new fact.",
          isCorrect: true,
        },
        {
          text: "Store the new fact independently when neither an existing event nor a standalone fact is selected as a match.",
          isCorrect: true,
        },
      ],
      explanation:
        "The event algorithm uses both existing events and standalone retrieved facts as possible context targets. Standalone matches can trigger new event creation, existing event matches update the event's fact set and metadata, and no-match cases remain stored independently.",
    },
    {
      id: "ai-agents-atommem-q29",
      chapter: 1,
      difficulty: "hard",
      type: "assertion-reason",
      prompt:
        "Assertion: AtomMem normalizes keyword, event, and temporal graph channels separately before fusing them.\n\nReason: Dense channels could otherwise overwhelm sparse but useful channels, and some facts may not have outgoing neighbors in every channel.",
      options: [
        { text: "Assertion is true, Reason is false.", isCorrect: false },
        { text: "Assertion is false, Reason is true.", isCorrect: false },
        { text: "Both are false.", isCorrect: false },
        {
          text: "Both are true, and the Reason is the correct explanation of the Assertion.",
          isCorrect: true,
        },
        {
          text: "Both are true, but the Reason is NOT the correct explanation of the Assertion.",
          isCorrect: false,
        },
      ],
      explanation:
        "Both statements are true, and the reason gives the system-level motivation. Channel-specific normalization and renormalized priors let graph propagation use available relation types without letting the densest relation family dominate by construction.",
    },
    {
      id: "ai-agents-atommem-q30",
      chapter: 1,
      difficulty: "hard",
      prompt:
        "Which statements correctly describe AtomMem's Personalized PageRank-style graph reranking?",
      options: [
        {
          text: "Seed retrieval scores are transformed into a personalized restart distribution.",
          isCorrect: true,
        },
        {
          text: "A higher restart probability keeps activation closer to seed facts and reduces drift.",
          isCorrect: true,
        },
        {
          text: "Dangling nodes redistribute probability according to the restart distribution so mass does not leak away.",
          isCorrect: true,
        },
        {
          text: "The iteration stops when it reaches a maximum number of steps or satisfies a convergence threshold.",
          isCorrect: true,
        },
      ],
      explanation:
        "The reranker iterates a random walk with restart until convergence or a maximum-step limit. Personalization anchors the walk, while graph propagation lets entity, event, and temporal associations surface non-seed facts.",
    },
    {
      id: "ai-agents-atommem-q31",
      chapter: 1,
      difficulty: "medium",
      prompt:
        "Which statements correctly interpret the LongMemEval supplementary evaluation?",
      options: [
        {
          text: "It was used as an independent benchmark to reduce concern that results were tailored only to LoCoMo.",
          isCorrect: true,
        },
        {
          text: "It evaluates categories such as single-session, multi-session, knowledge-update, temporal-reasoning, and preference-oriented memory questions.",
          isCorrect: true,
        },
        {
          text: "The subjective personalized category uses judge scoring because token overlap metrics are not suitable for rubric-based generation quality.",
          isCorrect: true,
        },
        {
          text: "It replaces LoCoMo in the main comparison because LoCoMo lacks long-term conversational interactions.",
          isCorrect: false,
        },
      ],
      explanation:
        "LongMemEval supplements rather than replaces LoCoMo. It checks memory behavior across additional categories, and some personalized-generation cases require judge-based assessment instead of exact token matching.",
    },
    {
      id: "ai-agents-atommem-q32",
      chapter: 1,
      difficulty: "medium",
      prompt:
        "What does AtomMem's latency breakdown imply about practical deployment?",
      options: [
        {
          text: "Most online latency comes from LLM inference stages such as query intent and answer generation.",
          isCorrect: true,
        },
        {
          text: "The memory retrieval pipeline adds relatively small overhead compared with total online processing.",
          isCorrect: true,
        },
        {
          text: "Graph reranking is reported as fast enough that structural retrieval does not dominate the online latency profile.",
          isCorrect: true,
        },
        {
          text: "The results show graph retrieval is the main bottleneck and therefore should be disabled for interactive chat.",
          isCorrect: false,
        },
      ],
      explanation:
        "The latency analysis reports total online processing in seconds, while the retrieval pipeline and graph reranking are measured in much smaller millisecond ranges. This suggests that LLM calls dominate online cost more than the graph retrieval structure.",
    },
    {
      id: "ai-agents-atommem-q33",
      chapter: 1,
      difficulty: "hard",
      prompt:
        "Which interpretation matches the graph-retrieval initial seed count analysis?",
      options: [
        {
          text: "Too few seed facts can under-activate the local memory graph and hurt recall.",
          isCorrect: true,
        },
        {
          text: "Too many seed facts can introduce irrelevant contexts and redundant edges that distort random-walk propagation.",
          isCorrect: true,
        },
        {
          text: "A seed count around 10 balances informational coverage and noise suppression in the reported configuration.",
          isCorrect: true,
        },
        {
          text: "Seed count has no practical effect once Personalized PageRank is used, because restart probability erases seed selection.",
          isCorrect: false,
        },
      ],
      explanation:
        "The seed count controls entry points into the localized graph, so it changes both coverage and noise. Personalized PageRank depends on the seed distribution, so poor seed selection can still degrade propagation.",
    },
    {
      id: "ai-agents-atommem-q34",
      chapter: 1,
      difficulty: "hard",
      prompt:
        "The compensatory fusion weight is tuned near 0.7 for event relevance. Which explanation best matches that result?",
      options: [
        {
          text: "A moderate event-heavy weight uses episode context to find missed facts while still retaining a fact-level precision term.",
          isCorrect: true,
        },
        {
          text: "Weights below the optimum underuse global event relevance and move compensatory recall closer to direct fact search.",
          isCorrect: true,
        },
        {
          text: "Weights above the optimum overemphasize event membership and can retrieve facts that fit the episode but not the query.",
          isCorrect: true,
        },
        {
          text: "The optimum means fact-level relevance should be removed from compensatory recall whenever event summaries are available.",
          isCorrect: false,
        },
      ],
      explanation:
        "The reported optimum is a tradeoff, not a rejection of fact-level relevance. Event context helps recover implicit evidence, but a direct fact relevance term keeps retrieved items aligned with the user's specific information need.",
    },
    {
      id: "ai-agents-atommem-q35",
      chapter: 1,
      difficulty: "hard",
      prompt:
        "Which hyperparameter pairings match the reported AtomMem retrieval configuration?",
      options: [
        {
          text: "Embedding similarity and keyword Jaccard are weighted 0.7 and 0.3 in the hybrid similarity score.",
          isCorrect: true,
        },
        {
          text: "The graph channel mixture gives the largest static prior to entity edges, followed closely by event edges, with a smaller temporal-edge prior.",
          isCorrect: true,
        },
        {
          text: "The final retrieved fact count and initial graph seed count are both set to 10.",
          isCorrect: true,
        },
        {
          text: "Graph construction is bounded by seed count, expansion hops, local node limits, neighbor limits, and temporal constraints.",
          isCorrect: true,
        },
      ],
      explanation:
        "The configuration combines dense and keyword similarity, uses all three graph channels with different priors, and limits graph construction by seeds, hops, nodes, neighbors, and temporal constraints. Random restart helps, but locality limits are still part of the design.",
    },
    {
      id: "ai-agents-atommem-q36",
      chapter: 1,
      difficulty: "medium",
      prompt:
        "Why is avoiding overlap between the supervised extraction dataset and LoCoMo test set important?",
      options: [
        {
          text: "It reduces the chance that benchmark gains come from memorizing evaluation entities, events, or themes.",
          isCorrect: true,
        },
        {
          text: "It makes the fact extractor's performance a better test of general extraction behavior across realistic scenarios.",
          isCorrect: true,
        },
        {
          text: "It keeps comparisons focused on memory-system behavior rather than leakage from training examples.",
          isCorrect: true,
        },
        {
          text: "It prevents AtomMem from using any human-refined data during supervised fine-tuning.",
          isCorrect: false,
        },
      ],
      explanation:
        "Benchmark leakage would make it hard to know whether the system learned a reusable extraction behavior or memorized test-specific content. The dataset can still use human-refined examples as long as those examples do not overlap with evaluation entities and events.",
    },
    {
      id: "ai-agents-atommem-q37",
      chapter: 1,
      difficulty: "medium",
      prompt:
        "A memory agent must answer, 'What did Alex decide after the meeting where the budget concern came up?' The relevant evidence is split across facts sharing an event, one keyword, and adjacent turns. Which AtomMem mechanism is most directly designed for this case?",
      options: [
        {
          text: "Graph-based associative recall seeded by primary and compensatory retrieval.",
          isCorrect: true,
        },
        {
          text: "A profile-only lookup that ignores episodic facts once a stable user attribute is available.",
          isCorrect: false,
        },
        {
          text: "Raw context replay of every session without atomic fact extraction or graph ranking.",
          isCorrect: false,
        },
        {
          text: "A single keyword Jaccard lookup with no event or temporal edges.",
          isCorrect: false,
        },
      ],
      explanation:
        "The scenario requires connecting evidence through multiple relation types, not only a profile field or a single keyword. AtomMem's associative graph can propagate across entity, event, and temporal edges after direct and compensatory retrieval identify useful seeds.",
    },
    {
      id: "ai-agents-atommem-q38",
      chapter: 1,
      difficulty: "hard",
      prompt:
        "Which design choice best protects original evidence when the system detects a conflict between a new fact and existing memory?",
      options: [
        {
          text: "Generate residual novel content and explicit update tuples rather than repeatedly rewriting the same stored entry without constraints.",
          isCorrect: true,
        },
        {
          text: "Let every new mention overwrite the closest existing fact because the latest text is always the correct user state.",
          isCorrect: false,
        },
        {
          text: "Reject all conflicting facts so the memory store never contains temporal changes or revised preferences.",
          isCorrect: false,
        },
        {
          text: "Collapse conflicts into a single event summary and remove fact-level support identifiers.",
          isCorrect: false,
        },
      ],
      explanation:
        "AtomMem treats conflicts as structured memory updates rather than unrestricted rewriting. Residual storage and update tuples let the system preserve novel evidence, manage consistency, and avoid losing the fact-level trace that supports later reasoning.",
    },
    {
      id: "ai-agents-atommem-q39",
      chapter: 1,
      difficulty: "hard",
      type: "assertion-reason",
      prompt:
        "Assertion: AtomMem's full architecture can improve over AtomMem-Flat even though AtomMem-Flat validates atomic facts as a strong representation.\n\nReason: Event memory, temporal profiles, and graph recall add organization and association mechanisms that flat fact retrieval lacks.",
      options: [
        { text: "Assertion is true, Reason is false.", isCorrect: false },
        { text: "Assertion is false, Reason is true.", isCorrect: false },
        { text: "Both are false.", isCorrect: false },
        {
          text: "Both are true, and the Reason is the correct explanation of the Assertion.",
          isCorrect: true,
        },
        {
          text: "Both are true, but the Reason is NOT the correct explanation of the Assertion.",
          isCorrect: false,
        },
      ],
      explanation:
        "Both statements are true, and the reason explains why the full system can outperform the flat variant. Atomic facts provide a strong storage substrate, while events, profiles, and graph recall help recover context, user state, and dispersed evidence.",
    },
    {
      id: "ai-agents-atommem-q40",
      chapter: 1,
      difficulty: "medium",
      prompt:
        "Which design lessons summarize AtomMem's approach to long-term agent memory?",
      options: [
        {
          text: "Use compact, standalone facts as a value-dense and faithful base representation.",
          isCorrect: true,
        },
        {
          text: "Organize facts into events and temporal profiles so episodic context and user-state evolution remain explicit.",
          isCorrect: true,
        },
        {
          text: "Use graph-based associative recall to recover dispersed evidence that isolated similarity search may miss.",
          isCorrect: true,
        },
        {
          text: "Avoid treating longer raw context and more frequent global rewrites as automatic substitutes for stable representation and controlled retrieval.",
          isCorrect: true,
        },
      ],
      explanation:
        "AtomMem's design is based on selective evidence preservation, structured organization, and bounded associative retrieval. The system argues against unbounded raw context and unconstrained rewriting because both can add noise, cost, or instability.",
    },
  ];
