# Lecture 7 Curriculum: RAG, Tool Calling, and Agents

Source materials:

- Transcript: `lecture 7 - transcript.md`
- Slides: `lecture 7 - slides.pdf`

## Course Role

This lecture moves from standalone LLMs to LLMs embedded in systems. It teaches how to overcome static knowledge, limited context, and inability to act through retrieval-augmented generation, tool calling, and agent loops.

## Learning Objectives

By the end, students should be able to:

- Explain the limitations that motivate RAG and tools: knowledge cutoff, limited context size, distraction by irrelevant information, token pricing, and inability to act.
- Design a basic RAG pipeline: collect documents, chunk, embed, retrieve, augment prompt, and generate.
- Choose and compare candidate retrieval methods: semantic embedding search, BM25 keyword search, and hybrid retrieval.
- Explain retrieval extensions: query/document embedding mismatch mitigation, contextual retrieval, prompt caching, and reranking/cross-encoders.
- Evaluate retrieval with top-k relevance metrics such as NDCG@k and reciprocal rank.
- Explain tool calling as model-mediated access to structured data, computation, or actions.
- Describe tool-calling stages: predict tool/arguments, execute backend function, synthesize response from tool output.
- Explain tool selection, tool descriptions, and Model Context Protocol (MCP) at a high level.
- Define an agent as a system that pursues goals and completes tasks on a user's behalf.
- Trace a ReAct-style loop: input, observe, plan, act, output.
- Explain multi-agent communication motivation and A2A-style standardization at a high level.
- Identify agent/tool safety risks such as data exfiltration and the need for training, safeguards, benchmarks, and observability.

## Prerequisite Assumptions

Students should know LLM generation, embeddings, context windows, reasoning models, basic evaluation, and the limitations of static pretrained weights.

## Curriculum Sequence

### 1. Motivate System-Augmented LLMs

Start with the lecture's list of vanilla LLM weaknesses: static knowledge, limited context size, distraction by irrelevant information, per-token cost, inability to perform actions, and evaluation difficulty. Position RAG, tools, and agents as escalating system patterns.

Active learning:

- Sort user requests into "answer from model memory", "RAG needed", "tool needed", and "agent loop needed."
- Ask what failure occurs if the model answers a fresh factual question without retrieval.

Assessment targets:

- Students can distinguish knowledge augmentation from action-taking.
- Students can explain why bigger context is not always sufficient.

### 2. Build a Knowledge Base for RAG

Teach RAG as augmenting the prompt with relevant information. Cover collecting documents, dividing into chunks, selecting chunk size/overlap, embedding chunks, and storing the resulting knowledge base.

Active learning:

- Given a long policy document, choose chunk boundaries and justify chunk size/overlap.
- Ask what information is lost when chunks are too small or too large.

Assessment targets:

- Students can explain why retrieval works over chunks rather than arbitrary full documents.
- Students can identify chunking hyperparameters and their tradeoffs.

### 3. Retrieve Candidates

Teach candidate retrieval as maximizing recall. Compare semantic search with embeddings, BM25 keyword search, and hybrid approaches. Include the transcript's example where exact keywords may matter and semantic similarity alone may miss the right document.

Active learning:

- Given a query and candidate chunks, decide whether semantic search, BM25, or hybrid retrieval is best.
- Ask which method handles synonyms and which handles exact identifiers.

Assessment targets:

- Students can explain why hybrid retrieval is often useful.
- Students can identify the mismatch between query embeddings and document/chunk embeddings.

### 4. Improve Retrieval Quality

Teach extensions: query expansion or fake-document approaches to mitigate query/document mismatch, contextual retrieval that adds a short document-level context to chunks, prompt caching to reduce repeated LLM-call cost, and reranking/cross-encoders to improve precision after recall-oriented retrieval.

Active learning:

- Write a short contextualization for a chunk using its whole-document context.
- Compare a bi-encoder retrieval step with a cross-encoder reranking step.

Assessment targets:

- Students can explain why rerankers are applied to a smaller candidate set.
- Students can explain why contextual retrieval can improve otherwise ambiguous chunks.

### 5. Evaluate Retrieval

Teach NDCG@k as rewarding relevant documents ranked near the top and reciprocal rank as measuring how early the first relevant result appears. Emphasize that retrieval quality is observable separately from final generation quality.

Active learning:

- Rank five retrieved chunks with binary or graded relevance and compute/interpret NDCG@k qualitatively.
- Ask how a bad answer can arise from good retrieval versus bad retrieval.

Assessment targets:

- Students can explain why retrieval metrics are evaluated at top-k.
- Students can separate retrieval failure from generation failure.

### 6. Add Tool Calling

Teach tool calling as exposing functions/APIs to the LLM for structured data, computation, or action. Trace the three stages: the model predicts the relevant function and arguments, the backend executes the function, and the model synthesizes the final response from tool output.

Active learning:

- Given a user request, choose a tool and infer arguments.
- Write a concise tool description/schema for a calculator, weather lookup, or database query.

Assessment targets:

- Students can identify tool prediction versus tool execution versus response synthesis.
- Students can explain why clear names, schemas, and descriptions matter.

### 7. Scale Tooling Through Selection and MCP

Teach the practical problem that too many tools can hurt performance and fill context. Explain tool selection/routing as reducing latency and improving model focus. Introduce MCP as a standard way to expose tools, prompts, and resources through servers/clients/hosts.

Active learning:

- Given a list of twenty tools, select the minimal subset for a request.
- Map a simple tool provider to MCP concepts: host, client, server, tools, prompts, resources.

Assessment targets:

- Students can explain why "more tools" is not always better.
- Students can explain MCP's value as avoiding duplicate custom integrations.

### 8. Build Agent Loops With ReAct

Define agents as systems that autonomously pursue goals and complete tasks on a user's behalf. Teach ReAct as reason + act: input, observe current state, plan the next task/tool call, act, observe the result, and continue until output.

Active learning:

- Trace the thermostat example: cold bear request -> observe unknown temperature -> plan to read room temperature -> call tool -> observe result -> plan to increase temperature -> act -> final response.
- Ask where reasoning, retrieval, and tools appear in the loop.

Assessment targets:

- Students can distinguish a single tool call from an agentic loop.
- Students can identify when an agent must stop and ask for clarification or approval.

### 9. Address Multi-Agent and Safety Concerns

Introduce the motivation for multiple specialized agents and A2A-like communication. Then focus on safety: real-world actions, data exfiltration, malicious tool use, runaway costs, and the need for safeguards, benchmarks, observability, and transparent intermediate steps.

Active learning:

- Identify safety risks in an agent with email, file, and web tools.
- Propose safeguards: permissioning, scopes, budget limits, allowlists, human approval, logging, and benchmark testing.

Assessment targets:

- Students can explain why agent safety is qualitatively more serious than ordinary text generation.
- Students can explain why transparency/observability improves trust and debuggability.

## Misconceptions to Address

- RAG does not update model weights; it changes context at inference time.
- Larger context does not guarantee better answers; irrelevant context can distract the model.
- Semantic search and keyword search solve different retrieval problems.
- A tool call is not the same as an agent; agents loop over observations, plans, and actions.
- Tool output must be interpretable by the model; backend correctness alone is not enough.
- Safety cannot be postponed until deployment because tools can act on external systems.

## Assessment Blueprint

Use design and debugging tasks:

- Design a RAG pipeline for a small document set.
- Choose retrieval methods and evaluate a top-k ranking.
- Write a tool schema and infer arguments from a user request.
- Diagnose whether a failure is retrieval, tool prediction, tool execution, or response synthesis.
- Trace a ReAct loop and identify stop conditions.
- Propose safety controls for an agent with real-world actions.

## Follow-Up Practice

- Build a RAG design checklist: data, chunking, embeddings, retrieval, reranking, generation, evaluation.
- Write three tool descriptions and test whether another student can choose the correct tool from the description alone.
- Read the ReAct paper summary and map its loop to a tool-using assistant you already use.
