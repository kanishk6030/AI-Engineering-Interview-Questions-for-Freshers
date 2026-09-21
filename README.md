# AI Engineering Interview Questions for Freshers

> **A focused cheat sheet for entry-level AI / GenAI / LLM roles: the questions that actually get asked, with short answers you can say out loud.**

These questions are helpful for roles such as:

- Junior / Associate AI Engineer
- GenAI Engineer (entry-level)
- LLM Application Developer
- Applied AI Engineer (entry-level)
- ML Engineer (entry-level)
- AI / ML Intern
- Data Scientist (GenAI-leaning, entry-level)

## How this list was built

- **Filtered for fresher level.** Kept concept-first questions you can answer from fundamentals plus one or two projects. Dropped infrastructure-scale and research-level topics (see [What to Skip as a Fresher](#what-to-skip-as-a-fresher)).
- **Added what the repo lacks.** Fresher interviews still screen classic ML basics (overfitting, bias-variance, precision/recall), so there is a dedicated [ML and Deep Learning Fundamentals](#ml-and-deep-learning-fundamentals) section.
- **Added short answers.** Every question has a crisp answer. Where the original repo has a detailed explanation, it is linked as **Deep dive**.
- **Marked priorities.** 🔥 = shows up again and again across fresher guides. Prepare these first.

## What fresher interviews usually look like

| Round | What is tested |
|---|---|
| Resume / project screen | Can you explain what you built and why you made each choice? |
| Technical concepts | ML basics, Transformers and tokens, prompting, RAG, embeddings, agents |
| Coding | Python fundamentals plus small LLM-app tasks (chunking, similarity search, API calls with retries) |
| Scenario and behavioral | "Your chatbot hallucinates. What do you do?", trade-offs, communication |

Interviewers generally probe three things: **conceptual understanding** of how LLMs work, **hands-on experience** with the GenAI toolchain, and **engineering maturity** (evaluation, failure handling, cost). The winning combination for freshers is solid fundamentals plus at least one deployed GenAI project you can explain end to end.

## Table of Contents

- [Must Know](#must-know)
- [ML and Deep Learning Fundamentals](#ml-and-deep-learning-fundamentals)
- [LLM Fundamentals](#llm-fundamentals)
- [Prompt Engineering](#prompt-engineering)
- [Retrieval-Augmented Generation (RAG)](#retrieval-augmented-generation-rag)
- [Vector Databases and Embeddings](#vector-databases-and-embeddings)
- [AI Agents, Tool Calling and MCP](#ai-agents-tool-calling-and-mcp)
- [Fine-Tuning and Model Adaptation](#fine-tuning-and-model-adaptation)
- [Evaluation, Safety and Responsible AI](#evaluation-safety-and-responsible-ai)
- [LLMOps and Production Basics](#llmops-and-production-basics)
- [Coding and Practical Implementation](#coding-and-practical-implementation)
- [Scenario-Based Questions](#scenario-based-questions)
- [Project and Behavioral Questions](#project-and-behavioral-questions)
- [What to Skip as a Fresher](#what-to-skip-as-a-fresher)
- [4-Week Study Plan](#4-week-study-plan)
- [Sources and Credits](#sources-and-credits)

---

## Must Know

If you only have a weekend, be able to explain each of these in two minutes without notes:

- LLM and the Transformer
- Tokens, embeddings and the context window
- Prompt engineering (zero-shot, few-shot, chain-of-thought)
- RAG (the full pipeline, and why it beats fine-tuning for changing knowledge)
- Agents, tool calling and MCP
- Fine-tuning (LoRA) vs RAG vs prompting
- Hallucinations and how to reduce them
- Evaluation (how you know your system works)

Start here: [AI Engineering Explained: LLM, RAG, MCP, Agent, Fine-Tuning, Quantization](https://www.youtube.com/watch?v=lnfWvX66FUk)

---

## ML and Deep Learning Fundamentals

> Not covered in the original repo, but regularly asked in fresher rounds. For more, see the author's [Machine Learning Interview Questions](https://github.com/amitshekhariitbhu/machine-learning-interview-questions).

- 🔥 **What is the difference between supervised, unsupervised, and reinforcement learning?**
  * **Answer:** Supervised learning trains on labeled data (spam detection, price prediction). Unsupervised learning finds structure without labels (clustering, dimensionality reduction). Reinforcement learning has an agent learn by trial and error from rewards. LLM pre-training is *self-supervised*: the labels come from the text itself (predict the next token).
- 🔥 **What are overfitting and underfitting? How do you prevent them?**
  * **Answer:** Overfitting: the model memorizes noise, so training accuracy is high but test accuracy is poor. Underfitting: the model is too simple and does poorly on both. Fix overfitting with more data, regularization, dropout, early stopping, data augmentation, or a simpler model. Fix underfitting with a bigger model, better features, or longer training.
- 🔥 **Explain the bias-variance tradeoff.**
  * **Answer:** High bias means the model is too simple and misses patterns (underfitting). High variance means it is too sensitive to the training data (overfitting). Lowering one usually raises the other, so the goal is the complexity level that minimizes total error on unseen data.
- **What is regularization? Explain L1 and L2.**
  * **Answer:** Regularization adds a penalty on large weights to the loss to reduce overfitting. L1 (Lasso) penalizes the sum of absolute weights and pushes some weights to exactly zero (feature selection). L2 (Ridge) penalizes squared weights and shrinks them smoothly.
- 🔥 **What are train, validation, and test sets? What is data leakage?**
  * **Answer:** Train fits the model, validation tunes hyperparameters, and test gives a final unbiased score (used once). Data leakage is when information from the test set or the future leaks into training (for example, scaling on the full dataset before splitting), which inflates results. K-fold cross-validation helps on small datasets.
- 🔥 **Explain precision, recall, and F1. Why is accuracy misleading on imbalanced data?**
  * **Answer:** Precision = TP / (TP + FP): of what you flagged, how much was right. Recall = TP / (TP + FN): of all real positives, how many you caught. F1 is their harmonic mean. If only 1% of transactions are fraud, a model that always says "not fraud" is 99% accurate and completely useless.
- **What are the confusion matrix and ROC-AUC?**
  * **Answer:** The confusion matrix counts TP, FP, TN, FN. ROC-AUC measures how well the model ranks positives above negatives across all thresholds (0.5 is random, 1.0 is perfect).
- **What is a loss function? Why is cross-entropy used for classification and LLMs?**
  * **Answer:** A loss function measures how wrong a prediction is; training minimizes it. Cross-entropy compares the predicted probability distribution to the true label, and heavily penalizes confident wrong answers. LLMs are trained with it to predict the next token. **Deep dive:** [Math Behind Cross-Entropy Loss](https://outcomeschool.com/blog/math-behind-cross-entropy-loss)
- 🔥 **What is gradient descent? What does the learning rate do?**
  * **Answer:** Gradient descent repeatedly moves the weights in the direction that reduces the loss. Learning rate too high: the loss bounces or diverges. Too low: training is slow or gets stuck. Variants: batch, stochastic, mini-batch; Adam is the common adaptive optimizer.
- **Explain epoch, batch size, and iteration.**
  * **Answer:** An epoch is one full pass over the training data. Batch size is how many samples are used per weight update. An iteration is one update, so iterations per epoch = dataset size / batch size.
- **What is backpropagation?**
  * **Answer:** A forward pass computes the prediction and loss. A backward pass uses the chain rule to compute how much each weight contributed to the loss (its gradient). The optimizer then updates the weights.
- **What are activation functions, and why do we need them?**
  * **Answer:** They add non-linearity; without them, stacked layers collapse into one linear function. ReLU is the default for hidden layers, sigmoid for binary outputs, softmax for multi-class probabilities.

---

## LLM Fundamentals

- 🔥 **What is a Large Language Model (LLM), and how does it work?**
  * **Answer:** An LLM is a Transformer neural network trained on huge amounts of text to predict the next token. At inference it generates one token at a time, appends it to the input, and repeats. **Deep dive:** [AI Engineering Explained (video)](https://www.youtube.com/watch?v=lnfWvX66FUk) and [Inside ChatGPT: What Happens After You Hit Enter?](https://outcomeschool.substack.com/p/inside-chatgpt-what-happens-after)
- 🔥 **What are foundation models, and how did they change AI engineering?**
  * **Answer:** Large models pre-trained on broad data that can be adapted to many tasks through prompting or fine-tuning. AI engineering shifted from training models from scratch to building applications on top of them (RAG, agents, evaluation). **Deep dive:** [AI Engineering Explained (video)](https://www.youtube.com/watch?v=lnfWvX66FUk)
- 🔥 **What is the Transformer architecture, and what are its key components?**
  * **Answer:** Token embeddings plus positional information, feeding a stack of blocks. Each block has multi-head self-attention, a feed-forward network, residual connections, and layer normalization. Its big advantage over RNNs is parallel training and better long-range dependencies. **Deep dive:** [Decoding Transformer Architecture](https://outcomeschool.com/blog/decoding-transformer-architecture)
- 🔥 **What is self-attention? Explain Query, Key, and Value.**
  * **Answer:** Each token produces a query ("what am I looking for?"), a key ("what do I contain?"), and a value ("what do I pass on?"). Attention weights = softmax(QKᵀ / √dₖ), and the output is the weighted sum of values, so every token can gather context from every other token. **Deep dive:** [Self Attention in Transformers](https://outcomeschool.com/blog/self-attention-in-transformers) and [Math behind Attention - Q, K, and V](https://outcomeschool.com/blog/math-behind-attention-qkv)
- **Why scale attention scores by √dₖ?**
  * **Answer:** Without scaling, dot products grow with dimension, which pushes softmax into saturated regions with tiny gradients and unstable training. **Deep dive:** [Math behind √dₖ Scaling Factor](https://outcomeschool.com/blog/scaling-dot-product-attention)
- **What is multi-head attention? Why use multiple heads?**
  * **Answer:** Several attention heads run in parallel, each with its own projections, so the model can capture different relationships at once (syntax, coreference, position). Outputs are concatenated and projected. **Deep dive:** [Multi-Head Attention in Transformers](https://outcomeschool.com/blog/multi-head-attention-in-transformers)
- **What is causal masking?**
  * **Answer:** In decoder models, each token may only attend to earlier tokens, so the model can't "see the future" during training. It is what makes next-token prediction valid. **Deep dive:** [Causal Masking in Attention](https://outcomeschool.com/blog/causal-masking-in-attention)
- **What is positional encoding, and why is it needed?**
  * **Answer:** Attention itself ignores word order, so position information is added to embeddings so the model knows "dog bites man" differs from "man bites dog". **Deep dive:** [Positional Embeddings in LLMs](https://outcomeschool.substack.com/p/positional-embeddings-in-llms)
- 🔥 **What is tokenization? Explain BPE.**
  * **Answer:** Tokenization splits text into subword units the model can process. Byte Pair Encoding starts from characters and repeatedly merges the most frequent adjacent pair, giving a fixed vocabulary that handles rare and unseen words. Tokens (not words) drive cost and context limits. **Deep dive:** [Tokenization in LLMs (video)](https://www.youtube.com/watch?v=sK2s9I84EVI) and [Byte Pair Encoding](https://outcomeschool.com/blog/bpe-in-llms)
- 🔥 **What are embeddings?**
  * **Answer:** Dense numeric vectors that represent meaning; texts with similar meaning land close together in vector space. They power semantic search and RAG. **Deep dive:** [Embeddings in Machine Learning (video)](https://www.youtube.com/watch?v=LedXW6xl21s)
- 🔥 **What is the context window, and why is it limited?**
  * **Answer:** The maximum number of tokens (input plus output) the model can consider at once; it is the model's working memory. It is limited because attention cost grows roughly quadratically with length and the KV cache eats GPU memory. **Deep dive:** [Why is the context window limited in LLMs?](https://www.youtube.com/watch?v=CGIhxIaOg3M&lc)
- 🔥 **What is temperature? Explain top-k and top-p sampling.**
  * **Answer:** Temperature rescales logits before softmax: low = focused and deterministic, high = diverse and creative. Top-k samples only from the k most likely tokens. Top-p (nucleus) samples from the smallest set of tokens whose cumulative probability reaches p. **Deep dive:** [How does Temperature control LLM output?](https://outcomeschool.com/blog/how-does-temperature-control-llm-output) and [Top-k and Top-p Sampling](https://outcomeschool.com/blog/how-do-top-k-and-top-p-sampling-work)
- **What are logits?**
  * **Answer:** The raw, unnormalized scores the model outputs for every token in the vocabulary. Softmax turns them into probabilities. **Deep dive:** [Understanding Logits](https://x.com/amitiitbhu/status/1927927814923207146)
- 🔥 **Encoder-only vs decoder-only vs encoder-decoder models?**
  * **Answer:** Encoder-only (BERT) understands text: classification, embeddings. Decoder-only (GPT, Claude, Llama) generates text and is the basis of modern LLMs. Encoder-decoder (T5) maps an input sequence to an output sequence: translation, summarization. **Deep dive:** [Encoder vs Decoder in Transformers](https://outcomeschool.com/blog/encoder-vs-decoder-in-transformers)
- **What is the KV cache, and why does it speed up inference?**
  * **Answer:** During generation, the keys and values of previous tokens are stored so they aren't recomputed for every new token. It makes decoding much faster at the cost of memory. **Deep dive:** [What is KV Cache in LLMs?](https://outcomeschool.com/blog/kv-cache-in-llms)
- **Why is the first token slower than the rest?**
  * **Answer:** The prefill phase processes the whole prompt at once (compute-heavy) before the first token can be produced; later tokens reuse the cache and are generated one at a time. **Deep dive:** [The First-Token Latency Problem](https://www.youtube.com/watch?v=XD8DD4cEHu0)
- **How do RNNs and Transformers differ?**
  * **Answer:** RNNs process tokens sequentially and struggle with long-range dependencies. Transformers process all tokens in parallel with attention, which trains faster and scales better. **Deep dive:** [How do RNNs and Transformers differ?](https://outcomeschool.com/blog/how-do-rnns-and-transformers-differ)
- **Open-source vs closed-source LLMs: when do you choose which?**
  * **Answer:** Closed APIs give top quality and zero infrastructure work. Open models give control, privacy, customization, and potentially lower cost at scale, but you own hosting and ops. Choose based on data sensitivity, budget, latency, and required quality.
- **What is Mixture of Experts (MoE)? (high level)**
  * **Answer:** Only a few specialized "expert" sub-networks activate per token, so the model has huge capacity at lower compute per token. **Deep dive:** [Mixture of Experts Explained](https://outcomeschool.com/blog/mixture-of-experts)
- **What is multimodal AI?**
  * **Answer:** Models that handle more than one data type (text, images, audio, video), for example describing an image or answering questions about a PDF page. **Deep dive:** [Multimodal AI](https://outcomeschool.com/blog/multimodal-ai)

---

## Prompt Engineering

- 🔥 **What is prompt engineering, and why does it matter?**
  * **Answer:** Designing the instructions, context, and examples given to a model so it produces reliable, useful output. It is the cheapest and fastest lever you have, and the first thing to try before RAG or fine-tuning.
- 🔥 **Explain zero-shot, one-shot, and few-shot prompting.**
  * **Answer:** Zero-shot: instruction only. One-shot: one example. Few-shot: several examples that show the format and style you want. Few-shot is great for consistent formatting and classification. **Deep dive:** [Zero-shot, one-shot, few-shot prompting](https://www.linkedin.com/posts/pallavi-shekhar_llm-prompting-ai-activity-7441801012472078336-JsHr)
- 🔥 **What is chain-of-thought (CoT) prompting? When do you use it?**
  * **Answer:** Asking the model to reason step by step before answering. It helps on multi-step reasoning, math, and logic. It adds tokens and latency, so skip it for simple lookups. **Deep dive:** [How does Chain-of-Thought Prompting work?](https://outcomeschool.com/blog/how-does-chain-of-thought-prompting-work)
- **What is self-consistency prompting?**
  * **Answer:** Sample several reasoning paths at a higher temperature and take the majority answer. It improves accuracy at higher cost.
- **What is ReAct prompting?**
  * **Answer:** The model alternates between *Reasoning* (thought) and *Acting* (calling a tool), then observes the result and continues. It is the basis of most simple agents. **Deep dive:** [ReAct Agent](https://outcomeschool.com/blog/react-agent)
- 🔥 **What is a system prompt, and how does it influence behavior?**
  * **Answer:** A high-priority instruction that sets the role, tone, rules, and output format for the whole conversation. User messages are interpreted inside the frame it sets.
- 🔥 **How do you get reliable structured output (JSON)?**
  * **Answer:** Define the schema in the prompt, add an example, use the provider's JSON mode / structured outputs / function calling, validate with something like Pydantic, and retry on failure. **Deep dive:** [How does Function Calling work in LLMs?](https://outcomeschool.com/blog/how-does-function-calling-work-in-llms)
- 🔥 **What is prompt injection, and how do you defend against it?**
  * **Answer:** Attacker-controlled text (direct, or hidden in documents and web pages: *indirect*) that overrides your instructions. Defenses: keep instructions and untrusted data separate, treat retrieved content as data, give tools least privilege, filter inputs and outputs, and require human approval for risky actions. There is no single perfect fix, so layer defenses. **Deep dive:** [Prompt Injection in LLMs](https://outcomeschool.com/blog/prompt-injection-in-llms)
- **What is jailbreaking?**
  * **Answer:** Prompting tricks (role-play, obfuscation, multi-turn pressure) that get a model to bypass its safety rules. Mitigate with alignment, guardrails, monitoring, and red teaming. **Deep dive:** [Prompt Injection in LLMs](https://outcomeschool.com/blog/prompt-injection-in-llms)
- **What is prompt chaining?**
  * **Answer:** Breaking a complex task into a sequence of smaller prompts where each output feeds the next (extract, then analyze, then write). It is easier to debug and often more accurate than one giant prompt. **Deep dive:** [How does Prompt Chaining work?](https://outcomeschool.com/blog/how-does-prompt-chaining-work)
- **What is the "lost in the middle" problem?**
  * **Answer:** Models use information at the start and end of a long prompt better than information in the middle. Put key facts at the edges and don't stuff the context. **Deep dive:** [The Lost in the Middle Problem](https://outcomeschool.com/blog/lost-in-the-middle-problem-in-llms)
- **How do you evaluate and iterate on prompts?**
  * **Answer:** Build a small test set with expected outputs, change one thing at a time, score results (rules, metrics, or LLM-as-judge), and version your prompts like code.
- **How do you reduce prompt cost and latency?**
  * **Answer:** Shorter prompts, cache repeated prefixes, smaller models for easy steps, capped output length, and fewer or better-chosen few-shot examples.
- **Prompt engineering vs prompt tuning vs fine-tuning?**
  * **Answer:** Prompt engineering changes the text input (no training). Prompt tuning learns small "soft prompt" vectors while the model stays frozen. Fine-tuning updates model weights.
- **How do you handle multi-turn conversations?**
  * **Answer:** Send relevant history each turn; when it gets long, use a sliding window, summarize older turns, or retrieve relevant memories. **Deep dive:** [AI Agent Memory](https://outcomeschool.com/blog/ai-agent-memory)

---

## Retrieval-Augmented Generation (RAG)

- 🔥 **What is RAG, and why is it important?**
  * **Answer:** RAG retrieves relevant documents at query time and gives them to the LLM as context. It grounds answers in real data, reduces hallucination, supports citations, and lets you update knowledge without retraining. **Deep dive:** [AI Engineering Explained (video)](https://www.youtube.com/watch?v=lnfWvX66FUk)
- 🔥 **Explain the architecture of a basic RAG pipeline.**
  * **Answer:** *Indexing:* load documents, chunk them, embed each chunk, store in a vector database. *Query time:* embed the question, retrieve top-k similar chunks (optionally rerank), build a prompt with the chunks, generate an answer with citations.
- 🔥 **What are chunking strategies? How do you pick chunk size?**
  * **Answer:** Chunks that are too small lose context; too large dilute relevance and waste tokens. Start around a few hundred tokens with 10-20% overlap, then tune against a retrieval eval set. **Deep dive:** [Chunking Strategies for RAG](https://outcomeschool.com/blog/chunking-strategies-for-rag)
- **Compare fixed-size, recursive, and semantic chunking.**
  * **Answer:** Fixed-size is simple but can cut sentences mid-thought. Recursive splits on natural boundaries (paragraph, sentence) and is a strong default. Semantic chunking splits where meaning shifts; better quality, more compute. **Deep dive:** [Chunking Strategies for RAG](https://outcomeschool.com/blog/chunking-strategies-for-rag)
- 🔥 **RAG vs fine-tuning: when do you use each?**
  * **Answer:** RAG for knowledge that changes, needs citations, or is private. Fine-tuning for style, format, or task behavior. They combine well. Try prompting first, then RAG, then fine-tuning. **Deep dive:** [AI Engineering Explained (video)](https://www.youtube.com/watch?v=lnfWvX66FUk)
- 🔥 **What is hybrid search, and why is it better than pure vector search?**
  * **Answer:** It combines keyword search (BM25: exact terms, IDs, jargon) with vector search (meaning) and merges the results. It fixes cases where embeddings miss exact matches. **Deep dive:** [How does Hybrid Search work?](https://outcomeschool.com/blog/how-does-hybrid-search-work)
- 🔥 **What is re-ranking?**
  * **Answer:** A second-stage model scores the top retrieved chunks against the query more precisely (for example, a cross-encoder) and reorders them, so the best context reaches the LLM. **Deep dive:** [How does a Reranker work?](https://outcomeschool.com/blog/how-does-a-reranker-work)
- 🔥 **How do you evaluate a RAG system?**
  * **Answer:** Evaluate retrieval (context precision and recall: did we fetch the right chunks?) and generation (faithfulness: is the answer supported by the context? answer relevance: does it address the question?). Use a labeled question set and track it over time. **Deep dive:** [LLM Evaluation](https://outcomeschool.com/blog/llm-evaluation)
- 🔥 **Your RAG system hallucinates even with the right context. How do you fix it?**
  * **Answer:** First check what was actually retrieved. Then tighten the prompt ("answer only from the context, else say you don't know"), lower temperature, reduce noisy chunks with reranking and smaller top-k, require citations, and add an answer-verification step.
- **How do you choose an embedding model?**
  * **Answer:** Consider quality on your domain (test on your own data), dimensions and cost, latency, language support, max input length, and hosted vs self-hosted.
- **What is metadata filtering?**
  * **Answer:** Restricting retrieval by attributes such as source, date, or user/team, applied before or during vector search. It also enables per-user access control. **Deep dive:** [How does a Vector Database work?](https://outcomeschool.com/blog/how-does-a-vector-database-work)
- **What is query transformation (HyDE, decomposition)?**
  * **Answer:** Rewriting the user's question to retrieve better: HyDE embeds a hypothetical answer instead of the question; decomposition splits a complex question into sub-questions. **Deep dive:** [How does HyDE work in RAG?](https://outcomeschool.com/blog/how-does-hyde-work)
- **What are Agentic RAG and GraphRAG? (awareness level)**
  * **Answer:** Agentic RAG lets an agent decide when and what to retrieve, and iterate. GraphRAG retrieves over a knowledge graph, which helps with multi-hop, relationship-heavy questions. **Deep dive:** [Agentic RAG](https://outcomeschool.com/blog/agentic-rag) and [GraphRAG](https://outcomeschool.com/blog/graphrag)
- **How do you keep a RAG knowledge base fresh?**
  * **Answer:** Re-index changed documents incrementally (upsert by document ID), store timestamps/versions in metadata, delete stale chunks, and schedule or trigger re-ingestion.
- **How do you handle PDFs with tables and complex layouts?**
  * **Answer:** Use layout-aware parsers or OCR, keep tables intact (convert to markdown/JSON), extract with structure-preserving chunking, and consider multimodal models for tricky pages.
- **How do you add citations and source attribution?**
  * **Answer:** Store source metadata (file, page, URL) with every chunk, pass chunk IDs into the prompt, ask the model to cite them, and verify that cited chunks exist.
- **Long context window vs RAG: when do you use which?**
  * **Answer:** Long context is simple for small, one-off documents. RAG wins when the corpus is large, changes often, needs access control, or cost/latency matters. Very long prompts also suffer from "lost in the middle". **Deep dive:** [The Lost in the Middle Problem](https://outcomeschool.com/blog/lost-in-the-middle-problem-in-llms)
- **How would you scale RAG to millions of documents?**
  * **Answer:** Use an ANN index (HNSW/IVF), metadata filters, hybrid search plus reranking, batch and cache embeddings, and shard the index. **Deep dive:** [Approximate Nearest Neighbor (ANN) search](https://outcomeschool.com/blog/how-does-approximate-nearest-neighbor-ann-search-work)

---

## Vector Databases and Embeddings

- 🔥 **What is a vector database, and how is it different from a traditional database?**
  * **Answer:** It stores embeddings and answers "find the most similar vectors" using similarity search and specialized indexes, whereas a traditional DB answers exact-match and range queries. Most also store metadata and support filtering. **Deep dive:** [How does a Vector Database work?](https://outcomeschool.com/blog/how-does-a-vector-database-work)
- 🔥 **Explain cosine similarity, dot product, and Euclidean distance.**
  * **Answer:** Cosine similarity measures the angle between vectors (direction, ignores length); dot product also factors in magnitude; Euclidean measures straight-line distance. For normalized embeddings, cosine and dot product give the same ranking. **Deep dive:** [How does a Vector Database work?](https://outcomeschool.com/blog/how-does-a-vector-database-work)
- **How does Approximate Nearest Neighbor (ANN) search work? Compare HNSW, IVF, and flat indexes.**
  * **Answer:** Exact (flat) search compares against every vector: accurate but slow at scale. ANN trades a little recall for big speedups. HNSW builds a layered graph (fast, high recall, more memory); IVF clusters vectors and searches only nearby clusters (less memory). **Deep dive:** [ANN search](https://outcomeschool.com/blog/how-does-approximate-nearest-neighbor-ann-search-work)
- **What is the difference between sparse and dense embeddings?**
  * **Answer:** Sparse vectors (BM25/TF-IDF) are mostly zeros and match exact terms. Dense vectors (neural embeddings) are compact and capture meaning. Hybrid search uses both.
- **How does embedding dimensionality affect performance and cost?**
  * **Answer:** More dimensions can capture more nuance but increase storage, memory, and search latency. Pick the smallest size that meets your quality target.
- **Your new embedding model has different dimensions or vectors. What do you do?**
  * **Answer:** You cannot mix embeddings from different models. Re-embed the whole corpus into a new versioned index, then switch traffic over.
- **Your vector search returns irrelevant results despite high similarity scores. How do you fix it?**
  * **Answer:** Inspect chunk quality and size, try hybrid search, add reranking and metadata filters, check that query and documents use the same embedding model, and consider domain fine-tuning.
- **How do you choose a vector database?**
  * **Answer:** FAISS or Chroma for local prototypes; pgvector if you already run Postgres; managed options (Pinecone, Qdrant, Weaviate, etc.) for scale and ops-free hosting. Compare filtering, hybrid support, cost, and latency.
- **What is contrastive learning, and how does it train embedding models?**
  * **Answer:** It pulls embeddings of related pairs (a question and its answer) closer and pushes unrelated pairs apart. **Deep dive:** [What is Contrastive Learning?](https://outcomeschool.com/blog/contrastive-learning)

---

## AI Agents, Tool Calling and MCP

- 🔥 **What is an AI agent, and how is it different from a simple LLM call?**
  * **Answer:** A single LLM call maps input to output once. An agent runs a loop: it plans, calls tools, observes results, and continues until the goal is met, keeping state along the way. **Deep dive:** [AI Agent Explained](https://outcomeschool.com/blog/ai-agent)
- 🔥 **What is tool use (function calling), and how does it enable agents?**
  * **Answer:** You describe tools (name, description, JSON parameters). The model returns a structured request to call one; your code executes it and returns the result to the model. The model never runs code itself. **Deep dive:** [How does Function Calling work in LLMs?](https://outcomeschool.com/blog/how-does-function-calling-work-in-llms)
- **What is the difference between structured output and function calling?**
  * **Answer:** Structured output forces the model's *reply* into a schema. Function calling has the model choose an *action* and its arguments for your code to execute. **Deep dive:** [Function Calling in LLMs](https://outcomeschool.com/blog/how-does-function-calling-work-in-llms)
- 🔥 **What is MCP (Model Context Protocol)? How does it differ from function calling?**
  * **Answer:** MCP is an open standard for connecting models to external tools and data through servers, so a tool is written once and works across many apps. Function calling is the model-level mechanism for choosing a tool; MCP standardizes how tools are exposed and discovered. **Deep dive:** [What is MCP?](https://outcomeschool.com/blog/what-is-mcp-model-context-protocol)
- 🔥 **How do you design good tools for an agent?**
  * **Answer:** Clear names and descriptions, narrow single-purpose functions, typed parameters with examples, few tools rather than many, and helpful error messages the model can act on.
- **Explain the ReAct and Plan-and-Execute patterns.**
  * **Answer:** ReAct interleaves think, act, and observe step by step. Plan-and-Execute writes a full plan first, then executes each step, which is better for long, structured tasks. **Deep dive:** [ReAct Agent](https://outcomeschool.com/blog/react-agent) and [Plan-and-Execute Agent](https://outcomeschool.com/blog/plan-and-execute-agent)
- **What is an agent loop, and how does it know when to stop?**
  * **Answer:** Repeat: model decides, tool runs, result returns. Stop when the model gives a final answer, a step or token budget is hit, or an error threshold is reached. **Deep dive:** [AI Agent Loop](https://outcomeschool.com/blog/ai-agent-loop)
- **What types of agent memory exist?**
  * **Answer:** Short-term (current context), long-term (stored facts and preferences, often in a vector store), and episodic (past interactions/experiences). **Deep dive:** [AI Agent Memory](https://outcomeschool.com/blog/ai-agent-memory)
- **Single-agent vs multi-agent systems?**
  * **Answer:** Start with a single agent. Multi-agent adds specialization and parallelism but also cost, coordination bugs, and harder debugging. **Deep dive:** [Multi-Agent Systems](https://outcomeschool.com/blog/multi-agent-systems)
- **Your agent is stuck in an infinite loop. How do you fix it?**
  * **Answer:** Enforce a max-step and token budget, detect repeated identical tool calls, add a fallback or escalation, and make tool errors informative. **Deep dive:** [Fix an infinite loop in an AI agent](https://www.linkedin.com/posts/pallavi-shekhar_ai-aiagents-machinelearning-share-7440257380707364864-5Ycc)
- **Your agent keeps picking the wrong tool. How do you improve tool selection?**
  * **Answer:** Sharpen tool descriptions, reduce overlapping tools, add examples, route by category first, and evaluate tool choice on a test set.
- **How do you prevent harmful or irreversible agent actions?**
  * **Answer:** Least-privilege permissions, human-in-the-loop approval for destructive actions, sandboxing, dry-run modes, and guardrails on inputs and outputs. **Deep dive:** [How do LLM guardrails work?](https://outcomeschool.com/blog/how-do-llm-guardrails-work)
- **How do you evaluate an AI agent?**
  * **Answer:** Measure task success, correctness of tool choices and arguments, number of steps, cost and latency, and safety, using scripted scenarios and logged traces. **Deep dive:** [AI Agent Evaluation](https://outcomeschool.com/blog/ai-agent-evaluation)
- **What are LangChain and LangGraph? (awareness level)**
  * **Answer:** LangChain is a framework for composing LLM apps (prompts, retrievers, tools). LangGraph models agent workflows as stateful graphs with loops and branching. **Deep dive:** [How does LangChain work?](https://outcomeschool.com/blog/how-does-langchain-work) and [How does LangGraph work?](https://outcomeschool.com/blog/how-does-langgraph-work)

---

## Fine-Tuning and Model Adaptation

- 🔥 **What is fine-tuning, and when should you fine-tune?**
  * **Answer:** Further training a pre-trained model on your data to change its behavior: style, format, domain language, or a narrow task. Use it when prompting and RAG aren't enough. It is not the best way to add fast-changing facts. **Deep dive:** [How does fine-tuning work?](https://outcomeschool.com/blog/how-does-fine-tuning-work)
- 🔥 **Full fine-tuning vs parameter-efficient fine-tuning (PEFT)?**
  * **Answer:** Full fine-tuning updates all weights (expensive, needs lots of GPU memory). PEFT trains a small number of extra parameters (LoRA, adapters), which is much cheaper with similar results for many tasks.
- 🔥 **What are LoRA and QLoRA?**
  * **Answer:** LoRA freezes the base model and trains small low-rank matrices added to certain layers. QLoRA does the same on a 4-bit quantized base model, so large models can be fine-tuned on modest hardware. **Deep dive:** [LoRA - Low-Rank Adaptation of LLMs](https://outcomeschool.com/blog/lora-low-rank-adaptation-of-llms)
- **Pre-training vs SFT vs RLHF/DPO?**
  * **Answer:** Pre-training learns language from massive raw text. Supervised fine-tuning (SFT) teaches instruction-following from example pairs. RLHF/DPO aligns outputs to human preferences (helpful, harmless). **Deep dive:** [Decoding InstructGPT](https://outcomeschool.com/blog/decoding-instructgpt) and [RLHF](https://outcomeschool.com/blog/reinforcement-learning-from-human-feedback-rlhf)
- **How do you prepare a dataset for fine-tuning?**
  * **Answer:** Collect high-quality, diverse examples in the target format, clean and deduplicate, remove PII, split into train/validation/test, and hand-inspect samples. Quality beats quantity.
- **What is catastrophic forgetting?**
  * **Answer:** The model loses general abilities after training on narrow data. Reduce it with lower learning rates, PEFT/LoRA, mixing in general data, and fewer epochs. **Deep dive:** [Continual Learning in LLMs](https://outcomeschool.com/blog/continual-learning-in-llms)
- **How do you evaluate a fine-tuned model?**
  * **Answer:** Compare against the base model on a held-out task set, plus regression checks on general ability and safety. **Deep dive:** [LLM Evaluation](https://outcomeschool.com/blog/llm-evaluation)
- 🔥 **What is model quantization?**
  * **Answer:** Storing weights in lower precision (FP16 to INT8/INT4) to cut memory and speed up inference, with a small quality trade-off. It's what lets big models run on smaller hardware. **Deep dive:** [How does Model Quantization work?](https://outcomeschool.com/blog/how-does-model-quantization-work)
- **What is knowledge distillation?**
  * **Answer:** Training a smaller "student" model to imitate a larger "teacher", giving a cheaper, faster model. **Deep dive:** [How does Knowledge Distillation work?](https://outcomeschool.com/blog/how-does-knowledge-distillation-work)

---

## Evaluation, Safety and Responsible AI

- 🔥 **What are hallucinations, and how do you mitigate them?**
  * **Answer:** Confident but false or unsupported output. Layer your defenses: ground answers with RAG, instruct the model to say "I don't know", lower temperature, require citations, verify outputs (rules or a checker model), and route uncertain cases to a human.
- 🔥 **How do you evaluate LLM outputs? What metrics do you use?**
  * **Answer:** Combine automatic checks (exact match, schema validity, unit tests), reference-based metrics (BLEU, ROUGE, BERTScore), LLM-as-a-judge, and human review. Pick metrics that match the task; overlap metrics alone can miss meaning. **Deep dive:** [LLM Evaluation](https://outcomeschool.com/blog/llm-evaluation)
- 🔥 **What is LLM-as-a-judge, and what are its limitations?**
  * **Answer:** Using a strong model to score outputs against a rubric. It is cheap and scalable, but can be biased (favoring longer answers or its own style), inconsistent, and needs calibration against human labels. **Deep dive:** [LLM as a Judge](https://outcomeschool.com/blog/llm-as-a-judge)
- **What is a golden dataset, and why build one?**
  * **Answer:** A curated set of inputs with trusted expected outputs. It is your regression test suite: run it on every prompt, model, or pipeline change to catch quality drops.
- **What are benchmarks (MMLU, HumanEval, GSM8K)? What is benchmark contamination?**
  * **Answer:** Standard tests for knowledge, coding, and math reasoning. Contamination is when test data leaked into training, inflating scores. Benchmarks don't replace evaluation on your own task. **Deep dive:** [LLM Evaluation](https://outcomeschool.com/blog/llm-evaluation)
- **What is red teaming?**
  * **Answer:** Deliberately attacking your own system (jailbreaks, prompt injection, harmful requests) before launch to find failures and fix them.
- **What are guardrails, and how do you implement them?**
  * **Answer:** Checks around the model: input filters (injection, PII, off-topic) and output filters (toxicity, policy, leaked data, schema). Implement with rules, classifiers, or a second model. **Deep dive:** [How do LLM guardrails work?](https://outcomeschool.com/blog/how-do-llm-guardrails-work)
- **How do you handle PII and privacy in LLM apps?**
  * **Answer:** Minimize what you send, mask or redact PII before calling the model, avoid logging sensitive data, control retention, and comply with laws such as GDPR and India's DPDP Act.
- **How do you detect and reduce bias?**
  * **Answer:** Test outputs across demographic groups, audit the training/eval data, use diverse examples and counterfactual tests, monitor in production, and add human review for high-stakes decisions.
- **What is explainability, and how is it different from interpretability?**
  * **Answer:** Interpretability is understanding how the model works internally. Explainability is giving understandable reasons for a specific decision to users or auditors.
- **What is AI alignment?**
  * **Answer:** Making models behave according to human intentions and values (helpful, honest, harmless), typically through SFT and RLHF/DPO. **Deep dive:** [Decoding InstructGPT](https://outcomeschool.com/blog/decoding-instructgpt)
- **How would you handle a model producing biased or harmful output in production?**
  * **Answer:** Contain it (disable the feature or add a filter), investigate with logs, fix the cause (prompt, data, guardrail), add the failing case to your eval set, and communicate transparently.

---

## LLMOps and Production Basics

- 🔥 **How is LLMOps different from traditional MLOps?**
  * **Answer:** Beyond deploying and monitoring models, you manage prompts and their versions, token cost, non-deterministic outputs, retrieval quality, guardrails, and continuous evaluation.
- 🔥 **How would you deploy an LLM application?**
  * **Answer:** Wrap the pipeline in an API (for example, FastAPI), containerize with Docker, deploy to a cloud service, and add streaming, timeouts, logging, and secrets management. Streamlit or Gradio are good for quick demos. **Deep dive:** [How does Token Streaming work?](https://outcomeschool.com/blog/how-does-token-streaming-work)
- 🔥 **How do you reduce LLM costs without hurting quality?**
  * **Answer:** Cache responses, shorten prompts, route easy requests to smaller models, cap output length, batch where possible, and use prompt caching. **Deep dive:** [LLM Routing](https://outcomeschool.com/blog/llm-routing) and [How does Semantic Caching work?](https://outcomeschool.com/blog/how-does-semantic-caching-work)
- **What is prompt caching vs semantic caching?**
  * **Answer:** Prompt caching reuses the computation for an identical, repeated prompt prefix (cheaper, faster). Semantic caching returns a stored answer when a *new* query is similar in meaning to an old one. **Deep dive:** [Prompt Caching](https://outcomeschool.com/blog/how-does-prompt-caching-work) and [Semantic Caching](https://outcomeschool.com/blog/how-does-semantic-caching-work)
- 🔥 **How do you handle rate limits and API failures?**
  * **Answer:** Retry retryable errors (429, 5xx, timeouts) with exponential backoff and jitter, respect `Retry-After`, queue and throttle requests, and fall back to another model or provider.
- **How do you monitor LLM apps in production?**
  * **Answer:** Log prompts, responses, latency, token usage, cost, and errors; trace each pipeline step (retrieval, tool calls); collect user feedback; alert on quality or cost drift. **Deep dive:** [AI Agent Observability](https://outcomeschool.com/blog/ai-agent-observability)
- **How do you reduce latency in an LLM app?**
  * **Answer:** Stream tokens to the user, use smaller or faster models, cache, run independent calls in parallel, and trim context.
- **How do you version and manage prompts?**
  * **Answer:** Keep prompts in version control or a registry, tie each version to eval results, and roll out changes gradually with A/B tests.
- **LLM API or self-hosted open-source model?**
  * **Answer:** APIs: fast to ship, best quality, pay per token. Self-hosting: data control and possible cost savings at high volume, but you handle GPUs, scaling, and updates.
- **How do you keep API keys and secrets safe?**
  * **Answer:** Use environment variables or a secrets manager, never commit keys to Git, scope and rotate keys, and keep them server-side (never in client code).

---

## Coding and Practical Implementation

Practice these until you can write them without looking anything up:

- 🔥 Implement a basic RAG pipeline (embed, store, retrieve, generate)
- 🔥 Implement cosine similarity and semantic search from scratch
- 🔥 Write a text chunker with overlap
- 🔥 Call an LLM API with retry and exponential backoff
- 🔥 Build a simple agent with tool use (calculator, web search) and a step limit
- Write a function-calling handler for an LLM API
- Implement streaming responses
- Implement conversation memory (sliding window, summary)
- Build a prompt template system with variable substitution
- Implement token counting and context-window management
- Build a simple caching layer for LLM responses
- Build an LLM-as-a-judge evaluation pipeline
- Extract text from PDFs and split it into chunks
- Write a basic PII / prompt-injection input check
- Stretch goal: implement scaled dot-product attention with a causal mask (NumPy or PyTorch)

Expect standard Python and data-structure questions too; many fresher loops still include a plain coding round.

**Cosine similarity from scratch**

```python
import numpy as np

def cosine_similarity(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denom) if denom else 0.0
```

**Simple chunker with overlap**

```python
def chunk_text(text, size=500, overlap=50):
    assert 0 <= overlap < size, "overlap must be smaller than size"
    step = size - overlap
    return [text[i:i + size] for i in range(0, max(len(text) - overlap, 1), step)]
```

**Retry with exponential backoff and jitter**

```python
import random, time

def call_with_retry(fn, max_retries=5, base=1.0, cap=30.0):
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception:  # in real code, catch only retryable errors (429, 5xx, timeouts)
            if attempt == max_retries - 1:
                raise
            delay = min(cap, base * 2 ** attempt)
            time.sleep(delay * random.uniform(0.5, 1.0))
```

**Minimal agent loop with a step budget**

```python
def run_agent(llm, tools, task, max_steps=8):
    messages = [{"role": "user", "content": task}]
    for _ in range(max_steps):
        reply = llm(messages, tools)              # returns text or a tool call
        if reply.get("tool_call") is None:
            return reply["text"]                  # final answer
        name, args = reply["tool_call"]["name"], reply["tool_call"]["args"]
        try:
            result = tools[name](**args)
        except Exception as e:
            result = f"Tool error: {e}"           # let the model recover
        messages.append({"role": "tool", "name": name, "content": str(result)})
    return "Stopped: step budget reached."
```

---

## Scenario-Based Questions

Interviewers love these. A good answer states a **diagnosis first**, then a **fix**, then how you would **verify** it.

- 🔥 **Your chatbot hallucinates facts. What do you do?**
  * **Answer:** Check whether retrieval fetched the right context. Ground the answer in retrieved text, instruct "answer only from context or say you don't know", add citations and a verification step, lower temperature, and add failures to your eval set.
- **Your LLM ignores the required output format.**
  * **Answer:** Use structured outputs or function calling, add a schema and few-shot examples, validate the response, and retry with the error message.
- **A document is too long for the context window.**
  * **Answer:** Chunk and retrieve relevant parts (RAG), or use map-reduce summarization (summarize chunks, then summarize the summaries).
- **Your LLM never says "I don't know."**
  * **Answer:** Explicitly permit abstaining in the prompt, ground answers in retrieved context, add a relevance threshold on retrieval, and evaluate abstention behavior.
- **Your chatbot forgets earlier turns after about 10 messages.**
  * **Answer:** Keep a sliding window of recent turns, summarize older ones, and store key facts in long-term memory that you retrieve when needed. **Deep dive:** [AI Agent Memory](https://outcomeschool.com/blog/ai-agent-memory)
- **Your LLM bill is too high.**
  * **Answer:** Measure cost per feature first, then cache, shorten prompts, route easy queries to cheaper models, and limit output length.
- **Users can make your chatbot reveal its system prompt.**
  * **Answer:** Assume the system prompt can leak, so keep secrets out of it. Add input and output filters, separate instructions from user data, and test with red-team prompts. **Deep dive:** [Prompt Injection in LLMs](https://outcomeschool.com/blog/prompt-injection-in-llms)
- **Your RAG app is slow.**
  * **Answer:** Profile each stage (embedding, search, rerank, LLM). Use an ANN index, cache embeddings and answers, retrieve fewer chunks, stream the response, and use a faster model where quality allows.
- **Your summarizer adds facts not in the source.**
  * **Answer:** Constrain the prompt to the source text, lower temperature, add a faithfulness check that compares claims to the source, and require quotes or citations.
- **A healthcare or finance chatbot gives advice it shouldn't.**
  * **Answer:** Add a scope guardrail and disclaimers, refuse or redirect out-of-scope requests, ground answers in vetted sources, and hand off to a human for high-risk queries. **Deep dive:** [How do LLM guardrails work?](https://outcomeschool.com/blog/how-do-llm-guardrails-work)

---

## Project and Behavioral Questions

Your projects are a large part of a fresher interview. Prepare a 3-minute story for your best GenAI project covering: **problem, architecture, key decisions, what failed, how you measured quality, and what you would improve.**

- 🔥 **Walk me through your best AI / LLM project.**
  * **Tip:** Lead with the problem and the result, then draw the architecture (data, chunking, embeddings, vector store, LLM, UI/API). Use real numbers (latency, accuracy, cost) if you have them.
- 🔥 **Why did you choose that chunk size, embedding model, vector DB, and LLM?**
  * **Tip:** Every choice needs a reason and a trade-off. "I tested 300, 500, and 800 tokens, and 500 gave the best retrieval hit rate" beats "it's the default".
- 🔥 **How did you evaluate your project, and how did you handle hallucinations?**
  * **Tip:** Describe your test set, the metrics, and one failure you found and fixed.
- **What was the hardest bug or failure, and what did you learn?**
  * **Tip:** Use STAR (Situation, Task, Action, Result); be honest about mistakes.
- **How would you scale or productionize your project?**
  * **Tip:** Talk about caching, monitoring, evaluation in CI, rate limits, cost controls, and security.
- **How do you decide whether a problem needs AI or a traditional solution?**
  * **Answer:** If rules or a simple query solve it reliably, use them: cheaper, faster, deterministic. Use AI when the input is unstructured, ambiguous, or too varied for rules, and when occasional errors are acceptable or checkable.
- **Describe a time you chose between accuracy and latency (or cost).**
  * **Tip:** Name the constraint, the options, the data you used to decide, and the outcome.
- **How do you explain LLM limitations to a non-technical stakeholder?**
  * **Answer:** Use plain analogies: it predicts likely text rather than looking up facts, so it can be confidently wrong. Then show mitigations (grounding, review steps) and agree on where errors are acceptable.
- **How would you build an AI feature with limited labeled data?**
  * **Answer:** Start with prompting and few-shot examples, use RAG for knowledge, generate and review synthetic data, label a small high-quality eval set, and fine-tune only if needed.
- **How do you stay current in AI?**
  * **Tip:** Name specific sources you follow (papers, model release notes, blogs) and something recent you tried hands-on.
- **What is AI Engineering, and how is it different from ML Engineering?**
  * **Answer:** ML engineering focuses on training and deploying models. AI engineering focuses on building products on top of foundation models: prompting, RAG, agents, evaluation, and production reliability.
- **Why are you interested in this role?**
  * **Tip:** Connect what you built to what the team builds; be specific.

---

## What to Skip as a Fresher

These topics from the original repo are usually senior, research, or infrastructure-level. Know that they exist, but go deep only if the job description asks for them:

- RL algorithm details: PPO, GRPO, RLVR, reward hacking
- Inference-engine internals: speculative decoding (Medusa, EAGLE), TensorRT-LLM, SGLang, prefill-decode disaggregation, chunked prefill
- Hardware and parallelism: GPU roofline math, TPU/LPU internals, tensor/pipeline parallelism, FSDP vs ZeRO
- Architecture details: RoPE, RMSNorm, Grouped-Query Attention, sliding-window attention, attention sinks, Flash Attention, DeepSeek internals
- Large-scale "Design X" questions: Sora, Midjourney, Suno, multi-region AI platforms, capacity planning
- Advanced governance and privacy: differential privacy, federated learning, NIST AI RMF, detailed EU AI Act compliance

---

## 4-Week Study Plan

| Week | Focus | Output |
|---|---|---|
| 1 | ML fundamentals, tokens, embeddings, Transformer and attention | Explain each concept aloud in 2 minutes |
| 2 | Prompt engineering, then RAG end to end | A working RAG app you can demo and explain |
| 3 | Agents and tool calling, evaluation, safety, and guardrails | Add an eval set and a simple agent to your project |
| 4 | Coding drills, scenario questions, mock interviews, project storytelling | A polished 3-minute project story |

---

## Sources and Credits

**Primary source (Apache-2.0):** [AI Engineering Interview Questions and Answers](https://github.com/amitshekhariitbhu/ai-engineering-interview-questions) by Amit Shekhar / [Outcome School](https://outcomeschool.com). Many **Deep dive** links point to their articles. This README is a curated and adapted derivative: questions were selected and reorganized for freshers, and short answers plus additional questions were added.

**Cross-checked against fresher and entry-level guides (2026):**

- [Generative AI Interview Questions, 2026 Update (NovelVista)](https://www.novelvista.com/blogs/ai-and-ml/generative-ai-interview-questions)
- [45+ AI Engineer Interview Questions (Aced / Exponent)](https://www.tryexponent.com/blog/ai-engineer-interview-questions)
- [45+ AI Engineer Interview Questions (Penn Career Services)](https://careerservices.upenn.edu/blog/2026/06/25/45-ai-engineer-interview-questions-answers-2026-guide/)
- [AI Engineer Interview Questions, 50+ for 2026 (Careery)](https://careery.pro/blog/ai-careers/ai-engineer-interview-questions)
- [Top LLM and GenAI Interview Questions (TopGenAIJobs)](https://www.topgenaijobs.com/blog/llm-interview-questions)
- [AI Interview Questions 2026: ML to LLMs (KodeKloud)](https://kodekloud.com/blog/ai-interview-questions/)
- [Top 35 AI Interview Questions (DataCamp)](https://www.datacamp.com/blog/ai-interview-questions)
- [Machine Learning Interview Questions (Aced / Exponent)](https://www.tryexponent.com/blog/top-machine-learning-interview-questions)
- [Top 60 AI and ML Interview Questions for Freshers (Cloud Soft Solutions)](https://cloudsoftsol.com/interview-questions/ai-ml-interview-questions-for-freshers-2026/)
- [Generative AI Interview Questions (InterviewBit)](https://www.interviewbit.com/generative-ai-interview-questions-and-answers/)

### License

This work is derived from an Apache-2.0 licensed project. If you publish it, keep the Apache License 2.0 text in a `LICENSE` file, retain the attribution above, and note that changes were made.
