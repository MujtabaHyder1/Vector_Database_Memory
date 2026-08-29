# Vector Database Memory for Stateful Agent Systems

## Overview

This project demonstrates how to build persistent vector database memory systems for AI agents that maintain context across conversations and improve responses over time. Moving beyond stateless query-response patterns, we implement a stateful agent architecture where agents learn from interactions and leverage accumulated knowledge.

### Building on Previous Weeks

- **Week 2**: Learned how vector embeddings and semantic search work (ChromaDB, cosine similarity)
- **Week 3**: Built RAG pipelines that retrieve documents and ground LLM responses
- **Week 4 (This)**: Integrate memory into agents so they become stateful, learning systems

## Key Concepts

### Vector Database Memory

Rather than treating each query independently, agents now maintain persistent memory across interactions:

```
User Query 1 → Agent Response + Store in Memory
User Query 2 → Retrieve relevant memory + Context-aware Response
User Query 3 → Recall patterns + Learn facts → Improved Response
```

### Memory Types

1. **Conversation Memory**: Stores past queries and responses for context continuity
2. **Learned Facts Memory**: Stores discovered or clarified information with confidence scores
3. **Interaction Pattern Memory**: Learns common user intents and successful strategies
4. **Error Memory**: Tracks mistakes to avoid repeating them

### Semantic Retrieval

Memories are retrieved using vector similarity, not keyword matching:
- "laptop request" and "I need a computer" map to similar vectors
- Agent finds relevant memories despite different wording
- Typo-resilient because similarity is semantic, not lexical

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│         MemoryAwareHRAgent (Stateful Agent)            │
├─────────────────────────────────────────────────────────┤
│  • Maintains improvement metrics across sessions       │
│  • Recalls context before generating responses        │
│  • Learns new facts and patterns                       │
│  • Tracks errors to improve over time                  │
└────────────┬──────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────┐
│      VectorMemoryManager (Multi-Type DB)               │
├─────────────────────────────────────────────────────────┤
│  • Manages 4 separate vector collections              │
│  • Handles embeddings with sentence-transformers      │
│  • Provides semantic retrieval across memory types    │
│  • Supports persistent storage with ChromaDB          │
└────────────┬──────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────┐
│       ChromaDB Vector Database (Persistent)            │
├─────────────────────────────────────────────────────────┤
│  • Conversation Memory (queries + responses)          │
│  • Learned Facts (discovered knowledge)               │
│  • Patterns (common behaviors)                        │
│  • Errors (mistakes to learn from)                    │
└─────────────────────────────────────────────────────────┘
```

## Quick Start

### Installation

```bash
pip install chromadb sentence-transformers pandas numpy
```

### Basic Usage

```python
from sentence_transformers import SentenceTransformer
import chromadb

# Initialize memory system
memory_manager = VectorMemoryManager(model_name='all-MiniLM-L6-v2')

# Create a stateful agent
hr_agent = MemoryAwareHRAgent(memory_manager)

# Agent learns from interactions
result = hr_agent.process_query("How do I request a new laptop?")

# Agent automatically uses learned knowledge for future queries
result2 = hr_agent.process_query("Can I get a computer for remote work?")
# ✓ Agent recalls previous conversation about laptops
# ✓ Uses learned facts about equipment policies
# ✓ Provides more informed response
```

## Notebook Structure

The notebook is organized into 16 sections:

1. **Setup and Dependencies** - Install and import required packages
2. **Memory Data Structures** - Define schemas for different memory types
3. **Multi-Type Vector Database Manager** - Core memory management class
4. **Stateful Agent with Memory Integration** - MemoryAwareHRAgent implementation
5. **Initialize and Demonstrate** - Set up the system
6. **Agent Learning Over Time** - Simulate multi-turn interactions
7. **Store Learned Facts** - Agent discovers and stores knowledge
8. **Store Error Examples** - Learning from mistakes
9. **Store Interaction Patterns** - Common behaviors and strategies
10. **Memory Recall** - Agent uses learned knowledge
11. **Memory Statistics** - Analyze learning progress
12. **Semantic Similarity** - Show how vectors find related concepts
13. **Memory Persistence** - State across sessions
14. **Multi-Agent Memory Sharing** - Organizational learning
15. **Key Takeaways** - Best practices and patterns
16. **Summary** - Recap and next steps

## Use Cases

### 1. Customer Service Agents
- Remember customer preferences and history
- Improve recommendations based on past interactions
- Avoid repeating mistakes or asking for information twice

### 2. Enterprise Knowledge Management
- Multiple agents share organizational knowledge
- Consistent responses across company
- Reduced need for training individual agents

### 3. Conversational AI Systems
- Maintain context across multi-turn conversations
- Learn common user intents and adapt responses
- Personalize interactions based on interaction history

### 4. Compliance and Audit Trails
- Store all interactions with timestamps
- Track decision reasoning and sources
- Maintain audit trails for regulatory requirements

## Technical Details

### Embedding Model
- **Model**: `all-MiniLM-L6-v2` (384-dimensional vectors)
- **Dimensions**: 384 (balance between speed and quality)
- **Training Data**: 215M diverse examples
- **Performance**: Fast inference, good semantic understanding

### Vector Database
- **Store**: ChromaDB (in-memory or persistent)
- **Similarity Metric**: Cosine similarity
- **Retrieval Method**: Approximate Nearest Neighbors (ANN)
- **Complexity**: O(log n) with proper indexing

### Memory Retrieval

For a query like "I need a new computer":

```
Step 1: Generate embedding for query
Step 2: Search across all memory types using cosine similarity
Step 3: Return top-k results (ranked by similarity)
Step 4: Agent integrates retrieved context into response

Results:
  Learned Fact: "Laptop policy: $2000 budget, 5 days approval"
  Pattern: "Equipment requests often for remote work"
  Past Error: "Don't forget to mention budget limits"
  
Response: [Informed by all retrieved memories]
```

## Performance Characteristics

| Operation | Time | Notes |
|-----------|------|-------|
| Embedding generation | ~10ms | Per query/document |
| Memory storage | <1ms | Per entry |
| Memory retrieval (top-3) | 1-5ms | From 100+ entries |
| Multi-type search | 5-15ms | Searching all 4 types |
| Full agent cycle | 50-100ms | Including all operations |

## Advanced Topics

### Memory Decay
Implement temporal relevance by weighting recent memories higher:
```python
weight = 1.0 / (1 + days_since_interaction)
```

### Confidence Scoring
Track reliability of learned facts:
```python
fact_entry.confidence = 0.95  # High confidence (verified)
fact_entry.confidence = 0.60  # Lower confidence (inferred)
```

### User-Specific Memory Isolation
Implement privacy by namespacing memory:
```python
memory_manager.store_memory(entry, namespace=f"user_{user_id}")
```

### Distributed Memory
Scale to multiple agents/regions with distributed vector databases:
- Milvus (open-source, distributed)
- Pinecone (managed, serverless)
- Weaviate (cloud-native)

## Extending This System

1. **Add Memory Decay**: Weight recent memories higher than old ones
2. **Implement User Namespacing**: Separate memories per user for privacy
3. **Add Feedback Loops**: Improve memory quality based on outcomes
4. **Visualize Memory**: Dashboard showing what agents have learned
5. **Export/Import**: Save memory to disk for backup/transfer
6. **Multi-modal Memory**: Store images, documents, not just text

## Requirements

- Python 3.8+
- ChromaDB for vector storage
- Sentence Transformers for embeddings
- Pandas for data manipulation
- NumPy for numerical operations

## Key Differences from Week 3 RAG

| Aspect | Week 3 RAG | Week 4 Memory |
|--------|-----------|---------------|
| Scope | Document retrieval | Agent state management |
| Memory Type | Static documents | Multiple dynamic types |
| Learning | No | Yes (facts, patterns, errors) |
| Persistence | Required | Optional but important |
| Query Focus | Single query | Multi-turn context |
| Agent State | Stateless | Stateful |

## Expected Outcomes

After completing this notebook, you'll understand:

✓ How to build persistent memory systems for agents  
✓ Semantic retrieval using vector similarity  
✓ Multiple memory types for different purposes  
✓ How agents improve over time through learning  
✓ Scaling memory to organizational level  
✓ Best practices for production memory systems  

## References

- ChromaDB Documentation: https://docs.trychroma.com/
- Sentence Transformers: https://www.sbert.net/
- Vector Databases: https://en.wikipedia.org/wiki/Vector_database
- Semantic Search: https://en.wikipedia.org/wiki/Semantic_search

## Next Steps

1. Run the notebook end-to-end
2. Experiment with different embedding models
3. Extend with persistent storage using ChromaDB's file backend
4. Implement memory decay for temporal relevance
5. Add user-specific memory isolation
6. Build a web interface to interact with the memory-aware agent

---

**Course**: Employability Life - Agentic AI & RAG  
**Sprint**: 2 (Vector Database Memory)  
**Author**: Mujtaba Hyder 
**Date**: 2026-08-29
