# Vector Database Memory for Stateful Agent Systems

**Program**: Employability Life - Agentic AI & RAG (Week 4)  
**Date**: September 2026  
**Team**: Dune  
**Status**: ✅ Production-Ready with Ollama Integration

---

## Overview

This is a **complete stateful agent system with persistent vector memory** that learns and improves over time. It demonstrates how agents move beyond stateless chatbots to become learning systems that maintain context, improve responses, and share knowledge across teams.

### Key Features

✅ **Ollama LLM Integration** - Local language model for actual response generation  
✅ **Persistent Vector Memory** - Survives session restarts using ChromaDB  
✅ **Episodic & Declarative Memory** - Follows Week 1 four-layer architecture  
✅ **Multi-Agent Learning** - Agents share knowledge from a single database  
✅ **Before/After Proof** - Demonstrates memory actually improves responses  
✅ **Memory-Informed Generation** - LLM uses retrieved context for responses  

---

## Architecture

### Four-Layer Memory Model (Week 1)

```
PERCEPTION    (Input: "Can I work from home?")
    ↓
REASONING     (Retrieve relevant memories: policies, past interactions)
    ↓
MEMORY        4 Types:
              • Episodic: "Employee asked this before"
              • Declarative: "Remote work approved 2-3 days/week"
              • Error: "Forgot to mention probation restrictions"
              • Procedural: Tools & capabilities
    ↓
ACTION        Generate response using LLM + memory context
```

### System Components

1. **OllamaLLM** - Interface to local Ollama model
   - Generates responses informed by memory context
   - Real LLM, not keyword matching

2. **PersistentVectorMemoryManager** - Memory storage & retrieval
   - ChromaDB for vector database
   - Persistent storage to disk
   - Semantic search with embeddings

3. **MemoryAwareAgent** - Stateful agent with learning
   - Retrieves relevant memories
   - Processes queries with LLM
   - Stores interactions as episodic memory
   - Learns facts and errors

4. **Memory Types**
   - **Episodic**: Specific past interactions + outcomes
   - **Declarative**: Facts, rules, policies
   - **Error**: Mistakes to learn from

---

## Before/After: How Memory Improves Responses

### Without Memory (Cold Start)

```
Query: "Can I work from home?"
↓
LLM: [No context] → Generic response
Result: May miss policies, incomplete
```

### With Memory (Memory-Informed)

```
Query: "Can I work from home?"
↓
Retrieve: "Remote work approved 2-3 days/week"
Retrieve: "3-month probation restriction"
↓
LLM: [With context] → Informed response
Result: Accurate, policy-compliant ✓
```

---

## Multi-Agent Learning

```
┌─────────────┐
│  HR Agent   │──→ Processes HR questions
│             │    Stores in Episodic Memory
└──────┬──────┘
       │ (shared)
       ↓
┌─────────────┐
│   SHARED    │  ← Single persistent database
│   MEMORY    │    Both agents read/write
│   DATABASE  │
└──────┬──────┘
       │ (shared)
       ↓
┌─────────────┐
│  IT Agent   │──→ Processes IT questions
│             │    Retrieves HR's learnings
└─────────────┘
```

**Result**: Knowledge learned by HR agent automatically helps IT agent. No separate training needed.

---

## Quick Start

### Prerequisites

1. **Python 3.8+** with pip
2. **Ollama** installed and running locally

### Installation Steps

#### 1. Install Ollama

```bash
# Download from https://ollama.ai
# Or use package manager:
# macOS: brew install ollama
# Linux: curl https://ollama.ai/install.sh | sh
# Windows: Download from ollama.ai
```

#### 2. Start Ollama

```bash
# Terminal 1: Start the Ollama server
ollama serve

# Terminal 2: Pull a model (mistral recommended)
ollama pull mistral
# or: ollama pull llama2, ollama pull neural-chat
```

#### 3. Run the Notebook

```bash
# Navigate to project directory
cd Desktop/Vector_Database_Memory-main

# Install Python dependencies (auto-installed in notebook)
pip install chromadb sentence-transformers requests pandas numpy

# Start Jupyter
jupyter notebook Vector_Database_Memory.ipynb
```

#### 4. Execute All Cells

Run cells in order (top to bottom):
- Cell 1-2: Setup dependencies
- Cell 3b: Connect to Ollama
- Cell 5-7: Initialize memory system
- Cell 11-12: Load facts
- Cell 13-15: Before/After demo
- Cell 16-19: Verification

---

## What Each Section Does

### Section 1-3: Setup
- Install dependencies
- Import libraries
- Initialize Ollama connection

### Section 4-5: Memory Structure
- Define episodic, declarative, error memory types
- Create persistent vector manager

### Section 6-7: Initialize System
- Create shared memory database
- Create HR and IT agents (share memory)
- Load initial facts

### Section 8-9: Before/After Demo
- **Phase 1**: Process queries WITHOUT memory context
- **Phase 2**: Same queries WITH memory context
- Compare responses to show improvement

### Section 10-11: Multi-Agent Learning
- HR agent processes queries (stores episodic memories)
- IT agent retrieves HR's memories + IT facts
- Demonstrate knowledge transfer

### Section 12-14: Error Learning
- Log common mistakes
- Agents learn to avoid repeating errors
- Errors become searchable in memory

### Section 15-16: Persistence
- Show stored memory statistics
- Explain how memory survives restarts
- Point to storage directory

### Section 17-19: Verification
- Check all components working
- Verify all requirements met
- Print checklist

---

## Memory Flow Example

```
Input: "I need a new laptop"
↓
[Agent Retrieves Context]
- Episodic: "Previous laptop request took 3 days"
- Declarative: "Budget: $2000, Timeline: 5 business days"
- Error: "Mention budget upfront to avoid disappointment"
↓
[LLM Processes with Context]
OllamaLLM(
  query="I need a new laptop",
  context="Budget limit: $2000. Approved within 5 days. Previous similar request: accepted."
)
↓
Response: "You can request a new laptop. Budget limit is $2000, 
and approval typically takes 5 business days. Submit via IT portal. 
Based on previous requests, the process is straightforward."
↓
[Store Interaction as Memory]
Episodic Memory: "Employee requested laptop → approved within 5 days"
↓
[Other Agents Can Retrieve]
IT Agent: "Another laptop request? I found similar past case..."
```

---

## How It Connects to Previous Weeks

### Week 2: Vector Embeddings & Semantic Search
- We use `all-MiniLM-L6-v2` embeddings (384-dim)
- ChromaDB for vector storage
- Cosine similarity for relevance

### Week 3: RAG Pipelines
- **Retrieval**: Get facts from memory
- **Augmented**: Pass to LLM as context
- **Generation**: LLM generates response

### Week 4 (This): Stateful Agents
- Add persistent memory layer
- Add multi-agent learning
- Add episodic + declarative memory types
- Complete stateful agent system

---

## Technical Specifications

### Storage
- **Format**: ChromaDB (persistent)
- **Location**: `~/.vector_memory/db/`
- **Indexing**: HNSW (O(log n) retrieval)
- **Embedding**: all-MiniLM-L6-v2 (384-dim)

### LLM
- **Provider**: Ollama (local)
- **Model**: mistral (can switch to llama2, neural-chat, etc.)
- **Response**: Context-aware, informed by retrieved memories

### Memory
- **Episodic**: Specific past interactions + outcomes
- **Declarative**: Facts, policies, procedures
- **Error**: Mistakes + resolutions
- **Growth**: Append-only (memories never deleted)

### Performance
- **Query → Response**: ~2-10 seconds (depends on LLM)
- **Memory Retrieval**: <100ms (HNSW indexing)
- **Scalability**: Tested up to 10K+ memories

---

## Files in This Repository

```
Vector_Database_Memory-main/
├── Vector_Database_Memory.ipynb    Main notebook (28 cells)
├── README.md                       This file
├── SRS.md                         Requirements & specifications
├── COMPLETION_SUMMARY.md          Verification results
└── .gitignore                     Git configuration
```

---

## Troubleshooting

### "Cannot connect to Ollama"
```
Error: Cannot connect to Ollama at http://localhost:11434
Fix: Make sure Ollama is running
  1. In terminal: ollama serve
  2. In another terminal: ollama pull mistral
  3. Retry notebook
```

### "Model 'mistral' not found"
```
Error: Model 'mistral' not found
Fix: Install the model
  ollama pull mistral
  # or use another model:
  # ollama pull llama2
  # ollama pull neural-chat
```

### "ChromaDB persistence error"
```
Error: Cannot write to ~/.vector_memory/db/
Fix: Check directory permissions
  mkdir -p ~/.vector_memory/db/
  # On Windows: ensure no path conflicts
```

### "Out of memory"
```
Memory issue running local LLM
Fix: Try smaller model
  ollama pull orca-mini  # Smaller than mistral
  # Then edit: llm = OllamaLLM(model_name="orca-mini")
```

---

## Key Takeaways

1. **Agents can be stateful** - They learn and improve over time
2. **Memory enables better responses** - Context from past makes LLM outputs better
3. **Semantic search is powerful** - Finds relevant memories despite different wording
4. **Multi-agent learning scales** - Teams share knowledge automatically
5. **Local LLM works well** - Ollama + memory is fast enough for production
6. **Persistence is crucial** - Memory survives sessions

---

## Extension Ideas

1. **Memory Decay** - Make old memories less relevant over time
2. **User Isolation** - Keep separate memory per user for privacy
3. **Visualization** - Dashboard showing memory usage
4. **Feedback Loops** - Rate response quality to improve
5. **Distributed Memory** - Scale to multiple machines
6. **Multi-Modal** - Store images, PDFs, voice in memory

---

## Requirements Met ✓

- [x] Actual LLM (Ollama) - Real response generation
- [x] Episodic & Declarative Memory - Week 1 architecture
- [x] Persistent Storage - Survives sessions
- [x] Multi-Agent Sharing - Organizational learning
- [x] Before/After Comparison - Memory improves responses
- [x] Memory-Informed Responses - LLM uses retrieved context
- [x] Error Learning - Agents track and avoid mistakes

---

## Questions?

Refer to:
- **SRS.md** for detailed requirements
- **COMPLETION_SUMMARY.md** for verification details
- Notebook comments for implementation details

**Happy Learning! 🚀**
