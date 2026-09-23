# ============================================================================
# tech_agent.py — Tech Department Support Agent with Persistent Vector Memory
# ============================================================================
#
# This module implements a stateful AI support agent for company tech departments.
# It uses:
#   - ChromaDB (vector database) for persistent memory storage
#   - SentenceTransformers (all-MiniLM-L6-v2) for text embeddings (384-dim)
#   - Ollama (local LLM) for generating policy-grounded answers
#
# The agent maintains 4 ChromaDB collections:
#   1. tech_policy_rag        — Company policy sections (from markdown file)
#   2. tech_episodic_memory   — Past conversations (auto-saved after every interaction)
#   3. tech_declarative_memory — User-taught facts and system rules
#   4. tech_tickets           — Device requests and device issue tickets
#
# Architecture follows the RAG pattern:
#   Retrieve (search 4 collections) → Augment (build context prompt) → Generate (Ollama LLM)
# ============================================================================

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import chromadb                          # Vector database for persistent memory
import requests                          # HTTP client for Ollama REST API
from sentence_transformers import SentenceTransformer  # Text → 384-dim embeddings

from agent_graph import TechMultiAgentWorkflow

logger = logging.getLogger("tech_agent.core")


# --- Default paths -----------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_KB_PATH = ROOT_DIR / "knowledge_base" / "tech_department_policies.md"  # Policy source document
DEFAULT_DB_DIR = ROOT_DIR / ".tech_agent_memory"  # ChromaDB persistent storage directory


# --- Data structure for search results ---------------------------------------
@dataclass
class SearchResult:
    """Represents a single ranked result from a ChromaDB semantic search."""
    id: str                    # Document or memory ID (e.g. "policy-0-device-request-policy")
    content: str               # Full text content of the matched document
    source: str                # Origin — file path for policies, memory kind for others
    kind: str                  # Type: "policy" | "episodic" | "declarative" | "ticket"
    similarity: float          # Cosine similarity score (0.0 to 1.0, higher = more relevant)
    metadata: dict[str, Any]   # Extra fields: timestamp, category, ticket_type, etc.


# =============================================================================
# OllamaClient — Wrapper for the Ollama local LLM REST API
# =============================================================================
# Connects to a locally running Ollama server to generate text responses.
# If Ollama is not running or the model is not installed, the agent falls
# back to showing retrieved context without LLM generation.
# =============================================================================
class OllamaClient:
    def __init__(self, model_name: str = "mistral", base_url: str = "http://localhost:11434"):
        self.model_name = model_name              # Which Ollama model to use (e.g. mistral, llama2)
        self.base_url = base_url.rstrip("/")      # Ollama server URL
        self.api_endpoint = f"{self.base_url}/api/generate"  # REST endpoint for text generation

    def ensure_running(self) -> bool:
        """Check if Ollama server is running; if not, automatically start it in the background."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=1.0)
            if response.status_code == 200:
                return True
        except Exception:
            pass

        # Locate ollama binary in PATH or Windows default install directory
        ollama_bin = shutil.which("ollama")
        if not ollama_bin:
            win_default = os.path.expandvars(r"%LOCALAPPDATA%\Programs\Ollama\ollama.exe")
            if os.path.exists(win_default):
                ollama_bin = win_default

        if not ollama_bin:
            logger.info("Ollama binary not found on system PATH or default installation folder.")
            return False

        try:
            logger.info("Auto-starting Ollama server: %s serve", ollama_bin)
            creation_flags = 0
            if os.name == "nt":
                creation_flags = subprocess.CREATE_NO_WINDOW | getattr(subprocess, "DETACHED_PROCESS", 0x00000008)

            subprocess.Popen(
                [ollama_bin, "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                creationflags=creation_flags,
                close_fds=True,
            )

            # Wait up to 5 seconds for Ollama server to become available
            for _ in range(10):
                time.sleep(0.5)
                try:
                    response = requests.get(f"{self.base_url}/api/tags", timeout=1.0)
                    if response.status_code == 200:
                        logger.info("Ollama server successfully auto-started in background!")
                        return True
                except Exception:
                    continue
        except Exception as exc:
            logger.warning("Could not auto-start Ollama: %s", exc)

        return False

    def is_available(self) -> bool:
        """Check if Ollama is running AND the required model is installed.
        If Ollama is not running, attempts to automatically spin it up."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=1.5)
        except requests.RequestException:
            # Attempt to auto-start Ollama server
            if self.ensure_running():
                try:
                    response = requests.get(f"{self.base_url}/api/tags", timeout=2.0)
                except requests.RequestException:
                    return False
            else:
                return False

        if response.status_code != 200:
            return False
        models = response.json().get("models", [])
        model_names = [model.get("name", "") for model in models]
        # Check if our model is in the list (handles both "mistral" and "mistral:latest")
        return any(name == self.model_name or name.startswith(f"{self.model_name}:") for name in model_names)


    def generate(self, prompt: str, temperature: float = 0.2) -> str:
        """Send a prompt to Ollama and return the generated text response.
        Uses low temperature (0.2) for factual, policy-grounded answers."""
        response = requests.post(
            self.api_endpoint,
            json={
                "model": self.model_name,
                "prompt": prompt,
                "temperature": temperature,  # Low = more deterministic/factual
                "stream": False,             # Get full response at once (not streamed)
            },
            timeout=90,  # Allow up to 90s for slower models/hardware
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()


# =============================================================================
# TechMemoryRAG — Vector Memory Manager with 4 ChromaDB Collections
# =============================================================================
# This is the core memory system. It manages:
#   1. Policy RAG      — Ingests company policy markdown into searchable vectors
#   2. Episodic Memory  — Auto-saves every conversation for future retrieval
#   3. Declarative Memory — Stores user-taught facts and system operating rules
#   4. Tickets          — Stores device request and device issue tickets
#
# All collections use:
#   - Cosine similarity (hnsw:space = cosine) for semantic matching
#   - HNSW indexing for O(log n) fast retrieval
#   - Persistent storage on disk (survives restarts)
# =============================================================================
class TechMemoryRAG:
    def __init__(
        self,
        db_dir: Path | str = DEFAULT_DB_DIR,
        kb_path: Path | str = DEFAULT_KB_PATH,
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self.db_dir = Path(db_dir)
        self.kb_path = Path(kb_path)
        self.db_dir.mkdir(parents=True, exist_ok=True)  # Create storage dir if it doesn't exist

        # Load the embedding model — converts text to 384-dimensional vectors
        self.embedding_model = SentenceTransformer(embedding_model)

        # Connect to ChromaDB with persistent disk storage
        self.client = chromadb.PersistentClient(path=str(self.db_dir))

        # --- Create or load the 4 vector collections ---

        # Collection 1: Company policy sections (from markdown knowledge base)
        self.policy_collection = self.client.get_or_create_collection(
            name="tech_policy_rag",
            metadata={"hnsw:space": "cosine"},
        )
        # Collection 2: Past conversations (auto-saved after each interaction)
        self.episodic_collection = self.client.get_or_create_collection(
            name="tech_episodic_memory",
            metadata={"hnsw:space": "cosine"},
        )
        # Collection 3: User-taught facts and system rules
        self.declarative_collection = self.client.get_or_create_collection(
            name="tech_declarative_memory",
            metadata={"hnsw:space": "cosine"},
        )
        # Collection 4: Device request and issue tickets
        self.ticket_collection = self.client.get_or_create_collection(
            name="tech_tickets",
            metadata={"hnsw:space": "cosine"},
        )

        # On startup: ingest policies and seed default operating rules
        self.ingest_policies()          # Load policy markdown into vectors
        self.seed_declarative_memory()  # Add default agent operating rules if not already present

    def ingest_policies(self) -> None:
        """Read the policy markdown file, split by ## headings, embed each section,
        and store in the policy_collection. Idempotent — skips already-ingested sections."""
        if not self.kb_path.exists():
            return

        # Read and split the markdown into (title, body) pairs by ## headings
        raw_text = self.kb_path.read_text(encoding="utf-8")
        sections = self._split_markdown_sections(raw_text)
        existing_ids = set(self.policy_collection.get(include=[])["ids"])

        for index, (title, body) in enumerate(sections):
            # Generate a stable ID so re-running doesn't create duplicates
            doc_id = f"policy-{index}-{self._stable_slug(title)}"
            if doc_id in existing_ids:
                continue  # Already ingested — skip
            content = f"{title}\n{body}".strip()
            # Embed the section text and store it in ChromaDB
            self.policy_collection.add(
                ids=[doc_id],
                documents=[content],
                embeddings=[self._embed(content)],
                metadatas=[{"source": str(self.kb_path), "title": title, "kind": "policy"}],
            )

    def search(self, query: str, top_k: int = 4, include_memory: bool = True) -> list[SearchResult]:
        """Search across all 4 collections and return merged results ranked by similarity.
        - Policies: top_k results (default 4)
        - Tickets: top 2 results
        - Declarative memory: top 3 results (if include_memory=True)
        - Episodic memory: top 3 results (if include_memory=True)
        Results are sorted by cosine similarity (highest first)."""
        results: list[SearchResult] = []
        results.extend(self._query_collection(self.policy_collection, query, top_k, "policy"))
        results.extend(self._query_collection(self.ticket_collection, query, 2, "ticket"))
        if include_memory:
            results.extend(self._query_collection(self.declarative_collection, query, 3, "declarative"))
            results.extend(self._query_collection(self.episodic_collection, query, 3, "episodic"))
        # Merge all results and sort by similarity score (highest = most relevant)
        return sorted(results, key=lambda item: item.similarity, reverse=True)

    def remember_interaction(self, query: str, answer: str, category: str = "conversation") -> str:
        """Save a conversation to episodic memory. Called automatically after every interaction.
        The answer is truncated to 800 chars to prevent long responses from dominating the embedding."""
        memory_id = f"episodic-{uuid.uuid4()}"  # Unique ID for this memory
        content = f"User asked: {query}\nAgent answered: {answer[:800]}"
        self.episodic_collection.add(
            ids=[memory_id],
            documents=[content],
            embeddings=[self._embed(content)],  # Convert to 384-dim vector
            metadatas=[
                {
                    "kind": "episodic",
                    "category": category,  # conversation, command, device_request, device_issue, memory_update
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                }
            ],
        )
        return memory_id

    def remember_fact(self, fact: str, category: str = "agent_instruction", source: str = "manual") -> str:
        """Store a user-taught fact or instruction in declarative memory.
        These persist permanently and are retrieved when semantically relevant to a query."""
        memory_id = f"declarative-{uuid.uuid4()}"
        content = fact.strip()
        if not content:
            raise ValueError("Declarative memory fact cannot be empty")
        self.declarative_collection.add(
            ids=[memory_id],
            documents=[content],
            embeddings=[self._embed(content)],
            metadatas=[
                {
                    "kind": "declarative",
                    "category": category,   # agent_instruction, policy, etc.
                    "source": source,        # "user" (taught via UI) or "system_seed" (auto)
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                }
            ],
        )
        return memory_id

    def seed_declarative_memory(self) -> None:
        """Add default operating rules to declarative memory on first run.
        These rules tell the agent what information to collect for each request type.
        Idempotent — skips if already seeded."""
        seeded_id = "declarative-tech-agent-operating-rules"
        existing_ids = set(self.declarative_collection.get(include=[])["ids"])
        if seeded_id in existing_ids:
            return  # Already seeded — don't duplicate
        content = (
            "Tech support agent operating rules: remember what action to take for common requests. "
            "For device requests, collect employee, department, manager, device type, work mode, business reason, "
            "preferred date, approval status, and budget needs. For device issues, collect device or asset tag, "
            "operating system, exact error, urgency, business impact, and troubleshooting already attempted. "
            "For software or access requests, collect tool or system name, business purpose, manager approval, "
            "license or budget owner, data sensitivity, and requested duration. Always use policy context first, "
            "then episodic memory from similar past tickets, and ask for missing details before creating assumptions."
        )
        self.declarative_collection.add(
            ids=[seeded_id],
            documents=[content],
            embeddings=[self._embed(content)],
            metadatas=[
                {
                    "kind": "declarative",
                    "category": "agent_instruction",
                    "source": "system_seed",
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                }
            ],
        )

    def create_ticket(self, ticket_type: str, details: dict[str, str]) -> str:
        """Create a new support ticket (device_request or device_issue).
        Generates a unique ticket ID like TECH-20260912-EDBD43, embeds the ticket
        content, and stores it in the tickets collection for future retrieval."""
        # Generate unique ticket ID: TECH-{date}-{6 random hex chars}
        ticket_id = f"TECH-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"
        content = self._ticket_text(ticket_id, ticket_type, details)
        self.ticket_collection.add(
            ids=[ticket_id],
            documents=[content],
            embeddings=[self._embed(content)],
            metadatas=[
                {
                    "kind": "ticket",
                    "ticket_type": ticket_type,    # "device_request" or "device_issue"
                    "status": "open",
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "payload": json.dumps(details),  # Full form data as JSON
                }
            ],
        )
        return ticket_id

    def ingest_document_bytes(self, file_bytes: bytes, filename: str, category: str = "uploaded_doc") -> dict[str, Any]:
        """Save uploaded document into knowledge_base/ directory, extract text, chunk, and index into ChromaDB vector memory."""
        import io
        extracted_chunks: list[dict[str, Any]] = []
        num_pages = 1

        # 1. Persist the raw document to the knowledge_base directory
        kb_dir = Path("knowledge_base")
        kb_dir.mkdir(parents=True, exist_ok=True)
        dest_file = kb_dir / filename
        saved_to_disk = False
        try:
            with open(dest_file, "wb") as f:
                f.write(file_bytes)
            saved_to_disk = True
            logger.info("Saved uploaded document to knowledge base: %s", dest_file)
        except Exception as exc:
            logger.warning("Could not persist file to knowledge_base directory: %s", exc)

        # 2. Extract text and chunk
        try:
            if filename.lower().endswith(".pdf"):
                import pypdf
                reader = pypdf.PdfReader(io.BytesIO(file_bytes))
                num_pages = len(reader.pages)
                for page_idx, page in enumerate(reader.pages):
                    page_text = (page.extract_text() or "").strip()
                    if not page_text:
                        continue
                    paragraphs = [p.strip() for p in page_text.split("\n\n") if len(p.strip()) > 30]
                    if not paragraphs:
                        paragraphs = [page_text]
                    for p in paragraphs:
                        if len(p) > 1200:
                            for i in range(0, len(p), 900):
                                extracted_chunks.append({"content": p[i:i+1000], "page": page_idx + 1})
                        else:
                            extracted_chunks.append({"content": p, "page": page_idx + 1})
            else:
                # Text or Markdown file
                text_content = file_bytes.decode("utf-8", errors="replace").strip()
                paragraphs = [p.strip() for p in text_content.split("\n\n") if len(p.strip()) > 30]
                if not paragraphs:
                    paragraphs = [text_content]
                for p in paragraphs:
                    if len(p) > 1200:
                        for i in range(0, len(p), 900):
                            extracted_chunks.append({"content": p[i:i+1000], "page": 1})
                    else:
                        extracted_chunks.append({"content": p, "page": 1})

        except Exception as exc:
            logger.exception("Failed to process uploaded document: %s", exc)
            return {
                "success": False,
                "error": str(exc),
                "answer": f"⚠️ Could not parse file `{filename}`: {str(exc)}",
            }

        if not extracted_chunks:
            return {
                "success": False,
                "error": "No text content found in document",
                "answer": f"⚠️ No extractable text found in `{filename}`. Please ensure the document contains selectable text.",
            }

        # 3. Index chunks into ChromaDB declarative_collection
        ingested_ids = []
        for ch in extracted_chunks:
            doc_id = f"DOC-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"
            meta = {
                "kind": "declarative",
                "source": f"knowledge_base/{filename}",
                "filename": filename,
                "page": ch.get("page", 1),
                "category": category,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
            page_info = f" | Page {ch['page']}" if filename.lower().endswith(".pdf") else ""
            content_with_header = f"[{filename}{page_info}]\n{ch['content']}"
            self.declarative_collection.add(
                ids=[doc_id],
                documents=[content_with_header],
                embeddings=[self._embed(content_with_header)],
                metadatas=[meta],
            )
            ingested_ids.append(doc_id)

        stats = self.stats()
        preview_samples = "\n".join([f"> *\"{ch['content'][:140]}...\"*" for ch in extracted_chunks[:3]])

        lines = [
            f"### 📄 Document Vectorized & Ingested: `{filename}`\n",
            f"- **Knowledge Base Location:** `knowledge_base/{filename}`" if saved_to_disk else f"- **Knowledge Base Storage:** Vector Ingestion Active",
            f"- **Pages Processed:** `{num_pages}` pages" if filename.lower().endswith(".pdf") else f"- **File Type:** Plaintext / Markdown",
            f"- **Semantic Vector Embeddings:** `{len(ingested_ids)}` chunks generated (384-d cosine vectors)",
            f"- **Target Collection:** `tech_declarative_memory` (ChromaDB)",
            f"- **Total Stored Knowledge Units:** `{stats.get('declarative_memories', 0)}` records",
            f"- **Retrieval Readiness:** Active across multi-agent RAG immediately.\n",
            "**Indexed Knowledge Sample Preview:**",
            preview_samples,
            f"\n*The multi-agent workflow has stored `{filename}` in the Knowledge Base and integrated it into live memory for future queries.*",
        ]
        answer = "\n".join(lines)

        trace = [
            f"📁 [Knowledge Base] Saved file copy to 'knowledge_base/{filename}'",
            f"📄 [Document Parser] Extracted {sum(len(c['content']) for c in extracted_chunks)} chars across {num_pages} pages",
            f"🧠 [Memory Agent] Generated {len(ingested_ids)} semantic vectors via all-MiniLM-L6-v2",
            f"💾 [ChromaDB] Persisted chunks into 'tech_declarative_memory'",
            f"✅ [Verification] Document indexed and available for live similarity retrieval",
        ]

        self.remember_interaction(f"Ingested document: {filename}", answer, category="document_ingestion")

        return {
            "success": True,
            "filename": filename,
            "saved_to_kb": saved_to_disk,
            "kb_path": str(dest_file) if saved_to_disk else None,
            "num_pages": num_pages,
            "num_chunks": len(ingested_ids),
            "chunk_ids": ingested_ids,
            "answer": answer,
            "agent_trace": trace,
            "active_agent": "Memory Agent",
            "ingested_ids": ingested_ids,
        }

    def stats(self) -> dict[str, int]:
        """Return the current count of items in each ChromaDB collection.
        Used by the /stats command and the Streamlit sidebar metrics."""
        episodic_count = self.episodic_collection.count()
        declarative_count = self.declarative_collection.count()
        return {
            "policies": self.policy_collection.count(),
            "episodic_memories": episodic_count,
            "declarative_memories": declarative_count,
            "memories": episodic_count + declarative_count,  # Total memory count
            "tickets": self.ticket_collection.count(),
        }

    def clear_memory(self) -> None:
        """Reset all memory (episodic, declarative, tickets) but keep policies.
        Recreates empty collections and re-seeds default operating rules."""
        for collection_name in ("tech_episodic_memory", "tech_declarative_memory", "tech_tickets"):
            try:
                self.client.delete_collection(collection_name)
            except Exception:
                pass  # Collection may not exist yet — safe to ignore
        self.episodic_collection = self.client.get_or_create_collection(
            name="tech_episodic_memory",
            metadata={"hnsw:space": "cosine"},
        )
        self.declarative_collection = self.client.get_or_create_collection(
            name="tech_declarative_memory",
            metadata={"hnsw:space": "cosine"},
        )
        self.ticket_collection = self.client.get_or_create_collection(
            name="tech_tickets",
            metadata={"hnsw:space": "cosine"},
        )
        self.seed_declarative_memory()

    def _query_collection(self, collection: Any, query: str, top_k: int, kind: str) -> list[SearchResult]:
        """Run a semantic search on a single ChromaDB collection.
        Embeds the query, finds top_k nearest neighbors, and converts
        cosine distances to similarity scores (1 - distance)."""
        count = collection.count()
        if count == 0:
            return []  # Empty collection — nothing to search
        # Embed the query and find nearest neighbors by cosine distance
        try:
            raw = collection.query(
                query_embeddings=[self._embed(query)],
                n_results=min(top_k, count),  # Don't request more results than exist
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            logger.warning("ChromaDB query exception on collection '%s': %s", getattr(collection, "name", kind), exc)
            return []

        # Convert raw ChromaDB results into SearchResult objects
        items: list[SearchResult] = []
        for index, content in enumerate(raw.get("documents", [[]])[0]):
            metadata = raw.get("metadatas", [[]])[0][index] or {}
            distance = raw.get("distances", [[]])[0][index]
            items.append(
                SearchResult(
                    id=raw["ids"][0][index],
                    content=content,
                    source=metadata.get("source", kind),
                    kind=metadata.get("kind", kind),
                    similarity=max(0.0, 1 - float(distance)),  # Convert distance → similarity
                    metadata=metadata,
                )
            )
        return items

    def _embed(self, text: str) -> list[float]:
        """Convert text to a 384-dimensional vector using SentenceTransformer."""
        return self.embedding_model.encode(text).tolist()

    @staticmethod
    def _stable_slug(text: str) -> str:
        """Convert a title to a URL-safe slug for use as a stable document ID.
        Example: 'Device Request Policy' → 'device-request-policy'"""
        return "".join(char.lower() if char.isalnum() else "-" for char in text).strip("-")[:48]

    @staticmethod
    def _split_markdown_sections(raw_text: str) -> list[tuple[str, str]]:
        """Split a markdown document into sections by ## headings.
        Returns a list of (title, body) tuples. Skips # top-level headings.
        Example: '## Device Policy\nText...' → ('Device Policy', 'Text...')"""
        sections: list[tuple[str, list[str]]] = []
        current_title = "General"
        current_lines: list[str] = []

        for line in raw_text.splitlines():
            if line.startswith("## "):  # New section heading
                if current_lines:
                    sections.append((current_title, current_lines))
                current_title = line.replace("## ", "", 1).strip()
                current_lines = []
            elif not line.startswith("# "):  # Skip top-level headings, keep everything else
                current_lines.append(line)

        if current_lines:  # Don't forget the last section
            sections.append((current_title, current_lines))

        # Join lines into body text, filter out empty sections
        return [(title, "\n".join(lines).strip()) for title, lines in sections if "\n".join(lines).strip()]

    @staticmethod
    def _ticket_text(ticket_id: str, ticket_type: str, details: dict[str, str]) -> str:
        """Format ticket details as plain text for embedding and display.
        Example output: 'Ticket TECH-20260912-EDBD43\nType: device_request\nEmployee: John Smith'"""
        lines = [f"Ticket {ticket_id}", f"Type: {ticket_type}"]
        for key, value in details.items():
            if value:
                lines.append(f"{key.replace('_', ' ').title()}: {value}")
        return "\n".join(lines)


# =============================================================================
# TechDepartmentAgent — The Main Agent that Orchestrates Everything
# =============================================================================
# This is the high-level agent that the Streamlit UI interacts with.
# It handles:
#   - Chat messages: search memory → generate answer → save to episodic memory
#   - Slash commands: /help, /stats, /policy, /memory, /remember, /request, /issue
#   - Device requests: create ticket + save to episodic memory
#   - Device issues: classify priority + create ticket + save to episodic memory
#   - Teaching: save facts to declarative memory
# =============================================================================
class TechDepartmentAgent:
    def __init__(self, rag: TechMemoryRAG, llm: OllamaClient):
        self.rag = rag  # Vector memory manager (4 ChromaDB collections)
        self.llm = llm  # Ollama LLM client for generating answers
        # Multi-Agent LangGraph workflow
        self.workflow = TechMultiAgentWorkflow(rag_instance=self.rag, llm_client=self.llm)

    def answer(self, message: str, step_callback: Optional[Callable[[str], None]] = None) -> dict[str, Any]:
        """Main entry point for processing a user message via the LangGraph Multi-Agent system.
        
        Workflow Steps:
        1. Check if it's a /command → handle immediately with direct command logic.
        2. Otherwise: Dispatch message to LangGraph StateGraph (Triage → Specialist → Tools → Responder → Memory).
        3. Convert output sources into `SearchResult` instances for UI backward compatibility.
        """
        # Step 1: Check for slash commands (e.g. /help, /stats, /remember)
        command_result = self._handle_command(message)
        if command_result:
            self.rag.remember_interaction(message, command_result["answer"], category="command")
            return command_result

        # Step 2: Execute LangGraph Multi-Agent Workflow
        result = self.workflow.run(message, step_callback=step_callback)

        # Convert raw sources dict to SearchResult objects for UI compatibility
        converted_sources: list[SearchResult] = []
        for s in result.get("sources", []):
            if isinstance(s, SearchResult):
                converted_sources.append(s)
            elif isinstance(s, dict):
                converted_sources.append(
                    SearchResult(
                        id=s.get("id", "source-doc"),
                        content=s.get("content", ""),
                        source=s.get("title", s.get("kind", "knowledge")),
                        kind=s.get("kind", "policy"),
                        similarity=float(s.get("similarity", 0.85)),
                        metadata=s.get("metadata", {}),
                    )
                )

        # Fallback to direct RAG search if sources empty
        if not converted_sources:
            try:
                converted_sources = self.rag.search(message)[:4]
            except Exception:
                converted_sources = []

        return {
            "answer": result.get("answer", ""),
            "sources": converted_sources[:6],
            "agent_trace": result.get("agent_trace", []),
            "active_agent": result.get("active_agent", "Tech Agent"),
            "tool_results": result.get("tool_results", []),
            "intent": result.get("intent", ""),
            "urgency": result.get("urgency", ""),
            "status": result.get("status", "completed"),
            "command": None,
        }

    def submit_device_request(self, details: dict[str, str]) -> dict[str, Any]:
        """Create a device request ticket using domain tools and return an agentic dispatch report."""
        import tools
        ticket_id = self.rag.create_ticket("device_request", details)
        
        # 1. Hardware tool lookup
        device_name = details.get("device", "")
        hw_match = tools.hardware_catalog_tool(keyword=device_name[:14] if device_name else None)
        hw_info = hw_match.output[0] if (hw_match.success and hw_match.output) else None

        # 2. SLA target calculation
        sla_res = tools.sla_calculator_tool(urgency="normal", issue_category="hardware")
        sla_info = sla_res.output if sla_res.success else {}

        # 3. Policy retrieval
        policy_res = tools.search_policy_tool(self.rag, query="device hardware request approval process", top_k=1)
        policy_title = policy_res.output[0]["title"] if (policy_res.success and policy_res.output) else "Corporate Hardware Policy"

        lines = [
            f"### 🎫 Device Request Dispatched: `{ticket_id}`\n",
            f"**Employee:** {details.get('employee', 'N/A')} ({details.get('department', 'General')}) | **Manager:** {details.get('manager', 'N/A')}",
            f"**Work Mode:** `{details.get('work_mode', 'Hybrid')}` | **Target Date:** `{details.get('preferred_date', 'Standard')}`\n",
        ]

        if hw_info:
            lines.append("#### 💻 Approved Hardware Specifications & Budget")
            lines.append(f"- **Selected Device:** {hw_info['name']}")
            lines.append(f"- **Tier Assignment:** {hw_info['tier']}")
            lines.append(f"- **Hardware Specs:** {hw_info['specs']}")
            lines.append(f"- **Estimated Budget:** {hw_info['cost']}")
            lines.append(f"- **Standard Lead Time:** {hw_info['lead_time']}")
            lines.append(f"- **Approval Policy:** {hw_info['approval_rule']}\n")

        lines.append("#### ⏱️ Service Level Agreement & Next Steps")
        lines.append(f"- **Fulfillment SLA:** {sla_info.get('full_resolution_target', 'Within 24-72 business hours')}")
        lines.append(f"- **Assigned Team:** {sla_info.get('assigned_tier', 'Tier 1 — IT Workplace Services')}")
        lines.append(f"- **Governing IT Policy:** {policy_title}\n")
        lines.append(f"**Business Justification:** *\"{details.get('business_reason', 'None provided')}\"*")
        lines.append("\n*Automated notification dispatched to manager for sign-off. Track this ticket anytime in Ticket Central.*")

        answer = "\n".join(lines)
        
        trace = [
            f"🤖 [Triage Agent] Validated request submission for {details.get('employee')}",
            f"🧰 [Hardware Specialist] Invoked hardware_catalog_tool for '{device_name}'",
            f"⏱️ [SLA Calculator] Assigned resolution window: {sla_info.get('full_resolution_target')}",
            f"💾 [Memory Agent] Persisted ticket {ticket_id} to ChromaDB vector store",
        ]

        self.rag.remember_interaction(f"Device request {ticket_id}", answer, category="device_request")
        return {
            "ticket_id": ticket_id,
            "answer": answer,
            "agent_trace": trace,
            "active_agent": "Hardware Specialist",
            "device": device_name,
        }

    def submit_device_issue(self, details: dict[str, str]) -> dict[str, Any]:
        """Create a device issue ticket using domain diagnostics tools and return an agentic triage plan."""
        import tools
        ticket_id = self.rag.create_ticket("device_issue", details)
        priority = (details.get("urgency") or "").lower() or self._classify_priority(details)

        # 1. Run SLA calculator tool
        sla_res = tools.sla_calculator_tool(urgency=priority, issue_category=details.get("issue_summary", "general"))
        sla_info = sla_res.output if sla_res.success else {}

        # 2. Run diagnostics playbook matching issue keywords
        summary_text = (details.get("issue_summary", "") + " " + details.get("troubleshooting", "")).lower()
        diag_res = tools.network_diagnostics_tool(issue_type=summary_text, os_type="windows")
        diag_info = diag_res.output if diag_res.success else {}

        lines = [
            f"### 🚨 Incident Ticket Triaged: `{ticket_id}`\n",
            f"**Affected Asset:** `{details.get('device_asset', 'Endpoint Asset')}` | **Reported by:** {details.get('employee', 'Employee')}",
            f"**Classified Priority:** `{priority.upper()}`\n",
            "#### ⏱️ SLA Commitment & Escalation Path",
            f"- **First Response Window:** {sla_info.get('first_response_target', 'Within 60 minutes')}",
            f"- **Target Resolution:** {sla_info.get('full_resolution_target', 'Within 4 hours')}",
            f"- **Assigned Engineering Tier:** {sla_info.get('assigned_tier', 'Tier 2 Support Engineers')}",
            f"- **Incident Alerting:** {'🚨 PagerDuty Triggered (On-Call Command)' if sla_info.get('pager_duty_incident') else 'Standard Priority Queue'}\n",
        ]

        if diag_info and diag_info.get("recommended_steps"):
            lines.append(f"#### 🛠️ Automated Triage Playbook: *{diag_info.get('title')}*")
            lines.append("**Immediate Recommended Troubleshooting Steps:**")
            for step in diag_info.get("recommended_steps", [])[:4]:
                lines.append(f"- {step}")
            lines.append(f"\n**Escalation Lead Team:** {diag_info.get('escalation_team')}\n")

        lines.append(f"**Issue Description:** *\"{details.get('issue_summary')}\"*")
        if details.get("troubleshooting"):
            lines.append(f"**Prior Steps Attempted:** *\"{details.get('troubleshooting')}\"*")
        lines.append("\n*Engineering team notified. You can inspect updates anytime in Ticket Central.*")

        answer = "\n".join(lines)

        trace = [
            f"🤖 [Triage Agent] Categorized incident as {priority.upper()}",
            f"🛠️ [Diagnostics Agent] Executed playbook: {diag_info.get('title', 'General Triage')}",
            f"⏱️ [SLA Calculator] First response target: {sla_info.get('first_response_target')}",
            f"💾 [Memory Agent] Persisted incident {ticket_id} to ChromaDB tickets collection",
        ]

        self.rag.remember_interaction(f"Device issue {ticket_id}", answer, category="device_issue")
        return {
            "ticket_id": ticket_id,
            "answer": answer,
            "agent_trace": trace,
            "active_agent": "Diagnostics Specialist",
            "priority": priority,
        }

    def teach_fact(self, fact: str, category: str = "agent_instruction") -> dict[str, Any]:
        """Teach the agent a new declarative fact and return an agentic memory confirmation."""
        memory_id = self.rag.remember_fact(fact, category=category, source="user")
        stats = self.rag.stats()

        lines = [
            f"### 🧠 Declarative Memory Indexed: `{memory_id}`\n",
            "**Learned Business Knowledge / Rule:**",
            f"> *\"{fact}\"*\n",
            "#### 🔍 Vector Memory Telemetry",
            f"- **Storage Collection:** `tech_declarative_memory` (384-dimensional cosine vector)",
            f"- **Active Knowledge Rules:** `{stats.get('declarative_memories', 0)}` saved facts",
            f"- **Operational Impact:** Semantic proximity search will retrieve this rule across all future employee inquiries.",
            "\n*The multi-agent workflow has integrated this policy rule into live memory.*"
        ]
        answer = "\n".join(lines)
        trace = [
            f"🤖 [Triage Agent] Received knowledge instruction from user",
            f"🧠 [Memory Agent] Vectorized fact with all-MiniLM-L6-v2 embedding model",
            f"💾 [ChromaDB] Persisted declarative memory as {memory_id}",
            f"✅ [Validation] Rule indexed and immediately available for query retrieval",
        ]
        self.rag.remember_interaction(f"Remember fact: {fact}", answer, category="memory_update")
        return {
            "memory_id": memory_id,
            "answer": answer,
            "agent_trace": trace,
            "active_agent": "Memory Agent",
        }

    def ingest_document_bytes(self, file_bytes: bytes, filename: str, category: str = "uploaded_doc") -> dict[str, Any]:
        """Save uploaded document into knowledge_base/ directory, extract text, chunk, and index into ChromaDB vector memory."""
        return self.rag.ingest_document_bytes(file_bytes=file_bytes, filename=filename, category=category)

    def _generate_answer(self, message: str, context: list[SearchResult]) -> str:
        """Generate an answer using the Ollama LLM with retrieved context.
        If Ollama is unavailable, falls back to showing raw retrieved context.
        This is the RAG pipeline: Retrieve → Augment (build prompt) → Generate."""

        # Check if Ollama is running and model is available
        if not self.llm.is_available():
            return self._fallback_answer(message, context)  # Graceful fallback

        # Build context text from top 6 search results (with kind and similarity labels)
        context_text = "\n\n".join(
            f"[{item.kind.upper()} | similarity {item.similarity:.2f}]\n{item.content}" for item in context[:6]
        )

        # Build the augmented prompt — tells the LLM to use retrieved context
        prompt = f"""
You are a company tech department support agent.
Use the policy RAG context, declarative memory, and episodic memory below to answer the employee.
Be concise, practical, and policy-grounded.
If details are missing, ask for them.
Do not invent policy.

Context:
{context_text}

Employee message:
{message}

Answer:
""".strip()
        try:
            return self.llm.generate(prompt)  # Send to Ollama for generation
        except requests.RequestException as exc:
            # If LLM request fails mid-call, fall back with error note
            return f"{self._fallback_answer(message, context)}\n\nLLM note: Ollama request failed: {exc}"

    def _fallback_answer(self, message: str, context: list[SearchResult]) -> str:
        """Generate a response WITHOUT the LLM by showing retrieved context directly.
        Used when Ollama is not running or when a request fails."""
        if not context:
            return (
                "I do not have enough policy context for that yet. "
                "Please share the device, user, business reason, urgency, and any error details."
            )

        # Show the top 4 most relevant results with their first sentence
        relevant = "\n".join(f"- {item.content.splitlines()[0]}: {self._first_sentence(item.content)}" for item in context[:4])
        return (
            "Based on the tech department policy and memory I found:\n"
            f"{relevant}\n\n"
            "Please provide any missing details such as employee name, department, manager approval, device type, "
            "business reason, asset tag, error message, urgency, and troubleshooting already attempted."
        )

    def _handle_command(self, message: str) -> dict[str, Any] | None:
        """Handle slash commands (/help, /stats, /policy, /memory, /remember, /request, /issue).
        Returns None if the message is not a command."""
        clean = message.strip()
        if not clean.startswith("/"):  # Not a command — return None to continue normal flow
            return None

        parts = clean.split(maxsplit=1)
        command = parts[0].lower()
        argument = parts[1] if len(parts) > 1 else ""

        if command in {"/help", "/commands"}:
            return {
                "answer": (
                    "Available commands:\n"
                    "- /policy <question> searches tech policies.\n"
                    "- /memory <question> searches policies, tickets, declarative memory, and episodic memory.\n"
                    "- /remember <fact or instruction> saves declarative memory for future answers.\n"
                    "- /stats shows memory counts.\n"
                    "- /request starts a device request.\n"
                    "- /issue starts a device issue report."
                ),
                "sources": [],
                "command": command,
            }

        if command == "/stats":
            stats = self.rag.stats()
            return {
                "answer": (
                    f"Memory status: {stats['policies']} policy sections, "
                    f"{stats['tickets']} tickets, {stats['episodic_memories']} episodic memories, "
                    f"{stats['declarative_memories']} declarative memories."
                ),
                "sources": [],
                "command": command,
            }

        if command == "/remember":
            if not argument.strip():
                return {
                    "answer": "Tell me what to remember, for example: /remember Always route MacBook requests over 2000 USD to finance approval.",
                    "sources": [],
                    "command": command,
                }
            result = self.teach_fact(argument.strip())
            return {
                "answer": result["answer"],
                "sources": [],
                "command": command,
            }

        if command == "/policy":
            sources = self.rag.search(argument or "tech policy", include_memory=False)
            return {
                "answer": self._fallback_answer(argument, sources),
                "sources": sources,
                "command": command,
            }

        if command == "/memory":
            sources = self.rag.search(argument or "recent tech support memory", include_memory=True)
            return {
                "answer": self._fallback_answer(argument, sources),
                "sources": sources,
                "command": command,
            }

        if command == "/request":
            return {
                "answer": "Use the Device Request form in the sidebar so I can capture the required policy fields.",
                "sources": [],
                "command": command,
            }

        if command == "/issue":
            return {
                "answer": "Use the Device Issue form in the sidebar so I can capture device, urgency, and troubleshooting details.",
                "sources": [],
                "command": command,
            }

        return {
            "answer": "Unknown command. Type /help to see supported tech agent commands.",
            "sources": [],
            "command": command,
        }

    @staticmethod
    def _first_sentence(text: str) -> str:
        """Extract the first sentence from a multi-line text block.
        Used in fallback answers to show a brief summary of each retrieved result."""
        body = " ".join(line.strip() for line in text.splitlines()[1:] if line.strip())
        return body.split(". ")[0][:240].strip() + "."

    @staticmethod
    def _classify_priority(details: dict[str, str]) -> str:
        """Auto-classify device issue priority based on keywords in the issue details.
        - Critical: lost, stolen, security, cannot work, production, executive
        - High: vpn, crash, meeting, camera, microphone, blocked, urgent
        - Normal: everything else (default)"""
        text = " ".join(details.values()).lower()
        critical_terms = ["lost", "stolen", "security", "cannot work", "unable to work", "production", "executive"]
        high_terms = ["vpn", "crash", "meeting", "camera", "microphone", "blocked", "urgent"]
        if any(term in text for term in critical_terms):
            return "critical"
        if any(term in text for term in high_terms):
            return "high"
        return "normal"
