from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import chromadb
import requests
from sentence_transformers import SentenceTransformer


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_KB_PATH = ROOT_DIR / "knowledge_base" / "tech_department_policies.md"
DEFAULT_DB_DIR = ROOT_DIR / ".tech_agent_memory"


@dataclass
class SearchResult:
    id: str
    content: str
    source: str
    kind: str
    similarity: float
    metadata: dict[str, Any]


class OllamaClient:
    def __init__(self, model_name: str = "mistral", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.api_endpoint = f"{self.base_url}/api/generate"

    def is_available(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            if response.status_code != 200:
                return False
            models = response.json().get("models", [])
            model_names = [model.get("name", "") for model in models]
            return any(name == self.model_name or name.startswith(f"{self.model_name}:") for name in model_names)
        except requests.RequestException:
            return False

    def generate(self, prompt: str, temperature: float = 0.2) -> str:
        response = requests.post(
            self.api_endpoint,
            json={
                "model": self.model_name,
                "prompt": prompt,
                "temperature": temperature,
                "stream": False,
            },
            timeout=90,
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()


class TechMemoryRAG:
    def __init__(
        self,
        db_dir: Path | str = DEFAULT_DB_DIR,
        kb_path: Path | str = DEFAULT_KB_PATH,
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self.db_dir = Path(db_dir)
        self.kb_path = Path(kb_path)
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_model = SentenceTransformer(embedding_model)
        self.client = chromadb.PersistentClient(path=str(self.db_dir))
        self.policy_collection = self.client.get_or_create_collection(
            name="tech_policy_rag",
            metadata={"hnsw:space": "cosine"},
        )
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
        self.ingest_policies()
        self.seed_declarative_memory()

    def ingest_policies(self) -> None:
        if not self.kb_path.exists():
            return

        raw_text = self.kb_path.read_text(encoding="utf-8")
        sections = self._split_markdown_sections(raw_text)
        existing_ids = set(self.policy_collection.get(include=[])["ids"])

        for index, (title, body) in enumerate(sections):
            doc_id = f"policy-{index}-{self._stable_slug(title)}"
            if doc_id in existing_ids:
                continue
            content = f"{title}\n{body}".strip()
            self.policy_collection.add(
                ids=[doc_id],
                documents=[content],
                embeddings=[self._embed(content)],
                metadatas=[{"source": str(self.kb_path), "title": title, "kind": "policy"}],
            )

    def search(self, query: str, top_k: int = 4, include_memory: bool = True) -> list[SearchResult]:
        results: list[SearchResult] = []
        results.extend(self._query_collection(self.policy_collection, query, top_k, "policy"))
        results.extend(self._query_collection(self.ticket_collection, query, 2, "ticket"))
        if include_memory:
            results.extend(self._query_collection(self.declarative_collection, query, 3, "declarative"))
            results.extend(self._query_collection(self.episodic_collection, query, 3, "episodic"))
        return sorted(results, key=lambda item: item.similarity, reverse=True)

    def remember_interaction(self, query: str, answer: str, category: str = "conversation") -> str:
        memory_id = f"episodic-{uuid.uuid4()}"
        content = f"User asked: {query}\nAgent answered: {answer[:800]}"
        self.episodic_collection.add(
            ids=[memory_id],
            documents=[content],
            embeddings=[self._embed(content)],
            metadatas=[
                {
                    "kind": "episodic",
                    "category": category,
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                }
            ],
        )
        return memory_id

    def remember_fact(self, fact: str, category: str = "agent_instruction", source: str = "manual") -> str:
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
                    "category": category,
                    "source": source,
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                }
            ],
        )
        return memory_id

    def seed_declarative_memory(self) -> None:
        seeded_id = "declarative-tech-agent-operating-rules"
        existing_ids = set(self.declarative_collection.get(include=[])["ids"])
        if seeded_id in existing_ids:
            return
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
        ticket_id = f"TECH-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"
        content = self._ticket_text(ticket_id, ticket_type, details)
        self.ticket_collection.add(
            ids=[ticket_id],
            documents=[content],
            embeddings=[self._embed(content)],
            metadatas=[
                {
                    "kind": "ticket",
                    "ticket_type": ticket_type,
                    "status": "open",
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "payload": json.dumps(details),
                }
            ],
        )
        return ticket_id

    def stats(self) -> dict[str, int]:
        episodic_count = self.episodic_collection.count()
        declarative_count = self.declarative_collection.count()
        return {
            "policies": self.policy_collection.count(),
            "episodic_memories": episodic_count,
            "declarative_memories": declarative_count,
            "memories": episodic_count + declarative_count,
            "tickets": self.ticket_collection.count(),
        }

    def clear_memory(self) -> None:
        for collection_name in ("tech_episodic_memory", "tech_declarative_memory", "tech_tickets"):
            try:
                self.client.delete_collection(collection_name)
            except Exception:
                pass
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
        count = collection.count()
        if count == 0:
            return []
        raw = collection.query(
            query_embeddings=[self._embed(query)],
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"],
        )
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
                    similarity=max(0.0, 1 - float(distance)),
                    metadata=metadata,
                )
            )
        return items

    def _embed(self, text: str) -> list[float]:
        return self.embedding_model.encode(text).tolist()

    @staticmethod
    def _stable_slug(text: str) -> str:
        return "".join(char.lower() if char.isalnum() else "-" for char in text).strip("-")[:48]

    @staticmethod
    def _split_markdown_sections(raw_text: str) -> list[tuple[str, str]]:
        sections: list[tuple[str, list[str]]] = []
        current_title = "General"
        current_lines: list[str] = []

        for line in raw_text.splitlines():
            if line.startswith("## "):
                if current_lines:
                    sections.append((current_title, current_lines))
                current_title = line.replace("## ", "", 1).strip()
                current_lines = []
            elif not line.startswith("# "):
                current_lines.append(line)

        if current_lines:
            sections.append((current_title, current_lines))

        return [(title, "\n".join(lines).strip()) for title, lines in sections if "\n".join(lines).strip()]

    @staticmethod
    def _ticket_text(ticket_id: str, ticket_type: str, details: dict[str, str]) -> str:
        lines = [f"Ticket {ticket_id}", f"Type: {ticket_type}"]
        for key, value in details.items():
            if value:
                lines.append(f"{key.replace('_', ' ').title()}: {value}")
        return "\n".join(lines)


class TechDepartmentAgent:
    def __init__(self, rag: TechMemoryRAG, llm: OllamaClient):
        self.rag = rag
        self.llm = llm

    def answer(self, message: str) -> dict[str, Any]:
        command_result = self._handle_command(message)
        if command_result:
            self.rag.remember_interaction(message, command_result["answer"], category="command")
            return command_result

        context = self.rag.search(message)
        answer = self._generate_answer(message, context)
        self.rag.remember_interaction(message, answer)
        return {
            "answer": answer,
            "sources": context[:6],
            "command": None,
        }

    def submit_device_request(self, details: dict[str, str]) -> dict[str, Any]:
        ticket_id = self.rag.create_ticket("device_request", details)
        answer = (
            f"Device request created as {ticket_id}. "
            "Tech will review manager approval, business reason, budget fit, and target date before fulfillment."
        )
        self.rag.remember_interaction(f"Device request {ticket_id}", answer, category="device_request")
        return {"ticket_id": ticket_id, "answer": answer}

    def submit_device_issue(self, details: dict[str, str]) -> dict[str, Any]:
        ticket_id = self.rag.create_ticket("device_issue", details)
        priority = self._classify_priority(details)
        answer = (
            f"Device issue logged as {ticket_id} with {priority} priority. "
            "Include screenshots or exact error messages if available so support can triage faster."
        )
        self.rag.remember_interaction(f"Device issue {ticket_id}", answer, category="device_issue")
        return {"ticket_id": ticket_id, "answer": answer, "priority": priority}

    def teach_fact(self, fact: str, category: str = "agent_instruction") -> dict[str, Any]:
        memory_id = self.rag.remember_fact(fact, category=category, source="user")
        answer = f"Declarative memory saved as {memory_id}. I will use it as remembered tech department knowledge."
        self.rag.remember_interaction(f"Remember fact: {fact}", answer, category="memory_update")
        return {"memory_id": memory_id, "answer": answer}

    def _generate_answer(self, message: str, context: list[SearchResult]) -> str:
        if not self.llm.is_available():
            return self._fallback_answer(message, context)

        context_text = "\n\n".join(
            f"[{item.kind.upper()} | similarity {item.similarity:.2f}]\n{item.content}" for item in context[:6]
        )
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
            return self.llm.generate(prompt)
        except requests.RequestException as exc:
            return f"{self._fallback_answer(message, context)}\n\nLLM note: Ollama request failed: {exc}"

    def _fallback_answer(self, message: str, context: list[SearchResult]) -> str:
        if not context:
            return (
                "I do not have enough policy context for that yet. "
                "Please share the device, user, business reason, urgency, and any error details."
            )

        relevant = "\n".join(f"- {item.content.splitlines()[0]}: {self._first_sentence(item.content)}" for item in context[:4])
        return (
            "Based on the tech department policy and memory I found:\n"
            f"{relevant}\n\n"
            "Please provide any missing details such as employee name, department, manager approval, device type, "
            "business reason, asset tag, error message, urgency, and troubleshooting already attempted."
        )

    def _handle_command(self, message: str) -> dict[str, Any] | None:
        clean = message.strip()
        if not clean.startswith("/"):
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
        body = " ".join(line.strip() for line in text.splitlines()[1:] if line.strip())
        return body.split(". ")[0][:240].strip() + "."

    @staticmethod
    def _classify_priority(details: dict[str, str]) -> str:
        text = " ".join(details.values()).lower()
        critical_terms = ["lost", "stolen", "security", "cannot work", "unable to work", "production", "executive"]
        high_terms = ["vpn", "crash", "meeting", "camera", "microphone", "blocked", "urgent"]
        if any(term in text for term in critical_terms):
            return "critical"
        if any(term in text for term in high_terms):
            return "high"
        return "normal"
