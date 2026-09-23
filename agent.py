#!/usr/bin/env python3
# Updated
"""

The LLM decides what to do each turn by emitting ONE JSON action:
    {"action": "list_dir",   "path": "."}
    {"action": "read_file",  "path": "notes.txt"}
    {"action": "write_file", "path": "out.txt", "content": "..."}
    {"action": "final_answer", "text": "..."}

Your code executes the action, feeds the result back, and the loop continues
until the model emits final_answer (or hits MAX_STEPS).

Run:
    python agent.py --model /path/to/model.gguf --workspace ./workspace
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from llama_cpp import Llama, LlamaGrammar
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

console = Console()

MAX_STEPS = 8
MAX_READ_CHARS = 6000  # truncate big files so we don't blow the context window

# --------------------------------------------------------------------------- #
# Grammar: forces the model to emit valid JSON matching our action schema.
# This is what makes small local models reliable enough to drive an agent.
# --------------------------------------------------------------------------- #
ACTION_GRAMMAR = r'''
root      ::= list | read | write | final
list      ::= "{" ws "\"action\"" ws ":" ws "\"list_dir\"" ws "," ws "\"path\"" ws ":" ws string ws "}"
read      ::= "{" ws "\"action\"" ws ":" ws "\"read_file\"" ws "," ws "\"path\"" ws ":" ws string ws "}"
write     ::= "{" ws "\"action\"" ws ":" ws "\"write_file\"" ws "," ws "\"path\"" ws ":" ws string ws "," ws "\"content\"" ws ":" ws string ws "}"
final     ::= "{" ws "\"action\"" ws ":" ws "\"final_answer\"" ws "," ws "\"text\"" ws ":" ws string ws "}"
string    ::= "\"" ( [^"\\\n] | "\\" ["\\/bfnrt] )* "\""
ws        ::= [ \t\n]*
'''

SYSTEM_PROMPT = """You are a careful file-handling agent working inside a sandboxed workspace folder.
You can only act by emitting exactly ONE JSON object per turn. Available actions:

  {"action": "list_dir", "path": "<relative dir>"}
  {"action": "read_file", "path": "<relative file>"}
  {"action": "write_file", "path": "<relative file>", "content": "<full file text>"}
  {"action": "final_answer", "text": "<message to the user>"}

Rules:
- Paths are relative to the workspace. Never use absolute paths or "..".
- To answer questions about a file, read it first. Do not guess its contents.
- After you receive a tool result, decide the next action.
- When the task is complete, emit final_answer with a short summary.
- Use "\\n" inside JSON strings for newlines."""


# --------------------------------------------------------------------------- #
# Tools (the only things the LLM can make the agent do)
# --------------------------------------------------------------------------- #
class Workspace:
    """Sandboxed file access. Every path is resolved and checked to stay inside root."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _safe(self, rel: str) -> Path:
        p = (self.root / rel).resolve()
        if p != self.root and self.root not in p.parents:
            raise PermissionError(f"path escapes workspace: {rel}")
        return p

    def list_dir(self, rel: str) -> str:
        p = self._safe(rel)
        if not p.is_dir():
            return f"ERROR: not a directory: {rel}"
        entries = sorted(p.iterdir(), key=lambda e: (e.is_file(), e.name.lower()))
        if not entries:
            return "(empty directory)"
        return "\n".join(f"{'[dir] ' if e.is_dir() else '      '}{e.name}" for e in entries)

    def read_file(self, rel: str) -> str:
        p = self._safe(rel)
        if not p.is_file():
            return f"ERROR: file not found: {rel}"
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return "ERROR: file is not valid UTF-8 text"
        if len(text) > MAX_READ_CHARS:
            return text[:MAX_READ_CHARS] + f"\n...[truncated, {len(text)} chars total]"
        return text

    def write_file(self, rel: str, content: str) -> str:
        p = self._safe(rel)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"OK: wrote {len(content)} chars to {rel}"


# --------------------------------------------------------------------------- #
# UI helpers
# --------------------------------------------------------------------------- #
def banner(model_name: str, workspace: Path) -> None:
    title = Text("MiniAgent - (Trial for SIH)", style="bold magenta", justify="center")
    body = Text.assemble(
        ("model      ", "dim"), (f"{model_name}\n", "cyan"),
        ("workspace  ", "dim"), (f"{workspace}\n", "cyan"),
        ("commands   ", "dim"), ("/exit  /clear", "green"),
    )
    console.print(Panel(body, title=title, border_style="magenta", padding=(1, 2)))


def show_action(step: int, action: dict) -> None:
    kind = action.get("action", "?")
    if kind == "write_file":
        header = f"step {step} · write_file → {action.get('path')}"
        console.print(Panel(
            Syntax(action.get("content", ""), _guess_lexer(action.get("path", "")),
                   theme="monokai", word_wrap=True),
            title=header, border_style="yellow"))
    else:
        detail = action.get("path") or ""
        console.print(f"[dim]step {step}[/] [bold yellow]{kind}[/] [cyan]{detail}[/]")


def _guess_lexer(path: str) -> str:
    return {".py": "python", ".dart": "dart", ".json": "json", ".md": "markdown",
            ".js": "javascript", ".html": "html", ".yaml": "yaml", ".yml": "yaml"
            }.get(Path(path).suffix.lower(), "text")


# --------------------------------------------------------------------------- #
# Agent loop
# --------------------------------------------------------------------------- #
class Agent:
    def __init__(self, llm: Llama, ws: Workspace, confirm_writes: bool):
        self.llm = llm
        self.ws = ws
        self.grammar = LlamaGrammar.from_string(ACTION_GRAMMAR)
        self.confirm_writes = confirm_writes
        self.history: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    def reset(self) -> None:
        self.history = self.history[:1]

    def _next_action(self) -> dict:
        out = self.llm.create_chat_completion(
            messages=self.history,
            grammar=self.grammar,
            temperature=0.2,
            max_tokens=5000,
        )
        raw = out["choices"][0]["message"]["content"]
        self.history.append({"role": "assistant", "content": raw})
        return json.loads(raw)

    def _execute(self, action: dict) -> str:
        kind = action["action"]
        try:
            if kind == "list_dir":
                return self.ws.list_dir(action["path"])
            if kind == "read_file":
                return self.ws.read_file(action["path"])
            if kind == "write_file":
                if self.confirm_writes and not _ask_yes_no(f"Allow write to {action['path']}?"):
                    return "DENIED: the user refused this write."
                return self.ws.write_file(action["path"], action["content"])
        except PermissionError as e:
            return f"ERROR: {e}"
        return f"ERROR: unknown action {kind}"

    def run(self, user_input: str) -> None:
        self.history.append({"role": "user", "content": user_input})

        for step in range(1, MAX_STEPS + 1):
            with console.status("[magenta]thinking...", spinner="dots"):
                try:
                    action = self._next_action()
                except json.JSONDecodeError:
                    console.print("[red]model produced invalid JSON, stopping.[/]")
                    return

            if action["action"] == "final_answer":
                console.print(Panel(action["text"], title="agent",
                                    border_style="green", padding=(1, 2)))
                return

            show_action(step, action)
            result = self._execute(action)
            console.print(f"[dim]  ↳ {result.splitlines()[0][:100] if result else ''}[/]")
            self.history.append({"role": "user", "content": f"TOOL RESULT:\n{result}"})

        console.print("[red]reached max steps without a final answer.[/]")


def _ask_yes_no(question: str) -> bool:
    ans = console.input(f"[bold red]? {question}[/] [dim](y/N)[/] ").strip().lower()
    return ans in ("y", "yes")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to .gguf file")
    ap.add_argument("--workspace", default="./workspace")
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--gpu-layers", type=int, default=-1, help="-1 = offload all")
    ap.add_argument("--no-confirm", action="store_true", help="skip write confirmation")
    args = ap.parse_args()

    model_path = Path(args.model)
    if not model_path.is_file():
        console.print(f"[red]model not found:[/] {model_path}")
        sys.exit(1)

    with console.status("[magenta]loading model...", spinner="dots"):
        llm = Llama(model_path=str(model_path), n_ctx=args.ctx,
                    n_gpu_layers=args.gpu_layers, verbose=False)

    ws = Workspace(Path(args.workspace))
    agent = Agent(llm, ws, confirm_writes=not args.no_confirm)

    banner(model_path.name, ws.root)

    session = PromptSession(history=InMemoryHistory())
    style = Style.from_dict({"prompt": "bold ansimagenta"})

    while True:
        try:
            user = session.prompt([("class:prompt", "❯ ")], style=style).strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user:
            continue
        if user == "/exit":
            break
        if user == "/clear":
            agent.reset()
            console.print("[dim]conversation cleared[/]")
            continue
        agent.run(user)

    console.print("[dim]bye.[/]")


if __name__ == "__main__":
    main()
