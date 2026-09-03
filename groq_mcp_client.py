#!/usr/bin/env python3
"""
groq_mcp_client.py
Groq-backed agent that talks to one or more MCP servers over stdio.

Usage:
    python groq_mcp_client.py --config servers.json
    python groq_mcp_client.py --server "python3 my_mcp_server.py"

Env:
    GROQ_API_KEY   required
    GROQ_MODEL     optional, default below

servers.json format:
{
  "servers": [
    {"name": "hexstrike", "command": "/home/haider/hexstrike-ai/venv/bin/python3", "args": ["hexstrike_mcp.py"]},
    {"name": "fs",        "command": "npx",      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/home/haider"]}
  ]
}

Note: hexstrike_server.py (Flask API, port 8888) must already be running in its
own terminal before you start this client. hexstrike_mcp.py is the stdio MCP
wrapper that talks to that Flask API internally -- that's the file this client
spawns, not hexstrike_server.py itself.
"""

import argparse
import asyncio
import json
import os
import sys
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any

from groq import Groq
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# ---- config -----------------------------------------------------------

DEFAULT_MODEL = "llama-3.3-70b-versatile"   # verify against console.groq.com/docs/models before relying on it
MAX_HISTORY_MESSAGES = 20                    # trim to keep token usage down
MAX_TOOL_OUTPUT_CHARS = 4000                 # truncate huge tool results before they hit the model

SYSTEM_PROMPT = """You are Haider's technical assistant, operating with MCP tool access.

Rules:
- Be direct. No filler, no repeating the request back, no disclaimers unless there's a real safety/legal issue.
- When a tool can answer the question, call it. Don't guess at data you can fetch.
- After tool results come back, give the conclusion first, details after.
- If a task is ambiguous, pick the most reasonable interpretation and say so in one line rather than asking.
- Keep responses tight. Bullets over paragraphs when listing things.
"""

# ---- data classes -------------------------------------------------------

@dataclass
class ServerSpec:
    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] | None = None


@dataclass
class ConnectedServer:
    spec: ServerSpec
    session: ClientSession
    tools: list[Any]


# ---- dashboard / output helpers -----------------------------------------

class Dash:
    """Minimal, dependency-free CLI dashboard."""

    RESET = "\033[0m"
    DIM = "\033[2m"
    BOLD = "\033[1m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    MAGENTA = "\033[35m"

    @staticmethod
    def header(text: str) -> None:
        line = "─" * max(len(text) + 2, 40)
        print(f"\n{Dash.CYAN}{line}\n {Dash.BOLD}{text}{Dash.RESET}{Dash.CYAN}\n{line}{Dash.RESET}")

    @staticmethod
    def status(servers: list[ConnectedServer]) -> None:
        print(f"{Dash.DIM}Connected servers:{Dash.RESET}")
        for s in servers:
            names = ", ".join(t.name for t in s.tools) or "(no tools)"
            print(f"  {Dash.GREEN}●{Dash.RESET} {s.spec.name:<12} {Dash.DIM}{names}{Dash.RESET}")
        print()

    @staticmethod
    def tool_call(name: str, args: dict) -> None:
        arg_str = json.dumps(args, ensure_ascii=False)
        if len(arg_str) > 120:
            arg_str = arg_str[:117] + "..."
        print(f"{Dash.YELLOW}⚙ tool call{Dash.RESET}  {Dash.BOLD}{name}{Dash.RESET}({arg_str})")

    @staticmethod
    def tool_result(ok: bool, preview: str) -> None:
        icon = f"{Dash.GREEN}✓" if ok else f"{Dash.RED}✗"
        preview = preview.replace("\n", " ")
        if len(preview) > 160:
            preview = preview[:157] + "..."
        print(f"{icon}{Dash.RESET} {Dash.DIM}{preview}{Dash.RESET}")

    @staticmethod
    def usage(prompt_tokens: int, completion_tokens: int, total: int) -> None:
        print(f"{Dash.DIM}tokens: {prompt_tokens} in / {completion_tokens} out / {total} total{Dash.RESET}")

    @staticmethod
    def assistant(text: str) -> None:
        print(f"\n{Dash.MAGENTA}{Dash.BOLD}assistant{Dash.RESET} {text}\n")

    @staticmethod
    def error(text: str) -> None:
        print(f"{Dash.RED}✗ {text}{Dash.RESET}")


# ---- MCP <-> Groq tool schema bridge -------------------------------------

def mcp_tools_to_groq_schema(tools: list[Any], server_name: str) -> list[dict]:
    """Convert MCP tool defs to OpenAI/Groq-style function schema.
    Tool names are namespaced as server__toolname to avoid collisions
    when multiple servers expose the same tool name.
    """
    schema = []
    for t in tools:
        schema.append({
            "type": "function",
            "function": {
                "name": f"{server_name}__{t.name}",
                "description": (t.description or "")[:1000],
                "parameters": t.inputSchema or {"type": "object", "properties": {}},
            },
        })
    return schema


def truncate(text: str, limit: int = MAX_TOOL_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated, {len(text) - limit} more chars]"


MAX_TOOLS_PER_REQUEST = 128  # hard cap enforced by Groq's API


def select_relevant_tools(query: str, full_schema: list[dict], limit: int = MAX_TOOLS_PER_REQUEST) -> list[dict]:
    """HexStrike exposes 150+ tools but Groq rejects requests with >128 tool
    defs. Score each tool by keyword overlap between the user's query and its
    name/description, keep the top N. Cheap and good enough -- it's picking
    a shortlist, not doing semantic search.
    """
    if len(full_schema) <= limit:
        return full_schema

    query_words = set(query.lower().replace("-", " ").replace("_", " ").split())

    def score(entry: dict) -> int:
        fn = entry["function"]
        text = (fn["name"] + " " + fn.get("description", "")).lower().replace("-", " ").replace("_", " ")
        tool_words = set(text.split())
        return len(query_words & tool_words)

    ranked = sorted(full_schema, key=score, reverse=True)
    return ranked[:limit]


# ---- core agent -----------------------------------------------------------

class GroqMCPAgent:
    def __init__(self, model: str = DEFAULT_MODEL):
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set in environment")
        self.client = Groq(api_key=api_key)
        self.model = model
        self.servers: list[ConnectedServer] = []
        self.tool_schema: list[dict] = []
        self.tool_index: dict[str, ConnectedServer] = {}  # namespaced name -> server
        self.messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        self._stack = AsyncExitStack()

    async def connect(self, specs: list[ServerSpec]) -> None:
        for spec in specs:
            # FIX: default to a full copy of the current environment rather than
            # letting the MCP SDK fall back to its filtered whitelist (PATH, HOME, etc).
            # Without this, GROQ_API_KEY / venv-related vars set in ~/.zshrc won't
            # reach the spawned server process.
            env = spec.env if spec.env is not None else os.environ.copy()
            params = StdioServerParameters(command=spec.command, args=spec.args, env=env)
            read, write = await self._stack.enter_async_context(stdio_client(params))
            session = await self._stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            tools_result = await session.list_tools()
            connected = ConnectedServer(spec=spec, session=session, tools=tools_result.tools)
            self.servers.append(connected)

            for schema_entry in mcp_tools_to_groq_schema(connected.tools, spec.name):
                self.tool_schema.append(schema_entry)
                self.tool_index[schema_entry["function"]["name"]] = connected

    async def close(self) -> None:
        await self._stack.aclose()

    def _trim_history(self) -> None:
        # keep system prompt + last N messages to control token usage
        if len(self.messages) > MAX_HISTORY_MESSAGES + 1:
            self.messages = [self.messages[0]] + self.messages[-MAX_HISTORY_MESSAGES:]

    async def _call_tool(self, namespaced_name: str, args: dict) -> str:
        server = self.tool_index.get(namespaced_name)
        if server is None:
            return f"error: unknown tool {namespaced_name}"
        real_name = namespaced_name.split("__", 1)[1]
        Dash.tool_call(real_name, args)
        try:
            result = await server.session.call_tool(real_name, args)
            parts = []
            for block in result.content:
                if hasattr(block, "text"):
                    parts.append(block.text)
                else:
                    parts.append(str(block))
            text = "\n".join(parts)
            is_error = getattr(result, "isError", False)
            Dash.tool_result(not is_error, text)
            return truncate(text)
        except Exception as e:  # noqa: BLE001
            Dash.tool_result(False, str(e))
            return f"error calling {real_name}: {e}"

    async def ask(self, user_input: str) -> str:
        self.messages.append({"role": "user", "content": user_input})
        self._trim_history()

        final_text = ""
        # tool-use loop: keep going until the model stops requesting tools
        for _ in range(8):  # hard cap to prevent runaway loops
            active_tools = select_relevant_tools(user_input, self.tool_schema) if self.tool_schema else []
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=active_tools or None,
                tool_choice="auto" if active_tools else None,
                temperature=0.3,
                max_tokens=1500,
            )
            choice = resp.choices[0]
            usage = resp.usage
            if usage:
                Dash.usage(usage.prompt_tokens, usage.completion_tokens, usage.total_tokens)

            msg = choice.message
            tool_calls = getattr(msg, "tool_calls", None)

            if not tool_calls:
                final_text = msg.content or ""
                self.messages.append({"role": "assistant", "content": final_text})
                break

            # record the assistant's tool-call turn
            self.messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in tool_calls
                ],
            })

            for tc in tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                result_text = await self._call_tool(tc.function.name, args)
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_text,
                })

            self._trim_history()

        return final_text


# ---- config loading --------------------------------------------------------

def load_specs(args: argparse.Namespace) -> list[ServerSpec]:
    if args.config:
        with open(args.config) as f:
            data = json.load(f)
        specs = []
        for s in data["servers"]:
            specs.append(ServerSpec(
                name=s["name"], command=s["command"],
                args=s.get("args", []), env=s.get("env"),
            ))
        return specs
    if args.server:
        parts = args.server.split()
        return [ServerSpec(name="server1", command=parts[0], args=parts[1:])]
    raise SystemExit("Provide --config servers.json or --server \"cmd args...\"")


# ---- main loop --------------------------------------------------------------

async def main() -> None:
    parser = argparse.ArgumentParser(description="Groq-backed MCP client")
    parser.add_argument("--config", help="path to servers.json")
    parser.add_argument("--server", help="single server as 'command arg1 arg2'")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    specs = load_specs(args)
    agent = GroqMCPAgent(model=args.model)

    Dash.header(f"groq-mcp-client  |  model={args.model}")
    try:
        await agent.connect(specs)
    except Exception as e:  # noqa: BLE001
        Dash.error(f"failed to connect to MCP server(s): {e}")
        return

    Dash.status(agent.servers)
    print("Type your request. Ctrl+C or 'exit' to quit.\n")

    try:
        while True:
            try:
                user_input = input(f"{Dash.BOLD}you{Dash.RESET} > ").strip()
            except EOFError:
                break
            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit"}:
                break
            try:
                reply = await agent.ask(user_input)
            except Exception as e:  # noqa: BLE001
                Dash.error(str(e))
                continue
            Dash.assistant(reply)
    except KeyboardInterrupt:
        pass
    finally:
        await agent.close()
        print(f"{Dash.DIM}disconnected.{Dash.RESET}")


if __name__ == "__main__":
    asyncio.run(main())
