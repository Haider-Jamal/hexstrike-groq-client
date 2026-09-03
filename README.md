# HexStrike Groq MCP Client

Custom Python MCP client connecting [HexStrike AI](https://github.com/0x4m4/hexstrike-ai)
(150+ pentesting tools) to Groq's Llama 3.3-70B for natural-language-directed
security tool orchestration over stdio.

## Features
- MCP stdio client -> Groq tool-calling bridge
- Keyword-relevance tool filtering to stay under Groq's 128-tool-per-request cap
- CLI dashboard: live tool-call/result logging, token usage per turn
- Multi-server support via servers.json

## Setup
1. Run HexStrike's `hexstrike_server.py` (Flask API backend)
2. `pip install mcp groq`
3. `export GROQ_API_KEY=your_key`
4. Copy `servers.json.example` -> `servers.json`, point `command` at your venv python
5. `python3 groq_mcp_client.py --config servers.json`

**Authorized testing only.**
EOFcat > README.md << 'EOF'
# HexStrike Groq MCP Client

Custom Python MCP client connecting [HexStrike AI](https://github.com/0x4m4/hexstrike-ai)
(150+ pentesting tools) to Groq's Llama 3.3-70B for natural-language-directed
security tool orchestration over stdio.

## Features
- MCP stdio client -> Groq tool-calling bridge
- Keyword-relevance tool filtering to stay under Groq's 128-tool-per-request cap
- CLI dashboard: live tool-call/result logging, token usage per turn
- Multi-server support via servers.json

## Setup
1. Run HexStrike's `hexstrike_server.py` (Flask API backend)
2. `pip install mcp groq`
3. `export GROQ_API_KEY=your_key`
4. Copy `servers.json.example` -> `servers.json`, point `command` at your venv python
5. `python3 groq_mcp_client.py --config servers.json`

**Authorized testing only.**
