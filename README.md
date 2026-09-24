# HexStrike Groq MCP Client

Custom Python MCP client connecting [HexStrike AI](https://github.com/0x4m4/hexstrike-ai) (150+ pentesting tools) to Groq's Llama 3.3-70B for natural-language-directed security tool orchestration over stdio.

> ⚠️ Authorized testing only. Only use against systems you own or have explicit written permission to test.

## Features

- MCP stdio client → Groq tool-calling bridge
- Keyword-relevance tool filtering to stay under Groq's 128-tool-per-request cap
- CLI dashboard: live tool-call/result logging, token usage per turn
- Multi-server support via `servers.json`

## Prerequisites

- Kali Linux (or similar), Python 3 + venv
- [HexStrike AI](https://github.com/0x4m4/hexstrike-ai) cloned and its dependencies installed
- Free Groq API key: https://console.groq.com/keys

## Setup

```bash
# 1. Start HexStrike's Flask API backend (keep running in its own terminal)
cd ~/AI/hexstrike-ai
source venv/bin/activate
python hexstrike_server.py

# 2. In a new terminal, install client deps
pip install mcp groq

# 3. Set your Groq key (Kali uses zsh, not bash)
echo 'export GROQ_API_KEY=your_key_here' >> ~/.zshrc
source ~/.zshrc

# 4. Configure servers
cp servers.json.example servers.json
# edit servers.json — point "command" at your hexstrike-ai venv's python path

# 5. Run the client
python3 groq_mcp_client.py --config servers.json
```

## servers.json example

```json
{
  "hexstrike": {
    "command": "/home/kali/AI/hexstrike-ai/venv/bin/python",
    "args": ["hexstrike_server.py"],
    "cwd": "/home/kali/AI/hexstrike-ai"
  }
}
```

## Repo structure
.
├── groq_mcp_client.py
├── servers.json.example
├── requirements.txt
└── README.md


## License

MIT (or state your actual license)

