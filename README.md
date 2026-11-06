# MultiAgentDebateGPT

Streamlit multi-agent debate demo: configure 2–5 GPT-4 agents with expertise and pro/con stances, run several iterations on a topic, and inspect the debate log as JSON.

Live app: [multiagentdebategpt.streamlit.app](https://multiagentdebategpt.streamlit.app)

## Features

- Sidebar: debate topic, number of agents (2–5), iterations (1–5)
- Per agent: expertise label and stance (`pro` / `con`)
- Agents reply asynchronously via LangChain `ChatOpenAI` (`gpt-4`), ~1–2 sentences aimed at a technical postdoctoral audience
- Live stream of arguments; **Show Debate Log** dumps session JSON; **Clear Session** resets

## Requirements

See `requirements.txt` (streamlit, openai, langchain-openai, asyncio, langchain).

## Setup

```bash
pip install -r requirements.txt
```

Configure Streamlit secrets:

```toml
openai_api_key = "YOUR_OPENAI_API_KEY"
```

## Usage

```bash
streamlit run DebateGPT.py
```

Set the topic and agents in the sidebar, then click **Start Debate**.
