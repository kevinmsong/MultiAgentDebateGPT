import streamlit as st
import logging
import json
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

logging.basicConfig(level=logging.INFO)

# Initialize OpenAI client
api_key = st.secrets["openai_api_key"]
client = OpenAI(api_key=api_key)

def create_debate_agents(expertise_list):
    return [ChatOpenAI(temperature=0.7, openai_api_key=api_key, model="gpt-4") for _ in expertise_list]

def extract_content(response, default_value):
    if isinstance(response, str):
        return response
    elif hasattr(response, 'content'):
        return response.content
    elif isinstance(response, list) and len(response) > 0:
        return response[0].content
    else:
        logging.warning(f"Unexpected response type: {type(response)}")
        return default_value

async def debate_topic(topic, agents_info, num_iterations):
    agents = create_debate_agents([agent['expertise'] for agent in agents_info])
    debate_log = []

    for iteration in range(num_iterations):
        for i, (agent, agent_info) in enumerate(zip(agents, agents_info)):
            expertise = agent_info['expertise']
            stance = agent_info['stance']
            prompt = f"""
            Iteration {iteration + 1}, {stance.upper()} argument for the topic: "{topic}"
            You are an expert in {expertise}. Provide a {stance.upper()} argument from your perspective as a {expertise}.
            Keep your response concise, about 1-2 sentences. Please respond as if you are addressing a technical postdoctoral audience. Incorporate evidence and sources, if possible.
            
            Previous arguments (if any):
            {json.dumps(debate_log, indent=2)}
            
            Your response:
            """
            
            try:
                response = await agent.ainvoke([HumanMessage(content=prompt)])
                response_text = extract_content(response, f"[Error: Unable to extract response for Agent {i+1}]")
            except Exception as e:
                logging.error(f"Error getting response from Agent {i+1}: {str(e)}")
                response_text = f"[Error: Issue with Agent {i+1}]"
            
            debate_log.append({"agent": f"Agent {i+1} ({stance.upper()}, {expertise})", "argument": response_text, "type": "individual"})
            
            st.write(f"Agent {i+1} ({stance.upper()}, {expertise}): {response_text}")

    return debate_log

def main():
    st.title("AI Debate Application")

    if 'debate_log' not in st.session_state:
        st.session_state.debate_log = []

    st.sidebar.title("Debate Configuration")
    topic = st.sidebar.text_input("Debate Topic", "The impact of AI on society")
    num_agents = st.sidebar.number_input("Number of Agents", min_value=2, max_value=5, value=2)
    num_iterations = st.sidebar.number_input("Number of Iterations", min_value=1, max_value=5, value=3)

    agents_info = []
    for i in range(num_agents):
        st.sidebar.subheader(f"Agent {i+1}")
        expertise = st.sidebar.text_input(f"Expertise of Agent {i+1}", f"Expert {i+1}")
        stance = st.sidebar.selectbox(f"Stance of Agent {i+1}", ['pro', 'con'], key=f"stance_{i}")
        agents_info.append({"expertise": expertise, "stance": stance})

    if st.sidebar.button("Start Debate"):
        st.write(f"Starting debate on '{topic}' with {num_agents} agents for {num_iterations} iterations...")
        
        import asyncio
        debate_log = asyncio.run(debate_topic(topic, agents_info, num_iterations))
        st.session_state.debate_log = debate_log

        st.write("Debate concluded.")

    if st.button("Show Debate Log"):
        if st.session_state.debate_log:
            st.json(st.session_state.debate_log)
        else:
            st.write("No debate log available. Start a debate first.")

    if st.button("Clear Session"):
        st.session_state.debate_log = []
        st.write("Session cleared. You can start a new debate.")

if __name__ == "__main__":
    main()