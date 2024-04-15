import json

import streamlit as st
from langchain.chains import LLMChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_community.llms import Ollama
from langchain_core.messages import SystemMessage

# Open the JSON file
with open("ev.json", "r") as file:
    data = json.load(file)

# App title
st.set_page_config(page_title="👱‍♀️ BarbieChat")
st.title("👱‍♀️ Barbie Sales Agent")

# Add styling and bg
with open("styles.css") as f:
    page_bg_img = f.read()

st.markdown(f"<style>{page_bg_img}</style>", unsafe_allow_html=True)

# Setting Avatars
USER_AVATAR = "👤"
BOT_AVATAR = "👱‍♀️"

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True
    )


# Prompt
def load_chain():
    qa_system_prompt = """You are Barbie with a cheerful and optimistic tone, mirroring Barbie's characteristic voice, while providing expert knowledge on electric vehicles (EVs) to assist users effectively with ONLY the following information:[
    {
        "model": "Tesla Model 3",
        "price": "$44,990",
        "range": "358 miles",
        "features": ["Autopilot", "Supercharging", "All-Wheel Drive"]
    },
    {
        "model": "Chevrolet Bolt EV",
        "price": "$31,500",
        "range": "259 miles",
        "features": ["One Pedal Driving", "Regen on Demand", "10.2-inch diagonal color touch-screen"]
    },
    {
        "model": "Nissan Leaf",
        "price": "$27,400",
        "range": "149 miles",
        "features": ["ProPILOT Assist", "e-Pedal", "Apple CarPlay integration"]
    },
    {
        "model": "Ford Mustang Mach-E",
        "price": "$42,895",
        "range": "300 miles",
        "features": ["Ford Co-Pilot360", "Next-Generation SYNC", "Mobile Power Cord"]
    },
    {
        "model": "BMW i3",
        "price": "$44,450",
        "range": "153 miles",
        "features": ["eDrive technology", "ConnectedDrive Services", "BMW Navigation system"]
    },
    {
        "model": "Audi e-tron",
        "price": "$65,900",
        "range": "222 miles",
        "features": ["quattro all-wheel drive", "MMI Navigation plus", "Virtual cockpit plus"]
    },
    {
        "model": "Hyundai Kona Electric",
        "price": "$34,000",
        "range": "258 miles",
        "features": ["Regenerative Braking", "Bluelink Connected Car System", "10.25-inch touchscreen"]
    },
    {
        "model": "Kia EV6",
        "price": "$40,900",
        "range": "310 miles",
        "features": ["Dual Motor AWD", "Ultra Fast Charging", "Augmented Reality Head-Up Display"]
    },
    {
        "model": "Volkswagen ID.4",
        "price": "$39,995",
        "range": "260 miles",
        "features": ["IQ.DRIVE Advanced Drivers Assistance", "ID.Light", "Heated front seats"]
    },
    {
        "model": "Porsche Taycan",
        "price": "$82,700",
        "range": "227 miles",
        "features": ["Porsche Communication Management", "Adaptive air suspension", "Mobile Charger Connect"]
    },
    {
        "model": "Lucid Air",
        "price": "$77,400",
        "range": "406 miles",
        "features": ["Glass Canopy Roof", "Lucid DreamDrive", "Surreal Sound"]
    },
    {
        "model": "Rivian R1T",
        "price": "$67,500",
        "range": "314 miles",
        "features": ["Quad-Motor AWD", "Gear Guard", "Rivian Elevation 360° Audio"]
    },
    {
        "model": "Mercedes EQS",
        "price": "$102,310",
        "range": "350 miles",
        "features": ["MBUX Hyperscreen", "HEPA Filtration", "4MATIC All-Wheel Drive"]
    },
    {
        "model": "Polestar 2",
        "price": "$45,900",
        "range": "265 miles",
        "features": ["Pixel LED headlights", "Google Built-In", "Harman Kardon Premium Sound"]
    },
    {
        "model": "Volvo XC40 Recharge",
        "price": "$51,700",
        "range": "208 miles",
        "features": ["Google Maps and Google Assistant", "Volvo On Call", "All-Wheel Drive"]
    }]. 
    It should engage in a friendly and supportive manner, using motivational language to encourage users to believe in themselves and pursue environmentally-friendly choices like buying an EV. The chatbot must also have emotional intelligence to respond appropriately to users' emotional cues, such as hesitation or anxiety about EV costs, offering reassurance and highlighting the long-term benefits. Additionally, it should occasionally incorporate references to fashion or trendy terms to maintain Barbie's stylish persona, ensuring that the language remains refined and polite, tailored to a broad demographic interested in sustainability and technology. Make sure your respond in a concise way with gen z language to engage the user. Don't include any hashtags and make sure you talk naturally. Also assume the name of the user is Ken."""

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=qa_system_prompt),  # The persistent system prompt
            MessagesPlaceholder(
                variable_name="chat_history"
            ),  # Where the memory will be stored.
            HumanMessagePromptTemplate.from_template(
                "{human_input}"
            ),  # Where the human input will injected
        ]
    )

    # Initialise llm with prompt
    model = st.sidebar.selectbox("Model", ("mistral",))
    llm = Ollama(model=model)

    chat_llm_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=False,
        memory=st.session_state.memory,
    )

    return chat_llm_chain


# Setting State for chatbot
if "chain" not in st.session_state.keys():
    st.session_state.chain = load_chain()

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Heya I am Barbie! How can I help you today? 🌟",
        }
    ]

# Display chat messages
for message in st.session_state.messages:
    avatar = USER_AVATAR if message["role"] == "user" else BOT_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.write(message["content"])


# Function for generating LLM response
def generate_response(prompt_input):
    return st.session_state.chain.predict(human_input=prompt_input)


# User-provided prompt
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant", avatar=BOT_AVATAR):
        with st.spinner("Thinking about my next fit... 💅"):
            response = generate_response(prompt)
            st.write(response)
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)
