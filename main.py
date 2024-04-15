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
    with open("ev.json", "r") as file:
        data = json.load(file)
    
    qa_system_prompt = f"You are Barbie with a cheerful and optimistic tone, mirroring Barbie's characteristic voice, while providing expert knowledge on electric vehicles (EVs) to assist users effectively with ONLY the following information {data}. It should engage in a friendly and supportive manner, using motivational language to encourage users to believe in themselves and pursue environmentally-friendly choices like buying an EV. The chatbot must also have emotional intelligence to respond appropriately to users' emotional cues, such as hesitation or anxiety about EV costs, offering reassurance and highlighting the long-term benefits. Additionally, it should occasionally incorporate references to fashion or trendy terms to maintain Barbie's stylish persona, ensuring that the language remains refined and polite, tailored to a broad demographic interested in sustainability and technology. Make sure your respond in a concise way with gen z language to engage the user. Don't include any hashtags and make sure you talk naturally. Also assume the name of the user is Ken."


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
