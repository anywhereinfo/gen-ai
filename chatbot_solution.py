import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv, find_dotenv

print(load_dotenv(find_dotenv()))

 
# Configure the language model
llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.7)

# Use StreamlitChatMessageHistory to store the chat history in session state
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = StreamlitChatMessageHistory(key="chat_history_key")

# Initialize the ConversationBufferMemory with StreamlitChatMessageHistory
memory = ConversationBufferMemory(chat_memory=st.session_state["chat_history"])

# Create the conversation chain
conversation_chain = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

# Streamlit UI setup
st.title("Conversational Chatbot with LangChain and Streamlit")

# Function to clear user input
def clear_text():
    st.session_state["user_input"] = ""

# User input handling
user_input = st.text_input("You:", key="user_input", placeholder="Type your message...")

# When the user submits their input
if st.button("Send") and user_input:
    # Run the conversation chain
    response = conversation_chain.run(input=user_input)
    
    # Add the conversation history manually
    st.session_state["chat_history"].add_user_message(user_input)
    st.session_state["chat_history"].add_ai_message(response)
    
    # Clear the input field
    st.session_state["user_input"] = ""
    # Rerun Streamlit to refresh the displayed chat
    st.experimental_rerun()

# Display chat history
for chat in st.session_state["chat_history"].messages:
    if chat["type"] == "user":
        st.write(f"You: {chat['content']}")
    else:
        st.write(f"Bot: {chat['content']}")