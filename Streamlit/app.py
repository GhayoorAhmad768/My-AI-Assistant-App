import streamlit as st
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import os

load_dotenv()  # reads GOOGLE_API_KEY from .env

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)
st.title("Beginner AI Assistent")
st.write("Enter a prompt and get a response from GPT!")


openai_api_key = st.text_input("Enter your OpenAI API Key", type="password")

# Prompt input from user
user_prompt = st.text_area("Your prompt", placeholder="Ask me anything...")

# Set up LLM and chain when API key is available
if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key

    # Initialize the LLM
    llm = ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo")

    # Create a basic prompt template
    prompt_template = PromptTemplate(
        input_variables=["question"],
        template="You are a helpful AI assistant. Answer the question clearly:\n\nQuestion: {question}\n\nAnswer:"
    )
    
    
    chain = LLMChain(llm=llm, prompt=prompt_template)

    # When user submits a prompt
    if st.button("Get Response"):
        if user_prompt.strip() == "":
            st.warning("Please enter a prompt.")
        else:
            with st.spinner("Thinking..."):
                response = chain.run(user_prompt)
                st.success("Here's the response:")
                st.write(response)
else:
    st.info("Please enter your OpenAI API key to start.")
