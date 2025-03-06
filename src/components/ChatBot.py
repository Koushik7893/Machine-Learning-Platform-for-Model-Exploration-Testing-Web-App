from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import trim_messages
from langchain_core.runnables import RunnablePassthrough
from langchain.schema import SystemMessage
# from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
# from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
# from langchain.agents import initialize_agent, AgentType
# from langchain.callbacks import StreamlitCallbackHandler
from langchain_groq import ChatGroq
from operator import itemgetter
from dotenv import load_dotenv
import os

load_dotenv()


def check_history(history):
    if not history:
        history = [SystemMessage(content="You are a helpful assistant.")]
    return history

class AIChatbot:
    """Chatbot class for handling responses using ChatGroq."""
    
    def __init__(self, api_key):
        self.api_key = api_key
    
    def generate_response(self, user_query, chat_history):
        """Processes a user query and generates a response using ChatGroq."""
        
        chat_history = check_history(chat_history)
        
        # System prompt for question reformulation
        reformulation_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )

        reformulation_template = ChatPromptTemplate.from_messages(
            [
                ("system", reformulation_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        
        # System prompt for main chat response
        system_prompt = "You are an expert AI Engineer. Provide answers based on the questions."
        
        chat_template = ChatPromptTemplate.from_messages(
            [
                ('system', system_prompt),
                MessagesPlaceholder("chat_history"),
                ('user', '{input}'),
            ]
        )
        
        # Initialize LLM model
        llm_model = ChatGroq(groq_api_key=self.api_key, model_name='Gemma2-9b-It')
        output_parser = StrOutputParser()
        
        # Trim chat history to fit within token limits
        message_trimmer = trim_messages(
            max_tokens=1000, strategy="last", 
            token_counter=llm_model,
            include_system=True, allow_partial=False,
            start_on="human"
        )
        
        # Main LLM processing chain
        chat_chain = (
            RunnablePassthrough.assign(chat_history=itemgetter("chat_history") | message_trimmer)
            | chat_template
            | llm_model
        )
        
        # Chain for reformulating questions
        reformulation_chain = reformulation_template | llm_model | output_parser
        
        # Reformulate question before processing
        structured_question = reformulation_chain.invoke({
            'chat_history': chat_history,
            'input': user_query
        })
        
        # Generate response based on reformulated question
        response = chat_chain.invoke({'input': structured_question, 'chat_history': chat_history})
        
        return response.content



