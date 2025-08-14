from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import trim_messages
from langchain_core.runnables import RunnablePassthrough
from langchain.schema import SystemMessage
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.callbacks import StreamlitCallbackHandler
import streamlit as st
# from langchain_community.retrievers import PineconeHybridSearchRetriever
# from pinecone import Pinecone
# from pinecone_text.sparse import BM25Encoder
from langchain_groq import ChatGroq
from operator import itemgetter
from dotenv import load_dotenv
import os


load_dotenv()

# index_name=os.getenv("PINECONE_INDEX_NAME")
# os.environ["HF_TOKEN"]=os.getenv("HF_TOKEN")



arxiv_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv=ArxivQueryRun(api_wrapper=arxiv_wrapper)

api_wrapper=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=200)
wiki=WikipediaQueryRun(api_wrapper=api_wrapper)

tools=[arxiv,wiki]
# st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)



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


output_parser = StrOutputParser()

        

def check_history(history):
    if not history:
        history = [SystemMessage(content="You are a helpful assistant.")]
    return history



# def pinecone_retrieval(question):
#     embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#     pc=Pinecone(api_key=os.getenv("PINECONE_TOKEN"))
#     index=pc.Index(index_name)
    
#     # bm25_encoder=BM25Encoder().load()
#     # retriever=PineconeHybridSearchRetriever(embeddings=embeddings,index=index) ## In case hybrid search sparse_encoder=bm25_encoder
    
#     query_vector = embeddings.embed_query(question)
#     results = index.query(vector=query_vector, top_k=5, include_metadata=True)

#     # Extracting relevant metadata (e.g., text content)
#     retrieved_texts = [match["metadata"]["text"] for match in results["matches"] if "metadata" in match]
    
#     return retrieved_texts 



def agent_retrieval(question, llm):
    search_agent=initialize_agent(tools,llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)
    try:
        response=search_agent.run(question)
    except Exception as e:
        response = f"Error in external search: {str(e)}"

    return response



class AIChatbot:
    """Chatbot class for handling responses using ChatGroq."""
    
    def __init__(self, api_key):
        self.api_key = api_key
    
    def generate_response(self, user_query, chat_history, search=False):
        """Processes a user query and generates a response using ChatGroq."""

        # Initialize LLM model
        llm_model = ChatGroq(groq_api_key=self.api_key, model_name='Gemma2-9b-It')
       
        # Gets Chat History
        chat_history = check_history(chat_history)
        

        # Chain for reformulating questions
        reformulation_chain = reformulation_template | llm_model | output_parser
              
              
        # Reformulate question before processing
        structured_question = reformulation_chain.invoke({
            'chat_history': chat_history,
            'input': user_query
        })
              
        

        # Trim chat history to fit within token limits
        message_trimmer = trim_messages(
            max_tokens=2000, strategy="last", 
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
        

        # Generate response based on reformulated question
        if search:
            agent_response = agent_retrieval(structured_question, llm_model)
            response = chat_chain.invoke({'input': f"Context: {agent_response}\nQuestion: {structured_question}", 'chat_history': chat_history})
            
        else:
            # pinecone_results = pinecone_retrieval(structured_question)
            pinecone_results = False
            if pinecone_results:
                context = "\n".join(pinecone_results)
                response = chat_chain.invoke({'input': f"Context: {context}\nQuestion: {structured_question}", 'chat_history': chat_history})
            else:
                response = chat_chain.invoke({'input': structured_question, 'chat_history': chat_history})
       
       
        return response.content



