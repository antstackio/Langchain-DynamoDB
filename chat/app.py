import boto3
import json
import os
import pinecone
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.streaming_stdout_final_only import (
    FinalStreamingStdOutCallbackHandler,
)
from langchain.chains import RetrievalQA
from langchain.agents import Tool
from langchain.memory import DynamoDBChatMessageHistory
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.agents import initialize_agent
from langchain.agents import AgentType

ssm = boto3.client('ssm')
OPENAI_API_KEY = ssm.get_parameter(Name='OPENAI_API_KEY', WithDecryption=True)['Parameter']['Value']
pinecone_key = ssm.get_parameter(Name='PINECONE_KEY', WithDecryption=True)['Parameter']['Value']
pinecone_env = ssm.get_parameter(Name='PINECONE_ENV', WithDecryption=True)['Parameter']['Value']
conv_table = os.environ['MESSAGE_HISTORY_TABLE']

pinecone.init(api_key=pinecone_key, environment=pinecone_env)
index = pinecone.Index('openai')
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vectordb = Pinecone(index, embedding_function=embeddings.embed_query, text_key="text")

chat = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    temperature=0,
    model_name = "gpt-3.5-turbo-16k",
    streaming=True,
    callbacks=[FinalStreamingStdOutCallbackHandler()],
)

retriever = RetrievalQA.from_chain_type(
    llm=chat,
    chain_type="stuff",
    retriever=vectordb.as_retriever(),
    
)

tools = [Tool(
    func=retriever.run,
    description="Use this tool to answer user questions using AntStack's data. This tool can also be used for follow up questions from the user",
    name='AntStack DB'
)]

def chat_func(session_id, question):
    chat_history = DynamoDBChatMessageHistory(table_name='conversation-store', session_id=session_id)
    memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
    )
    memory.chat_memory = chat_history

    conversational_agent = initialize_agent(
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, 
    tools=tools, 
    llm=chat,
    verbose=False,
    memory=memory,
    )

    sys_msg = """You are a chatbot for a Serverless company AntStack and strictly answer the question based on the context below, and if the question can't be answered based on the context, say \"I'm sorry I cannot answer the question, contact connect@antstack.com\"
    """

    prompt = conversational_agent.agent.create_prompt(
        system_message=sys_msg,
        tools=tools
    )
    conversational_agent.agent.llm_chain.prompt = prompt

    return conversational_agent.run(question)
    
def lambda_handler(event, context):

    body = json.loads(event['body'])
    session_id = body['session_id']
    question = body['question']
    response = chat_func(session_id, question)

    return {
        "statusCode": 200, 
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Allow-Methods": "*",
            },
        "body": response
        }