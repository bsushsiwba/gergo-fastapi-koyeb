from context import *
import os
import uuid
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from langchain.chains import LLMChain
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from langdetect import detect

# Initialize FastAPI app
app = FastAPI()

# Chat contexts
chat_contexts = {}
MAX_CONTEXTS = 5

def detect_language(text):
    """Detect the language of the input text."""
    try:
        return detect(text)
    except Exception:
        return "en"  # Default to English if detection fails

def find_answer_in_knowledge_base(question, language):
    """Search for an exact match in the knowledge base."""
    if question in knowledge_base:
        return knowledge_base[question]
    return None

class ChatRequest(BaseModel):
    question: str
    id: str = None

class ChatResponse(BaseModel):
    id: str
    response: str

# Get Groq API key and model
groq_api_key = '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'
model = 'llama3-8b-8192'
groq_chat = ChatGroq(
    groq_api_key=groq_api_key, 
    model_name=model
)

system_prompt = 'You are a friendly conversational chatbot who responds in the language of the user.'

@app.get("/")
async def welcome():
    return "The site is running correctly, use chat endpoint."

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    global chat_contexts

    # Detect language
    language = detect_language(request.question)

    # Check knowledge base for predefined answer
    predefined_answer = find_answer_in_knowledge_base(request.question, language)
    if predefined_answer:
        response = predefined_answer
        # Ensure chat_id is set if predefined answer is found
        chat_id = request.id if request.id else str(uuid.uuid4())
    else:
        # Retrieve or create chat context
        chat_id = request.id
        if not chat_id or chat_id not in chat_contexts:
            chat_id = str(uuid.uuid4())
            memory = ConversationBufferWindowMemory(k=5, memory_key="chat_history", return_messages=True)
            chat_contexts[chat_id] = memory
        else:
            memory = chat_contexts[chat_id]

        # Construct prompt and conversation chain
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("{human_input}"),
            ]
        )

        conversation = LLMChain(
            llm=groq_chat,
            prompt=prompt,
            verbose=False,
            memory=memory,
        )

        response = conversation.predict(human_input=request.question)

    # Maintain only the 5 most recent contexts
    if len(chat_contexts) > MAX_CONTEXTS:
        oldest_context_id = list(chat_contexts.keys())[0]
        del chat_contexts[oldest_context_id]

    return ChatResponse(id=chat_id, response=response)
