from uuid import uuid4
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import OpenAI
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient
import gradio as gr
import asyncio

load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = f"AIE1 - LangGraph - {uuid4().hex[0:8]}"
os.environ["LANGCHAIN_API_KEY"] = os.getenv('LANGCHAIN_API_KEY')
os.environ['MONGODB_ATLAS_CLUSTER_URI'] = os.getenv('MONGODB_ATLAS_CLUSTER_URI')

client = MongoClient(os.environ['MONGODB_ATLAS_CLUSTER_URI'])
DB_NAME = "phenx_data"
COLLECTION_NAME = "PhenX_langchain_loader"
MONGODB_COLLECTION = client[DB_NAME][COLLECTION_NAME]

# Instantiate the Embedding Model and Vector Store
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=os.environ['OPENAI_API_KEY'])
vectorStore = MongoDBAtlasVectorSearch(collection=MONGODB_COLLECTION, embedding=embeddings, index_name='default')

# Initialize RetrievalQA with an OpenAI model for follow-up questions
llm = OpenAI(openai_api_key=os.environ['OPENAI_API_KEY'], temperature=0)
retriever = vectorStore.as_retriever()
qa_chain = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=retriever)

with gr.Blocks() as demo:
    gr.Markdown("# Interactive Questionnaire Chatbot")
    probing_question = "What is your main area of interest or concern?"
    chatbot = gr.Chatbot(label="Conversation", show_label=False, value=[(None, probing_question)])
    textbox = gr.Textbox(placeholder="Type your response here...")

    # Initialize state variables
    state = gr.State({
        'questionnaire_selected': False,
        'questionnaire_id': None,
        'questions': [],
        'current_question_index': 0
    })

    def on_message(history, user_input, state):
        if not state['questionnaire_selected']:
            # Use the user's response to find the questionnaire
            user_interest = user_input
            # Step 1: Find the most relevant questionnaire based on user input
            docs = vectorStore.similarity_search(user_interest, K=1)
            if docs:
                # Assume the questionnaire's content includes multiple questions
                questionnaire_text = docs[0].page_content
                questionnaire_id = docs[0].metadata.get("protocol_id", "unknown")

                # Step 2: Split the questionnaire into individual questions
                state['questionnaire_id'] = questionnaire_id
                state['questions'] = questionnaire_text.split("\n")  # Adjust delimiter as needed
                state['current_question_index'] = 0
                state['questionnaire_selected'] = True

                # Proceed to ask the first question
                question = state['questions'][state['current_question_index']]
                state['current_question_index'] += 1
                # Update history
                history.append((user_input, question))
                return history, state
            else:
                # No relevant questionnaire found
                response = "Sorry, I couldn't find a relevant questionnaire for your topic."
                history.append((user_input, response))
                return history, state
        else:
            # Proceed with the questionnaire
            if state['current_question_index'] < len(state['questions']):
                # Store user's answer if needed (not shown here)
                # Ask next question
                question = state['questions'][state['current_question_index']]
                state['current_question_index'] += 1
                # Update history
                history.append((user_input, question))
                return history, state
            else:
                # End of questionnaire
                response = "Thank you! We've reached the end of the questionnaire."
                history.append((user_input, response))
                return history, state

    # Update the interface to pass state
    textbox.submit(on_message, [chatbot, textbox, state], [chatbot, state])

demo.launch()
