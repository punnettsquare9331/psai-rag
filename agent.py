from uuid import uuid4
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import OpenAI
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient
import gradio as gr

# Load environment variables
load_dotenv()

# Set up environment variables
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = f"AIE1 - LangGraph - {uuid4().hex[0:8]}"
os.environ["LANGCHAIN_API_KEY"] = os.getenv('LANGCHAIN_API_KEY')
os.environ['MONGODB_ATLAS_CLUSTER_URI'] = os.getenv('MONGODB_ATLAS_CLUSTER_URI')

# MongoDB setup
client = MongoClient(os.environ['MONGODB_ATLAS_CLUSTER_URI'])
DB_NAME = "phenx_data"
COLLECTION_NAME = "PhenX_langchain_loader_with_embeddings"  # Ensure this matches your data cleaning code
MONGODB_COLLECTION = client[DB_NAME][COLLECTION_NAME]

# Instantiate the Embedding Model and Vector Store
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=os.getenv('OPENAI_API_KEY'))
vectorStore = MongoDBAtlasVectorSearch(
    collection=MONGODB_COLLECTION,
    embedding=embeddings,  # Pass the OpenAIEmbeddings instance
    index_name='default',
    text_key='protocol_text'
)
# Initialize RetrievalQA with an OpenAI model for generating questions
llm = OpenAI(openai_api_key=os.getenv('OPENAI_API_KEY'), temperature=0.7)
retriever = vectorStore.as_retriever()
qa_chain = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=retriever)

# Function to generate follow-up questions using RAG
def generate_follow_up(protocol_text, user_input):
    prompt = f"""You are a compassionate psychiatrist conducting a mental health interview. Based on the following protocol text and the patient's response, generate a relevant follow-up question.

Protocol Text:
{protocol_text}

Patient Response:
{user_input}

Follow-up Question:"""
    response = llm(prompt)
    return response.strip()

# Function to find the most relevant questionnaire based on user input
def find_relevant_protocol(user_input):
    # Use raw user input for similarity search; vectorStore will embed it automatically
    docs = vectorStore.similarity_search(user_input, K=1)
    if docs:
        protocol = docs[0]
        protocol_id = protocol.metadata.get("protocol_id", "unknown")
        protocol_name = protocol.metadata.get("protocol_name", "Unknown Protocol")
        protocol_text = protocol.page_content
        return protocol_id, protocol_name, protocol_text
    else:
        return None, None, None

# Gradio Chat Interface
with gr.Blocks() as demo:
    gr.Markdown("# Psychiatrist Chatbot for Mental Health Concerns")
    chatbot = gr.Chatbot(label="Conversation", show_label=False)
    textbox = gr.Textbox(placeholder="Type your response here...")

    # Initialize state variables
    state = gr.State({
        'protocol_selected': False,
        'protocol_id': None,
        'protocol_name': None,
        'protocol_text': None
    })
    
    # Initial probing question
    initial_question = "How are you? What are your mental health concerns today? Please tell me how you are feeling."
    
    # Function to handle user messages
    def on_message(history, user_input, state):
        if len(history) == 0:
            # Display the initial question if this is the first interaction
            history.append(("Psychiatrist", initial_question))
            return history, state
          
        if not state['protocol_selected']:
            # First user input: match to a protocol
            protocol_id, protocol_name, protocol_text = find_relevant_protocol(user_input)
            if protocol_id:
                # Update state with selected protocol
                state['protocol_selected'] = True
                state['protocol_id'] = protocol_id
                state['protocol_name'] = protocol_name
                state['protocol_text'] = protocol_text
                
                # Generate the first follow-up question
                follow_up_question = generate_follow_up(protocol_text, user_input)
                
                # Update chat history
                history.append((user_input, follow_up_question))
                return history, state
            else:
                # No protocol found
                response = "I'm sorry, I couldn't find a relevant protocol for your concerns. Could you please provide more details?"
                history.append((user_input, response))
                return history, state
        else:
            # Subsequent user inputs: generate follow-up questions based on protocol_text
            protocol_text = state['protocol_text']
            follow_up_question = generate_follow_up(protocol_text, user_input)
            history.append((user_input, follow_up_question))
            return history, state

    # Function to handle submission
    def handle_submit(history, user_input, state):
        history, state = on_message(history, user_input, state)
        return history, ""

    # Set up Gradio interface
    textbox.submit(handle_submit, [chatbot, textbox, state], [chatbot, textbox])

demo.launch()
