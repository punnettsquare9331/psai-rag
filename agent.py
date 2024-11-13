import os
import threading
from uuid import uuid4
from datetime import datetime
from dotenv import load_dotenv
from pymongo import MongoClient
import gradio as gr

from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import OpenAI
from langchain_mongodb import MongoDBAtlasVectorSearch

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
CONVERSATION_COLLECTION = client[DB_NAME]["Conversations"]

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
    docs = vectorStore.similarity_search_with_score(user_input, K=2)
    if docs:
        protocol, score = docs[0]
        if score > 0.9:
            return None, None, None, score
        protocol_id = protocol.metadata.get("protocol_id", "unknown")
        protocol_name = protocol.metadata.get("protocol_name", "Unknown Protocol")
        protocol_text = protocol.page_content
        print('Confidence Score: ', score)
        return protocol_id, protocol_name, protocol_text, score
    else:
        return None, None, None, None

# Function to log conversations to MongoDB
def log_conversation(user_input, response, protocol_id):
    CONVERSATION_COLLECTION.insert_one({
        "protocol_id": protocol_id,
        "timestamp": datetime.utcnow(),
        "user_message": user_input,
        "psai_response": response
    })

# Function to fetch conversation history from MongoDB
def get_conversation_history():
    conversations = list(CONVERSATION_COLLECTION.find().sort("timestamp", 1))
    history_text = ""
    for conv in conversations:
        user_msg = conv.get("user_message", "")
        psai_response = conv.get("psai_response", "")
        timestamp = conv.get("timestamp", "").strftime("%Y-%m-%d %H:%M:%S")
        history_text += f"[{timestamp}]\n**User:** {user_msg}\n**PSAI:** {psai_response}\n\n"
    return history_text

# -------------------- Shared Resources -------------------- #

# Event to signal stopping the chatbot
stop_event = threading.Event()

# Lock to protect access to the current protocol information
protocol_lock = threading.Lock()
current_protocol = {
    'protocol_id': None,
    'protocol_name': None,
    'protocol_text': None,
    'score': None
}

# -------------------- User Interface -------------------- #

def create_user_interface():
    with gr.Blocks() as user_interface:
        gr.Markdown("# PS.AI")
        
        with gr.Row():
            chatbot = gr.Chatbot(
                label="Conversation", 
                show_label=False,
                value=[("Hi PSAI", "Hello, I'm PSAI. How are you feeling today? I'm here to listen and understand your current state of mind. Please share any thoughts, emotions, or concerns you're experiencing—no matter how big or small. Let's work through this together.")]  
            )
            # Live Status Indicator
            with gr.Column(scale=1, min_width=100):
                gr.Markdown(
                    """
                    <div style="display: flex; align-items: center;">
                        <span style="height: 15px; width: 15px; background-color: green; border-radius: 50%; display: inline-block; margin-right: 5px;"></span>
                        <span>Dr. Tester is live</span>
                    </div>
                    """,
                    elem_id="live_status"
                )
        
        textbox = gr.Textbox(placeholder="Type your response here...", label="Your Message")
        
        # Initialize state variables
        state = gr.State({
            'protocol_selected': False,
            'protocol_id': None,
            'protocol_name': None,
            'protocol_text': None,
            'score': None
        })
        
        # Function to handle user messages
        def on_message(history, user_input, state):
            if stop_event.is_set():
                response = "The chatbot has been stopped by a clinician. Please contact support for further assistance."
                history.append((user_input, response))
                log_conversation(user_input, response, state['protocol_id'])
                return history, state
            
            if not state['protocol_selected']:
                # First user input: match to a protocol
                protocol_id, protocol_name, protocol_text, score = find_relevant_protocol(user_input)
                with protocol_lock:
                    current_protocol['protocol_id'] = protocol_id
                    current_protocol['protocol_name'] = protocol_name
                    current_protocol['protocol_text'] = protocol_text
                    current_protocol['score'] = score
                if protocol_id:
                    # Update state with selected protocol
                    state['protocol_selected'] = True
                    state['protocol_id'] = protocol_id
                    state['protocol_name'] = protocol_name
                    state['protocol_text'] = protocol_text
                    state['score'] = score
                    
                    # Generate the first follow-up question
                    follow_up_question = generate_follow_up(protocol_text, user_input)
                    
                    # Update chat history
                    history.append((user_input, follow_up_question))
                    
                    # Log conversation
                    log_conversation(user_input, follow_up_question, protocol_id)
                    
                    return history, state
                else:
                    # No protocol found
                    response = "I'm sorry, I couldn't find a relevant protocol for your concerns. Could you please provide more details?"
                    history.append((user_input, response))
                    
                    # Log conversation
                    log_conversation(user_input, response, None)
                    
                    return history, state
            else:
                # Subsequent user inputs: generate follow-up questions based on protocol_text
                protocol_text = state['protocol_text']
                follow_up_question = generate_follow_up(protocol_text, user_input)
                history.append((user_input, follow_up_question))
                
                # Log conversation
                log_conversation(user_input, follow_up_question, state['protocol_id'])
                
                return history, state
        
        # Function to handle submission
        def handle_submit(history, user_input, state):
            history, state = on_message(history, user_input, state)
            return history, ""
        
        # Set up Gradio interface for conversation
        textbox.submit(handle_submit, [chatbot, textbox, state], [chatbot, textbox])
        
    return user_interface

# -------------------- Clinician Dashboard -------------------- #

def create_clinician_interface():
    with gr.Blocks() as clinician_interface:
        gr.Markdown("# Clinician Dashboard")
        
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("## Conversation History")
                clinician_history = gr.Markdown(value="Fetching conversation history...", label="Conversation Log")
                update_btn = gr.Button("Refresh Conversation History")
            with gr.Column(scale=1):
                gr.Markdown("## Current Protocol")
                protocol_info = gr.Markdown(value="No protocol selected.", label="Protocol Details")
                stop_btn = gr.Button("Stop Chatbot")
                #stop_btn.style(full_width=True, variant="stop")
        
        # Function to update conversation history
        def update_clinician_view():
            return get_conversation_history()
        
        # Function to update protocol information
        def update_protocol_info():
            with protocol_lock:
                if current_protocol['protocol_id']:
                    protocol_details = f"""
                    **Protocol ID:** {current_protocol['protocol_id']}  
                    **Protocol Name:** {current_protocol['protocol_name']}  
                    **Confidence Score:** {current_protocol['score']:.2f}
                    """
                else:
                    protocol_details = "No protocol selected."
            return protocol_details
        
        # Function to stop the chatbot
        def stop_chatbot():
            stop_event.set()
            return "Chatbot has been stopped.", update_protocol_info()
        
        # Set up button actions
        update_btn.click(fn=update_clinician_view, outputs=clinician_history)
        stop_btn.click(fn=stop_chatbot, outputs=[clinician_history, protocol_info])
        
        # Function to periodically update protocol info
        def auto_update_protocol():
            while not stop_event.is_set():
                protocol_info.value = update_protocol_info()
                time.sleep(5)
        
    return clinician_interface

# -------------------- Launch Both Interfaces -------------------- #

def launch_gradio_interfaces():
    user_interface = create_user_interface()
    clinician_interface = create_clinician_interface()
    
    # Launch user interface on port 7860
    user_thread = threading.Thread(target=user_interface.launch, kwargs={
        "server_name": "0.0.0.0",
        "server_port": 7860,
        "share": False
    }, daemon=True)
    
    # Launch clinician interface on port 7861
    clinician_thread = threading.Thread(target=clinician_interface.launch, kwargs={
        "server_name": "0.0.0.0",
        "server_port": 7861,
        "share": False
    }, daemon=True)
    
    user_thread.start()
    clinician_thread.start()
    
    user_thread.join()
    clinician_thread.join()

if __name__ == "__main__":
    launch_gradio_interfaces()
