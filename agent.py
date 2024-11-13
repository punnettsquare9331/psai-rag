
import os
import threading
from uuid import uuid4
from datetime import datetime
from dotenv import load_dotenv
from pymongo import MongoClient
import gradio as gr
import time
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
COLLECTION_NAME = "PhenX_langchain_loader_with_embeddings_v2"  # Ensure this matches your data cleaning code
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
llm = OpenAI(openai_api_key=os.getenv('OPENAI_API_KEY'), temperature=0.0)
retriever = vectorStore.as_retriever(search_kwargs={"k": 1}) # Limit it to 1 document
qa_chain = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=retriever)


def generate_follow_up_with_rag(protocol_text, user_input, chat_history):
    template = """Use only the following context to ask a follow-up question. If you do not know what to ask, say so and prompt the user to contact support. You should talk like you are a compassinate psychiatrist conducting a mental health interview. Based on the following protocol text and the patient's response, generate a relevant follow-up question. You will go through the protocol with the user question by question.

    Protocol Context: {protocol_text}

    Patient Response: {user_input}

    Chat Context: {chat_context}

    Follow-up Question From the Protocol In a Human Readable Format:"""
    return qa_chain.run(template).strip()

def generate_follow_up(protocol_text, user_input, chat_history):
    # Convert chat history into a formatted string
    history_text = ""
    for user_msg, ai_msg in chat_history:
        history_text += f"Patient: {user_msg}\nPsychiatrist: {ai_msg}\n"

    prompt = f"""You are a compassionate psychiatrist conducting a mental health interview. 
        Use the protocol text to guide your questions, and consider the previous conversation history.

        Protocol Text:
        {protocol_text}

        Previous Conversation:
        {history_text}

        Patient's Latest Response:
        {user_input}

        You Should Walk through the protocol with the user question by question.
        Based on the protocol and conversation history, first provide a compassionate response to the patient's latest response then provide your next question:"""
    response = llm(prompt)
    return response.strip()

def generate_score(protocol_text, user_input, chat_history):
    # Convert chat history into a formatted string
    history_text = ""
    for user_msg, ai_msg in chat_history:
        history_text += f"Patient: {user_msg}\nPsychiatrist: {ai_msg}\n"

    prompt = f"""You are a compassionate psychiatrist conducting a mental health interview. 
        Use the protocol text to guide your scoring, and consider the previous conversation history.

        Protocol Text:
        {protocol_text}

        Previous Conversation:
        {history_text}

        Patient's Latest Response:
        {user_input}

        You should attempt to score the patient's latest response based on the protocol and conversation history:"""
    response = llm(prompt)
    return response.strip()


# Function to find the most relevant questionnaire based on user input
def find_relevant_protocol(user_input):
    # Use raw user input for similarity search; vectorStore will embed it automatically
    docs = vectorStore.similarity_search_with_score(user_input, K=2)
    if docs:
        protocol, score = docs[0]
        if score > 0.95:
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

def clear_conversation_history():
    CONVERSATION_COLLECTION.delete_many({})

# Function to fetch conversation history from MongoDB
def get_conversation_history():
    # Fetch all conversation entries sorted by timestamp in ascending order
    conversations = list(CONVERSATION_COLLECTION.find().sort("timestamp", 1))
    history_text = ""
    
    for conv in conversations:
        # Retrieve user message and PSAI response
        user_msg = conv.get("user_message", "")
        psai_response = conv.get("psai_response", "")
        
        # Retrieve protocol_id; default to None if not present
        protocol_id = conv.get("protocol_id")
        
        # Format timestamp; default to empty string if not present
        timestamp = conv.get("timestamp")
        if timestamp:
            timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        else:
            timestamp_str = "Unknown Time"
        
        # Check if protocol_id exists and is valid
        if protocol_id and protocol_id != "unknown":
            # Construct the protocol link using the correct format
            protocol_link = f"https://www.phenxtoolkit.org/protocols/view/{protocol_id}"
            # Create a Markdown-formatted clickable link
            protocol_text = f"[Protocol {protocol_id}]({protocol_link})"
        else:
            protocol_text = "No protocol"
    
        # Append the formatted conversation entry to history_text
        history_text += f"[{timestamp_str}]\n**User:** {user_msg}\n**psai:** {psai_response}\n**Protocol:** {protocol_text}\n\n"
    
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

# Map to store protocol ID and its subsequent score
protocol_scores = {}

# -------------------- User Interface -------------------- #

def create_user_interface():
    with gr.Blocks(css="""
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600&display=swap');

        *, body, .gradio-container {
            font-family: 'Montserrat', sans-serif;
            background-color: #f9fbfd;
            color: #333333;
        }

        .gr-button-primary {
            background-color: #4A90E2;
            color: white;
            border-radius: 8px;
            padding: 10px 20px;
            font-weight: 600;
        }

        .gr-button-primary:hover {
            background-color: #357ABD;
        }

        .gradio-container h1, .gradio-container h2, .gradio-container h3 {
            color: #4A90E2;
        }

        /* Live status indicator styling */
        #live_status {
            font-size: 14px;
            color: #555555;
        }
    """) as user_interface:
        # Header Section
        with gr.Row():
            gr.Markdown("""
                <div style="text-align: center;">
                    <img src="https://resplendent-tartufo-44d2df.netlify.app/static/carbonift-7e7c45f6345b7ff58f1429b30a003654.png" alt="Clinic Logo" style="width: 120px; border-radius: 12px; margin-bottom: 10px;"/>
                    <h1 style="color: #4A90E2; font-weight: 600; margin: 0;">Welcome to PS.AI</h1>
                    <p style="font-size: 16px; color: #555; margin: 5px 0;">Your trusted mental health pre-screener</p>
                </div>
            """)

        # Chat and Status Section
        with gr.Row():
            # Chatbot Component
            chatbot = gr.Chatbot(
                label="Conversation",
                show_label=False,
                elem_id="chatbot",
                value=[
                    ("Hi psai", "Hello, I'm psai. How are you feeling today? I'm here to listen and understand your current state of mind. Please share any thoughts, emotions, or concerns you're experiencing—no matter how big or small. Let's work through this together.")
                ]
            )

            # Live Status Indicator
            with gr.Column(scale=1, min_width=150):
                gr.Markdown(
                    """
                    <div style="display: flex; align-items: center; justify-content: center; padding: 10px; background-color: #ffffff; border: 1px solid #e3e7eb; border-radius: 8px; box-shadow: 0px 2px 4px rgba(0, 0, 0, 0.1);">
                        <span style="height: 15px; width: 15px; background-color: #4CAF50; border-radius: 50%; display: inline-block; margin-right: 8px;"></span>
                        <span style="font-size: 14px; font-weight: 600; color: #555;">Dr. Tester is live</span>
                    </div>
                    """,
                    elem_id="live_status"
                )

        # Textbox Section
        textbox = gr.Textbox(
            placeholder="Type your response here...",
            label="Your Message",
            lines=1,
            elem_id="textbox",
            container=False,
            interactive=True
        )
        
        # Initialize state variables
        state = gr.State({
            'protocol_selected': False,
            'protocol_id': None,
            'protocol_name': None,
            'protocol_text': None,
            'score': None,
            'protocol_scores': {}
        })
        
        # Function to handle user messages
        def on_message(history, user_input, state):
            """             if stop_event.is_set():
                response = "The chatbot has been stopped by a clinician. Please contact support for further assistance."
                history.append((user_input, response))
                log_conversation(user_input, response, state['protocol_id'])
                return history, state
             """
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
                    
                    protocol_scores[protocol_id] = ""
                    # Generate the first follow-up question
                    follow_up_question = generate_follow_up(protocol_text, user_input, "")
                    
                    if ':' in follow_up_question:
                        follow_up_question_clean = follow_up_question.split(':', 1)[1].strip()
                    # Update chat history
                    history.append((user_input, follow_up_question_clean))
                    
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
                follow_up_question = generate_follow_up(protocol_text, user_input, history)
                score = generate_score(protocol_text, user_input, history)
                # Remove any text and colon before the response
                if ':' in follow_up_question:
                    follow_up_question_clean = follow_up_question.split(':', 1)[1].strip()
                history.append((user_input, follow_up_question_clean))

                protocol_scores[state['protocol_id']] = score
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

        # Function to update conversation history
        def update_clinician_view():
            return get_conversation_history()
        
        # Function to clear conversation history
        def clear_history():
            clear_conversation_history()
            update_clinician_view()
            return "Conversation history has been cleared."
        
        def update_protocol_scores():
            if not protocol_scores:
                return "No protocols have been selected yet."
            
            # Format the protocol scores as a markdown list
            scores_text = "### Selected Protocols:\n"
            for protocol_id, score in protocol_scores.items():
                protocol_link = f"https://www.phenxtoolkit.org/protocols/view/{protocol_id}"
                scores_text += f"- [{protocol_id}]({protocol_link}): {score}\n"
            
            return scores_text
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
        
        # Function to periodically update protocol info
        def auto_update_protocol():
            while not stop_event.is_set():
                protocol_info.value = update_protocol_info()
                time.sleep(5)
        
        # Function to stop the chatbot
        """         def stop_chatbot():
            stop_event.set()
            clear_conversation_history()
            return "Chatbot has been stopped.", update_protocol_info() """

        gr.Markdown("# Clinician Dashboard")
        
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("## Conversation History")
                clinician_history = gr.Markdown(value="Fetching conversation history...", label="Conversation Log")
                update_btn = gr.Button("Refresh Conversation History")
                # Add a button to clear conversation history
                clear_history_btn = gr.Button("Clear Conversation History")
                clear_history_btn.click(fn=clear_history, outputs=[clinician_history])
            with gr.Column(scale=1):
                gr.Markdown("## Current Protocol")
                protocol_info = gr.Markdown(value="No protocol selected.", label="Protocol Details")
                refresh_protocol_btn = gr.Button("Refresh Protocol Info")
                refresh_protocol_btn.click(fn=update_protocol_info, outputs=protocol_info)
                gr.Markdown("## Current Protocol Scoring")
                scoring_info = gr.Markdown(value="Fetching protocol scoring...", label="Protocol Scoring")
                refresh_scoring_btn = gr.Button("Refresh Protocol Scoring")
                refresh_scoring_btn.click(fn=update_protocol_scores, outputs=scoring_info)
                # stop_btn = gr.Button("Stop Chatbot")

                #stop_btn.style(full_width=True, variant="stop")
        # Set up button actions
        update_btn.click(fn=update_clinician_view, outputs=clinician_history)
        # stop_btn.click(fn=stop_chatbot, outputs=[clinician_history, protocol_info])
        

    return clinician_interface

# -------------------- Launch Both Interfaces -------------------- #

def launch_gradio_interfaces():
    user_interface = create_user_interface()
    clinician_interface = create_clinician_interface()
    
    # Launch user interface on port 7860
    user_thread = threading.Thread(target=user_interface.launch, kwargs={
        "server_name": "localhost",
        "server_port": 7860,
        "share": False
    }, daemon=True)
    
    # Launch clinician interface on port 7861
    clinician_thread = threading.Thread(target=clinician_interface.launch, kwargs={
        "server_name": "localhost",
        "server_port": 7861,
        "share": False
    }, daemon=True)
    user_thread.start()
    clinician_thread.start()
    
    user_thread.join()
    clinician_thread.join()

if __name__ == "__main__":
    launch_gradio_interfaces()

