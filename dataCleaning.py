import pandas as pd
import os
import docx
from uuid import uuid4
from dotenv import load_dotenv
from tqdm import tqdm
from pymongo import MongoClient

# Load environment variables
load_dotenv()

# Set up environment variables
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = f"AIE1 - LangGraph - {uuid4().hex[0:8]}"
os.environ["LANGCHAIN_API_KEY"] = os.getenv('LANGCHAIN_API_KEY')
os.environ['MONGODB_ATLAS_CLUSTER_URI'] = os.getenv('MONGODB_ATLAS_CLUSTER_URI')

# Load protocol ID and name mapping
protocol_df = pd.read_excel("protocol_id_name.xls", dtype=str)
protocol_mapping = dict(zip(protocol_df['PROTOCOL_ID'], protocol_df['NAME']))

def read_docx(file_path):
    """Read .docx files using python-docx."""
    doc = docx.Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])

def process_docx_file(file_path):
    """Process a .docx file and return its data in dictionary format."""
    filename_base = os.path.basename(file_path)
    protocol_id = filename_base.split('_')[1]  # Extract protocol_id
    protocol_name = protocol_mapping.get(protocol_id, "Unknown Protocol")  # Get protocol name
    
    # Extract text content from the .docx file
    doc_text = read_docx(file_path)
    
    # Return document data
    return {
        "filename": filename_base,
        "protocol_name": protocol_name,
        "text_content": doc_text
    }

def process_docx_directory(directory_path):
    """Process all .docx files in a directory and insert into MongoDB."""
    documents_to_insert = []
    
    # Iterate through all DOCX files in the directory
    for filename in tqdm(os.listdir(directory_path), desc="Processing .docx files"):
        if filename.endswith('.docx'):
            file_path = os.path.join(directory_path, filename)
            try:
                document_data = process_docx_file(file_path)
                if document_data:
                    documents_to_insert.append(document_data)
            except Exception as e:
                print(f"Error processing file {filename}: {e}")
    
    # Insert documents into MongoDB
    if documents_to_insert:
        try:
            MONGODB_COLLECTION.insert_many(documents_to_insert)
            print(f"Inserted {len(documents_to_insert)} documents successfully.")
        except Exception as e:
            print(f"Error inserting documents to MongoDB: {e}")
    
    return documents_to_insert

# MongoDB setup
client = MongoClient(os.environ['MONGODB_ATLAS_CLUSTER_URI'])
DB_NAME = "phenx_data"
COLLECTION_NAME = "PhenX_langchain_loader_docs"
MONGODB_COLLECTION = client[DB_NAME][COLLECTION_NAME]

# Process .docx documents in the specified directory
results = process_docx_directory('./')
