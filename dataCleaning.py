import pandas as pd
from uuid import uuid4
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from tqdm import tqdm  # Import tqdm for progress bars
from pymongo import MongoClient
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_community.llms import OpenAI
from langchain_community.document_loaders import DirectoryLoader, CSVLoader

load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = f"AIE1 - LangGraph - {uuid4().hex[0:8]}"
os.environ["LANGCHAIN_API_KEY"] = os.getenv('LANGCHAIN_API_KEY')
os.environ['MONGODB_ATLAS_CLUSTER_URI'] = os.getenv('MONGODB_ATLAS_CLUSTER_URI')


def process_csv_file(file_path):
    # Load the CSV file
    bad_line_count = 0

    # Define a function to handle bad lines
    def bad_line_handler(line):
        nonlocal bad_line_count
        bad_line_count += 1

    df = pd.read_csv(file_path, encoding='ISO-8859-1', on_bad_lines=bad_line_handler, engine='python')
    
    # Convert each row to a textual representation
    def row_to_text(row):
        # Create a string like "Column1: value1, Column2: value2, ..."
        return ', '.join([f"{col}: {row[col]}" for col in df.columns])

    # Apply this function to each row with a progress bar
    textual_representation = df.apply(row_to_text, axis=1)
    
    # print(f"Number of bad lines in {os.path.basename(file_path)}: {bad_line_count}")
    # Convert to list and return with filename
    return os.path.basename(file_path), textual_representation.tolist()

def process_csv_directory(directory_path):
    all_results = []
    
    # Iterate through all CSV files in directory with a progress bar
    for filename in tqdm(os.listdir(directory_path), desc="Processing files"):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory_path, filename)
            filename, text_list = process_csv_file(file_path)
            
            # Print results for each file
            # print(f"\nProcessing file: {filename}")
            for i, text in enumerate(text_list):
                print(f"Row {i+1}: {text}")
            
            all_results.append((filename, text_list))
    
    return all_results


client = MongoClient(os.environ['MONGODB_ATLAS_CLUSTER_URI'])

DB_NAME = "phenx_data"
COLLECTION_NAME = "PhenX_langchain_loader"

# ATLAS_VECTOR_SEARCH_INDEX_NAME = "langchain-test-index-vectorstores"

directory_loader = DirectoryLoader(
    path="./ALL_DD_CSV_Files",
    glob="*.csv",  # This ensures only CSV files are loaded
    loader_cls=CSVLoader,  # Specify the loader class to use for each file
    loader_kwargs={'encoding': 'ISO-8859-1'} 
)

# Load the data
# data = directory_loader.load()

# Each item in 'data' is a Document object representing a row in one of the CSV files
# for document in data:
#     print(document.page_content)  # The content of the row
#     print(document.metadata)      # Metadata, such as column names and file path

# Instantiate the Embedding Model
# embeddings = OpenAIEmbeddings(model="text-embedding-3-small",openai_api_key=os.environ['OPENAI_API_KEY'])

MONGODB_COLLECTION = client[DB_NAME][COLLECTION_NAME]

# vectorStore = MongoDBAtlasVectorSearch.from_documents(data, embeddings, collection=MONGODB_COLLECTION )
process_csv_directory('./ALL_DD_CSV_Files')