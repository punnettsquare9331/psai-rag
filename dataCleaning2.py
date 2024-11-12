from bs4 import BeautifulSoup
import os
from uuid import uuid4
from dotenv import load_dotenv
from pymongo import MongoClient
from langchain_openai import OpenAIEmbeddings

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
COLLECTION_NAME = "PhenX_langchain_loader_with_embeddings_v2"
MONGODB_COLLECTION = client[DB_NAME][COLLECTION_NAME]

# Initialize the embedding model
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=os.environ['OPENAI_API_KEY'])

# Load and parse the HTML file
html_file_path = "./phenxtoolkit_basic_report_20241112163341.html"
with open(html_file_path, "r", encoding="utf-8") as file:
    html_content = file.read()

soup = BeautifulSoup(html_content, "html.parser")

# Find all divs with the class "row border p-2"
divs = soup.find_all("div", class_="row border p-2")

print(len(divs));
# Ensure we are targeting the third div as specified
if len(divs) >= 3:
    target_div = divs[3]  # Target the third div with "row border p-2"
    
    # Find all tables within the targeted div
    tables = target_div.find_all("table", class_="table")

    # Loop through each table to extract data
    for table in tables:
        protocol_id = ""
        protocol_name = ""
        description = ""
        protocol_text = ""
        # print(len(tables));
        # Extract information from each row
        for row in table.find_all("tr"):
            cells = row.find_all("td")
            if len(cells) >= 2:

                header = cells[0].get_text(strip=True)
                value = cells[1].get_text(strip=True)

                if header == "Protocol ID":
                    protocol_id = value
                elif header == "Description of Protocol":
                    description = cells[1].get_text(strip=True)
                elif header == "Protocol Text" or header == "Protocol":
                    protocol_text = cells[1].get_text(" ", strip=True)  # Combine all <p> text into one string
            # Extract Protocol Name from the table header
            header_row = table.find("h3")
            if header_row:
                protocol_name = header_row.get_text(strip=True)

        # Generate embedding for the description
        embedding = embeddings_model.embed_query(description)

        # Prepare the document to insert in MongoDB
        document = {
            "protocol_id": protocol_id,
            "protocol_name": protocol_name,
            "embedding": embedding,
            "protocol_text": protocol_text
        }

        # Insert into MongoDB
        MONGODB_COLLECTION.insert_one(document)
        print(f"Inserted document with Protocol ID: {protocol_id}")

else:
    print("The specified div with class 'row border p-2' was not found in the expected position.")
