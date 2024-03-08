import os
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from sklearn.feature_extraction.text import TfidfVectorizer

uri = "mongodb+srv://team-all:HHcJOjFa0lD5zHma@lms-amg-rag.kqmslmy.mongodb.net/?retryWrites=true&w=majority"
client = MongoClient(uri, server_api=ServerApi("1"))

# Send a ping to confirm a successful connection
try:
    client.admin.command("ping")
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)


# Function to read markdown files and convert them into strings
def read_markdown_files(directory):
    markdown_texts = []
    for filename in os.listdir(directory):
        if filename.endswith(".md"):
            with open(os.path.join(directory, filename), "r", encoding="utf-8") as file:
                markdown_texts.append(file.read())
    return markdown_texts


# Function to generate TF-IDF vectors for the texts
def generate_tfidf_vectors(texts):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    return tfidf_matrix, feature_names


# Function to insert documents into MongoDB
def insert_documents(collection, documents):
    return collection.insert_many(documents)


# Function to create collections and insert data
def store_data_in_mongodb(texts, vectors, client):
    db = client["RAG"]

    # Create collections
    docs_collection = db["Docs"]
    metadata_collection = db["Metadata"]

    # Insert documents and metadata
    documents = []
    metadata = []
    for i, text in enumerate(texts):
        document = {
            "Doc_Title": f"Document {i+1}",
            "Category": "Some Category",
            "Vector": vectors[i].tolist(),
            "Text": text,
            "Flag": "AI or Actual",
            "ID": i + 1,  # Assuming unique ID for each document
        }
        documents.append(document)

        meta = {
            "Keywords": [
                "keyword1",
                "keyword2",
            ],  # Sample keywords, replace with actual data
            "People": ["person1", "person2"],  # Sample people, replace with actual data
            "Organisation": [
                "org1",
                "org2",
            ],  # Sample organizations, replace with actual data
            "Places": ["place1", "place2"],  # Sample places, replace with actual data
            "Email_ID": "example@example.com",  # Sample email, replace with actual data
            "Number": 1234567890,  # Sample number, replace with actual data
            "Doc_ID": i + 1,  # Foreign key reference to document
            "ID": i + 1,  # Assuming unique ID for each metadata entry
        }
        metadata.append(meta)

    # Inserting data into MongoDB
    insert_documents(docs_collection, documents)
    insert_documents(metadata_collection, metadata)


# Function to insert data into MongoDB Docs Collection from arguments
def insert_data_into_mongodb_collection(
    client, doc_title, category, vector, text, flag, keywords, people, org, places, email, number
):
    db = client["RAG"]
    docs_collection = db["Docs"]
    metadata_collection = db["Metadata"]

    document = {
        "Doc_Title": doc_title,
        "Category": category,
        "Vector": vector.tolist(),
        "Text": text,
        "Flag": flag, 
    }
    
    docs_collection.insert_one(document)
    doc_id = docs_collection.find_one({"Doc_Title": doc_title})["_id"]
    
    meta = {
            "Keywords": keywords,
            "People": people,
            "Organisation": org,
            "Places": places,
            "Email_ID": email,
            "Number": number,
            "Doc_ID": doc_id
        }

    return metadata_collection.insert_one(meta)

# Directory containing markdown files
# markdown_directory = "./data/business_docs"

# Read markdown files
# markdown_texts = read_markdown_files(markdown_directory)

# Generate TF-IDF vectors
# tfidf_matrix, feature_names = generate_tfidf_vectors(markdown_texts)

# Store data in MongoDB
# store_data_in_mongodb(markdown_texts, tfidf_matrix.toarray(), client)
