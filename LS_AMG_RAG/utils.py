import os
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from sklearn.feature_extraction.text import TfidfVectorizer
import pke
from pke.lang import stopwords
import string
import re
import pandas as pd
import spacy
import LS_AMG_RAG.prompt_utils as prompt_utils

# Cleaning the markdown formatting from the text
def remove_markdown_formatting(text):
    # Remove code blocks enclosed in triple backticks
    pattern_code_blocks = r"```[^\\S\\r\\n]*[a-z]*(?:\\n(?!```$).*)*\\n```"
    text_without_code_blocks = re.sub(pattern_code_blocks, '', text, 0, re.DOTALL)

    # Remove asterisks and hash symbols
    pattern_asterisks = r"\*+"
    pattern_hashes = r"#"
    text_without_formatting = re.sub(pattern_asterisks, '', text_without_code_blocks)
    text_without_formatting = re.sub(pattern_hashes, '', text_without_formatting)

    return text_without_formatting

# Function to read markdown files and convert them into strings
def read_markdown_files(root,filename):
    with open(os.path.join(root, filename), 'r') as f:
        raw_text = remove_markdown_formatting(f.read())
    return raw_text

def gemini_vector(text,title):
    gemini = prompt_utils.Gemini()
    response = gemini.gen_embeddings(text,title)
    return response

# Function to generate TF-IDF vectors for the texts
def generate_tfidf_vectors(texts):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    return tfidf_matrix, feature_names

# Extracting keywords using YAKE 
def keyword_yake(text):
    extractor = pke.unsupervised.YAKE()
    stoplist = stopwords.get('english')
    extractor.load_document(input=text,
                            language='en',
                            stoplist=stoplist,
                            normalization=None)
    extractor.candidate_selection(n=3)
    window = 2
    use_stems = False
    extractor.candidate_weighting(window=window,
                                use_stems=use_stems)
    threshold = 0.8
    keyphrases = extractor.get_n_best(n=10, threshold=threshold)
    keyphrases = [keyphrase for keyphrase, score in keyphrases]
    return keyphrases

# Function to extract metadata from the text using custom NER model
def extract_metadata(text,nlp=spacy.load(r"LS_AMG_RAG/metadata_extraction/custom_ner/output/model-best")):
    # nlp1 = spacy.load(r"LS_AMG_RAG/metadata_extraction/custom_ner/output/model-best")
    # nlp1 = spacy.load("../metadata_extraction/custom_ner/output/model-best")
    doc = nlp(text)
    label_dict = {
        "Person": set(),  # Use a set to store unique values
        "Places": set(),
        "Organization": set(),
        "Money": set(),
        "Email Id": set(),
        "Contact_Number": set(),
        # Add other entities here if needed
    }
    for ent in doc.ents:
        if ent.label_ in label_dict:
            label_dict[ent.label_].add(ent.text)  # Add the value to the set

    # Convert sets back to lists
    for label, values_set in label_dict.items():
        label_dict[label] = list(values_set)

    return label_dict

# Extracting dates from the text 
def date_extraction(text):
    # Updated regex pattern
    pattern = r'(\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{4}\b|\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b|\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}\b)'
    dates = re.findall(pattern, text)
    return dates

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
            # "Vector": vectors[i].tolist(),
            "Gemini_vector": vectors,
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
            "Money": [100, 200],  # Sample money, replace with actual data
            "Email_IDs": ["example1@example.com", "example2@example.com"],  # Sample email, replace with actual data
            "Contact_Numbers": [1234567890, 9876543210],  # Sample number, replace with actual data
            "Dates": ["date1", "date2"],  # Sample dates, replace with actual data
            "Doc_ID": i + 1,  # Foreign key reference to document
            "ID": i + 1,  # Assuming unique ID for each metadata entry
        }
        metadata.append(meta)

    # Inserting data into MongoDB
    insert_documents(docs_collection, documents)
    insert_documents(metadata_collection, metadata)


# Function to insert data into MongoDB Docs Collection from arguments
def insert_data_into_mongodb_collection(
    client, doc_title, category, vector, text, flag, keywords, people, org, places, money, email, contact_number, dates_mentioned
):
    db = client["RAG-2"]
    docs_collection = db["Docs"]
    metadata_collection = db["Metadata"]

    document = {
        "Doc_Title": doc_title,
        "Category": category,
        # "Vector": vector.tolist(),
        "Gemini_vector": vector,
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
            "Money": money,
            "Email_ID": email,
            "Contact_Number": contact_number,
            "Dates_Mentioned": dates_mentioned,
            "Doc_ID": doc_id
        }

    return metadata_collection.insert_one(meta)


if __name__ == "__main__":
    uri = "mongodb+srv://team-all:HHcJOjFa0lD5zHma@lms-amg-rag.kqmslmy.mongodb.net/?retryWrites=true&w=majority"
    client = MongoClient(uri, server_api=ServerApi("1"))

    # Send a ping to confirm a successful connection
    try:
        client.admin.command("ping")
        print("Pinged your deployment. You successfully connected to MongoDB!")
    except Exception as e:
        print(e)
    
    
    for root, dirs, files in os.walk('LS_AMG_RAG/data'):
        # print("Hello")
        # print(root, dirs, files)
        for file in files:
            # print("Hello")
            text = read_markdown_files(root,file)
            filename = file
            category = root.split('\\')[-1]
            embeddings = gemini_vector(text, filename)
            keywords = keyword_yake(text) 
            metadata = extract_metadata(text) # need to figure out about duplicates
            people = metadata["Person"]
            org = metadata["Organization"]
            places = metadata["Places"]
            money = metadata["Money"]
            email = metadata["Email Id"]
            contact_number = metadata["Contact_Number"]
            dates_mentioned = date_extraction(text)
            flag="AI"
            insert_data_into_mongodb_collection(client, filename, category, embeddings, text, flag, keywords, people, org, places, money, email, contact_number, dates_mentioned)
            print("Completed")
            # break
        # break
    # First step will be to run a loop which will read all the markdown files and and store their text along with doc name and category 

    # Second we will run tfidf over the text so this will give us our vectors

    # Third we will run keyword and metadata extraction functions over the text and store the metadata in a separate collection

    # Next we will pass all these variables to store_data_in_mongodb function which will store the data in the database