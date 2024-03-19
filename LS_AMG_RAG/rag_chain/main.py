from IPython.display import display, Markdown
import importlib
import json
import time

from LS_AMG_RAG import utils, prompt_utils
from LS_AMG_RAG.rag_chain import rag_utils

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

def get_keywords_and_metadata(query):
    keywords = utils.keyword_yake(query)
    metadata = utils.extract_metadata(query)
    return keywords, metadata

def get_relevant_documents(query, client, collection):
    result = list(rag_utils.keyword_search(client=client, query=query, collection=collection))
    doc_ids = [doc['Doc_ID'] for doc in result]
    titles = [client['RAG']['Docs'].find_one({'_id': doc_ids[idx]})['Doc_Title'] for idx in range(len(doc_ids))]
    scores = [doc['score'] for doc in result]
    return titles, scores

if __name__ == "__main__":

    uri = "mongodb+srv://team-all:HHcJOjFa0lD5zHma@lms-amg-rag.kqmslmy.mongodb.net/?retryWrites=true&w=majority"
    client = MongoClient(uri, server_api=ServerApi("1"))
    docs = client['RAG']['Docs']
    metadata = client['RAG']['Metadata']

    gemini = prompt_utils.Gemini()

    try:
        client.admin.command("ping")
        print("Pinged your deployment. You successfully connected to MongoDB!")
    except Exception as e:
        print(e)

    # Step 1: Get keywords and metadata from the query
    query = "Who are the members of Instagram's board of directors?"
    start_time = time.time()
    step1_start_time = time.time()
    keywords, metadata_query = get_keywords_and_metadata(query)
    print("Step 1: Get keywords and metadata from the query")
    print(f"Query: {query}")
    print(f"Keywords from query: {', '.join(keywords)}")
    print(f"Metadata from query: {json.dumps(metadata_query, indent=2)}")
    step1_end_time = time.time()
    print(f"Time taken for Step 1: {step1_end_time - step1_start_time} seconds")
    print("\n\n ---------------- \n\n")

    # Step 2: Get the most relevant documents using keyword search
    step2_start_time = time.time()
    titles, scores = get_relevant_documents(query, client, 'Metadata')
    print("Step 2: Get the most relevant documents using keyword search")
    print("Titles \t Scores")
    print("\n".join("{} \t {}".format(x, y) for x, y in zip(titles, scores)))
    step2_end_time = time.time()
    print(f"Time taken for Step 2: {step2_end_time - step2_start_time} seconds")
    print("\n\n ---------------- \n\n")
    exit()

    # Step 3: Get the most relevant documents using vector search from the filtered documents
    step3_start_time = time.time()
    vector_result = list(rag_utils.vector_search(client=client, query=query, titles=titles, collection='Docs'))
    print("Step 3: Get the most relevant documents using vector search from the filtered documents")
    print("Doc_Title \t Score")
    print("\n".join("{} \t {}".format(x['Doc_Title'], x['score']) for x in vector_result))
    step3_end_time = time.time()
    print(f"Time taken for Step 3: {step3_end_time - step3_start_time} seconds")
    print("\n\n ---------------- \n\n")

    # Step 4: Pass the most relevant document to the Gemini model
    step4_start_time = time.time()
    metaprompt = rag_utils.metaprompt.format(query=query, 
                                             relevant_document_title=vector_result[0]['Doc_Title'].split('.')[0], 
                                             relevant_document=vector_result[0]['Text'])

    result = gemini.send_message(metaprompt).text
    print("Step 4: Pass the most relevant document to the Gemini model")
    print(result)
    step4_end_time = time.time()
    print(f"Time taken for Step 4: {step4_end_time - step4_start_time} seconds")
    print("\n\n ---------------- \n\n")

    end_time = time.time()
    print(f"Total time taken: {end_time - start_time} seconds")