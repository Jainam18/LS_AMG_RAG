from IPython.display import display, Markdown
import importlib
import json
import time

from LS_AMG_RAG import utils, prompt_utils
from LS_AMG_RAG.rag_chain import rag_utils

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

top_k = {
    1: [],
    3: [],
    5: [],
    10: [],
}

step_times = {
    'Step 1': [],
    'Step 2': [],
    'Step 3': [],
    'Step 4': [],
    'Total': [],
}

queries = [
    {'query': "What is Instagram's current business proposal?",
     'file': "Business Proposal.md"},
    {'query': "What is the marketing plan for Instagram?",
     'file': "Marketing Plan.md"},
    {'query': "What information does the progress report of Instagram contain?",
     'file': "Progress Report.md"},
    {'query': "Who are the members of Instagram's board of directors?",
     'file': "Board of Directors.md"},
    {'query': "What are the diversity and inclusion initiatives implemented by Instagram?",
     'file': 'Diversity, Equity, and Inclusion.md'},
    {'query': "What is the Marketing Objective for Influencer Collaboration Services?",
     'file': 'Marketing Plan.md'},
    {'query': "Who is the target audience of Content Creation and Curation Services?",
     'file': 'Marketing Plan.md'},
    {'query': "What is the financial update for the Reels Optimization Project?",
     'file': 'Progress Report.md'},
    {'query': "Give me details about the progress report of the Stories Upgrade project.",
     'file': 'Progress Report.md'},
    {'query': "Compare the progress report of the Feed Redesign and Stories Upgrade project and draw a conclusion on the information.",
     'file': 'Progress Report.md'}
]

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

    for idx, query in enumerate(queries):
        print(f"Query {idx + 1}: {query['query']}")
        print(f"Number of documents in the collection: {docs.count_documents({})}")
        # Step 1: Get keywords and metadata from the query
        start_time = time.time()
        step1_start_time = time.time()
        # keywords, metadata_query = get_keywords_and_metadata(query['query'])
        print("Step 1: Keyword Extraction")
        # print(f"Query: {query['query']}")
        # print(f"Keywords from query: {', '.join(keywords)}")
        # print(f"Metadata from query: {json.dumps(metadata_query, indent=2)}")
        step1_end_time = time.time()
        print(f"Time taken for Keyword extraction: {step1_end_time - step1_start_time} seconds")
        print("\n\n ---------------- \n\n")

        # Step 2: Get the most relevant documents using keyword search
        step2_start_time = time.time()
        titles, scores = get_relevant_documents(query['query'], client, 'Metadata')
        print("Step 2: Filtering using Keyword Search")
        print("Titles \t Scores")
        print("\n".join("{} \t {}".format(x, y) for x, y in zip(titles, scores)))
        step2_end_time = time.time()
        print(f"Time taken for Filtering: {step2_end_time - step2_start_time} seconds")
        print(f"Number of documents after filtering (Keyword Search): {len(titles)}")
        print("\n\n ---------------- \n\n")

        # Step 3: Get the most relevant documents using vector search from the filtered documents
        step3_start_time = time.time()
        vector_result = list(rag_utils.vector_search(client=client, query=query['query'], titles=titles, collection='Docs'))
        print("Step 3: Vector Search from the filtered documents")
        print("Doc_Title \t Score")
        print("\n".join("{} \t {}".format(x['Doc_Title'], x['score']) for x in vector_result))
        step3_end_time = time.time()
        print(f"Time taken for Retrieval: {step3_end_time - step3_start_time} seconds")
        print(f"Number of documents after retrieval (Vector Search): {len(vector_result)}")
        print("Document with top score is passed to the Gemini model.")
        print("\n\n ---------------- \n\n")

        for k in top_k.keys():
            top_k[k].append(any(query['file'] in x['Doc_Title'] for x in vector_result[:k]))

        # Step 4: Pass the most relevant document to the Gemini model
        step4_start_time = time.time()
        metaprompt = rag_utils.metaprompt.format(query=query['query'], 
                                                relevant_document_title=vector_result[0]['Doc_Title'].split('.')[0], 
                                                relevant_document=vector_result[0]['Text'])

        result = gemini.send_message(metaprompt).text
        print("Step 4: Pass the most relevant document to the Gemini model")
        print(result)
        step4_end_time = time.time()
        print(f"Time taken for Generation: {step4_end_time - step4_start_time} seconds")
        print("\n\n ---------------- \n\n")

        end_time = time.time()
        print(f"Total time taken: {end_time - start_time} seconds")
        print("\n\n ---------------- \n\n")
        step_times['Step 1'].append(step1_end_time - step1_start_time)
        step_times['Step 2'].append(step2_end_time - step2_start_time)
        step_times['Step 3'].append(step3_end_time - step3_start_time)
        step_times['Step 4'].append(step4_end_time - step4_start_time)
        step_times['Total'].append(end_time - start_time)
        
    print("Top K Results")
    for k in top_k.keys():
        print(f"Top {k}: {(sum(top_k[k]) * 100) / len(queries):.2f}%")
    print("\n\n ---------------- \n\n")

    print("Average Step Times")
    print(f"Keyword Extraction: {sum(step_times['Step 1'])/len(queries):.2f} seconds")
    print(f"Keyword Search: {sum(step_times['Step 2'])/len(queries):.2f} seconds")
    print(f"Vector Search (Retrieval): {sum(step_times['Step 3'])/len(queries):.2f} seconds")
    print(f"Generation: {sum(step_times['Step 4'])/len(queries):.2f} seconds")
    print(f"Total: {sum(step_times['Total'])/len(queries):.2f} seconds")