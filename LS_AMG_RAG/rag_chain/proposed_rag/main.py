from IPython.display import display, Markdown
import importlib
import json
import time
import spacy
from LS_AMG_RAG import utils, prompt_utils
from LS_AMG_RAG.rag_chain.proposed_rag.rag import RAG

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

def get_relevant_documents(client, collection, query=None, keywords=None):
    rag_utils = RAG()
    result = list(rag_utils.keyword_search(client=client, query=query, keywords=keywords, collection=collection))
    doc_ids = [doc['Doc_ID'] for doc in result]
    titles = [client['RAG']['Docs'].find_one({'_id': doc_ids[idx]})['Doc_Title'] for idx in range(len(doc_ids))]
    scores = [doc['score'] for doc in result]
    return titles, scores

if __name__ == "__main__":

    rag_utils = RAG()

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

    for idx, query in enumerate(rag_utils.queries):
        print(f"Query {idx + 1}: {query['query']}")
        print(f"Number of documents in the collection: {docs.count_documents({})}")
        # Step 1: Get keywords and metadata from the query
        start_time = time.time()
        step1_start_time = time.time()
        nlp = spacy.load(r"LS_AMG_RAG/metadata_extraction/custom_ner/output/model-best")
        keywords, metadata_query = rag_utils.get_keywords_and_metadata(query['query'],nlp)
        print("Step 1: Keyword Extraction")
        print(f"Query: {query['query']}")
        print(f"Keywords from query: {', '.join(keywords)}")
        print(f"Metadata from query: {json.dumps(metadata_query, indent=2)}")
        step1_end_time = time.time()
        print(f"Time taken for Keyword extraction: {step1_end_time - step1_start_time} seconds")
        print("\n\n ---------------- \n\n")

        # Step 2: Get the most relevant documents using keyword search
        step2_start_time = time.time()
        # Decide whether to use keywords or query for keyword search
        titles, scores = get_relevant_documents(client=client, collection='Metadata', keywords=keywords)
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

        for k in rag_utils.top_k.keys():
            rag_utils.top_k[k].append(any(query['file'] in x['Doc_Title'] for x in vector_result[:k]))

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
        rag_utils.step_times['extract_keywords'].append(step1_end_time - step1_start_time)
        rag_utils.step_times['keyword_search'].append(step2_end_time - step2_start_time)
        rag_utils.step_times['vector_search'].append(step3_end_time - step3_start_time)
        rag_utils.step_times['llm_gen'].append(step4_end_time - step4_start_time)
        rag_utils.step_times['total'].append(end_time - start_time)
        
    print("Results")
    print("Top@K Results")
    for k in rag_utils.top_k.keys():
        print(f"Top@{k}: {(sum(rag_utils.top_k[k]) * 100) / len(rag_utils.queries):.2f}%")
    
    print("\n\n ---------------- \n\n")

    print("Average Step Times")
    print(f"Keyword Extraction: {sum(rag_utils.step_times['extract_keywords'])/len(rag_utils.queries):.2f} seconds")
    print(f"Keyword Search: {sum(rag_utils.step_times['keyword_search'])/len(rag_utils.queries):.2f} seconds")
    print(f"Vector Search: {sum(rag_utils.step_times['vector_search'])/len(rag_utils.queries):.2f} seconds")
    print(f"Retrieval (Keyword + Vector Search): {sum(rag_utils.step_times['keyword_search'])/len(rag_utils.queries) + sum(rag_utils.step_times['vector_search'])/len(rag_utils.queries):.2f} seconds")
    print(f"Generation: {sum(rag_utils.step_times['llm_gen'])/len(rag_utils.queries):.2f} seconds")
    print(f"Total: {sum(rag_utils.step_times['total'])/len(rag_utils.queries):.2f} seconds")