from IPython.display import display, Markdown
import importlib
import json
import time
import spacy
from LS_AMG_RAG import utils, prompt_utils
from LS_AMG_RAG.rag_chain.proposed_rag.rag import RAG
import pandas as pd

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

def get_relevant_documents(client, collection, query=None, keywords=None):
    rag_utils = RAG()
    result = list(rag_utils.keyword_search(client=client, query=query, keywords=keywords, collection=collection))
    doc_ids = [doc['Doc_ID'] for doc in result]
    titles = [client['RAG-2']['Docs'].find_one({'_id': doc_ids[idx]})['Doc_Title'] for idx in range(len(doc_ids))]
    scores = [doc['score'] for doc in result]
    return titles, scores

if __name__ == "__main__":

    rag_utils = RAG()

    uri = "mongodb+srv://team-all:HHcJOjFa0lD5zHma@lms-amg-rag.kqmslmy.mongodb.net/?retryWrites=true&w=majority"
    client = MongoClient(uri, server_api=ServerApi("1"))
    docs = client['RAG-2']['Docs']
    metadata = client['RAG-2']['Metadata']

    gemini = prompt_utils.Gemini()

    try:
        client.admin.command("ping")
        print("Pinged your deployment. You successfully connected to MongoDB!")
    except Exception as e:
        print(e)

    df = pd.DataFrame(columns=['query', 'true_document', 'proposed_retrieved_document', 'proposed_response', 'proposed_extract_keywords_time', 'proposed_keyword_search_time', 'proposed_vector_search_time', 'proposed_retrieval_time', 'proposed_generation_time', 'proposed_total_time', 'proposed_top_1', 'proposed_top_3', 'proposed_top_5', 'proposed_top_10'])

    for idx, query in enumerate(rag_utils.queries):
        print(f"Query {idx + 1}: {query['query']}")
        print(f"Number of documents in the collection: {docs.count_documents({})}")
        # Step 1: Get keywords and metadata from the query
        start_time = time.time()
        extract_keywords_start_time = time.time()
        nlp = spacy.load(r"LS_AMG_RAG/metadata_extraction/custom_ner/output/model-best")
        keywords, metadata_query = rag_utils.get_keywords_and_metadata(query['query'],nlp)
        print("Step 1: Keyword Extraction")
        print(f"Query: {query['query']}")
        print(f"Keywords from query: {', '.join(keywords)}")
        print(f"Metadata from query: {json.dumps(metadata_query, indent=2)}")
        extract_keywords_end_time = time.time()
        print(f"Time taken for Keyword extraction: {extract_keywords_end_time - extract_keywords_start_time} seconds")
        print("\n\n ---------------- \n\n")

        # Step 2: Get the most relevant documents using keyword search
        keyword_search_start_time = time.time()
        # Decide whether to use keywords or query for keyword search
        titles, scores = get_relevant_documents(client=client, collection='Metadata', keywords=keywords)
        print("Step 2: Filtering using Keyword Search")
        print("Titles \t Scores")
        print("\n".join("{} \t {}".format(x, y) for x, y in zip(titles, scores)))
        keyword_search_end_time = time.time()
        print(f"Time taken for Filtering: {keyword_search_end_time - keyword_search_start_time} seconds")
        print(f"Number of documents after filtering (Keyword Search): {len(titles)}")
        print("\n\n ---------------- \n\n")

        # Step 3: Get the most relevant documents using vector search from the filtered documents
        vector_search_start_time = time.time()
        vector_result = list(rag_utils.vector_search(client=client, query=query['query'], titles=titles, collection='Docs'))
        print("Step 3: Vector Search from the filtered documents")
        print("Doc_Title \t Score")
        print("\n".join("{} \t {}".format(x['Doc_Title'], x['score']) for x in vector_result))
        vector_search_end_time = time.time()
        print(f"Time taken for Retrieval: {vector_search_end_time - vector_search_start_time} seconds")
        print(f"Number of documents after retrieval (Vector Search): {len(vector_result)}")
        print("Document with top score is passed to the Gemini model.")
        print("\n\n ---------------- \n\n")

        for k in rag_utils.top_k.keys():
            rag_utils.top_k[k].append(any(query['file'] in x['Doc_Title'] for x in vector_result[:k]))

        # Step 4: Pass the most relevant document to the Gemini model
        llm_gen_start_time = time.time()
        metaprompt = rag_utils.metaprompt.format(query=query['query'], 
                                                relevant_document_title=vector_result[0]['Doc_Title'].split('.')[0], 
                                                relevant_document=vector_result[0]['Text'])

        result = gemini.send_message(metaprompt).text
        print("Step 4: Pass the most relevant document to the Gemini model")
        # print(result)
        llm_gen_end_time = time.time()
        print(f"Time taken for Generation: {llm_gen_end_time - llm_gen_start_time} seconds")
        print("\n\n ---------------- \n\n")

        end_time = time.time()
        print(f"Total time taken: {end_time - start_time} seconds")
        print("\n\n ---------------- \n\n")
        rag_utils.step_times['extract_keywords'].append(extract_keywords_end_time - extract_keywords_start_time)
        rag_utils.step_times['keyword_search'].append(keyword_search_end_time - keyword_search_start_time)
        rag_utils.step_times['vector_search'].append(vector_search_end_time - vector_search_start_time)
        rag_utils.step_times['llm_gen'].append(llm_gen_end_time - llm_gen_start_time)
        rag_utils.step_times['total'].append(end_time - start_time)
        
        df.loc[idx] = [query['query'], query['file'], vector_result[0]['Doc_Title'], result, extract_keywords_end_time - extract_keywords_start_time, keyword_search_end_time - keyword_search_start_time, vector_search_end_time - vector_search_start_time, vector_search_end_time - keyword_search_start_time, llm_gen_end_time - llm_gen_start_time, end_time - start_time, rag_utils.top_k[1][-1], rag_utils.top_k[3][-1], rag_utils.top_k[5][-1], rag_utils.top_k[10][-1]]
    print("Results")
    print("Top@K Results")
    for k in rag_utils.top_k.keys():
        print(f"Top@{k}: {(sum(rag_utils.top_k[k]) * 100) / len(rag_utils.queries):.2f}%")
    
    df.to_csv("LS_AMG_RAG/rag_chain/proposed_rag/proposed_results.csv", index=False)
    print("\n\n ---------------- \n\n")

    print("Average Step Times")
    print(f"Keyword Extraction: {sum(rag_utils.step_times['extract_keywords'])/len(rag_utils.queries):.2f} seconds")
    print(f"Keyword Search: {sum(rag_utils.step_times['keyword_search'])/len(rag_utils.queries):.2f} seconds")
    print(f"Vector Search: {sum(rag_utils.step_times['vector_search'])/len(rag_utils.queries):.2f} seconds")
    print(f"Retrieval (Keyword + Vector Search): {sum(rag_utils.step_times['keyword_search'])/len(rag_utils.queries) + sum(rag_utils.step_times['vector_search'])/len(rag_utils.queries):.2f} seconds")
    print(f"Generation: {sum(rag_utils.step_times['llm_gen'])/len(rag_utils.queries):.2f} seconds")
    print(f"Total: {sum(rag_utils.step_times['total'])/len(rag_utils.queries):.2f} seconds")