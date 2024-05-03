import time
import os
from tqdm import tqdm
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from LS_AMG_RAG.data_snythesis import prompt_utils
from LS_AMG_RAG.rag_chain.vanilla_rag.rag import RAG
import pandas as pd


if __name__ == "__main__":
    rag_utils = RAG(queries_path="LS_AMG_RAG/rag_chain/vanilla_rag/queries")

    # chroma_client = chromadb.PersistentClient(path="./") # to use the local database
    # chroma_client = chromadb.HttpClient(host='localhost', port=8000) # for local server
    chroma_client = chromadb.HttpClient(host="18.209.5.239", port=8000) # for AWS server

    gemini = prompt_utils.Gemini()
    google_ef  = embedding_functions.GoogleGenerativeAiEmbeddingFunction(api_key=os.environ['GEMINI_API_KEY'])

    try:
        collection = chroma_client.get_collection(
            name="my_collection",
            embedding_function=google_ef,
        )
        print("Collection already exists...")
    except:
        collection = chroma_client.create_collection(
            name="my_collection",
            embedding_function=google_ef,
            metadata={"hnsw:space": "cosine"})
        documents = []
        metadata = []
        ids = []

        for root, dirs, files in tqdm(os.walk("LS_AMG_RAG/data/"), desc="Indexing documents"):
            for file in files:
                if file.endswith(".md"):
                    category = root.split('\\')[-1]
                    with open(os.path.join(root, file), "r") as f:
                        file_contents = f.read()
                        documents.append(file_contents)
                        metadata.append({
                            "type": category,
                        })
                        ids.append(f"{category}_{file}")

        collection.add(
            documents=documents,
            metadatas=metadata,
            ids=ids,
        )
        print("New collection created and documents indexed...")
    finally:
        print(f"Total number of documents in the collection: {len(collection.get()['ids'])}")

    df = pd.DataFrame(columns=['query', 'true_document', 'vanilla_retrieved_document', 'vanilla_result', 'vanilla_retrieval_time', 'vanilla_generation_time', 'vanilla_total_time', 'vanilla_top_1', 'vanilla_top_3', 'vanilla_top_5', 'vanilla_top_10'])
    for idx, query in tqdm(enumerate(rag_utils.queries)):
        print(f"\nQuery {idx+1}: {query['query']}")
        total_start_time = time.time()
        retrieval_start_time = time.time()
        results = collection.query(
            query_texts=query['query'],
            n_results=20,
        )

        for k in rag_utils.top_k.keys():
            rag_utils.top_k[k].append(any(query['file'] in x for x in results['ids'][0][:k]))

        retrieval_end_time = time.time()
        print(f"True document: {query['file']}")
        print(f"Document retrieved: {results['ids'][0][0]}")
        print(f"Retrieval time: {retrieval_end_time - retrieval_start_time:.2f} seconds")

        gen_start_time = time.time()
        gemini_result = gemini.send_message(message=rag_utils.metaprompt.format(relevant_document=results['documents'][0][0], query=rag_utils.queries[0])).text
        gen_end_time = time.time()
        total_end_time = time.time()

        df.loc[idx] = [query['query'], query['file'], results['ids'][0][0], gemini_result, retrieval_end_time - retrieval_start_time, gen_end_time - gen_start_time, total_end_time - total_start_time, rag_utils.top_k[1][-1], rag_utils.top_k[3][-1], rag_utils.top_k[5][-1], rag_utils.top_k[10][-1]]

        rag_utils.step_times['vector_search'].append(retrieval_end_time - retrieval_start_time)
        rag_utils.step_times['llm_gen'].append(gen_end_time - gen_start_time)
        rag_utils.step_times['total'].append(total_end_time - total_start_time)
        
        print(f"Gemini time: {gen_end_time - gen_start_time:.2f} seconds")
        print(f"Total time: {total_end_time - total_start_time:.2f} seconds")
        print("\n-------------------\n")

    df.to_csv("LS_AMG_RAG/rag_chain/vanilla_rag/vanilla_results.csv", index=False)
    print("Results")
    print("Top@K Results")
    for k in rag_utils.top_k.keys():
        print(f"Top@{k}: {(sum(rag_utils.top_k[k]) * 100) / len(rag_utils.top_k[k]):.2f}%")

    print("\n\n ---------------- \n\n")

    print("Average Step Times")
    print(f"Retrieval (Vector Search): {sum(rag_utils.step_times['vector_search']) / len(rag_utils.step_times['vector_search']):.2f} seconds")
    print(f"Generation: {sum(rag_utils.step_times['llm_gen']) / len(rag_utils.step_times['llm_gen']):.2f} seconds")
    print(f"Total: {sum(rag_utils.step_times['total']) / len(rag_utils.step_times['total']):.2f} seconds")
