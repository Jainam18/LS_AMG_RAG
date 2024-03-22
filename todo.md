- [ ] NER Model comparision: Selecting the best model

- [ ] Chroma DB client-erver creation
- [ ] Running these questions and comparing the answers
- [ ] Creating Results section in ppt with the following parameters:
    - Top@k
    - Direct response comparision
    - Time
    - No. of tokens

How to make sure LS-AMG-RAG is working better than vanilla RAG?
1. Compare the metrics of both models
   1. ChromaDB in the client-server mode
   2. Use text splitters to split documents into smaller chunks 
      1. Direct response comparision
      2. Time taken
      3. Top@k
2. Enlarge the corpus to 3k documents using [this](https://www.kaggle.com/datasets/nltkdata/reuters/data)
   1. Calculate the same metrics on this corpus