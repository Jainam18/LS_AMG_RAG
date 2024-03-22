from LS_AMG_RAG import utils

def keyword_search(client, query, collection, limit=10):
    result = client['RAG'][collection].aggregate([
        {
            '$search': {
                'index': collection,
                'text': {
                    # 'query': ' '.join(utils.keyword_yake(query)), # to search for the keywords
                    'query': query, # to search for the whole query
                    'path': {
                        'wildcard': '*'
                    }
                },
                "scoreDetails": True,
            }
        },
        {
            '$limit': limit
        },
        {
            '$project': {
                'score': {
                    '$meta': 'searchScore'
                },
                # "scoreDetails": {"$meta": "searchScoreDetails"},
                'Doc_ID': 1,
                'Keywords': 1,
            }
        }
    ])

    return result  # not a dict

def metadata_search(client, query, collection, category, limit=10):
    result = client['RAG'][collection].aggregate([
        {
            '$search': {
                'index': collection,
                'text': {
                    'query': query, # to search for the whole query
                    # 'path': {
                    #     'wildcard': '*'
                    # }
                    'path': category
                },
                "scoreDetails": True,
            }
        },
        {
            '$limit': limit
        },
        {
            '$project': {
                'score': {
                    '$meta': 'searchScore'
                },
                # "scoreDetails": {"$meta": "searchScoreDetails"},
                'Doc_ID': 1,
            }
        }
    ])

    return result  # not a dict

def vector_search(client, query, titles, collection, limit=10):
    result = client['RAG'][collection].aggregate([
        {
            '$vectorSearch': {
                'index': "gemini_vector_index",
                'path': 'Gemini_vector',
                'filter': {
                    'Doc_Title': {
                        '$in': titles
                    }
                },
                'queryVector': utils.gemini_vector(query, "query"),
                'numCandidates': len(titles),
                'limit': len(titles) if len(titles) < limit else limit
            }
        },
        {
            '$project': {
                'score': {
                    '$meta': 'vectorSearchScore'
                },
                'Doc_Title': 1,
                'Text': 1,
            }
        }
    ])

    return result  # not a dict

metaprompt = """You are a helpful and informative bot that answers questions using text from the reference document included below. \
Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. \
However, you are talking to a non-technical audience, so be sure to break down complicated concepts and \
strike a friendly and converstional tone. \
Use your own knowledge base in addition to the information provided in the document to answer the question. \
Make relevant assumptions and use your best judgement to answer the question. \

DOCUMENT TITLE: {relevant_document_title}

DOCUMENT:

{relevant_document}

Use the above information from the document to answer the following question:
{query}

ANSWER:
"""