from LS_AMG_RAG import utils
import json


class RAG:
    '''
    RAG class to store the top_k documents for each query and the time taken for each step.

    Args:
        top_k (list): List of top_k values to store the documents for each query.
        queries_path (str): Path to the queries file. Default is None.

    Attributes:
        top_k (dict): Dictionary to store the top_k documents for each query.
        step_times (dict): Dictionary to store the time taken for each step.
        queries (list): List of queries to be used for the RAG.

    Methods:
        get_keywords_and_metadata: Get keywords and metadata for a given query.
        keyword_search: Search for the keywords in the collection.
        metadata_search: Search for the metadata in the collection.
        vector_search: Search for the vector in the collection.
    '''

    def __init__(self, top_k=[1, 3, 5, 10], queries_path=None):
        self.top_k = {k: [] for k in top_k}
        self.step_times = {
            'extract_keywords': [],
            'keyword_search': [],
            'vector_search': [],
            'llm_gen': [],
            'total': [],
        }
        if queries_path is None:
            self.queries = [
                {"query": "What is Instagram's current business proposal?",
                    "file": "Business Proposal.md"},
                {"query": "What is the marketing plan for Instagram?",
                    "file": "Marketing Plan.md"},
                {"query": "What information does the progress report of Instagram contain?",
                    "file": "Progress Report.md"},
                {"query": "Who are the members of Instagram's board of directors?",
                    "file": "Board of Directors.md"},
                {"query": "What are the diversity and inclusion initiatives implemented by Instagram?",
                    "file": "Diversity, Equity, and Inclusion.md"},
                {"query": "What is the Marketing Objective for Influencer Collaboration Services?",
                    "file": "Marketing Plan.md"},
                {"query": "Who is the target audience of Content Creation and Curation Services?",
                    "file": "Marketing Plan.md"},
                {"query": "What is the financial update for the Reels Optimization Project?",
                    "file": "Progress Report.md"},
                {"query": "Give me details about the progress report of the Stories Upgrade project.",
                    "file": "Progress Report.md"},
                {"query": "Compare the progress report of the Feed Redesign and Stories Upgrade project and draw a conclusion on the information.",
                    "file": "Progress Report.md"},
            ]
        else:
            self.queries = []
            with open(f'{queries_path}.txt') as f:
                for line in f:
                    line = line.replace('\n', '')
                    self.queries.append(json.loads(line))

        self.keyword_search_aggregation = [
            {
                '$search': {
                    'index': '',
                    'text': {
                        'query': '',
                        'path': {
                            'wildcard': '*'
                        }
                    },
                    "scoreDetails": True,
                }
            },
            {
                '$limit': '',
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
        ]

        self.vector_search_aggregation = [
            {
                '$vectorSearch': {
                    'index': 'gemini_vector_index',
                    'path': 'Gemini_vector',
                    'filter': {
                        'Doc_Title': {
                            '$in': None
                        }
                    },
                    'queryVector': None,
                    'numCandidates': None,
                    'limit': None,
                }
            },
            {
                '$project': {
                    'score': {
                        '$meta': 'vectorSearchScore'
                    },
                    'Doc_ID': 1,
                    'Doc_Title': 1,
                    'Text': 1,
                }
            }
        ]

        self.metaprompt = """You are a helpful and informative bot that answers questions using text from the reference document included below. \
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


    def get_keywords_and_metadata(self, query):
        keywords = utils.keyword_yake(query)
        metadata = utils.extract_metadata(query)
        return keywords, metadata

    def keyword_search(self, client, collection, query=None, keywords=None, limit=10):
        self.keyword_search_aggregation[0]['$search']['index'] = collection
        self.keyword_search_aggregation[1]['$limit'] = limit

        if keywords and query:
            raise ValueError("Both keywords and query cannot be provided.")
        elif keywords:
            self.keyword_search_aggregation[0]['$search']['text']['query'] = ' '.join(keywords)
        elif query:
            self.keyword_search_aggregation[0]['$search']['text']['query'] = query
        else:
            raise ValueError("Either keywords or query must be provided.")

        result = client['RAG'][collection].aggregate(self.keyword_search_aggregation)
        
        return result  # not a dict

    def vector_search(self, client, query, titles, collection, limit=10):

        self.vector_search_aggregation[0]['$vectorSearch']['filter']['Doc_Title']['$in'] = titles
        self.vector_search_aggregation[0]['$vectorSearch']['queryVector'] = utils.gemini_vector(query, "query")
        self.vector_search_aggregation[0]['$vectorSearch']['numCandidates'] = len(titles)
        self.vector_search_aggregation[0]['$vectorSearch']['limit'] = len(titles) if len(titles) < limit else limit

        result = client['RAG'][collection].aggregate(self.vector_search_aggregation)

        return result  # not a dict