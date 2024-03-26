from LS_AMG_RAG import utils
import json

class RAG:

    def __init__(self, top_k=[1, 3, 5, 10], queries_path=None):
        self.top_k = {k: [] for k in top_k}
        self.step_times = {
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

        self.metaprompt = """You are a helpful and informative bot that answers questions using text from the reference document included below. \
Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. \
However, you are talking to a non-technical audience, so be sure to break down complicated concepts and \
strike a friendly and converstional tone. \
Make relevant assumptions and use your best judgement to answer the question. \

DOCUMENT:

{relevant_document}

Use the above information from the document to answer the following question:
{query}

ANSWER:
"""

        