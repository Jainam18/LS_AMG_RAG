import google.generativeai as palm

import os
import time
from dotenv import load_dotenv

from IPython.display import display, Markdown

load_dotenv()
palm.configure(api_key=os.getenv("GEMINI_API_KEY"))

GENERATION_CONFIG = palm.types.GenerationConfig(
    candidate_count=None,
    stop_sequences=None,
    max_output_tokens=None,
    temperature=None
)


class Gemini:
    '''
    A class to interact with the Gemini chat model.

    Variables:
    model_name (str): The name of the model to use. Options are 'gemini-pro' (text only) and 'gemini-pro-vision' (text and images).
    model (google.generativeai.GenerativeModel): The generative model to use for the chat.
    chat (google.generativeai.generative_models.ChatSession): A chat session object that can be used to continue the conversation.
    '''

    def __init__(self, model_name='gemini-pro', chat_history=[]):
        '''
        Initialize the Gemini chat model.

        Args:
        model_name (str): The name of the model to use. Options are 'gemini-pro' (text only) and 'gemini-pro-vision' (text and images).
        chat_history (list): A list of strings that represent the chat history. The last string in the list is the most recent message.
        - Check format of chat_history here: https://ai.google.dev/tutorials/python_quickstart#chat_conversations
        '''
        self.model_name = model_name
        self.model = palm.GenerativeModel(model_name)
        self.chat = self.model.start_chat(history=chat_history)

    def send_message(self, message, generation_config=GENERATION_CONFIG, retry_time=10):
        '''
        Send a query to the chat model.

        Args:
        message (str): The message to send to the chat model.
        generation_config (google.generativeai.types.GenerationConfig): The generation config to use for the message.
        retry_time (int): The number of seconds to wait before retrying if there is an error.

        Returns:
        response (google.generativeai.generative_models.ChatResponse): The response from the chat model.
        '''
        try:
            response = self.chat.send_message(
                message, generation_config=generation_config)
            return response
        except Exception as e:
            print(e)
            print(f"Retrying in {retry_time} seconds...")
            time.sleep(retry_time)
            return self.send_message(self.chat, message)

    def count_tokens_in_message(self, text):
        '''
        Count the number of tokens in a given text.

        Args:
        text (str): The text to count the tokens for.

        Returns:
        count (int): The number of tokens in the text.
        '''
        count = self.model.count_tokens(text).total_tokens
        return count

    def count_tokens_in_chat(self):
        '''
        Count the number of tokens in the chat history.

        Args:
        None

        Returns:
        count (int): The number of tokens in the chat history.
        '''
        count = 0
        for message in self.chat.history:
            count += self.model.count_tokens(
                message.parts[0].text).total_tokens
        return count

    def display_chat(self, format='markdown'):
        '''
        Display the chat history in either MarkDown or simple print format.

        Args:
        None

        Returns:
        None
        '''
        chat_text = ''
        for i in range(0, len(self.chat.history), 2):
            chat_text += f'You: {self.chat.history[i]}\n'
            chat_text += f'Model: {self.chat.history[i+1]}\n\n'
            chat_text += '---\n\n'

        if format.lower == 'markdown':
            display(Markdown(chat_text))
        else:
            print(chat_text)

    def gen_embeddings(self, text, title=None):
        '''
        Generate embeddings for the given text.

        Args:
        text (str): The text to generate embeddings for.
        title (str): The title of the text (optional).

        Returns:
        embeddings (list): A list of embeddings for the text.
        '''
        embeddings = palm.embed_content(model = 'models/embedding-001',
                                        content=text,
                                        task_type="retrieval_document",
                                        title=title)
        return embeddings['embedding']
