# If you haven't already, create a .azure/access-key.json file with the following contents:
# {
#   "SPEECH_KEY": "YOUR_SUBSCRIPTION_KEY",
#   "SPEECH_REGION": "YOUR_SERVICE_REGION",
#   "OPENAI_KEY": "YOUR_OPENAI_API_KEY"
# }

import os
import json
import openai
import azure.cognitiveservices.speech as speechsdk
import pandas as pd
import tiktoken

from utils.core import microphone_to_text, answer_question, translate_to_speech
from utils.generic import remove_newlines, split_text_based_on_token_length, create_context

class ChatSocrates:
    '''Chat with Socrates'''

    def __init__(self,
                 access_key_fp="./.azure/access-key.json",
                 memory_csv_fp = "./.azure/memory.csv"):

        self.access_key_fp = access_key_fp
        self._setup_environmental_vars()

        self._setup_openai()

        self._setup_speech()

        self.memory_csv_fp = memory_csv_fp
        self._setup_memory_csv()
    
    
    
    def chat(self):
        '''Chat with Socrates'''
        # Get the question from the user using microphone.
        speech_question = microphone_to_text(self.speech_config)
        #speech_question = "What are the three key skills of summer program?"
        
        # Get the context from memory.
        aux_context = self.get_context_from_memory(speech_question)
        # Get the answer from OpenAI.
        text_answer = answer_question(question=speech_question, context=aux_context)
        # Translate the answer to speech.
        translate_to_speech(text_answer, self.speech_config)

        # Update memory with the question and answer.
        self._update_memory_csv(speech_question, text_answer)
    

    def get_context_from_memory(self, question):
        '''Create context for Socrates from memory csv'''

        context = create_context(question, self.memory_df)
        print("Context: ", context)

        return context


    # Private Methods
    def _update_memory_csv(self, question, answer):
        '''Update Memory CSV to include new question and answer'''
        
        memory_str = self._create_memory_str(question, answer)
        chunks = split_text_based_on_token_length(memory_str)

        print("Chunks: ", chunks)
        for new_memory_log in chunks:
            print("new_memory_log: ", new_memory_log)
            new_log = pd.DataFrame({'memory_log': [new_memory_log]})
            self.memory_df = pd.concat([self.memory_df, new_log], ignore_index=True)
        
        # Save the memory csv.
        self.memory_df.to_csv(self.memory_csv_fp, index=None)
        print("Memory CSV Updated.")
        print("\n", self.memory_df)

        return
    

    def _setup_memory_csv(self):
        '''Setup Memory CSV'''
        
        # Check if the memory csv file exists.
        if not os.path.exists(self.memory_csv_fp):
            # If not, create it.
            with open(self.memory_csv_fp, 'w') as f:
                f.write('memory_log')
        
        # Load memory CSV into memory.
        self.memory_df = pd.read_csv(self.memory_csv_fp, header=0)
        return
    

    def _create_memory_str(self, question, answer):
        '''Create Memory String'''
        question_i = len(self.memory_df) + 1
        memory_str = f"Question {question_i}: " + question.replace('?', '.') + " Answer: " + answer
        return memory_str


    def _setup_environmental_vars(self):
        '''Setup Environmental Variables'''
        # Extract the subscription key and region from "./.azure/access-key.json".
        with open(self.access_key_fp) as f:
            access_key = json.load(f)
            os.environ['SPEECH_KEY'] = access_key['SPEECH_KEY']
            os.environ['SPEECH_REGION'] = access_key['SPEECH_REGION']
            os.environ['OPENAI_KEY'] = access_key['OPENAI_KEY']
    

    def _setup_openai(self):
        '''Setup OpenAI API Key'''
        openai.organization = "org-sIfoOtpmfXX2eYU2RrkOdOzW"
        openai.api_key = os.getenv("OPENAI_KEY")
        #openai.Model.list()
    

    def _setup_speech(self):
        '''Setup Speech SDK'''
        # This example requires environment variables named "SPEECH_KEY" and "SPEECH_REGION"
        self.speech_config = speechsdk.SpeechConfig(subscription=os.environ.get('SPEECH_KEY'), region=os.environ.get('SPEECH_REGION'))
        # The language of the voice that speaks.
        self.speech_config.speech_synthesis_voice_name='en-US-JennyNeural'
        # The language of the text that is spoken.
        self.speech_config.speech_recognition_language="en-US"



if __name__ == "__main__":
    chat_socrates = ChatSocrates()
    chat_socrates.chat()