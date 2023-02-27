# Description: Generic functions used in the project. Includes transformations, embeddings, and other functions.
# References:
# https://platform.openai.com/docs/tutorials/web-qa-embeddings
import openai
import tiktoken
from openai.embeddings_utils import distances_from_embeddings
import numpy as np

# Load the cl100k_base tokenizer which is designed to work with the ada-002 model
TOKENIZER = tiktoken.get_encoding("cl100k_base")

# Function to split the text into chunks of a maximum number of tokens
def split_text_based_on_token_length(text, max_tokens=500):

    print("Splitting text into chunks of {} tokens".format(max_tokens))
    print("Text: {}".format(text))
    # Split the text into sentences
    sentences = text.split('. ')

    # Get the number of tokens for each sentence
    n_tokens = [len(TOKENIZER.encode(" " + sentence)) for sentence in sentences]
    
    chunks = []
    tokens_so_far = 0
    chunk = []

    if len(sentences) == 1:
        return [[sentences]]

    # Loop through the sentences and tokens joined together in a tuple
    for sentence, token in zip(sentences, n_tokens):

        # If the number of tokens so far plus the number of tokens in the current sentence is greater 
        # than the max number of tokens, then add the chunk to the list of chunks and reset
        # the chunk and tokens so far
        if tokens_so_far + token > max_tokens:
            chunks.append(". ".join(chunk) + ".")
            chunk = []
            tokens_so_far = 0

        # If the number of tokens in the current sentence is greater than the max number of 
        # tokens, go to the next sentence
        if token > max_tokens:
            continue

        # Otherwise, add the sentence to the chunk and add the number of tokens to the total
        chunk.append(sentence)
        tokens_so_far += token + 1

    # Add the last chunk to the list of chunks
    chunks.append(". ".join(chunk) + ".")

    return chunks


def create_context(
    question, df_in, max_len=1800, size="ada"
):
    """
    Create a context for a question by finding the most similar context from the dataframe
    """

    # Create copy of dataframe
    df = df_in.copy()
    df['memory_embeddings'] = df.memory_log.apply(lambda x: openai.Embedding.create(input=x, engine='text-embedding-ada-002')['data'][0]['embedding'])
    df['n_tokens'] = df.memory_log.apply(lambda x: len(TOKENIZER.encode(x)))
    df['memory_embeddings'] = df['memory_embeddings'].apply(np.array)
    #df['memory_embeddings'] = df['memory_embeddings'].apply(eval).apply(np.array)
    
    # Get the embeddings for the question
    q_embeddings = openai.Embedding.create(input=question, engine='text-embedding-ada-002')['data'][0]['embedding']

    # Get the distances from the embeddings
    df['distances'] = distances_from_embeddings(q_embeddings, df['memory_embeddings'].values, distance_metric='cosine')


    returns = []
    cur_len = 0

    # Sort by distance and add the text to the context until the context is too long
    for i, row in df.sort_values('distances', ascending=True).iterrows():
        
        # Add the length of the text to the current length
        cur_len += row['n_tokens'] + 4
        
        # If the context is too long, break
        if cur_len > max_len:
            break
        
        # Else add it to the text that is being returned
        returns.append(row["memory_log"])

    # Return the context
    return "\n\n###\n\n".join(returns)


# Function to remove newlines from a pandas series
def remove_newlines(serie):
    serie = serie.str.replace('\n', ' ')
    serie = serie.str.replace('\\n', ' ')
    serie = serie.str.replace('  ', ' ')
    serie = serie.str.replace('  ', ' ')
    return serie

