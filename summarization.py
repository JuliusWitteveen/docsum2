# -------------------------------------------------------------------
# summarization.py
# -------------------------------------------------------------------
"""
The summarization module implements the core functionality for summarizing text documents. It utilizes libraries like langchain, sklearn, numpy, and kneed for various tasks such as text splitting, embedding, clustering, and summarization using language models. This module is closely tied to the main application flow in 'main.py' and relies on configuration settings from the 'config' module.
"""

import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from sklearn.cluster import KMeans
import numpy as np
from kneed import KneeLocator
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import config  # Import the config module

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Use values from config module instead of defining them here
# 'DEFAULT_CHUNK_SIZE' and 'DEFAULT_CHUNK_OVERLAP' are configuration values imported from the 'config' module.
# They define default parameters for the text splitting process in the summarization.
DEFAULT_CHUNK_SIZE = config.DEFAULT_CHUNK_SIZE
DEFAULT_CHUNK_OVERLAP = config.DEFAULT_CHUNK_OVERLAP

def split_text(text, chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP):
    """
    Splits the input text into manageable chunks based on the specified chunk size and overlap.

    This function is a preparatory step in the summarization process, enabling handling of large texts
    by breaking them into smaller parts. It is used in the 'execute_summary' function.

    Args:
        text (str): The text to be split.
        chunk_size (int): The size of each chunk.
        chunk_overlap (int): The overlap between chunks.

    Returns:
        list: A list of tuples containing the chunk of text and its sequence index.

    Raises an exception if the text splitting process encounters an error.
    """
    try:
        text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", "\t"], chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        docs = text_splitter.create_documents([text])
        # Adding sequence identifiers to each chunk
        docs_with_id = [(doc, idx) for idx, doc in enumerate(docs)]
        
        # Logging the number of chunks created
        logging.info(f"Text split into {len(docs_with_id)} chunks.")

        return docs_with_id
    except Exception as e:
        logging.error(f"Error during text splitting: {e}")
        raise

def embed_text(docs, openai_api_key):
    """
    Embeds the provided text documents using OpenAI embeddings.

    This function is a crucial part of the summarization process, where it transforms
    the text chunks into vector representations for further clustering analysis.

    Args:
        docs (list): A list of text documents to be embedded.
        openai_api_key (str): The API key for accessing OpenAI services.

    Returns:
        list: A list of vector embeddings for the provided documents.

    Raises an exception if the embedding process encounters an error.
    """
    try:
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        vectors = embeddings.embed_documents([x.page_content for x in docs])
        return vectors
    except Exception as e:
        logging.error(f"Error during text embedding: {e}")
        raise

def determine_optimal_clusters(vectors, max_clusters=100):
    """
    Determines the optimal number of clusters for KMeans clustering of the text embeddings.

    This function is essential for identifying the best way to group text chunks based on
    their embeddings, which aids in creating a concise summary.

    Args:
        vectors (list): A list of vector embeddings.
        max_clusters (int): The maximum number of clusters to consider.

    Returns:
        int: The optimal number of clusters.

    Raises an exception if there's an error in determining the optimal clusters.
    """
    try:
        num_samples = len(vectors)
        if num_samples == 0:
            raise ValueError("No data points available for clustering.")

        max_clusters = min(num_samples, max_clusters)
        sse = []
        for k in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(vectors)
            sse.append(kmeans.inertia_)

        elbow_point = KneeLocator(range(1, len(sse) + 1), sse, curve='convex', direction='decreasing').elbow
        return elbow_point or 1
    except Exception as e:
        logging.error(f"Error determining optimal clusters: {e}")
        raise

def cluster_embeddings(vectors, num_clusters):
    """
    Clusters the provided embeddings into the specified number of clusters.

    After determining the optimal number of clusters, this function groups the embeddings,
    which is a step towards identifying key chunks for summarization.

    Args:
        vectors (list): The vector embeddings to be clustered.
        num_clusters (int): The number of clusters to form.

    Returns:
        list: Indices of the closest embeddings to the cluster centers.

    Raises an exception if the clustering process encounters an error.
    """
    try:
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10).fit(vectors)
        closest_indices = [np.argmin(np.linalg.norm(vectors - center, axis=1)) for center in kmeans.cluster_centers_]
        return sorted(closest_indices)
    except Exception as e:
        logging.error(f"Error during clustering embeddings: {e}")
        raise

def process_chunk(doc, llm3_turbo, map_prompt_template):
    """
    Processes a single text chunk for summarization using a language model.

    Each text chunk is individually summarized using the provided language model and prompt template.
    This function is part of the parallelized summarization process.

    Args:
        doc (str): The text chunk to be summarized.
        llm3_turbo (ChatOpenAI): The language model used for summarization.
        map_prompt_template (PromptTemplate): The template for summarization prompts.

    Returns:
        str: The summarized text of the chunk.

    Logs the first 100 characters of the processed chunk and raises an exception if an error occurs.
    """
    try:
        summary = load_summarize_chain(llm=llm3_turbo, chain_type="stuff", prompt=map_prompt_template).run([doc])
        # Log the first 100 characters of the processed chunk
        logging.info(f"Processed Chunk (first 100 characters): {summary[:100]}...")
        return summary
    except Exception as e:
        logging.error(f"Error summarizing document chunk: {e}")
        return ""

def generate_chunk_summaries(docs_with_id, selected_indices, openai_api_key, custom_prompt, max_workers=10):
    """
    Generates summaries for selected chunks of text in parallel.

    This function orchestrates the summarization of multiple text chunks, utilizing a ThreadPoolExecutor
    for parallel processing. It is a crucial part of the overall summarization workflow.

    Args:
        docs_with_id (list): A list of tuples containing text chunks and their indices.
        selected_indices (list): Indices of the chunks to be summarized.
        openai_api_key (str): The API key for OpenAI services.
        custom_prompt (str): The custom prompt for the summarization.
        max_workers (int): The maximum number of threads for parallel processing.

    Returns:
        list: A list of tuples containing the chunk index and its summary.

    Raises an exception if there's an error in the parallel summarization process.
    """
    try:
        llm3_turbo = ChatOpenAI(temperature=0, openai_api_key=openai_api_key, max_tokens=4096, model='gpt-3.5-turbo-16k')
        map_prompt_template = PromptTemplate(template=f"```{{text}}```\\n{custom_prompt}", input_variables=["text"])
        
        logging.info(f"Selected chunk indices for summarization: {selected_indices}")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_id = {executor.submit(process_chunk, docs_with_id[i][0], llm3_turbo, map_prompt_template): docs_with_id[i][1] for i in selected_indices}
            summary_list = []

            for future in as_completed(future_to_id):
                doc_id = future_to_id[future]
                try:
                    chunk_summary = future.result()
                    if not isinstance(chunk_summary, str):
                        chunk_summary = str(chunk_summary)  # Convert to string if not already
                    logging.info(f"Chunk {doc_id} Summary: {chunk_summary[:100]}...")  # Log first 100 characters of the summary
                    summary_list.append((doc_id, chunk_summary))
                except Exception as e:
                    logging.error(f"Error summarizing document chunk at index {doc_id}: {e}")

        return summary_list
    except Exception as e:
        logging.error(f"Error in generating chunk summaries: {e}")
        raise

def execute_summary(text, api_key, custom_prompt, chunk_size, chunk_overlap, use_clustering=False, progress_update_callback=None):
    """
    Executes the summarization process on the provided text.

    This function encompasses the entire workflow of summarizing a text document, including
    text splitting, embedding, optional clustering, and generating summaries for each chunk.

    Args:
        text (str): The text to be summarized.
        api_key (str): The API key for OpenAI services.
        custom_prompt (str): The custom prompt for the summarization.
        chunk_size (int): The size of each chunk for text splitting.
        chunk_overlap (int): The overlap between chunks.
        use_clustering (bool): Whether to use clustering for summarization.
        progress_update_callback (function): Optional callback function for progress updates.

    Returns:
        str: The final summarized text.

    Raises an exception if there's an error in the summarization process.
    """
    try:
        # Split the text into chunks
        docs_with_id = split_text(text, chunk_size, chunk_overlap)
        if progress_update_callback:
            progress_update_callback(30)

        # Embed the text chunks
        vectors = embed_text([doc[0] for doc in docs_with_id], api_key)
        if progress_update_callback:
            progress_update_callback(40)

        if use_clustering:
            logging.info("Applying clustering for summarization.")
            num_clusters = determine_optimal_clusters(vectors)
            logging.info(f"Number of clusters determined by elbow method: {num_clusters}")
            selected_indices = cluster_embeddings(vectors, num_clusters)
            logging.info(f"Selected chunk indices for summarization (clustering applied): {selected_indices}")
        else:
            logging.info("Summarizing without clustering.")
            selected_indices = range(len(docs_with_id))

        if progress_update_callback:
            progress_update_callback(50)

        # Generate summaries for selected chunks
        summaries_with_id = generate_chunk_summaries(docs_with_id, selected_indices, api_key, custom_prompt)

        # Sorting summaries based on their sequence identifier
        summaries_with_id.sort(key=lambda x: x[0])
        final_summary = "\n\n".join([summary for _, summary in summaries_with_id])

        if progress_update_callback:
            progress_update_callback(90)

        logging.info(f"Final Summary (first 500 characters): {final_summary[:500]}...")

        return final_summary
    except Exception as e:
        logging.error(f"Error in the summarization process: {e}")
        raise
