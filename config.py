# -------------------------------------------------------------------
# config.py
# -------------------------------------------------------------------

# Default prompt in English
# Default summarization prompt in English. This prompt is used in 'language_processing.py' and 'main.py'
# as a baseline for summarization and for translation into other languages if necessary.
DEFAULT_PROMPT_EN = "Summarize the text concisely and directly without prefatory phrases. Focus on presenting its key points and main ideas, ensuring that essential details are accurately conveyed in a straightforward manner."

# Default values for chunk size and chunk overlap
# Default values for chunk size and overlap used in the text splitting process in 'summarization.py'.
# These values dictate how text is divided for summarization.
DEFAULT_CHUNK_SIZE = 10000
DEFAULT_CHUNK_OVERLAP = 3000

# Standard directory for saving summaries
# Standard directory path for saving summarized documents. This path is used in 'file_handler.py'
# for determining where to save the output summaries.
SUMMARY_SAVE_PATH = "/path/to/summaries"

SUMMARY_SAVE_PATH = "/path/to/summaries"

# Supported file formats for saving and loading documents
# List of supported file formats for loading and saving documents, used in 'file_handler.py'.
# This ensures the application handles only specific, supported file types.
SUPPORTED_FILE_FORMATS = [".pdf", ".docx", ".rtf", ".txt"]

# Supported languages for translation
# Supported languages for translation in 'language_processing.py'. This list defines which
# languages the application can translate prompts and texts into.
SUPPORTED_LANGUAGES = ["en", "nl", "fr", "es"]

# Default language for translation (if applicable)
# Default language for translation, used as a fallback in language processing.
DEFAULT_TRANSLATION_LANGUAGE = "en"

# Standard directory for API key
# Standard directory path for the API key file, referenced in 'main.py' for retrieving the API key.
# This path is crucial for accessing external services like OpenAI for embeddings and summarization.
API_KEY_PATH = r'C:\\api_key.txt'
