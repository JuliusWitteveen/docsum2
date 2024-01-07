# -------------------------------------------------------------------
# language_processing.py
# -------------------------------------------------------------------
"""
The language_processing module is responsible for language detection and text translation.
It utilizes the 'langdetect' library for detecting the language of a given text and the 'translate' library
for translating text between supported languages. This module is primarily used in conjunction with
the 'main.py' and 'file_handler.py' modules to enhance the functionality of document summarization by
accommodating multiple languages.
"""

from langdetect import detect
from translate import Translator
import logging
import config  # Import the config module

# Configure logging
# Logging is used for debugging and tracking, particularly for language detection and translation processes.
# It helps in identifying issues with language-related operations.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Use values from config module
# 'SUPPORTED_LANGUAGES' and 'DEFAULT_TRANSLATION_LANGUAGE' are configuration values imported from the 'config' module.
# These values determine which languages are supported for translation and the default language for translation.
SUPPORTED_LANGUAGES = config.SUPPORTED_LANGUAGES
DEFAULT_TRANSLATION_LANGUAGE = config.DEFAULT_TRANSLATION_LANGUAGE


def detect_language(text):
    """
    Detects the language of the given text using the 'langdetect' library.

    This function is critical in the 'main.py' module to determine the language of the document being summarized,
    which influences the summarization prompt and possibly the translation of the prompt.

    Args:
        text (str): The text for which to detect the language.

    Returns:
        str: The detected language code (e.g., 'en' for English). Returns 'unknown' if detection fails.

    Exceptions are logged, and 'unknown' is returned in case of failure.
    """
    try:
        language = detect(text)
        logging.info(f"Detected language: {language}")
        return language
    except Exception as e:
        logging.error(f"Error in language detection: {e}")
        return "unknown"

def translate_prompt(prompt_text, target_language):
    """
    Translates the provided prompt text to the specified target language using the 'translate' library.

    This function is used in the 'main.py' module when the detected language of the document
    is not English, allowing the summarization prompt to be in the document's language.

    Args:
        prompt_text (str): The text to be translated.
        target_language (str): The target language code (e.g., 'nl' for Dutch).

    Returns:
        str: The translated text. Returns the original text if translation fails or is not supported.

    If translation to the target language is not supported or fails, the original text is returned.
    """
    if target_language not in SUPPORTED_LANGUAGES:
        logging.warning(f"Translation not supported for language: {target_language}")
        return prompt_text

    translator = Translator(to_lang=target_language)
    try:
        translated_text = translator.translate(prompt_text)
        logging.info(f"Translated text to {target_language}: {translated_text}")
        return translated_text
    except Exception as e:
        logging.error(f"Error in translating text to {target_language}: {e}")
        return prompt_text
