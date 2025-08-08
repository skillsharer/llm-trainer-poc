import pdfplumber
import json
import os

def load_txt(file_path: str) -> str:
    """
    Loads text content from a .txt file.

    Args:
        file_path (str): The full path to the .txt file.

    Returns:
        str: The content of the file as a string.
        
    Raises:
        FileNotFoundError: If the file is not found at the specified path.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: The file '{file_path}' was not found.")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"An error occurred while reading the .txt file: {e}")
        return ""

def load_pdf(file_path: str) -> str:
    """
    Extracts text content from all pages of a .pdf file.

    Args:
        file_path (str): The full path to the .pdf file.

    Returns:
        str: The concatenated text content from all pages.
        
    Raises:
        FileNotFoundError: If the file is not found at the specified path.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: The file '{file_path}' was not found.")
        
    full_text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    full_text += page_text + "\n"
        return full_text.strip()
    except Exception as e:
        print(f"An error occurred while reading the .pdf file: {e}")
        return ""
    
def load_json(file_path: str) -> str:
    """
    Loads text content from a .json file.

    Args:
        file_path (str): The full path to the .json file.

    Returns:
        str: The content of the file as a string.
        
    Raises:
        FileNotFoundError: If the file is not found at the specified path.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: The file '{file_path}' was not found.")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return json.dumps(data, indent=2)
    except Exception as e:
        print(f"An error occurred while reading the .json file: {e}")
        return ""