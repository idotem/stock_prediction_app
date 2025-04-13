import glob
import os


def get_indexed_graphrag_docs(docs_path):
    """
    Gets all documents from the graphrag-10k/input directory
    and returns them in a list with details about each file.

    Returns:
        list: List of dictionaries containing file information (name and size)
    """

    # Check if the directory exists
    if not os.path.exists(docs_path):
        print(f"Error: Directory '{docs_path}' not found.")
        return []

    # Get all files in the directory
    file_paths = glob.glob(os.path.join(docs_path, "*"))

    # Prepare the result list
    documents = []

    # Collect information about each file
    for file_path in file_paths:
        # Skip directories
        if os.path.isdir(file_path):
            continue

        # Get file name and size
        file_name = os.path.basename(file_path)
        file_size_bytes = os.path.getsize(file_path)

        # Convert size to appropriate unit
        if file_size_bytes < 1024:
            size_str = f"{file_size_bytes} bytes"
        elif file_size_bytes < 1024 * 1024:
            size_str = f"{file_size_bytes / 1024:.2f} KB"
        else:
            size_str = f"{file_size_bytes / (1024 * 1024):.2f} MB"

        # Add file information to documents list
        documents.append({
            "name": file_name,
            "size": size_str,
        })

    return documents
