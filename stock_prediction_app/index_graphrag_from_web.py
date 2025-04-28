import os
import shutil
import subprocess

def move_downloaded_files_and_index():
    os.environ["PATH"] += ":/home/meto/.local/bin"  # Adjust this path accordingly
    """
    Moves all files from 'data/next_docs_to_index' to 'graphrag-10k/input'
    and runs the 'graphrag index' command in the 'graphrag-10k' folder.
    If indexing fails, moves the files back to the original folder.
    """
    source_folder = "data/next_docs_to_index"
    destination_folder = "graphrag-10k/input"

    print("Start moving files to graphrag-10k/input...")
    # Store original working directory to return to it later
    original_dir = os.getcwd()

    # Track moved files to restore them if needed
    moved_files = []

    # Ensure destination directory exists
    os.makedirs(destination_folder, exist_ok=True)

    print("Moving files to graphrag-10k/input...")
    try:
        # Move files from source to destination
        for filename in os.listdir(source_folder):
            source_file = os.path.join(source_folder, filename)
            destination_file = os.path.join(destination_folder, filename)

            # Only move the file if it does not exist in destination
            if os.path.isfile(source_file) and not os.path.exists(destination_file):
                shutil.move(source_file, destination_file)
                moved_files.append((destination_file, source_file))  # Keep track of moved files

        # Use subprocess.run instead of os.system
        try:
            script_path = os.path.join(original_dir, "index-graphrag.sh")
            result = subprocess.run(
                [script_path],
                capture_output=True,
                text=True,
                check=False
            )
            return_code = result.returncode
            print(f"Return code: {return_code}")
            print(f"GraphRAG stdout: {result.stdout}")
            if result.stderr:
                print(f"GraphRAG stderr: {result.stderr}")
        except FileNotFoundError:
            print("Error: graphrag command not found. Make sure it's installed and in your PATH.")
            return_code = 127  # Standard shell code for command not found

        print(f"Moved files: {moved_files}")

        # Check if the command executed successfully
        if return_code == 0:
            print("GraphRAG indexing completed successfully!")
            return True
        else:
            print(f"GraphRAG indexing failed with return code: {return_code}")
            # Return to original directory before moving files back
            os.chdir(original_dir)

            # Move files back to their original location
            print("Moving files back to original location...")
            for dest_file, src_file in moved_files:
                if os.path.exists(dest_file):
                    shutil.move(dest_file, src_file)
                    print(f"Restored: {os.path.basename(src_file)}")
            return False
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        # Return to original directory before moving files back
        os.chdir(original_dir)

        # Move files back to their original location
        print("Moving files back to original location...")
        for dest_file, src_file in moved_files:
            if os.path.exists(dest_file):
                shutil.move(dest_file, src_file)
                print(f"Restored: {os.path.basename(src_file)}")

        return False
    finally:
        # Always make sure we return to the original directory
        os.chdir(original_dir)