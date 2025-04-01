import requests
from bs4 import BeautifulSoup
import gzip
import time
import os
import shutil

LOCAL = True
LOCAL_FILE = "C:/Users/leeka/Downloads/Kickstarter_2025-03-12T07_34_02_656Z.json.gz"
CHUNK_SIZE = 90 * 1024 * 1024  # 90MB chunks

def get_kickstarter_download_link():
    """
    Retrieves the download link from the Kickstarter datasets page
    """
    url = "https://webrobots.io/kickstarter-datasets/"
    
    try:
        # Fetch the webpage
        response = requests.get(url)
        response.raise_for_status()  # Raise exception for HTTP errors
        
        # Parse the HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find the div with class fusion-text
        fusion_text_div = soup.find('div', class_='fusion-text')
        if not fusion_text_div:
            return None
        
        # Find the first ul in this div
        ul = fusion_text_div.find('ul')
        if not ul:
            return None
        
        # Find the first li in this ul
        li = ul.find('li')
        if not li:
            return None
        
        # Find all 'a' tags in the li and look for one with text containing "name json"
        for a in li.find_all('a'):
            if a.text and "json" in a.text.lower():
                print(f"Found link: {a['href']}")
                return a.get('href')
        
        return None
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def download_file(url):
    """
    Downloads a file from a URL or reads from local path
    
    Args:
        url (str): URL or local path of the file to download
        
    Returns:
        bytes: The downloaded file data
    """
    if LOCAL:
        print("Using local file for testing.")
        with open(url, "rb") as f:
            file_bytes = f.read()
        print(f"Read {len(file_bytes)/(1024*1024):.2f} MB from local file")
    else:
        print("Downloading the file...")
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise exception for HTTP errors
            file_bytes = response.content
            print(f"Downloaded {len(file_bytes)/(1024*1024):.2f} MB")
        except Exception as e:
            print(f"Failed to download the file: {e}")
            return None
    
    return file_bytes

def split_into_chunks(file_data, chunk_size, dir_name="temp_chunks"):
    """
    Split the file data into chunks of specified size
    
    Args:
        file_data (bytes): The file data to chunk
        chunk_size (int): Size of each chunk in bytes
        dir_name (str): Directory to save chunks to
        
    Returns:
        list: List of file paths to the chunk files
    """
    chunk_files = []
    
    # Create a directory for chunks if it doesn't exist
    os.makedirs(dir_name, exist_ok=True)
    
    # Split the data into chunks and save each chunk
    for i in range(0, len(file_data), chunk_size):
        chunk = file_data[i:i + chunk_size]
        chunk_file_path = os.path.join(dir_name, f"chunk_{i//chunk_size}.part")
        with open(chunk_file_path, "wb") as f:
            f.write(chunk)
        chunk_files.append(chunk_file_path)
        print(f"Created chunk {i//chunk_size} with size {len(chunk)/(1024*1024):.2f} MB")
    
    return chunk_files

def decompress_gzip(input_file, output_filename):
    """
    Decompress a Gzip file to the specified output file
    
    Args:
        input_file (str): Path to the Gzip file
        output_filename (str): Name of the file to save the decompressed data
        
    Returns:
        bool: True if successful, False otherwise
    """
    print(f"Decompressing file {input_file}...")
    start_time = time.time()
    
    try:
        with gzip.open(input_file, "rb") as gz_file:
            with open(output_filename, "wb") as out_file:
                shutil.copyfileobj(gz_file, out_file)
                
        elapsed_time = time.time() - start_time
        print(f"Decompression completed in {elapsed_time:.2f} seconds")
        print(f"Decompressed data saved to {output_filename}")
        return True
        
    except Exception as e:
        print(f"Decompression failed: {e}")
        return False

def reconstruct_and_decompress(chunk_files, output_filename):
    """
    Reconstruct a file from chunks and decompress it
    
    Args:
        chunk_files (list): List of paths to chunk files
        output_filename (str): Name of the file to save the decompressed data
        
    Returns:
        bool: True if successful, False otherwise
    """
    print("Reconstructing from chunks and decompressing...")
    start_time = time.time()
    
    # Create a temp file containing all chunks combined
    temp_combined = "temp_combined.gz"
    try:
        with open(temp_combined, "wb") as combined:
            for chunk_file in chunk_files:
                with open(chunk_file, "rb") as f:
                    combined.write(f.read())
        
        # Decompress the combined file
        success = decompress_gzip(temp_combined, output_filename)
        
        # Clean up temp file
        if os.path.exists(temp_combined):
            os.remove(temp_combined)
            
        if success:
            elapsed_time = time.time() - start_time
            print(f"Reconstruction and decompression completed in {elapsed_time:.2f} seconds")
        
        return success
        
    except Exception as e:
        print(f"Reconstruction and decompression failed: {e}")
        if os.path.exists(temp_combined):
            os.remove(temp_combined)
        return False

def process_kickstarter_data(url=None, output_filename="decompressed.json", 
                             download=True, split=True, decompress=True, 
                             keep_chunks=True, chunk_size=CHUNK_SIZE):
    """
    Process Kickstarter data through download, splitting, and decompression
    
    Args:
        url (str): URL or local path of the file
        output_filename (str): Name of the file to save the decompressed data
        download (bool): Whether to download the file
        split (bool): Whether to split the file into chunks
        decompress (bool): Whether to decompress the file
        keep_chunks (bool): Whether to keep the chunks after processing
        chunk_size (int): Size of chunks in bytes
        
    Returns:
        dict: Dictionary with paths to the processed files
    """
    result = {"chunks": [], "decompressed": None}
    
    # Determine source URL
    if url is None:
        if LOCAL:
            url = LOCAL_FILE
        else:
            url = get_kickstarter_download_link()
            if not url:
                print("Failed to get download link")
                return result
    
    # STEP 1: Download
    file_bytes = None
    if download:
        file_bytes = download_file(url)
        if file_bytes is None:
            return result
    
    # STEP 2: Split into chunks
    chunk_dir = "gzip_chunks" if keep_chunks else "temp_chunks"
    chunk_files = []
    
    if split and file_bytes is not None:
        print(f"Breaking file into chunks of {chunk_size/(1024*1024):.2f} MB...")
        chunk_files = split_into_chunks(file_bytes, chunk_size, dir_name=chunk_dir)
        result["chunks"] = chunk_files
    
    # If we're not keeping chunks and not decompressing, we're done
    if not decompress:
        if not keep_chunks and os.path.exists("temp_chunks"):
            shutil.rmtree("temp_chunks")
        return result
    
    # STEP 3: Decompress
    decompression_success = False
    
    # Try direct decompression first if we have a URL
    if url and os.path.exists(url):
        decompression_success = decompress_gzip(url, output_filename)
    
    # If direct decompression failed or we only have chunks, try reconstruction
    if not decompression_success and chunk_files:
        decompression_success = reconstruct_and_decompress(chunk_files, output_filename)
    
    # Clean up chunks if not keeping them
    if not keep_chunks:
        for chunk_file in chunk_files:
            if os.path.exists(chunk_file):
                os.remove(chunk_file)
        if os.path.exists("temp_chunks") and not os.listdir("temp_chunks"):
            shutil.rmtree("temp_chunks")
    
    if decompression_success:
        result["decompressed"] = output_filename
    
    return result

if __name__ == "__main__":
    # Example 1: Full process - download, split, and decompress
    result = process_kickstarter_data(output_filename="decompressed.json", keep_chunks=True)
    
    # Example 2: Only download and split, no decompression
    # result = process_kickstarter_data(decompress=False, keep_chunks=True)
    
    # Example 3: Only decompress from existing chunks
    # chunk_files = [f"gzip_chunks/chunk_{i}.part" for i in range(10)]  # Adjust range based on your chunks
    # result = process_kickstarter_data(download=False, split=False, decompress=True)

    print("Processing completed!")
    if result["decompressed"]:
        print(f"Decompressed file: {result['decompressed']}")
    if result["chunks"]:
        print(f"Chunks: {len(result['chunks'])} files in {os.path.dirname(result['chunks'][0])}")