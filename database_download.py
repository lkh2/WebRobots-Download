import requests
from bs4 import BeautifulSoup
import gzip
import time
import os
import shutil
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import json

LOCAL = True
LOCAL_FILE = "C:/Users/leeka/Downloads/Kickstarter_2025-03-12T07_34_02_656Z.json.gz"
CHUNK_SIZE = 90 * 1024 * 1024  # 90MB chunks
HALVED = False  # Set to True to process only half of the entries

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

def json_to_parquet(json_file, parquet_file, halved=False):
    """
    Convert a JSON file to Parquet format, handling nested data structure
    
    Args:
        json_file (str): Path to the JSON file
        parquet_file (str): Path to save the Parquet file
        halved (bool): If True, only process half of the entries
        
    Returns:
        bool: True if successful, False otherwise
    """
    print(f"Converting {json_file} to Parquet format..." + (" (halved dataset)" if halved else ""))
    start_time = time.time()
    
    try:
        # Read the JSON file line by line to handle the nested structure
        records = []
        line_count = 0
        total_lines = 0
        
        # First pass to count lines if halved=True
        if halved:
            with open(json_file, 'r', encoding='utf-8', errors='replace') as f:
                total_lines = sum(1 for _ in f)
            target_lines = total_lines // 2
            print(f"Total JSON lines: {total_lines}, processing approximately {target_lines} lines")
        
        # Second pass to read data
        with open(json_file, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                line_count += 1
                
                # Skip if we've processed half the lines and halved flag is True
                if halved and line_count > total_lines // 2:
                    break
                    
                record = json.loads(line)
                if 'data' in record:
                    if 'run_id' in record:
                        record['data']['run_id'] = record['run_id']
                    records.append(record['data'])
        
        print(f"Processed {line_count} lines into {len(records)} records")
        df = pd.DataFrame(records)
        
        # Flatten nested fields and ensure they match expected names in streamlit_app.py
        for nested_field in ['category', 'creator', 'location', 'urls']:
            if nested_field in df.columns:
                sample_records = df[nested_field].dropna().head(10)
                for record in sample_records:
                    if isinstance(record, dict):
                        for key, value in record.items():
                            col_name = f'{nested_field}.{key}'
                            if col_name not in df.columns:
                                df[col_name] = df[nested_field].apply(
                                    lambda x: x.get(key) if isinstance(x, dict) else None
                                )
                            if isinstance(value, dict):
                                for subkey, subvalue in value.items():
                                    subcol_name = f'{nested_field}.{key}.{subkey}'
                                    if subcol_name not in df.columns:
                                        df[subcol_name] = df[nested_field].apply(
                                            lambda x: x.get(key, {}).get(subkey) 
                                            if isinstance(x, dict) and isinstance(x.get(key), dict)
                                            else None
                                        )
        
        # Write to Parquet
        df.to_parquet(parquet_file, engine='pyarrow', compression=None)
        
        elapsed_time = time.time() - start_time
        file_size_mb = os.path.getsize(parquet_file) / (1024 * 1024)
        print(f"Conversion completed in {elapsed_time:.2f} seconds")
        print(f"Parquet file saved to {parquet_file} ({file_size_mb:.2f} MB)")
        
        # Print sample of columns saved for verification
        print("Sample columns saved:", ", ".join(list(df.columns)[:10]))
        return True
        
    except Exception as e:
        print(f"JSON to Parquet conversion failed: {e}")
        return False

def compress_file(input_file, output_file):
    """
    Compress a file using gzip
    
    Args:
        input_file (str): Path to the input file
        output_file (str): Path to save the compressed file
        
    Returns:
        bool: True if successful, False otherwise
    """
    print(f"Compressing {input_file} to {output_file}...")
    start_time = time.time()
    
    try:
        with open(input_file, 'rb') as f_in:
            with gzip.open(output_file, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        elapsed_time = time.time() - start_time
        file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
        print(f"Compression completed in {elapsed_time:.2f} seconds")
        print(f"Compressed file saved to {output_file} ({file_size_mb:.2f} MB)")
        return True
        
    except Exception as e:
        print(f"Compression failed: {e}")
        return False

def split_file_into_chunks(file_path, chunk_size, dir_name):
    """
    Split a file into chunks of specified size
    
    Args:
        file_path (str): Path to the file to chunk
        chunk_size (int): Size of each chunk in bytes
        dir_name (str): Directory to save chunks to
        
    Returns:
        list: List of file paths to the chunk files
    """
    chunk_files = []
    
    # Create a directory for chunks if it doesn't exist
    os.makedirs(dir_name, exist_ok=True)
    
    try:
        file_size = os.path.getsize(file_path)
        
        # Split the file into chunks and save each chunk
        with open(file_path, "rb") as f:
            for i in range(0, file_size, chunk_size):
                chunk_file_path = os.path.join(dir_name, f"chunk_{i//chunk_size}.part")
                with open(chunk_file_path, "wb") as chunk_file:
                    chunk_data = f.read(chunk_size)
                    chunk_file.write(chunk_data)
                chunk_files.append(chunk_file_path)
                print(f"Created chunk {i//chunk_size} with size {len(chunk_data)/(1024*1024):.2f} MB")
        
        return chunk_files
        
    except Exception as e:
        print(f"Splitting file into chunks failed: {e}")
        return chunk_files

def process_kickstarter_data(url=None, output_filename="decompressed.json", 
                             download=True, split=True, decompress=True, 
                             keep_chunks=True, chunk_size=CHUNK_SIZE,
                             convert_to_parquet=True, compress_parquet=True, 
                             split_parquet=True, keep_parquet_chunks=True,
                             halved=False):
    """
    Process Kickstarter data through download, splitting, decompression, and conversion
    
    Args:
        url (str): URL or local path of the file
        output_filename (str): Name of the file to save the decompressed data
        download (bool): Whether to download the file
        split (bool): Whether to split the file into chunks
        decompress (bool): Whether to decompress the file
        keep_chunks (bool): Whether to keep the chunks after processing
        chunk_size (int): Size of chunks in bytes
        convert_to_parquet (bool): Whether to convert JSON to Parquet
        compress_parquet (bool): Whether to compress Parquet file
        split_parquet (bool): Whether to split compressed Parquet file
        keep_parquet_chunks (bool): Whether to keep Parquet chunks
        halved (bool): If True, only process half of the entries
        
    Returns:
        dict: Dictionary with paths to the processed files
    """
    result = {
        "chunks": [], 
        "decompressed": None,
        "parquet": None,
        "parquet_gz": None,
        "parquet_gz_chunks": []
    }
    
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
    else:
        return result  # Stop if decompression failed
    
    # STEP 4: Convert to Parquet
    parquet_file = output_filename.replace(".json", ".parquet")
    if convert_to_parquet and result["decompressed"]:
        conversion_success = json_to_parquet(output_filename, parquet_file, halved)
        if conversion_success:
            result["parquet"] = parquet_file
        else:
            return result  # Stop if conversion failed
    
    # STEP 5: Compress Parquet
    parquet_gz_file = parquet_file + ".gz"
    if compress_parquet and result["parquet"]:
        compression_success = compress_file(parquet_file, parquet_gz_file)
        if compression_success:
            result["parquet_gz"] = parquet_gz_file
        else:
            return result  # Stop if compression failed
    
    # STEP 6: Split Parquet GZ
    if split_parquet and result["parquet_gz"]:
        parquet_chunk_dir = "parquet_gz_chunks"
        parquet_chunk_files = split_file_into_chunks(parquet_gz_file, chunk_size, parquet_chunk_dir)
        result["parquet_gz_chunks"] = parquet_chunk_files
    
    return result

if __name__ == "__main__":
    result = process_kickstarter_data(
        output_filename="decompressed.json", 
        keep_chunks=True,
        convert_to_parquet=True,
        compress_parquet=True,
        split_parquet=True,
        keep_parquet_chunks=True,
        halved=HALVED  # Pass the HALVED flag
    )
    
    print("Processing completed!")
    if result["decompressed"]:
        print(f"Decompressed file: {result['decompressed']}")
    if result["chunks"]:
        print(f"Chunks: {len(result['chunks'])} files in {os.path.dirname(result['chunks'][0])}")
    if result["parquet"]:
        print(f"Parquet file: {result['parquet']}")
        if HALVED:
            print("Note: Parquet file contains only half of the original entries.")
    if result["parquet_gz"]:
        print(f"Compressed Parquet file: {result['parquet_gz']}")
    if result["parquet_gz_chunks"]:
        print(f"Parquet chunks: {len(result['parquet_gz_chunks'])} files in {os.path.dirname(result['parquet_gz_chunks'][0])}")