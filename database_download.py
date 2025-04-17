import requests
from bs4 import BeautifulSoup
import gzip
import time
import os
import shutil
import pandas as pd
import json
import polars as pl 
import re
import datetime

LOCAL = True
LOCAL_FILE = "C:/Users/leeka/Downloads/Kickstarter/Kickstarter_2025-03-12T07_34_02_656Z.json.gz"
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

# Define safe column access for Polars
def safe_column_access(df, possible_names):
    """Try multiple possible column names and return the first one that exists"""
    for col in possible_names:
        if col in df.columns:
            return col
    print(f"Warning: Could not find any column matching: {possible_names}. Returning None.")
    return None

def save_filter_metadata(df_final: pl.DataFrame, output_json_path: str):
    """
    Calculates filter options (unique values, maps) and min/max ranges
    from the final DataFrame and saves them to a JSON file.

    Args:
        df_final (pl.DataFrame): The final, processed DataFrame.
        output_json_path (str): Path to save the JSON metadata file.
    """
    print(f"Calculating filter metadata for {output_json_path}...")
    metadata = {
        'categories': ['All Categories'],
        'countries': ['All Countries'],
        'states': ['All States'],
        'subcategories': ['All Subcategories'], # For the 'All Categories' selection
        'category_subcategory_map': {'All Categories': ['All Subcategories']},
        'min_max_values': {
            'pledged': {'min': 0, 'max': 1000},
            'goal': {'min': 0, 'max': 10000},
            'raised': {'min': 0, 'max': 500}
        },
        'date_ranges': [ # Keep date ranges static for now
             'All Time', 'Last Month', 'Last 6 Months', 'Last Year',
             'Last 5 Years', 'Last 10 Years'
        ]
    }

    try:
        # Ensure df_final is not empty before proceeding
        if df_final.is_empty():
             print("Warning: Input DataFrame for metadata calculation is empty. Saving default metadata.")
             # Save default metadata and return
             with open(output_json_path, 'w', encoding='utf-8') as f:
                  json.dump(metadata, f, ensure_ascii=False, indent=4)
             print(f"Default filter metadata saved to {output_json_path}")
             return # Stop processing if df is empty


        schema = df_final.schema

        # --- Categories and Subcategories ---
        all_subcategories_set = set()
        if 'Category' in schema:
            # No .collect() needed here
            categories_unique_series = df_final.select(pl.col('Category')).unique().get_column('Category')
            valid_categories = sorted(categories_unique_series.filter(categories_unique_series.is_not_null() & (categories_unique_series != "N/A")).to_list())
            metadata['categories'] += valid_categories
            for cat in valid_categories:
                metadata['category_subcategory_map'][cat] = [] # Initialize empty list

            if 'Subcategory' in schema:
                # No .collect() needed here
                cat_subcat_pairs = df_final.select(['Category', 'Subcategory']).unique().drop_nulls()
                for row in cat_subcat_pairs.iter_rows(named=True):
                    category = row['Category']
                    subcategory = row['Subcategory']
                    if category and subcategory and category != "N/A" and subcategory != "N/A":
                        if category in metadata['category_subcategory_map']:
                             metadata['category_subcategory_map'][category].append(subcategory)
                        all_subcategories_set.add(subcategory)
            # Sort subcategories within each category
            for cat in metadata['category_subcategory_map']:
                 subcats = metadata['category_subcategory_map'][cat]
                 prefix = []
                 rest = []
                 if 'All Subcategories' in subcats: # Should not happen here, but safety check
                     prefix = ['All Subcategories']
                     rest = sorted([s for s in subcats if s != 'All Subcategories'])
                 else:
                      rest = sorted(subcats)
                 metadata['category_subcategory_map'][cat] = prefix + rest # Ensure prefix is added if needed later

        # Get all unique subcategories for the 'All Categories' dropdown
        if 'Subcategory' in schema:
             if not all_subcategories_set: # If map wasn't built (e.g., no Category col)
                 # No .collect() needed here
                 subcategories_unique_series = df_final.select(pl.col('Subcategory')).unique().get_column('Subcategory')
                 all_subcategories_set.update(subcategories_unique_series.filter(subcategories_unique_series.is_not_null() & (subcategories_unique_series != "N/A")).to_list())

             metadata['subcategories'] += sorted(list(all_subcategories_set))
             metadata['category_subcategory_map']['All Categories'] += sorted(list(all_subcategories_set))


        # --- Countries ---
        if 'Country' in schema:
            # No .collect() needed here
            countries_unique_series = df_final.select(pl.col('Country')).unique().get_column('Country')
            metadata['countries'] += sorted(countries_unique_series.filter(countries_unique_series.is_not_null() & (countries_unique_series != "N/A")).to_list())

        # --- States ---
        # Extract plain state names if they are in HTML format
        if 'State' in schema and schema['State'] == pl.Utf8:
            # Check if the state column contains the HTML wrapper
            # Get the first state safely
            state_series = df_final.select('State').head(1).get_column('State')
            sample_state_result = None
            if len(state_series) > 0:
                 sample_state_result = state_series[0] # Access item using index

            if sample_state_result and isinstance(sample_state_result, str) and sample_state_result.startswith('<div class="state_cell state-'):
                 # Extract the state name between > and <
                 # No .collect() needed here
                 states_unique_series = df_final.select(
                     pl.col('State').str.extract(r'>([a-zA-Z]+)<', 1).alias("extracted_state")
                 ).unique().get_column('extracted_state')
                 plain_states = states_unique_series.filter(states_unique_series.is_not_null() & (states_unique_series != "")).to_list()
                 # Capitalize for display in the filter dropdown
                 metadata['states'] += sorted([s.capitalize() for s in plain_states if s.lower() != 'unknown'])
            else:
                 # Assume plain state names if no HTML detected
                 # No .collect() needed here
                 states_unique_series = df_final.select(pl.col('State')).unique().get_column('State')
                 plain_states = states_unique_series.filter(states_unique_series.is_not_null() & (states_unique_series != "N/A")).to_list()
                 metadata['states'] += sorted([s.capitalize() for s in plain_states]) # Capitalize for consistency


        # --- Min/Max Values ---
        required_minmax_cols = ['Raw Pledged', 'Raw Goal', 'Raw Raised']
        if all(col in schema for col in required_minmax_cols):
            try:
                # Aggregations like min/max on an eager DataFrame return a new DataFrame, no .collect()
                min_max_df = df_final.select([
                    pl.min('Raw Pledged').alias('min_pledged'),
                    pl.max('Raw Pledged').alias('max_pledged'),
                    pl.min('Raw Goal').alias('min_goal'),
                    pl.max('Raw Goal').alias('max_goal'),
                    pl.min('Raw Raised').alias('min_raised'),
                    pl.max('Raw Raised').alias('max_raised') # Use Raw Raised directly
                ])

                # Access values using .item() or indexing [0]
                metadata['min_max_values']['pledged']['min'] = int(min_max_df.item(0, 'min_pledged')) if min_max_df.item(0, 'min_pledged') is not None else 0
                metadata['min_max_values']['pledged']['max'] = int(min_max_df.item(0, 'max_pledged')) if min_max_df.item(0, 'max_pledged') is not None else 1000
                metadata['min_max_values']['goal']['min'] = int(min_max_df.item(0, 'min_goal')) if min_max_df.item(0, 'min_goal') is not None else 0
                metadata['min_max_values']['goal']['max'] = int(min_max_df.item(0, 'max_goal')) if min_max_df.item(0, 'max_goal') is not None else 10000
                metadata['min_max_values']['raised']['min'] = int(min_max_df.item(0, 'min_raised')) if min_max_df.item(0, 'min_raised') is not None else 0
                # Use calculated max_raised, ensure it's an int
                max_raised_calc_val = min_max_df.item(0, 'max_raised')
                metadata['min_max_values']['raised']['max'] = int(max_raised_calc_val) if max_raised_calc_val is not None else 500

                print("Min/max ranges calculated:", metadata['min_max_values'])
            except Exception as e:
                print(f"Warning: Error calculating min/max filter ranges: {e}. Using defaults.")
        else:
            print("Warning: Missing columns required for min/max filter ranges. Using defaults.")


        # --- Save to JSON ---
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4)
        print(f"Filter metadata saved successfully to {output_json_path}")

    except Exception as e:
        import traceback
        print(f"Error calculating or saving filter metadata: {e}")
        print(traceback.format_exc())

def extract_date_from_filename(filename):
    """
    Extracts the date (YYYY-MM-DD) from a filename matching the pattern
    'Kickstarter_YYYY-MM-DDTHH_MM_SS_MSZ...'
    """
    match = re.search(r'Kickstarter_(\d{4}-\d{2}-\d{2})T', os.path.basename(filename))
    if match:
        date_str = match.group(1)
        try:
            # Return a pandas Timestamp which Polars can handle
            return pd.to_datetime(date_str)
        except ValueError:
            print(f"Warning: Could not parse date '{date_str}' from filename '{filename}'.")
            return None
    else:
        print(f"Warning: Could not find date pattern 'Kickstarter_YYYY-MM-DDTHH' in filename '{filename}'.")
        return None

def json_to_parquet(json_file, parquet_file, halved=False):
    """
    Convert a JSON file to Parquet format, performing calculations, cleaning,
    deduplicating based on project URL, and applying specific filters.

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
        # --- Extract Dataset Creation Date ---
        dataset_creation_datetime = extract_date_from_filename(json_file)

        # Read the JSON file line by line
        records = []
        line_count = 0
        total_lines = 0

        if halved:
            with open(json_file, 'r', encoding='utf-8', errors='replace') as f:
                total_lines = sum(1 for _ in f)
            target_lines = total_lines // 2
            print(f"Total JSON lines: {total_lines}, processing approximately {target_lines} lines")

        with open(json_file, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                line_count += 1
                if halved and line_count > total_lines // 2:
                    break
                try:
                    record = json.loads(line)
                    if 'data' in record:
                        # Store run_id if present at the top level
                        if 'run_id' in record:
                            record['data']['run_id'] = record['run_id']
                        # Basic type checking for numeric fields
                        for key in ['goal', 'pledged', 'backers_count', 'converted_pledged_amount', 'state_changed_at', 'updated_at', 'run_id', 'created_at', 'deadline']: # Added date fields
                            if key in record['data'] and not isinstance(record['data'][key], (int, float)):
                                try:
                                    # Try casting to float first, then int if it fails or key is run_id/date
                                    if key in ['run_id', 'created_at', 'deadline', 'state_changed_at', 'updated_at']:
                                         record['data'][key] = int(record['data'][key]) if record['data'][key] is not None else 0
                                    else:
                                         record['data'][key] = float(record['data'][key]) if record['data'][key] is not None else 0.0
                                except (ValueError, TypeError):
                                     # Ensure 'state' gets a default if conversion fails, other numeric default to 0
                                     if key == 'state':
                                         record['data'][key] = "unknown" # Or another appropriate default
                                     else:
                                         record['data'][key] = 0
                        records.append(record['data'])
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON line: {line_count}")
                    continue
                except Exception as e:
                    print(f"Error processing line {line_count}: {e}")
                    continue


        print(f"Processed {line_count} lines into {len(records)} records")
        if not records:
             print("No valid records found. Aborting Parquet conversion.")
             return False

        # Convert list of dicts to Polars DataFrame directly
        df = pl.DataFrame(records)

        print("Initial Polars DataFrame created. Starting transformations...")

        # --- Start Polars Transformations ---

        # Flatten nested JSON structures needed for deduplication key (URL)
        # and potentially for sort key (though run_id and state_changed_at are usually top-level)
        url_flattened = False
        if 'urls' in df.columns:
             print("Flattening column: urls")
             # Ensure 'urls' is struct type before accessing fields
             if isinstance(df['urls'].dtype, pl.Struct):
                 # Check for urls.web.project path specifically
                 if 'web' in df['urls'].struct.fields and isinstance(df['urls'].struct.field('web').dtype, pl.Struct) and 'project' in df['urls'].struct.field('web').struct.fields:
                     df = df.with_columns(pl.col('urls').struct.field('web').struct.field('project').alias('urls.web.project'))
                     url_flattened = True
                 else:
                      print("Warning: Could not find 'urls.web.project' structure.")
             else:
                 print("Warning: 'urls' column is not a struct type, cannot flatten.")


        # --- Deduplication Step ---
        deduplication_key_col = 'urls.web.project'
        sort_key_options = ['state_changed_at', 'run_id', 'updated_at', 'created_at'] # Order of preference

        if not url_flattened or deduplication_key_col not in df.columns:
             print(f"Error: Deduplication key column '{deduplication_key_col}' not found or couldn't be flattened. Skipping deduplication.")
        else:
             # Find the best available sort key column
             sort_key_col = None
             for key in sort_key_options:
                 if key in df.columns:
                     sort_key_col = key
                     print(f"Using '{sort_key_col}' as the sort key for deduplication.")
                     break

             if sort_key_col:
                 # Ensure the sort key is numeric (Int or Float) or Datetime for proper sorting
                 # We'll assume timestamps are Unix epoch seconds (Int64) based on previous logic
                 # Handle potential nulls by filling them with a value that sorts last (0 or epoch start)
                 print(f"Preparing sort key '{sort_key_col}'...")
                 df = df.with_columns(
                     pl.col(sort_key_col).cast(pl.Int64, strict=False).fill_null(0).alias(sort_key_col)
                 )
                 # Also ensure the deduplication key (URL) is Utf8 and handle nulls
                 df = df.with_columns(
                     pl.col(deduplication_key_col).cast(pl.Utf8, strict=False).fill_null("__MISSING_URL__").alias(deduplication_key_col)
                 )

                 original_rows = df.height
                 print(f"Sorting by '{deduplication_key_col}' (asc) and '{sort_key_col}' (desc) for deduplication...")
                 # Sort by URL, then by the sort key descending (latest/highest first)
                 df = df.sort(deduplication_key_col, sort_key_col, descending=[False, True])

                 print(f"Applying unique constraint on '{deduplication_key_col}', keeping first (latest)...")
                 df = df.unique(subset=[deduplication_key_col], keep='first', maintain_order=True) # maintain_order=True is default but explicit
                 removed_rows = original_rows - df.height
                 print(f"Removed {removed_rows} duplicate rows based on project URL.")

             else:
                 print("Warning: No suitable sort key found among options. Skipping deduplication based on latest update.")
                 # Optionally, could deduplicate just by URL keeping *any* first instance:
                 # print(f"Applying unique constraint on '{deduplication_key_col}', keeping first encountered...")
                 # df = df.with_columns(pl.col(deduplication_key_col).cast(pl.Utf8, strict=False).fill_null("__MISSING_URL__").alias(deduplication_key_col))
                 # original_rows = df.height
                 # df = df.unique(subset=[deduplication_key_col], keep='first', maintain_order=True)
                 # removed_rows = original_rows - df.height
                 # print(f"Removed {removed_rows} duplicate rows based on project URL (kept first encountered).")


        # --- Continue with other transformations on the deduplicated DataFrame ---

        # Flatten remaining nested structures if needed for selection
        nested_structures = {
            'category': ['name', 'parent_name'],
            'creator': ['name'],
            'location': ['country', 'expanded_country'],
            # 'urls' was handled above for deduplication
        }
        for base_col, fields in nested_structures.items():
             if base_col in df.columns and isinstance(df[base_col].dtype, pl.Struct): # Check if it's a struct
                 print(f"Flattening column: {base_col}")
                 expressions = []
                 for field in fields:
                      # Check if field exists within the struct
                      if field in df[base_col].struct.fields:
                          new_col_name = f"{base_col}.{field}"
                          expressions.append(pl.col(base_col).struct.field(field).alias(new_col_name))
                      else:
                          print(f"Warning: Field '{field}' not found in struct '{base_col}'.")
                 if expressions:
                      df = df.with_columns(expressions)


        print("Nested structure flattening complete.")

        # Find the correct column names using safe_column_access
        goal_col = safe_column_access(df, ['goal'])
        exchange_rate_col = safe_column_access(df, ['usd_exchange_rate', 'static_usd_rate'])
        pledged_col = safe_column_access(df, ['converted_pledged_amount', 'pledged'])
        created_col = safe_column_access(df, ['created_at'])
        deadline_col = safe_column_access(df, ['deadline']) # Keep original name for processing
        backers_col = safe_column_access(df, ['backers_count'])
        name_col = safe_column_access(df, ['name'])
        creator_col = safe_column_access(df, ['creator.name'])
        link_col = safe_column_access(df, ['urls.web.project']) # Use the flattened URL column
        country_expanded_col = safe_column_access(df, ['location.expanded_country'])
        state_col = safe_column_access(df, ['state']) # Keep original name for processing
        category_col = safe_column_access(df, ['category.parent_name'])
        subcategory_col = safe_column_access(df, ['category.name'])
        country_code_col = safe_column_access(df, ['location.country'])
        # Add safe access for the displayable country name
        country_displayable_name_col = safe_column_access(df, ['country_displayable_name'])

        # Calculate and store raw values
        if goal_col and exchange_rate_col:
             df = df.with_columns([
                  pl.col(goal_col).cast(pl.Float64, strict=False).fill_null(0.0),
                  pl.col(exchange_rate_col).cast(pl.Float64, strict=False).fill_null(1.0)
             ])
             df = df.with_columns(
                  (pl.col(goal_col) * pl.col(exchange_rate_col)).alias('Raw Goal')
             )
             df = df.with_columns(
                  pl.when(pl.col('Raw Goal') < 1.0).then(1.0).otherwise(pl.col('Raw Goal')).alias('Raw Goal')
             )
        else:
             print("Warning: 'goal' or 'usd_exchange_rate'/'static_usd_rate' column not found. Setting 'Raw Goal' to 1.0.")
             df = df.with_columns(pl.lit(1.0).cast(pl.Float64).alias('Raw Goal'))

        if pledged_col:
             df = df.with_columns(
                  pl.col(pledged_col).cast(pl.Float64, strict=False).fill_null(0.0).alias('Raw Pledged')
             )
        else:
             print("Warning: 'converted_pledged_amount'/'pledged' column not found. Setting 'Raw Pledged' to 0.0.")
             df = df.with_columns(pl.lit(0.0).cast(pl.Float64).alias('Raw Pledged'))

        df = df.with_columns(
             pl.when((pl.col('Raw Pledged') <= 0.0) | (pl.col('Raw Goal') <= 0.0))
             .then(0.0)
             .otherwise((pl.col('Raw Pledged') / pl.col('Raw Goal')) * 100.0)
             .alias('Raw Raised')
             .cast(pl.Float64)
        )

        # Dates: Convert Unix timestamp
        default_date = pl.lit(pd.to_datetime('2000-01-01')).cast(pl.Datetime)
        if created_col:
            # Try converting assuming seconds epoch first, fallback for milliseconds if error
            try:
                 df = df.with_columns(
                      pl.from_epoch(pl.col(created_col).cast(pl.Int64, strict=True), time_unit="s")
                      .fill_null(default_date)
                      .alias('Raw Date')
                 )
            except pl.ComputeError:
                 print(f"Warning: Casting '{created_col}' to Int64 seconds epoch failed, trying milliseconds.")
                 try:
                      df = df.with_columns(
                           pl.from_epoch(pl.col(created_col).cast(pl.Int64, strict=True), time_unit="ms")
                           .fill_null(default_date)
                           .alias('Raw Date')
                      )
                 except Exception as e_ms:
                      print(f"Warning: Casting '{created_col}' to Int64 ms epoch also failed ({e_ms}). Setting default date.")
                      df = df.with_columns(default_date.alias('Raw Date'))
            except Exception as e_other:
                 print(f"Warning: Error casting '{created_col}' ({e_other}). Setting default date.")
                 df = df.with_columns(default_date.alias('Raw Date'))
        else:
            print("Warning: 'created_at' column not found. Setting 'Raw Date' to default.")
            df = df.with_columns(default_date.alias('Raw Date'))

        if deadline_col:
             try:
                  df = df.with_columns(
                       pl.from_epoch(pl.col(deadline_col).cast(pl.Int64, strict=True), time_unit="s")
                       .fill_null(default_date)
                       .alias('Raw Deadline')
                  )
             except pl.ComputeError:
                  print(f"Warning: Casting '{deadline_col}' to Int64 seconds epoch failed, trying milliseconds.")
                  try:
                       df = df.with_columns(
                           pl.from_epoch(pl.col(deadline_col).cast(pl.Int64, strict=True), time_unit="ms")
                           .fill_null(default_date)
                           .alias('Raw Deadline')
                       )
                  except Exception as e_ms:
                       print(f"Warning: Casting '{deadline_col}' to Int64 ms epoch also failed ({e_ms}). Setting default date.")
                       df = df.with_columns(default_date.alias('Raw Deadline'))
             except Exception as e_other:
                  print(f"Warning: Error casting '{deadline_col}' ({e_other}). Setting default date.")
                  df = df.with_columns(default_date.alias('Raw Deadline'))
        else:
            print("Warning: 'deadline' column not found. Setting 'Raw Deadline' to default.")
            df = df.with_columns(default_date.alias('Raw Deadline'))

        # --- Apply Deadline/State Filter ---
        # Filter rows where Raw Deadline is before the dataset creation date
        # AND the original 'state' (before aliasing/selection) is 'live', 'submitted', or 'started'.
        if dataset_creation_datetime:
            dataset_creation_date_pl = pl.lit(dataset_creation_datetime).cast(pl.Datetime)
            # Check if required columns exist for filtering
            if state_col and state_col in df.columns and 'Raw Deadline' in df.columns and df['Raw Deadline'].dtype == pl.Datetime:
                initial_rows = df.height
                print(f"Applying filter: Removing rows where Raw Deadline < {dataset_creation_datetime.date()} AND state is live/submitted/started...")

                # Ensure 'state' column is treated as string for comparison
                state_col_expr = pl.col(state_col).cast(pl.Utf8, strict=False).fill_null("").str.to_lowercase()

                # Condition to identify rows TO REMOVE
                remove_condition = (
                    (pl.col('Raw Deadline') < dataset_creation_date_pl) &
                    (state_col_expr.is_in(['live', 'submitted', 'started']))
                )

                # Filter to KEEP rows that DO NOT match the remove condition
                df = df.filter(~remove_condition)

                removed_count = initial_rows - df.height
                print(f"Removed {removed_count} rows based on deadline/state filter.")
            else:
                missing_cols_warn = []
                if not (state_col and state_col in df.columns):
                    missing_cols_warn.append(f"'{state_col}' (original state)")
                if 'Raw Deadline' not in df.columns or df['Raw Deadline'].dtype != pl.Datetime:
                     missing_cols_warn.append("'Raw Deadline' (Datetime)")
                print(f"Warning: Skipping deadline/state filter due to missing/incorrect columns: {', '.join(missing_cols_warn)}.")
        else:
            print("Warning: Skipping deadline/state filter because dataset creation date could not be determined from filename.")


        # Backer count
        if backers_col:
            df = df.with_columns(
                 pl.col(backers_col).cast(pl.Int64, strict=False).fill_null(0).alias('Backer Count')
            )
        else:
            print("Warning: 'backers_count' column not found. Setting 'Backer Count' to 0.")
            df = df.with_columns(pl.lit(0).cast(pl.Int64).alias('Backer Count'))


        # Format display columns
        df = df.with_columns(
            pl.col('Raw Goal').fill_null(0.0).round(2).alias('Goal'),
            pl.col('Raw Pledged').fill_null(0.0).round(2).alias('Pledged Amount'),
            pl.col('Raw Raised').fill_null(0.0).round(2).alias('%Raised'),
            pl.col('Raw Date').dt.strftime('%Y-%m-%d').fill_null('N/A').alias('Date'),
            pl.col('Raw Deadline').dt.strftime('%Y-%m-%d').fill_null('N/A').alias('Deadline')
        )

        # --- Select final columns ---
        # Use the Polars DataFrame `df` which has been filtered and deduplicated
        select_expressions = []
        # Add calculated display columns
        calculated_display_cols = ['Pledged Amount', 'Date', 'Deadline', 'Goal', '%Raised']
        for col in calculated_display_cols:
            if col in df.columns: select_expressions.append(pl.col(col))
            else: print(f"Error: Display column '{col}' missing.")

        # Add raw values needed for frontend logic/filters
        raw_value_cols = ['Raw Goal', 'Raw Pledged', 'Raw Raised', 'Raw Date', 'Raw Deadline', 'Backer Count']
        for col in raw_value_cols:
             if col in df.columns: select_expressions.append(pl.col(col))
             else: print(f"Error: Raw value column '{col}' missing.")

        # --- Build Special Columns: Country and Category ---

        # Country: Prioritize expanded > displayable > code. Clean "the ".
        country_expr = pl.lit("N/A").cast(pl.Utf8) # Default value
        # Build the expression conditionally based on column existence and content
        # Innermost fallback: location.country (country_code_col)
        if country_code_col and country_code_col in df.columns:
             country_expr = pl.when((pl.col(country_code_col).is_not_null()) & (pl.col(country_code_col) != "N/A")) \
                              .then(pl.col(country_code_col)) \
                              .otherwise(country_expr)
        # Middle fallback: country_displayable_name
        if country_displayable_name_col and country_displayable_name_col in df.columns:
             country_expr = pl.when((pl.col(country_displayable_name_col).is_not_null()) & (pl.col(country_displayable_name_col) != "N/A")) \
                              .then(pl.col(country_displayable_name_col)) \
                              .otherwise(country_expr) # Fallback to location.country or "N/A"
        # Highest priority: location.expanded_country
        if country_expanded_col and country_expanded_col in df.columns:
             country_expr = pl.when((pl.col(country_expanded_col).is_not_null()) & (pl.col(country_expanded_col) != "N/A")) \
                              .then(pl.col(country_expanded_col)) \
                              .otherwise(country_expr) # Fallback to displayable_name or location.country or "N/A"

        # Apply string replacements and alias
        select_expressions.append(
             country_expr.fill_null("N/A")
                         .str.replace("the United States", "United States", literal=True)
                         .str.replace("the United Kingdom", "United Kingdom", literal=True)
                         .alias('Country')
        )

        # Category: Prioritize parent_name > name.
        category_expr = pl.lit("N/A").cast(pl.Utf8) # Default value
        # Build the expression conditionally
        # Fallback: category.name (subcategory_col)
        if subcategory_col and subcategory_col in df.columns:
            category_expr = pl.when((pl.col(subcategory_col).is_not_null()) & (pl.col(subcategory_col) != "N/A")) \
                             .then(pl.col(subcategory_col)) \
                             .otherwise(category_expr)
        # Primary: category.parent_name (category_col)
        if category_col and category_col in df.columns:
            category_expr = pl.when((pl.col(category_col).is_not_null()) & (pl.col(category_col) != "N/A")) \
                             .then(pl.col(category_col)) \
                             .otherwise(category_expr) # Fallback to category.name or "N/A"

        # Add the expression, ensuring nulls are handled and aliased
        select_expressions.append(
             category_expr.fill_null("N/A").alias('Category')
        )


        # Add other required columns (potentially flattened)
        # Remove Country and Category as they are handled above
        other_required_cols = {
            'Project Name': name_col,
            'Creator': creator_col,
            'Link': link_col, # Use the flattened URL column name used for deduplication
            # 'Country': country_expanded_col, # Handled above
            'State': state_col, # Use original state col name, will be selected/aliased
            # 'Category': category_col, # Handled above
            'Subcategory': subcategory_col,
            'Country Code': country_code_col,
        }
        for final_name, source_col_name in other_required_cols.items():
            if source_col_name and source_col_name in df.columns:
                # Select and alias if necessary, ensure Utf8 and fill nulls
                select_expressions.append(
                    pl.col(source_col_name).cast(pl.Utf8, strict=False).fill_null("N/A").alias(final_name)
                )
            else:
                print(f"Warning: Source column '{source_col_name}' for '{final_name}' not found. Creating default 'N/A'.")
                select_expressions.append(pl.lit("N/A").cast(pl.Utf8).alias(final_name))

        # Select the final set of columns
        df_final = df.select(select_expressions)

        print("Columns in df_final before popularity score:", df_final.columns)

        # --- Calculate Popularity Score on the final DataFrame ---
        now_dt_expr = pl.lit(pd.Timestamp.now()).cast(pl.Datetime) # Keep as expression
        if 'Raw Date' not in df_final.columns or df_final['Raw Date'].dtype != pl.Datetime:
             print("Error: 'Raw Date' column missing/wrong type for popularity score. Setting time_factor to 0.")
             df_final = df_final.with_columns(pl.lit(0.0).alias('time_factor'))
        else:
             # Define expression for days difference
             days_diff_expr = (now_dt_expr - pl.col('Raw Date')).dt.total_days()

             # Compute the maximum positive days difference *eagerly*
             # Filter out rows where calculation might fail or be invalid before computing max
             try:
                 valid_days_diff = df_final.select(days_diff_expr.alias('diff')).filter(pl.col('diff').is_not_nan() & pl.col('diff').is_not_null() & (pl.col('diff') > 0))
                 if not valid_days_diff.is_empty():
                     computed_max_days = valid_days_diff.select(pl.max('diff')).item()
                 else:
                      computed_max_days = None
             except Exception as e:
                  print(f"Warning: Error computing max days difference: {e}. Setting time factor to 0.")
                  computed_max_days = None


             # Now use the computed scalar value in the Python if statement
             if computed_max_days is None or computed_max_days <= 0:
                  print("Warning: Could not determine a valid positive max_days for time factor. Setting time_factor to 0.")
                  time_factor_expr = pl.lit(0.0)
             else:
                 # Use the computed_max_days scalar in the expression for normalization
                 # This expression will be applied row-wise by with_columns
                 time_factor_expr = (1.0 - (days_diff_expr / computed_max_days)).clip(0.0, 1.0) # Ensure it's between 0 and 1

             # Add the resulting time_factor column using the appropriate expression
             df_final = df_final.with_columns(time_factor_expr.fill_null(0.0).alias('time_factor'))


        required_norm_cols_final = ['Backer Count', 'Raw Pledged', 'Raw Raised']
        missing_norm_cols_final = [col for col in required_norm_cols_final if col not in df_final.columns]
        if missing_norm_cols_final:
            print(f"Error: Missing columns for normalization in final DF: {missing_norm_cols_final}. Pop score inaccurate.")
            df_final = df_final.with_columns(pl.lit(0.0).alias('Popularity Score')) # Add default score col
        else:
            # --- Popularity Score Calculations (using df_final) ---
            # Capped Percentage Expression (can be defined once)
            capped_percentage_expr = pl.col('Raw Raised').clip(upper_bound=500.0).fill_null(0.0).alias('capped_percentage')

            # Function for normalization (safer inside the scope)
            def normalize_col_final_safe(df_context, col_name):
                 if col_name not in df_context.columns:
                      print(f"Warning: Column '{col_name}' not found for normalization.")
                      return pl.lit(0.0)

                 # Eagerly compute min/max on the context DataFrame
                 min_max_df = df_context.select(
                     pl.min(col_name).alias('min_val'),
                     pl.max(col_name).alias('max_val')
                 )
                 min_val = min_max_df.item(0, 'min_val')
                 max_val = min_max_df.item(0, 'max_val')

                 if min_val is None or max_val is None:
                      print(f"Warning: Min/max for '{col_name}' could not be determined. Normalization returns 0.")
                      return pl.lit(0.0)

                 range_val = max_val - min_val
                 if range_val == 0:
                      return pl.lit(0.0) # Avoid division by zero
                 else:
                      # Return the expression for normalization
                      return ((pl.col(col_name) - min_val) / range_val).fill_null(0.0)

            # Apply normalization expressions
            df_final = df_final.with_columns([
                 normalize_col_final_safe(df_final, 'Backer Count').alias('normalized_backers'),
                 normalize_col_final_safe(df_final, 'Raw Pledged').alias('normalized_pledged'),
                 capped_percentage_expr # Add the capped percentage column
            ])

            # Normalize the capped percentage separately
            # Need to compute its min/max *after* it's added
            normalized_percentage_expr = normalize_col_final_safe(df_final, 'capped_percentage')


            # Calculate Popularity Score
            pop_score_components = ['normalized_backers', 'normalized_pledged', 'time_factor']
            missing_pop_components = [c for c in pop_score_components if c not in df_final.columns]

            if missing_pop_components:
                 print(f"Error: Missing pop score components in final DF: {missing_pop_components}. Setting score to 0.")
                 df_final = df_final.with_columns(pl.lit(0.0).alias('Popularity Score'))
            else:
                 df_final = df_final.with_columns(
                     (
                          pl.col('normalized_backers') * 0.2778 +
                          pl.col('normalized_pledged') * 0.3889 +
                          normalized_percentage_expr * 0.2222 + # Use the normalized capped percentage expression
                          pl.col('time_factor') * 0.1111
                     ).alias('Popularity Score').cast(pl.Float64).fill_null(0.0)
                 )

            # Clean up temporary normalization columns
            cols_to_drop = ['normalized_backers', 'normalized_pledged', 'time_factor', 'capped_percentage']
            existing_cols_to_drop = [col for col in cols_to_drop if col in df_final.columns]
            if existing_cols_to_drop:
                 df_final = df_final.drop(existing_cols_to_drop)


        print("Polars transformations complete.")
        # --- End Polars Transformations ---

        # Write final Polars DataFrame to Parquet
        # Ensure the DataFrame is not empty before writing
        if df_final.height == 0:
            print("Warning: Final DataFrame is empty. Skipping Parquet write.")
            return False

        df_final.write_parquet(parquet_file, compression='snappy', use_pyarrow=True)

        elapsed_time = time.time() - start_time
        file_size_mb = os.path.getsize(parquet_file) / (1024 * 1024)
        print(f"Conversion completed in {elapsed_time:.2f} seconds")
        print(f"Parquet file saved to {parquet_file} ({file_size_mb:.2f} MB)")

        print("Final columns saved:", ", ".join(list(df_final.columns)))
        # print("Sample data head:\n", df_final.head(3)) # Limit output size

        # --- Calculate and save filter metadata ---
        filter_metadata_file = "filter_metadata.json"
        save_filter_metadata(df_final, filter_metadata_file)
        # We return True indicating parquet success, metadata saving is best-effort
        return True

    except Exception as e:
        import traceback
        print(f"JSON to Parquet conversion failed: {e}")
        print(traceback.format_exc()) # Print full traceback
        return False

def extract_base_name(file_path):
    """
    Extracts the base name from a file path without any extensions.
    
    Args:
        file_path (str): Path to the file
        
    Returns:
        str: Base name without extensions
    """
    # Extract just the filename without directory
    file_name = os.path.basename(file_path)
    
    # Remove all extensions (.json.gz -> base_name)
    base_name = file_name.split('.')[0]
    
    return base_name

def process_kickstarter_data(url=None, output_filename=None,
                             download=True, split=True, decompress=True,
                             keep_chunks=True, chunk_size=CHUNK_SIZE,
                             convert_to_parquet=True,
                             halved=False):
    """
    Process Kickstarter data through download, splitting, decompression, and conversion to Parquet.
    Also generates a metadata file for frontend filters.

    Args:
        url (str): URL or local path of the file
        output_filename (str): Name of the file to save the decompressed JSON data
        download (bool): Whether to download the file
        split (bool): Whether to split the downloaded file into chunks
        decompress (bool): Whether to decompress the file (from chunks or directly)
        keep_chunks (bool): Whether to keep the downloaded chunks after processing
        chunk_size (int): Size of chunks in bytes for download splitting
        convert_to_parquet (bool): Whether to convert JSON to Parquet
        halved (bool): If True, only process half of the entries

    Returns:
        dict: Dictionary with paths to the processed files (chunks, decompressed JSON, Parquet, filter metadata)
    """
    result = {
        "chunks": [],
        "decompressed": None,
        "parquet": None,
        "filter_metadata": None # Added key for metadata file
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
    
    # Extract base name from URL/file for consistent naming
    base_name = extract_base_name(url)
    
    # Set output filenames with proper extensions
    if output_filename is None:
        output_filename = f"{base_name}.json"
    
    # Make sure output_filename has .json extension
    if not output_filename.lower().endswith('.json'):
        output_filename = f"{os.path.splitext(output_filename)[0]}.json"
    
    # Define parquet filename based on the same base name
    parquet_file = f"{base_name}.parquet"
    
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
    filter_metadata_file = "filter_metadata.json" # Keep this name as specified
    if convert_to_parquet and result["decompressed"]:
        conversion_success = json_to_parquet(output_filename, parquet_file, halved)
        if conversion_success:
            result["parquet"] = parquet_file
            # Check if metadata file was created
            if os.path.exists(filter_metadata_file):
                 result["filter_metadata"] = filter_metadata_file
            else:
                 print(f"Warning: Parquet conversion succeeded, but filter metadata file '{filter_metadata_file}' was not found.")
        else:
            # If conversion fails, keep the decompressed JSON if it exists
            if not os.path.exists(output_filename):
                 result["decompressed"] = None
            return result # Stop if conversion failed

    return result

if __name__ == "__main__":
    result = process_kickstarter_data(
        output_filename=None,  # Let the function determine the filename based on input
        keep_chunks=True,      # Keep downloaded gzip chunks
        convert_to_parquet=True,
        halved=HALVED          # Pass the HALVED flag
    )

    print("\nProcessing completed!")
    if result["decompressed"]:
        print(f"Intermediate decompressed JSON file: {result['decompressed']}")
    if result["chunks"]:
        print(f"Downloaded Gzip Chunks: {len(result['chunks'])} files in {os.path.dirname(result['chunks'][0])}")
    if result["parquet"]:
        print(f"Final Parquet file: {result['parquet']}")
        if HALVED:
            print("Note: Parquet file contains only half of the original entries.")
    if result["filter_metadata"]:
        print(f"Filter metadata file: {result['filter_metadata']}")
    else:
        print("Filter metadata file was not generated.")