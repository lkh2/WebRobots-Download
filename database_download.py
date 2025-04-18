import gzip
import time
import os
import pandas as pd
import numpy as np
import json
import re
import datetime
from pathlib import Path
import country_converter as coco
from pandas.errors import ParserError

# Directory containing the Kickstarter JSON.gz files
KICKSTARTER_DATA_DIR = Path("C:/Users/leeka/Downloads/Kickstarter")
# Output filename for filter metadata
FILTER_METADATA_FILENAME = "filter_metadata.json"
# Progress reminder frequency
PROGRESS_REMINDER_INTERVAL = 50000

def parse_date_from_filename(filepath: Path):
    """
    Extracts the date (YYYY-MM-DD) from a filename matching the pattern
    'Kickstarter_YYYY-MM-DDTHH_MM_SS_MSZ...'.
    Returns a datetime.date object or None if parsing fails.
    """
    match = re.search(r'Kickstarter_(\d{4}-\d{2}-\d{2})T', filepath.name)
    if match:
        date_str = match.group(1)
        try:
            return pd.to_datetime(date_str).date()
        except (ValueError, ParserError):
            print(f"Warning: Could not parse date '{date_str}' from filename '{filepath.name}'.")
    return None

def find_kickstarter_files(directory: Path):
    """
    Finds all Kickstarter*.json.gz files in the directory,
    parses their dates, and returns them sorted by date descending (latest first).
    """
    files_with_dates = []
    print(f"Scanning directory: {directory}")
    for filepath in directory.glob("Kickstarter*.json.gz"):
        file_date = parse_date_from_filename(filepath)
        if file_date:
            files_with_dates.append((filepath, file_date))
        else:
            print(f"Skipping file due to missing/invalid date in name: {filepath.name}")

    if not files_with_dates:
        print("No valid Kickstarter*.json.gz files found.")
        return [], None

    files_with_dates.sort(key=lambda item: item[1], reverse=True)

    print(f"Found {len(files_with_dates)} files. Latest: {files_with_dates[0][0].name} ({files_with_dates[0][1]})")
    return files_with_dates, files_with_dates[0][1] 

def safe_column_access(df, possible_names):
    """Try multiple possible column names and return the first one that exists"""
    for col in possible_names:
        if col in df.columns:
            return col
    return None

def save_filter_metadata(df_final: pd.DataFrame, output_json_path: str):
    """
    Calculates filter options (unique values, maps) and min/max ranges
    from the final DataFrame and saves them to a JSON file using Pandas.
    """
    print(f"Calculating filter metadata for {output_json_path}...")
    metadata = {
        'categories': ['All Categories'],
        'countries': ['All Countries'],
        'states': ['All States'],
        'subcategories': ['All Subcategories'],
        'category_subcategory_map': {'All Categories': ['All Subcategories']},
        'min_max_values': {
            'pledged': {'min': 0, 'max': 1000},
            'goal': {'min': 0, 'max': 10000},
            'raised': {'min': 0, 'max': 500}
        },
        'date_ranges': [ 
             'All Time', 'Last Month', 'Last 6 Months', 'Last Year',
             'Last 5 Years', 'Last 10 Years'
        ]
    }

    try:
        if df_final.empty: 
             print("Warning: Input DataFrame for metadata calculation is empty. Saving default metadata.")
             with open(output_json_path, 'w', encoding='utf-8') as f:
                  json.dump(metadata, f, ensure_ascii=False, indent=4)
             print(f"Default filter metadata saved to {output_json_path}")
             return 

        # --- Categories and Subcategories ---
        all_subcategories_set = set()
        if 'Category' in df_final.columns:
            categories_unique = df_final['Category'].dropna().unique()
            valid_categories = sorted([cat for cat in categories_unique if cat != "N/A"]) 
            metadata['categories'] += valid_categories
            for cat in valid_categories:
                metadata['category_subcategory_map'][cat] = []

            if 'Subcategory' in df_final.columns:
                cat_subcat_pairs = df_final[['Category', 'Subcategory']].dropna().drop_duplicates() 
                for index, row in cat_subcat_pairs.iterrows():
                    category = row['Category']
                    subcategory = row['Subcategory']
                    if category and subcategory and category != "N/A" and subcategory != "N/A":
                        if category in metadata['category_subcategory_map']:
                             metadata['category_subcategory_map'][category].append(subcategory)
                        all_subcategories_set.add(subcategory)
            for cat in metadata['category_subcategory_map']:
                 subcats = metadata['category_subcategory_map'][cat]
                 prefix = []
                 rest = []
                 if 'All Subcategories' in subcats: 
                     prefix = ['All Subcategories']
                     rest = sorted([s for s in subcats if s != 'All Subcategories'])
                 else:
                      rest = sorted(subcats)
                 metadata['category_subcategory_map'][cat] = prefix + rest 

        if 'Subcategory' in df_final.columns:
             if not all_subcategories_set:
                 subcategories_unique = df_final['Subcategory'].dropna().unique()
                 all_subcategories_set.update([sub for sub in subcategories_unique if sub != "N/A"])

             metadata['subcategories'] += sorted(list(all_subcategories_set))
             metadata['category_subcategory_map']['All Categories'] += sorted(list(all_subcategories_set))

        # --- Countries ---
        if 'Country' in df_final.columns:
            countries_unique = df_final['Country'].dropna().unique()
            metadata['countries'] += sorted([c for c in countries_unique if c != "N/A"])

        # --- States ---
        if 'State' in df_final.columns and pd.api.types.is_string_dtype(df_final['State']): 
            sample_state_result = df_final['State'].dropna().iloc[0] if not df_final['State'].dropna().empty else None

            if sample_state_result and isinstance(sample_state_result, str) and sample_state_result.startswith('<div class="state_cell state-'):
                 states_extracted = df_final['State'].str.extract(r'>([a-zA-Z\s]+)<', expand=False).dropna().unique() 
                 plain_states = [s for s in states_extracted if s != ""]
                 metadata['states'] += sorted([s.strip().capitalize() for s in plain_states if s.lower().strip() != 'unknown'])
            else:
                 states_unique = df_final['State'].dropna().unique()
                 plain_states = [s for s in states_unique if s != "N/A"]
                 metadata['states'] += sorted([s.capitalize() for s in plain_states]) 

        # --- Min/Max Values ---
        required_minmax_cols = ['Raw Pledged', 'Raw Goal', 'Raw Raised']
        if all(col in df_final.columns for col in required_minmax_cols):
            try:
                min_max_stats = df_final[required_minmax_cols].agg(['min', 'max'])

                metadata['min_max_values']['pledged']['min'] = int(min_max_stats.loc['min', 'Raw Pledged']) if pd.notna(min_max_stats.loc['min', 'Raw Pledged']) else 0
                metadata['min_max_values']['pledged']['max'] = int(min_max_stats.loc['max', 'Raw Pledged']) if pd.notna(min_max_stats.loc['max', 'Raw Pledged']) else 1000
                metadata['min_max_values']['goal']['min'] = int(min_max_stats.loc['min', 'Raw Goal']) if pd.notna(min_max_stats.loc['min', 'Raw Goal']) else 0
                metadata['min_max_values']['goal']['max'] = int(min_max_stats.loc['max', 'Raw Goal']) if pd.notna(min_max_stats.loc['max', 'Raw Goal']) else 10000
                metadata['min_max_values']['raised']['min'] = int(min_max_stats.loc['min', 'Raw Raised']) if pd.notna(min_max_stats.loc['min', 'Raw Raised']) else 0

                max_raised_calc_val = min_max_stats.loc['max', 'Raw Raised']
                min_raised = metadata['min_max_values']['raised']['min']
                calculated_max = int(max_raised_calc_val) if pd.notna(max_raised_calc_val) else 500
                metadata['min_max_values']['raised']['max'] = max(min_raised, calculated_max)


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

def process_kickstarter_dataframe(df: pd.DataFrame, latest_dataset_date: datetime.date, parquet_output_path: str):
    """
    Processes the combined Kickstarter DataFrame using Pandas: flattens, calculates fields,
    deduplicates, filters, converts country codes, calculates popularity,
    selects final columns, and saves to Parquet.

    Args:
        df (pd.DataFrame): Combined DataFrame from all JSON files.
        latest_dataset_date (datetime.date): The date extracted from the latest input file.
        parquet_output_path (str): Path to save the final Parquet file.

    Returns:
        bool: True if successful, False otherwise
    """
    print(f"\n--- Starting DataFrame Processing ({len(df)} rows) ---")
    start_time = time.time()

    try:
        if df.empty: 
             print("Input DataFrame is empty. Skipping processing.")
             return False

        # --- Flatten nested structures ---
        def safe_get(data, keys, default=None):
            """Safely get nested dictionary keys."""
            try:
                for key in keys:
                    data = data[key]
                return data
            except (KeyError, TypeError, IndexError):
                return default

        # Flatten 'urls.web.project'
        try:
            web_project_url_col = 'urls.web.project'
            if 'urls' in df.columns and isinstance(df['urls'].iloc[0], dict):
                 df[web_project_url_col] = df['urls'].apply(lambda x: safe_get(x, ['web', 'project']))
                 print("Flattened 'urls.web.project' using .apply()")
            elif web_project_url_col not in df.columns: 
                 print("Warning: 'urls' column or nested structure 'web.project' not found. Creating dummy column.")
                 df[web_project_url_col] = None
            else:
                 print(f"Column '{web_project_url_col}' seems to exist or 'urls' is not a dict column.")
                 pass 

        except Exception as e:
            print(f"Warning: Error trying to flatten 'urls.web.project': {e}. Creating dummy column.")
            df['urls.web.project'] = None
            web_project_url_col = 'urls.web.project'

        # --- Flatten remaining structures needed ---
        flatten_cols = {
            'category': ['name', 'parent_name'],
            'creator': ['name'],
            'location': ['country', 'expanded_country', 'displayable_name']
        }
        for base_col, fields in flatten_cols.items():
             if base_col in df.columns and isinstance(df[base_col].iloc[0], dict): 
                 print(f"Flattening column: {base_col}")
                 for field in fields:
                     new_col_name = f"{base_col}.{field}"
                     try:
                         if field in df[base_col].dropna().iloc[0]:
                             df[new_col_name] = df[base_col].apply(lambda x: safe_get(x, [field]))
                         else:
                              print(f"  Warning: Field '{field}' not found in first non-null item of '{base_col}'. Skipping {new_col_name}.")
                              if new_col_name not in df.columns: df[new_col_name] = None 
                     except IndexError: 
                          print(f"  Warning: Column '{base_col}' contains no non-null dictionary values. Skipping flattening.")
                          if new_col_name not in df.columns: df[new_col_name] = None
                     except Exception as e:
                          print(f"  Warning: Error flattening {base_col}.{field}: {e}. Skipping {new_col_name}.")
                          if new_col_name not in df.columns: df[new_col_name] = None
             elif base_col in df.columns:
                 print(f"  Warning: Column '{base_col}' exists but is not a dict type. Cannot flatten further.")
             else:
                 for field in fields:
                     new_col_name = f"{base_col}.{field}"
                     if new_col_name not in df.columns: df[new_col_name] = None


        print("Structure flattening complete.")

        # --- Column Access and Calculations ---
        goal_col = safe_column_access(df, ['goal'])
        exchange_rate_col = safe_column_access(df, ['usd_exchange_rate', 'static_usd_rate'])
        pledged_col = safe_column_access(df, ['converted_pledged_amount', 'pledged'])
        created_col = safe_column_access(df, ['created_at'])
        deadline_col = safe_column_access(df, ['deadline'])
        backers_col = safe_column_access(df, ['backers_count'])
        name_col = safe_column_access(df, ['name'])
        creator_name_col = safe_column_access(df, ['creator.name'])
        loc_country_col = safe_column_access(df, ['location.country'])
        state_col = safe_column_access(df, ['state'])
        category_parent_col = safe_column_access(df, ['category.parent_name']) 
        category_name_col = safe_column_access(df, ['category.name']) 
        top_level_country_col = safe_column_access(df, ['country']) 

        # --- Calculations (Raw Goal, Pledged, Raised) ---
        if goal_col:
             df[goal_col] = pd.to_numeric(df[goal_col], errors='coerce').fillna(0.0)
        if exchange_rate_col:
             df[exchange_rate_col] = pd.to_numeric(df[exchange_rate_col], errors='coerce').fillna(1.0)
        if pledged_col:
            df[pledged_col] = pd.to_numeric(df[pledged_col], errors='coerce').fillna(0.0)

        # Goal Calculation
        if goal_col and exchange_rate_col:
            df['Raw Goal'] = (df[goal_col] * df[exchange_rate_col])
            df['Raw Goal'] = df['Raw Goal'].clip(lower=1.0)
        elif goal_col:
             print("Warning: Exchange rate column not found. Using 'goal' directly for 'Raw Goal'.")
             df['Raw Goal'] = df[goal_col].clip(lower=1.0) 
        else:
            print("Warning: 'goal' column not found. Setting 'Raw Goal' to default 1.0.")
            df['Raw Goal'] = 1.0

        df['Raw Goal'] = df['Raw Goal'].astype(float)

        # Pledged Calculation
        if pledged_col:
            df['Raw Pledged'] = df[pledged_col].astype(float) 
        else:
            print("Warning: 'pledged'/'converted_pledged_amount' column not found. Setting 'Raw Pledged' to 0.0.")
            df['Raw Pledged'] = 0.0
        df['Raw Pledged'] = df['Raw Pledged'].astype(float)

        # Raised Calculation
        df['Raw Raised'] = np.where(
            (df['Raw Pledged'] <= 0.0) | (df['Raw Goal'] <= 0.0), 
            0.0,                                                
            (df['Raw Pledged'] / df['Raw Goal']) * 100.0     
        )
        df['Raw Raised'] = df['Raw Raised'].fillna(0.0).astype(float)

        # --- Date Conversions (Raw Date, Raw Deadline) ---
        default_datetime_pd = pd.Timestamp("2000-01-01", tz='UTC') 

        def convert_epoch_to_datetime_pd(series):
            """Helper to convert epoch (s or ms) Series to Datetime, handling errors."""
            dt_series = pd.to_datetime(series, unit='s', errors='coerce', utc=True)
            first_valid = dt_series.dropna().iloc[0] if not dt_series.dropna().empty else None
            if dt_series.isnull().sum() > len(series) * 0.5 or (first_valid and first_valid.year < 1971):
                 print("  Trying milliseconds conversion for some values...")
                 dt_series_ms = pd.to_datetime(series, unit='ms', errors='coerce', utc=True)
                 dt_series = dt_series.fillna(dt_series_ms)

            return dt_series.fillna(default_datetime_pd) 

        if created_col and created_col in df.columns:
             df[created_col] = pd.to_numeric(df[created_col], errors='coerce')
             df['Raw Date'] = convert_epoch_to_datetime_pd(df[created_col])
        else:
             print(f"Warning: Column '{created_col}' not found for date conversion. Setting 'Raw Date' to default.")
             df['Raw Date'] = default_datetime_pd

        if deadline_col and deadline_col in df.columns:
             df[deadline_col] = pd.to_numeric(df[deadline_col], errors='coerce')
             df['Raw Deadline'] = convert_epoch_to_datetime_pd(df[deadline_col])
        else:
             print(f"Warning: Column '{deadline_col}' not found for date conversion. Setting 'Raw Deadline' to default.")
             df['Raw Deadline'] = default_datetime_pd

        df['Raw Date'] = pd.to_datetime(df['Raw Date'], utc=True)
        df['Raw Deadline'] = pd.to_datetime(df['Raw Deadline'], utc=True)

        # --- Backer Count ---
        if backers_col and backers_col in df.columns:
             df['Backer Count'] = pd.to_numeric(df[backers_col], errors='coerce').fillna(0).astype('Int64')
        else:
            print("Warning: 'backers_count' column not found. Setting 'Backer Count' to 0.")
            df['Backer Count'] = 0
            df['Backer Count'] = df['Backer Count'].astype('Int64') 

        # --- Country Code Replacement ---
        print("Applying country code conversion...")
        country_col_final = 'Country'
        country_code_source_col = '__country_code_source' 

        df[country_code_source_col] = None 
        if loc_country_col and loc_country_col in df.columns:
            df[country_code_source_col] = df[loc_country_col]
            print(f"Using '{loc_country_col}' as primary country code source.")
            if top_level_country_col and top_level_country_col in df.columns:
                df[country_code_source_col] = df[country_code_source_col].fillna(df[top_level_country_col])
                print(f"Using '{top_level_country_col}' as fallback country code source.")
            else:
                print("Top-level 'country' column not found for fallback.")
        elif top_level_country_col and top_level_country_col in df.columns:
            df[country_code_source_col] = df[top_level_country_col]
            print(f"Primary source '{loc_country_col}' not found. Using '{top_level_country_col}' as country code source.")
        else:
            print(f"Warning: Neither '{loc_country_col}' nor '{top_level_country_col}' found. Cannot determine country codes.")
            df[country_col_final] = "N/A" 

        if country_code_source_col in df.columns:
            country_mapping = {}
            try:
                 unique_codes = df[country_code_source_col].dropna().astype(str).unique()
            except Exception as e:
                 print(f"Warning: Could not extract unique country codes from '{country_code_source_col}': {e}. Mapping might be incomplete.")
                 unique_codes = []

            for code in unique_codes:
                if isinstance(code, str) and len(code) == 2:
                    try:
                        country = coco.convert(names=code.upper(), to='name_short')
                        if country:
                            country_mapping[code.upper()] = country
                    except Exception:
                        pass
                    
            country_mapping['XK'] = 'Kosovo' # Add Kosovo mapping

            print(f"Found {len(country_mapping)} valid country code mappings (including XK).")

            # --- Apply mapping using .map() ---
            df[country_col_final] = df[country_code_source_col].astype(str).str.upper().map(country_mapping).fillna("N/A")
            df[country_col_final] = df[country_col_final].str.replace("the United States", "United States", regex=False)
            df[country_col_final] = df[country_col_final].str.replace("the United Kingdom", "United Kingdom", regex=False)

            print(f"Created '{country_col_final}' column using codes from '{country_code_source_col}' via .map(). Invalid/missing codes set to 'N/A'.")
            df.drop(columns=[country_code_source_col], inplace=True) 

        elif country_col_final not in df.columns:
             df[country_col_final] = "N/A"

        # --- Category Selection ---
        category_col_final = 'Category'
        subcategory_col_final = 'Subcategory'

        raw_parent_col_name = 'category.parent_name'
        raw_name_col_name = 'category.name'

        category_parent_col = safe_column_access(df, [raw_parent_col_name])
        if not category_parent_col:
            print(f"Warning: Source column for parent category ('{raw_parent_col_name}') not found. Creating dummy.")
            df[raw_parent_col_name] = "N/A"
            category_parent_col = raw_parent_col_name 
        df[category_parent_col] = df[category_parent_col].astype(str).fillna("N/A")

        category_name_col = safe_column_access(df, [raw_name_col_name])
        if not category_name_col:
            print(f"Warning: Source column for category name ('{raw_name_col_name}') not found. Creating dummy.")
            df[raw_name_col_name] = "N/A"
            category_name_col = raw_name_col_name 
        df[category_name_col] = df[category_name_col].astype(str).fillna("N/A")

        df[category_col_final] = np.where(
            (df[category_parent_col] != "N/A") & (df[category_parent_col] != ""), 
            df[category_parent_col],   
            np.where( 
                 (df[category_name_col] != "N/A") & (df[category_name_col] != ""),
                 df[category_name_col],                               
                 "N/A"                                    
            )
        )

        df[subcategory_col_final] = df[category_name_col]

        # --- Format Display Columns ---
        df['Goal'] = df['Raw Goal'].round(2)
        df['Pledged Amount'] = df['Raw Pledged'].round(2)
        df['%Raised'] = df['Raw Raised'].round(2)

        df['Date'] = df['Raw Date'].dt.strftime('%Y-%m-%d').fillna('N/A')
        df['Deadline'] = df['Raw Deadline'].dt.strftime('%Y-%m-%d').fillna('N/A')

        print("Adjusting categories: Imputing missing Category based on Subcategory mode...")

        if category_col_final not in df.columns:
             print(f"Error: Column '{category_col_final}' missing before final adjustment. Creating default.")
             df[category_col_final] = "N/A"
        if subcategory_col_final not in df.columns:
             print(f"Error: Column '{subcategory_col_final}' missing before final adjustment. Creating default.")
             df[subcategory_col_final] = "N/A"
        df[category_col_final] = df[category_col_final].astype(str).fillna("N/A")
        df[subcategory_col_final] = df[subcategory_col_final].astype(str).fillna("N/A")

        invalid_values = ['N/A', 'None', ''] 

        category_is_invalid_mask = df[category_col_final].isnull() | df[category_col_final].isin(invalid_values)
        print(f"Found {category_is_invalid_mask.sum()} rows with invalid Category.")

        if category_is_invalid_mask.any():
            valid_category_mask = ~df[category_col_final].isin(invalid_values) & df[category_col_final].notnull()
            valid_subcategory_mask = ~df[subcategory_col_final].isin(invalid_values) & df[subcategory_col_final].notnull()
            valid_pairs_df = df.loc[valid_category_mask & valid_subcategory_mask, [category_col_final, subcategory_col_final]].copy()

            subcategory_to_most_common_category = {} 
            valid_category_set = set()         

            if not valid_pairs_df.empty:
                print("Calculating most common category for each subcategory (for fallback)...")
                try:
                    mode_series = valid_pairs_df.groupby(subcategory_col_final)[category_col_final].agg(lambda x: x.mode()[0] if not x.mode().empty else "N/A")
                    subcategory_to_most_common_category = mode_series.to_dict()
                    print(f"Created mode mapping for {len(subcategory_to_most_common_category)} subcategories.")
                except Exception as e:
                    print(f"Warning: Error calculating subcategory modes: {e}. Mode fallback might be incomplete.")

                print("Identifying valid categories (where Category != Subcategory)...")
                valid_categories_where_different = valid_pairs_df[valid_pairs_df[category_col_final] != valid_pairs_df[subcategory_col_final]][category_col_final]
                valid_category_set = set(valid_categories_where_different.unique())
                valid_category_set = {cat for cat in valid_category_set if cat not in invalid_values and pd.notna(cat)}
                print(f"Found {len(valid_category_set)} unique valid categories meeting the criteria.")

            else:
                print("No valid Category/Subcategory pairs found to build imputation map or valid category set.")

            print("Applying imputation logic to rows with invalid Category...")
            imputed_count_direct = 0
            imputed_count_mode = 0
            failed_imputation_count = 0

            indices_to_impute = df[category_is_invalid_mask].index

            new_categories = df[category_col_final].copy()

            for idx in indices_to_impute:
                current_subcat = df.loc[idx, subcategory_col_final]

                if current_subcat in invalid_values or pd.isna(current_subcat):
                     new_categories.loc[idx] = "N/A" 
                     failed_imputation_count += 1
                     continue

                if current_subcat in valid_category_set:
                    new_categories.loc[idx] = current_subcat
                    imputed_count_direct += 1
                else:
                    most_common_cat = subcategory_to_most_common_category.get(current_subcat, "N/A")
                    new_categories.loc[idx] = most_common_cat
                    if most_common_cat != "N/A":
                        imputed_count_mode += 1
                    else:
                        failed_imputation_count += 1 

            df[category_col_final] = new_categories

            print(f"Imputation summary: ")
            print(f"  - Directly imputed using Subcategory value: {imputed_count_direct}")
            print(f"  - Imputed using mode Category: {imputed_count_mode}")
            print(f"  - Failed to impute (remains N/A): {failed_imputation_count}")

        else:
             print("No invalid categories found requiring adjustment.")

        df[category_col_final] = df[category_col_final].astype(str).fillna("N/A")
        df[subcategory_col_final] = df[subcategory_col_final].astype(str).fillna("N/A")

        # --- Select and Alias Final Columns ---
        print("Selecting final columns...")
        final_columns_mapping = {
            'Project Name': name_col,
            'Creator': creator_name_col,
            'Link': web_project_url_col,
            'Date': 'Date', 
            'Deadline': 'Deadline',
            'State': state_col,
            'Country': country_col_final,
            'Category': category_col_final,
            'Subcategory': subcategory_col_final,
            'Goal': 'Goal', 
            'Pledged Amount': 'Pledged Amount',
            '%Raised': '%Raised',
            'Backer Count': 'Backer Count',
            'Raw Goal': 'Raw Goal',
            'Raw Pledged': 'Raw Pledged',
            'Raw Raised': 'Raw Raised',
            'Raw Date': 'Raw Date',
            'Raw Deadline': 'Raw Deadline',
            'Country Code': loc_country_col
        }

        select_cols = []
        rename_map = {}
        missing_original_codes = []
        for final_name, source_col_ref in final_columns_mapping.items():
             if source_col_ref and source_col_ref in df.columns:
                 select_cols.append(source_col_ref)
                 if final_name != source_col_ref:
                      rename_map[source_col_ref] = final_name
             else:
                 print(f"Warning: Source column '{source_col_ref}' for final column '{final_name}' not found. Column will be missing.")
                 missing_original_codes.append(final_name)

        if not select_cols:
             print("Error: No columns could be selected for the final DataFrame.")
             return False

        df_final = df[select_cols].copy() 
        df_final.rename(columns=rename_map, inplace=True)

        for missing_col in missing_original_codes:
             if missing_col not in df_final.columns:
                 print(f"Adding missing final column '{missing_col}' with default value 'N/A'.")
                 df_final[missing_col] = "N/A" 

        string_like_cols = ['Project Name', 'Creator', 'Link', 'State', 'Country', 'Category', 'Subcategory', 'Country Code', 'Date', 'Deadline']
        for col in df_final.columns:
             if col in string_like_cols:
                 df_final[col] = df_final[col].astype(str).fillna("N/A")

        print(f"Selected {len(df_final.columns)} final columns.")

        # --- Calculate Popularity Score on the final DataFrame ---
        print("Calculating popularity score...")
        if 'Raw Date' not in df_final.columns or not pd.api.types.is_datetime64_any_dtype(df_final['Raw Date']):
             print("Error: 'Raw Date' column missing or wrong type for popularity score. Setting time_factor to 0.")
             df_final['time_factor'] = 0.0
        else:
             now_dt = pd.Timestamp.now(tz='UTC')
             time_delta_days = (now_dt - df_final['Raw Date']).dt.total_seconds() / (24 * 3600)

             valid_days_diff = time_delta_days[time_delta_days > 0]
             computed_max_days = valid_days_diff.max() if not valid_days_diff.empty else None

             if computed_max_days is None or computed_max_days <= 0:
                  print("Warning: Could not determine valid positive max_days for time factor. Setting time_factor to 0.")
                  df_final['time_factor'] = 0.0
             else:
                 df_final['time_factor'] = (1.0 - (time_delta_days / computed_max_days)).clip(0.0, 1.0)

             df_final['time_factor'] = df_final['time_factor'].fillna(0.0).astype(float)

        required_norm_cols_final = ['Backer Count', 'Raw Pledged', 'Raw Raised']
        missing_norm_cols_final = [col for col in required_norm_cols_final if col not in df_final.columns]
        if missing_norm_cols_final:
            print(f"Error: Missing columns for normalization in final DF: {missing_norm_cols_final}. Popularity score will be inaccurate (set to 0).")
            df_final['Popularity Score'] = 0.0
        else:
            # --- Popularity Score Calculations (using df_final) ---
            df_final['capped_percentage'] = df_final['Raw Raised'].clip(upper=500.0).fillna(0.0)

            def normalize_col_pd_safe(series):
                 min_val = series.min()
                 max_val = series.max()
                 if pd.isna(min_val) or pd.isna(max_val):
                     print(f"Warning: Min/max for normalization is None/NaN for {series.name}. Normalization returns 0.")
                     return pd.Series(0.0, index=series.index) 

                 range_val = max_val - min_val
                 if range_val == 0:
                      return pd.Series(0.0, index=series.index) 
                 else:
                      return ((series - min_val) / range_val).fillna(0.0).astype(float)

            # Apply normalization
            df_final['Backer Count'] = pd.to_numeric(df_final['Backer Count'], errors='coerce').fillna(0)
            df_final['Raw Pledged'] = pd.to_numeric(df_final['Raw Pledged'], errors='coerce').fillna(0)
            df_final['capped_percentage'] = pd.to_numeric(df_final['capped_percentage'], errors='coerce').fillna(0)

            df_final['normalized_backers'] = normalize_col_pd_safe(df_final['Backer Count'])
            df_final['normalized_pledged'] = normalize_col_pd_safe(df_final['Raw Pledged'])
            df_final['normalized_percentage'] = normalize_col_pd_safe(df_final['capped_percentage'])

            # Calculate Popularity Score
            if 'time_factor' not in df_final.columns: df_final['time_factor'] = 0.0

            df_final['Popularity Score'] = (
                 df_final['normalized_backers'] * 0.2778 +
                 df_final['normalized_pledged'] * 0.3889 +
                 df_final['normalized_percentage'] * 0.2222 +
                 df_final['time_factor'] * 0.1111
            ).astype(float).fillna(0.0)

            cols_to_drop = ['normalized_backers', 'normalized_pledged', 'time_factor', 'capped_percentage', 'normalized_percentage']
            existing_cols_to_drop = [col for col in cols_to_drop if col in df_final.columns]
            if existing_cols_to_drop:
                 df_final.drop(columns=existing_cols_to_drop, inplace=True)


        print("Popularity score calculation complete.")

        if df_final.empty:
            print("Warning: Final DataFrame is empty after processing and filtering. Skipping Parquet write.")
            return False

        print(f"\n--- Writing Final DataFrame ({len(df_final)} rows) ---")
        print(f"Output path: {parquet_output_path}")
        print("Final Schema:")
        print(df_final.info()) 

        Path(parquet_output_path).parent.mkdir(parents=True, exist_ok=True)
        df_final.to_parquet(parquet_output_path, compression='snappy', engine='pyarrow', index=False) 

        elapsed_time = time.time() - start_time
        file_size_mb = os.path.getsize(parquet_output_path) / (1024 * 1024) if os.path.exists(parquet_output_path) else 0
        print(f"\nProcessing and Parquet conversion completed in {elapsed_time:.2f} seconds")
        print(f"Parquet file saved to {parquet_output_path} ({file_size_mb:.2f} MB)")
        print("Final columns saved:", ", ".join(list(df_final.columns)))

        save_filter_metadata(df_final, FILTER_METADATA_FILENAME)

        return True

    except Exception as e:
        import traceback
        print(f"\n--- ERROR during DataFrame processing ---")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {e}")
        print("Traceback:")
        print(traceback.format_exc())
        if 'df' in locals() and isinstance(df, pd.DataFrame):
             print("\nDataFrame sample at error point:")
             try:
                  print(df.head(3))
             except Exception as e_print:
                  print(f"(Could not print DataFrame head: {e_print})")
        if 'df_final' in locals() and isinstance(df_final, pd.DataFrame):
             print("\nFinal DataFrame sample at error point:")
             try:
                  print(df_final.head(3))
             except Exception as e_print:
                  print(f"(Could not print df_final head: {e_print})")

        return False

if __name__ == "__main__":
    print("--- Starting Kickstarter Data Processing ---")
    overall_start_time = time.time()

    sorted_files, latest_date = find_kickstarter_files(KICKSTARTER_DATA_DIR)

    if not sorted_files:
        print("No files to process. Exiting.")
        exit()

    latest_file_path = sorted_files[0][0]
    output_parquet_filename = f"{latest_file_path.stem.split('.')[0]}.parquet" 
    output_dir = Path(output_parquet_filename).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    filter_metadata_path = output_dir / FILTER_METADATA_FILENAME 

    all_records = []
    seen_ids = set()
    total_lines_processed = 0

    print(f"\n--- Reading and Filtering Records ({len(sorted_files)} files) ---")
    for file_path, file_date in sorted_files:
        print(f"Processing file: {file_path.name} ({file_date})")
        file_start_time = time.time()
        lines_in_file = 0
        records_added_from_file = 0
        reminder_threshold = PROGRESS_REMINDER_INTERVAL

        try:
            with gzip.open(file_path, 'rt', encoding='utf-8', errors='replace') as f:
                for line in f:
                    lines_in_file += 1
                    total_lines_processed += 1

                    if lines_in_file >= reminder_threshold:
                        print(f"  ... processed {lines_in_file:,} lines in {file_path.name}")
                        reminder_threshold += PROGRESS_REMINDER_INTERVAL

                    try:
                        record = json.loads(line)
                        if 'data' in record and isinstance(record['data'], dict) and 'id' in record['data']:
                            project_id = record['data']['id']
                            try:
                                project_id_int = int(project_id)
                            except (ValueError, TypeError):
                                continue 

                            if project_id_int not in seen_ids:
                                seen_ids.add(project_id_int)

                                data = record['data']
                                if 'state' not in data or data['state'] is None:
                                     data['state'] = "unknown"
                                else:
                                     data['state'] = str(data['state'])

                                all_records.append(data)
                                records_added_from_file += 1

                    except json.JSONDecodeError:
                        print(f"  Warning: Skipping invalid JSON line {lines_in_file} in {file_path.name}")
                        continue
                    except Exception as e_line:
                        print(f"  Error processing line {lines_in_file} in {file_path.name}: {e_line}")
                        continue 

            file_elapsed = time.time() - file_start_time
            print(f"  Finished {file_path.name}: Read {lines_in_file:,} lines, added {records_added_from_file:,} unique records. Took {file_elapsed:.2f}s.")

        except FileNotFoundError:
             print(f"Error: File not found: {file_path}. Skipping.")
        except Exception as e_file:
             print(f"Error reading or processing file {file_path.name}: {e_file}")


    print(f"\n--- Initial Record Collection Complete ---")
    print(f"Total lines processed across all files: {total_lines_processed:,}")
    print(f"Total unique project records found: {len(all_records):,}")

    if not all_records:
        print("No records collected. Exiting.")
        exit()

    print("\n--- Creating Pandas DataFrame ---")
    try:
        df_combined = pd.DataFrame(all_records)
        print(f"Initial DataFrame created with {len(df_combined)} rows and {len(df_combined.columns)} columns.")
        del all_records
        import gc
        gc.collect()
    except Exception as e_df:
        print(f"Error creating DataFrame: {e_df}")
        exit()

    success = process_kickstarter_dataframe(df_combined, latest_date, output_parquet_filename)

    overall_elapsed = time.time() - overall_start_time
    print("\n--- Processing Summary ---")
    if success:
        print(f"Successfully processed data and saved to: {output_parquet_filename}")
        if os.path.exists(filter_metadata_path):
             print(f"Filter metadata saved to: {filter_metadata_path}")
        else:
             print(f"Warning: Filter metadata file '{filter_metadata_path}' was not generated.")
        print(f"Total execution time: {overall_elapsed:.2f} seconds")

    else:
        print("Processing failed. Please review the error messages above.")
        print(f"Total execution time: {overall_elapsed:.2f} seconds")