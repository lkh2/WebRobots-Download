# Kickstarter Data Preprocessing for CrowdInsight

This script (`database_download.py`) preprocesses raw Kickstarter project data downloaded as gzipped JSON line files (`.json.gz`) for use with the [CrowdInsight Streamlit application](https://github.com/lkh2/CrowdInsight). It processes multiple data dumps, deduplicates projects, cleans and transforms the data, calculates derived metrics like popularity, and outputs a consolidated Parquet file and a JSON metadata file suitable for the application's filters.

## Input

The script requires a directory containing Kickstarter data files.

1.  **Directory:** You must specify the path to this directory by setting the `KICKSTARTER_DATA_DIR` variable within the `database_download.py` script.
2.  **Files:** The script looks for files matching the pattern `Kickstarter_YYYY-MM-DDTHH_MM_SS_MSZ*.json.gz` within the specified directory. The date in the filename is used to determine the latest version of each project.

## Process

The script performs the following steps:

1.  **File Discovery:** Locates all `Kickstarter*.json.gz` files in the `KICKSTARTER_DATA_DIR`.
2.  **Date Parsing & Sorting:** Extracts the date from each filename and sorts the files in descending order (newest first).
3.  **Data Loading & Deduplication:** Reads records line by line from the gzipped JSON files. It keeps only the first encountered version of each unique project ID (based on the sorted file list, this means the _latest_ version is kept).
4.  **Initial DataFrame Creation:** Combines the unique records into a Pandas DataFrame.
5.  **Data Flattening:** Extracts relevant data from nested JSON structures (e.g., `category`, `creator`, `location`, `urls`).
6.  **Column Calculation & Cleaning:**
    - Calculates goal and pledged amounts in USD using exchange rates where available.
    - Computes the percentage raised.
    - Converts epoch timestamps for creation and deadline dates into datetime objects and then formatted strings.
    - Cleans and standardizes backer counts.
    - Uses `country_converter` to map country codes (from `location.country` or `country`) to full country names. Handles missing or invalid codes.
    - Determines `Category` and `Subcategory`, attempting to impute missing parent categories based on subcategory information.
    - Cleans `State` information.
7.  **Final Column Selection:** Selects and renames columns needed for the CrowdInsight app. Fills missing essential columns with "N/A".
8.  **Popularity Score Calculation:** Computes a 'Popularity Score' based on normalized backer count, pledged amount, percentage raised (capped), and project age (time factor).
9.  **Output Generation:**
    - Saves the final processed DataFrame to a Parquet file (`.parquet`) using snappy compression. The filename is derived from the latest input data file (e.g., `Kickstarter_2023-10-26T...json` becomes `Kickstarter_2023-10-26T....parquet`).
    - Calculates and saves metadata (unique categories, countries, states, subcategories, min/max values for numerical filters, date ranges) to `filter_metadata.json` in the same directory as the Parquet file. This file powers the dynamic filters in the Streamlit app.

## Output

1.  **`[LatestInputFileNameWithoutExtension].parquet`**: A Parquet file containing the cleaned, processed, and consolidated Kickstarter project data. This is the primary data source for the CrowdInsight application.
2.  **`filter_metadata.json`**: A JSON file containing lists of unique values (categories, countries, etc.) and min/max ranges derived from the processed data. This is used to populate the filter options in the CrowdInsight UI.

## Usage

1.  **Set Input Directory:** Modify the `KICKSTARTER_DATA_DIR` variable near the top of `database_download.py` to point to the directory containing your `.json.gz` files.
    ```python
    # Directory containing the Kickstarter JSON.gz files
    # !!! IMPORTANT: Set this path to your data directory !!!
    KICKSTARTER_DATA_DIR = Path("path/to/your/kickstarter/data")
    ```
2.  **Install Dependencies:** Make sure you have the required libraries installed:
    ```bash
    pip install pandas numpy country_converter pyarrow openpyxl
    ```
    _(Note: `openpyxl` might be needed indirectly by pandas depending on the operation, it's good practice to include it)._
3.  **Run the Script:** Execute the script from your terminal:
    ```bash
    python database_download.py
    ```

The script will print progress information to the console and save the output files in the directory specified by `output_parquet_filename` (which defaults to the same directory as the script unless `KICKSTARTER_DATA_DIR` points elsewhere).

## Dependencies

- pandas
- numpy
- country_converter
- pyarrow
- pathlib (standard library)
- datetime (standard library)
- json (standard library)
- re (standard library)
- gzip (standard library)
- os (standard library)
- time (standard library)
