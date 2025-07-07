import gspread
import pandas as pd
import numpy as np
from oauth2client.service_account import ServiceAccountCredentials
from gspread_dataframe import set_with_dataframe

SHEET_NAME = "tmt_forecasting_data"
SHEET_ID = "1EpErhtDmjRKLY4bgM_CVrQU5KDPM4GIBctyYn1bUblk"


def authorize_google_sheets(credentials_path="credentials.json"):
    """Authorize using a Google service account JSON key file."""
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive"
    ]
    try:
        creds = ServiceAccountCredentials.from_json_keyfile_name(credentials_path, scope)
        client = gspread.authorize(creds)
        return client
    except Exception as e:
        print(f"❌ Failed to authorize Google Sheets. Check credentials.json. Error: {e}")
        raise


def read_sheet_as_df(sheet_tab_name, credentials_path="credentials.json"):
    """Read a Google Sheet tab into a pandas DataFrame with proper typing."""
    client = authorize_google_sheets(credentials_path)
    sheet = client.open_by_key(SHEET_ID).worksheet(sheet_tab_name)
    data = sheet.get_all_values()

    if not data or len(data) < 2:
        return pd.DataFrame()

    headers = [h.strip() for h in data[0]]
    df = pd.DataFrame(data[1:], columns=headers)

    # Strip whitespace from all string entries
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    # Convert columns to appropriate types
    for col in df.columns:
        col_clean = col.lower()
        if col_clean == 'date':
            df[col] = pd.to_datetime(df[col], dayfirst=True, errors='coerce')
        elif col_clean not in ['month', 'city']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def write_df_to_sheet(df, sheet_tab_name, credentials_path="credentials.json"):
    """Write a pandas DataFrame to a Google Sheet tab with proper formatting."""
    client = authorize_google_sheets(credentials_path)
    sheet = client.open_by_key(SHEET_ID).worksheet(sheet_tab_name)
    sheet.clear()

    df = df.copy()

    # Standardize and clean column names
    df.columns = df.columns.str.strip()

    # Format 'date' column as string in DD-MM-YYYY for user display
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.strftime('%d-%m-%Y')

    # Safely handle missing data
    df.replace([np.nan, None], "", inplace=True)

    # Use gspread's helper to push DataFrame with correct types
    set_with_dataframe(sheet, df, include_index=False, include_column_header=True, resize=True)


def append_row_to_sheet(row_dict, sheet_tab_name, credentials_path="credentials.json"):
    """Append a single row to a Google Sheet tab based on matching headers."""
    client = authorize_google_sheets(credentials_path)
    sheet = client.open_by_key(SHEET_ID).worksheet(sheet_tab_name)

    # Get and clean header row
    headers = [h.strip() for h in sheet.row_values(1)]

    # Format values properly
    formatted_row = []
    for col in headers:
        val = row_dict.get(col, "")
        if col.lower() == "date" and isinstance(val, (pd.Timestamp, str)):
            # Convert to DD-MM-YYYY if needed
            try:
                val = pd.to_datetime(val, dayfirst=True).strftime('%d-%m-%Y')
            except:
                val = str(val)
        elif isinstance(val, float):
            val = round(val, 2)  # Limit float precision
        formatted_row.append(str(val))

    # Append using USER_ENTERED so Google Sheets parses types automatically
    sheet.append_row(formatted_row, value_input_option="USER_ENTERED")
