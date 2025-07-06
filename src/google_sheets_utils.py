import gspread
import pandas as pd
from oauth2client.service_account import ServiceAccountCredentials

SHEET_NAME = "tmt_forecasting_data"

def authorize_google_sheets(credentials_path="credentials.json"):
    scope = ["https://spreadsheets.google.com/feeds",
             "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name(credentials_path, scope)
    client = gspread.authorize(creds)
    return client

def read_sheet_as_df(sheet_tab_name, credentials_path="credentials.json"):
    client = authorize_google_sheets(credentials_path)
    sheet = client.open(SHEET_NAME).worksheet(sheet_tab_name)
    data = sheet.get_all_values()
    df = pd.DataFrame(data[1:], columns=data[0])
    # Parse date column if present
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], dayfirst=True)
    return df

def write_df_to_sheet(df, sheet_tab_name, credentials_path="credentials.json"):
    client = authorize_google_sheets(credentials_path)
    sheet = client.open(SHEET_NAME).worksheet(sheet_tab_name)
    sheet.clear()
    data = [df.columns.tolist()] + df.astype(str).values.tolist()
    sheet.update(data)

def append_row_to_sheet(row_dict, sheet_tab_name, credentials_path="credentials.json"):
    client = authorize_google_sheets(credentials_path)
    sheet = client.open(SHEET_NAME).worksheet(sheet_tab_name)
    headers = sheet.row_values(1)
    new_row = [row_dict.get(col, "") for col in headers]
    sheet.append_row(new_row)