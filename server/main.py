import os
from typing import Union, List, Optional
import json
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Depends, Body, UploadFile
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from fastapi import Query
import pandas as pd
import pandas_ta as ta
from datastore.providers.database import DatabaseManager
from psycopg2.extras import RealDictCursor
from pydantic import BaseModel
from datetime import date
from pydantic import ValidationError


import finnhub
from datetime import datetime, timedelta

from models.api import (
    DeleteRequest,
    DeleteResponse,
    QueryRequest,
    FilenamesRequest,
    QueryResponse,
    UpsertRequest,
    UpsertResponse,
    JsonResponse,
)
from datastore.factory import get_datastore
from services.file import get_document_from_file

from models.models import (
    DocumentMetadata, 
    Source,
    DocumentMetadataFilter,
    Query as ApiQuery, 
    FormType,   
)

FINHUB_API_KEY = os.environ.get("FINHUB_API_KEY")
finnhub_client = finnhub.Client(api_key=FINHUB_API_KEY)

bearer_scheme = HTTPBearer()
BEARER_TOKEN = os.environ.get("BEARER_TOKEN")
assert BEARER_TOKEN is not None


def validate_token(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    if credentials.scheme != "Bearer" or credentials.credentials != BEARER_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing token")
    return credentials


app = FastAPI(dependencies=[Depends(validate_token)])

# Initialize DatabaseManager
db_manager = DatabaseManager()

@app.post(
    "/upsert-file",
    response_model=UpsertResponse,
)
async def upsert_file(
    file: UploadFile = File(...),
    metadata: Optional[str] = Form(None),
):
    try:
        metadata_obj = (
            DocumentMetadata.parse_raw(metadata)
            if metadata
            else DocumentMetadata(source=Source.file)
        )
    except:
        metadata_obj = DocumentMetadata(source=Source.file)

    document = await get_document_from_file(file, metadata_obj)

    try:
        ids = await datastore.upsert([document])
        return UpsertResponse(ids=ids)
    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail=f"str({e})")


@app.post(
    "/upsert",
    response_model=UpsertResponse,
)
async def upsert(
    request: UpsertRequest = Body(...),
):
    try:
        print("GOT UPSERT REQUEST:")
        ids = await datastore.upsert(request.documents)
        print("IDs ARE:")
        print(ids)
        return UpsertResponse(ids=ids)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal Service Error")


@app.post(
    "/query",
    response_model=QueryResponse,
)
async def query_main(
    request: QueryRequest = Body(...),
):
    try:
        # print("GOT QUERY REQUEST:")
        # print(request)
        results = await datastore.query(
            request.queries,
        )
        return QueryResponse(results=results)
    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail="Internal Service Error")


@app.get(
    "/technicals"
)
async def technicals_main(
    symbol: str = Query(...)
):
    try:

        # Convert datetime objects to Unix timestamps
        start_date = int((datetime.utcnow() - timedelta(days=60)).timestamp())
        end_date = int(datetime.utcnow().timestamp())

        # Use Unix timestamps in the API call
        data = finnhub_client.stock_candles(symbol, 'D', start_date, end_date)

        # Convert to DataFrame
        df = pd.DataFrame(data)
        df['t'] = pd.to_datetime(df['t'], unit='s')  # Convert 't' to a datetime format
        df.set_index('t', inplace=True)

        # Calculate technical indicators
        # Moving Averages
        df['SMA_10'] = ta.sma(df['c'], length=10)
        df['EMA_10'] = ta.ema(df['c'], length=10)

        # Relative Strength Index (RSI)
        df['RSI_14'] = ta.rsi(df['c'], length=14)

        # Moving Average Convergence Divergence (MACD)
        macd = ta.macd(df['c'])
        df = pd.concat([df, macd], axis=1)

        # Bollinger Bands
        bollinger = ta.bbands(df['c'])
        df = pd.concat([df, bollinger], axis=1)

        # Stochastic Oscillator
        stoch = ta.stoch(df['h'], df['l'], df['c'])
        df = pd.concat([df, stoch], axis=1)

        # ADX (Average Directional Index)
        adx = ta.adx(df['h'], df['l'], df['c'])
        df = pd.concat([df, adx], axis=1)

        # Convert datetime columns to string format
        df.index = df.index.strftime('%Y-%m-%d %H:%M:%S')

        # Convert DataFrame to JSON using Pandas' to_json
        json_response_data = df[::-1].to_json(orient='index')

        print(json_response_data)
        return JsonResponse(results=json_response_data)

    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail="Internal Service Error")


def extract_financial_ratios(data):
    # Define the keys for the ratios we are interested in
    keys = {
        'price_earnings_ratio': 'peTTM',  # Price to Earnings Ratio
        'price_sales_ratio': 'psTTM',  # Price to Sales Ratio
        'price_free_cash_flow_ratio': 'pfcfShareTTM',  # Price to Free Cash Flow Ratio
        'price_book_ratio': 'pbQuarterly',  # Price to Book Ratio
        'revenue_growth_3_year': "revenueGrowth3Y",
        'revenue_growth_5_year': "revenueGrowth5Y",
        'revenue_growth_ttm': "revenueGrowthTTMYoy",
        'eps_growth_3_year': "epsGrowth3Y",
        'eps_growth_5_year': "epsGrowth5Y",
        'eps_growth_ttm': "epsGrowthTTMYoy",
        'cash_flow_per_share_ttm': "cashFlowPerShareTTM",
        'market_cap': "marketCapitalization",
        'dividend_growth_5_year': "dividendGrowthRate5Y",
        'dividend_per_share_ttm': "dividendPerShareTTM",
        'ebitda_growth_5_year': "ebitdaCagr5Y",
        'ebitda_per_share_ttm': "ebitdPerShareTTM",
    }

    # Extract and label the ratios
    extracted_ratios = {label: data.get(key, None) for label, key in keys.items()}

    return extracted_ratios

def update_with_ebitda(cash_flow):
    for financial in cash_flow["financials"]:
        net_income = financial.get("netIncomeStartingLine", 0)
        interest = financial.get("cashInterestPaid", 0)
        taxes = financial.get("cashTaxesPaid", 0)
        depreciation_amortization = financial.get("depreciationAmortization", 0)
        
        # Calculate EBITDA
        ebitda = net_income + interest + taxes + depreciation_amortization

        # Update the dictionary
        financial["EBITDA"] = ebitda

def extract_metrics(data, region, sector):
    """
    Extracts metrics for a given region and sector from the provided data.

    :param data: The dataset containing sector information and metrics.
    :param region: The region for which the metrics are required.
    :param sector: The sector for which the metrics are required.
    :return: A dictionary containing the metrics for the specified region and sector.
    """
    # Check if the provided region matches the data's region
    if data['region'] != region:
        return "Region not found in the data."

    # Search for the specified sector in the data
    for entry in data['data']:
        if entry['sector'] == sector:
            # Extract only the median values from the metrics
            median_metrics = {key: value['m'] for key, value in entry['metrics'].items()}
            return median_metrics

    # If the sector is not found in the data
    return "Sector not found in the data."


def get_sector_by_symbol(symbol):
    try:
        conn = db_manager.get_conn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                query = """
                    SELECT sector
                    FROM stocks
                    WHERE symbol = %s
                """
                cursor.execute(query, (symbol,))
                result = cursor.fetchone()
                if result:
                    return result['sector']
                else:
                    return None
        finally:
            db_manager.put_conn(conn)
    except Exception as e:
        print("An error occurred:", e)
        # Handle the exception as needed
        return None


def extract_texts(query_results):
    # Extract texts from the list of QueryResult objects
    texts = []
    for query_result in query_results:
        for document_chunk in query_result.results:
            texts.append(document_chunk.text)
    return texts

def extract_urls(query_results):
    # Extract texts from the list of QueryResult objects
    urls = []
    for query_result in query_results:
        for document_chunk in query_result.results:
            url = document_chunk.metadata.url

            if 'chat.aitickerchat.com' in url:
                formatted_url = f"[{document_chunk.metadata.form_type}]('https://chat.aitickerchat.com/')"
            else:
                formatted_url = f"[{document_chunk.metadata.form_type}]({url})"

            urls.append(formatted_url)
    return urls


# def get_latest_press_release_filenames(symbol):
#     try:
#         # Connect to your PostgreSQL database
#         conn = db_manager.get_conn()
        
#         filenames = []  # Initialize an empty list to store filenames

#         with conn.cursor(cursor_factory=RealDictCursor) as cursor:
#             # Fetch the most recent published_date for earnings_transcript
#             query = """
#             SELECT published_date FROM source_file_metadata 
#             WHERE symbol = %s AND form_type = 'earnings_transcript'
#             ORDER BY published_date DESC
#             LIMIT 1;
#             """
#             cursor.execute(query, (symbol,))

#             result = cursor.fetchone()
#             if result:
#                 published_date = result["published_date"]
#                 start_date = published_date - timedelta(days=2)
#                 end_date = published_date + timedelta(days=2)

#                 # Fetch all filenames within the date range for specified form_types
#                 earnings_press_release_query = """
#                 SELECT filename FROM source_file_metadata 
#                 WHERE symbol = %s AND form_type IN ('8-K', '6-K') AND published_date BETWEEN %s AND %s AND
#                 to_tsvector('english', source_file_metadata.description) @@
#                 to_tsquery('english', 'earnings | (financial & results) | (reports & fiscal)  | (results & fiscal) | (reports & quarter) | (results & quarter)')
#                 """
#                 cursor.execute(earnings_press_release_query, (symbol, start_date, end_date))
                
#                 # Process all fetched rows for filenames
#                 rows = cursor.fetchall()
#                 filenames = [row["filename"] for row in rows]  # Extract filenames

#         db_manager.put_conn(conn)  # Release the connection back to the pool
#         return filenames

#     except Exception as e:
#         print("Error:", e)
#         return []  # Return an empty list in case of error



# class UserRequest(BaseModel):
#     symbol: str
#     from_press_release: Optional[bool] = False
#     xbrl_only: Optional[bool] = True

# async def financial_statement_user_request(request: UserRequest, statement: str, json_structure: str):

#     symbol = request.symbol
#     from_press_release = request.from_press_release
#     xbrl_only = request.xbrl_only

#     print(f"symbol: {symbol}")
#     print(f"from_press_release: {from_press_release}")
#     print(f"xbrl_only: {xbrl_only}")

#     financial_statement_data = []

#     if from_press_release:
#         filenames = get_latest_press_release_filenames(symbol)
#         print(f"LATEST PRESS RELEASE FILENAME: {filenames}")
#         filter = DocumentMetadataFilter(
#             symbol=symbol,
#             filenames=filenames,
#             xbrl_only=xbrl_only
#         )
#     else:
#         filter = DocumentMetadataFilter(
#             symbol=symbol,
#             form_types=[FormType._10_K, FormType._10_Q],
#             xbrl_only=xbrl_only
#         )

#     try:            
#         queries = [
#             ApiQuery(
#                 query=statement,
#                 filter=filter,
#                 sort_order="desc",
#                 limit=1,
#                 top_k=40,
#             ),
#         ]

#         financial_statement_data = await datastore.query(queries)

#         # Prepare system and user messages
#         system_message = f"You are an assistant expert at parsing {statement} data and converting it to a JSON.  ALWAYS GET QUARTERLY DATA if possible and then convert that data to a JSON that looks like this {json_structure}.  You will always convert the ? to the values in millions (for both dollars and shares) from the data provided and you will replace YYYY-MM-DD with the reporting date.  For the frequency put in either annual or quarterly.  Add the correct currency.  Always try for quarterly data if possible.  Never put in comments to this output, only include the JSON data.  ALWAYS infer missing data from other data if it is not explicitly provided."
        
#         # Handle None for datastore query
#         if financial_statement_data is None:
#             serialized_financial_statement_data = []
#         else:
#             serialized_financial_statement_data = [result.to_dict() for result in financial_statement_data if result is not None]

#         json_response_data = json.dumps(serialized_financial_statement_data)

#         user_message = f"Create the {statement} JSON from this context:  {json_response_data}"

#         # Connect to your PostgreSQL database
#         conn = db_manager.get_conn()

#         with conn.cursor(cursor_factory=RealDictCursor) as cursor:

#             # Check for existing rows
#             check_query = """
#             SELECT content FROM fine_tunes 
#             WHERE symbol = %s AND role = 'assistant' AND prompt_name = %s;
#             """
#             cursor.execute(check_query, (symbol, statement,))
#             existing_rows = cursor.fetchall()

#             if existing_rows:

#                 # If rows exist, use their content instead of inserting new data
#                 existing_content = existing_rows[0]['content']
#                 json_response_data = json.dumps(existing_content)

#             else:

#                 # Delete existing rows matching symbol and roles 'system' or 'user'
#                 delete_query = """
#                 DELETE FROM fine_tunes 
#                 WHERE symbol = %s AND prompt_name = %s;
#                 """
#                 cursor.execute(delete_query, (symbol, statement,))

#                 # Find the current largest message_example_id
#                 cursor.execute("SELECT MAX(message_example_id) AS max_row FROM fine_tunes WHERE prompt_name = %s;", (statement,))
#                 max_id_row = cursor.fetchone()
#                 print(f"max_id_row: {max_id_row}")
#                 next_message_example_id = max_id_row['max_row'] + 1 if max_id_row['max_row'] is not None else 1

#                 # Insert rows
#                 insert_query = """
#                 INSERT INTO fine_tunes (symbol, content, prompt_name, role, training_data, message_example_id, created_at, updated_at) VALUES 
#                 (%s, %s, %s, 'system', TRUE, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
#                 (%s, %s, %s, 'user', TRUE, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP);
#                 """
#                 cursor.execute(insert_query, (symbol, system_message, statement, next_message_example_id, symbol, user_message, statement, next_message_example_id))
#                 conn.commit()

#         db_manager.put_conn(conn)

#     except Exception as e:
#         print("Error:", e)

#     return JsonResponse(results=json_response_data)

# @app.post("/fine-tune-income-statement-user-request")
# async def fine_tune_income_statement_user_request(request: UserRequest):
#     return await financial_statement_user_request(request, "Income Statement", """{"period":"YYYY-MM-DD", "revenue":?, "costOfGoodsSold":?, "grossIncome":?, "researchDevelopment":?, "sgaExpense":?, "otherOperatingExpensesTotal":?, "totalOperatingExpense":?, "ebit":?, "interestIncome":?, "interestExpense":?, "interestIncomeExpense":?, "otherIncomeExpenseNet":?, "pretaxIncome":?, "provisionforIncomeTaxes":?, "netIncome":?, "netIncomeAfterTaxes":?, "dilutedAverageSharesOutstanding":?, "dilutedEPS":?, "nonRecurringItems":?, "gainLossOnDispositionOfAssets":?, "minorityInterest" :?, "equityEarningsAffiliates":? , "fiscal_year":?, "fiscal_quarter":?, "frequency": "annual or quarterly","currency":?}""")

# @app.post("/fine-tune-balance-sheet-user-request")
# async def fine_tune_balance_sheet_user_request(request: UserRequest):
#     return await financial_statement_user_request(request, "Balance Sheet", """{"period":"YYYY-MM-DD", "accountsPayable":?, "accountsReceivables":?, "accruedLiability":?, "accumulatedDepreciation":?, "additionalPaidInCapital":?, "cash":?, "cashEquivalents":?, "cashShortTermInvestments":?, "commonStock":?, "currentAssets":?, "currentLiabilities":?, "currentPortionLongTermDebt":?, "inventory":?, "liabilitiesShareholdersEquity":?, "longTermDebt":?, "longTermInvestments":?, "netDebt":?, "otherCurrentAssets":?, "otherCurrentliabilities":?, "otherEquity":?, "otherLiabilities":?, "otherLongTermAssets":?, "otherReceivables":?, "propertyPlantEquipment":?, "retainedEarnings":?, "sharesOutstanding":?, "shortTermDebt":?, "shortTermInvestments":?, "tangibleBookValueperShare":?, "totalAssets":?, "totalDebt":?, "totalEquity":?, "totalLiabilities":?, "totalReceivables":?, "fiscal_year":?, "fiscal_quarter":?, "frequency": "annual or quarterly","currency":?}""")

# @app.post("/fine-tune-cash-flow-user-request")
# async def fine_tune_cash_flow_user_request(request: UserRequest):
#     return await financial_statement_user_request(request, "Cash Flow", """{"period":"YYYY-MM-DD", "capex":?, "cashDividendsPaid":?, "cashTaxesPaid":?, "changeinCash":?, "changesinWorkingCapital":?, "depreciationAmortization":?, "fcf":?, "issuanceReductionCapitalStock":?, "issuanceReductionDebtNet":?, "netCashFinancingActivities":?, "netIncomeStartingLine":?, "netInvestingCashFlow":?, "netOperatingCashFlow":?, "otherFundsFinancingItems":?, "otherFundsNonCashItems":?, "otherInvestingCashFlowItemsTotal":?, "stockBasedCompensation":?, "fiscal_year":?, "fiscal_quarter":?, "frequency": "annual or quarterly","currency":?}""")



# class AssistantResponse(BaseModel):
#     symbol: str
#     assistant_message: str

# async def financial_statement_assistant_response(request: AssistantResponse, statement: str):

#     assistant_message = request.assistant_message
#     symbol = request.symbol

#     try:
#         # Connect to your PostgreSQL database
#         conn = db_manager.get_conn()

#         with conn.cursor(cursor_factory=RealDictCursor) as cursor:

#             # Delete existing rows matching symbol and role 'assistant'
#             delete_query = """
#             DELETE FROM fine_tunes 
#             WHERE symbol = %s AND prompt_name = %s AND role = 'assistant';
#             """
#             cursor.execute(delete_query, (symbol, statement,))

#             # Find the message_example_id for a matching 'user' role entry
#             select_query = """
#             SELECT message_example_id FROM fine_tunes 
#             WHERE symbol = %s AND prompt_name = %s AND role = 'user'
#             ORDER BY created_at DESC
#             LIMIT 1;
#             """
#             cursor.execute(select_query, (symbol, statement,))
#             match_row = cursor.fetchone()

#             # Proceed only if a matching user entry exists
#             if match_row:
#                 message_example_id = match_row['message_example_id']

#                 insert_query = """
#                 INSERT INTO fine_tunes (symbol, content, prompt_name, role, training_data, message_example_id, created_at, updated_at) VALUES 
#                 (%s, %s, %s, 'assistant', TRUE, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP);
#                 """
#                 cursor.execute(insert_query, (symbol, assistant_message, statement, message_example_id))
#                 conn.commit()
#             else:
#                 print("No matching user entry found for symbol:", symbol)

#         db_manager.put_conn(conn)

#     except Exception as e:
#         print("Error:", e)

#     return {"message": f"{symbol} {statement} inserted successfully"}

# @app.post("/fine-tune-income-statement-assistant-response")
# async def fine_tune_income_statement_assistant_response(request: AssistantResponse):
#     return await financial_statement_assistant_response(request, "Income Statement")

# @app.post("/fine-tune-balance-sheet-assistant-response")
# async def fine_tune_balance_sheet_assistant_response(request: AssistantResponse):
#     return await financial_statement_assistant_response(request, "Balance Sheet")

# @app.post("/fine-tune-cash-flow-assistant-response")
# async def fine_tune_cash_flow_assistant_response(request: AssistantResponse):
#     return await financial_statement_assistant_response(request, "Cash Flow")



# async def list_financial_statement(statement: str):
#     symbols = []  # Initialize outside try block to ensure scope visibility
#     try:
#         # Connect to your PostgreSQL database
#         conn = db_manager.get_conn()
#         try:
#             with conn.cursor(cursor_factory=RealDictCursor) as cursor:
#                 cursor.execute("""
#                 SELECT symbol FROM fine_tunes 
#                 WHERE prompt_name = %s AND role = 'assistant'
#                 ORDER BY created_at DESC;
#                 """, (statement,))
#                 symbols = [row['symbol'] for row in cursor.fetchall()]
#         finally:
#             db_manager.put_conn(conn)
#     except Exception as e:
#         print("Error:", e)

#     return {"message": f"The following symbols have a {statement} saved: {symbols}"}

# @app.get("/list-fine-tune-income-statement")
# async def list_fine_tune_income_statement():
#     return await list_financial_statement("Income Statement")

# @app.get("/list-fine-tune-balance-sheet")
# async def list_fine_tune_balance_sheet():
#     return await list_financial_statement("Balance Sheet")

# @app.get("/list-fine-tune-cash-flow")
# async def list_fine_tune_cash_flow():
#     return await list_financial_statement("Cash Flow")


# class DeleteRequest(BaseModel):
#     symbol: str

# async def delete_financial_statement(request: DeleteRequest, statement: str):
#     symbol = request.symbol

#     try:
#         # Connect to your PostgreSQL database
#         conn = db_manager.get_conn()
#         try:
#             with conn.cursor(cursor_factory=RealDictCursor) as cursor:
#                 cursor.execute("""
#                 DELETE FROM fine_tunes 
#                 WHERE symbol = %s AND prompt_name = %s;
#                 """, (symbol, statement,))
#                 conn.commit()
#         finally:
#             db_manager.put_conn(conn)
#     except Exception as e:
#         print("Error:", e)

#     return {"message": f"{symbol} {statement} deleted successfully"}

# @app.post("/delete-fine-tune-income-statement")
# async def delete_fine_tune_income_statement(request: DeleteRequest):
#     return await delete_financial_statement(request, "Income Statement")

# @app.post("/delete-fine-tune-balance-sheet")
# async def delete_fine_tune_balance_sheet(request: DeleteRequest):
#     return await delete_financial_statement(request, "Balance Sheet")

# @app.post("/delete-fine-tune-cash-flow")
# async def delete_fine_tune_cash_flow(request: DeleteRequest):
#     return await delete_financial_statement(request, "Cash Flow")



@app.get(
    "/analyze",
)
async def analyze_main(
    symbol: str = Query(...)
):


    key_risks = []
    key_risk_urls = []
    try:

        queries = [
            ApiQuery(
                query="What are the top business performance risks the company is facing based on analyst questions and management commentary?",
                filter=DocumentMetadataFilter(
                    symbol=symbol,
                    form_types=[FormType._20_F, FormType._10_K, FormType._10_Q, FormType.earnings_transcript]
                ),
                sort_order="desc",
                limit=2,
                top_k=15
            ),
        ]

        # Handle None for datastore query
        documents = await datastore.query(queries)
        # print(f"DOCUMENTS: {documents}")

        if documents is None:
            documents = []
        key_risks = extract_texts(documents)
        key_risk_urls = extract_urls(documents)

        # print(f"KEY RISKS: {key_risks}")

    except Exception as e:
        print("Error:", e)


    key_opportunities = []
    key_opportunity_urls = []
    try:

        queries = [
            ApiQuery(
                query="What are the top opportunities the company has based on analyst questions and management commentary?",
                filter=DocumentMetadataFilter(
                    symbol=symbol,
                    form_types=[FormType._20_F, FormType._10_K, FormType._10_Q, FormType.earnings_transcript]
                ),
                sort_order="desc",
                limit=2,
                top_k=15
            ),
        ]

        # Handle None for datastore query
        documents = await datastore.query(queries)

        if documents is None:
            documents = []
        key_opportunities = extract_texts(documents)
        key_opportunity_urls = extract_urls(documents)

        # print(f"KEY OPPORTUNITIES: {key_opportunities}")

    except Exception as e:
        print("Error:", e)


    forward_guidance = []
    forward_guidance_urls = []
    try:

        queries = [
            ApiQuery(
                query="Management Guidance on Future Performance",
                filter=DocumentMetadataFilter(
                    symbol=symbol,
                    form_types=[FormType._20_F, FormType._10_K, FormType._10_Q, FormType.earnings_transcript]
                ),
                sort_order="desc",
                limit=2,
                top_k=10
            )
        ]

        # Handle None for datastore query
        documents = await datastore.query(queries)

        if documents is None:
            documents = []
        forward_guidance = extract_texts(documents)
        forward_guidance_urls = extract_urls(documents)

        # print(f"FORWARD GUIDANCE: {forward_guidance}")

    except Exception as e:
        print("Error:", e)


    # try:
    #     news = finnhub_client.company_news(symbol, _from=(datetime.utcnow() - timedelta(days=5)).strftime('%Y-%m-%d'), to=datetime.utcnow().strftime('%Y-%m-%d'))
    #     if news is None:
    #         news = []  # Default value if None
    #     print(f"NEWS: {news}")
    # except Exception as e:
    #     print("Error:", e)

    income_statements = {"financials": []}  # Default value if None
    try:
        income_statements = finnhub_client.financials(symbol, "ic", "quarterly")
        if income_statements is None or income_statements["financials"] is None:
            income_statements = {"financials": []}  # Default value if None
        else:
            income_statements = income_statements["financials"][0]
        # print(income_statements)
    except Exception as e:
        print("Error:", e)

    cash_flow = {"financials": []}  # Default value if None
    try:
        cash_flow = finnhub_client.financials(symbol, "cf", "quarterly")
        if cash_flow is None or cash_flow["financials"] is None:
            cash_flow = {"financials": []}  # Default value if None
        else:
            cash_flow = cash_flow["financials"][0]
        # print(cash_flow)
    except Exception as e:
        print("Error:", e)

    balance_sheet = {"financials": []}  # Default value if None
    try:
        balance_sheet = finnhub_client.financials(symbol, "bs", "quarterly")
        if balance_sheet is None or balance_sheet["financials"] is None:
            balance_sheet = {"financials": []}  # Default value if None
        else:
            balance_sheet = balance_sheet["financials"][0]
        # print(balance_sheet)
    except Exception as e:
        print("Error:", e)

    annual_income_statements = {"financials": []}  # Default value if None
    try:
        annual_income_statements = finnhub_client.financials(symbol, "ic", "annual")
        if annual_income_statements is None or annual_income_statements["financials"] is None:
            annual_income_statements = {"financials": []}  # Default value if None
        else:
            annual_income_statements = annual_income_statements["financials"][0]
        # print(annual_income_statements)
    except Exception as e:
        print("Error:", e)

    key_ratios = {"metric": {}}  # Default value if None
    try:
        key_ratios = finnhub_client.company_basic_financials(symbol, "all")
        if key_ratios is None:
            key_ratios = {"metric": {}}  # Default value if None
        else:
            key_ratios = extract_financial_ratios(key_ratios["metric"])
        # print(key_ratios)
    except Exception as e:
        print("Error:", e)


    sector_ratios = {"metric": {}}  # Default value if None
    try:
        raw_sector_ratios = finnhub_client.sector_metric('NA')
        if raw_sector_ratios is None:
            sector_ratios = {"metric": {}}  # Default value if None
        else:
            sector_ratios = extract_financial_ratios(extract_metrics(raw_sector_ratios, 'NA', get_sector_by_symbol(symbol)))
        # print(sector_ratios)
    except Exception as e:
        print("Error:", e)


    # Prefixing 'sector_' to the keys of sector_ratios and merging it into key_ratios
    for key, value in sector_ratios.items():
        prefixed_key = f'sector_{key}'
        key_ratios[prefixed_key] = value

    # revenue_estimates = {"data": []}  # Default value if None
    # try:
    #     revenue_estimates = finnhub_client.company_revenue_estimates(symbol, "quarterly")
    #     if revenue_estimates is None:
    #         revenue_estimates = {"data": []}  # Default value if None
    #     print(revenue_estimates)
    # except Exception as e:
    #     print("Error:", e)

    # ebit_estimates = {"data": []}  # Default value if None
    # try:
    #     ebit_estimates = finnhub_client.company_ebit_estimates(symbol, "quarterly")
    #     if ebit_estimates is None:
    #         ebit_estimates = {"data": []}  # Default value if None
    #     print(ebit_estimates)
    # except Exception as e:
    #     print("Error:", e)

    eps_estimates = {"data": []}  # Default value if None
    try:
        eps = finnhub_client.company_eps_estimates(symbol, "annual")
        if eps is None:
            eps_estimates = {"data": []}  # Default value if None
        else:
            eps_estimates = [entry for entry in eps['data'] if entry['year'] == annual_income_statements['year'] + 1][0]

        # print(eps_estimates)
    except Exception as e:
        print("Error:", e)

    price_target={}
    try:
        price_target = finnhub_client.price_target(symbol)
        # print(price_target)
    except Exception as e:
        print("Error:", e)

    recommendation_trends=[]
    try:
        recommendation_trends = finnhub_client.recommendation_trends(symbol)[0]
        # print(recommendation_trends)
    except Exception as e:
        print("Error:", e)

    dividends=[]
    try:
        dividends = finnhub_client.stock_dividends(symbol, _from=(datetime.utcnow() - timedelta(days=2*365)).strftime('%Y-%m-%d'), to=datetime.utcnow().strftime('%Y-%m-%d'))
        # print(dividends)
    except Exception as e:
        print("Error:", e)

    insider_transactions={"data": []}
    try:
        insider_transactions = finnhub_client.stock_insider_transactions(symbol, (datetime.utcnow() - timedelta(days=60)).strftime('%Y-%m-%d'), datetime.utcnow().strftime('%Y-%m-%d'))
        # print(insider_transactions)
    except Exception as e:
        print("Error:", e)

    quote={}
    try:
        quote = finnhub_client.quote(symbol)
        # print(quote)
    except Exception as e:
        print("Error:", e)


    # Merge the three arrays into a single array
    risks_opportunities_and_guidance = key_risks + key_opportunities + forward_guidance
    risks_opportunities_and_guidance_urls = list(set(key_risk_urls + key_opportunity_urls + forward_guidance_urls))

    # Construct the final response
    response_data = {
        "quarterly_income_statement": income_statements,
        "quarterly_cash_flow": cash_flow,
        "quarterly_balance_sheet": balance_sheet,
        "annual_income_statement": annual_income_statements,
        "eps_estimates": eps_estimates,
        "price_target": price_target,
        "recommendation_trends": recommendation_trends,
        "dividends": dividends,
        "insider_transactions": insider_transactions["data"][:20],
        "current_price": quote,
        "key_financials": key_ratios,
        "risks_opportunities_and_guidance": risks_opportunities_and_guidance,
        "source_document_urls": risks_opportunities_and_guidance_urls,
    }

    json_response_data = json.dumps(response_data)
    print(f"/analyze output for stock {symbol}: {json_response_data}")

    return JsonResponse(results=json_response_data)




# @app.post(
#     "/query",
#     response_model=QueryResponse,
# )
# async def query_main(
#     request: QueryRequest = Body(...),
# ):
#     try:
#         print("GOT QUERY REQUEST:")
#         print(request)
#         results = await datastore.query(


@app.post(
    "/filenames"
)
async def filename_main(
    request: FilenamesRequest = Body(...)
):
    try:
        # print(request)
        symbols=request.symbols
        conn = db_manager.get_conn()
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            query = """
                SELECT filename, description, form_type, fiscal_year, fiscal_quarter, 
                TO_CHAR(published_date, 'YYYY-MM-DD') AS published_date_str
                FROM source_file_metadata
                WHERE form_type != 'private_document' AND in_vector_db = true AND symbol = ANY(%s)
                ORDER BY published_date desc
            """
            cursor.execute(query, (symbols,))
            filenames = cursor.fetchall()

        json_response_data = json.dumps(filenames)
        print(f"/filenames output for stocks {symbols}: {json_response_data}")
        return JsonResponse(results=json_response_data)

    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail="Internal Service Error")
    finally:
        if conn:
            db_manager.put_conn(conn)


@app.get(
    "/financial-statements",
)
async def financial_statements_main(
    statement: str = Query(...),
    freq: str = Query(...),
    symbol: str = Query(...),
):
    try:
        body = finnhub_client.financials(symbol, statement, freq)
        return JsonResponse(results=json.dumps(body))
    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail="Internal Service Error")


@app.get(
    "/quote",
)
async def quote_main(
    symbol: str = Query(...),
):
    try:
        body = finnhub_client.quote(symbol)
        return JsonResponse(results=json.dumps(body))

    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail="Internal Service Error")


@app.get(
    "/metrics",
)
async def metrics_main(
    symbol: str = Query(...),
):
    try:
        body = finnhub_client.company_basic_financials(symbol, "all")
        key_ratios = body["metric"]
        print(json.dumps(key_ratios))
        return JsonResponse(results=json.dumps(key_ratios))

    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail="Internal Service Error")


# Function to sort each list of dictionaries by the 'period' key
def sort_by_period(data):
    for key in data:
        data[key].sort(key=lambda x: x['period'])
    return data


@app.get(
    "/metrics-time-series",
)
async def metrics_time_series_main(
    symbol: str = Query(...),
    freq: str = Query(...)
):
    try:
        body = finnhub_client.company_basic_financials(symbol, "all")
        if freq == 'quarterly':
            key_ratios = sort_by_period(body["series"]["quarterly"])
        else:
            key_ratios = sort_by_period(body["series"]["annual"])
        print(json.dumps(key_ratios))
        return JsonResponse(results=json.dumps(key_ratios))

    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail="Internal Service Error")


@app.get(
    "/candles"
)
async def candles_main(
    symbol: str = Query(...),
    resolution: str = Query(...),
    from_timestamp: int = Query(...),
    to_timestamp: int = Query(...)
):
    try:
        body = finnhub_client.stock_candles(symbol, resolution, from_timestamp, to_timestamp)
        return JsonResponse(results=json.dumps(body))

    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail="Internal Service Error")


@app.get(
    "/company-profile",
)
async def company_profile_main(
    symbol: str = Query(...),
):
    try:
        body = finnhub_client.company_profile(symbol=symbol)
        return JsonResponse(results=json.dumps(body))

    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail="Internal Service Error")



@app.get("/dividend")
async def dividend_main(
    symbol: str = Query(...),
):
    try:
        body = finnhub_client.stock_dividends(symbol, _from=(datetime.utcnow() - timedelta(days=10*365)).strftime('%Y-%m-%d'), to=datetime.utcnow().strftime('%Y-%m-%d'))
        return JsonResponse(results=json.dumps(body))
    except ValueError as ve:
        print(f"Timestamp conversion error: {ve}")
        raise HTTPException(status_code=400, detail="Invalid timestamp format")
    except Exception as e:
        print(f"Error in dividend_main: {e}")
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/price-metric")
async def price_metric_main(
    symbol: str = Query(...),
):
    try:
        body = finnhub_client.price_metrics(symbol, datetime.utcnow().strftime('%Y-%m-%d'))
        return JsonResponse(results=json.dumps(body))
    except ValueError as ve:
        print(f"Timestamp conversion error: {ve}")
        raise HTTPException(status_code=400, detail="Invalid timestamp format")
    except Exception as e:
        print(f"Error in price_metric_main: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/revenue-estimates",
)
async def revenue_estimates_main(
    symbol: str = Query(...),
    freq: str = Query(...),
):
    try:
        body = finnhub_client.company_revenue_estimates(symbol, freq)
        return JsonResponse(results=json.dumps(body))

    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail="Internal Service Error")


@app.get(
    "/revenue-breakdown",
)
async def revenue_breakdown_main(
    symbol: str = Query(...),
):
    try:
        body = finnhub_client.stock_revenue_breakdown(symbol)
        data = body["data"][0]["breakdown"]
        print(json.dumps(data))
        return JsonResponse(results=json.dumps(data))

    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail="Internal Service Error")


@app.get(
    "/eps-estimates",
)
async def eps_estimates_main(
    symbol: str = Query(...),
    freq: str = Query(...),
):
    try:
        body = finnhub_client.company_eps_estimates(symbol, freq)
        return JsonResponse(results=json.dumps(body))

    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail="Internal Service Error")


@app.get(
    "/ebitda-estimates",
)
async def ebitda_estimates_main(
    symbol: str = Query(...),
    freq: str = Query(...),
):
    try:
        body = finnhub_client.company_ebitda_estimates(symbol, freq)
        return JsonResponse(results=json.dumps(body))

    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail="Internal Service Error")


@app.get(
    "/ebit-estimates",
)
async def ebit_estimates_main(
    symbol: str = Query(...),
    freq: str = Query(...),
):
    try:
        body = finnhub_client.company_ebit_estimates(symbol, freq)
        return JsonResponse(results=json.dumps(body))

    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail="Internal Service Error")


@app.get(
    "/price-targets",
)
async def price_targets_main(
    symbol: str = Query(...)
):
    try:
        body = finnhub_client.price_target(symbol)
        return JsonResponse(results=json.dumps(body))

    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail="Internal Service Error")


@app.get(
    "/recommendation-trends",
)
async def recommendation_trends_main(
    symbol: str = Query(...)
):
    try:
        body = finnhub_client.recommendation_trends(symbol)
        return JsonResponse(results=json.dumps(body))

    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail="Internal Service Error")

@app.get(
    "/earnings-calendar",
)
async def earnings_calendar_main(
    symbol: str = Query(...),
):
    try:
        body = finnhub_client.earnings_calendar(_from=(datetime.utcnow() - timedelta(days=5*365)).strftime('%Y-%m-%d'), to=(datetime.utcnow() + timedelta(days=5*365)), symbol=symbol)
        return JsonResponse(results=json.dumps(body))

    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail="Internal Service Error")


@app.get(
    "/insider-transactions",
)
async def insider_transactions(
    symbol: str = Query(...),
):
    try:
        body = finnhub_client.stock_insider_transactions(symbol, (datetime.utcnow() - timedelta(days=365)).strftime('%Y-%m-%d'), datetime.utcnow().strftime('%Y-%m-%d'))["data"][:1000]
        print(body)
        return JsonResponse(results=json.dumps(body))

    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail="Internal Service Error")


@app.delete(
    "/delete",
    response_model=DeleteResponse,
)
async def delete(
    request: DeleteRequest = Body(...),
):
    if not (request.ids or request.filter or request.delete_all):
        raise HTTPException(
            status_code=400,
            detail="One of ids, filter, or delete_all is required",
        )
    try:
        success = await datastore.delete(
            ids=request.ids,
            filter=request.filter,
            delete_all=request.delete_all,
        )
        return DeleteResponse(success=success)
    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail="Internal Service Error")


@app.on_event("startup")
async def startup():
    global datastore
    datastore = await get_datastore()


def start():
    uvicorn.run("server.main:app", host="0.0.0.0", port=8000, reload=True)
