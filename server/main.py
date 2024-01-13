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
        print(request)
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


@app.get(
    "/analyze",
)
async def analyze_main(
    symbol: str = Query(...)
):


    key_risks = []
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
        # print(f"KEY RISKS: {key_risks}")

    except Exception as e:
        print("Error:", e)


    key_opportunities = []
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

        # print(f"KEY OPPORTUNITIES: {key_opportunities}")

    except Exception as e:
        print("Error:", e)


    forward_guidance = []
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
                WHERE in_vector_db = true AND symbol = ANY(%s)
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


class FinancialFilter(BaseModel):
    financial_name: str
    comparison: str  # ">=" or "<="
    value: float


# Define a new request model
class SearchRequest(BaseModel):
    market_cap_min: Optional[int] = None
    market_cap_max: Optional[int] = None
    country: Optional[str] = None
    industry: Optional[str] = None
    symbols: Optional[List[str]] = None
    query: Union[List[str], str]  # Depending on the assistant, it will send either a string or an list of strings.
    top_k: Optional[int] = 50 
    financial_filters: Optional[List[FinancialFilter]] = None
    published_before_date: Optional[date] = None
    published_after_date: Optional[date] = None
    published_before_days_before_current: Optional[int] = None
    published_after_days_before_current: Optional[int] = None
    form_types: Optional[List[str]] = None 

@app.post("/search")
async def search_main(request: SearchRequest = Body(...)):
    conn = db_manager.get_conn()
    try:

        # Check if the query input is a string
        if isinstance(request.query, str):
            # If the string is empty, replace it with the default query
            if request.query.strip() == '':
                request.query = ["General information about the company"]
            else:
                # If it's a non-empty string, convert it to a list
                request.query = [request.query]
        elif isinstance(request.query, list):
            # If the query is a list, replace any empty strings with the default query
            request.query = [q if q.strip() != '' else "General information about the company" for q in request.query]

        # Construct the query for stocks table with a JOIN on source_file_metadata
        query_parts = ["1=1"]  # This is a placeholder to simplify query building
        params = []

        # Add conditions for existing filters
        if request.market_cap_min is not None or request.market_cap_max is not None:
            market_cap_min = request.market_cap_min if request.market_cap_min is not None else 0
            market_cap_max = request.market_cap_max if request.market_cap_max is not None else 1e12  # A large number to represent 'infinity'
            query_parts.append("market_cap BETWEEN %s AND %s")
            params.extend([market_cap_min, market_cap_max])

        if request.country:
            query_parts.append("country = %s")
            params.append(request.country)

        if request.industry:
            query_parts.append("industry = %s")
            params.append(request.industry)

        if request.symbols:
            symbols_placeholders = ', '.join(['%s'] * len(request.symbols))
            query_parts.append(f"symbol IN ({symbols_placeholders})")
            params.extend(request.symbols)

        if request.financial_filters:
            # Ensure to include the basic_financials table in the join
            query_parts.append("stocks.symbol = basic_financials.symbol")

            for filter in request.financial_filters:
                # Directly use the comparison operator from the input
                query_parts.append(f"basic_financials.{filter.financial_name} {filter.comparison} %s")
                params.append(filter.value)

            # Adjust the query to include the JOIN with basic_financials
            query = f"""
                SELECT DISTINCT stocks.symbol 
                FROM stocks 
                JOIN basic_financials ON {" AND ".join(query_parts)}
            """
        else:

            # Construct the final JOIN query
            query = f"""
                SELECT DISTINCT symbol 
                FROM stocks 
                WHERE {" AND ".join(query_parts)}
            """

        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query, tuple(params))
            symbols = [row['symbol'] for row in cursor.fetchall()]

        query_parts = ["1=1"]  # This is a placeholder to simplify query building
        params = []

        # Add conditions for new filters (joining with source_file_metadata table)
        if request.published_before_date:
            query_parts.append("published_date <= %s")
            params.append(request.published_before_date)

        if request.published_before_days_before_current:
            days_before = request.published_before_days_before_current
            target_date = datetime.now() - timedelta(days=days_before)
            query_parts.append("published_date <= %s")
            params.append(target_date.date())  # Use .date() to get the date part without time

        if request.published_after_date:
            query_parts.append("published_date >= %s")
            params.append(request.published_after_date)

        if request.published_after_days_before_current:
            days_before = request.published_after_days_before_current
            target_date = datetime.now() - timedelta(days=days_before)
            query_parts.append("published_date >= %s")
            params.append(target_date.date())  # Use .date() to get the date part without time

        if request.form_types:
            form_types_placeholders = ', '.join(['%s'] * len(request.form_types))
            query_parts.append(f"form_type IN ({form_types_placeholders})")
            params.extend(request.form_types)

        if symbols:
            query = f"""
                SELECT filename
                FROM source_file_metadata
                WHERE in_vector_db = true AND symbol IN ('{"', '".join(symbols)}') AND {" AND ".join(query_parts)}
            """
        else:
            query = f"""
                SELECT filename
                FROM source_file_metadata
                WHERE in_vector_db = true AND {" AND ".join(query_parts)}
            """

        # Execute query to get filenames
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query, tuple(params))
            filenames = [row['filename'] for row in cursor.fetchall()]

        # Execute the datastore query
        queries = [
            ApiQuery(
                query=q,
                filter=DocumentMetadataFilter(
                    filenames=filenames,
                ),
                top_k=request.top_k 
            ) for q in request.query
        ]
        document_results = await datastore.query(queries)

        # Handle None for datastore query
        if document_results is None:
            serialized_document_results = []
        else:
            serialized_document_results = [result.to_dict() for result in document_results if result is not None]

        # Extract unique symbols from document results
        unique_symbols = set()
        for result in serialized_document_results:
            for doc in result['results']:
                if doc['metadata']['symbol']:
                    unique_symbols.add(doc['metadata']['symbol'])

        # Fetch financial data for these symbols
        financial_query = """
            SELECT 
                eps_growth_5y,
                eps_growth_ttm_yoy,
                current_dividend_yield_ttm,
                current_ratio_annual,
                dividend_growth_rate_5y,
                ebitda_cagr_5y,
                gross_margin_5y,
                gross_margin_ttm,
                insider_dollars_bought_one_month,
                insider_dollars_bought_three_months,
                insider_dollars_bought_ttm,
                insider_dollars_sold_one_month,
                insider_dollars_sold_three_months,
                insider_dollars_sold_ttm,
                insider_dollars_buy_sell_ratio_one_month,
                insider_dollars_buy_sell_ratio_three_months,
                insider_dollars_buy_sell_ratio_ttm,
                long_term_debt_equity_annual,
                market_capitalization,
                net_profit_margin_5y,
                net_profit_margin_ttm,
                operating_margin_5y,
                operating_margin_ttm,
                payout_ratio_ttm,
                pe_ttm,
                pfcf_share_ttm,
                ps_ttm,
                quick_ratio_annual,
                revenue_growth_5y,
                revenue_growth_ttm_yoy,
                roa_5y,
                roa_ttm,
                roe_5y,
                roe_ttm,
                roi_5y,
                roi_ttm,
                total_debt_total_equity_annual 
            FROM basic_financials 
            WHERE symbol IN %s
        """
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(financial_query, (tuple(unique_symbols),))
            financial_data = cursor.fetchall()

        # Combine the document results with the financial data
        combined_results = []
        for result in serialized_document_results:
            for doc in result['results']:
                symbol = doc['metadata']['symbol']
                financial_info = next((item for item in financial_data if item['symbol'] == symbol), None)
                combined_result = {
                    'document_data': doc,
                    'financial_data': financial_info
                }
                combined_results.append(combined_result)

        json_response_data = json.dumps(combined_results)
        print(f"/search output for request {request}: {json_response_data}")

        return JsonResponse(results=json_response_data)

    except ValidationError as e:
        print("Validation Error:", e.json())
        raise HTTPException(status_code=400, detail=f"Input validation error: {e}")
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
