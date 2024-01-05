import os
from typing import Optional, List
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
        print("GOT QUERY REQUEST:")
        print(request)
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
        'price_book_ratio': 'pbQuarterly'  # Price to Book Ratio
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


@app.get(
    "/analyze",
)
async def analyze_main(
    symbol: str = Query(...)
):

    key_risks = []
    key_opportunities = []
    forward_guidance = []
    try:

        queries = [
            ApiQuery(
                query="Top Risks",
                filter=DocumentMetadataFilter(
                    symbol=symbol,
                    form_types=[FormType._20_F, FormType._10_K, FormType._10_Q]
                ),
                sort_order="desc",
                limit=2,
                top_k=15
            ),
            ApiQuery(
                query="Top Opportunities",
                filter=DocumentMetadataFilter(
                    symbol=symbol,
                    form_types=[FormType._20_F, FormType._10_K, FormType._10_Q]
                ),
                sort_order="desc",
                limit=2,
                top_k=15
            ),
            ApiQuery(
                query="Forward Looking Guidance",
                filter=DocumentMetadataFilter(
                    symbol=symbol,
                    form_types=[FormType.earnings_transcript]
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
        query_response_dict = QueryResponse(results=documents).to_dict()
        key_risks = query_response_dict['results'][0]['results'] if len(query_response_dict['results']) > 0 else []
        key_opportunities = query_response_dict['results'][1]['results'] if len(query_response_dict['results']) > 1 else []
        forward_guidance = query_response_dict['results'][2]['results'] if len(query_response_dict['results']) > 2 else []

        print(f"KEY RISKS: {key_risks}")
        print(f"KEY OPPORTUNITIES: {key_opportunities}")
        print(f"FORWARD GUIDANCE: {forward_guidance}")

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
        print(income_statements)
    except Exception as e:
        print("Error:", e)

    cash_flow = {"financials": []}  # Default value if None
    try:
        cash_flow = finnhub_client.financials(symbol, "cf", "quarterly")
        if cash_flow is None or cash_flow["financials"] is None:
            cash_flow = {"financials": []}  # Default value if None
        print(cash_flow)
    except Exception as e:
        print("Error:", e)

    balance_sheet = {"financials": []}  # Default value if None
    try:
        balance_sheet = finnhub_client.financials(symbol, "bs", "quarterly")
        if balance_sheet is None or balance_sheet["financials"] is None:
            balance_sheet = {"financials": []}  # Default value if None
        print(balance_sheet)
    except Exception as e:
        print("Error:", e)

    annual_income_statements = {"financials": []}  # Default value if None
    try:
        annual_income_statements = finnhub_client.financials(symbol, "ic", "annual")
        if annual_income_statements is None or annual_income_statements["financials"] is None:
            annual_income_statements = {"financials": []}  # Default value if None
        print(annual_income_statements)
    except Exception as e:
        print("Error:", e)

    key_ratios = {"metric": {}}  # Default value if None
    try:
        key_ratios = finnhub_client.company_basic_financials(symbol, "all")
        if key_ratios is None:
            key_ratios = {"metric": {}}  # Default value if None
        else:
            key_ratios = extract_financial_ratios(key_ratios["metric"])
        print(key_ratios)
    except Exception as e:
        print("Error:", e)

    revenue_estimates = {"data": []}  # Default value if None
    try:
        revenue_estimates = finnhub_client.company_revenue_estimates(symbol, "quarterly")
        if revenue_estimates is None:
            revenue_estimates = {"data": []}  # Default value if None
        print(revenue_estimates)
    except Exception as e:
        print("Error:", e)

    ebit_estimates = {"data": []}  # Default value if None
    try:
        ebit_estimates = finnhub_client.company_ebit_estimates(symbol, "quarterly")
        if ebit_estimates is None:
            ebit_estimates = {"data": []}  # Default value if None
        print(ebit_estimates)
    except Exception as e:
        print("Error:", e)

    eps_estimates = {"data": []}  # Default value if None
    try:
        eps_estimates = finnhub_client.company_eps_estimates(symbol, "quarterly")
        if eps_estimates is None:
            eps_estimates = {"data": []}  # Default value if None
        print(eps_estimates)
    except Exception as e:
        print("Error:", e)

    price_target={}
    try:
        price_target = finnhub_client.price_target(symbol)
        print(price_target)
    except Exception as e:
        print("Error:", e)

    recommendation_trends=[]
    try:
        recommendation_trends = finnhub_client.recommendation_trends(symbol)
        print(recommendation_trends)
    except Exception as e:
        print("Error:", e)

    dividends=[]
    try:
        dividends = finnhub_client.stock_dividends(symbol, _from=(datetime.utcnow() - timedelta(days=5*365)).strftime('%Y-%m-%d'), to=datetime.utcnow().strftime('%Y-%m-%d'))
        print(dividends)
    except Exception as e:
        print("Error:", e)

    insider_transactions={}
    try:
        insider_transactions = finnhub_client.stock_insider_transactions(symbol, (datetime.utcnow() - timedelta(days=60)).strftime('%Y-%m-%d'), datetime.utcnow().strftime('%Y-%m-%d'))
        print(insider_transactions)
    except Exception as e:
        print("Error:", e)

    quote={}
    try:
        quote = finnhub_client.quote(symbol)
        print(quote)
    except Exception as e:
        print("Error:", e)


    # Construct the final response
    response_data = {
        "quarterly_income_statements": income_statements["financials"][:5],
        "quarterly_cash_flow": cash_flow["financials"][:5],
        "quarterly_balance_sheet": balance_sheet["financials"][:5],
        "annual_income_statements": annual_income_statements["financials"][:5],
        "revenue_estimates": revenue_estimates["data"][:3],
        "ebit_estimates": ebit_estimates["data"][:3],
        "eps_estimates": eps_estimates["data"][:3],
        "price_target": price_target,
        "recommendation_trends": recommendation_trends[:5],
        "dividends": dividends,
        "insider_transactions": insider_transactions["data"][:20],
        "current_price": quote,
        "key_ratios": key_ratios,
        "key_risks": key_risks,
        "key_opportunities": key_opportunities,
        "forward_guidance": forward_guidance,
    }

    print(response_data)
    json_response_data = json.dumps(response_data)

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
        print(request)
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
        return JsonResponse(results=json_response_data)

    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail="Internal Service Error")
    finally:
        if conn:
            db_manager.put_conn(conn)


# Define a new request model
class SearchRequest(BaseModel):
    market_cap_min: Optional[int] = None
    market_cap_max: Optional[int] = None
    country: Optional[str] = None
    industry: Optional[str] = None
    query: str
    published_before_date: Optional[date] = None
    published_after_date: Optional[date] = None
    published_before_days_before_current: Optional[int] = None
    published_after_days_before_current: Optional[int] = None
    form_types: Optional[List[str]] = None 

@app.post("/search")
async def search_main(request: SearchRequest = Body(...)):
    conn = db_manager.get_conn()
    try:

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

        # Construct the final JOIN query
        query = f"""
            SELECT DISTINCT symbol 
            FROM stocks 
            WHERE {" AND ".join(query_parts)}
        """

        print(query)
        print(params)

        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query, tuple(params))
            symbols = [row['symbol'] for row in cursor.fetchall()]
        print(symbols)

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

        print(query)
        print(params)

        # Continue with the existing logic using the obtained symbols or all symbols if none were filtered
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query, tuple(params))
            filenames = [row['filename'] for row in cursor.fetchall()]

        print(f"FILENAMES: {filenames}")

        queries = [
            ApiQuery(
                query=request.query,
                filter=DocumentMetadataFilter(
                    filenames=filenames,
                ),
                top_k=50
            )
        ]

        # Handle None for datastore query
        document_results = await datastore.query(queries)
        if document_results is None:
            serialized_document_results = []
        else:
            serialized_document_results = [result.dict() for result in document_results if result is not None]

        json_response_data = json.dumps(serialized_document_results)

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
