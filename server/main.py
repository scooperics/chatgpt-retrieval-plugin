import os
from typing import Optional
import json
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Depends, Body, UploadFile
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from fastapi import Query
import pandas as pd
import pandas_ta as ta
import time


import finnhub
from datetime import datetime, timedelta

from models.api import (
    DeleteRequest,
    DeleteResponse,
    QueryRequest,
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


@app.get(
    "/analyze",
)
async def analyze_main(
    symbol: str = Query(...)
):
    try:

        # Please help me create queries
        queries = [
            ApiQuery(
                query="Top Risks",
                filter=DocumentMetadataFilter(
                    symbol=symbol,
                    form_types=[FormType._10_K, FormType._10_Q]
                ),
                sort_order="desc",
                limit=1,
                top_k=5
            ),
            ApiQuery(
                query="Top Opportunities",
                filter=DocumentMetadataFilter(
                    symbol=symbol,
                    form_types=[FormType._10_K, FormType._10_Q]
                ),
                sort_order="desc",
                limit=1,
                top_k=5
            ),
            ApiQuery(
                query="Forward Looking Guidance",
                filter=DocumentMetadataFilter(
                    symbol=symbol,
                    form_types=[FormType.earnings_transcript]
                ),
                sort_order="desc",
                limit=1,
                top_k=5
            ),
        ]

        # Handle None for datastore query
        document_results = await datastore.query(queries)
        if document_results is None:
            serialized_document_results = []
        else:
            serialized_document_results = [result.dict() for result in document_results if result is not None]
        print(document_results)

        income_statements = finnhub_client.financials(symbol, "ic", "quarterly")
        if income_statements is None or income_statements["financials"] is None:
            income_statements = {"financials": []}  # Default value if None
        print(income_statements)

        annual_income_statements = finnhub_client.financials(symbol, "ic", "annual")
        if annual_income_statements is None or annual_income_statements["financials"] is None:
            annual_income_statements = {"financials": []}  # Default value if None
        print(annual_income_statements)

        key_ratios = finnhub_client.company_basic_financials(symbol, "all")
        if key_ratios is None:
            key_ratios = {"metric": {}}  # Default value if None
        else:
            key_ratios = key_ratios["metric"]
        print(key_ratios)

        revenue_estimates = finnhub_client.company_revenue_estimates(symbol, "quarterly")
        if revenue_estimates is None:
            revenue_estimates = {"data": []}  # Default value if None
        print(revenue_estimates)

        ebit_estimates = finnhub_client.company_ebit_estimates(symbol, "quarterly")
        if ebit_estimates is None:
            ebit_estimates = {"data": []}  # Default value if None
        print(ebit_estimates)

        eps_estimates = finnhub_client.company_eps_estimates(symbol, "quarterly")
        if eps_estimates is None:
            eps_estimates = {"data": []}  # Default value if None
        print(eps_estimates)

        price_target = finnhub_client.price_target(symbol)
        print(price_target)

        recommendation_trends = finnhub_client.recommendation_trends(symbol)
        print(recommendation_trends)

        dividends = finnhub_client.stock_dividends(symbol, _from=(datetime.utcnow() - timedelta(days=5*365)).strftime('%Y-%m-%d'), to=datetime.utcnow().strftime('%Y-%m-%d'))
        print(dividends)

        insider_transactions = finnhub_client.stock_insider_transactions(symbol, (datetime.utcnow() - timedelta(days=60)).strftime('%Y-%m-%d'), datetime.utcnow().strftime('%Y-%m-%d'))
        print(insider_transactions)

        # Construct the final response
        response_data = {
            "document_results": serialized_document_results,
            "quarterly_income_statements": income_statements["financials"][:10],
            "annual_income_statements": annual_income_statements["financials"][:5],
            "key_ratios": key_ratios,
            "revenue_estimates": revenue_estimates["data"][:10],
            "ebit_estimates": ebit_estimates["data"][:10],
            "eps_estimates": eps_estimates["data"][:10],
            "price_target": price_target,
            "recommendation_trends": recommendation_trends[:10],
            "dividends": dividends,
            "insider_transactions": insider_transactions["data"][:10],
        }

        print(response_data)
        json_response_data = json.dumps(response_data)

        return JsonResponse(results=json_response_data)

    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail="Internal Service Error")


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
