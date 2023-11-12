import os
from typing import Optional
import json
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Depends, Body, UploadFile
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from fastapi import Query

import finnhub
from datetime import datetime

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

from models.models import DocumentMetadata, Source

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
            key_ratios = body["series"]["quarterly"]
        else:
            key_ratios = body["series"]["annual"]
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
    from_timestamp: int = Query(...),
    to_timestamp: int = Query(...)
):
    try:
        body = finnhub_client.stock_dividends(symbol, _from=datetime.utcfromtimestamp(from_timestamp).strftime('%Y-%m-%d'), to=datetime.utcfromtimestamp(to_timestamp).strftime('%Y-%m-%d'))
        return JsonResponse(results=json.dumps(body))
    except ValueError as ve:
        print(f"Timestamp conversion error: {ve}")
        raise HTTPException(status_code=400, detail="Invalid timestamp format")
    except Exception as e:
        print(f"Error in dividend_main: {e}")
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
    from_timestamp: int = Query(...),
    to_timestamp: int = Query(...)
):
    try:
        body = finnhub_client.earnings_calendar(_from=datetime.utcfromtimestamp(from_timestamp).strftime('%Y-%m-%d'), to=datetime.utcfromtimestamp(to_timestamp).strftime('%Y-%m-%d'), symbol=symbol)
        return JsonResponse(results=json.dumps(body))

    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail="Internal Service Error")


@app.get(
    "/insider-transactions",
)
async def insider_transactions(
    symbol: str = Query(...),
    from_timestamp: int = Query(...),
    to_timestamp: int = Query(...)
):
    try:
        body = finnhub_client.stock_insider_transactions(symbol, datetime.utcfromtimestamp(from_timestamp).strftime('%Y-%m-%d'), datetime.utcfromtimestamp(to_timestamp).strftime('%Y-%m-%d'))
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
