import os
from typing import Optional
import json
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Depends, Body, UploadFile
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
import finnhub
from datetime import datetime

from models.api import (
    DeleteRequest,
    DeleteResponse,
    QueryRequest,
    QueryResponse,
    UpsertRequest,
    UpsertResponse,
    FinancialStatement,
    SymbolOnly,
    Candle,
    Estimate,
    TimedResponse,
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
app.mount("/.well-known", StaticFiles(directory=".well-known"), name="static")

# Create a sub-application, in order to access just the query endpoint in an OpenAPI schema, found at http://0.0.0.0:8000/sub/openapi.json when the app is running locally
sub_app = FastAPI(
    title="AITickerChat Retrieval Plugin API",
    description="A retrieval API for querying and filtering SEC filing documents and earnings call transcripts based on natural language queries and metadata",
    version="1.0.0",
    servers=[{"url": "https://stock-advisor.com"}],
    dependencies=[Depends(validate_token)],
)
app.mount("/sub", sub_app)


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


@app.post(
    "/financial-statements",
)
async def financial_statements_main(
    request: FinancialStatement = Body(...),
):
    try:
        body = finnhub_client.financials(request.symbol, request.statement, request.freq)
        return json.dumps(body)

    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail="Internal Service Error")


@app.post(
    "/quote",
)
async def quote_main(
    request: SymbolOnly = Body(...),
):
    try:
        body = finnhub_client.quote(request.symbol)
        return json.dumps(body)

    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail="Internal Service Error")


@app.post(
    "/metrics",
)
async def metrics_main(
    request: SymbolOnly = Body(...),
):
    try:
        body = finnhub_client.company_basic_financials(request.symbol, "all")
        return json.dumps(body)

    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail="Internal Service Error")


@app.post(
    "/candles",
)
async def candles_main(
    request: Candle = Body(...),
):
    try:
        body = finnhub_client.stock_candles(request.symbol, request.resolution, request.from_timestamp, request.to_timestamp)
        return json.dumps(body)

    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail="Internal Service Error")


@app.post(
    "/company-profile",
)
async def company_profile_main(
    request: SymbolOnly = Body(...),
):
    try:
        body = finnhub_client.company_profile2(request.symbol)
        return json.dumps(body)

    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail="Internal Service Error")


@app.post(
    "/dividend",
)
async def dividend_main(
    request: TimedResponse = Body(...),
):
    try:
        body = finnhub_client.stock_dividends(request.symbol, _from=datetime.utcfromtimestamp(request.from_timestamp).strftime('%Y-%m-%d'), to=datetime.utcfromtimestamp(request.to_timestamp).strftime('%Y-%m-%d'))
        return json.dumps(body)

    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail="Internal Service Error")


@app.post(
    "/revenue-estimates",
)
async def revenue_estimates_main(
    request: Estimate = Body(...),
):
    try:
        body = finnhub_client.company_revenue_estimates(request.symbol, request.freq)
        return json.dumps(body)

    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail="Internal Service Error")


@app.post(
    "/revenue-breakdown",
)
async def revenue_breakdown_main(
    request: SymbolOnly = Body(...),
):
    try:
        body = finnhub_client.stock_revenue_breakdown(request.symbol)
        return json.dumps(body)

    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail="Internal Service Error")


@app.post(
    "/eps-estimates",
)
async def eps_estimates_main(
    request: Estimate = Body(...),
):
    try:
        body = finnhub_client.company_eps_estimates(request.symbol, request.freq)
        return json.dumps(body)

    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail="Internal Service Error")


@app.post(
    "/ebitda-estimates",
)
async def ebitda_estimates_main(
    request: Estimate = Body(...),
):
    try:
        body = finnhub_client.company_ebitda_estimates(request.symbol, request.freq)
        return json.dumps(body)

    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail="Internal Service Error")


@app.post(
    "/ebit-estimates",
)
async def ebit_estimates_main(
    request: Estimate = Body(...),
):
    try:
        body = finnhub_client.company_ebit_estimates(request.symbol, request.freq)
        return json.dumps(body)

    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail="Internal Service Error")


@app.post(
    "/price-targets",
)
async def price_targets_main(
    request: SymbolOnly = Body(...),
):
    try:
        body = finnhub_client.price_target(request.symbol)
        return json.dumps(body)

    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail="Internal Service Error")


@app.post(
    "/recommendation-trends",
)
async def recommendation_trends_main(
    request: SymbolOnly = Body(...),
):
    try:
        body = finnhub_client.recommendation_trends(request.symbol)
        return json.dumps(body)

    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail="Internal Service Error")

@app.post(
    "/earnings-calendar",
)
async def earnings_calendar_main(
    request: TimedResponse = Body(...),
):
    try:
        body = finnhub_client.earnings_calendar(_from=datetime.utcfromtimestamp(request.from_timestamp).strftime('%Y-%m-%d'), to=datetime.utcfromtimestamp(request.to_timestamp).strftime('%Y-%m-%d'), symbol=request.symbol)
        return json.dumps(body)

    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail="Internal Service Error")


@app.post(
    "/insider-transactions",
)
async def insider_transactions(
    request: TimedResponse = Body(...),
):
    try:
        body = finnhub_client.stock_insider_transactions(request.symbol, datetime.utcfromtimestamp(request.from_timestamp).strftime('%Y-%m-%d'), datetime.utcfromtimestamp(request.to_timestamp).strftime('%Y-%m-%d'))
        return json.dumps(body)

    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail="Internal Service Error")


@sub_app.post(
    "/query",
    response_model=QueryResponse,
    # NOTE: We are describing the shape of the API endpoint input due to a current limitation in parsing arrays of objects from OpenAPI schemas. This will not be necessary in the future.
    description="Accepts search query objects array each with query and optional filter. Break down complex questions into sub-questions. Refine results by criteria, e.g. time / source, don't do this often. Split queries if ResponseTooLargeError occurs.",
)
async def query(
    request: QueryRequest = Body(...),
):
    try:
        results = await datastore.query(
            request.queries,
        )
        return QueryResponse(results=results)
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
