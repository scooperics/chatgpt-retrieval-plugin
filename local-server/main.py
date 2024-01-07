# This is a version of the main.py file found in ../../../server/main.py for testing the plugin locally.
# Use the command `poetry run dev` to run this.
from typing import Optional
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Body, UploadFile

from models.api import (
    DeleteRequest,
    DeleteResponse,
    QueryRequest,
    QueryResponse,
    UpsertRequest,
    UpsertResponse,
)
import os
from datastore.factory import get_datastore
from services.file import get_document_from_file

from starlette.responses import FileResponse

from models.models import DocumentMetadata, Source
from fastapi.middleware.cors import CORSMiddleware


from typing import List
import json
import uvicorn
from fastapi import Query
from psycopg2.extras import RealDictCursor

from datastore.providers.database import DatabaseManager

import finnhub
from datetime import datetime, timedelta

from models.api import (
    JsonResponse,
)

from models.models import (
    DocumentMetadata, 
    Source,
    DocumentMetadataFilter,
    Query as ApiQuery, 
    FormType,   
)

FINHUB_API_KEY = os.environ.get("FINHUB_API_KEY")
finnhub_client = finnhub.Client(api_key=FINHUB_API_KEY)

app = FastAPI()

PORT = 3333

origins = [
    f"http://localhost:{PORT}",
    "https://chat.openai.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.route("/.well-known/ai-plugin.json")
async def get_manifest(request):
    file_path = "./local-server/ai-plugin.json"
    return FileResponse(file_path, media_type="text/json")


@app.route("/.well-known/logo.png")
async def get_logo(request):
    file_path = "./local-server/logo.png"
    return FileResponse(file_path, media_type="text/json")


@app.route("/.well-known/openapi.yaml")
async def get_openapi(request):
    file_path = "./local-server/openapi.yaml"
    return FileResponse(file_path, media_type="text/json")


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
        ids = await datastore.upsert(request.documents)
        return UpsertResponse(ids=ids)
    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail="Internal Service Error")


@app.post("/query", response_model=QueryResponse)
async def query_main(request: QueryRequest = Body(...)):
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


# Initialize DatabaseManager
db_manager = DatabaseManager()


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
                query="Top Risks",
                filter=DocumentMetadataFilter(
                    symbol=symbol,
                    form_types=[FormType._20_F, FormType._10_K, FormType._10_Q]
                ),
                sort_order="desc",
                limit=2,
                top_k=15
            ),
        ]

        # Handle None for datastore query
        documents = await datastore.query(queries)

        print(f"DOCUMENTS: {documents}")
        if documents is None:
            documents = []
        key_risks = extract_texts(documents)
        print(f"KEY RISKS: {key_risks}")

    except Exception as e:
        print("Error:", e)


    key_opportunities = []
    try:

        queries = [
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
        ]

        # Handle None for datastore query
        documents = await datastore.query(queries)

        if documents is None:
            documents = []
        key_opportunities = extract_texts(documents)

        print(f"KEY OPPORTUNITIES: {key_opportunities}")

    except Exception as e:
        print("Error:", e)



    forward_guidance = []
    try:

        queries = [
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
        forward_guidance = extract_texts(documents)

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
        else:
            income_statements = income_statements["financials"][0]
        print(income_statements)
    except Exception as e:
        print("Error:", e)

    cash_flow = {"financials": []}  # Default value if None
    try:
        cash_flow = finnhub_client.financials(symbol, "cf", "quarterly")
        if cash_flow is None or cash_flow["financials"] is None:
            cash_flow = {"financials": []}  # Default value if None
        else:
            cash_flow = cash_flow["financials"][0]
        print(cash_flow)
    except Exception as e:
        print("Error:", e)

    balance_sheet = {"financials": []}  # Default value if None
    try:
        balance_sheet = finnhub_client.financials(symbol, "bs", "quarterly")
        if balance_sheet is None or balance_sheet["financials"] is None:
            balance_sheet = {"financials": []}  # Default value if None
        else:
            balance_sheet = balance_sheet["financials"][0]
        print(balance_sheet)
    except Exception as e:
        print("Error:", e)

    annual_income_statements = {"financials": []}  # Default value if None
    try:
        annual_income_statements = finnhub_client.financials(symbol, "ic", "annual")
        if annual_income_statements is None or annual_income_statements["financials"] is None:
            annual_income_statements = {"financials": []}  # Default value if None
        else:
            annual_income_statements = annual_income_statements["financials"][0]
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


    sector_ratios = {"metric": {}}  # Default value if None
    try:
        raw_sector_ratios = finnhub_client.sector_metric('NA')
        if raw_sector_ratios is None:
            sector_ratios = {"metric": {}}  # Default value if None
        else:
            sector_ratios = extract_financial_ratios(extract_metrics(raw_sector_ratios, 'NA', get_sector_by_symbol(symbol)))
        print(sector_ratios)
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
        recommendation_trends = finnhub_client.recommendation_trends(symbol)[0]
        print(recommendation_trends)
    except Exception as e:
        print("Error:", e)

    dividends=[]
    try:
        dividends = finnhub_client.stock_dividends(symbol, _from=(datetime.utcnow() - timedelta(days=2*365)).strftime('%Y-%m-%d'), to=datetime.utcnow().strftime('%Y-%m-%d'))
        print(dividends)
    except Exception as e:
        print("Error:", e)

    insider_transactions={"data": []}
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
        "key_risks": key_risks,
        "key_opportunities": key_opportunities,
        "forward_guidance": forward_guidance,
    }

    print(response_data)
    json_response_data = json.dumps(response_data)

    return JsonResponse(results=json_response_data)





def start():
    uvicorn.run("local-server.main:app", host="localhost", port=PORT, reload=True)
