# This is a version of the main.py file found in ../../../server/main.py for testing the plugin locally.
# Use the command `poetry run dev` to run this.
from typing import Union, List, Optional
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
from datetime import date
from pydantic import ValidationError

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
from pydantic import BaseModel

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




class UserRequest(BaseModel):
    symbol: str

@app.post("/fine-tune-income-statement-user-request")
async def fine_tune_income_statement_user_request(request: UserRequest):
    symbol = request.symbol
    financial_statement_data = []
    try:

        queries = [
            ApiQuery(
                query='Income Statement',
                filter=DocumentMetadataFilter(
                    symbol=symbol,
                    form_types=[FormType._10_K, FormType._10_Q],
                    xbrl_only=True
                ),
                sort_order="desc",
                limit=1,
                top_k=15,
            ),
        ]

        # Handle None for datastore query
        documents = await datastore.query(queries)

        if documents is None:
            documents = []
        financial_statement_data = extract_texts(documents)
        print(f"Financial Statement Data: {financial_statement_data}")

    except Exception as e:
        print("Error:", e)

    # Prepare system and user messages
    system_message = """You are an assistant expert at parsing income_statement data and 
    converting it to a JSON that looks like this {"costOfGoodsSold":?,"dilutedAverageSharesOutstanding":?,"dilutedEPS":?,"ebit":?,"grossIncome":?,"netIncome":?,"netIncomeAfterTaxes":?,"period":"YYYY-MM-DD","pretaxIncome":?,"provisionforIncomeTaxes":?,"researchDevelopment":?,"revenue":?,"sgaExpense":?,"totalOperatingExpense":?,"totalOtherIncomeExpenseNet":?}.
    You will always convert the ? to the values in millions from the context in the user message and you will replace YYYY-MM-DD with the reporting date."""
    
    user_message = f"Create the income statement JSON from this context:  {financial_statement_data}"

    try:
        # Connect to your PostgreSQL database
        conn = db_manager.get_conn()

        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            # Find the current largest message_example_id
            cursor.execute("SELECT MAX(message_example_id) FROM fine_tunes WHERE prompt_name = 'Income Statement';")
            max_id_row = cursor.fetchone()
            next_message_example_id = max_id_row[0] + 1 if max_id_row[0] is not None else 1

            # Insert rows
            insert_query = """
            INSERT INTO fine_tunes (content, prompt_name, role, training_data, message_example_id, created_at, updated_at) VALUES 
            (%s, 'Income Statement', 'system', TRUE, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
            (%s, 'Income Statement', 'user', TRUE, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP);
            """
            cursor.execute(insert_query, (system_message, next_message_example_id, user_message, next_message_example_id))
            conn.commit()

        db_manager.put_conn(conn)

    except Exception as e:
        print("Error:", e)

    return {"message": "Data inserted successfully"}


class AssistantResponse(BaseModel):
    assistant_message: str

@app.post("/fine-tune-income-statement-assistant-response")
async def fine_tune_income_statement_assistant_response(request: AssistantResponse):
    assistant_message = request.assistant_message
    try:
        # Connect to your PostgreSQL database
        conn = db_manager.get_conn()

        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            # Find the current largest message_example_id
            cursor.execute("SELECT MAX(message_example_id) FROM fine_tunes WHERE prompt_name = 'Income Statement';")
            max_id_row = cursor.fetchone()

            insert_query = """
            INSERT INTO fine_tunes (content, prompt_name, role, training_data, message_example_id, created_at, updated_at) VALUES 
            ($1, 'Income Statement', 'assistant', TRUE, $2, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP);
            """
            cursor.execute(insert_query, (assistant_message, max_id_row))
            conn.commit()

        db_manager.put_conn(conn)

    except Exception as e:
        print("Error:", e)

    return {"message": "Data inserted successfully"}



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

        print(f"KEY OPPORTUNITIES: {key_opportunities}")

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

        print(f"FORWARD GUIDANCE: {forward_guidance}")

    except Exception as e:
        print("Error:", e)


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
    top_k: Optional[int] = 30 
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


        # If financial filters were set, get the actual financials that were filtered on and add them to 
        # the document metadata!  (brilliant eh?)
        if request.financial_filters:

            response = QueryResponse(results=document_results)

            # get the unique symbols
            unique_symbols = set()

            for result_index, result in enumerate(response.results):
                if result:
                    for doc_index, document_chunk in enumerate(result.results):
                        if document_chunk and document_chunk.metadata:
                            symbol = document_chunk.metadata.symbol
                            if symbol:
                                unique_symbols.add(symbol)

            query_facts = []
            for filter in request.financial_filters:
                # Directly use the comparison operator from the input
                query_facts.append(f"basic_financials.{filter.financial_name}")

            # Now unique_symbols contains all unique stock symbols from the query results
            print(f"Unique symbols: {unique_symbols}")

            # Fetch financial data for these symbols
            financial_query = f"""
                SELECT 
                    symbol,
                    {', '.join(query_facts)}
                FROM basic_financials 
                WHERE symbol IN %s
            """

            financial_data_dict = {}

            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(financial_query, (tuple(unique_symbols),))
                financial_rows = cursor.fetchall()
                
                for row in financial_rows:
                    symbol = row['symbol']
                    financial_data_dict[symbol] = row

            # now add the financial_data to the metadata.
            for result_index, result in enumerate(response.results):
                if result:
                    for doc_index, document_chunk in enumerate(result.results):
                        if document_chunk and document_chunk.metadata:
                            document_chunk.metadata.financial_data = json.dumps(financial_data_dict[document_chunk.metadata.symbol])

        # Handle None for datastore query
        if document_results is None:
            serialized_document_results = []
        else:
            serialized_document_results = [result.to_dict() for result in document_results if result is not None]

        # Now `financial_data_dict` is a dictionary with each symbol as a key
        json_response_data = json.dumps(serialized_document_results)
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




def start():
    uvicorn.run("local-server.main:app", host="localhost", port=PORT, reload=True)
