import os
import psycopg2


# since there are no 10-Qs in Q4, Chat GPT sometimes gets confused and asks for it.  In these cases, we need to add 10-Ks to the filter criteria
# or there will be no response.
def add_form_10K_if_needed(form_types, fiscal_quarter):
    if fiscal_quarter == 4 and '10-Q' in form_types and '10-K' not in form_types:
        form_types.append('10-K')

    return form_types


def lookup_documents(sort_order, limit, symbol, form_types, fiscal_quarter, fiscal_year):

    POSTGRES_DB = os.environ.get("POSTGRES_DB")
    POSTGRES_USER = os.environ.get("POSTGRES_USER")
    POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD")
    POSTGRES_HOST = os.environ.get("POSTGRES_HOST")
    POSTGRES_PORT = os.environ.get("POSTGRES_PORT")
    POSTGRES_SSLMODE = os.environ.get("POSTGRES_SSLMODE")

    try:
        # Connect to the PostgreSQL database
        print("connecting to db")
        conn = psycopg2.connect(database=POSTGRES_DB, user=POSTGRES_USER, password=POSTGRES_PASSWORD, host=POSTGRES_HOST, port=POSTGRES_PORT, sslmode = POSTGRES_SSLMODE)
        cursor = conn.cursor()

        # Convert the symbol to cik
        cik = None
        if symbol is not None:
            print("getting cik")
            cursor.execute("SELECT cik FROM stocks WHERE symbol = %s", (symbol,))
            result = cursor.fetchone()
            if result:
                cik = result[0]
                print(f"cik: {cik}")
    
        # Construct the SQL query
        query = "SELECT filename FROM source_file_metadata WHERE 1=1"
        params = []

        # Apply filters based on provided parameters
        if cik is not None:
            query += " AND cik = %s"
            params.append(cik)
        # if start_date is not None:
        #     query += " AND published_date >= %s"
        #     params.append(start_date)
        # if end_date is not None:
        #     query += " AND published_date <= %s"
        #     params.append(end_date)
        if form_types is not None:
            add_form_10K_if_needed(form_types, fiscal_quarter)
            query += " AND form_type = ANY(%s)"
            params.append(form_types)
        if fiscal_quarter is not None:
            query += " AND fiscal_quarter = %s"
            params.append(str(fiscal_quarter))
        if fiscal_year is not None:
            query += " AND fiscal_year = %s"
            params.append(str(fiscal_year))
 
        # Apply sorting and limiting if all three parameters are not None
        if sort_order is not None and limit is not None:
            query += f" ORDER BY published_date {sort_order} LIMIT %s"
            params.append(limit)
        elif limit is not None:
            query += f" ORDER BY published_date desc LIMIT %s"
            params.append(limit)

        if len(params) == 0:
            cursor.close()
            conn.close()
            return[]

        # Execute the SQL query
        print(f"query: {query}")
        print(f"params: {params}")
        cursor.execute(query, params)

        # Fetch the results
        results = cursor.fetchall()

        # Close the cursor and connection
        cursor.close()
        conn.close()

        # Extract filenames from the results
        filenames = [row[0] for row in results]

        return filenames

    except psycopg2.Error as e:
        # Handle database-related exceptions
        print(f"Database error occurred: {e}")
        return []

    except Exception as e:
        # Handle other exceptions
        print(f"An error occurred: {e}")
        return []