import os
import psycopg2
from psycopg2 import sql

class DatabaseManager:
    
    def __init__(self):
        self.POSTGRES_DB = os.environ.get("POSTGRES_DB")
        self.POSTGRES_USER = os.environ.get("POSTGRES_USER")
        self.POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD")
        self.POSTGRES_HOST = os.environ.get("POSTGRES_HOST")
        self.POSTGRES_PORT = os.environ.get("POSTGRES_PORT")
        self.POSTGRES_SSLMODE = os.environ.get("POSTGRES_SSLMODE")

        self.conn = psycopg2.connect(database=self.POSTGRES_DB, user=self.POSTGRES_USER, password=self.POSTGRES_PASSWORD, host=self.POSTGRES_HOST, port=self.POSTGRES_PORT, sslmode=self.POSTGRES_SSLMODE)
        self.cursor = self.conn.cursor()


    def __del__(self):
        self.cursor.close()
        self.conn.close()


    @staticmethod
    def add_form_10K_if_needed(form_types, fiscal_quarter):
        if fiscal_quarter == 4 and '10-Q' in form_types and '10-K' not in form_types:
            form_types.append('10-K')
        return form_types


    def lookup_documents(self, sort_order, limit, symbol, form_types, fiscal_quarter, fiscal_year):
        try:
            cik = None
            if symbol is not None:
                self.cursor.execute("SELECT cik FROM stocks WHERE symbol = %s", (symbol,))
                result = self.cursor.fetchone()
                if result:
                    cik = result[0]

            query = "SELECT filename FROM source_file_metadata WHERE 1=1"
            params = []

            if cik is not None:
                query += " AND cik = %s"
                params.append(cik)

            if form_types is not None:
                self.add_form_10K_if_needed(form_types, fiscal_quarter)
                query += " AND form_type = ANY(%s)"
                params.append(form_types)
            if fiscal_quarter is not None:
                query += " AND fiscal_quarter = %s"
                params.append(str(fiscal_quarter))
            if fiscal_year is not None:
                query += " AND fiscal_year = %s"
                params.append(str(fiscal_year))

            if sort_order is not None and limit is not None:
                query += f" ORDER BY published_date {sort_order} LIMIT %s"
                params.append(limit)
            elif limit is not None:
                query += f" ORDER BY published_date desc LIMIT %s"
                params.append(limit)

            if len(params) == 0:
                return []

            self.cursor.execute(query, params)
            results = self.cursor.fetchall()

            filenames = [row[0] for row in results]
            return filenames

        except psycopg2.Error as e:
            print(f"Database error occurred: {e}")
            return []

        except Exception as e:
            print(f"An error occurred: {e}")
            return []


    def insert_query_log(self, document_ids, filenames, fiscal_quarter, fiscal_year, form_types, query, symbol, xbrl_only, sort_order, limit, top_k, result_ids):
        try:
            insert = sql.SQL("""
                INSERT INTO query_logs (document_ids, filenames, fiscal_quarter, fiscal_year, form_types, query, symbol, xbrl_only, sort_order, limit, top_k, result_ids)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """)

            values = (document_ids, filenames, fiscal_quarter, fiscal_year, form_types, query, symbol, xbrl_only, sort_order, limit, top_k, result_ids)

            self.cursor.execute(insert, values)
            self.conn.commit()

        except psycopg2.Error as e:
            # Handle database-related exceptions
            print(f"Database error occurred: {e}")
            return

        except Exception as e:
            # Handle other exceptions
            print(f"An error occurred: {e}")
            return
