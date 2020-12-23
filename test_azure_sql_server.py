from unittest import TestCase
import urllib.parse
import os
import sqlalchemy as db
import pandas as pd


class TestSqlServerConnection(TestCase):
    def test_get_db_engine(self):
        try:
            driver = r'{ODBC Driver 17 for SQL Server}'
            server = os.environ['AZURE_SQL_SERVER']
            database = 'mlflowtracking'
            username = os.environ['AZURE_SQL_SERVER_ADMIN']
            password = os.environ['AZURE_SQL_SERVER_ADMIN_PW']
            param_str = f'Driver={driver};Server={server};Database={database};Uid={username};Pwd={password}' \
                        f';Encrypt=no;TrustServerCertificate=no;Connection Timeout=30;'
            params = urllib.parse.quote_plus(param_str)
            connection_string = f'mssql+pyodbc:///?odbc_connect={params}'
            engine = db.create_engine(connection_string, pool_pre_ping=True)

            with engine.begin() as connection:
                table = pd.read_sql_table(table_name='runs', schema='dbo', con=connection)
                print('Done')

            with engine.begin() as connection:
                query = f"INSERT dbo.test VALUES('Confetti')"
                connection.execute(query)
                query = f'SELECT ID FROM dbo.test'
                batch_id = connection.execute(query).fetchone()[0]
                print(batch_id)

            print('Done')
        except Exception as e:
            print(str(e))
