import urllib.parse
import os
import pandas as pd
import sqlalchemy as db

class SqlServerConnection:
    def __init__(self):
        pass

    def get_db_engine(self):
        driver = r'{ODBC Driver 17 for SQL Server}'
        server = os.environ['AZURE_SQL_SERVER']
        database = 'mlflowtracking'
        username = os.environ['AZURE_SQL_SERVER_ADMIN']
        password = os.environ['AZURE_SQL_SERVER_ADMIN_PW']
        param_str = f'Driver={driver};Server={server};Database={database};Uid={username};Pwd={password}' \
                    f';Encrypt=no;TrustServerCertificate=no;Connection Timeout=30;'
        params = urllib.parse.quote_plus(param_str)
        connection_string = f'mssql+pyodbc:///?odbc_connect={params}'
        return db.create_engine(connection_string, pool_pre_ping=True)

