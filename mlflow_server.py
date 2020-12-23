import os
import urllib.parse
import subprocess
import logging

log_lvl = logging.DEBUG

logging.basicConfig(filename=f'logs\mlflow.log',
                    level=log_lvl,
                    filemode='a+',
                    format='%(asctime)s %(name)s - %(levelname)s - %(message)s')


# mlflow server --backend-store-uri %MLFLOW_TRACKING_URI% --default-artifact-root %MLFLOW_ARTIFACT_URI%


def run_server_command():
    try:
        cmd = get_cmd(as_list=False)
        # TODO: This needs to be debugged, runs fine in terminal, but not with venv
        # result = await call_cmd(cmd)
        print(cmd)
    except Exception as e:
        print(str(e))


def get_cmd(as_list: bool):
    """Returns the command to start the mlflow server."""
    backend_store = get_tracking_uri()
    logging.debug(str(backend_store))
    # artifact_uri = 'C:\\Users\\dhamacher\\Documents\\AzureDevOps\\mlflow-lab\\mlruns'
    # cmd_str = f'mlflow server --backend-store-uri {backend_store} --default-artifact-root {artifact_uri} --host 0.0.0.0'

    artifact_uri = 'wasbs://machinelearning@stdhamacher001.blob.core.windows.net/mlruns/'
    cmd_str = f'mlflow server --backend-store-uri {backend_store} --default-artifact-root {artifact_uri} --host 0.0.0.0'


    cmd = ['mlflow server',
           '--backend-store-uri',
           f'{backend_store}',
           f'--default-artifact-root',
           f'{artifact_uri}',
           '--host 0.0.0.0']

    logging.debug(f'Command String: {cmd_str}')
    if as_list:
        return cmd
    else:
        return cmd_str


def get_tracking_uri() -> str:
    driver = r'{ODBC Driver 17 for SQL Server}'
    server = os.environ['AZURE_SQL_SERVER']
    database = 'mlflowtracking'
    username = os.environ['AZURE_SQL_SERVER_ADMIN']
    password = os.environ['AZURE_SQL_SERVER_ADMIN_PW']
    param_str = f'Driver={driver};Server={server};Database={database};Uid={username};Pwd={password}' \
                f';Encrypt=no;TrustServerCertificate=no;Connection Timeout=30;'
    params = urllib.parse.quote_plus(param_str)
    backend_store = f'mssql+pyodbc:///?odbc_connect={params}'
    return backend_store


async def call_cmd(cmd) -> subprocess.CompletedProcess:
    try:
        logging.debug('Start MLflow server.')
        proc = subprocess.Popen(cmd,
                                stdout = subprocess.PIPE,
                                stderr = subprocess.PIPE,
                                preexec_fn = os.setsid)
        pgid = os.getpgid(proc.task.pid)
        logging.debug(f'PROCESS ID: {pgid}')
        print("Waiting for output")
        return proc
    except Exception as e:
        logging.exception(str(e))


run_server_command()
