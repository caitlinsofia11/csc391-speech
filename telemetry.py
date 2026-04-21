import os
from azure.monitor.opentelemetry import configure_azure_monitor

def init_telemetry():
    connection_string = os.environ.get("APPLICATIONINSIGHTS_CONNECTION_STRING")
    if not connection_string:
        raise ValueError("APPLICATIONINSIGHTS_CONNECTION_STRING not set in .env")
    configure_azure_monitor(connection_string=connection_string)
    print("Application Insights initialized.")