from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    DataCollector,
    DeploymentCollection,
    ManagedOnlineDeployment,
    ManagedOnlineEndpoint,
    Environment,
    Model,
)
from azure.ai.ml.constants import AssetTypes
from azure.core.exceptions import ResourceNotFoundError

import argparse
import datetime


def get_data_collector() -> DataCollector:
    return DataCollector(
        collections={
            "model_inputs": DeploymentCollection(enabled="true"),
            "model_outputs": DeploymentCollection(enabled="true"),
        }
    )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--subscription-id", dest="subscription_id", required=True)
    parser.add_argument("--resource-group", dest="resource_group", required=True)
    parser.add_argument("--workspace", dest="workspace", required=True)
    parser.add_argument("--endpoint-name", dest="endpoint_name", default="diabetes-endpoint")
    parser.add_argument("--deployment-name", dest="deployment_name", default="blue")
    parser.add_argument("--model_name", dest="model_name", required=True)
    

    return parser.parse_args()


def get_ml_client(subscription_id: str, resource_group: str, workspace: str) -> MLClient:
    credential = DefaultAzureCredential()
    return MLClient(
        credential=credential,
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace,
    )


def ensure_endpoint(ml_client: MLClient, endpoint_name: str) -> ManagedOnlineEndpoint:
    endpoint = ManagedOnlineEndpoint(
        name=endpoint_name,
        description="Online endpoint for MLflow diabetes model",
        auth_mode="key",
    )
    try:
        existing_endpoint = ml_client.online_endpoints.get(name=endpoint_name)
        print(f"Endpoint '{endpoint_name}' state: {existing_endpoint.provisioning_state}")
        
        # If the endpoint is stuck or failed, re-provision it
        if existing_endpoint.provisioning_state != "Succeeded":
            print(f"Endpoint is in '{existing_endpoint.provisioning_state}' state. Re-creating endpoint...")
            return ml_client.online_endpoints.begin_create_or_update(endpoint).result()
        
        return existing_endpoint
    except ResourceNotFoundError:
        print(f"Endpoint '{endpoint_name}' not found. Creating new endpoint...")
        return ml_client.online_endpoints.begin_create_or_update(endpoint).result()


def create_or_update_deployment(
    ml_client: MLClient,
    endpoint_name: str,
    deployment_name: str,
    model_name: str,
) -> ManagedOnlineDeployment:
    print(f"Fetching model '{model_name}:latest' from Azure ML registry...")
    
    # Get the latest version registered by your train-prod workflow
    model = ml_client.models.get(name=model_name, label="latest")

    # --- ADD THIS ENVIRONMENT BLOCK ---
    env = Environment(
        name=f"{model_name}-inference-env",
        description="Environment with data collection support",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
        conda_file={
            "name": "inference-env",
            "channels": ["conda-forge"],
            "dependencies": [
                "python=3.10",
                "pip",
                {
                    "pip": [
                        "azureml-defaults",
                        "azureml-ai-monitoring",  # <--- Fixes missing Collector
                        "mlflow",
                        "pandas",
                        "scikit-learn",
                        "numpy",
                    ]
                },
            ],
        },
    )

    deployment = ManagedOnlineDeployment(
        name=deployment_name,
        endpoint_name=endpoint_name,
        model=model,
        environment=env,  # <--- ADD THIS LINE
        instance_type="Standard_D2as_v4",
        instance_count=1,
        data_collector=get_data_collector(),
    )

    return ml_client.online_deployments.begin_create_or_update(deployment).result()


def set_traffic_to_deployment(ml_client: MLClient, endpoint_name: str, deployment_name: str) -> None:
    endpoint = ml_client.online_endpoints.get(name=endpoint_name)
    endpoint.traffic = {deployment_name: 100}
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()


def main() -> None:
    args = parse_args()

    print("Connecting to Azure Machine Learning workspace...")
    ml_client = get_ml_client(
        subscription_id=args.subscription_id,
        resource_group=args.resource_group,
        workspace=args.workspace,
    )

    print(f"Ensuring online endpoint '{args.endpoint_name}' exists...")
    endpoint = ensure_endpoint(ml_client, args.endpoint_name)
    print(f"Using endpoint: {endpoint.name}")

    print(f"Creating or updating deployment '{args.deployment_name}'...")
    deployment = create_or_update_deployment(
        ml_client=ml_client,
        endpoint_name=endpoint.name,
        deployment_name=args.deployment_name,
        model_name=args.model_name,
    )
    print(f"Deployment state: {deployment.provisioning_state}")

    print("Directing 100% of traffic to the deployment...")
    set_traffic_to_deployment(ml_client, endpoint.name, args.deployment_name)

    endpoint = ml_client.online_endpoints.get(name=endpoint.name)
    print(f"Deployment complete. Scoring URI: {endpoint.scoring_uri}")


if __name__ == "__main__":
    main()
