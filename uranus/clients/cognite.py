__all__ = ["Cognite"]

import pandas as pd

from uranus.exceptions import CogniteConnectionError
from datetime import datetime
from typing import List, Tuple
from pprint import pprint
from cognite.client import CogniteClient, ClientConfig
from cognite.client.credentials import OAuthClientCredentials
from loguru import logger


# NOTE: this came from https://hub.cognite.com/open-industrial-data-211/openid-connect-on-open-industrial-data-993
CLIENT_ID     = "1b90ede3-271e-401b-81a0-a4d52bea3273" 


class Cognite:
    
    def __init__(self,
                     name           : str,
                     client_secret  : str,
                     tenant_id      : str,
                     project        : str="publicdata",
                     client_id      : str=CLIENT_ID,
                    ):
            """
            Initializes a new instance of the class.

            Parameters:
            ----------
            name : str
                The name of the client.
            client_id : str
                The client ID for authentication.
            client_secret : str
                The client secret for authentication.
            tenant_id : str
                The tenant ID for the Azure Active Directory.
            project : str, optional
                The project name (default is "publicdata").

            Raises:
            ------
            CogniteConnectionError
                If there is an error connecting to Cognite.
            """
            
            self.name = name
            self.tenant_id = tenant_id
            self.project = project
            self.base_url = f"https://api.cognitedata.com"
            self.url = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
            self._client_id = client_id
            self._client_secret = client_secret 
            self._credentials = OAuthClientCredentials(
                token_url = self.url,
                client_id = self._client_id,
                client_secret = self._client_secret,
                scopes = [f"{self.base_url}/.default"],
            )
            self._config = ClientConfig(
                client_name=name , 
                project=project, 
                credentials=self._credentials, 
                base_url=self.base_url)
            self.client = CogniteClient( self._config )

            try:
                token_status = self.client.iam.token.inspect()
                logger.info(f"{token_status.projects[0].project_url_name}")
            except Exception as e:
                print(e)
                raise CogniteConnectionError(self.name)

    def __call__(self) -> CogniteClient:
        return self.client

    def search(self, description: str) -> List:
            """
            Searches for assets based on the provided description.

            Args:
                description (str): The description to search for in the assets.

            Returns:
                List: A list of assets that match the search criteria.
            """
            
            return self.client.assets.search(description=description, limit=None)
    
    def get_time_series_names(self, asset_name: str) -> List[str]:
            """
            Retrieves the names (descriptions) of all time series associated with a given asset.

            Args:
                asset_name (str): The name of the asset.

            Returns:
                List[str]: A list of time series descriptions or names.
            """
            asset = self.client.assets.search(name=asset_name, limit=1)[0]
            ts_list = self.client.time_series.list(asset_ids=[asset.id], limit=None)
            return [ts.description or ts.name or ts.external_id for ts in ts_list]
    
    def get_time_series_range(self, asset_name: str) -> pd.DataFrame:
            """
            Retrieves the first and last data point timestamps for each time series associated with an asset.

            Args:
                asset_name (str): The name of the asset.

            Returns:
                pd.DataFrame: A DataFrame containing the metric name, external ID, first and last timestamps.
            """
            asset = self.client.assets.search(name=asset_name, limit=1)[0]
            ts_list = self.client.time_series.list(asset_ids=[asset.id], limit=None)
            external_ids = [ts.external_id for ts in ts_list if ts.external_id]
            
            # Fetch latest points in batch
            latest_dps = self.client.time_series.data.retrieve_latest(external_id=external_ids)
            
            ranges = []
            for ts in ts_list:
                name = ts.description or ts.name or ts.external_id
                
                # Fetch first data point
                first_dp = self.client.time_series.data.retrieve(id=ts.id, start=0, limit=1)
                first_time = datetime.fromtimestamp(first_dp[0].timestamp / 1000.0) if len(first_dp) > 0 else None
                
                # Find matching latest point
                last_time = None
                for dp_list in latest_dps:
                    # In some SDK versions, it's dp_list.id or dp_list.external_id
                    if (dp_list.id == ts.id or dp_list.external_id == ts.external_id) and len(dp_list) > 0:
                        last_time = datetime.fromtimestamp(dp_list[0].timestamp / 1000.0)
                        break
                        
                ranges.append({
                    "metric": name,
                    "external_id": ts.external_id,
                    "first_timestamp": first_time,
                    "last_timestamp": last_time
                })
            
            return pd.DataFrame(ranges)

    def get_dataframe( self, 
                           asset_name  : str, 
                           start_time  : datetime, 
                           end_time    : datetime,
                           granularity : str='2s',
                           agregates   : List[str] = ['average'],
                           ) -> Tuple[pd.DataFrame, pd.DataFrame]:
            """
            Retrieves time series data for a specified asset within a given time range.

            Parameters:
            asset_name (str): The name of the asset for which to retrieve data.
            start_time (datetime): The start time for the data retrieval.
            end_time (datetime): The end time for the data retrieval.
            granularity (str, optional): The granularity of the data points. Default is '2s'.
            agregates (List[str], optional): List of aggregation functions to apply. Default is ['average'].

            Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing two DataFrames:
                - The first DataFrame contains the time series data for the specified asset.
                - The second DataFrame contains metadata information about the time series.
            """
            
            asset = self.client.assets.search(name=asset_name, limit=1)[0]
            ts_list = self.client.time_series.list(asset_ids=[asset.id], limit=None)
            id_to_description = {ts.external_id: ts.description for ts in ts_list if ts.description}
            pprint(id_to_description)
            df_info = ts_list.to_pandas()
            external_ids_interesse = df_info['external_id'].tolist()
            logger.info(f"Searching data from {start_time} to {end_time} for asset '{asset_name}'")
            # Recupera os dados com uma granularidade de 5 minutos
            df_data = self.client.time_series.data.retrieve_dataframe(
                    external_id=external_ids_interesse,
                    start=start_time,
                    end=end_time, # Período de 3 meses
                    granularity=granularity, # 5 minutos entre cada ponto de dados
                    aggregates=agregates, # Média para granulidade
                    ignore_unknown_ids=True # Ignora IDs não encontrados
            )

            # Renomeia as colunas para remover o sufixo do agregado e usar a descrição
            df_data.columns = [col.split('|')[0] for col in df_data.columns]
            df_data = df_data.rename(columns=id_to_description)
            return df_data, df_info
