import os
import urllib.request as request
import zipfile
import gdown
from CNN_Classifier import logger
from CNN_Classifier.utils.common import get_size
from CNN_Classifier.entity.config_entity import DataIngestionConfig

class DataIngestion:
    """
    class DataIngestion:
        This class is responsible for downloading the dataset from the url
        and extracting the data from the zip file into the data directory
    
        Args:
            config (DataIngestionConfig): The configuration object containing the parameters required for data ingestion
    """
    def __init__(self,config: DataIngestionConfig):
        """
        Initializes the DataIngestion object with the given configuration

        Args:
            config (DataIngestionConfig): The DataIngestionConfig object containing the parameters required for data ingestion
        """
        self.config = config


    def download_file(self)->str:
        '''
        fetch data from the url
        '''

        try:
            dataset_url = self.config.source_URL
            zip_download_dir = self.config.local_data_file
            os.makedirs("artifacts/data_ingestion",exist_ok=True)
            logger.info(f"Download data from{dataset_url} into file {zip_download_dir}")
            
            file_id = dataset_url.split("/")[-2]
            prefix = 'https://drive.google.com/uc?/export=download&id='
            gdown.download(prefix+file_id,zip_download_dir)

            logger.info(f"Dowloaded data from {dataset_url} into file {zip_download_dir}")

        except Exception as e:
            raise e
        

    def extract_zip_file(self):
        '''
            zip file path : str
            extracts the zip file into the data directory
            fucntion returns none
            '''
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path,exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file,'r') as zip_ref:
            zip_ref.extractall(unzip_path)