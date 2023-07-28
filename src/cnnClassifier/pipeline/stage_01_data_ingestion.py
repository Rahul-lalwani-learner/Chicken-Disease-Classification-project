from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.data_ingestion import DataIngestion
from cnnClassifier import logger

STAGE_NAME = "Data Ingestion Stage"

class DataIngestionTrainingPipeline: 
    def __init__(self): 
        pass

    def main(self): 
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(data_ingestion_config)
        data_ingestion.download_file()
        data_ingestion.extract_zip_file()


if __name__ == "__main__":
    try: 
        logger.info(">>> Stage {} started <<<".format(STAGE_NAME))
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(">>> Stage {} completed <<<\n\n X===========X".format(STAGE_NAME))
    
    except Exception as e:
        logger.exception(e)
        raise e