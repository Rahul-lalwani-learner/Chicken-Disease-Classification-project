from cnnClassifier import logger
from cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline

STAGE_NAME = "Data Ingestion Stage"
try: 
    logger.info(">>> Stage {} started <<<".format(STAGE_NAME))
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(">>> Stage {} completed <<<\n\n X===========X".format(STAGE_NAME))

except Exception as e:
    logger.exception(e)
    raise e