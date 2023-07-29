from cnnClassifier import logger
from cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from cnnClassifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline

STAGE_NAME = "Data Ingestion Stage"
try: 
    logger.info(">>> Stage {} started <<<".format(STAGE_NAME))
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(">>> Stage {} completed <<<\n\n X===========X".format(STAGE_NAME))

except Exception as e:
    logger.exception(e)
    raise e



STAGE_NAME = "Prepare Base Model Stage"
try: 
    logger.info("************************")
    logger.info(f">>>>>>>>> Running stage: {STAGE_NAME} started <<<<<<<<<<")
    obj = PrepareBaseModelTrainingPipeline()
    obj.main()
    logger.info(f">>>>>>> stage: {STAGE_NAME} completed <<<<<<<<<\n\nX================X")
except Exception as e:
    logger.error(f"Error while running stage: {STAGE_NAME} - Error message: {e}")
    raise e