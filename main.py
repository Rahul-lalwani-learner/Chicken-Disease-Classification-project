from cnnClassifier import logger
from cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from cnnClassifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
from cnnClassifier.pipeline.stage_03_training import ModelTrainingPipeline
from cnnClassifier.pipeline.stage_04_evaluation import EvaluationPipeline

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


STAGE_NAME = "Training Stage"
try: 
    logger.info(f"***********************")
    logger.info(f">>>>>>>>>>>> {STAGE_NAME} started <<<<<<<<<<<")
    obj = ModelTrainingPipeline()
    obj.main()
    logger.info(f">>>>>>>>>>>> {STAGE_NAME} completed <<<<<<<<<<<")
except Exception as e: 
    logger.exception(f"Error in {STAGE_NAME} pipeline: {e}")
    raise e


STAGE_NAME = "Evaluation Stage"
try: 
    logger.info(f"******************")
    logger.info(f">>>>> {STAGE_NAME} started <<<<<")
    obj = EvaluationPipeline()
    obj.main()
    logger.info(f">>>>>>>>>> {STAGE_NAME} completed <<<<<<<<<<\n\nX=====================X")

except Exception as e:
    logger.exception(f"Exception occured in {STAGE_NAME} : {e}")
    raise e