from CNN_Classifier import logger
from CNN_Classifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from CNN_Classifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
from CNN_Classifier.pipeline.stage_03_model_trainer import ModelTrainerTrainingPipeline

STAGE_NAME = "Data Ingestion Stage"

try:
    logger.info(f">>>>>>>>>stage {STAGE_NAME} started<<<<<<<<<")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f">>>>>>>>>stage {STAGE_NAME} completed<<<<<\n\n<<<<<<<<")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Prepare Base Model"
try:
        logger.info(f"************************")
        logger.info(f">>>>>>> STAGE {STAGE_NAME} STARTED <<<<<<<<<<<<<<<<<")
        prepare_base_model = PrepareBaseModelTrainingPipeline()
        prepare_base_model.main()
        logger.info(f">>>>>>STAGE {STAGE_NAME} COMPLETED <<<<<<< \n \n ******************")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Model Training"
try:
    logger.info(f"************************")
    logger.info(f">>>>>>> STAGE {STAGE_NAME} STARTED <<<<<<<<<<<<<<<<<")
    obj = ModelTrainerTrainingPipeline()
    obj.main()
    logger.info(f">>>>>>STAGE {STAGE_NAME} COMPLETED <<<<<<< \n \n ******************")
except Exception as e:
    logger.exception(e)
    raise e