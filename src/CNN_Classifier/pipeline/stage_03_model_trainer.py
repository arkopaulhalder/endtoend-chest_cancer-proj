from CNN_Classifier.config.configuration import ConfigurationManager
from CNN_Classifier.components.model_trainer import Training
from CNN_Classifier import logger

STAGE_NAME = "Model Training Stage"


class ModelTrainerTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        training_config = config.get_training_config()
        training = Training(config=training_config)
        training.get_base_model()
        training.train_valid_generator()
        training.train()


if __name__ == '__main__':
    try:
        logger.info(f"************************")
        logger.info(f">>>>>>> STAGE {STAGE_NAME} STARTED <<<<<<<<<<<<<<<<<")
        obj = ModelTrainerTrainingPipeline()
        obj.main()
        logger.info(f">>>>>>STAGE {STAGE_NAME} COMPLETED <<<<<<< \n \n ******************")
    except Exception as e:
        logger.exception(e)
        raise e