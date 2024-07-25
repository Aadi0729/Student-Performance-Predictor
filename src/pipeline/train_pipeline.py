# From the pipeline, we will be calling other components

import sys
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from src.components import data_ingestion as di
from src.components import data_transformation as dt
from src.components import model_trainer as mt


class TrainPipeline:
    def __init__(self, raw_data_path=None):
        self.raw_data_path = raw_data_path

    def train(self):
        try:
            logging.info("Initiating data ingestion")
            di_obj = di.DataIngestion()
            train_data, test_data = di_obj.initiate_data_ingestion(raw_data_path=self.raw_data_path)
            logging.info("Data ingestion completed")
            
            logging.info("Initiating data transformation")
            dt_obj = dt.DataTransformation() # We call DataTransformation here, just for the sake of demonstration.
            train_arr, test_arr,_ = dt_obj.initiate_data_transformation(train_data,test_data)
            logging.info("Data transformation completed and saved preprocessor object")
            
            
            logging.info("Training the model")
            mt_obj = mt.ModelTrainer()
            print(f"Best Models r2_score: {mt_obj.initiate_model_trainer(train_arr, test_arr)}")
            logging.info("Model training completed and saved the best model")
    
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    train_pipeline_obj = TrainPipeline("data/NewsPerformance.csv")
    train_pipeline_obj.train()