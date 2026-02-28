from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.pipeline.train_pipeline import ModelTrainer 


if __name__ == "__main__":
    obj = DataIngestion()
    train_data_path, test_data_path = obj.initiate_data_ingestion()

    transformation_obj = DataTransformation()
    train_arr, test_arr = transformation_obj.data_transformation(train_data_path,test_data_path)

    train = ModelTrainer()
    report_type,report_priority,best_type_name,best_priority_name=train.initiate_model_trainer(train_arr, test_arr)

    print(best_type_name,best_priority_name)