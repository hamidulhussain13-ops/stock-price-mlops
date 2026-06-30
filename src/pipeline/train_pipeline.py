from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


def main():

    print("Step 1: Data Ingestion")
    ingestion = DataIngestion()
    data = ingestion.initiate_data_ingestion()

    print("Step 2: Data Validation")
    validator = DataValidation(data)
    validator.validate_data()

    print("Step 3: Data Transformation")
    transformer = DataTransformation()
    X_train, X_test, y_train, y_test = transformer.initiate_data_transformation(data)

    print("Step 4: Model Training Started")
    trainer = ModelTrainer()
    score = trainer.train_model(X_train, y_train, X_test, y_test)

    print(f"Step 5: Model Training Completed | Best Score: {score}")

    print("Pipeline Completed Successfully 🚀")


if __name__ == "__main__":
    main()