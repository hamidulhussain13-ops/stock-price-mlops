import pandas as pd

class DataIngestion:

    def initiate_data_ingestion(self):
        print("Data ingestion started")

        df = pd.read_csv("notebook/train.csv")

        return df   # 🔥 MUST return DataFrame