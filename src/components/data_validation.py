class DataValidation:
    def __init__(self, data):
        self.data = data

    def validate_data(self):
        print("Data validation started")

        if self.data is None:
            raise Exception("Data is empty")

        if len(self.data) == 0:
            raise Exception("No rows in data")

        print("Data is valid ✅")
        return True