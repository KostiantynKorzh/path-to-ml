from csv import DictReader


class Tester:

    def __init__(self, dataset_path):
        self.dataset = self.create_dataset_from_csv(dataset_path)

    def create_dataset_from_csv(self, dataset_path):
        with open(dataset_path, newline='') as csvfile:
            csv_reader = DictReader(csvfile, delimiter=',')
            header = next(csv_reader)
            dataset = []
            for row in csv_reader:
                dataset.append([row['X4 number of convenience stores'], row['Y house price of unit area']])

        return dataset
