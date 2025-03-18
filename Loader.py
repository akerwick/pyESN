import pickle
import cloudpickle
import os
import numpy as np
import json
from concurrent.futures import ProcessPoolExecutor
import torch

IS_PARALLEL_PROCESSING = True

class Loader():
    def __init__(self):
        pass

    def load_model(self, filename):
        model_file = os.path.join(self.directory, filename)
        # with open(model_file, 'rb') as f:
        #     model = cloudpickle.load(f)
        model = torch.load(model_file)
        return model

    def mount(self, directory: str, verbose = True, metadata = True) -> None:
            '''
            Mounts the loader to load data from a specified directory.

            This method sets the base directory for the loader to search for pickle files.
            It ensures that the directory exists and prepares the loader to handle multiple
            `.pkl` files from that directory.

            Args:
                directory (str): The path to the directory where the pickle files are stored.

            Raises:
                FileNotFoundError: If the specified directory does not exist.
                NotADirectoryError: If the specified path is not a directory.
            '''
            if not os.path.exists(directory):
                raise FileNotFoundError(f"The directory {directory} does not exist.")

            if not os.path.isdir(directory):
                raise NotADirectoryError(f"The path {directory} is not a directory.")

            self.directory = directory


            if verbose:
                print(f"Loader successfully mounted to the directory: {self.directory}")

            if metadata:
                # Check if the 'metadata' subdirectory exists inside the main directory
                metadata_directory = os.path.join(directory, 'metadata')

                if not os.path.exists(metadata_directory):
                    raise FileNotFoundError(f"The 'metadata' subdirectory does not exist inside {directory}.")

                if not os.path.isdir(metadata_directory):
                    raise NotADirectoryError(f"The path {metadata_directory} is not a directory.")
                self.metadata_directory = metadata_directory
                if verbose:
                    print(f"Metadata successfully mounted to: {self.metadata_directory}")


    def get_sample(self, A, fc, fm):
        x_file, y_file, pt_file = self.generate_filenames(float(A),float(fc),float(fm))
        x_file, y_file, pt_file = self.directory + x_file, self.directory + y_file, self.directory + pt_file
        x_sample_dbfile = open(x_file, 'rb')
        y_sample_dbfile = open(y_file, 'rb')
        pt_sample_dbfile = open(pt_file, 'rb')
        x_sample = pickle.load(x_sample_dbfile)
        y_sample = pickle.load(y_sample_dbfile)
        pt_sample = pickle.load(pt_sample_dbfile)
        #print(type(x_sample), type(y_sample), type(pt_sample))
        return x_sample, y_sample, pt_sample

    def generate_filenames(self, A: float, Fc: float, Fm: float) -> tuple[str, str, str]:
        ID = self.get_ID(A, Fc, Fm)
        return f'x_{ID}.pkl', f'y_{ID}.pkl', f'pt_{ID}.pkl'

    def get_ID(self, A: float, Fc: float, Fm: float):
        #print(f'FC: {Fc}')
        metadata_file = f"sample_{A:.1f}_{Fc:.1f}_{Fm:.1f}_0.1_500.0.txt"
        metadata_path = os.path.join(self.metadata_directory, metadata_file)
        # Check if the metadata file exists
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"The metadata file {metadata_file} does not exist at {metadata_path}.")

        # Open and read the file to extract the ID
        with open(metadata_path, 'r') as file:
            for line in file:
                # Assuming the file contains a line like 'ID: 755'
                if line.startswith("ID:"):
                    # Split the line to get the ID value (after "ID: ")
                    ID = line.split(":")[1].strip()  # .strip() removes leading/trailing whitespace
                    return ID

        # If no line starting with "ID:" was found, raise an error
        raise ValueError(f"ID not found in the file {metadata_file} at {metadata_path}.")



    def load_db(self, A_list : list, fc_list: list, fm_list: list):
        #sample0 = self.get_sample(A_list[0], fc_list[0], fm_list[0])
        x, y, pt = [], [], []
        items = [(A, fc, fm) for fm in fm_list for fc in fc_list for A in A_list]
        if IS_PARALLEL_PROCESSING:
            with ProcessPoolExecutor(max_workers=10) as pool:
                results = pool.map(self.get_sample, *zip(*items))
        else:
            results = []
            for item in items:
                results.append(self.get_sample(*item))
        for result in results:
            x.append(result[0])
            y.append(result[1])
            pt.append(result[2])

        # for A in A_list:
        #     for fc in fc_list:
        #         for fm in fm_list:
        #             x_sample, y_sample, pt_sample = self.get_sample(A, fc, fm)
        #             x.append(x_sample)
        #             y.append(y_sample)
        #             pt.append(pt_sample)
        try:
            x = np.stack(x, axis=0)
        except Exception as e:  # Catch any exception and bind it to variable 'e'
            print(f"An error occurred: {e}")
            print("A:", A)
            print("fc:", fc)
            print("fm:", fm)
            print("x0: ", x[0].shape)
            print("x1000: ", x[1000].shape)
        y = np.stack(y, axis=0)
        pt = np.stack(pt, axis=0)
        return x, y, pt


def main():
    loader = Loader()
    loader.mount('data/')
    A_list = [i for i in range(150, 3001, 285)]
    fc_list = [i for i in range(1000, 10001, 500)]
    fm_list = [i for i in range(10, 71, 10)]
    x, y, pt = loader.load_db(A_list, fc_list, fm_list)
    print(x.shape, y.shape, pt.shape)
    loader.mount('trained_models/')
    for key in fc_list:
        model = loader.load_model(key)
        print(model)

if __name__ == "__main__":
    main()
