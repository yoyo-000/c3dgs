import numpy as np
import os

def merge_npz_files(input_files, output_file):
    data = {}
    for file in input_files:
        with np.load(file) as npz_file:
            for key, value in npz_file.items():
                print(f"Key: {key}, Size: {value.shape}")
                # if key in data:
                #     data[key] = np.concatenate((data[key], value))
                # else:
                #     data[key] = value

    np.savez(output_file, **data)

if __name__ == "__main__":
    data_dir = r'D:\code\GS\c3dgs\output\w-b'
    input_files = []
    for i in range(1, 3):
        input_files.append(os.path.join(data_dir, str(i),"point_cloud","iteration_35000","point_cloud.npz"))
    output_file = os.path.join(data_dir, 'output.npz')
    
    merge_npz_files(input_files, output_file)