import csv
import numpy as np

def csv_files_equal(file1, file2):
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        reader1 = csv.reader(f1)
        reader2 = csv.reader(f2)

        for i, (row1, row2) in enumerate(zip(reader1, reader2), start=1):
            if row1 != row2:
                print(f"第 {i} 行不相等:")
                print(f"文件1: {row1}")
                print(f"文件2: {row2}")
        
        try:
            next(reader1)
            print("文件1有剩余行")
            return False
        except StopIteration:
            pass

        try:
            next(reader2)
            print("文件2有剩余行")
            return False
        except StopIteration:
            pass 



def numpy_files_equal(file1, file2):
    data1 = np.load(file1, allow_pickle=True)
    data2 = np.load(file2, allow_pickle=True)

    if not np.array_equal(data1, data2):
        print("数组内容不相等")
        if data1.shape == data2.shape:
            diff_indices = np.where(data1 != data2)
            for index in zip(*diff_indices):
                print(f"位置 {index} 不同: 文件1 = {data1[index]}, 文件2 = {data2[index]}")
        else:
            print("数组形状不同:")
            print(f"文件1形状: {data1.shape}")
            print(f"文件2形状: {data2.shape}")
    
    else:
        print("两个NumPy文件内容相等")


if __name__ == "__main__":
    file1 = './data/fill_inside/6531eca5c059807382725fd5.npy'
    file2 = './data/fill_inside/HCH_PRT.npy'

    numpy_files_equal(file1, file2)
