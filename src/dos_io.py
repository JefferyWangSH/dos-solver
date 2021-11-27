import numpy as np


"""
    subroutine for the file output of calculated dos data
"""
def write_dos(freq_list, dos_list, output_file_name):
    assert(len(freq_list) == len(dos_list))
    assert(isinstance(output_file_name, str))
    
    # file output of data
    with open(output_file_name, mode="w") as outfile:
        for data in zip(freq_list, dos_list):
            outfile.write("{:<25.12f}{:<25.12f}\n".format(data[0], data[1]))
    outfile.close()


"""
    subroutine for reading dos data from file
"""
def read_dos(input_file_name):
    assert(isinstance(input_file_name, str))

    # read input data from file
    freq_list, dos_list = [], []
    with open(input_file_name, mode='r') as infile:
        for line in infile:
            if len(line) != 0:
                str_split = line.split()
                freq, dos = list(map(float, str_split))
                freq_list.append(freq)
                dos_list.append(dos)
    infile.close()
    return np.array(freq_list), np.array(dos_list)
   