import numpy as np
import yaml
import json
import pandas as pd
import os
import hashlib

def load_data(file_path):
    try:
        return np.loadtxt(file_path)
    except IOError as e:
        print(f"Error loading file {file_path}: {e}")
        return None

def process_and_plot_data(data, ax0):
    size_list = np.size(data[:, 0])
    stability_current = data[0, 2]
    load_1 = np.array([data[0, 0]])
    energy_1 = np.array([data[0, 1]])
    
    for ind_e in range(size_list):
        load_t = data[ind_e, 0]
        if -stability_current == data[ind_e, 2] or ind_e == (size_list - 1):
            load_mean = (data[ind_e - 1, 0] + data[ind_e, 0]) / 2
            e_mean = (data[ind_e - 1, 1] + data[ind_e, 1]) / 2
            # print('np.array([%.10f, %.10f])' % (load_mean, e_mean))
            if np.size(load_1) > 1:
                load_1[-1] = load_mean
                energy_1[-1] = e_mean
            if stability_current == 1:
                ax0.plot(load_1, energy_1, c='C0', linewidth=3)
            elif stability_current == -1:
                ax0.plot(load_1, energy_1, c='C1', linewidth=3)
            stability_current = data[ind_e, 2]
            load_1 = np.array(load_mean)
            energy_1 = np.array(e_mean)
            # ax0.axvline(load_1)
        else:
            load_1 = np.append(load_1, data[ind_e, 0])
            energy_1 = np.append(energy_1, data[ind_e, 1])


def load_json_data(rootdir):

    # with open(rootdir + '/parameters.pkl', 'r') as f:
    # 	params = json.load(f)

    with open(rootdir + '/parameters.yaml') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    try:
        with open(rootdir + '/time_data.json', 'r') as f:
            data = json.load(f)
            dataf = pd.DataFrame(data).sort_values('load')
        # Continue with your code using the dataf DataFrame
    except FileNotFoundError:
        print("File 'time_data.json' not found. Handle this case accordingly.")
        dataf = pd.DataFrame()

    if os.path.isfile(rootdir + '/signature.md5'):
        #         print('sig file found')
        with open(rootdir + '/signature.md5', 'r') as f:
            signature = f.read()
    else:
        print('no sig file found')
        signature = hashlib.md5(str(params).encode('utf-8')).hexdigest()

    return params, dataf, signature
