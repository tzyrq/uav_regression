import numpy as np

if __name__ == "__main__":

    '''
    data: (3000, 60, 15, 5)
    label: (3000, 60, 15, 5)
    init_data: (3000, 100, 100)
    init_label: (3000, 100, 100)
    '''

    data_path = '/data/zzhao/uav_regression/main_test/data_tasks.npy'
    label_path = '/data/zzhao/uav_regression/main_test/training_label_density.npy'
    init_path = '/data/zzhao/uav_regression/main_test/data_init_density.npy'
    init_label = '/data/zzhao/uav_regression/main_test/label_T1_10s.npy'

    data = np.load(data_path)
    label = np.load(label_path)
    init_data = np.load(init_path)
    init_label = np.load(init_label)

    print("data:", data.shape)
    print("label:", data.shape)
    print("init_data:", init_data.shape)
    print("init_label:", init_label.shape)
