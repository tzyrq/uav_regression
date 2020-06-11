import numpy as np
import matplotlib
import matplotlib.pyplot as plt


if __name__ == "__main__":

    '''
    data: (3000, 60, 15, 5)
    label: (3000, 60, 15, 5)
    init_data: (3000, 100, 100)
    init_label: (3000, 100, 100)
    '''

    """
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
    """

    pathInitMin = "/home/share_uav/ruiz/data/minmax/dataset/initialMin.npy"
    pathInitMax = "/home/share_uav/ruiz/data/minmax/dataset/initialMax.npy"
    pathLabelMin = "/home/share_uav/ruiz/data/minmax/dataset/labelMin.npy"
    pathLabelMax = "/home/share_uav/ruiz/data/minmax/dataset/labelMax.npy"

    initMin = np.load(pathInitMin)
    initMax = np.load(pathInitMax)
    labelMin = np.load(pathLabelMin)
    labelMax = np.load(pathLabelMax)

    dataInitMin = np.squeeze(initMin[0])
    dataInitMax = np.squeeze(initMax[0])
    dataLabelMin = np.squeeze(labelMin[0])
    dataLabelMax = np.squeeze(labelMax[0])

    # fig, ax = plt.subplots(2, 2, dpi=200, figsize=(10, 10))
    # im0 = ax[0, 0].imshow(dataInitMin)
    # ax[0, 0].set_title("initial min map", fontsize=20)
    #
    # im1 = ax[0, 1].imshow(dataInitMax)
    # ax[0, 1].set_title("initial max map", fontsize=20)
    #
    # im2 = ax[1, 0].imshow(dataLabelMin)
    # ax[1, 0].set_title("label min map")
    #
    # im3 = ax[1, 1].imshow(dataLabelMax)
    # ax[1, 1].set_title("label max map")
    #
    # plt.show()
    # plt.close(fig)




