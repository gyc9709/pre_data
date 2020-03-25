import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from scipy import interpolate
from scipy.interpolate import interp1d
from scipy.interpolate import spline

def softsaccade():

    path = 'D:/Desktop/softSaccadicModel/results_gyc'
    outpath = 'D:\Desktop\softSaccadicModel\oliver_result500'
    ID = []
    for i in range(1, 3 + 1):
        for j in range(1, 500 + 1):
            ID.append([i, j])
    ID = np.array(ID)
    for each in os.listdir(path):
        data = np.loadtxt(path + '/' + each)
        ID_list = []   # 随机得到3个ID
        for i in range(3):
            ID_list.append(random.randint(1, 10))
        ID_list.sort()
        time = int(len(data)/10)
        sam_data = []
        data = np.ndarray.tolist(data)
        for i in ID_list:
            for index, item in enumerate(data):
                if item[0] == i:
                    sam_data.append(item)
        # print(len(sam_data))
        sam_data = np.array(sam_data).T
        lon_new_list = []
        lat_new_list = []
        sac_list = []
        for i in range(3):
            lon = np.array(sam_data[2][i*10: (i+1)*10])
            lat = np.array(sam_data[3][i*10: (i+1)*10])
            # print(lon)
            # print(lat)
            # print(each)
            # print('-----------------------------')

            x = np.array(range(10))
            y = np.array(range(2))
            a = sam_data[2:4, i*10: (i+1)*10]
            f = interpolate.interp2d(x, y, a, kind='linear')
            xnew = np.linspace(0, 9, 500)
            ynew = np.array([0, 1])
            znew = f(xnew, ynew)
            # print(znew[0])
            # print(znew[1])
            znew = np.array(znew).T
            znew = np.ndarray.tolist(znew)
            sac_list = sac_list + znew
        sac_list = np.array(sac_list)

        data_out = np.hstack((ID, sac_list))
        each = each.replace('_sal', '')
        out = outpath + '/' + each
        np.savetxt(out, data_out)
        # print(data_out)
        # print(len(data_out))


            #  B 样化插值
            # tck, u = interpolate.splprep([lon, lat], s=0)
            # lon_new, lat_new = interpolate.splev(np.linspace(0, 1, 500), tck)
            # lon_new_list = lon_new_list + lon_new
            # lat_new_list = lat_new_list + lat_new


            # plt.figure()
            # plt.subplot(121)
            # plt.plot(lon, lat)
            # plt.subplot(122)
            # plt.plot(znew[0], znew[1])
            # plt.show()
# softsaccade()


def Zhu():
    path = 'D:/Desktop/Omnidirectional/results/week14/ResultsZhu/H/Scanpaths'
    outpath = 'D:/Desktop/Omnidirectional/results/week14/ResultsZhu/Zhu_result500'
    ID = []
    for i in range(1, 3 + 1):
        for j in range(1, 500 + 1):
            ID.append([i, j])
    ID = np.array(ID)


    for each in os.listdir(path):
        each_path = path + '/' + each
        f = open(each_path, 'r')
        lines = f.readlines()
        del lines[0]
        sac75 = []
        sac60 = []
        sac40 = []
        for index, line in enumerate(lines):
            if index <= 75:
                order = line.split(',')[0]
                lon = float(line.split(',')[1])*360-180
                lat = float(line.split(',')[2])*180-90
                sac75.append([lon, lat])
            elif index > 75 and index <= 136:
                order = line.split(',')[0]
                lon = float(line.split(',')[1])*360-180
                lat = float(line.split(',')[2])*180-90
                sac60.append([lon, lat])
            else:
                order = float(line.split(',')[0])
                lon = float(line.split(',')[1])*360-180
                lat = float(line.split(',')[2])*180-90
                sac40.append([lon, lat])
        sac75 = np.array(sac75).T
        sac60 = np.array(sac60).T
        sac40 = np.array(sac40).T

        #75
        x = np.array(range(76))
        y = np.array(range(2))
        a = sac75
        f = interpolate.interp2d(x, y, a, kind='linear')
        xnew = np.linspace(0, 75, 500)
        ynew = np.array([0, 1])
        znew = f(xnew, ynew)
        sac75 = znew.T


        #60
        x = np.array(range(61))
        y = np.array(range(2))
        a = sac60
        f = interpolate.interp2d(x, y, a, kind='linear')
        xnew = np.linspace(0, 60, 500)
        ynew = np.array([0, 1])
        znew = f(xnew, ynew)
        sac60 = znew.T


        #40
        x = np.array(range(41))
        y = np.array(range(2))
        a = sac40
        f = interpolate.interp2d(x, y, a, kind='linear')
        xnew = np.linspace(0, 40, 500)
        ynew = np.array([0, 1])
        znew = f(xnew, ynew)
        sac40 = znew.T

        sac = np.vstack((sac75, sac60))
        sac = np.vstack((sac, sac40))

        sac = np.hstack((ID, sac))
        print(sac)
        print(sac.shape)
        each_outpath = outpath + '/' + each
        np.savetxt(each_outpath, sac)

# Zhu()


def Salient360():

    path = 'D:/Desktop/Omnidirectional/code/Github/saliency-360salient-2017/results_all'
    out_path = 'D:\Desktop\Omnidirectional\code\Github\saliency-360salient-2017\salient500'
    ID = []
    for i in range(1, 3 + 1):
        for j in range(1, 500 + 1):
            ID.append([i, j])
    ID = np.array(ID)
    for each in os.listdir(path):
        sac_img = np.array([0, 0])
        each_path = path + '/' + each
        rawdata = np.loadtxt(each_path, usecols=(3, 4))/(40/9)-np.array([180, 90])
        times = int(len(rawdata)/1000)
        # print(times)

        ID_list = []  # 随机得到3个ID
        for i in range(3):
            ID_list.append(random.randint(1, 20))
        ID_list.sort()
        # print(ID_list)
        for i in ID_list:
            sac = rawdata[(i-1)*1000: (i-1)*1000+10].T
            x = np.array(range(10))
            y = np.array(range(2))
            a = sac
            f = interpolate.interp2d(x, y, a, kind='linear')
            xnew = np.linspace(0, 9, 500)
            ynew = np.array([0, 1])
            znew = f(xnew, ynew)
            sac = znew.T  # 竖
            sac_img = np.vstack((sac_img, sac))
        sac_img = np.delete(sac_img, 0, axis=0)
        out = np.hstack((ID, sac_img))
        out_path_img =out_path + '/' + each
        np.savetxt(out_path_img, out)


# Salient360()

def name_change_Salient360():
    path = 'D:/Desktop/Omnidirectional/code/Github/saliency-360salient-2017/results_all'
    name_path = 'D:\Desktop\Omnidirectional\code\Github\saliency-360salient-2017\SaltiNet_name.txt'
    f = open(name_path, 'r')
    lines = f.readlines()
    i = 0
    for each in os.listdir(path):
        name = lines[i].replace('.jpg\n', '') + '.txt'
        i += 1
        os.rename(path + '/' + each, path + '/' + name)
        print(name)
# name_change_Salient360()


def data_2rdn_500():
    path = r'D:\Desktop\Omnidirectional\dataset\data_2rdn_vision'
    out_path = 'D:\Desktop\Omnidirectional\dataset\data_2rdn_vision_500'
    ID = []
    for i in range(1, 30 + 1):
        for j in range(1, 500 + 1):
            ID.append([i, j])
    ID = np.array(ID)
    for each in os.listdir(path):
        each_path = path + '/' + each
        data = np.loadtxt(each_path, usecols=(0, 1, 2, 3))
        times = 30
        data_sam = []
        for i in range(1, 30+1):
            data_id = []
            for item in data:
                if int(item[0]) == i:
                    data_id.append(item[2:4])
            # print(len(data_id))
            data_sam_id = []
            for j in range(500):
                data_sam_id.append(data_id[j*2])
            # print(len(data_sam_id))
            data_sam = data_sam + data_sam_id
        data_sam = np.array(data_sam)
        data_sam = np.hstack((ID, data_sam))
        each_out_path = out_path + '/' + each
        np.savetxt(each_out_path, data_sam)
        # print(data_sam.shape)

# data_2rdn_500()

def select_test():
    test_path = r'test_list_name.txt'
    path = r'D:\Desktop\Omnidirectional\results\week14\ResultsZhu\DTW_Zhu'
    outpath = r'D:\Desktop\Omnidirectional\results\week15\DTW_Zhu_test'
    list = []
    fo = open(test_path)
    for line in fo.readlines():
        line = line.replace('\n', '.txt')
        list.append(line)
    # print(list)
    for each in os.listdir(path):
        if each in list:
            data = np.loadtxt(path + '/' + each)
            np.savetxt(outpath + '/' + each, data)

# select_test()


def mean_var():
    path = r'D:\Desktop\Omnidirectional\results\week15\DTW_Zhu_test'
    mean_sum = 0
    std_sum = 0
    for each in os.listdir(path):
        data = np.loadtxt(path + '/' + each, usecols=2)
        mean = np.mean(data)
        mean_sum += mean
        std = np.std(data)
        std_sum += std
    print(mean_sum/180, std_sum/180)
# mean_var()


def mean_var_video_DTW():
    path = r'D:\Desktop\Omnidirectional\code\DTW_video\DTW_video_1127_3col.txt'
    data = np.loadtxt(path)
    print(np.mean(data), np.std(data))

# mean_var_video_DTW()


def figure_trajectory():
    path = r'D:\Desktop\softSaccadicModel\results_gyc\human_P0_sal.txt'
    data = np.loadtxt(path, usecols=(2, 3)).T
    plt.figure()
    plt.plot(data[0, 10:20], data[1, 10:20])
    plt.show()
    print(min(data[0]), min(data[1]))

figure_trajectory()



def test():
    x = np.array([23, 24, 24, 25, 25])
    y = np.array([13, 12, 13, 12, 13])

    # append the starting x,y coordinates
    # x = np.r_[x, x[0]]
    # y = np.r_[y, y[0]]

    print(x)
    print(y)

    tck, u = interpolate.splprep([x, y], s=0, per=False)

    # evaluate the spline fits for 1000 evenly spaced distance values
    xi, yi = interpolate.splev(np.linspace(0, 1, 1000), tck)
    plt.figure()
    # plot the result
    fig, ax = plt.subplots(1, 1)
    ax.plot(x, y, 'or')
    ax.plot(xi, yi, '-b')
    plt.show()


def test2():
    import numpy as np
    import pylab as pl
    from scipy import interpolate
    import matplotlib.pyplot as plt

    x = np.linspace(0, 2 * np.pi + np.pi / 4, 10)
    y = np.sin(x)

    x_new = np.linspace(0, 2 * np.pi + np.pi / 4, 100)
    f_linear = interpolate.interp1d(x, y)
    print(f_linear(x_new))

    plt.xlabel(u'安培/A')
    plt.ylabel(u'伏特/V')

    plt.plot(x, y, "o", label=u"原始数据")
    plt.plot(x_new, f_linear(x_new), label=u"线性插值")

    pl.legend()
    pl.show()

# test2()
