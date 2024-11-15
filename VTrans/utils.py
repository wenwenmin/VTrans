import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import math, copy, time
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
import random
from torch import optim
from torch.autograd import Variable
import datetime
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, roc_curve, auc, recall_score, precision_score, accuracy_score, \
    matthews_corrcoef, f1_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt
from lifelines.statistics import logrank_test
import argparse




def KM(y_pred_list):
    t = []
    for i in range(len(y_pred_list)):
        for j in range(len(y_pred_list[i])):
            f = y_pred_list[i][j].cpu()
            t.append(f)
    t = np.array(t)

    indices_of_ones_1 = [index for index, label in enumerate(t) if label == 1]
    indices_of_ones_0 = [index for index, label in enumerate(t) if label == 0]
    test = pd.read_csv(
        'D:/TCGA/Data03_cBioPortal_tcga_microbiome_data/' + type + '_tcga_pan_can_atlas_2018' + '/test_kfold.csv',
        usecols=lambda column: column not in ['ENTITY_STABLE_ID'])

    test_id = np.array(test.columns).tolist()

    selected_user_ids_1 = [test_id[index] for index in indices_of_ones_1]
    selected_user_ids_0 = [test_id[index] for index in indices_of_ones_0]

    sm = pd.read_csv(
        'D:/TCGA/Data03_cBioPortal_tcga_microbiome_data/' + type + '_tcga_pan_can_atlas_2018' + '/Test_not_sort_data_kfold.csv')
    id = np.array(sm['ID']).tolist()
    months = np.array(sm['Months']).tolist()
    status = np.array(sm['Status']).tolist()
    indices_in_list_1 = [id.index(element) for element in selected_user_ids_1]
    indices_in_list_0 = [id.index(element) for element in selected_user_ids_0]

    time_1 = [months[index] for index in indices_in_list_1]
    status_1 = [status[index] for index in indices_in_list_1]

    time_0 = [months[index] for index in indices_in_list_0]
    status_0 = [status[index] for index in indices_in_list_0]

    for i in range(len(status_1)):
        if status_1[i] == '1:DECEASED':
            status_1[i] = 1
        else:
            status_1[i] = 0

    for i in range(len(status_0)):
        if status_0[i] == '1:DECEASED':
            status_0[i] = 1
        else:
            status_0[i] = 0

    kmf_group1 = KaplanMeierFitter()
    kmf_group2 = KaplanMeierFitter()

    kmf_group1.fit(durations=time_1, event_observed=status_1)
    kmf_group2.fit(durations=time_0, event_observed=status_0)

    results = logrank_test(time_1, time_0, event_observed_A=status_1, event_observed_B=status_0)
    p_value = results.p_value

    kmf_group1.plot_survival_function(label='Low-Risk', color='red', linestyle='-')
    kmf_group2.plot_survival_function(label='High-Risk', color='blue', linestyle='-')

    plt.title('')
    plt.xlabel('')
    plt.ylabel('')

    plt.fill_between(kmf_group1.survival_function_.index, 0, 1, facecolor='white', edgecolor='white')
    plt.fill_between(kmf_group2.survival_function_.index, 0, 1, facecolor='white', edgecolor='white')
    # 显示图例
    plt.legend(prop={'size': 12}, frameon=False)
    plt.text(30, 0.8, f'Log-Rank p-value: {p_value:.4f}', fontsize=15, color='black')
    # 显示图形
    plt.show()

    return t


def compute_saliency_maps(x, y, model, t, type):
    model.load_state_dict(torch.load('D:/TCGA/Data03_cBioPortal_tcga_microbiome_data/model/best_model_VTrans.pth'))
    model.eval()
    x, y = x.cuda(), y.cuda()
    x.requires_grad_()

    y_pred_1 = model(x)[0]

    target_score_indices = torch.argmax(y_pred_1, 1)
    target_scores = y_pred_1.gather(1, target_score_indices.view(-1, 1)).squeeze()

    loss = torch.sum(target_scores)
    loss.backward()

    hot_pred = y_pred_1.argmax(1)
    saliencies = x.grad.data.abs().detach().cpu()
    row_min = torch.min(saliencies, dim=1, keepdim=True)[0]
    row_max = torch.max(saliencies, dim=1, keepdim=True)[0]
    normalized_sa = (saliencies - row_min) / (row_max - row_min)

    # t = y.clone().detach()
    # t = torch.tensor(t)
    t = hot_pred.clone().detach()
    indices_1 = torch.nonzero(t).view(-1)
    indices_0 = torch.nonzero(t == 0).view(-1)
    print('predict_0 sample number: ', indices_0.shape)
    print('predict_1 sample number: ', indices_1.shape)

    data = pd.read_csv(
        'D:/TCGA/Data03_cBioPortal_tcga_microbiome_data/' + type + '_tcga_pan_can_atlas_2018' + '/test_kfold.csv')
    data = np.array(data)

    data_1 = data[:, indices_1.cpu()]
    data_0 = data[:, indices_0.cpu()]

    normalized_sa_1 = normalized_sa[indices_1]
    normalized_sa_0 = normalized_sa[indices_0]

    normalized_sa_0_squared = torch.pow(normalized_sa_0, 2)
    normalized_sa_0_sum = torch.sum(normalized_sa_0_squared, dim=0)
    scores_0 = torch.sqrt(normalized_sa_0_sum)

    top_values_0, top_indices_0 = torch.topk(scores_0, k=50)

    selected_elements_0 = data_0[top_indices_0]
    selected_elements_1 = data_1[top_indices_0]
    selected_elements_0_0 = torch.tensor(selected_elements_0)
    selected_elements_0_1 = torch.tensor(selected_elements_1)

    concatenated_tensor1 = torch.cat((selected_elements_0_0, selected_elements_0_1), dim=1)
    # normalized_concatenated_tensor1 = 6 * (concatenated_tensor1 - concatenated_tensor1.min(dim=1, keepdim=True)[0]) / (concatenated_tensor1.max(dim=1, keepdim=True)[0] - concatenated_tensor1.min(dim=1, keepdim=True)[0]) - 1
    min_val = concatenated_tensor1.min()
    max_val = concatenated_tensor1.max()
    normalized_concatenated_tensor1 = 20 * ((concatenated_tensor1 - min_val) / (max_val - min_val)) - 10

    y_labels = ["M1", "", "", "", "", "", "", "", "", "M10",
                "", "", "", "", "", "", "", "", "", "M20",
                "", "", "", "", "", "", "", "", "", "M30",
                "", "", "", "", "", "", "", "", "", "M40",
                "", "", "", "", "", "", "", "", "", "M50"]
    # 画热图
    plt.imshow(concatenated_tensor1, cmap='viridis', interpolation='nearest')
    # 添加颜色条
    plt.colorbar()
    plt.yticks(range(len(y_labels)), y_labels)
    # 显示图形
    plt.show()

    g0 = normalized_sa_0[:, top_indices_0]
    g1 = normalized_sa_1[:, top_indices_0]
    gg = torch.cat((g0, g1), dim=0)

    y_labels = ["M1", "", "", "", "", "", "", "", "", "M10",
                "", "", "", "", "", "", "", "", "", "M20",
                "", "", "", "", "", "", "", "", "", "M30",
                "", "", "", "", "", "", "", "", "", "M40",
                "", "", "", "", "", "", "", "", "", "M50"]
    # 画热图
    plt.imshow(gg.t(), cmap='viridis', interpolation='nearest')
    # 添加颜色条
    plt.colorbar()
    plt.yticks(range(len(y_labels)), y_labels)
    # 显示图形
    plt.show()

    normalized_sa_1_squared = torch.pow(normalized_sa_1, 2)
    normalized_sa_1_sum = torch.sum(normalized_sa_1_squared, dim=0)
    scores_1 = torch.sqrt(normalized_sa_1_sum)

    top_values_1, top_indices_1 = torch.topk(scores_1, k=50)

    selected_elements_0 = data_0[top_indices_1]
    selected_elements_1 = data_1[top_indices_1]
    selected_elements_1_0 = torch.tensor(selected_elements_0)
    selected_elements_1_1 = torch.tensor(selected_elements_1)

    concatenated_tensor2 = torch.cat((selected_elements_1_0, selected_elements_1_1), dim=1)
    # normalized_concatenated_tensor2 = 2 * (concatenated_tensor2 - concatenated_tensor2.min(dim=1, keepdim=True)[0]) / (concatenated_tensor2.max(dim=1, keepdim=True)[0] - concatenated_tensor2.min(dim=1, keepdim=True)[0]) - 1
    min_val = concatenated_tensor2.min()
    max_val = concatenated_tensor2.max()
    normalized_concatenated_tensor2 = 20 * ((concatenated_tensor2 - min_val) / (max_val - min_val)) - 10

    concatenated_tensor3 = torch.cat((concatenated_tensor1, concatenated_tensor2), dim=0)

    y_labels = ["M1", "", "", "", "", "", "", "", "", "M10",
                "", "", "", "", "", "", "", "", "", "M20",
                "", "", "", "", "", "", "", "", "", "M30",
                "", "", "", "", "", "", "", "", "", "M40",
                "", "", "", "", "", "", "", "", "", "M50"]
    # 画热图
    plt.imshow(concatenated_tensor2, cmap='viridis', interpolation='nearest')
    # 添加颜色条
    plt.colorbar()
    plt.yticks(range(len(y_labels)), y_labels)
    # 显示图形
    plt.show()

    g0 = normalized_sa_0[:, top_indices_1]
    g1 = normalized_sa_1[:, top_indices_1]
    gg = torch.cat((g0, g1), dim=0)

    y_labels = ["M1", "", "", "", "", "", "", "", "", "M10",
                "", "", "", "", "", "", "", "", "", "M20",
                "", "", "", "", "", "", "", "", "", "M30",
                "", "", "", "", "", "", "", "", "", "M40",
                "", "", "", "", "", "", "", "", "", "M50"]
    # 画热图
    plt.imshow(gg.t(), cmap='viridis', interpolation='nearest')
    # 添加颜色条
    plt.colorbar()
    plt.yticks(range(len(y_labels)), y_labels)
    # 显示图形
    plt.show()

    mirs_index_0 = top_indices_0
    mirs_index_1 = top_indices_1

    # km_number = 4
    for km_number in range(50):
        print(km_number + 1, top_indices_0[km_number])
        label_0 = []
        label_1 = []
        test = pd.read_csv(
            'D:/TCGA/Data03_cBioPortal_tcga_microbiome_data/' + type + '_tcga_pan_can_atlas_2018' + '/test_kfold.csv',
            header=0, index_col=False, usecols=lambda column: column not in ['ENTITY_STABLE_ID'])
        names = test.columns.tolist()
        test = np.array(test)
        m1 = test[mirs_index_0[km_number]]
        threshold = np.median(m1)

        for i in range(len(m1)):
            if m1[i] >= threshold:
                label_1.append(names[i])
            else:
                label_0.append(names[i])

        selected_user_ids_1 = label_1
        selected_user_ids_0 = label_0

        sm = pd.read_csv(
            'D:/TCGA/Data03_cBioPortal_tcga_microbiome_data/' + type + '_tcga_pan_can_atlas_2018' + '/Test_not_sort_data_kfold.csv')
        id = np.array(sm['ID']).tolist()
        months = np.array(sm['Months']).tolist()
        status = np.array(sm['Status']).tolist()
        indices_in_list_1 = [id.index(element) for element in selected_user_ids_1]
        indices_in_list_0 = [id.index(element) for element in selected_user_ids_0]

        time_1 = [months[index] for index in indices_in_list_1]
        status_1 = [status[index] for index in indices_in_list_1]

        time_0 = [months[index] for index in indices_in_list_0]
        status_0 = [status[index] for index in indices_in_list_0]

        for i in range(len(status_1)):
            if status_1[i] == '1:DECEASED':
                status_1[i] = 1
            else:
                status_1[i] = 0

        for i in range(len(status_0)):
            if status_0[i] == '1:DECEASED':
                status_0[i] = 1
            else:
                status_0[i] = 0

        kmf_group1 = KaplanMeierFitter()
        kmf_group2 = KaplanMeierFitter()

        kmf_group1.fit(durations=time_1, event_observed=status_1)
        kmf_group2.fit(durations=time_0, event_observed=status_0)

        results = logrank_test(time_1, time_0, event_observed_A=status_1, event_observed_B=status_0)
        p_value = results.p_value

        kmf_group1.plot_survival_function(label='Low-Risk', color='red', linestyle='-')
        kmf_group2.plot_survival_function(label='High-Risk', color='blue', linestyle='-')

        plt.title('')
        plt.xlabel('')
        plt.ylabel('')

        plt.fill_between(kmf_group1.survival_function_.index, 0, 1, facecolor='white', edgecolor='white')
        plt.fill_between(kmf_group2.survival_function_.index, 0, 1, facecolor='white', edgecolor='white')
        # 显示图例
        plt.legend(prop={'size': 12}, frameon=False)
        plt.text(30, 0.8, f'Log-Rank p-value: {p_value:.4f}', fontsize=15, color='black')
        # 显示图形
        plt.show()

    for km_number in range(50):
        print(km_number + 1, top_indices_1[km_number])
        label_0 = []
        label_1 = []
        test = pd.read_csv(
            'D:/TCGA/Data03_cBioPortal_tcga_microbiome_data/' + type + '_tcga_pan_can_atlas_2018' + '/test_kfold.csv',
            header=0, index_col=False, usecols=lambda column: column not in ['ENTITY_STABLE_ID'])
        names = test.columns.tolist()
        test = np.array(test)
        m1 = test[mirs_index_1[km_number]]
        threshold = np.median(m1)

        for i in range(len(m1)):
            if m1[i] >= threshold:
                label_1.append(names[i])
            else:
                label_0.append(names[i])

        selected_user_ids_1 = label_1
        selected_user_ids_0 = label_0

        sm = pd.read_csv(
            'D:/TCGA/Data03_cBioPortal_tcga_microbiome_data/' + type + '_tcga_pan_can_atlas_2018' + '/Test_not_sort_data_kfold.csv')
        id = np.array(sm['ID']).tolist()
        months = np.array(sm['Months']).tolist()
        status = np.array(sm['Status']).tolist()
        indices_in_list_1 = [id.index(element) for element in selected_user_ids_1]
        indices_in_list_0 = [id.index(element) for element in selected_user_ids_0]

        time_1 = [months[index] for index in indices_in_list_1]
        status_1 = [status[index] for index in indices_in_list_1]

        time_0 = [months[index] for index in indices_in_list_0]
        status_0 = [status[index] for index in indices_in_list_0]

        for i in range(len(status_1)):
            if status_1[i] == '1:DECEASED':
                status_1[i] = 1
            else:
                status_1[i] = 0

        for i in range(len(status_0)):
            if status_0[i] == '1:DECEASED':
                status_0[i] = 1
            else:
                status_0[i] = 0

        kmf_group1 = KaplanMeierFitter()
        kmf_group2 = KaplanMeierFitter()

        kmf_group1.fit(durations=time_1, event_observed=status_1)
        kmf_group2.fit(durations=time_0, event_observed=status_0)

        results = logrank_test(time_1, time_0, event_observed_A=status_1, event_observed_B=status_0)
        p_value = results.p_value

        kmf_group1.plot_survival_function(label='Low-Risk', color='red', linestyle='-')
        kmf_group2.plot_survival_function(label='High-Risk', color='blue', linestyle='-')

        plt.title('')
        plt.xlabel('')
        plt.ylabel('')

        plt.fill_between(kmf_group1.survival_function_.index, 0, 1, facecolor='white', edgecolor='white')
        plt.fill_between(kmf_group2.survival_function_.index, 0, 1, facecolor='white', edgecolor='white')
        # 显示图例
        plt.legend(prop={'size': 12}, frameon=False)
        plt.text(30, 0.8, f'Log-Rank p-value: {p_value:.4f}', fontsize=15, color='black')
        # 显示图形
        plt.show()

    sp = pd.read_csv('D:/TCGA/Data03_cBioPortal_tcga_microbiome_data/' + '/microbiome_name.csv',
                     usecols=['ENTITY_STABLE_ID'])
    mic_names = np.array(sp)[:, 0]
    mic_0_scores = np.array(scores_1)

    # 获取排序后的索引
    sorted_indices = np.argsort(mic_0_scores)

    # 对数组进行排序
    sorted_a1 = mic_0_scores[sorted_indices][-50:]
    sorted_a2 = mic_names[sorted_indices][-20:]

    x = np.arange(50)
    y = sorted_a1

    print(sorted_a2)
    print(top_indices_0)

    # 绘制散点图
    plt.scatter(x, y, color='blue', marker='o', label='microbiomes')

    # 用线连接点
    plt.plot(x, y, color='red', linestyle='-', linewidth=2)

    # plt.xticks(x[::1], person_names[::1], rotation=45, ha="right")

    # 显示图例
    plt.legend()

    # 显示图形
    plt.show()


t = KM(y_pred_list_true)
compute_saliency_maps(data_set[3].features, data_set[3].labels, model, t, config.type)