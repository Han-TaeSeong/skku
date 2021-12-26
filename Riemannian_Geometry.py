import numpy as np
from numpy import linalg
import torch
import matplotlib.pyplot as plt
import time

start = time.time()


################################################################################ 평균이용
accu = list()
accu_sum = 0
for z in range(1,10):
    z = str(z)
    train_file = np.load('.\\Post_Research\\Preprocessed_Data\\3s - 5.5s\\4-40Hz_BPF\\A0'+z+'T.npz')
    test_file = np.load('.\\Post_Research\\Preprocessed_Data\\3s - 5.5s\\4-40Hz_BPF\\A0'+z+'E.npz')
    x_train = train_file['x']
    y_train = train_file['y']
    x_test = test_file['x']
    y_test = test_file['y']

    x_train_SCM = ""  #### Sample Covariance Matrix
    for i in range(0, 288):
        scm = x_train[i]
        scm = (np.matmul(scm, scm.transpose())) / 625  #### time_series  로 나누어 주기(scaling)
        if i == 0:
            x_train_SCM = np.expand_dims(scm, axis=0)
        else:
            scm = np.expand_dims(scm, axis=0)
            x_train_SCM = np.concatenate((x_train_SCM, scm), axis=0)

    x_test_SCM = ""
    for i in range(0, 288):
        scm = x_test[i]
        scm = (np.matmul(scm, scm.transpose())) / 625  #### time_series  로 나누어 주기(scaling)
        if i == 0:
            x_test_SCM = np.expand_dims(scm, axis=0)
        else:
            scm = np.expand_dims(scm, axis=0)
            x_test_SCM = np.concatenate((x_test_SCM, scm), axis=0)

    correct = 0
    for i in range(0, 288):
        P1 = x_test_SCM[i]
        P1_inverse = linalg.inv(P1)
        distance_1 = 0
        distance_2 = 0
        distance_3 = 0
        distance_4 = 0

        test_class = 0
        for j in range(0, 288):
            P2 = x_train_SCM[j]

            matrix = np.matmul(P1_inverse, P2)
            eigenvalue, eigenvector = linalg.eig(matrix)
            sum = 0
            for k in eigenvalue:
                t = np.log(k)
                s = t ** 2
                sum += s

            if y_train[j] == 1:
                distance_1 += sum
            elif y_train[j] == 2:
                distance_2 += sum
            elif y_train[j] == 3:
                distance_3 += sum
            else:
                distance_4 += sum
        # print(f' Distance : {distance}  Index : {index} Class : {test_class}')
        xx = np.array([distance_1, distance_2, distance_3, distance_4])
        test_class = np.argmin(xx) + 1

        if test_class == y_test[i]:
            correct += 1

    accuracy = correct / 288
    accu_sum += accuracy
    print(f'Accuracy of subject{z} : {accuracy}')
    accu.append(accuracy)
accu_avg=accu_sum/9
print(accu_avg)

sub = list()
for i in range(1, 10):
    i = str(i)
    sub.append('subject'+i)
accu.append(accu_avg)
sub.append('average')

plt.bar(sub, accu)
plt.title(f'Riemannian')
plt.xlabel('Subject')
plt.ylabel('Accuracy')
plt.show()
################################################################################ 평균이용




######################################################################## cross subject
# accu = list()
# accu_sum = 0
# for z in range(1,10):
#     z = str(z)
#     x_train = None  ##2592,22,625
#     y_train = None
#     for w in range(1, 10):
#         w = str(w)
#         train_file = np.load('.\\Preprocessed_Data\\4_40Hz\\A0' + w + 'T.npz')
#         if x_train is None:
#             x_train = train_file['x']
#             y_train = train_file['y']
#         else:
#             x_train = np.concatenate((x_train, train_file['x']), axis=0)
#             y_train = np.append(y_train,train_file['y'])
#     test_file = np.load('.\\Preprocessed_Data\\4_40Hz\\A0'+z+'E.npz')
#     x_test = test_file['x']
#     y_test = test_file['y']
#     x_train_SCM = ""  #### Sample Covariance Matrix
#     for i in range(0, 2592):
#         scm = x_train[i]
#         scm = (np.matmul(scm, scm.transpose())) / 625  #### time_series  로 나누어 주기(scaling)
#         if i == 0:
#             x_train_SCM = np.expand_dims(scm, axis=0)
#         else:
#             scm = np.expand_dims(scm, axis=0)
#             x_train_SCM = np.concatenate((x_train_SCM, scm), axis=0)
#
#     x_test_SCM = ""
#     for i in range(0, 288):
#         scm = x_test[i]
#         scm = (np.matmul(scm, scm.transpose())) / 625  #### time_series  로 나누어 주기(scaling)
#         if i == 0:
#             x_test_SCM = np.expand_dims(scm, axis=0)
#         else:
#             scm = np.expand_dims(scm, axis=0)
#             x_test_SCM = np.concatenate((x_test_SCM, scm), axis=0)
#
#     correct = 0
#     for i in range(0, 288):
#         P1 = x_test_SCM[i]
#         P1_inverse = linalg.inv(P1)
#         distance_1 = 0
#         distance_2 = 0
#         distance_3 = 0
#         distance_4 = 0
#
#         test_class = 0
#         for j in range(0, 2592):
#             P2 = x_train_SCM[j]
#
#             matrix = np.matmul(P1_inverse, P2)
#             eigenvalue, eigenvector = linalg.eig(matrix)
#             sum = 0
#             for k in eigenvalue:
#                 t = np.log(k)
#                 s = t ** 2
#                 sum += s
#
#             if y_train[j] == 1:
#                 distance_1 += sum
#             elif y_train[j] == 2:
#                 distance_2 += sum
#             elif y_train[j] == 3:
#                 distance_3 += sum
#             else:
#                 distance_4 += sum
#         # print(f' Distance : {distance}  Index : {index} Class : {test_class}')
#         xx = np.array([distance_1, distance_2, distance_3, distance_4])
#         test_class = np.argmin(xx) + 1
#
#         if test_class == y_test[i]:
#             correct += 1
#
#     accuracy = correct / 288
#     accu_sum += accuracy
#     print(f'Accuracy of subject{z} : {accuracy}')
#     accu.append(accuracy)
# accu_avg=accu_sum/9
# print(accu_avg)
#
# sub = list()
# for i in range(1, 10):
#     i = str(i)
#     sub.append('subject'+i)
# accu.append(accu_avg)
# sub.append('average')
#
# plt.bar(sub, accu)
# plt.title(f'Riemannian')
# plt.xlabel('Subject')
# plt.ylabel('Accuracy')
# plt.show()
######################################################################## cross subject





################################################################## pilot data
# train_file = np.load('.\\Preprocessed_Data\\0.5_40Hz\\A06T.npz')
# test_file = np.load('.\\Preprocessed_Data\\0.5_40Hz\\A06E.npz')
#
# x_train = train_file['x']
# y_train = train_file['y']
# x_test = test_file['x']
# y_test = test_file['y']
################################################################## pilot data

##################################################################  Pi 구현
# x_train_SCM = ""  #### Sample Covariance Matrix
# for i in range(0, 288):
#     z = x_train[i]
#     z = (np.matmul(z, z.transpose())) / 649  #### time_series -1 로 나누어 주기(scaling)
#     if i == 0:
#         x_train_SCM = np.expand_dims(z, axis=0)
#     else:
#         z = np.expand_dims(z, axis=0)
#         x_train_SCM = np.concatenate((x_train_SCM, z), axis=0)
#
# x_test_SCM = ""
# for i in range(0, 288):
#     z = x_test[i]
#     z = (np.matmul(z, z.transpose())) / 649  #### time_series -1 로 나누어 주기(scaling)
#     if i == 0:
#         x_test_SCM = np.expand_dims(z, axis=0)
#     else:
#         z = np.expand_dims(z, axis=0)
#         x_test_SCM = np.concatenate((x_test_SCM, z), axis=0)
##################################################################  Pi 구현


##################################################################### 모델 기초
#### test Pi 와 Riemmanian geodesic distance 구하기   inverse(P1) * P2  --> eigenvalue 구하고-->로그취하고 --> summation
# P1_inverse = linalg.inv(x_test_SCM[0])
# P2 = x_test_SCM[1]
#
# matrix = np.matmul(P1_inverse, P2)
# eigenvalue, eigenvector = linalg.eig(matrix)
#
# sum = 0
# for i in eigenvalue:
#     t = np.log(i)
#     s = t ** 2
#     sum += s
#
# distance = np.sqrt(sum)
# print(distance)
##################################################################### 모델 기초


################################################################################
##################### 제일 가까운 거리를 이용한 분류 모델
# correct =0
# for i in range(0, 288):
#     P1 = x_test_SCM[i]
#     P1_inverse = linalg.inv(P1)
#     distance = 0
#     index = 0
#     test_class = y_train[0]
#     for j in range(0, 288):
#         P2 = x_train_SCM[j]
#
#         matrix = np.matmul(P1_inverse, P2)
#         eigenvalue, eigenvector = linalg.eig(matrix)
#         sum = 0
#         for k in eigenvalue:
#             t = np.log(k)
#             s = t ** 2
#             sum += s
#
#         if j == 0:
#             # distance = np.sqrt(sum)
#             distance = sum
#         else:
#             # if distance > np.sqrt(sum):
#             if distance > sum:
#                 # distance = np.sqrt(sum)
#                 distance = sum
#                 index = j
#                 test_class = y_train[j]
#
#     # print(f' Distance : {distance}  Index : {index} Class : {test_class}')
#     if test_class == y_test[i]:
#         correct +=1
#
# print(correct/288)
##################### 제일 가까운 거리를 이용한 분류 모델
################################################################################


################################################################################
#############################  클래스간 평균 방법
# correct = 0
# a = 0
# b = 0
# c = 0
# d = 0
# for i in range(0, 288):
#     P1 = x_test_SCM[i]
#     P1_inverse = linalg.inv(P1)
#
#     distance_1 = 0
#     distance_2 = 0
#     distance_3 = 0
#     distance_4 = 0
#
#     test_class = 0
#     for j in range(0, 288):
#         P2 = x_train_SCM[j]
#
#         matrix = np.matmul(P1_inverse, P2)
#         eigenvalue, eigenvector = linalg.eig(matrix)
#         sum = 0
#         for k in eigenvalue:
#             t = np.log(k)
#             s = t ** 2
#             sum += s
#
#             # sum = np.sqrt(sum)
#
#         if y_train[j] == 1:
#             distance_1 += sum
#         elif y_train[j] == 2:
#             distance_2 += sum
#         elif y_train[j] == 3:
#             distance_3 += sum
#         else:
#             distance_4 += sum
#     # print(f' Distance : {distance}  Index : {index} Class : {test_class}')
#     xx = np.array([distance_1, distance_2, distance_3, distance_4])
#     test_class = np.argmin(xx) + 1
#
#     if test_class == y_test[i]:
#         correct += 1
#
#     if test_class == 1:
#         a += 1
#     elif test_class == 2:
#         b += 1
#     elif test_class == 3:
#         c += 1
#     else:
#         d += 1
#
# print(a, b, c, d)
# print(correct / 288)
#############################  클래스간 평균 방법
################################################################################



################################################################################ 최소거리 이용
# accu = list()
# accu_sum = 0
# for z in range(1,10):
#     z = str(z)
#     train_file = np.load('.\\Preprocessed_Data\\0.5_40Hz\\A0'+z+'T.npz')
#     test_file = np.load('.\\Preprocessed_Data\\0.5_40Hz\\A0'+z+'E.npz')
#     x_train = train_file['x']
#     y_train = train_file['y']
#     x_test = test_file['x']
#     y_test = test_file['y']
#
#     x_train_SCM = ""  #### Sample Covariance Matrix
#     for i in range(0, 288):
#         scm = x_train[i]
#         scm = (np.matmul(scm, scm.transpose())) / 625  #### time_series -1 로 나누어 주기(scaling)
#         if i == 0:
#             x_train_SCM = np.expand_dims(scm, axis=0)
#         else:
#             scm = np.expand_dims(scm, axis=0)
#             x_train_SCM = np.concatenate((x_train_SCM, scm), axis=0)
#
#     x_test_SCM = ""
#     for i in range(0, 288):
#         scm = x_test[i]
#         scm = (np.matmul(scm, scm.transpose())) / 625  #### time_series -1 로 나누어 주기(scaling)
#         if i == 0:
#             x_test_SCM = np.expand_dims(scm, axis=0)
#         else:
#             scm = np.expand_dims(scm, axis=0)
#             x_test_SCM = np.concatenate((x_test_SCM, scm), axis=0)
#
#     correct = 0
#     for i in range(0, 288):
#         P1 = x_test_SCM[i]
#         P1_inverse = linalg.inv(P1)
#         distance = 0
#         index = 0
#         test_class = y_train[0]
#         for j in range(0, 288):
#             P2 = x_train_SCM[j]
#
#             matrix = np.matmul(P1_inverse, P2)
#             eigenvalue, eigenvector = linalg.eig(matrix)
#             sum = 0
#             for k in eigenvalue:
#                 t = np.log(k)
#                 s = t ** 2
#                 sum += s
#
#             if j == 0:
#                 # distance = np.sqrt(sum)
#                 distance = sum
#             else:
#                 # if distance > np.sqrt(sum):
#                 if distance > sum:
#                     # distance = np.sqrt(sum)
#                     distance = sum
#                     index = j
#                     test_class = y_train[j]
#
#         # print(f' Distance : {distance}  Index : {index} Class : {test_class}')
#         if test_class == y_test[i]:
#             correct += 1
#
#     accuracy = correct / 288
#     accu_sum += accuracy
#     print(f'Accuracy of subject{z} : {accuracy}')
#     accu.append(accuracy)
# accu_avg=accu_sum/9
# print(accu_avg)
#
# sub = list()
# for i in range(1, 10):
#     i = str(i)
#     sub.append('subject'+i)
# accu.append(accu_avg)
# sub.append('average')
#
# plt.bar(sub, accu)
# plt.title(f'Riemannian')
# plt.xlabel('Subject')
# plt.ylabel('Accuracy')
# plt.show()
################################################################################ 최소거리 이용






t = time.time() - start
if t > 60:
    print(f'{t / 60:.2f} Min')
else:
    print(f'{t:.2f} Sec')