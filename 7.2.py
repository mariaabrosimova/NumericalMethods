import numpy as np
import matplotlib.pyplot as plt
import math
h = 0.0001
a = 0
b = 1
n = round((b - a)/h)
A = np.array([
    [-269.264, -47.169, -33.887],
    [-56.642, -18.312, 115.518],
    [12.843, -119.698, -8.424],
    ])
Y0 = np.array([1.6, 4.4, 2.4])
B = np.array([
    [-11.485, 63.977, -62.832],
    [-64.673, -11.25, 86.831],
    [62.115, -87.346, -11.265],
    ])
Z0 = np.array([0.8, 4.4, 1.6])

A_lambda =np.linalg.eig(A)[0]
print ("Массив из собственных чисел матрицы А:", A_lambda)

B_lambda =np.linalg.eig(B)[0]
print ("Массив из собственных чисел матрицы B:", B_lambda)

def system_stiffness(x):
    return np.max(np.abs(x.real))/np.min(np.abs(x.real))
print("Жесткость системы А:", system_stiffness(A_lambda.real))
print("Жесткость системы B:", system_stiffness(B_lambda.real))

abs_lambda = np.max(abs(A_lambda.real))
print("|lambda|:", abs_lambda)
h_t= 2/abs_lambda
print("Теоретическая оценка: h<= ", h_t)

#---------------------------------------метод Эйлера----------------------------------------------------
def system_Eiler(matrix, y0, h, n):
    y_t = [y0]
    for i in range(1, n + 1):
        y_t.append(y_t[i - 1] + h * (matrix @ y_t[i - 1]))
    #print (y_t)
    return np.array(y_t)

def output(arr, t):
    fig, axs = plt.subplots(3,1,figsize = (21, 10))
    #print("00",arr[0:, 0])
    axs[0].plot(t, arr[0:, 0], label='1 компонента', color="r")
    axs[0].legend()
    #print("01",arr[0:, 1])
    axs[1].plot(t, arr[0:, 1], label='2 компонента', color="g")
    axs[1].legend()

    axs[2].plot(t, arr[0:, 2], label='3 компонента', color="b")
    axs[2].legend()

    plt.show()

t = np.linspace(a, b, n + 1)
AY = system_Eiler(A, Y0, h, n)
output(AY, t)


BZ = system_Eiler(B, Z0, h, n)
#output(BZ, t)

#---------------------------------------метод Эйлера-Коши----------------------------------------------------
def system_Eiler_Cauchy(matrix, y0, h, n):
    buf_y = [y0]
    y_t = [y0]

    for i in range(1, n + 1):
        buf_y.append(y_t[i - 1] + h * (matrix @ y_t[i - 1]))
        y_t.append(y_t[i - 1] + 0.5* h *(matrix @ buf_y[i-1]))
    return np.array(y_t)

AY = system_Eiler_Cauchy(A, Y0, h, n)
output(AY, t)

#BZ = system_Eiler_Cauchy(B, Z0, h, n)
#output(BZ, t)
#---------------------------------------неявный метод Эйлера----------------------------------------------------
def implicit_Eiler(matrix, y0, h, n):
    y_t = [y0]
    for i in range(1, n + 1):
        y_t.append(np.linalg.inv(np.eye(y0.shape[0]) - h * matrix) @ y_t[i - 1])
    return np.array(y_t)

AY = implicit_Eiler(A, Y0, h, n)
output(AY, t)

#BZ = implicit_Eiler(B, Z0, h, n)
#output(BZ, t)