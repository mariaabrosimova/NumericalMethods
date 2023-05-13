import numpy as np
import matplotlib.pyplot as plt 
import math
from scipy import integrate
#6.1
def dF(f, x, h):
    return (8/21*f(x) + 12/7*f(x - 7*h) - 3/56*f(x - 14*h) - 49/24*f(x - 6 *h))/h
def RightDifferenceDerivative(f,x, h):
    return (f(x + h) - f(x))/h
def F5_1(x):
    return 0.6+1.3*x+1.2*x**3+1.9*x**4

def dF5_1(x):
    return 1.3+3.6*x**2+7.6*x**3
def F_2(x):
    return x**2
def dF_2(x):
    return x*2

a = 1
b = 3
h0 = 0.0001
x = np.linspace(a, b, 1000)
#график точной производной
fig, axs = plt.subplots(1, 5, figsize = (50, 5)) #3
#plt.interactive(True)
axs[0].plot(x, dF5_1(x), label = "точная производная", color = "red")
axs[0].grid()
axs[0].legend()

axs[1].plot(x, dF(F5_1, x, h0), label = "ф-ла численного интегрирования", color = "blue")
axs[1].grid()
axs[1].legend()

axs[2].plot(x, RightDifferenceDerivative(F5_1,x, h0), label = "ф-ла правой разрастной производной", color = "pink")
axs[2].grid()
axs[2].legend()
#график погрешности индивидуальной формулы численного дифференцирования
axs[3].plot(x, abs(dF5_1(x) - dF(F5_1, x, h0)), label = "погрешность формулы численного дифференцирования", color = "green",linewidth=1.0)
axs[3].grid()
axs[3].legend()
#график погрешности формулы правой разностной производной
axs[4].plot(x,abs(dF5_1(x) - RightDifferenceDerivative( F5_1,x, h0)), label = "погрешность формулы правой разностной производной", color = "cyan")
axs[4].grid()
axs[4].legend()
plt.show()
#axs.legend.show()
#п.4
def f4(x):
   return (np.exp(x)*np.sin(math.pi*x))

def df4(x):
    return np.exp(x)*(np.sin(math.pi*x) + math.pi*np.cos(math.pi*x))

def optimization(func, f, df, x):
    h0 = 0.1
    y = np.zeros(15)
    R = np.zeros(15)
    for k in range(15):
        h = h0* 10**(-k)
        y[k] = func(f, x, h)
        R[k] = abs(y[k] - df(x))
    return y, R
x0=2

y1, R1 = optimization(dF, f4, df4, x0)
print("численное дифференцирование")
print(y1)
print("погрешность численного дифференцирования")
print(R1)

y2,R2 = optimization(RightDifferenceDerivative, f4, df4, x0)
print("формула правой разностной производной")
print(y2)
print("погрешность формулы правой разностной производной")
print(R2)

print("  i  |         шаг h           |      R численного дифференцирования             |        R правой разностной производной  ")
print("-------------------------------------------------------------------------------------------------------------------------")
for i in range(0, 15):
    print("%i    |   %.15lf     |  %.18lf                           | %.18lf    " % (i, 0.1**i, R1[i], R2[i]))

fig, axs = plt.subplots(1, 2, figsize = (50, 5)) #3
axs[0].plot(np.linspace(1, 14, 13), R1[1:14], label = "погрешность численного дифференцирования", color = "red")
axs[0].grid()
axs[0].legend()

axs[1].plot(np.linspace(1, 14, 13), R2[1:14], label = "погрешность формулы правой разностной производной", color = "violet")
axs[1].grid()
axs[1].legend()
plt.show()
#------------------------------------------------------------------------------------------
#6.2
def f(t, y):
    return -np.sin(0.5 * t)/np.cos(0.5 * t)*y

def y(t):
    return np.cos(0.5 * t)**2 #аналитическое решение задачи Коши

eps = 10**(-6)
t0 = 0
T = np.pi/2
y0 = 1

def Euler(t0, T, y0, eps, f, p):
    flag = True
    n = 1
    h = (T-t0)/n
    y = np.zeros(n+1)
    y[0] = y0
    y[1] = y[0] + h*f(t0, y[0])
    while flag:
        flag = False
        y1 = y.copy()
        n = 2*n
        h = (T-t0)/n
        y = np.zeros(n+1)
        y[0] = y0
        for i in range(1, int(n/2) + 1):
            y[2*i-1] = y[2*i-2] + h*f(t0 + (2*i - 2)*h, y[2*i-2])
            y[2*i] = y[2*i-1] + h*f(t0 + (2*i - 1)*h, y[2*i-1])
            if abs(y[2*i] - y1[i])/(2**p - 1) >= eps:
                flag = True
    return y, n, h
y_arr, n, h = Euler(t0, T, y0, eps, f, 1)
print("метод Эйлера с использованием правила Рунге")
print("n = ", n)
print("h = ", h)

x = np.linspace(t0, T, n + 1)
fig, axs = plt.subplots(1, 2, figsize = (20, 5)) #3
axs[0].plot(x,  y_arr, label = "Решение методом Эйлера", color = "red")
#axs[0].plot(x,  y(x), label = "Аналитическое решение задачи Коши", color = "blue")
axs[0].grid()
axs[0].legend()

axs[1].plot(x,abs(y(x) - y_arr), label = "погрешность решения методом Эйлера", color = "violet")
axs[1].grid()
axs[1].legend()
plt.show()
#Эйлер-Коши
def Euler_Cauchy(t0, T, y0, eps, f, p):
    flag = True
    n = 1
    h = (T-t0)/n
    y = np.zeros(n+1)
    y[0] = y0
    y_ = y[0] + h*(f(t0, y[0]))
    y[1] = y[0] + h/2*(f(t0, y[0]) + f(T, y_))
    while flag:
        flag = False
        y1 = y.copy()
        n = 2*n
        h = (T-t0)/n
        y = np.zeros(n+1)
        y[0] = y0
        for i in range(1, int(n/2) + 1):
            y_ = y[2*i-2] + h*f(t0 + (2*i - 2)*h, y[2*i-2])
            y[2*i-1] = y[2*i-2] + h/2*(f(t0 + (2*i - 2)*h, y[2*i-2]) +
                                       f(t0 + (2*i - 1)*h, y_))
            y_ = y[2*i-1] + h*f(t0 + (2*i - 1)*h, y[2*i-1])
            y[2*i] = y[2*i-1] + h/2*(f(t0 + (2*i - 1)*h, y[2*i-1]) +
                                       f(t0 + (2*i)*h, y_))
            if abs(y[2*i] - y1[i])/(2**p - 1) >= eps:
                flag = True
    return y, n, h

y_arr2, n2, h2= Euler_Cauchy(t0, T, y0, eps, f, 2)
print("метод Эйлера-Коши")
print("n = ", n2)
print ("h = ", h2)


x = np.linspace(t0, T, n2+ 1)
fig, axs = plt.subplots(1, 2, figsize = (20, 5)) #3
axs[0].plot(x,  y_arr2, label = "Решение методом Эйлера-Коши", color = "red")
axs[0].grid()
axs[0].legend()

axs[1].plot(x,abs(y(x) - y_arr2), label = "погрешность решения методом Эйлера-Коши", color = "blue")
axs[1].grid()
axs[1].legend()
plt.show()





