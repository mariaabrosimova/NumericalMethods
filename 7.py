import numpy as np
import matplotlib.pyplot as plt 
#7.1
a = 0
b = 1
h = 0.01
n = round((b-a)/h)
u0 = 1
v0 = 1
t0 = 0
t = np.linspace(a, b, n + 1)
def func_du(u, v, t):
    return u + t
def func_dv(u,v,t):
    return u*v - 1

#----------------------------------------метод Эйлера----------------------------------------
def Euler(t0, u0, v0, h, n, func_du, func_dv):
    vec_u, vec_v = [u0], [v0]
    buft = t0
    for i in range(1, n + 1):
        vec_u.append(vec_u[i - 1] + h * func_du(vec_u[i - 1],  vec_v[i - 1], buft))
        vec_v.append(vec_v[i - 1] + h * func_dv(vec_u[i - 1], vec_v[i - 1], buft))
        buft += h
    return np.array(vec_u), np.array(vec_v)

def ErrorRunge(y_2h,y_1h, p):
    buflist = []
    N = round(n/2)
    for i in range(0, N):
        buflist.append(abs(y_2h[i]-y_1h[2*i])/(2**p-1))
    max_number = max(buflist)
    return max_number


u_1h, v_1h = Euler(t0, u0, v0, h, n, func_du, func_dv)
u_2h, v_2h = Euler(t0, u0, v0, 2*h, round(n/2), func_du, func_dv)
print("-----------------------------------------------------------")
p = 1
print("метод Эйлера")
print("n = ", n)
print ("h = ", h)
err_u = ErrorRunge(u_2h, u_1h, p)
print("Погрешность по правилу Рунге для u: ", err_u)
err_v = ErrorRunge(v_2h, v_1h, p)
print("Погрешность по правилу Рунге для v: ", err_v)

#----------------------------------------метод Эйлера-Коши----------------------------------------
def Euler_Cauchy(t0, u0, v0, h, n, func_du, func_dv):
    u_solution = np.zeros(n + 1)
    v_solution= np.zeros(n + 1)
    u_solution[0] = u0
    v_solution[0] = v0
    u_buf = np.copy(u_solution)
    v_buf = np.copy(v_solution)
    buft = t0
    for i in range(1, n + 1):
        u_buf[i] = u_solution[i - 1] + h * func_du(u_solution[i - 1], v_solution[i - 1], buft)
        v_buf[i] = v_solution[i - 1] + h * func_dv(u_solution[i - 1], v_solution[i - 1], buft)

        u_solution[i] =u_solution[i - 1] + 0.5* h *(func_du(u_solution[i - 1], v_solution[i - 1], buft)
                                                    + func_du(u_buf[i], v_buf[i], buft + h))
        v_solution[i] = v_solution[i - 1] + 0.5 * h * (func_dv(u_solution[i - 1], v_solution[i - 1], buft)
                                                    + func_dv(u_buf[i], v_buf[i], buft + h))
        buft += h
    return u_solution, v_solution

print("-----------------------------------------------------------")

u_1h_Euler_Cauchy, v_1h_Euler_Cauchy = Euler_Cauchy(t0, u0, v0, h, n, func_du, func_dv)
u_2h_Euler_Cauchy, v_2h_Euler_Cauchy = Euler_Cauchy(t0, u0, v0, 2*h, round(n/2) , func_du, func_dv)


print("метод Эйлера-Коши")
print("n = ", n)
print ("h = ", h)

p = 2
err_u = ErrorRunge(u_2h_Euler_Cauchy, u_1h_Euler_Cauchy, p)
print("Погрешность по правилу Рунге для u: ", err_u)
err_v = ErrorRunge(v_2h_Euler_Cauchy, v_1h_Euler_Cauchy, p)
print("Погрешность по правилу Рунге для v: ", err_v)

fig, axs = plt.subplots(1, 2, figsize = (10,5))

axs[0].plot(t, u_1h, label='Методом Эйлера u(t)', color = 'violet')
axs[0].plot(t, u_1h_Euler_Cauchy, label='Методом Эйлера-Коши U(t)', color = 'b',linestyle='dashed', linewidth= 1)
axs[0].grid()
axs[0].legend()

axs[1].plot(t, v_1h, label='Методом Эйлера v(t)', color = 'y')
axs[1].plot(t, v_1h_Euler_Cauchy, label='Методом Эйлера-Коши V(t)',color = 'red',linestyle='dashdot', linewidth= 1)
axs[1].grid()
axs[1].legend()

plt.show()



