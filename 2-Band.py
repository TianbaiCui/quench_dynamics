#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 15:45:31 2016

@author: Tianbai
"""

from numpy import *
from matplotlib.pyplot import *
from scipy.integrate import simps
from scipy.integrate import quad
from scipy.optimize import fsolve
import multiprocessing


vi = 0.19
vf = 0.2
r_i = 0.8
r_f = 0.8
r = r_f
r_coup = 0
ui = -r_coup * vi
uf = -r_coup * vf


def Gap_Eq(x, u1, u2, v, r):
    return [x[0] + u1 * x[0] * arcsinh(1/abs(x[0])) + r * v * x[1] * arcsinh(1/abs(x[1])), x[1] + r * u2 * x[1] * arcsinh(1/abs(x[1])) + v * x[0] * arcsinh(1/abs(x[0]))]

#def Gap_Eq(x, u, v):
#    return [x[0] + u * x[0] * arcsinh(1/abs(x[0])) + v * x[1] * arcsinh(1/abs(x[1])), x[1] + r * u * x[1] * arcsinh(1/abs(x[1])) + v * x[0] * arcsinh(1/abs(x[0]))]

D1_i, D2_i = fsolve(Gap_Eq,[0.08, -0.09], args=(ui, uf, vi, r), xtol=1e-20)
D1_f, D2_f = fsolve(Gap_Eq,[0.1, -0.1], args=(uf, uf, vf, r), xtol=1e-20)

del1_i = D1_i/D1_f
del2_i = D2_i/D1_f

def rK3(a, b, c, fa, fb, fc, hs):
    a1 = fa(a, b, c)*hs
    b1 = fb(a, b, c)*hs
    c1 = fc(a, b, c)*hs
    ak = a + a1*0.5
    bk = b + b1*0.5
    ck = c + c1*0.5
    a2 = fa(ak, bk, ck)*hs
    b2 = fb(ak, bk, ck)*hs
    c2 = fc(ak, bk, ck)*hs
    ak = a + a2*0.5
    bk = b + b2*0.5
    ck = c + c2*0.5
    a3 = fa(ak, bk, ck)*hs
    b3 = fb(ak, bk, ck)*hs
    c3 = fc(ak, bk, ck)*hs
    ak = a + a3
    bk = b + b3
    ck = c + c3
    a4 = fa(ak, bk, ck)*hs
    b4 = fb(ak, bk, ck)*hs
    c4 = fc(ak, bk, ck)*hs
    a = a + (a1 + 2*(a2 + a3) + a4)/6
    b = b + (b1 + 2*(b2 + b3) + b4)/6
    c = c + (c1 + 2*(c2 + c3) + c4)/6
    return a, b, c

def theta1_i(x):
    return arcsin(del1_i / (x ** 2 + del1_i ** 2)**0.5)

def phi1_i(x):
    return 0

def theta2_i(x):
    return arcsin(del2_i / (x ** 2 + del2_i ** 2)**0.5)

def phi2_i(x):
    return 0


D1 = []
D2 = []
th_phi_en1 = []
th_phi_en2 = []

D1.append(quad(lambda x: (-uf * sin(theta1_i(x)) - r * vf * sin(theta2_i(x))), 0, 1/D1_f)[0])
D2.append(quad(lambda x: (-r * uf * sin(theta2_i(x)) - vf * sin(theta1_i(x))), 0, 1/D1_f)[0])


def endot(en, th, phi):
    return 0

def thdot(en, th, phi):
    return (2 * en * sin(phi))

def phidot1(en, th, phi):
    return 2 * (D1[0] - en * tan(th) * cos(phi))

def phidot2(en, th, phi):
    return 2 * (D2[0] - en * tan(th) * cos(phi))

hs = 0.001

t = [0]

engrid = linspace(0, 1/D1_f, 10000)

for j in range(len(engrid)):
    en = engrid[j]
    th_temp1 = theta1_i(en)
    phi_temp1= phi1_i(en)
    th_temp2 = theta2_i(en)
    phi_temp2 = phi2_i(en)
    th_phi_en1.append(rK3(en, th_temp1, phi_temp1, endot, thdot, phidot1, hs))
    th_phi_en2.append(rK3(en, th_temp2, phi_temp2, endot, thdot, phidot2, hs))

x1 = transpose(list(zip(*th_phi_en1)[0])).astype('float64')
y1 = transpose(list(zip(*th_phi_en1)[1])).astype('float64')
z1 = transpose(list(zip(*th_phi_en1)[2])).astype('float64')

#x2 = transpose(list(zip(*th_phi_en2)[0])).astype('float64')
y2 = transpose(list(zip(*th_phi_en2)[1])).astype('float64')
z2 = transpose(list(zip(*th_phi_en2)[2])).astype('float64')

#nkt = []

#nkt.append([(engrid[j], cos(y[j])*cos(z[j])) for j in range(len(engrid))])

#th = interp1d(x, y, kind = 'cubic')
#phi = interp1d(x, z, kind = 'cubic')

#def phidot2(en, th2, phi2):
#        return 2 * (D_temp - en * tan(th2) * cos(phi2))

def phidot3(en, th2, phi2, delta):
        return 2 * (delta - en * tan(th2) * cos(phi2))

def rK3_2(a, b, c, fa, fb, fc, hs, d):
    a1 = fa(a, b, c)*hs
    b1 = fb(a, b, c)*hs
    c1 = fc(a, b, c, d)*hs
    ak = a + a1*0.5
    bk = b + b1*0.5
    ck = c + c1*0.5
    a2 = fa(ak, bk, ck)*hs
    b2 = fb(ak, bk, ck)*hs
    c2 = fc(ak, bk, ck, d)*hs
    ak = a + a2*0.5
    bk = b + b2*0.5
    ck = c + c2*0.5
    a3 = fa(ak, bk, ck)*hs
    b3 = fb(ak, bk, ck)*hs
    c3 = fc(ak, bk, ck, d)*hs
    ak = a + a3
    bk = b + b3
    ck = c + c3
    a4 = fa(ak, bk, ck)*hs
    b4 = fb(ak, bk, ck)*hs
    c4 = fc(ak, bk, ck, d)*hs
    a = a + (a1 + 2*(a2 + a3) + a4)/6
    b = b + (b1 + 2*(b2 + b3) + b4)/6
    c = c + (c1 + 2*(c2 + c3) + c4)/6
    return b, c

def parallelK1((delta, j)):
    en = engrid[j]
    th_temp = y1[j]
    phi_temp = z1[j]
    th_phi_en = (rK3_2(en, th_temp, phi_temp, endot, thdot, phidot3, hs, delta))
    return th_phi_en

def parallelK2((delta, j)):
    en = engrid[j]
    th_temp = y2[j]
    phi_temp = z2[j]
    th_phi_en = (rK3_2(en, th_temp, phi_temp, endot, thdot, phidot3, hs, delta))
    return th_phi_en

def swaverk1(i):
    delgrid = [D1_temp] * len(engrid)
    if __name__ == '__main__':
        pool = multiprocessing.Pool()
        temp_result = pool.map(parallelK1, zip(delgrid, range(len(engrid))))
        result = zip(*temp_result)
    pool.close()
    pool.join()
    return result

def swaverk2(i):
    delgrid2 = [D2_temp] * len(engrid)
    if __name__ == '__main__':
        pool = multiprocessing.Pool()
        temp_result = pool.map(parallelK2, zip(delgrid2, range(len(engrid))))
        result = zip(*temp_result)
    pool.close()
    pool.join()
    return result
#fname_template1 = "n1_t={time2}.dat"
#fname_template2 = "n2_t={time2}.dat"
# i controls the time step
th_phi_en = []
for i in range(1, 100001):
    t.append(i * hs)
    sinth1 = map(sin, y1)
    sinth2 = map(sin, y2)
    I1 = simps(sinth1, x1)
    I2 = simps(sinth2, x1)
#    integrand1 = map(lambda x: -vf * x, sinth1)
#    integrand2 = map(lambda x: -vf * x, sinth2)
    D1.append(-uf * I1 - r * vf * I2)
    D2.append(-r * uf * I2 - vf * I1)

    D1_temp = D1[i]
    D2_temp = D2[i]

    th_phi_en1 = swaverk1(i)
    th_phi_en2 = swaverk2(i)

    y1 = th_phi_en1[0]
    z1 = th_phi_en1[1]
    y2 = th_phi_en2[0]
    z2 = th_phi_en2[1]
#    if i in (1000, 10000, 20000, 30000, 40000, 50000, 52000, 54000, 56000, 58000, 60000, 80000, 100000,140000):
#        nk1 = [(engrid[j], cos(y1[j])*cos(z1[j])*((engrid[j]**2 + del1_i**2)**0.5)/engrid[j]) for j in range(1,len(engrid))]
#	nk2 = [(engrid[j], cos(y2[j])*cos(z2[j])*((engrid[j]**2 + del2_i**2)**0.5)/engrid[j]) for j in range(1,len(engrid))]
#        savetxt(fname_template1.format(time2=int(i*hs)), nk1)
#	savetxt(fname_template2.format(time2=int(i*hs)), nk2)

savetxt("D1_vi=0.19_vf=0.2_r=0.8.dat", zip(t,D1))
savetxt("D2_vi=0.19_vf=0.2_r=0.8.dat", zip(t,D2))
