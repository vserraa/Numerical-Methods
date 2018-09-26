import math
import numpy as np
from sympy import*
import matplotlib.pyplot as plt

class Solver:
	def __init__(self, f, t0, y0, h, nsteps):
		self.f = f
		self.t0 = t0
		self.y0 = y0
		self.h = h
		self.nsteps = nsteps

	def euler(self):
		ans = []
		ans.append([self.t0, self.y0])
		t, y = self.t0, self.y0
		for i in range(1, self.nsteps+1):
			y = y + self.h*self.f(t, y)
			t = t + self.h
			ans.append([t, y])

		return ans

	def inverse_euler(self):
		ans = []
		ans.append([self.t0, self.y0])
		t, y = self.t0, self.y0
		for i in range(1, self.nsteps+1):
			k = y + self.h*self.f(t, y)
			y = y + self.h*self.f(t + self.h, k)
			t = t + self.h
			ans.append([t, y])

		return ans		

	def improved_euler(self):
		ans = []
		ans.append([self.t0, self.y0])
		t, y = self.t0, self.y0
		for i in range(1, self.nsteps+1):
			k = y + self.h*self.f(t, y)
			y = y + 0.5*self.h*(self.f(t + self.h, k) + self.f(t, y))
			t = t + self.h
			ans.append([t, y])

		return ans		

	def runge_kutta(self):
		ans = []
		t, y = self.t0, self.y0
		ans.append([t, y])
		for i in range(1, self.nsteps+1):
			k1 = f(t, y)
			k2 = f(t + 0.5*h, y + 0.5*h*k1)
			k3 = f(t + 0.5*h, y + 0.5*h*k2)
			k4 = f(t + h, y + h*k3)
			y = y + h*(k1 + 2*k2 + 2*k3 + k4)/6
			t = t + h
			ans.append([t, y])

		return ans

	def adam_bashforth_by_euler(self, order):
		ans = self.euler()
	
		for i in range(order, self.nsteps+1):
			if order == 5:
				ans[i][1] = ans[i-1][1] + self.h*((1901/720)*self.f(ans[i-1][0], ans[i-1][1]) - (1387/360)*self.f(ans[i-2][0], ans[i-2][1])
					+ (109/30)*self.f(ans[i-3][0], ans[i-3][1]) - (637/360)*self.f(ans[i-4][0], ans[i-4][1])
					+ (251/720)*self.f(ans[i-5][0], ans[i-5][1]))
				ans[i][0] = ans[i-1][0] + h
			elif order == 6:
				ans[i][1] = ans[i-1][1] + self.h*(4277*self.f(ans[i-1][0], ans[i-1][1]) - 2641*3*self.f(ans[i-2][0], ans[i-2][1])
					+ 4991*2*self.f(ans[i-3][0], ans[i-3][1]) - 3649*2*self.f(ans[i-4][0], ans[i-4][1])
					+ 959*3*self.f(ans[i-5][0], ans[i-5][1]) - 95*5*self.f(ans[i-6][0], ans[i-6][1]))/1440

				ans[i][0] = ans[i-1][0] + h
		
		return ans

#Main part of the code
#We wish to find an approximate solution to the equation dy/dt = f(t, y)

f = open("in.txt")
for line in f:
	entrada = line.split()
	method = entrada[0]
	t0, y0 = int(entrada[1]), int(entrada[2])
	h = float(entrada[3])
	nsteps = int(entrada[4])
	expr = sympify(entrada[5])
	t, y = symbols("t y")
	f = lambdify((t, y), expr, "numpy")

	solver = Solver(f, t0, y0, h, nsteps)
	pts = []
	if method == "euler":
		pts = solver.euler()
		print("Metodo de Euler")
	elif method == "euler_inverso":
		pts = solver.inverse_euler()
		print("Metodo de Euler Inverso")
	elif method == "euler_aprimorado":
		pts = solver.improved_euler()
		print("Metodo de Euler Aprimorado")
	elif method == "runge_kutta":
		pts = solver.runge_kutta()
		print("Metodo de Runge-Kutta")
	elif method == "adam_bashforth_by_euler":
		order = int(entrada[6])
		print("Metodo de Adam-Bashforth por Euler")
		pts = solver.adam_bashforth_by_euler(order);

	for [x, y] in pts:
		format(y, '.12g')
		print("%lf %.10lf" %(x, y))

	#ploting the solution
	#plt.plot(pts[:, 0], pts[:, 1], ls = '-', color = 'black', linewidth = 1)
	#plt.show()

	print("\n")