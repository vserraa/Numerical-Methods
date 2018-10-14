import math
import numpy as np
from sympy import*
import matplotlib.pyplot as plt

class Solver:
	def __init__(self, f, t0, y0, h, nsteps, inital_points):
		self.f = f
		self.t0 = t0
		self.y0 = y0
		self.h = h
		self.nsteps = nsteps
		self.inital_points = inital_points;

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

	def adam_bashforth_by_method(self, order, method):
		if method == 'euler':
			ans = self.euler()
		elif method == 'inverse euler':
			ans = self.inverse_euler()
		elif method == 'improved_euler':
			ans = self.improved_euler()
		elif method == 'runge kutta':
			ans = self.runge_kutta()
		elif method == 'list':
			ans = self.inital_points

		h, f = self.h, self.f
		for i in range(order, self.nsteps+1):
			if len(ans) == i:
				ans.append([0, 0])

			if order == 2:
				ans[i][1] = ans[i-1][1] + h*((3/2)*f(ans[i-1][0], ans[i-1][1]) - (1/2)*f(ans[i-2][0], ans[i-2][1]))
			elif order == 3:
				ans[i][1] = ans[i-1][1] + h*((23/12)*f(ans[i-1][0], ans[i-1][1]) - (4/3)*f(ans[i-2][0], ans[i-2][1])
					+ (5/12)*f(ans[i-3][0], ans[i-3][1]))
			elif order == 4:
				ans[i][1] = ans[i-1][1] + h*((55/24)*f(ans[i-1][0], ans[i-1][1]) - (59/24)*f(ans[i-2][0], ans[i-2][1])
					+ (37/24)*f(ans[i-3][0], ans[i-3][1]) - (3/8)*f(ans[i-4][0], ans[i-4][1]))		
			elif order == 5:
				ans[i][1] = ans[i-1][1] + h*((1901/720)*f(ans[i-1][0], ans[i-1][1]) - (1387/360)*f(ans[i-2][0], ans[i-2][1])
					+ (109/30)*f(ans[i-3][0], ans[i-3][1]) - (637/360)*f(ans[i-4][0], ans[i-4][1]) + (251/720)*f(ans[i-5][0], ans[i-5][1]))
			elif order == 6:
				ans[i][1] = ans[i-1][1] + h*(4277*f(ans[i-1][0], ans[i-1][1]) - 2641*3*f(ans[i-2][0], ans[i-2][1])
					+ 4991*2*f(ans[i-3][0], ans[i-3][1]) - 3649*2*f(ans[i-4][0], ans[i-4][1])
					+ 959*3*f(ans[i-5][0], ans[i-5][1]) - 95*5*f(ans[i-6][0], ans[i-6][1]))/1440
			elif order == 7: 
				ans[i][1] = ans[i-1][1] + h*((198721/60480)*f(ans[i-1][0], ans[i-1][1]) - (18367/2520)*f(ans[i-2][0], ans[i-2][1])
					+ (235183/20160)*f(ans[i-3][0], ans[i-3][1]) - (10754/945)*f(ans[i-4][0], ans[i-4][1]) + (135713/20160)*f(ans[i-5][0], ans[i-5][1])
					- (5603/2520)*f(ans[i-6][0], ans[i-6][1]) + (19087/60480)*f(ans[i-7][0], ans[i-7][1]))	
			elif order == 8:
				ans[i][1] = ans[i-1][1] + h*((16083/4480)*f(ans[i-1][0], ans[i-1][1]) - (1152169/120960)*f(ans[i-2][0], ans[i-2][1])
					+ (242653/13440)*f(ans[i-3][0], ans[i-3][1]) - (296053/13440)*f(ans[i-4][0], ans[i-4][1]) + (2102243/120960)*f(ans[i-5][0], ans[i-5][1])
					- (115747/13440)*f(ans[i-6][0], ans[i-6][1]) + (32863/13440)*f(ans[i-7][0], ans[i-7][1])
					- (5257/17280)*f(ans[i-8][0], ans[i-8][1]))

			ans[i][0] = ans[i-1][0] + h
		
		return ans

#Main part of the code
#We wish to find an approximate solution to the equation dy/dt = f(t, y)

f = open("in.txt")
for line in f:
	entrada = line.split()
	method = entrada[0]
	ini_pts = []
	if method == 'adam_bashforth' or method == 'adam_multon' or method == 'formula_inversa':
		order = int(entrada[-1])
		expr = sympify(entrada[-2])
		t, y = symbols("t y")
		f = lambdify((t, y), expr, "numpy")
		nsteps = int(entrada[-3])
		h = float(entrada[-4])
		t0, y0 = float(entrada[-5]), 0
		for i in range(1, 1 + order):
			ini_pts.append([t0 + (i-1)*h, float(entrada[i])])
	else:	
		t0, y0 = float(entrada[1]), float(entrada[2])
		h = float(entrada[3])
		nsteps = int(entrada[4])
		expr = sympify(entrada[5])
		t, y = symbols("t y")
		f = lambdify((t, y), expr, "numpy")
	
	solver = Solver(f, t0, y0, h, nsteps, ini_pts)
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
		pts = solver.adam_bashforth_by_method(order, 'euler')
	elif method == 'adam_bashforth_by_euler_inverso':
		order = int(entrada[6])
		print("Metodo de Adam-Bashforth por Euler Inverso")
		pts = solver.adam_bashforth_by_method(order, 'inverse euler')
	elif method == 'adam_bashforth_by_euler_aprimorado':
		order = int(entrada[6])
		print("Metodo de Adam-Bashforth por Euler Aprimorado")
		pts = solver.adam_bashforth_by_method(order, 'improved euler')
	elif method == 'adam_bashforth_by_runge_kutta':
		order = int(entrada[6])
		print("Metodo de Adam-Bashforth por Runge Kutta")
		pts = solver.adam_bashforth_by_method(order, 'runge kutta')
	elif method == 'adam_bashforth':
		print("Metodo de Adam-Bashforth")
		pts = solver.adam_bashforth_by_method(order, 'list')
	
	for [x, y] in pts:
		format(y, '.12g')
		print("%lf %.10lf" %(x, y))

#	ploting the solution
	plt.plot(pts[:, 0], pts[:, 1], ls = '-', color = 'black', linewidth = 1)
	plt.show()

	print("\n")
