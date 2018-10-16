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
		self.coef_ab = [
			[1],
			[1],
			[3.0/2.0, 1.0/2.0],
			[23.0/12.0, -4.0/3.0, 5.0/12.0],
			[55.0/24.0, -59.0/24.0, 37.0/24.0, -3.0/8.0],
			[1901.0/720.0, -1387.0/360.0, 109.0/30.0, -637.0/360.0, 251.0/720.0],
			[4277.0/1440.0, -2641.0/480.0, 4991.0/720.0, -3649.0/720.0, 959.0/480.0, -95.0/288.0],
			[198721.0/60480.0, 18367.0/2520.0, 235183.0/20160.0, 10754.0/945.0, 135713.0/20160.0, 5603.0/2520.0, 19087.0/60480.0],
			[16083.0/4480.0, 1152169.0/120960.0, 242653.0/13440.0, 296053.0/13440.0, 2102243.0/120960.0, 115747.0/13440.0, 32863.0/13440.0, 5257.0/17280.0]
		]
		self.coef_am = [
			[1],
			[1.0/2.0, 1.0/2.0],
			[5.0/12.0, 2.0/3.0, -1.0/12.0],
			[3.0/8.0, 19.0/24.0, -5.0/24.0, 1.0/24.0],
			[251.0/720.0, 323.0/360.0, -11.0/30.0, 53.0/360.0, 19.0/720.0],
			[95.0/288.0, 1427.0/1440.0, -133.0/240.0, 241.0/720.0, -173.0/1440.0, 3.0/760.0],
			[19087.0/60480.0, 2713.0/2520.0, -15487.0/20160.0, 586.0/945.0, -6737.0/20160.0, 263.0/2520.0, -863.0/60480.0],
			[5257.0/17280.0, 139849.0/120960.0, -4511.0/4480.0, 123133.0/120960.0, -88547.0/120960.0, 1537.0/4480.0, -11351.0/120960.0, 275.0/24192.0]
		]
		self.coef_inv = [
			[1],
			[1],
			[2.0/3.0, 4.0/3.0, -1.0/3.0],
			[6.0/11.0, 18.0/11.0, -9.0/11.0, 2.0/11.0],
			[12.0/25.0, 48.0/25.0, -36.0/25.0, 16.0/25.0, -3.0/25.0],
			[60.0/137.0, 300.0/137.0, -300.0/137.0, 200.0/137.0, -75.0/137.0, 12.0/137.0],
			[60.0/147.0, 360.0/147.0, -450.0/147.0, 400.0/147.0, -255.0/147.0, 72.0/147.0, -10.0/147.0]		
		]

	def get_ab(self, ans, idx, order):
		value = ans[idx-1][1]
		for i in range(1, order+1):
			value += (self.h)*(self.coef_ab[order][i-1])*self.f(ans[idx-i][0], ans[idx-i][1])
		return value	

	def get_am(self, ans, idx, order):
		value = ans[idx-1][1]
		ans[idx][1] = self.get_ab(ans, idx, order-1)
		ans[idx][0] = ans[idx-1][0] + self.h
		for i in range(0, order):
			value += self.h*self.coef_am[order][i]*self.f(ans[idx-i][0], ans[idx-i][1])
		return value 
	
	def	get_inv(self, ans, idx, order):
		ans[idx][1] = self.get_ab(ans, idx, order)
		ans[idx][0] = ans[idx-1][0] + self.h
		value = self.coef_inv[order][0]*self.h*self.f(ans[idx][0], ans[idx][1])
		for i in range (1, order+1):
			value += self.coef_inv[order][i]*ans[idx-i][1]
			
		return value

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
		elif method == 'improved euler':
			ans = self.improved_euler()
		elif method == 'runge kutta':
			ans = self.runge_kutta()
		elif method == 'list':
			ans = self.inital_points

		h, f = self.h, self.f
		for i in range(order, self.nsteps+1):
			if len(ans) == i:
				ans.append([0, 0])
	
			ans[i][1] = self.get_ab(ans, i, order)
			ans[i][0] = ans[i-1][0] + h
		
		return ans

	def adam_multon_by_method(self, order, method):
		if method == 'euler':
			ans = self.euler()
		elif method == 'inverse euler':
			ans = self.inverse_euler()
		elif method == 'improved euler':
			ans = self.improved_euler()
		elif method == 'runge kutta':
			ans = self.runge_kutta()
		elif method == 'list':
			ans = self.inital_points

		h, f = self.h, self.f
		for i in range(order-1, self.nsteps+1):
			if len(ans) == i:
				ans.append([0, 0])
	
			ans[i][1] = self.get_am(ans, i, order)
			ans[i][0] = ans[i-1][0] + h
		
		return ans
	
	def backward_diff(self, order, method):
		if method == 'euler':
			ans = self.euler()
		elif method == 'inverse euler':
			ans = self.inverse_euler()
		elif method == 'improved euler':
			ans = self.improved_euler()
		elif method == 'runge kutta':
			ans = self.runge_kutta()
		elif method == 'list':
			ans = self.inital_points

		h, f = self.h, self.f
		for i in range(order, self.nsteps+1):
			if len(ans) == i:
				ans.append([0, 0])
		
			ans[i][1] = self.get_inv(ans, i, order)
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
	elif method == 'adam_multon':
		print("Metodo de Adam-Multon")
		pts = solver.adam_multon_by_method(order, 'list')
	elif method == 'adam_multon_by_euler':
		order = int(entrada[6])
		print("Metodo de Adam-Multon por Euler")
		pts = solver.adam_multon_by_method(order, 'euler')
	elif method == 'adam_multon_by_euler_inverso':
		order = int(entrada[6])
		print("Metodo de Adam-Multon por Euler Inverso")
		pts = solver.adam_multon_by_method(order, 'inverse euler')
	elif method == 'adam_multon_by_euler_aprimorado':
		order = int(entrada[6])
		print("Metodo de Adam-Multon por Euler Aprimorado")
		pts = solver.adam_multon_by_method(order, 'improved euler')
	elif method == 'adam_multon_by_runge_kutta':
		order = int(entrada[6])
		print("Metodo de Adam-Multon por Runge Kutta")
		pts = solver.adam_multon_by_method(order, 'runge kutta')
	elif method == 'formula_inversa':
		print("Metodo Formula Inversa de Diferenciacao")
		pts = solver.backward_diff(order, 'list')
	elif method == 'formula_inversa_by_euler':
		order = int(entrada[6])
		print("order is %d" %order)
		print("Metodo Formula Inversa de Diferenciacao por Euler")
		pts = solver.backward_diff(order-1, 'euler')
	elif method == 'formula_inversa_by_euler_inverso':
		order = int(entrada[6])
		print("Metodo Formula Inversa de Diferenciacao por Euler Inverso")
		pts = solver.backward_diff(order-1, 'inverse euler')
	elif method == 'formula_inversa_by_euler_aprimorado':
		order = int(entrada[6])
		print("Metodo Formula Inversa de Diferenciacao por Euler Aprimorado")
		pts = solver.backward_diff(order-1, 'improved euler')
	elif method == 'formula_inversa_by_runge_kutta':
		order = int(entrada[6])
		print("Metodo Formula Inversa de Diferenciacao por Runge Kutta")
		pts = solver.backward_diff(order-1, 'runge kutta')

	
	for [x, y] in pts:
		format(y, '.12g')
		print("%lf %.10lf" %(x, y))

#########################	ploting the solution	##############################
#	plt.plot(pts[:, 0], pts[:, 1], ls = '-', color = 'black', linewidth = 1)
#	plt.show()
##################################################################################

	print("\n")
