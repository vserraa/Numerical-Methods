import math
import numpy as np
from sympy import*
import matplotlib.pyplot as plt

def euler(f, x0, y0, h, xmax):
	ans = []
	ans.append([x0, y0])
	x, y = x0, y0
	while x < xmax:
		y = y + h*f(x, y)
		x = x + h
		ans.append([x, y])

	ans = np.array(ans)
	return ans

#Main part of the code
#We wish to find an approximate solution to the equation dy/dx = f(x, y)
expr = sympify(input("f(x, y) = "))
x, y = symbols('x y')
f = lambdify((x, y), expr, "numpy")
x0 = int(input("x0 = "))
y0 = int(input("y0 = "))
pts = euler(f, x0, y0, 0.1, 500)
#ploting the solution
plt.plot(pts[:, 0], pts[:, 1], ls = '-', color = 'black', linewidth = 1)
plt.show()