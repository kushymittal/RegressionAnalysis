import pandas
import numpy
import math

# For linear regression
from sklearn import linear_model

# For polynomial regression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# For plotting
import plotly.plotly as py
import plotly.graph_objs as go
import matplotlib.pyplot as plt

def linear_reg():
	filename = "physical.txt"
	my_data = pandas.read_csv(filename, sep="\t", header=0)

	x_values = my_data[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]].values
	y_values = my_data[[0]].values

	model = linear_model.LinearRegression()
	model.fit(x_values, y_values)

	print model.coef_			# [[ 1.78181867  0.1550904   0.18913544 -0.48183705 -0.02931235  0.66144124		0.31784645  0.44589018  0.29721231 -0.91956267]]

	# Plot the residual
	residual = y_values - model.predict(x_values)
	plt.scatter(y_values, residual, color='black')
	plt.ylabel("Residual")
	plt.xlabel("Fitted Values")
	plt.show()


def cube_root_reg():
	filename = "physical.txt"
	my_data = pandas.read_csv(filename, sep="\t", header=0)

	x_values = my_data[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]].values
	y_values = my_data[[0]].values

	y_values_cr = numpy.array([math.pow(y_values[i][0], 0.3333) for i in range(len(y_values))]).reshape(-1, 1)

	# Regress the cube root of mass against the dimensions
	model = linear_model.LinearRegression()
	model.fit(x_values, y_values_cr)

	print model.coef_		# [[ 0.0279648   0.00414277  0.0010515  -0.00253113  0.0008099   0.01114955		0.00577241  0.01065397  0.00791703 -0.01244873]]

	# Plot residual in original co-orid
	y_predicted_cr = model.predict(x_values)
	y_predicted = numpy.array([math.pow(y_predicted_cr[i][0], 3) for i in range(len(y_values))]).reshape(-1, 1)

	residual = y_values - y_predicted

	plt.scatter(y_values, residual, color='green')
	plt.title("Original Co-ordinates")
	plt.ylabel("Residual")
	plt.xlabel("Fitted Values")
	plt.show()

	residual_cr = y_values_cr - y_predicted_cr
	plt.scatter(y_values_cr, residual_cr, color='green')
	plt.title("Cube Root Co-ordinates")
	plt.ylabel("Residual")
	plt.xlabel("Fitted Values")
	plt.show()



def main():
	#linear_reg()

	cube_root_reg()

if __name__ == '__main__':
	main()