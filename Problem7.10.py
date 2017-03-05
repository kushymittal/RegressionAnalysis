import pandas
import numpy

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

	print (x_values)
	print (y_values)

	model = linear_model.LinearRegression()
	model.fit(x_values, y_values)

	print model.coef_			# [[ 1.78181867  0.1550904   0.18913544 -0.48183705 -0.02931235  0.66144124		0.31784645  0.44589018  0.29721231 -0.91956267]]

	# Plot the regression line
	residual = y_values - model.predict(x_values)
	plt.scatter(y_values, residual, color='black')
	plt.ylabel("Residual")
	plt.xlabel("Fitted Values")
	plt.show()


def cube_root_reg():
	filename = "sulfate_dataset.txt"
	my_data = pandas.read_csv(filename, sep="\t", header=0)


def main():
	linear_reg()

if __name__ == '__main__':
	main()