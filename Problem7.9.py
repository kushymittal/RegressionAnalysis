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

def regress_curve():
	filename = "sulfate_dataset.txt"
	my_data = pandas.read_csv(filename, sep="\t", header=0)

	x_values = my_data["Hours"].values.reshape(-1, 1)
	y_values = my_data["Sulfate"].values.reshape(-1,1) 

	colors = ['teal', 'yellowgreen', 'gold', 'blue']
	plt.scatter(x_values, y_values, color='black', s=30, marker='o', label='Data Points')

	for degree in [2]:
		model = make_pipeline(PolynomialFeatures(degree), Ridge())
		model.fit(x_values, y_values)
		plt.plot(x_values, y_values, color=colors[degree-2], linewidth=2, label="Degree: %d" % degree)
		plt.legend(loc='lower left')
		plt.show()


def regress_log_log():
	filename = "sulfate_dataset.txt"
	my_data = pandas.read_csv(filename, sep="\t", header=0)
	my_data = numpy.log(my_data)

	x_values = my_data["Hours"].values.reshape(-1, 1)
	y_values = my_data["Sulfate"].values.reshape(-1,1) 

	model = linear_model.LinearRegression()
	model.fit(x_values, y_values)

	print model.coef_			# [[-0.24704615]]

	plt.scatter(x_values, y_values, color='black')
	plt.plot(x_values, model.predict(x_values), color='blue', linewidth=2)
	plt.title("Blood Sulfate in a Baboon Named Brunhilda")
	plt.ylabel("Sulfate (log)")
	plt.xlabel("Hours (log)")
	plt.show()

def main():
	regress_curve()

if __name__ == '__main__':
	main()