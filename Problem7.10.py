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

	#x_values = my_data[:, 1:11].values
	y_values = my_data["Mass"].values.reshape(-1, 1)
	#print x_values
	print y_values

	#model = linear_model.LinearRegression()
	#model.fit(x_values, y_values)


def cube_root_reg():
	filename = "sulfate_dataset.txt"
	my_data = pandas.read_csv(filename, sep="\t", header=0)


def main():
	linear_reg()

if __name__ == '__main__':
	main()