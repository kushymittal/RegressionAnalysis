import pandas
import numpy

import math

# For linear regression
from sklearn import linear_model

# FOr plotting
import matplotlib.pyplot as plt

def log_regress():
	filename = "abalone_dataset.txt"
	my_data = pandas.read_csv(filename, sep=",", usecols = [1, 2, 3, 4, 5, 6, 7, 8], skip_blank_lines=True)
	my_data = my_data.replace(0, 0.00000001)
	
	x_values = my_data[[0, 1, 2, 3, 4, 5, 6]].values
	y_values_1 = my_data[[7]].values

	# Dataset description says add 1.5 to get age in year
	y_values_1 = y_values_1 + 1.5

	# Take the log of the entire dataset
	#x_values = numpy.log(x_values)
	y_values = numpy.log(y_values_1)

	model = linear_model.LinearRegression()
	model.fit(x_values, y_values)

	print model.coef_			# [[ 0.25603512  1.39372311  1.11692882  0.58249223 -1.49644684 -0.65516573		0.54529821]]

	y_predicted = model.predict(x_values)
	y_predicted_transformed = numpy.array([math.pow(y_predicted[i][0], math.e) for i in range(len(y_values))]).reshape(-1, 1)

	residual = y_values_1 - y_predicted_transformed
	plt.scatter(y_values_1, residual, color='green')
	plt.title("Residual Against Log Fitted Values Without Gender")
	plt.ylabel("Residual Error")
	plt.xlabel("Age (In Years)")
	plt.show()


def log_regress_with_gender():						# FIX ME
	filename = "abalone_dataset.txt"
	my_data = pandas.read_csv(filename, sep=",", usecols = [0, 1, 2, 3, 4, 5, 6, 7, 8], skip_blank_lines=True)
	
	# Convert Gender to numerical values
	my_data = my_data.replace("F", 1)
	my_data = my_data.replace("M", -1)
	my_data = my_data.replace("I", 0)

	x_values = my_data[[0, 1, 2, 3, 4, 5, 6, 7]].values
	y_values_1 = my_data[[8]].values

	# Dataset description says add 1.5 to get age in year
	y_values_1 = y_values_1 + 1.5

	# Take the log of the entire dataset
	#x_values = numpy.log(x_values)
	y_values = numpy.log(y_values_1)

	model = linear_model.LinearRegression()
	model.fit(x_values, y_values)

	print model.coef_

	y_predicted = model.predict(x_values)
	y_predicted_transformed = numpy.array([math.pow(y_predicted[i][0], math.e) for i in range(len(y_values))]).reshape(-1, 1)

	residual = y_values_1 - y_predicted_transformed
	plt.scatter(y_values_1, residual, color='green')
	plt.title("Residual Against Log Fitted Values With Gender")
	plt.ylabel("Residual Error")
	plt.xlabel("Age (In Years)")
	plt.show()

def basic_regression():
	filename = "abalone_dataset.txt"
	my_data = pandas.read_csv(filename, sep=",", usecols = [1, 2, 3, 4, 5, 6, 7, 8])

	x_values = my_data[[0, 1, 2, 3, 4, 5, 6]].values
	y_values = my_data[[7]].values

	# Dataset description says add 1.5 to get age in year
	y_values = y_values + 1.5

	model = linear_model.LinearRegression()
	model.fit(x_values, y_values)

	print model.coef_	# [[ -1.51186355  13.25861227  11.90036056   9.23991828 -20.20887007	-9.8080347    8.58703255]]

	residual = y_values - model.predict(x_values)
	plt.scatter(y_values, residual, color='green')
	plt.title("Residual Against Fitted Values Without Gender")
	plt.ylabel("Residual Error")
	plt.xlabel("Age (In Years)")
	plt.show()

def regression_with_gender():
	filename = "abalone_dataset.txt"
	my_data = pandas.read_csv(filename, sep=",")

	# Convert Gender to numerical values
	my_data.replace("F", 1, inplace="True")
	my_data.replace("M", -1, inplace="True")
	my_data.replace("I", 0, inplace="True")

	x_values = my_data[[0, 1, 2, 3, 4, 5, 6, 7]].values
	y_values = my_data[[8]].values

	# Dataset description says add 1.5 to get age in year
	y_values = y_values + 1.5

	model = linear_model.LinearRegression()
	model.fit(x_values, y_values)

	print model.coef_		# [[ -0.06164878  -1.51755916  13.31723051  11.93677868   9.24296413	-20.273804    -9.74184398   8.5911218 ]]

	residual = y_values - model.predict(x_values)
	plt.scatter(y_values, residual, color='green')
	plt.title("Residual Against Fitted Values With Gender")
	plt.ylabel("Residual Error")
	plt.xlabel("Age (In Years)")
	plt.show()

def main():
	# Part a
	#basic_regression()

	# part b
	#regression_with_gender()

	# part c
	#log_regress()

	# part d
	log_regress_with_gender()

if __name__ == '__main__':
	main()