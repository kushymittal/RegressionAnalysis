import pandas
import numpy
import math

# For linear regression
from sklearn import linear_model

# For transformations
from scipy import stats

# For polynomial regression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# For plotting
import plotly.plotly as py
import plotly.graph_objs as go
import matplotlib.pyplot as plt


def regress_curve_2():
	filename = "sulfate_dataset.txt"
	my_data = pandas.read_csv(filename, sep="\t", header=0)

	x = my_data["Hours"].values
	y = my_data["Sulfate"].values

	# Box Cox Transformtaion on Y
	y_transformed, lamb = stats.boxcox(y)		
	y = y.reshape(-1, 1)
	x = x.reshape(-1, 1)

	model = linear_model.LinearRegression()
	model.fit(x, y_transformed)

	print lamb				# lamb = -1.10251654861
	print model.coef_		# [[-0.00061966]]
	
	# Predict y
	y_predicted = model.predict(x)
	print y_predicted
	# Transform the predicted y
	y_predicted_transformed = numpy.array([math.pow(y_predicted[i], lamb) for i in range(len(y))]).reshape(-1, 1)

	plt.scatter(x, y, color='green', label="Original Data Points")	# Original data points transformed
	plt.plot(x, y_predicted_transformed, 'o', color='red', label="Regression Curve")								# plot predicted y vs x curve
	plt.title("Blood Sulfate in a Baboon Named Brunhilda Regression Curve")
	plt.ylabel("Sulfate")
	plt.xlabel("Hours")
	plt.legend(loc='upper right')
	plt.show()
	

	residual = y - y_predicted_transformed
	plt.scatter(y, residual, color='green')
	plt.title("Residual Against Fitted Values")
	plt.ylabel("Residual Error")
	plt.xlabel("Fitted Values")
	plt.show()

# Delete this
def regress_curve():
	filename = "sulfate_dataset.txt"
	my_data = pandas.read_csv(filename, sep="\t", header=0)

	x = my_data["Hours"].values
	y = my_data["Sulfate"].values
	z = numpy.polyfit(x, y, 2)
	residual = numpy.array([0 for i in range(21)])
	for i in range(21):
		residual[i] = y[i] - (z[0]*x[i]*x[i] + z[1]*x[i] + z[2])
	residual = residual.reshape(-1, 1)

	x_values = my_data["Hours"].values.reshape(-1, 1)
	y_values = my_data["Sulfate"].values.reshape(-1,1) 

	colors = ['yellowgreen', 'gold', 'blue']
	plt.scatter(x_values, y_values, color='black', s=30, marker='o', label='Data Points')

	# Plot regressed curve
	degree = 2
	model = make_pipeline(PolynomialFeatures(degree), Ridge())
	model.fit(x_values, y_values)
	plt.plot(x_values, y_values, color=colors[degree-2], linewidth=2, label="Degree: %d" % degree)
	plt.legend(loc='lower left')
	plt.show()

	# Plot residuals against the fitted values
	plt.scatter(y_values, residual, color='black')
	plt.title("Residual Against Fitted Values")
	plt.ylabel("Residual Error")
	plt.xlabel("Fitted Values")
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

	# Plot the regression line
	plt.scatter(x_values, y_values, color='black')
	plt.plot(x_values, model.predict(x_values), color='blue', linewidth=2)
	plt.title("Blood Sulfate in a Baboon Named Brunhilda")
	plt.ylabel("Sulfate (log)")
	plt.xlabel("Hours (log)")
	plt.show()

	# Plot the residual against the fitted values
	y_predicted = model.predict(x_values)

	y_predicted_transformed = numpy.array([math.pow(y_predicted[i][0], math.e) for i in range(len(y_predicted))]).reshape(-1, 1)

	residual = y_values - y_predicted_transformed		# change to e^
	plt.scatter(y_values, residual, color='black')
	plt.title("Residual Against Fitted Values")
	plt.ylabel("Residual Error")
	plt.xlabel("Fitted Values")
	plt.show()

def main():
	#regress_log_log()

	#regress_curve()

	regress_curve_2()

if __name__ == '__main__':
	main()