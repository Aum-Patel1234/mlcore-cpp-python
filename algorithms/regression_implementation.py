from math import sqrt
import pandas as pd
import numpy as np


class Regression2Variables:
    def __init__(self):
        pass

    def get_mean(self, series: pd.Series) -> float:  # returns mean of float data type
        return series.mean()

    def get_variance_or_deviation(
        self,
        series: pd.Series,
        deviation=False,
        get_population_variance=True,
        populationAndSample=False,
    ) -> float:
        try:
            series = series.astype("float")
            if series.isnull().any():
                return "There are null values present in data..."
        except:
            return "The data type of the column is not convertible to float."

        sumOfSquareOfDeviations = 0.0
        mean = self.get_mean(series)
        # print(series.values)

        for num in series.values:
            sumOfSquareOfDeviations += (num - mean) ** 2

        if deviation:
            return sumOfSquareOfDeviations  #  square of deviations

        population_variance = sumOfSquareOfDeviations
        sample_variance = sumOfSquareOfDeviations

        population_variance /= series.size
        sample_variance /= series.size - 1

        if populationAndSample:
            return population_variance, sample_variance

        if get_population_variance:
            return population_variance

        return sample_variance

    def get_deviation_of_point_from_mean(self, x: pd.Series, y: pd.Series):
        # alternate way to erase error
        x = pd.to_numeric(x, errors="coerce")
        y = pd.to_numeric(y, errors="coerce")
        if x.isnull().any() or y.isnull().any():
            return "The data contains non-numeric values, which have been converted to NaN."

        # logic with correct data
        deviation = 0.0
        meanX = self.get_mean(x)
        meanY = self.get_mean(y)

        for num1, num2 in zip(x, y):
            deviation += (num1 - meanX) * (num2 - meanY)

        return deviation

    def get_coefficient_of_correlation(self, x: pd.Series, y: pd.Series):
        """
                    ∑ (x-X)(y-Y)
        r =  ---------------------------
              sqrt( (x-X)**2 . (y-Y)**2 )
        """

        sumOfSquareOfDeviations_x = self.get_variance_or_deviation(x, deviation=True)
        sumOfSquareOfDeviations_y = self.get_variance_or_deviation(y, deviation=True)
        deviation = self.get_deviation_of_point_from_mean(x, y)

        return (deviation) / sqrt(sumOfSquareOfDeviations_x * sumOfSquareOfDeviations_y)

    def find_line_of_regression(self, x: pd.Series, y: pd.Series):
        """
        ∑y = Na + b∑x
        ∑xy = a.∑x + b∑(x**2)           never thought of how to solve equations using code so doing anoter way for now

        let, X - mean of x , Y - mean of y

        y - Y = byx. (x - X)         -> eq y on x
        x - X = bxy. (y- Y)         -> eq x on y

        byx = r. SD(y)/SD(x)            where,      SD(value) = standard deveiation = sqrt of variance  x
        bxy = r. SD(x)/SD(y)
        """

        meanX = self.get_mean(x)
        meanY = self.get_mean(y)
        sd_x = sqrt(self.get_variance_or_deviation(x))
        sd_y = sqrt(self.get_variance_or_deviation(y))

        r = self.get_coefficient_of_correlation(
            x, y
        )  # we need this for calculation as it is a 2 variable regression

        byx = r * sd_y / sd_x
        bxy = r * sd_x / sd_y

        print(f"\nTherefore the line of y on x :   y - {meanY} = {byx} * (x - {meanX})")
        print(
            f"\n                  and x on y :   x - {meanX} = {bxy} * (y - {meanY})\n\n"
        )

        # predict value of linear regression

        dataX = int(input("Enter age and predict the salary : "))

        if sqrt(bxy * byx) > 1:  # r = sqrt(byx*bxy)  and  r cannot be greater than 1
            return (
                "cannot calculate the prediction as the r > 0, try flipping the columns"
            )

        dataY = (byx * (dataX - meanX)) + meanY

        print(f"The salary expected with experience {dataX} is = {dataY}")


def main():
    # main function
    file_path = r"basics\data\years_experience_salary.xlsx"  # Constant file path

    df = pd.read_excel(file_path)
    # print(df)
    my_regression_class = Regression2Variables()
    population_variance, sample_variance = (
        my_regression_class.get_variance_or_deviation(
            df.iloc[:, 0], populationAndSample=True
        )
    )

    print(
        f"My variance = {population_variance} and the numpy variance is - {np.var(df.iloc[:, 0].astype(float))} \n\n pandas variance = {df.iloc[: ,0].var()} and the sample variance = {sample_variance}\n\nHence numpy calculates by dividing by n (population_variance) whereas pandas calculate dividing by n-1 (sample_variance)!!\n\n"
    )

    print(
        f"r = {my_regression_class.get_coefficient_of_correlation(df.iloc[:, 0], df.iloc[:, 1])} and the in built r = {df.corr()}\n\n"
    )

    my_regression_class.find_line_of_regression(df.iloc[:, 0], df.iloc[:, 1])


if __name__ == "__main__":  # Prevents execution when imported
    main()

