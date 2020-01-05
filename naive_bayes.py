"""
Includes Naive Bayes class and demonstration.
"""
from csv import reader
from math import sqrt
from math import exp
from math import pi
from math import log
import numpy as np
import time

class NaiveBayes:
    """
    Naive Bayes class.

    Attributes:
        dataset (list): all training data loaded from CSV file
        labels (list): list of label names
        divided (dict): dataset divided into labels
        means_devs (dict): calculated means and standard deviations of training data
    """
    def load_csv(self, csv_path):
        """
        Initialize NaiveBayes object and parse CSV data.

        Args:
            csv_path (str): path to CSV file
        """
        self.dataset = list()
        self.labels = list()
        with open(csv_path) as data_file:
            csv_reader = reader(data_file, delimiter=",")
            first_line = True
            for row in csv_reader:
                if first_line:
                    first_line = False
                else:
                    count = 0
                    for col in row[:-1]:
                        row[count] = float(row[count])
                        count += 1
                    row[-1] = self.get_label_id(row[-1])
                    self.dataset.append(row)

    def get_label_id(self, label):
        """
        Return label id or add new label to list.

        Args:
            label (str/int): label value

        Returns:
            int: id of label
        """
        if label in self.labels:
            return self.labels.index(label)
        self.labels.append(label)
        return len(self.labels)-1

    def pdf(self, x, stdev, mean):
        """
        Calculate Gaussian PDF for input.

        Args:
            x (int/float): input to calculate PDF for
            stdev (float): standard deviation value
            mean (float): mean value

        Returns:
            float: probability score
        """
        return (1 / (sqrt(2 * pi) * stdev)) * exp(-((x-mean)**2 / (2 * stdev**2)))

    def fit(self, x, y):
        """
        Train model with currently loaded csv data.

        x (list): nested list of data
        y (list): list of integers representing labels
        """
        # create dict for label separation
        self.divided = dict()
        # create list for each label
        count = 0
        while count < len(y):
            self.divided[count] = list()
            count += 1
        # add each row to their individual list in dict
        for row in x:
            self.divided[row[-1]].append(row)
        # calculate mean and standard deviation of each attribute for each category
        self.means_devs = dict()
        for label, data in self.divided.items():
            # create lists and dicts for collecting data
            self.means_devs[label] = dict()
            self.means_devs[label]["means"] = list()
            self.means_devs[label]["devs"] = list()
            col_amt = len(data[0])-1
            temp_nums = dict()
            for n in range(0, col_amt):
                temp_nums[n] = list()
            for row in data:
                # add all data to temporary lists
                count = 0
                for col in row[:-1]:
                    temp_nums[count].append(col)
                    count += 1
            count = 0
            for attr in temp_nums:
                # calculate means and standard deviation values
                self.means_devs[label]["means"].append(np.mean(temp_nums[count]))
                self.means_devs[label]["devs"].append(np.std(temp_nums[count]))
                count += 1

    def predict_one(self, x):
        """
        Calculate and return a prediction class for an input example.

        Args:
            list: attributes of example

        Returns:
            int: integer representing class
        """
        c_sum = 0 # sum of the probabilities for all categories
        p_scores = list() # probabilities for each category
        for label in self.labels:
            log_sum = 0
            count = 0
            for attr in x:
                # calculate probability for each attribute
                # then use logarithm and add to row sum
                md = self.means_devs[self.get_label_id(label)]
                log_sum += log(self.pdf(attr, md["devs"][count], md["means"][count]))
                count += 1
            # transform log product to original form, att to c_sum
            # and store in p_scores
            res = exp(log_sum)
            p_scores.append(res)
            c_sum += res
        # normalize p scores
        normalized = list()
        for p in p_scores:
            normalized.append(p/c_sum)
        # return index of highest value in normalized list
        return np.argmax(normalized)

    def predict(self, x):
        """
        Classifies examples X and returns a list of predictions.

        Args:
            x (list): nested list of examples

        Returns:
            list: list of integers representing class predictions
        """
        # get amount of attributes
        attr_count = len(self.means_devs[0]["means"])
        res = list()
        # get class prediction for each input row
        for row in x:
            res.append(self.predict_one(row[:attr_count]))
        return res

    def get_labels(self):
        """
        Return a list of the labels for every row in the training dataset.

        Returns:
            list: list of integers representing labels
        """
        res = list()
        for row in self.dataset:
            res.append(row[-1])
        return res

    def accuracy_score(self, preds, y):
        """
        Calculates accuracy score for a list of predictions.

        Args:
            preds (list): predictions
            y (list): correct labels
        """
        # get amount of rows
        rows = len(preds)
        count = 0
        matches = 0
        # calculate accuracies
        while count < rows:
            if preds[count] == y[count]:
                matches += 1
            count += 1
        return "{}% ({}/{})".format(round((matches/rows)*100, 2), matches, rows)

    def confusion_matrix(self, preds, y):
        """
        Generates a confusion matrix and returns the result
        as an integer matrix.
        """
        # get amount of labels
        amount = y[-1]+1
        # get amount of rows
        rows = len(preds)
        # create empty matrix
        matrix = [[0 for x in range(amount)] for y in range(amount)]
        count = 0
        # add data to matrix
        while count < rows:
            matrix[y[count]][preds[count]] += 1
            count += 1
        return matrix

    def demonstration(self, csv_path):
        """
        Prints results of the different class methods
        based on a given CSV dataset.

        Args:
            csv_path (str): path to CSV file
        """
        print("")
        # load csv data
        self.load_csv(csv_path)
        d = self.dataset
        # train model
        start = time.time()
        self.fit(d, self.labels)
        end = time.time()
        training_time = round(end - start, 3)
        l = self.get_labels()
        start = time.time()
        p = self.predict(d)
        end = time.time()
        classificiation_time = round(end - start, 3)
        # generate accuracy score and confusion matrix
        accuracy = self.accuracy_score(p, l)
        matrix = self.confusion_matrix(p, l)
        # print statistics
        print("Dataset: " + csv_path)
        print("Accuracy: " + accuracy)
        print("Training time: " + str(training_time))
        print("Classification time: " + str(classificiation_time))
        print("Confusion matrix:\n")
        count = 0
        # print confusion matrix and related data
        s = ""
        while count < len(matrix):
            s += str(count).ljust(5)
            count += 1
        print(s)
        s = ""
        while count < len(matrix)*5:
            s += "-"
            count += 1
        print(s)
        count = 0
        for row in matrix:
            s = ""
            for col in row:
                s += str(col).ljust(5)
            h = str(count) + " -> " + str(self.labels[count])
            print(s + h)
            count += 1
        print("")

"""
Demonstration of the Naive Bayes class
using the Iris and Banknote Authentication datasets.
"""
iris = NaiveBayes()
iris.demonstration("iris.csv")

banknote = NaiveBayes()
banknote.demonstration("banknote_authentication.csv")
