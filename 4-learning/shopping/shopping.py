import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """

    # Constants used for data transformation (Month into int, remaining to bool)
    MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
    VISITORTYPE = {"Other": 0,
                   "New_Visitor": 0,
                   "Returning_Visitor": 1}
    WEEKEND = ("FALSE", "TRUE")
    REVENUE = ("FALSE", "TRUE")

    evidence = []
    labels = []

    with open(filename, mode='r') as INFILE:
        file = csv.reader(INFILE)
        next(file)
        i = 0
        for row in file:
            j = 0
            evidence.append([])
            labels.append([])

            for col in row[:17]:

                # Converting values to ints or float
                if j in [0, 2, 4, 11, 12, 13, 14]:
                    col = int(col)
                elif j in [1, 3, 5, 6, 7, 8, 9]:
                    col = float(col)
                elif j == 10:
                    col = MONTHS.index(col)
                elif j == 15:
                    col = VISITORTYPE[col]
                elif j == 16:
                    col = WEEKEND.index(col)

                # Adding each column to list for its row
                evidence[i].append(col)
                j += 1

            for col in row[17:]:
                labels[i].append(REVENUE.index(col))
            
            i += 1
    return evidence, labels


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    k_neigh = KNeighborsClassifier(n_neighbors=1)
    return k_neigh.fit(evidence, labels)


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """

    i = 0
    sensitivity = 0
    specificity = 0

    while i < len(labels):

        # If values match, checks if a positive or negative
        # prediction was accurately made
        if labels[i] == predictions[i]:
            if labels[i] == 1:
                sensitivity += 1
            else:
                specificity += 1
        i += 1

    # Converting sum of accurate predictions into proportions
    sensitivity /= sum(labels)
    specificity /= len(labels) - sum(labels)

    return sensitivity, specificity


if __name__ == "__main__":
    main()
