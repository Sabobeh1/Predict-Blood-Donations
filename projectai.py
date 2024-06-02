import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.datasets import load_iris
from sklearn import tree
import itertools
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
import matplotlib
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import messagebox
from tkinter import *


matplotlib.use('TkAgg')

# %matplotlib inline
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

global best_DT_model, best_model_NB, best_model_ann, best_y_pred_DT, best_y_pred_NB, y_pred_best_ann, x_train, x_test, y_train, y_test, old_x_test, old_y_test

best_DT_model = 0
best_model_NB = 0
best_model_ann = 0
best_y_pred_DT = 0
y_pred_best_ann = 0
old_x_test = 0
old_y_test = 0

path = "transfusion.data"
dataset = pd.read_csv(path)
dataset.head()

#convert to data frame 
dataset.dropna(inplace=True)

dataset.describe().round(2)

dataset.rename(columns={'whether he/she donated blood in March 2007': 'goal'}, inplace=True)
#dataset = dataset.drop(columns='Time (months)')
x = dataset.drop(columns='goal')

x.head()

y = dataset['goal']
y.value_counts()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1, stratify=y)
# x_train, x_test, y_train, y_test = train_test_split(x, y,test_size= 0.3,random_state= 1,stratify=y)
# x_train, x_test, y_train, y_test = train_test_split(x, y,test_size= 0.25,random_state= 1,stratify=y)

old_x_test = x_test
old_y_test = y_test
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

def plot_decision_tree(tree_model):
    iris = load_iris()

    # tree.export_graphviz(tree_model.fit(iris.data, iris.target), out_file=dot_out_file)

    # fig = plt.figure(figsize=(25,20))
    tree.plot_tree(tree_model.fit(iris.data, iris.target), feature_names=iris.feature_names,
                   class_names=iris.target_names, filled=True)
    plt.show()


class ClassificationReportTable(tk.Tk):
    def __init__(self, classification_report_str, accuracy, type, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        self.title("Classification Report Table")
        self.geometry("600x600")

        # Display accuracy from the passed variable
        accuracy_label_text = f"Accuracy: {accuracy:.3%}"
        accuracy_label = tk.Label(self, text=accuracy_label_text)
        accuracy_label.pack()

        classification_report_text = f"Classification Report:\n {classification_report_str}"
        report_label = tk.Label(self, text=classification_report_text)
        report_label.pack()

        b = tk.Button(self, text="Optimize Using Different Hyper Parameters", command=lambda: self.optimize(type))
        b.pack(pady=5)

    def optimize(self, type):
        global best_DT_model, best_model_NB, best_model_ann, best_y_pred_DT, best_y_pred_NB, y_pred_best_ann, x_train, x_test, y_train, y_test
        if (type == 0):
            # Define the parameter grid you want to search
            param_grid = {

                'max_depth': [None, 10, 20, 30, 40, 50],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]

            }

            # Create the Decision Tree classifier
            DT_model = DecisionTreeClassifier(random_state=1)

            # Create the GridSearchCV object
            grid_search = GridSearchCV(DT_model, param_grid, cv=5, scoring='accuracy')

            # Fit the grid search to your training data
            grid_search.fit(x_train, y_train)

            # Get the best hyperparameters from the grid search
            best_params = grid_search.best_params_

            # Get the best estimator (classifier) from the grid search
            best_DT_model = grid_search.best_estimator_

            plot_decision_tree(best_DT_model)

            # Print the best hyperparameters and the best estimator
            print("Best Hyperparameters:")
            print(best_params)
            print("\nBest Estimator:")
            print(best_DT_model)

            # Use the best estimator to make predictions
            best_y_pred_DT = best_DT_model.predict(x_test)

            # Calculate accuracy and classification report for the best estimator
            accuracy_best = accuracy_score(y_test, best_y_pred_DT)
            report_best = classification_report(y_test, best_y_pred_DT)


            # Print the accuracy and classification report for the best estimator
            print(f"\nAccuracy with Best Estimator: {accuracy_best}\n")
            print("Confusion Matrix:\n", confusion_matrix(y_test, best_y_pred_DT))
            print(f"Classification Report with Best Estimator:\n{report_best}")
            self.destroy()

        elif (type == 1):
            # Creating the Gaussian Naive Bayes model
            NB_model = GaussianNB()

            # Parameters grid. Note: GaussianNB doesn't have many hyperparameters to tune.
            # Here, we are only tuning 'var_smoothing'
            param_grid = {'var_smoothing': np.logspace(0, -9, num=100)}

            # Creating the GridSearchCV object
            grid_search = GridSearchCV(estimator=NB_model, param_grid=param_grid, cv=5, n_jobs=-1)

            # Fitting the grid search to the data
            grid_search.fit(x_train, y_train)

            # Extracting the best estimator
            best_model_NB = grid_search.best_estimator_

            # Predicting the Test set results
            best_y_pred_NB = best_model_NB.predict(x_test)

            # Calculating accuracy
            accuracy_best = accuracy_score(y_test, best_y_pred_NB)
            report_best = classification_report(y_test, best_y_pred_NB)

            # Print the best hyperparameters and the best estimator
            print("Best Hyperparameters:")
            print(grid_search.best_params_)
            print("\nBest Estimator:")
            print(best_model_NB)

            # Print the accuracy and classification report for the best estimator
            print(f"\nAccuracy with Best Estimator: {accuracy_best}\n")
            print("Confusion Matrix:\n", confusion_matrix(y_test, best_y_pred_NB))
            print(f"Classification Report with Best Estimator:\n{report_best}")
            self.destroy()
        else:
            # Define a parameter grid to search for best parameters for MLP
            parameter_space = {
                'hidden_layer_sizes': [(50,), (100,), (100, 50)],
                'activation': ['tanh', 'relu'],
                'solver': ['sgd', 'adam'],
                'alpha': [0.0001, 0.05],
                'learning_rate': ['constant', 'adaptive'],
            }

            # Create a GridSearchCV instance with MLPClassifier
            best_model_ann = GridSearchCV(MLPClassifier(max_iter=300, random_state=1),
                                          parameter_space, n_jobs=-1, cv=3)

            # Fit the model
            best_model_ann.fit(x_train, y_train)

            # Best parameter set
            # print('Best parameters found:\n', mlp_gs.best_params_)

            # All results
            means = best_model_ann.cv_results_['mean_test_score']
            stds = best_model_ann.cv_results_['std_test_score']
            for mean, std, params in zip(means, stds, best_model_ann.cv_results_['params']):
                print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

            # Predict on test data with the best parameters
            y_pred_best_ann = best_model_ann.predict(x_test)

            # Print evaluation results
            print("Accuracy:", accuracy_score(y_test, y_pred_best_ann))
            print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_best_ann))
            print("Classification Report:\n", classification_report(y_test, y_pred_best_ann))
            self.destroy()

def visualize_data():
    train_counts = y_train.value_counts().sort_index()
    test_counts = y_test.value_counts().sort_index()
    fig, ax = plt.subplots()
    barWidth = 0.35
    r1 = range(len(train_counts))
    r2 = [x + barWidth for x in r1]

    ax.bar(r1, train_counts, color='blue', width=barWidth, edgecolor='grey', label='Training Set')
    ax.bar(r2, test_counts, color='green', width=barWidth, edgecolor='grey', label='Test Set')

    plt.xlabel('Class', fontweight='bold')
    plt.ylabel('Count', fontweight='bold')
    plt.xticks([r + barWidth / 2 for r in range(len(train_counts))], train_counts.index)

    plt.legend()
    plt.title('Training and Test Set Distribution')
    plt.show()

def create_decision_tree(show_optmize):
    DT_model = DecisionTreeClassifier(random_state=1)
    DT_model.fit(x_train, y_train)
    y_pred_DT = DT_model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred_DT)
    report = classification_report(y_test, y_pred_DT)
    
    plot_decision_tree(DT_model)
    if(show_optmize):
        app = ClassificationReportTable(report, accuracy, 0)
        app.mainloop()

    return y_pred_DT[len(y_pred_DT)-1]


def create_naive_bayes_model(show_optmize):
    NB_model = GaussianNB()
    NB_model.fit(x_train, y_train)

    y_pred_NB = NB_model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred_NB)

    if(show_optmize):
        app = ClassificationReportTable(classification_report(y_test, y_pred_NB), accuracy, 1)
        app.mainloop()

    return y_pred_NB[len(y_pred_NB)-1]


def create_ann_model(show_optmize):
    # Create an MLPClassifier instance
    # Here, we use one hidden layer with 100 neurons, and relu as the activation function
    mlp = MLPClassifier(hidden_layer_sizes=(100,), activation='relu',
                        max_iter=300, solver='adam', random_state=1)

    # Train the model
    mlp.fit(x_train, y_train)

    # Predictions and evaluation
    predictions = mlp.predict(x_test)

    if(show_optmize):
        app = ClassificationReportTable(
            "Confusion Matrix:\n" + str(confusion_matrix(y_test, predictions)) + '\nClassification Report:\n' +
            str(classification_report(y_test, predictions)), accuracy_score(y_test, predictions), 2)
        app.mainloop()

    return predictions[len(predictions)-1]


def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()


def compare_all_methods():
    global best_DT_model, best_model_NB, best_model_ann, best_y_pred_DT, best_y_pred_NB, y_pred_best_ann, x_train, x_test, y_train, y_test
    # Define the parameter grid you want to search
    param_grid = {

        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]

    }

    # Create the Decision Tree classifier
    DT_model = DecisionTreeClassifier(random_state=1)

    # Create the GridSearchCV object
    grid_search = GridSearchCV(DT_model, param_grid, cv=5, scoring='accuracy')

    # Fit the grid search to your training data
    grid_search.fit(x_train, y_train)

    # Get the best hyperparameters from the grid search
    best_params = grid_search.best_params_

    # Get the best estimator (classifier) from the grid search
    best_DT_model = grid_search.best_estimator_

    # Use the best estimator to make predictions
    best_y_pred_DT = best_DT_model.predict(x_test)

    # Creating the Gaussian Naive Bayes model
    NB_model = GaussianNB()

    # Parameters grid. Note: GaussianNB doesn't have many hyperparameters to tune.
    # Here, we are only tuning 'var_smoothing'
    param_grid = {'var_smoothing': np.logspace(0, -9, num=100)}

    # Creating the GridSearchCV object
    grid_search = GridSearchCV(estimator=NB_model, param_grid=param_grid, cv=5, n_jobs=-1)

    # Fitting the grid search to the data
    grid_search.fit(x_train, y_train)

    # Extracting the best estimator
    best_model_NB = grid_search.best_estimator_

    # Predicting the Test set results
    best_y_pred_NB = best_model_NB.predict(x_test)

    # Define a parameter grid to search for best parameters for MLP
    parameter_space = {
        'hidden_layer_sizes': [(50,), (100,), (100, 50)],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.05],
        'learning_rate': ['constant', 'adaptive'],
    }

    # Create a GridSearchCV instance with MLPClassifier
    best_model_ann = GridSearchCV(MLPClassifier(max_iter=300, random_state=1),
                                  parameter_space, n_jobs=-1, cv=3)

    # Fit the model
    best_model_ann.fit(x_train, y_train)

    # Best parameter set
    # print('Best parameters found:\n', mlp_gs.best_params_)

    # All results
    means = best_model_ann.cv_results_['mean_test_score']
    stds = best_model_ann.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, best_model_ann.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

    # Predict on test data with the best parameters
    y_pred_best_ann = best_model_ann.predict(x_test)

    models = [best_DT_model, best_model_NB, best_model_ann]
    model_names = ["Decision Tree", "Naive Bayes", "Artificial Nerual Network"]
    y_preds = [best_y_pred_DT, best_y_pred_NB, y_pred_best_ann]

    for i, model in enumerate(models):
        cm = confusion_matrix(y_test, y_preds[i])
        plt.figure()
        plot_confusion_matrix(cm, classes=['Not Donated', 'Donated'], title=f'Confusion Matrix for {model_names[i]}')

    comparisons = pd.DataFrame(columns=["Model", "Accuracy", "Precision", "Recall", "F1-Score"])
    for i, model in enumerate(models):
        comparisons = comparisons._append({
            "Model": model_names[i],
            "Accuracy": accuracy_score(y_test, y_preds[i]),
            "Precision": precision_score(y_test, y_preds[i]),
            "Recall": recall_score(y_test, y_preds[i]),
            "F1-Score": f1_score(y_test, y_preds[i])
        }, ignore_index=True)

    print(comparisons)

    def plot_model_accuracies(df, title='Model Accuracies'):
        colors = plt.cm.viridis(np.linspace(0, 1, len(df)))

        fig, ax = plt.subplots()
        y_pos = np.arange(len(df))
        bars = ax.barh(y_pos, df["Accuracy"], align='center', color=colors, ecolor='black')

        for bar, model in zip(bars, df["Model"]):
            bar.set_label(model)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(df["Model"])
        ax.invert_yaxis()  # labels read top-to-bottom
        ax.set_xlabel('Accuracy')
        ax.set_title(title)
        # ax.legend()
        # plt.legend(loc="lower right")
        plt.tight_layout()
        plt.show()
        return fig, ax

    fig, ax = plot_model_accuracies(comparisons)

def add_new_test_data_row():
    global x_test, y_test, old_x_test, old_y_test, x_train
    def checkInputs():
        global x_test, y_test, old_x_test, old_y_test, x_train
        input1_val = input1.get()
        input2_val = input2.get()
        input3_val = input3.get()
        input4_val = input4.get()

        if not (input1_val.isdigit() and input2_val.isdigit() and input3_val.isdigit() and input4_val.isdigit()):
            messagebox.showerror("Error", "Inputs must be numbers")
            return False
        
        old_x_test.loc[len(old_x_test)] = [input1_val, input2_val, input3_val, input4_val]
        old_y_test.loc[len(old_y_test)] = choice.get()
        x_test = old_x_test
        scaler = MinMaxScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        y_test = old_y_test
        return True

    def decision_tree():
        # Check if inputs are numeric
        if not checkInputs():
            return
        
        pred = create_decision_tree(False)
        if(pred == choice.get()):
            print("Model predictted the value correctly (" + str(pred) + ")")
        else:
            print("Model didn't predict the value correctly (" + str(pred) + ")")
    
    def naive_bias():
        # Check if inputs are numeric
        if not checkInputs():
            return

        pred = create_naive_bayes_model(False)
        if(pred == choice.get()):
            print("Model predictted the value correctly (" + str(pred) + ")")
        else:
            print("Model didn't predict the value correctly (" + str(pred) + ")")

    def ann():
        # Check if inputs are numeric
        if not checkInputs():
            return
        
        pred = create_ann_model(False)
        if(pred == choice.get()):
            print("Model predictted the value correctly (" + str(pred) + ")")
        else:
            print("Model didn't predict the value correctly (" + str(pred) + ")")

    # Create main window
    new_window = tk.Tk()
    new_window.title("New test row")

    # Create input fields
    input1_label = tk.Label(new_window, text="Recency:")
    input1_label.grid(row=0, column=0, padx=5, pady=5)
    input1 = tk.Entry(new_window)
    input1.grid(row=0, column=1, padx=5, pady=5)

    input2_label = tk.Label(new_window, text="Frequency:")
    input2_label.grid(row=1, column=0, padx=5, pady=5)
    input2 = tk.Entry(new_window)
    input2.grid(row=1, column=1, padx=5, pady=5)

    input3_label = tk.Label(new_window, text="Monetary:")
    input3_label.grid(row=2, column=0, padx=5, pady=5)
    input3 = tk.Entry(new_window)
    input3.grid(row=2, column=1, padx=5, pady=5)

    input4_label = tk.Label(new_window, text="Time:")
    input4_label.grid(row=3, column=0, padx=5, pady=5)
    input4 = tk.Entry(new_window)
    input4.grid(row=3, column=1, padx=5, pady=5)

    def set_choice(value):
        choice.set(value)

    # Create radio buttons
    choice = tk.IntVar()
    choice_label = tk.Label(new_window, text="Actual Value:")
    choice_label.grid(row=4, column=0, padx=5, pady=5)
    radio_button_0 = tk.Radiobutton(new_window, text="0", variable=choice, value=0, command=lambda: set_choice(0))
    radio_button_0.grid(row=4, column=1, padx=5, pady=5)
    radio_button_1 = tk.Radiobutton(new_window, text="1", variable=choice, value=1, command=lambda: set_choice(1))
    radio_button_1.grid(row=4, column=2, padx=5, pady=5)

    # Create submit buttons
    dt_button = tk.Button(new_window, text="Use Decision Tree", command=lambda: decision_tree())
    dt_button.grid(row=5, column=0, columnspan=3, padx=5, pady=5)

    nb_button = tk.Button(new_window, text="Use NB model", command=lambda: naive_bias())
    nb_button.grid(row=6, column=0, columnspan=3, padx=5, pady=5)

    ann_button = tk.Button(new_window, text="Use ANN Model", command=lambda: ann())
    ann_button.grid(row=7, column=0, columnspan=3, padx=5, pady=5)

    # Run the main event loop
    new_window.mainloop()


# Create the main window\\
window = tk.Tk()
window.title("Prediction Algorithms")

# Create a label
label = tk.Label(window, text="Which Prediction Method You want to use: ")
label.pack(pady=10)

# Create three buttons
button0 = tk.Button(window, text="Visualize Data", command=lambda: visualize_data())
button0.pack(pady=5)

button1 = tk.Button(window, text="Decision Tree", command=lambda: create_decision_tree(True))
button1.pack(pady=5)

button2 = tk.Button(window, text="Naive Bayes", command=lambda: create_naive_bayes_model(True))
button2.pack(pady=5)

button3 = tk.Button(window, text="Artificial Nerual Network", command=lambda: create_ann_model(True))
button3.pack(pady=5)

button4 = tk.Button(window, text="Compare All Three Methods", command=lambda: compare_all_methods())
button4.pack(pady=5)

button5 = tk.Button(window, text="Add new test Row", command=lambda: add_new_test_data_row())
button5.pack(pady=5)

# Start the GUI event loop
window.mainloop()