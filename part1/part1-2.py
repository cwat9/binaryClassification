import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def load_data(file_path):
    mat = scipy.io.loadmat(file_path)
    return mat['train_fea1'].astype(np.float32) / 255.0, mat['train_gnd1']

def get_class_indices(data_gnd, label, num_samples=1500):
    class_indices = np.where(data_gnd[:, 0] == label)[0][:num_samples]
    return class_indices

def split_data(class1_idx, class2_idx):
    train_idx = np.concatenate([class1_idx[:500], class2_idx[:500]])
    validation_idx = np.concatenate([class1_idx[500:1000], class2_idx[500:1000]])
    test_idx = np.concatenate([class1_idx[1000:1500], class2_idx[1000:1500]])
    return train_idx, validation_idx, test_idx

def train_logistic_regression(x_train, y_train, C):
    model = LogisticRegression(solver='liblinear', C=C, tol=1e-7, max_iter=1000)
    model.fit(x_train, y_train.ravel())
    return model

def train_random_forest(x_train, y_train, n_estimators):
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(x_train, y_train.ravel())
    return model

def evaluate_model(model, x, y):
    return model.score(x, y)

def plot_scores(c_values, val_scores, train_scores, best_C, best_val_score, test_score,
                c_values_rf, val_scores_rf, train_scores_rf, best_n_estimators, best_val_score_rf, test_score_rf):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Logistic Regression Plot
    ax1.semilogx(c_values, val_scores, color="blue", label="True Score (Logistic Regression)")
    ax1.semilogx(c_values, train_scores, color="red", label="Train Score (Logistic Regression)")
    ax1.text(best_C, best_val_score, f' {best_C} , {best_val_score}')
    ax1.semilogx(best_C, test_score, marker="o", markersize=5, markeredgecolor="green", markerfacecolor="green", label='Test Score (Logistic Regression)')

    # Random Forest Plot
    ax2.semilogx(c_values_rf, val_scores_rf, color="green", label="True Score (Random Forest)")
    ax2.semilogx(c_values_rf, train_scores_rf, color="orange", label="Train Score (Random Forest)")
    ax2.text(best_n_estimators, best_val_score_rf, f' {best_n_estimators} , {best_val_score_rf}')
    ax2.semilogx(best_n_estimators, test_score_rf, marker="o", markersize=5, markeredgecolor="blue", markerfacecolor="blue", label='Test Score (Random Forest)')

    ax1.legend(loc="lower right")
    ax2.legend(loc="lower right")

    ax1.set(xlabel="C Values by Powers of 10", ylabel="Score Values", ylim=(0.8, 1))
    ax2.set(xlabel="Number of Trees", ylabel="Score Values", ylim=(0.8, 1))

    plt.show()

def main():
    data_fea, data_gnd = load_data('MNISTmini.mat')

    # testing 5 vs 9
    labels = [5, 9]

    c_values = [10**i for i in range(-7, 5)]
    n_estimators_values = [50, 100, 150, 200, 1000]

    plotC, plotval, plottrain = [], [], []
    plotC_rf, plotval_rf, plottrain_rf = [], [], []

    for C in c_values:
        class1_idx = get_class_indices(data_gnd, labels[0])
        class2_idx = get_class_indices(data_gnd, labels[1])

        train_idx, validation_idx, test_idx = split_data(class1_idx, class2_idx)

        x_train, x_val, y_train, y_val = train_test_split(
            data_fea[train_idx, :], data_gnd[train_idx, :].ravel(), test_size=0.2, random_state=42)

        # Logistic Regression
        model_lr = train_logistic_regression(x_train, y_train, C)

        val_score_lr = evaluate_model(model_lr, data_fea[validation_idx, :], data_gnd[validation_idx, :])
        train_score_lr = evaluate_model(model_lr, x_train, y_train)

        plotval.append(val_score_lr)
        plottrain.append(train_score_lr)
        plotC.append(C)

        # Random Forest
        for n_estimators in n_estimators_values:
            model_rf = train_random_forest(x_train, y_train, n_estimators)

            val_score_rf = evaluate_model(model_rf, data_fea[validation_idx, :], data_gnd[validation_idx, :])
            train_score_rf = evaluate_model(model_rf, x_train, y_train)

            plotval_rf.append(val_score_rf)
            plottrain_rf.append(train_score_rf)
            plotC_rf.append(n_estimators)

    best_val_score = max(plotval)
    best_val_idx = plotval.index(best_val_score)
    best_C = plotC[best_val_idx]

    best_model = train_logistic_regression(data_fea[train_idx, :], data_gnd[train_idx, :].ravel(), best_C)
    test_score = evaluate_model(best_model, data_fea[test_idx, :], data_gnd[test_idx, :])

    print(f"Logistic Regression Score against the test data: {test_score}")

    best_val_score_rf = max(plotval_rf)
    best_val_idx_rf = plotval_rf.index(best_val_score_rf)
    best_n_estimators = plotC_rf[best_val_idx_rf]

    best_model_rf = train_random_forest(data_fea[train_idx, :], data_gnd[train_idx, :].ravel(), best_n_estimators)
    test_score_rf = evaluate_model(best_model_rf, data_fea[test_idx, :], data_gnd[test_idx, :])

    print(f"Random Forest Score against the test data: {test_score_rf}")

    plot_scores(plotC, plotval, plottrain, best_C, best_val_score, test_score,
                plotC_rf, plotval_rf, plottrain_rf, best_n_estimators, best_val_score_rf, test_score_rf)

if __name__ == "__main__":
    main()
