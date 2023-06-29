# function to do actual conformal prediction
import numpy as np
def conformal(cal_smx, cal_labels, val_labels, val_smx, size_cal_set):
    cal_scores = 1 - cal_smx[np.arange(size_cal_set), cal_labels]
    alphas = []
    prediction_sets = []
    actual_labels = []
    i = 0

    for i, image in enumerate(val_smx):
        # try alphas until there's only one class in the prediction set
        for alpha in np.arange(0.01, 1, 0.0001):
            q_level = np.ceil((size_cal_set + 1) * (1 - alpha)) / size_cal_set
            qhat = np.quantile(cal_scores, q_level, interpolation='higher')

            prediction_set = image >= (1 - qhat)

            if np.sum(prediction_set) == 1:
                prediction_sets.append(prediction_set)
                alphas.append(alpha)
                actual_labels.append(val_labels[i])
                break

    return prediction_sets, alphas, actual_labels

# generate empirical coverage of conformal prediction
def evaluate_conformal(prediction_sets, actual_labels):
    prediction_sets = np.array(prediction_sets)
    actual_labels = np.array(actual_labels)
    empirical_coverage = prediction_sets[np.arange(prediction_sets.shape[0]),actual_labels].mean()
    print(f"The empirical coverage is: {empirical_coverage}")
    return empirical_coverage

def create_fake_logits(alphas):
    fake_logits = 1 - np.array(alphas)
    return fake_logits