import numpy as np
import pandas as pd
from utils import *
import warnings
import matplotlib.pyplot as plt
#from sklearn import metrics

def evaluation_metrics(y_true, y_predicted):
    tp = np.sum(np.logical_and(np.equal(y_true, 1), np.equal(y_predicted, 1)))
    tn = np.sum(np.logical_and(np.equal(y_true, 0), np.equal(y_predicted, 0)))
    fp = np.sum(np.logical_and(np.equal(y_true, 0), np.equal(y_predicted, 1)))
    fn = np.sum(np.logical_and(np.equal(y_true, 1), np.equal(y_predicted, 0)))

    confusion_matrix = [[tp, fn], [fp, tn]]
    accuracy = np.mean(np.equal(y_true, y_predicted))
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    print(f"Accuracy {accuracy} ")
    print(f"Precision {precision} ")
    print(f"Recall {recall} ")
    print(f"Confusion Matrix {confusion_matrix} ")
    return accuracy

def count_table(docs, vocab):
    vocab_size = len(vocab)
    counts_table = []

    docs = pd.DataFrame(docs)
    for review in docs.itertuples():
        review = review[1::]
        review_df = pd.DataFrame(review)
        word_count = review_df.value_counts()
        doc_count = [0] * vocab_size
        for word, count in word_count.items():
            word = word[0]
            doc_count[vocab.index(word)] = count
        counts_table.append(doc_count)

    return np.array(counts_table)

def probability_prediction(test_data,
                           prob_positive, prob_negative,
                           positive_word_freq, positive_total_word_count,
                           negative_word_freq, negative_total_word_count,
                           laplace_param, vocab_list, log_enable):
    predicted_labels = []
    positive_probs = []
    negative_probs = []

    for j in range(len(test_data)):
        doc = test_data[j]
        if log_enable == 1:
            positive_prob = np.log(prob_positive)
            negative_prob = np.log(prob_negative)
        else:
            positive_prob = prob_positive
            negative_prob = prob_negative

        for w in doc:
            if w in vocab_list:
                if log_enable == 1:
                    positive_prob += np.log((positive_word_freq[vocab_list.index(w)] + laplace_param) /
                                            (positive_total_word_count + laplace_param * len(vocab_list)))
                    negative_prob += np.log((negative_word_freq[vocab_list.index(w)] + laplace_param) /
                                            (negative_total_word_count + laplace_param * len(vocab_list)))
                else:
                    positive_prob *= ((positive_word_freq[vocab_list.index(w)] + laplace_param) /
                                      (positive_total_word_count + laplace_param * len(vocab_list)))
                    negative_prob *= ((negative_word_freq[vocab_list.index(w)] + laplace_param) /
                                      (negative_total_word_count + laplace_param * len(vocab_list)))
            else:
                if log_enable == 1:
                    positive_prob += np.log(
                        laplace_param / (positive_total_word_count + laplace_param * len(vocab_list)))
                    negative_prob += np.log(
                        laplace_param / (negative_total_word_count + laplace_param * len(vocab_list)))
                else:
                    positive_prob *= (laplace_param / (positive_total_word_count + laplace_param * len(vocab_list)))
                    negative_prob *= (laplace_param / (negative_total_word_count + laplace_param * len(vocab_list)))
        positive_probs.append(positive_prob)
        negative_probs.append(negative_prob)

    for j in range(len(positive_probs)):
        if positive_probs[j] >= negative_probs[j]:
            predicted_labels.append(1)
        else:
            predicted_labels.append(0)

    return predicted_labels

def naive_bayes():
    percentage_positive_instances_train = 0.2
    percentage_negative_instances_train = 0.2
    percentage_positive_instances_test = 0.2
    percentage_negative_instances_test = 0.2

    (pos_train, neg_train, vocab) = load_training_set(percentage_positive_instances_train,
                                                      percentage_negative_instances_train)
    (pos_test, neg_test) = load_test_set(percentage_positive_instances_test, percentage_negative_instances_test)

    prob_positive = len(pos_train) / (len(pos_train) + len(neg_train))
    prob_negative = len(neg_train) / (len(pos_train) + len(neg_train))

    ##if(prob_negative==prob_positive)

    print("Number of positive training instances:", len(pos_train))
    print("Number of negative training instances:", len(neg_train))
    print("Number of positive test instances:", len(pos_test))
    print("Number of negative test instances:", len(neg_test))
    print("Prob of positive test instances:", prob_positive)
    print("Prob of negative test instances:", prob_negative)

    vocab_list = list(vocab)
    positive_train_table = count_table(pos_train, vocab_list)
    negative_train_table = count_table(neg_train, vocab_list)

    positive_word_freq = np.sum(positive_train_table, axis=0)  # total frequency count of each word
    positive_total_word_count = np.sum(positive_train_table)

    # print("positive_word_freq", positive_word_freq)
    # print("positive_total_word_count ",positive_total_word_count)

    negative_word_freq = np.sum(negative_train_table, axis=0)
    negative_total_word_count = np.sum(negative_train_table)

    # print("negative_word_freq", negative_word_freq)
    # print("negative_total_word_count", negative_total_word_count)

    test_data = pos_test + neg_test

    log_enable = 0
    laplace_param = 10 ** -5

    test_label_predicted = probability_prediction(test_data,
                                                  prob_positive, prob_negative,
                                                  positive_word_freq, positive_total_word_count,
                                                  negative_word_freq, negative_total_word_count,
                                                  laplace_param, vocab_list, log_enable)

    log_enable = 1
    laplace_param = 10 ** -5
    log_test_label_predicted = probability_prediction(test_data,
                                                      prob_positive, prob_negative,
                                                      positive_word_freq, positive_total_word_count,
                                                      negative_word_freq, negative_total_word_count,
                                                      laplace_param, vocab_list, log_enable)

    true_test_label = np.concatenate((np.ones(len(pos_test)), np.zeros(len(neg_test))), axis=0)


    print("\n")
    print("ANSWER 1")
    print("WITHOUT LOG PROBABILITY  ")
    evaluation_metrics(true_test_label, test_label_predicted)

    print("\n")
    print("WITH LOG PROBABILITY")
    evaluation_metrics(true_test_label, log_test_label_predicted)

    print("\n")
    print("ANSWER 2")
    print("EVALUATION METRICS FOR ALPHA=1")

    laplace_param = 1
    test_label_predicted = probability_prediction(test_data,
                                                  prob_positive, prob_negative,
                                                  positive_word_freq, positive_total_word_count,
                                                  negative_word_freq, negative_total_word_count,
                                                  laplace_param, vocab_list, log_enable)
    evaluation_metrics(true_test_label, test_label_predicted)

    print("\n")
    print(" PLOT OF ALPHA VS ACCURACY ")
    laplace_params = [0.0001, 0.001, 0.01, 0.1, 1.0, 10, 100, 1000]
    accuracy_list = []

    for j in laplace_params:
        laplace_test_label_predicted = probability_prediction(test_data,
                                                              prob_positive, prob_negative,
                                                              positive_word_freq, positive_total_word_count,
                                                              negative_word_freq, negative_total_word_count,
                                                              j, vocab_list, log_enable)
        print("for alpha ", j)
        accuracy_list.append(evaluation_metrics(true_test_label, laplace_test_label_predicted))

    plt.plot(laplace_params, accuracy_list)
    plt.scatter(laplace_params, accuracy_list)
    plt.xticks((np.asarray(laplace_params)))
    plt.xscale("log")
    plt.xlabel("ALPHA ")
    plt.ylabel("ACCURACY ")
    plt.title("PLOT OF ALPHA VS ACCURACY")
    plt.show()

    print("\n")
    print("ANSWER 3")

    percentage_positive_instances_train = 1
    percentage_negative_instances_train = 1
    percentage_positive_instances_test = 1
    percentage_negative_instances_test = 1

    (pos_train, neg_train, vocab) = load_training_set(percentage_positive_instances_train,
                                                      percentage_negative_instances_train)
    (pos_test, neg_test) = load_test_set(percentage_positive_instances_test, percentage_negative_instances_test)

    prob_positive = len(pos_train) / (len(pos_train) + len(neg_train))
    prob_negative = len(neg_train) / (len(pos_train) + len(neg_train))

    print("Number of positive training instances:", len(pos_train))
    print("Number of negative training instances:", len(neg_train))
    print("Number of positive test instances:", len(pos_test))
    print("Number of negative test instances:", len(neg_test))
    print("Prob of positive test instances:", prob_positive)
    print("Prob of negative test instances:", prob_negative)

    vocab_list = list(vocab)
    positive_train_table = count_table(pos_train, vocab_list)
    negative_train_table = count_table(neg_train, vocab_list)

    positive_word_freq = np.sum(positive_train_table, axis=0)
    positive_total_word_count = np.sum(positive_train_table)

    negative_word_freq = np.sum(negative_train_table, axis=0)
    negative_total_word_count = np.sum(negative_train_table)

    test_data = pos_test + neg_test
    log_enable = 1
    laplace_param = 10

    test_label_predicted = probability_prediction(test_data,
                                                  prob_positive, prob_negative,
                                                  positive_word_freq, positive_total_word_count,
                                                  negative_word_freq, negative_total_word_count,
                                                  laplace_param, vocab_list, log_enable)

    true_test_label = np.concatenate((np.ones(len(pos_test)), np.zeros(len(neg_test))), axis=0)
    evaluation_metrics(true_test_label, test_label_predicted)

    print("\n")
    print("ANSWER 4")
    print("50% : 100% :: TRAINING : TEST DATA")

    percentage_positive_instances_train = 0.5
    percentage_negative_instances_train = 0.5
    percentage_positive_instances_test = 1
    percentage_negative_instances_test = 1

    (pos_train, neg_train, vocab) = load_training_set(percentage_positive_instances_train,
                                                      percentage_negative_instances_train)
    (pos_test, neg_test) = load_test_set(percentage_positive_instances_test, percentage_negative_instances_test)

    prob_positive = len(pos_train) / (len(pos_train) + len(neg_train))
    prob_negative = len(neg_train) / (len(pos_train) + len(neg_train))

    print("Number of positive training instances:", len(pos_train))
    print("Number of negative training instances:", len(neg_train))
    print("Number of positive test instances:", len(pos_test))
    print("Number of negative test instances:", len(neg_test))
    print("Prob of positive test instances:", prob_positive)
    print("Prob of negative test instances:", prob_negative)

    vocab_list = list(vocab)
    positive_train_table = count_table(pos_train, vocab_list)
    negative_train_table = count_table(neg_train, vocab_list)

    positive_word_freq = np.sum(positive_train_table, axis=0)
    positive_total_word_count = np.sum(positive_train_table)

    negative_word_freq = np.sum(negative_train_table, axis=0)
    negative_total_word_count = np.sum(negative_train_table)

    test_data = pos_test + neg_test

    log_enable = 1
    laplace_param = 10

    test_label_predicted = probability_prediction(test_data,
                                                  prob_positive, prob_negative,
                                                  positive_word_freq, positive_total_word_count,
                                                  negative_word_freq, negative_total_word_count,
                                                  laplace_param, vocab_list, log_enable)

    true_test_label = np.concatenate((np.ones(len(pos_test)), np.zeros(len(neg_test))), axis=0)
    evaluation_metrics(true_test_label, test_label_predicted)

    print("\n")
    print("ANSWER 6")
    print("UNBALANCED DATASET")

    percentage_positive_instances_train = 0.1
    percentage_negative_instances_train = 0.5
    percentage_positive_instances_test = 1
    percentage_negative_instances_test = 1

    (pos_train, neg_train, vocab) = load_training_set(percentage_positive_instances_train,
                                                      percentage_negative_instances_train)
    (pos_test, neg_test) = load_test_set(percentage_positive_instances_test, percentage_negative_instances_test)

    prob_positive = len(pos_train) / (len(pos_train) + len(neg_train))
    prob_negative = len(neg_train) / (len(pos_train) + len(neg_train))

    print("Number of positive training instances:", len(pos_train))
    print("Number of negative training instances:", len(neg_train))
    print("Number of positive test instances:", len(pos_test))
    print("Number of negative test instances:", len(neg_test))
    print("Prob of positive test instances:", prob_positive)
    print("Prob of negative test instances:", prob_negative)

    vocab_list = list(vocab)
    positive_train_table = count_table(pos_train, vocab_list)
    negative_train_table = count_table(neg_train, vocab_list)

    positive_word_freq = np.sum(positive_train_table, axis=0)
    positive_total_word_count = np.sum(positive_train_table)

    negative_word_freq = np.sum(negative_train_table, axis=0)
    negative_total_word_count = np.sum(negative_train_table)

    test_data = pos_test + neg_test

    log_enable = 1
    laplace_param = 10

    test_label_predicted = probability_prediction(test_data,
                                                  prob_positive, prob_negative,
                                                  positive_word_freq, positive_total_word_count,
                                                  negative_word_freq, negative_total_word_count,
                                                  laplace_param, vocab_list, log_enable)

    true_test_label = np.concatenate((np.ones(len(pos_test)), np.zeros(len(neg_test))), axis=0)
    evaluation_metrics(true_test_label, test_label_predicted)


if __name__ == "__main__":
    warnings.simplefilter('ignore')
    naive_bayes()
