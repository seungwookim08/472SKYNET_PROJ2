import numpy as np
import math
import re
import sys
import time


output_line_counter = 1
max_filter_length = 9
min_filter_length = 2


def get_token_count(label, length_filter, stopword_filter):
    global max_filter_length, min_filter_length
    stopwords = open("English-Stop-Words.txt", "r").readlines()
    tokens = dict()
    file_count = 0
    i = 1
    while True:
        # print("/train/train-{}-{:05d}.txt".format(label, i))
        try:
            with open(".\\train\\train-{}-{:05d}.txt".format(label, i), encoding="latin-1") as test_file:
                txt = test_file.read().lower()
                tokenized = list(filter(None, re.split('[^a-zA-Z]', txt)))

                if stopword_filter:
                    tokenized = [token for token in tokenized if token not in stopwords]

                if length_filter:
                    tokenized = [token for token in tokenized if min_filter_length < len(token) < max_filter_length]

                for token in tokenized:
                    if token in tokens:
                        tokens[token] += 1
                    else:
                        tokens[token] = 1
                file_count += 1
        except FileNotFoundError:
            #print('Build model for {} is done'.format(label))
            break
        i += 1
    return tokens, file_count


def build_model(model_filename, length_filter=False, stopword_filter=False):
    ham_token, ham_file_count = get_token_count('ham', length_filter, stopword_filter)
    spam_token, spam_file_count = get_token_count('spam', length_filter, stopword_filter)

    words = set(list(ham_token.keys()) + list(spam_token.keys()))

    for word in words:
        if word not in ham_token.keys():
            ham_token[word] = 0
        if word not in spam_token.keys():
            spam_token[word] = 0

    ham_token_smooth = {key: ham_token[key] + 0.5 for key in words}
    spam_token_smooth = {key: spam_token[key] + 0.5 for key in words}
    total_ham = sum(list(ham_token_smooth.values()))
    total_spam = sum(list(spam_token_smooth.values()))

    ham_model = {key: ham_token_smooth[key]/total_ham for key in words}
    spam_model = {key: spam_token_smooth[key]/total_spam for key in words}

    linecount = 1
    # now sort token by alphabetic order
    sorted_keys = sorted(words)
    model_file = open(model_filename, "w+")
    for key in sorted_keys:
        model_file.write('{}  {}  {}  {}  {}  {}\n'.format(
            linecount, key, ham_token[key], ham_model[key], spam_token[key], spam_model[key]))
        linecount += 1
    return ham_file_count, spam_file_count


def classify_set(prior_ham, prior_spam, fileset, nb):
    i = 1
    spam_count = 0
    ham_count = 0
    classifications = list()
    while True:
        try:
            classification = ""
            score_ham = 0
            score_spam = 0
            with open(".\\test\\test-{}-{:05d}.txt".format(fileset, i), encoding="latin-1") as test_file:
                txt = test_file.read().lower()
                tokenized = re.split('[^a-zA-Z]', txt)
                for token in tokenized:
                    if token in nb:
                        ham_token_score, spam_token_score = nb[token]
                        score_ham = score_ham + math.log10(float(ham_token_score))
                        score_spam = score_spam + math.log10(float(spam_token_score))
                    else:
                        pass
                # add prior
                score_ham = score_ham + math.log10(prior_ham)
                score_spam = score_spam + math.log10(prior_spam)
                if score_spam >= score_ham:
                    spam_count += 1
                    classification = "spam"
                else:
                    ham_count += 1
                    classification = "ham"
                classifications.append((classification, score_spam, score_ham))
        except FileNotFoundError:
            #print('Classification test for %s done.' % fileset)
            break
        i += 1
    return classifications, spam_count, ham_count


def load_model(model_filename):
    model_file = open(model_filename)
    lines = model_file.readlines()
    model = dict()
    for line in lines:
        words = line.split("  ")
        model[words[1]] = (words[3], words[5])
    return model


def output_results(classifications, output_file, class_under_test):
    global output_line_counter
    for classification_index, (classification, score_spam, score_ham) in enumerate(classifications):
        judgement = "wrong"
        if classification == class_under_test:
            judgement = "right"
        output_file.write("{}  test-{}-{:05d}.txt  {}  {}  {}  {}  {}\n".format(output_line_counter, class_under_test, classification_index, classification, score_ham, score_spam, class_under_test, judgement))
        output_line_counter += 1


def NB_Classifer(ham_file_count, spam_file_count, model_filename, output_filename):
    nb = load_model(model_filename)
    output_file = open(output_filename, "w+")

    # compute priors
    prior_ham = ham_file_count/(ham_file_count+spam_file_count)
    prior_spam = spam_file_count/(ham_file_count+spam_file_count)

    hamtest_classifications, hamtest_wrong_count, hamtest_right_count = classify_set(prior_ham, prior_spam, "ham", nb)
    output_results(hamtest_classifications, output_file, "ham")
    print('accuracy for ham ', hamtest_right_count / (hamtest_right_count + hamtest_wrong_count))

    spamtest_classifications, spamtest_right_count, spamtest_wrong_count = classify_set(prior_ham, prior_spam, "spam", nb)
    output_results(spamtest_classifications, output_file, "spam")
    print('accuracy for spam ', spamtest_right_count / (spamtest_right_count + spamtest_wrong_count))


def __main__():
    num_runs = 1
    baseline_total_time = 0
    baseline_build_time = 0
    baseline_class_time = 0

    stopword_total_time = 0
    stopword_build_time = 0
    stopword_class_time = 0

    wordlength_total_time = 0
    wordlength_build_time = 0
    wordlength_class_time = 0

    hybrid_total_time = 0
    hybrid_build_time = 0
    hybrid_class_time = 0

    for i in range(0, num_runs):

        print("Starting baseline tests")
        start_time = time.time()*1000
        ham_file_count, spam_file_count = build_model("baseline-model.txt")
        build_end = time.time() * 1000
        # print("Model build computation time: %fms" % (build_end-start_time))
        baseline_build_time = baseline_build_time + (build_end-start_time)

        NB_Classifer(ham_file_count, spam_file_count, "baseline-model.txt", "baseline-result.txt")
        end_time = time.time()*1000
        # print("Classification computation time: %fms" % (end_time-build_end))
        baseline_class_time = baseline_class_time + (end_time-build_end)

        #print("Total computation time: %fms\n" % (end_time-start_time))
        baseline_total_time = baseline_total_time + (end_time-start_time)



        print("Starting stopword tests")
        start_time = time.time()*1000
        ham_file_count, spam_file_count = build_model("stopword-model.txt", stopword_filter=True)
        build_end = time.time() * 1000
        #print("Model build computation time: %fms" % (build_end-start_time))
        stopword_build_time = stopword_build_time + (build_end-start_time)

        NB_Classifer(ham_file_count, spam_file_count, "stopword-model.txt", "stopword-result.txt")
        end_time = time.time()*1000
        #print("Classification computation time: %fms" % (end_time-build_end))
        stopword_class_time = stopword_class_time + (end_time-build_end)

        #print("Total computation time: %fms\n" % (end_time-start_time))
        stopword_total_time = stopword_total_time + (end_time-start_time)


        print("Starting wordlength tests")
        start_time = time.time()*1000
        ham_file_count, spam_file_count = build_model("wordlength-model.txt", length_filter=True)
        build_end = time.time() * 1000
        #print("Model build computation time: %fms" % (build_end-start_time))
        wordlength_build_time = wordlength_build_time + (build_end-start_time)

        NB_Classifer(ham_file_count, spam_file_count, "wordlength-model.txt", "wordlength-result.txt")
        end_time = time.time()*1000
        #print("Classification computation time: %fms" % (end_time-build_end))
        wordlength_class_time = wordlength_class_time + (end_time-build_end)

        #print("Total computation time: %fms\n" % (end_time-start_time))
        wordlength_total_time = wordlength_total_time + (end_time-start_time)


        print("Starting hybrid tests")
        start_time = time.time()*1000
        ham_file_count, spam_file_count = build_model("hybrid-model.txt", stopword_filter=True, length_filter=True)
        build_end = time.time() * 1000
        #print("Model build computation time: %fms" % (build_end-start_time))
        hybrid_build_time = hybrid_build_time + (build_end-start_time)

        NB_Classifer(ham_file_count, spam_file_count, "hybrid-model.txt", "hybrid-result.txt")
        end_time = time.time()*1000
        #print("Classification computation time: %fms" % (end_time-build_end))
        hybrid_class_time = hybrid_class_time + (end_time-build_end)

        #print("Total computation time: %fms\n" % (end_time-start_time))
        hybrid_total_time = hybrid_total_time + (end_time-start_time)

    print("\nBaseline:")
    print("Total time: %f" % (baseline_total_time/num_runs))
    print("Buld time: %f" % (baseline_build_time/num_runs))
    print("Classification time: %f" % (baseline_class_time/num_runs))

    print("\nStopwords:")
    print("Total time: %f" % (stopword_total_time/num_runs))
    print("Buld time: %f" % (stopword_build_time/num_runs))
    print("Classification time: %f" % (stopword_class_time/num_runs))

    print("\nWordlength:")
    print("Total time: %f" % (wordlength_total_time/num_runs))
    print("Buld time: %f" % (wordlength_build_time/num_runs))
    print("Classification time: %f" % (wordlength_class_time/num_runs))

    print("\nHybrid:")
    print("Total time: %f" % (hybrid_total_time/num_runs))
    print("Buld time: %f" % (hybrid_build_time/num_runs))
    print("Classification time: %f" % (hybrid_class_time/num_runs))


if __name__ == "__main__":
    __main__()
