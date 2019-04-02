import numpy as np
import math, re, sys

def get_tocken_count(label):
    tokens = dict()
    file_count = 0
    i = 1
    while True:
        # print("/train/train-{}-{:05d}.txt".format(label, i))
        try:
            with open(".\\train\\train-{}-{:05d}.txt".format(label, i), encoding='latin-1') as test_file:
                txt = test_file.read().lower()
                tokenized = re.split('[^a-zA-Z]', txt)
                for token in tokenized:
                    if token in tokens:
                        tokens[token] += 1
                    else:
                        tokens[token] = 1
                # print(tokenized)
                file_count += 1
        except FileNotFoundError:
            print('Build model for {} is done'.format(label))
            break
        i += 1
    # remove dummy count
    del tokens['']
    return tokens, file_count


def build_model():
    ham_tocken, ham_file_count = get_tocken_count('ham')
    spam_tocken, spam_file_count = get_tocken_count('spam')

    words = set(list(ham_tocken.keys()) + list(spam_tocken.keys()))

    for word in words:
        if word not in ham_tocken.keys():
            ham_tocken[word] = 0
        if word not in spam_tocken.keys():
            spam_tocken[word] = 0

    ham_tocken_smooth = {key: ham_tocken[key] + 0.5 for key in words}
    spam_tocken_smooth = {key: spam_tocken[key] + 0.5 for key in words}
    total_ham = sum(list(ham_tocken_smooth.values()))
    total_spam = sum(list(spam_tocken_smooth.values()))

    ham_model = {key: ham_tocken_smooth[key]/total_ham for key in words}
    spam_model = {key: spam_tocken_smooth[key]/total_spam for key in words}

    linecount = 1
    # now sort token by alhabetic order
    sorted_keys = sorted(words)
    model_file = open("model1.txt", "w+")
    for key in sorted_keys:
        model_file.write('{}  {}  {}  {}  {}  {}\n'.format(
            linecount, key, ham_tocken[key], ham_model[key], spam_tocken[key], spam_model[key]))
        linecount += 1
    return ham_file_count, spam_file_count


def NB_Classifer(ham_file_count, spam_file_count,file):
    output_list=[]
    line_counter = 1
    nb=dict()
    # compute prior
    prior_ham = ham_file_count/(ham_file_count+spam_file_count)
    prior_spam = spam_file_count/(ham_file_count+spam_file_count)
    # find conditional probability for each word in file and build a dictionary
    lines = file.readlines()
    for line in lines:
        words = line.split("  ")
        nb[words[1]] = [words[3], words[5]]

    i = 1
    right_count = 0
    wrong_count = 0
    while True:
        try:
            score_ham = 0
            score_spam = 0
            with open(".\\test\\test-ham-{:05d}.txt".format(i), encoding='latin-1') as test_file:
                txt = test_file.read().lower()
                tokenized = re.split('[^a-zA-Z]',txt)
                for token in tokenized:
                    if token in nb:
                        list = nb[token]
                        # p (w|ham): list[1]
                        score_ham = score_ham + math.log10(float(list[0]))
                        score_spam = score_spam + math.log10(float(list[1]))
                    else:
                        # if the word did not appear in test set, make it 0 for now
                        score_ham = score_ham + 0
                        score_spam = score_spam + 0
                # add prior
                score_ham = score_ham + math.log10(prior_ham)
                score_spam = score_spam + math.log10(prior_spam)
                if score_spam >= score_ham:
                    outputstring = line_counter,"test-ham-{:05d}.txt".format(i),"spam",score_ham,score_spam,"ham","wrong"
                    output_list.append(outputstring)
                    # print(outputstring[0])
                    wrong_count += 1
                else:
                    outputstring=line_counter, "test-ham-{:05d}.txt".format(i), "ham", score_ham, score_spam, "ham", "right"
                    output_list.append(outputstring)
                    # print(outputstring)
                    right_count += 1
                line_counter += 1
        except FileNotFoundError:
            print('test for ham class is done')
            break
        i += 1
    print('accuracy for ham ', right_count / (right_count + wrong_count))

    i = 1
    right_count = 0
    wrong_count = 0
    while True:
        try:
            score_ham = 0
            score_spam = 0
            with open(".\\test\\test-spam-{:05d}.txt".format(i), encoding='latin-1') as test_file:
                txt = test_file.read().lower()
                tokenized = re.split('[^a-zA-Z]',txt)
                for token in tokenized:
                    if token in nb:
                        list = nb[token]
                        # p (w|ham): list[1]
                        score_ham = score_ham + math.log10(float(list[0]))
                        score_spam = score_spam + math.log10(float(list[1]))
                    else:
                        # if the word did not appear in test set, make it 0 for now
                        score_ham = score_ham + 0
                        score_spam = score_spam + 0
                # add prior
                score_ham = score_ham + math.log10(prior_ham)
                score_spam = score_spam + math.log10(prior_spam)
                if score_spam > score_ham:
                    outputstring=line_counter,"test-spam-{:05d}.txt".format(i),"spam",score_ham,score_spam,"spam","right"
                    output_list.append(outputstring)
                    # print(outputstring)
                    right_count += 1
                else:
                    outputstring=line_counter, "test-spam-{:05d}.txt".format(i), "ham", score_ham, score_spam, "spam", "wrong"
                    output_list.append(outputstring)
                    # print(outputstring)
                    wrong_count += 1
                line_counter += 1
        except FileNotFoundError:
            print('test for spam class is done')
            break
        i += 1
    print('accuracy for spam ', right_count/(right_count+wrong_count))

    # make output
    model_file = open("baseline-result.txt", "w+")
    for item in output_list:
        model_file.write('{}  {}  {}  {}  {}  {}  {}\n'.format(item[0],item[1],item[2],item[3],item[4],item[5],item[6]))



def __main__():
    is_build = input('Do you want to build a model? (Y/N)')
    if is_build.lower() == 'y':
        ham_file_count, spam_file_count = build_model()
        print(ham_file_count, spam_file_count)
        # 1000 997
        file = open("model1.txt")
        NB_Classifer(ham_file_count, spam_file_count,file)

if __name__ == "__main__":
    __main__()
