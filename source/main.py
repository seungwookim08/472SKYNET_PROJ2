import numpy as np
import math, re, sys
def build_model():
    tokens = dict()
    ham_file_count = 0
    spam_file_count = 0
    i = 1
    while True:
        print("/train/train-ham-{:05d}.txt".format(i))
        try:
            with open(".\\train\\train-ham-{:05d}.txt".format(i)) as test_file:
                txt = test_file.read().lower()
                tokenized = re.split('[^a-zA-Z]',txt)
                for token in tokenized:
                    if token in tokens:
                        tokens[token]['ham'] += 1
                    else:
                        tokens[token] = {'ham': 1}
                # print(tokenized)
                ham_file_count+=1
        except FileNotFoundError:
            print('Build model for ham is done')
            break
        except UnicodeDecodeError:
            # utf-8 code can't decode, then try with byte code
            with open(".\\train\\train-ham-{:05d}.txt".format(i), 'rb') as test_file:
                txt = test_file.read()
                txt = txt.decode('ISO-8859-1').lower()
                tokenized = re.split('[^a-zA-Z]',txt)
                for token in tokenized:
                    if token in tokens:
                        tokens[token]['ham'] += 1
                    else:
                        tokens[token] = {'ham': 1}
                # print(tokenized)
                ham_file_count += 1
        i += 1
    i = 1
    while True:
        print("/train/train-spam-{:05d}.txt".format(i))
        try:
            with open(".\\train\\train-spam-{:05d}.txt".format(i)) as test_file:
                txt = test_file.read().lower()
                tokenized = re.split('[^a-zA-Z]',txt)
                for token in tokenized:
                    if token in tokens:
                        if len(tokens[token]) > 1:
                            tokens[token]['spam'] += 1
                    else:
                        tokens[token] = {'spam': 1}
                # print(tokenized)
                spam_file_count+=1
        except FileNotFoundError:
            print('Build model for spam is done')
            break
        except UnicodeDecodeError:
            # utf-8 code can't decode, then try with byte code
            with open(".\\train\\train-spam-{:05d}.txt".format(i), 'rb') as test_file:
                txt = test_file.read()
                txt = txt.decode('ISO-8859-1').lower()
                tokenized = re.split('[^a-zA-Z]',txt)
                for token in tokenized:
                    if token in tokens:
                        if len(tokens[token]) > 1:
                            tokens[token]['spam'] += 1
                    else:
                        tokens[token] = {'spam': 1}
                # print(tokenized)
                spam_file_count += 1
        i += 1
    
    # To have a deep copy for smoothed token, we need both original frequency and smoothed probability
    smoothed_tokens = dict()
    for key, value in tokens.items():
        for key2, value2 in tokens[key].items():
            smoothed_tokens.update({key: {'{}'.format(key2): value2}})
    
    total_ham = 0
    total_spam = 0
    # Applying smoothed_tokens
    for key, value in smoothed_tokens.items():
        if 'ham' in value:
            value['ham'] += 0.5
        else:
            value['ham'] = 0.5
            tokens[key].update(dict(ham=0))
        if 'spam' in value:
            value['spam'] += 0.5
        else:
            value['spam'] = 0.5
            tokens[key].update(dict(spam=0))
        total_ham += value['ham']
        total_spam += value['spam']
    
    i = 1
    # now sort token by alhabetic order
    sorted_keys = sorted(tokens)
    model_file = open("model1.txt".format(i), "w+")
    for key in sorted_keys:
        model_file.write('{}  {}  {}  {}  {}  {}\n'.format(
               i, key, tokens[key]['ham'], smoothed_tokens[key]['ham']/total_ham, tokens[key]['spam'], smoothed_tokens[key]['spam']/total_spam ))
        i=i+1
    return tokens, tokens, ham_file_count, spam_file_count
def __main__():
    is_build = input('Do you want to build a model? (Y/N)')
    if is_build.lower() == 'y':
        a, b, ham_file_count, spam_file_count = build_model()
        print(ham_file_count, spam_file_count)
        # 1000 997
        file = open("model1.txt")

if __name__ == "__main__":
    __main__()
def Classifer(ham_file_count, spam_file_count,file):
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
    while True:
        try:
            score_ham = 0
            score_spam = 0
            with open(".\\test\\test-ham-{:05d}.txt".format(i)) as test_file:
                txt = test_file.read().lower()
                tokenized = re.split('[^a-zA-Z]',txt)
                for token in tokenized:
                    if token in nb:
                        list = nb[token]
                        # p (w|ham): list[1]
                        score_ham = score_ham + math.log10(list[0])
                        score_spam = score_spam + math.log10(list[1])
                    else:
                        # if the word did not appear in test set, make it 0 for now
                        score_ham = score_ham + 0
                        score_spam = score_spam + 0
                # add prior
                score_ham = score_ham + math.log10(prior_ham)
                score_spam = score_spam + math.log10(prior_spam)
                if score_spam >= score_ham:
                    print(line_counter,"test-ham-{:05d}.txt".format(i),"spam",score_ham,score_spam,"ham","wrong")
                else:
                    print(line_counter, "test-ham-{:05d}.txt".format(i), "ham", score_ham, score_spam, "ham", "right")

        except FileNotFoundError:
            print('test is done')
            break
        except UnicodeDecodeError:
            # utf-8 code can't decode, then try with byte code
            score_ham = 0
            score_spam = 0
            with open(".\\train\\train-ham-{:05d}.txt".format(i), 'rb') as test_file:
                txt = test_file.read()
                txt = txt.decode('ISO-8859-1').lower()
                tokenized = re.split('[^a-zA-Z]',txt)
                for token in tokenized:
                    if token in nb:
                        list = nb[token]
                        # p (w|ham): list[1]
                        score_ham = score_ham + math.log10(list[0])
                        score_spam = score_spam + math.log10(list[1])
                    else:
                        # if the word did not appear in test set, make it 0 for now
                        score_ham = score_ham + 0
                        score_spam = score_spam + 0
                # add prior
                score_ham = score_ham + math.log10(prior_ham)
                score_spam = score_spam + math.log10(prior_spam)
                if score_spam >= score_ham:
                    print(line_counter,"test-ham-{:05d}.txt".format(i),"spam",score_ham,score_spam,"ham","wrong")
                else:
                    print(line_counter, "test-ham-{:05d}.txt".format(i), "ham", score_ham, score_spam, "ham", "right")
        i += 1

    i = 1
    while True:
        try:
            score_ham = 0
            score_spam = 0
            with open(".\\test\\test-spam-{:05d}.txt".format(i)) as test_file:
                txt = test_file.read().lower()
                tokenized = re.split('[^a-zA-Z]',txt)
                for token in tokenized:
                    if token in nb:
                        list = nb[token]
                        # p (w|ham): list[1]
                        score_ham = score_ham + math.log10(list[0])
                        score_spam = score_spam + math.log10(list[1])
                    else:
                        # if the word did not appear in test set, make it 0 for now
                        score_ham = score_ham + 0
                        score_spam = score_spam + 0
                # add prior
                score_ham = score_ham + math.log10(prior_ham)
                score_spam = score_spam + math.log10(prior_spam)
                if score_spam > score_ham:
                    print(line_counter,"test-ham-{:05d}.txt".format(i),"spam",score_ham,score_spam,"spam","right")
                else:
                    print(line_counter, "test-ham-{:05d}.txt".format(i), "ham", score_ham, score_spam, "spam", "wrong")
        except FileNotFoundError:
            print('test is done')
            break
        except UnicodeDecodeError:
            # utf-8 code can't decode, then try with byte code
            with open(".\\train\\train-spam-{:05d}.txt".format(i), 'rb') as test_file:
                txt = test_file.read()
                txt = txt.decode('ISO-8859-1').lower()
                tokenized = re.split('[^a-zA-Z]', txt)
                for token in tokenized:
                    if token in nb:
                        list = nb[token]
                        # p (w|ham): list[1]
                        score_ham = score_ham + math.log10(list[0])
                        score_spam = score_spam + math.log10(list[1])
                    else:
                        # if the word did not appear in test set, make it 0 for now
                        score_ham = score_ham + 0
                        score_spam = score_spam + 0
                # add prior
                score_ham = score_ham + math.log10(prior_ham)
                score_spam = score_spam + math.log10(prior_spam)
                if score_spam > score_ham:
                    print(line_counter,"test-ham-{:05d}.txt".format(i),"spam",score_ham,score_spam,"spam","right")
                else:
                    print(line_counter, "test-ham-{:05d}.txt".format(i), "ham", score_ham, score_spam, "spam", "wrong")

        i += 1
# change