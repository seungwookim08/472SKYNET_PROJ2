import numpy as np
import math, re, sys

def build_model():
    tokens = dict()
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
        i+=1
    return tokens, tokens
def __main__():    
    is_build = input('Do you want to build a model? (Y/N)')
    if is_build.lower() == 'y':
        a, b = build_model()
    # print(a, b)
if __name__ == "__main__":
    __main__()
