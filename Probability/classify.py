import os
import math

#These first two functions require os operations and so are completed for you
#Completed for you
def load_training_data(vocab, directory):
    """ Create the list of dictionaries """
    top_level = os.listdir(directory)
    dataset = []
    for d in top_level:
        if d.startswith('.'):
            # ignore hidden files
            continue
        if d[-1] == '/':
            label = d[:-1]
            subdir = d
        else:
            label = d
            subdir = d+"/"
        files = os.listdir(directory+subdir)
        for f in files:
            bow = create_bow(vocab, directory+subdir+f)
            dataset.append({'label': label, 'bow': bow})
    return dataset

#Completed for you
def create_vocabulary(directory, cutoff):
    """ Create a vocabulary from the training directory
        return a sorted vocabulary list
    """

    top_level = os.listdir(directory)
    vocab = {}
    for d in top_level:
        if d.startswith('.'):
            # ignore hidden files
            continue
        subdir = d if d[-1] == '/' else d+'/'
        files = os.listdir(directory+subdir)
        for f in files:
            with open(directory+subdir+f,'r', encoding='utf-8') as doc:
                for word in doc:
                    word = word.strip()
                    if not word in vocab and len(word) > 0:
                        vocab[word] = 1
                    elif len(word) > 0:
                        vocab[word] += 1
    return sorted([word for word in vocab if vocab[word] >= cutoff])

#The rest of the functions need modifications ------------------------------
#Needs modifications
def create_bow(vocab, filepath):
    """ Create a single dictionary for the data
        Note: label may be None
    """
    bow = {}
    # TODO: add your code here
    OOV = 0
    document = []

    with open(filepath, encoding='utf-8') as f:
        for line in f:
            document.append(line.strip())
    
    for token in document:
        if token in vocab and token in bow.keys():
            bow[token] += 1
        elif token in vocab:
            bow[token] = 1
        else:
            OOV += 1

    if OOV != 0:
        bow[None] = OOV

    return bow

#Needs modifications
def prior(training_data, label_list):
    """ return the prior probability of the label in the training set
        => frequency of DOCUMENTS
    """

    smooth = 1 # smoothing factor
    logprob = {}
    # TODO: add your code here
    for label in label_list:
        no_files = 0
        for data in training_data:
            if data['label'] == label:
                no_files += 1
        
        logprob[label] = math.log((no_files + smooth)/(len(training_data) + 2), math.e)


    return logprob

#Needs modifications
def p_word_given_label(vocab, training_data, label):
    """ return the class conditional probability of label over all words, with smoothing """

    smooth = 1 # smoothing factor
    word_prob = {}
    # TODO: add your code here

    for word in vocab:
        word_prob[word] = 0

    word_prob[None] = 0

    total_wc = 0
    for data in training_data:
        if data['label'] == label:
            total_wc += sum(data['bow'].values())
            for word in data['bow']:
                if word in word_prob:
                    word_prob[word] += data['bow'][word]
                else:
                    word_prob[None] += data['bow'][word]

    for word in vocab:
        word_prob[word] = math.log((word_prob[word] + smooth*1)/(total_wc + smooth*(len(vocab) + 1)) , math.e)

    word_prob[None] = math.log((word_prob[None] + smooth*1)/(total_wc + smooth*(len(vocab) + 1)) , math.e)

    return word_prob


##################################################################################
#Needs modifications
def train(training_directory, cutoff):
    """ return a dictionary formatted as follows:
            {
             'vocabulary': <the training set vocabulary>,
             'log prior': <the output of prior()>,
             'log p(w|y=2016)': <the output of p_word_given_label() for 2016>,
             'log p(w|y=2020)': <the output of p_word_given_label() for 2020>
            }
    """
    retval = {}
    label_list = [f for f in os.listdir(training_directory) if not f.startswith('.')] # ignore hidden files
    # TODO: add your code here
    retval['vocabulary'] = create_vocabulary(training_directory, cutoff)
    training_data = load_training_data(retval['vocabulary'], training_directory)

    retval['log prior'] = prior(training_data, ['2020', '2016'])
    retval['log p(w|y=2016)'] = p_word_given_label(retval['vocabulary'], training_data, '2016')
    retval['log p(w|y=2020)'] = p_word_given_label(retval['vocabulary'], training_data, '2020')

    return retval

#Needs modifications
def classify(model, filepath):
    """ return a dictionary formatted as follows:
            {
             'predicted y': <'2016' or '2020'>,
             'log p(y=2016|x)': <log probability of 2016 label for the document>,
             'log p(y=2020|x)': <log probability of 2020 label for the document>
            }
    """
    retval = {}
    # TODO: add your code here
    prob_2016 = model['log prior']['2016']
    prob_2020 = model['log prior']['2020']
    file_bow = create_bow(model['vocabulary'], filepath)

    for word in file_bow:
        prob_2016 += model['log p(w|y=2016)'][word] * file_bow[word]

    for word in file_bow:
        prob_2020 += model['log p(w|y=2020)'][word] * file_bow[word]

    retval['log p(y=2016|x)'] = prob_2016
    retval['log p(y=2020|x)'] = prob_2020


    if retval['log p(y=2016|x)'] >= retval['log p(y=2020|x)']:
        retval['predicted y'] = '2016'
    else:
        retval['predicted y'] = '2020'

    return retval


