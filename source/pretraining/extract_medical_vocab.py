import os
import pandas as pd
from glob import glob
import re
import numpy as np
import nltk
from nltk.corpus import stopwords
import itertools


def extract_medical_vocab(data_dir):
    nltk.download('stopwords')
    en_stops = list(set(stopwords.words('english')))
    with open(os.path.join(data_dir, "wordlist.txt"), "r") as wordlist:
        total_wordlist = wordlist.read().splitlines()

    csvs = glob(data_dir+"/*.csv")
    csv_wordlist = []
    for csv in csvs:
        csv_input = pd.read_csv(csv)
        csv_dataframe = pd.DataFrame(csv_input, columns=["Abbreviation/Shorthand"])
        csv_wordlist += [csv_dataframe["Abbreviation/Shorthand"][i] for i in range(len(csv_dataframe["Abbreviation/Shorthand"]))]
    
    final_list_ = total_wordlist + csv_wordlist
    final_list = []
    for word in final_list_:
        if word not in en_stops:
            final_list.append(word)
    return final_list

def sort_length_wise(list):
    new_list = sorted(list, key=lambda x: len(x.split()))
    return new_list

def sort_alphabet_wise(list):
    new_list = sorted(list, key=len)
    return new_list

def word_in_list(found_words, list):
    f_list = found_words.split(" ")
    for word in f_list:
        if word not in list:
            return False
    return True

def get_idx_info(found, sentence):
    new_found = []
    for f in found:
        matches = []
        for match in re.finditer(f, sentence, re.S):
            matches.append(match)
        for i in range(len(matches)):
            start_idx = matches[i].start()
            end_idx = matches[i].end()
            if start_idx != 0 and sentence[start_idx-1] != " ":
                pass
            else:
                info = [f, start_idx, end_idx]
                new_found.append(info)
    return new_found

def remove_overlaps(found, sl, sa, sentence):
    new_found = found.copy()
    for a, b in itertools.combinations(found, 2):
        if a != b:
            if (a[1] <= b[1] and b[1] <= a[2]) or (b[1] <= a[1] and a[1] <= b[2]):#overlaps
                a_sentence = a[0].split(" ")
                b_sentence = b[0].split(" ")
                if len(a_sentence) == len(b_sentence):
                    a_alpha = "".join(a_sentence)
                    b_alpha = "".join(b_sentence)
                    if len(a_alpha) > len(b_alpha):
                        try:
                            new_found.remove(b)
                        except: pass
                    else:
                        try:
                            new_found.remove(a)
                        except: pass
                elif len(a_sentence) > len(b_sentence):
                    try:
                        new_found.remove(b)
                    except: pass
                elif len(a_sentence) < len(b_sentence):
                    try:
                        new_found.remove(a)
                    except: pass
    return new_found

def where_to_mask(sentence, meaningful_list=[], tokenizer=None):
    if tokenizer:
        sentence_list = tokenizer(sentence)
    else:
        sentence_list = sentence.split(" ")
    idx = np.zeros(len(sentence_list))

    if len(meaningful_list) == 0:
        return [int(i) for i in idx]

    found = [] #indicates that there is no meaningful vocab or phrase in the sentence
    for element in meaningful_list:
        if element in sentence:
            found.append(element)
    if found:#if matching vocab/phrase exists
        sl = sort_length_wise(found)
        sa = sort_alphabet_wise(found)    
        info_found = get_idx_info(found, sentence)
        if tokenizer:
            sentence_list = tokenizer(sentence)
        else:
            sentence_list = sentence.split(" ")
        idx = np.zeros(len(sentence_list))
        mask_val = 0
        info_found = remove_overlaps(info_found, sl, sa, sentence)
        for f in info_found:
            f_string = f[0]
            if tokenizer:
                f_list = tokenizer(f_string)
            else:
                f_list = f_string.split(" ")
            if len(f_list) == 1:#this is entity masking
                try:
                    found_idx = sentence_list.index(f_list[0])
                    mask_val += 1
                    idx[found_idx] = mask_val
                except:#if the word is not found,
                #this can occur when we think of the following case:
                #sentence: the abmortal ...
                #key_vocab: abo and abmortal
                #notice that both are technically "in" this sentence but we don't want abo
                    pass
            else:#this is phrase masking
                if word_in_list(f_string, sentence_list):
                    mask_val += 1
                    start_idx = f[1]
                    end_idx = f[2]
                    count = 0
                    for i, word in enumerate(sentence_list):
                        if count == start_idx:
                            idx[i] = mask_val
                        elif count > start_idx and count < end_idx:
                            idx[i] = mask_val
                        count += len(word) + 1
    return [int(i) for i in idx]

if __name__ == "__main__":
    meaningful_list = extract_medical_vocab('data/medical_words')

    
    import pickle
    with open('/home/kaeunkim/NLP-RRED/data/phrase_vocab.pkl', 'rb') as f: 
        meaningful_list = pickle.load(f)

    # sentence = "there is the is a sentence for test abortive there is an ad abmortal adsfsadfasf dfsadfsd fsa 'clips seen"
    sentence2 = "Cardiomediastinal contours are within normal mediastinal contours pleural effusion pneumothorax present acute mediastinal contours"
    # phrase_list = ["there is an", "there is the"]
    # where_to_mask(sentence, phrase_list)
    where_to_mask(sentence2, meaningful_list)