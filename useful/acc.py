import numpy as np
import sys
from collections import defaultdict


def read_prediction(file_name, lables):

    wrong_counts = np.zeros((len(lables), len(lables)))
    label_int = dict()
    for i in range(len(lables)):
        label_int[lables[i]] = i
    lable_dict = dict()
    for i in range(len(lables)):
        lable_dict[i] = lables[i]
    evaluation = defaultdict(list)
    embryo_predict = dict() #for majority
    embryo_predict_max = dict()
    embryo_predict_nb = dict()
    embryo_cnt = dict()
    embryo_label = dict()
    f = open(file_name, 'r')
    for line in f:
        parsed = line.replace('\n', '').split('***') # Split by '***' first
        if len(parsed) < 2: # Check if split was successful, handle unexpected lines
            print(f"Warning: Unexpected line format: {line.strip()}")
            continue # Skip to the next line

        image_name = parsed[0]
        score_str = parsed[1]
        score = []
        prob_strs = score_str.split('>>')[1:] # Split probability string by '>>' and remove the first empty element if any
        if not prob_strs: # Check if probabilities were parsed
            print(f"Warning: No probabilities found for image: {image_name.strip()}")
            continue # Skip to the next line

        for prob_str in prob_strs:
            try:
                score.append(float(prob_str))
            except ValueError:
                print(f"Warning: Could not convert probability to float: {prob_str.strip()} for image: {image_name.strip()}")
                continue # Skip probability if conversion fails


        embryo_name = image_name
        embryo_name = embryo_name.replace('../../Images/test/','') #embryo_name is the name of embryo. all the images are grouped as a single embryo_name.

        ind = embryo_name.rfind('_')
        embryo_name = embryo_name[:ind]
        ind = embryo_name.rfind('_')
        embryo_name = embryo_name[:ind]
        ind = embryo_name.find('_')
        embryo_name = embryo_name[ind+1:]

        if embryo_name not in embryo_predict:
            embryo_predict[embryo_name] = 0
            embryo_cnt[embryo_name] = 0
            embryo_predict_max[embryo_name] = [1]*len(lables) # Initialize with ones, length based on labels
            embryo_predict_nb[embryo_name] = [0]*len(lables) # Initialize with zeros, length based on labels

        current_cnt = embryo_predict_nb[embryo_name]
        predicted_class_index = np.argmax(score)
        current_cnt[predicted_class_index]+=1 # Use predicted_class_index
        embryo_predict_nb[embryo_name] = current_cnt

        current_score = embryo_predict_max[embryo_name]
        for i in range(len(lables)):
            current_score[i] *= score[i] # multiplication
        embryo_predict_max[embryo_name] = current_score

        t_label = -1
        image_name_ind = image_name.rfind('/')
        image_name = image_name[image_name_ind+1:]
        t_flag = True

        for lable in lables:
            if image_name.find(lable + '_') != -1 and t_flag:
                t_label = lable
                t_flag = False
        embryo_label[embryo_name] = t_label
        if t_label == -1:
            print (image_name)
            print('Error: True label not found in image name')
            exit()
        p_label = lable_dict[predicted_class_index] # Use predicted_class_index
        embryo_cnt[embryo_name]+=1
        if p_label == t_label:
            embryo_predict[embryo_name]+=1
        evaluation[image_name] = [t_label, score] # Store score list

    c = 0
    t = 0
    for embryo_name in embryo_predict.keys():
        t += 1
        if embryo_label[embryo_name] == lable_dict[np.argmax(embryo_predict_nb[embryo_name])]:
            c+=1
        else:
            wrong_counts[label_int[embryo_label[embryo_name]], np.argmax(embryo_predict_nb[embryo_name])] += 1
    print("embryo accuracy:", c, t, c/ t)
    print("number of misclassified images from different classes:: ")
    print(wrong_counts)
    return evaluation, lables, lable_dict


def acc(evaluation, lables, lable_dict):
    c = 0
    t = 0
    lables_count = dict()
    for label in lables:
        lables_count[label] = 0
    for image in evaluation:
        t_label = evaluation[image][0]
        p_label = lable_dict[np.argmax(evaluation[image][1])]
        if t_label == p_label:
            c+=1
        lables_count[t_label]+=1
        t+=1

    acc = 0
    if t != 0:
        acc = c * 1. / t
    return c, t, acc


if __name__ == '__main__':
    file_name = 'output.txt'

    lables1 = ['good','poor']
    evaluation1, lables1, lable_dict1 = read_prediction(file_name, lables1)
    print("accuracy per image: ", acc(evaluation1, lables1, lable_dict1))