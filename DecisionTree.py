import pandas as pd
import numpy as np


def entropy(p):
    return -p*np.log(p) - (1-p) * np.log(1-p)

def choosing_parameter_info_gain(x,y):
    entropy_list = []
    for i in range(4):
        avg = np.mean(x[:,i])
        left_class0,left_class1,right_class0,right_class1 = [],[],[],[]
        for j in range(len(x[:,i])):
            if x[j,i]<avg and y[j] == 0 :
                left_class0.append(j)
            elif x[j,i]<avg and y[j] == 1 :
                left_class1.append(j)
            elif x[j,i]>=avg and y[j] == 0 :
                right_class0.append(j)
            elif x[j,i]>=avg and y[j] == 1 :
                right_class1.append(j)
        lef_len = len(left_class0)+len(left_class1)
        right_len = len(right_class0)+len(right_class1)
        total_len = lef_len + right_len
        entropy_cur = lef_len/total_len * entropy(len(left_class0)/lef_len) + right_len/total_len * entropy(len(right_class0)/right_len)
        entropy_list.append(entropy_cur)
    max_index = entropy_list.index(max(entropy_list))
    avg = np.mean(x[:, max_index])
    return max_index,x[x[:,max_index]<avg,:],y[x[:,max_index]<avg],x[x[:,max_index]>=avg,:],y[x[:,max_index]>=avg],avg


def choosing_parameter_gain_ratio(x,y):
    zeros,ones = 0 ,0
    for i in range(y.shape[0]):
        if y[i] == 0:
            zeros += 1
        else :
            ones += 1
    entropy_before = entropy(zeros/(zeros+ones))
    entropy_list = []
    for i in range(4):
        avg = np.mean(x[:,i])
        left_class0,left_class1,right_class0,right_class1 = [],[],[],[]
        for j in range(len(x[:,i])):
            if x[j,i]<avg and y[j] == 0 :
                left_class0.append(j)
            elif x[j,i]<avg and y[j] == 1 :
                left_class1.append(j)
            elif x[j,i]>=avg and y[j] == 0 :
                right_class0.append(j)
            elif x[j,i]>=avg and y[j] == 1 :
                right_class1.append(j)
        lef_len = len(left_class0)+len(left_class1)
        right_len = len(right_class0)+len(right_class1)
        total_len = lef_len + right_len
        entropy_cur = lef_len/total_len * entropy(len(left_class0)/lef_len) + right_len/total_len * entropy(len(right_class0)/right_len)
        entropy_list.append((entropy_before-entropy_cur)/entropy_cur)
    max_index = entropy_list.index(max(entropy_list))
    avg = np.mean(x[:, max_index])
    return max_index,x[x[:,max_index]<avg,:],y[x[:,max_index]<avg],x[x[:,max_index]>=avg,:],y[x[:,max_index]>=avg],avg


## DATA PREPARATION
df = pd.read_csv("iris.csv")
df.head()
data = df.to_numpy()
data[data[:,4]=='Iris-setosa',4] = 1
data[data[:,4]=='Iris-versicolor',4] = 0
labels = data[:,4]
data = np.delete(data,4,axis=1)
x_train = np.concatenate((data[:40,:],data[50:90,:]),axis=0)
x_test = np.concatenate((data[40:50,:],data[90:,:]),axis=0)
y_train = np.concatenate((labels[:40],labels[50:90]),axis=0)
y_test = np.concatenate((labels[40:50],labels[90:]),axis=0)


## Information Gain Decision Tree
top_condition, x_l, y_l, x_r,y_r,top_avg = choosing_parameter_info_gain(x_train,y_train)
l_condition,x_ll,y_ll,x_lr,y_lr,l_avg = choosing_parameter_info_gain(x_l, y_l)
r_condition,x_rl,y_rl,x_rr,y_rr,r_avg = choosing_parameter_info_gain(x_r, y_r)
ll_condition,x_lll,y_lll,x_llr,y_llr,ll_avg = choosing_parameter_info_gain(x_ll, y_ll)
lr_condition,x_lrl,y_lrl,x_lrr,y_lrr,lr_avg = choosing_parameter_info_gain(x_lr, y_lr)
rl_condition,x_rll,y_rll,x_rlr,y_rlr,rl_avg = choosing_parameter_info_gain(x_rl, y_rl)
rr_condition,x_rrl,y_rrl,x_rrr,y_rrr,rr_avg = choosing_parameter_info_gain(x_rr, y_rr)

lll_answer = 1 if np.sum(y_lll)>np.shape(y_lll)[0]/2 else 0
llr_answer = 1 if np.sum(y_llr)>np.shape(y_llr)[0]/2 else 0
lrl_answer = 1 if np.sum(y_lrl)>np.shape(y_lrl)[0]/2 else 0
lrr_answer = 1 if np.sum(y_lrr)>np.shape(y_lrr)[0]/2 else 0
rll_answer = 1 if np.sum(y_rll)>np.shape(y_rll)[0]/2 else 0
rlr_answer = 1 if np.sum(y_rlr)>np.shape(y_rlr)[0]/2 else 0
rrl_answer = 1 if np.sum(y_rrl)>np.shape(y_rrl)[0]/2 else 0
rrr_answer = 1 if np.sum(y_rrr)>np.shape(y_rrr)[0]/2 else 0


correct_answers = 0
for i in range(20):
    x = x_test[i,:]
    if x[top_condition] > top_avg :
        if x[r_condition] > r_avg :
            if x[rr_condition] > rr_avg :
                estimate = rrr_answer
            else:
                estimate = rrl_answer
        else:
            if x[rl_condition] > rl_avg:
                estimate = rlr_answer
            else:
                estimate = rll_answer
    else:
        if x[l_condition]> l_avg:
            if x[lr_condition] > lr_avg:
                estimate = lrr_answer
            else :
                estimate = lrl_answer
        else:
            if x[ll_condition]>ll_avg:
                estimate = llr_answer
            else:
                estimate = lll_answer
    if y_test[i] == estimate :
        correct_answers += 1

print("Information gain accuracy : {}".format(correct_answers/20))



## Gain Ration Decision Tree

top_condition, x_l, y_l, x_r,y_r,top_avg = choosing_parameter_gain_ratio(x_train,y_train)
l_condition,x_ll,y_ll,x_lr,y_lr,l_avg = choosing_parameter_gain_ratio(x_l, y_l)
r_condition,x_rl,y_rl,x_rr,y_rr,r_avg = choosing_parameter_gain_ratio(x_r, y_r)
ll_condition,x_lll,y_lll,x_llr,y_llr,ll_avg = choosing_parameter_gain_ratio(x_ll, y_ll)
lr_condition,x_lrl,y_lrl,x_lrr,y_lrr,lr_avg = choosing_parameter_gain_ratio(x_lr, y_lr)
rl_condition,x_rll,y_rll,x_rlr,y_rlr,rl_avg = choosing_parameter_gain_ratio(x_rl, y_rl)
rr_condition,x_rrl,y_rrl,x_rrr,y_rrr,rr_avg = choosing_parameter_gain_ratio(x_rr, y_rr)

lll_answer = 1 if np.sum(y_lll)>np.shape(y_lll)[0]/2 else 0
llr_answer = 1 if np.sum(y_llr)>np.shape(y_llr)[0]/2 else 0
lrl_answer = 1 if np.sum(y_lrl)>np.shape(y_lrl)[0]/2 else 0
lrr_answer = 1 if np.sum(y_lrr)>np.shape(y_lrr)[0]/2 else 0
rll_answer = 1 if np.sum(y_rll)>np.shape(y_rll)[0]/2 else 0
rlr_answer = 1 if np.sum(y_rlr)>np.shape(y_rlr)[0]/2 else 0
rrl_answer = 1 if np.sum(y_rrl)>np.shape(y_rrl)[0]/2 else 0
rrr_answer = 1 if np.sum(y_rrr)>np.shape(y_rrr)[0]/2 else 0


correct_answers = 0
for i in range(20):
    x = x_test[i,:]
    if x[top_condition] > top_avg :
        if x[r_condition] > r_avg :
            if x[rr_condition] > rr_avg :
                estimate = rrr_answer
            else:
                estimate = rrl_answer
        else:
            if x[rl_condition] > rl_avg:
                estimate = rlr_answer
            else:
                estimate = rll_answer
    else:
        if x[l_condition]> l_avg:
            if x[lr_condition] > lr_avg:
                estimate = lrr_answer
            else :
                estimate = lrl_answer
        else:
            if x[ll_condition]>ll_avg:
                estimate = llr_answer
            else:
                estimate = lll_answer
    if y_test[i] == estimate :
        correct_answers += 1

print("Gain Ration accuracy : {}".format(correct_answers/20))