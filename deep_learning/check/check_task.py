# import numpy as np
# import sklearn.metrics
f = open("responses.txt")


tp = 0
tn = 0
fp = 0
fn = 0

for s in f.readlines():
    s.split(' ')
    if s[0] == 'positive' and s[1] == 'positive':
        tp +=1
    elif s[0] == 'negative' and s[1] == 'negative':
        tn +=1
    elif s[0] == 'positive' and s[1] == 'negative':
        fp +=1
    elif s[0] == 'negative' and s[1] == 'positive':
        fn +=1

precision = tp / (tp + fp)
recall = tp / (tp + fn)

f1 = 2 * (precision*recall)/(precision+recall)
accuracy = (tp + tn) / (tp + tn + fp+ fn)

print(precision, recall, f1, accuracy)



