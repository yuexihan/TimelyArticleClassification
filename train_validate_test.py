import random

f_p = open('positive.data', 'rb')
f_p_train = open('positive.train', 'wb')
f_p_validate = open('positive.validate', 'wb')
f_p_test = open('positive.test', 'wb')

lines = []
for line in f_p:
    lines.append(line)
    if len(lines) == 10:
        random.shuffle(lines)
        for x in lines[:7]:
            f_p_train.write(x)
        for x in lines[7:8]:
            f_p_validate.write(x)
        for x in lines[8:10]:
            f_p_test.write(x)
        lines = []
random.shuffle(lines)
for x in lines[:7]:
    f_p_train.write(x)
for x in lines[7:8]:
    f_p_validate.write(x)
for x in lines[8:]:
    f_p_test.write(x)
lines = []
f_p.close()
f_p_train.close()
f_p_validate.close()
f_p_test.close()

f_n = open('negative.data', 'rb')
f_n_train = open('negative.train', 'wb')
f_n_validate = open('negative.validate', 'wb')
f_n_test = open('negative.test', 'wb')

for line in f_n:
    lines.append(line)
    if len(lines) == 178:
        random.shuffle(lines)
        for x in lines[:7]:
            f_n_train.write(x)
        for x in lines[7:8]:
            f_n_validate.write(x)
        for x in lines[8:10]:
            f_n_test.write(x)
        lines = []
random.shuffle(lines)
for x in lines[:7]:
    f_n_train.write(x)
for x in lines[7:8]:
    f_n_validate.write(x)
for x in lines[8:]:
    f_n_test.write(x)
lines = []
f_n.close()
f_n_train.close()
f_n_validate.close()
f_n_test.close()
