import argparse
import math
import pprint

# Create the parser and pass in the arguments as inputs for attribute,training and testing file
parser = argparse.ArgumentParser(description='Decision tree for IDS')
parser.add_argument('--attr',default='ids-attr.txt',
                    help='attribute file')
parser.add_argument('--train',default='ids-train.txt',
                    help='training file')
parser.add_argument('--test',default='ids-test.txt',
                    help='testing file')
args = parser.parse_args()

# create 3 lists for data in the 3 input files(attribute,training and testing file)
attr_list = []
with open(args.attr) as f:
	for i in f:
		attr_list.append(i.replace("\n","").split(" "))

train_list = []
with open(args.train) as f:
	for i in f:
		train_list.append(i.replace("\n","").split(" "))

test_list = []
with open(args.test) as f:
	for i in f:
		test_list.append(i.replace("\n","").split(" "))

# initiate decision tree as a python object/dictionary
Decision_tree = {}

# define entropy function
def Inf(p,n):
	if p!=0 and n!=0:
		return -((p/(p+n))*math.log2(p/(p+n)))-((n/(p+n))*math.log2(n/(p+n)))
	else:
		return 0

# calculate entropy for training dataset
normal = 0
neptune = 0
for i in train_list:
	if i[-1]=="normal":
		normal+=1
	if i[-1]=="neptune":
		neptune+=1
I = Inf(normal,neptune)

# define the information gain as gain = I - remainder (see section 18.4 of Russell and Norvig’s book “Artificial Intelligence, A Modern Approach”)
def gain(k,kk):
	gains = []
	for m,i in enumerate(k):
		remainder = 0
		status = []
		for j in i[1:]:
			st = [j]
			pos = 0
			nep = 0
			for n in kk:
				if n[m]==j:
					if n[-1]=="normal":
						pos+=1
				if n[m]==j:
					if n[-1]=="neptune":
						nep+=1
			I1 = Inf(pos,nep)*((pos+nep)/(normal+neptune))
			remainder += I1
			if pos!=0 and nep==0:
				st.append(1)
			elif pos==0 and nep!=0:
				st.append(2)
			elif pos!=0 and nep!=0:
				st.append(3)
			elif pos==0 and nep==0:
				st.append(4)
			status.append(st)
		gain = I-remainder
		gains.append([gain,status])
	return gains

# define the tree to return the decision tree(recursive function)
def tree(d,attr_list,train_list):
	if len(attr_list)==0:
		return
	p = gain(attr_list[:-1],train_list)
	ind = p.index(max(p))
	d[attr_list[ind][0]] = {}
	for j,i in enumerate(attr_list[ind][1:]):
		if p[ind][1][j][1]==3:
			attr_list1 = attr_list.copy()
			del attr_list1[ind]
			train_list1 = []
			for u in train_list:
				if u[ind]==i:
					train_list1.append(u)
			d[attr_list[ind][0]][i] = tree({},attr_list1,train_list1)
		elif p[ind][1][j][1]==1:
			d[attr_list[ind][0]][i] = {"normal"}
		elif p[ind][1][j][1]==2:
			d[attr_list[ind][0]][i] = {"neptune"}
		# elif p[ind][1][j][1]==4:
		# 	d[attr_list[ind][0]][i] = {"no example observed"}
	return d

# making the tree and fitting it to training dataset and printing the final decision tree
kkk = tree(Decision_tree,attr_list,train_list)
pprint.pprint(kkk)

# defining the prediction function
def pred(dictio,t,mode):
	if dictio=={'normal'} or dictio=={'neptune'}:
		return dictio
	o = None
	d = None
	for i in dictio:
		for k,v in enumerate(attr_list):
			if v[0]==i:
				if mode=="train":
					d = train_list[t][k]
				elif mode=="test":
					d = test_list[t][k]
				o = i
	return pred(dictio[o][d],t,mode)

# calculating training accuracy
acc_train = 0
for m,b in enumerate(train_list):
	y = b[-1]
	y_hat = pred(kkk,m,"train")
	if {y}==y_hat:
		acc_train+=1
print("Training dataset accuracy:",str(acc_train*100/len(train_list))+"%")

# calculating testing accuracy
acc_test = 0
for m,b in enumerate(test_list):
	y = b[-1]
	y_hat = pred(kkk,m,"test")
	if {y}==y_hat:
		acc_test+=1
print("Testing dataset accuracy:",str(acc_test*100/len(test_list))+"%")