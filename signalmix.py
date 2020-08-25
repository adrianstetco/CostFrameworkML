import pandas as pd
import numpy as np
import random as random
from numpy.random import normal
import re
import math
import seaborn as sns
from scipy.stats import powerlaw
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import History
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.layers import Dense, Activation, Flatten, BatchNormalization, Input
from sklearn.model_selection import cross_val_score
import matplotlib._color_data as mcd
from sklearn.utils import shuffle
from matplotlib.ticker import MaxNLocator

sns.set_style("whitegrid", {'axes.grid' : False})

def knapsack_greedy(accuracies, cost, budget):
    value = accuracies/cost   
    index_sorted = np.flip(np.argsort(value))
    solution_cost=0
    solution_vector = np.zeros(len(value))
    for i in range(len(value)):
        if (solution_cost+cost[index_sorted[i]])>budget:
            break
        else:
            solution_cost+= cost[index_sorted[i]]
            solution_vector[index_sorted[i]]=1
    solution_accuracy= sum(accuracies[solution_vector.astype(bool)])
    return solution_vector.astype(int), solution_cost, solution_accuracy

def knapsack_dynamic(accuracies, costs, budget):
    if budget==0:
        return 0,0,0
    mat = np.zeros([len(costs), budget+1])
    solution = np.zeros(len(costs))
    mat.fill(np.nan)
    def knapsack_new(n, b):
        b= int(b)        
        if np.isnan(mat[n-1, b])==False: return mat[n-1, b]
        if n==0 or b==0:
            result= 0
        elif costs[n-1]<=b:
            aux1 = knapsack_new(n-1, b)
            aux2 = accuracies[n-1] + knapsack_new(n-1, b - costs[n-1])
            result = max(aux1, aux2)
        else:
            result = knapsack_new(n-1, b) 
        mat[n-1, b]= result
        return result
    total_accuracy = knapsack_new(len(costs), budget)
    #print(pd.DataFrame(mat))
    i=len(costs)-1;w=budget
    while (i>=0 and w>0):
        if (mat[i,w] != mat[i-1, w] ):
            solution[i]=1
            w = int(w - costs[i])
            i=i-1            
        else:
            i = i-1 
    return solution, sum(costs * solution), sum(accuracies * solution)



# Classifcation Neural Network
def model(feature_size, output_size, act, learning_rate):    
    inputs = Input(shape=(feature_size,))
    # layer 1
    dense1 = Dense(feature_size*2)(inputs)
    bn1   =  BatchNormalization()(dense1)
    act1  =  Activation(act)(bn1)    
    # layer 2
    dense2 = Dense(feature_size)(act1)
    bn2   =  BatchNormalization()(dense2)
    act2  =  Activation(act)(bn2)    
    # layer 3
    dense3 = Dense(5)(act2)
    bn3   =  BatchNormalization()(dense3)
    act3  =  Activation(act)(bn3)    
    # output layer, softmax activation
    dense = Dense(output_size, activation="softmax")(act3)
    opt =   Adam(lr=learning_rate, beta_1=0.5, beta_2=0.999, epsilon=1e-8)
    model = Model(inputs=inputs, outputs=dense)
    model.compile(loss='categorical_crossentropy', optimizer=opt,  metrics=['accuracy'])
    print(model.summary())
    return model
    

# Generats a vector which sums to a given value and comes from a uniform or normal distribution
def generate_weights(nr_signals, sumTo=1, distribution = "uniform", powerlaw_param=0.5):
    if   distribution=="normal":
        r = [normal(0, 3) for i in range(0, nr_signals)]
    elif distribution=="uniform":
        r = np.zeros(nr_signals) + 1/nr_signals
    elif distribution=="powerlaw":
        r = powerlaw.rvs(powerlaw_param, size=nr_signals)
        r = [round(i/sum(r),3) *sumTo for i in r ]
    s = sum(r)
    return [ i/s *sumTo for i in r ]

# parameters for generating data
features_nr   = 20
features_size = 10000
categorical_var_values = 3 
total_cost   = 10000

#specify the weighting on the mixture of sines (features), how the target variable is computed
#powerlaw.pdf(x, a) = a * x**(a-1)
mix_weights    = generate_weights(features_nr, 1, "powerlaw", powerlaw_param=0.5)
#specify the costsof aquiring the features 
costs_uniform      = generate_weights(features_nr, total_cost, "uniform")
costs_powerlaw      = generate_weights(features_nr, total_cost, "powerlaw")
# generate dataframe of sines + target variable, weighted sum of sines
random.seed(1)
df = pd.DataFrame()
for i in range(features_nr):
    amplitude = normal(0, 1)
    period    = 2*np.pi/normal(0.5, 0.1)
    shift     = normal(0, 1)
    vertical  = normal(0, 0.5)
    # generates a Pandas(pd) series from sin(period*(x+shift))
    y = pd.Series([math.sin(period*(i+shift))+vertical for i in np.linspace(0,10,features_size)])
    y.name= "S"+ str(i)
    df = pd.concat([df, pd.Series(y)], axis=1)
dependent_series = (df*mix_weights).sum(axis=1)
dependent_series_cut = pd.cut(dependent_series, categorical_var_values).astype(str)
dependent_series.name = "Target Variable"
df = pd.concat([df, dependent_series_cut], axis=1)
#df = pd.concat([df, dependent_series], axis=1)
 
encoder = LabelBinarizer()
train_x, test_x, train_y, test_y = train_test_split(df.iloc[:,0:features_nr], df.iloc[:,-1], test_size=0.2)
train_y = encoder.fit_transform(train_y)
test_y  = encoder.transform(test_y)


# Train a classification network on all the features
act = "relu";
learning_rate = 0.001;
epochs = 512;batch_size = 256;
nn      = model(features_nr, categorical_var_values,   act=act, learning_rate= learning_rate)
history = nn.fit(train_x, train_y, epochs=epochs,  batch_size=batch_size)
print(nn.evaluate(test_x, test_y))

#fig, (ax1, ax2) = plt.subplots(1, 2)
#pd.Series(nn.history['acc'], name="Accuracy").plot(legend=True, ax=ax1)
#pd.Series(nn.history['loss'], name="Loss").plot(legend=True, ax=ax2)

total_acc = history.history["accuracy"][-1]*100
alpha=0.5
feature_accuracies = {}
train_x2 = train_x.copy()
test_x2  = test_x.copy()
for i in range(features_nr):
    #randomize across training and test sets
    print("Feature" + str(i))
    # random by sampling from normal distribution
    #train_x2.iloc[:,i]= np.random.normal(0, 0.1, train_x2.iloc[:,i].shape[0])
    #test_x2.iloc[:,i]= np.random.normal(0, 0.1, test_x2.iloc[:,i].shape[0])
    #random by shuffling the vectors
    train_x2.iloc[:,i]= np.array(shuffle(train_x2.iloc[:,i]))
    test_x2.iloc[:,i] = np.array(shuffle(test_x2.iloc[:,i]))   
    train_accuracy = nn.evaluate(train_x2, train_y)[1]*100
    test_accuracy  = nn.evaluate(test_x2, test_y)[1]*100
    print(str(train_accuracy) + " " + str(test_accuracy)) #debug
    # accuracy ratio computation, alpha is a parameter controlling the accuracy on training vs testing
    # accuracy_ratio = ((a*train_accuracy) + ((1-a)*test_accuracy) * 1/features_nr) / total_accuracy 
    feature_accuracies[train_x2.iloc[:,i].name]= total_acc - (alpha*train_accuracy + (1-alpha)*test_accuracy)
    train_x2 = train_x.copy()
    test_x2  = test_x.copy()

print(feature_accuracies)
fig, ax1 = plt.subplots(1)
v = list(feature_accuracies.values())
k = list(feature_accuracies.keys())
v_one = [i/sum(v) for i in v] #make them sum to one to be one the same scale
p = pd.DataFrame(list(zip(v_one,mix_weights, costs_uniform)), columns=["A","W","Cost"], index=k)
ax1.plot(p.A, label="Feature Importance", marker="o")
ax1.plot(p.W, label="MixWeights", marker="o")
ax1.legend()
plt.xticks(rotation=90)

#knapsack
budget=1000
solution_vector, solution_cost, solution_accuracy = knapsack_greedy(p.A, costs_uniform, budget)
print("Solution includes features {0} and costs {1} having total accuracy {2}".format(solution_vector, solution_cost, solution_accuracy))


budgets = np.linspace(0, sum(costs_uniform), 500)

accuracy_for_budget_greedy = np.zeros(len(budgets)) 
accuracy_for_budget_greedy2 = np.zeros(len(budgets)) 
accuracy_for_budget_dynamic = np.zeros(len(budgets)) 
accuracy_for_budget_dynamic2 = np.zeros(len(budgets)) 

k=0
for i in budgets:    
    print(str(i)+"/"+str(len(budgets)))
    #cost 1
    _, _, solution_accuracy_greedy = knapsack_greedy(p.A, costs_uniform, int(i))
    accuracy_for_budget_greedy[k] = solution_accuracy_greedy
    _, _, solution_accuracy_dynamic = knapsack_dynamic(p.A, costs_uniform, int(i))
    accuracy_for_budget_dynamic[k] = solution_accuracy_dynamic
    
    #cost 2
    _, _, solution_accuracy_greedy = knapsack_greedy(p.A, costs_powerlaw, int(i))
    accuracy_for_budget_greedy2[k] = solution_accuracy_greedy    
    _, _, solution_accuracy_dynamic = knapsack_dynamic(p.A, costs_powerlaw, int(i))
    accuracy_for_budget_dynamic2[k] = solution_accuracy_dynamic
    k=k+1

fig, ax1 = plt.subplots(1)
df = pd.DataFrame([np.asarray(budgets), np.asarray(accuracy_for_budget_greedy), np.asarray(accuracy_for_budget_greedy2), np.asarray(accuracy_for_budget_dynamic), 
                   np.asarray(accuracy_for_budget_dynamic2)], index=["Budget","Accuracy_Cost1_Greedy", "Accuracy_Cost2_Greedy","Accuracy_Cost1_Dynamic", "Accuracy_Cost2_Dynamic"]).T

ax1.plot(df.iloc[:,0],df.iloc[:,1])
    
    
    sns.lineplot(x="Budget", y="Accuracy_Cost1_Greedy", data=df, ax=ax1, label = "Cost1 Greedy", color="blue")
sns.lineplot(x="Budget", y="Accuracy_Cost2_Greedy", data=df, ax=ax1, label = "Cost2 Greedy", color="red")

sns.lineplot(x="Budget", y="Accuracy_Cost1_Dynamic",data=df, ax=ax1, label = "Cost1 Dynamic", color="blue", hue="logic", style="logic", markers=["o", "o"])
sns.lineplot(x="Budget", y="Accuracy_Cost2_Dynamic", data=df, ax=ax1, label = "Cost2 Dynamic", color="red",  hue="logic", style="logic", markers=["o", "o"])


#ax1.xaxis.label.set_size(15)
#ax1.yaxis.label.set_size(15)
ax1.set_ylabel("Accuracy")
ax1.legend()





## plot, skip this part
#axes = df.iloc[0:1000,0:10].plot(subplots=False, legend=False)
axes = dependent_series.plot(legend=True, markersize=1)
colors = np.random.choice(np.array(list(mcd.XKCD_COLORS.values())), categorical_var_values)
colors = dict(zip(dependent_series_cut.unique(), colors))
for i in range(len(dependent_series)):
 plt.plot(i, dependent_series[i], ".", color=colors[df[i]], markersize=4)
plt.plot(l)
plt.plot(mix_weights_powerlaw)
plt.plot(np.sum(nn.layers[1].get_weights()[0], axis=1))
