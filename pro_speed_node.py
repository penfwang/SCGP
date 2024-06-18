import numpy as np
from deap import base,creator,gp
from ParallelToolbox import ParallelToolbox
from selection import *
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score,confusion_matrix
from sklearn.model_selection import cross_val_score,cross_val_predict,train_test_split,StratifiedKFold
from function_set import add,subtract,multiply,protectedDiv,concat1,concat2,concat3,concat4,Array,analytic_quotient
import sys,saveFile,itertools,math,time,random,gp_restrict,ea_simple_elitism,operator
from sklearn.utils import resample
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from evolutionary_forest.component.evaluation import quick_evaluate


def evaluate(individual, toolbox, data, labels,pset):
    X1 = quick_evaluate(individual, pset, data, prefix='f')
    X1 = np.array(X1)
    X = np.array(X1.reshape(data.shape[0], -1))
    final_scores = []
    skf = StratifiedKFold(n_splits=5)
    whole_pre = []
    for train_index, test_index in skf.split(X, labels):
        x_train_new, x_test_new = X[train_index,:], X[test_index,:]
        y_train_new, y_test_new = labels[train_index], labels[test_index]
        model = KNeighborsClassifier(n_neighbors=5)
        # model = SVC(kernel='linear',random_state=0)  ##random_state=1,kernel='linear'
        model.fit(x_train_new, y_train_new)
        pre_label = model.predict(x_test_new)
        whole_pre.extend(pre_label)
        
        scores = balanced_accuracy_score(y_test_new, pre_label)
        
        # cnf_matrix = confusion_matrix(y_test_new, pre_label)
        # FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
        # TP = np.diag(cnf_matrix)
        # FN = FN.astype(float)
        # TP = TP.astype(float)
        # TPR = TP / (TP + FN)
        # scores= np.mean(TPR)
        
        final_scores.append(scores)
    # return 1*(1-np.mean(final_scores))+ 0.000001*X.shape[1],
    return 1*(1-np.mean(final_scores))+ 0.000001*len(individual),

def init_stats():
    fitness_stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats = tools.MultiStatistics(fitness=fitness_stats)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    return stats


def eval_wrapper(*args, **kwargs):
    return evaluate(*args, **kwargs, toolbox=rd['toolbox'], data=rd['data'], labels=rd['labels'], pset =rd['pset'])


# copies data over from parent process
def init_data(rundata):
    global rd
    rd = rundata



def main(seed,dataset_name):
    random.seed(int(seed))
    ##################################loading the data
    folder1 = '/nesi/project/vuw03334/split_73' + '/' + 'train' + str(dataset_name) + ".npy"
    x_train = np.load(folder1)
    #label_to_see = list(set(x_train[:,0]))
    
    feature_data = x_train[:, 1:]
    variance = feature_data.var(axis = 0)
    index_remove = np.argwhere(variance <= 0.00000001)
    index_remove = [m[0] for m in index_remove]

    if len(index_remove) == 0:
           feature_num = x_train.shape[1] - 1
           training_data = feature_data
    else:
        feature_num = x_train.shape[1] - 1 - len(index_remove)
        training_data = np.delete(feature_data,index_remove,axis=1)
    
    scaler = MinMaxScaler()
    scaler.fit(training_data)
    training_data_norm = scaler.transform(training_data)
    # training_data_norm = preprocessing.normalize(training_data)
    
    pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(list, feature_num), Array, "f")
    pset.context["array"] = np.array
    pset.addPrimitive(add, [list, list], list, name='add')
    pset.addPrimitive(subtract, [list, list], list, name='sub')
    pset.addPrimitive(multiply, [list, list], list, name='mul')
    pset.addPrimitive(protectedDiv, [list, list], list, name='pro')
    # pset.addPrimitive(analytic_quotient, [list, list], list, name='AQ')
    pset.addPrimitive(concat1, [list, list], Array, name='c1')
    pset.addPrimitive(concat2, [list, Array], Array, name='c2')
    pset.addPrimitive(concat3, [Array, Array], Array, name='c3')
    
    weights = (-1.,)
    creator.create("FitnessMin", base.Fitness, weights=weights)
    # set up toolbox
    toolbox = ParallelToolbox()  # base.Toolbox()
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin, pset=pset)
        # toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=6)
    toolbox.register("expr", gp_restrict.genHalfAndHalfMD, pset=pset, min_=1, max_=6)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register("mate", gp.cxOnePoint)
        # toolbox.register("expr_mut", gp.genFull, min_= 1, max_=6)
    toolbox.register("expr_mut", gp_restrict.genFull, min_=1, max_=6)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    #if x_train.shape[1] > 2000:
        #value= 17
    #else:
    value=9
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=value))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=value))

    toolbox.register("select", selElitistAndTournament, tournsize=7, elitism=ELITISM)
    #toolbox.register("select", selElitistAndselEpsilonLexicase, tournsize=7, elitism=ELITISM)
    toolbox.register("evaluate", eval_wrapper)
    
    
    rd['data'] = training_data_norm
    rd['labels'] = x_train[:, 0]
    #################################data information
    rd['num_classes'] = len(set(rd['labels']))
    rd['num_instances'] = rd['data'].shape[0]
    rd['num_features'] = rd['data'].shape[1]

    rd['toolbox'] = toolbox
    rd['pset'] = pset
    
    pop = toolbox.population(n=POP_SIZE)
    stats = init_stats()
    hof = tools.HallOfFame(1)
    pop, logbook,min_fitness,min_pop = ea_simple_elitism.eaSimple(pop, rd, CXPB, MUTPB, ELITISM, NGEN, stats,halloffame=hof, verbose=True)
    best = hof[0]
    return min_fitness,min_pop,best




POP_SIZE = 1024
NGEN = 50
CXPB = 0.8
MUTPB = 0.19
ELITISM = 10

rd = {}
if __name__ == "__main__":
    dataset_name = str(sys.argv[1])
    seed = str(sys.argv[2])
    #dataset_name = 'dataSet_ion'
    #seed = str(1)
    random.seed(int(seed))
    start = time.time()
    min_fitness,min_pop,p_one= main(seed, dataset_name)
    end = time.time()
    running_time = end - start
    saveFile.save_individual(seed, dataset_name, p_one)
    saveFile.save_pop(seed, dataset_name, min_pop)
    saveFile.saveAllfeature7(seed, dataset_name, running_time)
    saveFile.saveAllfeature6(seed, dataset_name, min_fitness)

