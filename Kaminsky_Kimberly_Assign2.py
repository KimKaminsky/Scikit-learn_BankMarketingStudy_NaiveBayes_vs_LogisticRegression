# Kimberly Kaminsky - Assign #2 
# Evaluating Classification Methods

####################
# Import Libraries #
####################

# import base packages into the namespace for this program
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages # Allows printing multiple graphs to pdf
import PyPDF2 as pp                      # Allows pdf file manipulation

# cross-validation scoring code adapted from Scikit Learn documentation
from sklearn.metrics import roc_auc_score

# specify the set of classifiers being evaluated
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
names = ["Naive_Bayes", "Logistic_Regression"]
classifiers = [BernoulliNB(alpha=1.0, binarize=0.5, 
                           class_prior = [0.5, 0.5], fit_prior=False), 
               LogisticRegression()]
               
                                           
# specify the k-fold cross-validation design
from sklearn.model_selection import KFold

# create test train split
from sklearn.cross_validation import train_test_split

#################### 
# Define constants #
####################

# ten-fold cross-validation employed here
N_FOLDS = 10

# seed value for random number generators to obtain reproducible results
RANDOM_SEED = 11

################################
# Functions for use in program #
################################

# function to clear output console
def clear():
    print("\033[H\033[J")
    
# map target from probability cutoff specified
def prob_to_pred(x, cutoff):
    if(x > cutoff):
        return(1)
    else:
        return(0)
        
def evaluate_classifier(predicted, observed):
    import pandas as pd 
    if(len(predicted) != len(observed)):
        print('\nevaluate_classifier error:',\
             ' predicted and observed must be the same length\n')
        return(None) 
    if(len(set(predicted)) != 2):
        print('\nevaluate_classifier error:',\
              ' predicted must be binary\n')
        return(None)          
    if(len(set(observed)) != 2):
        print('\nevaluate_classifier error:',\
              ' observed must be binary\n')
        return(None)          

    predicted_data = predicted
    observed_data = observed
    input_data = {'predicted': predicted_data,'observed':observed_data}
    input_data_frame = pd.DataFrame(input_data)
    
    cmat = pd.crosstab(input_data_frame['predicted'],\
        input_data_frame['observed']) 
    a = float(cmat.loc[0,0])
    b = float(cmat.loc[0,1])
    c = float(cmat.loc[1,0]) 
    d = float(cmat.loc[1,1])
    n = a + b + c + d
    predictive_accuracy = (a + d)/n
    true_positive_rate = a / (a + c)
    false_positive_rate = b / (b + d)
    precision = a / (a + b)
    specificity = 1 - false_positive_rate   
    expected_accuracy = (((a + b)*(a + c)) + ((b + d)*(c + d)))/(n * n)
    kappa = (predictive_accuracy - expected_accuracy)\
       /(1 - expected_accuracy)   
    return(a, b, c, d, predictive_accuracy, true_positive_rate, specificity,\
        false_positive_rate, precision, expected_accuracy, kappa)

#########################
# Import and clean data #
#########################

# Import the dataset
datapath = os.path.join("D:/","Kim MSPA", "Predict 422", "Assignments", "Assignment2", "")
bank = pd.read_csv(datapath + 'bank.csv', sep = ';')  

# examine the shape of original input data
print(bank.shape)  #4521 observations with 17 attributes

# drop observations with missing data, if any
bank2 = bank.dropna()
# examine the shape of input data after dropping missing data
print(bank2.shape)  # No missing data

# look at the list of column names, note that y is the response
list(bank.columns.values)

# look at the beginning of the DataFrame
bank.head()

# mapping function to convert text no/yes to integer 0/1
convert_to_binary = {'no' : 0, 'yes' : 1}

# define binary variable for having credit in default
default = bank['default'].map(convert_to_binary)

# define binary variable for having a mortgage or housing loan
housing = bank['housing'].map(convert_to_binary)

# define binary variable for having a personal loan
loan = bank['loan'].map(convert_to_binary)

# define response variable to use in the model
response = bank['response'].map(convert_to_binary)

# Create a dataframe for analysis later in program
d_data = pd.DataFrame(default)
h_data = pd.DataFrame(housing)
l_data = pd.DataFrame(loan)
r_data = pd.DataFrame(response)
d_data = d_data.join(h_data)
d_data = d_data.join(l_data)
bank_data = d_data.join(r_data)


# gather three explanatory variables and response into a numpy array 
# here we use .T to obtain the transpose for the structure we want
model_data = np.array([np.array(default), np.array(housing), np.array(loan), 
    np.array(response)]).T

# examine the shape of model_data, which we will use in subsequent modeling
print(model_data.shape)


######################################
# Create Frequency/Proportion Tables #
######################################

# Turn off interactive mode since a large number of plots are generated
# Plots will be saved off in pdf files
mpl.is_interactive()
plt.ioff()

# Default Frequency and Proportion Table
defaultCT = pd.crosstab(index=default,columns=response, margins=True)
defaultCT.columns = ["No_Deposit", "Deposit", "Row_Total"]
defaultCT.index = ["No_Default", "Default", "Col_Total"]
defaultProp = defaultCT/defaultCT.loc["Col_Total","Row_Total"]

# Housing Frequency and Proportion Table
housingCT = pd.crosstab(index=housing,columns=response, margins=True)
housingCT.columns = ["No_Deposit", "Deposit", "Row_Total"]
housingCT.index = ["No_Housing", "Housing", "Col_Total"]
housingProp = housingCT/housingCT.loc["Col_Total","Row_Total"]

# Loan Frequency and Proportion Table
loanCT = pd.crosstab(index=loan,columns=response, margins=True)
loanCT.columns = ["No_Deposit", "Deposit", "Row_Total"]
loanCT.index = ["No_Loan", "Loan", "Col_Total"]
loanProp = loanCT/loanCT.loc["Col_Total","Row_Total"]

PositiveLoanProp = loanCT.div(loanCT["Row_Total"],axis=0)

print(defaultCT)
print(defaultProp)

print(housingCT)
print(housingProp)

print(loanCT)
print(loanProp)
print(PositiveLoanProp)

# Create pdf file to store tgraphs
with PdfPages(datapath + 'output1.pdf') as pdf:

    # Bar charts of variable frequencies
    defaultPLT = pd.crosstab(index=default,columns=response)
    fig = defaultPLT.plot.bar().get_figure()
    pdf.savefig(fig) 
    
    housingPLT = pd.crosstab(index=housing,columns=response)
    fig = housingPLT.plot.bar().get_figure()
    pdf.savefig(fig) 
    
    loanPLT = pd.crosstab(index=loan,columns=response)
    fig = loanPLT.plot.bar().get_figure()
    pdf.savefig(fig) 


#################################
# Select best modeling method  #
#################################

# set up numpy array for storing results
cv_results = np.zeros((N_FOLDS, len(names)))

# set up kfolds
kf = KFold(n_splits = N_FOLDS, shuffle=False, random_state = RANDOM_SEED)

# check the splitting process by looking at fold observation counts
index_for_fold = 0  # fold count initialized 

# Train and test the Logistic Regression and Naive Bayes models on each of the folds
for train_index, test_index in kf.split(model_data):
    print('\nFold index:', index_for_fold,
          '------------------------------------------')
#   note that 0:model_data.shape[1]-1 slices for explanatory variables
#   and model_data.shape[1]-1 is the index for the response variable    
    X_train = model_data[train_index, 0:model_data.shape[1]-1]
    X_test = model_data[test_index, 0:model_data.shape[1]-1]
    y_train = model_data[train_index, model_data.shape[1]-1]
    y_test = model_data[test_index, model_data.shape[1]-1]   
    print('\nShape of input data for this fold:',
          '\nData Set: (Observations, Variables)')
    print('X_train:', X_train.shape)
    print('X_test:',X_test.shape)
    print('y_train:', y_train.shape)
    print('y_test:',y_test.shape)

    index_for_method = 0  # initialize
    for name, clf in zip(names, classifiers):
        print('\nClassifier evaluation for:', name)
        print('  Scikit Learn method:', clf)
        clf.fit(X_train, y_train)  # fit on the train set for this fold
        # evaluate on the test set for this fold
        y_test_predict = clf.predict_proba(X_test)
        fold_method_result = roc_auc_score(y_test, y_test_predict[:,1]) 
        print('Area under ROC curve:', fold_method_result)
        cv_results[index_for_fold, index_for_method] = fold_method_result
        index_for_method += 1
  
    index_for_fold += 1

cv_results_df = pd.DataFrame(cv_results)
cv_results_df.columns = names

# Print the mean result from the kfolds for each of the models for comparison
print('\n----------------------------------------------')
print('Average results from ', N_FOLDS, '-fold cross-validation\n',
      '\nMethod                 Area under ROC Curve', sep = '')     
print(cv_results_df.mean())   


########################################
# Fit model using best modeling method #
########################################

# Fit model using LogisticRegression utilizing all the sample data
logreg = LogisticRegression()
logreg.fit(model_data[:, 0:model_data.shape[1]-1], model_data[:, model_data.shape[1]-1])

# Try to find a cutoff to figure out how to classify 
bank_data['pred_prob'] = logreg.predict_proba(model_data[:, 0:model_data.shape[1]-1])[:,1]

# try cutoff set at 0.50
bank_data['pred_logit_50'] =\
    bank_data['pred_prob'].\
    apply(lambda d: prob_to_pred(d, cutoff = 0.50))    
print('\nConfusion matrix for 0.50 cutoff\n',\
    pd.crosstab(bank_data.pred_logit_50, bank_data.response, margins = True))    
# cutoff 0.50 does not work for targeting... all predictions 0 or No  

# try cutoff set at 0.10
bank_data['pred_logit_10'] =\
    bank_data['pred_prob'].\
    apply(lambda d: prob_to_pred(d, cutoff = 0.10))    
print('\nConfusion matrix for 0.10 cutoff\n',\
    pd.crosstab(bank_data.pred_logit_10, bank_data.response, margins = True)) 
    
# try cutoff set at 0.09
bank_data['pred_logit_09'] =\
    bank_data['pred_prob'].\
    apply(lambda d: prob_to_pred(d, cutoff = 0.09))    
print('\nConfusion matrix for 0.09 cutoff\n',\
    pd.crosstab(bank_data.pred_logit_09, bank_data.response, margins = True)) 
# This cutoff is more appropriate 

# Do a check of the logistic regression performance
print('\n Logistic Regression Performance (0.10 cutoff)\n',\
    'Percentage of Targets Correctly Classified:',\
    100 * round(evaluate_classifier(bank_data['pred_logit_10'],\
    bank_data['response'])[4], 3),'\n')
    
# Investigate classes of customers
cust_classes = np.array([[0,0,0], [1,0,0], [1,1,0],  [1,0,1], [1,1,1],
                        [0,1,0], [0,1,1], [0,0,1]])
                        
cust_predictions = pd.DataFrame(data=cust_classes)

cust_predictions.columns = ["Default", "Housing", "Loan"]

cust_predictions['pred_prob'] = logreg.predict_proba(cust_classes)[:,1]
print(cust_predictions)


# Now provide a visual analysis of the findings
# data to plot
n_groups = 8
Actual_positive_prop = (len(bank_data.loc[(bank_data['default'] == 0) & (bank_data['housing'] == 0) & (bank_data['loan'] == 0) & (bank_data['response'] ==1)])\
                    /len(bank_data.loc[(bank_data['default'] == 0) & (bank_data['housing'] == 0) & (bank_data['loan'] == 0)]) ,
                   len(bank_data.loc[(bank_data['default'] == 1) & (bank_data['housing'] == 0) & (bank_data['loan'] == 0) & (bank_data['response'] ==1)])\
                    /len(bank_data.loc[(bank_data['default'] == 1) & (bank_data['housing'] == 0) & (bank_data['loan'] == 0)]),
                   len(bank_data.loc[(bank_data['default'] == 1) & (bank_data['housing'] == 1) & (bank_data['loan'] == 0) & (bank_data['response'] ==1)])\
                    /len(bank_data.loc[(bank_data['default'] == 1) & (bank_data['housing'] == 1) & (bank_data['loan'] == 0)]),
                    len(bank_data.loc[(bank_data['default'] == 1) & (bank_data['housing'] == 0) & (bank_data['loan'] == 1) & (bank_data['response'] ==1)])\
                    /len(bank_data.loc[(bank_data['default'] == 1) & (bank_data['housing'] == 0) & (bank_data['loan'] == 1)]), 
                   len(bank_data.loc[(bank_data['default'] == 1) & (bank_data['housing'] == 1) & (bank_data['loan'] == 1) & (bank_data['response'] ==1)])\
                    /len(bank_data.loc[(bank_data['default'] == 1) & (bank_data['housing'] == 1) & (bank_data['loan'] == 1)]),                 
                   len(bank_data.loc[(bank_data['default'] == 0) & (bank_data['housing'] == 1) & (bank_data['loan'] == 0) & (bank_data['response'] ==1)])\
                    /len(bank_data.loc[(bank_data['default'] == 0) & (bank_data['housing'] == 1) & (bank_data['loan'] == 0)]),                     
                   len(bank_data.loc[(bank_data['default'] == 0) & (bank_data['housing'] == 1) & (bank_data['loan'] == 1) & (bank_data['response'] ==1)])\
                    /len(bank_data.loc[(bank_data['default'] == 0) & (bank_data['housing'] == 1) & (bank_data['loan'] == 1)]), 
                   len(bank_data.loc[(bank_data['default'] == 0) & (bank_data['housing'] == 0) & (bank_data['loan'] == 1) & (bank_data['response'] ==1)])\
                    /len(bank_data.loc[(bank_data['default'] == 0) & (bank_data['housing'] == 0) & (bank_data['loan'] == 1)]))   
                    
objects = ('Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 'Class 6', 'Class 7', 'Class 8')

# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8
 
rects1 = plt.bar(index, Actual_positive_prop, bar_width,
                 alpha=opacity,
                 color='b',
                 label='Actual Positive Proportion')
 
rects2 = plt.bar(index + bar_width, cust_predictions['pred_prob'], bar_width,
                 alpha=opacity,
                 color='g',
                 label='Predicted Positive Proportion')
 
plt.xlabel('Classes')
plt.ylabel('Proportion of Positive Responses')
plt.title('Comparison of Actual and Predicted Positive Proportions')
plt.xticks(index + bar_width, ('Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 'Class 6', 'Class 7'))
plt.legend()
 
plt.tight_layout()
plt.savefig(datapath+'output2.pdf') 

#########################################
# Combine all pdf files into 1 pdf file #
#########################################

pdfWriter = pp.PdfFileWriter()
pdfOne = pp.PdfFileReader(open(datapath + "output1.pdf", "rb"))
pdfTwo = pp.PdfFileReader(open(datapath + "output2.pdf", "rb"))

for pageNum in range(pdfOne.numPages):        
    pageObj = pdfOne.getPage(pageNum)
    pdfWriter.addPage(pageObj)


for pageNum in range(pdfTwo.numPages):        
    pageObj = pdfTwo.getPage(pageNum)
    pdfWriter.addPage(pageObj)


outputStream = open(datapath + r"Assign2_Output.pdf", "wb")
pdfWriter.write(outputStream)
outputStream.close()
