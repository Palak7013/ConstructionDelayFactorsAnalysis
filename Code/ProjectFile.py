#Importing Libraries
import matplotlib.pyplot as plt
import numpy as np

# %matplotlib qt

#Importing Dataset
import pandas as pd

dataset = pd.read_csv('FrequencyCopy_RoadConstruction2.csv')

X = dataset.iloc[:, 1:45]        # All Factors
x = dataset.iloc[:, 1:45].values

Y = dataset.iloc[:, 45]          # Type of Organization
y = dataset.iloc[:, 45].values    


# Implementing SVC on Dataset before factor analysis

# Splitting the Dataset into Test set and Training set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.20, random_state = 0)

from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, Y_train)

# Predicting the Test set results
Y_pred = classifier.predict(X_test)
print(Y_pred)
print(Y_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_before = confusion_matrix(Y_test, Y_pred)                                                                 # ACCURACY = 58% (test size = 0.15)
print(cm_before)                                                                                             # ACCURACY = 66.6% (test size = 0.10)
                                                  # ACCURACY = 51.7% (test size = 0.20)

# Correlation Matrix of 44 Factors
corr = X.corr()

# Correlation Heatmap of 44 Factors
import seaborn as sns

ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)

ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
)

# PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 44)
X_pca = pca.fit_transform(X)

#Finding Explained Varience Ratio
per_var = pca.explained_variance_
var = np.round(per_var, decimals=2)

#Plotting PCA Results
plt.title('Dataset\n', fontsize = 18)
plt.xlabel('Principal Components', fontsize=16)
plt.ylabel(' Varience', fontsize = 16)
plt.plot(var,'ro-')

#Factor Analysis (Bartlett’s Test) ( p_value should be zero)
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
chi_square_value,p_value=calculate_bartlett_sphericity(x)
print('\nChi Square Value (Dataset): ',chi_square_value)
print('\np Value (Dataset): ',p_value)


#Calculating Eignvalues (value greater than 1 is considered)
from factor_analyzer import FactorAnalyzer
fa = FactorAnalyzer()
fa.fit(x)

# Check & Uniqueness Eigenvalues
ev, v = fa.get_eigenvalues()
r_ev = np.round(ev, decimals=2)
print('\nEignValues (Dataset):\n',ev)

#Plotting EigenVaues
plt.title(' Dataset ')
plt.xlabel('Factors')
plt.ylabel('EigenValues')
plt.plot(r_ev,'ro-')

# Performing factor Analysis (11 factors have EigenValues more than 1)
fa = FactorAnalyzer(n_factors = 11)
fa.fit(x,11)
loadings = fa.loadings_  
r_loadings = np.round(loadings, decimals=4) 
print('Loadings (Dataset):\n',r_loadings)    

# Calculating Varience for factor analysis
ss_loading1, proportion_var1, cumulative_var1 = fa.get_factor_variance()
print("\nSS Loading (Category 1):\n ",ss_loading1)
print("\nProportion Var (Category 1):\n ",proportion_var1)
print("\nCumulative Var (Category 1):\n ",cumulative_var1)


# Selecting Top 11 features

# Import the necessary libraries first
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# Feature extraction
test = SelectKBest(score_func=chi2, k=11)
fit = test.fit(x, y)

# Summarize scores
np.set_printoptions(precision=2)
print(fit.scores_)


#Dataset with top 11 features only
X_top = x[:,[6, 11, 12, 24, 25, 26, 27, 30, 32, 33 ,34]]
X_top = x[:,0:11]


# Implementing SVC (Support Vector Classification)

# Splitting the Dataset into Test set and Training set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_top, y, test_size = 0.20, random_state = 0)

from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, Y_train)

# Predicting the Test set results
Y_pred = classifier.predict(X_test)
print(Y_pred)
print(Y_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_after = confusion_matrix(Y_test, Y_pred)                                                                 # ACCURACY = 54% (test size = 0.15)  72.7%
print(cm_after)                                                                                             # ACCURACY = 53.3% (test size = 0.10)  73%
                                           # ACCURACY = 51.7% (test size = 0.20)  58.6%



#__________________________________________________Category1_____________________________________________________

X1 = dataset.iloc[:, 1:6]
x1 = dataset.iloc[:, 1:6].values



# Correlation Matrix
corr1 = X1.corr()

# Correlation Heatmap
import seaborn as sns

ax1 = sns.heatmap(
    corr1, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)

ax1.set_xticklabels(
    ax1.get_xticklabels(),
    rotation=15,
    horizontalalignment='right'
)

# PCA
from sklearn.decomposition import PCA
pca1 = PCA(n_components = 5)
X1_pca = pca1.fit_transform(X1)

#Finding Explained Varience Ratio
per_var1 = pca1.explained_variance_
var1 = np.round(per_var1, decimals=2)

#Plotting PCA Results
plt.title('Materials')
plt.xlabel('Principal Components')
plt.ylabel(' Varience')
plt.plot(var1,'ro-')


#Factor Analysis (Bartlett’s Test)
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
chi_square1,p_value1 = calculate_bartlett_sphericity(x1)
print('\nChi Square Value (Category 1): ',chi_square1)    # 235.0309110
print('\np Value (Category 1): ',p_value1)                # 3.65447374402961e-45

#Calculating Eignvalues (value greater than 1 is considered)
from factor_analyzer import FactorAnalyzer
fa1 = FactorAnalyzer()
fa1.fit(x1)

# Check Eigenvalues
ev1, v1 = fa1.get_eigenvalues()
r_ev1 = np.round(ev1, decimals=2)
print('\nEigenValues (Category 1):\n',ev1)   # 2.63442753 0.81696845 0.72714019 0.63474533 0.18671851

#Plotting EigenVaues
plt.title('Materials')
plt.xlabel('Factors')
plt.ylabel('EigenValues')
plt.plot(r_ev1,'ro-')


# Selecting Top 1 features

# Import the necessary libraries first
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# Feature extraction
test = SelectKBest(score_func=chi2, k=1)
fit = test.fit(x1, y)

# Summarize scores
np.set_printoptions(precision=2)
print(fit.scores_)

#Dataset with top 1 feature only
#X1_top = x1[:,[4]]
X1_top = x1[:,[0]]


#_________________________________________________Category2______________________________________________________

X2 = dataset.iloc[:, 6:14]
x2 = dataset.iloc[:, 6:14].values

# Correlation Matrix
corr2 = X2.corr()

# Correlation heatmap
import seaborn as sns

ax2 = sns.heatmap(
    corr2, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)

ax2.set_xticklabels(
    ax2.get_xticklabels(),
    rotation=15,
    horizontalalignment='right'
)

# PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 8)
X2_pca = pca.fit_transform(X2)

#Finding Explained Varience Ratio
per_var = pca.explained_variance_
var2 = np.round(per_var, decimals=2)


#Plotting PCA Results
plt.title('Labor & Equipments')
plt.xlabel('Principal Components')
plt.ylabel(' Varience')
plt.plot(var2,'ro-')


#Factor Analysis (Bartlett’s Test)
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
chi_square2,p_value2 = calculate_bartlett_sphericity(x2)
print('\nChi Square Value (Category 2): ',chi_square2)
print('\np Value (Category 2): ',p_value2)

#Calculating Eignvalues (value greater than 1 is considered)
from factor_analyzer import FactorAnalyzer
fa2 = FactorAnalyzer()
fa2.fit(x2)

# Check  Uniqueness & Eigenvalues
ev2, v2 = fa2.get_eigenvalues()
r_ev2 = np.round(ev2, decimals=2)
print('\nEigenValues (Category 2):\n',ev2)  #  [5.36 0.66 0.59 0.42 0.37 0.26 0.19 0.15]

#Plotting EigenVaues
plt.title('Labor & Equipments')
plt.xlabel('Factors')
plt.ylabel('EigenValues')
plt.plot(r_ev2,'ro-')


# Selecting Top 1 features
# Import the necessary libraries first
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# Feature extraction
test = SelectKBest(score_func=chi2, k=1)
fit = test.fit(x2, y)

# Summarize scores
np.set_printoptions(precision=2)
print(fit.scores_)

#Dataset with top 1 feature only
X2_top = x2[:,[6]]



#__________________________________________________Category3______________________________________________________

X3 = dataset.iloc[:, 14:19]
x3 = dataset.iloc[:, 14:19].values

# Correlation Matrix
corr3 = X3.corr()

# Correlation Heatmap
import seaborn as sns

ax3 = sns.heatmap(
    corr3, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)

ax3.set_xticklabels(
    ax3.get_xticklabels(),
    rotation=15,
    horizontalalignment='right'
)

# PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 5)
X3_pca = pca.fit_transform(X3)

#Finding Explained Varience Ratio
per_var = pca.explained_variance_
var3 = np.round(per_var, decimals=2)


#Plotting PCA Results
plt.title('Financing\n')
plt.xlabel('Principal Components')
plt.ylabel(' Varience')
plt.plot(var3,'ro-')

#Factor Analysis (Bartlett’s Test)
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
chi_square3, p_value3 = calculate_bartlett_sphericity(x3)
print('\nChi Square Value (Category 3): ',chi_square3)
print('\np Value (Category 3): ',p_value3)

#Calculating Eignvalues (value greater than 1 is considered)
from factor_analyzer import FactorAnalyzer
fa3 = FactorAnalyzer()
fa3.fit_transform(x3)

# Check Eigenvalues
ev3, v3 = fa3.get_eigenvalues()
r_ev3 = np.round(ev3, decimals=2)
print('\nEigenValues (Category 3):\n',ev3)

#Plotting EigenVaues
plt.title('Financing\n')
plt.xlabel('Factors')
plt.ylabel('EigenValues')
plt.plot(r_ev3,'ro-')


# Selecting Top 2 features
# Import the necessary libraries first
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# Feature extraction
test = SelectKBest(score_func=chi2, k=2)
fit = test.fit(x3, y)

# Summarize scores
np.set_printoptions(precision=2)
print(fit.scores_)

#Dataset with top 1 feature only
X3_top = x3[:,[2, 4]]

#__________________________________________________Category4______________________________________________________

X4 = dataset.iloc[:, 19:25]
x4 = dataset.iloc[:, 19:25].values

# Correlation Matrix
corr4 = X4.corr()

# Correlation Heatmap
import seaborn as sns

ax4 = sns.heatmap(
    corr4, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)

ax4.set_xticklabels(
    ax4.get_xticklabels(),
    rotation=15,
    horizontalalignment='right'
)

# PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 6)
X4_pca = pca.fit_transform(X4)

#Finding Explained Varience Ratio
per_var = pca.explained_variance_
var4 = np.round(per_var, decimals=2)


#Plotting PCA Results
plt.title('Design & Documentation\n')
plt.xlabel('Principal Components')
plt.ylabel(' Varience')
plt.plot(var4,'ro-')

#Factor Analysis (Bartlett’s Test)
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
chi_square4, p_value4 = calculate_bartlett_sphericity(x4)
print('\nChi Square Value (Category 4): ',chi_square4)
print('\np Value (Category 4): ',p_value4)

#Calculating Eignvalues (value greater than 1 is considered)
from factor_analyzer import FactorAnalyzer
fa = FactorAnalyzer()
fa.fit_transform(x4)

# Check Eigenvalues
ev4, v4 = fa.get_eigenvalues()
r_ev4 = np.round(ev4, decimals=2)
print('\nEigenValues (Category 4):\n',ev4)  #[3.35 0.87 0.65 0.44 0.34 0.34]

#Plotting EigenVaues
plt.title('Design & Documentation\n')
plt.xlabel('Factors')
plt.ylabel('EigenValues')
plt.plot(r_ev4,'ro-')


# Selecting Top 1 features
# Import the necessary libraries first
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# Feature extraction
test = SelectKBest(score_func=chi2, k=1)
fit = test.fit(x4, y)

# Summarize scores
np.set_printoptions(precision=2)
print(fit.scores_)

#Dataset with top 1 feature only
X4_top = x4[:,[5]]


#__________________________________________________Category5______________________________________________________

X5 = dataset.iloc[:, 25:31]
x5 = dataset.iloc[:, 25:31].values

# Correlation Matrix
corr5 = X5.corr()

# Correlation Heatmap
import seaborn as sns

ax5 = sns.heatmap(
    corr5, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)

ax5.set_xticklabels(
    ax5.get_xticklabels(),
    rotation=15,
    horizontalalignment='right'
)
 

# PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 6)
X5_pca = pca.fit_transform(X5)

#Finding Explained Varience Ratio
per_var = pca.explained_variance_
var5 = np.round(per_var, decimals=2)


#Plotting PCA Results
plt.title('Management & Organization\n')
plt.xlabel('Principal Components')
plt.ylabel(' Varience')
plt.plot(var5,'ro-')

#Factor Analysis (Bartlett’s Test)
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
chi_square5, p_value5 = calculate_bartlett_sphericity(x5)
print('\nChi Square Value (Category 5): ',chi_square5)
print('\np Value (Category 5): ',p_value5)

#Calculating Eignvalues (value greater than 1 is considered)
from factor_analyzer import FactorAnalyzer
fa = FactorAnalyzer()
fa.fit_transform(x5)

# Check Eigenvalues
ev5, v5 = fa.get_eigenvalues()
r_ev5 = np.round(ev5, decimals=2)
print('\nEigenValues (Category 5):\n',ev5)

#Plotting EigenVaues
plt.title('Management & Organization\n')
plt.xlabel('Factors')
plt.ylabel('EigenValues')
plt.plot(r_ev5,'ro-')


# Selecting Top 2 features
# Import the necessary libraries first
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# Feature extraction
test = SelectKBest(score_func=chi2, k=2)
fit = test.fit(x5, y)

# Summarize scores
np.set_printoptions(precision=2)
print(fit.scores_)

#Dataset with top 2 feature only
X5_top = x5[:,[0,3]]

#__________________________________________________Category6______________________________________________________

X6 = dataset.iloc[:, 31:33]
x6 = dataset.iloc[:, 31:33].values

# Correlation Matrix
corr6 = X6.corr()

# Correlation Heatmap
import seaborn as sns

ax6 = sns.heatmap(
    corr6, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)

ax6.set_xticklabels(
    ax6.get_xticklabels(),
    rotation=0.5,
    horizontalalignment='center'
)

ax6.set_yticklabels(
    ax6.get_yticklabels(),
    rotation = 0.5,
    horizontalalignment='right'
)

# PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
X6_pca = pca.fit_transform(X6)

#Finding Explained Varience Ratio
per_var = pca.explained_variance_
var6 = np.round(per_var, decimals=2)


#Plotting PCA Results
plt.title('Schedule\n')
plt.xlabel('Principal Components')
plt.ylabel(' Varience')
plt.plot(var6,'ro-')

#Factor Analysis (Bartlett’s Test)
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
chi_square6, p_value6 = calculate_bartlett_sphericity(x6)
print('\nChi Square Value (Category 6): ',chi_square6)
print('\np Value (Category 6): ',p_value6)

#Calculating Eignvalues (value greater than 1 is considered)
from factor_analyzer import FactorAnalyzer
fa6 = FactorAnalyzer()
fa6.fit_transform(x6)

# Check Eigenvalues
ev6, v6 = fa6.get_eigenvalues()
r_ev6 = np.round(ev6, decimals=2)
print('\nEigenValues (Category 6):\n',ev6)

#Plotting EigenVaues
plt.title('Schedule\n')
plt.xlabel('Factors')
plt.ylabel('EigenValues')
plt.plot(r_ev6,'ro-')

# Selecting Top 1 features
# Import the necessary libraries first
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# Feature extraction
test = SelectKBest(score_func=chi2, k=1)
fit = test.fit(x6, y)

# Summarize scores
np.set_printoptions(precision=2)
print(fit.scores_)

#Dataset with top 1 feature only
X6_top = x6[:,[0]]

#__________________________________________________Category7_____________________________________________________

X7 = dataset.iloc[:, 33:38]
x7 = dataset.iloc[:, 33:38].values

# Correlation Matrix
corr7 = X7.corr()

# Correlation Heatmap
import seaborn as sns

ax7 = sns.heatmap(
    corr7, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)

ax7.set_xticklabels(
    ax7.get_xticklabels(),
    rotation=15,
    horizontalalignment='right'
)

# PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 5)
X7_pca = pca.fit_transform(X7)

#Finding Explained Varience Ratio
per_var7 = pca.explained_variance_
var7 = np.round(per_var7, decimals=2)


#Plotting PCA Results
plt.title('Contractual Issues\n')
plt.xlabel('Principal Components')
plt.ylabel(' Varience')
plt.plot(var7,'ro-')

#Factor Analysis (Bartlett’s Test)
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
chi_square7 ,p_value7 = calculate_bartlett_sphericity(x7)
print('\nChi Square Value (Category 7): ',chi_square7)
print('\np Value (Category 7): ',p_value7)

#Calculating Eignvalues (value greater than 1 is considered)
from factor_analyzer import FactorAnalyzer
fa7 = FactorAnalyzer()
fa7.fit_transform(x7)

# Check Eigenvalues
ev7, v7 = fa7.get_eigenvalues()
r_ev7 = np.round(ev7, decimals=2)
print('\nEigenValues (Category 7):\n',ev7)

#Plotting EigenVaues
plt.title('Contractual Issues\n')
plt.xlabel('Factors')
plt.ylabel('EigenValues')
plt.plot(r_ev7,'ro-')

# Selecting Top 1 features
# Import the necessary libraries first
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# Feature extraction
test = SelectKBest(score_func=chi2, k=1)
fit = test.fit(x7, y)

# Summarize scores
np.set_printoptions(precision=2)
print(fit.scores_)

#Dataset with top 1 feature only
X7_top = x7[:,[1]]
#__________________________________________________Category8_____________________________________________________

X8 = dataset.iloc[:, 38:42]
x8 = dataset.iloc[:, 38:42].values

# Correlation Matrix
corr8 = X8.corr()

# Correlation Heatmap
import seaborn as sns

ax8 = sns.heatmap(
    corr8, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)

ax8.set_xticklabels(
    ax8.get_xticklabels(),
    rotation=15,
    horizontalalignment='right'
)

# PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 4)
X8_pca = pca.fit_transform(X8)

#Finding Explained Varience Ratio
per_var8 = pca.explained_variance_
var8 = np.round(per_var8, decimals=2)


#Plotting PCA Results
plt.title('Scope of Work\n')
plt.xlabel('Principal Components')
plt.ylabel(' Varience')
plt.plot(var8,'ro-')

#Factor Analysis (Bartlett’s Test)
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
chi_square8, p_value8 = calculate_bartlett_sphericity(x8)
print('\nChi Square Value (Category 8): ',chi_square8)
print('\np Value (Category 8): ',p_value8)

#Calculating Eignvalues (value greater than 1 is considered)
from factor_analyzer import FactorAnalyzer
fa8 = FactorAnalyzer()
fa8.fit_transform(x8)

# Check Eigenvalues
ev8, v8 = fa8.get_eigenvalues()
r_ev8 = np.round(ev8, decimals=2)
print('\nEigenValues (Category 8):\n',ev8)

#Plotting EigenVaues
plt.title('Scope of Work\n')
plt.xlabel('Factors')
plt.ylabel('EigenValues')
plt.plot(r_ev8,'ro-')

# Selecting Top 1 features
# Import the necessary libraries first
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# Feature extraction
test = SelectKBest(score_func=chi2, k=1)
fit = test.fit(x8, y)

# Summarize scores
np.set_printoptions(precision=2)
print(fit.scores_)

#Dataset with top 1 feature only
X8_top = x8[:,[1]]
#__________________________________________________Category9_____________________________________________________

X9 = dataset.iloc[:, 42:45]
x9 = dataset.iloc[:, 42:45].values

# Correlation Matrix
corr9 = X9.corr()

# Correlation Heatmap
import seaborn as sns

ax9 = sns.heatmap(
    corr9, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)

ax9.set_xticklabels(
    ax9.get_xticklabels(),
    rotation=15,
    horizontalalignment='right'
)

# PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 3)
X9_pca = pca.fit_transform(X9)

#Finding Explained Varience Ratio
per_var9 = pca.explained_variance_
var9 = np.round(per_var9, decimals=2)

#Plotting PCA Results
plt.title('External Issues\n')
plt.xlabel('Principal Components')
plt.ylabel(' Varience')
plt.plot(var9,'ro-')

#Factor Analysis (Bartlett’s Test)
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
chi_square9, p_value9 = calculate_bartlett_sphericity(x9)
print('\nChi Square Value (Category 9): ',chi_square9)
print('\np Value (Category 9): ',p_value9)



#Calculating Eignvalues (value greater than 1 is considered)
from factor_analyzer import FactorAnalyzer
fa9 = FactorAnalyzer()
fa9.fit_transform(x9)

# Check Eigenvalues
ev9, v9 = fa9.get_eigenvalues()
r_ev9 = np.round(ev9, decimals=2)
print('\nEigenValues (Category 9):\n',ev9)

#Plotting EigenVaues
plt.title('External Issues\n')
plt.xlabel('Factors')
plt.ylabel('EigenValues')
plt.plot(r_ev9,'ro-')

# Selecting Top 1 features

# Import the necessary libraries first
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# Feature extraction
test = SelectKBest(score_func=chi2, k=2)
fit = test.fit(x9, y)

# Summarize scores
np.set_printoptions(precision=2)
print(fit.scores_)

#Dataset with top 1 feature only
X9_top = x9[:,[0,1]]

#********************************************* After Analysis *****************************************************
# Combining resylts of all categories

#Dataset with top features only

#Xafter_top = x[:,[1, 6, 14, 15, 19, 25, 26, 31, 33, 38, 42 ,43]]

Xafter_top = x[:,[1,12,16,18,24,25,28,31,34,39,42,43]]
# Implementing SVC (Support Vector Classification)

# Splitting the Dataset into Test set and Training set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(Xafter_top, y, test_size = 0.20, random_state = 0)

from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, Y_train)

# Predicting the Test set results
Y_pred = classifier.predict(X_test)
print(Y_pred)
print(Y_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cmconclusion_after = confusion_matrix(Y_test, Y_pred)                                                       # ACCURACY = 50% (test size = 0.15)    59%
print(cmconclusion_after)                                                                                   # ACCURACY = 53.3% (test size = 0.10)  53.3%
                                                       # ACCURACY = 55.17% (test size = 0.20) 51.72%






