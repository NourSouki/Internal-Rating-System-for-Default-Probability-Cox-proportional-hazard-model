#!/usr/bin/env python
# coding: utf-8

# # Internal Rating System for Probability of Default using Cox proportional hazard model

# The objectif is to predict Default probability using a dataset full of relevant financial indicators. the project took a multifaceted approach, integrating exploratory data analysis, statistical methodologies especially the Cox Model:

# ### Import necessary packages

# In[378]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
#from catboost import CatBoostClassifier
#import xgboost as xgb
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score


# ### Read data

# In[379]:


data=pd.read_excel("SNI.xlsx")
data.head()


# ### Check Data

# In[380]:


for col in data.columns:
    if data[col].dtype=="object":
        unique_values = data[col].unique()
        print(f"Unique values in {col} column:")
        print(unique_values)
        print("\n")


# Some steps of data validation and cleaning are necessary before moving on

# In[381]:


data["DIVERSITE_CLIENTS"].replace('Diversification très forte par produits,clients, situation geographique', 'Diversification tres forte par produits, clients, situation geographique', inplace=True)
data["DIVERSITE_CLIENTS"].replace('Diversification très forte par produits, clients, situation geographique', 'Diversification tres forte par produits, clients, situation geographique', inplace=True)
data["DIVERSITE_FOURNISSEURS"].replace('Tres grande diversite', 'Très grande diversite', inplace=True)
data["IMPACT_SOCIAUX_ENVIRONNEMENTAL"].replace('Aucun impact social ou environnemental, soumis e une reglementation', 'Aucun impact social ou environnemental, soumis à une reglementation', inplace=True)
data["NIVEAU_COMPETITIVITE"].replace('Tres forte concurrence', 'Très forte concurrence', inplace=True)
data["REPUTATION"].replace('Tres bonne', 'Très bonne', inplace=True)
for col in data.columns:
    if data[col].dtype=="object":
        unique_values = data[col].unique()
        print(f"Unique values in {col} column:")
        print(unique_values)
        print("\n")


# In[382]:


float_columns=data.select_dtypes(include='float64').columns
len(float_columns)


# ### Data visualization

# In[383]:


num_cols = 4
num_rows = len(float_columns) // num_cols + (len(float_columns) % num_cols > 0)

fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 12))

axes = axes.flatten()

for i, col in enumerate(float_columns):
    ax = axes[i]
    data.boxplot(column=col, ax=ax)
    ax.set_title(f'Boxplot of {col}')

for i in range(len(float_columns), num_rows * num_cols):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.show()


# ### Visualize the default items

# In[384]:



columns = ['EXPERIENCE_MANAGEMENT_MOYENNE_DIRIGEANT', 'DIVERSITE_CLIENTS',
           'DIVERSITE_FOURNISSEURS', 'IMPACT_SOCIAUX_ENVIRONNEMENTAL',
           'NIVEAU_COMPETITIVITE', 'QUALITE_INFORMATION_FINANCIERE', 'REPUTATION',
           'STRUCTUREDUMANAGEMENT', 'SUPPORT', 'POSITIONNEMENTMARCHE',
           'Categorie_juridique', 'Cote en bourse', 'Appartenance a un groupe',
           'Secteurs']


if 'defaut' in data.columns:
    for column in columns:
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))

        # All data
        sns.countplot(x=data[column], ax=ax[0])
        ax[0].set_title(column + ' (all data)')
        ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=90)

        # Category 0
        sns.countplot(x=data[column][data['defaut'] == 0], ax=ax[1])
        ax[1].set_title(column + ' (category 0)')
        ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=90)

        # Category 1
        sns.countplot(x=data[column][data['defaut'] == 1], ax=ax[2])
        ax[2].set_title(column + ' (category 1)')
        ax[2].set_xticklabels(ax[2].get_xticklabels(), rotation=90)

        plt.show()
else:
    print("'defaut' column not found in the DataFrame.")


# In[385]:


data.shape


# ### Check null values

# In[386]:


# Count zeros in each column
zero_counts = data.eq(0).sum()

print(zero_counts)


# null values in columns like (default, stock, cote en bourse, appartenance a un groupe) are normal but in columns like (equity, total assets, liabilities..) are not evident and may generate errors and problems later 

# In[387]:


# Sort the DataFrame by 'numtiers' and 'Annee'
sorted_data = data.sort_values(by=['numtiers', 'Annee'])


# In[388]:


l_cap_pro=list(sorted_data[sorted_data['CAPITAUX_PROPRES']==0]['numtiers'])
l_tot_act=list(sorted_data[sorted_data['TOTAL_ACTIF']==0]['numtiers'])
l_pass_cir=list(sorted_data[sorted_data['PASSIF_CIRCULANT']==0]['numtiers'])


# ### Impute null values by the next year's value for the same company 

# In[389]:


print(sorted_data[sorted_data['numtiers'].isin(l_cap_pro)].sort_values(by=['numtiers','Annee'])[['numtiers','Annee', 'CAPITAUX_PROPRES']])

# Replace 0 values with NaN in 'CAPITAUX_PROPRES' column
sorted_data['CAPITAUX_PROPRES'].replace(0, np.nan, inplace=True)
# Forward fill null values in 'CAPITAUX_PROPRES' column within each 'numtiers' group
sorted_data['CAPITAUX_PROPRES'] = sorted_data.groupby('numtiers')['CAPITAUX_PROPRES'].transform(lambda x: x.fillna(method='ffill'))
# If there are still null values remaining, fill them with the next year's value
sorted_data['CAPITAUX_PROPRES'] = sorted_data.groupby('numtiers')['CAPITAUX_PROPRES'].transform(lambda x: x.fillna(method='bfill'))

print()
print(sorted_data[sorted_data['numtiers'].isin(l_cap_pro)].sort_values(by=['numtiers','Annee'])[['numtiers','Annee', 'CAPITAUX_PROPRES']])


# In[390]:


print(sorted_data[sorted_data['numtiers'].isin(l_tot_act)].sort_values(by=['numtiers','Annee'])[['numtiers','Annee', 'TOTAL_ACTIF']])
# Replace 0 values with NaN in 'TOTAL_ACTIF' column
sorted_data['TOTAL_ACTIF'].replace(0, np.nan, inplace=True)
# Forward fill null values in 'TOTAL_ACTIF' column within each 'numtiers' group
sorted_data['TOTAL_ACTIF'] = sorted_data.groupby('numtiers')['TOTAL_ACTIF'].transform(lambda x: x.fillna(method='ffill'))
# If there are still null values remaining, fill them with the next year's value
sorted_data['TOTAL_ACTIF'] = sorted_data.groupby('numtiers')['TOTAL_ACTIF'].transform(lambda x: x.fillna(method='bfill'))

print()
print(sorted_data[sorted_data['numtiers'].isin(l_tot_act)].sort_values(by=['numtiers','Annee'])[['numtiers','Annee', 'TOTAL_ACTIF']])


# In[391]:


print(sorted_data[sorted_data['numtiers'].isin(l_pass_cir)].sort_values(by=['numtiers','Annee'])[['numtiers','Annee', 'PASSIF_CIRCULANT']])
# Replace 0 values with NaN in 'PASSIF_CIRCULANT' column
sorted_data['PASSIF_CIRCULANT'].replace(0, np.nan, inplace=True)
# Forward fill null values in 'PASSIF_CIRCULANT' column within each 'numtiers' group
sorted_data['PASSIF_CIRCULANT'] = sorted_data.groupby('numtiers')['PASSIF_CIRCULANT'].transform(lambda x: x.fillna(method='ffill'))
# If there are still null values remaining, fill them with the next year's value
sorted_data['PASSIF_CIRCULANT'] = sorted_data.groupby('numtiers')['PASSIF_CIRCULANT'].transform(lambda x: x.fillna(method='bfill'))

print()
print(sorted_data[sorted_data['numtiers'].isin(l_pass_cir)].sort_values(by=['numtiers','Annee'])[['numtiers','Annee', 'PASSIF_CIRCULANT']])


# In[392]:


data=sorted_data.copy()


# In[393]:


# Show unique values in the 'Secteurs' column
unique_secteurs = data['Secteurs'].unique()
unique_secteurs


# ### some mistakes happened while date entry. The default year is the last year in the dataset .

# In[394]:


# Boolean indexing to filter rows where 'defaut' is equal to 1
defaut_1_rows = sorted_data[sorted_data['defaut'] == 1][["numtiers","Annee","defaut"]]
print(defaut_1_rows.shape)
#☺Imputation by nearest neighbors
for index, row in defaut_1_rows.iterrows():
    id_entrep=row['numtiers']
    subset_id=data[data['numtiers']==id_entrep]
    print(row['numtiers'],row["Annee"])
    print(subset_id[["numtiers","Annee","defaut"]])
    print()


# In[396]:


import pandas as pd

# Assuming 'data' is your DataFrame containing the dataset
# Reindex the dataset
data = data.reset_index(drop=True)

# Group by 'numtiers' and sort by 'Annee' within each group
data.sort_values(by=['numtiers', 'Annee'], inplace=True)

# Group by 'numtiers' and get the last row for each group
last_rows = data.groupby(['numtiers','Annee']).tail(1)

# Find the rows where 'default' is 1
default_1_rows = last_rows[last_rows['defaut'] == 1]

# Update 'default' to 1 for the last year's rows where it was originally 1
data.loc[default_1_rows.index, 'defaut'] = 1


# In[397]:


# Boolean indexing to filter rows where 'defaut' is equal to 1
defaut_1_rows = data[data['defaut'] == 1][["numtiers","Annee","defaut"]]
print(defaut_1_rows.shape)
#☺Imputation by nearest neighbors
for index, row in defaut_1_rows.iterrows():
    id_entrep=row['numtiers']
    subset_id=data[data['numtiers']==id_entrep]
    print(row['numtiers'],row["Annee"])
    print(subset_id[["numtiers","Annee","defaut"]])
    print()


# ### The best way to compare companies and make decisions is by using ratios. (The companies do not have the same scale, niether the features' values)

# In[398]:


# Calculate financial ratios
data['rentabilité'] = data['RESULTAT_NET'] / data['CHIFFRE_AFFAIRES']
data['Return_on_Assets'] = data['RESULTAT_NET'] / data['TOTAL_ACTIF']#utiliser ROA au lieu de ROE
data['endettement'] = data['CAPITAUX_PROPRES'] / data['TOTAL_ACTIF'] #ya plusieurs zéro => capitaux propres/total actif
data['liquidité'] = data['TRESORIE_NETTE'] / data['PASSIF_CIRCULANT'] #liquidité immédiate/cash ratio
data['rotation'] = data['CHIFFRE_AFFAIRES'] / data['TOTAL_ACTIF']

print(data)


# In[399]:


ratios_columns=["rentabilité","Return_on_Assets","endettement","liquidité","rotation"]


# In[400]:


data[ratios_columns].describe()


# In[401]:


R_0=data[data['rentabilité']<0]
R_1=data[data['rentabilité']>1]


# In[402]:


print(R_1[R_1["defaut"]==1].shape[0], "stés ont une rentabilité >1 et ont fait défaut")
print(R_0[R_0["defaut"]==1].shape[0], "stés ont une rentabilité <0 et ont fait défaut")
print()
print(R_1[R_1["defaut"]==0].shape[0], "stés ont une rentabilité >1 et n'ont pas fait défaut")
print(R_0[R_0["defaut"]==0].shape[0], "stés ont une rentabilité <0 et n'ont pas fait défaut")


# In[403]:


E_0=data[data['endettement']<0]
E_0[['numtiers','Annee','Secteurs','DETTE_FINANCIERE','CAPITAUX_PROPRES']].shape


# In[404]:


#☺Imputation by nearest neighbors
for index, row in E_0.iterrows():
    id_entrep=row['numtiers']
    subset_id=data[data['numtiers']==id_entrep]
    print(row['numtiers'],row["Annee"],row["endettement"])
    print(subset_id[["numtiers","Annee","endettement","defaut"]])
    print()


# ### Visualize the ratios data

# In[406]:


bins = 20  # Number of bins for the histograms

# Create a figure and axis object
fig, axs = plt.subplots(len(ratios_columns), 1, figsize=(8, 6 * len(ratios_columns)))

# Plot histograms for each ratio column with and without default
for i, column in enumerate(ratios_columns):
    # Plot histogram for default = 1
    axs[i].hist(data[data['defaut'] == 1][column], bins=bins, color='red', alpha=0.7, label='default = 1')
    
    # Set labels and title for each subplot
    axs[i].set_xlabel(column)
    axs[i].set_ylabel('Frequency')
    axs[i].set_title(f'Histogram of {column}')
    axs[i].legend()  # Add legend
    
plt.tight_layout()  # Adjust spacing between subplots
plt.show()


# In[407]:


means = data.groupby('defaut')[ratios_columns].mean()

means.plot(kind='bar', figsize=(10, 6))
plt.xlabel('Default')
plt.ylabel('Mean or Median')
plt.title('Mean of Ratios by Default Status')
plt.xticks(np.arange(2), ['Non-Default', 'Default'], rotation=0)
plt.grid(axis='y')
plt.legend(title='Ratio Variables', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


# ### Discretization : We can use this technique to optimize the work in a later version

# In[408]:


from sklearn.preprocessing import KBinsDiscretizer
# Initialize the discretizer with desired parameters
discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')

# Fit and transform the selected columns
discretized_data = discretizer.fit_transform(data[ratios_columns])

# Convert the discretized data back to a DataFrame
discretized_df = pd.DataFrame(discretized_data, columns=ratios_columns)
ratios_df=data
# Merge the discretized data back into the original DataFrame
ratios_df[ratios_columns] = discretized_df


# In[409]:


discretized_df


# In[410]:


bins = 5  # Number of bins for the histograms

# Create a figure and axis object
fig, axs = plt.subplots(len(ratios_columns), 1, figsize=(8, 6 * len(ratios_columns)))

# Plot histograms for each ratio column with and without default
for i, column in enumerate(ratios_columns):
    # Plot histogram for default = 1
    axs[i].hist(ratios_df[ratios_df['defaut'] == 1][column], bins=bins, color='blue', alpha=0.7, label='default = 1')
    
    # Set labels and title for each subplot
    axs[i].set_xlabel(column)
    axs[i].set_ylabel('Frequency')
    axs[i].set_title(f'Histogram of {column}')
    axs[i].legend()  # Add legend
plt.tight_layout()  # Adjust spacing between subplots
plt.show()


# In[411]:


ratios_df.head()


# ### Create the age variable

# In[412]:


from datetime import datetime

# Assuming the "creation_date" column is in datetime format
data['DATE_DE_CREATION_ENTREP'] = pd.to_datetime(data['DATE_DE_CREATION_ENTREP'])

# Calculate the age based on the current date
current_date = datetime.now()
data['age'] = (current_date - data['DATE_DE_CREATION_ENTREP']).dt.days/365.25

# Print the DataFrame with the new 'age' column
print(data['age'])


# In[414]:


data_encoded = pd.get_dummies(data, columns=['Secteurs'])


# ### We won't use all the columns as inputs: We'll proceed to calculate the weight of evidence and information value of the features in order to select the most important:

# In[415]:


def calculate_woe_iv(feature_data, target_data):
    total_events = np.sum(target_data)
    total_non_events = len(target_data) - total_events
    woe_iv = 0
    
    for value in feature_data.unique():
        event_count = np.sum(target_data[feature_data == value])
        non_event_count = np.sum((feature_data == value) & (target_data == 0))
        
        if event_count == 0 or non_event_count == 0:
            continue
        
        proportion_of_events = event_count / total_events
        proportion_of_non_events = non_event_count / total_non_events
        woe = np.log(proportion_of_events / proportion_of_non_events)
        iv = (proportion_of_events - proportion_of_non_events) * woe
        woe_iv += iv
    
    return woe_iv

def calculate_iv(df, target):
    iv_values = {}
    
    for feature in df.columns:
        if feature != target:
            feature_data = df[feature]
            target_data = df[target]
            
            if feature_data.dtype == 'object':
                feature_data = feature_data.astype('category').cat.codes
            
            iv_values[feature] = calculate_woe_iv(feature_data, target_data)
    
    return iv_values


# In[416]:


# Assuming you have already defined the function calculate_iv()

# Calculate IV values for each feature in your data
iv_values = calculate_iv(data, 'defaut')

# Print IV values for each feature
for feature, iv in iv_values.items():
    print(f"IV for {feature}: {iv}")


# In[417]:


data


# ### Time for the Cox model !

# In[418]:


from lifelines import CoxPHFitter
from imblearn.over_sampling import SMOTENC
from sklearn.preprocessing import LabelEncoder


# In[419]:


# Select relevant columns for survival analysis, such as 'defaut' (event), 'age', and other relevant features
selected_columns = [ 'defaut','NIVEAU_COMPETITIVITE', 'QUALITE_INFORMATION_FINANCIERE', 'REPUTATION',
       'Secteurs', 'rentabilité', 'Return_on_Assets', 'endettement',
       'liquidité', 'rotation','age']

data_selected = data[selected_columns]

# Convert floating-point numbers to integers with rounding up if decimal part >= 0.5
data_selected.loc[:, 'age'] = data_selected['age'].apply(lambda x: int(x) + 1 if isinstance(x, float) and x - int(x) >= 0.5 else x)

# Replace 'Plus de 10 ans' with a numerical value (e.g., 11)
data_selected.loc[:, 'age'] = data_selected['age'].replace('Plus de 10 ans', 11)

# Séparation des données en ensembles d'entraînement et de test
X = data_selected.drop('defaut', axis=1)
y = data_selected['defaut']

# Liste des caractéristiques catégorielles
categorical_features = ['REPUTATION', 'Secteurs']

# Convertir les noms des caractéristiques en indices
categorical_features_indices = [X.columns.get_loc(col) for col in categorical_features]

# Application de SMOTE
smote = SMOTENC(categorical_features=categorical_features_indices)


le = LabelEncoder()
for col in X.columns[X.dtypes == "object"]:
    X[col] = le.fit_transform(X[col])
X_NC, y_NC = smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_NC, y_NC, test_size=0.2, random_state=42)

X_train['defaut'] = y_train
X_test['defaut'] = y_test

# Fit the Cox proportional hazards model
cox_model = CoxPHFitter()
cox_model.fit(X_train, duration_col='age', event_col='defaut')


# In[420]:


# Print the summary of the fitted model
cox_model.print_summary()


# ### We have a concordance of 0.74 ! Amazing !

# In[421]:


# Access specific coefficients and hazard ratios
coefficients = cox_model.summary['coef']
hazard_ratios = cox_model.summary['exp(coef)']


# In[422]:


coefficients
hazard_ratios


# In[423]:


# Check the shape of hazard_ratios
print(hazard_ratios.shape)


# In[424]:


import matplotlib.pyplot as plt

# Check the shape of hazard_ratios
print("Shape of hazard_ratios:", hazard_ratios.shape)

# Check the shape of data_selected.columns[1:]
print("Shape of columns:", data_selected.columns[1:].shape)

# Make sure hazard_ratios is not empty and has non-zero size
if hazard_ratios is not None and len(hazard_ratios) > 0:
    # Check if shapes are compatible for plotting
    if hazard_ratios.shape[0] == data_selected.columns[1:].shape[0]:
        # Plotting hazard ratios
        plt.figure(figsize=(10, 6))
        plt.barh(data_selected.columns[1:], hazard_ratios.values[1:], color='skyblue')
        plt.xlabel('Hazard Ratio')
        plt.title('Hazard Ratios for Selected Features')
        plt.show()
    else:
        print("Shapes are not compatible for plotting.")
else:
    print("No hazard ratios data to plot.")


# In[425]:


# Make sure hazard_ratios is not empty and has non-zero size
if hazard_ratios is not None and len(hazard_ratios) > 0:
    # Plotting hazard ratios
    plt.figure(figsize=(10, 6))
    plt.barh(data_selected.columns[1:], hazard_ratios.values[0], color='skyblue')  # Use hazard_ratios.values[0] instead of hazard_ratios.values[1:]
    plt.xlabel('Hazard Ratio')
    plt.title('Hazard Ratios for Selected Features')
    plt.show()
else:
    print("No hazard ratios data to plot.")


# In[426]:


cox_model.check_assumptions(X_train, p_value_threshold = 0.05)


# In[427]:


from lifelines.statistics import proportional_hazard_test


# In[428]:


proportional_hazard_test(fitted_cox_model=cox_model, training_df=X_train, time_transform='log')


# In[434]:



c_index = cox_model.score(X_train, scoring_method="concordance_index")
print(f"Concordance Index: {c_index}")


# In[429]:



c_index = cox_model.score(X_test, scoring_method="concordance_index")
print(f"Concordance Index: {c_index}")


# In[430]:


test_predictions = cox_model.predict_expectation(X_test)


# In[431]:


# Add the predicted survival times to the test set
test_set_with_predictions = X_test.assign(predicted_survival=test_predictions.values)

# Sort the individuals by predicted survival time
sorted_test_set = test_set_with_predictions.sort_values(by='predicted_survival')

sorted_test_set.head(20)


# ### Clearly, the results can be improved. We can see that the data issues resulted in non logical results like the statistical tests output.

# 
# ## We can try other algorithms like random forest, logistic regression...

# In[ ]:


from sklearn.metrics import classification_report
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
# Prédiction sur l'ensemble de test
y_pred = rf_model.predict(X_test)

# Affichage du rapport de classification
print(classification_report(y_test, y_pred))


# In[ ]:


# Prédiction des probabilités de la classe positive
y_pred_proba = rf_model.predict_proba(X_test)[::,1]

# Calcul de la courbe ROC
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

# Calcul de l'aire sous la courbe ROC
roc_auc = auc(fpr, tpr)

# Tracé de la courbe ROC
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='Courbe ROC (aire = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taux de faux positifs')
plt.ylabel('Taux de vrais positifs')
plt.title('Courbe ROC')
plt.legend(loc="lower right")
plt.show()


# In[ ]:


lg_model = LogisticRegression()
lg_model.fit(X_train, y_train)
y_pred_lg = lg_model.predict(X_test)
print(classification_report(y_test, y_pred_lg))


# ## At the end, this was an opportunity to practise and build a default probibilty predictor from scratch using the Cox model, which is suitable for our data type (panel data)

# ### You can use this project for academic purposes
