import pandas
import numpy as np
import xgboost
import matplotlib.pyplot as pp
from dtreeviz.trees import dtreeviz

dummy = [
    'BusinessTravel',
    'Department',
    'EducationField',
    'JobRole',
    'MaritalStatus'
]
df = pandas.read_csv(
    'WA_Fn-UseC_-HR-Employee-Attrition.csv', index_col='EmployeeNumber')
df = pandas.get_dummies(df, columns=dummy)
df['Attrition'] = [int(df.loc[i,'Attrition'] == 'Yes') for i in df.index]
df['Gender'] = [int(df.loc[i,'Gender'] == 'Male') for i in df.index]
df['Over18'] = [int(df.loc[i,'Over18'] == 'Y') for i in df.index]
df['OverTime'] = [int(df.loc[i,'OverTime'] == 'Yes') for i in df.index]
df, y = df.drop(columns='Attrition'), df['Attrition']

test_set = np.random.permutation(df.index)
test_set = test_set[:len(df) // 4]
test_set = np.array([i in test_set for i in df.index])

soln = xgboost.XGBClassifier(max_depth=4, verbosity=2)
soln.fit(df[~test_set], y[~test_set])
test = soln.predict(df[test_set])
print(np.mean(test == y[test_set]))

xgboost.plot_importance(soln)
pp.show()
tree = dtreeviz(
    soln,
    df[~test_set],
    y[~test_set],
    feature_names=list(df.columns),
    target_name='Attrition',
    class_names=['Not likely to quit', 'Likely to quit'],
    tree_index=0,
    X=df.values[0]
)
tree.save('Tree.svg')