import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn import svm

times = 20
x = np.linspace(-np.pi, times * np.pi, times * 201)
y = np.sin(x)

datums = {'x' : x, 'y' : y, 'label' : np.zeros(len(y))}

df = pd.DataFrame(data=datums)

df.loc[(df['y'] > 0.5) & (df['y'] < 1), 'label'] = 1

# print(df)
print(len(df), len(df[df['label'] == 1]))

clf = svm.SVC(kernel='linear')

train_x = df[['x', 'y']].as_matrix()
train_y = df['label'].values
print(train_x)
print(train_y)

clf.fit(train_x, train_y)
print(clf)

fit_y = clf.predict(train_x)

print(fit_y)

plt.plot(x, y)
plt.scatter(x, train_y)
plt.scatter(x, fit_y)
plt.xlabel('Angle [rad]')
plt.ylabel('sin(x)')
plt.axis('tight')
plt.show()