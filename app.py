from datetime import datetime
from numpy import percentile
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import squarify
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import r2_score
from catboost import CatBoostRegressor
import statsmodels.api as sm

cars = pd.read_csv('./dataset/germany-cars-zenrows.csv') # read dataset

print(cars.sample(frac=1).head(20)) # view dataset
print(cars.gear.unique()) # view gear types
print(cars.offerType.unique()) # view offers types
print(cars.describe(percentiles=[.01, .25, .5, .75, .99]).apply(lambda s: s.apply('{0:.2f}'.format))) # view describe of dataset

# visualisation car makes
makes = pd.DataFrame(cars.make.value_counts())
makes.reset_index(level=0, inplace=True)
makes = makes.sort_values(by='make', ascending=False).head(20)
makes.columns = ('make', 'size')

group = cars.groupby(cars.make)
mean_price = pd.DataFrame(group.price.mean())
mean_price.reset_index(level=0, inplace=True)
makes = pd.merge(makes, mean_price, how='left', on='make')

labels = ["%s\n%d items\nMean price:\n %d€" % (label) for label in zip(makes['make'], makes['size'], makes['price'])]
squarify.plot(sizes=makes['size'], label=labels, alpha=.8, color=plt.cm.tab20c.colors, edgecolor='white', linewidth=2)
plt.axis('off')

cars['fuel'] = cars['fuel'].replace(['Electric/Gasoline', 'Electric/Diesel', 'Electric'], 'Electric')
cars['fuel'] = cars['fuel'].replace(['CNG', 'LPG', 'Others', '-/- (Fuel)', 'Ethanol', 'Hydrogen'], 'Others')

fuels = pd.DataFrame(cars['fuel'].value_counts())
group = cars.groupby(cars['fuel'])
mean_price = pd.DataFrame(group.price.mean())
mean_price.reset_index(level=0, inplace=True)
fuels.reset_index(level=0, inplace=True)
fuels.columns = ('fuel', 'size')
fuels = pd.merge(fuels, mean_price, how='left', on='fuel')

labels = ["%s\n%d items\nMean price: %d€" % (label) for label in zip(fuels['fuel'], fuels['size'], fuels['price'])]
fig1, ax1 = plt.subplots()

# pie diagram
ax1.pie(fuels['size'], labels=labels, autopct='%1.1f%%', startangle=15, colors=plt.cm.Set1.colors)
ax1.axis('equal')

# bar diagram
sns.countplot(x='year', hue='fuel', data=cars)

# clear dataset
cars['age'] = datetime.now().year - cars['year']
cars = cars.drop('year', axis=1)

cars = cars.dropna()

makeDummies = pd.get_dummies(cars.make)
cars = cars.join(makeDummies)
cars = cars.drop('make', axis=1)

modelDummies = pd.get_dummies(cars.model)
# cars = cars.join(modelDummies)
# cars = cars.merge(modelDummies)
cars = cars.drop('model', axis=1)

fuelDummies = pd.get_dummies(cars.fuel)
# cars = cars.join(fuelDummies)
cars = cars.drop('fuel', axis=1)

cars = cars[stats.zscore(cars.price) < 3]
cars = cars[stats.zscore(cars.hp) < 3]
cars = cars[stats.zscore(cars.mileage) < 3]

offerTypeDummies = pd.get_dummies(cars.offerType)
cars = cars.join(offerTypeDummies)
cars = cars.drop('offerType', axis=1)

gearDummies = pd.get_dummies(cars.gear)
cars = cars.join(gearDummies)
cars = cars.drop('gear', axis=1)

# correlation
sns.heatmap(cars.corr(), annot=True, cmap='coolwarm')

# regression line
sns.set_theme(style='darkgrid')
sns.jointplot(x='hp', y='price', data=cars, kind='reg', color='m', line_kws={'color': 'red'})

sns.set_theme(style='darkgrid')
sns.jointplot(x='mileage', y='price', data=cars, kind='reg', color='m', line_kws={'color': 'red'})

# create train & test models
X = cars.drop('price', axis=1)
Y = cars.price
X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.7, test_size=0.3, random_state=100)

lm = linear_model.LinearRegression()
lm.fit(X_train, y_train)
y_pred = lm.predict(X_test)
print(r2_score(y_true=y_test, y_pred=y_pred))

model = CatBoostRegressor(iterations=6542, learning_rate=0.03)
model.fit(X_train, y_train, eval_set=(X_test, y_test))
print(model.score(X, Y))

# X = cars[['mileage', 'hp', 'age']]
# model = sm.OLS(Y, X).fit()
# prediction = model.predict(X)
# print(model.rsquared)

sorted_feature_importance = model.get_feature_importance().argsort()[-20:]
plt.barh(cars.columns[sorted_feature_importance], model.feature_importances_[sorted_feature_importance])
plt.xlabel('Feature Importance')

plt.show()

realData = pd.DataFrame.from_records([
  {'mileage': 87000, 'make': 'Volkswagen', 'model': 'Gold', 'fuel': 'Gasoline', 'gear': 'Manual', 'offerType': 'Used', 'price': 12990, 'hp': 125, 'year': 2015},
	{'mileage': 230000, 'make': 'Opel', 'model': 'Zafira Tourer', 'fuel': 'CNG', 'gear': 'Manual', 'offerType': 'Used', 'price': 5200, 'hp': 150, 'year': 2012},
	{'mileage': 5, 'make': 'Mazda', 'model': '3', 'hp': 122, 'gear': 'Manual', 'offerType': 'Employee\'s car', 'fuel': 'Gasoline', 'price': 20900, 'year': 2020}
])

realData = realData.drop('price', axis=1)
realData['age'] = datetime.now().year - realData['year']
realData = realData.drop('year', axis=1)

# all the other transformations and dummies go here
offerTypeDummies = pd.get_dummies(realData.offerType)
realData = realData.join(offerTypeDummies)
realData = realData.drop('offerType', axis=1)

gearDummies = pd.get_dummies(realData.gear)
realData = realData.join(gearDummies)
realData = realData.drop('gear', axis=1)

makeDummies = pd.get_dummies(realData.make)
realData = realData.join(makeDummies)
realData = realData.drop('make', axis=1)

modelDummies = pd.get_dummies(realData.model)
realData = realData.join(modelDummies)
realData = realData.drop('model', axis=1)

fuelDummies = pd.get_dummies(realData.fuel)
realData = realData.join(fuelDummies)
realData = realData.drop('fuel', axis=1)

fitModel = pd.DataFrame(columns=cars.columns)
fitModel = pd.concat([fitModel, realData], ignore_index=True)
fitModel = fitModel.fillna(0)

preds = model.predict(fitModel)
print(preds)  # [12213.35324984 5213.058479 20674.08838559]
