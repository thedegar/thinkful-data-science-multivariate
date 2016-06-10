import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

df = pd.read_csv('RawData\loans\loansData.csv', index_col=0)
df.dropna(inplace=True)

# Clean data
df['Interest.Rate'] = df['Interest.Rate'].apply(lambda x: x[:-1]).astype(float)
# Code categorical variable
df['Home.Ownership.Ord'] = pd.Categorical(df['Home.Ownership']).codes

# Predictor variables
X = df['Monthly.Income']
X1 = df[['Monthly.Income', 'Home.Ownership.Ord']]
X2 = df['Home.Ownership.Ord']
X = sm.add_constant(X)
X1 = sm.add_constant(X1)
X2 = sm.add_constant(X2)

# Response variable
y = df['Interest.Rate']

# fit a OLS model with intercept on Monthly Income
est = sm.OLS(y, X).fit()

# fit a OLS model with intercept on Monthly Income and Home Ownership
est1 = sm.OLS(y, X1).fit()

# fit a OLS model with intercept on Monthly Income and Home Ownership and their interaction
est2 = smf.ols('y ~ X + X2 + X*X2', data=df).fit()

# or

est2 = smf.ols('y ~ X:X2', data=df).fit()

# Both smf.ols('formulas') fail due to ValueError: For numerical factors, num_columns must be an int
# I found: https://groups.google.com/forum/#!topic/pystatsmodels/KcSzNqDxv-Q
# which implies it is due to patsy version 0.4.0.  I have not been successful upgrading to 0.4.1 in anaconda (ipython)
# plan to try at home on my own machine.

