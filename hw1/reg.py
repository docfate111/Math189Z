import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
raw=pd.read_csv('covid19-in-italy/covid19_italy_region.csv')
df=raw.groupby(['Date']).aggregate({'TotalPositiveCases': 'sum'})
#df.reset_index(level=0, inplace=True)
df.reset_index(level=0, inplace=True)
df['days since 2/24']=df.index
slope, intercept, r_value, p_value, std_err = stats.linregress(df['days since 2/24'], df['TotalPositiveCases'])
print("slope: %f   \nintercept: %f" % (slope, intercept))
print("R-squared: %f" % r_value**2)
print("P-value: %f" % p_value)
print("Std Error %f" % std_err)
x=df['days since 2/24']
y=df['TotalPositiveCases']
plt.xlabel('Days since 2/24')
plt.ylabel('Total Positive Cases')
plt.plot(x, y, 'o', label='original data')
plt.plot(x, intercept + slope*x, 'r', label='fitted line')
plt.title("Cases by day")
plt.show()
plt.close()
