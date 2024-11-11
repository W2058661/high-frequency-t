#!/usr/bin/env python
# coding: utf-8

# In[2]:


# import packages

import scipy.io as sio  # for loading matlab data

import numpy as np      # for numerical libs

from matplotlib.ticker import FuncFormatter # for custom bar plot labels
dat
import matplotlib.pyplot as plt  # for plotting

import matplotlib.pylab as pylab
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15, 5),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)


# In[3]:


# import packages

import scipy.io as sio  # for loading matlab data

import numpy as np      # for numerical libs

from matplotlib.ticker import FuncFormatter # for custom bar plot labels

import matplotlib.pyplot as plt  # for plotting

import matplotlib.pylab as pylab
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15, 5),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)


# In[4]:


data = sio.loadmat( 'FB_20141126.mat')


# In[5]:


LOB=data['LOB']

t = (np.array((LOB['EventTime'][0][0][:,0]))-3600000*9.5)*1e-3
bid = np.array(LOB['BuyPrice'][0][0]*1e-4)
bidvol = np.array(LOB['BuyVolume'][0][0]*1.0)
ask = np.array(LOB['SellPrice'][0][0]*1e-4)
askvol = np.array(LOB['SellVolume'][0][0]*1.0)
MO=np.array(LOB['MO'][0][0]*1.0)
dt = t[1]-t[0]


# In[6]:


midprice = 0.5*(bid[:,0]+ask[:,0])
microprice= (bid[:,0]*askvol[:,0]+ask[:,0]*bidvol[:,0])/(bidvol[:,0]+askvol[:,0])
spread = ask[:,0]-bid[:,0]


# In[7]:


plt.plot(t, microprice)
plt.title('Microprice')
plt.ylabel('price')
plt.xlabel('time (seconds from trading start)')
plt.show()


# In[8]:


plt.plot(t, microprice-midprice)
plt.title('Microprice - Midprice')
plt.ylabel('diff')
plt.xlabel('time (seconds from trading start)')
plt.show()


# In[9]:


plt.plot(t,spread)
plt.title('Spread')
plt.ylabel('spread')
plt.xlabel('time (seconds from trading start)')
plt.show()


# In[10]:


plt.hist(spread,bins=[0.01,0.02,0.03], width=0.001,align='mid') 
plt.title("Hist of spread")
plt.xlabel(r'spread')
plt.ylabel('Freq')
plt.show()


# In[11]:


rho = np.array((bidvol[:,0]-askvol[:,0])/(bidvol[:,0]+askvol[:,0]),ndmin=2).T


# In[12]:


plt.plot(t, rho)
plt.title('Imbalance')
plt.ylabel(r'$\rho$')
plt.xlabel('time (seconds from trading start)')

a = plt.axes([.65, .6, .2, .2])
idx = (t>3600) & (t<=3600+60)
plt.plot( t[idx], rho[idx])
plt.title('a')
plt.ylabel(r'$\rho$')
plt.xlabel(r'$t$')
plt.xticks([3600,3630,3660])

plt.show()


# In[13]:


plt.acorr(rho[:,0]-np.mean(rho[:,0]),maxlags=6000)  # maximum one minute 
plt.title('Autocorrelation of imbalance')
plt.xlim([0,6000])
plt.show()


# In[14]:


plt.hist(rho, bins=np.linspace(-1, 1, num=50)) 
plt.title("Hist of imbalance")
plt.xlabel(r'$\rho$')
plt.ylabel('Freq')
plt.show()


# In[46]:


# import packages

import scipy.io as sio  # for loading matlab data

import numpy as np      # for numerical libs

import pandas as pd



# In[47]:


data = sio.loadmat('FTSE_sample.mat')  #load matlab data


# In[48]:


bestask = np.squeeze(data['Bestask'])
bestbid = np.squeeze(data['Bestbid'])


# In[49]:


a=[bestbid,bestask]
columns=['bestbid','bestask']
data=pd.DataFrame(np.transpose(a),columns=columns)
data['mid']=(data['bestbid']+data['bestask'])/2


# In[ ]:





# In[50]:


data['returns']=np.log(data['mid'] / data['mid'].shift(1)) 
data['lreturns']= data['returns'].shift(1) 
# we now have the column of our dependent variable (return) and independent variable (lag return)


# In[51]:


import statsmodels.formula.api as smf


# In[52]:


model = smf.ols(formula='returns ~lreturns', data=data).fit()
#this is our simple linear regression model


# In[53]:


model.summary()


# In[54]:


import scipy.io as sio

# Load the .mat file
file_path = 'FTSE_sample.mat'
data = sio.loadmat(file_path)

# Display the structure and keys of the loaded data to understand its contents
data.keys()


# In[55]:


# Check the structure (shape) of each array to understand the time series length and dimensions
bestask = data['Bestask']
bestbid = data['Bestbid']
mo = data['MO']

# Display the shape of each array
bestask.shape, bestbid.shape, mo.shape


# In[56]:


import statsmodels.api as sm
net_order_flow = mo.flatten()

sm.graphics.tsa.plot_acf(net_order_flow, lags=20)


# In[57]:


import pandas as pd

# Assuming net_order_flow is a numpy array
net_order_flow_series = pd.Series(net_order_flow)

# Now you can apply the rolling sum
cumulative_flow = net_order_flow_series.rolling(window=30).sum()


# In[58]:


import matplotlib.pyplot as plt
import statsmodels.api as sm

# Flatten the 'MO' array to a 1D array for processing
net_order_flow = mo.flatten()

# Plot the autocorrelation function (ACF) up to 20 lags
lags = 20
plt.figure(figsize=(10, 5))
sm.graphics.tsa.plot_acf(net_order_flow, lags=lags, title="Autocorrelation of Net Order Flow (Up to 20 Lags)")
plt.xlabel("Lag")
plt.ylabel("Autocorrelation")
plt.show()


# In[64]:


import numpy as np
import pandas as pd

bestask = np.squeeze(data['Bestask'])
bestbid = np.squeeze(data['Bestbid'])


# Calculate mid prices and minutely returns using the extracted variables directly
mid_prices = (bestbid + bestask) / 2
minutely_returns = np.diff(mid_prices) / mid_prices[:-1]

# Define the rolling window for cumulative order flow (30 minutes)
window = 30

# Calculate cumulative order flows for the past 30 minutes
cumulative_order_flow = pd.Series(net_order_flow_series).rolling(window).sum().shift(1)  # Shift to avoid look-ahead bias

# Generate trading positions based on cumulative order flow
# 1 for long position, -1 for short position
positions = np.where(cumulative_order_flow > 0, 1, -1)

# Calculate strategy returns: if long, return as is; if short, invert the return
strategy_returns = positions[1:] * minutely_returns

# Calculate cumulative returns of the strategy and the stock itself
cumulative_strategy_return = np.cumsum(strategy_returns)
cumulative_stock_return = np.cumsum(minutely_returns[1:])

# Display the cumulative returns of both the strategy and the stock
print("Cumulative Return of Strategy:", cumulative_strategy_return[-1])
print("Cumulative Return of Stock:", cumulative_stock_return[-1])


# In[74]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

order_flows=np.squeeze(data['MO'])

# Define the rolling window for cumulative order flow (30 minutes)
window = 30

# Calculate cumulative order flows for the past 30 minutes
# Define the market orders series directly
cumulative_order_flow = pd.Series(market_orders).rolling(window).sum().shift(100)  # Shift to avoid look-ahead bias

# Generate trading positions based on cumulative order flow
# 1 for long position, -1 for short position
positions = np.where(cumulative_order_flow > 0, 1, -1)

# Calculate strategy returns: if long, return as is; if short, invert the return
strategy_returns = positions[1:] * minutely_returns

# Calculate cumulative returns of the strategy and the stock itself
cumulative_strategy_return = np.cumsum(strategy_returns)
cumulative_stock_return = np.cumsum(minutely_returns[1:])

# Plot the cumulative returns
plt.figure(figsize=(12, 6))
plt.plot(cumulative_strategy_return, label="Cumulative Return of Strategy", color="blue")
plt.plot(cumulative_stock_return, label="Cumulative Return of Stock", color="red")
plt.xlabel("Time (Minutes)")
plt.ylabel("Cumulative Return")
plt.title("Cumulative Return of Trading Strategy vs Stock")
plt.legend()
plt.grid(True)
plt.show()


# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Parameters
p = 0.2  # Probability V = VH
alpha = 0.8  # Proportion of informed traders
VH = 2  # High value
VL = 1  # Low value
time_steps = 50  # Number of simulation steps

# Initial probabilities
P_VH = p
P_VL = 1 - p

# Lists to store A and B values over time
ask_prices = []
bid_prices = []

for t in range(time_steps):
    # Probabilities of buy and sell given V=VH and V=VL
    P_Buy_given_VH = 0.5 + 0.5 * alpha
    P_Buy_given_VL = 0.5 * (1 - alpha)
    P_Sell_given_VH = 0.5 * (1 - alpha)
    P_Sell_given_VL = 0.5 + 0.5 * alpha

    # Total probability of Buy and Sell
    P_Buy = P_Buy_given_VH * P_VH + P_Buy_given_VL * P_VL
    P_Sell = P_Sell_given_VH * P_VH + P_Sell_given_VL * P_VL

    # Bayesian updates
    P_VH_given_Buy = (P_Buy_given_VH * P_VH) / P_Buy
    P_VL_given_Buy = (P_Buy_given_VL * P_VL) / P_Buy
    P_VH_given_Sell = (P_Sell_given_VH * P_VH) / P_Sell
    P_VL_given_Sell = (P_Sell_given_VL * P_VL) / P_Sell

    # Calculate Ask and Bid prices
    A = P_VH_given_Buy * VH + P_VL_given_Buy * VL
    B = P_VH_given_Sell * VH + P_VL_given_Sell * VL
    
    # Append prices for plotting
    ask_prices.append(A)
    bid_prices.append(B)
    
    # Randomly simulate a buy or sell action
    action = np.random.choice(["buy", "sell"])
    if action == "buy":
        P_VH, P_VL = P_VH_given_Buy, P_VL_given_Buy
    else:
        P_VH, P_VL = P_VH_given_Sell, P_VL_given_Sell

# Plotting Ask and Bid prices
plt.figure(figsize=(12, 6))
plt.plot(ask_prices, label="Ask Price (A)", color="blue")
plt.plot(bid_prices, label="Bid Price (B)", color="red")
plt.xlabel("Time Steps")
plt.ylabel


# In[ ]:




