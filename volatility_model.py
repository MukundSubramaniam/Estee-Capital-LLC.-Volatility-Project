import pandas as pd
import math
from scipy.stats import norm
import matplotlib.pyplot as plt

################################################################################

#Black-Scholes Formula
def d(sigma, S, K, r, t):
    d1 = 1 / (sigma * math.sqrt(t)) * (math.log(S/K) + (r + sigma **2/2)* t)
    d2 = d1 - sigma * math.sqrt(t)
    return (d1, d2)

def call_price(sigma, S, K, r ,t, d1, d2):
    C = norm.cdf(d1) * S - norm.cdf(d2) * K * math.exp(-r * t)
    return C
    
#Calculates Implied Volatility
def getSigma(S, K, t, r, c0):
    
    #Tolerances
    tol = math.e**-3
    epsilon = 1
    
    count = 0
    max_iter = 1000
    
    vol = 0.50
    
    while epsilon > tol:
        count += 1
        if count >= max_iter:
            print('Breaking on count')
            break
    
        orig_vol = vol
        d1, d2 = d(vol, S, K, r, t)
        function_value_call = call_price(vol, S, K, r, t, d1, d2) - c0
        vega = S * norm.pdf(d1) * math.sqrt(t)
        
        if vega != 0:
            #Newton's Method
            vol = -function_value_call / vega + vol
        
        epsilon = abs((vol - orig_vol) / orig_vol)
        
    return vol

def readFile(path):
    with open(path, "rt") as f:
        return f.read()

def graphVolatility(L, T):
    fig = plt.figure()
    fig.suptitle('Volatility of Call Options with 11300 Strike')
    #plt.plot(L[0], label = 'Implied Volatility (Ask Price)')
    #plt.plot(L[1], label = 'Implied Volatility (Bid Price)')
    plt.plot(T[0], label = 'Realized Volatility')
    plt.legend()
    plt.xlabel('Index')
    plt.ylabel('Volatility')
    plt.show()

def impliedVolatility(L):
    K = 11300.0
    r = 0.069
    IVListAsk = []
    IVListBid = []
    prices = []
    returns = []
    RVDict = dict()
    IVolSeries = pd.DataFrame()
    HVolSeries = pd.DataFrame()
    prevStock = None
    count = 0
    print(len(L))
    for i in range(13,len(L), 7776):
        count += 1
        
        t = (4.0 - ((float(L[i][11:13]) - 9) * 3600 + (float(L[i][14:16]) - 15) 
        * 60 + float(L[i][17:19]) + float(L[i][19:]))/22500)/ 365.0
        
        S = (float(L[i+1])+float(L[i+2]))/2 * (1/(1 + r)**t)
        c0 = float(L[i+3])
        
        IVListAsk.append(getSigma(S, K, t, r, c0))
        IVListBid.append(getSigma(S, K, t, r, float(L[i+4])))
        
        if prevStock != None:
            prices.append(S)
            
        if count % 280 == 0:
            print(returns)
            RVDict[count] = (pd.Series(returns).std() * math.sqrt(252)) 
            returns = []

        '''
        if len(prices) != 0:
            avg = sum(prices)/len(prices)
            
        for j in range(len(prices)):
            returns.append((prices[j] - avg)**2)
        
        if len(returns) != 0:
            RVList.append(math.sqrt(sum(returns)/len(returns)))
'''
        returns = []
        prevStock = S
        
    IVolSeries[0] = IVListAsk[1:]
    IVolSeries[1] = IVListBid[1:]
    HVolSeries[0] = RVDict
    graphVolatility(IVolSeries, HVolSeries)

impliedVolatility(readFile('/Users/mukundsubramaniam/Downloads/11300Call.csv').split(','))




