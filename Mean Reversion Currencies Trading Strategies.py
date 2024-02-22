
'If you don’t have any of these libraries please paste the command related to the library you need into an Anaconda terminal.'
# ex : pip install arch_model

#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from arch import arch_model
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor


def dataPrep(data):
    # Calculate daily variations
    data['Variations'] = data['Close'].diff()
    data['Return'] = data['Close'].pct_change()
    # Add a constant term for the intercept in OLS regression
    data['Constant'] = 1

    data['Lagged_X'] = data['Close'].shift(1)

    if np.any(np.isinf(data['Close'])) or np.any(np.isnan(data['Close'])):
        data['Close'] = data['Close'].replace([np.inf, -np.inf], np.nan)
        data['Close'].fillna(method='ffill', inplace=True)
    data.dropna(inplace= True)

def OLSRegression(endo, exog):
    # Perform OLS regression
    model = sm.OLS(endo, exog)
    results = model.fit()

    # Extract alpha (intercept) and beta (slope) coefficients
    alpha, beta = results.params['Constant'], results.params['Lagged_X']

    # Extract t-statistic for the beta coefficient
    t_stat_beta = results.tvalues['Lagged_X']

    # Print results
    print(results.summary())

    print("\n Alpha (Intercept):", alpha)
    print("Beta (Slope):", beta)
    print("T-Statistic for Beta:", t_stat_beta)

    # Test for mean reversion based on the t-statistic
    critical_value = -1  # You can change this value as needed

    if t_stat_beta < critical_value:
        print("\n Beta is statiscally significant, The full sample exhibits mean reversion.")
    else:
        print("\n Beta is not statiscally significant, The full sample does not exhibit mean reversion.")

def TradingStrat(estimation_window, critical_value, data):
    ListOfSignals = [0] * estimation_window
    # Implement the trading strategy
    for i in range(estimation_window, len(data)):
        # Extract the sub-sample for estimation
        sub_sample = data.iloc[i - estimation_window:i]

        # Perform OLS regression for mean reversion
        model = sm.OLS(sub_sample['Variations'], sub_sample[['Constant', 'Lagged_X']])
        results = model.fit()

        # Extract alpha (intercept) and beta (slope) coefficients
        alpha_hat, beta_hat = results.params['Constant'], results.params['Lagged_X']

        # Extract t-statistic for the beta coefficient
        t_stat_beta = results.tvalues['Lagged_X']

        # Check mean reversion conditions
        if alpha_hat > 0 and beta_hat < 0 and t_stat_beta < critical_value:
            # Compute the expected change in exchange rate
            expected_change = alpha_hat + beta_hat * data['Close'].iloc[i]

            # Make a trading decision based on the expected change
            if expected_change > 0:
                ListOfSignals.append(1) # Buy Euro
            elif expected_change < 0:
                ListOfSignals.append(-1) # Buy USD
            else:
                ListOfSignals.append(0) # Do nothing
        else:
            ListOfSignals.append(0)
                
    return ListOfSignals

def CalcMetrics(ListOfSignalsShifted, ListOfReturn, confidence_level):
    ListOfReturns = ListOfReturn * ListOfSignalsShifted
    ListOfCumulativePerformance = (1+ListOfReturns).cumprod()
    
    daily_returns = ListOfReturns.dropna()
    sharpe_ratio = np.sqrt(252) * (daily_returns.mean() / daily_returns.std())
    max_drawdown = (ListOfCumulativePerformance / ListOfCumulativePerformance.cummax() - 1).min()

    #Sortino ratio 
    excess_return = daily_returns #take the returns
    negativ_yields = excess_return[excess_return < 0]#only take the negativ returns in one hand
    vol_negativ = np.std(negativ_yields)#comput the negativs yield
    sortino_ratio = np.mean(excess_return)/vol_negativ #sortino ratio
    
    #Calmar Ratio
    compound_annual_yield = np.mean(daily_returns)*252 #annual yield/return
    calmar_ratio = compound_annual_yield/abs(max_drawdown)#Calmar ratio

    
    #Value at risk -> take the value of the quantile with the confidence level
    sorted_returns = ListOfReturns.sort_values()
    ValueAtRisk = -sorted_returns.quantile(confidence_level)
    
    # Calculate CVaR
    CValueAtRisk = -sorted_returns[sorted_returns <= ValueAtRisk].mean()
    
    return ListOfReturns, ListOfCumulativePerformance, sharpe_ratio, max_drawdown, ValueAtRisk, CValueAtRisk, sortino_ratio,calmar_ratio

def garchModel(ListOfVariations):
    model_garch = arch_model(ListOfVariations * 100, vol='Garch', p=1, q=1)
    results_garch = model_garch.fit()
    
    print(results_garch.summary())
    
    ListOfVolatilities = results_garch.conditional_volatility
    return ListOfVolatilities

def TradingStratWithVolatility(estimation_window, critical_value, data, stop_loss, take_profit):
    ListOfSignals = [0] * 80
    for i in range(estimation_window, len(data)):
        # Extract the sub-sample for estimation
        sub_sample = data.iloc[i - estimation_window:i]

        # Perform OLS regression for mean reversion
        model_v2 = sm.OLS(sub_sample['Variations'], sub_sample[['Constant', 'Lagged_X']])
        resultsVolat = model_v2.fit()

        # Extract alpha (intercept) and beta (slope) coefficients
        alpha_hat, beta_hat = resultsVolat.params['Constant'], resultsVolat.params['Lagged_X']

        # Extract t-statistic for the beta coefficient
        t_stat_beta = resultsVolat.tvalues['Lagged_X']

        # Check mean reversion conditions
        if alpha_hat > 0 and beta_hat < 0 and t_stat_beta < critical_value:
            # Compute the expected change in exchange rate
            expected_change = alpha_hat + beta_hat * data['Close'].iloc[i]

            if expected_change > 0 and expected_change > take_profit * dataCustomStrat['Volatility_Conditional'].iloc[i]:
                ListOfSignals.append(1)
            elif expected_change < 0 and expected_change < stop_loss * dataCustomStrat['Volatility_Conditional'].iloc[i]:
                ListOfSignals.append(-1)
            else:
                ListOfSignals.append(0)
        else:
            ListOfSignals.append(0)
            
    return ListOfSignals

# Function to load and preprocess data for a given currency pair
def load_and_prep_data(file_path):
    data = pd.read_csv(file_path, parse_dates=True, index_col='Date')
    dataPrep(data)
    return data

def mean_reversion_strategy(data, estimation_window, critical_value, confidence_level):
    data['Signal'] = TradingStrat(estimation_window, critical_value, data)
    data['Strategy_Return'], data['Cumulative_Performance'], sharpe_ratio, max_drawdown, ValueAtRisk, CValueAtRisk,sortino_ratio,calmar_ratio = CalcMetrics(data['Signal'].shift(1), data['Return'], confidence_level)
    return sharpe_ratio, max_drawdown, ValueAtRisk, CValueAtRisk,sortino_ratio,calmar_ratio

# Function to implement the joint trading strategy
def joint_trading_strategy(estimation_window, critical_value, data, eur_gbp_data, gbp_usd_data, confidence_level):

    # Create a DataFrame for joint strategy
    joint_data = pd.DataFrame(index=data.index)
    
    joint_data['Strategy_Return'] = (data['Strategy_Return'] + gbp_usd_data['Strategy_Return'] + eur_gbp_data['Strategy_Return'] + aud_usd_data['Strategy_Return']) / 4
    joint_data['Cumulative_Performance'] = (data['Cumulative_Performance'] + gbp_usd_data['Cumulative_Performance'] + eur_gbp_data['Cumulative_Performance'] + aud_usd_data['Cumulative_Performance']) / 4
    
    
    daily_returns = joint_data['Strategy_Return'].dropna()
    sharpe_ratio = np.sqrt(252) * (daily_returns.mean() / daily_returns.std())
    max_drawdown = (joint_data['Cumulative_Performance'] / joint_data['Cumulative_Performance'].cummax() - 1).min()
    
    #Sortino ratio 
    excess_return = daily_returns #take the returns
    negativ_yields = excess_return[excess_return < 0]
    vol_negativ = np.std(negativ_yields)
    sortino_ratio = np.mean(excess_return)/vol_negativ
    
    #Calmar Ratio
    compound_annual_yield = np.mean(daily_returns)*252
    calmar_ratio = compound_annual_yield/abs(max_drawdown)
    
    sorted_returns = joint_data['Strategy_Return'].sort_values()
    
    ValueAtRisk = -sorted_returns.quantile(confidence_level)
    
    # Calculate CVaR
    CValueAtRisk = -sorted_returns[sorted_returns <= ValueAtRisk].mean()

    return joint_data, sharpe_ratio, max_drawdown, ValueAtRisk, CValueAtRisk,sortino_ratio,calmar_ratio


def joint_trading_strategy2(estimation_window, critical_value, data, eur_gbp_data, gbp_usd_data, aud_usd_data, confidence_level):

    returns = pd.concat([data['Return'], eur_gbp_data['Return'], gbp_usd_data['Return'], aud_usd_data['Return']], axis=1)
    returns.columns = ['EUR/USD', 'EUR/GBP', 'GBP/USD', 'AUD/USD']
    returns.dropna(inplace=True)
    
    variations = pd.concat([data['Variations'], eur_gbp_data['Variations'], gbp_usd_data['Variations'], aud_usd_data['Variations']], axis=1)
    variations.columns = ['EUR/USD', 'EUR/GBP', 'GBP/USD', 'AUD/USD']
    variations.dropna(inplace=True)

    lagged_prices = pd.concat([data['Lagged_X'], eur_gbp_data['Lagged_X'], gbp_usd_data['Lagged_X'], aud_usd_data['Lagged_X']], axis=1)
    lagged_prices.columns = ['EUR/USD', 'EUR/GBP', 'GBP/USD', 'AUD/USD']
    lagged_prices.dropna(inplace=True)


    prices = pd.concat([data['Close'], eur_gbp_data['Close'], gbp_usd_data['Close'], aud_usd_data['Close']], axis=1)
    prices.columns = ['EUR/USD', 'EUR/GBP', 'GBP/USD', 'AUD/USD']
    prices.dropna(inplace=True)

    perfs = pd.concat([data['Cumulative_Performance'], eur_gbp_data['Cumulative_Performance'], gbp_usd_data['Cumulative_Performance'], aud_usd_data['Cumulative_Performance']], axis=1)
    perfs.columns = ['EUR/USD', 'EUR/GBP', 'GBP/USD', 'AUD/USD']
    perfs.dropna(inplace=True)
    
    correlation_matrix = returns.corr()

    threshold = 0.2 #define a threshold for the correlation ex if correlation is higher than this threshold

    # Initialize a dictionary to store trading signals for each pair

    trading_signals = {}
    # Loop through pairs and generate trading signals

    for pair in variations.columns:
        
        signal_key = f'Signal_{pair}'

        for other_pair in variations.columns:

        # Generate trading signal based on correlation and OLS regression

            if correlation_matrix.loc[pair, other_pair] > threshold:

                    X = sm.add_constant(lagged_prices[[other_pair]])
                    y = variations[[pair]]
                    model = sm.OLS(y, X).fit()
                    expected_change = model.params['const'] + model.params[other_pair] * prices[other_pair]
                    trading_signals[signal_key] = np.where(expected_change > 0, 1, 0)

    # Create a DataFrame to hold trading positions
    positions = pd.DataFrame(index=returns.index)

    # Loop through trading signals and create positions

    for pair in returns.columns:
        signal_key = f'Signal_{pair}'
        position_key = f'Position_{pair}'
    
        positions[position_key] = np.where(trading_signals[signal_key] == 1, 1, -1)
    # Calculate portfolio returns
    portfolio_returns = positions['Position_EUR/USD'] * returns['EUR/USD'] + positions['Position_EUR/GBP'] * returns['EUR/GBP'] + positions['Position_GBP/USD'] * returns['GBP/USD'] + positions['Position_AUD/USD'] * returns['AUD/USD']

    # Calculate cumulative portfolio returns
    cumulative_portfolio_returns = (1 + portfolio_returns).cumprod()

    daily_returns = portfolio_returns.dropna()
    
    sharpe_ratio = np.sqrt(252) * (daily_returns.mean() / daily_returns.std())
    max_drawdown = (cumulative_portfolio_returns / cumulative_portfolio_returns.cummax() - 1).min()
    sorted_returns = portfolio_returns.sort_values()
    ValueAtRisk = -sorted_returns.quantile(confidence_level)

    # Calculate CVaR

    CValueAtRisk = -sorted_returns[sorted_returns <= ValueAtRisk].mean()

    #Sortino ratio 

    excess_return = daily_returns #take the returns

    negativ_yields = excess_return[excess_return < 0]

    vol_negativ = np.std(negativ_yields)

    sortino_ratio = np.mean(excess_return)/vol_negativ

    #Calmar Ratio
    compound_annual_yield = np.mean(daily_returns)*252
    calmar_ratio = compound_annual_yield/abs(max_drawdown)
    
    plt.figure(figsize=(10, 6))
    plt.plot(variations.index, cumulative_portfolio_returns, label='Joint Strategy Cumulative Performance', color='blue')
    plt.title('Test Joint Strategy Cumulative Performance')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Performance')
    plt.legend()
    plt.show()
    
    return sharpe_ratio, max_drawdown, ValueAtRisk, CValueAtRisk, sortino_ratio, calmar_ratio


#Question 1

data = load_and_prep_data('EURUSD=X.csv')
print("---------------------------EUR/USD datas ----------------------------")
print(data)

plt.figure(figsize=(10, 6))
plt.plot(data['Close'], label='EUR/USD Exchange Rate')
plt.title('EUR/USD Exchange Rate Time Series between 2010 and 2023')
plt.legend()
plt.show()

OLSRegression(data['Variations'], data[['Constant', 'Lagged_X']])

#Question 2

# Define parameters for the trading strategy
estimation_window = 80
critical_value = -1
critical_value2 = -0.30
confidence_level = 0.05

data['Signal'] = TradingStrat(estimation_window, critical_value, data)
data['Strategy_Return'], data['Cumulative_Performance'], sharpe_ratio, max_drawdown, ValueAtRisk, CValueAtRisk, sortino_ratio,calmar_ratio = CalcMetrics(data['Signal'].shift(1), data['Return'], confidence_level)

plt.figure(figsize=(10, 6))
plt.plot(data['Cumulative_Performance'])
plt.title('Cumulative Performance of the trading strategy over time')
plt.xlabel('Date')
plt.ylabel('Value of holdings')
plt.legend()
plt.show()

print(' ---- METRICS for the Initial Strategy: -------')
print(f'Criticial value =  : {critical_value}')
print(f'Sharpe Ratio: {sharpe_ratio:.4f}')
print(f'Maximum Drawdown: {max_drawdown:.4f}')
print(f'Sortino Ratio : {sortino_ratio:.4f}')
print(f'Calmar ratio: {calmar_ratio:.4f}')
print(f'Value at risk with 95% confidence level: {ValueAtRisk:.4f}')
print(f'Condiational Value at risk: {CValueAtRisk:.4f}')



#Question 3

#Changing the critical value parameter

critical_value2 = -0.30
data['Signal2'] = TradingStrat(estimation_window, critical_value2, data)
data['Strategy_Return2'], data['Cumulative_Performance2'], sharpe_ratio2, max_drawdown2,  ValueAtRisk2, CValueAtRisk2,sortino_ratio2,calmar_ratio2 = CalcMetrics(data['Signal2'].shift(1), data['Return'], confidence_level)

print('Initial Strategy with a different critical value:')
print(f'Criticial value =  : {critical_value2}')
print(f'Sharpe Ratio: {sharpe_ratio2:.4f}')
print(f'Maximum Drawdown: {max_drawdown2:.4f}')
print(f'Sortino Ratio : {sortino_ratio2:.4f}')
print(f'Calmar ratio: {calmar_ratio2:.4f}')
print(f'Value at risk with 95% confidence level: {ValueAtRisk2:.4f}')
print(f'Condiational Value at risk: {CValueAtRisk2:.4f}')
#Here we plot the new strat
plt.plot(data['Cumulative_Performance2'])
plt.title('Cumulative Performance of the trading strategy over time ( with critical value of -0.3')
plt.xlabel('Date')
plt.ylabel('Value of holdings')
plt.legend()
plt.show()

#Changing the estimation window
estimation_window = 60 # we copy paste the baic code of the question 2 and replace this parameter.
data['Signal3'] = TradingStrat(estimation_window, critical_value, data)
data['Strategy_Return3'], data['Cumulative_Performance3'], sharpe_ratio3, max_drawdown3,  ValueAtRisk3, CValueAtRisk3,sortino_ratio3,calmar_ratio3 = CalcMetrics(data['Signal3'].shift(1), data['Return'], confidence_level)

print('Initial strategy with 60 days estim:ation window')
print(f'Criticial value =  : {critical_value}')
print(f'Sharpe Ratio: {sharpe_ratio3:.4f}')
print(f'Maximum Drawdown: {max_drawdown3:.4f}')
print(f'Sortino Ratio : {sortino_ratio3:.4f}')
print(f'Calmar ratio: {calmar_ratio3:.4f}')
print(f'Value at risk with 95% confidence level: {ValueAtRisk3:.4f}')
print(f'Condiational Value at risk: {CValueAtRisk3:.4f}')
#Here we plot the new strat
plt.plot(data['Cumulative_Performance3'])
plt.title('Cumulative Performance of the trading strategy over time ( with esimation window of 60 days')
plt.xlabel('Date')
plt.ylabel('Value of holdings')
plt.legend()
plt.show()

#Changing the strategy by adding volatility as a parameter for the function

estimation_window = 80 #reput 80 days for the estimation window

dataCustomStrat = data.copy()
dataCustomStrat['Volatility_Conditional'] = garchModel(dataCustomStrat['Variations'])
stop_loss = -0.005  # Stop-loss
take_profit = 0.005 # Take-profit

dataCustomStrat['Signal'] = TradingStratWithVolatility(estimation_window, critical_value, data, stop_loss, take_profit)

dataCustomStrat['Strategy_Return'], dataCustomStrat['Cumulative_Performance'], sharpe_ratioCustom, max_drawdownCustom, ValueAtRiskCustom, CValueAtRiskCustom,sortino_ratioCustom,calmar_ratioCustom = CalcMetrics(dataCustomStrat['Signal'].shift(1),dataCustomStrat['Return'], confidence_level)
print('Custom Strategy including volatility:')
print(f'Sharpe Ratio: {sharpe_ratioCustom:.4f}')
print(f'Maximum Drawdown: {max_drawdownCustom:.4f}')
print(f'Sortino Ratio : {sortino_ratioCustom:.4f}')
print(f'Calmar ratio: {calmar_ratioCustom:.4f}')
print(f'Value at risk with 95% confidence level: {ValueAtRiskCustom:.4f}')
print(f'Condiational Value at risk: {CValueAtRiskCustom:.4f}')
plt.plot(dataCustomStrat['Cumulative_Performance'])
plt.title('Cumulative Performance of the trading strategy with volatility')
plt.xlabel('Date')
plt.ylabel('Value of holdings')
plt.legend()
plt.show()

#Changing the strategy by using Sklearn Linear Regression model to predict the performance

dataTrain = load_and_prep_data('EURUSD=X-2.csv')

plt.figure(figsize=(10, 6))
plt.plot(dataTrain['Close'], label='EUR/USD Exchange Rate')
plt.title('EUR/USD Exchange Rate Time Series between 2003 and 2009 used to train the Ml models')
plt.legend()
plt.show()

dataMl = data.copy()

X_train = dataTrain[['Constant', 'Lagged_X']]
y_train = dataTrain['Variations']

X_test = dataMl[['Constant', 'Lagged_X']]
y_test = dataMl['Variations']

modelMl = LinearRegression()
modelMl.fit(X_train, y_train)

y_pred = modelMl.predict(X_test)

dataMl.loc[y_pred > 0, 'Signal'] = 1  # Buy Euro
dataMl.loc[y_pred < 0, 'Signal'] = -1  # Buy USD

dataMl['Strategy_Return'], dataMl['Cumulative_Performance'], sharpe_ratioMl, max_drawdownMl, ValueAtRiskLR, CValueAtRiskLR,sortino_ratioLR,calmar_ratioLR = CalcMetrics(dataMl['Signal'].shift(1), dataMl['Return'], confidence_level)
plt.plot(dataMl['Cumulative_Performance'])
plt.title('Cumulative Performance of the trading strategy with Sklearn')
plt.xlabel('Date')
plt.ylabel('Value of holdings')
plt.legend()
plt.show()
print('------------------Linear regression:--------------------')
print(f'Sharpe Ratio: {sharpe_ratioMl:.4f}')
print(f'Maximum Drawdown: {max_drawdownMl:.4f}')
print(f'Sortino Ratio : {sortino_ratioLR:.4f}')
print(f'Calmar ratio: {calmar_ratioLR:.4f}')
print(f'Value at risk with 95% confidence level: {ValueAtRiskLR:.4f}')
print(f'Condiational Value at risk: {CValueAtRiskLR:.4f}')




#Changing the strategy by using Sklearn Gradient boosting Regressor to predict the performance

modelMl2 = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
modelMl2.fit(X_train, y_train)

y_predGradBoost = modelMl2.predict(X_test)

dataMl.loc[y_predGradBoost > 0, 'Signal2'] = 1  # Buy Euro
dataMl.loc[y_predGradBoost < 0, 'Signal2'] = -1  # Buy USD

dataMl['Strategy_Return2'], dataMl['Cumulative_Performance2'], sharpe_ratioMl2, max_drawdownMl2, ValueAtRiskGDB, CValueAtRiskGDB,sortino_ratioGDB,calmar_ratioGDB = CalcMetrics(dataMl['Signal2'].shift(1), dataMl['Return'], confidence_level)
plt.plot(dataMl['Cumulative_Performance2'])
plt.title('Cumulative Performance of the trading strategy with Sklearn Boosting Regressor')
plt.xlabel('Date')
plt.ylabel('Value of holdings')
plt.legend()
plt.show()
print('-------------------GradientBoostingRegressor: -----------')
print(f'Sharpe Ratio: {sharpe_ratioMl2:.4f}')
print(f'Maximum Drawdown: {max_drawdownMl2:.4f}')
print(f'Sortino Ratio : {sortino_ratioGDB:.4f}')
print(f'Calmar ratio: {calmar_ratioGDB:.4f}')
print(f'Value at risk with 95% confidence level: {ValueAtRiskGDB:.4f}')
print(f'Condiational Value at risk: {CValueAtRiskGDB:.4f}')

#PLOTING the differents strategy

plt.figure(figsize=(15, 7))
plt.plot(data.index, data['Cumulative_Performance3'], label='Strategy Cumulative Performance for estimation window = 60 ', color='blue')
plt.plot(data.index, data['Cumulative_Performance2'], label=f'Strategy Cumulative Performance for crit value = {critical_value2}', color='red')
plt.plot(data.index, data['Cumulative_Performance'], label=f'Strategy Cumulative Performance for crit value = {critical_value}', color = 'black')
plt.plot(dataMl.index, dataMl['Cumulative_Performance'], label='Strategy Cumulative Performance for Linear regressor',color='gray')
plt.plot(dataMl.index, dataMl['Cumulative_Performance2'], label='Strategy Cumulative Performance for GradientBoosting regressor', color="orange")
plt.plot(dataCustomStrat.index, dataCustomStrat['Cumulative_Performance'], label='Strategy Cumulative Performance with Volatility threshold', color='green')
plt.title('Mean Reversion Trading Strategy Cumulative Performance')
plt.xlabel('Date')
plt.ylabel('Cumulative Performance')
plt.legend()
plt.show()

#RECAP
print("---------------START OF THE RECAP OF  THE METRICS OF THE STRATEGIES ----------------")
print(' ---- METRICS for the Initial Strategy: -------')
print(f'Criticial value =  : {critical_value}')
print(f'Sharpe Ratio: {sharpe_ratio:.4f}')
print(f'Maximum Drawdown: {max_drawdown:.4f}')
print(f'Sortino Ratio : {sortino_ratio:.4f}')
print(f'Calmar ratio: {calmar_ratio:.4f}')
print(f'Value at risk with 95% confidence level: {ValueAtRisk:.4f}')
print(f'Condiational Value at risk: {CValueAtRisk:.4f}')

print('-----------------Initial Strategy with a different critical value: -----------------------------')
print(f'Criticial value =  : {critical_value2}')
print(f'Sharpe Ratio: {sharpe_ratio2:.4f}')
print(f'Maximum Drawdown: {max_drawdown2:.4f}')
print(f'Sortino Ratio : {sortino_ratio2:.4f}')
print(f'Calmar ratio: {calmar_ratio2:.4f}')
print(f'Value at risk with 95% confidence level: {ValueAtRisk2:.4f}')
print(f'Condiational Value at risk: {CValueAtRisk2:.4f}')

print('-------------------Initial strategy with 60 days estim:ation window-----------------')
print(f'Criticial value =  : {critical_value}')
print(f'Sharpe Ratio: {sharpe_ratio3:.4f}')
print(f'Maximum Drawdown: {max_drawdown3:.4f}')
print(f'Sortino Ratio : {sortino_ratio3:.4f}')
print(f'Calmar ratio: {calmar_ratio3:.4f}')
print(f'Value at risk with 95% confidence level: {ValueAtRisk3:.4f}')
print(f'Condiational Value at risk: {CValueAtRisk3:.4f}')

print('--------------Custom Strategy including volatility: -------------------')
print(f'Sharpe Ratio: {sharpe_ratioCustom:.4f}')
print(f'Maximum Drawdown: {max_drawdownCustom:.4f}')
print(f'Sortino Ratio : {sortino_ratioCustom:.4f}')
print(f'Calmar ratio: {calmar_ratioCustom:.4f}')
print(f'Value at risk with 95% confidence level: {ValueAtRiskCustom:.4f}')
print(f'Condiational Value at risk: {CValueAtRiskCustom:.4f}')

print('------------------Linear regression:--------------------')
print(f'Sharpe Ratio: {sharpe_ratioMl:.4f}')
print(f'Maximum Drawdown: {max_drawdownMl:.4f}')
print(f'Sortino Ratio : {sortino_ratioLR:.4f}')
print(f'Calmar ratio: {calmar_ratioLR:.4f}')
print(f'Value at risk with 95% confidence level: {ValueAtRiskLR:.4f}')
print(f'Condiational Value at risk: {CValueAtRiskLR:.4f}')

print('-------------------GradientBoostingRegressor: -----------')
print(f'Sharpe Ratio: {sharpe_ratioMl2:.4f}')
print(f'Maximum Drawdown: {max_drawdownMl2:.4f}')
print(f'Sortino Ratio : {sortino_ratioGDB:.4f}')
print(f'Calmar ratio: {calmar_ratioGDB:.4f}')
print(f'Value at risk with 95% confidence level: {ValueAtRiskGDB:.4f}')
print(f'Condiational Value at risk: {CValueAtRiskGDB:.4f}')

print('----------------- END OF RECAP --------------------- ')
#Question 4

#Using the initial strategy with the pair GBP/EUR

eur_gbp_data = load_and_prep_data('GBPEUR=X.csv')

plt.figure(figsize=(10, 6))
plt.plot(eur_gbp_data['Close'], label='GBP/EUR Exchange Rate', color='green')
plt.title('GBP/EUR Exchange Rate Time Series between 2010 and 2023')
plt.legend()
plt.show()

sharpe_ratio_eur_gbp, max_drawdown_eur_gbp, ValueAtRisk_eur_gbp, CValueAtRisk_eur_gbp,sortino_ratio_eur_gbp,calmar_ratio_eur_gbp = mean_reversion_strategy(eur_gbp_data, estimation_window, critical_value, confidence_level)
print('--------------------Initial Strategy with GBP/EUR :---------------------------')
print(f'Sharpe Ratio: {sharpe_ratio_eur_gbp:.4f}')
print(f'Maximum Drawdown: {max_drawdown_eur_gbp:.4f}')
print(f'Sortino Ratio : {sortino_ratio_eur_gbp:.4f}')
print(f'Calmar ratio: {calmar_ratio_eur_gbp:.4f}')
print(f'Value at risk with 95% confidence level: {ValueAtRisk_eur_gbp:.4f}')
print(f'Condiational Value at risk: {CValueAtRisk_eur_gbp:.4f}')
plt.plot(eur_gbp_data.index, eur_gbp_data['Cumulative_Performance'], label='Strategy Cumulative Performance for GBP/EUR', color='red')
plt.title('Cumulative Performance of the trading strategy ')
plt.xlabel('Date')
plt.ylabel('Value of holdings')
plt.legend()
plt.show()
#Using the initial strategy with the pair GBP/USD

gbp_usd_data = load_and_prep_data('GBPUSD=X.csv')

plt.figure(figsize=(10, 6))
plt.plot(gbp_usd_data['Close'], label='GBP/USD Exchange Rate', color='orange')
plt.title('GBP/USD Exchange Rate Time Series between 2010 and 2023')
plt.legend()
plt.show()

sharpe_ratio_gbp_usd, max_drawdown_gbp_usd, ValueAtRisk_gbp_usd, CValueAtRisk_gbp_usd,sortino_ratio_gbp_usd,calmar_ratio_gbp_usd = mean_reversion_strategy(gbp_usd_data, estimation_window, critical_value, confidence_level)

print('-----------Initial Strategy with GBP/USD :-----------------------')
print(f'Sharpe Ratio: {sharpe_ratio_gbp_usd:.4f}')
print(f'Maximum Drawdown: {max_drawdown_gbp_usd:.4f}')
print(f'Sortino Ratio : {sortino_ratio_gbp_usd:.4f}')
print(f'Calmar ratio: {calmar_ratio_gbp_usd:.4f}')
print(f'Value at risk with 95% confidence level: {ValueAtRisk_gbp_usd:.4f}')
print(f'Condiational Value at risk: {CValueAtRisk_gbp_usd:.4f}')
plt.plot(gbp_usd_data.index, gbp_usd_data['Cumulative_Performance'], label='Strategy Cumulative Performance for GBP/USD', color="red")
plt.title('Cumulative Performance of the trading strategy (GBP/USD)')
plt.xlabel('Date')
plt.ylabel('Value of holdings')
plt.legend()
plt.show()




#Using the initial strategy with the pair USD/AUD

aud_usd_data = load_and_prep_data('AUD=X.csv')

plt.figure(figsize=(10, 6))
plt.plot(aud_usd_data['Close'], label='USD/AUD Exchange Rate', color='orange')
plt.title('USD/AUD Exchange Rate Time Series between 2010 and 2023')
plt.legend()
plt.show()

sharpe_ratio_aud_usd, max_drawdown_aud_usd, ValueAtRisk_aud_usd, CValueAtRisk_aud_usd,sortino_ratio_aud_usd,calmar_ratio_aud_usd = mean_reversion_strategy(aud_usd_data, estimation_window, critical_value, confidence_level)

print('-------------Initial Strategy with USD/AUD--------------:')
print(f'Sharpe Ratio: {sharpe_ratio_aud_usd:.4f}')
print(f'Maximum Drawdown: {max_drawdown_aud_usd:.4f}')
print(f'Sortino Ratio : {sortino_ratio_aud_usd:.4f}')
print(f'Calmar ratio: {calmar_ratio_aud_usd:.4f}')
print(f'Value at risk with 95% confidence level: {ValueAtRisk_aud_usd:.4f}')
print(f'Condiational Value at risk: {CValueAtRisk_aud_usd:.4f}')




plt.figure(figsize=(10, 6))
plt.plot(aud_usd_data.index, aud_usd_data['Cumulative_Performance'], label=f'Strategy Cumulative Performance for crit value = {critical_value}', color='blue')
plt.title('Strategy Cumulative Performance for USD/AUD')
plt.xlabel('Date')
plt.ylabel('Value of holdings')
plt.legend()
plt.show()

plt.figure(figsize=(10,6))
plt.plot(data.index, data['Cumulative_Performance'], label='Strategy Cumulative Performance for EUR/USD ', color='Red')
plt.plot(eur_gbp_data.index, eur_gbp_data['Cumulative_Performance'], label='Strategy Cumulative Performance for GBP/EUR', color='grey')
plt.plot(gbp_usd_data.index, gbp_usd_data['Cumulative_Performance'], label='Strategy Cumulative Performance for GBP/USD', color="orange")
plt.plot(aud_usd_data.index, aud_usd_data['Cumulative_Performance'], label='Strategy Cumulative Performance for USD/AUD', color='blue')
plt.title('Strategy Cumulative Performance for several currencies')
plt.xlabel('Date')
plt.ylabel('Cumulative performance')
plt.legend()
plt.show()

#Joint trading strategy
joint_data, sharpe_ratio_joint, max_drawdown_joint, ValueAtRisk_joint, CValueAtRisk_joint,sortino_ratio_joint,calmar_ratio_joint = joint_trading_strategy(estimation_window, critical_value, data, eur_gbp_data, gbp_usd_data, confidence_level)

plt.figure(figsize=(10, 6))
plt.plot(joint_data.index, joint_data['Cumulative_Performance'])
plt.title('Strategy Cumulative Performance for joint strategy')
plt.xlabel('Date')
plt.ylabel('Cumulative Performance')
plt.legend()
plt.show()


print('------------- BASIC Joint Strategy--------------:')
print(f'Sharpe Ratio: {sharpe_ratio_joint:.4f}')
print(f'Maximum Drawdown: {max_drawdown_joint:.4f}')
print(f'Sortino Ratio : {sortino_ratio_joint:.4f}')
print(f'Calmar ratio: {calmar_ratio_joint:.4f}')
print(f'Value at risk with 95% confidence level: {ValueAtRisk_joint:.4f}')
print(f'Condiational Value at risk: {CValueAtRisk_joint:.4f}')


# NOW we try to implement a new  strategy based on correlation and compute OLS for correlation pairs in the sample
#Joint trading strategy2

sharpe_ratio_joint2, max_drawdown_joint2, ValueAtRisk_joint2, CValueAtRisk_joint2,sortino_ratio_joint2,calmar_ratio_joint2  = joint_trading_strategy2(estimation_window, critical_value, data, eur_gbp_data, gbp_usd_data, aud_usd_data, confidence_level)

print('-------------TEST Joint Strategy--------------:')
print(f'Sharpe Ratio: {sharpe_ratio_joint2:.4f}')
print(f'Maximum Drawdown: {max_drawdown_joint2:.4f}')
print(f'Sortino Ratio : {sortino_ratio_joint2:.4f}')
print(f'Calmar ratio: {calmar_ratio_joint2:.4f}')
print(f'Value at risk with 95% confidence level: {ValueAtRisk_joint2:.4f}')
print(f'Condiational Value at risk: {CValueAtRisk_joint2:.4f}')

