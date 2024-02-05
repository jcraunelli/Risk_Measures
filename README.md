# Risk_Measures
This repository explains and has code for the following risk measures: Var, CVar, Greeks

## VaR
    
    "The maximum loss with a confidence level over a predetermined period."
   
The implied factor or variable is our current position, or the value of our current portfolio or individual stock(s). The second one is over a specific time period. with a confidence level or probability
    
    Example #1: On February 7, 2017, we own 300 shares of IBM's stocks worth $52,911. The maximum loss tomorrow, that is, February 8, 2017, is $ 1,951 with a 99% confidence level.

    Example #2: Our mutual fund has a value of $10 million today. The maximum loss over the next 3 months is $0.5 million at a 95% confidence level.

    Example #3: The value of our bank is $200 million. The VaR of our bank is $10m with a 1% probability over the next 6 months.
    
Two methods to estimate a VaR. The first method is based on the assumption that our security or portfolio returns follow a normal distribution, while the second method depends on the ranking of the historical returns.

For the VaR estimation, usually we would choose two confidence levels of 95% and 99%. For the 95% (99%) confidence level, we actually look at the left tail with a 5% (1%) probability.

Usually, there are two methods to estimate a VaR. The first method is based on the assumption that our security or portfolio returns follow a normal distribution, while the second method depends on the ranking of the historical returns. Before discussing the first method, let's review the concepts with respect to a normal distribution.

One function called spicy.stats.norm.pdf() could be used to estimate the density. The function has three input values: x, μ, and σ. The following code calls this function and verifies the results manually
