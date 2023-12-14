"""
This module contains functions for backtesting the intraday straddle strategy. The logic
for this strategy is a vanilla short straddle entering at 9:45am and exiting at 3:15pm.
There is a stoploss and trailing stoploss set for this strategy. The strategy runs on
the SPX weekly options and runs only on the expiration date of the options, as it follows
the European options model. The strategy is backtested on the SPX from 2021 to 2023.
"""