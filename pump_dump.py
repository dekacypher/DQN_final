from pyine.indicators import *

# //@version=3 - using the 3rd version of pine script
# strategy("[BoTo] Pump&Dump Strategy", shorttitle = "[BoTo] P&D Strategy", default_qty_type = strategy.percent_of_equity, default_qty_value = 100, pyramiding = 0)

# //Settings
multiplier = input(3.0)
length = input(100)
stop = input(100.0, title = "Stop loss, %")

# //Indicator
body = abs(close - open)
sma = sma(body, length) * multiplier
# plot(body, color = gray, linewidth = 1, transp = 0, title = "Body")
# plot(sma, color = gray, style = area, linewidth = 0, transp = 90, title = "Avg.body * Multiplier")

# //Signals
pump = body > sma and close > open
dump = body > sma and close < open
color = lambda pump, green, dump: green if pump else dump: bgcolor(color, transp = 0)

# //Stops
# size = strategy.position_size
autostop = 0.0 

if (pump == 0) and (size == 0): 
    autostop = low
else:
    autostop = autostop[1]

userstop = 0.0

if (pump == 0) and (size == 0): 
    userstop = close - (close / 100 * stop)
else:
    userstop = userstop[1]


# //Strategy
if pump: 
    None
#     strategy.entry("Pump", strategy.long)
if (dump < autostop) or (low < autostop) or (low < userstop): 
#     strategy.close_all()