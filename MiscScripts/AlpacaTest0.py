# PKJM62G0UIEAA3T1DIHA  API Key ID
# 08PcErcQBQMmMFLPafd2zeVel6DeBkAyhEUpKHFe  Secret Key
from pprint import pprint

import alpaca_trade_api as tradeapi

# API LINK
api = tradeapi.REST('PKJM62G0UIEAA3T1DIHA', '08PcErcQBQMmMFLPafd2zeVel6DeBkAyhEUpKHFe',
                    base_url='https://paper-api.alpaca.markets')  # or use ENV Vars shown below
# Cache account details?
account = api.get_account()


# This is linked to the PAPER i.e. not real money account, so i have run this maybe 10 times and when it opens monday
# they will go through.
# CMCM is cheetah mobile, going at about 2.34 a share. I was going to use it to test real money
# with TDA but that never really worked. Should find a smaller share to test with real money
# although if we just use the paper account we can test all day long with whichever stocks obviously.
# api.submit_order('CMCM', 1, 'buy', 'market', 'day')
# print(api.list_orders())

# For your text analysis, this will return a double line separated listing of the news
# listings which had text summaries in them. There are links to urls in all of them which
# I suppose we could also scrape for information. In any case this is a very easy way to piggyback off
# of a ton of Bloomberg Yahoo other industry outlets and then pipe it straight to your NLP program.
def get_summaries(symbol):
    sret = ""
    for i in api.polygon.news(symbol):
        if i.summary:
            sret += i.title
            sret += i.summary
        # sret += "\n\n"
    return sret


# Testing it with Apple then an example of the full output with Tesla
# jacob = get_summaries('AAPL')
# print(jacob)
# pprint(api.polygon.news('TSLA')[0:2])

# Maybe you want things in an array instead, this adds the title and summary to each array object.
def summariesArr(symbol):
    sret = []
    for i in api.polygon.news(symbol):
        if i.summary:
            s = ""
            s += i.title
            s += i.summary
            sret.append(s)

    return sret


# Test
# pprint(summariesArr('AAPL'))

# Pull out just the float askprice from the returned Quote Entity
js = api.get_last_quote('CMCM').askprice

qt = api.polygon.gainers_losers(direction="gainers")

# pprint(api.polygon.grouped_daily('2019-02-01'))
# hg = api.polygon.all_tickers()


"""
# Not sure where I was headed with this, seems like we can get open/close
# data and do caluclations on the fly when the markets are open
hold = api.polygon.grouped_daily('2019-02-01')
items = hold.items()

run = items.__iter__()
for i in range(1):
    jc = (run.__next__())
    # jc = jc.open
    print(type(jc[1]))
    """

api.polygon.last_quote('AAPL').askprice


# Experimenting with this method
# holdon = api.polygon.previous_day_bar('AAPL')[0]
# print(holdon)


# Just a test comparing some of the data pulled with previous_day_bar
# Had to take the attribute after specifying index 0, even though
# there is no index 1, i dont know why this works/doesnt work.
def movedSome(symbol):
    temp = api.polygon.previous_day_bar(symbol)[0]
    if temp.close > temp.open:
        return True
    else:
        return False


# Reaches into the current quote object which is returned and hands back just the floating
# point percent change
def getChangePercent(symbol):
    try:
        iret = api.alpha_vantage.current_quote(symbol)["10. change percent"]
        return iret
    except KeyError:
        return False



print(getChangePercent('TORC'))

# print(api.polygon.exchanges()[2])
print(api.polygon.snapshot('AAPL'))
