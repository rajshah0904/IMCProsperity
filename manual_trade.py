import numpy as np
import statistics as stats
'''
start = 900
end = 1000
probs = np.linspace(0, 1, end - start + 1)
probs = probs / np.sum(probs)

new_probs = []

for i in probs:
    new_probs.append(round(i, 8))

print(probs)
print(np.sum(probs))

print(new_probs)
print(np.sum(new_probs))
'''


def sim_trades(low_bid, high_bid):
    total_pofit = 0
    start = 900
    end = 1000
    dataset_size = 10000
    probs = np.linspace(0, 1, end - start + 1)
    probs = probs / np.sum(probs)

    new_probs = []

    for i in probs:
        new_probs.append(round(i, 8))

    probs = new_probs

    prices = np.random.choice(np.arange(start, end + 1), size = dataset_size, p = probs)
    
    for i in prices:
        #if high_bid >= i:
        #    total_pofit += 1000 - high_bid
        #elif low_bid >= i:
        #    total_pofit += 1000 - low_bid
        if low_bid >= i:
            total_pofit += 1000 - low_bid
        elif high_bid >= i:
            total_pofit += 1000 - high_bid
    
    return total_pofit


def find_optimal_prices():
    best_low_bid = 900
    best_high_bid = 1000
    best_total_profit = sim_trades(best_low_bid, best_high_bid)

    for low_bid in range(900, 1000):
        for high_bid in range(low_bid + 1, 1001):
            sim = sim_trades(low_bid, high_bid)
            if sim >= best_total_profit:
                best_low_bid = low_bid
                best_high_bid = high_bid
                best_total_profit = sim
                #print(best_low_bid)
                #print(best_high_bid)
                #print("------------------")
    return (best_low_bid, best_high_bid)

def find_optimal_low():
    best_low_bid = 900
    best_high_bid = 966
    best_total_profit = sim_trades(best_low_bid, best_high_bid)

    for low_bid in range(900, best_high_bid):
        sim = sim_trades(low_bid, best_high_bid)
        if sim >= best_total_profit:
            best_low_bid = low_bid
            best_high_bid = best_high_bid
            best_total_profit = sim
            #print(best_low_bid)
            #print(best_high_bid)
            #print("------------------")
    return best_low_bid

def monte_carlo_low(iters = 1000):
    low_bids = []

    for i in range(iters):
        sim = find_optimal_low()
        low_bids.append(sim)

    return (np.mean(low_bids), np.median(low_bids), stats.mode(low_bids))

def monte_carlo(iters = 100):
    low_bids = []
    high_bids = []

    for i in range(iters):
        sim = find_optimal_prices()
        low_bids.append(sim[0])
        high_bids.append(sim[1])
        print(sim[0])
        print(sim[1])
        print("------------------")

    avg_low = np.mean(low_bids)
    avg_high = np.mean(high_bids)
    return(avg_low, avg_high)

simulation = monte_carlo()
print(simulation)

#print(monte_carlo_low())