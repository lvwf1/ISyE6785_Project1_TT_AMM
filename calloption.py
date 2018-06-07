import itertools
import numpy as np
import scipy.stats
import math
import matplotlib.pyplot as plt


class CallOption(object):
    def __init__(self, S0, K, rf, divR, sigma, tyears):
        self.S0 = S0
        self.K = K
        self.rf = rf
        self.divR = divR
        self.sigma = sigma
        self.tyears = tyears

    def BinomialTreeEuroCallPrice(self, N=10):
        deltaT = self.tyears / float(N)
        # create the size of up-move and down-move
        u = np.exp(self.sigma * np.sqrt(deltaT))
        d = 1.0 / u

        # Let fs store the value of the option
        fs = [0.0 for j in range(N + 1)]
        fs_pre = [0.0 for j in range(N + 1)]

        # Compute the risk-neutral probability of moving up: q
        a = np.exp(self.rf * deltaT)
        q = (a - d) / (u - d)

        # Compute the value of the European Call option at maturity time tyears:
        for j in range(N + 1):
            fs[j] = max(self.S0 * np.power(u, j) * np.power(d, N - j) - self.K, 0)
        fs_pre = fs
        #print('Call option value at maturity is: ', fs)

        # Apply the recursive pricing equation to get the option value in periods: N-1, N-2, ... , 0
        for t in range(N - 1, -1, -1):
            fs = [0.0 for j in range(t + 1)]  # initialize the value of options at all nodes in period t to 0.0
            for j in range(t + 1):
                # The following line is the recursive option pricing equation:
                fs[j] = np.exp(-self.rf * deltaT) * (q * fs_pre[j + 1] + (1 - q) * fs_pre[j])
            fs_pre = fs
        return fs[0]

    def BS_d1(self):
        return (np.log(self.S0 / self.K) + (self.rf + self.sigma ** 2 / 2.0) * self.tyears) / (
                    self.sigma * np.sqrt(self.tyears))

    def BS_d2(self):
        return (np.log(self.S0 / self.K) + (self.rf - self.sigma ** 2 / 2.0) * self.tyears) / (
                    self.sigma * np.sqrt(self.tyears))

    def BS_CallPrice(self):
        return self.S0 * scipy.stats.norm.cdf(self.BS_d1()) - self.K * np.exp(
            -self.rf * self.tyears) * scipy.stats.norm.cdf(self.BS_d2())

    def BS_CallPriceDO(self, H):
        return self.BS_CallPrice() - np.power(H/self.S0,2*(self.rf-(self.sigma ** 2/2.0)))*(H**2/self.S0 * scipy.stats.norm.cdf(self.BS_d1()) - self.K * np.exp(
            -self.rf * self.tyears) * scipy.stats.norm.cdf(self.BS_d2()))

    def BS_CallPriceDI(self, H):
        return self.BS_CallPrice()-self.BS_CallPriceDO(H)

    def TrinomialTreeEuroCallPrice(self, N=10, H=100):
        deltaT = self.tyears / float(N)
        X0 = np.log(self.S0)
        alpha = self.rf - self.divR - np.power(self.sigma, 2.0) / 2.0
        h = np.sqrt(3.0 * deltaT) * self.sigma

        # Risk-neutral probabilities:
        qU = 1.0 / 6.0
        qM = 2.0 / 3.0
        qD = 1.0 / 6.0

        # Initialize the stock prices and option values at maturity with 0.0
        stk = [0.0 for i in range(2 * N + 1)]
        fs = [0.0 for i in range(2 * N + 1)]
        fs_pre = [0.0 for i in range(2 * N + 1)]

        nd_idx = 0
        pre_price = X0 - float(N + 1) * h
        # Compute the stock prices and option values at maturity

        time_move = []
        move = [-1, 0, 1]
        for indices in itertools.product(range(len(tuple(move))), repeat=N):
            time_move.append([tuple(move)[i] for i in indices])

        # Compute the stock prices and option values at maturity
        for m in time_move:
            cur_price = X0
            down_and_in = False
            time_counter = 0
            for t in m:
                cur_price = cur_price + t * h
                time_counter = time_counter+1
                if cur_price < np.log(H) :
                    down_and_in = True
                if time_counter == N:
                    if (cur_price - pre_price) > h / 1000.0:
                        stk[nd_idx] = np.exp(cur_price + alpha * self.tyears)
                        if down_and_in:
                            fs_pre[nd_idx] = max(stk[nd_idx] - self.K, 0)
                        else:
                            fs_pre[nd_idx] = 0
                        pre_price = cur_price
                        nd_idx = nd_idx + 1
        #print('Call option value at maturity is: ', fs_pre)

        # Backward recursion for computing option prices in time periods N-1, N-2, ... , 0
        for t in range(N - 1, -1, -1):
            fs = []
            cur_optP = 0.0
            for i in range(2 * t + 1):
                cur_optP = np.exp(-self.rf * deltaT) * (qU * fs_pre[i + 2] + qM * fs_pre[i + 1] + qD * fs_pre[i])
                fs.append(cur_optP)
            fs_pre = fs
        return fs[0]


if __name__ == '__main__':
    S0 = 100.0
    K = 100.0
    rf = 0.1
    divR = 0.0
    sigma = 0.3
    T = 0.6  # unit is in years

    n_periods = 5
    H = 99
    call_test = CallOption(S0, K, rf, divR, sigma, T)
    call_tri = call_test.TrinomialTreeEuroCallPrice(n_periods, H)
    call_bs = call_test.BS_CallPriceDI(H)
    print('Trinomial Tree Call option price is: ', call_tri)
    print('Black-Scholes Call option price is: ', call_bs)