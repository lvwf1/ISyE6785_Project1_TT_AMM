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

    def BS_d1(self, S=100):
        return (np.log(S / self.K) + (self.rf + self.sigma ** 2 / 2.0) * self.tyears) / (
                    self.sigma * np.sqrt(self.tyears))

    def BS_d2(self,S = 100):
        return (np.log(S / self.K) + (self.rf - self.sigma ** 2 / 2.0) * self.tyears) / (
                    self.sigma * np.sqrt(self.tyears))

    def BS_CallPrice(self, S = 100):
        return S * scipy.stats.norm.cdf(self.BS_d1(S)) - self.K * np.exp(
            -self.rf * self.tyears) * scipy.stats.norm.cdf(self.BS_d2(S))

    def BS_CallPriceDI(self, H):
        return np.power(H/self.S0,2*self.rf/self.sigma**2)*self.BS_CallPrice(H**2/self.S0)

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

        nd_idx = N
        pre_price = X0 - float(N + 1) * h
        # Compute the stock prices and option values at maturity

        move = [1,0,-1]

        # Compute the stock prices and option values at maturity
        for indices in itertools.product(move, repeat=N):
            cur_price = X0
            time_counter = 0
            down_and_in=False
            for i in indices:
                cur_price = cur_price + move[i] * h
                time_counter = time_counter+1
                if cur_price < np.log(H):
                    down_and_in=True
                if time_counter == N and down_and_in and (cur_price - pre_price) > h / 1000.0:
                    stk[nd_idx] = np.exp(cur_price + alpha * self.tyears)
                    fs_pre[nd_idx] = max(stk[nd_idx] - self.K, 0)
                    pre_price = cur_price
                    nd_idx = nd_idx + 1
        print('Call option value at maturity is: ', fs_pre)

        # Backward recursion for computing option prices in time periods N-1, N-2, ... , 0
        for t in range(N - 1, -1, -1):
            fs = []
            cur_optP = 0.0
            for i in range(2 * t + 1):
                cur_optP = np.exp(-self.rf * deltaT) * (qU * fs_pre[i + 2] + qM * fs_pre[i + 1] + qD * fs_pre[i])
                fs.append(cur_optP)
            fs_pre = fs
        return fs[0]

    def AdaptiveMeshEuroCallPrice(self, N=10, H=100):
        deltaT = self.tyears / float(N)
        X0 = np.log(self.S0)
        alpha = self.rf - self.divR - np.power(self.sigma, 2.0) / 2.0
        h = np.sqrt(3.0 * deltaT) * self.sigma

        # Initialize the stock prices and option values at maturity with 0.0
        a_stk = [0.0 for i in range(2 * N + 1)]
        fs = [0.0 for i in range(2 * N + 1)]
        fs_pre = [0.0 for i in range(2 * N + 1)]

        nd_idx = 0
        pre_price = X0 - float(N + 1) * h
        # Compute the stock prices and option values at maturity
        for i in range(N + 1):
            for j in range(N + 1):
                k = max(N - i - j, 0)
                cur_price = X0 + (i - k) * h
                if (cur_price - pre_price) > h / 1000.0:
                    a_stk[nd_idx] = np.exp(cur_price + alpha * self.tyears)
                    # Compute the option value at the cur_price level
                    fs_pre[nd_idx] = max(a_stk[nd_idx] - self.K, 0)
                    pre_price = cur_price
                    nd_idx = nd_idx + 1
        print('Call option value at maturity is: ', fs_pre)

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
    H = 99.9
    call_test = CallOption(S0, K, rf, divR, sigma, T)
    call_tri = call_test.TrinomialTreeEuroCallPrice(n_periods, H)
    call_bs = call_test.BS_CallPriceDI(H)
    print('Trinomial Tree Call option price is: ', call_tri)
    print('Black-Scholes Call option price is: ', call_bs)