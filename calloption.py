import numpy as np
import scipy.stats
import math
import time
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

    def BS_d1(self, S0 = 100):
        return (np.log(S0 / self.K) + (self.rf + self.sigma ** 2 / 2.0) * self.tyears) / (self.sigma * np.sqrt(self.tyears))

    def BS_d2(self, S0 = 100):
        return (np.log(S0 / self.K) + (self.rf - self.sigma ** 2 / 2.0) * self.tyears) / (self.sigma * np.sqrt(self.tyears))

    def BS_CallPrice(self, S0 = 100):
        return S0 * scipy.stats.norm.cdf(self.BS_d1(S0)) - self.K * np.exp(-self.rf * self.tyears) * scipy.stats.norm.cdf(self.BS_d2(S0))

    def BS_CallPriceDI(self, S0 = 100, H = 90):
        return np.power(H/S0,2*self.rf-self.sigma**2) * self.BS_CallPrice(H**2/S0)

    def BS_CallPriceDO(self, S0 = 100, H = 90):
        return self.BS_CallPrice(S0)- self.BS_CallPriceDI(S0, H)

    def TrinomialTreeEuroCallPriceDO(self, S0=100, N=10, H=100):
        deltaT = self.tyears / float(N)
        X0 = np.log(S0)
        alpha = self.rf - self.divR - np.power(self.sigma, 2.0) / 2.0
        h = X0 - np.log(H)

        # Risk-neutral probabilities:
        qU = 1.0 / 6.0
        qM = 2.0 / 3.0
        qD = 1.0 / 6.0

        # Initialize the stock prices and option values at maturity with 0.0
        stk = [0.0 for i in range(2 * N + 1)]
        fs = [0.0 for i in range(2 * N + 1)]
        fs_pre = [0.0 for i in range(2 * N + 1)]

        nd_idx = 2*N
        pre_price = X0 - float(N + 1) * h

        # Compute the stock prices and option values at maturity
        for i in range(N + 1):
            for j in range(N + 1):
                k = max(N - i - j, 0)
                cur_price = X0 + (i - k) * h
                if (cur_price - pre_price) > h / 1000.0:
                    stk[nd_idx] = np.exp(cur_price + alpha * self.tyears)
                    # Compute the option value at the cur_price level
                    fs_pre[nd_idx] = max(stk[nd_idx] - self.K, 0)
                    pre_price = cur_price
                    nd_idx = nd_idx - 1
        #print('Call option value at maturity is: ', fs_pre)

        # Backward recursion for computing option prices in for each time periods N-1, N-2, ... , 0
        for t in range(1, N + 1):
            fs = []
            for i in range(1, 2 * (N - t) + 2):  # number of nodes at step j
                fs.append(0.0)
                if (t != N):
                    if X0 + (N - t - i + 1) * h > np.log(H):
                        fs[-1] = np.exp(-self.rf * deltaT) * (qU * fs_pre[i - 1] + qM * fs_pre[i] + qD * fs_pre[i + 1])
                else:
                    fs[-1] = np.exp(-self.rf * deltaT) * (qU * fs_pre[i - 1] + qM * fs_pre[i] + qD * fs_pre[i + 1])
            fs_pre = fs

        return fs[0]

    def TrinomialTreeEuroCallPriceDI(self, S=100, N=10, H=100):
        # Use Regular call option value minus Down-and-in call option value to get down-and-out call option value
        return self.TrinomialTreeEuroCallPrice(S,N)-self.TrinomialTreeEuroCallPriceDO(S,N,H)

    def TrinomialTreeEuroCallPriceRTM(self, S0=100, H=100):

        startTime = time.time()

        # Constant Parameters
        X0 = np.log(S0)
        alpha = self.rf - (self.sigma ** 2) / 2
        h = X0 - np.log(H)
        k = self.tyears / math.floor((3.0 * self.sigma ** 2 / h ** 2) * self.tyears)
        N = int(self.tyears / k)

        # Risk-neutral probabilities:
        [pD, pM, pU] = self.computeProbas(alpha, h, k)

        # Initialize the stock prices and option values at maturity with X0
        stk = [X0]

        for i in range(N):
            stk = [stk[0] + h] + stk + [stk[-1] - h]

        fs_pre = stk

        # Perform Down-And-Out Call Option Lattice
        for i in range(2 * N + 1):
            if stk[i] > np.log(H):
                fs_pre[i] = max(np.exp(stk[i]) - K, 0.0)

        # Backward recursion for computing option prices in for each time periods N-1, N-2, ... , 0
        for j in range(1, N + 1):
            fs = []
            for i in range(1, 2 * (N - j) + 2):  # number of nodes at step j
                fs.append(0.0)
                if (j != N):
                    if (np.log(S0) + (N - j - i + 1) * h > np.log(H)):
                        fs[-1] = np.exp(-self.rf * k) * (pU * fs_pre[i - 1] + pM * fs_pre[i] + pD * fs_pre[i + 1])
                else:
                    fs[-1] = np.exp(-self.rf * k) * (pU * fs_pre[i - 1] + pM * fs_pre[i] + pD * fs_pre[i + 1])
            fs_pre = fs
        return [fs[0],time.time()-startTime]

    def TrinomialTreeEuroCallPrice(self, S0=100, N=10):
        deltaT = self.tyears / float(N)
        X0 = np.log(S0)
        alpha = self.rf - self.divR - np.power(self.sigma, 2.0) / 2.0
        h = np.sqrt(3.0 * deltaT) * self.sigma

        # Risk-neutral probabilities:
        qU = 1.0 / 6.0
        qM = 2.0 / 3.0
        qD = 1.0 / 6.0

        # Initialize the stock prices and option values at maturity with 0.0
        stk = [0.0 for i in range(2 * N + 1)]
        fs_pre = [0.0 for i in range(2 * N + 1)]

        nd_idx = 2 * N
        pre_price = X0 - float(N + 1) * h
        # Compute the stock prices and option values at maturity
        for i in range(N + 1):
            for j in range(N + 1):
                k = max(N - i - j, 0)
                cur_price = X0 + (i - k) * h
                if (cur_price - pre_price) > h / 1000.0:
                    stk[nd_idx] = np.exp(cur_price + alpha * self.tyears)
                    # Compute the option value at the cur_price level
                    fs_pre[nd_idx] = max(stk[nd_idx] - self.K, 0)
                    pre_price = cur_price
                    nd_idx = nd_idx - 1
        #print('Call option value at maturity is: ', fs_pre)

        return self.ComputeTrinomialTree(N, deltaT, qU, qM, qD, fs_pre)

    def ComputeTrinomialTree(self, N, deltaT, qU, qM, qD, fs_pre):
        # Backward recursion for computing option prices in time periods N-1, N-2, ... , 0
        for t in range(N - 1, -1, -1):
            fs = []
            for i in range(2 * t + 1):
                cur_optP = np.exp(-self.rf * deltaT) * (qU * fs_pre[i + 2] + qM * fs_pre[i + 1] + qD * fs_pre[i])
                fs.append(cur_optP)
            fs_pre = fs
        return fs[0]

    def AdaptiveMeshEuroCallPrice(self, S0, M, H):

        startTime = time.time()

        # Constant Parameters
        X0 = np.log(S0)
        alpha = self.rf - (self.sigma ** 2) / 2
        h = (2 ** M) * (X0 - np.log(H))
        k = self.tyears / math.floor((3.0 * self.sigma ** 2 / h ** 2) * self.tyears)
        N = int(self.tyears / k)

        # Initialize the stock prices and option values at maturity with X0
        fs_pre = []
        for i in range(N + 1):
            stk = []
            for j in range(2 * i + 1):
                stk.append((np.log(H) + h - i * h) + j * h)
            fs_pre.append(stk)

        # Compute the payoff of the lattice A
        fs_A = []
        finalPayoffA = []
        for i in range(len(fs_pre[N])):
            finalPayoffA.append(max(np.exp(fs_pre[N][i])-self.K,0))
            fs_A.append(finalPayoffA)
        # Calculate Risk-neutral probabilities:
        [pD, pM, pU] = self.computeProbas(alpha, h, k)
        for i in range(1, N + 1):
            opTree = []
            for j in range(2 * (N - i) + 1):
                if (np.exp(fs_pre[N - i][j]) > H):
                    C = np.exp(-self.rf * k) * (pD * fs_A[0][j] + pM * fs_A[0][j + 1] + pU * fs_A[0][j + 2])
                else:
                    C = 0.0
                opTree.append(C)
            fs_A.insert(0, opTree)
        finalValue = fs_A[0][0]
        # Construction of the lattice B
        delta = 0
        gamma = 0
        if M > 0:
            fs_B = []
            fs_B.append([0, 0, fs_pre[N][N]])
            j = 1
            B = [0, 0, 0]
            for i in range(1, N * 4 + 1):
                stepA = int(np.ceil(N - i / 4.0))
                [pD, pM, pU] = self.computeProbas(alpha, h / 2, k / 4)
                B[1] = np.exp(-self.rf * k / 4) * (pD * 0 + pM * fs_B[0][1] + pU * fs_B[0][2])
                if (j > 0):
                    [pD, pM, pU] = self.computeProbas(alpha, h, j * k / 4)
                    B[2] = np.exp(-self.rf * j * k / 4) * (pD * 0 + pM * fs_A[stepA][stepA] + pU * fs_A[stepA][stepA + 1])
                else:
                    B[2] = fs_A[stepA][stepA]

                fs_B.insert(0, B)
                j += 1
                if (j > 3):
                    j = 0
            finalValue = fs_B[0][1]
            delta = (fs_B[0][2] - fs_B[0][0]) / (2.0 * h / 2 * S0)
            gamma = (1.0 / S0 ** 2) * (((fs_B[0][2] + fs_B[0][0] - 2 * fs_B[0][1]) / (h / 2 ** 2)) - (fs_B[0][2] - fs_B[0][0]) / (2.0 * h / 2))
            # Construction of the lattice C
            if M > 1:
                fs_C = []
                fs_C.append([0, 0, fs_pre[N][N]])
                j = 1
                C = [0, 0, 0]
                for i in range(1, N * 16 + 1):
                    stepA = int(np.ceil(4 * N - i / 4.0))
                    [pD, pM, pU] = self.computeProbas(alpha, h / 4, k / 16)
                    C[1] = np.exp(-self.rf * k / 16) * (pD * 0 + pM * fs_C[0][1] + pU * fs_C[0][2])
                    if (j > 0):
                        [pD, pM, pU] = self.computeProbas(alpha, h / 2, j * k / 16)
                        C[2] = np.exp(-self.rf * j * k / 16) * (pD * 0 + pM * fs_B[stepA][1] + pU * fs_B[stepA][2])
                    else:
                        C[2] = fs_B[stepA][1]

                    fs_C.insert(0, C)
                    j += 1
                    if (j > 3):
                        j = 0
                finalValue = fs_C[0][1]
                delta = (fs_C[0][2] - fs_C[0][0]) / (2.0 * h / 4 * S0)
                gamma = (1.0 / S0 ** 2) * (((fs_C[0][2] + fs_C[0][0] - 2 * fs_C[0][1]) / (h / 4 ** 2)) - (fs_C[0][2] - fs_C[0][0]) / (2.0 * h / 4))
                # Construction of the lattice D
                if M > 2:
                    fs_D = []
                    fs_D.append([0, 0, 0])
                    j = 1
                    D = [0, 0, 0]
                    for i in range(1, N * 64 + 1):
                        stepA = int(np.ceil(16 * N - i / 4.0))
                        [pD, pM, pU] = self.computeProbas(alpha, h / 8, k / 64)
                        D[1] = np.exp(-self.rf * k / 64) * (pD * 0 + pM * fs_D[0][1] + pU * fs_D[0][2])
                        if (j > 0):
                            [pD, pM, pU] = self.computeProbas(alpha, h / 4, j * k / 64)
                            D[2] = np.exp(-self.rf * j * k / 64) * (pD * 0 + pM * fs_C[stepA][1] + pU * fs_C[stepA][2])
                        else:
                            D[2] = fs_C[stepA][1]

                        fs_D.insert(0, D)
                        j += 1
                        if (j > 3):
                            j = 0
                    finalValue = fs_D[0][1]
                    delta = (fs_D[0][2] - fs_D[0][0]) / (2.0 * h / 8 * S0)
                    gamma = (1.0 / S0 ** 2) * (((fs_D[0][2] + fs_D[0][0] - 2 * fs_D[0][1]) / (h / 8 ** 2)) - (fs_D[0][2] - fs_D[0][0]) / (2.0 * h / 8))
                    # Construction of the lattice E
                    if M > 3:
                        fs_E = []
                        fs_E.append([0, 0, 0])
                        j = 1
                        E = [0, 0, 0]
                        for i in range(1, N * 256 + 1):
                            stepA = int(np.ceil(16 * N - i / 4.0))
                            [pD, pM, pU] = self.computeProbas(alpha, h / 16, k / 256)
                            E[1] = np.exp(-self.rf * k / 256) * (pD * 0 + pM * fs_E[0][1] + pU * fs_E[0][2])
                            if (j > 0):
                                [pD, pM, pU] = self.computeProbas(alpha, h / 16, j * k / 256)
                                E[2] = np.exp(-self.rf * j * k / 256) * (pD * 0 + pM * fs_D[stepA][1] + pU * fs_D[stepA][2])
                            else:
                                E[2] = fs_D[stepA][1]

                            fs_E.insert(0, E)
                            j += 1
                            if (j > 3):
                                j = 0
                        finalValue = fs_E[0][1]
                        delta = (fs_E[0][2] - fs_E[0][0]) / (2.0 * h / 16 * S0)
                        gamma = (1.0 / S0 ** 2) * (((fs_E[0][2] + fs_E[0][0] - 2 * fs_E[0][1]) / (h / 16 ** 2)) - (fs_E[0][2] - fs_E[0][0]) / (2.0 * h / 17))

        return [finalValue, time.time() - startTime, delta, gamma]

    def computeProbas(self, alpha, h, k):
        pU = 0.5 * ((self.sigma ** 2) * (k / (h ** 2)) + (alpha ** 2) * ((k ** 2) / (h ** 2)) + alpha * (k / h))
        pD = 0.5 * ((self.sigma ** 2) * (k / (h ** 2)) + (alpha ** 2) * ((k ** 2) / (h ** 2)) - alpha * (k / h))
        pM = 1 - pD - pU
        return [pD, pM, pU]

    def TTDI_timer(self, S, N, H):
        start = time.time()
        self.TrinomialTreeEuroCallPriceDI(S, N, H)
        end = time.time()
        return end-start

if __name__ == '__main__':


#1
    S0 = 100.0
    K = 100.0
    rf = 0.1
    divR = 0.0
    sigma = 0.3
    T = 0.6  # unit is in years

    n_periods = 200
    H = 99.5

    call_test = CallOption(S0, K, rf, divR, sigma, T)
    call_tri_di = call_test.TrinomialTreeEuroCallPriceDI(S0, n_periods, H)
    call_tri_do = call_test.TrinomialTreeEuroCallPriceDO(S0, n_periods, H)
    call_bs_di = call_test.BS_CallPriceDI(S0, H)
    call_bs_do = call_test.BS_CallPriceDO(S0, H)
    print('Trinomial Tree Call down-and-in option price is: ', call_tri_di)
    print('Trinomial Tree Call down-and-out option price is: ', call_tri_do)
    print('Black-Scholes Call down-and-in option price is: ', call_bs_di)
    print('Black-Scholes Call down-and-out option price is: ', call_bs_do)


    axis_n = np.arange(50, 1000, 50)
    TTDI_vec = [call_test.TrinomialTreeEuroCallPriceDI(S0,n,H) for n in axis_n]
    TTDO_vec = [call_test.TrinomialTreeEuroCallPriceDO(S0,n,H) for n in axis_n]
    BSDI_vec = [call_test.BS_CallPriceDI(S0,H) for n in axis_n]
    BSDO_vec = [call_test.BS_CallPriceDO(S0,H) for n in axis_n]
    print('Trinomial Tree Call down-and-in option price from 50 - 1000 periods is: ',TTDI_vec)
    print('Trinomial Tree Call down-and-out option price from 50 - 1000 periods is: ',TTDO_vec)
    print('Black-Scholes Call down-and-in option price from 50 - 1000 periods is: ', BSDI_vec)
    print('Black-Scholes Call down-and-out option price from 50 - 1000 periods is: ', BSDO_vec)
    plt.plot(axis_n, TTDI_vec, 'r-', lw=2, label='TTDI')
    plt.plot(axis_n, BSDI_vec, 'b-', lw=2, label='BSDI')
    label = ['TTDI','BSDI']
    plt.xlabel("Number of Periods")
    plt.ylabel("Option Price")
    plt.title("European Call Down-And-In Option Price vs. Number of Periods in a Lattice")
    plt.legend(label)
    plt.grid(True)
    plt.show()

    axis_h = np.array([95, 99.5, 99.9])
    TTDI_vec2 = np.array([call_test.TrinomialTreeEuroCallPriceDI(S0, n_periods, H) for H in axis_h])
    BSDI_vec2 = np.array([call_test.BS_CallPriceDI(S0,H) for H in axis_h])
    plt.plot(axis_h, TTDI_vec2/BSDI_vec2, 'r-', lw=2)
    label = ['TTDI Accuracy']
    plt.xlabel("Barrier Option")
    plt.ylabel("Accuracy Percentage")
    plt.title("Price Accuracy Percentage vs. Barrier Option")
    plt.legend(label)
    plt.grid(True)
    plt.show()
    TTDI_vec3 = np.array([call_test.TTDI_timer(S0, n_periods, H) for H in axis_h])
    TTDI_vec4 = np.array([call_test.TTDI_timer(S0, n_periods, H) for n_periods in axis_n])
    plt.subplot(211)
    plt.plot(axis_h, TTDI_vec3, 'b-', lw=2)
    label = ['TTDI Computation Time']
    plt.xlabel("Barrier Option")
    plt.ylabel("Computational Time")
    plt.title("Computational Time vs. Barrier Option")
    plt.legend(label)
    plt.grid(True)
    plt.subplot(212)
    plt.plot(axis_n, TTDI_vec4, 'b-', lw=2)
    label = ['TTDI Computation Time']
    plt.xlabel("Time Steps")
    plt.ylabel("Computational Time")
    plt.title("Computational Time vs. Number of Time Steps")
    plt.legend(label)
    plt.grid(True)
    plt.show()


#2
    S0 = 92
    K = 100.0
    rf = 0.1
    divR = 0.0
    sigma = 0.25
    T = 1.0  # unit is in years

    n_periods = 10
    H = 90

    call_test2 = CallOption(S0, K, rf, divR, sigma, T)

    axis_s = [92,91,90.5,90.25]
    axis_sn = [92,91,90.5,90.25]
    axis_sm = [(92,0), (91,1),(90.5,2),(90.25,3),(90.125,4)]
    BS_vec = [call_test2.BS_CallPriceDO(s,H) for s in axis_s]
    print('Black-Scholes Call down-and-out option price from 92, 91, 90, 90.5, 90.25, 90.125 barrier is: ',BS_vec)
    TT_vec = [call_test2.TrinomialTreeEuroCallPriceRTM(s,H) for s in axis_sn]
    print('Trinomial Tree Call down-and-out option price from 92, 91, 90, 90.5, 90.25, 90.125 barrier is: ',[vec[0] for vec in TT_vec])
    print('Trinomial Tree Call computation time from 92, 91, 90, 90.5, 90.25, 90.125 barrier is: ',[vec[1] for vec in TT_vec])
    AMM_vec = [call_test2.AdaptiveMeshEuroCallPrice(s,M,H) for (s,M) in axis_sm]
    print(AMM_vec)
    print('Adaptive Mesh Call down-and-out option price from 92, 91, 90, 90.5, 90.25, 90.125 barrier with 0,1,2,3,4 mesh level is: ',[vec[0] for vec in AMM_vec])
    print('Adaptive Mesh Call computation time from 92, 91, 90, 90.5, 90.25, 90.125 barrier with 0,1,2,3,4 mesh level is: ',[vec[1] for vec in AMM_vec])
    print('Adaptive Mesh Call Delta from 92, 91, 90, 90.5, 90.25, 90.125 barrier with 0,1,2,3,4 mesh level is: ',[vec[2] for vec in AMM_vec])
    print('Adaptive Mesh Call Gamma from 92, 91, 90, 90.5, 90.25, 90.125 barrier with 0,1,2,3,4 mesh level is: ',[vec[3] for vec in AMM_vec])

    plt.subplot(211)
    plt.plot(axis_s, BS_vec, 'r-', lw=2, label='BS')
    plt.plot(axis_s, [vec[0] for vec in AMM_vec], 'b-', lw=2, label='AMM')
    label = ['BS', 'AMM']
    plt.xlabel("Current Price")
    plt.ylabel("Option Price")
    plt.title("European Call Option Price vs. Current Price Closed to Barrier Option (Adaptive Mesh vs. Black-Scholes)")
    plt.legend(label)
    plt.grid(True)

    plt.subplot(212)
    plt.plot(axis_s, BS_vec, 'r-', lw=2, label='BS')
    plt.plot(axis_s, [vec[0] for vec in TT_vec], 'b-', lw=2, label='TT')
    label = ['BS', 'TT']
    plt.xlabel("Current Price")
    plt.ylabel("Option Price")
    plt.title("European Call Option Price vs. Current Price Closed to Barrier Option (Trinomial Tree vs. Black-Scholes)")
    plt.legend(label)
    plt.grid(True)
    plt.show()

    plt.plot(axis_s, [vec[1] for vec in TT_vec], 'r-', lw=2, label='TT')
    plt.plot(axis_s, [vec[1] for vec in AMM_vec], 'b-', lw=2, label='AMM')
    label = ['TT', 'AMM']
    plt.xlabel("Barrier Option")
    plt.ylabel("Computation Time")
    plt.title("European Trinomial Tree Call Option Price vs. Current Price Closed to Barrier Option Performance")
    plt.legend(label)
    plt.grid(True)
    plt.show()

#3

    axis_time = [0,1,2,3]
    plt.plot(axis_time, [vec[2] for vec in AMM_vec], 'r-', lw=2, label='Delta')
    plt.plot(axis_time, [vec[3] for vec in AMM_vec], 'b-', lw=2, label='Gamma')
    label = ['Delta','Gamma']
    plt.xlabel("Level of Mesh")
    plt.ylabel("Delta and Gamma Value")
    plt.title("Adaptive Mesh Delta and Gamma vs. Level Of Mesh")
    plt.legend(label)
    plt.grid(True)
    plt.show()
