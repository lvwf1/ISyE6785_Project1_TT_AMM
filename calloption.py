import itertools
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
        return self.BS_d1(S0) - self.sigma * np.sqrt(self.tyears)

    def BS_CallPrice(self, S0 = 100):
        return S0 * scipy.stats.norm.cdf(self.BS_d1(S0)) - self.K * np.exp(-self.rf * self.tyears) * scipy.stats.norm.cdf(self.BS_d2(S0))

    def BS_CallPriceDI(self, S0 = 100, H = 90):
        return np.power(H/S0,2*self.rf-self.sigma**2) * H**2/S0 * scipy.stats.norm.cdf((np.log(H**2/S0 / self.K) + (self.rf + self.sigma ** 2 / 2.0) * self.tyears) / (self.sigma * np.sqrt(self.tyears))) - self.K * np.exp(-self.rf * self.tyears) * scipy.stats.norm.cdf((np.log(H**2/S0 / self.K) + (self.rf + self.sigma ** 2 / 2.0) * self.tyears) / (self.sigma * np.sqrt(self.tyears)) - self.sigma * np.sqrt(self.tyears))

    def BS_CallPriceDO(self, S0 = 100, H = 90):
        return self.BS_CallPrice(S0)- self.BS_CallPriceDI(S0, H)

    def TrinomialTreeEuroCallPriceDI(self, S0=100, N=10, H=100):
        deltaT = self.tyears / float(N)
        X0 = np.log(H**2/S0)
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

        # Initialize the stock price movement
        move = [1,0,-1]

        # Compute the stock prices and option values at maturity
        for indices in itertools.product(move, repeat=N):
            cur_price = X0
            time_counter = 0
            down_and_in=False
            for i in indices:
                cur_price = cur_price + move[i] * h
                time_counter = time_counter+1
                # Only Compute the stock price if the price hits the barrier
                if cur_price < np.log(H):
                    down_and_in=True
                if time_counter == N and down_and_in and (cur_price - pre_price) > h / 1000.0:
                    stk[nd_idx] = np.exp(cur_price + alpha * self.tyears)
                    fs_pre[nd_idx] = max(stk[nd_idx] - self.K, 0)
                    pre_price = cur_price
                    nd_idx = nd_idx + 1

        return self.ComputeTrinomialTree(N, deltaT, qU, qM, qD, fs_pre)

    def TrinomialTreeEuroCallPriceDO(self, S=100, N=10, H=100):
        # Use Regular call option value minus Down-and-in call option value to get down-and-out call option value
        return self.TrinomialTreeEuroCallPrice(S,N)-self.TrinomialTreeEuroCallPriceDI(S,N,H)

    def TrinomialTreeDelta(self, S0=100, N=10, e=0.01):
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

        nd_idx = 0
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
                    nd_idx = nd_idx + 1
        #print('Call option value at maturity is: ', fs_pre)

        fs_pre_a = [x + e for x in fs_pre]
        fs_pre_s = [x - e for x in fs_pre]

        return (self.ComputeTrinomialTree(N, deltaT, qU, qM, qD, fs_pre_a) - self.ComputeTrinomialTree(N, deltaT, qU, qM, qD, fs_pre_s))/(2*e)/self.S0

    def TrinomialTreeGamma(self, S0=100, N=10, e=0.01):
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

        nd_idx = 0
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
                    nd_idx = nd_idx + 1
        #print('Call option value at maturity is: ', fs_pre)

        # Compute Trinomial perturbation
        fs_pre_a = [x + e for x in fs_pre]
        fs_pre_s = [x - e for x in fs_pre]

        return ((self.ComputeTrinomialTree(N, deltaT, qU, qM, qD, fs_pre_a) - self.ComputeTrinomialTree(N, deltaT, qU, qM, qD, fs_pre_s) + 2 * self.ComputeTrinomialTree(N, deltaT, qU, qM, qD, fs_pre)) / (e ** 2) - (self.ComputeTrinomialTree(N, deltaT, qU, qM, qD, fs_pre_a) - self.ComputeTrinomialTree(N, deltaT, qU, qM, qD, fs_pre_s)) / (2 * e)) / self.S0 ** 2

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

        nd_idx = 0
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
                    nd_idx = nd_idx + 1
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

    def AdaptiveMeshEuroCallPrice(self, S0=100, H=100, M=1):
        X0 = np.log(S0)
        alpha = self.rf - self.divR - np.power(self.sigma, 2.0) / 2.0
        h = 2 ** M * (X0 - np.log(H))
        k = self.tyears / math.floor((3.0 * sigma ** 2 / h ** 2)*self.tyears)
        N = int(self.tyears / k)

        # Initialize the stock prices and option values at maturity with 0.0
        stk = [0.0 for i in range(2*N+1)]
        a_mesh = [0.0 for i in range(2*N+1)]

        nd_idx = 0
        pre_price = X0 - float(N + 1) * h

        # Compute the stock prices and option values at maturity
        for i in range(N + 1):
            for j in range(N + 1):
                l = max(N - i - j, 0)
                cur_price = X0 + (i - l) * h
                if (cur_price - pre_price) > h / 1000.0:
                    stk[nd_idx] = np.exp(cur_price + alpha * self.tyears)
                    # Compute the option value at the cur_price level
                    if stk[nd_idx] > H :
                        a_mesh[nd_idx] = max(stk[nd_idx] - self.K, 0)
                        pre_price = cur_price
                        nd_idx = nd_idx + 1
        #print('Call option value at maturity is: ', a_mesh)

        return self.ComputeAMM(H, h, k, N, a_mesh, alpha, M)

    def AdaptiveMeshDelta(self, S0=100, M=1, e=0.01):
        X0 = np.log(S0)
        alpha = self.rf - self.divR - np.power(self.sigma, 2.0) / 2.0
        h = 2 ** M * (X0 - np.log(H))
        k = self.tyears / math.floor((3.0 * sigma ** 2 / h ** 2)*self.tyears)
        N = int(self.tyears / k)

        # Initialize the stock prices and option values at maturity with 0.0
        stk = [0.0 for i in range(2*N+1)]
        a_mesh = [0.0 for i in range(2*N+1)]

        nd_idx = 0
        pre_price = X0 - float(N + 1) * h

        # Compute the stock prices and option values at maturity
        for i in range(N + 1):
            for j in range(N + 1):
                l = max(N - i - j, 0)
                cur_price = X0 + (i - l) * h
                if (cur_price - pre_price) > h / 1000.0:
                    stk[nd_idx] = np.exp(cur_price + alpha * self.tyears)
                    # Compute the option value at the cur_price level
                    if stk[nd_idx] > H :
                        a_mesh[nd_idx] = max(stk[nd_idx] - self.K, 0)
                        pre_price = cur_price
                        nd_idx = nd_idx + 1
        #print('Call option value at maturity is: ', a_mesh)

        a_mesh_a = [x+e for x in a_mesh]
        a_mesh_s = [x-e for x in a_mesh]

        return (self.ComputeAMM(H, h, k, N, a_mesh_a, alpha, M)-self.ComputeAMM(H, h, k, N, a_mesh_s, alpha, M))/(2*e)/self.S0

    def AdaptiveMeshGamma(self, S0=100, M=1, e=0.01):
        X0 = np.log(S0)
        alpha = self.rf - self.divR - np.power(self.sigma, 2.0) / 2.0
        h = 2 ** M * (X0 - np.log(H))
        k = self.tyears / math.floor((3.0 * sigma ** 2 / h ** 2)*self.tyears)
        N = int(self.tyears / k)

        # Initialize the stock prices and option values at maturity with 0.0
        stk = [0.0 for i in range(2*N+1)]
        a_mesh = [0.0 for i in range(2*N+1)]

        nd_idx = 0
        pre_price = X0 - float(N + 1) * h
        # Compute the stock prices and option values at maturity
        for i in range(N + 1):
            for j in range(N + 1):
                l = max(N - i - j, 0)
                cur_price = X0 + (i - l) * h
                if (cur_price - pre_price) > h / 1000.0:
                    stk[nd_idx] = np.exp(cur_price + alpha * self.tyears)
                    # Compute the option value at the cur_price level
                    if stk[nd_idx] > H :
                        a_mesh[nd_idx] = max(stk[nd_idx] - self.K, 0)
                        pre_price = cur_price
                        nd_idx = nd_idx + 1
        #print('Call option value at maturity is: ', a_mesh)

        a_mesh_a = [x+e for x in a_mesh]
        a_mesh_s = [x-e for x in a_mesh]

        return ((self.ComputeAMM(H, h, k, N, a_mesh_a, alpha, M)-self.ComputeAMM(H, h, k, N, a_mesh_s, alpha, M)+2*self.ComputeAMM(H, h, k, N, a_mesh, alpha, M))/(e**2)-(self.ComputeAMM(H, h, k, N, a_mesh_a, alpha, M)-self.ComputeAMM(H, h, k, N, a_mesh_s, alpha, M))/(2*e))/self.S0**2

    def ComputeAMM(self, H, h, k, N, a_mesh, alpha, M):

        # Initialize the mid mesh with log(H) + h / 2
        b_mesh_mid = np.log(H) + h / 2
        c_mesh_mid = np.log(H) + h / 4
        d_mesh_mid = np.log(H) + h / 8
        e_mesh_mid = np.log(H) + h / 16

        # Backward recursion for computing option prices in time periods N-k, N-2k, ... , 0
        for t in range(N):
            pre_amesh = a_mesh
            for t_a in range(N):
                if t_a + 2 < N - t:
                    a_mesh[t_a + 1] = np.exp(-self.rf * k) * (
                                self.qU(alpha, k, h) * pre_amesh[t_a + 2] + self.qM(alpha, k, h) * pre_amesh[
                            t_a + 1] + self.qD(alpha, k, h) * pre_amesh[t_a])
            for t_a in range(4):
                h_a = h
                k_a = k * t_a / 4
                a_mesh_mid = (np.exp(-self.rf * k_a) * (
                            self.qU(alpha, k_a, h_a) * pre_amesh[2] + self.qM(alpha, k_a, h_a) * pre_amesh[1] + self.qD(
                        alpha, k_a, h_a) * pre_amesh[0]))
                if M > 0:
                    for t_b in range(4):
                        h_b = h / 2
                        k_b = k / 4
                        b_mesh_mid = (np.exp(-self.rf * k_b) * (
                                    self.qU(alpha, k_b, h_b) * a_mesh_mid + self.qM(alpha, k_b,
                                                                                    h_b) * b_mesh_mid + self.qD(alpha,
                                                                                                                k_b,
                                                                                                                h_b) *
                                    pre_amesh[0]))
                        if M > 1:
                            for t_c in range(4):
                                h_c = h / 4
                                k_c = k / 16
                                c_mesh_mid = (np.exp(-self.rf * k_c) * (
                                            self.qU(alpha, k_c, h_c) * b_mesh_mid + self.qM(alpha, k_c,
                                                                                            h_c) * c_mesh_mid + self.qD(
                                        alpha, k_c, h_c) * pre_amesh[0]))
                                if M > 2:
                                    for t_d in range(4):
                                        h_d = h / 8
                                        k_d = k / 32
                                        d_mesh_mid = (np.exp(-self.rf * k_d) * (
                                                    self.qU(alpha, k_d, h_d) * c_mesh_mid + self.qM(alpha, k_d,
                                                                                                    h_d) * d_mesh_mid + self.qD(
                                                alpha, k_d, h_d) * pre_amesh[0]))
                                        if M > 3:
                                            for t_d in range(4):
                                                h_e = h / 16
                                                k_e = k / 64
                                                e_mesh_mid = (np.exp(-self.rf * k_e) * (
                                                            self.qU(alpha, k_e, h_e) * d_mesh_mid + self.qM(alpha, k_e,
                                                                                                            h_e) * e_mesh_mid + self.qD(
                                                        alpha, k_e, h_e) * pre_amesh[0]))

        # Selectively return the final mesh value depending on mesh level
        if M > 3:
            return e_mesh_mid
        elif M > 2:
            return d_mesh_mid
        elif M > 1:
            return c_mesh_mid
        elif M > 0:
            return b_mesh_mid
        else:
            return a_mesh_mid

    def qU(self, alpha, k, h):
        #Compute the risk-neutral probability upward
        return 1 / 2 * (self.sigma ** 2 * k / h ** 2 + alpha ** 2 * k ** 2 / h ** 2 + alpha * k / h)

    def qD(self, alpha, k, h):
        # Compute the risk-neutral probability downward
        return 1 / 2 * (self.sigma ** 2 * k / h ** 2 + alpha ** 2 * k ** 2 / h ** 2 - alpha * k / h)

    def qM(self, alpha, k, h):
        # Compute the risk-neutral probability unchanged
        return 1 - self.qU(alpha, k,h) - self.qD(alpha, k,h)

    def TTDI_timer(self, S, N, H):
        start = time.time()
        self.TrinomialTreeEuroCallPriceDI(S, N, H)
        end = time.time()
        return end-start

    def TTDO_timer(self, S, N, H):
        start = time.time()
        self.TrinomialTreeEuroCallPriceDO(S, N, H)
        end = time.time()
        return end-start

    def AMM_timer(self, S, H, M):
        start = time.time()
        self.AdaptiveMeshEuroCallPrice(S,H,M)
        end = time.time()
        return end-start

    def AMM_timer_Delta(self, H, M, e):
        start = time.time()
        self.AdaptiveMeshDelta(H, M, e)
        end = time.time()
        return end-start

    def AMM_timer_Gamma(self, H, M, e):
        start = time.time()
        self.AdaptiveMeshGamma(H, M, e)
        end = time.time()
        return end-start

    def TT_timer_Delta(self, S, N, e):
        start = time.time()
        self.TrinomialTreeDelta(S, N, e)
        end = time.time()
        return end - start

    def TT_timer_Gamma(self, S, N, e):
        start = time.time()
        self.TrinomialTreeGamma(S, N,e)
        end = time.time()
        return end - start

if __name__ == '__main__':

#1
    S0 = 100.0
    K = 100.0
    rf = 0.1
    divR = 0.0
    sigma = 0.3
    T = 0.6  # unit is in years

    n_periods = 10
    H = 99.9

    call_test = CallOption(S0, K, rf, divR, sigma, T)
    call_tri_di = call_test.TrinomialTreeEuroCallPriceDI(S0, n_periods, H)
    call_tri_do = call_test.TrinomialTreeEuroCallPriceDO(S0, n_periods, H)
    call_bs_di = call_test.BS_CallPriceDI(S0, H)
    call_bs_do = call_test.BS_CallPriceDO(S0, H)
    print('Trinomial Tree Call down-and-in option price is: ', call_tri_di)
    print('Trinomial Tree Call down-and-out option price is: ', call_tri_do)
    print('Black-Scholes Call down-and-in option price is: ', call_bs_di)
    print('Black-Scholes Call down-and-out option price is: ', call_bs_do)


    axis_n = np.arange(4, 12, 1)
    TTDI_vec = [call_test.TrinomialTreeEuroCallPriceDI(S0,n,H) for n in axis_n]
    TTDO_vec = [call_test.TrinomialTreeEuroCallPriceDO(S0,n,H) for n in axis_n]
    BSDI_vec = [call_test.BS_CallPriceDI(S0,H) for n in axis_n]
    BSDO_vec = [call_test.BS_CallPriceDO(S0,H) for n in axis_n]
    print('Trinomial Tree Call down-and-in option price from 4 - 12 periods is: ',TTDI_vec)
    print('Trinomial Tree Call down-and-out option price from 4 - 12 periods is: ',TTDO_vec)
    print('Black-Scholes Call down-and-in option price from 4 - 12 periods is: ', BSDI_vec)
    print('Black-Scholes Call down-and-out option price from 4 - 12 periods is: ', BSDO_vec)
    plt.plot(axis_n, TTDI_vec, 'r-', lw=2, label='TTDI')
    plt.plot(axis_n, TTDO_vec, 'c-', lw=2, label='TTDO')
    plt.plot(axis_n, BSDI_vec, 'g-', lw=2, label='BSDI')
    plt.plot(axis_n, BSDO_vec, 'b-', lw=2, label='BSDO')
    label = ['TTDI','TTDO','BSDI','BSDO']
    plt.xlabel("Number of Periods")
    plt.ylabel("Option Price")
    plt.title("European Call Option Price vs. Number of Periods in a Lattice")
    plt.legend(label)
    plt.grid(True)
    plt.show()
    axis_h = np.array([95, 99, 99.9])
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
    plt.plot(axis_h, TTDI_vec3, 'b-', lw=2)
    label = ['TTDI Computation Time']
    plt.xlabel("Barrier Option")
    plt.ylabel("Computational Time")
    plt.title("Computational Time vs. Barrier Option")
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
    M = 1

    call_test2 = CallOption(S0, K, rf, divR, sigma, T)


    axis_s = [92,91,90.5,90.25,90.125]
    axis_sn = [(92,6), (91,8), (90.5,10), (90.25,12), (90.125,14)]
    axis_sm = [(92,0), (91,1), (90.5,2), (90.25,3), (90.125,4)]
    BS_vec = [call_test2.BS_CallPriceDO(s,H) for s in axis_s]
    print('Black-Scholes Call down-and-out option price from 92, 91, 90, 90.5, 90.25, 90.125 barrier is: ',BS_vec)
    TT_vec = [call_test2.TrinomialTreeEuroCallPriceDO(s,n,H) for (s,n) in axis_sn]
    print('Trinomial Tree Call down-and-out option price from 92, 91, 90, 90.5, 90.25, 90.125 barrier is: ',TT_vec)
    AMM_vec = [call_test2.AdaptiveMeshEuroCallPrice(s,H,M) for (s,M) in axis_sm]
    print('Adaptive Mesh Call down-and-out option price from 92, 91, 90, 90.5, 90.25, 90.125 barrier with 0,1,2,3,4 mesh level is: ',AMM_vec)

    plt.subplot(211)
    plt.plot(axis_s, BS_vec, 'r-', lw=2, label='BS')
    plt.plot(axis_s, AMM_vec, 'b-', lw=2, label='AMM')
    label = ['BS', 'AMM']
    plt.xlabel("Current Price")
    plt.ylabel("Option Price")
    plt.title("European Call Option Price vs. Current Price Closed to Barrier Option (Adaptive Mesh vs. Black-Scholes)")
    plt.legend(label)
    plt.grid(True)

    plt.subplot(212)
    plt.plot(axis_s, BS_vec, 'r-', lw=2, label='BS')
    plt.plot(axis_s, TT_vec, 'b-', lw=2, label='TT')
    label = ['BS', 'TT']
    plt.xlabel("Current Price")
    plt.ylabel("Option Price")
    plt.title("European Call Option Price vs. Current Price Closed to Barrier Option (Trinomial Tree vs. Black-Scholes)")
    plt.legend(label)
    plt.grid(True)
    plt.show()


    TT_time_vec=[call_test2.TTDO_timer(s,n,H) for (s,n) in axis_sn]
    AMM_time_vec=[call_test2.AMM_timer(s,H,M) for (s,M) in axis_sm]

    plt.plot(axis_s, TT_time_vec, 'r-', lw=2, label='TT')
    plt.plot(axis_s, AMM_time_vec, 'b-', lw=2, label='AMM')
    label = ['TT', 'AMM']
    plt.xlabel("Barrier Option")
    plt.ylabel("Computation Time")
    plt.title("European Trinomial Tree Call Option Price vs. Current Price Closed to Barrier Option Performance")
    plt.legend(label)
    plt.grid(True)
    plt.show()

#3
    S0 = 90.5
    K = 100.0
    rf = 0.1
    divR = 0.0
    sigma = 0.25
    T = 1.0  # unit is in years

    n_periods = 10
    H = 90
    M = 2

    call_test3 = CallOption(S0, K, rf, divR, sigma, T)
    AMM_delta_vec = [call_test3.AMM_timer_Delta(S0,M,e) for (S0,M,e) in [(92,0,0.01),(91,1,0.01),(90.5,2,0.01),(90.25,3,0.01)]]
    print('Adaptive Mesh Delta for Mesh 92, 91, 90.5, 90.25 with e=0.01 is: ', AMM_delta_vec)
    AMM_gamma_vec = [call_test3.AMM_timer_Gamma(S0,M,e) for (S0,M,e) in [(92,0,0.01),(91,1,0.01),(90.5,2,0.01),(90.25,3,0.01)]]
    print('Adaptive Mesh Gamma for Mesh 92, 91, 90.5, 90.25 with e=0.01 is: ', AMM_gamma_vec)
    TT_delta_vec = [call_test3.TT_timer_Delta(S0,n,e) for n,e in [(25,0.01),(50,0.01),(250,0.01),(1000,0.01)]]
    print('Trinomial Tree Delta for n=25 n=50 n=250 n=1000 with e=0.01 is: ', TT_delta_vec)
    TT_gamma_vec = [call_test3.TT_timer_Gamma(S0,n,e) for n,e in [(25,0.01),(50,0.01),(250,0.01),(1000,0.01)]]
    print('Trinomial Tree Gamma for n=25 n=50 n=250 n=1000 with e=0.01 is: ', TT_gamma_vec)
    axis_time = [0,1,2,3]
    plt.plot(axis_time, AMM_delta_vec, 'r-', lw=2, label='Delta')
    plt.plot(axis_time, AMM_gamma_vec, 'b-', lw=2, label='Gamma')
    label = ['Delta','Gamma']
    plt.xlabel("Level of Mesh")
    plt.ylabel("Computation time for Delta and Gamma")
    plt.title("Adaptive Mesh Delta and Gamma vs. Level Of Mesh")
    plt.legend(label)
    plt.grid(True)
    plt.show()