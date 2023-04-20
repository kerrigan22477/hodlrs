import time
from main import Main
import numpy as np
from solveForX import SolveForX
from numpy.linalg import inv
import matplotlib.pyplot as plt

m = Main()

def Time(r, size):
                                 # r, k
    hodlr, root, covMat = m.main(r, 2, size)
    # solve For X
    np.set_printoptions(suppress=True)
    s = SolveForX()
    b = np.arange(len(covMat)).T

    test_start = time.time()
    test = np.linalg.solve(covMat, b)
    test_end = time.time()
    test_time = test_end - test_start
    # print(test.round(2))

    start = time.time()
    max_level = np.log2(len(covMat)) - 1
    y, Kli, update, next_update = s.solveForX(b, root, int(max_level))
    end = time.time()
    my_time = end - start

    return test_time, my_time

# profile for different size matrices and different sizes of R
# without R, should be cubic and n log n - double check with paper
# log t on x log n on y plots
ts = []
ms = []
sizes = [2**i for i in range(3,12)]

# increasing sizes, no R
for i in range(3,12):
    num_points = 2**i
    r = 1
    t_time, m_time = Time(r, num_points)
    ts.append(t_time)
    ms.append(m_time)
    print(i)
plt.loglog(sizes, ts, label='Numpy SolveForX')
plt.loglog(sizes, ms, label='Hodlr SolveForX')
plt.xlabel('Number of Data Points')
plt.ylabel('Runtimes')
plt.legend()
plt.show()

'''
trs = []
mrs = []
rs = [i for i in range(128,32,-1)]

# decreasing R same size
for r in range(128, 32, -1):
    num_points = r // 8
    t_time, m_time = Time(r, num_points)
    trs.append(t_time)
    mrs.append(m_time)
plt.loglog(rs, trs, label='Numpy SolveForX')
plt.loglog(rs, mrs, label='Hodlr SolveForX')
plt.xlabel('size of R')
plt.ylabel('Runtimes')
plt.legend()
plt.show()'''


'''plt.loglog(dt,ec_us,'r-', label='Euler Cromer Slope = ' + str(round(ec_slope, 1)))
plt.loglog(dt,e_us,'b-', label='Euler Slope = ' + str(round(e_slope, 1)))
plt.xlabel('dT')
plt.ylabel('1-step Energy Error')
plt.legend()'''