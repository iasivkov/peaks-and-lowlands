import numpy as np
import sys
sys.path.insert(0,"..")
from src import Landshaft
import time

np.random.seed(12)

def main():
    mean, sigma, n = [0, 8, 3, 14, 18], [1, 1.5, 0.5, 2, 0.7], 200000
    x1 = np.random.normal(mean[0], sigma[0], n)
    x2 = np.random.normal(mean[1], sigma[1], n)
    x3 = np.random.normal(mean[2], sigma[2], n)
    x4 = np.random.normal(mean[3], sigma[3], n)
    x5 = np.random.normal(mean[4], sigma[4], n)
    x = np.concatenate([x1,x2,x3,x4,x5])   
    x=x

    landshaft = Landshaft(x=x,q_num=50, threshold=0.33)
    start = time.time()
    landshaft.build_landshaft()
    end = time.time()
    print("Elapsed time: ", end - start)
    landshaft.plot_ponds_and_peaks()
 
if __name__=="__main__":
    main()
    