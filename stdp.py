# coding: utf-8
import numpy as np

# 只有一个简单的二层结构，看看这个二元关系能学到什么

#biReLU
def activate(X, w):
    return np.minimum(np.maximum(np.dot(X, w), 0), 1)

def main():
    # 7 * 7 (kernel_size = 7) -> 100 (n_features = 100)
    # w = 49 * 100
    n_features = 100
    kernel_size = 7

    inputs = np.asarray([ np.eye(kernel_size) ])
    inputs = inputs.reshape(-1, kernel_size ** 2)

    weights = np.random.normal(0, 0.01, kernel_size ** 2 * n_features)
    weights = weights.reshape(kernel_size ** 2, n_features)

    # >0ms 70% 40ms以内 <0ms 40% -40ms以内
    print(activate(inputs, weights))

    # update weights
    # select max one to update

    for tmp in range(100):
        a = activate(inputs, weights)
        max_idx = np.argmax(a)
        max_weight = 1
        for i in range(kernel_size ** 2):
            # 下沉
            weights[i][max_idx] = weights[i][max_idx] * 1.6
            max_weight = weights[i][max_idx]

        if max_weight > 1:
            weights = weights / max_weight
   
        print(a)

main()
