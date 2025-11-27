import numpy as np
def permutation_test(data1, data2, num_permutations=1000):

    observed_diff = np.abs(np.mean(data1) - np.mean(data2))
    combined = np.concatenate([data1, data2])
    count = 0

    for _ in range(num_permutations):
        np.random.shuffle(combined)
        new_data1 = combined[:len(data1)]
        new_data2 = combined[len(data1):]
        new_diff = np.abs(np.mean(new_data1) - np.mean(new_data2))
        if abs(new_diff) >= abs(observed_diff):
            count += 1

    p_value = count / num_permutations
    return p_value
