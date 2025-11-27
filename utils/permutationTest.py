import numpy as np
def permutation_test(data1, data2, num_permutations=1000):

    observed_diff = np.abs(np.mean(data1) - np.mean(data2))
    combined = np.concatenate([data1, data2])
    count = 0
    shuffDiffs = np.zeros(num_permutations)

    for i in range(num_permutations):
        np.random.shuffle(combined)
        new_data1 = combined[:len(data1)]
        new_data2 = combined[len(data1):]
        new_diff = np.abs(np.mean(new_data1) - np.mean(new_data2))
        shuffDiffs[i] = new_diff

    p_value = np.sum(shuffDiffs >= observed_diff) / num_permutations

    return p_value, shuffDiffs, observed_diff
