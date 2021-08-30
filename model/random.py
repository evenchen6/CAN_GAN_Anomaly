# refer: https://blog.csdn.net/weixin_42686879/article/details/111058885
import numpy as np

def randomint_plus(low, high=None, cutoff=None, size=None):
    """
        Used to generate random integers or sequences that do not contain some intermediate values

    Parameters
    ----------
    low : int
    high: int
    cutoff: int/list
    size: tuple
    Notes
    -----
    1. During the call, if high, cutoff, and size are all defaulted, it will return a value of [0, low) by default
    2. If cutoff, size are default, return a random integer value of [low, high)
    3. If size is default, it returns a random integer value (single) of [low, cutoff) U (cutoff, high)
    4. Return a matrix of a given size, the matrix elements are random integer values of [low, cutoff) U (cutoff, high)

    See Also
    --------
    np.random.randint()

    Usage
        randomint_plus(low=1, high=9, cutoff=[1,2], (2,4))

        >> out
        [[6 3 3 7]
            [7 7 7 4]]

    :returns
        if size is None, return int
        else return numpy.ndarray
    """
    if high is None:
        assert low is not None, "Error"
        high = low
        low = 0

    number = 1
    if size is not None:
        for i in range(len(size)):
            number = number * size[i]

    if cutoff is None:
        random_result = np.random.randint(low, high, size=size)
    else:
        if number == 1:
            random_result = randint_digit(low, high, cutoff)
        else:
            random_result = np.zeros(number, )
            for i in range(number):
                random_result[i] = randint_digit(low, high, cutoff)
            random_result = random_result.reshape(size)

    if number > 1:
        random_result = np.array(random_result).astype(int)

    return random_result


def randint_digit(low, high, cutoff):
    """
        Used to generate a random integer in the interval (low, high) excluding cutoff

    Parameters
    ----------
    low: int
    high: int
    cutoff: int/list

    Usage:
        randint_digit(low=1, high=9, cutoff=[2,4])

        >> out
        1

    :returns int
    """
    digit_list = list(range(low, high))
    if type(cutoff) is int:
        if cutoff in digit_list:
            digit_list.remove(cutoff)
    else:
        for i in cutoff:
            if i not in digit_list:
                continue
            digit_list.remove(i)

    np.random.shuffle(digit_list)

    return digit_list.pop()

if __name__ == '__main__':
    # DEMO
    cutoff1 = 3
    cutoff2 = [1, 2]
    low = 1
    high = 9

    result = randint_digit(low, high, cutoff1)
    print(result)

    result = randomint_plus(low, high, cutoff2, (64, ))
    print(result)