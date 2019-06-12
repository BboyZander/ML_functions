from functool import reduce
from math import ceil

def deep_flatten(arr):
    """
    Extend your list of lists into one list
    :param arr: list of lists
    """
    def spread(arg):
        """
        # Выравнивание списка при помощи рекурсии.
        Используется list.extend() вместе с пустым массивом (result) и
        функция spread для сглаживания каждого элемента списка
        """
        ret = []
        for i in arg:
            if isinstance(i, list):
                ret.extend(i)
            else:
                ret.append(i)
        return ret

    result = []
    result.extend(spread(list(map(lambda x: deep_flatten(x) if type(x) == list else x, arr))))
    return result

def count_occurences(arr, val):
    """
    return count of val from your array
    """
    return reduce((lambda x, y: x + 1 if y == val and type(y) == type(val) else x + 0), arr)

def chunk(arr, size):
    """
    Cut array into 'size' splits
    """
    return list(
        map(lambda x: arr[x * size:x * size + size],
            list(range(0, ceil(len(arr) / size)))))