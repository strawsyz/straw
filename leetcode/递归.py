def df(n):
    if n < 2:
        return n
    return df(n - 1) + df(n - 2)
if __name__ == '__main__':

    res = df(40)
    print(res)
