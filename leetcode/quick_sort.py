def quick_sort(lst):
    # print(lis)
    # if len(lis) == 0:
    #     return []
    nums = len(lst)
    if nums <= 1:
        return lst
    temp = lst[0]
    smaller_numbers = []
    bigger_numbers = []
    res = []
    for i in range(1, nums):
        if temp > lst[i]:
            smaller_numbers.append(lst[i])
        else:
            bigger_numbers.append(lst[i])
    if smaller_numbers != []:
        res = res + quick_sort(smaller_numbers)
    res = res + [temp]
    if bigger_numbers != []:
        res = res + quick_sort(bigger_numbers)
        # res.append(quick_sort(bigger_numbers))
    # res.append(quick_sort(smaller_numbers)).append(temp).append(bigger_numbers)
    return res


if __name__ == '__main__':
    # temp =
    res = quick_sort([1, 2, 2, 4, 6, 3, 45, 5, 6])
    print(res)
