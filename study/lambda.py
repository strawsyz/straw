two_sum = lambda x, y: x + y
sum = lambda x, y=100: x + y
# sum_with_100 = sum(100)
# result = sum_with_100(200)
# print(result)

lower = lambda x, y: x if x < y else y
print(lower(7, 10))

d = [{"order": 3}, {"order": 1}, {"order": 2}]
d.sort(key=lambda x: x['order'])
print(d)
