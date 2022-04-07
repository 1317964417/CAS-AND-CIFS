import numpy as np

# 对一个行或列升序排列并扩展到其他行或列（降序排列则取负即可）
a=np.array([2,1,4,3])
b=np.array([4,3,2,1])

a_idx=np.argsort(a)
print(a_idx)
# [1 0 3 2]

a_sort=a[a_idx]
print(a_sort)
# [1 2 3 4]

b_sort_like_a = b[a_idx]
print(b_sort_like_a)
# [3 4 1 2]
print(a_sort)
# [1 2 3 4]



