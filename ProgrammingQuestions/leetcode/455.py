class Solution:
    def findContentChildren(self, g: List[int], s: List[int]) -> int:
        g.sort()
        s.sort()
#         获得在s中最大的数字
        while g and s:
            max_s = s.pop()
            left = 0
            right = len(g) -1
            while left<=right:
                mid = (left+right)//2
                if g[mid] <max_s:
                    left = mid + 1
                elif g[mid]>max_s:
                    right = mid-1
                else:
                    break
            content = 0
            # right所在的位置就是刚好小于max_s的数字在g中的位置
            if right < 1:
                return right+1
            else:
                content+=1
                g = g[:right]
                # s.pop()