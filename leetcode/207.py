
# DFS
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        # 存储每种课的依赖的所有课程
        # 根据课程的依赖关系建立的图结构
        graphic = [[] for _ in range(numCourses)]
        for pre in prerequisites:
            graphic[pre[0]].append(pre[1])
        # 存储每个课程的调查状态
        # 0表示还没有调查，初始状态
        # 1表示不清楚调查结果
        # 2表示没有构成环
        status = [0] * numCourses
        for i in range(numCourses):
            if self.exist_cycle(status, graphic, i):
                # 如果检测到环了，就无法结束所有的课程了
                return False
        return True

    def exist_cycle(self, status, graphic, cur_node):
        # 如果遇到了一个正在调查的点，说明已经遇到一个环
        if status[cur_node] == 1:
            return True
        if status[cur_node] == 2:
            return False
        # 标志该课程为1，表示正在调查该课程
        status[cur_node] = 1
        # 遍历学习该课程前需要学习的课程
        for next_node in graphic[cur_node]:
            if self.exist_cycle(status, graphic, next_node):
                # 如果发现有一个环，说明，当前这个图有环
                return True
        # 走出循环，说明已经遍历了所有预备课程，没有形成环
        status[cur_node] = 2
        return False
