import numpy as np
import matplotlib.pyplot as plt
import MetaHeuristic.HC as HC
import MetaHeuristic.SA as SA

class TS:
    def __init__(self, num_runs, num_iters, num_pattern, file_path, tabuListSize):
        # 定義接收參數
        self.x_num_runs = num_runs  # 執行數
        self.x_num_iters = num_iters  # 迭代數
        self.x_num_patterns_sol = num_pattern  # 解的大小
        self.x_filename_ini = file_path  # 初始值路徑
        self.x_Tabu_Size = tabuListSize  # Tabu list大小
        # 定義所需參數
        self.sol = []  # 历史解
        self.tmp_sol = []  # 当前解
        self.obj_value = 0  # 历史值
        self.avg_obj_value = 0.0  # 平均最優值
        self.avg_obj_value_iter = []  # 迭代平均最優值
        self.best_obj_value = 0  # 最優值
        self.best_sol = []  # 最優解
        # 定義TabuSearch所需變量
        self.TabuList = []  # 存放最近解集
        self.TabuKey = 0  # 記錄最近解所在位置

    def run(self):
        # 初始化迭代記錄
        self.avg_obj_value = 0.0
        self.avg_obj_value_iter = np.zeros(self.x_num_iters)
        # 開始Tabu Search
        for r in range(0, self.x_num_runs):
            # 0. Initialization
            self.init()
            self.obj_value = self.evaluate(self.sol)
            for i in range(0, self.x_num_iters):
                # 1. Transition
                self.tmp_sol = self.transit(self.sol)
                # 2. Evaluation
                tmp_obj_value = self.evaluate(self.tmp_sol)
                # 3. Determination
                if self.determine(tmp_obj_value, self.obj_value):
                    self.obj_value = tmp_obj_value
                    self.sol = self.tmp_sol
                    # 更新 Tabu List
                    if len(self.TabuList) < self.x_Tabu_Size:
                        self.TabuList.append(list(self.sol))  # Tabu List未滿，加入新解
                    else:
                        self.TabuList[self.TabuKey] = list(self.sol)  # FIFO 原則更新 Tabu List
                        self.TabuKey = (self.TabuKey + 1) % self.x_Tabu_Size
                # 更新最優解
                if self.obj_value > self.best_obj_value:
                    self.best_obj_value = self.obj_value
                    self.best_sol = self.sol
                # 記錄當前迭代最優解
                self.avg_obj_value_iter[i] = self.avg_obj_value_iter[i] + self.best_obj_value
            # 累加最優質解
            self.avg_obj_value += self.best_obj_value

        # 4.Output
        self.avg_obj_value /= self.x_num_runs
        for i in range(0, self.x_num_iters):
            self.avg_obj_value_iter[i] = self.avg_obj_value_iter[i] / self.x_num_runs
            # 輸出每次迭代的平均最優解
            # print(self.avg_obj_value_iter[i])
        print("===============best_value=============")
        print(self.avg_obj_value)
        return self.avg_obj_value_iter

    def init(self):
        self.sol = []
        self.obj_value = 0
        self.best_obj_value = 0
        self.TabuList = []
        self.TabuKey = 0
        # 隨機生成解集或讀入外部解集
        if len(self.x_filename_ini) > 0:
            print("外部讀入解集功能尚未實現!")
        else:
            for i in range(0, self.x_num_patterns_sol):
                self.sol.append(np.random.randint(0, 100) % 2)

    def evaluate(self, solution):
        count = 0
        for i in range(0, self.x_num_patterns_sol):
            count += solution[i]
        return count

    def transit(self, solution):
        i = np.random.randint(0, self.x_num_patterns_sol * 100) % self.x_num_patterns_sol
        tmp_sol = list(solution)
        tmp_sol[i] = 1 - tmp_sol[i]
        # 判斷 Tabu list 是否填滿
        if_tabu = False
        if len(self.TabuList) >= self.x_Tabu_Size:
            # 檢查 tabu list
            for tabu in self.TabuList:
                if tabu == tmp_sol:
                    if_tabu = True
        # 若存在于tabu list，返回原解
        if if_tabu:
            return solution
        else:
            return tmp_sol

    def determine(self, tmp_obj_value, obj_value):
        return tmp_obj_value > obj_value


ts = TS(30, 1024, 100, "", 10)
avg_obj_value_iter = ts.run()
# print(avg_obj_value_iter)
# SA
sa = SA.SA(30, 1024, 100, "", 0.00001, 1.0)
y2 = sa.run()
# HC
hc = HC.HC(30, 1024, 100, "")
y3 = hc.run()
# 繪圖比較HC / SA / TS
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
x = [i for i in range(0,len(avg_obj_value_iter))]
ax1.plot(x, avg_obj_value_iter, color='blue')
ax1.plot(x, y2, color='red')
ax1.plot(x, y3, color='green')
plt.show()
