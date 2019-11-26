import numpy as np
import matplotlib.pyplot as plt
import random as rand


class GA:
    def __init__(self, num_runs, num_iters, num_pattern, num_pop, mutation_prob, file_path):
        # 接收參數
        self.x_num_runs = num_runs  # 執行數
        self.x_num_iters = num_iters  # 迭代數
        self.x_num_patterns_sol = num_pattern  # 基因的大小
        self.x_num_population = num_pop  # 族群的大小
        self.x_mutation_prob = mutation_prob
        self.x_filename_ini = file_path  # 初始值路徑
        # 定義GA所需參數
        self.sol = []  # 历史解
        self.sol_value = []  # 當前解集的結果值
        self.parent = []  # 每次 generation 的親代解
        self.children = []  # 每次 generation 的子代解s
        self.parent_size = self.x_num_population // 2  # 親代個數
        self.children_size = self.x_num_population - self.parent_size  # 子代個數
        # 定義迭代所需參數
        self.avg_obj_value = 0.0  # 平均最優值
        self.avg_obj_value_iter = []  # 迭代平均最優值
        self.best_obj_value = 0  # 最優值
        self.best_sol = []  # 最優解
        # 定義輪盤法所需參數
        self.WheelProb = []
        # 定義競爭法所需參數
        self.Tour_candi = []

    def run(self):
        # 初始化迭代記錄
        self.avg_obj_value = 0.0
        self.avg_obj_value_iter = np.zeros(self.x_num_iters)
        # 開始一次 Run
        for r in range(0, self.x_num_runs):
            # 0. Initialization
            self.init()
            # print(self.sol)
            # 開始一次 Generation
            for i in range(0, self.x_num_iters):
                # 1. Evaluation
                (self.best_obj_value, self.best_sol) = self.evaluate(self.sol)
                # print(self.sol_value)
                # 2. Selection
                # ====================================
                # self.parent = self.selection_wheel(self.sol)
                # self.parent = self.selection_wheel_sq(self.sol)
                # self.parent = self.selection_wheel_trans(self.sol)
                # ====================================
                self.parent = self.selection_tournament(self.sol)
                # ====================================
                # 3. crossover
                self.children = self.crossover(self.parent)
                # 4. Mutation
                self.children = self.mutation(self.children)
                self.sol = list(self.parent + self.children)
                # 打亂 list 次序
                rand.shuffle(self.sol)
                # 記錄當前迭代最優解
                self.avg_obj_value_iter[i] = self.avg_obj_value_iter[i] + self.best_obj_value
            # 累加最優質解
            self.avg_obj_value += self.best_obj_value
            print(self.best_obj_value)
        # output
        self.avg_obj_value /= self.x_num_runs
        for i in range(0, self.x_num_iters):
            self.avg_obj_value_iter[i] = self.avg_obj_value_iter[i] / self.x_num_runs
        # print(self.avg_obj_value)
        return self.avg_obj_value_iter

    def init(self):
        print("=============Initialization!============")
        self.sol = []  # 历史解
        self.sol_value = []  # 當前解集的結果值
        self.parent = []  # 每次 generation 的親代解
        self.children = []  # 每次 generation 的子代解
        self.best_obj_value = 0  # 最優值
        self.WheelProb = []  # 定義輪盤法所需參數
        self.Tour_candi = []  # 定義競爭法所需參數
        # 隨機生成解集或讀入外部解集
        if len(self.x_filename_ini) > 0:
            print("外部讀入解集功能尚未實現!")
        else:
            for i in range(0, self.x_num_population):
                tmp_gene = []
                for j in range(0, self.x_num_patterns_sol):
                    tmp_gene.append(np.random.randint(0, 100) % 2)
                self.sol.append(list(tmp_gene))

    def crossover(self, sol):
        self.children = []
        for i in range(0, self.parent_size // 2):
            cross_point = np.random.randint(0, self.x_num_patterns_sol)
            tmp_children1 = list(self.parent[2 * i][0:cross_point] + self.parent[2 * i + 1][cross_point:])
            tmp_children2 = list(self.parent[2 * i + 1][0:cross_point] + self.parent[2 * i][cross_point:])
            self.children.append(tmp_children1)
            self.children.append(tmp_children2)
        return self.children

    def mutation(self, tmp_sol):
        for i in range(0, self.children_size):
            if_mutate = (np.random.rand() < self.x_mutation_prob)
            if if_mutate:
                mutate_point = np.random.randint(0, self.x_num_patterns_sol)
                self.children[i][mutate_point] = 1 - self.children[i][mutate_point]
        return self.children

    def evaluate(self, sol):
        tmp_value = []
        self.sol_value = []
        best_value = 0
        best_sol = []
        for i in range(0, self.x_num_population):
            count = 0
            for j in range(0, self.x_num_patterns_sol):
                count += sol[i][j]
            self.sol_value.append(count)
            # update best_value
            if count > best_value:
                best_value = count
                best_sol = list(sol[i])
        return best_value, best_sol

    def selection_wheel(self, sol):
        # print((self.evaluate(sol))[0])
        self.parent = []
        self.WheelProb = []
        # 計算被選概率
        sum_fitness = sum(i for i in self.sol_value)
        for i in range(0, self.x_num_population):
            self.WheelProb.append(self.sol_value[i] / sum_fitness)
        # 進行輪盤選擇
        # print(sum(self.WheelProb))
        for i in range(0, self.parent_size):
            prob = np.random.rand()
            select_num = -1
            while prob > 0:
                select_num += 1
                prob -= self.WheelProb[select_num]
            self.parent.append(list(self.sol[select_num]))
        # print((self.evaluate(self.parent + self.parent))[0])
        return self.parent

    def selection_wheel_sq(self, sol):
        # print((self.evaluate(sol))[0])
        self.parent = []
        self.WheelProb = []
        # 計算被選概率
        sum_fitness = sum(np.square(i) for i in self.sol_value)
        for i in range(0, self.x_num_population):
            self.WheelProb.append(np.square(self.sol_value[i]) / sum_fitness)
        # 進行輪盤選擇
        for i in range(0, self.parent_size):
            prob = np.random.rand()
            select_num = -1
            while prob > 0:
                select_num += 1
                prob -= self.WheelProb[select_num]
            # print("select_num")
            # print(select_num)
            self.parent.append(list(self.sol[select_num]))
        # print((self.evaluate(self.parent + self.parent))[0])
        return self.parent

    def selection_wheel_trans(self, sol):
        # print((self.evaluate(sol))[0])
        self.parent = []
        self.WheelProb = []
        # 將 value 標準化
        min_value = min(self.sol_value)
        value_trans = list(i - min_value + 0.1 for i in self.sol_value)
        # 計算被選概率
        sum_fitness = sum(i for i in value_trans)
        for i in range(0, self.x_num_population):
            self.WheelProb.append(value_trans[i] / sum_fitness)
        # 進行輪盤選擇
        # print(sum(self.WheelProb))
        for i in range(0, self.parent_size):
            prob = np.random.rand()
            select_num = -1
            while prob > 0:
                select_num += 1
                prob -= self.WheelProb[select_num]
            self.parent.append(list(self.sol[select_num]))
        # print((self.evaluate(self.parent + self.parent))[0])
        # print(len(self.parent))
        return self.parent

    def selection_tournament(self, sol):
        # 初始化
        self.Tour_candi = [i for i in range(0, len(sol))]
        rand.shuffle(self.Tour_candi)
        self.parent = []
        # 開始錦標賽
        for i in range(0, len(self.Tour_candi)//2):
            if self.sol_value[self.Tour_candi[2*i]] > self.sol_value[self.Tour_candi[2*i + 1]]:
                self.parent.append(list(sol[self.Tour_candi[2*i]]))
            else:
                self.parent.append(list(sol[self.Tour_candi[2 * i + 1]]))
        print("=============self.parent==========")
        print(len(self.parent))
        print(self.parent)
        return self.parent


ga = GA(10, 1000, 100, 16, 0.1, "")
avg_obj_value_iter = ga.run()
x = [i for i in range(0, len(avg_obj_value_iter))]
plt.plot(x, avg_obj_value_iter, color='blue')
plt.show()
