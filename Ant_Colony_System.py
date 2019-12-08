import numpy as np
import matplotlib.pyplot as plt
import random as rand
import math


class ACO:
    def __init__(self, num_runs, num_iters, num_ants, tsp_file_path, alpha, beta):
        # 接收參數
        self.tsp_path = tsp_file_path  # TSP 文件位置
        self.x_num_ants = num_ants  # 螞蟻只數
        self.x_num_runs = num_runs  # 回合數
        self.x_num_iters = num_iters  # 迭代數
        self.alpha = alpha  # 費洛蒙衰減率
        self.beta = beta  # 距離權重
        # 定義ACO所需參數
        self.q0 = 0.9  # 隨機/輪盤選擇概率
        self.rou = 0.1  # 局部費洛蒙衰減率
        self.f0 = 0  # f0 = （n*lnn）^-1
        self.Q = 0.8  # Ant_Q 的學習率     
        # 定義ACO所需變量
        self.city_location = []  # TSP城市位置
        self.city_num = 0  # TSP城市數量
        self.tour_table = []  # 存放每隻螞蟻的路徑(m*n); m：螞蟻數，n: TSP點數
        self.ant_distance = []  # 存放每隻螞蟻的路徑長度(m)
        self.phero_table = []  # 存取每條路徑的費洛蒙量(n*n)
        self.dist_table = []  # 存取TSP點間的路徑距離(n*n)
        self.left_city = []  # 存取每隻螞蟻尚未經過的城市（m*n`）
        # 定義迭代所需參數
        self.avg_obj_value = 0.0  # 平均最優值
        self.avg_obj_value_iter = []  # 迭代平均最優值
        self.best_obj_value = 0  # 最優值
        self.best_sol = []  # 最優解

    def run(self):
        # 初始化迭代記錄
        self.avg_obj_value = 0.0
        self.avg_obj_value_iter = np.zeros(self.x_num_iters)
        # 0.讀入路徑文件並生成距離矩陣
        self.read_TSP(self.tsp_path)
        # 開始一次 Run
        for r in range(0, self.x_num_runs):
            print("=====================run=========================")
            # 1. Initialization
            self.init()
            for i in range(0, self.x_num_iters):
                # print("=========iteration==============")
                # 2. Update the tour list
                self.update_tour(self.tour_table)
                # 3. Update global best
                self.update_global()
                # 4. 記錄當前迭代最優解
                self.avg_obj_value_iter[i] = self.avg_obj_value_iter[i] + self.best_obj_value
            # 累加 run 最優質解
            self.avg_obj_value += self.best_obj_value
            # print(self.best_obj_value)
        # 5.Output
        self.avg_obj_value /= self.x_num_runs
        for i in range(0, self.x_num_iters):
            self.avg_obj_value_iter[i] = self.avg_obj_value_iter[i] / self.x_num_runs
        return self.avg_obj_value_iter

    def read_TSP(self, path):
        self.city_location = []
        self.city_num = 0
        # 讀取tsp文件並記錄城市位置
        f = open(path)
        for line in f:
            if line[0:2] == "EOF":
                break
            elif line[0].isdigit():
                loc = line.split()
                self.city_location.append([int(loc[1]), int(loc[2])])   # 保存(x, y)位置
                self.city_num += 1
        # 記錄距離矩陣
        self.dist_table = np.zeros((self.city_num, self.city_num))  # 存取TSP點間的路徑距離(n*n)
        for i in range(0, self.city_num):
            for j in range(0, self.city_num):
                # 歐氏距離計算城市間距
                dist = math.sqrt((self.city_location[j][0] - self.city_location[i][0]) ** 2 +
                                 (self.city_location[j][1] - self.city_location[i][1]) ** 2)
                if dist == 0:
                    dist = float('inf')  # 對角線距離設為無窮
                self.dist_table[i][j] = dist

    def init(self):
        self.best_obj_value = float('inf')  # 初始化迭代記錄
        self.best_sol = []
        self.f0 = 0
        self.tour_table = np.zeros((self.x_num_ants, self.city_num), dtype=int)  # 初始化螞蟻路徑
        self.phero_table = np.zeros((self.city_num, self.city_num))  # 初始化費洛蒙
        self.left_city = []  # 初始化螞蟻尚未經過的城市
        # 1) 隨機分佈起始點
        for i in range(0, self.x_num_ants):
            start = np.random.randint(0, self.city_num)
            self.tour_table[i][0] = start  # 作為路徑起點
        # 2) 計算Lnn距離
        all_city = []
        for i in range(0, self.city_num):
            all_city.append(i)
        (lnn, lnn_tour) = self.lNN(all_city)
        # 3) 初始化費洛蒙表：f0 = （n*lnn）^-1
        self.f0 = 1 / (self.city_num * lnn)
        for i in range(0, self.city_num):
            for j in range(0, self.city_num):
                self.phero_table[i][j] = self.f0
        '''
        # 可選：4) 基於 L_nn 更新費洛蒙表
        for i in range(0, self.city_num - 1):
            sp = lnn_tour[i]
            ep = lnn_tour[i + 1]
            self.phero_table[sp][ep] = (1 - self.alpha) * self.phero_table[sp][ep] + self.alpha * (1 / lnn)
            self.phero_table[ep][sp] = self.phero_table[sp][ep]
        # update the last point
        sp = lnn_tour[self.city_num - 1]  # 終點
        ep = lnn_tour[0]  # 起點
        self.phero_table[sp][ep] = (1 - self.alpha) * self.phero_table[sp][ep] + self.alpha * (1 / lnn)
        self.phero_table[ep][sp] = self.phero_table[sp][ep]
        # print(lnn)
        '''

    def update_tour(self, tour_table):
        # 0) 初始化路徑距離表
        self.ant_distance = np.zeros(self.x_num_ants)
        self.update_tour_table()
        # 開始移動
        for i in range(0, self.city_num):
            # 1) 判斷是否走完全程
            if i < self.city_num - 1:
                # 2) 每隻螞蟻各走一步 # print("==========ant=========")
                for j in range(0, self.x_num_ants):
                    step = -1
                    # 3) 計算 p_j(r,s)
                    (tour_prob, maxStep) = self.cal_prob(i, j)  
                    q = np.random.rand()  # 判斷是否 maxStep
                    # 4) 基於最大值或輪盤法選擇下一步 
                    # exploitation
                    if q < self.q0:
                        step = maxStep
                    # biased maxStep
                    else:
                        step = self.wheels(tour_prob, j)
                    # 5) 基於 step 更新 tour 和 left 並記錄路徑距離
                    self.tour_table[j][i + 1] = int(self.left_city[j][step])
                    self.ant_distance[j] += self.dist_table[self.left_city[j][step]][tour_table[j][i]]
                    self.left_city[j] = np.delete(self.left_city[j], step)
            else:
                # 6) 螞蟻回到起點
                for j in range(0, self.x_num_ants):
                    self.ant_distance[j] += self.dist_table[self.tour_table[j][self.city_num-1]][self.tour_table[j][0]]
            # 7) local update the pheromone table
            # ===================simply Local update==================
            self.local_update_simply(self.phero_table)
            # ===================Local update with rou0 = 0 ==================
            # self.local_ACS_0()
            # =================== Ant - Q==================
            # self.ant_q(self.phero_table)

    def update_global(self):
        # 0) 費洛蒙總體衰減: (1 - alpha)*p(r, s)
        for i in range(0, self.city_num):
            for j in range(0, self.city_num):
                self.phero_table[i][j] = (1 - self.alpha) * self.phero_table[i][j]
        # 1) 找到最優路徑 L best
        index_best = -1
        L_best = float('inf')
        for i in range(0, self.x_num_ants):
            if self.ant_distance[i] < L_best:
                index_best = i
                L_best = self.ant_distance[i]
        # 2) update the global best value
        Best_tour = self.tour_table[index_best]
        if L_best < self.best_obj_value:
            self.best_obj_value = L_best
            self.best_sol = Best_tour
        # 3) update the pheromone table: p(r, s) + alpha*delta(r, s)
        for i in range(0, self.city_num - 1):
            sp = Best_tour[i]
            ep = Best_tour[i + 1]
            self.phero_table[sp][ep] = self.phero_table[sp][ep] + self.alpha * (1 / L_best)
            self.phero_table[ep][sp] = self.phero_table[sp][ep]
        # 4) update the last point
        sp = Best_tour[self.city_num - 1]  # 終點
        ep = Best_tour[0]  # 起點
        self.phero_table[sp][ep] = self.phero_table[sp][ep] + self.alpha * (1 / L_best)
        self.phero_table[ep][sp] = self.phero_table[sp][ep]

    def array_delete(self, param, start):
        # 刪除數列param中值為start的位置
        tmp_set = []
        for j in range(0, len(param)):
            if param[j] == start:
                tmp_set = np.delete(param, j)
                break
        return tmp_set

    def lNN(self, all_city):
        lnn = 0
        tmp_city = 0
        next_city = 0
        lnn_tour = [tmp_city]
        for i in range(0, self.city_num - 1):
            all_city = self.array_delete(all_city, tmp_city)
            # 1) 找到當前城市的最近鄰
            min_dist = 100000
            for i in range(0, len(all_city)):
                temp_dist = self.dist_table[tmp_city][all_city[i]]
                if temp_dist < min_dist:
                    min_dist = temp_dist
                    next_city = all_city[i]
            # 2) 計算距離並更新當前城市
            tmp_city = next_city
            lnn += min_dist
            lnn_tour.append(tmp_city)
        lnn += self.dist_table[tmp_city][0]
        # print(lnn_tour)
        return lnn, lnn_tour

    def cal_prob(self, i, j):
        max_s = -1
        max_prob = 0
        ant_tour_prob = np.zeros(self.city_num)
        # 1) 計算分子
        for k in range(0, len(self.left_city[j])):
            ci = self.left_city[j][k]  # city index
            start_p = self.tour_table[j][i]  # 起點
            # 應用 p_k(r, s) 公式
            ant_tour_prob[ci] = self.phero_table[start_p][ci] * pow(1 / self.dist_table[start_p][ci], self.beta)
            # 更新最大值
            if ant_tour_prob[ci] > max_prob:
                max_s = k
                max_prob = ant_tour_prob[ci]
        # 2) 計算分母
        sum_tour_prob = sum(ant_tour_prob)
        # 3) 計算概率
        # sum_test = 0  # 測試總和是否為0
        for k in range(0, len(ant_tour_prob)):
            ant_tour_prob[k] = ant_tour_prob[k] / sum_tour_prob
            # sum_test += ant_tour_prob[k]
        return ant_tour_prob, max_s

    def update_tour_table(self):
        # 初始化 tour_table 除起點外全部位置
        for i in range(0, self.x_num_ants):
            for j in range(1, self.city_num):
                self.tour_table[i][j] = -1
        # 補充 left_city
        self.left_city = []  # 存取每隻螞蟻尚未經過的城市（m*n`）
        all_city = []
        for i in range(0, self.city_num):
            all_city.append(i)
        for i in range(0, self.x_num_ants):
            self.left_city.append(all_city)
        # 從left中刪除起點
        for i in range(0, self.x_num_ants):
            self.left_city[i] = self.array_delete(self.left_city[i], self.tour_table[i][0])

    def wheels(self, tour_prob, j):
        p1 = np.random.rand()
        # print("====================wheels============")
        select_num = -1
        maxStep = -1
        # 輪盤法選擇目標城市
        while p1 > 0:
            select_num += 1
            p1 -= tour_prob[select_num]
        # 查找目標城市在 left 中的位置
        for k in range(0, len(self.left_city[j])):
            ci = self.left_city[j][k]
            if ci == select_num:
                maxStep = k
                break
        return maxStep

    def local_ACS_0(self):
        # delta(rou(r, s)) = 0
        for i in range(0, self.city_num):
            for j in range(0, self.city_num):
                self.phero_table[i][j] = (1 - self.rou) * self.phero_table[i][j]

    def local_update_simply(self, phero_table):
        # delta(rou(r, s)) = rou0
        for i in range(0, self.city_num):
            for j in range(0, self.city_num):
                self.phero_table[i][j] = (1 - self.rou) * self.phero_table[i][j] + self.rou * self.f0

    def ant_q(self, phero_table):
        # Search the z s.t. arg max(r(i,z)
        max_z = []
        for i in range(0, self.city_num):
            max_index = -1
            max_phero = 0
            for j in range(0, self.city_num):
                if self.phero_table[i][j] > max_phero:
                    max_phero = self.phero_table[i][j]
                    max_index = j
            max_z.append(max_index)
        # Local update by max_z
        for i in range(0, self.city_num):
            for j in range(0, self.city_num):
                r_z = self.phero_table[j][max_z[j]]
                self.phero_table[i][j] = (1 - self.rou) * self.phero_table[i][j] + self.rou*self.Q * r_z

aco = ACO(2, 5000, 20, "eil51_tsp.txt", 0.1, 2)
avg_obj_value_iter = aco.run()
x = [i for i in range(0, len(avg_obj_value_iter))]
plt.plot(x, avg_obj_value_iter, color='blue')
print(avg_obj_value_iter)
plt.show()
