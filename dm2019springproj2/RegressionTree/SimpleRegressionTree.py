class SimpleRegressionTree():
    def __init__(self):
        return
    def err(self, dataSet):
        return np.var(dataSet[:, -1]) * shape(dataSet)[0]
    
    def splitDataSet(self, dataSet, feature, value):
        '''
        Input:
            dataSet：当前数据集
            feature：切分变量[列名]
            value：划分点
        Output:
            dataSet1：在feature上<=value的子数据集
            dataSet2：在feature上>value的子数据集
        '''
        dataSet1 = dataSet[dataSet[:, feature] <= value] # 左边
        dataSet2 = dataSet[dataSet[:, feature] > value] # 右边
        return dataSet1, dataSet2
    
    def chooseBestFeature(self, dataSet, min_sample=4, epsilon=0.5):
        '''
        Input:
            dataSet：当前数据集
            min_sample：每次划分后，每部分最少的数据数量
            epsilon：误差下降阈值，值越大树的深度越大
        Output:
            bestColumn：最优划分属性
            bestValue：最优划分点
        '''
        features = dataSet.shape[1] - 1 # 特征数量（除去最后一列的标签值）
        sErr = err(dataSet) # 当前数据集的损失
        minErr = np.inf # 初始化最小误差
        bestColumn = 0 # 最优划分特征
        bestValue = 0 # 最优划分值
        nowErr = 0 # 当前误差

        # 如果数据都是一类，无须进行划分
        if len(np.unique(dataSet[:, -1].T.tolist())) == 1:
            return None, np.mean(dataSet[:, -1])
        # 每个特征循环，寻找最优特征
        for feature in range(0, features):
            # 遍历每一行数据，寻找最优划分点
            for row in range(0, dataSet.shape[0]):
                dataSet1, dataSet2 = splitDataSet(dataSet, feature, dataSet[row, feature]) # 划分后的数据
                # 不满足min_sample，直接跳过这种不合法的划分
                if len(dataSet1) < min_sample or len(dataSet2) < min_sample:
                    continue
                # 计算当前这种划分的误差
                nowErr = err(dataSet1) + err(dataSet2)
                # 维护最优的划分（最优属性和对应的最优划分点）
                if nowErr < minErr:
                    minErr = nowErr
                    bestColumn = feature
                    bestValue = dataSet[row, feature]
        # 当划分前后误差下降较小时，直接返回
        if (sErr - minErr) < epsilon:
            return None, np.mean(dataSet[:, -1])

        # 获得当前最优划分
        dataSet1, dataSet2 = splitDataSet(dataSet, bestColumn, bestValue)
        if len(dataSet1) < min_sample or len(dataSet2) < min_sample:
            return None, np.mean(dataSet[:, -1])

        return bestColumn, bestValue

    def createTree(self, dataSet):
        '''
        Input:
            dataSet: 数据集D
        Output:
            决策树T
        '''
        bestColumn, bestValue = chooseBestFeature(dataSet)
        if bestColumn == None:
            return bestValue
        retTree = {} # 初始化决策树
        retTree['spCol'] = bestColumn # 最优划分属性（列）
        retTree['spVal'] = bestValue # 最优分割值
        lSet,rSet = splitDataSet(dataSet, bestColumn, bestValue) # 最优划分
        retTree['left'] = createTree(lSet)
        retTree['right'] = createTree(rSet)
        return retTree
    
    def prune(self, tree, testData):
        if shape(testData)[0] == 0:
            return getMean(tree)
        if (isTree(tree['right']) or isTree(tree['left'])):
            lSet, rSet = splitDataSet(testData, tree['spCol'], tree['spVal'])
        if isTree(tree['left']):
            tree['left'] = prune(tree['left'], lSet)
        if isTree(tree['right']):
            tree['right'] = prune(tree['right'], rSet)

        # 如果两个分支不再是子树，合并
        # 合并前后的误差进行比较，如果合并后的误差比较小，则合并，否则不操作
        if not isTree(tree['left']) and not isTree(tree['right']):
            lSet, rSet = splitDataSet(testData, tree['spCol'], tree['spVal'])
            errMerge = err(dataSet)
            errNoMerge = err(lSet) + err(rSet)
            if errMerge > errNoMerge:
                print('merging')
                return (tree['left'] + tree['right']) / 2.0
            else:
                return tree

    def isTree(self, obj):
        return (type(obj).__name__ == 'dict')

    def getMean(self, obj):
        if isTree(tree['right']):
            tree['right'] = getMean(tree['right'])
            tree['left'] = getMean(tree['left'])
            return (tree['left'] + tree['right']) / 2.0
        
    # 导入数据集
    def loadData(self, filaName):
        dataSet = []
        fr = open(filaName)
        for line in fr.readlines():
            curLine = line.strip().split('\t')
            theLine = []
            for item in curLine:
                item = float(item)
                theLine.append(item)
            dataSet.append(theLine)
        return dataSet
    def getTree(self):
        return 