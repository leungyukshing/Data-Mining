## 中山大学数据挖掘项目2--并行化随机森林

## Intro

中山大学数据挖掘课程项目2项目代码

使用MPI实现并行化随机森林。

## 文件结构

+ `bagging`：后期模型集成代码
+ `code`：主要代码
+ `data`：数据集
+ `model`：存放模型
+ `RegressionTree`：自定义决策回归树
+ `result`：存放中间及输出结果
+ `test`：部分测试代码（可忽略）

## How to run?

+ Prerequisite: Python3.6+, MPI

+ You have to create the folders manually, following the structure stated in the last part! Make sure put train and test data into `data/` .

+ Open your cmd, enter the `code/`:

  ```bash
  $ mpiexec -n 5 python baseline_train.py
  ```

  Then you will find models will be generated and stored in the `model/`. 

  ```bash
  $ mpiexec -n 36 python baseline_test.py
  ```

  Then you will find there are 6 result files stored in `result/`.

  ```bash
  $ python combine_result.py
  ```

  Use this script to combine the results into one file, which is required to summit to the website.



For any questions or suggestions, please feel free to contact me!