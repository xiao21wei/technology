# 趋势预测报告（prophet模型）

[参考代码链接](**https://github.com/xiao21wei/technology**)

该报告重点关注prophet模型的参数调优问题，通过使用不同的参数组合来构建模型，比较不同模型下的预测结果，比较结果，得出最优参数。最后我们使用得到的最优参数组合来构建模型，并使用该模型进行趋势预测。

我们依次对prophet模型的各个常用参数进行调优。

```python
param_grid = {
        'n_changepoints': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
    }
```

此时最优参数为30

```python
param_grid = {
        'n_changepoints': [26, 27, 28, 29, 30, 31, 32, 33, 34],
    }
```

此时最优参数为30

说明，对于参数`n_changepoints`，最优参数为30

```
param_grid = {
        'n_changepoints': [30],
        'changepoint_range': [0.5, 0.6, 0.7, 0.8, 0.9],
    }
```

此时最优参数为0.6

```
param_grid = {
        'n_changepoints': [30],
        'changepoint_range': [0.50, 0.55, 0.6, 0.65, 0.70],
    }
```

此时最优参数为0.6



```
param_grid = {
        'n_changepoints': [30],
        'changepoint_range': [0.55, 0.56, 0.57, 0.58, 0.59, 0.60, 0.61, 0.62, 0.63, 0.64, 0.65],
    }
```

此时最优参数为0.6

```
'changepoint_prior_scale': [0.01, 0.1, 0.5, 1],
param_grid = {
        'n_changepoints': [30],
        'changepoint_range': [0.6],
        'changepoint_prior_scale': [0.01, 0.1, 0.5, 1],
    }
```

此时最优参数是0.5

```
param_grid = {
        'n_changepoints': [30],
        'changepoint_range': [0.6],
        'changepoint_prior_scale': [0.5],
        'daily_seasonality': [True],
        'seasonality_mode': ['multiplicative', 'additive'],
    }
```

此时最优参数为additive

```
param_grid = {
        'n_changepoints': [30],
        'changepoint_range': [0.6],
        'changepoint_prior_scale': [0.5],
        'daily_seasonality': [True],
        'seasonality_mode': ['additive'],
        'seasonality_prior_scale': [0.1, 0.5, 1, 5, 10, 15, 20],
    }
```

此时最优参数为1

```
param_grid = {
        'n_changepoints': [30],
        'changepoint_range': [0.6],
        'changepoint_prior_scale': [0.5],
        'daily_seasonality': [True],
        'seasonality_mode': ['additive'],
        'seasonality_prior_scale': [1],
        'interval_width': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    }
```

此时最优参数为0.1

```
param_grid = {
        'n_changepoints': [30],
        'changepoint_range': [0.6],
        'changepoint_prior_scale': [0.5],
        'daily_seasonality': [True],
        'seasonality_mode': ['additive'],
        'seasonality_prior_scale': [1],
        'interval_width': [0.1],
        'uncertainty_samples': [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000],
    }
```

此时最优参数为500

经过以上的测试，我们得出了对于该预测情景下，最优的参数组合

```
best_params = {
        'n_changepoints': 30,
        'changepoint_range': 0.6,
        'changepoint_prior_scale': 0.5,
        'daily_seasonality': True,
        'seasonality_mode': 'additive',
        'seasonality_prior_scale': 1,
        'interval_width': 0.1,
        'uncertainty_samples': 500,
    }
```

我们使用这组最优参数来搭建prophet模型，并使用前7天的测点数据进行训练，并在第8天的数据上进行测试，比较预测得到的数据和真实数据

使用默认参数时的预测情况：

![image-20230522170001283](趋势预测报告（prophet模型）.assets/image-20230522170001283.png)

![image-20230522170014130](趋势预测报告（prophet模型）.assets/image-20230522170014130.png)

使用最优参数时的预测情况：

![image-20230522165842564](趋势预测报告（prophet模型）.assets/image-20230522165842564.png)

![image-20230522165904064](趋势预测报告（prophet模型）.assets/image-20230522165904064.png)

对比之后发现，通过调整模型从参数并不能有效提升该模型的预测性能，说明该情景下，出现prophet模型的预测效果较差的情况，参数并不是主要影响因素。

通过观察我们用于训练和测试的数据可以发现，我们使用的数据的时间跨度比较大，但是，由于数据量过大，数据的集中出现的范围内的数据量也很大，所以，模型在预测数据时，有可能会认为预测数据出现在数据的集中范围的正确率较高，所以，我们最后就得到了一条近乎为直线的趋势预测曲线。

所以，我们接下来尝试从用于训练和测试的数据本身入手，通过对数据进行一定程度的处理，来提升模型的性能。

以上的趋势预测用于训练和测试的数据量分别为133436和20254

我们尝试每10个数据取一个进行训练和测试

此时用于训练和测试的数据量分别为13344和2026

![image-20230522170638053](趋势预测报告（prophet模型）.assets/image-20230522170638053.png)

![image-20230522170648360](趋势预测报告（prophet模型）.assets/image-20230522170648360.png)

我们发现数据在变化趋势上有了明显的波动，但目前仍然无法评价该模型的预测效果，因为数据的变化趋势依旧不明显，所以，接下来，我们重点关注这部分的数据调整内容。

我们尝试每100个数据取一个进行训练和测试

此时用于训练和测试的数据量分别为1335和203

![image-20230522171030271](趋势预测报告（prophet模型）.assets/image-20230522171030271.png)



![image-20230522171042043](趋势预测报告（prophet模型）.assets/image-20230522171042043.png)

我们尝试每300个数据取一个进行训练和测试

此时用于训练和测试的数据量分别为445和68

![image-20230524094223065](趋势预测报告（prophet模型）.assets/image-20230524094223065.png)

![image-20230524094235439](趋势预测报告（prophet模型）.assets/image-20230524094235439.png)

我们尝试每500个数据取一个进行训练和测试

此时用于训练和测试的数据量分别为267和41

![image-20230524093738940](趋势预测报告（prophet模型）.assets/image-20230524093738940.png)

![image-20230524093754946](趋势预测报告（prophet模型）.assets/image-20230524093754946.png)

我们尝试每1000个数据取一个进行训练和测试

此时用于训练和测试的数据量分别为134和21

![image-20230522171902584](趋势预测报告（prophet模型）.assets/image-20230522171902584.png)

![image-20230522171917987](趋势预测报告（prophet模型）.assets/image-20230522171917987.png)

我们发现预测趋势的变化幅度进一步增大，这是增大数据间隔的必然结果。随着数据点之间的间距不断扩大，相邻数据点之间的影响逐渐减小，数据变化较大的数据点在数据集中的比重增大，在进行测试数据集的预测时，可以明显发现，预测数据的趋势变化和真实数据的变化情况出现了较为严重的偏差，现在这种情况下，该模型并不适合用来对后续的数据进行预测。

比较上述的几次测试结果，可以发现，当每300个数据进行取值时，得到的预测结果的总体变化趋势和真实情况是最符合的。所以，我们在使用统计学模型进行趋势预测时，可以考虑对参数进行一定程度的调整，来优化模型的性能。