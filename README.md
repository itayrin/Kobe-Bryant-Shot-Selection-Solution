# Kobe-Bryant-Shot-Selection-Solution
Solution for Kaggle Problem : Kobe Bryant  Shot selection problem.
Though the competition closed the submission, the rank can be validated using the solution.
Rank 150/1170 


In order to run without the hyper parameter optimization module:
First Tab:
```sh
$ python prediction.py
```

In order to run with the hyper parameter optimization module(longer simulation time):
First Tab:
```sh
$ python main.py
```

Report.pdf contains the design steps and inferences during the process of solving the problem.
Note: A detailed simulation is not explored for heavy hyper parameter optimal search.The rank
can be firther improved. In addition the number of cores is left floating, so upon running 
the XgBoost library, all the cores will be engaged. In order to limit the number of cores 
running, modify ** nthread ** in self.space = {} for hyper parameter optimization. Further details 
to configuration of XgBoost can be found [Xgboost Knobs](http://xgboost.readthedocs.io/en/latest//parameter.html)

