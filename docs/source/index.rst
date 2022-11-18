Welcome to Scope Example's documentation!
=========================================

Here you'll find several Scope Examples that you can use to learn how to use Scope algorithm with autodiff. 

.. toctree::
   :maxdepth: 2
   
Linear Model (variable selection)
------------------------------------------------------------------------------------
.. code-block:: python
   :caption: Sparse Linear Model
   :linenos:
   :emphasize-lines: 3,5

   from abess import ConvexSparseSolver, make_glm_data
   import numpy as np
   import scope_model
   ## setting
   np.random.seed(3)
   n = 30
   p = 5
   k = 3
   family = "gaussian"
   ## generate data
   data = make_glm_data(n=n, p=p, k=k, family=family)
   ## define model
   model = ConvexSparseSolver(
      model_size=p, # number of features
      support_size=k, # number of selected features
      aux_para_size=1, # number of auxiliary parameters which is intercept in this case
      sample_size=n, # number of samples, not important if support_size is given
   )
   ## set data
   model.set_data(scope_model.CustomData(data.x, data.y))
   ## set loss function
   model.set_model_autodiff(scope_model.linear_model)
   ## start to solve
   model.fit()
   ## print results
   print("Estimated coefficients: ", model.coef_)
   print("True coefficients: ", data.coef_)


Logistic Model with Cross Validation
------------------------------------------------------------------------------------
Here is an example of logistic regression model, and the number of selected features will be choosen by cross validation.
For this, users have to offer a callback function to tell the algorithm how to split data.

.. code-block:: python
   :caption: Sparse Logistic Model
   :linenos:
   :emphasize-lines: 3,5

   from abess import ConvexSparseSolver, make_glm_data
   import numpy as np
   import scope_model
   ## setting
   np.random.seed(3)
   n = 100
   p = 5
   k = 3
   family = "binomial"
   ## generate data
   data = make_glm_data(n=n, p=p, k=k, family=family)
   ## define model
   model = ConvexSparseSolver(
      model_size=p, # number of features
      sample_size=n, # number of samples, neccessary if cv > 1
      cv = 5, # number of folds in cross validation
   )
   ## set data
   model.set_data(scope_model.CustomData(data.x, data.y))
   ## set loss function
   model.set_model_autodiff(scope_model.logistic_model)
   ## set split and deleter callback function
   model.set_split_method(scope_model.split_sample, scope_model.deleter)
   ## start to solve
   model.fit()
   ## print results
   print("Estimated coefficients: ", model.coef_)
   print("True coefficients: ", data.coef_)


MultiLinear Model (group variable selection)
------------------------------------------------------------------------------------
Here is an example of MultiLinear regression model, which each feature corresponds to a colmun of parameters
For this, users have to offer a group information.

.. code-block:: python
   :caption: MultiLinear Model
   :linenos:
   :emphasize-lines: 3,5

   from abess import ConvexSparseSolver, make_multivariate_glm_data
   import numpy as np
   import scope_model
   ## setting
   np.random.seed(3)
   n = 100 
   p = 5
   k = 3
   M = 3
   family = "multigaussian"
   ## generate data
   data = make_multivariate_glm_data(n=n, p=p, k=k, M=M, family=family)
   ## define model
   model = ConvexSparseSolver(
      model_size=p * M, # there are M groups, each group has p features
      support_size=k, # number of selected groups of features
      group=[i for i in range(p) for j in range(M)] # group information
   )
   ## set data
   model.set_data(scope_model.CustomData(data.x, data.y))
   ## set loss function
   model.set_model_autodiff(scope_model.multi_linear_model)
   ## start to solve
   model.fit()
   ## print results
   print("Estimated coefficients: ", model.coef_)
   print("True coefficients: ", data.coef_)
