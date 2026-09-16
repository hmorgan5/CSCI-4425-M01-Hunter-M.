# California Housing Dataset Exploration

## How to Run

Install required packages:

```bash
pip install pandas scikit-learn torch matplotlib tabulate

Run Program: python hw2.py

## Dataset Description

```
.. _california_housing_dataset:

California Housing dataset
--------------------------

**Data Set Characteristics:**

:Number of Instances: 20640

:Number of Attributes: 8 numeric, predictive attributes and the target

:Attribute Information:
    - MedInc        median income in block group
    - HouseAge      median house age in block group
    - AveRooms      average number of rooms per household
    - AveBedrms     average number of bedrooms per household
    - Population    block group population
    - AveOccup      average number of household members
    - Latitude      block group latitude
    - Longitude     block group longitude

:Missing Attribute Values: None

This dataset was obtained from the StatLib:
https://lib.stat.cmu.edu/datasets/houses.zip

The target variable is the median house value for California districts,
expressed in hundreds of thousands of dollars ($100,000).

This dataset was derived from the 1990 U.S. census, using one row per census
block group. A block group is the smallest geographical unit for which the U.S.
Census Bureau publishes sample data (a block group typically has a population
of 600 to 3,000 people).

A household is a group of people residing within a home. Since the average
number of rooms and bedrooms in this dataset are provided per household, these
columns may take surprisingly large values for block groups with few households
and many empty houses, such as vacation resorts.

It can be downloaded/loaded using the
:func:`sklearn.datasets.fetch_california_housing` function.

.. rubric:: References

- Pace, R. Kelley and Ronald Barry, Sparse Spatial Autoregressions,
  Statistics and Probability Letters, 33:291-297, 1997.

```

## First 5 Rows

|    |   MedInc |   HouseAge |   AveRooms |   AveBedrms |   Population |   AveOccup |   Latitude |   Longitude |   MedHouseVal |
|---:|---------:|-----------:|-----------:|------------:|-------------:|-----------:|-----------:|------------:|--------------:|
|  0 |   8.3252 |         41 |    6.98413 |     1.02381 |          322 |    2.55556 |      37.88 |     -122.23 |         4.526 |
|  1 |   8.3014 |         21 |    6.23814 |     0.97188 |         2401 |    2.10984 |      37.86 |     -122.22 |         3.585 |
|  2 |   7.2574 |         52 |    8.28814 |     1.07345 |          496 |    2.80226 |      37.85 |     -122.24 |         3.521 |
|  3 |   5.6431 |         52 |    5.81735 |     1.07306 |          558 |    2.54795 |      37.85 |     -122.25 |         3.413 |
|  4 |   3.8462 |         52 |    6.28185 |     1.08108 |          565 |    2.18147 |      37.85 |     -122.25 |         3.422 |

## Summary Statistics

|       |      MedInc |   HouseAge |     AveRooms |    AveBedrms |   Population |     AveOccup |    Latitude |   Longitude |   MedHouseVal |
|:------|------------:|-----------:|-------------:|-------------:|-------------:|-------------:|------------:|------------:|--------------:|
| count | 20640       | 20640      | 20640        | 20640        |     20640    | 20640        | 20640       | 20640       |   20640       |
| mean  |     3.87067 |    28.6395 |     5.429    |     1.09668  |      1425.48 |     3.07066  |    35.6319  |  -119.57    |       2.06856 |
| std   |     1.89982 |    12.5856 |     2.47417  |     0.473911 |      1132.46 |    10.386    |     2.13595 |     2.00353 |       1.15396 |
| min   |     0.4999  |     1      |     0.846154 |     0.333333 |         3    |     0.692308 |    32.54    |  -124.35    |       0.14999 |
| 25%   |     2.5634  |    18      |     4.44072  |     1.00608  |       787    |     2.42974  |    33.93    |  -121.8     |       1.196   |
| 50%   |     3.5348  |    29      |     5.22913  |     1.04878  |      1166    |     2.81812  |    34.26    |  -118.49    |       1.797   |
| 75%   |     4.74325 |    37      |     6.05238  |     1.09953  |      1725    |     3.28226  |    37.71    |  -118.01    |       2.64725 |
| max   |    15.0001  |    52      |   141.909    |    34.0667   |     35682    |  1243.33     |    41.95    |  -114.31    |       5.00001 |

# Model Evaluation

| Model | MSE | RMSE |
|-------|------|------|
| Linear Regression | 0.5290 | 0.7273 |
| Neural Network | 0.4434 | 0.6659 |

## Analysis

The Neural Network performed better because it achieved lower MSE and RMSE values on the test set. Lower values indicate that the model's predictions were closer to the actual house values.

## Neural Network Loss Curve

![Losslot.png

### Loss Analysis

The training loss decreased as the number of epochs increased, indicating that the neural network was learning from the training data. The largest reduction in loss occurred during the early epochs, followed by smaller improvements as training continued. This behavior suggests that the model was converging toward a stable solution and improving its predictive performance over time.

