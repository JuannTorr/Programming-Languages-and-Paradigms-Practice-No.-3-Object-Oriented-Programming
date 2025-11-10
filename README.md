# Programming-Languages-and-Paradigms-Practice-No.-3-Object-Oriented-Programming

Practice #3

Object-Oriented Programming - Linear Regression

This project is an implementation of a Multiple Linear Regression framework from scratch in Java, as part of Practice No. 3 for the "Programming Languages and Paradigms" course.

Authors:
* Juan Esteban Torres Peña
* Mathias Velez Londoño
* Alejandro Restrepo Osorio

Links:
* YouTube Video: 
* GitHub Repository: https://github.com/JuannTorr/Programming-Languages-and-Paradigms-Practice-No.-3-Object-Oriented-Programming
---

## 1. Code Explanation

The objective was to design and implement an object-oriented class to compute multiple linear regression models without using external statistical or matrix-processing libraries. The solution is centered on the `LinearRegression.java` class.

### LinearRegression.java

This main class encapsulates all the model's logic.

#### Key Attributes
* `private double[] weights;`: An array to store the weights (slopes) for each input feature.
* `private double bias;`: The bias term (or intercept).
* `private int numFeatures;`: Stores the number of input variables (columns).
* `private double[] featureMeans;`: Array to store the mean of each feature (for scaling).
* `private double[] featureStdDevs;`: Array to store the standard deviation of each feature.
* `private double y_mean;` and `private double y_stdDev;`: Store the mean and standard deviation of the target (y), crucial for un-scaling predictions.

#### Required Methods

`public void fit(double[][] X, double[] y)`
This is the model's training engine.
1.  Data Scaling: It first scales the input data `X` and target data `y` using Z-Score Normalization (`(value - mean) / stdDev`). This fulfills the `data_scaling()` requirement. The means and standard deviations are saved as class attributes.
2.  Gradient Descent: It runs an iterative optimization loop (for a set number of `epochs`) to find the optimal `weights` and `bias`.
3.  In each epoch, it calculates the current predictions, computes the error, and then calculates the gradients (`dw` and `db`).
4.  It updates the `weights` and `bias` in the opposite direction of the gradient, moderated by the `learningRate`.

`public double[] predict(double[][] X_test)`
This method takes new, unseen data and generates predictions.
1.  Scale `X_test`: It scales the input `X_test` data using the `featureMeans` and `featureStdDevs` *saved* during the `fit()` process.
2.  Calculate Prediction: It applies the linear formula `y_hat = (X_scaled * weights) + bias`.
3.  Un-scale Result: It converts the scaled prediction back into the original data's range using the saved `y_mean` and `y_stdDev`.

`public double score(double[] y_true, double[] y_pred)`
This method calculates the model's error. We implemented Mean Squared Error (MSE). It compares the true values (`y_true`) against the model's predictions (`y_pred`) and returns the average of the squared errors. A lower score is better.

---

## 2. Test Results

Two separate `Main` files were created to test the class in both required scenarios, demonstrating its functionality for both simple and multiple regression.

### Test 1: Multiple Regression (student_exam_scores.csv)

`Main.java` was used to test a 4-feature model to predict student exam scores.

Console Output:
```
Iniciando Práctica 3: Regresión Lineal Múltiple
Features: 4 | Epochs: 1000 | LR: 0.01
...
¡Entrenamiento completado!
--- Resultados del Modelo ---
Weights (escalados): [0.35..., -0.12..., 0.08..., 0.46...]
Bias (escalado): -0.01...
--- Probando Predicciones ---
Prediciendo...
Estudiante 1: Predicción=31.61... (Real=35.0)
Estudiante 2: Predicción=30.04... (Real=30.0)
Calculando score (Error Cuadrático Medio - MSE)...
Score (MSE) del modelo en datos de prueba: 5.72...
```

### Test 2: Simple Regression (ice_cream_selling_data.csv)

`Main_Simple.java` was used to test a 1-feature model to predict ice cream sales based on temperature. The *same* `LinearRegression.java` class was reused without modification.

Console Output:
```
Iniciando Práctica 3: Regresión Lineal SIMPLE
Features: 1 | Epochs: 5000 | LR: 0.001
...
Epoch 4999 | Bias (escalado): -5.55...E-17
¡Entrenamiento completado!
--- Resultados del Modelo (Simple) ---
Weight (Pendiente): [0.9877...]
Bias (Intercepto): -5.55...E-17
--- Probando Predicciones (Simple) ---
Prediciendo...
Predicción para 25°C: 118.45... (Real=120.0)
Calculando score (Error Cuadrático Medio - MSE)...
Score (MSE) del modelo simple: 2.38...
```

---

## 3. Troubles and Solutions

This section details the main issues encountered during development, as required by the practice report.

| Problem | Description | Solution |
| :--- | :--- | :--- |
| Syntax: `incompatible types` | Java threw this error when initializing the `X_train` and `y_train` arrays in `Main_Simple.java`. | Discovered that the `{...}` array shortcut syntax in Java is only valid if used on the *same line* as the variable declaration. |
| Error: `ArrayIndexOutOfBoundsException` | The program failed when calling `mlr.predict()`. The error trace pointed to `scaleTestFeatures`. | The `X_test` data in `Main.java` was incorrectly defined (e.g., `{7.0}` instead of `{7.0, 8.0, 75.0, 50}`). The test data must have the same number of features (4) as the training data. |
| Logic: Exploding Bias | In the simple regression test, the scaled `bias` grew to a large negative number (-5) instead of converging near zero. | The `learningRate` (0.01) was too high for this dataset, causing the model to overshoot. Solution was to lower the `learningRate` to `0.001` and increase `epochs` to `5000` to stabilize the training. |
| Logic: Incorrect Predictions | Initial predictions were 10x smaller than expected (e.g., `3.1` instead of `31.0`). | We were not scaling the target variable (`y`). The fix was to scale `y` in `fit()` and un-scale the final prediction in `predict()`. |

---

## 4. Conclusions

At least three conclusions are required for the report.

1.  Object-Oriented design enabled reusability. The project's main success was using the *same* `LinearRegression.java` class to handle both multiple (4-feature) and simple (1-feature) regression without modification. The logic was successfully encapsulated.
2.  Data pre-processing is critical for convergence. The model was non-functional without first scaling (normalizing) both `X` and `y` data. Z-Score scaling was essential for Gradient Descent to converge correctly.
3.  Hyperparameters are problem-specific. The `learningRate` that worked for the multiple regression test (0.01) caused the simple regression model to fail. This proved that hyperparameters must be tuned for each specific dataset to achieve stability and accuracy.
