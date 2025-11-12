public class LinearRegression {

    // --- Atributos Requeridos ---
    private double[] weights; 
    private double bias;

    // --- Atributos Adicionales para Escalado ---
    private double[] featureMeans;
    private double[] featureStdDevs;
    private int numFeatures;
    private double y_mean;
    private double y_stdDev;

    // --- Hiperparámetros del Modelo ---
    private double learningRate;
    private int epochs;

    //Constructor para inicializar el modelo.

    public LinearRegression(int numFeatures, double learningRate, int epochs) {
        this.numFeatures = numFeatures;
        this.learningRate = learningRate;
        this.epochs = epochs;
        
        // Inicializamos los pesos aleatoriamente (ayuda a la convergencia)
        this.weights = new double[numFeatures];
        for (int i = 0; i < numFeatures; i++) {
             this.weights[i] = 0.01 * Math.random(); 
        }
        this.bias = 0.0;
        
        // Inicializamos los arreglos para los parámetros de escalado
        this.featureMeans = new double[numFeatures];
        this.featureStdDevs = new double[numFeatures];
        this.y_mean = 0.0;
        this.y_stdDev = 1.0; 
    }

    //Escala internamente los datos X e y.
    public void fit(double[][] X_train, double[] y_train) {
        System.out.println("Entrenando modelo...");
        int numSamples = X_train.length;
        
        // 1. Escalar X (features)
        double[][] X_scaled = scaleFeatures(X_train);
        
        // 2. Escalar Y (target)
        this.y_mean = calculateMean(y_train);
        this.y_stdDev = calculateStdDev(y_train, this.y_mean);
        
        double[] y_scaled = new double[numSamples];
        for (int i = 0; i < numSamples; i++) {
            if (this.y_stdDev > 0) {
                y_scaled[i] = (y_train[i] - this.y_mean) / this.y_stdDev;
            } else {
                y_scaled[i] = 0;
            }
        }
        System.out.println("Datos de entrenamiento escalados (X e y).");

        // 3. Implementación de Gradient Descent 
        System.out.println("Iniciando Gradient Descent...");
        for (int epoch = 0; epoch < this.epochs; epoch++) {
            
            // a. Calcular predicciones (escaladas)
            double[] y_pred_scaled = new double[numSamples];
            for (int i = 0; i < numSamples; i++) {
                y_pred_scaled[i] = predictInternal(X_scaled[i]);
            }
            
            // b. Calcular gradientes
            double[] dw = new double[this.numFeatures];
            double db = 0.0; 

            for (int i = 0; i < numSamples; i++) {
                double error = y_pred_scaled[i] - y_scaled[i]; 
                db += error;
                for (int j = 0; j < this.numFeatures; j++) {
                    dw[j] += error * X_scaled[i][j];
                }
            }
            
            // Promediar gradientes
            db /= numSamples;
            for (int j = 0; j < this.numFeatures; j++) {
                dw[j] /= numSamples;
            }

            // c. Actualizar weights y bias
            this.bias -= this.learningRate * db;
            for (int j = 0; j < this.numFeatures; j++) {
                this.weights[j] -= this.learningRate * dw[j];
            }

            // Opcional: Imprimir progreso
            if (epoch % (epochs / 10) == 0 || epoch == this.epochs - 1) {
                System.out.println("Epoch " + epoch + " | Bias (escalado): " + this.bias);
            }
        }
        
        System.out.println("¡Entrenamiento completado!");
    }

    //Predice los valores de salida para un nuevo conjunto de datos X.
    public double[] predict(double[][] X_test) {
        System.out.println("Prediciendo...");
        
        // 1. Escalar los datos de prueba (X_test)
        double[][] X_test_scaled = scaleTestFeatures(X_test); 

        // 2. Calcular predicciones (aún escaladas)
        double[] predictions_scaled = new double[X_test.length];
        for (int i = 0; i < X_test.length; i++) {
            predictions_scaled[i] = predictInternal(X_test_scaled[i]);
        }
        
        // y_real = (y_scaled * y_stdDev) + y_mean
        double[] predictions_unscaled = new double[X_test.length];
        for (int i = 0; i < X_test.length; i++) {
            predictions_unscaled[i] = (predictions_scaled[i] * this.y_stdDev) + this.y_mean;
        }
        
        return predictions_unscaled; 
    }

    //Calcula el error de las predicciones (score) usando el error Cuadrático Medio (MSE).
   
    public double score(double[] y_true, double[] y_pred) {
        System.out.println("Calculando score (Error Cuadrático Medio - MSE)...");

        if (y_true.length != y_pred.length) {
            System.out.println("Error: Los arreglos y_true y y_pred tienen tamaños diferentes.");
            return -1.0; 
        }

        int n = y_true.length;
        double sumSquaredError = 0.0;

        // SUM( (y_true[i] - y_pred[i])^2 )
        for (int i = 0; i < n; i++) {
            double error = y_true[i] - y_pred[i];
            sumSquaredError += Math.pow(error, 2);
        }

        // (1/n) * SUM(...)
        return sumSquaredError / n;
    }
    
    //Métodos Getters (Requeridos por la práctica) 
    public double[] getWeights() { 
        return weights; 
    }

    //Retorna el sesgo (bias) del modelo.
     
    public double getBias() { 
        return bias; 
    }


    //MÉTODOS PRIVADOS AUXILIARES
    private double predictInternal(double[] x_scaled_row) {
        double prediction = this.bias;
        for (int j = 0; j < this.numFeatures; j++) {
            prediction += this.weights[j] * x_scaled_row[j];
        }
        return prediction;
    }

    //Escala la matriz de entrenamiento X (scaleFeatures) y Calcula y *guarda* la media y stdDev de cada columna.
     
    private double[][] scaleFeatures(double[][] X) {
        int numRows = X.length;
        int numCols = this.numFeatures;
        double[][] X_scaled = new double[numRows][numCols];

        for (int j = 0; j < numCols; j++) {
            double[] column = getColumn(X, j);
            this.featureMeans[j] = calculateMean(column);
            this.featureStdDevs[j] = calculateStdDev(column, this.featureMeans[j]);

            for (int i = 0; i < numRows; i++) {
                if (this.featureStdDevs[j] > 0) {
                    X_scaled[i][j] = (X[i][j] - this.featureMeans[j]) / this.featureStdDevs[j];
                } else {
                    X_scaled[i][j] = 0;
                }
            }
        }
        return X_scaled;
    }

    //Escala la matriz de prueba X_test (scaleTestFeatures).
  
    private double[][] scaleTestFeatures(double[][] X_test) {
        int numRows = X_test.length;
        int numCols = this.numFeatures;
        double[][] X_scaled = new double[numRows][numCols];

        for (int j = 0; j < numCols; j++) {
            for (int i = 0; i < numRows; i++) {
                // Asegurarse de que X_test[i] tenga suficientes columnas
                if (X_test[i].length > j) {
                    if (this.featureStdDevs[j] > 0) {
                        X_scaled[i][j] = (X_test[i][j] - this.featureMeans[j]) / this.featureStdDevs[j];
                    } else {
                        X_scaled[i][j] = 0;
                    }
                } else {
                    // Esto no debería pasar si los datos de X_test son correctos
                    System.out.println("Error: Fila de X_test no tiene suficientes features.");
                    X_scaled[i][j] = 0;
                }
            }
        }
        return X_scaled;
    }
    
    // Helper para extraer una sola columna de una matriz 2D.
    
    private double[] getColumn(double[][] matrix, int colIndex) {
        double[] column = new double[matrix.length];
        for (int i = 0; i < matrix.length; i++) {
            column[i] = matrix[i][colIndex];
        }
        return column;
    }

    //Calcula la media (promedio) de un arreglo.
     
    private double calculateMean(double[] data) {
        double sum = 0.0;
        for (double val : data) {
            sum += val;
        }
        return sum / data.length;
    }

    //Calcula la desviación estándar de un arreglo.
    private double calculateStdDev(double[] data, double mean) {
        double sumSquaredDiff = 0.0;
        for (double val : data) {
            sumSquaredDiff += Math.pow(val - mean, 2);
        }
        double variance = sumSquaredDiff / data.length;
        // Evitar raíz cuadrada de 0 o números muy pequeños
        return (variance > 0) ? Math.sqrt(variance) : 0;
    }
}
