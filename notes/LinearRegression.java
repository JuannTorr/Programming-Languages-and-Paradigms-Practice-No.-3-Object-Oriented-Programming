
public class LinearRegression {

    //Atributos Requeridos
    private double[] weights; 
    private double bias;

    //Atributos Adicionales para Escalado
    private double[] featureMeans;
    private double[] featureStdDevs;
    private int numFeatures;
    private double y_mean;
    private double y_stdDev;

    //Hiperparámetros del Modelo
    private double learningRate;
    private int epochs;

    public LinearRegression(int numFeatures, double learningRate, int epochs) {
        this.numFeatures = numFeatures;
        this.learningRate = learningRate;
        this.epochs = epochs;
        
        //Inicializamos los pesos aleatoriamente 
        this.weights = new double[numFeatures];
        for (int i = 0; i < numFeatures; i++) {
             this.weights[i] = 0.01 * Math.random(); 
        }
        this.bias = 0.0;
        
        //Inicializamos los arreglos para los parámetros de escalado
        this.featureMeans = new double[numFeatures];
        this.featureStdDevs = new double[numFeatures];
        this.y_mean = 0.0;
        this.y_stdDev = 1.0; 
    }
    public void fit(double[][] X_train, double[] y_train) {
        System.out.println("Entrenando modelo...");
        int numSamples = X_train.length;
  
        double[][] X_scaled = scaleFeatures(X_train);

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
            
        System.out.println("Iniciando Gradient Descent...");
        for (int epoch = 0; epoch < this.epochs; epoch++) {

            double[] y_pred_scaled = new double[numSamples];
            for (int i = 0; i < numSamples; i++) {
                y_pred_scaled[i] = predictInternal(X_scaled[i]);
            }
            
            double[] dw = new double[this.numFeatures];
            double db = 0.0; 

            for (int i = 0; i < numSamples; i++) {
                double error = y_pred_scaled[i] - y_scaled[i]; 
                db += error;
                for (int j = 0; j < this.numFeatures; j++) {
                    dw[j] += error * X_scaled[i][j];
                }
            }

            db /= numSamples;
            for (int j = 0; j < this.numFeatures; j++) {
                dw[j] /= numSamples;
            }

            this.bias -= this.learningRate * db;
            for (int j = 0; j < this.numFeatures; j++) {
                this.weights[j] -= this.learningRate * dw[j];
            }

            if (epoch % (epochs / 10) == 0 || epoch == this.epochs - 1) {
                System.out.println("Epoch " + epoch + " | Bias (escalado): " + this.bias);
            }
        }
        
        System.out.println("¡Entrenamiento completado!");
    }

    public double[] predict(double[][] X_test) {
        System.out.println("Prediciendo...");

        double[][] X_test_scaled = scaleTestFeatures(X_test); 

        double[] predictions_scaled = new double[X_test.length];
        for (int i = 0; i < X_test.length; i++) {
            predictions_scaled[i] = predictInternal(X_test_scaled[i]);
        }
        
        double[] predictions_unscaled = new double[X_test.length];
        for (int i = 0; i < X_test.length; i++) {
            predictions_unscaled[i] = (predictions_scaled[i] * this.y_stdDev) + this.y_mean;
        }
        
        return predictions_unscaled; 
    }

    public double score(double[] y_true, double[] y_pred) {
        System.out.println("Calculando score (Error Cuadrático Medio - MSE)...");

        if (y_true.length != y_pred.length) {
            System.out.println("Error: Los arreglos y_true y y_pred tienen tamaños diferentes.");
            return -1.0; 
        }

        int n = y_true.length;
        double sumSquaredError = 0.0;

        for (int i = 0; i < n; i++) {
            double error = y_true[i] - y_pred[i];
            sumSquaredError += Math.pow(error, 2);
        }

        return sumSquaredError / n;
    }
    
    public double[] getWeights() { 
        return weights; 
    }

    public double getBias() { 
        return bias; 
    }

    private double predictInternal(double[] x_scaled_row) {
        double prediction = this.bias;
        for (int j = 0; j < this.numFeatures; j++) {
            prediction += this.weights[j] * x_scaled_row[j];
        }
        return prediction;
    }

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

    private double[][] scaleTestFeatures(double[][] X_test) {
        int numRows = X_test.length;
        int numCols = this.numFeatures;
        double[][] X_scaled = new double[numRows][numCols];

        for (int j = 0; j < numCols; j++) {
            for (int i = 0; i < numRows; i++) {
                if (X_test[i].length > j) {
                    if (this.featureStdDevs[j] > 0) {
                        X_scaled[i][j] = (X_test[i][j] - this.featureMeans[j]) / this.featureStdDevs[j];
                    } else {
                        X_scaled[i][j] = 0;
                    }
                } else {
                    System.out.println("Error: Fila de X_test no tiene suficientes features.");
                    X_scaled[i][j] = 0;
                }
            }
        }
        return X_scaled;
    }

    private double[] getColumn(double[][] matrix, int colIndex) {
        double[] column = new double[matrix.length];
        for (int i = 0; i < matrix.length; i++) {
            column[i] = matrix[i][colIndex];
        }
        return column;
    }

    private double calculateMean(double[] data) {
        double sum = 0.0;
        for (double val : data) {
            sum += val;
        }
        return sum / data.length;
    }

    private double calculateStdDev(double[] data, double mean) {
        double sumSquaredDiff = 0.0;
        for (double val : data) {
            sumSquaredDiff += Math.pow(val - mean, 2);
        }
        double variance = sumSquaredDiff / data.length;

        return (variance > 0) ? Math.sqrt(variance) : 0;
    }
}
