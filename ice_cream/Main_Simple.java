import java.util.Arrays;

/**
 * Clase principal para probar el caso de Regresión Simple
 * (Datos de Venta de Helados).
 * ¡Utiliza la MISMA clase LinearRegression.java!
 */
public class Main_Simple {

    public static void main(String[] args) {
        
        // --- 1. Datos de entrenamiento (Ice_cream_selling_data.csv) ---
        
        // Datos X (Temperature)
        double[][] X_train = {
            {-4.66},
            {-4.31},
            {-4.21},
            {-3.94},
            {-3.57},
            {0.0},
            {5.0},
            {10.0},
            {15.0},
            {20.0}
        };

        // Datos y (Sales)
        double[] y_train = {
            41.84,
            34.66,
            39.38,
            37.53,
            32.28,
            50.0,
            70.0,
            85.0,
            95.0,
            110.0
        };
        
        // --- 2. Definir Hiperparámetros ---
        
        int numFeatures = X_train[0].length; 
        
        // (CORRECCIÓN AQUÍ)
        // Reducimos la tasa de aprendizaje para evitar que el Bias "explote"
        double learningRate = 0.001;
        // Aumentamos las épocas para que tenga tiempo de aprender más lento
        int epochs = 5000;          
        
        
        System.out.println("Iniciando Práctica 3: Regresión Lineal SIMPLE");
        System.out.println("Features: " + numFeatures + " | Epochs: " + epochs + " | LR: " + learningRate);
        
        // --- 3. Crear y Entrenar ---
        LinearRegression mlr_simple = new LinearRegression(numFeatures, learningRate, epochs);
        
        mlr_simple.fit(X_train, y_train);
        
        // --- 4. Resultados del Modelo ---
        System.out.println("--- Resultados del Modelo (Simple) ---");
        System.out.println("Weight (Pendiente): " + Arrays.toString(mlr_simple.getWeights()));
        System.out.println("Bias (Intercepto): " + mlr_simple.getBias());

        // --- 5. Probar Predicción ---
        // ¿Cuánto venderemos si la temperatura es 25°C?
        double[][] X_test = {
            {25.0} 
        };
        
        // Valor real supuesto (para calcular el score)
        double[] y_test = { 120.0 };

        System.out.println("\n--- Probando Predicciones (Simple) ---");
        
        double[] y_hat = mlr_simple.predict(X_test); 

        System.out.println("Predicción para 25°C: " + y_hat[0] + " (Real=" + y_test[0] + ")");
        
        // --- 6. Calcular el Score ---
        double mse = mlr_simple.score(y_test, y_hat);
        System.out.println("Score (MSE) del modelo simple: " + mse);
    }
}
