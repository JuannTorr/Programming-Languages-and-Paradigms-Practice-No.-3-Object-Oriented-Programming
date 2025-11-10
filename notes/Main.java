import java.util.Arrays; // Para imprimir los arreglos

/**
 * Clase principal para probar el framework de Regresión Lineal.
 * Sigue el pseudocódigo de la Práctica 3.
 */
public class Main {

    public static void main(String[] args) {
        
        // --- 1. Datos de entrenamiento (basados en student_exam_scores.csv) ---
        // X_train = [hours_studied, sleep_hours, attendance_percent, previous_scores]
        double[][] X_train = {
            {8.0, 8.8, 72.1, 45},
            {1.3, 8.6, 60.7, 55},
            {4.0, 8.2, 73.7, 86},
            {3.5, 4.8, 95.1, 66},
            {9.1, 6.4, 89.8, 71},
            {5.5, 7.5, 80.0, 70},
            {2.0, 5.0, 65.0, 50}
        };

        // y_train = [exam_score]
        double[] y_train = {30.2, 25.0, 35.8, 34.0, 40.3, 38.0, 28.0};
        
        // El número de features es 4
        int numFeatures = X_train[0].length;
        
        // --- 2. Definir Hiperparámetros ---
        double learningRate = 0.01; // Tasa de aprendizaje
        int epochs = 1000;          // Número de iteraciones
        
        System.out.println("Iniciando Práctica 3: Regresión Lineal Múltiple");
        System.out.println("Features: " + numFeatures + " | Epochs: " + epochs + " | LR: " + learningRate);
        
        // --- 3. Crear y Entrenar (Pseudocódigo: "mlr = new LinearRegression...") ---
        LinearRegression mlr = new LinearRegression(numFeatures, learningRate, epochs);
        
        // (Pseudocódigo: "mlr.fit(...)")
        mlr.fit(X_train, y_train);
        
        // --- 4. Resultados del Modelo (Pseudocódigo: "print("weights: ...") ---
        System.out.println("--- Resultados del Modelo ---");
        System.out.println("Weights (escalados): " + Arrays.toString(mlr.getWeights()));
        System.out.println("Bias (escalado): " + mlr.getBias());

        // --- 5. Probar Predicción y Score ---
        
        // (Pseudocódigo: "X_test [][] = ...")
     
        double[][] X_test = {
            {7.0, 8.0, 75.0, 50}, // Estudiante 1 (4 features)
            {2.5, 6.0, 70.0, 60}  // Estudiante 2 (4 features)
        };
        
        // Valores Reales para X_test (y_test)
        double[] y_test = { 35.0, 30.0 }; // Valores supuestos para calcular el score

        System.out.println("\n--- Probando Predicciones ---");
        
        // (Pseudocódigo: "y_hat[] = mlr.predict(...)")
        double[] y_hat = mlr.predict(X_test); 

        // (Pseudocódigo: "for (yi in y_hat) print...")
        for (int i = 0; i < y_hat.length; i++) {
            System.out.println("Estudiante " + (i+1) + ": Predicción=" + y_hat[i] + " (Real=" + y_test[i] + ")");
        }
        
        // --- 6. Calcular el Score (Pseudocódigo: "print("score: ...") ---
        double mse = mlr.score(y_test, y_hat);
        System.out.println("Score (MSE) del modelo en datos de prueba: " + mse);
    }
}
                
