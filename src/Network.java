import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.BackPropagation;
import org.neuroph.util.TransferFunctionType;

import java.sql.Time;

public class Network {

    //Siec neuronowa wykorzystujaca bibloteke Neuroph w wersji 2.92
    public static void main(String[] args) {

        int testSuccessCount = 0;


        // Utworzenie danych uczących
        DataSet trainingData = new DataSet(35, 26);
        trainingData.setLabel("TrainingData");
        trainingData.addRow(new DataSetRow(new double[]{0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1}, new double[]{1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}));
        trainingData.addRow(new DataSetRow(new double[]{1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0}, new double[]{0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}));
        trainingData.addRow(new DataSetRow(new double[]{0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0}, new double[]{0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}));
        trainingData.addRow(new DataSetRow(new double[]{1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0}, new double[]{0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}));
        trainingData.addRow(new DataSetRow(new double[]{1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1}, new double[]{0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}));
        trainingData.addRow(new DataSetRow(new double[]{1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0}, new double[]{0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}));
        trainingData.addRow(new DataSetRow(new double[]{0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0}, new double[]{0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}));
        trainingData.addRow(new DataSetRow(new double[]{1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1}, new double[]{0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}));
        trainingData.addRow(new DataSetRow(new double[]{0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0}, new double[]{0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}));
        trainingData.addRow(new DataSetRow(new double[]{1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0}, new double[]{0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}));
        //10
        trainingData.addRow(new DataSetRow(new double[]{1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1}, new double[]{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}));
        trainingData.addRow(new DataSetRow(new double[]{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1}, new double[]{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}));
        trainingData.addRow(new DataSetRow(new double[]{1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1}, new double[]{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}));
        trainingData.addRow(new DataSetRow(new double[]{1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1}, new double[]{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}));
        trainingData.addRow(new DataSetRow(new double[]{0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0}, new double[]{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}));
        trainingData.addRow(new DataSetRow(new double[]{1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0}, new double[]{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}));
        trainingData.addRow(new DataSetRow(new double[]{0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1}, new double[]{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0}));
        trainingData.addRow(new DataSetRow(new double[]{1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1}, new double[]{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0}));
        trainingData.addRow(new DataSetRow(new double[]{0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0}, new double[]{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0}));
        trainingData.addRow(new DataSetRow(new double[]{1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0}, new double[]{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0}));
        //20
        trainingData.addRow(new DataSetRow(new double[]{1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0}, new double[]{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0}));
        trainingData.addRow(new DataSetRow(new double[]{1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0}, new double[]{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0}));
        trainingData.addRow(new DataSetRow(new double[]{1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0}, new double[]{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0}));
        trainingData.addRow(new DataSetRow(new double[]{1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1}, new double[]{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0}));
        trainingData.addRow(new DataSetRow(new double[]{1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0}, new double[]{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0}));
        trainingData.addRow(new DataSetRow(new double[]{1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1}, new double[]{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1}));


        //Wybór algorytmu oraz nadanie wartosci wspolczynnikowi uczenia i maksymalnemu bledu uczenia
        BackPropagation backPropagation = new BackPropagation();
        backPropagation.setMaxError(0.1);
        backPropagation.setLearningRate(0.1);

        //Stworzenie sieci składającej się z 3 warstw: 35 neuronow wejsc, 10 neuronow warstwie ukrytej, 26 neuronow wyjscia
        //Wybor funkcji aktywacji, tutaj to sigmoidalna
        MultiLayerPerceptron multiLayerPerceptron = new MultiLayerPerceptron(TransferFunctionType.SIGMOID, 35, 10, 26);
        multiLayerPerceptron.setLabel("Rozpoznawanie liter za pomoca sieci neuronowej");

        // Ustawienie algorytmu wstecznej propagacji jako algorytmu rozwiązywania sieci neuronowej

        multiLayerPerceptron.setLearningRule(backPropagation);

        // Uczenie sieci oraz zapisane do pliku "my.nnet"
        System.out.println("Uczenie: " + multiLayerPerceptron.getLabel() + ", dane uczace: " + trainingData.getLabel());
        System.out.println("Uzycie algorytmu propagacji wstecznej z ustawieniami parametrow: \n Maksymalny blad uczenia: " + backPropagation.getMaxError() + "\n wspolczynnik uczenia: " + backPropagation.getLearningRate());
        multiLayerPerceptron.learn(trainingData);
        System.out.println("Uczenie zakonczone:");
        // Wyswietlenie liczby iteracji oraz całkowitego błędu uczenia w sieci neuronowej
        System.out.println(" Liczba iteracji: " + backPropagation.getCurrentIteration());
        System.out.println(" Blad ogolny: " + backPropagation.getErrorFunction().getTotalError());

        multiLayerPerceptron.save("my.nnet");

        // wczytanie utworzonej sieci
        NeuralNetwork neuralNetwork = NeuralNetwork.createFromFile("my.nnet");

        // przeprowadzenie testu sieci i porównanie żądanego outputu z otrzymanym
        int letter = 1;
        for (DataSetRow dataSetRow : trainingData.getRows()) {
            System.out.println();
            System.out.println();
            System.out.println("Litera: " + letter);
            double[] desiredOutput = dataSetRow.getDesiredOutput();

            neuralNetwork.setInput(dataSetRow.getInput());
            neuralNetwork.calculate();
            double[] output = neuralNetwork.getOutput();

            System.out.println("Oczekiwany wynik:");
            for (int i = 0; i < 26; i++) {
                System.out.print(desiredOutput[i] + "  ");
            }
            System.out.println();
            System.out.println("Oczekiwana litera: " + checkLetter(desiredOutput));

            System.out.println("Wynik:");
            for (int i = 0; i < 26; i++) {
                System.out.print(output[i] + "  ");
            }
            System.out.println();
            System.out.println("Twoja Litera: " + checkLetter(output));
            if ((int) checkLetter(output) - 64 == letter)
                testSuccessCount++;
            letter++;
        }
        System.out.println("Liczba dobrze rozpoznanych liter = " + testSuccessCount);
    }

    public static char checkLetter(double[] letter) {
        char yourLetter = 'X';
        double point = letter[0];
        int index = 0;

        // ustalenie litery
        for (int i = 1; i < 26; i++) {
            if (point < letter[i]) {
                point = letter[i];
                index = i;
            }
        }
        switch (index) {
            case 0:
                yourLetter = 'A';
                break;
            case 1:
                yourLetter = 'B';
                break;
            case 2:
                yourLetter = 'C';
                break;
            case 3:
                yourLetter = 'D';
                break;
            case 4:
                yourLetter = 'E';
                break;
            case 5:
                yourLetter = 'F';
                break;
            case 6:
                yourLetter = 'G';
                break;
            case 7:
                yourLetter = 'H';
                break;
            case 8:
                yourLetter = 'I';
                break;
            case 9:
                yourLetter = 'J';
                break;
            case 10:
                yourLetter = 'K';
                break;
            case 11:
                yourLetter = 'L';
                break;
            case 12:
                yourLetter = 'M';
                break;
            case 13:
                yourLetter = 'N';
                break;
            case 14:
                yourLetter = 'O';
                break;
            case 15:
                yourLetter = 'P';
                break;
            case 16:
                yourLetter = 'Q';
                break;
            case 17:
                yourLetter = 'R';
                break;
            case 18:
                yourLetter = 'S';
                break;
            case 19:
                yourLetter = 'T';
                break;
            case 20:
                yourLetter = 'U';
                break;
            case 21:
                yourLetter = 'V';
                break;
            case 22:
                yourLetter = 'W';
                break;
            case 23:
                yourLetter = 'X';
                break;
            case 24:
                yourLetter = 'Y';
                break;
            case 25:
                yourLetter = 'Z';
                break;
        }
        return yourLetter;
    }
}