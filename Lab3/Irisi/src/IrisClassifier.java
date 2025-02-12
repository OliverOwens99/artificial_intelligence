import java.io.File;

import org.encog.ml.data.MLData;
import org.encog.ml.data.versatile.NormalizationHelper;
import org.encog.ml.data.versatile.VersatileMLDataSet;
import org.encog.ml.data.versatile.columns.ColumnType;
import org.encog.util.csv.CSVFormat;
import org.encog.util.csv.ReadCSV;
import org.encog.util.simple.EncogUtility;
import org.encog.ml.data.versatile.columns.ColumnDefinition;
import org.encog.ml.data.versatile.sources.VersatileDataSource;
import org.encog.ml.factory.MLMethodFactory;
import org.encog.ml.model.EncogModel;
import org.encog.ml.data.versatile.sources.CSVDataSource;
import org.encog.Encog;
import org.encog.ml.*;

public class IrisClassifier {
    public void go(String file) {
    VersatileDataSource source = new CSVDataSource(new File(file), false, CSVFormat.DECIMAL_POINT);
    VersatileMLDataSet data = new VersatileMLDataSet(source);
    data.defineSourceColumn("sepal-length", 0, ColumnType.continuous);
    data.defineSourceColumn("sepal-width", 1, ColumnType.continuous);
    data.defineSourceColumn("petal-length", 2, ColumnType.continuous);
    data.defineSourceColumn("petal-width", 3, ColumnType.continuous);
    ColumnDefinition outputColumn = data.defineSourceColumn("species", 4, ColumnType.nominal);
    
    data.analyze();
    data.defineSingleOutputOthersInput(outputColumn);

    EncogModel model = new EncogModel(data);
    model.selectMethod(data, MLMethodFactory.TYPE_FEEDFORWARD);
    data.normalize(); 
    
    model.holdBackValidation(0.3, true, 1001);
    model.selectTrainingType(data);
    MLRegression bestMethod = (MLRegression) model.crossvalidate(5, true);
    System.out.println( "Training error: " +
    EncogUtility.calculateRegressionError(bestMethod, model.getTrainingDataset()));
    System.out.println( "Validation error: " +
    EncogUtility.calculateRegressionError(bestMethod, model.getValidationDataset()));

    NormalizationHelper helper = data.getNormHelper();
    ReadCSV csv = new ReadCSV(new File(file), false, CSVFormat.DECIMAL_POINT);
    String[] line = new String[4];
    MLData input = helper.allocateInputVector();
    while (csv.next()) {
    line[0] = csv.get(0);
    line[1] = csv.get(1);
    line[2] = csv.get(2);
    line[3] = csv.get(3);
    String expected = csv.get(4); 
    helper.normalizeInputVector(line, input.getData(), false);
    MLData output = bestMethod.compute(input); 
    
    String actual = helper.denormalizeOutputVectorToString(output)[0]; 
    System.out.println("Expected: " + actual + " Actual: " + expected);
    }

    Encog.getInstance().shutdown(); 

    
    }
   
    public static void main(String[] args) {
        new IrisClassifier().go("./iris.txt");

    }
   }
