/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package mlclassifierapp;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import javax.swing.JOptionPane;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;

/**
 *
 * @author Tivas-Tech
 */
public class ClassifierApp {
    // Define a Weka Classifier Object
    private Classifier mClassifier = null;
    public void classify(){
        //AssetManager assetManager = getAssets();
        try 
        {
            InputStream modelInputStream=this.getClass().getResourceAsStream("afamTrainedModels.model");
            mClassifier = (Classifier)
                    weka.core.SerializationHelper.read(modelInputStream);
            
        } 
        catch (Exception e) {
            // Handle Weka model still failed to load
            e.printStackTrace();
            JOptionPane.showMessageDialog(null, "afamTrainedModels.model NOT found");
        }
        // Set up the Attributes. These need to match the Weka Iris.arff dataset
        final Attribute attributeSepalLength = new Attribute("sepallength");
        final Attribute attributeSepalWidth = new Attribute("sepalwidth");
        final Attribute attributePetalLength = new Attribute("petallength");
        final Attribute attributePetalWidth = new Attribute("petalwidth");
        // Set up the Classes. For this dataset se have 3. These also needs to match the Iris.arff dataset
        final List<String> classes = new ArrayList<String>() {{
            add("Iris-setosa");
            add("Iris-versicolor");
            add("Iris-virginica");}
        };
        // Set up an ArrayList to hold the values for the untested samples we wish to classify.
        // In this example, we will just have a single untested sample.
            ArrayList<Attribute> attributeList = new ArrayList<Attribute>(2) {{
            add(attributeSepalLength);
            add(attributeSepalWidth);
            add(attributePetalLength);
            add(attributePetalWidth);

            Attribute attributeClass = new Attribute("@@class@@", classes);
            add(attributeClass);}
        };
        // Create a new Instance which we will classify
        Instances dataUnpredicted = new Instances("TestInstances",attributeList, 1);
        // The last feature is the target variable
        dataUnpredicted.setClassIndex(dataUnpredicted.numAttributes() - 1);
        DenseInstance newInstance = new DenseInstance(dataUnpredicted.numAttributes()) {{
            setValue(attributeSepalLength, 4.95);
            setValue(attributeSepalWidth, 3.50);
            setValue(attributePetalLength, 2.00);
            setValue(attributePetalWidth, 0.45);}
        };
        // Define the dataset
        newInstance.setDataset(dataUnpredicted);
// Predict the new sample
        try {
            double result = mClassifier.classifyInstance(newInstance);
            String prediction = classes.get(new Double(result).intValue());
            JOptionPane.showMessageDialog(null, prediction);
        } catch (Exception e) {
            // Oops, need to handle the Weka prediction exception
            e.printStackTrace();
        }
    
    }
}
