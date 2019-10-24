package Assignment2;

import Sarb.LUTInterface;

import java.io.File;
import java.io.IOException;

public class LUT implements LUTInterface {

    public LUT(int argNumInputs,
               int[] argVariableFloor,
               int[] argVariableCeiling) {

    }

    public void initialiseLUT() {}

    public int indexFor(double[] X) {}

    public double outputFor(double[] X) {}

    public double train(double[] X, double argValue) {}

    public void save(File argFile) {}

    public void load(String argFileName) throws IOException {}

}
