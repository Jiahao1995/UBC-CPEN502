package Sarb;

import java.io.File;
import java.io.IOException;

public interface CommonInterface {

    public double outputFor(double[] X);

    public double train(double[] X, double argValue);

    public void save(File argFile);

    public void load(String argFileName) throws IOException;

}
