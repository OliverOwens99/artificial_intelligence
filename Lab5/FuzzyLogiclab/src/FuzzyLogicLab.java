import net.sourceforge.jFuzzyLogic.FIS;

public class FuzzyLogicLab {
    public static void main(String[] args) {
        FIS fis = FIS.load("./FCL/funding.fcl", true); //Load and parse the FCL
        fis.setVariable("funding", 60); //Apply a value to a variable
        fis.setVariable("staffing", 14);
        fis.evaluate(); //Execute the fuzzy inference engine
        System.out.println(fis.getVariable("risk").getValue()); //Output
        
        // Display charts of membership functions
        fis.chart();
    }
}