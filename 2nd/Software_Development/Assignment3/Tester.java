public class Tester {

    //Set
    private Mystery1 m;
    private final String breakLine = "-------------------------------------------------------------------------------------------";
    //Construct
    public Tester() {

        header(true);
        header(true);
        header(true);

        System.out.println();
        System.out.println("\t\t\t2ME3 Tester Class");
        System.out.println("\t\t\tPlease run this.test() in your main function."); //just to get rid of yellow squiggle lol, i could just run it here
        System.out.println();

        header(true);
        header(true);
        header(true);
    }

    public static void main(String args[]){
        Tester t = new Tester();
        t.test();
    }

    public void test() {
        
        //
        // Adding Nodes + Edges Correct Behaviour
        //

        header();
        f1("a", "a");
        f2("a", 0);

        header();
        f1("a", "b");
        f2("a", 0);
        f2("b", 1);


        header();
        f1("a", "b");
        f1("b", "a");
        f2("a", 1);
        f2("b", 1);


        header();
        f1("a", "b");
        f1("a", "b");
        f1("a", "b");
        f1("a", "b");
        f2("a", 0);
        f2("b", 1);


        header();
        f1("a", "b");
        f1("b", "c");
        f1("c", "d");
        f1("d", "a");
        f2("a", 1);
        f2("b", 1);
        f2("c", 1);
        f2("d", 1);


        header();
        f2("unadded", 0);

        //
        // f3 correct behaviour
        //

        header();
        f1("a", "b");
        f1("b", "c");
        f1("c", "d");
        f1("d", "a");
        f3("a", "d", true); //Good
        f3("a", "b", false); //no back path
        f3("a", "c", false); //no back path
        f3("a", "u", false); //no connection at all
        f3("a", "a", false); //no back path/ no connection
        f3("u", "u", false); //no back path/ no connection


        header();
        f1("a", "b");
        f3("a", "b", false); //no back path
        

        header();
        f1("a", "b");
        f1("b", "a");
        f3("a", "b", true);

        header();
        f1("a", "b");
        f1("b", "d");
        f1("d", "c");
        f1("c", "d");
        f1("c", "a");

        f3("a", "c", true);
        f3("b", "a", true);
        f3("d", "c", true);
        f3("c", "d", true);
        f3("b", "a", true);

        f3("c", "a", false);
        f3("c", "b", false);
        f3("b", "c", false);
        f3("a", "b", false);
          
        header();
        f1("i", "a");
        f1("a", "i");
        f1("d", "a");
        f1("a", "b");
        f1("b", "e");
        f1("b", "f");
        f1("d", "c");
        f1("f", "c");
        f1("f", "g");
        f1("g", "e");
        f1("g", "c");
        f1("c", "g");
        f1("g", "d");
        f1("g", "g");

        String[] nodes = { "a", "b", "c", "d", "e", "f", "g", "i", "u" };
        String[] passingEdges = { "ai", "ad", "ba", "cd", "cf", "cg", "dg", "fb", "gc", "gf", "ia",};

        for (String s1 : nodes) {
            for (String s2 : nodes) {
                
                //Check If Should Pass
                boolean shouldPass = false;
                String edgeName = s1 + s2;
                for (String pass : passingEdges) {
                    if (pass.equals(edgeName)) {
                        shouldPass = true;
                        break;
                    }
                }

                //Perform Test
                f3(s1, s2, shouldPass);
            }    
        }
        nodes = null;
        passingEdges = null;

        //
        //

        header();
        f1("i", "a");
        f1("a", "i");
        f1("d", "a");
        f1("a", "b");
        f1("b", "e");
        f1("b", "f");
        f1("d", "c");
        f1("f", "c");
        f1("f", "g");
        f1("g", "e");
        f1("g", "c");
        f1("c", "g");
        f1("g", "d");
        f1("g", "g");

        f1("c", "z");
        f1("z", "e");
        f1("e", "z");
        f1("d", "i");

        String[] nodesII = { "a", "b", "c", "d", "e", "f", "g", "i", "u", "z",};
        String[] passingEdgesII = { "ai", "ad", "ba", "cd", "cf", "cg", "dg", "fb", "gc", "gf", "ia", "ez", "ze", "id"};

        for (String s1 : nodesII) {
            for (String s2 : nodesII) {
                
                //Check If Should Pass
                boolean shouldPass = false;
                String edgeName = s1 + s2;
                for (String pass : passingEdgesII) {
                    if (pass.equals(edgeName)) {
                        shouldPass = true;
                        break;
                    }
                }

                //Perform Test
                f3(s1, s2, shouldPass);
            }    
        }
        nodesII = null;
        passingEdgesII = null;

        header();
        System.out.println("Testing Complete. If you see this in terminal, you passed all my test cases.");
        header();
    }

    //
    // FUNCTIONS
    //

    //Reset
    private void reset() {
        this.m = new Mystery1();
    }
    //Write
    private void header(boolean flag) {
        System.out.println(breakLine);
    }
    private void header() {
        header(true);
        reset();
    }

    //
    //
    //
    private void f1(String s1, String s2) {

        //Addition
        this.m.f1(s1, s2);
        //Write
        System.out.println("f1: Added X(\"" + s1 + "\", \"" + s2 + "\") to m.");

    }

    private void f2(String s, int expected) {
        int result = this.m.f2(s);

        String head = "f2(\"" + s + "\"):";

        //Check
        if (result == expected) {
            System.out.println(head + " PASS (" + result + " == " + expected + ")");
        } else {
            throw new AssertionError(head + " FAIL \t\t" + "(Result= " + result + ". Expected= " + expected + ")");
        }
    }

    private void f3(String s1, String s2, boolean expected) {

        //Get Result
        boolean result = m.f3(s1, s2);
        String header = "f3(\"" + s1 + "\", \"" + s2 + "\"):";

        String resultStatus = (result == expected) ? "PASS" : "FAIL";
        String resultString = resultStatus + ": Result=" + result + ". Expected=" + expected;

        if (result != expected) {
            throw new AssertionError(header + "\t" + resultString);
        }

        System.out.println(header + " " + resultStatus + " (" + result +"=="+ result +")");

    }

    //
    public String toString() {
        return "Current M1 State:\n" + this.m.toString();
    }
}