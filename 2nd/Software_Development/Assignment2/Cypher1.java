public class Cypher1 extends CypherDecorator {

    public Cypher1(Cypher newCypher) {
        super(newCypher);
    }

    private int key = 5;

    @Override
    public String encrypt(String m) {
        return tempCypher.encrypt(m) + " Encrypted with: " + key;
    }

    @Override
    public String decrypt(String m) {
        return "(Decrypt2: " + tempCypher.decrypt(m) + " With key: " + key + ")";
    }



}