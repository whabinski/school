public class Cypher2 extends CypherDecorator{

    public Cypher2(Cypher newCypher) {
        super(newCypher);
    }

    private String key = "Hello";

    @Override
    public String encrypt(String m) {
        return tempCypher.encrypt(m) + " Encrypted with: " + key;
    }

    @Override
    public String decrypt(String m) {
        return "(Decrypt2: " + tempCypher.decrypt(m) + " With key: " + key + ")";
    }



}
