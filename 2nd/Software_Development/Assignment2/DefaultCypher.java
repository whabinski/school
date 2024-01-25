public class DefaultCypher implements Cypher{

    private int key = 10;

    @Override
    public String encrypt(String m) {
        return "(Encrypt1: " + m + " With key: " + key + ")";
    }

    @Override
    public String decrypt(String m) {
        return "(Decrypt1: " + m + " With key: " + key + ")";
    }

    
}