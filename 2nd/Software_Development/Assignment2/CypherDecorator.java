public abstract class CypherDecorator implements Cypher{

    protected Cypher tempCypher;

    public CypherDecorator(Cypher newCypher){
        this.tempCypher = newCypher;
    }

    public String encrypt(String m) {
        return tempCypher.encrypt(m);
    }

    public String decrypt(String m) {
        return tempCypher.decrypt(m);
    }
    
}