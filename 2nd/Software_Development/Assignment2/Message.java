public abstract class Message {

    public abstract void recieve(String m);
    public abstract void updateCypher(Cypher cypher);
    public abstract String getMessage();
    public abstract void setName(String newName);
    public abstract String getName();

    public void send(String m, Cypher c, Message reciever){
        reciever.recieve(c.encrypt(m));
    }

    public String decrypt(String m, Cypher c){
        return c.decrypt(m);
    }
    
}