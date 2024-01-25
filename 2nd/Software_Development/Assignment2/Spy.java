public class Spy extends ASpy{

    private FieldBase fieldBase;
    private String name;
    public Boolean isAlive;
    private Cypher cypher;
    private String message;

    public Spy(String name){
        this.name = name;
        this.isAlive = true;
    }

    public void unAlive(){
        this.isAlive = false;
        System.out.println(this.name + " has been terminated");
        fieldBase.agentTerminated(this);
    }

    public void setFieldBase(FieldBase fieldBase){
        if (this.fieldBase == null){
            this.fieldBase = fieldBase;
        }
    }
 
    public void updateCypher(Cypher cypher){
        this.cypher = cypher;
    }

    public void send(String m, Message reciever){
        super.send(m, this.cypher, reciever);
    }
    public void recieve(String m){
        this.message = super.decrypt(m, this.cypher);
    }

    public String getMessage(){
        return this.message;
    }

    public void setName(String newName){
        this.name = newName;
    }

    public String getName(){
        return this.name;
    }

    public FieldBase getFieldBase(){
        return this.fieldBase;
    }

}
