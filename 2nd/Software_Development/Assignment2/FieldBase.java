import java.util.ArrayList;

public class FieldBase extends AFieldBase{

    private HomeBase homeBase;
    private String name;
    private ArrayList<Spy> spies; 
    private Cypher cypher;
    private String message;

    public FieldBase(String name){
        this.homeBase = HomeBase.getInstance();
        this.name = name;
        this.spies = new ArrayList<Spy>();
        homeBase.registerFieldBase(this);
    }

    public void goDark(){
        homeBase.unregisterFieldBase(this);
        homeBase = null;
    }

    public void goLight(){
        homeBase = HomeBase.getInstance();
    }

    public void registerSpy(Spy spy){
        if (spy.isAlive == true) {
            spy.setFieldBase(this);
            spies.add(spy);
            spy.updateCypher(this.cypher);
            return;
        }
        System.out.println("Cannot register an unalived Spy");
    }

    public void agentTerminated(Spy spy){
        spy.setFieldBase(null);
        this.spies.remove(spy);
    }

    public void updateCypher(Cypher cypher){
        this.cypher = cypher;
        for (Spy s : spies){
            s.updateCypher(cypher);
        }
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

    public void setName(String newName) {
        this.name = newName; 
    }

    public String getName(){
        return this.name;
    }

    public HomeBase getHomeBase(){
        return this.homeBase;
    }

}