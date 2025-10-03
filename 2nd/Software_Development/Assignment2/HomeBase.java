import java.util.ArrayList;

public class HomeBase extends AHomebase{

    private static HomeBase instance = new HomeBase();
    private String name;
    private ArrayList<FieldBase> fieldBases;
    private Cypher cypher;
    private String message;
    
    private HomeBase(){
        this.name = "HYDRA";
        this.fieldBases = new ArrayList<FieldBase>();
        this.cypher = new DefaultCypher();
    }

    public static HomeBase getInstance(){
        return instance;
    }

    public void registerFieldBase(FieldBase fieldbase){
        fieldBases.add(fieldbase);
        fieldbase.updateCypher(this.cypher);
    }

    public void unregisterFieldBase(FieldBase fieldbase){
        fieldBases.remove(fieldbase);
    }

    public void setCypher(Cypher cypher){
        this.cypher = cypher;
        updateCypher(cypher);
    }

    public void updateCypher(Cypher cypher) {
        for (FieldBase fb : fieldBases){
            fb.updateCypher(cypher);
        }
    }

    public void setName(String newName){
        instance.name = newName;
    }

    public String getName(){
        return instance.name;
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

}
