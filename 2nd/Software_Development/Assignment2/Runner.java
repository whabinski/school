public class Runner {
    
    public static void main (String[] args){

        HomeBase home = HomeBase.getInstance();

        FieldBase one = new FieldBase("one");
        FieldBase two = new FieldBase("two");
        FieldBase three = new FieldBase("three");
        FieldBase four = new FieldBase("four");


        Spy agentA = new Spy("Agent A");
        one.registerSpy(agentA);
        Spy agentB = new Spy("Agent B");
        one.registerSpy(agentB);
        Spy agentC = new Spy("Agent C");
        two.registerSpy(agentC);
        Spy agentD = new Spy("Agent D");
        three.registerSpy(agentD);
        Spy agentE = new Spy("Agent E");
        four.registerSpy(agentE);


        System.out.println(agentA.getName());
        System.out.println(agentA.getFieldBase().getName());
        System.out.println(agentA.getFieldBase().getHomeBase().getName());

        System.out.println("\n");

        System.out.println(agentB.getName());
        System.out.println(agentB.getFieldBase().getName());
        System.out.println(agentB.getFieldBase().getHomeBase().getName());
        
        System.out.println("\nChange Name Here!\n");
        home.setName("SHIELD");
        
        System.out.println(agentC.getName());
        System.out.println(agentC.getFieldBase().getName());
        System.out.println(agentC.getFieldBase().getHomeBase().getName());

        System.out.println("\n");

        System.out.println(agentD.getName());
        System.out.println(agentD.getFieldBase().getName());
        System.out.println(agentD.getFieldBase().getHomeBase().getName());

        System.out.println("\n");
        agentD.unAlive();
        System.out.println("\n");

        System.out.println(agentD.getName());
        try{
            System.out.println(agentD.getFieldBase().getName());
            System.out.println(agentD.getFieldBase().getHomeBase().getName());
        } catch (Exception e){
            System.out.println("null");
        }   

        System.out.println("\n");
        System.out.println("re register dead spy: AgentD");
        one.registerSpy(agentD);

        System.out.println("\n");

        System.out.println(agentE.getName());
        System.out.println(agentE.getFieldBase().getName());
        System.out.println(agentE.getFieldBase().getHomeBase().getName());

        System.out.println("\n");
        System.out.println("four going dark");
        System.out.println("\n");
        four.goDark();

        System.out.println(agentE.getName());
        System.out.println(agentE.getFieldBase().getName());
        try{
            System.out.println(agentE.getFieldBase().getHomeBase().getName());
        }catch(Exception e){
            System.out.println("Field Base four is dark");
        }

        System.out.println("\n");
        System.out.println("four back online");
        System.out.println("\n");
        four.goLight();

        System.out.println(agentE.getName());
        System.out.println(agentE.getFieldBase().getName());
        System.out.println(agentE.getFieldBase().getHomeBase().getName());

        home.setCypher(new Cypher1(new Cypher2(new DefaultCypher())));

        System.out.println("\nSecret Message:");
        two.send("This is the message", agentA);
        System.out.println(agentA.getMessage());


        
    

    }

}