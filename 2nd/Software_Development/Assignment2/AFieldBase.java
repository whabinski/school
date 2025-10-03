
public abstract class AFieldBase extends Message {

    public abstract void goDark();
    public abstract void goLight();
    public abstract void registerSpy(Spy spy);
    public abstract void agentTerminated(Spy spy);
    public abstract HomeBase getHomeBase();
}