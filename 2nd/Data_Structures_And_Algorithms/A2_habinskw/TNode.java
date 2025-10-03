public class TNode<T>{

    public T key;
    public TNode<T> left;
    public TNode<T> right;
    public TNode<T> middle;

    public TNode(T init){
        key = init;
        left = null;
        right = null;
        middle = null;
    }
}
