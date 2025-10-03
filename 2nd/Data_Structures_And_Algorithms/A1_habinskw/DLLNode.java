public class DLLNode<T> {
	public T key;
	public DLLNode<T> next;
	public DLLNode<T> prev;
	
	public DLLNode(T init) {
		key = init;
		next = null;
		prev = null;
	}
}
