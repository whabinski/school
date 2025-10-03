public class Stack<T> {
	public SLL<T> stack;

	public Stack() {
		stack = new SLL<T>();
	}

	public boolean isEmpty() {
		return stack.head == null;
	}

	public void push(T x) {
		SLLNode<T> node = new SLLNode<T>(x);
		stack.insert(node);
	}

	public T pop() {
		T x = stack.head.key;
		stack.delete(stack.head);
		return x;
	}

	public int count() {
		return stack.count();
	}

}
