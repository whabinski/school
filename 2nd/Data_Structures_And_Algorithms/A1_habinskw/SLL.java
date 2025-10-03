
public class SLL<T> {

	public SLLNode<T> head;

	public SLLNode<T> search(T k) {
		SLLNode<T> x = this.head;
		while (x != null && x.key != k) {
			x = x.next;
		}
		return x; 
	}

	public int count() {
		int c = 0;
		SLLNode<T> walk = this.head;
		while (walk != null) {
			c++;
			walk = walk.next;
		}		
		return c;
	}

	public void insert(SLLNode<T> x) {
		x.next = head;
		head = x;
	}

	public void delete(SLLNode<T> x) {
		if (x == null) {
			return;
		}
		if (x == this.head) {
			this.head = x.next;
			return;
		}

		SLLNode<T> walk = this.head;
		while (walk.next != x) {
			walk = walk.next;
		}
		walk.next = walk.next.next; 
	}

	public SLL() {
		head = null;
	}
}
