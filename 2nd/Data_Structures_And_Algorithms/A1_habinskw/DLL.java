public class DLL<T> {
	public DLLNode<T> head, tail;

	public void insert(DLLNode<T> x) {
		if(head == null) {  
            head = tail = x;   
            head.prev = null;   
            tail.next = null;  
        }  
        else {    
            tail.next = x;  
            x.prev = tail;  
            tail = x;    
            tail.next = null;  
        }  
	}

	public int count() {
		int c = 0;
		DLLNode<T> walk = this.head;
		while (walk != null) {
			c++;
			walk = walk.next;
		}		
		return c;
	}

	public void traverse(){
		DLLNode<T> walk = this.head;
		while (walk != null){
			System.out.println(walk.key);
			walk = walk.next;
		}
	}

	public DLL() {
		head = null;
	}
}
