public class selectionSort{

    public static DLL<Integer> l = new DLL<>();
    public static DLL<Integer> l2 = new DLL<>();

    public static void tester(){

        var n1 = new DLLNode<Integer>(6);
        var n2 = new DLLNode<Integer>(19);
        var n3 = new DLLNode<Integer>(5);
        var n4 = new DLLNode<Integer>(24);
        var n5 = new DLLNode<Integer>(12);
        var n6 = new DLLNode<Integer>(9);

        l.insert(n5);
        l.insert(n6);
        l.insert(n1);
        l.insert(n4);
        l.insert(n3);
        l.insert(n2);
  
        l.traverse();
    }

    public static DLL<Integer> sort(DLL<Integer> l){

        DLLNode<Integer> walk = l.head;
        DLLNode<Integer> current = l.head;
        DLLNode<Integer> min = l.head;


        while (walk != null){
            while (current != null){
                if(current.key < min.key){
                    min = current;
                }
                current = current.next;
            }

            if (walk == l.head)
                l.head = min;

            DLLNode<Integer> temp = walk.next;

            walk.next = min.next;
            min.next = temp;
        
            if (min.next != null){
                min.next.prev = min;
            }   
            if (walk.next != null){
                walk.next.prev = walk;
            }
                
            temp = walk.prev;
            walk.prev = min.prev;
            min.prev = temp;
        
            if (min.prev != null){
                min.prev.next = min;
            }
            if (walk.prev != null){
                walk.prev.next = walk;
            }
            
            walk = min.next;
            current = walk;
            min = walk;
        }

        return l;
    }

    public static void main (String[] args) {
        tester();
        System.out.println("------");
        l = sort(l);
        l.traverse();
    }   
}