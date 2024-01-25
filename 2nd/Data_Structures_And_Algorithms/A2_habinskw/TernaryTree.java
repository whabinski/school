public class TernaryTree<T>{

    public TNode<T> root;

    public void insert(TNode<T> x){

        if (root == null){
            root = x;
			return;
        }

        TNode<T> current = root;
        while (true){

			if ((x.key.toString().compareTo(current.key.toString())) < 0){
				if (current.left == null){
					current.left = x;
					return;
				}
					current = current.left;
            }
            else if ((x.key.toString().compareTo(current.key.toString())) > 0){
				if (current.right == null){
					current.right = x;
					return;
				}
					current = current.right;
            }
            else{
				if (current.middle == null){
					current.middle = x;
					return;
				}
					current = current.middle;
            }
        }
    }

	public TNode<T> search(T val){

		TNode<T> current = root;

		while (current != null) {
			if (current.key.toString().compareTo(val.toString()) == 0){
				return current;
			}
			else if (current.key.toString().compareTo(val.toString()) < 0){
				current = current.right;
			}
			else if (current.key.toString().compareTo(val.toString()) > 0){
				current = current.left;
			}
		}

		return null;
	}

    public void print(TNode<T> t){

		System.out.println(t.key);

        if (t.left != null){
			System.out.print("parent: " + t.key + "       ");
            print(t.left);
        }
		if (t.middle != null){
			System.out.print("parent: " + t.key + "       ");
            print(t.middle);
        }
        if (t.right != null){
			System.out.print("parent: " + t.key + "       ");
            print(t.right);
        }
    

    }

    public static void main(String[] args){
    }

}