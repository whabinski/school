import java.util.*;

public class Mystery1 {

    public ArrayList<Tuple> S;
    
    public Mystery1(){
        S = new ArrayList<Tuple>();
    }

    public void f1(String s1, String s2){

        boolean newTuple = true;
        Tuple t = find(s1);

        if (s1 == s2){
            return;
        }

        if (t != null && (t.first == s1 && t.last.contains(s2))){
            newTuple = false;
        }
        else if (t != null && (t.first == s1)){
            t.last.add(s2);
            newTuple = false;
        }

        if (newTuple == true){
            S.add(new Tuple(s1,s2));
        }
    }

    public int f2(String s){
        int count = 0;

        for (Tuple t : this.S){
            if (t.last.contains(s)){
                count ++;
            }
        }

        return count;
    }

    public boolean f3(String s1, String s2){

        if (DFS(s1, new ArrayList<Tuple>(), new ArrayList<String>()).contains(s2)){
            if(find(s2).last.contains(s1)){
                return true;
            }
        }
        
        return false;
    }

    private Tuple find(String s){
        for (Tuple t : this.S){
            if (t.first == s){
                return t;
            }
        }
        return null;
    }

    private ArrayList<String> DFS(String s, ArrayList<Tuple> marked, ArrayList<String> results){

        Tuple t = find(s);
        if (t == null){ return results; }
        
        if (!marked.contains(t)){
            results.add(t.first);
            marked.add(t);
            for (String str : t.last){
                DFS(str, marked, results);
            }
        }
        return results;
    }

    private class Tuple{

        String first;
        ArrayList<String> last;
    
        public Tuple(String s1, String s2){
            this.last = new ArrayList<>();
            this.first = s1;
            this.last.add(s2);
        }
    
    }

}
