public class parenthesisMatching {

    public static boolean isValid(String exp){

        Stack<String> s = new Stack<>();

        for (int i = 0; i < exp.length(); i++){
            if (exp.charAt(i) == '('){
                s.push("(");
            }
            else if (exp.charAt(i) == ')'){
                if (s.isEmpty()){
                    return false;
                }
            s.pop();
            }
        }

        if(!s.isEmpty()){
            return false;
        }
        return true;
    }

    public static void main (String[] args) {
        String exp = "())()))()()((()()((()()";
        String exp2 = "((())((()))())";
        String exp3 = "))((";
        String exp4 = "(())(()";
        System.out.println(isValid(exp2));
    }   
}