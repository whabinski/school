import java.util.Scanner;
public class HumanPlayer extends Player{

    private Scanner s = new Scanner(System.in);

    public HumanPlayer(char symbol, Board board, String name) {
        super(symbol, board, name);
    }

    public void makeMove(Board board) {

        System.out.print(super.name + ", please input your move: ");
        int input = Integer.valueOf(s.nextLine());

        while (!board.setMove(input, super.symbol)){
            System.out.print("Invalid entry, please input another move: ");
            input = Integer.valueOf(s.nextLine());
        }
    }
    
}