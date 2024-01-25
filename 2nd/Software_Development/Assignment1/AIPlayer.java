import java.util.*;
import java.util.Random;
public class AIPlayer extends Player{

    private Random random = new Random();

    public AIPlayer(char symbol, Board board, String name) {
        super(symbol, board, name);
    }

    public void makeMove(Board board) {
        int num = random.nextInt(7);
        ArrayList<Integer> possibleWins = board.moveToWin(super.symbol);
        ArrayList<Integer> possibleBlocks = board.moveToBlock(super.symbol);

        if (!possibleWins.isEmpty()){
            for (Integer i : possibleWins){
                if (board.setMove(i + 1, super.symbol)){
                    num = i + 1;
                    break;
                }
            }
        }
        else if (!possibleBlocks.isEmpty()){
            for (Integer i : possibleBlocks){
                if (board.setMove(i + 1, super.symbol)){
                    num = i + 1;
                    break;
                }
            }
        }
        else{
            while (!board.setMove(num, super.symbol)){
                num = random.nextInt(7);
            }
        }

        System.out.println(super.name + " chooses: " + num);
    }
    
    
}
