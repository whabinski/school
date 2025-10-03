import java.util.*;
public class Board {

	private final int NUM_OF_COLUMNS = 7;
	private final int NUM_OF_ROW = 6;

	private char[][] b;
	
	/* 
	 * The board object must contain the board state in some manner.
	 * You must decide how you will do this.
	 * 
	 * You may add addition private/public methods to this class is you wish.
	 * However, you should use best OO practices. That is, you should not expose
	 * how the board is being implemented to other classes. Specifically, the
	 * Player classes.
	 * 
	 * You may add private and public methods if you wish. In fact, to achieve
	 * what the assignment is asking, you'll have to
	 * 
	 */
	
	public Board() {
	
		this.b = new char[NUM_OF_ROW][NUM_OF_COLUMNS];
		reset();
		
	}
	
	public void printBoard() {

		for (int i = 0; i < 6; i++){
			System.out.println("| " + this.b[i][0] + " | " + this.b[i][1] + " | " + this.b[i][2] + " | " + this.b[i][3] + " | "+ this.b[i][4] + " | " + this.b[i][5] + " | " + this.b[i][6] + " |");
		}
		System.out.println("-----------------------------");
		
	}
	
	public boolean containsWin() {

		//across
		for(int row = 0; row < NUM_OF_ROW; row++){
			for (int col = 0; col < NUM_OF_COLUMNS - 3; col++){
				if (b[row][col] != ' ' && (b[row][col] == b[row][col+1] && b[row][col] == b[row][col+2] && b[row][col] == b[row][col+3])){
					return true;
				}
			}			
		}

		//up and down
		for(int row = 0; row < NUM_OF_ROW - 3; row++){
			for(int col = 0; col < NUM_OF_COLUMNS; col++){
				if (b[row][col] != ' ' && (b[row][col] == b[row+1][col] && b[row][col] == b[row+2][col] && b[row][col] == b[row+3][col])){
					return true;
				}
			}
		}

		//down right diagonal
		for(int row = 0; row < NUM_OF_ROW - 3; row++){
			for(int col = 0; col < NUM_OF_COLUMNS - 3; col++){
				if (b[row][col] != ' ' && (b[row][col] == b[row+1][col+1] && b[row][col] == b[row+2][col+2] && b[row][col] == b[row+3][col+3])){
					return true;
				}
			}
		}

		//up right diagonal
		for(int row = 3; row < NUM_OF_ROW; row++){
			for(int col = 0; col < NUM_OF_COLUMNS - 3; col++){
				if (b[row][col] != ' ' && (b[row][col] == b[row-1][col+1] && b[row][col] == b[row-2][col+2] && b[row][col] == b[row-3][col+3])){
					return true;
				}
			}
		}

		return false;
		
	}
	
	public boolean isTie() {

		if (b[0][0] != ' ' && b[0][1] != ' ' && b[0][2] != ' ' && b[0][3] != ' ' && b[0][4] != ' ' && b[0][5] != ' ' && b[0][6] != ' '){
			return true;
		} 
		return false;
	}
	
	public void reset() {

		for (int i = 0; i < NUM_OF_ROW; i++){
			for (int j = 0; j < NUM_OF_COLUMNS; j++){
				this.b[i][j] = ' ';
			}
		}

	}

	public boolean setMove(int col, char symbol){

		col = col - 1;

		if(col < 0 || col > 6){
			return false;
		}

		if (this.b[5][col] == ' '){
			this.b[5][col] = symbol;
		}
		else if (this.b[4][col] == ' '){
			this.b[4][col] = symbol;
		}
		else if (this.b[3][col] == ' '){
			this.b[3][col] = symbol;
		}
		else if (this.b[2][col] == ' '){
			this.b[2][col] = symbol;
		}
		else if (this.b[1][col] == ' '){
			this.b[1][col] = symbol;
		}
		else if (this.b[0][col] == ' '){
			this.b[0][col] = symbol;
		}
		else {
			return false;
		}

		return true;
	}

	public ArrayList<Integer> moveToWin(char Symbol){
        
        ArrayList<Integer> possibilities = new ArrayList<Integer>();

        //across
		for(int row = 0; row < NUM_OF_ROW; row++){
			for (int col = 0; col < NUM_OF_COLUMNS - 3; col++){
				if (b[row][col] == ' ' && b[row][col+1] == Symbol && b[row][col+2] == Symbol && b[row][col+3] == Symbol){
                    if (findRowHeight(col) == row){
						possibilities.add(col);
					}
				}
                else if (b[row][col] == Symbol && b[row][col+1] == ' ' && b[row][col+2] == Symbol && b[row][col+3] == Symbol){
                    if (findRowHeight(col + 1) == row){
						possibilities.add(col + 1);
					}
				}
                else if (b[row][col] == Symbol && b[row][col+1] == Symbol && b[row][col+2] == ' ' && b[row][col+3] == Symbol){
                    if (findRowHeight(col + 2) == row){
						possibilities.add(col + 2);
					}
				}
                else if (b[row][col] == Symbol && b[row][col+1] == Symbol && b[row][col+2] == Symbol && b[row][col+3] == ' '){
                    if (findRowHeight(col + 3) == row){
						possibilities.add(col + 3);
					}
				}
			}			
		}

        //up and down
		for(int row = 0; row < NUM_OF_ROW - 3; row++){
			for(int col = 0; col < NUM_OF_COLUMNS; col++){
				if (b[row][col] == ' ' && b[row+1][col] == Symbol && b[row+2][col] == Symbol && b[row+3][col] == Symbol){
					possibilities.add(col);
				}
			}
		}

        //down right diagonal
		for(int row = 0; row < NUM_OF_ROW - 3; row++){
			for(int col = 0; col < NUM_OF_COLUMNS - 3; col++){
				if (b[row][col] == ' ' && b[row+1][col+1] == Symbol && b[row+2][col+2] == Symbol && b[row+3][col+3] == Symbol){
                    if (findRowHeight(col) == row){
						possibilities.add(col);
					}
                }
                else if (b[row][col] == Symbol && b[row+1][col+1] == ' ' && b[row+2][col+2] == Symbol && b[row+3][col+3] == Symbol){
                    if (findRowHeight(col + 1) == (row + 1)){
						possibilities.add(col + 1);
					}
                }
                else if (b[row][col] == Symbol && b[row+1][col+1] == Symbol && b[row+2][col+2] == ' ' && b[row+3][col+3] == Symbol){
                    if (findRowHeight(col + 2) == (row + 2)){
						possibilities.add(col + 2);
					}
                }
                else if (b[row][col] == Symbol && b[row+1][col+1] == Symbol && b[row+2][col+2] == Symbol && b[row+3][col+3] == ' '){
                    if (findRowHeight(col + 3) == row + 3){
						possibilities.add(col + 3);
					}
                }
			}
		}

        //up right diagonal
        for(int row = 3; row < NUM_OF_ROW; row++){
			for(int col = 0; col < NUM_OF_COLUMNS - 3; col++){
				if (b[row][col] == ' ' && b[row-1][col+1] == Symbol && b[row-2][col+2] == Symbol && b[row-3][col+3] == Symbol){
                    if (findRowHeight(col) == row){
						possibilities.add(col);
					}
                }
                else if (b[row][col] == Symbol && b[row-1][col+1] == ' ' && b[row-2][col+2] == Symbol && b[row-3][col+3] == Symbol){
                    if (findRowHeight(col + 1) == (row - 1)){
						possibilities.add(col + 1);
					}
                }
                else if (b[row][col] == Symbol && b[row-1][col+1] == Symbol && b[row-2][col+2] == ' ' && b[row-3][col+3] == Symbol){
                    if (findRowHeight(col + 2) == (row - 2)){
						possibilities.add(col + 2);
					}
                }
                else if (b[row][col] == Symbol && b[row-1][col+1] == Symbol && b[row-2][col+2] == Symbol && b[row-3][col+3] == ' '){
                    if (findRowHeight(col + 3) == (row - 3)){
						possibilities.add(col + 3);
					}
                }
			}
		}

        return possibilities;

    }

	public ArrayList<Integer> moveToBlock(char Symbol){
        
        ArrayList<Integer> possibilities = new ArrayList<Integer>();

        //across
		for(int row = 0; row < NUM_OF_ROW; row++){
			for (int col = 0; col < NUM_OF_COLUMNS - 3; col++){
				if (b[row][col] == ' ' && (b[row][col+1] != Symbol && b[row][col+1] != ' ') && (b[row][col+2] != Symbol && b[row][col+2] != ' ') && (b[row][col+3] != Symbol && b[row][col+3] != ' ')){
                    if (findRowHeight(col) == row){
						possibilities.add(col);
					}
				}
                else if ((b[row][col] != Symbol && b[row][col] != ' ') && b[row][col+1] == ' ' && (b[row][col+2] != Symbol && b[row][col+2] != ' ') && (b[row][col+3] != Symbol && b[row][col+3] != ' ')){
					if (findRowHeight(col + 1) == row){
						possibilities.add(col + 1);
					}
				}
                else if ((b[row][col] != Symbol && b[row][col] != ' ') && (b[row][col+1] != Symbol && b[row][col+1] != ' ') && b[row][col+2] == ' ' && (b[row][col+3] != Symbol && b[row][col+3] != ' ')){
                    if (findRowHeight(col + 2) == row){
						possibilities.add(col + 2);
					}
				}
                else if ((b[row][col] != Symbol && b[row][col] != ' ') && (b[row][col+1] != Symbol && b[row][col+1] != ' ') && (b[row][col+2] != Symbol && b[row][col+2] != ' ') && b[row][col+3] == ' '){
					if (findRowHeight(col + 3) == row){
						possibilities.add(col + 3);
					}
				}
			}			
		}

        //up and down
		for(int row = 0; row < NUM_OF_ROW - 3; row++){
			for(int col = 0; col < NUM_OF_COLUMNS; col++){
				if (b[row][col] == ' ' && (b[row+1][col] != Symbol && b[row+1][col] != ' ') && (b[row+2][col] != Symbol && b[row+2][col] != ' ') && (b[row+3][col] != Symbol && b[row+3][col] != ' ')){
					possibilities.add(col);
				}
			}
		}

		//down right diagonal
		for(int row = 0; row < NUM_OF_ROW - 3; row++){
			for(int col = 0; col < NUM_OF_COLUMNS - 3; col++){
				if (b[row][col] == ' ' && (b[row+1][col+1] != Symbol && b[row+1][col+1] != ' ') && (b[row+2][col+2] != Symbol && b[row+2][col+2] != ' ') && (b[row+3][col+3] != Symbol && b[row+3][col+3] != ' ')){
                    if (findRowHeight(col) == row){
						possibilities.add(col);
					}
                }
                else if ((b[row][col] != Symbol && b[row][col] != ' ') && b[row+1][col+1] == ' ' && (b[row+2][col+2] != Symbol && b[row+2][col+2] != ' ') && (b[row+3][col+3] != Symbol && b[row+3][col+3] != ' ')){
                    if (findRowHeight(col + 1) == (row + 1)){
						possibilities.add(col + 1);
					}
                }
                else if ((b[row][col] != Symbol && b[row][col] != ' ') && (b[row+1][col+1] != Symbol && b[row+1][col+1] != ' ') && b[row+2][col+2] == ' ' && (b[row+3][col+3] != Symbol && b[row+3][col+3] != ' ')){
                    if (findRowHeight(col + 2) == (row + 2)){
						possibilities.add(col + 2);
					}
                }
                else if ((b[row][col] != Symbol && b[row][col] != ' ') && (b[row+1][col+1] != Symbol && b[row+1][col+1] != ' ') && (b[row+2][col+2] != Symbol && b[row+2][col+2] != ' ') && b[row+3][col+3] == ' '){
                    if (findRowHeight(col + 3) == (row + 3)){
						possibilities.add(col + 3);
					}
                }
			}
		}

		//up right diagonal
        for(int row = 3; row < NUM_OF_ROW; row++){
			for(int col = 0; col < NUM_OF_COLUMNS - 3; col++){
				if (b[row][col] == ' ' && (b[row-1][col+1] != Symbol && b[row-1][col+1] != ' ') && (b[row-2][col+2] != Symbol && b[row-2][col+2] != ' ') && (b[row-3][col+3] != Symbol && b[row-3][col+3] != ' ')){
					if (findRowHeight(col) == row){
						possibilities.add(col);
					}
                }
                else if ((b[row][col] != Symbol && b[row][col] != ' ') && b[row-1][col+1] == ' ' && (b[row-2][col+2] != Symbol && b[row-2][col+2] != ' ') && (b[row-3][col+3] != Symbol && b[row-3][col+3] != ' ')){
					if (findRowHeight(col + 1) == (row - 1)){
						possibilities.add(col + 1);
					}
                }
                else if ((b[row][col] != Symbol && b[row][col] != ' ') && (b[row-1][col+1] != Symbol && b[row-1][col+1] != ' ') && b[row-2][col+2] == ' ' && (b[row-3][col+3] != Symbol && b[row-3][col+3] != ' ')){
                    if (findRowHeight(col + 2) == (row - 2)){
						possibilities.add(col + 2);
					}
                }
                else if ((b[row][col] != Symbol && b[row][col] != ' ') && (b[row-1][col+1] != Symbol && b[row-1][col+1] != ' ') && (b[row-2][col+2] != Symbol && b[row-2][col+2] != ' ') && b[row-3][col+3] == ' '){
                    if (findRowHeight(col + 3) == (row - 3)){
						possibilities.add(col + 3);
					}
                }
			}
		}

        return possibilities;
    }

	private int findRowHeight(int col){
		
		if (this.b[5][col] == ' '){
			return 5;
		}
		else if (this.b[4][col] == ' '){
			return 4;
		}
		else if (this.b[3][col] == ' '){
			return 3;
		}
		else if (this.b[2][col] == ' '){
			return 2;
		}
		else if (this.b[1][col] == ' '){
			return 1;
		}
		else {
			return 0;
		}
	}

}
