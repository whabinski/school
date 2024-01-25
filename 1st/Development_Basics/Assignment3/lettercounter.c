#include <stdio.h>

void countText(int Letters[], char Text[], int len);
void makeTable(int count[], char letters[], int len);

int main(){

    // Initializers
    int letterCount[26] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    char Letters[26] = {'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'};
    char Text[1024];
    int textLength;

    // Prompts and recieves input from the user
    printf("Enter Text for Analysis: \n");
    scanf("%[^\n]", Text);


    // Finds length of the Text
    for (int i = 0; Text[i]; i++){
        textLength = i;
    }

    //Calls Helper Functions
    countText(letterCount,Text,textLength);
    makeTable(letterCount, Letters, textLength);

    return 0;
}

// Helper function to count how many letters are in the text
void countText(int Letters[], char Text[], int len){
    
    // loops through each character in the string
    for (int i = 0; i <= len; i++){
        // Switch statement to add to the counter of each letter when a letter is found in the string
        switch (Text[i])
        {
        case 'A':
        case 'a':
            Letters[0] += 1;
            break;
        case 'B':
        case 'b':
            Letters[1] += 1;
            break;
        case 'C':
        case 'c':
            Letters[2] += 1;
            break;
        case 'D':
        case 'd':
            Letters[3] += 1;
            break;
        case 'E':
        case 'e':
            Letters[4] += 1;
            break;
        case 'F':
        case 'f':
            Letters[5] += 1;
            break;
        case 'G':
        case 'g':
            Letters[6] += 1;
            break;
        case 'H':
        case 'h':
            Letters[7] += 1;
            break;
        case 'I':
        case 'i':
            Letters[8] += 1;
            break;
        case 'J':
        case 'j':
            Letters[9] += 1;
            break;
        case 'K':
        case 'k':
            Letters[10] += 1;
            break;
        case 'L':
        case 'l':
            Letters[11] += 1;
            break;
        case 'M':
        case 'm':
            Letters[12] += 1;
            break;
        case 'N':
        case 'n':
            Letters[13] += 1;
            break;
        case 'O':
        case 'o':
            Letters[14] += 1;
            break;
        case 'P':
        case 'p':
            Letters[15] += 1;
            break;
        case 'Q':
        case 'q':
            Letters[16] += 1;
            break;
        case 'R':
        case 'r':
            Letters[17] += 1;
            break;
        case 'S':
        case 's':
            Letters[18] += 1;
            break;
        case 'T':
        case 't':
            Letters[19] += 1;
            break;
        case 'U':
        case 'u':
            Letters[20] += 1;
            break;
        case 'V':
        case 'v':
            Letters[21] += 1;
            break;
        case 'W':
        case 'w':
            Letters[22] += 1;
            break;
        case 'X':
        case 'x':
            Letters[23] += 1;
            break;
        case 'Y':
        case 'y':
            Letters[24] += 1;
            break;
        case 'Z':
        case 'z':
            Letters[25] += 1;
            break;
        default:
            break;
        }
    }

}

// Helper function to make the table and output the data
void makeTable(int count[], char letters[], int len){

    int max = count[0], min = count[0];

    printf("\nMax = %d", max);
    printf("\nMin = %d", min);

    printf("\nLetter Analysis Complete\n");
    printf("%-10s%-15s%-15s\n","Letter","Occurances","Percentage");
    printf("****************************************\n");

    // Loop through all the letters and outputs the corresponding row of data
    for (int i = 0; i < 26; i++){
        float percentage = (((float) count[i])/((float) (len + 1)))*100;
        printf("%-10c%-15d%-15.2f\n", letters[i], count[i], percentage);

        if (count[i] >= count[max]) {
            max = i;}
        if (count[i] <= count[min]) {
            min = i;}
    }

    // Outputs the final frequent letters
    printf("\nThe most frequently occuring character is %c\n", letters[max]);
    printf("The least frequently occuring character is %c\n", letters[min]);

}