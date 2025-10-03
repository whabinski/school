#include <stdio.h>

const int NUMS = 3;


int main(){

    int highs, lows;

    // Output Table Title
    printf(">----------------- Temperature Analyzer -------------------<\n");


    // For loop for each day in NUMS
    for (int i = 0; i < NUMS; i++){
        
        do{
        printf("Enter the high value for day %d: ", (i + 1));
        scanf("%d", &highs);
        printf("Enter the low value for day %d: ", (i + 1));
        scanf("%d", &lows);
        //Check if values are acceptable
        if ((!(highs > lows) || !(highs < 41) || !(lows > -41))){
            printf("Incorrect values, temperatures must be in the range -40 to 40, high must be greater than low.\n");
        }
        } while (!(highs > lows) || !(highs < 41) || !(lows > -41)); 

    }

    return 0;
}
        