#include <stdio.h> 
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

char toFloat(float arg);
void toBin(char c);
   
int main(){    

    float num;

    printf("Enter any number:");

    scanf("%f",&num);

    printf("\nThe Char value is: %c", toFloat(num));
    printf("\n");
    toBin(toFloat(num));


    return 0;
}


char toFloat(float arg){

//out of bounds ranges
    if (arg > 31 || arg < -31){
        return ' ';
    }

    int *exp = (int *) calloc(3, sizeof(int));
    int *bin = (int *) calloc(8, sizeof(int));
    int *bigNum = (int *) calloc(4, sizeof(int));
    int *smallNum = (int *) calloc(4, sizeof(int));

    int *newMantissa = (int *) calloc(4, sizeof(int));
    int *newNumber = (int *) calloc(12, sizeof(int));

    int wholeNum, countBigNum = 0, countSmallNum = 0;
    int bias = 3;
    int sign = (arg >= 0) ? 0 : 1;
    float decimalNum;

    arg = fabsf(arg); // absolute val of float
    wholeNum = arg; // wholeNum from float
    decimalNum = arg - wholeNum; // fraction from float

//calculate whole number to binary
    while(wholeNum != 0){
        bigNum[countBigNum] = wholeNum%2;
        wholeNum = wholeNum / 2;
        countBigNum += 1;
    }

//calulate fraction to binary
    if (decimalNum > 0){
        for (int i = 0; i < 5; i++){
            decimalNum = decimalNum*2;
            smallNum[i] = decimalNum;
            if ((int) decimalNum == 1){
                decimalNum = decimalNum - 1;
            }
            countSmallNum += 1;
        }
    }

//calc 4 bit mantissa
    int newCount = 0;
    for (int i = countBigNum - 1; i >= 0; i--){
        newNumber[newCount] = bigNum[i];
        newCount += 1;
    }

    for (int i = 0; i < countSmallNum; i ++){
        newNumber[newCount] = smallNum[i];
        newCount += 1;
    }

    for (int i = 0; i < 4; i++){
        if (newNumber[i] == 1){
            newMantissa[0] = newNumber[i+1];
            newMantissa[1] = newNumber[i+2];
            newMantissa[2] = newNumber[i+3];
            newMantissa[3] = newNumber[i+4];
            break;
        }
    }

// calculate exp
    int biasedExp = log2f(arg) + bias;
    int countExp = 0;
    while(biasedExp != 0){
        exp[countExp] = biasedExp%2;
        biasedExp = biasedExp / 2;
        countExp += 1;
    }

//calculate 8-bit float
    bin[0] = sign;
    bin[1] = exp[2];
    bin[2] = exp[1];
    bin[3] = exp[0];
    bin[4] = newMantissa[0];
    bin[5] = newMantissa[1];
    bin[6] = newMantissa[2];
    bin[7] = newMantissa[3];

    //convert int array to char array
    char result[8];
    for (int i = 0; i < 8; i++){
        if (bin[i] == 1){
            result[i] = '1';
        }
        else{
            result[i] = '0';
        }
    }
    return strtol(result,0,2);
}

void toBin(char c){

    int *bin = calloc(8, sizeof(int));
    int count = 0;
    
    // calculate 8 - bit float from char
    for (int i = 7; i >= 0; i--){
        bin[count] = ((c & (1 << i)) ? 1 : 0);
        count += 1;
    }

    // convert int array to char 
    char result[8];
    for (int i = 0; i < 8; i++){
        if (bin[i] == 1){
            result[i] = '1';
        }
        else if(bin[0] == 0){
            result[i] = '0';
        }
    }

    printf("The Binary value is: ");
    for (int i = 0; i < 8; i++){
        printf("%c", result[i]);
    }
}