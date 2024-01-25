#include <stdio.h> 
#include <stdlib.h>
#include <stdbool.h>

void partOne();
void partTwo();
   
int main(){    

    partOne();
    partTwo();

    return 0;
}

// Little Endian or Big Endian checker
void partOne(){

    // Allocates byte array in memory
    char *bytes = (char *) calloc(8, sizeof(char));

    // Interprets array pointer as int
    int *ptr = (int*) bytes;
    // Stores hex number in memory
    *ptr = 0x04030201;

    //Fluff
    printf("------------------------------------------------\n");
    printf("PART ONE\n");
    printf("------------------------------------------------\n");
    printf("The bytes: \n");

    //Prints array of bytes
    for (int i = 0; i < 8; i++){
        printf("%d ", bytes[i]);
    }

    printf("\n");

    // Determines if the integer was stored in big endian vs little endian
    if (bytes[0] == 1 && bytes[1] == 2 && bytes[2] == 3 && bytes[3] == 4){
        printf("Therefore the integer was stored in little endian\n");
    } 
    else if (bytes[0] == 4 && bytes[1] == 3 && bytes[2] == 2 && bytes[3] == 1){
        printf("Therefore the integer was stored in big endian\n");
    }

    printf("------------------------------------------------\n");

}

void partTwo(){

        char *bytes = (char *) calloc(8, sizeof(char));

        // Interprets array pointer as int
        int *ptr = (int*) bytes;
        
        *ptr = 46546;

        //10110101 11010010 = 46546
        //2s = -75 -46 in little endian it should be -46 -75
        //signed = -53 -82 in little endian it should be -82 -53

        printf("------------------------------------------------\n");
        printf("PART TWO\n");
        printf("------------------------------------------------\n");
        printf("The bytes: \n");

        // prints arrary of bytes
        for (int i = 0; i < 8; i++){
            printf("%d ", bytes[i]);
        }

        if (bytes[0] == -46 && bytes[1] == -75){
            printf("\nThe architecture uses twos complement");
        }
        else if (bytes[0] == -82 && bytes[1] == -53){
            printf("\nThe architecture uses signed magnitude");
        }
}


