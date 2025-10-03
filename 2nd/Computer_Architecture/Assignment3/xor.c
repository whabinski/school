#include <stdio.h> 
#include <stdlib.h>
#include <stdbool.h>

extern int axor(int a, int b);

int xor(int a, int b){
    return a^b;
}

int main(){

    printf("5^9: %d\n",xor(5, 9));
    printf("17^4: %d\n",xor(17, 4));
    printf("2^90: %d\n",xor(2, 90));
    printf("8^12: %d\n",xor(8, 12));

    printf("With axor.s implementation (5^9): %d\n", axor(5,9));
    printf("With axor.s implementation (17^4): %d\n", axor(17,4));

    return 0;
}
