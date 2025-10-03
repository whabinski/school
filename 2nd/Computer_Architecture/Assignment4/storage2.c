#include <stdio.h>
#include <string.h>
#include <stdlib.h>


// Allocates 2D Char Array
char *two_d_alloc(size_t N, size_t M, size_t sz){
    return (char *)malloc(N * M * sz);
}

// Deallocates 2D Char Array
void two_d_dealloc(char *array){
    free(array);
}

// Stores Int into 2D Char Array
int two_d_store_int(int arg, char *array, size_t i, size_t j, size_t M, size_t N){
    if ((i >= N || i < 0) || (j >= M || j < 0) || (array == NULL)){
        printf("Input is invalid. ");
        return -1;
    }
    array[(i * M + j) * sizeof(int)] = arg;
    return 0;
}

// Fetches Int from 2D Char Array
int two_d_fetch_int(char *array, size_t i, size_t j, size_t M, size_t N){
    if ((i >= N || i < 0) || (j >= M || j < 0) || (array == NULL)){
        printf("Input is invalid. ");
        return -1;
    }
    return array[(i * M + j) * sizeof(int)];
}


// Stores any data type in 2D Char Array
int two_d_store(void *arg, char *array, size_t i, size_t j, size_t M, size_t N, size_t sz){
    if ((i >= N || i < 0) || (j >= M || j < 0) || (array == NULL)){
        printf("Input is invalid. ");
        return -1;
    }
    memcpy((void *)(array + (i * M + j) * sz), arg, sz);
    return 0;
}

// Fetches item of any data type from 2D Array
void *two_d_fetch(char *array, size_t i, size_t j, size_t M, size_t N, size_t sz){
    if ((i >= N || i < 0) || (j >= M || j < 0) || (array == NULL)){
        printf("Input is invalid. ");
        return NULL;
    }
    return (void *)(array + ((i * M + j) * sz));
}


int main(){
    // Testing allocating
    char *test = two_d_alloc(3, 3, sizeof(int));

    // Storing numbers in 4 randomly selected locations
    printf("Storing 10 at Row 1, Column 3\n");
    two_d_store_int(10, test, 0, 2, 3, 3);
    printf("Storing 4 at Row 2, Column 1\n");
    two_d_store_int(4, test, 1, 0, 3, 3);
    printf("Storing 123 at Row 3, Column 2\n");
    two_d_store_int(123, test, 2, 1, 3, 3);
    printf("Storing 66 at Row 1, Column 1\n");
    two_d_store_int(66, test, 0, 0, 3, 3);
    
    // Prints the numbers from the 4 locations
    printf("Fetching from Row 1, Column 3. Should be 10:\n");
    printf("%d", two_d_fetch_int(test, 0, 2, 3, 3));
    printf("\nFetching from Row 2, Column 1. Should be 4:\n");
    printf("%d", two_d_fetch_int(test, 1, 0, 3, 3));
    printf("\nFetching from Row 3, Column 2. Should be 123:\n");
    printf("%d", two_d_fetch_int(test, 2, 1, 3, 3));
    printf("\nFetching from Row 1, Column 1. Should be 66:\n");
    printf("%d", two_d_fetch_int(test, 0, 0, 3, 3));

    // Frees memory
    two_d_dealloc(test);

    // Allocating to test Out of Bounds
    char *test2 = two_d_alloc(3, 3, sizeof(int));

    // Tests every edge case. Should print 4 errors
    two_d_store_int(123, test2, -1, 2, 3, 3);
    two_d_store_int(123, test2, 2, -1, 3, 3);
    two_d_store_int(123, test2, 4, 2, 3, 3);
    two_d_store_int(123, test2, 2, 4, 3, 3);

    // Frees memory
    two_d_dealloc(test2);

    // Allocates float array
    char *test3 = two_d_alloc(3, 3, sizeof(float));

     // Storing numbers in 4 randomly selected locations
    printf("Storing 10.5 at Row 1, Column 3\n");
    float float1 = 10.5;
    void *floatPointer = &float1;
    two_d_store(floatPointer, test3, 0, 2, 3, 3, sizeof(float));
    printf("Storing 4.91 at Row 2, Column 1\n");
    float float2 = 4.91;
    void *floatPointer2 = &float2;
    two_d_store(floatPointer2, test3, 1, 0, 3, 3, sizeof(float));
    printf("Storing -12.002 at Row 3, Column 2\n");
    float float3 = -12.002;
    void *floatPointer3 = &float3;
    two_d_store(floatPointer3, test3, 2, 1, 3, 3, sizeof(float));
    printf("Storing 66.1234 at Row 1, Column 1\n");
    float float4 = 66.666;
    void *floatPointer4 = &float4;
    two_d_store(floatPointer4, test3, 0, 0, 3, 3, sizeof(float));
    
    // Prints the numbers from the 4 locations
    printf("Fetching from Row 1, Column 3. Should be 10.5:\n");
    printf("%.1f", (*((float *)two_d_fetch(test3, 0, 2, 3, 3, sizeof(float)))));
    printf("\nFetching from Row 2, Column 1. Should be 4.91:\n");
    printf("%.2f", (*((float *)two_d_fetch(test3, 1, 0, 3, 3, sizeof(float)))));
    printf("\nFetching from Row 3, Column 2. Should be -12.002:\n");
    printf("%.3f", (*((float *)two_d_fetch(test3, 2, 1, 3, 3, sizeof(float)))));
    printf("\nFetching from Row 1, Column 1. Should be 66.666:\n");
    printf("%.3f", (*((float *)two_d_fetch(test3, 0, 0, 3, 3, sizeof(float)))));

    // Frees memory
    two_d_dealloc(test);

    
    return 0;
}
