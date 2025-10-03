#include <stdio.h> 
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>

char* two_d_alloc(size_t N, size_t M, size_t sz);
void two_d_dealloc(char* array);
int two_d_store_int(int arg, char* array, size_t i, size_t j, size_t M, size_t N);
int two_d_fetch_int(char* array, size_t i, size_t j, size_t M, size_t N);
int two_d_store(void* arg, char* array, size_t i, size_t j, size_t M, size_t N, size_t sz);
void* two_d_fetch(char* array, size_t i, size_t j, size_t M, size_t N, size_t sz);

void fillArray(char* arr, size_t N, size_t M, size_t sz);
void printArray(char* arr, size_t N, size_t M, size_t sz);
void printSmallArray(char* arr, size_t N, size_t M, size_t sz);

// Main Function Call
int main(){    

    int row = 3, col = 3;
    size_t size = sizeof(float);
    float arg = 9.25;

    char *buf = two_d_alloc(row,col, size);

    //Fill array with 1,2,3,4,5....
    fillArray(buf, row, col, size);
    //Store int
    printf("%d\n", two_d_store_int(-1, buf, 1, 2, col, row));
    //Fetch int
    printf("%d\n",two_d_fetch_int(buf, 1, 2, col, row));
    //Store value
    printf("Want to store: 9.25 at row: 2, col: 2\n");
    printf("%d\n",two_d_store(&arg, buf, 2, 2, col, row, size));
    //Fetch value
    printf("%f\n",*((float*)(two_d_fetch(buf, 2, 2, col, row, size))));
    //Print the array
    //printArray(buf, row, col, size);
    //Deallocate array
    two_d_dealloc(buf);
    return 0;
}

// Allocates an N by M "2D" array of size sz
char* two_d_alloc(size_t N, size_t M, size_t sz){
    return malloc(N * M * sz);
}

// Deallocates / frees the array
void two_d_dealloc(char* array){
    free(array);
}

// Stores an integer value arg in row i and column j of an N by M matrix
int two_d_store_int(int arg, char* array, size_t i, size_t j, size_t M, size_t N){

    if (array == NULL || i > N || i < 0 || j > M || j < 0) {
        printf("Error, inputs are inccorect ");
        return -1;
    }

    printf("Want to store: %d at row: %zu, col: %zu\n", arg, i, j);

    size_t sz = sizeof(int); 
    array[i * (M * sz) + (j * sz)] = arg;
    return 0;
}

// Fetches the value at row i and column j of an N by M matrix
int two_d_fetch_int(char* array, size_t i, size_t j, size_t M, size_t N){

    if (array == NULL || i > N || i < 0 || j > M || j < 0) {
        printf("Error, inputs are inccorect ");
        return -1;
    }

    printf("Want to fetch value at row: %zu, col: %zu\n", i, j);

    size_t sz = sizeof(int);
    return array[i * (M * sz) + (j * sz)];

    return -1;
}

// Store value arg at row and col
int two_d_store(void* arg, char* array, size_t i, size_t j, size_t M, size_t N, size_t sz){

    if (array == NULL || i > N || i < 0 || j > M || j < 0) {
        printf("Error, inputs are inccorect ");
        return -1;
    }

    char* location = array;
    location += i * M * sz;
    location += j * sz;

    memcpy(location, arg, sz);
    
    return 0;

}

// Fetch value at row and col
void* two_d_fetch(char* array, size_t i, size_t j, size_t M, size_t N, size_t sz){

    if (array == NULL || i > N || i < 0 || j > M || j < 0) {
        printf("Error, inputs are inccorect ");
        return NULL;
    }

    printf("Want to fetch value at row: %zu, col: %zu\n", i, j);

    char* location = array;
    location += i * M * sz;
    location += j * sz;
    return location;
}


// Fill array with 1,2,3,4.....
void fillArray(char* arr, size_t N, size_t M, size_t sz){
    int count = 0;
    for (int i = 0; i < (N * M * sz); i+=sz)
    arr[i] = ++count;

}

// Print array with bytes 
void printArray(char* array, size_t N, size_t M, size_t sz){
    for (int i = 0; i < N ; i++) {
        for (int j = 0; j < M * sz ; j++){
            printf("%hx ", array[i * (M*sz) + j]);
        }
        
    printf("\n");
    }
}

void printSmallArray(char* array, size_t N, size_t M, size_t sz){
    for (int i = 0; i < N ; i++) {
        for (int j = 0; j < M; j++){
            
            printf("%hx ", array[i * (M*sz) + (j*sz)]);
        }
        
    printf("\n");
    
    }
}

