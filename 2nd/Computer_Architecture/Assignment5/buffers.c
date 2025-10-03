#include <stdio.h> 
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>

const int buffer_size = 4;

char fname_in[] = "test_in.txt";
int fd_in = -1;
char buffer_in[buffer_size];
int in_pos = 0;
bool eof = false;

char fname_out[] = "test_out.txt";
int fd_out = -1;
char buffer_out[buffer_size];
int out_pos = 0;


// Function to read a character from the file using a buffer
char buf_in() {

    // If the buffer is empty, read in 16 bytes from the file
    if (in_pos == 0) {
        read(fd_in, buffer_in, buffer_size);
    }

    // If the buffer isn't empty
    if (in_pos != 0) {
        char temp[1];
        if(pread(fd_in, temp, 1*sizeof(char), in_pos + (buffer_size - 1)) == 0){
            eof = true;
        }
        buffer_in[(in_pos - 1) % buffer_size] = temp[0]; 
    }

    // Return the next byte (character) from the buffer and update the read position
    char ch = buffer_in[in_pos % buffer_size];
    in_pos += 1;
    return ch;
}

// Helper function for buf_out
void buf_flush(){
    write(fd_out, buffer_out, buffer_size);
}

// Function to write a character to a file using a buffer
void buf_out(char data){
    
    if (out_pos == buffer_size){
        buf_flush();
        out_pos = 0;

        memset(buffer_out, 0, buffer_size);
        //for (int i = 0; i < strlen(buffer_out); i ++)
    }

    buffer_out[out_pos] = data;
    out_pos += 1;

}

// Function to test buf_in
void testIn(){
    for (int i = 0; i < 1024; i++) printf("%c", buf_in());
}

// Function to test buf_out
void testOut(){
    char str[] = "This is a test for writing to a file using a buffer";
    int len = strlen(str);
    
    for (int i = 0; i < len; i++){
        buf_out(str[i]);
    }

    buf_flush();
}

void readAndWrite(){
    for (int i = 0; i < 1024; i++){  
      buf_out(buf_in());  
      if (eof == true){
        for(int i = 0; i < (buffer_size - 2); i++)
        buf_out(buf_in());
        buf_flush();
        return;
      }
    }
}

// Main Function
int main() {

    printf("----------------------------------------------------\nRead a file, and copy contents to a destination file\n----------------------------------------------------\n");
    printf("Please enter the name of a file you want to copy from: ");
    scanf("%s", fname_in);
    printf("Please enter the name of a destination file you want to copy to: ");
    scanf("%s", fname_out);

    // Open the file for reading / writing
    fd_in = open(fname_in, O_RDONLY);
    fd_out = open(fname_out, O_WRONLY);

    // Methods for testing functions
    // TASK 2.1
    //testIn();
    // TASK 2.2
    //testOut();

    // USE THIS METHOD to read from a file and copy to a destination file
    // TASK 2.3
    readAndWrite();

    // Close the file descriptors
    close(fd_in);
    close(fd_out);
    return 0;
}