#include <stdio.h>
#include <string.h>
#include <stdlib.h>

// HELPER FUNCTIONS
void inputHouse();
void printHouses();
void yesOrNo();
void selectionSort();

// INITIALIZE HOUSE TYPE
typedef struct house{
char city[50];
int price;
}House;

// INITIALIZE ARRAY OF TYPE HOUSE
House *houses;
int numHouses = 0; // number of houses in array counter

int main(){

    // Allocate 1 element to array
    houses = (House *) malloc(sizeof(House));
    printf("*********** HOUSING ***********\n");

    // Function to add another house
    inputHouse();
    
    // Frees the dynamically allocated array
    free(houses);

    return 0;
}

// Function that takes in user input and adds house element to array
void inputHouse(){

    printf("\nEnter the name of the city: \n");
    scanf("%s", houses[numHouses].city);
    printf("Enter the average house price: \n");
    scanf("%d", &houses[numHouses].price);

    numHouses += 1;

    // Calls function to determine if the user wants to add another house or not
    yesOrNo();

}

// Function that asks the user if they would like to add another house
// If yes, then inputHouse() is called again, otherwise the program moves on to printing the data
void yesOrNo(){

    char anotherCity[1];
    printf("Enter another City? (y/n): ");
    scanf("%s",anotherCity);

    int val = strcmp(anotherCity, "y");
    if (val == 0) {
        inputHouse();
        houses = (House *)realloc(houses, sizeof(House)); // If another House data is going to be inputed, the array must reallocate another element of House
    }
    // Function that prints the data
    else printHouses();

}

// Function that loops through the array and prints all unsorted data
void printHouses(){

    printf("\n\nList of Cities and Prices: \n");

    for (int i = 0; i < numHouses; i ++){
        printf("\nCity: %s Price: %d", houses[i].city, houses[i].price); 
    }

    selectionSort();
}

// Function that sorts the house array based on housing price from lowest to highest, then prints the data again
void selectionSort(){

    int smallest;

    for (int i = 0; i < (numHouses - 1); i++){
        smallest = i;

        for (int j = i+1; j < numHouses; j++){
            if (houses[j].price < houses[smallest].price){
                House temp = houses[j];
                houses[j] = houses[smallest];
                houses[smallest] = temp;
            }
        }
    }

    printf("\n\nSorted: \n");
    for (int i = 0; i < numHouses; i ++){
        printf("\nCity: %s Price: %d", houses[i].city, houses[i].price); 
    }

    printf("\n\nThe city with the highest house price: %s", houses[numHouses - 1].city);

}
