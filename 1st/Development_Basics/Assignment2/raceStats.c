#include <stdio.h>


int main(){

int laps;
float distance;

// Collecting input for distance of track
do{
printf("Enter the distance of the track: ");
scanf("%f", &distance);
if(distance <= 0) printf("The value of distance must be greater than 0\n");
}while(distance <= 0);

// Collecting input for number of laps
do{
printf("Enter the number of laps: ");
scanf("%d", &laps);
if(laps <= 0) printf("The value of number of laps must greater than 0\n");
}while(laps <= 0);

// Table Header
printf("%-10s%-20s%-20s%-20s\n", "# of laps", "Distance", "Speed", "Time");
printf("******************************************************\n");

float speed[laps],time[laps];

// Collecting input for speed of each lap
for (int i = 0; i < laps; i++){
    do{
    printf("Enter the speed of lap %d: ", (i + 1));
    scanf("%f", &speed[i]);
    if(speed[i] <= 0) printf("The value of speed must be greater than 0\n");
    // Output data to table
    else {
    time[i] = distance/speed[i];
    printf("\n%-10d%-20.2f%-20.2f%-20.2f\n",(i+1),distance,speed[i],time[i]);
    }
    }while(speed[i] <= 0);
}

// Output Toal data line
float totalLaps = distance * laps;
float totalSpeed = 0, totalTime = 0;

for (int i = 0; i < laps; i++){
totalSpeed += speed[i];
totalTime += time[i];
}

printf("\n%-10s%-20.2f%-20.2f%-20.2f","Total:",totalLaps,(totalSpeed / laps),totalTime);
return 0;
}
