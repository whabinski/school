/*******************************************************************************
*
* Purpose: Square area calculator.  Outputs calculated square areas from
* side length provided via standard input... outputs them with a unit
* provided as 2nd argv values.  i.e. if we run with ""./square inches" expect
* output of the format: 50 inches
*
*******************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include "library.h"
 
int main(int argc, char *argv[])
{
 int m = 0;
 
 // if argument number is incorrect output this notice
 if (argc != 2)
 {
   printf("Incorrect number of arguments provided.\nPlease enter the corresponding units");
   return 0;
 }
 
 // read in ints one at a time and multiply by the 2nd argv value
 while (scanf("%d", &m) != EOF)
 {
   printf("%d %s\n", square_area(m), argv[1]);
 }
 
 return 0;
}
