/**
 * @file course.h
 * @author Wyatt Habinski
 * @date 2022-04-12
 * @brief Course library for managing courses. Including course type definition and course functions
 * 
 */


#include "student.h"
#include <stdbool.h>
 
/**
 * Course type that stores course name, course code, students in the course, 
 * and total number of students in the course
 * 
 */
typedef struct _course 
{
  char name[100];
  char code[10];
  Student *students;
  int total_students;
} Course;

void enroll_student(Course *course, Student *student);
void print_course(Course *course);
Student *top_student(Course* course);
Student *passing(Course* course, int *total_passing);


