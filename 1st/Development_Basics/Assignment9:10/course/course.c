/**
 * @file course.c
 * @author Wyatt Habinski
 * @date 2022-04-12
 * @brief course library for managing courses, incluing definitions of course functions
 * 
 */

#include "course.h"
#include <stdlib.h>
#include <stdio.h>
 
/**
 * Adds a student to the course. Updates total number of students in course
 * Uses calloc if its the first student in the course
 * Otherwise uses realloc
 * 
 * @param course 
 * @param student 
 * @return nothing
 */
void enroll_student(Course *course, Student *student)
{
  course->total_students++;
  if (course->total_students == 1) 
  {
    course->students = calloc(1, sizeof(Student));
  }
  else 
  {
    course->students = 
      realloc(course->students, course->total_students * sizeof(Student)); 
  }
  course->students[course->total_students - 1] = *student;
}

/**
 * Prints the course name, code, total number of students in the course, and prints each student in the course
 * 
 * @param course 
 * @return nothing
 */
void print_course(Course* course)
{
  printf("Name: %s\n", course->name);
  printf("Code: %s\n", course->code);
  printf("Total students: %d\n\n", course->total_students);
  printf("****************************************\n\n");
  for (int i = 0; i < course->total_students; i++) 
    print_student(&course->students[i]);
}

/**
 * Returns the student with the highest grade average in the course
 * 
 * @param course 
 * @return Student* 
 */
Student* top_student(Course* course)
{
  if (course->total_students == 0) return NULL;
 
  double student_average = 0;
  double max_average = average(&course->students[0]);
  Student *student = &course->students[0];
 
  for (int i = 1; i < course->total_students; i++) /**< loops through all students in the course */
  {
    student_average = average(&course->students[i]);
    if (student_average > max_average) 
    {
      max_average = student_average;
      student = &course->students[i];
    }   
  }

  return student;
}

/**
 * Returns an array of students that are passing the course
 * Also updates the number of total students passing the course
 * 
 * @param course 
 * @param total_passing 
 * @return Student* 
 */
Student *passing(Course* course, int *total_passing)
{
  int count = 0;
  Student *passing = NULL;
  
  for (int i = 0; i < course->total_students; i++)  /**< loops through all student in the course*/
    if (average(&course->students[i]) >= 50) count++;
  
  passing = calloc(count, sizeof(Student));

  int j = 0;
  for (int i = 0; i < course->total_students; i++) /**< loops through all student in the course*/
  {
    if (average(&course->students[i]) >= 50)
    {
      passing[j] = course->students[i];
      j++; 
    }
  }

  *total_passing = count;

  return passing;
}