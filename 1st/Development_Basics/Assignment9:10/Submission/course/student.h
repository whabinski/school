 /**
  * @file student.h
  * @author Wyatt Habinski
  * @date 2022-04-12
  * @brief Student library for managing students. Contains student type definition
  *         student functions.
  */

/**
 * Student type stores student's firstname, lastname, ID, grades, and number of grades
 * 
 */
typedef struct _student 
{ 
  char first_name[50]; /**< the student's first name */
  char last_name[50]; /**< the student's last name */
  char id[11]; /**< the studen's ID number */
  double *grades; /**< a pointer to the students grades */
  int num_grades; /**< the number of grades the student has */
} Student;

void add_grade(Student *student, double grade);
double average(Student *student);
void print_student(Student *student);
Student* generate_random_student(int grades); 
