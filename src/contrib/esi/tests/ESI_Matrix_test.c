
/*
       Tests the ESI_Matrix interface
*/
#include "ESI.h"

extern int ESI_Object_test(ESI_Object *);

int ESI_Matrix_test(ESI_Matrix *matrix,ESI_Vector *vector,ESI_Vector *bvector)
{
  int            ierr;

  ierr = ESI_Object_test((ESI_Object*)matrix); 
  if (ierr) {printf("error calling ESI_Object_test()\n");return ierr;}

  ierr = matrix->setup();
  if (ierr) {printf("error calling matrix->setup() \n");return ierr;}

  ierr = matrix->matvec(*vector,*bvector);
  if (ierr) {printf("error calling matrix->matvec() \n");return ierr;}

  ierr = matrix->apply(*vector,*bvector);
  if (ierr) {printf("error calling matrix->apply() \n");return ierr;}

  return 0;
}







