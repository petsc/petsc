
/*
       Tests the ESI_VectorPointerAccess interface
*/
#include "ESI.h"

extern int ESI_Vector_test(ESI_Vector *);

int ESI_VectorPointerAccess_test(ESI_VectorPointerAccess *vector)
{
  int    ierr,localsize,i;
  double *values,norm2;

  ierr = ESI_Vector_test((ESI_Vector*)vector);
  if (ierr) {printf("error calling ESI_Vector_test()\n");return ierr;}

  ierr = vector->getArrayPointer(values,localsize);
  if (ierr) {printf("error calling ->getArrayPointer()\n");return ierr;}

  for (i=0; i<localsize; i++) {
    values[i] = (double) i;
  }

  ierr = vector->norm2(norm2);
  if (ierr) {printf("error calling vector->norm2() \n");return ierr;}
  printf("ESI_VectorPointerAccess_test: norm2 %g\n",norm2);

  ierr = vector->restoreArrayPointer(values,localsize);
  if (ierr) {printf("error calling ->restoreArrayPointer()\n");return ierr;}

  return 0;
}







