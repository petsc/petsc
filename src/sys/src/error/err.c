
#include "petsc.h"
#include <stdio.h>

int PetscErrorHandler(line,file,message,number)
int line,number;
char *file, *message;
{
  fprintf(stderr,"%s %d %s %d\n",file,line,message,number);
  return number;
}
