#include "stdio.h"
#include "petsc.h"
#include "pinclude/ptime.h"

int main( int argc, char **argv)
{
  double x, y;
  
  PetscInitialize(&argc, &argv,0,0,0);

   PetscTime(x);
   PetscTime(y);
   PetscTime(y);
   PetscTime(y);
   PetscTime(y);
   PetscTime(y);
   PetscTime(y);
   PetscTime(y);
   PetscTime(y);
   PetscTime(y);
   PetscTime(y);

  fprintf(stderr,"%-15s : %e sec\n","PetscTime",(y-x)/10.0);
  PetscFinalize();
  return 0;
}
