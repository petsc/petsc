
#include "PETSc_Matrix.h"

extern int ESI_Matrix_test(ESI_Matrix *,ESI_Vector *,ESI_Vector*);

int main(int argc,char **args)
{
  int    ierr;

  PetscInitialize(&argc,&args,0,0);

  PETSc_Map    *map     = new PETSc_Map(MPI_COMM_WORLD,5,PETSC_DECIDE);
  PETSc_Vector *vector  = new PETSc_Vector((ESI_MapAlgebraic *)map);
  PETSc_Vector *bvector = new PETSc_Vector((ESI_MapAlgebraic *)map);
  PETSc_Matrix *matrix  = new PETSc_Matrix((ESI_MapAlgebraic *)map,(ESI_MapAlgebraic *)map);

  ierr = ESI_Matrix_test(matrix,vector,bvector);
  if (ierr) {printf("error calling ESI_Matrix_test()\n");return ierr;}

  
  delete matrix;
  delete vector;
  delete bvector;
  delete map;
  PetscFinalize();

  return 0;
}
