

#include "PETSc_Vector.h"


extern int ESI_VectorPointerAccess_test(ESI_VectorPointerAccess *);
int main(int argc,char **args)
{
  int    ierr;
  double norm,dot;

  PetscInitialize(&argc,&args,0,0);

  PETSc_Map *map = new PETSc_Map(MPI_COMM_WORLD,5,PETSC_DECIDE);

  MPI_Comm comm;
  map->getCommunicator(&comm);
  int rank;
  MPI_Comm_rank(comm,&rank);


  PETSc_Vector *vector = new PETSc_Vector((ESI_MapAlgebraic *)map);

  ierr = ESI_VectorPointerAccess_test((ESI_VectorPointerAccess *)vector);
  if (ierr) {printf("Error in ESI_Vector_Test()\n");return ierr;}

  const ESI_Map *gmap; vector->getMap(gmap);

  delete map;

  vector->put(3.0);
  vector->scale(4.2);
  vector->norm1(norm);
  vector->dot(*vector,dot);

  PetscPrintf(comm,"norm %g dot %g\n",norm,dot);

  double *array; int silly;

  vector->getArrayPointer(array,silly);
  array[0] = 22.3;
  vector->restoreArrayPointer(array,silly);
  vector->norm1(norm);
  vector->dot(*vector,dot);

  PetscPrintf(comm,"norm %g dot %g\n",norm,dot);

  delete vector;
  PetscFinalize();

  return 0;
}
