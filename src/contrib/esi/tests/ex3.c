
#include "petsc/matrix.h"

extern int ESI_Matrix_test(esi::Operator<double,int> *,esi::Vector<double,int> *,esi::Vector<double,int>*);

int main(int argc,char **args)
{
  int    ierr;

  PetscInitialize(&argc,&args,0,0);

  esi::petsc::Map<int>    *map     = new esi::petsc::Map<int>(MPI_COMM_WORLD,5,PETSC_DECIDE);
  esi::petsc::Vector<double,int> *vector  = new esi::petsc::Vector<double,int>((esi::MapPartition<int> *)map);
  esi::petsc::Vector<double,int> *bvector = new esi::petsc::Vector<double,int>((esi::MapPartition<int> *)map);
  esi::petsc::Matrix<double,int> *matrix  = new esi::petsc::Matrix<double,int>((esi::MapPartition<int> *)map,(esi::MapPartition<int> *)map);

  ierr = ESI_Matrix_test(matrix,vector,bvector);
  if (ierr) {printf("error calling ESI_Matrix_test()\n");return ierr;}

  
  delete matrix;
  delete vector;
  delete bvector;
  delete map;
  PetscFinalize();

  return 0;
}
