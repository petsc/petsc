
#include "esi/petsc/matrix.h"

extern int ESI_Matrix_test(esi::Operator<double,int> *,esi::Vector<double,int> *,esi::Vector<double,int>*);

int main(int argc,char **args)
{
  int    ierr;

  PetscInitialize(&argc,&args,0,0);

  esi::petsc::IndexSpace<int>    *map     = new esi::petsc::IndexSpace<int>(MPI_COMM_WORLD,5,PETSC_DECIDE);
  esi::petsc::Vector<double,int> *vector  = new esi::petsc::Vector<double,int>(map);
  esi::petsc::Vector<double,int> *bvector = new esi::petsc::Vector<double,int>(map);
  esi::petsc::Matrix<double,int> *matrix  = new esi::petsc::Matrix<double,int>(map,map);

  ierr = ESI_Matrix_test(matrix,vector,bvector);
  if (ierr) {printf("error calling ESI_Matrix_test()\n");return ierr;}

  
  delete matrix;
  delete vector;
  delete bvector;
  delete map;
  PetscFinalize();

  return 0;
}
