
static char help[] = "Tests the routines VecScatterCreateToAll(), VecScatterCreateToZero()\n\n";

#include <petscvec.h>

int main(int argc,char **argv)
{
  PetscInt       n = 3,i,len,start,end;
  PetscMPIInt    size,rank;
  PetscScalar    value,*yy;
  Vec            x,y,z,y_t;
  VecScatter     toall,tozero;

  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));

  /* create two vectors */
  PetscCall(VecCreateMPI(PETSC_COMM_WORLD,PETSC_DECIDE,size*n,&x));

  /* each processor inserts its values */

  PetscCall(VecGetOwnershipRange(x,&start,&end));
  for (i=start; i<end; i++) {
    value = (PetscScalar) i;
    PetscCall(VecSetValues(x,1,&i,&value,INSERT_VALUES));
  }
  PetscCall(VecAssemblyBegin(x));
  PetscCall(VecAssemblyEnd(x));
  PetscCall(VecView(x,PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(VecScatterCreateToAll(x,&toall,&y));
  PetscCall(VecScatterBegin(toall,x,y,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(VecScatterEnd(toall,x,y,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(VecScatterDestroy(&toall));

  /* Cannot view the above vector with VecView(), so place it in an MPI Vec
     and do a VecView() */
  PetscCall(VecGetArray(y,&yy));
  PetscCall(VecGetLocalSize(y,&len));
  PetscCall(VecCreateMPIWithArray(PETSC_COMM_WORLD,1,len,PETSC_DECIDE,yy,&y_t));
  PetscCall(VecView(y_t,PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(VecDestroy(&y_t));
  PetscCall(VecRestoreArray(y,&yy));

  PetscCall(VecScatterCreateToAll(x,&tozero,&z));
  PetscCall(VecScatterBegin(tozero,x,z,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(VecScatterEnd(tozero,x,z,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(VecScatterDestroy(&tozero));
  if (rank == 0) {
    PetscCall(VecView(z,PETSC_VIEWER_STDOUT_SELF));
  }
  PetscCall(VecDestroy(&z));

  PetscCall(VecScatterCreateToZero(x,&tozero,&z));
  PetscCall(VecScatterBegin(tozero,x,z,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(VecScatterEnd(tozero,x,z,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(VecScatterDestroy(&tozero));
  PetscCall(VecDestroy(&z));

  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&y));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      nsize: 4

TEST*/
