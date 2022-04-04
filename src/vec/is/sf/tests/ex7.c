static char help[]= "Test vecscatter of different block sizes across processes\n\n";

#include <petscvec.h>
int main(int argc,char **argv)
{
  PetscInt           i,bs,n,low,high;
  PetscMPIInt        nproc,rank;
  Vec                x,y,z;
  IS                 ix,iy;
  VecScatter         vscat;
  const PetscScalar  *yv;

  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&nproc));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  PetscCheck(nproc == 2,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"This test can only run on two MPI ranks");

  /* Create an MPI vector x of size 12 on two processes, and set x = {0, 1, 2, .., 11} */
  PetscCall(VecCreateMPI(PETSC_COMM_WORLD,6,PETSC_DECIDE,&x));
  PetscCall(VecGetOwnershipRange(x,&low,&high));
  for (i=low; i<high; i++) PetscCall(VecSetValue(x,i,(PetscScalar)i,INSERT_VALUES));
  PetscCall(VecAssemblyBegin(x));
  PetscCall(VecAssemblyEnd(x));

  /* Create a seq vector y, and a parallel to sequential (PtoS) vecscatter to scatter x to y */
  if (rank == 0) {
    /* On rank 0, seq y is of size 6. We will scatter x[0,1,2,6,7,8] to y[0,1,2,3,4,5] using IS with bs=3 */
    PetscInt idx[2]={0,2};
    PetscInt idy[2]={0,1};
    n    = 6;
    bs   = 3;
    PetscCall(VecCreateSeq(PETSC_COMM_SELF,n,&y));
    PetscCall(ISCreateBlock(PETSC_COMM_SELF,bs,2,idx,PETSC_COPY_VALUES,&ix));
    PetscCall(ISCreateBlock(PETSC_COMM_SELF,bs,2,idy,PETSC_COPY_VALUES,&iy));
  } else {
    /* On rank 1, seq y is of size 4. We will scatter x[4,5,10,11] to y[0,1,2,3] using IS with bs=2 */
    PetscInt idx[2]= {2,5};
    PetscInt idy[2]= {0,1};
    n    = 4;
    bs   = 2;
    PetscCall(VecCreateSeq(PETSC_COMM_SELF,n,&y));
    PetscCall(ISCreateBlock(PETSC_COMM_SELF,bs,2,idx,PETSC_COPY_VALUES,&ix));
    PetscCall(ISCreateBlock(PETSC_COMM_SELF,bs,2,idy,PETSC_COPY_VALUES,&iy));
  }
  PetscCall(VecScatterCreate(x,ix,y,iy,&vscat));

  /* Do the vecscatter */
  PetscCall(VecScatterBegin(vscat,x,y,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(VecScatterEnd(vscat,x,y,INSERT_VALUES,SCATTER_FORWARD));

  /* Print y. Since y is sequential, we put y in a parallel z to print its value on both ranks */
  PetscCall(VecGetArrayRead(y,&yv));
  PetscCall(VecCreateMPIWithArray(PETSC_COMM_WORLD,1,n,PETSC_DECIDE,yv,&z));
  PetscCall(VecView(z,PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(VecRestoreArrayRead(y,&yv));

  PetscCall(ISDestroy(&ix));
  PetscCall(ISDestroy(&iy));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&y));
  PetscCall(VecDestroy(&z));
  PetscCall(VecScatterDestroy(&vscat));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      nsize: 2
      args:
      requires:
TEST*/
