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

  CHKERRQ(PetscInitialize(&argc,&argv,(char*)0,help));
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&nproc));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  PetscCheckFalse(nproc != 2,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"This test can only run on two MPI ranks");

  /* Create an MPI vector x of size 12 on two processes, and set x = {0, 1, 2, .., 11} */
  CHKERRQ(VecCreateMPI(PETSC_COMM_WORLD,6,PETSC_DECIDE,&x));
  CHKERRQ(VecGetOwnershipRange(x,&low,&high));
  for (i=low; i<high; i++) CHKERRQ(VecSetValue(x,i,(PetscScalar)i,INSERT_VALUES));
  CHKERRQ(VecAssemblyBegin(x));
  CHKERRQ(VecAssemblyEnd(x));

  /* Create a seq vector y, and a parallel to sequential (PtoS) vecscatter to scatter x to y */
  if (rank == 0) {
    /* On rank 0, seq y is of size 6. We will scatter x[0,1,2,6,7,8] to y[0,1,2,3,4,5] using IS with bs=3 */
    PetscInt idx[2]={0,2};
    PetscInt idy[2]={0,1};
    n    = 6;
    bs   = 3;
    CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,n,&y));
    CHKERRQ(ISCreateBlock(PETSC_COMM_SELF,bs,2,idx,PETSC_COPY_VALUES,&ix));
    CHKERRQ(ISCreateBlock(PETSC_COMM_SELF,bs,2,idy,PETSC_COPY_VALUES,&iy));
  } else {
    /* On rank 1, seq y is of size 4. We will scatter x[4,5,10,11] to y[0,1,2,3] using IS with bs=2 */
    PetscInt idx[2]= {2,5};
    PetscInt idy[2]= {0,1};
    n    = 4;
    bs   = 2;
    CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,n,&y));
    CHKERRQ(ISCreateBlock(PETSC_COMM_SELF,bs,2,idx,PETSC_COPY_VALUES,&ix));
    CHKERRQ(ISCreateBlock(PETSC_COMM_SELF,bs,2,idy,PETSC_COPY_VALUES,&iy));
  }
  CHKERRQ(VecScatterCreate(x,ix,y,iy,&vscat));

  /* Do the vecscatter */
  CHKERRQ(VecScatterBegin(vscat,x,y,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(vscat,x,y,INSERT_VALUES,SCATTER_FORWARD));

  /* Print y. Since y is sequential, we put y in a parallel z to print its value on both ranks */
  CHKERRQ(VecGetArrayRead(y,&yv));
  CHKERRQ(VecCreateMPIWithArray(PETSC_COMM_WORLD,1,n,PETSC_DECIDE,yv,&z));
  CHKERRQ(VecView(z,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(VecRestoreArrayRead(y,&yv));

  CHKERRQ(ISDestroy(&ix));
  CHKERRQ(ISDestroy(&iy));
  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&y));
  CHKERRQ(VecDestroy(&z));
  CHKERRQ(VecScatterDestroy(&vscat));

  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   test:
      nsize: 2
      args:
      requires:
TEST*/
