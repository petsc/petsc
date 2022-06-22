static char help[]= "  Test VecScatter with x, y on different communicators\n\n";

#include <petscvec.h>

int main(int argc,char **argv)
{
  PetscInt           i,n=5,rstart;
  PetscScalar        *val;
  const PetscScalar  *dat;
  Vec                x,y1,y2;
  MPI_Comm           newcomm;
  PetscMPIInt        nproc,rank;
  IS                 ix;
  VecScatter         vscat1,vscat2;

  PetscFunctionBegin;
  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&nproc));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));

  PetscCheck(nproc == 2,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"This test must run with exactly two MPI ranks");

  /* Create MPI vectors x and y, which are on the same comm (i.e., MPI_IDENT) */
  PetscCall(VecCreateMPI(PETSC_COMM_WORLD,n,PETSC_DECIDE,&x));
  PetscCall(VecDuplicate(x,&y1));
  PetscCall(VecGetOwnershipRange(x,&rstart,NULL));

  /* Set x's value locally. x would be {0., 1., 2., ..., 9.} */
  PetscCall(VecGetArray(x,&val));
  for (i=0; i<n; i++) val[i] = rstart + i;
  PetscCall(VecRestoreArray(x,&val));

  /* Create index set ix = {0, 1, 2, ..., 9} */
  PetscCall(ISCreateStride(PETSC_COMM_WORLD,n,rstart,1,&ix));

  /* Create newcomm that reverses processes in x's comm, and then create y2 on it*/
  PetscCallMPI(MPI_Comm_split(PETSC_COMM_WORLD,0/*color*/,-rank/*key*/,&newcomm));
  PetscCall(VecCreateMPI(newcomm,n,PETSC_DECIDE,&y2));

  /* It looks vscat1/2 are the same, but actually not. y2 is on a different communicator than x */
  PetscCall(VecScatterCreate(x,ix,y1,ix,&vscat1));
  PetscCall(VecScatterCreate(x,ix,y2,ix,&vscat2));

  PetscCall(VecScatterBegin(vscat1,x,y1,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(VecScatterBegin(vscat2,x,y2,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(VecScatterEnd  (vscat1,x,y1,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(VecScatterEnd  (vscat2,x,y2,INSERT_VALUES,SCATTER_FORWARD));

  /* View on rank 0 of x's comm, which is PETSC_COMM_WORLD */
  if (rank == 0) {
    /* Print the part of x on rank 0, which is 0 1 2 3 4 */
    PetscCall(PetscPrintf(PETSC_COMM_SELF,"\nOn rank 0 of PETSC_COMM_WORLD, x  = "));
    PetscCall(VecGetArrayRead(x,&dat));
    for (i=0; i<n; i++) PetscCall(PetscPrintf(PETSC_COMM_SELF," %.0g",(double)PetscRealPart(dat[i])));
    PetscCall(VecRestoreArrayRead(x,&dat));

    /* Print the part of y1 on rank 0, which is 0 1 2 3 4 */
    PetscCall(PetscPrintf(PETSC_COMM_SELF,"\nOn rank 0 of PETSC_COMM_WORLD, y1 = "));
    PetscCall(VecGetArrayRead(y1,&dat));
    for (i=0; i<n; i++) PetscCall(PetscPrintf(PETSC_COMM_SELF," %.0g",(double)PetscRealPart(dat[i])));
    PetscCall(VecRestoreArrayRead(y1,&dat));

    /* Print the part of y2 on rank 0, which is 5 6 7 8 9 since y2 swapped the processes of PETSC_COMM_WORLD */
    PetscCall(PetscPrintf(PETSC_COMM_SELF,"\nOn rank 0 of PETSC_COMM_WORLD, y2 = "));
    PetscCall(VecGetArrayRead(y2,&dat));
    for (i=0; i<n; i++) PetscCall(PetscPrintf(PETSC_COMM_SELF," %.0g",(double)PetscRealPart(dat[i])));
    PetscCall(VecRestoreArrayRead(y2,&dat));
    PetscCall(PetscPrintf(PETSC_COMM_SELF,"\n"));
  }

  PetscCall(ISDestroy(&ix));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&y1));
  PetscCall(VecDestroy(&y2));
  PetscCall(VecScatterDestroy(&vscat1));
  PetscCall(VecScatterDestroy(&vscat2));
  PetscCallMPI(MPI_Comm_free(&newcomm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      nsize: 2
      requires: double
TEST*/
