static char help[]= "  Test VecScatter with x, y on different communicators\n\n";

#include <petscvec.h>

int main(int argc,char **argv)
{
  PetscErrorCode     ierr;
  PetscInt           i,n=5,rstart;
  PetscScalar        *val;
  const PetscScalar  *dat;
  Vec                x,y1,y2;
  MPI_Comm           newcomm;
  PetscMPIInt        nproc,rank;
  IS                 ix;
  VecScatter         vscat1,vscat2;

  PetscFunctionBegin;
  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&nproc));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));

  PetscCheckFalse(nproc != 2,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"This test must run with exactly two MPI ranks");

  /* Create MPI vectors x and y, which are on the same comm (i.e., MPI_IDENT) */
  CHKERRQ(VecCreateMPI(PETSC_COMM_WORLD,n,PETSC_DECIDE,&x));
  CHKERRQ(VecDuplicate(x,&y1));
  CHKERRQ(VecGetOwnershipRange(x,&rstart,NULL));

  /* Set x's value locally. x would be {0., 1., 2., ..., 9.} */
  CHKERRQ(VecGetArray(x,&val));
  for (i=0; i<n; i++) val[i] = rstart + i;
  CHKERRQ(VecRestoreArray(x,&val));

  /* Create index set ix = {0, 1, 2, ..., 9} */
  CHKERRQ(ISCreateStride(PETSC_COMM_WORLD,n,rstart,1,&ix));

  /* Create newcomm that reverses processes in x's comm, and then create y2 on it*/
  CHKERRMPI(MPI_Comm_split(PETSC_COMM_WORLD,0/*color*/,-rank/*key*/,&newcomm));
  CHKERRQ(VecCreateMPI(newcomm,n,PETSC_DECIDE,&y2));

  /* It looks vscat1/2 are the same, but actually not. y2 is on a different communicator than x */
  CHKERRQ(VecScatterCreate(x,ix,y1,ix,&vscat1));
  CHKERRQ(VecScatterCreate(x,ix,y2,ix,&vscat2));

  CHKERRQ(VecScatterBegin(vscat1,x,y1,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterBegin(vscat2,x,y2,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd  (vscat1,x,y1,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd  (vscat2,x,y2,INSERT_VALUES,SCATTER_FORWARD));

  /* View on rank 0 of x's comm, which is PETSC_COMM_WORLD */
  if (rank == 0) {
    /* Print the part of x on rank 0, which is 0 1 2 3 4 */
    CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"\nOn rank 0 of PETSC_COMM_WORLD, x  = "));
    CHKERRQ(VecGetArrayRead(x,&dat));
    for (i=0; i<n; i++) CHKERRQ(PetscPrintf(PETSC_COMM_SELF," %.0g",(double)PetscRealPart(dat[i])));
    CHKERRQ(VecRestoreArrayRead(x,&dat));

    /* Print the part of y1 on rank 0, which is 0 1 2 3 4 */
    CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"\nOn rank 0 of PETSC_COMM_WORLD, y1 = "));
    CHKERRQ(VecGetArrayRead(y1,&dat));
    for (i=0; i<n; i++) CHKERRQ(PetscPrintf(PETSC_COMM_SELF," %.0g",(double)PetscRealPart(dat[i])));
    CHKERRQ(VecRestoreArrayRead(y1,&dat));

    /* Print the part of y2 on rank 0, which is 5 6 7 8 9 since y2 swapped the processes of PETSC_COMM_WORLD */
    CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"\nOn rank 0 of PETSC_COMM_WORLD, y2 = "));
    CHKERRQ(VecGetArrayRead(y2,&dat));
    for (i=0; i<n; i++) CHKERRQ(PetscPrintf(PETSC_COMM_SELF," %.0g",(double)PetscRealPart(dat[i])));
    CHKERRQ(VecRestoreArrayRead(y2,&dat));
    CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"\n"));
  }

  CHKERRQ(ISDestroy(&ix));
  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&y1));
  CHKERRQ(VecDestroy(&y2));
  CHKERRQ(VecScatterDestroy(&vscat1));
  CHKERRQ(VecScatterDestroy(&vscat2));
  CHKERRMPI(MPI_Comm_free(&newcomm));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      nsize: 2
      requires: double
TEST*/
