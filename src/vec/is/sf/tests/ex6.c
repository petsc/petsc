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
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&nproc);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);

  if (nproc != 2) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"This test must run with exactly two MPI ranks\n");

  /* Create MPI vectors x and y, which are on the same comm (i.e., MPI_IDENT) */
  ierr = VecCreateMPI(PETSC_COMM_WORLD,n,PETSC_DECIDE,&x);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&y1);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(x,&rstart,NULL);CHKERRQ(ierr);

  /* Set x's value locally. x would be {0., 1., 2., ..., 9.} */
  ierr = VecGetArray(x,&val);CHKERRQ(ierr);
  for (i=0; i<n; i++) val[i] = rstart + i;
  ierr = VecRestoreArray(x,&val);CHKERRQ(ierr);

  /* Create index set ix = {0, 1, 2, ..., 9} */
  ierr = ISCreateStride(PETSC_COMM_WORLD,n,rstart,1,&ix);CHKERRQ(ierr);

  /* Create newcomm that reverses processes in x's comm, and then create y2 on it*/
  ierr = MPI_Comm_split(PETSC_COMM_WORLD,0/*color*/,-rank/*key*/,&newcomm);CHKERRMPI(ierr);
  ierr = VecCreateMPI(newcomm,n,PETSC_DECIDE,&y2);CHKERRQ(ierr);

  /* It looks vscat1/2 are the same, but actually not. y2 is on a different communicator than x */
  ierr = VecScatterCreate(x,ix,y1,ix,&vscat1);CHKERRQ(ierr);
  ierr = VecScatterCreate(x,ix,y2,ix,&vscat2);CHKERRQ(ierr);

  ierr = VecScatterBegin(vscat1,x,y1,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterBegin(vscat2,x,y2,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd  (vscat1,x,y1,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd  (vscat2,x,y2,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

  /* View on rank 0 of x's comm, which is PETSC_COMM_WORLD */
  if (rank == 0) {
    /* Print the part of x on rank 0, which is 0 1 2 3 4 */
    ierr = PetscPrintf(PETSC_COMM_SELF,"\nOn rank 0 of PETSC_COMM_WORLD, x  = ");CHKERRQ(ierr);
    ierr = VecGetArrayRead(x,&dat);CHKERRQ(ierr);
    for (i=0; i<n; i++) {ierr = PetscPrintf(PETSC_COMM_SELF," %.0g",(double)PetscRealPart(dat[i]));CHKERRQ(ierr);}
    ierr = VecRestoreArrayRead(x,&dat);CHKERRQ(ierr);

    /* Print the part of y1 on rank 0, which is 0 1 2 3 4 */
    ierr = PetscPrintf(PETSC_COMM_SELF,"\nOn rank 0 of PETSC_COMM_WORLD, y1 = ");CHKERRQ(ierr);
    ierr = VecGetArrayRead(y1,&dat);CHKERRQ(ierr);
    for (i=0; i<n; i++) {ierr = PetscPrintf(PETSC_COMM_SELF," %.0g",(double)PetscRealPart(dat[i]));CHKERRQ(ierr);}
    ierr = VecRestoreArrayRead(y1,&dat);CHKERRQ(ierr);

    /* Print the part of y2 on rank 0, which is 5 6 7 8 9 since y2 swapped the processes of PETSC_COMM_WORLD */
    ierr = PetscPrintf(PETSC_COMM_SELF,"\nOn rank 0 of PETSC_COMM_WORLD, y2 = ");CHKERRQ(ierr);
    ierr = VecGetArrayRead(y2,&dat);CHKERRQ(ierr);
    for (i=0; i<n; i++) {ierr = PetscPrintf(PETSC_COMM_SELF," %.0g",(double)PetscRealPart(dat[i]));CHKERRQ(ierr);}
    ierr = VecRestoreArrayRead(y2,&dat);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"\n");CHKERRQ(ierr);
  }

  ierr = ISDestroy(&ix);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&y1);CHKERRQ(ierr);
  ierr = VecDestroy(&y2);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&vscat1);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&vscat2);CHKERRQ(ierr);
  ierr = MPI_Comm_free(&newcomm);CHKERRMPI(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      nsize: 2
      requires: double
TEST*/

