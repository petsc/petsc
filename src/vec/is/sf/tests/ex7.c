static char help[]= "Test vecscatter of different block sizes across processes\n\n";

#include <petscvec.h>
int main(int argc,char **argv)
{
  PetscErrorCode     ierr;
  PetscInt           i,bs,n,low,high;
  PetscMPIInt        nproc,rank;
  Vec                x,y,z;
  IS                 ix,iy;
  VecScatter         vscat;
  const PetscScalar  *yv;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&nproc);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);
  if (nproc != 2) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"This test can only run on two MPI ranks");

  /* Create an MPI vector x of size 12 on two processes, and set x = {0, 1, 2, .., 11} */
  ierr = VecCreateMPI(PETSC_COMM_WORLD,6,PETSC_DECIDE,&x);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(x,&low,&high);CHKERRQ(ierr);
  for (i=low; i<high; i++) {ierr = VecSetValue(x,i,(PetscScalar)i,INSERT_VALUES);CHKERRQ(ierr);}
  ierr = VecAssemblyBegin(x);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(x);CHKERRQ(ierr);

  /* Create a seq vector y, and a parallel to sequential (PtoS) vecscatter to scatter x to y */
  if (!rank) {
    /* On rank 0, seq y is of size 6. We will scatter x[0,1,2,6,7,8] to y[0,1,2,3,4,5] using IS with bs=3 */
    PetscInt idx[2]={0,2};
    PetscInt idy[2]={0,1};
    n    = 6;
    bs   = 3;
    ierr = VecCreateSeq(PETSC_COMM_SELF,n,&y);CHKERRQ(ierr);
    ierr = ISCreateBlock(PETSC_COMM_SELF,bs,2,idx,PETSC_COPY_VALUES,&ix);CHKERRQ(ierr);
    ierr = ISCreateBlock(PETSC_COMM_SELF,bs,2,idy,PETSC_COPY_VALUES,&iy);CHKERRQ(ierr);
  } else {
    /* On rank 1, seq y is of size 4. We will scatter x[4,5,10,11] to y[0,1,2,3] using IS with bs=2 */
    PetscInt idx[2]= {2,5};
    PetscInt idy[2]= {0,1};
    n    = 4;
    bs   = 2;
    ierr = VecCreateSeq(PETSC_COMM_SELF,n,&y);CHKERRQ(ierr);
    ierr = ISCreateBlock(PETSC_COMM_SELF,bs,2,idx,PETSC_COPY_VALUES,&ix);CHKERRQ(ierr);
    ierr = ISCreateBlock(PETSC_COMM_SELF,bs,2,idy,PETSC_COPY_VALUES,&iy);CHKERRQ(ierr);
  }
  ierr = VecScatterCreate(x,ix,y,iy,&vscat);CHKERRQ(ierr);

  /* Do the vecscatter */
  ierr = VecScatterBegin(vscat,x,y,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(vscat,x,y,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

  /* Print y. Since y is sequential, we put y in a parallel z to print its value on both ranks */
  ierr = VecGetArrayRead(y,&yv);CHKERRQ(ierr);
  ierr = VecCreateMPIWithArray(PETSC_COMM_WORLD,1,n,PETSC_DECIDE,yv,&z);CHKERRQ(ierr);
  ierr = VecView(z,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(y,&yv);CHKERRQ(ierr);

  ierr = ISDestroy(&ix);CHKERRQ(ierr);
  ierr = ISDestroy(&iy);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);
  ierr = VecDestroy(&z);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&vscat);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      nsize: 2
      args:
      requires:
TEST*/

