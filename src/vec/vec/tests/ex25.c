
static char help[] = "Scatters from a parallel vector to a sequential vector.  In\n\
this case processor zero is as long as the entire parallel vector; rest are zero length.\n\n";

#include <petscvec.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscInt       n = 5,N,low,high,iglobal,i;
  PetscMPIInt    size,rank;
  PetscScalar    value,zero = 0.0;
  Vec            x,y;
  IS             is1,is2;
  VecScatter     ctx;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);

  /* create two vectors */
  N    = size*n;
  ierr = VecCreate(PETSC_COMM_WORLD,&y);CHKERRQ(ierr);
  ierr = VecSetSizes(y,PETSC_DECIDE,N);CHKERRQ(ierr);
  ierr = VecSetFromOptions(y);CHKERRQ(ierr);
  if (rank == 0) {
    ierr = VecCreateSeq(PETSC_COMM_SELF,N,&x);CHKERRQ(ierr);
  } else {
    ierr = VecCreateSeq(PETSC_COMM_SELF,0,&x);CHKERRQ(ierr);
  }

  /* create two index sets */
  if (rank == 0) {
    ierr = ISCreateStride(PETSC_COMM_SELF,N,0,1,&is1);CHKERRQ(ierr);
    ierr = ISCreateStride(PETSC_COMM_SELF,N,0,1,&is2);CHKERRQ(ierr);
  } else {
    ierr = ISCreateStride(PETSC_COMM_SELF,0,0,1,&is1);CHKERRQ(ierr);
    ierr = ISCreateStride(PETSC_COMM_SELF,0,0,1,&is2);CHKERRQ(ierr);
  }

  ierr = VecSet(x,zero);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(y,&low,&high);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    iglobal = i + low; value = (PetscScalar) (i + 10*rank);
    ierr    = VecSetValues(y,1,&iglobal,&value,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(y);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(y);CHKERRQ(ierr);
  ierr = VecView(y,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = VecScatterCreate(y,is2,x,is1,&ctx);CHKERRQ(ierr);
  ierr = VecScatterBegin(ctx,y,x,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx,y,x,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&ctx);CHKERRQ(ierr);

  if (rank == 0) {
    printf("----\n");
    ierr = VecView(x,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  }

  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);
  ierr = ISDestroy(&is1);CHKERRQ(ierr);
  ierr = ISDestroy(&is2);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      nsize: 3

TEST*/
