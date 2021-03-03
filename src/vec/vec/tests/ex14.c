
static char help[] = "Scatters from a sequential vector to a parallel vector.\n\
This does the tricky case.\n\n";

#include <petscvec.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscInt       n = 5,N;
  PetscMPIInt    size,rank;
  PetscScalar    value,zero = 0.0;
  Vec            x,y;
  IS             is1,is2;
  VecScatter     ctx = 0;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);

  /* create two vectors */
  N    = size*n;
  ierr = VecCreate(PETSC_COMM_WORLD,&y);CHKERRQ(ierr);
  ierr = VecSetSizes(y,PETSC_DECIDE,N);CHKERRQ(ierr);
  ierr = VecSetFromOptions(y);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,N,&x);CHKERRQ(ierr);

  /* create two index sets */
  ierr = ISCreateStride(PETSC_COMM_SELF,n,0,1,&is1);CHKERRQ(ierr);
  ierr = ISCreateStride(PETSC_COMM_SELF,n,rank,1,&is2);CHKERRQ(ierr);

  value = rank+1;
  ierr  = VecSet(x,value);CHKERRQ(ierr);
  ierr  = VecSet(y,zero);CHKERRQ(ierr);

  ierr = VecScatterCreate(x,is1,y,is2,&ctx);CHKERRQ(ierr);
  ierr = VecScatterBegin(ctx,x,y,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx,x,y,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&ctx);CHKERRQ(ierr);

  ierr = VecView(y,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);
  ierr = ISDestroy(&is1);CHKERRQ(ierr);
  ierr = ISDestroy(&is2);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}



/*TEST

   test:
      nsize: 2

TEST*/
