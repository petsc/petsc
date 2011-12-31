
static char help[] = "Tests VecSetValues and VecSetValuesBlocked() on MPI vectors.\n\
Where atleast a couple of mallocs will occur in the stash code.\n\n";

#include <petscvec.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;
  PetscInt       i,j,n = 50,bs;
  PetscScalar    val,*vals,zero=0.0;
  Vec            x;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr); 
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  bs = size;

  ierr = PetscOptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
  ierr = VecSetSizes(x,PETSC_DECIDE,n*bs);CHKERRQ(ierr);
  ierr = VecSetBlockSize(x,bs);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);

  for (i=0; i<n*bs; i++) {
    val  = i*1.0;
    ierr = VecSetValues(x,1,&i,&val,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(x);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(x);CHKERRQ(ierr);

  ierr = VecView(x,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /* Now do the blocksetvalues */
  ierr = VecSet(x,zero);CHKERRQ(ierr);
  ierr = PetscMalloc(bs*sizeof(PetscScalar),&vals);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    for (j=0; j<bs; j++) {
      vals[j] = (i*bs+j)*1.0;
    }
    ierr = VecSetValuesBlocked(x,1,&i,vals,INSERT_VALUES);CHKERRQ(ierr);
  }

  ierr = VecAssemblyBegin(x);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(x);CHKERRQ(ierr);

  ierr = VecView(x,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = PetscFree(vals);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
 
