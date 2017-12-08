
static char help[] = "Tests VecMax() with index.\n\
  -n <length> : vector length\n\n";

#include <petscvec.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscInt       n = 5,idx;
  PetscReal      value;
  Vec            x;
  PetscScalar    one = 1.0;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);

  /* create vector */
  ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
  ierr = VecSetSizes(x,PETSC_DECIDE,n);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);


  ierr = VecSet(x,one);CHKERRQ(ierr);
  ierr = VecSetValue(x,0,0.0,INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecSetValue(x,n-1,2.0,INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(x);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(x);CHKERRQ(ierr);

  ierr = VecView(x,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = VecMax(x,&idx,&value);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Maximum value %g index %D\n",(double)value,idx);CHKERRQ(ierr);
  ierr = VecMin(x,&idx,&value);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Minimum value %g index %D\n",(double)value,idx);CHKERRQ(ierr);

  ierr = VecDestroy(&x);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}



/*TEST
   test:

   test:
      suffix: 2
      nsize: 2

TEST*/
