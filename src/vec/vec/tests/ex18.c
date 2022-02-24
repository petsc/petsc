
static char help[] = "Compares BLAS dots on different machines. Input\n\
arguments are\n\
  -n <length> : local vector length\n\n";

#include <petscvec.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscInt       n = 15,i;
  PetscScalar    v;
  Vec            x,y;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  if (n < 5) n = 5;

  /* create two vectors */
  CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,n,&x));
  CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,n,&y));

  for (i=0; i<n; i++) {
    v    = ((PetscReal)i) + 1.0/(((PetscReal)i) + .35);
    CHKERRQ(VecSetValues(x,1,&i,&v,INSERT_VALUES));
    v   += 1.375547826473644376;
    CHKERRQ(VecSetValues(y,1,&i,&v,INSERT_VALUES));
  }
  CHKERRQ(VecAssemblyBegin(y));
  CHKERRQ(VecAssemblyEnd(y));

  CHKERRQ(VecDot(x,y,&v));
  CHKERRQ(PetscFPrintf(PETSC_COMM_WORLD,stdout,"Vector inner product %16.12e\n",(double)PetscRealPart(v)));

  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&y));

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:

TEST*/
