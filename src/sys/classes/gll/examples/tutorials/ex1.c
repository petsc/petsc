
#include <petscgll.h>
#include <petscviewer.h>

int main(int argc,char **args)
{
  PetscErrorCode ierr;
  PetscGLL       gll_la,gll_n;
  PetscInt       n = 3,i;

  ierr = PetscInitialize(&argc,&args,NULL,NULL);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
  ierr = PetscGLLCreate(n,PETSCGLL_VIA_LINEARALGEBRA,&gll_la);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_SELF,"Computed via linear algebra: ");CHKERRQ(ierr);
  ierr = PetscGLLView(&gll_la,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);

  ierr = PetscGLLCreate(n,PETSCGLL_VIA_NEWTON,&gll_n);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_SELF,"Computed via Newton: ");CHKERRQ(ierr);
  ierr = PetscGLLView(&gll_n,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);

  for (i=0; i<n; i++) {
    gll_la.nodes[i]   -= gll_n.nodes[i];
    gll_la.weights[i] -= gll_n.weights[i];
  }
  ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_SELF,"Difference: ");CHKERRQ(ierr);
  ierr = PetscGLLView(&gll_la,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);

  ierr = PetscGLLDestroy(&gll_la);CHKERRQ(ierr);
  ierr = PetscGLLDestroy(&gll_n);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:

TEST*/
