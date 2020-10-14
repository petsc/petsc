#include <petscfv.h>

static char help[] = "Test memory allocation of PetscFV arrays used in PetscFVComputeGradient";

int main(int argc, char **argv)
{
    PetscErrorCode ierr;
    PetscFV        fvm;
    PetscInt       dim, numFaces;
    PetscScalar    *dx, *grad;

    PetscFunctionBeginUser;
    ierr = PetscInitialize(&argc, &argv, PETSC_NULL, help); if (ierr) return ierr;

    /*
      Working with a 2D mesh, made of triangles, and using the 2nd neighborhood
      to reconstruct the cell gradient with a least square method, we use numFaces = 9
      The array dx is not initialised, but it doesn't matter here
      */
    dim = 2;
    numFaces = 9;
    ierr = PetscMalloc2(dim * numFaces, &dx, dim * numFaces, &grad);CHKERRQ(ierr);
    ierr = PetscFVCreate(PETSC_COMM_WORLD, &fvm);CHKERRQ(ierr);
    ierr = PetscFVSetType(fvm, PETSCFVLEASTSQUARES);CHKERRQ(ierr);
    ierr = PetscFVLeastSquaresSetMaxFaces(fvm, numFaces);CHKERRQ(ierr);

    /* Issue here */
    ierr = PetscFVComputeGradient(fvm, numFaces, dx, grad);CHKERRQ(ierr);

    ierr = PetscFVDestroy(&fvm);CHKERRQ(ierr);
    ierr = PetscFree2(dx, grad);CHKERRQ(ierr);
    ierr = PetscFinalize();CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

/*TEST
  test:
    suffix: 1
TEST*/
