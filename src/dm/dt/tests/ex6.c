static char help[] = "Tests 1D Gauss-Lobatto-Legendre discretization on [-1, 1].\n\n";

#include <petscdt.h>
#include <petscviewer.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;

  PetscInt       n = 3,i;
  PetscReal      *la_nodes,*la_weights,*n_nodes,*n_weights;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));

  CHKERRQ(PetscMalloc1(n,&la_nodes));
  CHKERRQ(PetscMalloc1(n,&la_weights));
  CHKERRQ(PetscMalloc1(n,&n_nodes));
  CHKERRQ(PetscMalloc1(n,&n_weights));
  CHKERRQ(PetscDTGaussLobattoLegendreQuadrature(n,PETSCGAUSSLOBATTOLEGENDRE_VIA_LINEAR_ALGEBRA,la_nodes,la_weights));

  CHKERRQ(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_SELF,"Gauss-Lobatto-Legendre nodes and weights computed via linear algebra: \n"));
  CHKERRQ(PetscRealView(n,la_nodes,PETSC_VIEWER_STDOUT_SELF));
  CHKERRQ(PetscRealView(n,la_weights,PETSC_VIEWER_STDOUT_SELF));
  CHKERRQ(PetscDTGaussLobattoLegendreQuadrature(n,PETSCGAUSSLOBATTOLEGENDRE_VIA_NEWTON,n_nodes,n_weights));
  CHKERRQ(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_SELF,"Gauss-Lobatto-Legendre nodes and weights computed via Newton: \n"));
  CHKERRQ(PetscRealView(n,n_nodes,PETSC_VIEWER_STDOUT_SELF));
  CHKERRQ(PetscRealView(n,n_weights,PETSC_VIEWER_STDOUT_SELF));

  for (i=0; i<n; i++) {
    la_nodes[i]   -= n_nodes[i];
    la_weights[i] -= n_weights[i];
  }
  CHKERRQ(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_SELF,"Difference: \n"));
  CHKERRQ(PetscRealView(n,la_nodes,PETSC_VIEWER_STDOUT_SELF));
  CHKERRQ(PetscRealView(n,la_weights,PETSC_VIEWER_STDOUT_SELF));

  CHKERRQ(PetscFree(la_nodes));
  CHKERRQ(PetscFree(la_weights));
  CHKERRQ(PetscFree(n_nodes));
  CHKERRQ(PetscFree(n_weights));

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:

TEST*/
