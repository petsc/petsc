static char help[] = "Tests 1D Gauss-Lobatto-Legendre discretization on [-1, 1].\n\n";

#include <petscdt.h>
#include <petscviewer.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;

  PetscInt       n = 3,i;
  PetscReal      *la_nodes,*la_weights,*n_nodes,*n_weights;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);

  ierr = PetscMalloc1(n,&la_nodes);CHKERRQ(ierr);
  ierr = PetscMalloc1(n,&la_weights);CHKERRQ(ierr);
  ierr = PetscMalloc1(n,&n_nodes);CHKERRQ(ierr);
  ierr = PetscMalloc1(n,&n_weights);CHKERRQ(ierr);
  ierr = PetscDTGaussLobattoLegendreQuadrature(n,PETSCGAUSSLOBATTOLEGENDRE_VIA_LINEAR_ALGEBRA,la_nodes,la_weights);CHKERRQ(ierr);

  ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_SELF,"Gauss-Lobatto-Legendre nodes and weights computed via linear algebra: \n");CHKERRQ(ierr);
  ierr = PetscRealView(n,la_nodes,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  ierr = PetscRealView(n,la_weights,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  ierr = PetscDTGaussLobattoLegendreQuadrature(n,PETSCGAUSSLOBATTOLEGENDRE_VIA_NEWTON,n_nodes,n_weights);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_SELF,"Gauss-Lobatto-Legendre nodes and weights computed via Newton: \n");CHKERRQ(ierr);
  ierr = PetscRealView(n,n_nodes,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  ierr = PetscRealView(n,n_weights,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);

  for (i=0; i<n; i++) {
    la_nodes[i]   -= n_nodes[i];
    la_weights[i] -= n_weights[i];
  }
  ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_SELF,"Difference: \n");CHKERRQ(ierr);
  ierr = PetscRealView(n,la_nodes,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  ierr = PetscRealView(n,la_weights,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);

  ierr = PetscFree(la_nodes);CHKERRQ(ierr);
  ierr = PetscFree(la_weights);CHKERRQ(ierr);
  ierr = PetscFree(n_nodes);CHKERRQ(ierr);
  ierr = PetscFree(n_weights);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:

TEST*/
