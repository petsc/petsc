static char help[] = "Tests 1D Gauss-Lobatto-Legendre discretization on [-1, 1].\n\n";

#include <petscdt.h>
#include <petscviewer.h>

int main(int argc, char **argv)
{
  PetscInt   n = 3, i;
  PetscReal *la_nodes, *la_weights, *n_nodes, *n_weights;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));

  PetscCall(PetscMalloc1(n, &la_nodes));
  PetscCall(PetscMalloc1(n, &la_weights));
  PetscCall(PetscMalloc1(n, &n_nodes));
  PetscCall(PetscMalloc1(n, &n_weights));
  PetscCall(PetscDTGaussLobattoLegendreQuadrature(n, PETSCGAUSSLOBATTOLEGENDRE_VIA_LINEAR_ALGEBRA, la_nodes, la_weights));

  PetscCall(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_SELF, "Gauss-Lobatto-Legendre nodes and weights computed via linear algebra: \n"));
  PetscCall(PetscRealView(n, la_nodes, PETSC_VIEWER_STDOUT_SELF));
  PetscCall(PetscRealView(n, la_weights, PETSC_VIEWER_STDOUT_SELF));
  PetscCall(PetscDTGaussLobattoLegendreQuadrature(n, PETSCGAUSSLOBATTOLEGENDRE_VIA_NEWTON, n_nodes, n_weights));
  PetscCall(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_SELF, "Gauss-Lobatto-Legendre nodes and weights computed via Newton: \n"));
  PetscCall(PetscRealView(n, n_nodes, PETSC_VIEWER_STDOUT_SELF));
  PetscCall(PetscRealView(n, n_weights, PETSC_VIEWER_STDOUT_SELF));

  for (i = 0; i < n; i++) {
    la_nodes[i] -= n_nodes[i];
    la_weights[i] -= n_weights[i];
  }
  PetscCall(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_SELF, "Difference: \n"));
  PetscCall(PetscRealView(n, la_nodes, PETSC_VIEWER_STDOUT_SELF));
  PetscCall(PetscRealView(n, la_weights, PETSC_VIEWER_STDOUT_SELF));

  PetscCall(PetscFree(la_nodes));
  PetscCall(PetscFree(la_weights));
  PetscCall(PetscFree(n_nodes));
  PetscCall(PetscFree(n_weights));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:

TEST*/
