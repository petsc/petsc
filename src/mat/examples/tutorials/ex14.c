
static char help[] = "Tests basic assembly of MATIJ using the edges of a 2d parallel DA.\n";

#include <petscmat.h>
#include <petscdmda.h>
#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  PetscErrorCode ierr;
  DM       da;
  Mat      A;
  PetscInt i,j,k,i0,j0,m,n,gi0,gj0,gm,gn,M = 4,N = 4, e0[2],e1[2];
#if 0
  PetscBool preallocate,flag;
#endif

  PetscInitialize(&argc,&args,(char *)0,help);

  /* Create the DA and extract its size info. */
  ierr = DMDACreate2d(PETSC_COMM_WORLD, DMDA_BOUNDARY_NONE, DMDA_BOUNDARY_NONE, DMDA_STENCIL_STAR,-M,-N,PETSC_DECIDE,PETSC_DECIDE,1,1,PETSC_NULL,PETSC_NULL,&da); CHKERRQ(ierr);
  ierr = DMDAGetInfo(da, PETSC_NULL, &M,&N,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL); CHKERRQ(ierr);
  ierr = DMDAGetCorners(da,&i0,&j0,PETSC_NULL,&m,&n,PETSC_NULL); CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(da,&gi0,&gj0,PETSC_NULL,&gm,&gn,PETSC_NULL); CHKERRQ(ierr);

  /* Create MATIJ with m*n local rows (and columns). */
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n);CHKERRQ(ierr);
  ierr = MatSetType(A,MATIJ);CHKERRQ(ierr);
#if 0
  ierr = PetscOptionsGetBool(PETSC_NULL, "--preallocate", &preallocate, &flag); CHKERRQ(ierr);
  if (preallocate) {
  }
#endif

  /*
   Add local and ghosted edges to A: grid points are indexed by i first,
   so that points with the same i-index differ by a multiple of M.
   */
  for (j = j0; j < j0+n; ++j) {
    for (i = i0; i < i0+m; ++i) {
      k = 0;
      if (i+1 < gi0+gm) {/* there is a point to the right, so draw an edge to it.*/
        e0[k] = i*M+j; e1[k] = (i+i)*M+j;
        ++k;
      }
      if (j+1 < gj0+gn) {/* there is a point above, so draw an edge to it.*/
        e0[k] = i*M+j; e1[k] = (i)*M+j+1;
        ++k;
      }
      MatIJSetEdges(A,k,e0,e1); CHKERRQ(ierr);

    }
  }
  ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);    CHKERRQ(ierr);

  /* View A. */
  ierr = MatView(A, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);

  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
