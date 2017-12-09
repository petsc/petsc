
static char help[] = "Test VecGetArray4d()\n";

#include <petscvec.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscInt       cnt,i,j,k,l,m = 2,n = 3,p = 4,q = 5;
  Vec            x;
  PetscScalar    ****xx;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = VecCreateSeq(PETSC_COMM_WORLD,m*n*p*q,&x);CHKERRQ(ierr);
  ierr = VecGetArray4d(x,m,n,p,q,0,0,0,0,&xx);CHKERRQ(ierr);
  cnt  = 0;
  for (i=0; i<m; i++)
    for (j=0; j<n; j++)
      for (k=0; k<p; k++)
        for (l=0; l<q; l++)
          xx[i][j][k][l] = cnt++;

  ierr = VecRestoreArray4d(x,m,n,p,q,0,0,0,0,&xx);CHKERRQ(ierr);
  ierr = VecView(x,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}



/*TEST

   test:

TEST*/
