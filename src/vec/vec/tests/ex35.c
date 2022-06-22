
static char help[] = "Test VecGetArray4d()\n";

#include <petscvec.h>

int main(int argc,char **argv)
{
  PetscInt       cnt,i,j,k,l,m = 2,n = 3,p = 4,q = 5;
  Vec            x;
  PetscScalar    ****xx;

  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCall(VecCreateSeq(PETSC_COMM_WORLD,m*n*p*q,&x));
  PetscCall(VecGetArray4d(x,m,n,p,q,0,0,0,0,&xx));
  cnt  = 0;
  for (i=0; i<m; i++)
    for (j=0; j<n; j++)
      for (k=0; k<p; k++)
        for (l=0; l<q; l++)
          xx[i][j][k][l] = cnt++;

  PetscCall(VecRestoreArray4d(x,m,n,p,q,0,0,0,0,&xx));
  PetscCall(VecView(x,PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(VecDestroy(&x));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:

TEST*/
