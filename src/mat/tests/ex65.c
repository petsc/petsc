
static char help[] = "Saves a rectangular sparse matrix to disk.\n\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat            A;
  PetscErrorCode ierr;
  PetscInt       m = 100,n = 11,js[11],i,j,cnt;
  PetscScalar    values[11];
  PetscViewer    view;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(MatCreateSeqAIJ(PETSC_COMM_WORLD,m,n,20,0,&A));

  for (i=0; i<n; i++) values[i] = (PetscReal)i;

  for (i=0; i<m; i++) {
    cnt = 0;
    if (i % 2) {
      for (j=0; j<n; j += 2) {
        js[cnt++] = j;
      }
    } else {
      ;
    }
    CHKERRQ(MatSetValues(A,1,&i,cnt,js,values,INSERT_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,"rect",FILE_MODE_WRITE,&view));
  CHKERRQ(MatView(A,view));
  CHKERRQ(PetscViewerDestroy(&view));

  CHKERRQ(MatDestroy(&A));

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:

TEST*/
