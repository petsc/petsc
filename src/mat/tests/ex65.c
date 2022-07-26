
static char help[] = "Saves a rectangular sparse matrix to disk.\n\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat            A;
  PetscInt       m = 100,n = 11,js[11],i,j,cnt;
  PetscScalar    values[11];
  PetscViewer    view;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCall(MatCreateSeqAIJ(PETSC_COMM_WORLD,m,n,20,0,&A));

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
    PetscCall(MatSetValues(A,1,&i,cnt,js,values,INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,"rect",FILE_MODE_WRITE,&view));
  PetscCall(MatView(A,view));
  PetscCall(PetscViewerDestroy(&view));

  PetscCall(MatDestroy(&A));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:

TEST*/
