
static char help[] = "Tests MatDenseGetArray() and MatView_SeqDense_Binary(), MatView_MPIDense_Binary().\n\n";

#include <petscmat.h>
#include <petscviewer.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Mat              A;
  PetscInt         i,j,m = 3,n = 2,rstart,rend;
  PetscErrorCode   ierr;
  PetscScalar      v,*array;
  PetscViewer      view;

  PetscInitialize(&argc,&args,(char *)0,help);

  /*
      Create a parallel dense matrix shared by all processors
  */
  ierr = MatCreateDense(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,m,n,PETSC_NULL,&A);CHKERRQ(ierr);

  /*
     Set values into the matrix
  */
  for (i=0; i<m; i++) {
    for (j=0; j<n; j++) {
      v = 9.0/(i+j+1); ierr = MatSetValues(A,1,&i,1,&j,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /*
       Print the matrix to the screen
  */
  ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);


  /*
      Print the local portion of the matrix to the screen
  */
  ierr = MatDenseGetArray(A,&array);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(A,&rstart,&rend);CHKERRQ(ierr);
  for (i=rstart; i<rend; i++) {
    for (j=0; j<n; j++) {
      PetscSynchronizedPrintf(PETSC_COMM_WORLD,"%6.4e ",PetscRealPart(array[j*(rend-rstart)+i-rstart]));
    }
    PetscSynchronizedPrintf(PETSC_COMM_WORLD,"\n");
  }
  PetscSynchronizedFlush(PETSC_COMM_WORLD);
  ierr = MatDenseRestoreArray(A,&array);CHKERRQ(ierr);

  /*
      Store the binary matrix to a file
  */
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD, "matrix.dat", FILE_MODE_WRITE, &view);CHKERRQ(ierr);
  ierr = PetscViewerSetFormat(view,PETSC_VIEWER_NATIVE);CHKERRQ(ierr);
  ierr = MatView(A,view);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&view);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);

  /*
     Now reload the matrix and view it
  */
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"matrix.dat",FILE_MODE_READ,&view);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetType(A,MATMPIDENSE);CHKERRQ(ierr);
  ierr = MatLoad(A,view);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&view);CHKERRQ(ierr);
  ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return 0;
}
