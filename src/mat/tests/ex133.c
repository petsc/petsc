
static char help[] = "Test saving SeqSBAIJ matrix that is missing diagonal entries.";

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat            A;
  PetscInt       bs=3,m=4,i,j,val = 10,row[2],col[3],rstart;
  PetscErrorCode ierr;
  PetscMPIInt    size;
  PetscScalar    x[6][9];

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
  PetscAssertFalse(size > 1,PETSC_COMM_WORLD,PETSC_ERR_SUP,"Test is only for sequential");
  ierr   = MatCreateSeqSBAIJ(PETSC_COMM_SELF,bs,m*bs,m*bs,1,NULL,&A);CHKERRQ(ierr);
  ierr   = MatSetOption(A,MAT_IGNORE_LOWER_TRIANGULAR,PETSC_TRUE);CHKERRQ(ierr);
  rstart = 0;

  row[0] =rstart+0;  row[1] =rstart+2;
  col[0] =rstart+0;  col[1] =rstart+1;  col[2] =rstart+3;
  for (i=0; i<6; i++) {
    for (j =0; j< 9; j++) x[i][j] = (PetscScalar)val++;
  }
  ierr = MatSetValuesBlocked(A,2,row,3,col,&x[0][0],INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatView(A,PETSC_VIEWER_BINARY_WORLD);CHKERRQ(ierr);

  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}
