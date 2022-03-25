
static char help[] = "Test saving SeqSBAIJ matrix that is missing diagonal entries.";

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat            A;
  PetscInt       bs=3,m=4,i,j,val = 10,row[2],col[3],rstart;
  PetscMPIInt    size;
  PetscScalar    x[6][9];

  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheckFalse(size > 1,PETSC_COMM_WORLD,PETSC_ERR_SUP,"Test is only for sequential");
  PetscCall(MatCreateSeqSBAIJ(PETSC_COMM_SELF,bs,m*bs,m*bs,1,NULL,&A));
  PetscCall(MatSetOption(A,MAT_IGNORE_LOWER_TRIANGULAR,PETSC_TRUE));
  rstart = 0;

  row[0] =rstart+0;  row[1] =rstart+2;
  col[0] =rstart+0;  col[1] =rstart+1;  col[2] =rstart+3;
  for (i=0; i<6; i++) {
    for (j =0; j< 9; j++) x[i][j] = (PetscScalar)val++;
  }
  PetscCall(MatSetValuesBlocked(A,2,row,3,col,&x[0][0],INSERT_VALUES));
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatView(A,PETSC_VIEWER_BINARY_WORLD));

  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return 0;
}
