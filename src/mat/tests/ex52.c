
static char help[] = "Tests various routines in MatMPIBAIJ format.\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat            A;
  PetscInt       m=2,bs=1,M,row,col,start,end,i,j,k;
  PetscMPIInt    rank,size;
  PetscScalar    data=100;
  PetscBool      flg;

  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));

  /* Test MatSetValues() and MatGetValues() */
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-mat_block_size",&bs,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-mat_size",&m,NULL));

  M    = m*bs*size;
  PetscCall(MatCreateBAIJ(PETSC_COMM_WORLD,bs,PETSC_DECIDE,PETSC_DECIDE,M,M,PETSC_DECIDE,NULL,PETSC_DECIDE,NULL,&A));

  PetscCall(MatGetOwnershipRange(A,&start,&end));
  PetscCall(PetscOptionsHasName(NULL,NULL,"-column_oriented",&flg));
  if (flg) {
    PetscCall(MatSetOption(A,MAT_ROW_ORIENTED,PETSC_FALSE));
  }

  /* inproc assembly */
  for (row=start; row<end; row++) {
    for (col=start; col<end; col++,data+=1) {
      PetscCall(MatSetValues(A,1,&row,1,&col,&data,INSERT_VALUES));
    }
  }
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  /* offproc assembly */
  data = 5.0;
  row  = (M+start-1)%M;
  for (col=0; col<M; col++) {
    PetscCall(MatSetValues(A,1,&row,1,&col,&data,ADD_VALUES));
  }
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  /* Test MatSetValuesBlocked() */
  PetscCall(PetscOptionsHasName(NULL,NULL,"-test_setvaluesblocked",&flg));
  if (flg) {
    PetscScalar *bval;
    row /= bs;
    col  = start/bs;
    PetscCall(PetscMalloc1(bs*bs,&bval));
    k = 1;
    /* row oriented - default */
    for (i=0; i<bs; i++) {
      for (j=0; j<bs; j++) {
        bval[i*bs+j] = (PetscScalar)k; k++;
      }
    }
    PetscCall(MatSetValuesBlocked(A,1,&row,1,&col,bval,INSERT_VALUES));
    PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
    PetscCall(PetscFree(bval));
  }

  PetscCall(MatView(A,PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      suffix: 1
      nsize: 3
      args: -mat_block_size 2 -test_setvaluesblocked

   test:
      suffix: 2
      nsize: 3
      args: -mat_block_size 2 -test_setvaluesblocked -column_oriented

   test:
      suffix: 3
      nsize: 3
      args: -mat_block_size 1 -test_setvaluesblocked

   test:
      suffix: 4
      nsize: 3
      args: -mat_block_size 1 -test_setvaluesblocked -column_oriented

TEST*/
