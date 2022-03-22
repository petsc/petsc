static char help[] = "Testing MatCreateMPIAIJWithSplitArrays().\n\n";

#include <petscmat.h>

int main(int argc,char **argv)
{
  Mat               A,B;
  PetscInt          i,j,column,*ooj;
  PetscInt          *di,*dj,*oi,*oj,nd;
  const PetscInt    *garray;
  PetscScalar       *oa,*da;
  PetscScalar       value;
  PetscRandom       rctx;
  PetscBool         equal,done;
  Mat               AA,AB;
  PetscMPIInt       size,rank;

  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheck(size > 1,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"Must run with 2 or more processes");
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));

  /* Create a mpiaij matrix for checking */
  PetscCall(MatCreateAIJ(PETSC_COMM_WORLD,5,5,PETSC_DECIDE,PETSC_DECIDE,0,NULL,0,NULL,&A));
  PetscCall(MatSetOption(A,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_FALSE));
  PetscCall(MatSetUp(A));
  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD,&rctx));
  PetscCall(PetscRandomSetFromOptions(rctx));

  for (i=5*rank; i<5*rank+5; i++) {
    for (j=0; j<5*size; j++) {
      PetscCall(PetscRandomGetValue(rctx,&value));
      column = (PetscInt) (5*size*PetscRealPart(value));
      PetscCall(PetscRandomGetValue(rctx,&value));
      PetscCall(MatSetValues(A,1,&i,1,&column,&value,INSERT_VALUES));
    }
  }
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  PetscCall(MatMPIAIJGetSeqAIJ(A,&AA,&AB,&garray));
  PetscCall(MatGetRowIJ(AA,0,PETSC_FALSE,PETSC_FALSE,&nd,(const PetscInt**)&di,(const PetscInt**)&dj,&done));
  PetscCall(MatSeqAIJGetArray(AA,&da));
  PetscCall(MatGetRowIJ(AB,0,PETSC_FALSE,PETSC_FALSE,&nd,(const PetscInt**)&oi,(const PetscInt**)&oj,&done));
  PetscCall(MatSeqAIJGetArray(AB,&oa));

  PetscCall(PetscMalloc1(oi[5],&ooj));
  PetscCall(PetscArraycpy(ooj,oj,oi[5]));
  /* modify the column entries in the non-diagonal portion back to global numbering */
  for (i=0; i<oi[5]; i++) {
    ooj[i] = garray[ooj[i]];
  }

  PetscCall(MatCreateMPIAIJWithSplitArrays(PETSC_COMM_WORLD,5,5,PETSC_DETERMINE,PETSC_DETERMINE,di,dj,da,oi,ooj,oa,&B));
  PetscCall(MatSetUp(B));
  PetscCall(MatEqual(A,B,&equal));

  PetscCall(MatRestoreRowIJ(AA,0,PETSC_FALSE,PETSC_FALSE,&nd,(const PetscInt**)&di,(const PetscInt**)&dj,&done));
  PetscCall(MatSeqAIJRestoreArray(AA,&da));
  PetscCall(MatRestoreRowIJ(AB,0,PETSC_FALSE,PETSC_FALSE,&nd,(const PetscInt**)&oi,(const PetscInt**)&oj,&done));
  PetscCall(MatSeqAIJRestoreArray(AB,&oa));

  PetscCheck(equal,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Likely a bug in MatCreateMPIAIJWithSplitArrays()");

  /* Free spaces */
  PetscCall(PetscFree(ooj));
  PetscCall(PetscRandomDestroy(&rctx));
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&B));
  PetscCall(PetscFree(oj));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      nsize: 3

TEST*/
