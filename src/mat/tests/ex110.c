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
  PetscErrorCode    ierr;
  PetscBool         equal,done;
  Mat               AA,AB;
  PetscMPIInt       size,rank;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheckFalse(size == 1,PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,"Must run with 2 or more processes");
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));

  /* Create a mpiaij matrix for checking */
  CHKERRQ(MatCreateAIJ(PETSC_COMM_WORLD,5,5,PETSC_DECIDE,PETSC_DECIDE,0,NULL,0,NULL,&A));
  CHKERRQ(MatSetOption(A,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_FALSE));
  CHKERRQ(MatSetUp(A));
  CHKERRQ(PetscRandomCreate(PETSC_COMM_WORLD,&rctx));
  CHKERRQ(PetscRandomSetFromOptions(rctx));

  for (i=5*rank; i<5*rank+5; i++) {
    for (j=0; j<5*size; j++) {
      CHKERRQ(PetscRandomGetValue(rctx,&value));
      column = (PetscInt) (5*size*PetscRealPart(value));
      CHKERRQ(PetscRandomGetValue(rctx,&value));
      CHKERRQ(MatSetValues(A,1,&i,1,&column,&value,INSERT_VALUES));
    }
  }
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  CHKERRQ(MatMPIAIJGetSeqAIJ(A,&AA,&AB,&garray));
  CHKERRQ(MatGetRowIJ(AA,0,PETSC_FALSE,PETSC_FALSE,&nd,(const PetscInt**)&di,(const PetscInt**)&dj,&done));
  CHKERRQ(MatSeqAIJGetArray(AA,&da));
  CHKERRQ(MatGetRowIJ(AB,0,PETSC_FALSE,PETSC_FALSE,&nd,(const PetscInt**)&oi,(const PetscInt**)&oj,&done));
  CHKERRQ(MatSeqAIJGetArray(AB,&oa));

  CHKERRQ(PetscMalloc1(oi[5],&ooj));
  CHKERRQ(PetscArraycpy(ooj,oj,oi[5]));
  /* modify the column entries in the non-diagonal portion back to global numbering */
  for (i=0; i<oi[5]; i++) {
    ooj[i] = garray[ooj[i]];
  }

  CHKERRQ(MatCreateMPIAIJWithSplitArrays(PETSC_COMM_WORLD,5,5,PETSC_DETERMINE,PETSC_DETERMINE,di,dj,da,oi,ooj,oa,&B));
  CHKERRQ(MatSetUp(B));
  CHKERRQ(MatEqual(A,B,&equal));

  CHKERRQ(MatRestoreRowIJ(AA,0,PETSC_FALSE,PETSC_FALSE,&nd,(const PetscInt**)&di,(const PetscInt**)&dj,&done));
  CHKERRQ(MatSeqAIJRestoreArray(AA,&da));
  CHKERRQ(MatRestoreRowIJ(AB,0,PETSC_FALSE,PETSC_FALSE,&nd,(const PetscInt**)&oi,(const PetscInt**)&oj,&done));
  CHKERRQ(MatSeqAIJRestoreArray(AB,&oa));

  PetscCheck(equal,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Likely a bug in MatCreateMPIAIJWithSplitArrays()");

  /* Free spaces */
  CHKERRQ(PetscFree(ooj));
  CHKERRQ(PetscRandomDestroy(&rctx));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&B));
  CHKERRQ(PetscFree(oj));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      nsize: 3

TEST*/
