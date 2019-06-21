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
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  if (size == 1) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,"Must run with 2 or more processes");
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

  /* Create a mpiaij matrix for checking */
  ierr = MatCreateAIJ(PETSC_COMM_WORLD,5,5,PETSC_DECIDE,PETSC_DECIDE,0,NULL,0,NULL,&A);CHKERRQ(ierr);
  ierr = MatSetOption(A,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_FALSE);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);
  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&rctx);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rctx);CHKERRQ(ierr);

  for (i=5*rank; i<5*rank+5; i++) {
    for (j=0; j<5*size; j++) {
      ierr   = PetscRandomGetValue(rctx,&value);CHKERRQ(ierr);
      column = (PetscInt) (5*size*PetscRealPart(value));
      ierr   = PetscRandomGetValue(rctx,&value);CHKERRQ(ierr);
      ierr   = MatSetValues(A,1,&i,1,&column,&value,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = MatMPIAIJGetSeqAIJ(A,&AA,&AB,&garray);CHKERRQ(ierr);
  ierr = MatGetRowIJ(AA,0,PETSC_FALSE,PETSC_FALSE,&nd,(const PetscInt**)&di,(const PetscInt**)&dj,&done);CHKERRQ(ierr);
  ierr = MatSeqAIJGetArray(AA,&da);CHKERRQ(ierr);
  ierr = MatGetRowIJ(AB,0,PETSC_FALSE,PETSC_FALSE,&nd,(const PetscInt**)&oi,(const PetscInt**)&oj,&done);CHKERRQ(ierr);
  ierr = MatSeqAIJGetArray(AB,&oa);CHKERRQ(ierr);

  ierr = PetscMalloc1(oi[5],&ooj);CHKERRQ(ierr);
  ierr = PetscArraycpy(ooj,oj,oi[5]);CHKERRQ(ierr);
  /* modify the column entries in the non-diagonal portion back to global numbering */
  for (i=0; i<oi[5]; i++) {
    ooj[i] = garray[ooj[i]];
  }

  ierr = MatCreateMPIAIJWithSplitArrays(PETSC_COMM_WORLD,5,5,PETSC_DETERMINE,PETSC_DETERMINE,di,dj,da,oi,ooj,oa,&B);CHKERRQ(ierr);
  ierr = MatSetUp(B);CHKERRQ(ierr);
  ierr = MatEqual(A,B,&equal);CHKERRQ(ierr);

  ierr = MatRestoreRowIJ(AA,0,PETSC_FALSE,PETSC_FALSE,&nd,(const PetscInt**)&di,(const PetscInt**)&dj,&done);CHKERRQ(ierr);
  ierr = MatSeqAIJRestoreArray(AA,&da);CHKERRQ(ierr);
  ierr = MatRestoreRowIJ(AB,0,PETSC_FALSE,PETSC_FALSE,&nd,(const PetscInt**)&oi,(const PetscInt**)&oj,&done);CHKERRQ(ierr);
  ierr = MatSeqAIJRestoreArray(AB,&oa);CHKERRQ(ierr);

  if (!equal) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Likely a bug in MatCreateMPIAIJWithSplitArrays()");

  /* Free spaces */
  ierr = PetscFree(ooj);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rctx);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = PetscFree(oj);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}


/*TEST

   test:
      nsize: 3

TEST*/
