static char help[] = "Testing MatCreateSeqBAIJWithArrays() and MatCreateSeqSBAIJWithArrays().\n\n";

#include <petscmat.h>

int main(int argc,char **argv)
{
  Mat            A,B,As;
  const PetscInt *ai,*aj;
  PetscInt       i,j,k,nz,n,asi[]={0,2,3,4,6,7};
  PetscInt       asj[]={0,4,1,2,3,4,4};
  PetscScalar    asa[7],*aa;
  PetscRandom    rctx;
  PetscErrorCode ierr;
  PetscMPIInt    size;
  PetscBool      flg;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheckFalse(size != 1,PETSC_COMM_WORLD,PETSC_ERR_SUP,"This is a uniprocessor example only!");

  /* Create a aij matrix for checking */
  CHKERRQ(MatCreateSeqAIJ(PETSC_COMM_SELF,5,5,2,NULL,&A));
  CHKERRQ(PetscRandomCreate(PETSC_COMM_WORLD,&rctx));
  CHKERRQ(PetscRandomSetFromOptions(rctx));

  k = 0;
  for (i=0; i<5; i++) {
    nz = asi[i+1] - asi[i];  /* length of i_th row of A */
    for (j=0; j<nz; j++) {
      CHKERRQ(PetscRandomGetValue(rctx,&asa[k]));
      CHKERRQ(MatSetValues(A,1,&i,1,&asj[k],&asa[k],INSERT_VALUES));
      CHKERRQ(MatSetValues(A,1,&i,1,&asj[k],&asa[k],INSERT_VALUES));
      if (i != asj[k]) { /* insert symmetric entry */
        CHKERRQ(MatSetValues(A,1,&asj[k],1,&i,&asa[k],INSERT_VALUES));
      }
      k++;
    }
  }
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  /* Create a baij matrix using MatCreateSeqBAIJWithArrays() */
  CHKERRQ(MatGetRowIJ(A,0,PETSC_FALSE,PETSC_FALSE,&n,&ai,&aj,&flg));
  CHKERRQ(MatSeqAIJGetArray(A,&aa));
  /* WARNING: This sharing is dangerous if either A or B is later assembled */
  CHKERRQ(MatCreateSeqBAIJWithArrays(PETSC_COMM_SELF,1,5,5,(PetscInt*)ai,(PetscInt*)aj,aa,&B));
  CHKERRQ(MatSeqAIJRestoreArray(A,&aa));
  CHKERRQ(MatRestoreRowIJ(A,0,PETSC_FALSE,PETSC_FALSE,&n,&ai,&aj,&flg));
  CHKERRQ(MatMultEqual(A,B,10,&flg));
  PetscCheckFalse(!flg,PETSC_COMM_SELF,PETSC_ERR_ARG_NOTSAMETYPE,"MatMult(A,B) are NOT equal");

  /* Create a sbaij matrix using MatCreateSeqSBAIJWithArrays() */
  CHKERRQ(MatCreateSeqSBAIJWithArrays(PETSC_COMM_SELF,1,5,5,asi,asj,asa,&As));
  CHKERRQ(MatMultEqual(A,As,10,&flg));
  PetscCheckFalse(!flg,PETSC_COMM_SELF,PETSC_ERR_ARG_NOTSAMETYPE,"MatMult(A,As) are NOT equal");

  /* Free spaces */
  CHKERRQ(PetscRandomDestroy(&rctx));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&B));
  CHKERRQ(MatDestroy(&As));
  ierr = PetscFinalize();
  return ierr;
}
