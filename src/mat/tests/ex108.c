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
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  if (size != 1) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"This is a uniprocessor example only!");

  /* Create a aij matrix for checking */
  ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,5,5,2,NULL,&A);CHKERRQ(ierr);
  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&rctx);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rctx);CHKERRQ(ierr);

  k = 0;
  for (i=0; i<5; i++) {
    nz = asi[i+1] - asi[i];  /* length of i_th row of A */
    for (j=0; j<nz; j++) {
      ierr = PetscRandomGetValue(rctx,&asa[k]);CHKERRQ(ierr);
      ierr = MatSetValues(A,1,&i,1,&asj[k],&asa[k],INSERT_VALUES);CHKERRQ(ierr);
      ierr = MatSetValues(A,1,&i,1,&asj[k],&asa[k],INSERT_VALUES);CHKERRQ(ierr);
      if (i != asj[k]) { /* insert symmetric entry */
        ierr = MatSetValues(A,1,&asj[k],1,&i,&asa[k],INSERT_VALUES);CHKERRQ(ierr);
      }
      k++;
    }
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* Create a baij matrix using MatCreateSeqBAIJWithArrays() */
  ierr = MatGetRowIJ(A,0,PETSC_FALSE,PETSC_FALSE,&n,&ai,&aj,&flg);CHKERRQ(ierr);
  ierr = MatSeqAIJGetArray(A,&aa);CHKERRQ(ierr);
  /* WARNING: This sharing is dangerous if either A or B is later assembled */
  ierr = MatCreateSeqBAIJWithArrays(PETSC_COMM_SELF,1,5,5,(PetscInt*)ai,(PetscInt*)aj,aa,&B);CHKERRQ(ierr);
  ierr = MatSeqAIJRestoreArray(A,&aa);CHKERRQ(ierr);
  ierr = MatRestoreRowIJ(A,0,PETSC_FALSE,PETSC_FALSE,&n,&ai,&aj,&flg);CHKERRQ(ierr);
  ierr = MatMultEqual(A,B,10,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_NOTSAMETYPE,"MatMult(A,B) are NOT equal");

  /* Create a sbaij matrix using MatCreateSeqSBAIJWithArrays() */
  ierr = MatCreateSeqSBAIJWithArrays(PETSC_COMM_SELF,1,5,5,asi,asj,asa,&As);CHKERRQ(ierr);
  ierr = MatMultEqual(A,As,10,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_NOTSAMETYPE,"MatMult(A,As) are NOT equal");

  /* Free spaces */
  ierr = PetscRandomDestroy(&rctx);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = MatDestroy(&As);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}
