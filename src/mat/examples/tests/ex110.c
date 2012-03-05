static char help[] = "Testing MatCreateMPIAIJWithSplitArrays().\n\n";

#include <petscmat.h>
#include <../src/mat/impls/aij/mpi/mpiaij.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv) {
  Mat            A,B;
  PetscInt       i,j,column;
  PetscInt       *di,*dj,*oi,*oj;
  PetscScalar    *oa,*da,value; 
  PetscRandom    rctx;
  PetscErrorCode ierr;
  PetscBool      equal;
  Mat_SeqAIJ     *daij,*oaij;
  Mat_MPIAIJ     *Ampiaij;
  PetscMPIInt    size,rank;

  PetscInitialize(&argc,&argv,(char *)0,help);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  if (size == 1) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,"Must run with 2 or more processes");CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

  /* Create a mpiaij matrix for checking */
  ierr = MatCreateAIJ(PETSC_COMM_WORLD,5,5,PETSC_DECIDE,PETSC_DECIDE,0,PETSC_NULL,0,PETSC_NULL,&A);CHKERRQ(ierr);
  ierr = MatSetOption(A,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_FALSE);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);
  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&rctx);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rctx);CHKERRQ(ierr);
 
  for (i=5*rank; i<5*rank+5; i++) {
    for (j=0; j<5*size; j++){
      ierr   = PetscRandomGetValue(rctx,&value);CHKERRQ(ierr);
      column = (PetscInt) (5*size*PetscRealPart(value));
      ierr   = PetscRandomGetValue(rctx,&value);CHKERRQ(ierr);
      ierr   = MatSetValues(A,1,&i,1,&column,&value,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  Ampiaij = (Mat_MPIAIJ*) A->data;
  daij    = (Mat_SeqAIJ*) Ampiaij->A->data;
  oaij    = (Mat_SeqAIJ*) Ampiaij->B->data;
  
  di      = daij->i;
  dj      = daij->j;
  da      = daij->a;

  oi      = oaij->i;
  oa      = oaij->a;
  ierr    = PetscMalloc(oi[5]*sizeof(PetscInt),&oj);CHKERRQ(ierr);
  ierr    = PetscMemcpy(oj,oaij->j,oi[5]*sizeof(PetscInt));CHKERRQ(ierr);
  /* modify the column entries in the non-diagonal portion back to global numbering */
  for (i=0; i<oi[5]; i++) {
    oj[i] = Ampiaij->garray[oj[i]];
  }

  ierr = MatCreateMPIAIJWithSplitArrays(PETSC_COMM_WORLD,5,5,PETSC_DETERMINE,PETSC_DETERMINE,di,dj,da,oi,oj,oa,&B);CHKERRQ(ierr);
  ierr = MatSetUp(B);CHKERRQ(ierr);
  ierr = MatEqual(A,B,&equal);CHKERRQ(ierr);

  if (!equal) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Likely a bug in MatCreateMPIAIJWithSplitArrays()");

  /* Free spaces */
  ierr = PetscRandomDestroy(&rctx);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = PetscFree(oj);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return(0);
}
