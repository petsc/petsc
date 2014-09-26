static char help[] = "Testing MatCreateMPIAIJConcatenateSeqAIJ() and MatCreateMPIBAIJConcatenateSeqBAIJ().\n\n";

#include <petscmat.h>
#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  Mat            seqaijmat,mpiaijmat;
  PetscMPIInt    rank;
  PetscScalar    value[3];
  PetscInt       i,col[3],n=10,bs=1;
  PetscBool      isAIJ,isBAIJ;
  
  PetscInitialize(&argc,&argv,(char*)0,help);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,"-bs",&bs,NULL);CHKERRQ(ierr);

  /* Create seqaij matrices of size (n+rank) by n */
  ierr = MatCreate(PETSC_COMM_SELF,&seqaijmat);CHKERRQ(ierr);
  ierr = MatSetSizes(seqaijmat,(n+rank)*bs,PETSC_DECIDE,PETSC_DECIDE,n*bs);CHKERRQ(ierr);
  ierr = MatSetFromOptions(seqaijmat);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(seqaijmat,3,NULL);CHKERRQ(ierr);
  ierr = MatSeqBAIJSetPreallocation(seqaijmat,1,3,NULL);CHKERRQ(ierr);

  if (bs == 1) {
    value[0] = -1.0; value[1] = 2.0; value[2] = -1.0;
    for (i=1; i<n-1; i++) {
      col[0] = i-1; col[1] = i; col[2] = i+1;
      ierr   = MatSetValues(seqaijmat,1,&i,3,col,value,INSERT_VALUES);CHKERRQ(ierr);
    }
    i = n - 1; col[0] = n - 2; col[1] = n - 1;
    ierr = MatSetValues(seqaijmat,1,&i,2,col,value,INSERT_VALUES);CHKERRQ(ierr);

    i = 0; col[0] = 0; col[1] = 1; value[0] = 2.0; value[1] = -1.0;
    ierr = MatSetValues(seqaijmat,1,&i,2,col,value,INSERT_VALUES);CHKERRQ(ierr);
  } else {
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Not supported yet");
  }
  ierr = MatAssemblyBegin(seqaijmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(seqaijmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (rank == 1) {
    printf("[%d] seqmat:\n",rank);
    ierr = MatView(seqaijmat,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  }

  ierr = PetscObjectTypeCompare((PetscObject)seqaijmat,MATSEQAIJ,&isAIJ);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)seqaijmat,MATSEQBAIJ,&isBAIJ);CHKERRQ(ierr);
  if (isAIJ) {
    /* Concatenate seqaij matrices into a single mpiaij matrix */
    ierr = MatCreateMPIAIJConcatenateSeqAIJ(PETSC_COMM_WORLD,seqaijmat,PETSC_DECIDE,MAT_INITIAL_MATRIX,&mpiaijmat);CHKERRQ(ierr);
    ierr = MatCreateMPIAIJConcatenateSeqAIJ(PETSC_COMM_WORLD,seqaijmat,PETSC_DECIDE,MAT_REUSE_MATRIX,&mpiaijmat);CHKERRQ(ierr);
  } else if (isBAIJ){
    ierr = MatCreateMPIBAIJConcatenateSeqBAIJ(PETSC_COMM_WORLD,seqaijmat,PETSC_DECIDE,MAT_INITIAL_MATRIX,&mpiaijmat);CHKERRQ(ierr);
    ierr = MatCreateMPIAIJConcatenateSeqAIJ(PETSC_COMM_WORLD,seqaijmat,PETSC_DECIDE,MAT_REUSE_MATRIX,&mpiaijmat);CHKERRQ(ierr);
  } else {
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Not supported yet");
  }
  ierr = MatView(mpiaijmat,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = MatDestroy(&seqaijmat);CHKERRQ(ierr);
  ierr = MatDestroy(&mpiaijmat);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}

