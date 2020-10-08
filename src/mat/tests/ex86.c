static char help[] = "Testing MatCreateMPIMatConcatenateSeqMat().\n\n";

#include <petscmat.h>
int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  Mat            seqmat,mpimat;
  PetscMPIInt    rank;
  PetscScalar    value[3],*vals;
  PetscInt       i,col[3],n=5,bs=1;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-bs",&bs,NULL);CHKERRQ(ierr);

  /* Create seqaij matrices of size (n+rank) by n */
  ierr = MatCreate(PETSC_COMM_SELF,&seqmat);CHKERRQ(ierr);
  ierr = MatSetSizes(seqmat,(n+rank)*bs,PETSC_DECIDE,PETSC_DECIDE,n*bs);CHKERRQ(ierr);
  ierr = MatSetFromOptions(seqmat);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(seqmat,3*bs,NULL);CHKERRQ(ierr);
  ierr = MatSeqBAIJSetPreallocation(seqmat,bs,3,NULL);CHKERRQ(ierr);

  if (bs == 1) {
    value[0] = -1.0; value[1] = 2.0; value[2] = -1.0;
    for (i=1; i<n-1; i++) {
      col[0] = i-1; col[1] = i; col[2] = i+1;
      ierr   = MatSetValues(seqmat,1,&i,3,col,value,INSERT_VALUES);CHKERRQ(ierr);
    }
    i = n - 1; col[0] = n - 2; col[1] = n - 1;
    ierr = MatSetValues(seqmat,1,&i,2,col,value,INSERT_VALUES);CHKERRQ(ierr);

    i = 0; col[0] = 0; col[1] = 1; value[0] = 2.0; value[1] = -1.0;
    ierr = MatSetValues(seqmat,1,&i,2,col,value,INSERT_VALUES);CHKERRQ(ierr);
  } else {
    PetscInt *rows,*cols,j;
    ierr = PetscMalloc3(bs*bs,&vals,bs,&rows,bs,&cols);CHKERRQ(ierr);
    /* diagonal blocks */
    for (i=0; i<bs*bs; i++) vals[i] = 2.0;
    for (i=0; i<n*bs; i+=bs) {
      for (j=0; j<bs; j++) {rows[j] = i+j; cols[j] = i+j;}
      ierr = MatSetValues(seqmat,bs,rows,bs,cols,vals,INSERT_VALUES);CHKERRQ(ierr);
    }
    /* off-diagonal blocks */
    for (i=0; i<bs*bs; i++) vals[i] = -1.0;
    for (i=0; i<(n-1)*bs; i+=bs) {
      for (j=0; j<bs; j++) {rows[j] = i+j; cols[j] = i+bs+j;}
      ierr = MatSetValues(seqmat,bs,rows,bs,cols,vals,INSERT_VALUES);CHKERRQ(ierr);
    }

    ierr = PetscFree3(vals,rows,cols);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(seqmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(seqmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (!rank) {
    ierr = PetscPrintf(PETSC_COMM_SELF,"[%d] seqmat:\n",rank);CHKERRQ(ierr);
    ierr = MatView(seqmat,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  }

  ierr = MatCreateMPIMatConcatenateSeqMat(PETSC_COMM_WORLD,seqmat,PETSC_DECIDE,MAT_INITIAL_MATRIX,&mpimat);CHKERRQ(ierr);
  ierr = MatCreateMPIMatConcatenateSeqMat(PETSC_COMM_WORLD,seqmat,PETSC_DECIDE,MAT_REUSE_MATRIX,&mpimat);CHKERRQ(ierr);
  ierr = MatView(mpimat,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = MatDestroy(&seqmat);CHKERRQ(ierr);
  ierr = MatDestroy(&mpimat);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}



/*TEST

   test:
      nsize: 3

   test:
      suffix: 2
      nsize: 3
      args: -mat_type baij

   test:
      suffix: 3
      nsize: 3
      args: -mat_type baij -bs 2

TEST*/
