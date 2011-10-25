static char help[] = "Test sequential MatMatMult() and MatPtAP() for AIJ matrices.\n\n";

#include <petscmat.h>
#include <../src/mat/impls/aij/seq/aij.h> /*I "petscmat.h" I*/
#include <../src/mat/utils/freespace.h>
#include <petscbt.h>
#include <../src/mat/impls/dense/seq/dense.h> /*I "petscmat.h" I*/

#undef __FUNCT__  
#define __FUNCT__ "MatMultTransposeColoringApply"
PetscErrorCode  MatMultTransposeColoringApply(Mat B,Mat Btdense,MatMultTransposeColoring coloring)
{
  PetscErrorCode ierr;
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)B->data;
  Mat_SeqDense   *atdense = (Mat_SeqDense*)Btdense->data;
  PetscInt       m=Btdense->rmap->n,n=Btdense->cmap->n;

  PetscFunctionBegin;    
  PetscValidHeaderSpecific(B,MAT_CLASSID,1);
  PetscValidHeaderSpecific(Btdense,MAT_CLASSID,1);
  PetscValidHeaderSpecific(coloring,MAT_MULTTRANSPOSECOLORING_CLASSID,3);
 
  //printf("MatMultTransposeColoringApply Btdense: %d,%d\n",Btdense->rmap->n,Btdense->cmap->n);
  ierr = PetscMemzero(atdense->v,(m*n)*sizeof(MatScalar));CHKERRQ(ierr);

  PetscInt j,k,l,col,anz;
  for (k=0; k<coloring->ncolors; k++) { 
    for (l=0; l<coloring->ncolumns[k]; l++) { /* insert a row of B to a column of Btdense */
      col = coloring->columns[k][l];   // =row of B
      anz = a->i[col+1] - a->i[col];
      //printf("Brow %d, nz %d\n",col,anz);
      for (j=0; j<anz; j++){
        PetscInt  *atcol = a->j + a->i[col],brow,bcol;
        MatScalar *atval = a->a + a->i[col],*bval=atdense->v;
        brow = atcol[j]; bcol=k;
        bval[bcol*m+brow] = atval[j];
        //printf("  B(%d,%d) to Btdense (%d,%d)\n",col,atcol[j],atcol[j],k);
      }
    }
  }
  //ierr = MatView(Btdense,PETSC_VIEWER_STDOUT_WORLD);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv) {
  Mat            A,B,C,C_dense,C_sparse;
  PetscInt       I,J;
  PetscErrorCode ierr;
  MatScalar      one=1.0,val;

  PetscInitialize(&argc,&argv,(char *)0,help);
  /* Create A */
  ierr = MatCreate(PETSC_COMM_SELF,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,4,4,4,4);CHKERRQ(ierr);
  ierr = MatSetType(A,MATSEQAIJ);CHKERRQ(ierr);
  I=0; J=0; val=1.0; ierr = MatSetValues(A,1,&I,1,&J,&val,ADD_VALUES);CHKERRQ(ierr);
  I=1; J=3; val=2.0; ierr = MatSetValues(A,1,&I,1,&J,&val,ADD_VALUES);CHKERRQ(ierr);
  I=2; J=2; val=3.0; ierr = MatSetValues(A,1,&I,1,&J,&val,ADD_VALUES);CHKERRQ(ierr);
  I=3; J=0; val=4.0; ierr = MatSetValues(A,1,&I,1,&J,&val,ADD_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(A,"A_");CHKERRQ(ierr);
  ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,"\n");CHKERRQ(ierr);
 
  /* Create B */
  ierr = MatCreate(PETSC_COMM_SELF,&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,2,4,2,4);CHKERRQ(ierr);
  ierr = MatSetType(B,MATSEQAIJ);CHKERRQ(ierr);
  I=0; J=0; ierr = MatSetValues(B,1,&I,1,&J,&one,ADD_VALUES);CHKERRQ(ierr);
  I=0; J=1; ierr = MatSetValues(B,1,&I,1,&J,&one,ADD_VALUES);CHKERRQ(ierr);

  I=1; J=1; ierr = MatSetValues(B,1,&I,1,&J,&one,ADD_VALUES);CHKERRQ(ierr);
  I=1; J=2; ierr = MatSetValues(B,1,&I,1,&J,&one,ADD_VALUES);CHKERRQ(ierr);
  I=1; J=3; ierr = MatSetValues(B,1,&I,1,&J,&one,ADD_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(B,"B_");CHKERRQ(ierr);
  ierr = MatView(B,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,"\n");CHKERRQ(ierr);

  /* C = A*B^T */
  ierr = MatMatMultTranspose(A,B,MAT_INITIAL_MATRIX,2.0,&C);CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(C,"C_");CHKERRQ(ierr);
  ierr = MatView(C,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,"\n");CHKERRQ(ierr);
 
  /* Create MatMultTransposeColoring from symbolic C=A*B^T */
  MatMultTransposeColoring  matfdcoloring = 0;
  ISColoring                iscoloring;
  ierr = MatGetColoring(C,MATCOLORINGSL,&iscoloring);CHKERRQ(ierr); //modify to C!
  ierr = MatMultTransposeColoringCreate(C,iscoloring,&matfdcoloring);CHKERRQ(ierr);
  //ierr = MatFDColoringSetFromOptions(matfdcoloring);CHKERRQ(ierr);
  //ierr = MatFDColoringView((MatFDColoring)matfdcoloring,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  //ierr = MatFDColoringView(matfdcoloring,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);
  ierr = ISColoringDestroy(&iscoloring);CHKERRQ(ierr);

  /* Create Bt_dense */
  Mat      Bt_dense;
  PetscInt m,n;
  ierr = MatCreate(PETSC_COMM_WORLD,&Bt_dense);CHKERRQ(ierr);
  ierr = MatSetSizes(Bt_dense,A->cmap->n,matfdcoloring->ncolors,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = MatSetType(Bt_dense,MATDENSE);CHKERRQ(ierr);
  ierr = MatSeqDenseSetPreallocation(Bt_dense,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(Bt_dense,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Bt_dense,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatGetLocalSize(Bt_dense,&m,&n);CHKERRQ(ierr);
  printf("Bt_dense: %d,%d\n",m,n);

  /* Get Bt_dense by Apply MatMultTransposeColoring to B */
  ierr = MatMultTransposeColoringApply(B,Bt_dense,matfdcoloring);CHKERRQ(ierr);

  /* C_dense = A*Bt_dense */
  ierr = MatMatMult(A,Bt_dense,MAT_INITIAL_MATRIX,2.0,&C_dense); CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(C_dense,"C_dense_");CHKERRQ(ierr);
  //ierr = MatView(C_dense,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  //ierr = PetscPrintf(PETSC_COMM_SELF,"\n");CHKERRQ(ierr);

  /* Recover C from C_dense */
  PetscInt    k,l,row,col;
  PetscScalar *ca,*cval;
  ierr = MatGetLocalSize(C_dense,&m,&n);CHKERRQ(ierr);
  ierr = MatDuplicate(C,MAT_DO_NOT_COPY_VALUES,&C_sparse);CHKERRQ(ierr);
  ierr = MatGetLocalSize(C,&m,&n);CHKERRQ(ierr);
  ierr = MatGetArray(C_dense,&ca);CHKERRQ(ierr);
  cval = ca;
  for (k=0; k<matfdcoloring->ncolors; k++) { 
    for (l=0; l<matfdcoloring->nrows[k]; l++){
      row  = matfdcoloring->rows[k][l];             /* local row index */
      col  = matfdcoloring->columnsforrow[k][l];    /* global column index */
      ierr = MatSetValues(C_sparse,1,&row,1,&col,cval+row,INSERT_VALUES);CHKERRQ(ierr);
      //printf("MatSetValues() row %d, col %d\n",row,col);
    }
    cval += m;
  }
  ierr = MatRestoreArray(C_dense,&ca);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(C_sparse,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C_sparse,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(C_sparse,"C_sparse_");CHKERRQ(ierr);
  ierr = MatView(C_sparse,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,"\n");CHKERRQ(ierr);
 
  /* Free spaces */
  ierr = MatDestroy(&C_dense);CHKERRQ(ierr);
  ierr = MatDestroy(&C_sparse);CHKERRQ(ierr);
  ierr = MatDestroy(&Bt_dense);CHKERRQ(ierr);
  ierr = MatMultTransposeColoringDestroy(&matfdcoloring);CHKERRQ(ierr);
  ierr = MatDestroy(&C);CHKERRQ(ierr); 
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  PetscFinalize();
  return(0);
}
