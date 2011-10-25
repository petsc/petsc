static char help[] = "Test sequential MatMatMult() and MatPtAP() for AIJ matrices.\n\n";

#include <petscmat.h>
#include <../src/mat/impls/aij/seq/aij.h> /*I "petscmat.h" I*/
#include <../src/mat/utils/freespace.h>
#include <petscbt.h>
#include <../src/mat/impls/dense/seq/dense.h> /*I "petscmat.h" I*/

#if defined(MV)
#undef __FUNCT__  
#define __FUNCT__ "MatMultTransposeColoringDestroy"
/*@
    MatMultTransposeColoringDestroy - Destroys a coloring context for matrix product C=A*B^T that was created
    via MatMultTransposeColoringCreate().

    Collective on MatMultTransposeColoring

    Input Parameter:
.   c - coloring context

    Level: intermediate

.seealso: MatMultTransposeColoringCreate()
@*/
PetscErrorCode  MatMultTransposeColoringDestroy(MatMultTransposeColoring *c)
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  if (!*c) PetscFunctionReturn(0);
  if (--((PetscObject)(*c))->refct > 0) {*c = 0; PetscFunctionReturn(0);}

  for (i=0; i<(*c)->ncolors; i++) {
    ierr = PetscFree((*c)->columns[i]);CHKERRQ(ierr);
    ierr = PetscFree((*c)->rows[i]);CHKERRQ(ierr);
    ierr = PetscFree((*c)->columnsforrow[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree((*c)->ncolumns);CHKERRQ(ierr);
  ierr = PetscFree((*c)->columns);CHKERRQ(ierr);
  ierr = PetscFree((*c)->nrows);CHKERRQ(ierr);
  ierr = PetscFree((*c)->rows);CHKERRQ(ierr);
  ierr = PetscFree((*c)->columnsforrow);CHKERRQ(ierr);
  ierr = PetscHeaderDestroy(c);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif

#undef __FUNCT__
#define __FUNCT__ "MatMultTransposeColoringCreate_SeqAIJ"
/*@
   MatMultTransposeColoringCreate - Creates a matrix coloring context for finite difference 
   computation of Jacobians.

   Collective on Mat

   Input Parameters:
+  mat - the matrix containing the nonzero structure of the Jacobian
-  iscoloring - the coloring of the matrix; usually obtained with MatGetColoring() or DMGetColoring()

    Output Parameter:
.   color - the new coloring context
   
    Level: intermediate

.seealso: MatFDColoringDestroy(),SNESDefaultComputeJacobianColor(), ISColoringCreate(),
          MatFDColoringSetFunction(), MatFDColoringSetFromOptions(), MatFDColoringApply(),
          MatFDColoringView(), MatFDColoringSetParameters(), MatGetColoring(), DMGetColoring()
@*/
PetscErrorCode MatMultTransposeColoringCreate_SeqAIJ(Mat mat,ISColoring iscoloring,MatMultTransposeColoring *color)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* MatFDColoringCreate() */
  MatMultTransposeColoring  c;
  MPI_Comm       comm;
  ierr = PetscObjectGetComm((PetscObject)mat,&comm);CHKERRQ(ierr);
  ierr = PetscHeaderCreate(c,_p_MatMultTransposeColoring,int,MAT_FDCOLORING_CLASSID,0,"MatMultTransposeColoring","Jacobian computation via finite differences with coloring","Mat",comm,MatFDColoringDestroy,MatFDColoringView);CHKERRQ(ierr);

  c->ctype = iscoloring->ctype;
  // Modified MatFDColoringCreate_SeqAIJ(Mat mat,ISColoring iscoloring,MatFDColoring c)
  //--------------------------------
  PetscInt       i,n,nrows,N,j,k,m,*rows,*ci,*cj,ncols,col;
  const PetscInt *is;
  PetscInt       nis = iscoloring->n,*rowhit,*columnsforrow,bs = 1;
  IS             *isa;
  PetscBool      done;
  PetscBool      flg1,flg2;

  PetscFunctionBegin;
  if (!mat->assembled) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Matrix must be assembled by calls to MatAssemblyBegin/End();");

  ierr = ISColoringGetIS(iscoloring,PETSC_IGNORE,&isa);CHKERRQ(ierr);
  /* this is ugly way to get blocksize but cannot call MatGetBlockSize() because AIJ can have bs > 1 */
  ierr = PetscTypeCompare((PetscObject)mat,MATSEQBAIJ,&flg1);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)mat,MATMPIBAIJ,&flg2);CHKERRQ(ierr);
  if (flg1 || flg2) {
    ierr = MatGetBlockSize(mat,&bs);CHKERRQ(ierr);
  }

  N          = mat->cmap->N/bs;
  c->M       = mat->rmap->N/bs;  /* set total rows, columns and local rows */
  c->N       = mat->cmap->N/bs;
  c->m       = mat->rmap->N/bs;
  c->rstart  = 0;

  c->ncolors = nis;
  ierr       = PetscMalloc(nis*sizeof(PetscInt),&c->ncolumns);CHKERRQ(ierr);
  ierr       = PetscMalloc(nis*sizeof(PetscInt*),&c->columns);CHKERRQ(ierr); 
  ierr       = PetscMalloc(c->m*sizeof(PetscInt),&c->nrows);CHKERRQ(ierr); // square matrix?
  ierr       = PetscMalloc(c->m*sizeof(PetscInt*),&c->rows);CHKERRQ(ierr); // square matrix?
  ierr       = PetscMalloc(c->m*sizeof(PetscInt*),&c->columnsforrow);CHKERRQ(ierr); // nis -> m

  ierr = MatGetColumnIJ(mat,0,PETSC_FALSE,PETSC_FALSE,&ncols,&ci,&cj,&done);CHKERRQ(ierr);
  if (!done) SETERRQ1(((PetscObject)mat)->comm,PETSC_ERR_SUP,"MatGetColumnIJ() not supported for matrix type %s",((PetscObject)mat)->type_name);

  //ierr = PetscMalloc((N+1)*sizeof(PetscInt),&rowhit);CHKERRQ(ierr);
  //ierr = PetscMalloc((N+1)*sizeof(PetscInt),&columnsforrow);CHKERRQ(ierr);
  ierr = PetscMalloc((c->m+1)*sizeof(PetscInt),&rowhit);CHKERRQ(ierr);
  ierr = PetscMalloc((c->m+1)*sizeof(PetscInt),&columnsforrow);CHKERRQ(ierr);

  for (i=0; i<nis; i++) {
    ierr = ISGetLocalSize(isa[i],&n);CHKERRQ(ierr);
    ierr = ISGetIndices(isa[i],&is);CHKERRQ(ierr);
    c->ncolumns[i] = n;
    if (n) {
      ierr = PetscMalloc(n*sizeof(PetscInt),&c->columns[i]);CHKERRQ(ierr);
      ierr = PetscMemcpy(c->columns[i],is,n*sizeof(PetscInt));CHKERRQ(ierr);
    } else {
      c->columns[i]  = 0;
    }

      /* fast, crude version requires O(N*N) work */
      //ierr = PetscMemzero(rowhit,N*sizeof(PetscInt));CHKERRQ(ierr);
      ierr = PetscMemzero(rowhit,c->m*sizeof(PetscInt));CHKERRQ(ierr);
      /* loop over columns*/
      for (j=0; j<n; j++) {
        col  = is[j];
        rows = cj + ci[col]; 
        m    = ci[col+1] - ci[col];
        /* loop over columns marking them in rowhit */
        for (k=0; k<m; k++) {
          rowhit[*rows++] = col + 1;
        }
      }
      /* count the number of hits */
      nrows = 0;
      //for (j=0; j<N; j++) {
      for (j=0; j<c->m; j++) {
        if (rowhit[j]) nrows++;
      }
      c->nrows[i] = nrows;
      ierr        = PetscMalloc((nrows+1)*sizeof(PetscInt),&c->rows[i]);CHKERRQ(ierr);
      ierr        = PetscMalloc((nrows+1)*sizeof(PetscInt),&c->columnsforrow[i]);CHKERRQ(ierr);
      nrows       = 0;
      //for (j=0; j<N; j++) {
      for (j=0; j<c->m; j++) {
        if (rowhit[j]) {
          c->rows[i][nrows]          = j;
          c->columnsforrow[i][nrows] = rowhit[j] - 1;
          nrows++;
        }
      }
    ierr = ISRestoreIndices(isa[i],&is);CHKERRQ(ierr);  
  }
  ierr = MatRestoreColumnIJ(mat,0,PETSC_FALSE,PETSC_FALSE,&ncols,&ci,&cj,&done);CHKERRQ(ierr);

  ierr = PetscFree(rowhit);CHKERRQ(ierr);
  ierr = PetscFree(columnsforrow);CHKERRQ(ierr);
  ierr = ISColoringRestoreIS(iscoloring,&isa);CHKERRQ(ierr);
  // --------end of Modified MatFDColoringCreate_SeqAIJ()-----------------
  *color = c;
  //ierr = MatDestroy(&mat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

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
  PetscValidHeaderSpecific(coloring,MAT_FDCOLORING_CLASSID,3);
 
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
  ISColoring            iscoloring;
  ierr = MatGetColoring(C,MATCOLORINGSL,&iscoloring);CHKERRQ(ierr); //modify to C!
  ierr = MatMultTransposeColoringCreate_SeqAIJ(C,iscoloring,&matfdcoloring);CHKERRQ(ierr);
  //ierr = MatFDColoringSetFromOptions(matfdcoloring);CHKERRQ(ierr);
  ierr = MatFDColoringView((MatFDColoring)matfdcoloring,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
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
