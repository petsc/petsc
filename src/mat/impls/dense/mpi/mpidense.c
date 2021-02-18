
/*
   Basic functions for basic parallel dense matrices.
*/

#include <../src/mat/impls/dense/mpi/mpidense.h>    /*I   "petscmat.h"  I*/
#include <../src/mat/impls/aij/mpi/mpiaij.h>
#include <petscblaslapack.h>

/*@

      MatDenseGetLocalMatrix - For a MATMPIDENSE or MATSEQDENSE matrix returns the sequential
              matrix that represents the operator. For sequential matrices it returns itself.

    Input Parameter:
.      A - the Seq or MPI dense matrix

    Output Parameter:
.      B - the inner matrix

    Level: intermediate

@*/
PetscErrorCode MatDenseGetLocalMatrix(Mat A,Mat *B)
{
  Mat_MPIDense   *mat = (Mat_MPIDense*)A->data;
  PetscErrorCode ierr;
  PetscBool      flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidPointer(B,2);
  ierr = PetscObjectBaseTypeCompare((PetscObject)A,MATMPIDENSE,&flg);CHKERRQ(ierr);
  if (flg) *B = mat->A;
  else {
    ierr = PetscObjectBaseTypeCompare((PetscObject)A,MATSEQDENSE,&flg);CHKERRQ(ierr);
    if (!flg) SETERRQ1(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Not for matrix type %s",((PetscObject)A)->type_name);
    *B = A;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatCopy_MPIDense(Mat A, Mat B, MatStructure s)
{
  Mat_MPIDense   *Amat = (Mat_MPIDense*)A->data;
  Mat_MPIDense   *Bmat = (Mat_MPIDense*)B->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCopy(Amat->A,Bmat->A,s);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatGetRow_MPIDense(Mat A,PetscInt row,PetscInt *nz,PetscInt **idx,PetscScalar **v)
{
  Mat_MPIDense   *mat = (Mat_MPIDense*)A->data;
  PetscErrorCode ierr;
  PetscInt       lrow,rstart = A->rmap->rstart,rend = A->rmap->rend;

  PetscFunctionBegin;
  if (row < rstart || row >= rend) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"only local rows");
  lrow = row - rstart;
  ierr = MatGetRow(mat->A,lrow,nz,(const PetscInt**)idx,(const PetscScalar**)v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatRestoreRow_MPIDense(Mat A,PetscInt row,PetscInt *nz,PetscInt **idx,PetscScalar **v)
{
  Mat_MPIDense   *mat = (Mat_MPIDense*)A->data;
  PetscErrorCode ierr;
  PetscInt       lrow,rstart = A->rmap->rstart,rend = A->rmap->rend;

  PetscFunctionBegin;
  if (row < rstart || row >= rend) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"only local rows");
  lrow = row - rstart;
  ierr = MatRestoreRow(mat->A,lrow,nz,(const PetscInt**)idx,(const PetscScalar**)v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode  MatGetDiagonalBlock_MPIDense(Mat A,Mat *a)
{
  Mat_MPIDense   *mdn = (Mat_MPIDense*)A->data;
  PetscErrorCode ierr;
  PetscInt       m = A->rmap->n,rstart = A->rmap->rstart;
  PetscScalar    *array;
  MPI_Comm       comm;
  PetscBool      flg;
  Mat            B;

  PetscFunctionBegin;
  ierr = MatHasCongruentLayouts(A,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Only square matrices supported.");
  ierr = PetscObjectQuery((PetscObject)A,"DiagonalBlock",(PetscObject*)&B);CHKERRQ(ierr);
  if (!B) { /* This should use MatDenseGetSubMatrix (not create), but we would need a call like MatRestoreDiagonalBlock */

    ierr = PetscObjectTypeCompare((PetscObject)mdn->A,MATSEQDENSECUDA,&flg);CHKERRQ(ierr);
    if (flg) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Not coded for %s. Send an email to petsc-dev@mcs.anl.gov to request this feature",MATSEQDENSECUDA);
    ierr = PetscObjectGetComm((PetscObject)(mdn->A),&comm);CHKERRQ(ierr);
    ierr = MatCreate(comm,&B);CHKERRQ(ierr);
    ierr = MatSetSizes(B,m,m,m,m);CHKERRQ(ierr);
    ierr = MatSetType(B,((PetscObject)mdn->A)->type_name);CHKERRQ(ierr);
    ierr = MatDenseGetArrayRead(mdn->A,(const PetscScalar**)&array);CHKERRQ(ierr);
    ierr = MatSeqDenseSetPreallocation(B,array+m*rstart);CHKERRQ(ierr);
    ierr = MatDenseRestoreArrayRead(mdn->A,(const PetscScalar**)&array);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject)A,"DiagonalBlock",(PetscObject)B);CHKERRQ(ierr);
    *a   = B;
    ierr = MatDestroy(&B);CHKERRQ(ierr);
  } else *a = B;
  PetscFunctionReturn(0);
}

PetscErrorCode MatSetValues_MPIDense(Mat mat,PetscInt m,const PetscInt idxm[],PetscInt n,const PetscInt idxn[],const PetscScalar v[],InsertMode addv)
{
  Mat_MPIDense   *A = (Mat_MPIDense*)mat->data;
  PetscErrorCode ierr;
  PetscInt       i,j,rstart = mat->rmap->rstart,rend = mat->rmap->rend,row;
  PetscBool      roworiented = A->roworiented;

  PetscFunctionBegin;
  for (i=0; i<m; i++) {
    if (idxm[i] < 0) continue;
    if (idxm[i] >= mat->rmap->N) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Row too large");
    if (idxm[i] >= rstart && idxm[i] < rend) {
      row = idxm[i] - rstart;
      if (roworiented) {
        ierr = MatSetValues(A->A,1,&row,n,idxn,v+i*n,addv);CHKERRQ(ierr);
      } else {
        for (j=0; j<n; j++) {
          if (idxn[j] < 0) continue;
          if (idxn[j] >= mat->cmap->N) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Column too large");
          ierr = MatSetValues(A->A,1,&row,1,&idxn[j],v+i+j*m,addv);CHKERRQ(ierr);
        }
      }
    } else if (!A->donotstash) {
      mat->assembled = PETSC_FALSE;
      if (roworiented) {
        ierr = MatStashValuesRow_Private(&mat->stash,idxm[i],n,idxn,v+i*n,PETSC_FALSE);CHKERRQ(ierr);
      } else {
        ierr = MatStashValuesCol_Private(&mat->stash,idxm[i],n,idxn,v+i,m,PETSC_FALSE);CHKERRQ(ierr);
      }
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatGetValues_MPIDense(Mat mat,PetscInt m,const PetscInt idxm[],PetscInt n,const PetscInt idxn[],PetscScalar v[])
{
  Mat_MPIDense   *mdn = (Mat_MPIDense*)mat->data;
  PetscErrorCode ierr;
  PetscInt       i,j,rstart = mat->rmap->rstart,rend = mat->rmap->rend,row;

  PetscFunctionBegin;
  for (i=0; i<m; i++) {
    if (idxm[i] < 0) continue; /* SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Negative row"); */
    if (idxm[i] >= mat->rmap->N) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Row too large");
    if (idxm[i] >= rstart && idxm[i] < rend) {
      row = idxm[i] - rstart;
      for (j=0; j<n; j++) {
        if (idxn[j] < 0) continue; /* SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Negative column"); */
        if (idxn[j] >= mat->cmap->N) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Column too large");
        ierr = MatGetValues(mdn->A,1,&row,1,&idxn[j],v+i*n+j);CHKERRQ(ierr);
      }
    } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Only local values currently supported");
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseGetLDA_MPIDense(Mat A,PetscInt *lda)
{
  Mat_MPIDense   *a = (Mat_MPIDense*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatDenseGetLDA(a->A,lda);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseSetLDA_MPIDense(Mat A,PetscInt lda)
{
  Mat_MPIDense   *a = (Mat_MPIDense*)A->data;
  PetscBool      iscuda;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!a->A) {
    if (a->matinuse) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ORDER,"Need to call MatDenseRestoreSubMatrix() first");
    ierr = PetscLayoutSetUp(A->rmap);CHKERRQ(ierr);
    ierr = PetscLayoutSetUp(A->cmap);CHKERRQ(ierr);
    ierr = MatCreate(PETSC_COMM_SELF,&a->A);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)A,(PetscObject)a->A);CHKERRQ(ierr);
    ierr = MatSetSizes(a->A,A->rmap->n,A->cmap->N,A->rmap->n,A->cmap->N);CHKERRQ(ierr);
    ierr = PetscObjectTypeCompare((PetscObject)A,MATMPIDENSECUDA,&iscuda);CHKERRQ(ierr);
    ierr = MatSetType(a->A,iscuda ? MATSEQDENSECUDA : MATSEQDENSE);CHKERRQ(ierr);
  }
  ierr = MatDenseSetLDA(a->A,lda);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseGetArray_MPIDense(Mat A,PetscScalar **array)
{
  Mat_MPIDense   *a = (Mat_MPIDense*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (a->matinuse) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ORDER,"Need to call MatDenseRestoreSubMatrix() first");
  ierr = MatDenseGetArray(a->A,array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseGetArrayRead_MPIDense(Mat A,const PetscScalar **array)
{
  Mat_MPIDense   *a = (Mat_MPIDense*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (a->matinuse) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ORDER,"Need to call MatDenseRestoreSubMatrix() first");
  ierr = MatDenseGetArrayRead(a->A,array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseGetArrayWrite_MPIDense(Mat A,PetscScalar **array)
{
  Mat_MPIDense   *a = (Mat_MPIDense*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (a->matinuse) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ORDER,"Need to call MatDenseRestoreSubMatrix() first");
  ierr = MatDenseGetArrayWrite(a->A,array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDensePlaceArray_MPIDense(Mat A,const PetscScalar *array)
{
  Mat_MPIDense   *a = (Mat_MPIDense*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (a->vecinuse) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ORDER,"Need to call MatDenseRestoreColumnVec() first");
  if (a->matinuse) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ORDER,"Need to call MatDenseRestoreSubMatrix() first");
  ierr = MatDensePlaceArray(a->A,array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseResetArray_MPIDense(Mat A)
{
  Mat_MPIDense   *a = (Mat_MPIDense*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (a->vecinuse) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ORDER,"Need to call MatDenseRestoreColumnVec() first");
  if (a->matinuse) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ORDER,"Need to call MatDenseRestoreSubMatrix() first");
  ierr = MatDenseResetArray(a->A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseReplaceArray_MPIDense(Mat A,const PetscScalar *array)
{
  Mat_MPIDense   *a = (Mat_MPIDense*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (a->vecinuse) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ORDER,"Need to call MatDenseRestoreColumnVec() first");
  if (a->matinuse) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ORDER,"Need to call MatDenseRestoreSubMatrix() first");
  ierr = MatDenseReplaceArray(a->A,array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatCreateSubMatrix_MPIDense(Mat A,IS isrow,IS iscol,MatReuse scall,Mat *B)
{
  Mat_MPIDense      *mat  = (Mat_MPIDense*)A->data,*newmatd;
  PetscErrorCode    ierr;
  PetscInt          lda,i,j,rstart,rend,nrows,ncols,Ncols,nlrows,nlcols;
  const PetscInt    *irow,*icol;
  const PetscScalar *v;
  PetscScalar       *bv;
  Mat               newmat;
  IS                iscol_local;
  MPI_Comm          comm_is,comm_mat;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)A,&comm_mat);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)iscol,&comm_is);CHKERRQ(ierr);
  if (comm_mat != comm_is) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_NOTSAMECOMM,"IS communicator must match matrix communicator");

  ierr = ISAllGather(iscol,&iscol_local);CHKERRQ(ierr);
  ierr = ISGetIndices(isrow,&irow);CHKERRQ(ierr);
  ierr = ISGetIndices(iscol_local,&icol);CHKERRQ(ierr);
  ierr = ISGetLocalSize(isrow,&nrows);CHKERRQ(ierr);
  ierr = ISGetLocalSize(iscol,&ncols);CHKERRQ(ierr);
  ierr = ISGetSize(iscol,&Ncols);CHKERRQ(ierr); /* global number of columns, size of iscol_local */

  /* No parallel redistribution currently supported! Should really check each index set
     to comfirm that it is OK.  ... Currently supports only submatrix same partitioning as
     original matrix! */

  ierr = MatGetLocalSize(A,&nlrows,&nlcols);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(A,&rstart,&rend);CHKERRQ(ierr);

  /* Check submatrix call */
  if (scall == MAT_REUSE_MATRIX) {
    /* SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Reused submatrix wrong size"); */
    /* Really need to test rows and column sizes! */
    newmat = *B;
  } else {
    /* Create and fill new matrix */
    ierr = MatCreate(PetscObjectComm((PetscObject)A),&newmat);CHKERRQ(ierr);
    ierr = MatSetSizes(newmat,nrows,ncols,PETSC_DECIDE,Ncols);CHKERRQ(ierr);
    ierr = MatSetType(newmat,((PetscObject)A)->type_name);CHKERRQ(ierr);
    ierr = MatMPIDenseSetPreallocation(newmat,NULL);CHKERRQ(ierr);
  }

  /* Now extract the data pointers and do the copy, column at a time */
  newmatd = (Mat_MPIDense*)newmat->data;
  ierr = MatDenseGetArray(newmatd->A,&bv);CHKERRQ(ierr);
  ierr = MatDenseGetArrayRead(mat->A,&v);CHKERRQ(ierr);
  ierr = MatDenseGetLDA(mat->A,&lda);CHKERRQ(ierr);
  for (i=0; i<Ncols; i++) {
    const PetscScalar *av = v + lda*icol[i];
    for (j=0; j<nrows; j++) {
      *bv++ = av[irow[j] - rstart];
    }
  }
  ierr = MatDenseRestoreArrayRead(mat->A,&v);CHKERRQ(ierr);
  ierr = MatDenseRestoreArray(newmatd->A,&bv);CHKERRQ(ierr);

  /* Assemble the matrices so that the correct flags are set */
  ierr = MatAssemblyBegin(newmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(newmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* Free work space */
  ierr = ISRestoreIndices(isrow,&irow);CHKERRQ(ierr);
  ierr = ISRestoreIndices(iscol_local,&icol);CHKERRQ(ierr);
  ierr = ISDestroy(&iscol_local);CHKERRQ(ierr);
  *B   = newmat;
  PetscFunctionReturn(0);
}

PetscErrorCode MatDenseRestoreArray_MPIDense(Mat A,PetscScalar **array)
{
  Mat_MPIDense   *a = (Mat_MPIDense*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatDenseRestoreArray(a->A,array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatDenseRestoreArrayRead_MPIDense(Mat A,const PetscScalar **array)
{
  Mat_MPIDense   *a = (Mat_MPIDense*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatDenseRestoreArrayRead(a->A,array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatDenseRestoreArrayWrite_MPIDense(Mat A,PetscScalar **array)
{
  Mat_MPIDense   *a = (Mat_MPIDense*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatDenseRestoreArrayWrite(a->A,array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatAssemblyBegin_MPIDense(Mat mat,MatAssemblyType mode)
{
  Mat_MPIDense   *mdn = (Mat_MPIDense*)mat->data;
  PetscErrorCode ierr;
  PetscInt       nstash,reallocs;

  PetscFunctionBegin;
  if (mdn->donotstash || mat->nooffprocentries) PetscFunctionReturn(0);

  ierr = MatStashScatterBegin_Private(mat,&mat->stash,mat->rmap->range);CHKERRQ(ierr);
  ierr = MatStashGetInfo_Private(&mat->stash,&nstash,&reallocs);CHKERRQ(ierr);
  ierr = PetscInfo2(mdn->A,"Stash has %D entries, uses %D mallocs.\n",nstash,reallocs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatAssemblyEnd_MPIDense(Mat mat,MatAssemblyType mode)
{
  Mat_MPIDense   *mdn=(Mat_MPIDense*)mat->data;
  PetscErrorCode ierr;
  PetscInt       i,*row,*col,flg,j,rstart,ncols;
  PetscMPIInt    n;
  PetscScalar    *val;

  PetscFunctionBegin;
  if (!mdn->donotstash && !mat->nooffprocentries) {
    /*  wait on receives */
    while (1) {
      ierr = MatStashScatterGetMesg_Private(&mat->stash,&n,&row,&col,&val,&flg);CHKERRQ(ierr);
      if (!flg) break;

      for (i=0; i<n;) {
        /* Now identify the consecutive vals belonging to the same row */
        for (j=i,rstart=row[j]; j<n; j++) {
          if (row[j] != rstart) break;
        }
        if (j < n) ncols = j-i;
        else       ncols = n-i;
        /* Now assemble all these values with a single function call */
        ierr = MatSetValues_MPIDense(mat,1,row+i,ncols,col+i,val+i,mat->insertmode);CHKERRQ(ierr);
        i    = j;
      }
    }
    ierr = MatStashScatterEnd_Private(&mat->stash);CHKERRQ(ierr);
  }

  ierr = MatAssemblyBegin(mdn->A,mode);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mdn->A,mode);CHKERRQ(ierr);

  if (!mat->was_assembled && mode == MAT_FINAL_ASSEMBLY) {
    ierr = MatSetUpMultiply_MPIDense(mat);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatZeroEntries_MPIDense(Mat A)
{
  PetscErrorCode ierr;
  Mat_MPIDense   *l = (Mat_MPIDense*)A->data;

  PetscFunctionBegin;
  ierr = MatZeroEntries(l->A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatZeroRows_MPIDense(Mat A,PetscInt n,const PetscInt rows[],PetscScalar diag,Vec x,Vec b)
{
  Mat_MPIDense      *l = (Mat_MPIDense*)A->data;
  PetscErrorCode    ierr;
  PetscInt          i,len,*lrows;

  PetscFunctionBegin;
  /* get locally owned rows */
  ierr = PetscLayoutMapLocal(A->rmap,n,rows,&len,&lrows,NULL);CHKERRQ(ierr);
  /* fix right hand side if needed */
  if (x && b) {
    const PetscScalar *xx;
    PetscScalar       *bb;

    ierr = VecGetArrayRead(x, &xx);CHKERRQ(ierr);
    ierr = VecGetArrayWrite(b, &bb);CHKERRQ(ierr);
    for (i=0;i<len;++i) bb[lrows[i]] = diag*xx[lrows[i]];
    ierr = VecRestoreArrayRead(x, &xx);CHKERRQ(ierr);
    ierr = VecRestoreArrayWrite(b, &bb);CHKERRQ(ierr);
  }
  ierr = MatZeroRows(l->A,len,lrows,0.0,NULL,NULL);CHKERRQ(ierr);
  if (diag != 0.0) {
    Vec d;

    ierr = MatCreateVecs(A,NULL,&d);CHKERRQ(ierr);
    ierr = VecSet(d,diag);CHKERRQ(ierr);
    ierr = MatDiagonalSet(A,d,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecDestroy(&d);CHKERRQ(ierr);
  }
  ierr = PetscFree(lrows);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatMult_SeqDense(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatMultAdd_SeqDense(Mat,Vec,Vec,Vec);
PETSC_INTERN PetscErrorCode MatMultTranspose_SeqDense(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatMultTransposeAdd_SeqDense(Mat,Vec,Vec,Vec);

PetscErrorCode MatMult_MPIDense(Mat mat,Vec xx,Vec yy)
{
  Mat_MPIDense      *mdn = (Mat_MPIDense*)mat->data;
  PetscErrorCode    ierr;
  const PetscScalar *ax;
  PetscScalar       *ay;
  PetscMemType      axmtype,aymtype;

  PetscFunctionBegin;
  ierr = VecGetArrayReadAndMemType(xx,&ax,&axmtype);CHKERRQ(ierr);
  ierr = VecGetArrayAndMemType(mdn->lvec,&ay,&aymtype);CHKERRQ(ierr);
  ierr = PetscSFBcastWithMemTypeBegin(mdn->Mvctx,MPIU_SCALAR,axmtype,ax,aymtype,ay);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(mdn->Mvctx,MPIU_SCALAR,ax,ay);CHKERRQ(ierr);
  ierr = VecRestoreArrayAndMemType(mdn->lvec,&ay);CHKERRQ(ierr);
  ierr = VecRestoreArrayReadAndMemType(xx,&ax);CHKERRQ(ierr);
  ierr = (*mdn->A->ops->mult)(mdn->A,mdn->lvec,yy);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultAdd_MPIDense(Mat mat,Vec xx,Vec yy,Vec zz)
{
  Mat_MPIDense      *mdn = (Mat_MPIDense*)mat->data;
  PetscErrorCode    ierr;
  const PetscScalar *ax;
  PetscScalar       *ay;
  PetscMemType      axmtype,aymtype;

  PetscFunctionBegin;
  ierr = VecGetArrayReadAndMemType(xx,&ax,&axmtype);CHKERRQ(ierr);
  ierr = VecGetArrayAndMemType(mdn->lvec,&ay,&aymtype);CHKERRQ(ierr);
  ierr = PetscSFBcastWithMemTypeBegin(mdn->Mvctx,MPIU_SCALAR,axmtype,ax,aymtype,ay);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(mdn->Mvctx,MPIU_SCALAR,ax,ay);CHKERRQ(ierr);
  ierr = VecRestoreArrayAndMemType(mdn->lvec,&ay);CHKERRQ(ierr);
  ierr = VecRestoreArrayReadAndMemType(xx,&ax);CHKERRQ(ierr);
  ierr = (*mdn->A->ops->multadd)(mdn->A,mdn->lvec,yy,zz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultTranspose_MPIDense(Mat A,Vec xx,Vec yy)
{
  Mat_MPIDense      *a = (Mat_MPIDense*)A->data;
  PetscErrorCode    ierr;
  const PetscScalar *ax;
  PetscScalar       *ay;
  PetscMemType      axmtype,aymtype;

  PetscFunctionBegin;
  ierr = VecSet(yy,0.0);CHKERRQ(ierr);
  ierr = (*a->A->ops->multtranspose)(a->A,xx,a->lvec);CHKERRQ(ierr);
  ierr = VecGetArrayReadAndMemType(a->lvec,&ax,&axmtype);CHKERRQ(ierr);
  ierr = VecGetArrayAndMemType(yy,&ay,&aymtype);CHKERRQ(ierr);
  ierr = PetscSFReduceWithMemTypeBegin(a->Mvctx,MPIU_SCALAR,axmtype,ax,aymtype,ay,MPIU_SUM);CHKERRQ(ierr);
  ierr = PetscSFReduceEnd(a->Mvctx,MPIU_SCALAR,ax,ay,MPIU_SUM);CHKERRQ(ierr);
  ierr = VecRestoreArrayReadAndMemType(a->lvec,&ax);CHKERRQ(ierr);
  ierr = VecRestoreArrayAndMemType(yy,&ay);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultTransposeAdd_MPIDense(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_MPIDense      *a = (Mat_MPIDense*)A->data;
  PetscErrorCode    ierr;
  const PetscScalar *ax;
  PetscScalar       *ay;
  PetscMemType      axmtype,aymtype;

  PetscFunctionBegin;
  ierr = VecCopy(yy,zz);CHKERRQ(ierr);
  ierr = (*a->A->ops->multtranspose)(a->A,xx,a->lvec);CHKERRQ(ierr);
  ierr = VecGetArrayReadAndMemType(a->lvec,&ax,&axmtype);CHKERRQ(ierr);
  ierr = VecGetArrayAndMemType(zz,&ay,&aymtype);CHKERRQ(ierr);
  ierr = PetscSFReduceWithMemTypeBegin(a->Mvctx,MPIU_SCALAR,axmtype,ax,aymtype,ay,MPIU_SUM);CHKERRQ(ierr);
  ierr = PetscSFReduceEnd(a->Mvctx,MPIU_SCALAR,ax,ay,MPIU_SUM);CHKERRQ(ierr);
  ierr = VecRestoreArrayReadAndMemType(a->lvec,&ax);CHKERRQ(ierr);
  ierr = VecRestoreArrayAndMemType(zz,&ay);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatGetDiagonal_MPIDense(Mat A,Vec v)
{
  Mat_MPIDense      *a    = (Mat_MPIDense*)A->data;
  PetscErrorCode    ierr;
  PetscInt          lda,len,i,n,m = A->rmap->n,radd;
  PetscScalar       *x,zero = 0.0;
  const PetscScalar *av;

  PetscFunctionBegin;
  ierr = VecSet(v,zero);CHKERRQ(ierr);
  ierr = VecGetArray(v,&x);CHKERRQ(ierr);
  ierr = VecGetSize(v,&n);CHKERRQ(ierr);
  if (n != A->rmap->N) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Nonconforming mat and vec");
  len  = PetscMin(a->A->rmap->n,a->A->cmap->n);
  radd = A->rmap->rstart*m;
  ierr = MatDenseGetArrayRead(a->A,&av);CHKERRQ(ierr);
  ierr = MatDenseGetLDA(a->A,&lda);CHKERRQ(ierr);
  for (i=0; i<len; i++) {
    x[i] = av[radd + i*lda + i];
  }
  ierr = MatDenseRestoreArrayRead(a->A,&av);CHKERRQ(ierr);
  ierr = VecRestoreArray(v,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatDestroy_MPIDense(Mat mat)
{
  Mat_MPIDense   *mdn = (Mat_MPIDense*)mat->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
#if defined(PETSC_USE_LOG)
  PetscLogObjectState((PetscObject)mat,"Rows=%D, Cols=%D",mat->rmap->N,mat->cmap->N);
#endif
  ierr = MatStashDestroy_Private(&mat->stash);CHKERRQ(ierr);
  if (mdn->vecinuse) SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_ORDER,"Need to call MatDenseRestoreColumnVec() first");
  if (mdn->matinuse) SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_ORDER,"Need to call MatDenseRestoreSubMatrix() first");
  ierr = MatDestroy(&mdn->A);CHKERRQ(ierr);
  ierr = VecDestroy(&mdn->lvec);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&mdn->Mvctx);CHKERRQ(ierr);
  ierr = VecDestroy(&mdn->cvec);CHKERRQ(ierr);
  ierr = MatDestroy(&mdn->cmat);CHKERRQ(ierr);

  ierr = PetscFree(mat->data);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)mat,NULL);CHKERRQ(ierr);

  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatDenseGetLDA_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatDenseSetLDA_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatDenseGetArray_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatDenseRestoreArray_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatDenseGetArrayRead_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatDenseRestoreArrayRead_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatDenseGetArrayWrite_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatDenseRestoreArrayWrite_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatDensePlaceArray_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatDenseResetArray_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatDenseReplaceArray_C",NULL);CHKERRQ(ierr);
#if defined(PETSC_HAVE_ELEMENTAL)
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatConvert_mpidense_elemental_C",NULL);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_SCALAPACK)
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatConvert_mpidense_scalapack_C",NULL);CHKERRQ(ierr);
#endif
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatMPIDenseSetPreallocation_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatProductSetFromOptions_mpiaij_mpidense_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatProductSetFromOptions_mpidense_mpiaij_C",NULL);CHKERRQ(ierr);
#if defined (PETSC_HAVE_CUDA)
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatProductSetFromOptions_mpiaijcusparse_mpidense_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatProductSetFromOptions_mpidense_mpiaijcusparse_C",NULL);CHKERRQ(ierr);
#endif
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatDenseGetColumn_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatDenseRestoreColumn_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatDenseGetColumnVec_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatDenseRestoreColumnVec_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatDenseGetColumnVecRead_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatDenseRestoreColumnVecRead_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatDenseGetColumnVecWrite_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatDenseRestoreColumnVecWrite_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatDenseGetSubMatrix_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatDenseRestoreSubMatrix_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatView_SeqDense(Mat,PetscViewer);

#include <petscdraw.h>
static PetscErrorCode MatView_MPIDense_ASCIIorDraworSocket(Mat mat,PetscViewer viewer)
{
  Mat_MPIDense      *mdn = (Mat_MPIDense*)mat->data;
  PetscErrorCode    ierr;
  PetscMPIInt       rank;
  PetscViewerType   vtype;
  PetscBool         iascii,isdraw;
  PetscViewer       sviewer;
  PetscViewerFormat format;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)mat),&rank);CHKERRMPI(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERDRAW,&isdraw);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerGetType(viewer,&vtype);CHKERRQ(ierr);
    ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
    if (format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
      MatInfo info;
      ierr = MatGetInfo(mat,MAT_LOCAL,&info);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPushSynchronized(viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"  [%d] local rows %D nz %D nz alloced %D mem %D \n",rank,mat->rmap->n,(PetscInt)info.nz_used,(PetscInt)info.nz_allocated,(PetscInt)info.memory);CHKERRQ(ierr);
      ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopSynchronized(viewer);CHKERRQ(ierr);
      ierr = PetscSFView(mdn->Mvctx,viewer);CHKERRQ(ierr);
      PetscFunctionReturn(0);
    } else if (format == PETSC_VIEWER_ASCII_INFO) {
      PetscFunctionReturn(0);
    }
  } else if (isdraw) {
    PetscDraw draw;
    PetscBool isnull;

    ierr = PetscViewerDrawGetDraw(viewer,0,&draw);CHKERRQ(ierr);
    ierr = PetscDrawIsNull(draw,&isnull);CHKERRQ(ierr);
    if (isnull) PetscFunctionReturn(0);
  }

  {
    /* assemble the entire matrix onto first processor. */
    Mat         A;
    PetscInt    M = mat->rmap->N,N = mat->cmap->N,m,row,i,nz;
    PetscInt    *cols;
    PetscScalar *vals;

    ierr = MatCreate(PetscObjectComm((PetscObject)mat),&A);CHKERRQ(ierr);
    if (!rank) {
      ierr = MatSetSizes(A,M,N,M,N);CHKERRQ(ierr);
    } else {
      ierr = MatSetSizes(A,0,0,M,N);CHKERRQ(ierr);
    }
    /* Since this is a temporary matrix, MATMPIDENSE instead of ((PetscObject)A)->type_name here is probably acceptable. */
    ierr = MatSetType(A,MATMPIDENSE);CHKERRQ(ierr);
    ierr = MatMPIDenseSetPreallocation(A,NULL);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)mat,(PetscObject)A);CHKERRQ(ierr);

    /* Copy the matrix ... This isn't the most efficient means,
       but it's quick for now */
    A->insertmode = INSERT_VALUES;

    row = mat->rmap->rstart;
    m   = mdn->A->rmap->n;
    for (i=0; i<m; i++) {
      ierr = MatGetRow_MPIDense(mat,row,&nz,&cols,&vals);CHKERRQ(ierr);
      ierr = MatSetValues_MPIDense(A,1,&row,nz,cols,vals,INSERT_VALUES);CHKERRQ(ierr);
      ierr = MatRestoreRow_MPIDense(mat,row,&nz,&cols,&vals);CHKERRQ(ierr);
      row++;
    }

    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = PetscViewerGetSubViewer(viewer,PETSC_COMM_SELF,&sviewer);CHKERRQ(ierr);
    if (!rank) {
      ierr = PetscObjectSetName((PetscObject)((Mat_MPIDense*)(A->data))->A,((PetscObject)mat)->name);CHKERRQ(ierr);
      ierr = MatView_SeqDense(((Mat_MPIDense*)(A->data))->A,sviewer);CHKERRQ(ierr);
    }
    ierr = PetscViewerRestoreSubViewer(viewer,PETSC_COMM_SELF,&sviewer);CHKERRQ(ierr);
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
    ierr = MatDestroy(&A);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatView_MPIDense(Mat mat,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscBool      iascii,isbinary,isdraw,issocket;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERBINARY,&isbinary);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERSOCKET,&issocket);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERDRAW,&isdraw);CHKERRQ(ierr);

  if (iascii || issocket || isdraw) {
    ierr = MatView_MPIDense_ASCIIorDraworSocket(mat,viewer);CHKERRQ(ierr);
  } else if (isbinary) {
    ierr = MatView_Dense_Binary(mat,viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatGetInfo_MPIDense(Mat A,MatInfoType flag,MatInfo *info)
{
  Mat_MPIDense   *mat = (Mat_MPIDense*)A->data;
  Mat            mdn  = mat->A;
  PetscErrorCode ierr;
  PetscLogDouble isend[5],irecv[5];

  PetscFunctionBegin;
  info->block_size = 1.0;

  ierr = MatGetInfo(mdn,MAT_LOCAL,info);CHKERRQ(ierr);

  isend[0] = info->nz_used; isend[1] = info->nz_allocated; isend[2] = info->nz_unneeded;
  isend[3] = info->memory;  isend[4] = info->mallocs;
  if (flag == MAT_LOCAL) {
    info->nz_used      = isend[0];
    info->nz_allocated = isend[1];
    info->nz_unneeded  = isend[2];
    info->memory       = isend[3];
    info->mallocs      = isend[4];
  } else if (flag == MAT_GLOBAL_MAX) {
    ierr = MPIU_Allreduce(isend,irecv,5,MPIU_PETSCLOGDOUBLE,MPI_MAX,PetscObjectComm((PetscObject)A));CHKERRQ(ierr);

    info->nz_used      = irecv[0];
    info->nz_allocated = irecv[1];
    info->nz_unneeded  = irecv[2];
    info->memory       = irecv[3];
    info->mallocs      = irecv[4];
  } else if (flag == MAT_GLOBAL_SUM) {
    ierr = MPIU_Allreduce(isend,irecv,5,MPIU_PETSCLOGDOUBLE,MPI_SUM,PetscObjectComm((PetscObject)A));CHKERRQ(ierr);

    info->nz_used      = irecv[0];
    info->nz_allocated = irecv[1];
    info->nz_unneeded  = irecv[2];
    info->memory       = irecv[3];
    info->mallocs      = irecv[4];
  }
  info->fill_ratio_given  = 0; /* no parallel LU/ILU/Cholesky */
  info->fill_ratio_needed = 0;
  info->factor_mallocs    = 0;
  PetscFunctionReturn(0);
}

PetscErrorCode MatSetOption_MPIDense(Mat A,MatOption op,PetscBool flg)
{
  Mat_MPIDense   *a = (Mat_MPIDense*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  switch (op) {
  case MAT_NEW_NONZERO_LOCATIONS:
  case MAT_NEW_NONZERO_LOCATION_ERR:
  case MAT_NEW_NONZERO_ALLOCATION_ERR:
    MatCheckPreallocated(A,1);
    ierr = MatSetOption(a->A,op,flg);CHKERRQ(ierr);
    break;
  case MAT_ROW_ORIENTED:
    MatCheckPreallocated(A,1);
    a->roworiented = flg;
    ierr = MatSetOption(a->A,op,flg);CHKERRQ(ierr);
    break;
  case MAT_FORCE_DIAGONAL_ENTRIES:
  case MAT_KEEP_NONZERO_PATTERN:
  case MAT_USE_HASH_TABLE:
  case MAT_SORTED_FULL:
    ierr = PetscInfo1(A,"Option %s ignored\n",MatOptions[op]);CHKERRQ(ierr);
    break;
  case MAT_IGNORE_OFF_PROC_ENTRIES:
    a->donotstash = flg;
    break;
  case MAT_SYMMETRIC:
  case MAT_STRUCTURALLY_SYMMETRIC:
  case MAT_HERMITIAN:
  case MAT_SYMMETRY_ETERNAL:
  case MAT_IGNORE_LOWER_TRIANGULAR:
  case MAT_IGNORE_ZERO_ENTRIES:
    ierr = PetscInfo1(A,"Option %s ignored\n",MatOptions[op]);CHKERRQ(ierr);
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"unknown option %s",MatOptions[op]);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatDiagonalScale_MPIDense(Mat A,Vec ll,Vec rr)
{
  Mat_MPIDense      *mdn = (Mat_MPIDense*)A->data;
  const PetscScalar *l;
  PetscScalar       x,*v,*vv,*r;
  PetscErrorCode    ierr;
  PetscInt          i,j,s2a,s3a,s2,s3,m=mdn->A->rmap->n,n=mdn->A->cmap->n,lda;

  PetscFunctionBegin;
  ierr = MatDenseGetArray(mdn->A,&vv);CHKERRQ(ierr);
  ierr = MatDenseGetLDA(mdn->A,&lda);CHKERRQ(ierr);
  ierr = MatGetLocalSize(A,&s2,&s3);CHKERRQ(ierr);
  if (ll) {
    ierr = VecGetLocalSize(ll,&s2a);CHKERRQ(ierr);
    if (s2a != s2) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Left scaling vector non-conforming local size, %D != %D", s2a, s2);
    ierr = VecGetArrayRead(ll,&l);CHKERRQ(ierr);
    for (i=0; i<m; i++) {
      x = l[i];
      v = vv + i;
      for (j=0; j<n; j++) { (*v) *= x; v+= lda;}
    }
    ierr = VecRestoreArrayRead(ll,&l);CHKERRQ(ierr);
    ierr = PetscLogFlops(1.0*n*m);CHKERRQ(ierr);
  }
  if (rr) {
    const PetscScalar *ar;

    ierr = VecGetLocalSize(rr,&s3a);CHKERRQ(ierr);
    if (s3a != s3) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Right scaling vec non-conforming local size, %d != %d.", s3a, s3);
    ierr = VecGetArrayRead(rr,&ar);CHKERRQ(ierr);
    ierr = VecGetArray(mdn->lvec,&r);CHKERRQ(ierr);
    ierr = PetscSFBcastBegin(mdn->Mvctx,MPIU_SCALAR,ar,r);CHKERRQ(ierr);
    ierr = PetscSFBcastEnd(mdn->Mvctx,MPIU_SCALAR,ar,r);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(rr,&ar);CHKERRQ(ierr);
    for (i=0; i<n; i++) {
      x = r[i];
      v = vv + i*lda;
      for (j=0; j<m; j++) (*v++) *= x;
    }
    ierr = VecRestoreArray(mdn->lvec,&r);CHKERRQ(ierr);
    ierr = PetscLogFlops(1.0*n*m);CHKERRQ(ierr);
  }
  ierr = MatDenseRestoreArray(mdn->A,&vv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatNorm_MPIDense(Mat A,NormType type,PetscReal *nrm)
{
  Mat_MPIDense      *mdn = (Mat_MPIDense*)A->data;
  PetscErrorCode    ierr;
  PetscInt          i,j;
  PetscMPIInt       size;
  PetscReal         sum = 0.0;
  const PetscScalar *av,*v;

  PetscFunctionBegin;
  ierr = MatDenseGetArrayRead(mdn->A,&av);CHKERRQ(ierr);
  v    = av;
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)A),&size);CHKERRMPI(ierr);
  if (size == 1) {
    ierr = MatNorm(mdn->A,type,nrm);CHKERRQ(ierr);
  } else {
    if (type == NORM_FROBENIUS) {
      for (i=0; i<mdn->A->cmap->n*mdn->A->rmap->n; i++) {
        sum += PetscRealPart(PetscConj(*v)*(*v)); v++;
      }
      ierr = MPIU_Allreduce(&sum,nrm,1,MPIU_REAL,MPIU_SUM,PetscObjectComm((PetscObject)A));CHKERRQ(ierr);
      *nrm = PetscSqrtReal(*nrm);
      ierr = PetscLogFlops(2.0*mdn->A->cmap->n*mdn->A->rmap->n);CHKERRQ(ierr);
    } else if (type == NORM_1) {
      PetscReal *tmp,*tmp2;
      ierr = PetscCalloc2(A->cmap->N,&tmp,A->cmap->N,&tmp2);CHKERRQ(ierr);
      *nrm = 0.0;
      v    = av;
      for (j=0; j<mdn->A->cmap->n; j++) {
        for (i=0; i<mdn->A->rmap->n; i++) {
          tmp[j] += PetscAbsScalar(*v);  v++;
        }
      }
      ierr = MPIU_Allreduce(tmp,tmp2,A->cmap->N,MPIU_REAL,MPIU_SUM,PetscObjectComm((PetscObject)A));CHKERRQ(ierr);
      for (j=0; j<A->cmap->N; j++) {
        if (tmp2[j] > *nrm) *nrm = tmp2[j];
      }
      ierr = PetscFree2(tmp,tmp2);CHKERRQ(ierr);
      ierr = PetscLogFlops(A->cmap->n*A->rmap->n);CHKERRQ(ierr);
    } else if (type == NORM_INFINITY) { /* max row norm */
      PetscReal ntemp;
      ierr = MatNorm(mdn->A,type,&ntemp);CHKERRQ(ierr);
      ierr = MPIU_Allreduce(&ntemp,nrm,1,MPIU_REAL,MPIU_MAX,PetscObjectComm((PetscObject)A));CHKERRQ(ierr);
    } else SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"No support for two norm");
  }
  ierr = MatDenseRestoreArrayRead(mdn->A,&av);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatTranspose_MPIDense(Mat A,MatReuse reuse,Mat *matout)
{
  Mat_MPIDense   *a    = (Mat_MPIDense*)A->data;
  Mat            B;
  PetscInt       M = A->rmap->N,N = A->cmap->N,m,n,*rwork,rstart = A->rmap->rstart;
  PetscErrorCode ierr;
  PetscInt       j,i,lda;
  PetscScalar    *v;

  PetscFunctionBegin;
  if (reuse == MAT_INITIAL_MATRIX || reuse == MAT_INPLACE_MATRIX) {
    ierr = MatCreate(PetscObjectComm((PetscObject)A),&B);CHKERRQ(ierr);
    ierr = MatSetSizes(B,A->cmap->n,A->rmap->n,N,M);CHKERRQ(ierr);
    ierr = MatSetType(B,((PetscObject)A)->type_name);CHKERRQ(ierr);
    ierr = MatMPIDenseSetPreallocation(B,NULL);CHKERRQ(ierr);
  } else B = *matout;

  m    = a->A->rmap->n; n = a->A->cmap->n;
  ierr = MatDenseGetArrayRead(a->A,(const PetscScalar**)&v);CHKERRQ(ierr);
  ierr = MatDenseGetLDA(a->A,&lda);CHKERRQ(ierr);
  ierr = PetscMalloc1(m,&rwork);CHKERRQ(ierr);
  for (i=0; i<m; i++) rwork[i] = rstart + i;
  for (j=0; j<n; j++) {
    ierr = MatSetValues(B,1,&j,m,rwork,v,INSERT_VALUES);CHKERRQ(ierr);
    v   += lda;
  }
  ierr = MatDenseRestoreArrayRead(a->A,(const PetscScalar**)&v);CHKERRQ(ierr);
  ierr = PetscFree(rwork);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (reuse == MAT_INITIAL_MATRIX || reuse == MAT_REUSE_MATRIX) {
    *matout = B;
  } else {
    ierr = MatHeaderMerge(A,&B);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDuplicate_MPIDense(Mat,MatDuplicateOption,Mat*);
PETSC_INTERN PetscErrorCode MatScale_MPIDense(Mat,PetscScalar);

PetscErrorCode MatSetUp_MPIDense(Mat A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLayoutSetUp(A->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(A->cmap);CHKERRQ(ierr);
  if (!A->preallocated) {
    ierr = MatMPIDenseSetPreallocation(A,NULL);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatAXPY_MPIDense(Mat Y,PetscScalar alpha,Mat X,MatStructure str)
{
  PetscErrorCode ierr;
  Mat_MPIDense   *A = (Mat_MPIDense*)Y->data, *B = (Mat_MPIDense*)X->data;

  PetscFunctionBegin;
  ierr = MatAXPY(A->A,alpha,B->A,str);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatConjugate_MPIDense(Mat mat)
{
  Mat_MPIDense   *a = (Mat_MPIDense*)mat->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatConjugate(a->A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatRealPart_MPIDense(Mat A)
{
  Mat_MPIDense   *a = (Mat_MPIDense*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatRealPart(a->A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatImaginaryPart_MPIDense(Mat A)
{
  Mat_MPIDense   *a = (Mat_MPIDense*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatImaginaryPart(a->A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatGetColumnVector_MPIDense(Mat A,Vec v,PetscInt col)
{
  PetscErrorCode ierr;
  Mat_MPIDense   *a = (Mat_MPIDense*) A->data;

  PetscFunctionBegin;
  if (!a->A) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Missing local matrix");
  if (!a->A->ops->getcolumnvector) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Missing get column operation");
  ierr = (*a->A->ops->getcolumnvector)(a->A,v,col);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatGetColumnNorms_SeqDense(Mat,NormType,PetscReal*);

PetscErrorCode MatGetColumnNorms_MPIDense(Mat A,NormType type,PetscReal *norms)
{
  PetscErrorCode ierr;
  PetscInt       i,n;
  Mat_MPIDense   *a = (Mat_MPIDense*) A->data;
  PetscReal      *work;

  PetscFunctionBegin;
  ierr = MatGetSize(A,NULL,&n);CHKERRQ(ierr);
  ierr = PetscMalloc1(n,&work);CHKERRQ(ierr);
  ierr = MatGetColumnNorms_SeqDense(a->A,type,work);CHKERRQ(ierr);
  if (type == NORM_2) {
    for (i=0; i<n; i++) work[i] *= work[i];
  }
  if (type == NORM_INFINITY) {
    ierr = MPIU_Allreduce(work,norms,n,MPIU_REAL,MPIU_MAX,A->hdr.comm);CHKERRQ(ierr);
  } else {
    ierr = MPIU_Allreduce(work,norms,n,MPIU_REAL,MPIU_SUM,A->hdr.comm);CHKERRQ(ierr);
  }
  ierr = PetscFree(work);CHKERRQ(ierr);
  if (type == NORM_2) {
    for (i=0; i<n; i++) norms[i] = PetscSqrtReal(norms[i]);
  }
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_CUDA)
static PetscErrorCode MatDenseGetColumnVec_MPIDenseCUDA(Mat A,PetscInt col,Vec *v)
{
  Mat_MPIDense   *a = (Mat_MPIDense*)A->data;
  PetscErrorCode ierr;
  PetscInt       lda;

  PetscFunctionBegin;
  if (a->vecinuse) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ORDER,"Need to call MatDenseRestoreColumnVec() first");
  if (a->matinuse) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ORDER,"Need to call MatDenseRestoreSubMatrix() first");
  if (!a->cvec) {
    ierr = VecCreateMPICUDAWithArray(PetscObjectComm((PetscObject)A),A->rmap->bs,A->rmap->n,A->rmap->N,NULL,&a->cvec);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)A,(PetscObject)a->cvec);CHKERRQ(ierr);
  }
  a->vecinuse = col + 1;
  ierr = MatDenseGetLDA(a->A,&lda);CHKERRQ(ierr);
  ierr = MatDenseCUDAGetArray(a->A,(PetscScalar**)&a->ptrinuse);CHKERRQ(ierr);
  ierr = VecCUDAPlaceArray(a->cvec,a->ptrinuse + (size_t)col * (size_t)lda);CHKERRQ(ierr);
  *v   = a->cvec;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseRestoreColumnVec_MPIDenseCUDA(Mat A,PetscInt col,Vec *v)
{
  Mat_MPIDense   *a = (Mat_MPIDense*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!a->vecinuse) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseGetColumnVec() first");
  if (!a->cvec) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Missing internal column vector");
  a->vecinuse = 0;
  ierr = MatDenseCUDARestoreArray(a->A,(PetscScalar**)&a->ptrinuse);CHKERRQ(ierr);
  ierr = VecCUDAResetArray(a->cvec);CHKERRQ(ierr);
  *v   = NULL;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseGetColumnVecRead_MPIDenseCUDA(Mat A,PetscInt col,Vec *v)
{
  Mat_MPIDense   *a = (Mat_MPIDense*)A->data;
  PetscInt       lda;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (a->vecinuse) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ORDER,"Need to call MatDenseRestoreColumnVec() first");
  if (a->matinuse) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ORDER,"Need to call MatDenseRestoreSubMatrix() first");
  if (!a->cvec) {
    ierr = VecCreateMPICUDAWithArray(PetscObjectComm((PetscObject)A),A->rmap->bs,A->rmap->n,A->rmap->N,NULL,&a->cvec);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)A,(PetscObject)a->cvec);CHKERRQ(ierr);
  }
  a->vecinuse = col + 1;
  ierr = MatDenseGetLDA(a->A,&lda);CHKERRQ(ierr);
  ierr = MatDenseCUDAGetArrayRead(a->A,&a->ptrinuse);CHKERRQ(ierr);
  ierr = VecCUDAPlaceArray(a->cvec,a->ptrinuse + (size_t)col * (size_t)lda);CHKERRQ(ierr);
  ierr = VecLockReadPush(a->cvec);CHKERRQ(ierr);
  *v   = a->cvec;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseRestoreColumnVecRead_MPIDenseCUDA(Mat A,PetscInt col,Vec *v)
{
  Mat_MPIDense   *a = (Mat_MPIDense*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!a->vecinuse) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ORDER,"Need to call MatDenseGetColumnVec() first");
  if (!a->cvec) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Missing internal column vector");
  a->vecinuse = 0;
  ierr = MatDenseCUDARestoreArrayRead(a->A,&a->ptrinuse);CHKERRQ(ierr);
  ierr = VecLockReadPop(a->cvec);CHKERRQ(ierr);
  ierr = VecCUDAResetArray(a->cvec);CHKERRQ(ierr);
  *v   = NULL;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseGetColumnVecWrite_MPIDenseCUDA(Mat A,PetscInt col,Vec *v)
{
  Mat_MPIDense   *a = (Mat_MPIDense*)A->data;
  PetscErrorCode ierr;
  PetscInt       lda;

  PetscFunctionBegin;
  if (a->vecinuse) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ORDER,"Need to call MatDenseRestoreColumnVec() first");
  if (a->matinuse) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ORDER,"Need to call MatDenseRestoreSubMatrix() first");
  if (!a->cvec) {
    ierr = VecCreateMPICUDAWithArray(PetscObjectComm((PetscObject)A),A->rmap->bs,A->rmap->n,A->rmap->N,NULL,&a->cvec);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)A,(PetscObject)a->cvec);CHKERRQ(ierr);
  }
  a->vecinuse = col + 1;
  ierr = MatDenseGetLDA(a->A,&lda);CHKERRQ(ierr);
  ierr = MatDenseCUDAGetArrayWrite(a->A,(PetscScalar**)&a->ptrinuse);CHKERRQ(ierr);
  ierr = VecCUDAPlaceArray(a->cvec,a->ptrinuse + (size_t)col * (size_t)lda);CHKERRQ(ierr);
  *v   = a->cvec;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseRestoreColumnVecWrite_MPIDenseCUDA(Mat A,PetscInt col,Vec *v)
{
  Mat_MPIDense   *a = (Mat_MPIDense*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!a->vecinuse) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ORDER,"Need to call MatDenseGetColumnVec() first");
  if (!a->cvec) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Missing internal column vector");
  a->vecinuse = 0;
  ierr = MatDenseCUDARestoreArrayWrite(a->A,(PetscScalar**)&a->ptrinuse);CHKERRQ(ierr);
  ierr = VecCUDAResetArray(a->cvec);CHKERRQ(ierr);
  *v   = NULL;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseCUDAPlaceArray_MPIDenseCUDA(Mat A, const PetscScalar *a)
{
  Mat_MPIDense   *l = (Mat_MPIDense*) A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (l->vecinuse) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ORDER,"Need to call MatDenseRestoreColumnVec() first");
  if (l->matinuse) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ORDER,"Need to call MatDenseRestoreSubMatrix() first");
  ierr = MatDenseCUDAPlaceArray(l->A,a);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseCUDAResetArray_MPIDenseCUDA(Mat A)
{
  Mat_MPIDense   *l = (Mat_MPIDense*) A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (l->vecinuse) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ORDER,"Need to call MatDenseRestoreColumnVec() first");
  if (l->matinuse) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ORDER,"Need to call MatDenseRestoreSubMatrix() first");
  ierr = MatDenseCUDAResetArray(l->A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseCUDAReplaceArray_MPIDenseCUDA(Mat A, const PetscScalar *a)
{
  Mat_MPIDense   *l = (Mat_MPIDense*) A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (l->vecinuse) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ORDER,"Need to call MatDenseRestoreColumnVec() first");
  if (l->matinuse) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ORDER,"Need to call MatDenseRestoreSubMatrix() first");
  ierr = MatDenseCUDAReplaceArray(l->A,a);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseCUDAGetArrayWrite_MPIDenseCUDA(Mat A, PetscScalar **a)
{
  Mat_MPIDense   *l = (Mat_MPIDense*) A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatDenseCUDAGetArrayWrite(l->A,a);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseCUDARestoreArrayWrite_MPIDenseCUDA(Mat A, PetscScalar **a)
{
  Mat_MPIDense   *l = (Mat_MPIDense*) A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatDenseCUDARestoreArrayWrite(l->A,a);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseCUDAGetArrayRead_MPIDenseCUDA(Mat A, const PetscScalar **a)
{
  Mat_MPIDense   *l = (Mat_MPIDense*) A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatDenseCUDAGetArrayRead(l->A,a);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseCUDARestoreArrayRead_MPIDenseCUDA(Mat A, const PetscScalar **a)
{
  Mat_MPIDense   *l = (Mat_MPIDense*) A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatDenseCUDARestoreArrayRead(l->A,a);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseCUDAGetArray_MPIDenseCUDA(Mat A, PetscScalar **a)
{
  Mat_MPIDense   *l = (Mat_MPIDense*) A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatDenseCUDAGetArray(l->A,a);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseCUDARestoreArray_MPIDenseCUDA(Mat A, PetscScalar **a)
{
  Mat_MPIDense   *l = (Mat_MPIDense*) A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatDenseCUDARestoreArray(l->A,a);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseGetColumnVecWrite_MPIDense(Mat,PetscInt,Vec*);
static PetscErrorCode MatDenseGetColumnVecRead_MPIDense(Mat,PetscInt,Vec*);
static PetscErrorCode MatDenseGetColumnVec_MPIDense(Mat,PetscInt,Vec*);
static PetscErrorCode MatDenseRestoreColumnVecWrite_MPIDense(Mat,PetscInt,Vec*);
static PetscErrorCode MatDenseRestoreColumnVecRead_MPIDense(Mat,PetscInt,Vec*);
static PetscErrorCode MatDenseRestoreColumnVec_MPIDense(Mat,PetscInt,Vec*);
static PetscErrorCode MatDenseRestoreSubMatrix_MPIDense(Mat,Mat*);

static PetscErrorCode MatBindToCPU_MPIDenseCUDA(Mat mat,PetscBool bind)
{
  Mat_MPIDense   *d = (Mat_MPIDense*)mat->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (d->vecinuse) SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_ORDER,"Need to call MatDenseRestoreColumnVec() first");
  if (d->matinuse) SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_ORDER,"Need to call MatDenseRestoreSubMatrix() first");
  if (d->A) {
    ierr = MatBindToCPU(d->A,bind);CHKERRQ(ierr);
  }
  mat->boundtocpu = bind;
  if (!bind) {
    PetscBool iscuda;

    ierr = PetscObjectTypeCompare((PetscObject)d->cvec,VECMPICUDA,&iscuda);CHKERRQ(ierr);
    if (!iscuda) {
      ierr = VecDestroy(&d->cvec);CHKERRQ(ierr);
    }
    ierr = PetscObjectTypeCompare((PetscObject)d->cmat,MATMPIDENSECUDA,&iscuda);CHKERRQ(ierr);
    if (!iscuda) {
      ierr = MatDestroy(&d->cmat);CHKERRQ(ierr);
    }
    ierr = PetscObjectComposeFunction((PetscObject)mat,"MatDenseGetColumnVec_C",MatDenseGetColumnVec_MPIDenseCUDA);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)mat,"MatDenseRestoreColumnVec_C",MatDenseRestoreColumnVec_MPIDenseCUDA);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)mat,"MatDenseGetColumnVecRead_C",MatDenseGetColumnVecRead_MPIDenseCUDA);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)mat,"MatDenseRestoreColumnVecRead_C",MatDenseRestoreColumnVecRead_MPIDenseCUDA);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)mat,"MatDenseGetColumnVecWrite_C",MatDenseGetColumnVecWrite_MPIDenseCUDA);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)mat,"MatDenseRestoreColumnVecWrite_C",MatDenseRestoreColumnVecWrite_MPIDenseCUDA);CHKERRQ(ierr);
  } else {
    ierr = PetscObjectComposeFunction((PetscObject)mat,"MatDenseGetColumnVec_C",MatDenseGetColumnVec_MPIDense);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)mat,"MatDenseRestoreColumnVec_C",MatDenseRestoreColumnVec_MPIDense);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)mat,"MatDenseGetColumnVecRead_C",MatDenseGetColumnVecRead_MPIDense);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)mat,"MatDenseRestoreColumnVecRead_C",MatDenseRestoreColumnVecRead_MPIDense);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)mat,"MatDenseGetColumnVecWrite_C",MatDenseGetColumnVecWrite_MPIDense);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)mat,"MatDenseRestoreColumnVecWrite_C",MatDenseRestoreColumnVecWrite_MPIDense);CHKERRQ(ierr);
  }
  if (d->cmat) {
    ierr = MatBindToCPU(d->cmat,bind);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatMPIDenseCUDASetPreallocation(Mat A, PetscScalar *d_data)
{
  Mat_MPIDense   *d = (Mat_MPIDense*)A->data;
  PetscErrorCode ierr;
  PetscBool      iscuda;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  ierr = PetscObjectTypeCompare((PetscObject)A,MATMPIDENSECUDA,&iscuda);CHKERRQ(ierr);
  if (!iscuda) PetscFunctionReturn(0);
  ierr = PetscLayoutSetUp(A->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(A->cmap);CHKERRQ(ierr);
  if (!d->A) {
    ierr = MatCreate(PETSC_COMM_SELF,&d->A);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)A,(PetscObject)d->A);CHKERRQ(ierr);
    ierr = MatSetSizes(d->A,A->rmap->n,A->cmap->N,A->rmap->n,A->cmap->N);CHKERRQ(ierr);
  }
  ierr = MatSetType(d->A,MATSEQDENSECUDA);CHKERRQ(ierr);
  ierr = MatSeqDenseCUDASetPreallocation(d->A,d_data);CHKERRQ(ierr);
  A->preallocated = PETSC_TRUE;
  PetscFunctionReturn(0);
}
#endif

static PetscErrorCode MatSetRandom_MPIDense(Mat x,PetscRandom rctx)
{
  Mat_MPIDense   *d = (Mat_MPIDense*)x->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatSetRandom(d->A,rctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMissingDiagonal_MPIDense(Mat A,PetscBool  *missing,PetscInt *d)
{
  PetscFunctionBegin;
  *missing = PETSC_FALSE;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMatTransposeMultSymbolic_MPIDense_MPIDense(Mat,Mat,PetscReal,Mat);
static PetscErrorCode MatMatTransposeMultNumeric_MPIDense_MPIDense(Mat,Mat,Mat);
static PetscErrorCode MatTransposeMatMultNumeric_MPIDense_MPIDense(Mat,Mat,Mat);
static PetscErrorCode MatTransposeMatMultSymbolic_MPIDense_MPIDense(Mat,Mat,PetscReal,Mat);
static PetscErrorCode MatEqual_MPIDense(Mat,Mat,PetscBool*);
static PetscErrorCode MatLoad_MPIDense(Mat,PetscViewer);

/* -------------------------------------------------------------------*/
static struct _MatOps MatOps_Values = { MatSetValues_MPIDense,
                                        MatGetRow_MPIDense,
                                        MatRestoreRow_MPIDense,
                                        MatMult_MPIDense,
                                /*  4*/ MatMultAdd_MPIDense,
                                        MatMultTranspose_MPIDense,
                                        MatMultTransposeAdd_MPIDense,
                                        NULL,
                                        NULL,
                                        NULL,
                                /* 10*/ NULL,
                                        NULL,
                                        NULL,
                                        NULL,
                                        MatTranspose_MPIDense,
                                /* 15*/ MatGetInfo_MPIDense,
                                        MatEqual_MPIDense,
                                        MatGetDiagonal_MPIDense,
                                        MatDiagonalScale_MPIDense,
                                        MatNorm_MPIDense,
                                /* 20*/ MatAssemblyBegin_MPIDense,
                                        MatAssemblyEnd_MPIDense,
                                        MatSetOption_MPIDense,
                                        MatZeroEntries_MPIDense,
                                /* 24*/ MatZeroRows_MPIDense,
                                        NULL,
                                        NULL,
                                        NULL,
                                        NULL,
                                /* 29*/ MatSetUp_MPIDense,
                                        NULL,
                                        NULL,
                                        MatGetDiagonalBlock_MPIDense,
                                        NULL,
                                /* 34*/ MatDuplicate_MPIDense,
                                        NULL,
                                        NULL,
                                        NULL,
                                        NULL,
                                /* 39*/ MatAXPY_MPIDense,
                                        MatCreateSubMatrices_MPIDense,
                                        NULL,
                                        MatGetValues_MPIDense,
                                        MatCopy_MPIDense,
                                /* 44*/ NULL,
                                        MatScale_MPIDense,
                                        MatShift_Basic,
                                        NULL,
                                        NULL,
                                /* 49*/ MatSetRandom_MPIDense,
                                        NULL,
                                        NULL,
                                        NULL,
                                        NULL,
                                /* 54*/ NULL,
                                        NULL,
                                        NULL,
                                        NULL,
                                        NULL,
                                /* 59*/ MatCreateSubMatrix_MPIDense,
                                        MatDestroy_MPIDense,
                                        MatView_MPIDense,
                                        NULL,
                                        NULL,
                                /* 64*/ NULL,
                                        NULL,
                                        NULL,
                                        NULL,
                                        NULL,
                                /* 69*/ NULL,
                                        NULL,
                                        NULL,
                                        NULL,
                                        NULL,
                                /* 74*/ NULL,
                                        NULL,
                                        NULL,
                                        NULL,
                                        NULL,
                                /* 79*/ NULL,
                                        NULL,
                                        NULL,
                                        NULL,
                                /* 83*/ MatLoad_MPIDense,
                                        NULL,
                                        NULL,
                                        NULL,
                                        NULL,
                                        NULL,
                                /* 89*/ NULL,
                                        NULL,
                                        NULL,
                                        NULL,
                                        NULL,
                                /* 94*/ NULL,
                                        NULL,
                                        MatMatTransposeMultSymbolic_MPIDense_MPIDense,
                                        MatMatTransposeMultNumeric_MPIDense_MPIDense,
                                        NULL,
                                /* 99*/ MatProductSetFromOptions_MPIDense,
                                        NULL,
                                        NULL,
                                        MatConjugate_MPIDense,
                                        NULL,
                                /*104*/ NULL,
                                        MatRealPart_MPIDense,
                                        MatImaginaryPart_MPIDense,
                                        NULL,
                                        NULL,
                                /*109*/ NULL,
                                        NULL,
                                        NULL,
                                        MatGetColumnVector_MPIDense,
                                        MatMissingDiagonal_MPIDense,
                                /*114*/ NULL,
                                        NULL,
                                        NULL,
                                        NULL,
                                        NULL,
                                /*119*/ NULL,
                                        NULL,
                                        NULL,
                                        NULL,
                                        NULL,
                                /*124*/ NULL,
                                        MatGetColumnNorms_MPIDense,
                                        NULL,
                                        NULL,
                                        NULL,
                                /*129*/ NULL,
                                        NULL,
                                        MatTransposeMatMultSymbolic_MPIDense_MPIDense,
                                        MatTransposeMatMultNumeric_MPIDense_MPIDense,
                                        NULL,
                                /*134*/ NULL,
                                        NULL,
                                        NULL,
                                        NULL,
                                        NULL,
                                /*139*/ NULL,
                                        NULL,
                                        NULL,
                                        NULL,
                                        NULL,
                                        MatCreateMPIMatConcatenateSeqMat_MPIDense,
                                /*145*/ NULL,
                                        NULL,
                                        NULL
};

PetscErrorCode  MatMPIDenseSetPreallocation_MPIDense(Mat mat,PetscScalar *data)
{
  Mat_MPIDense   *a = (Mat_MPIDense*)mat->data;
  PetscBool      iscuda;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (a->matinuse) SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_ORDER,"Need to call MatDenseRestoreSubMatrix() first");
  ierr = PetscLayoutSetUp(mat->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(mat->cmap);CHKERRQ(ierr);
  if (!a->A) {
    ierr = MatCreate(PETSC_COMM_SELF,&a->A);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)mat,(PetscObject)a->A);CHKERRQ(ierr);
    ierr = MatSetSizes(a->A,mat->rmap->n,mat->cmap->N,mat->rmap->n,mat->cmap->N);CHKERRQ(ierr);
  }
  ierr = PetscObjectTypeCompare((PetscObject)mat,MATMPIDENSECUDA,&iscuda);CHKERRQ(ierr);
  ierr = MatSetType(a->A,iscuda ? MATSEQDENSECUDA : MATSEQDENSE);CHKERRQ(ierr);
  ierr = MatSeqDenseSetPreallocation(a->A,data);CHKERRQ(ierr);
  mat->preallocated = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_ELEMENTAL)
PETSC_INTERN PetscErrorCode MatConvert_MPIDense_Elemental(Mat A, MatType newtype,MatReuse reuse,Mat *newmat)
{
  Mat            mat_elemental;
  PetscErrorCode ierr;
  PetscScalar    *v;
  PetscInt       m=A->rmap->n,N=A->cmap->N,rstart=A->rmap->rstart,i,*rows,*cols;

  PetscFunctionBegin;
  if (reuse == MAT_REUSE_MATRIX) {
    mat_elemental = *newmat;
    ierr = MatZeroEntries(*newmat);CHKERRQ(ierr);
  } else {
    ierr = MatCreate(PetscObjectComm((PetscObject)A), &mat_elemental);CHKERRQ(ierr);
    ierr = MatSetSizes(mat_elemental,PETSC_DECIDE,PETSC_DECIDE,A->rmap->N,A->cmap->N);CHKERRQ(ierr);
    ierr = MatSetType(mat_elemental,MATELEMENTAL);CHKERRQ(ierr);
    ierr = MatSetUp(mat_elemental);CHKERRQ(ierr);
    ierr = MatSetOption(mat_elemental,MAT_ROW_ORIENTED,PETSC_FALSE);CHKERRQ(ierr);
  }

  ierr = PetscMalloc2(m,&rows,N,&cols);CHKERRQ(ierr);
  for (i=0; i<N; i++) cols[i] = i;
  for (i=0; i<m; i++) rows[i] = rstart + i;

  /* PETSc-Elemental interface uses axpy for setting off-processor entries, only ADD_VALUES is allowed */
  ierr = MatDenseGetArray(A,&v);CHKERRQ(ierr);
  ierr = MatSetValues(mat_elemental,m,rows,N,cols,v,ADD_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(mat_elemental, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mat_elemental, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatDenseRestoreArray(A,&v);CHKERRQ(ierr);
  ierr = PetscFree2(rows,cols);CHKERRQ(ierr);

  if (reuse == MAT_INPLACE_MATRIX) {
    ierr = MatHeaderReplace(A,&mat_elemental);CHKERRQ(ierr);
  } else {
    *newmat = mat_elemental;
  }
  PetscFunctionReturn(0);
}
#endif

static PetscErrorCode MatDenseGetColumn_MPIDense(Mat A,PetscInt col,PetscScalar **vals)
{
  Mat_MPIDense   *mat = (Mat_MPIDense*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatDenseGetColumn(mat->A,col,vals);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseRestoreColumn_MPIDense(Mat A,PetscScalar **vals)
{
  Mat_MPIDense   *mat = (Mat_MPIDense*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatDenseRestoreColumn(mat->A,vals);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatCreateMPIMatConcatenateSeqMat_MPIDense(MPI_Comm comm,Mat inmat,PetscInt n,MatReuse scall,Mat *outmat)
{
  PetscErrorCode ierr;
  Mat_MPIDense   *mat;
  PetscInt       m,nloc,N;

  PetscFunctionBegin;
  ierr = MatGetSize(inmat,&m,&N);CHKERRQ(ierr);
  ierr = MatGetLocalSize(inmat,NULL,&nloc);CHKERRQ(ierr);
  if (scall == MAT_INITIAL_MATRIX) { /* symbolic phase */
    PetscInt sum;

    if (n == PETSC_DECIDE) {
      ierr = PetscSplitOwnership(comm,&n,&N);CHKERRQ(ierr);
    }
    /* Check sum(n) = N */
    ierr = MPIU_Allreduce(&n,&sum,1,MPIU_INT,MPI_SUM,comm);CHKERRQ(ierr);
    if (sum != N) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Sum of local columns %D != global columns %D",sum,N);

    ierr = MatCreateDense(comm,m,n,PETSC_DETERMINE,N,NULL,outmat);CHKERRQ(ierr);
    ierr = MatSetOption(*outmat,MAT_NO_OFF_PROC_ENTRIES,PETSC_TRUE);CHKERRQ(ierr);
  }

  /* numeric phase */
  mat = (Mat_MPIDense*)(*outmat)->data;
  ierr = MatCopy(inmat,mat->A,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(*outmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*outmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_CUDA)
PetscErrorCode MatConvert_MPIDenseCUDA_MPIDense(Mat M,MatType type,MatReuse reuse,Mat *newmat)
{
  Mat            B;
  Mat_MPIDense   *m;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (reuse == MAT_INITIAL_MATRIX) {
    ierr = MatDuplicate(M,MAT_COPY_VALUES,newmat);CHKERRQ(ierr);
  } else if (reuse == MAT_REUSE_MATRIX) {
    ierr = MatCopy(M,*newmat,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  }

  B    = *newmat;
  ierr = MatBindToCPU_MPIDenseCUDA(B,PETSC_TRUE);CHKERRQ(ierr);
  ierr = PetscFree(B->defaultvectype);CHKERRQ(ierr);
  ierr = PetscStrallocpy(VECSTANDARD,&B->defaultvectype);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)B,MATMPIDENSE);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatConvert_mpidensecuda_mpidense_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatProductSetFromOptions_mpiaij_mpidensecuda_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatProductSetFromOptions_mpiaijcusparse_mpidensecuda_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatProductSetFromOptions_mpidensecuda_mpiaij_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatProductSetFromOptions_mpidensecuda_mpiaijcusparse_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatDenseCUDAGetArray_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatDenseCUDAGetArrayRead_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatDenseCUDAGetArrayWrite_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatDenseCUDARestoreArray_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatDenseCUDARestoreArrayRead_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatDenseCUDARestoreArrayWrite_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatDenseCUDAPlaceArray_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatDenseCUDAResetArray_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatDenseCUDAReplaceArray_C",NULL);CHKERRQ(ierr);
  m    = (Mat_MPIDense*)(B)->data;
  if (m->A) {
    ierr = MatConvert(m->A,MATSEQDENSE,MAT_INPLACE_MATRIX,&m->A);CHKERRQ(ierr);
    ierr = MatSetUpMultiply_MPIDense(B);CHKERRQ(ierr);
  }
  B->ops->bindtocpu = NULL;
  B->offloadmask    = PETSC_OFFLOAD_CPU;
  PetscFunctionReturn(0);
}

PetscErrorCode MatConvert_MPIDense_MPIDenseCUDA(Mat M,MatType type,MatReuse reuse,Mat *newmat)
{
  Mat            B;
  Mat_MPIDense   *m;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (reuse == MAT_INITIAL_MATRIX) {
    ierr = MatDuplicate(M,MAT_COPY_VALUES,newmat);CHKERRQ(ierr);
  } else if (reuse == MAT_REUSE_MATRIX) {
    ierr = MatCopy(M,*newmat,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  }

  B    = *newmat;
  ierr = PetscFree(B->defaultvectype);CHKERRQ(ierr);
  ierr = PetscStrallocpy(VECCUDA,&B->defaultvectype);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)B,MATMPIDENSECUDA);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatConvert_mpidensecuda_mpidense_C",                    MatConvert_MPIDenseCUDA_MPIDense);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatProductSetFromOptions_mpiaij_mpidensecuda_C",        MatProductSetFromOptions_MPIAIJ_MPIDense);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatProductSetFromOptions_mpiaijcusparse_mpidensecuda_C",MatProductSetFromOptions_MPIAIJ_MPIDense);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatProductSetFromOptions_mpidensecuda_mpiaij_C",        MatProductSetFromOptions_MPIDense_MPIAIJ);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatProductSetFromOptions_mpidensecuda_mpiaijcusparse_C",MatProductSetFromOptions_MPIDense_MPIAIJ);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatDenseCUDAGetArray_C",                                MatDenseCUDAGetArray_MPIDenseCUDA);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatDenseCUDAGetArrayRead_C",                            MatDenseCUDAGetArrayRead_MPIDenseCUDA);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatDenseCUDAGetArrayWrite_C",                           MatDenseCUDAGetArrayWrite_MPIDenseCUDA);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatDenseCUDARestoreArray_C",                            MatDenseCUDARestoreArray_MPIDenseCUDA);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatDenseCUDARestoreArrayRead_C",                        MatDenseCUDARestoreArrayRead_MPIDenseCUDA);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatDenseCUDARestoreArrayWrite_C",                       MatDenseCUDARestoreArrayWrite_MPIDenseCUDA);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatDenseCUDAPlaceArray_C",                              MatDenseCUDAPlaceArray_MPIDenseCUDA);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatDenseCUDAResetArray_C",                              MatDenseCUDAResetArray_MPIDenseCUDA);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatDenseCUDAReplaceArray_C",                            MatDenseCUDAReplaceArray_MPIDenseCUDA);CHKERRQ(ierr);
  m    = (Mat_MPIDense*)(B)->data;
  if (m->A) {
    ierr = MatConvert(m->A,MATSEQDENSECUDA,MAT_INPLACE_MATRIX,&m->A);CHKERRQ(ierr);
    ierr = MatSetUpMultiply_MPIDense(B);CHKERRQ(ierr);
    B->offloadmask = PETSC_OFFLOAD_BOTH;
  } else {
    B->offloadmask = PETSC_OFFLOAD_UNALLOCATED;
  }
  ierr = MatBindToCPU_MPIDenseCUDA(B,PETSC_FALSE);CHKERRQ(ierr);

  B->ops->bindtocpu = MatBindToCPU_MPIDenseCUDA;
  PetscFunctionReturn(0);
}
#endif

PetscErrorCode MatDenseGetColumnVec_MPIDense(Mat A,PetscInt col,Vec *v)
{
  Mat_MPIDense   *a = (Mat_MPIDense*)A->data;
  PetscErrorCode ierr;
  PetscInt       lda;

  PetscFunctionBegin;
  if (a->vecinuse) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ORDER,"Need to call MatDenseRestoreColumnVec() first");
  if (a->matinuse) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ORDER,"Need to call MatDenseRestoreSubMatrix() first");
  if (!a->cvec) {
    ierr = VecCreateMPIWithArray(PetscObjectComm((PetscObject)A),A->rmap->bs,A->rmap->n,A->rmap->N,NULL,&a->cvec);CHKERRQ(ierr);
  }
  a->vecinuse = col + 1;
  ierr = MatDenseGetLDA(a->A,&lda);CHKERRQ(ierr);
  ierr = MatDenseGetArray(a->A,(PetscScalar**)&a->ptrinuse);CHKERRQ(ierr);
  ierr = VecPlaceArray(a->cvec,a->ptrinuse + (size_t)col * (size_t)lda);CHKERRQ(ierr);
  *v   = a->cvec;
  PetscFunctionReturn(0);
}

PetscErrorCode MatDenseRestoreColumnVec_MPIDense(Mat A,PetscInt col,Vec *v)
{
  Mat_MPIDense   *a = (Mat_MPIDense*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!a->vecinuse) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseGetColumnVec() first");
  if (!a->cvec) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Missing internal column vector");
  a->vecinuse = 0;
  ierr = MatDenseRestoreArray(a->A,(PetscScalar**)&a->ptrinuse);CHKERRQ(ierr);
  ierr = VecResetArray(a->cvec);CHKERRQ(ierr);
  *v   = NULL;
  PetscFunctionReturn(0);
}

PetscErrorCode MatDenseGetColumnVecRead_MPIDense(Mat A,PetscInt col,Vec *v)
{
  Mat_MPIDense   *a = (Mat_MPIDense*)A->data;
  PetscErrorCode ierr;
  PetscInt       lda;

  PetscFunctionBegin;
  if (a->vecinuse) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ORDER,"Need to call MatDenseRestoreColumnVec() first");
  if (a->matinuse) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ORDER,"Need to call MatDenseRestoreSubMatrix() first");
  if (!a->cvec) {
    ierr = VecCreateMPIWithArray(PetscObjectComm((PetscObject)A),A->rmap->bs,A->rmap->n,A->rmap->N,NULL,&a->cvec);CHKERRQ(ierr);
  }
  a->vecinuse = col + 1;
  ierr = MatDenseGetLDA(a->A,&lda);CHKERRQ(ierr);
  ierr = MatDenseGetArrayRead(a->A,&a->ptrinuse);CHKERRQ(ierr);
  ierr = VecPlaceArray(a->cvec,a->ptrinuse + (size_t)col * (size_t)lda);CHKERRQ(ierr);
  ierr = VecLockReadPush(a->cvec);CHKERRQ(ierr);
  *v   = a->cvec;
  PetscFunctionReturn(0);
}

PetscErrorCode MatDenseRestoreColumnVecRead_MPIDense(Mat A,PetscInt col,Vec *v)
{
  Mat_MPIDense   *a = (Mat_MPIDense*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!a->vecinuse) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ORDER,"Need to call MatDenseGetColumnVec() first");
  if (!a->cvec) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Missing internal column vector");
  a->vecinuse = 0;
  ierr = MatDenseRestoreArrayRead(a->A,&a->ptrinuse);CHKERRQ(ierr);
  ierr = VecLockReadPop(a->cvec);CHKERRQ(ierr);
  ierr = VecResetArray(a->cvec);CHKERRQ(ierr);
  *v   = NULL;
  PetscFunctionReturn(0);
}

PetscErrorCode MatDenseGetColumnVecWrite_MPIDense(Mat A,PetscInt col,Vec *v)
{
  Mat_MPIDense   *a = (Mat_MPIDense*)A->data;
  PetscErrorCode ierr;
  PetscInt       lda;

  PetscFunctionBegin;
  if (a->vecinuse) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ORDER,"Need to call MatDenseRestoreColumnVec() first");
  if (a->matinuse) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ORDER,"Need to call MatDenseRestoreSubMatrix() first");
  if (!a->cvec) {
    ierr = VecCreateMPIWithArray(PetscObjectComm((PetscObject)A),A->rmap->bs,A->rmap->n,A->rmap->N,NULL,&a->cvec);CHKERRQ(ierr);
  }
  a->vecinuse = col + 1;
  ierr = MatDenseGetLDA(a->A,&lda);CHKERRQ(ierr);
  ierr = MatDenseGetArrayWrite(a->A,(PetscScalar**)&a->ptrinuse);CHKERRQ(ierr);
  ierr = VecPlaceArray(a->cvec,a->ptrinuse + (size_t)col * (size_t)lda);CHKERRQ(ierr);
  *v   = a->cvec;
  PetscFunctionReturn(0);
}

PetscErrorCode MatDenseRestoreColumnVecWrite_MPIDense(Mat A,PetscInt col,Vec *v)
{
  Mat_MPIDense   *a = (Mat_MPIDense*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!a->vecinuse) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ORDER,"Need to call MatDenseGetColumnVec() first");
  if (!a->cvec) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Missing internal column vector");
  a->vecinuse = 0;
  ierr = MatDenseRestoreArrayWrite(a->A,(PetscScalar**)&a->ptrinuse);CHKERRQ(ierr);
  ierr = VecResetArray(a->cvec);CHKERRQ(ierr);
  *v   = NULL;
  PetscFunctionReturn(0);
}

PetscErrorCode MatDenseGetSubMatrix_MPIDense(Mat A,PetscInt cbegin,PetscInt cend,Mat *v)
{
  Mat_MPIDense   *a = (Mat_MPIDense*)A->data;
  Mat_MPIDense   *c;
  PetscErrorCode ierr;
  MPI_Comm       comm;
  PetscBool      setup = PETSC_FALSE;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)A,&comm);CHKERRQ(ierr);
  if (a->vecinuse) SETERRQ(comm,PETSC_ERR_ORDER,"Need to call MatDenseRestoreColumnVec() first");
  if (a->matinuse) SETERRQ(comm,PETSC_ERR_ORDER,"Need to call MatDenseRestoreSubMatrix() first");
  if (!a->cmat) {
    setup = PETSC_TRUE;
    ierr = MatCreate(comm,&a->cmat);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)A,(PetscObject)a->cmat);CHKERRQ(ierr);
    ierr = MatSetType(a->cmat,((PetscObject)A)->type_name);CHKERRQ(ierr);
    ierr = PetscLayoutReference(A->rmap,&a->cmat->rmap);CHKERRQ(ierr);
    ierr = PetscLayoutSetSize(a->cmat->cmap,cend-cbegin);CHKERRQ(ierr);
    ierr = PetscLayoutSetUp(a->cmat->cmap);CHKERRQ(ierr);
  } else if (cend-cbegin != a->cmat->cmap->N) {
    setup = PETSC_TRUE;
    ierr = PetscLayoutDestroy(&a->cmat->cmap);CHKERRQ(ierr);
    ierr = PetscLayoutCreate(comm,&a->cmat->cmap);CHKERRQ(ierr);
    ierr = PetscLayoutSetSize(a->cmat->cmap,cend-cbegin);CHKERRQ(ierr);
    ierr = PetscLayoutSetUp(a->cmat->cmap);CHKERRQ(ierr);
  }
  c = (Mat_MPIDense*)a->cmat->data;
  if (c->A) SETERRQ(comm,PETSC_ERR_ORDER,"Need to call MatDenseRestoreSubMatrix() first");
  ierr = MatDenseGetSubMatrix(a->A,cbegin,cend,&c->A);CHKERRQ(ierr);
  if (setup) { /* do we really need this? */
    ierr = MatSetUpMultiply_MPIDense(a->cmat);CHKERRQ(ierr);
  }
  a->cmat->preallocated = PETSC_TRUE;
  a->cmat->assembled = PETSC_TRUE;
  a->matinuse = cbegin + 1;
  *v = a->cmat;
  PetscFunctionReturn(0);
}

PetscErrorCode MatDenseRestoreSubMatrix_MPIDense(Mat A,Mat *v)
{
  Mat_MPIDense   *a = (Mat_MPIDense*)A->data;
  Mat_MPIDense   *c;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!a->matinuse) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ORDER,"Need to call MatDenseGetSubMatrix() first");
  if (!a->cmat) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Missing internal matrix");
  if (*v != a->cmat) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Not the matrix obtained from MatDenseGetSubMatrix()");
  a->matinuse = 0;
  c    = (Mat_MPIDense*)a->cmat->data;
  ierr = MatDenseRestoreSubMatrix(a->A,&c->A);CHKERRQ(ierr);
  *v   = NULL;
  PetscFunctionReturn(0);
}

/*MC
   MATMPIDENSE - MATMPIDENSE = "mpidense" - A matrix type to be used for distributed dense matrices.

   Options Database Keys:
. -mat_type mpidense - sets the matrix type to "mpidense" during a call to MatSetFromOptions()

  Level: beginner

.seealso: MatCreateDense()

M*/
PETSC_EXTERN PetscErrorCode MatCreate_MPIDense(Mat mat)
{
  Mat_MPIDense   *a;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr      = PetscNewLog(mat,&a);CHKERRQ(ierr);
  mat->data = (void*)a;
  ierr      = PetscMemcpy(mat->ops,&MatOps_Values,sizeof(struct _MatOps));CHKERRQ(ierr);

  mat->insertmode = NOT_SET_VALUES;

  /* build cache for off array entries formed */
  a->donotstash = PETSC_FALSE;

  ierr = MatStashCreate_Private(PetscObjectComm((PetscObject)mat),1,&mat->stash);CHKERRQ(ierr);

  /* stuff used for matrix vector multiply */
  a->lvec        = NULL;
  a->Mvctx       = NULL;
  a->roworiented = PETSC_TRUE;

  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatDenseGetLDA_C",MatDenseGetLDA_MPIDense);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatDenseSetLDA_C",MatDenseSetLDA_MPIDense);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatDenseGetArray_C",MatDenseGetArray_MPIDense);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatDenseRestoreArray_C",MatDenseRestoreArray_MPIDense);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatDenseGetArrayRead_C",MatDenseGetArrayRead_MPIDense);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatDenseRestoreArrayRead_C",MatDenseRestoreArrayRead_MPIDense);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatDenseGetArrayWrite_C",MatDenseGetArrayWrite_MPIDense);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatDenseRestoreArrayWrite_C",MatDenseRestoreArrayWrite_MPIDense);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatDensePlaceArray_C",MatDensePlaceArray_MPIDense);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatDenseResetArray_C",MatDenseResetArray_MPIDense);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatDenseReplaceArray_C",MatDenseReplaceArray_MPIDense);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatDenseGetColumnVec_C",MatDenseGetColumnVec_MPIDense);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatDenseRestoreColumnVec_C",MatDenseRestoreColumnVec_MPIDense);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatDenseGetColumnVecRead_C",MatDenseGetColumnVecRead_MPIDense);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatDenseRestoreColumnVecRead_C",MatDenseRestoreColumnVecRead_MPIDense);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatDenseGetColumnVecWrite_C",MatDenseGetColumnVecWrite_MPIDense);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatDenseRestoreColumnVecWrite_C",MatDenseRestoreColumnVecWrite_MPIDense);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatDenseGetSubMatrix_C",MatDenseGetSubMatrix_MPIDense);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatDenseRestoreSubMatrix_C",MatDenseRestoreSubMatrix_MPIDense);CHKERRQ(ierr);
#if defined(PETSC_HAVE_ELEMENTAL)
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatConvert_mpidense_elemental_C",MatConvert_MPIDense_Elemental);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_SCALAPACK)
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatConvert_mpidense_scalapack_C",MatConvert_Dense_ScaLAPACK);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_CUDA)
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatConvert_mpidense_mpidensecuda_C",MatConvert_MPIDense_MPIDenseCUDA);CHKERRQ(ierr);
#endif
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatMPIDenseSetPreallocation_C",MatMPIDenseSetPreallocation_MPIDense);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatProductSetFromOptions_mpiaij_mpidense_C",MatProductSetFromOptions_MPIAIJ_MPIDense);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatProductSetFromOptions_mpidense_mpiaij_C",MatProductSetFromOptions_MPIDense_MPIAIJ);CHKERRQ(ierr);
#if defined(PETSC_HAVE_CUDA)
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatProductSetFromOptions_mpiaijcusparse_mpidense_C",MatProductSetFromOptions_MPIAIJ_MPIDense);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatProductSetFromOptions_mpidense_mpiaijcusparse_C",MatProductSetFromOptions_MPIDense_MPIAIJ);CHKERRQ(ierr);
#endif

  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatDenseGetColumn_C",MatDenseGetColumn_MPIDense);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatDenseRestoreColumn_C",MatDenseRestoreColumn_MPIDense);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)mat,MATMPIDENSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
   MATMPIDENSECUDA - MATMPIDENSECUDA = "mpidensecuda" - A matrix type to be used for distributed dense matrices on GPUs.

   Options Database Keys:
. -mat_type mpidensecuda - sets the matrix type to "mpidensecuda" during a call to MatSetFromOptions()

  Level: beginner

.seealso:

M*/
#if defined(PETSC_HAVE_CUDA)
PETSC_EXTERN PetscErrorCode MatCreate_MPIDenseCUDA(Mat B)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscCUDAInitializeCheck();CHKERRQ(ierr);
  ierr = MatCreate_MPIDense(B);CHKERRQ(ierr);
  ierr = MatConvert_MPIDense_MPIDenseCUDA(B,MATMPIDENSECUDA,MAT_INPLACE_MATRIX,&B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif

/*MC
   MATDENSE - MATDENSE = "dense" - A matrix type to be used for dense matrices.

   This matrix type is identical to MATSEQDENSE when constructed with a single process communicator,
   and MATMPIDENSE otherwise.

   Options Database Keys:
. -mat_type dense - sets the matrix type to "dense" during a call to MatSetFromOptions()

  Level: beginner


.seealso: MATSEQDENSE,MATMPIDENSE,MATDENSECUDA
M*/

/*MC
   MATDENSECUDA - MATDENSECUDA = "densecuda" - A matrix type to be used for dense matrices on GPUs.

   This matrix type is identical to MATSEQDENSECUDA when constructed with a single process communicator,
   and MATMPIDENSECUDA otherwise.

   Options Database Keys:
. -mat_type densecuda - sets the matrix type to "densecuda" during a call to MatSetFromOptions()

  Level: beginner

.seealso: MATSEQDENSECUDA,MATMPIDENSECUDA,MATDENSE
M*/

/*@C
   MatMPIDenseSetPreallocation - Sets the array used to store the matrix entries

   Collective

   Input Parameters:
.  B - the matrix
-  data - optional location of matrix data.  Set data=NULL for PETSc
   to control all matrix memory allocation.

   Notes:
   The dense format is fully compatible with standard Fortran 77
   storage by columns.

   The data input variable is intended primarily for Fortran programmers
   who wish to allocate their own matrix memory space.  Most users should
   set data=NULL.

   Level: intermediate

.seealso: MatCreate(), MatCreateSeqDense(), MatSetValues()
@*/
PetscErrorCode  MatMPIDenseSetPreallocation(Mat B,PetscScalar *data)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(B,MAT_CLASSID,1);
  ierr = PetscTryMethod(B,"MatMPIDenseSetPreallocation_C",(Mat,PetscScalar*),(B,data));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   MatDensePlaceArray - Allows one to replace the array in a dense matrix with an
   array provided by the user. This is useful to avoid copying an array
   into a matrix

   Not Collective

   Input Parameters:
+  mat - the matrix
-  array - the array in column major order

   Notes:
   You can return to the original array with a call to MatDenseResetArray(). The user is responsible for freeing this array; it will not be
   freed when the matrix is destroyed.

   Level: developer

.seealso: MatDenseGetArray(), MatDenseResetArray(), VecPlaceArray(), VecGetArray(), VecRestoreArray(), VecReplaceArray(), VecResetArray()

@*/
PetscErrorCode  MatDensePlaceArray(Mat mat,const PetscScalar *array)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  ierr = PetscUseMethod(mat,"MatDensePlaceArray_C",(Mat,const PetscScalar*),(mat,array));CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)mat);CHKERRQ(ierr);
#if defined(PETSC_HAVE_CUDA)
  mat->offloadmask = PETSC_OFFLOAD_CPU;
#endif
  PetscFunctionReturn(0);
}

/*@
   MatDenseResetArray - Resets the matrix array to that it previously had before the call to MatDensePlaceArray()

   Not Collective

   Input Parameters:
.  mat - the matrix

   Notes:
   You can only call this after a call to MatDensePlaceArray()

   Level: developer

.seealso: MatDenseGetArray(), MatDensePlaceArray(), VecPlaceArray(), VecGetArray(), VecRestoreArray(), VecReplaceArray(), VecResetArray()

@*/
PetscErrorCode  MatDenseResetArray(Mat mat)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  ierr = PetscUseMethod(mat,"MatDenseResetArray_C",(Mat),(mat));CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)mat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   MatDenseReplaceArray - Allows one to replace the array in a dense matrix with an
   array provided by the user. This is useful to avoid copying an array
   into a matrix

   Not Collective

   Input Parameters:
+  mat - the matrix
-  array - the array in column major order

   Notes:
   The memory passed in MUST be obtained with PetscMalloc() and CANNOT be
   freed by the user. It will be freed when the matrix is destroyed.

   Level: developer

.seealso: MatDenseGetArray(), VecReplaceArray()
@*/
PetscErrorCode  MatDenseReplaceArray(Mat mat,const PetscScalar *array)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  ierr = PetscUseMethod(mat,"MatDenseReplaceArray_C",(Mat,const PetscScalar*),(mat,array));CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)mat);CHKERRQ(ierr);
#if defined(PETSC_HAVE_CUDA)
  mat->offloadmask = PETSC_OFFLOAD_CPU;
#endif
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_CUDA)
/*@C
   MatDenseCUDAPlaceArray - Allows one to replace the GPU array in a dense matrix with an
   array provided by the user. This is useful to avoid copying an array
   into a matrix

   Not Collective

   Input Parameters:
+  mat - the matrix
-  array - the array in column major order

   Notes:
   You can return to the original array with a call to MatDenseCUDAResetArray(). The user is responsible for freeing this array; it will not be
   freed when the matrix is destroyed. The array must have been allocated with cudaMalloc().

   Level: developer

.seealso: MatDenseCUDAGetArray(), MatDenseCUDAResetArray()
@*/
PetscErrorCode  MatDenseCUDAPlaceArray(Mat mat,const PetscScalar *array)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  ierr = PetscUseMethod(mat,"MatDenseCUDAPlaceArray_C",(Mat,const PetscScalar*),(mat,array));CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)mat);CHKERRQ(ierr);
  mat->offloadmask = PETSC_OFFLOAD_GPU;
  PetscFunctionReturn(0);
}

/*@C
   MatDenseCUDAResetArray - Resets the matrix array to that it previously had before the call to MatDenseCUDAPlaceArray()

   Not Collective

   Input Parameters:
.  mat - the matrix

   Notes:
   You can only call this after a call to MatDenseCUDAPlaceArray()

   Level: developer

.seealso: MatDenseCUDAGetArray(), MatDenseCUDAPlaceArray()

@*/
PetscErrorCode  MatDenseCUDAResetArray(Mat mat)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  ierr = PetscUseMethod(mat,"MatDenseCUDAResetArray_C",(Mat),(mat));CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)mat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   MatDenseCUDAReplaceArray - Allows one to replace the GPU array in a dense matrix with an
   array provided by the user. This is useful to avoid copying an array
   into a matrix

   Not Collective

   Input Parameters:
+  mat - the matrix
-  array - the array in column major order

   Notes:
   This permanently replaces the GPU array and frees the memory associated with the old GPU array.
   The memory passed in CANNOT be freed by the user. It will be freed
   when the matrix is destroyed. The array should respect the matrix leading dimension.

   Level: developer

.seealso: MatDenseCUDAGetArray(), MatDenseCUDAPlaceArray(), MatDenseCUDAResetArray()
@*/
PetscErrorCode  MatDenseCUDAReplaceArray(Mat mat,const PetscScalar *array)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  ierr = PetscUseMethod(mat,"MatDenseCUDAReplaceArray_C",(Mat,const PetscScalar*),(mat,array));CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)mat);CHKERRQ(ierr);
  mat->offloadmask = PETSC_OFFLOAD_GPU;
  PetscFunctionReturn(0);
}

/*@C
   MatDenseCUDAGetArrayWrite - Provides write access to the CUDA buffer inside a dense matrix.

   Not Collective

   Input Parameters:
.  A - the matrix

   Output Parameters
.  array - the GPU array in column major order

   Notes:
   The data on the GPU may not be updated due to operations done on the CPU. If you need updated data, use MatDenseCUDAGetArray(). The array must be restored with MatDenseCUDARestoreArrayWrite() when no longer needed.

   Level: developer

.seealso: MatDenseCUDAGetArray(), MatDenseCUDARestoreArray(), MatDenseCUDARestoreArrayWrite(), MatDenseCUDAGetArrayRead(), MatDenseCUDARestoreArrayRead()
@*/
PetscErrorCode MatDenseCUDAGetArrayWrite(Mat A, PetscScalar **a)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  ierr = PetscUseMethod(A,"MatDenseCUDAGetArrayWrite_C",(Mat,PetscScalar**),(A,a));CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   MatDenseCUDARestoreArrayWrite - Restore write access to the CUDA buffer inside a dense matrix previously obtained with MatDenseCUDAGetArrayWrite().

   Not Collective

   Input Parameters:
+  A - the matrix
-  array - the GPU array in column major order

   Notes:

   Level: developer

.seealso: MatDenseCUDAGetArray(), MatDenseCUDARestoreArray(), MatDenseCUDAGetArrayWrite(), MatDenseCUDARestoreArrayRead(), MatDenseCUDAGetArrayRead()
@*/
PetscErrorCode MatDenseCUDARestoreArrayWrite(Mat A, PetscScalar **a)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  ierr = PetscUseMethod(A,"MatDenseCUDARestoreArrayWrite_C",(Mat,PetscScalar**),(A,a));CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)A);CHKERRQ(ierr);
  A->offloadmask = PETSC_OFFLOAD_GPU;
  PetscFunctionReturn(0);
}

/*@C
   MatDenseCUDAGetArrayRead - Provides read-only access to the CUDA buffer inside a dense matrix. The array must be restored with MatDenseCUDARestoreArrayRead() when no longer needed.

   Not Collective

   Input Parameters:
.  A - the matrix

   Output Parameters
.  array - the GPU array in column major order

   Notes:
   Data can be copied to the GPU due to operations done on the CPU. If you need write only access, use MatDenseCUDAGetArrayWrite().

   Level: developer

.seealso: MatDenseCUDAGetArray(), MatDenseCUDARestoreArray(), MatDenseCUDARestoreArrayWrite(), MatDenseCUDAGetArrayWrite(), MatDenseCUDARestoreArrayRead()
@*/
PetscErrorCode MatDenseCUDAGetArrayRead(Mat A, const PetscScalar **a)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  ierr = PetscUseMethod(A,"MatDenseCUDAGetArrayRead_C",(Mat,const PetscScalar**),(A,a));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   MatDenseCUDARestoreArrayRead - Restore read-only access to the CUDA buffer inside a dense matrix previously obtained with a call to MatDenseCUDAGetArrayRead().

   Not Collective

   Input Parameters:
+  A - the matrix
-  array - the GPU array in column major order

   Notes:
   Data can be copied to the GPU due to operations done on the CPU. If you need write only access, use MatDenseCUDAGetArrayWrite().

   Level: developer

.seealso: MatDenseCUDAGetArray(), MatDenseCUDARestoreArray(), MatDenseCUDARestoreArrayWrite(), MatDenseCUDAGetArrayWrite(), MatDenseCUDAGetArrayRead()
@*/
PetscErrorCode MatDenseCUDARestoreArrayRead(Mat A, const PetscScalar **a)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscUseMethod(A,"MatDenseCUDARestoreArrayRead_C",(Mat,const PetscScalar**),(A,a));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   MatDenseCUDAGetArray - Provides access to the CUDA buffer inside a dense matrix. The array must be restored with MatDenseCUDARestoreArray() when no longer needed.

   Not Collective

   Input Parameters:
.  A - the matrix

   Output Parameters
.  array - the GPU array in column major order

   Notes:
   Data can be copied to the GPU due to operations done on the CPU. If you need write only access, use MatDenseCUDAGetArrayWrite(). For read-only access, use MatDenseCUDAGetArrayRead().

   Level: developer

.seealso: MatDenseCUDAGetArrayRead(), MatDenseCUDARestoreArray(), MatDenseCUDARestoreArrayWrite(), MatDenseCUDAGetArrayWrite(), MatDenseCUDARestoreArrayRead()
@*/
PetscErrorCode MatDenseCUDAGetArray(Mat A, PetscScalar **a)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  ierr = PetscUseMethod(A,"MatDenseCUDAGetArray_C",(Mat,PetscScalar**),(A,a));CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   MatDenseCUDARestoreArray - Restore access to the CUDA buffer inside a dense matrix previously obtained with MatDenseCUDAGetArray().

   Not Collective

   Input Parameters:
+  A - the matrix
-  array - the GPU array in column major order

   Notes:

   Level: developer

.seealso: MatDenseCUDAGetArray(), MatDenseCUDARestoreArrayWrite(), MatDenseCUDAGetArrayWrite(), MatDenseCUDARestoreArrayRead(), MatDenseCUDAGetArrayRead()
@*/
PetscErrorCode MatDenseCUDARestoreArray(Mat A, PetscScalar **a)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  ierr = PetscUseMethod(A,"MatDenseCUDARestoreArray_C",(Mat,PetscScalar**),(A,a));CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)A);CHKERRQ(ierr);
  A->offloadmask = PETSC_OFFLOAD_GPU;
  PetscFunctionReturn(0);
}
#endif

/*@C
   MatCreateDense - Creates a matrix in dense format.

   Collective

   Input Parameters:
+  comm - MPI communicator
.  m - number of local rows (or PETSC_DECIDE to have calculated if M is given)
.  n - number of local columns (or PETSC_DECIDE to have calculated if N is given)
.  M - number of global rows (or PETSC_DECIDE to have calculated if m is given)
.  N - number of global columns (or PETSC_DECIDE to have calculated if n is given)
-  data - optional location of matrix data.  Set data=NULL (PETSC_NULL_SCALAR for Fortran users) for PETSc
   to control all matrix memory allocation.

   Output Parameter:
.  A - the matrix

   Notes:
   The dense format is fully compatible with standard Fortran 77
   storage by columns.
   Note that, although local portions of the matrix are stored in column-major
   order, the matrix is partitioned across MPI ranks by row.

   The data input variable is intended primarily for Fortran programmers
   who wish to allocate their own matrix memory space.  Most users should
   set data=NULL (PETSC_NULL_SCALAR for Fortran users).

   The user MUST specify either the local or global matrix dimensions
   (possibly both).

   Level: intermediate

.seealso: MatCreate(), MatCreateSeqDense(), MatSetValues()
@*/
PetscErrorCode  MatCreateDense(MPI_Comm comm,PetscInt m,PetscInt n,PetscInt M,PetscInt N,PetscScalar *data,Mat *A)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr = MatCreate(comm,A);CHKERRQ(ierr);
  ierr = MatSetSizes(*A,m,n,M,N);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);
  if (size > 1) {
    PetscBool havedata = (PetscBool)!!data;

    ierr = MatSetType(*A,MATMPIDENSE);CHKERRQ(ierr);
    ierr = MatMPIDenseSetPreallocation(*A,data);CHKERRQ(ierr);
    ierr = MPIU_Allreduce(MPI_IN_PLACE,&havedata,1,MPIU_BOOL,MPI_LOR,comm);CHKERRQ(ierr);
    if (havedata) {  /* user provided data array, so no need to assemble */
      ierr = MatSetUpMultiply_MPIDense(*A);CHKERRQ(ierr);
      (*A)->assembled = PETSC_TRUE;
    }
  } else {
    ierr = MatSetType(*A,MATSEQDENSE);CHKERRQ(ierr);
    ierr = MatSeqDenseSetPreallocation(*A,data);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_CUDA)
/*@C
   MatCreateDenseCUDA - Creates a matrix in dense format using CUDA.

   Collective

   Input Parameters:
+  comm - MPI communicator
.  m - number of local rows (or PETSC_DECIDE to have calculated if M is given)
.  n - number of local columns (or PETSC_DECIDE to have calculated if N is given)
.  M - number of global rows (or PETSC_DECIDE to have calculated if m is given)
.  N - number of global columns (or PETSC_DECIDE to have calculated if n is given)
-  data - optional location of GPU matrix data.  Set data=NULL for PETSc
   to control matrix memory allocation.

   Output Parameter:
.  A - the matrix

   Notes:

   Level: intermediate

.seealso: MatCreate(), MatCreateDense()
@*/
PetscErrorCode  MatCreateDenseCUDA(MPI_Comm comm,PetscInt m,PetscInt n,PetscInt M,PetscInt N,PetscScalar *data,Mat *A)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr = MatCreate(comm,A);CHKERRQ(ierr);
  PetscValidLogicalCollectiveBool(*A,!!data,6);
  ierr = MatSetSizes(*A,m,n,M,N);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);
  if (size > 1) {
    ierr = MatSetType(*A,MATMPIDENSECUDA);CHKERRQ(ierr);
    ierr = MatMPIDenseCUDASetPreallocation(*A,data);CHKERRQ(ierr);
    if (data) {  /* user provided data array, so no need to assemble */
      ierr = MatSetUpMultiply_MPIDense(*A);CHKERRQ(ierr);
      (*A)->assembled = PETSC_TRUE;
    }
  } else {
    ierr = MatSetType(*A,MATSEQDENSECUDA);CHKERRQ(ierr);
    ierr = MatSeqDenseCUDASetPreallocation(*A,data);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
#endif

static PetscErrorCode MatDuplicate_MPIDense(Mat A,MatDuplicateOption cpvalues,Mat *newmat)
{
  Mat            mat;
  Mat_MPIDense   *a,*oldmat = (Mat_MPIDense*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  *newmat = NULL;
  ierr    = MatCreate(PetscObjectComm((PetscObject)A),&mat);CHKERRQ(ierr);
  ierr    = MatSetSizes(mat,A->rmap->n,A->cmap->n,A->rmap->N,A->cmap->N);CHKERRQ(ierr);
  ierr    = MatSetType(mat,((PetscObject)A)->type_name);CHKERRQ(ierr);
  a       = (Mat_MPIDense*)mat->data;

  mat->factortype   = A->factortype;
  mat->assembled    = PETSC_TRUE;
  mat->preallocated = PETSC_TRUE;

  mat->insertmode = NOT_SET_VALUES;
  a->donotstash   = oldmat->donotstash;

  ierr = PetscLayoutReference(A->rmap,&mat->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutReference(A->cmap,&mat->cmap);CHKERRQ(ierr);

  ierr = MatDuplicate(oldmat->A,cpvalues,&a->A);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)mat,(PetscObject)a->A);CHKERRQ(ierr);
  ierr = MatSetUpMultiply_MPIDense(mat);CHKERRQ(ierr);

  *newmat = mat;
  PetscFunctionReturn(0);
}

PetscErrorCode MatLoad_MPIDense(Mat newMat, PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscBool      isbinary;
#if defined(PETSC_HAVE_HDF5)
  PetscBool      ishdf5;
#endif

  PetscFunctionBegin;
  PetscValidHeaderSpecific(newMat,MAT_CLASSID,1);
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  /* force binary viewer to load .info file if it has not yet done so */
  ierr = PetscViewerSetUp(viewer);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERBINARY,&isbinary);CHKERRQ(ierr);
#if defined(PETSC_HAVE_HDF5)
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERHDF5,  &ishdf5);CHKERRQ(ierr);
#endif
  if (isbinary) {
    ierr = MatLoad_Dense_Binary(newMat,viewer);CHKERRQ(ierr);
#if defined(PETSC_HAVE_HDF5)
  } else if (ishdf5) {
    ierr = MatLoad_Dense_HDF5(newMat,viewer);CHKERRQ(ierr);
#endif
  } else SETERRQ2(PetscObjectComm((PetscObject)newMat),PETSC_ERR_SUP,"Viewer type %s not yet supported for reading %s matrices",((PetscObject)viewer)->type_name,((PetscObject)newMat)->type_name);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatEqual_MPIDense(Mat A,Mat B,PetscBool  *flag)
{
  Mat_MPIDense   *matB = (Mat_MPIDense*)B->data,*matA = (Mat_MPIDense*)A->data;
  Mat            a,b;
  PetscBool      flg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  a    = matA->A;
  b    = matB->A;
  ierr = MatEqual(a,b,&flg);CHKERRQ(ierr);
  ierr = MPIU_Allreduce(&flg,flag,1,MPIU_BOOL,MPI_LAND,PetscObjectComm((PetscObject)A));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatDestroy_MatTransMatMult_MPIDense_MPIDense(void *data)
{
  PetscErrorCode        ierr;
  Mat_TransMatMultDense *atb = (Mat_TransMatMultDense *)data;

  PetscFunctionBegin;
  ierr = PetscFree2(atb->sendbuf,atb->recvcounts);CHKERRQ(ierr);
  ierr = MatDestroy(&atb->atb);CHKERRQ(ierr);
  ierr = PetscFree(atb);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatDestroy_MatMatTransMult_MPIDense_MPIDense(void *data)
{
  PetscErrorCode        ierr;
  Mat_MatTransMultDense *abt = (Mat_MatTransMultDense *)data;

  PetscFunctionBegin;
  ierr = PetscFree2(abt->buf[0],abt->buf[1]);CHKERRQ(ierr);
  ierr = PetscFree2(abt->recvcounts,abt->recvdispls);CHKERRQ(ierr);
  ierr = PetscFree(abt);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatTransposeMatMultNumeric_MPIDense_MPIDense(Mat A,Mat B,Mat C)
{
  Mat_MPIDense          *a=(Mat_MPIDense*)A->data, *b=(Mat_MPIDense*)B->data, *c=(Mat_MPIDense*)C->data;
  Mat_TransMatMultDense *atb;
  PetscErrorCode        ierr;
  MPI_Comm              comm;
  PetscMPIInt           size,*recvcounts;
  PetscScalar           *carray,*sendbuf;
  const PetscScalar     *atbarray;
  PetscInt              i,cN=C->cmap->N,cM=C->rmap->N,proc,k,j;
  const PetscInt        *ranges;

  PetscFunctionBegin;
  MatCheckProduct(C,3);
  if (!C->product->data) SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Product data empty");
  atb = (Mat_TransMatMultDense *)C->product->data;
  recvcounts = atb->recvcounts;
  sendbuf = atb->sendbuf;

  ierr = PetscObjectGetComm((PetscObject)A,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);

  /* compute atbarray = aseq^T * bseq */
  ierr = MatTransposeMatMult(a->A,b->A,atb->atb ? MAT_REUSE_MATRIX : MAT_INITIAL_MATRIX,PETSC_DEFAULT,&atb->atb);CHKERRQ(ierr);

  ierr = MatGetOwnershipRanges(C,&ranges);CHKERRQ(ierr);

  /* arrange atbarray into sendbuf */
  ierr = MatDenseGetArrayRead(atb->atb,&atbarray);CHKERRQ(ierr);
  for (proc=0, k=0; proc<size; proc++) {
    for (j=0; j<cN; j++) {
      for (i=ranges[proc]; i<ranges[proc+1]; i++) sendbuf[k++] = atbarray[i+j*cM];
    }
  }
  ierr = MatDenseRestoreArrayRead(atb->atb,&atbarray);CHKERRQ(ierr);

  /* sum all atbarray to local values of C */
  ierr = MatDenseGetArrayWrite(c->A,&carray);CHKERRQ(ierr);
  ierr = MPI_Reduce_scatter(sendbuf,carray,recvcounts,MPIU_SCALAR,MPIU_SUM,comm);CHKERRMPI(ierr);
  ierr = MatDenseRestoreArrayWrite(c->A,&carray);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatTransposeMatMultSymbolic_MPIDense_MPIDense(Mat A,Mat B,PetscReal fill,Mat C)
{
  PetscErrorCode        ierr;
  MPI_Comm              comm;
  PetscMPIInt           size;
  PetscInt              cm=A->cmap->n,cM,cN=B->cmap->N;
  Mat_TransMatMultDense *atb;
  PetscBool             cisdense;
  PetscInt              i;
  const PetscInt        *ranges;

  PetscFunctionBegin;
  MatCheckProduct(C,3);
  if (C->product->data) SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Product data not empty");
  ierr = PetscObjectGetComm((PetscObject)A,&comm);CHKERRQ(ierr);
  if (A->rmap->rstart != B->rmap->rstart || A->rmap->rend != B->rmap->rend) {
    SETERRQ4(comm,PETSC_ERR_ARG_SIZ,"Matrix local dimensions are incompatible, A (%D, %D) != B (%D,%D)",A->rmap->rstart,A->rmap->rend,B->rmap->rstart,B->rmap->rend);
  }

  /* create matrix product C */
  ierr = MatSetSizes(C,cm,B->cmap->n,A->cmap->N,B->cmap->N);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompareAny((PetscObject)C,&cisdense,MATMPIDENSE,MATMPIDENSECUDA,"");CHKERRQ(ierr);
  if (!cisdense) {
    ierr = MatSetType(C,((PetscObject)A)->type_name);CHKERRQ(ierr);
  }
  ierr = MatSetUp(C);CHKERRQ(ierr);

  /* create data structure for reuse C */
  ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);
  ierr = PetscNew(&atb);CHKERRQ(ierr);
  cM   = C->rmap->N;
  ierr = PetscMalloc2((size_t)cM*(size_t)cN,&atb->sendbuf,size,&atb->recvcounts);CHKERRQ(ierr);
  ierr = MatGetOwnershipRanges(C,&ranges);CHKERRQ(ierr);
  for (i=0; i<size; i++) atb->recvcounts[i] = (ranges[i+1] - ranges[i])*cN;

  C->product->data    = atb;
  C->product->destroy = MatDestroy_MatTransMatMult_MPIDense_MPIDense;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMatTransposeMultSymbolic_MPIDense_MPIDense(Mat A, Mat B, PetscReal fill, Mat C)
{
  PetscErrorCode        ierr;
  MPI_Comm              comm;
  PetscMPIInt           i, size;
  PetscInt              maxRows, bufsiz;
  PetscMPIInt           tag;
  PetscInt              alg;
  Mat_MatTransMultDense *abt;
  Mat_Product           *product = C->product;
  PetscBool             flg;

  PetscFunctionBegin;
  MatCheckProduct(C,4);
  if (C->product->data) SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Product data not empty");
  /* check local size of A and B */
  if (A->cmap->n != B->cmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Matrix local column dimensions are incompatible, A (%D) != B (%D)",A->cmap->n,B->cmap->n);

  ierr = PetscStrcmp(product->alg,"allgatherv",&flg);CHKERRQ(ierr);
  alg  = flg ? 0 : 1;

  /* setup matrix product C */
  ierr = MatSetSizes(C,A->rmap->n,B->rmap->n,A->rmap->N,B->rmap->N);CHKERRQ(ierr);
  ierr = MatSetType(C,MATMPIDENSE);CHKERRQ(ierr);
  ierr = MatSetUp(C);CHKERRQ(ierr);
  ierr = PetscObjectGetNewTag((PetscObject)C,&tag);CHKERRQ(ierr);

  /* create data structure for reuse C */
  ierr = PetscObjectGetComm((PetscObject)C,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);
  ierr = PetscNew(&abt);CHKERRQ(ierr);
  abt->tag = tag;
  abt->alg = alg;
  switch (alg) {
  case 1: /* alg: "cyclic" */
    for (maxRows = 0, i = 0; i < size; i++) maxRows = PetscMax(maxRows, (B->rmap->range[i + 1] - B->rmap->range[i]));
    bufsiz = A->cmap->N * maxRows;
    ierr = PetscMalloc2(bufsiz,&(abt->buf[0]),bufsiz,&(abt->buf[1]));CHKERRQ(ierr);
    break;
  default: /* alg: "allgatherv" */
    ierr = PetscMalloc2(B->rmap->n * B->cmap->N, &(abt->buf[0]), B->rmap->N * B->cmap->N, &(abt->buf[1]));CHKERRQ(ierr);
    ierr = PetscMalloc2(size,&(abt->recvcounts),size+1,&(abt->recvdispls));CHKERRQ(ierr);
    for (i = 0; i <= size; i++) abt->recvdispls[i] = B->rmap->range[i] * A->cmap->N;
    for (i = 0; i < size; i++) abt->recvcounts[i] = abt->recvdispls[i + 1] - abt->recvdispls[i];
    break;
  }

  C->product->data    = abt;
  C->product->destroy = MatDestroy_MatMatTransMult_MPIDense_MPIDense;
  C->ops->mattransposemultnumeric = MatMatTransposeMultNumeric_MPIDense_MPIDense;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMatTransposeMultNumeric_MPIDense_MPIDense_Cyclic(Mat A, Mat B, Mat C)
{
  Mat_MPIDense          *a=(Mat_MPIDense*)A->data, *b=(Mat_MPIDense*)B->data, *c=(Mat_MPIDense*)C->data;
  Mat_MatTransMultDense *abt;
  PetscErrorCode        ierr;
  MPI_Comm              comm;
  PetscMPIInt           rank,size, sendsiz, recvsiz, sendto, recvfrom, recvisfrom;
  PetscScalar           *sendbuf, *recvbuf=NULL, *cv;
  PetscInt              i,cK=A->cmap->N,k,j,bn;
  PetscScalar           _DOne=1.0,_DZero=0.0;
  const PetscScalar     *av,*bv;
  PetscBLASInt          cm, cn, ck, alda, blda = 0, clda;
  MPI_Request           reqs[2];
  const PetscInt        *ranges;

  PetscFunctionBegin;
  MatCheckProduct(C,3);
  if (!C->product->data) SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Product data empty");
  abt  = (Mat_MatTransMultDense*)C->product->data;
  ierr = PetscObjectGetComm((PetscObject)C,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRMPI(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);
  ierr = MatDenseGetArrayRead(a->A,&av);CHKERRQ(ierr);
  ierr = MatDenseGetArrayRead(b->A,&bv);CHKERRQ(ierr);
  ierr = MatDenseGetArrayWrite(c->A,&cv);CHKERRQ(ierr);
  ierr = MatDenseGetLDA(a->A,&i);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(i,&alda);CHKERRQ(ierr);
  ierr = MatDenseGetLDA(b->A,&i);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(i,&blda);CHKERRQ(ierr);
  ierr = MatDenseGetLDA(c->A,&i);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(i,&clda);CHKERRQ(ierr);
  ierr = MatGetOwnershipRanges(B,&ranges);CHKERRQ(ierr);
  bn   = B->rmap->n;
  if (blda == bn) {
    sendbuf = (PetscScalar*)bv;
  } else {
    sendbuf = abt->buf[0];
    for (k = 0, i = 0; i < cK; i++) {
      for (j = 0; j < bn; j++, k++) {
        sendbuf[k] = bv[i * blda + j];
      }
    }
  }
  if (size > 1) {
    sendto = (rank + size - 1) % size;
    recvfrom = (rank + size + 1) % size;
  } else {
    sendto = recvfrom = 0;
  }
  ierr = PetscBLASIntCast(cK,&ck);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(c->A->rmap->n,&cm);CHKERRQ(ierr);
  recvisfrom = rank;
  for (i = 0; i < size; i++) {
    /* we have finished receiving in sending, bufs can be read/modified */
    PetscInt nextrecvisfrom = (recvisfrom + 1) % size; /* which process the next recvbuf will originate on */
    PetscInt nextbn = ranges[nextrecvisfrom + 1] - ranges[nextrecvisfrom];

    if (nextrecvisfrom != rank) {
      /* start the cyclic sends from sendbuf, to recvbuf (which will switch to sendbuf) */
      sendsiz = cK * bn;
      recvsiz = cK * nextbn;
      recvbuf = (i & 1) ? abt->buf[0] : abt->buf[1];
      ierr = MPI_Isend(sendbuf, sendsiz, MPIU_SCALAR, sendto, abt->tag, comm, &reqs[0]);CHKERRMPI(ierr);
      ierr = MPI_Irecv(recvbuf, recvsiz, MPIU_SCALAR, recvfrom, abt->tag, comm, &reqs[1]);CHKERRMPI(ierr);
    }

    /* local aseq * sendbuf^T */
    ierr = PetscBLASIntCast(ranges[recvisfrom + 1] - ranges[recvisfrom], &cn);CHKERRQ(ierr);
    if (cm && cn && ck) PetscStackCallBLAS("BLASgemm",BLASgemm_("N","T",&cm,&cn,&ck,&_DOne,av,&alda,sendbuf,&cn,&_DZero,cv + clda * ranges[recvisfrom],&clda));

    if (nextrecvisfrom != rank) {
      /* wait for the sends and receives to complete, swap sendbuf and recvbuf */
      ierr = MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);CHKERRMPI(ierr);
    }
    bn = nextbn;
    recvisfrom = nextrecvisfrom;
    sendbuf = recvbuf;
  }
  ierr = MatDenseRestoreArrayRead(a->A,&av);CHKERRQ(ierr);
  ierr = MatDenseRestoreArrayRead(b->A,&bv);CHKERRQ(ierr);
  ierr = MatDenseRestoreArrayWrite(c->A,&cv);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMatTransposeMultNumeric_MPIDense_MPIDense_Allgatherv(Mat A, Mat B, Mat C)
{
  Mat_MPIDense          *a=(Mat_MPIDense*)A->data, *b=(Mat_MPIDense*)B->data, *c=(Mat_MPIDense*)C->data;
  Mat_MatTransMultDense *abt;
  PetscErrorCode        ierr;
  MPI_Comm              comm;
  PetscMPIInt           size;
  PetscScalar           *cv, *sendbuf, *recvbuf;
  const PetscScalar     *av,*bv;
  PetscInt              blda,i,cK=A->cmap->N,k,j,bn;
  PetscScalar           _DOne=1.0,_DZero=0.0;
  PetscBLASInt          cm, cn, ck, alda, clda;

  PetscFunctionBegin;
  MatCheckProduct(C,3);
  if (!C->product->data) SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Product data empty");
  abt  = (Mat_MatTransMultDense*)C->product->data;
  ierr = PetscObjectGetComm((PetscObject)A,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);
  ierr = MatDenseGetArrayRead(a->A,&av);CHKERRQ(ierr);
  ierr = MatDenseGetArrayRead(b->A,&bv);CHKERRQ(ierr);
  ierr = MatDenseGetArrayWrite(c->A,&cv);CHKERRQ(ierr);
  ierr = MatDenseGetLDA(a->A,&i);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(i,&alda);CHKERRQ(ierr);
  ierr = MatDenseGetLDA(b->A,&blda);CHKERRQ(ierr);
  ierr = MatDenseGetLDA(c->A,&i);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(i,&clda);CHKERRQ(ierr);
  /* copy transpose of B into buf[0] */
  bn      = B->rmap->n;
  sendbuf = abt->buf[0];
  recvbuf = abt->buf[1];
  for (k = 0, j = 0; j < bn; j++) {
    for (i = 0; i < cK; i++, k++) {
      sendbuf[k] = bv[i * blda + j];
    }
  }
  ierr = MatDenseRestoreArrayRead(b->A,&bv);CHKERRQ(ierr);
  ierr = MPI_Allgatherv(sendbuf, bn * cK, MPIU_SCALAR, recvbuf, abt->recvcounts, abt->recvdispls, MPIU_SCALAR, comm);CHKERRMPI(ierr);
  ierr = PetscBLASIntCast(cK,&ck);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(c->A->rmap->n,&cm);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(c->A->cmap->n,&cn);CHKERRQ(ierr);
  if (cm && cn && ck) PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&cm,&cn,&ck,&_DOne,av,&alda,recvbuf,&ck,&_DZero,cv,&clda));
  ierr = MatDenseRestoreArrayRead(a->A,&av);CHKERRQ(ierr);
  ierr = MatDenseRestoreArrayRead(b->A,&bv);CHKERRQ(ierr);
  ierr = MatDenseRestoreArrayWrite(c->A,&cv);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMatTransposeMultNumeric_MPIDense_MPIDense(Mat A, Mat B, Mat C)
{
  Mat_MatTransMultDense *abt;
  PetscErrorCode        ierr;

  PetscFunctionBegin;
  MatCheckProduct(C,3);
  if (!C->product->data) SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Product data empty");
  abt = (Mat_MatTransMultDense*)C->product->data;
  switch (abt->alg) {
  case 1:
    ierr = MatMatTransposeMultNumeric_MPIDense_MPIDense_Cyclic(A, B, C);CHKERRQ(ierr);
    break;
  default:
    ierr = MatMatTransposeMultNumeric_MPIDense_MPIDense_Allgatherv(A, B, C);CHKERRQ(ierr);
    break;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatDestroy_MatMatMult_MPIDense_MPIDense(void *data)
{
  PetscErrorCode   ierr;
  Mat_MatMultDense *ab = (Mat_MatMultDense*)data;

  PetscFunctionBegin;
  ierr = MatDestroy(&ab->Ce);CHKERRQ(ierr);
  ierr = MatDestroy(&ab->Ae);CHKERRQ(ierr);
  ierr = MatDestroy(&ab->Be);CHKERRQ(ierr);
  ierr = PetscFree(ab);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_ELEMENTAL)
PetscErrorCode MatMatMultNumeric_MPIDense_MPIDense(Mat A,Mat B,Mat C)
{
  PetscErrorCode   ierr;
  Mat_MatMultDense *ab;

  PetscFunctionBegin;
  MatCheckProduct(C,3);
  if (!C->product->data) SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Missing product data");
  ab   = (Mat_MatMultDense*)C->product->data;
  ierr = MatConvert_MPIDense_Elemental(A,MATELEMENTAL,MAT_REUSE_MATRIX, &ab->Ae);CHKERRQ(ierr);
  ierr = MatConvert_MPIDense_Elemental(B,MATELEMENTAL,MAT_REUSE_MATRIX, &ab->Be);CHKERRQ(ierr);
  ierr = MatMatMultNumeric_Elemental(ab->Ae,ab->Be,ab->Ce);CHKERRQ(ierr);
  ierr = MatConvert(ab->Ce,MATMPIDENSE,MAT_REUSE_MATRIX,&C);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMatMultSymbolic_MPIDense_MPIDense(Mat A,Mat B,PetscReal fill,Mat C)
{
  PetscErrorCode   ierr;
  Mat              Ae,Be,Ce;
  Mat_MatMultDense *ab;

  PetscFunctionBegin;
  MatCheckProduct(C,4);
  if (C->product->data) SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Product data not empty");
  /* check local size of A and B */
  if (A->cmap->rstart != B->rmap->rstart || A->cmap->rend != B->rmap->rend) {
    SETERRQ4(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_SIZ,"Matrix local dimensions are incompatible, A (%D, %D) != B (%D,%D)",A->rmap->rstart,A->rmap->rend,B->rmap->rstart,B->rmap->rend);
  }

  /* create elemental matrices Ae and Be */
  ierr = MatCreate(PetscObjectComm((PetscObject)A), &Ae);CHKERRQ(ierr);
  ierr = MatSetSizes(Ae,PETSC_DECIDE,PETSC_DECIDE,A->rmap->N,A->cmap->N);CHKERRQ(ierr);
  ierr = MatSetType(Ae,MATELEMENTAL);CHKERRQ(ierr);
  ierr = MatSetUp(Ae);CHKERRQ(ierr);
  ierr = MatSetOption(Ae,MAT_ROW_ORIENTED,PETSC_FALSE);CHKERRQ(ierr);

  ierr = MatCreate(PetscObjectComm((PetscObject)B), &Be);CHKERRQ(ierr);
  ierr = MatSetSizes(Be,PETSC_DECIDE,PETSC_DECIDE,B->rmap->N,B->cmap->N);CHKERRQ(ierr);
  ierr = MatSetType(Be,MATELEMENTAL);CHKERRQ(ierr);
  ierr = MatSetUp(Be);CHKERRQ(ierr);
  ierr = MatSetOption(Be,MAT_ROW_ORIENTED,PETSC_FALSE);CHKERRQ(ierr);

  /* compute symbolic Ce = Ae*Be */
  ierr = MatCreate(PetscObjectComm((PetscObject)C),&Ce);CHKERRQ(ierr);
  ierr = MatMatMultSymbolic_Elemental(Ae,Be,fill,Ce);CHKERRQ(ierr);

  /* setup C */
  ierr = MatSetSizes(C,A->rmap->n,B->cmap->n,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = MatSetType(C,MATDENSE);CHKERRQ(ierr);
  ierr = MatSetUp(C);CHKERRQ(ierr);

  /* create data structure for reuse Cdense */
  ierr = PetscNew(&ab);CHKERRQ(ierr);
  ab->Ae = Ae;
  ab->Be = Be;
  ab->Ce = Ce;

  C->product->data    = ab;
  C->product->destroy = MatDestroy_MatMatMult_MPIDense_MPIDense;
  C->ops->matmultnumeric = MatMatMultNumeric_MPIDense_MPIDense;
  PetscFunctionReturn(0);
}
#endif
/* ----------------------------------------------- */
#if defined(PETSC_HAVE_ELEMENTAL)
static PetscErrorCode MatProductSetFromOptions_MPIDense_AB(Mat C)
{
  PetscFunctionBegin;
  C->ops->matmultsymbolic = MatMatMultSymbolic_MPIDense_MPIDense;
  C->ops->productsymbolic = MatProductSymbolic_AB;
  PetscFunctionReturn(0);
}
#endif

static PetscErrorCode MatProductSetFromOptions_MPIDense_AtB(Mat C)
{
  Mat_Product *product = C->product;
  Mat         A = product->A,B=product->B;

  PetscFunctionBegin;
  if (A->rmap->rstart != B->rmap->rstart || A->rmap->rend != B->rmap->rend)
    SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Matrix local dimensions are incompatible, (%D, %D) != (%D,%D)",A->rmap->rstart,A->rmap->rend,B->rmap->rstart,B->rmap->rend);
  C->ops->transposematmultsymbolic = MatTransposeMatMultSymbolic_MPIDense_MPIDense;
  C->ops->productsymbolic = MatProductSymbolic_AtB;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatProductSetFromOptions_MPIDense_ABt(Mat C)
{
  PetscErrorCode ierr;
  Mat_Product    *product = C->product;
  const char     *algTypes[2] = {"allgatherv","cyclic"};
  PetscInt       alg,nalg = 2;
  PetscBool      flg = PETSC_FALSE;

  PetscFunctionBegin;
  /* Set default algorithm */
  alg = 0; /* default is allgatherv */
  ierr = PetscStrcmp(product->alg,"default",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = MatProductSetAlgorithm(C,(MatProductAlgorithm)algTypes[alg]);CHKERRQ(ierr);
  }

  /* Get runtime option */
  if (product->api_user) {
    ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)C),((PetscObject)C)->prefix,"MatMatTransposeMult","Mat");CHKERRQ(ierr);
    ierr = PetscOptionsEList("-matmattransmult_mpidense_mpidense_via","Algorithmic approach","MatMatTransposeMult",algTypes,nalg,algTypes[alg],&alg,&flg);CHKERRQ(ierr);
    ierr = PetscOptionsEnd();CHKERRQ(ierr);
  } else {
    ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)C),((PetscObject)C)->prefix,"MatProduct_ABt","Mat");CHKERRQ(ierr);
    ierr = PetscOptionsEList("-matproduct_abt_mpidense_mpidense_via","Algorithmic approach","MatProduct_ABt",algTypes,nalg,algTypes[alg],&alg,&flg);CHKERRQ(ierr);
    ierr = PetscOptionsEnd();CHKERRQ(ierr);
  }
  if (flg) {
    ierr = MatProductSetAlgorithm(C,(MatProductAlgorithm)algTypes[alg]);CHKERRQ(ierr);
  }

  C->ops->mattransposemultsymbolic = MatMatTransposeMultSymbolic_MPIDense_MPIDense;
  C->ops->productsymbolic          = MatProductSymbolic_ABt;
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatProductSetFromOptions_MPIDense(Mat C)
{
  PetscErrorCode ierr;
  Mat_Product    *product = C->product;

  PetscFunctionBegin;
  switch (product->type) {
#if defined(PETSC_HAVE_ELEMENTAL)
  case MATPRODUCT_AB:
    ierr = MatProductSetFromOptions_MPIDense_AB(C);CHKERRQ(ierr);
    break;
#endif
  case MATPRODUCT_AtB:
    ierr = MatProductSetFromOptions_MPIDense_AtB(C);CHKERRQ(ierr);
    break;
  case MATPRODUCT_ABt:
    ierr = MatProductSetFromOptions_MPIDense_ABt(C);CHKERRQ(ierr);
    break;
  default:
    break;
  }
  PetscFunctionReturn(0);
}
