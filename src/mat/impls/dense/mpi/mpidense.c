
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
  PetscBool      flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidPointer(B,2);
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)A,MATMPIDENSE,&flg));
  if (flg) *B = mat->A;
  else {
    PetscCall(PetscObjectBaseTypeCompare((PetscObject)A,MATSEQDENSE,&flg));
    PetscCheck(flg,PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Not for matrix type %s",((PetscObject)A)->type_name);
    *B = A;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatCopy_MPIDense(Mat A, Mat B, MatStructure s)
{
  Mat_MPIDense   *Amat = (Mat_MPIDense*)A->data;
  Mat_MPIDense   *Bmat = (Mat_MPIDense*)B->data;

  PetscFunctionBegin;
  PetscCall(MatCopy(Amat->A,Bmat->A,s));
  PetscFunctionReturn(0);
}

PetscErrorCode MatShift_MPIDense(Mat A,PetscScalar alpha)
{
  Mat_MPIDense   *mat = (Mat_MPIDense*)A->data;
  PetscInt       j,lda,rstart = A->rmap->rstart,rend = A->rmap->rend,rend2;
  PetscScalar    *v;

  PetscFunctionBegin;
  PetscCall(MatDenseGetArray(mat->A,&v));
  PetscCall(MatDenseGetLDA(mat->A,&lda));
  rend2 = PetscMin(rend,A->cmap->N);
  if (rend2>rstart) {
    for (j=rstart; j<rend2; j++) v[j-rstart+j*lda] += alpha;
    PetscCall(PetscLogFlops(rend2-rstart));
  }
  PetscCall(MatDenseRestoreArray(mat->A,&v));
  PetscFunctionReturn(0);
}

PetscErrorCode MatGetRow_MPIDense(Mat A,PetscInt row,PetscInt *nz,PetscInt **idx,PetscScalar **v)
{
  Mat_MPIDense   *mat = (Mat_MPIDense*)A->data;
  PetscInt       lrow,rstart = A->rmap->rstart,rend = A->rmap->rend;

  PetscFunctionBegin;
  PetscCheck(row >= rstart && row < rend,PETSC_COMM_SELF,PETSC_ERR_SUP,"only local rows");
  lrow = row - rstart;
  PetscCall(MatGetRow(mat->A,lrow,nz,(const PetscInt**)idx,(const PetscScalar**)v));
  PetscFunctionReturn(0);
}

PetscErrorCode MatRestoreRow_MPIDense(Mat A,PetscInt row,PetscInt *nz,PetscInt **idx,PetscScalar **v)
{
  Mat_MPIDense   *mat = (Mat_MPIDense*)A->data;
  PetscInt       lrow,rstart = A->rmap->rstart,rend = A->rmap->rend;

  PetscFunctionBegin;
  PetscCheck(row >= rstart && row < rend,PETSC_COMM_SELF,PETSC_ERR_SUP,"only local rows");
  lrow = row - rstart;
  PetscCall(MatRestoreRow(mat->A,lrow,nz,(const PetscInt**)idx,(const PetscScalar**)v));
  PetscFunctionReturn(0);
}

PetscErrorCode  MatGetDiagonalBlock_MPIDense(Mat A,Mat *a)
{
  Mat_MPIDense   *mdn = (Mat_MPIDense*)A->data;
  PetscInt       m = A->rmap->n,rstart = A->rmap->rstart;
  PetscScalar    *array;
  MPI_Comm       comm;
  PetscBool      flg;
  Mat            B;

  PetscFunctionBegin;
  PetscCall(MatHasCongruentLayouts(A,&flg));
  PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_SUP,"Only square matrices supported.");
  PetscCall(PetscObjectQuery((PetscObject)A,"DiagonalBlock",(PetscObject*)&B));
  if (!B) { /* This should use MatDenseGetSubMatrix (not create), but we would need a call like MatRestoreDiagonalBlock */

    PetscCall(PetscObjectTypeCompare((PetscObject)mdn->A,MATSEQDENSECUDA,&flg));
    PetscCheck(!flg,PETSC_COMM_SELF,PETSC_ERR_SUP,"Not coded for %s. Send an email to petsc-dev@mcs.anl.gov to request this feature",MATSEQDENSECUDA);
    PetscCall(PetscObjectGetComm((PetscObject)(mdn->A),&comm));
    PetscCall(MatCreate(comm,&B));
    PetscCall(MatSetSizes(B,m,m,m,m));
    PetscCall(MatSetType(B,((PetscObject)mdn->A)->type_name));
    PetscCall(MatDenseGetArrayRead(mdn->A,(const PetscScalar**)&array));
    PetscCall(MatSeqDenseSetPreallocation(B,array+m*rstart));
    PetscCall(MatDenseRestoreArrayRead(mdn->A,(const PetscScalar**)&array));
    PetscCall(PetscObjectCompose((PetscObject)A,"DiagonalBlock",(PetscObject)B));
    *a   = B;
    PetscCall(MatDestroy(&B));
  } else *a = B;
  PetscFunctionReturn(0);
}

PetscErrorCode MatSetValues_MPIDense(Mat mat,PetscInt m,const PetscInt idxm[],PetscInt n,const PetscInt idxn[],const PetscScalar v[],InsertMode addv)
{
  Mat_MPIDense   *A = (Mat_MPIDense*)mat->data;
  PetscInt       i,j,rstart = mat->rmap->rstart,rend = mat->rmap->rend,row;
  PetscBool      roworiented = A->roworiented;

  PetscFunctionBegin;
  for (i=0; i<m; i++) {
    if (idxm[i] < 0) continue;
    PetscCheck(idxm[i] < mat->rmap->N,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Row too large");
    if (idxm[i] >= rstart && idxm[i] < rend) {
      row = idxm[i] - rstart;
      if (roworiented) {
        PetscCall(MatSetValues(A->A,1,&row,n,idxn,v+i*n,addv));
      } else {
        for (j=0; j<n; j++) {
          if (idxn[j] < 0) continue;
          PetscCheck(idxn[j] < mat->cmap->N,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Column too large");
          PetscCall(MatSetValues(A->A,1,&row,1,&idxn[j],v+i+j*m,addv));
        }
      }
    } else if (!A->donotstash) {
      mat->assembled = PETSC_FALSE;
      if (roworiented) {
        PetscCall(MatStashValuesRow_Private(&mat->stash,idxm[i],n,idxn,v+i*n,PETSC_FALSE));
      } else {
        PetscCall(MatStashValuesCol_Private(&mat->stash,idxm[i],n,idxn,v+i,m,PETSC_FALSE));
      }
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatGetValues_MPIDense(Mat mat,PetscInt m,const PetscInt idxm[],PetscInt n,const PetscInt idxn[],PetscScalar v[])
{
  Mat_MPIDense   *mdn = (Mat_MPIDense*)mat->data;
  PetscInt       i,j,rstart = mat->rmap->rstart,rend = mat->rmap->rend,row;

  PetscFunctionBegin;
  for (i=0; i<m; i++) {
    if (idxm[i] < 0) continue; /* negative row */
    PetscCheck(idxm[i] < mat->rmap->N,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Row too large");
    if (idxm[i] >= rstart && idxm[i] < rend) {
      row = idxm[i] - rstart;
      for (j=0; j<n; j++) {
        if (idxn[j] < 0) continue; /* negative column */
        PetscCheck(idxn[j] < mat->cmap->N,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Column too large");
        PetscCall(MatGetValues(mdn->A,1,&row,1,&idxn[j],v+i*n+j));
      }
    } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Only local values currently supported");
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseGetLDA_MPIDense(Mat A,PetscInt *lda)
{
  Mat_MPIDense   *a = (Mat_MPIDense*)A->data;

  PetscFunctionBegin;
  PetscCall(MatDenseGetLDA(a->A,lda));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseSetLDA_MPIDense(Mat A,PetscInt lda)
{
  Mat_MPIDense   *a = (Mat_MPIDense*)A->data;
  PetscBool      iscuda;

  PetscFunctionBegin;
  if (!a->A) {
    PetscCheck(!a->matinuse,PetscObjectComm((PetscObject)A),PETSC_ERR_ORDER,"Need to call MatDenseRestoreSubMatrix() first");
    PetscCall(PetscLayoutSetUp(A->rmap));
    PetscCall(PetscLayoutSetUp(A->cmap));
    PetscCall(MatCreate(PETSC_COMM_SELF,&a->A));
    PetscCall(PetscLogObjectParent((PetscObject)A,(PetscObject)a->A));
    PetscCall(MatSetSizes(a->A,A->rmap->n,A->cmap->N,A->rmap->n,A->cmap->N));
    PetscCall(PetscObjectTypeCompare((PetscObject)A,MATMPIDENSECUDA,&iscuda));
    PetscCall(MatSetType(a->A,iscuda ? MATSEQDENSECUDA : MATSEQDENSE));
  }
  PetscCall(MatDenseSetLDA(a->A,lda));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseGetArray_MPIDense(Mat A,PetscScalar **array)
{
  Mat_MPIDense   *a = (Mat_MPIDense*)A->data;

  PetscFunctionBegin;
  PetscCheck(!a->matinuse,PetscObjectComm((PetscObject)A),PETSC_ERR_ORDER,"Need to call MatDenseRestoreSubMatrix() first");
  PetscCall(MatDenseGetArray(a->A,array));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseGetArrayRead_MPIDense(Mat A,const PetscScalar **array)
{
  Mat_MPIDense   *a = (Mat_MPIDense*)A->data;

  PetscFunctionBegin;
  PetscCheck(!a->matinuse,PetscObjectComm((PetscObject)A),PETSC_ERR_ORDER,"Need to call MatDenseRestoreSubMatrix() first");
  PetscCall(MatDenseGetArrayRead(a->A,array));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseGetArrayWrite_MPIDense(Mat A,PetscScalar **array)
{
  Mat_MPIDense   *a = (Mat_MPIDense*)A->data;

  PetscFunctionBegin;
  PetscCheck(!a->matinuse,PetscObjectComm((PetscObject)A),PETSC_ERR_ORDER,"Need to call MatDenseRestoreSubMatrix() first");
  PetscCall(MatDenseGetArrayWrite(a->A,array));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDensePlaceArray_MPIDense(Mat A,const PetscScalar *array)
{
  Mat_MPIDense   *a = (Mat_MPIDense*)A->data;

  PetscFunctionBegin;
  PetscCheck(!a->vecinuse,PetscObjectComm((PetscObject)A),PETSC_ERR_ORDER,"Need to call MatDenseRestoreColumnVec() first");
  PetscCheck(!a->matinuse,PetscObjectComm((PetscObject)A),PETSC_ERR_ORDER,"Need to call MatDenseRestoreSubMatrix() first");
  PetscCall(MatDensePlaceArray(a->A,array));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseResetArray_MPIDense(Mat A)
{
  Mat_MPIDense   *a = (Mat_MPIDense*)A->data;

  PetscFunctionBegin;
  PetscCheck(!a->vecinuse,PetscObjectComm((PetscObject)A),PETSC_ERR_ORDER,"Need to call MatDenseRestoreColumnVec() first");
  PetscCheck(!a->matinuse,PetscObjectComm((PetscObject)A),PETSC_ERR_ORDER,"Need to call MatDenseRestoreSubMatrix() first");
  PetscCall(MatDenseResetArray(a->A));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseReplaceArray_MPIDense(Mat A,const PetscScalar *array)
{
  Mat_MPIDense   *a = (Mat_MPIDense*)A->data;

  PetscFunctionBegin;
  PetscCheck(!a->vecinuse,PetscObjectComm((PetscObject)A),PETSC_ERR_ORDER,"Need to call MatDenseRestoreColumnVec() first");
  PetscCheck(!a->matinuse,PetscObjectComm((PetscObject)A),PETSC_ERR_ORDER,"Need to call MatDenseRestoreSubMatrix() first");
  PetscCall(MatDenseReplaceArray(a->A,array));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatCreateSubMatrix_MPIDense(Mat A,IS isrow,IS iscol,MatReuse scall,Mat *B)
{
  Mat_MPIDense      *mat  = (Mat_MPIDense*)A->data,*newmatd;
  PetscInt          lda,i,j,rstart,rend,nrows,ncols,Ncols,nlrows,nlcols;
  const PetscInt    *irow,*icol;
  const PetscScalar *v;
  PetscScalar       *bv;
  Mat               newmat;
  IS                iscol_local;
  MPI_Comm          comm_is,comm_mat;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)A,&comm_mat));
  PetscCall(PetscObjectGetComm((PetscObject)iscol,&comm_is));
  PetscCheck(comm_mat == comm_is,PETSC_COMM_SELF,PETSC_ERR_ARG_NOTSAMECOMM,"IS communicator must match matrix communicator");

  PetscCall(ISAllGather(iscol,&iscol_local));
  PetscCall(ISGetIndices(isrow,&irow));
  PetscCall(ISGetIndices(iscol_local,&icol));
  PetscCall(ISGetLocalSize(isrow,&nrows));
  PetscCall(ISGetLocalSize(iscol,&ncols));
  PetscCall(ISGetSize(iscol,&Ncols)); /* global number of columns, size of iscol_local */

  /* No parallel redistribution currently supported! Should really check each index set
     to comfirm that it is OK.  ... Currently supports only submatrix same partitioning as
     original matrix! */

  PetscCall(MatGetLocalSize(A,&nlrows,&nlcols));
  PetscCall(MatGetOwnershipRange(A,&rstart,&rend));

  /* Check submatrix call */
  if (scall == MAT_REUSE_MATRIX) {
    /* SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Reused submatrix wrong size"); */
    /* Really need to test rows and column sizes! */
    newmat = *B;
  } else {
    /* Create and fill new matrix */
    PetscCall(MatCreate(PetscObjectComm((PetscObject)A),&newmat));
    PetscCall(MatSetSizes(newmat,nrows,ncols,PETSC_DECIDE,Ncols));
    PetscCall(MatSetType(newmat,((PetscObject)A)->type_name));
    PetscCall(MatMPIDenseSetPreallocation(newmat,NULL));
  }

  /* Now extract the data pointers and do the copy, column at a time */
  newmatd = (Mat_MPIDense*)newmat->data;
  PetscCall(MatDenseGetArray(newmatd->A,&bv));
  PetscCall(MatDenseGetArrayRead(mat->A,&v));
  PetscCall(MatDenseGetLDA(mat->A,&lda));
  for (i=0; i<Ncols; i++) {
    const PetscScalar *av = v + lda*icol[i];
    for (j=0; j<nrows; j++) {
      *bv++ = av[irow[j] - rstart];
    }
  }
  PetscCall(MatDenseRestoreArrayRead(mat->A,&v));
  PetscCall(MatDenseRestoreArray(newmatd->A,&bv));

  /* Assemble the matrices so that the correct flags are set */
  PetscCall(MatAssemblyBegin(newmat,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(newmat,MAT_FINAL_ASSEMBLY));

  /* Free work space */
  PetscCall(ISRestoreIndices(isrow,&irow));
  PetscCall(ISRestoreIndices(iscol_local,&icol));
  PetscCall(ISDestroy(&iscol_local));
  *B   = newmat;
  PetscFunctionReturn(0);
}

PetscErrorCode MatDenseRestoreArray_MPIDense(Mat A,PetscScalar **array)
{
  Mat_MPIDense   *a = (Mat_MPIDense*)A->data;

  PetscFunctionBegin;
  PetscCall(MatDenseRestoreArray(a->A,array));
  PetscFunctionReturn(0);
}

PetscErrorCode MatDenseRestoreArrayRead_MPIDense(Mat A,const PetscScalar **array)
{
  Mat_MPIDense   *a = (Mat_MPIDense*)A->data;

  PetscFunctionBegin;
  PetscCall(MatDenseRestoreArrayRead(a->A,array));
  PetscFunctionReturn(0);
}

PetscErrorCode MatDenseRestoreArrayWrite_MPIDense(Mat A,PetscScalar **array)
{
  Mat_MPIDense   *a = (Mat_MPIDense*)A->data;

  PetscFunctionBegin;
  PetscCall(MatDenseRestoreArrayWrite(a->A,array));
  PetscFunctionReturn(0);
}

PetscErrorCode MatAssemblyBegin_MPIDense(Mat mat,MatAssemblyType mode)
{
  Mat_MPIDense   *mdn = (Mat_MPIDense*)mat->data;
  PetscInt       nstash,reallocs;

  PetscFunctionBegin;
  if (mdn->donotstash || mat->nooffprocentries) PetscFunctionReturn(0);

  PetscCall(MatStashScatterBegin_Private(mat,&mat->stash,mat->rmap->range));
  PetscCall(MatStashGetInfo_Private(&mat->stash,&nstash,&reallocs));
  PetscCall(PetscInfo(mdn->A,"Stash has %" PetscInt_FMT " entries, uses %" PetscInt_FMT " mallocs.\n",nstash,reallocs));
  PetscFunctionReturn(0);
}

PetscErrorCode MatAssemblyEnd_MPIDense(Mat mat,MatAssemblyType mode)
{
  Mat_MPIDense   *mdn=(Mat_MPIDense*)mat->data;
  PetscInt       i,*row,*col,flg,j,rstart,ncols;
  PetscMPIInt    n;
  PetscScalar    *val;

  PetscFunctionBegin;
  if (!mdn->donotstash && !mat->nooffprocentries) {
    /*  wait on receives */
    while (1) {
      PetscCall(MatStashScatterGetMesg_Private(&mat->stash,&n,&row,&col,&val,&flg));
      if (!flg) break;

      for (i=0; i<n;) {
        /* Now identify the consecutive vals belonging to the same row */
        for (j=i,rstart=row[j]; j<n; j++) {
          if (row[j] != rstart) break;
        }
        if (j < n) ncols = j-i;
        else       ncols = n-i;
        /* Now assemble all these values with a single function call */
        PetscCall(MatSetValues_MPIDense(mat,1,row+i,ncols,col+i,val+i,mat->insertmode));
        i    = j;
      }
    }
    PetscCall(MatStashScatterEnd_Private(&mat->stash));
  }

  PetscCall(MatAssemblyBegin(mdn->A,mode));
  PetscCall(MatAssemblyEnd(mdn->A,mode));
  PetscFunctionReturn(0);
}

PetscErrorCode MatZeroEntries_MPIDense(Mat A)
{
  Mat_MPIDense   *l = (Mat_MPIDense*)A->data;

  PetscFunctionBegin;
  PetscCall(MatZeroEntries(l->A));
  PetscFunctionReturn(0);
}

PetscErrorCode MatZeroRows_MPIDense(Mat A,PetscInt n,const PetscInt rows[],PetscScalar diag,Vec x,Vec b)
{
  Mat_MPIDense      *l = (Mat_MPIDense*)A->data;
  PetscInt          i,len,*lrows;

  PetscFunctionBegin;
  /* get locally owned rows */
  PetscCall(PetscLayoutMapLocal(A->rmap,n,rows,&len,&lrows,NULL));
  /* fix right hand side if needed */
  if (x && b) {
    const PetscScalar *xx;
    PetscScalar       *bb;

    PetscCall(VecGetArrayRead(x, &xx));
    PetscCall(VecGetArrayWrite(b, &bb));
    for (i=0;i<len;++i) bb[lrows[i]] = diag*xx[lrows[i]];
    PetscCall(VecRestoreArrayRead(x, &xx));
    PetscCall(VecRestoreArrayWrite(b, &bb));
  }
  PetscCall(MatZeroRows(l->A,len,lrows,0.0,NULL,NULL));
  if (diag != 0.0) {
    Vec d;

    PetscCall(MatCreateVecs(A,NULL,&d));
    PetscCall(VecSet(d,diag));
    PetscCall(MatDiagonalSet(A,d,INSERT_VALUES));
    PetscCall(VecDestroy(&d));
  }
  PetscCall(PetscFree(lrows));
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatMult_SeqDense(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatMultAdd_SeqDense(Mat,Vec,Vec,Vec);
PETSC_INTERN PetscErrorCode MatMultTranspose_SeqDense(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatMultTransposeAdd_SeqDense(Mat,Vec,Vec,Vec);

PetscErrorCode MatMult_MPIDense(Mat mat,Vec xx,Vec yy)
{
  Mat_MPIDense      *mdn = (Mat_MPIDense*)mat->data;
  const PetscScalar *ax;
  PetscScalar       *ay;
  PetscMemType      axmtype,aymtype;

  PetscFunctionBegin;
  if (!mdn->Mvctx) PetscCall(MatSetUpMultiply_MPIDense(mat));
  PetscCall(VecGetArrayReadAndMemType(xx,&ax,&axmtype));
  PetscCall(VecGetArrayAndMemType(mdn->lvec,&ay,&aymtype));
  PetscCall(PetscSFBcastWithMemTypeBegin(mdn->Mvctx,MPIU_SCALAR,axmtype,ax,aymtype,ay,MPI_REPLACE));
  PetscCall(PetscSFBcastEnd(mdn->Mvctx,MPIU_SCALAR,ax,ay,MPI_REPLACE));
  PetscCall(VecRestoreArrayAndMemType(mdn->lvec,&ay));
  PetscCall(VecRestoreArrayReadAndMemType(xx,&ax));
  PetscCall((*mdn->A->ops->mult)(mdn->A,mdn->lvec,yy));
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultAdd_MPIDense(Mat mat,Vec xx,Vec yy,Vec zz)
{
  Mat_MPIDense      *mdn = (Mat_MPIDense*)mat->data;
  const PetscScalar *ax;
  PetscScalar       *ay;
  PetscMemType      axmtype,aymtype;

  PetscFunctionBegin;
  if (!mdn->Mvctx) PetscCall(MatSetUpMultiply_MPIDense(mat));
  PetscCall(VecGetArrayReadAndMemType(xx,&ax,&axmtype));
  PetscCall(VecGetArrayAndMemType(mdn->lvec,&ay,&aymtype));
  PetscCall(PetscSFBcastWithMemTypeBegin(mdn->Mvctx,MPIU_SCALAR,axmtype,ax,aymtype,ay,MPI_REPLACE));
  PetscCall(PetscSFBcastEnd(mdn->Mvctx,MPIU_SCALAR,ax,ay,MPI_REPLACE));
  PetscCall(VecRestoreArrayAndMemType(mdn->lvec,&ay));
  PetscCall(VecRestoreArrayReadAndMemType(xx,&ax));
  PetscCall((*mdn->A->ops->multadd)(mdn->A,mdn->lvec,yy,zz));
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultTranspose_MPIDense(Mat A,Vec xx,Vec yy)
{
  Mat_MPIDense      *a = (Mat_MPIDense*)A->data;
  const PetscScalar *ax;
  PetscScalar       *ay;
  PetscMemType      axmtype,aymtype;

  PetscFunctionBegin;
  if (!a->Mvctx) PetscCall(MatSetUpMultiply_MPIDense(A));
  PetscCall(VecSet(yy,0.0));
  PetscCall((*a->A->ops->multtranspose)(a->A,xx,a->lvec));
  PetscCall(VecGetArrayReadAndMemType(a->lvec,&ax,&axmtype));
  PetscCall(VecGetArrayAndMemType(yy,&ay,&aymtype));
  PetscCall(PetscSFReduceWithMemTypeBegin(a->Mvctx,MPIU_SCALAR,axmtype,ax,aymtype,ay,MPIU_SUM));
  PetscCall(PetscSFReduceEnd(a->Mvctx,MPIU_SCALAR,ax,ay,MPIU_SUM));
  PetscCall(VecRestoreArrayReadAndMemType(a->lvec,&ax));
  PetscCall(VecRestoreArrayAndMemType(yy,&ay));
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultTransposeAdd_MPIDense(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_MPIDense      *a = (Mat_MPIDense*)A->data;
  const PetscScalar *ax;
  PetscScalar       *ay;
  PetscMemType      axmtype,aymtype;

  PetscFunctionBegin;
  if (!a->Mvctx) PetscCall(MatSetUpMultiply_MPIDense(A));
  PetscCall(VecCopy(yy,zz));
  PetscCall((*a->A->ops->multtranspose)(a->A,xx,a->lvec));
  PetscCall(VecGetArrayReadAndMemType(a->lvec,&ax,&axmtype));
  PetscCall(VecGetArrayAndMemType(zz,&ay,&aymtype));
  PetscCall(PetscSFReduceWithMemTypeBegin(a->Mvctx,MPIU_SCALAR,axmtype,ax,aymtype,ay,MPIU_SUM));
  PetscCall(PetscSFReduceEnd(a->Mvctx,MPIU_SCALAR,ax,ay,MPIU_SUM));
  PetscCall(VecRestoreArrayReadAndMemType(a->lvec,&ax));
  PetscCall(VecRestoreArrayAndMemType(zz,&ay));
  PetscFunctionReturn(0);
}

PetscErrorCode MatGetDiagonal_MPIDense(Mat A,Vec v)
{
  Mat_MPIDense      *a    = (Mat_MPIDense*)A->data;
  PetscInt          lda,len,i,n,m = A->rmap->n,radd;
  PetscScalar       *x,zero = 0.0;
  const PetscScalar *av;

  PetscFunctionBegin;
  PetscCall(VecSet(v,zero));
  PetscCall(VecGetArray(v,&x));
  PetscCall(VecGetSize(v,&n));
  PetscCheck(n == A->rmap->N,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Nonconforming mat and vec");
  len  = PetscMin(a->A->rmap->n,a->A->cmap->n);
  radd = A->rmap->rstart*m;
  PetscCall(MatDenseGetArrayRead(a->A,&av));
  PetscCall(MatDenseGetLDA(a->A,&lda));
  for (i=0; i<len; i++) {
    x[i] = av[radd + i*lda + i];
  }
  PetscCall(MatDenseRestoreArrayRead(a->A,&av));
  PetscCall(VecRestoreArray(v,&x));
  PetscFunctionReturn(0);
}

PetscErrorCode MatDestroy_MPIDense(Mat mat)
{
  Mat_MPIDense   *mdn = (Mat_MPIDense*)mat->data;

  PetscFunctionBegin;
#if defined(PETSC_USE_LOG)
  PetscLogObjectState((PetscObject)mat,"Rows=%" PetscInt_FMT ", Cols=%" PetscInt_FMT,mat->rmap->N,mat->cmap->N);
#endif
  PetscCall(MatStashDestroy_Private(&mat->stash));
  PetscCheck(!mdn->vecinuse,PetscObjectComm((PetscObject)mat),PETSC_ERR_ORDER,"Need to call MatDenseRestoreColumnVec() first");
  PetscCheck(!mdn->matinuse,PetscObjectComm((PetscObject)mat),PETSC_ERR_ORDER,"Need to call MatDenseRestoreSubMatrix() first");
  PetscCall(MatDestroy(&mdn->A));
  PetscCall(VecDestroy(&mdn->lvec));
  PetscCall(PetscSFDestroy(&mdn->Mvctx));
  PetscCall(VecDestroy(&mdn->cvec));
  PetscCall(MatDestroy(&mdn->cmat));

  PetscCall(PetscFree(mat->data));
  PetscCall(PetscObjectChangeTypeName((PetscObject)mat,NULL));

  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatDenseGetLDA_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatDenseSetLDA_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatDenseGetArray_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatDenseRestoreArray_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatDenseGetArrayRead_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatDenseRestoreArrayRead_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatDenseGetArrayWrite_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatDenseRestoreArrayWrite_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatDensePlaceArray_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatDenseResetArray_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatDenseReplaceArray_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatConvert_mpiaij_mpidense_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatConvert_mpidense_mpiaij_C",NULL));
#if defined(PETSC_HAVE_ELEMENTAL)
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatConvert_mpidense_elemental_C",NULL));
#endif
#if defined(PETSC_HAVE_SCALAPACK)
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatConvert_mpidense_scalapack_C",NULL));
#endif
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatMPIDenseSetPreallocation_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatProductSetFromOptions_mpiaij_mpidense_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatProductSetFromOptions_mpidense_mpiaij_C",NULL));
#if defined (PETSC_HAVE_CUDA)
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatProductSetFromOptions_mpiaijcusparse_mpidense_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatProductSetFromOptions_mpidense_mpiaijcusparse_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatConvert_mpidense_mpidensecuda_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatConvert_mpidensecuda_mpidense_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatProductSetFromOptions_mpiaij_mpidensecuda_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatProductSetFromOptions_mpiaijcusparse_mpidensecuda_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatProductSetFromOptions_mpidensecuda_mpiaij_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatProductSetFromOptions_mpidensecuda_mpiaijcusparse_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatDenseCUDAGetArray_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatDenseCUDAGetArrayRead_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatDenseCUDAGetArrayWrite_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatDenseCUDARestoreArray_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatDenseCUDARestoreArrayRead_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatDenseCUDARestoreArrayWrite_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatDenseCUDAPlaceArray_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatDenseCUDAResetArray_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatDenseCUDAReplaceArray_C",NULL));
#endif
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatDenseGetColumn_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatDenseRestoreColumn_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatDenseGetColumnVec_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatDenseRestoreColumnVec_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatDenseGetColumnVecRead_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatDenseRestoreColumnVecRead_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatDenseGetColumnVecWrite_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatDenseRestoreColumnVecWrite_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatDenseGetSubMatrix_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatDenseRestoreSubMatrix_C",NULL));

  PetscCall(PetscObjectCompose((PetscObject)mat,"DiagonalBlock",NULL));
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatView_SeqDense(Mat,PetscViewer);

#include <petscdraw.h>
static PetscErrorCode MatView_MPIDense_ASCIIorDraworSocket(Mat mat,PetscViewer viewer)
{
  Mat_MPIDense      *mdn = (Mat_MPIDense*)mat->data;
  PetscMPIInt       rank;
  PetscViewerType   vtype;
  PetscBool         iascii,isdraw;
  PetscViewer       sviewer;
  PetscViewerFormat format;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)mat),&rank));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERDRAW,&isdraw));
  if (iascii) {
    PetscCall(PetscViewerGetType(viewer,&vtype));
    PetscCall(PetscViewerGetFormat(viewer,&format));
    if (format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
      MatInfo info;
      PetscCall(MatGetInfo(mat,MAT_LOCAL,&info));
      PetscCall(PetscViewerASCIIPushSynchronized(viewer));
      PetscCall(PetscViewerASCIISynchronizedPrintf(viewer,"  [%d] local rows %" PetscInt_FMT " nz %" PetscInt_FMT " nz alloced %" PetscInt_FMT " mem %" PetscInt_FMT " \n",rank,mat->rmap->n,(PetscInt)info.nz_used,(PetscInt)info.nz_allocated,(PetscInt)info.memory));
      PetscCall(PetscViewerFlush(viewer));
      PetscCall(PetscViewerASCIIPopSynchronized(viewer));
      if (mdn->Mvctx) PetscCall(PetscSFView(mdn->Mvctx,viewer));
      PetscFunctionReturn(0);
    } else if (format == PETSC_VIEWER_ASCII_INFO) {
      PetscFunctionReturn(0);
    }
  } else if (isdraw) {
    PetscDraw draw;
    PetscBool isnull;

    PetscCall(PetscViewerDrawGetDraw(viewer,0,&draw));
    PetscCall(PetscDrawIsNull(draw,&isnull));
    if (isnull) PetscFunctionReturn(0);
  }

  {
    /* assemble the entire matrix onto first processor. */
    Mat         A;
    PetscInt    M = mat->rmap->N,N = mat->cmap->N,m,row,i,nz;
    PetscInt    *cols;
    PetscScalar *vals;

    PetscCall(MatCreate(PetscObjectComm((PetscObject)mat),&A));
    if (rank == 0) {
      PetscCall(MatSetSizes(A,M,N,M,N));
    } else {
      PetscCall(MatSetSizes(A,0,0,M,N));
    }
    /* Since this is a temporary matrix, MATMPIDENSE instead of ((PetscObject)A)->type_name here is probably acceptable. */
    PetscCall(MatSetType(A,MATMPIDENSE));
    PetscCall(MatMPIDenseSetPreallocation(A,NULL));
    PetscCall(PetscLogObjectParent((PetscObject)mat,(PetscObject)A));

    /* Copy the matrix ... This isn't the most efficient means,
       but it's quick for now */
    A->insertmode = INSERT_VALUES;

    row = mat->rmap->rstart;
    m   = mdn->A->rmap->n;
    for (i=0; i<m; i++) {
      PetscCall(MatGetRow_MPIDense(mat,row,&nz,&cols,&vals));
      PetscCall(MatSetValues_MPIDense(A,1,&row,nz,cols,vals,INSERT_VALUES));
      PetscCall(MatRestoreRow_MPIDense(mat,row,&nz,&cols,&vals));
      row++;
    }

    PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
    PetscCall(PetscViewerGetSubViewer(viewer,PETSC_COMM_SELF,&sviewer));
    if (rank == 0) {
      PetscCall(PetscObjectSetName((PetscObject)((Mat_MPIDense*)(A->data))->A,((PetscObject)mat)->name));
      PetscCall(MatView_SeqDense(((Mat_MPIDense*)(A->data))->A,sviewer));
    }
    PetscCall(PetscViewerRestoreSubViewer(viewer,PETSC_COMM_SELF,&sviewer));
    PetscCall(PetscViewerFlush(viewer));
    PetscCall(MatDestroy(&A));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatView_MPIDense(Mat mat,PetscViewer viewer)
{
  PetscBool      iascii,isbinary,isdraw,issocket;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERBINARY,&isbinary));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERSOCKET,&issocket));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERDRAW,&isdraw));

  if (iascii || issocket || isdraw) {
    PetscCall(MatView_MPIDense_ASCIIorDraworSocket(mat,viewer));
  } else if (isbinary) PetscCall(MatView_Dense_Binary(mat,viewer));
  PetscFunctionReturn(0);
}

PetscErrorCode MatGetInfo_MPIDense(Mat A,MatInfoType flag,MatInfo *info)
{
  Mat_MPIDense   *mat = (Mat_MPIDense*)A->data;
  Mat            mdn  = mat->A;
  PetscLogDouble isend[5],irecv[5];

  PetscFunctionBegin;
  info->block_size = 1.0;

  PetscCall(MatGetInfo(mdn,MAT_LOCAL,info));

  isend[0] = info->nz_used; isend[1] = info->nz_allocated; isend[2] = info->nz_unneeded;
  isend[3] = info->memory;  isend[4] = info->mallocs;
  if (flag == MAT_LOCAL) {
    info->nz_used      = isend[0];
    info->nz_allocated = isend[1];
    info->nz_unneeded  = isend[2];
    info->memory       = isend[3];
    info->mallocs      = isend[4];
  } else if (flag == MAT_GLOBAL_MAX) {
    PetscCall(MPIU_Allreduce(isend,irecv,5,MPIU_PETSCLOGDOUBLE,MPI_MAX,PetscObjectComm((PetscObject)A)));

    info->nz_used      = irecv[0];
    info->nz_allocated = irecv[1];
    info->nz_unneeded  = irecv[2];
    info->memory       = irecv[3];
    info->mallocs      = irecv[4];
  } else if (flag == MAT_GLOBAL_SUM) {
    PetscCall(MPIU_Allreduce(isend,irecv,5,MPIU_PETSCLOGDOUBLE,MPI_SUM,PetscObjectComm((PetscObject)A)));

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

  PetscFunctionBegin;
  switch (op) {
  case MAT_NEW_NONZERO_LOCATIONS:
  case MAT_NEW_NONZERO_LOCATION_ERR:
  case MAT_NEW_NONZERO_ALLOCATION_ERR:
    MatCheckPreallocated(A,1);
    PetscCall(MatSetOption(a->A,op,flg));
    break;
  case MAT_ROW_ORIENTED:
    MatCheckPreallocated(A,1);
    a->roworiented = flg;
    PetscCall(MatSetOption(a->A,op,flg));
    break;
  case MAT_FORCE_DIAGONAL_ENTRIES:
  case MAT_KEEP_NONZERO_PATTERN:
  case MAT_USE_HASH_TABLE:
  case MAT_SORTED_FULL:
    PetscCall(PetscInfo(A,"Option %s ignored\n",MatOptions[op]));
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
    PetscCall(PetscInfo(A,"Option %s ignored\n",MatOptions[op]));
    break;
  default:
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"unknown option %s",MatOptions[op]);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatDiagonalScale_MPIDense(Mat A,Vec ll,Vec rr)
{
  Mat_MPIDense      *mdn = (Mat_MPIDense*)A->data;
  const PetscScalar *l;
  PetscScalar       x,*v,*vv,*r;
  PetscInt          i,j,s2a,s3a,s2,s3,m=mdn->A->rmap->n,n=mdn->A->cmap->n,lda;

  PetscFunctionBegin;
  PetscCall(MatDenseGetArray(mdn->A,&vv));
  PetscCall(MatDenseGetLDA(mdn->A,&lda));
  PetscCall(MatGetLocalSize(A,&s2,&s3));
  if (ll) {
    PetscCall(VecGetLocalSize(ll,&s2a));
    PetscCheck(s2a == s2,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Left scaling vector non-conforming local size, %" PetscInt_FMT " != %" PetscInt_FMT, s2a, s2);
    PetscCall(VecGetArrayRead(ll,&l));
    for (i=0; i<m; i++) {
      x = l[i];
      v = vv + i;
      for (j=0; j<n; j++) { (*v) *= x; v+= lda;}
    }
    PetscCall(VecRestoreArrayRead(ll,&l));
    PetscCall(PetscLogFlops(1.0*n*m));
  }
  if (rr) {
    const PetscScalar *ar;

    PetscCall(VecGetLocalSize(rr,&s3a));
    PetscCheck(s3a == s3,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Right scaling vec non-conforming local size, %" PetscInt_FMT " != %" PetscInt_FMT ".", s3a, s3);
    PetscCall(VecGetArrayRead(rr,&ar));
    if (!mdn->Mvctx) PetscCall(MatSetUpMultiply_MPIDense(A));
    PetscCall(VecGetArray(mdn->lvec,&r));
    PetscCall(PetscSFBcastBegin(mdn->Mvctx,MPIU_SCALAR,ar,r,MPI_REPLACE));
    PetscCall(PetscSFBcastEnd(mdn->Mvctx,MPIU_SCALAR,ar,r,MPI_REPLACE));
    PetscCall(VecRestoreArrayRead(rr,&ar));
    for (i=0; i<n; i++) {
      x = r[i];
      v = vv + i*lda;
      for (j=0; j<m; j++) (*v++) *= x;
    }
    PetscCall(VecRestoreArray(mdn->lvec,&r));
    PetscCall(PetscLogFlops(1.0*n*m));
  }
  PetscCall(MatDenseRestoreArray(mdn->A,&vv));
  PetscFunctionReturn(0);
}

PetscErrorCode MatNorm_MPIDense(Mat A,NormType type,PetscReal *nrm)
{
  Mat_MPIDense      *mdn = (Mat_MPIDense*)A->data;
  PetscInt          i,j;
  PetscMPIInt       size;
  PetscReal         sum = 0.0;
  const PetscScalar *av,*v;

  PetscFunctionBegin;
  PetscCall(MatDenseGetArrayRead(mdn->A,&av));
  v    = av;
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)A),&size));
  if (size == 1) {
    PetscCall(MatNorm(mdn->A,type,nrm));
  } else {
    if (type == NORM_FROBENIUS) {
      for (i=0; i<mdn->A->cmap->n*mdn->A->rmap->n; i++) {
        sum += PetscRealPart(PetscConj(*v)*(*v)); v++;
      }
      PetscCall(MPIU_Allreduce(&sum,nrm,1,MPIU_REAL,MPIU_SUM,PetscObjectComm((PetscObject)A)));
      *nrm = PetscSqrtReal(*nrm);
      PetscCall(PetscLogFlops(2.0*mdn->A->cmap->n*mdn->A->rmap->n));
    } else if (type == NORM_1) {
      PetscReal *tmp,*tmp2;
      PetscCall(PetscCalloc2(A->cmap->N,&tmp,A->cmap->N,&tmp2));
      *nrm = 0.0;
      v    = av;
      for (j=0; j<mdn->A->cmap->n; j++) {
        for (i=0; i<mdn->A->rmap->n; i++) {
          tmp[j] += PetscAbsScalar(*v);  v++;
        }
      }
      PetscCall(MPIU_Allreduce(tmp,tmp2,A->cmap->N,MPIU_REAL,MPIU_SUM,PetscObjectComm((PetscObject)A)));
      for (j=0; j<A->cmap->N; j++) {
        if (tmp2[j] > *nrm) *nrm = tmp2[j];
      }
      PetscCall(PetscFree2(tmp,tmp2));
      PetscCall(PetscLogFlops(A->cmap->n*A->rmap->n));
    } else if (type == NORM_INFINITY) { /* max row norm */
      PetscReal ntemp;
      PetscCall(MatNorm(mdn->A,type,&ntemp));
      PetscCall(MPIU_Allreduce(&ntemp,nrm,1,MPIU_REAL,MPIU_MAX,PetscObjectComm((PetscObject)A)));
    } else SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"No support for two norm");
  }
  PetscCall(MatDenseRestoreArrayRead(mdn->A,&av));
  PetscFunctionReturn(0);
}

PetscErrorCode MatTranspose_MPIDense(Mat A,MatReuse reuse,Mat *matout)
{
  Mat_MPIDense   *a    = (Mat_MPIDense*)A->data;
  Mat            B;
  PetscInt       M = A->rmap->N,N = A->cmap->N,m,n,*rwork,rstart = A->rmap->rstart;
  PetscInt       j,i,lda;
  PetscScalar    *v;

  PetscFunctionBegin;
  if (reuse == MAT_INITIAL_MATRIX || reuse == MAT_INPLACE_MATRIX) {
    PetscCall(MatCreate(PetscObjectComm((PetscObject)A),&B));
    PetscCall(MatSetSizes(B,A->cmap->n,A->rmap->n,N,M));
    PetscCall(MatSetType(B,((PetscObject)A)->type_name));
    PetscCall(MatMPIDenseSetPreallocation(B,NULL));
  } else B = *matout;

  m    = a->A->rmap->n; n = a->A->cmap->n;
  PetscCall(MatDenseGetArrayRead(a->A,(const PetscScalar**)&v));
  PetscCall(MatDenseGetLDA(a->A,&lda));
  PetscCall(PetscMalloc1(m,&rwork));
  for (i=0; i<m; i++) rwork[i] = rstart + i;
  for (j=0; j<n; j++) {
    PetscCall(MatSetValues(B,1,&j,m,rwork,v,INSERT_VALUES));
    v   += lda;
  }
  PetscCall(MatDenseRestoreArrayRead(a->A,(const PetscScalar**)&v));
  PetscCall(PetscFree(rwork));
  PetscCall(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));
  if (reuse == MAT_INITIAL_MATRIX || reuse == MAT_REUSE_MATRIX) {
    *matout = B;
  } else {
    PetscCall(MatHeaderMerge(A,&B));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDuplicate_MPIDense(Mat,MatDuplicateOption,Mat*);
PETSC_INTERN PetscErrorCode MatScale_MPIDense(Mat,PetscScalar);

PetscErrorCode MatSetUp_MPIDense(Mat A)
{
  PetscFunctionBegin;
  PetscCall(PetscLayoutSetUp(A->rmap));
  PetscCall(PetscLayoutSetUp(A->cmap));
  if (!A->preallocated) {
    PetscCall(MatMPIDenseSetPreallocation(A,NULL));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatAXPY_MPIDense(Mat Y,PetscScalar alpha,Mat X,MatStructure str)
{
  Mat_MPIDense   *A = (Mat_MPIDense*)Y->data, *B = (Mat_MPIDense*)X->data;

  PetscFunctionBegin;
  PetscCall(MatAXPY(A->A,alpha,B->A,str));
  PetscFunctionReturn(0);
}

PetscErrorCode MatConjugate_MPIDense(Mat mat)
{
  Mat_MPIDense   *a = (Mat_MPIDense*)mat->data;

  PetscFunctionBegin;
  PetscCall(MatConjugate(a->A));
  PetscFunctionReturn(0);
}

PetscErrorCode MatRealPart_MPIDense(Mat A)
{
  Mat_MPIDense   *a = (Mat_MPIDense*)A->data;

  PetscFunctionBegin;
  PetscCall(MatRealPart(a->A));
  PetscFunctionReturn(0);
}

PetscErrorCode MatImaginaryPart_MPIDense(Mat A)
{
  Mat_MPIDense   *a = (Mat_MPIDense*)A->data;

  PetscFunctionBegin;
  PetscCall(MatImaginaryPart(a->A));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatGetColumnVector_MPIDense(Mat A,Vec v,PetscInt col)
{
  Mat_MPIDense   *a = (Mat_MPIDense*) A->data;

  PetscFunctionBegin;
  PetscCheck(a->A,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Missing local matrix");
  PetscCheck(a->A->ops->getcolumnvector,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Missing get column operation");
  PetscCall((*a->A->ops->getcolumnvector)(a->A,v,col));
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatGetColumnReductions_SeqDense(Mat,PetscInt,PetscReal*);

PetscErrorCode MatGetColumnReductions_MPIDense(Mat A,PetscInt type,PetscReal *reductions)
{
  PetscInt       i,m,n;
  Mat_MPIDense   *a = (Mat_MPIDense*) A->data;
  PetscReal      *work;

  PetscFunctionBegin;
  PetscCall(MatGetSize(A,&m,&n));
  PetscCall(PetscMalloc1(n,&work));
  if (type == REDUCTION_MEAN_REALPART) {
    PetscCall(MatGetColumnReductions_SeqDense(a->A,(PetscInt)REDUCTION_SUM_REALPART,work));
  } else if (type == REDUCTION_MEAN_IMAGINARYPART) {
    PetscCall(MatGetColumnReductions_SeqDense(a->A,(PetscInt)REDUCTION_SUM_IMAGINARYPART,work));
  } else {
    PetscCall(MatGetColumnReductions_SeqDense(a->A,type,work));
  }
  if (type == NORM_2) {
    for (i=0; i<n; i++) work[i] *= work[i];
  }
  if (type == NORM_INFINITY) {
    PetscCall(MPIU_Allreduce(work,reductions,n,MPIU_REAL,MPIU_MAX,A->hdr.comm));
  } else {
    PetscCall(MPIU_Allreduce(work,reductions,n,MPIU_REAL,MPIU_SUM,A->hdr.comm));
  }
  PetscCall(PetscFree(work));
  if (type == NORM_2) {
    for (i=0; i<n; i++) reductions[i] = PetscSqrtReal(reductions[i]);
  } else if (type == REDUCTION_MEAN_REALPART || type == REDUCTION_MEAN_IMAGINARYPART) {
    for (i=0; i<n; i++) reductions[i] /= m;
  }
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_CUDA)
PetscErrorCode MatShift_MPIDenseCUDA(Mat A,PetscScalar alpha)
{
  PetscScalar *da;
  PetscInt    lda;

  PetscFunctionBegin;
  PetscCall(MatDenseCUDAGetArray(A,&da));
  PetscCall(MatDenseGetLDA(A,&lda));
  PetscCall(PetscInfo(A,"Performing Shift on backend\n"));
  PetscCall(MatShift_DenseCUDA_Private(da,alpha,lda,A->rmap->rstart,A->rmap->rend,A->cmap->N));
  PetscCall(MatDenseCUDARestoreArray(A,&da));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseGetColumnVec_MPIDenseCUDA(Mat A,PetscInt col,Vec *v)
{
  Mat_MPIDense   *a = (Mat_MPIDense*)A->data;
  PetscInt       lda;

  PetscFunctionBegin;
  PetscCheck(!a->vecinuse,PetscObjectComm((PetscObject)A),PETSC_ERR_ORDER,"Need to call MatDenseRestoreColumnVec() first");
  PetscCheck(!a->matinuse,PetscObjectComm((PetscObject)A),PETSC_ERR_ORDER,"Need to call MatDenseRestoreSubMatrix() first");
  if (!a->cvec) {
    PetscCall(VecCreateMPICUDAWithArray(PetscObjectComm((PetscObject)A),A->rmap->bs,A->rmap->n,A->rmap->N,NULL,&a->cvec));
    PetscCall(PetscLogObjectParent((PetscObject)A,(PetscObject)a->cvec));
  }
  a->vecinuse = col + 1;
  PetscCall(MatDenseGetLDA(a->A,&lda));
  PetscCall(MatDenseCUDAGetArray(a->A,(PetscScalar**)&a->ptrinuse));
  PetscCall(VecCUDAPlaceArray(a->cvec,a->ptrinuse + (size_t)col * (size_t)lda));
  *v   = a->cvec;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseRestoreColumnVec_MPIDenseCUDA(Mat A,PetscInt col,Vec *v)
{
  Mat_MPIDense   *a = (Mat_MPIDense*)A->data;

  PetscFunctionBegin;
  PetscCheck(a->vecinuse,PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseGetColumnVec() first");
  PetscCheck(a->cvec,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Missing internal column vector");
  a->vecinuse = 0;
  PetscCall(MatDenseCUDARestoreArray(a->A,(PetscScalar**)&a->ptrinuse));
  PetscCall(VecCUDAResetArray(a->cvec));
  *v   = NULL;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseGetColumnVecRead_MPIDenseCUDA(Mat A,PetscInt col,Vec *v)
{
  Mat_MPIDense   *a = (Mat_MPIDense*)A->data;
  PetscInt       lda;

  PetscFunctionBegin;
  PetscCheck(!a->vecinuse,PetscObjectComm((PetscObject)A),PETSC_ERR_ORDER,"Need to call MatDenseRestoreColumnVec() first");
  PetscCheck(!a->matinuse,PetscObjectComm((PetscObject)A),PETSC_ERR_ORDER,"Need to call MatDenseRestoreSubMatrix() first");
  if (!a->cvec) {
    PetscCall(VecCreateMPICUDAWithArray(PetscObjectComm((PetscObject)A),A->rmap->bs,A->rmap->n,A->rmap->N,NULL,&a->cvec));
    PetscCall(PetscLogObjectParent((PetscObject)A,(PetscObject)a->cvec));
  }
  a->vecinuse = col + 1;
  PetscCall(MatDenseGetLDA(a->A,&lda));
  PetscCall(MatDenseCUDAGetArrayRead(a->A,&a->ptrinuse));
  PetscCall(VecCUDAPlaceArray(a->cvec,a->ptrinuse + (size_t)col * (size_t)lda));
  PetscCall(VecLockReadPush(a->cvec));
  *v   = a->cvec;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseRestoreColumnVecRead_MPIDenseCUDA(Mat A,PetscInt col,Vec *v)
{
  Mat_MPIDense   *a = (Mat_MPIDense*)A->data;

  PetscFunctionBegin;
  PetscCheck(a->vecinuse,PetscObjectComm((PetscObject)A),PETSC_ERR_ORDER,"Need to call MatDenseGetColumnVec() first");
  PetscCheck(a->cvec,PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Missing internal column vector");
  a->vecinuse = 0;
  PetscCall(MatDenseCUDARestoreArrayRead(a->A,&a->ptrinuse));
  PetscCall(VecLockReadPop(a->cvec));
  PetscCall(VecCUDAResetArray(a->cvec));
  *v   = NULL;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseGetColumnVecWrite_MPIDenseCUDA(Mat A,PetscInt col,Vec *v)
{
  Mat_MPIDense   *a = (Mat_MPIDense*)A->data;
  PetscInt       lda;

  PetscFunctionBegin;
  PetscCheck(!a->vecinuse,PetscObjectComm((PetscObject)A),PETSC_ERR_ORDER,"Need to call MatDenseRestoreColumnVec() first");
  PetscCheck(!a->matinuse,PetscObjectComm((PetscObject)A),PETSC_ERR_ORDER,"Need to call MatDenseRestoreSubMatrix() first");
  if (!a->cvec) {
    PetscCall(VecCreateMPICUDAWithArray(PetscObjectComm((PetscObject)A),A->rmap->bs,A->rmap->n,A->rmap->N,NULL,&a->cvec));
    PetscCall(PetscLogObjectParent((PetscObject)A,(PetscObject)a->cvec));
  }
  a->vecinuse = col + 1;
  PetscCall(MatDenseGetLDA(a->A,&lda));
  PetscCall(MatDenseCUDAGetArrayWrite(a->A,(PetscScalar**)&a->ptrinuse));
  PetscCall(VecCUDAPlaceArray(a->cvec,a->ptrinuse + (size_t)col * (size_t)lda));
  *v   = a->cvec;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseRestoreColumnVecWrite_MPIDenseCUDA(Mat A,PetscInt col,Vec *v)
{
  Mat_MPIDense   *a = (Mat_MPIDense*)A->data;

  PetscFunctionBegin;
  PetscCheck(a->vecinuse,PetscObjectComm((PetscObject)A),PETSC_ERR_ORDER,"Need to call MatDenseGetColumnVec() first");
  PetscCheck(a->cvec,PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Missing internal column vector");
  a->vecinuse = 0;
  PetscCall(MatDenseCUDARestoreArrayWrite(a->A,(PetscScalar**)&a->ptrinuse));
  PetscCall(VecCUDAResetArray(a->cvec));
  *v   = NULL;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseCUDAPlaceArray_MPIDenseCUDA(Mat A, const PetscScalar *a)
{
  Mat_MPIDense   *l = (Mat_MPIDense*) A->data;

  PetscFunctionBegin;
  PetscCheck(!l->vecinuse,PetscObjectComm((PetscObject)A),PETSC_ERR_ORDER,"Need to call MatDenseRestoreColumnVec() first");
  PetscCheck(!l->matinuse,PetscObjectComm((PetscObject)A),PETSC_ERR_ORDER,"Need to call MatDenseRestoreSubMatrix() first");
  PetscCall(MatDenseCUDAPlaceArray(l->A,a));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseCUDAResetArray_MPIDenseCUDA(Mat A)
{
  Mat_MPIDense   *l = (Mat_MPIDense*) A->data;

  PetscFunctionBegin;
  PetscCheck(!l->vecinuse,PetscObjectComm((PetscObject)A),PETSC_ERR_ORDER,"Need to call MatDenseRestoreColumnVec() first");
  PetscCheck(!l->matinuse,PetscObjectComm((PetscObject)A),PETSC_ERR_ORDER,"Need to call MatDenseRestoreSubMatrix() first");
  PetscCall(MatDenseCUDAResetArray(l->A));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseCUDAReplaceArray_MPIDenseCUDA(Mat A, const PetscScalar *a)
{
  Mat_MPIDense   *l = (Mat_MPIDense*) A->data;

  PetscFunctionBegin;
  PetscCheck(!l->vecinuse,PetscObjectComm((PetscObject)A),PETSC_ERR_ORDER,"Need to call MatDenseRestoreColumnVec() first");
  PetscCheck(!l->matinuse,PetscObjectComm((PetscObject)A),PETSC_ERR_ORDER,"Need to call MatDenseRestoreSubMatrix() first");
  PetscCall(MatDenseCUDAReplaceArray(l->A,a));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseCUDAGetArrayWrite_MPIDenseCUDA(Mat A, PetscScalar **a)
{
  Mat_MPIDense   *l = (Mat_MPIDense*) A->data;

  PetscFunctionBegin;
  PetscCall(MatDenseCUDAGetArrayWrite(l->A,a));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseCUDARestoreArrayWrite_MPIDenseCUDA(Mat A, PetscScalar **a)
{
  Mat_MPIDense   *l = (Mat_MPIDense*) A->data;

  PetscFunctionBegin;
  PetscCall(MatDenseCUDARestoreArrayWrite(l->A,a));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseCUDAGetArrayRead_MPIDenseCUDA(Mat A, const PetscScalar **a)
{
  Mat_MPIDense   *l = (Mat_MPIDense*) A->data;

  PetscFunctionBegin;
  PetscCall(MatDenseCUDAGetArrayRead(l->A,a));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseCUDARestoreArrayRead_MPIDenseCUDA(Mat A, const PetscScalar **a)
{
  Mat_MPIDense   *l = (Mat_MPIDense*) A->data;

  PetscFunctionBegin;
  PetscCall(MatDenseCUDARestoreArrayRead(l->A,a));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseCUDAGetArray_MPIDenseCUDA(Mat A, PetscScalar **a)
{
  Mat_MPIDense   *l = (Mat_MPIDense*) A->data;

  PetscFunctionBegin;
  PetscCall(MatDenseCUDAGetArray(l->A,a));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseCUDARestoreArray_MPIDenseCUDA(Mat A, PetscScalar **a)
{
  Mat_MPIDense   *l = (Mat_MPIDense*) A->data;

  PetscFunctionBegin;
  PetscCall(MatDenseCUDARestoreArray(l->A,a));
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

  PetscFunctionBegin;
  PetscCheck(!d->vecinuse,PetscObjectComm((PetscObject)mat),PETSC_ERR_ORDER,"Need to call MatDenseRestoreColumnVec() first");
  PetscCheck(!d->matinuse,PetscObjectComm((PetscObject)mat),PETSC_ERR_ORDER,"Need to call MatDenseRestoreSubMatrix() first");
  if (d->A) PetscCall(MatBindToCPU(d->A,bind));
  mat->boundtocpu = bind;
  if (!bind) {
    PetscBool iscuda;

    PetscCall(PetscObjectTypeCompare((PetscObject)d->cvec,VECMPICUDA,&iscuda));
    if (!iscuda) {
      PetscCall(VecDestroy(&d->cvec));
    }
    PetscCall(PetscObjectTypeCompare((PetscObject)d->cmat,MATMPIDENSECUDA,&iscuda));
    if (!iscuda) {
      PetscCall(MatDestroy(&d->cmat));
    }
    PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatDenseGetColumnVec_C",MatDenseGetColumnVec_MPIDenseCUDA));
    PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatDenseRestoreColumnVec_C",MatDenseRestoreColumnVec_MPIDenseCUDA));
    PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatDenseGetColumnVecRead_C",MatDenseGetColumnVecRead_MPIDenseCUDA));
    PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatDenseRestoreColumnVecRead_C",MatDenseRestoreColumnVecRead_MPIDenseCUDA));
    PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatDenseGetColumnVecWrite_C",MatDenseGetColumnVecWrite_MPIDenseCUDA));
    PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatDenseRestoreColumnVecWrite_C",MatDenseRestoreColumnVecWrite_MPIDenseCUDA));
    mat->ops->shift                   = MatShift_MPIDenseCUDA;
  } else {
    PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatDenseGetColumnVec_C",MatDenseGetColumnVec_MPIDense));
    PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatDenseRestoreColumnVec_C",MatDenseRestoreColumnVec_MPIDense));
    PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatDenseGetColumnVecRead_C",MatDenseGetColumnVecRead_MPIDense));
    PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatDenseRestoreColumnVecRead_C",MatDenseRestoreColumnVecRead_MPIDense));
    PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatDenseGetColumnVecWrite_C",MatDenseGetColumnVecWrite_MPIDense));
    PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatDenseRestoreColumnVecWrite_C",MatDenseRestoreColumnVecWrite_MPIDense));
    mat->ops->shift                   = MatShift_MPIDense;
  }
  if (d->cmat) PetscCall(MatBindToCPU(d->cmat,bind));
  PetscFunctionReturn(0);
}

PetscErrorCode MatMPIDenseCUDASetPreallocation(Mat A, PetscScalar *d_data)
{
  Mat_MPIDense   *d = (Mat_MPIDense*)A->data;
  PetscBool      iscuda;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscCall(PetscObjectTypeCompare((PetscObject)A,MATMPIDENSECUDA,&iscuda));
  if (!iscuda) PetscFunctionReturn(0);
  PetscCall(PetscLayoutSetUp(A->rmap));
  PetscCall(PetscLayoutSetUp(A->cmap));
  if (!d->A) {
    PetscCall(MatCreate(PETSC_COMM_SELF,&d->A));
    PetscCall(PetscLogObjectParent((PetscObject)A,(PetscObject)d->A));
    PetscCall(MatSetSizes(d->A,A->rmap->n,A->cmap->N,A->rmap->n,A->cmap->N));
  }
  PetscCall(MatSetType(d->A,MATSEQDENSECUDA));
  PetscCall(MatSeqDenseCUDASetPreallocation(d->A,d_data));
  A->preallocated = PETSC_TRUE;
  A->assembled = PETSC_TRUE;
  PetscFunctionReturn(0);
}
#endif

static PetscErrorCode MatSetRandom_MPIDense(Mat x,PetscRandom rctx)
{
  Mat_MPIDense   *d = (Mat_MPIDense*)x->data;

  PetscFunctionBegin;
  PetscCall(MatSetRandom(d->A,rctx));
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
static PetscErrorCode MatTransposeMatMultSymbolic_MPIDense_MPIDense(Mat,Mat,PetscReal,Mat);
static PetscErrorCode MatTransposeMatMultNumeric_MPIDense_MPIDense(Mat,Mat,Mat);
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
                                        MatShift_MPIDense,
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
                                        MatGetColumnReductions_MPIDense,
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
                                        NULL,
                                        NULL,
                                        NULL
};

PetscErrorCode  MatMPIDenseSetPreallocation_MPIDense(Mat mat,PetscScalar *data)
{
  Mat_MPIDense   *a = (Mat_MPIDense*)mat->data;
  PetscBool      iscuda;

  PetscFunctionBegin;
  PetscCheck(!a->matinuse,PetscObjectComm((PetscObject)mat),PETSC_ERR_ORDER,"Need to call MatDenseRestoreSubMatrix() first");
  PetscCall(PetscLayoutSetUp(mat->rmap));
  PetscCall(PetscLayoutSetUp(mat->cmap));
  if (!a->A) {
    PetscCall(MatCreate(PETSC_COMM_SELF,&a->A));
    PetscCall(PetscLogObjectParent((PetscObject)mat,(PetscObject)a->A));
    PetscCall(MatSetSizes(a->A,mat->rmap->n,mat->cmap->N,mat->rmap->n,mat->cmap->N));
  }
  PetscCall(PetscObjectTypeCompare((PetscObject)mat,MATMPIDENSECUDA,&iscuda));
  PetscCall(MatSetType(a->A,iscuda ? MATSEQDENSECUDA : MATSEQDENSE));
  PetscCall(MatSeqDenseSetPreallocation(a->A,data));
  mat->preallocated = PETSC_TRUE;
  mat->assembled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatConvert_MPIAIJ_MPIDense(Mat A, MatType newtype,MatReuse reuse,Mat *newmat)
{
  Mat            B,C;

  PetscFunctionBegin;
  PetscCall(MatMPIAIJGetLocalMat(A,MAT_INITIAL_MATRIX,&C));
  PetscCall(MatConvert_SeqAIJ_SeqDense(C,MATSEQDENSE,MAT_INITIAL_MATRIX,&B));
  PetscCall(MatDestroy(&C));
  if (reuse == MAT_REUSE_MATRIX) {
    C = *newmat;
  } else C = NULL;
  PetscCall(MatCreateMPIMatConcatenateSeqMat(PetscObjectComm((PetscObject)A),B,A->cmap->n,!C?MAT_INITIAL_MATRIX:MAT_REUSE_MATRIX,&C));
  PetscCall(MatDestroy(&B));
  if (reuse == MAT_INPLACE_MATRIX) {
    PetscCall(MatHeaderReplace(A,&C));
  } else if (reuse == MAT_INITIAL_MATRIX) *newmat = C;
  PetscFunctionReturn(0);
}

PetscErrorCode MatConvert_MPIDense_MPIAIJ(Mat A, MatType newtype,MatReuse reuse,Mat *newmat)
{
  Mat            B,C;

  PetscFunctionBegin;
  PetscCall(MatDenseGetLocalMatrix(A,&C));
  PetscCall(MatConvert_SeqDense_SeqAIJ(C,MATSEQAIJ,MAT_INITIAL_MATRIX,&B));
  if (reuse == MAT_REUSE_MATRIX) {
    C = *newmat;
  } else C = NULL;
  PetscCall(MatCreateMPIMatConcatenateSeqMat(PetscObjectComm((PetscObject)A),B,A->cmap->n,!C?MAT_INITIAL_MATRIX:MAT_REUSE_MATRIX,&C));
  PetscCall(MatDestroy(&B));
  if (reuse == MAT_INPLACE_MATRIX) {
    PetscCall(MatHeaderReplace(A,&C));
  } else if (reuse == MAT_INITIAL_MATRIX) *newmat = C;
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_ELEMENTAL)
PETSC_INTERN PetscErrorCode MatConvert_MPIDense_Elemental(Mat A, MatType newtype,MatReuse reuse,Mat *newmat)
{
  Mat            mat_elemental;
  PetscScalar    *v;
  PetscInt       m=A->rmap->n,N=A->cmap->N,rstart=A->rmap->rstart,i,*rows,*cols;

  PetscFunctionBegin;
  if (reuse == MAT_REUSE_MATRIX) {
    mat_elemental = *newmat;
    PetscCall(MatZeroEntries(*newmat));
  } else {
    PetscCall(MatCreate(PetscObjectComm((PetscObject)A), &mat_elemental));
    PetscCall(MatSetSizes(mat_elemental,PETSC_DECIDE,PETSC_DECIDE,A->rmap->N,A->cmap->N));
    PetscCall(MatSetType(mat_elemental,MATELEMENTAL));
    PetscCall(MatSetUp(mat_elemental));
    PetscCall(MatSetOption(mat_elemental,MAT_ROW_ORIENTED,PETSC_FALSE));
  }

  PetscCall(PetscMalloc2(m,&rows,N,&cols));
  for (i=0; i<N; i++) cols[i] = i;
  for (i=0; i<m; i++) rows[i] = rstart + i;

  /* PETSc-Elemental interface uses axpy for setting off-processor entries, only ADD_VALUES is allowed */
  PetscCall(MatDenseGetArray(A,&v));
  PetscCall(MatSetValues(mat_elemental,m,rows,N,cols,v,ADD_VALUES));
  PetscCall(MatAssemblyBegin(mat_elemental, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(mat_elemental, MAT_FINAL_ASSEMBLY));
  PetscCall(MatDenseRestoreArray(A,&v));
  PetscCall(PetscFree2(rows,cols));

  if (reuse == MAT_INPLACE_MATRIX) {
    PetscCall(MatHeaderReplace(A,&mat_elemental));
  } else {
    *newmat = mat_elemental;
  }
  PetscFunctionReturn(0);
}
#endif

static PetscErrorCode MatDenseGetColumn_MPIDense(Mat A,PetscInt col,PetscScalar **vals)
{
  Mat_MPIDense   *mat = (Mat_MPIDense*)A->data;

  PetscFunctionBegin;
  PetscCall(MatDenseGetColumn(mat->A,col,vals));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseRestoreColumn_MPIDense(Mat A,PetscScalar **vals)
{
  Mat_MPIDense   *mat = (Mat_MPIDense*)A->data;

  PetscFunctionBegin;
  PetscCall(MatDenseRestoreColumn(mat->A,vals));
  PetscFunctionReturn(0);
}

PetscErrorCode MatCreateMPIMatConcatenateSeqMat_MPIDense(MPI_Comm comm,Mat inmat,PetscInt n,MatReuse scall,Mat *outmat)
{
  Mat_MPIDense   *mat;
  PetscInt       m,nloc,N;

  PetscFunctionBegin;
  PetscCall(MatGetSize(inmat,&m,&N));
  PetscCall(MatGetLocalSize(inmat,NULL,&nloc));
  if (scall == MAT_INITIAL_MATRIX) { /* symbolic phase */
    PetscInt sum;

    if (n == PETSC_DECIDE) {
      PetscCall(PetscSplitOwnership(comm,&n,&N));
    }
    /* Check sum(n) = N */
    PetscCall(MPIU_Allreduce(&n,&sum,1,MPIU_INT,MPI_SUM,comm));
    PetscCheck(sum == N,PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Sum of local columns %" PetscInt_FMT " != global columns %" PetscInt_FMT,sum,N);

    PetscCall(MatCreateDense(comm,m,n,PETSC_DETERMINE,N,NULL,outmat));
    PetscCall(MatSetOption(*outmat,MAT_NO_OFF_PROC_ENTRIES,PETSC_TRUE));
  }

  /* numeric phase */
  mat = (Mat_MPIDense*)(*outmat)->data;
  PetscCall(MatCopy(inmat,mat->A,SAME_NONZERO_PATTERN));
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_CUDA)
PetscErrorCode MatConvert_MPIDenseCUDA_MPIDense(Mat M,MatType type,MatReuse reuse,Mat *newmat)
{
  Mat            B;
  Mat_MPIDense   *m;

  PetscFunctionBegin;
  if (reuse == MAT_INITIAL_MATRIX) {
    PetscCall(MatDuplicate(M,MAT_COPY_VALUES,newmat));
  } else if (reuse == MAT_REUSE_MATRIX) {
    PetscCall(MatCopy(M,*newmat,SAME_NONZERO_PATTERN));
  }

  B    = *newmat;
  PetscCall(MatBindToCPU_MPIDenseCUDA(B,PETSC_TRUE));
  PetscCall(PetscFree(B->defaultvectype));
  PetscCall(PetscStrallocpy(VECSTANDARD,&B->defaultvectype));
  PetscCall(PetscObjectChangeTypeName((PetscObject)B,MATMPIDENSE));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatConvert_mpidensecuda_mpidense_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatProductSetFromOptions_mpiaij_mpidensecuda_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatProductSetFromOptions_mpiaijcusparse_mpidensecuda_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatProductSetFromOptions_mpidensecuda_mpiaij_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatProductSetFromOptions_mpidensecuda_mpiaijcusparse_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatDenseCUDAGetArray_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatDenseCUDAGetArrayRead_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatDenseCUDAGetArrayWrite_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatDenseCUDARestoreArray_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatDenseCUDARestoreArrayRead_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatDenseCUDARestoreArrayWrite_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatDenseCUDAPlaceArray_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatDenseCUDAResetArray_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatDenseCUDAReplaceArray_C",NULL));
  m    = (Mat_MPIDense*)(B)->data;
  if (m->A) {
    PetscCall(MatConvert(m->A,MATSEQDENSE,MAT_INPLACE_MATRIX,&m->A));
  }
  B->ops->bindtocpu = NULL;
  B->offloadmask    = PETSC_OFFLOAD_CPU;
  PetscFunctionReturn(0);
}

PetscErrorCode MatConvert_MPIDense_MPIDenseCUDA(Mat M,MatType type,MatReuse reuse,Mat *newmat)
{
  Mat            B;
  Mat_MPIDense   *m;

  PetscFunctionBegin;
  if (reuse == MAT_INITIAL_MATRIX) {
    PetscCall(MatDuplicate(M,MAT_COPY_VALUES,newmat));
  } else if (reuse == MAT_REUSE_MATRIX) {
    PetscCall(MatCopy(M,*newmat,SAME_NONZERO_PATTERN));
  }

  B    = *newmat;
  PetscCall(PetscFree(B->defaultvectype));
  PetscCall(PetscStrallocpy(VECCUDA,&B->defaultvectype));
  PetscCall(PetscObjectChangeTypeName((PetscObject)B,MATMPIDENSECUDA));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatConvert_mpidensecuda_mpidense_C",                    MatConvert_MPIDenseCUDA_MPIDense));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatProductSetFromOptions_mpiaij_mpidensecuda_C",        MatProductSetFromOptions_MPIAIJ_MPIDense));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatProductSetFromOptions_mpiaijcusparse_mpidensecuda_C",MatProductSetFromOptions_MPIAIJ_MPIDense));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatProductSetFromOptions_mpidensecuda_mpiaij_C",        MatProductSetFromOptions_MPIDense_MPIAIJ));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatProductSetFromOptions_mpidensecuda_mpiaijcusparse_C",MatProductSetFromOptions_MPIDense_MPIAIJ));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatDenseCUDAGetArray_C",                                MatDenseCUDAGetArray_MPIDenseCUDA));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatDenseCUDAGetArrayRead_C",                            MatDenseCUDAGetArrayRead_MPIDenseCUDA));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatDenseCUDAGetArrayWrite_C",                           MatDenseCUDAGetArrayWrite_MPIDenseCUDA));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatDenseCUDARestoreArray_C",                            MatDenseCUDARestoreArray_MPIDenseCUDA));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatDenseCUDARestoreArrayRead_C",                        MatDenseCUDARestoreArrayRead_MPIDenseCUDA));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatDenseCUDARestoreArrayWrite_C",                       MatDenseCUDARestoreArrayWrite_MPIDenseCUDA));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatDenseCUDAPlaceArray_C",                              MatDenseCUDAPlaceArray_MPIDenseCUDA));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatDenseCUDAResetArray_C",                              MatDenseCUDAResetArray_MPIDenseCUDA));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatDenseCUDAReplaceArray_C",                            MatDenseCUDAReplaceArray_MPIDenseCUDA));
  m    = (Mat_MPIDense*)(B->data);
  if (m->A) {
    PetscCall(MatConvert(m->A,MATSEQDENSECUDA,MAT_INPLACE_MATRIX,&m->A));
    B->offloadmask = PETSC_OFFLOAD_BOTH;
  } else {
    B->offloadmask = PETSC_OFFLOAD_UNALLOCATED;
  }
  PetscCall(MatBindToCPU_MPIDenseCUDA(B,PETSC_FALSE));

  B->ops->bindtocpu = MatBindToCPU_MPIDenseCUDA;
  PetscFunctionReturn(0);
}
#endif

PetscErrorCode MatDenseGetColumnVec_MPIDense(Mat A,PetscInt col,Vec *v)
{
  Mat_MPIDense   *a = (Mat_MPIDense*)A->data;
  PetscInt       lda;

  PetscFunctionBegin;
  PetscCheck(!a->vecinuse,PetscObjectComm((PetscObject)A),PETSC_ERR_ORDER,"Need to call MatDenseRestoreColumnVec() first");
  PetscCheck(!a->matinuse,PetscObjectComm((PetscObject)A),PETSC_ERR_ORDER,"Need to call MatDenseRestoreSubMatrix() first");
  if (!a->cvec) {
    PetscCall(VecCreateMPIWithArray(PetscObjectComm((PetscObject)A),A->rmap->bs,A->rmap->n,A->rmap->N,NULL,&a->cvec));
  }
  a->vecinuse = col + 1;
  PetscCall(MatDenseGetLDA(a->A,&lda));
  PetscCall(MatDenseGetArray(a->A,(PetscScalar**)&a->ptrinuse));
  PetscCall(VecPlaceArray(a->cvec,a->ptrinuse + (size_t)col * (size_t)lda));
  *v   = a->cvec;
  PetscFunctionReturn(0);
}

PetscErrorCode MatDenseRestoreColumnVec_MPIDense(Mat A,PetscInt col,Vec *v)
{
  Mat_MPIDense   *a = (Mat_MPIDense*)A->data;

  PetscFunctionBegin;
  PetscCheck(a->vecinuse,PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseGetColumnVec() first");
  PetscCheck(a->cvec,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Missing internal column vector");
  a->vecinuse = 0;
  PetscCall(MatDenseRestoreArray(a->A,(PetscScalar**)&a->ptrinuse));
  PetscCall(VecResetArray(a->cvec));
  *v   = NULL;
  PetscFunctionReturn(0);
}

PetscErrorCode MatDenseGetColumnVecRead_MPIDense(Mat A,PetscInt col,Vec *v)
{
  Mat_MPIDense   *a = (Mat_MPIDense*)A->data;
  PetscInt       lda;

  PetscFunctionBegin;
  PetscCheck(!a->vecinuse,PetscObjectComm((PetscObject)A),PETSC_ERR_ORDER,"Need to call MatDenseRestoreColumnVec() first");
  PetscCheck(!a->matinuse,PetscObjectComm((PetscObject)A),PETSC_ERR_ORDER,"Need to call MatDenseRestoreSubMatrix() first");
  if (!a->cvec) {
    PetscCall(VecCreateMPIWithArray(PetscObjectComm((PetscObject)A),A->rmap->bs,A->rmap->n,A->rmap->N,NULL,&a->cvec));
  }
  a->vecinuse = col + 1;
  PetscCall(MatDenseGetLDA(a->A,&lda));
  PetscCall(MatDenseGetArrayRead(a->A,&a->ptrinuse));
  PetscCall(VecPlaceArray(a->cvec,a->ptrinuse + (size_t)col * (size_t)lda));
  PetscCall(VecLockReadPush(a->cvec));
  *v   = a->cvec;
  PetscFunctionReturn(0);
}

PetscErrorCode MatDenseRestoreColumnVecRead_MPIDense(Mat A,PetscInt col,Vec *v)
{
  Mat_MPIDense   *a = (Mat_MPIDense*)A->data;

  PetscFunctionBegin;
  PetscCheck(a->vecinuse,PetscObjectComm((PetscObject)A),PETSC_ERR_ORDER,"Need to call MatDenseGetColumnVec() first");
  PetscCheck(a->cvec,PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Missing internal column vector");
  a->vecinuse = 0;
  PetscCall(MatDenseRestoreArrayRead(a->A,&a->ptrinuse));
  PetscCall(VecLockReadPop(a->cvec));
  PetscCall(VecResetArray(a->cvec));
  *v   = NULL;
  PetscFunctionReturn(0);
}

PetscErrorCode MatDenseGetColumnVecWrite_MPIDense(Mat A,PetscInt col,Vec *v)
{
  Mat_MPIDense   *a = (Mat_MPIDense*)A->data;
  PetscInt       lda;

  PetscFunctionBegin;
  PetscCheck(!a->vecinuse,PetscObjectComm((PetscObject)A),PETSC_ERR_ORDER,"Need to call MatDenseRestoreColumnVec() first");
  PetscCheck(!a->matinuse,PetscObjectComm((PetscObject)A),PETSC_ERR_ORDER,"Need to call MatDenseRestoreSubMatrix() first");
  if (!a->cvec) {
    PetscCall(VecCreateMPIWithArray(PetscObjectComm((PetscObject)A),A->rmap->bs,A->rmap->n,A->rmap->N,NULL,&a->cvec));
  }
  a->vecinuse = col + 1;
  PetscCall(MatDenseGetLDA(a->A,&lda));
  PetscCall(MatDenseGetArrayWrite(a->A,(PetscScalar**)&a->ptrinuse));
  PetscCall(VecPlaceArray(a->cvec,a->ptrinuse + (size_t)col * (size_t)lda));
  *v   = a->cvec;
  PetscFunctionReturn(0);
}

PetscErrorCode MatDenseRestoreColumnVecWrite_MPIDense(Mat A,PetscInt col,Vec *v)
{
  Mat_MPIDense *a = (Mat_MPIDense*)A->data;

  PetscFunctionBegin;
  PetscCheck(a->vecinuse,PetscObjectComm((PetscObject)A),PETSC_ERR_ORDER,"Need to call MatDenseGetColumnVec() first");
  PetscCheck(a->cvec,PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Missing internal column vector");
  a->vecinuse = 0;
  PetscCall(MatDenseRestoreArrayWrite(a->A,(PetscScalar**)&a->ptrinuse));
  PetscCall(VecResetArray(a->cvec));
  *v   = NULL;
  PetscFunctionReturn(0);
}

PetscErrorCode MatDenseGetSubMatrix_MPIDense(Mat A,PetscInt rbegin,PetscInt rend,PetscInt cbegin,PetscInt cend,Mat *v)
{
  Mat_MPIDense *a = (Mat_MPIDense*)A->data;
  Mat_MPIDense *c;
  MPI_Comm     comm;
  PetscInt     pbegin, pend;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)A,&comm));
  PetscCheck(!a->vecinuse,comm,PETSC_ERR_ORDER,"Need to call MatDenseRestoreColumnVec() first");
  PetscCheck(!a->matinuse,comm,PETSC_ERR_ORDER,"Need to call MatDenseRestoreSubMatrix() first");
  pbegin = PetscMax(0,PetscMin(A->rmap->rend,rbegin)-A->rmap->rstart);
  pend = PetscMin(A->rmap->n,PetscMax(0,rend-A->rmap->rstart));
  if (!a->cmat) {
    PetscCall(MatCreate(comm,&a->cmat));
    PetscCall(PetscLogObjectParent((PetscObject)A,(PetscObject)a->cmat));
    PetscCall(MatSetType(a->cmat,((PetscObject)A)->type_name));
    if (rend-rbegin==A->rmap->N) PetscCall(PetscLayoutReference(A->rmap,&a->cmat->rmap));
    else {
      PetscCall(PetscLayoutSetLocalSize(a->cmat->rmap,pend-pbegin));
      PetscCall(PetscLayoutSetSize(a->cmat->rmap,rend-rbegin));
      PetscCall(PetscLayoutSetUp(a->cmat->rmap));
    }
    PetscCall(PetscLayoutSetSize(a->cmat->cmap,cend-cbegin));
    PetscCall(PetscLayoutSetUp(a->cmat->cmap));
  } else {
    PetscBool same = (PetscBool)(rend-rbegin == a->cmat->rmap->N);
    if (same && a->cmat->rmap->N != A->rmap->N) {
      same = (PetscBool)(pend-pbegin == a->cmat->rmap->n);
      PetscCall(MPIU_Allreduce(MPI_IN_PLACE,&same,1,MPIU_BOOL,MPI_LAND,PetscObjectComm((PetscObject)A)));
    }
    if (!same) {
      PetscCall(PetscLayoutDestroy(&a->cmat->rmap));
      PetscCall(PetscLayoutCreate(comm,&a->cmat->rmap));
      PetscCall(PetscLayoutSetLocalSize(a->cmat->rmap,pend-pbegin));
      PetscCall(PetscLayoutSetSize(a->cmat->rmap,rend-rbegin));
      PetscCall(PetscLayoutSetUp(a->cmat->rmap));
    }
    if (cend-cbegin != a->cmat->cmap->N) {
      PetscCall(PetscLayoutDestroy(&a->cmat->cmap));
      PetscCall(PetscLayoutCreate(comm,&a->cmat->cmap));
      PetscCall(PetscLayoutSetSize(a->cmat->cmap,cend-cbegin));
      PetscCall(PetscLayoutSetUp(a->cmat->cmap));
    }
  }
  c = (Mat_MPIDense*)a->cmat->data;
  PetscCheck(!c->A,comm,PETSC_ERR_ORDER,"Need to call MatDenseRestoreSubMatrix() first");
  PetscCall(MatDenseGetSubMatrix(a->A,pbegin,pend,cbegin,cend,&c->A));
  a->cmat->preallocated = PETSC_TRUE;
  a->cmat->assembled = PETSC_TRUE;
  a->matinuse = cbegin + 1;
  *v = a->cmat;
  PetscFunctionReturn(0);
}

PetscErrorCode MatDenseRestoreSubMatrix_MPIDense(Mat A,Mat *v)
{
  Mat_MPIDense *a = (Mat_MPIDense*)A->data;
  Mat_MPIDense *c;

  PetscFunctionBegin;
  PetscCheck(a->matinuse,PetscObjectComm((PetscObject)A),PETSC_ERR_ORDER,"Need to call MatDenseGetSubMatrix() first");
  PetscCheck(a->cmat,PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Missing internal matrix");
  PetscCheck(*v == a->cmat,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Not the matrix obtained from MatDenseGetSubMatrix()");
  a->matinuse = 0;
  c = (Mat_MPIDense*)a->cmat->data;
  PetscCall(MatDenseRestoreSubMatrix(a->A,&c->A));
  *v = NULL;
  PetscFunctionReturn(0);
}

/*MC
   MATMPIDENSE - MATMPIDENSE = "mpidense" - A matrix type to be used for distributed dense matrices.

   Options Database Keys:
. -mat_type mpidense - sets the matrix type to "mpidense" during a call to MatSetFromOptions()

  Level: beginner

.seealso: `MatCreateDense()`

M*/
PETSC_EXTERN PetscErrorCode MatCreate_MPIDense(Mat mat)
{
  Mat_MPIDense   *a;

  PetscFunctionBegin;
  PetscCall(PetscNewLog(mat,&a));
  mat->data = (void*)a;
  PetscCall(PetscMemcpy(mat->ops,&MatOps_Values,sizeof(struct _MatOps)));

  mat->insertmode = NOT_SET_VALUES;

  /* build cache for off array entries formed */
  a->donotstash = PETSC_FALSE;

  PetscCall(MatStashCreate_Private(PetscObjectComm((PetscObject)mat),1,&mat->stash));

  /* stuff used for matrix vector multiply */
  a->lvec        = NULL;
  a->Mvctx       = NULL;
  a->roworiented = PETSC_TRUE;

  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatDenseGetLDA_C",MatDenseGetLDA_MPIDense));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatDenseSetLDA_C",MatDenseSetLDA_MPIDense));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatDenseGetArray_C",MatDenseGetArray_MPIDense));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatDenseRestoreArray_C",MatDenseRestoreArray_MPIDense));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatDenseGetArrayRead_C",MatDenseGetArrayRead_MPIDense));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatDenseRestoreArrayRead_C",MatDenseRestoreArrayRead_MPIDense));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatDenseGetArrayWrite_C",MatDenseGetArrayWrite_MPIDense));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatDenseRestoreArrayWrite_C",MatDenseRestoreArrayWrite_MPIDense));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatDensePlaceArray_C",MatDensePlaceArray_MPIDense));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatDenseResetArray_C",MatDenseResetArray_MPIDense));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatDenseReplaceArray_C",MatDenseReplaceArray_MPIDense));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatDenseGetColumnVec_C",MatDenseGetColumnVec_MPIDense));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatDenseRestoreColumnVec_C",MatDenseRestoreColumnVec_MPIDense));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatDenseGetColumnVecRead_C",MatDenseGetColumnVecRead_MPIDense));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatDenseRestoreColumnVecRead_C",MatDenseRestoreColumnVecRead_MPIDense));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatDenseGetColumnVecWrite_C",MatDenseGetColumnVecWrite_MPIDense));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatDenseRestoreColumnVecWrite_C",MatDenseRestoreColumnVecWrite_MPIDense));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatDenseGetSubMatrix_C",MatDenseGetSubMatrix_MPIDense));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatDenseRestoreSubMatrix_C",MatDenseRestoreSubMatrix_MPIDense));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatConvert_mpiaij_mpidense_C",MatConvert_MPIAIJ_MPIDense));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatConvert_mpidense_mpiaij_C",MatConvert_MPIDense_MPIAIJ));
#if defined(PETSC_HAVE_ELEMENTAL)
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatConvert_mpidense_elemental_C",MatConvert_MPIDense_Elemental));
#endif
#if defined(PETSC_HAVE_SCALAPACK)
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatConvert_mpidense_scalapack_C",MatConvert_Dense_ScaLAPACK));
#endif
#if defined(PETSC_HAVE_CUDA)
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatConvert_mpidense_mpidensecuda_C",MatConvert_MPIDense_MPIDenseCUDA));
#endif
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatMPIDenseSetPreallocation_C",MatMPIDenseSetPreallocation_MPIDense));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatProductSetFromOptions_mpiaij_mpidense_C",MatProductSetFromOptions_MPIAIJ_MPIDense));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatProductSetFromOptions_mpidense_mpiaij_C",MatProductSetFromOptions_MPIDense_MPIAIJ));
#if defined(PETSC_HAVE_CUDA)
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatProductSetFromOptions_mpiaijcusparse_mpidense_C",MatProductSetFromOptions_MPIAIJ_MPIDense));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatProductSetFromOptions_mpidense_mpiaijcusparse_C",MatProductSetFromOptions_MPIDense_MPIAIJ));
#endif

  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatDenseGetColumn_C",MatDenseGetColumn_MPIDense));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatDenseRestoreColumn_C",MatDenseRestoreColumn_MPIDense));
  PetscCall(PetscObjectChangeTypeName((PetscObject)mat,MATMPIDENSE));
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
#include <petsc/private/deviceimpl.h>
PETSC_EXTERN PetscErrorCode MatCreate_MPIDenseCUDA(Mat B)
{
  PetscFunctionBegin;
  PetscCall(PetscDeviceInitialize(PETSC_DEVICE_CUDA));
  PetscCall(MatCreate_MPIDense(B));
  PetscCall(MatConvert_MPIDense_MPIDenseCUDA(B,MATMPIDENSECUDA,MAT_INPLACE_MATRIX,&B));
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

.seealso: `MATSEQDENSE`, `MATMPIDENSE`, `MATDENSECUDA`
M*/

/*MC
   MATDENSECUDA - MATDENSECUDA = "densecuda" - A matrix type to be used for dense matrices on GPUs.

   This matrix type is identical to MATSEQDENSECUDA when constructed with a single process communicator,
   and MATMPIDENSECUDA otherwise.

   Options Database Keys:
. -mat_type densecuda - sets the matrix type to "densecuda" during a call to MatSetFromOptions()

  Level: beginner

.seealso: `MATSEQDENSECUDA`, `MATMPIDENSECUDA`, `MATDENSE`
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

.seealso: `MatCreate()`, `MatCreateSeqDense()`, `MatSetValues()`
@*/
PetscErrorCode  MatMPIDenseSetPreallocation(Mat B,PetscScalar *data)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(B,MAT_CLASSID,1);
  PetscTryMethod(B,"MatMPIDenseSetPreallocation_C",(Mat,PetscScalar*),(B,data));
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

.seealso: `MatDenseGetArray()`, `MatDenseResetArray()`, `VecPlaceArray()`, `VecGetArray()`, `VecRestoreArray()`, `VecReplaceArray()`, `VecResetArray()`

@*/
PetscErrorCode  MatDensePlaceArray(Mat mat,const PetscScalar *array)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscUseMethod(mat,"MatDensePlaceArray_C",(Mat,const PetscScalar*),(mat,array));
  PetscCall(PetscObjectStateIncrease((PetscObject)mat));
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

.seealso: `MatDenseGetArray()`, `MatDensePlaceArray()`, `VecPlaceArray()`, `VecGetArray()`, `VecRestoreArray()`, `VecReplaceArray()`, `VecResetArray()`

@*/
PetscErrorCode  MatDenseResetArray(Mat mat)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscUseMethod(mat,"MatDenseResetArray_C",(Mat),(mat));
  PetscCall(PetscObjectStateIncrease((PetscObject)mat));
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

.seealso: `MatDenseGetArray()`, `VecReplaceArray()`
@*/
PetscErrorCode  MatDenseReplaceArray(Mat mat,const PetscScalar *array)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscUseMethod(mat,"MatDenseReplaceArray_C",(Mat,const PetscScalar*),(mat,array));
  PetscCall(PetscObjectStateIncrease((PetscObject)mat));
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

.seealso: `MatDenseCUDAGetArray()`, `MatDenseCUDAResetArray()`
@*/
PetscErrorCode  MatDenseCUDAPlaceArray(Mat mat,const PetscScalar *array)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscUseMethod(mat,"MatDenseCUDAPlaceArray_C",(Mat,const PetscScalar*),(mat,array));
  PetscCall(PetscObjectStateIncrease((PetscObject)mat));
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

.seealso: `MatDenseCUDAGetArray()`, `MatDenseCUDAPlaceArray()`

@*/
PetscErrorCode  MatDenseCUDAResetArray(Mat mat)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscUseMethod(mat,"MatDenseCUDAResetArray_C",(Mat),(mat));
  PetscCall(PetscObjectStateIncrease((PetscObject)mat));
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

.seealso: `MatDenseCUDAGetArray()`, `MatDenseCUDAPlaceArray()`, `MatDenseCUDAResetArray()`
@*/
PetscErrorCode  MatDenseCUDAReplaceArray(Mat mat,const PetscScalar *array)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscUseMethod(mat,"MatDenseCUDAReplaceArray_C",(Mat,const PetscScalar*),(mat,array));
  PetscCall(PetscObjectStateIncrease((PetscObject)mat));
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

.seealso: `MatDenseCUDAGetArray()`, `MatDenseCUDARestoreArray()`, `MatDenseCUDARestoreArrayWrite()`, `MatDenseCUDAGetArrayRead()`, `MatDenseCUDARestoreArrayRead()`
@*/
PetscErrorCode MatDenseCUDAGetArrayWrite(Mat A, PetscScalar **a)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscUseMethod(A,"MatDenseCUDAGetArrayWrite_C",(Mat,PetscScalar**),(A,a));
  PetscCall(PetscObjectStateIncrease((PetscObject)A));
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

.seealso: `MatDenseCUDAGetArray()`, `MatDenseCUDARestoreArray()`, `MatDenseCUDAGetArrayWrite()`, `MatDenseCUDARestoreArrayRead()`, `MatDenseCUDAGetArrayRead()`
@*/
PetscErrorCode MatDenseCUDARestoreArrayWrite(Mat A, PetscScalar **a)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscUseMethod(A,"MatDenseCUDARestoreArrayWrite_C",(Mat,PetscScalar**),(A,a));
  PetscCall(PetscObjectStateIncrease((PetscObject)A));
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

.seealso: `MatDenseCUDAGetArray()`, `MatDenseCUDARestoreArray()`, `MatDenseCUDARestoreArrayWrite()`, `MatDenseCUDAGetArrayWrite()`, `MatDenseCUDARestoreArrayRead()`
@*/
PetscErrorCode MatDenseCUDAGetArrayRead(Mat A, const PetscScalar **a)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscUseMethod(A,"MatDenseCUDAGetArrayRead_C",(Mat,const PetscScalar**),(A,a));
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

.seealso: `MatDenseCUDAGetArray()`, `MatDenseCUDARestoreArray()`, `MatDenseCUDARestoreArrayWrite()`, `MatDenseCUDAGetArrayWrite()`, `MatDenseCUDAGetArrayRead()`
@*/
PetscErrorCode MatDenseCUDARestoreArrayRead(Mat A, const PetscScalar **a)
{
  PetscFunctionBegin;
  PetscUseMethod(A,"MatDenseCUDARestoreArrayRead_C",(Mat,const PetscScalar**),(A,a));
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

.seealso: `MatDenseCUDAGetArrayRead()`, `MatDenseCUDARestoreArray()`, `MatDenseCUDARestoreArrayWrite()`, `MatDenseCUDAGetArrayWrite()`, `MatDenseCUDARestoreArrayRead()`
@*/
PetscErrorCode MatDenseCUDAGetArray(Mat A, PetscScalar **a)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscUseMethod(A,"MatDenseCUDAGetArray_C",(Mat,PetscScalar**),(A,a));
  PetscCall(PetscObjectStateIncrease((PetscObject)A));
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

.seealso: `MatDenseCUDAGetArray()`, `MatDenseCUDARestoreArrayWrite()`, `MatDenseCUDAGetArrayWrite()`, `MatDenseCUDARestoreArrayRead()`, `MatDenseCUDAGetArrayRead()`
@*/
PetscErrorCode MatDenseCUDARestoreArray(Mat A, PetscScalar **a)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscUseMethod(A,"MatDenseCUDARestoreArray_C",(Mat,PetscScalar**),(A,a));
  PetscCall(PetscObjectStateIncrease((PetscObject)A));
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

.seealso: `MatCreate()`, `MatCreateSeqDense()`, `MatSetValues()`
@*/
PetscErrorCode  MatCreateDense(MPI_Comm comm,PetscInt m,PetscInt n,PetscInt M,PetscInt N,PetscScalar *data,Mat *A)
{
  PetscFunctionBegin;
  PetscCall(MatCreate(comm,A));
  PetscCall(MatSetSizes(*A,m,n,M,N));
  PetscCall(MatSetType(*A,MATDENSE));
  PetscCall(MatSeqDenseSetPreallocation(*A,data));
  PetscCall(MatMPIDenseSetPreallocation(*A,data));
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

.seealso: `MatCreate()`, `MatCreateDense()`
@*/
PetscErrorCode  MatCreateDenseCUDA(MPI_Comm comm,PetscInt m,PetscInt n,PetscInt M,PetscInt N,PetscScalar *data,Mat *A)
{
  PetscFunctionBegin;
  PetscCall(MatCreate(comm,A));
  PetscValidLogicalCollectiveBool(*A,!!data,6);
  PetscCall(MatSetSizes(*A,m,n,M,N));
  PetscCall(MatSetType(*A,MATDENSECUDA));
  PetscCall(MatSeqDenseCUDASetPreallocation(*A,data));
  PetscCall(MatMPIDenseCUDASetPreallocation(*A,data));
  PetscFunctionReturn(0);
}
#endif

static PetscErrorCode MatDuplicate_MPIDense(Mat A,MatDuplicateOption cpvalues,Mat *newmat)
{
  Mat            mat;
  Mat_MPIDense   *a,*oldmat = (Mat_MPIDense*)A->data;

  PetscFunctionBegin;
  *newmat = NULL;
  PetscCall(MatCreate(PetscObjectComm((PetscObject)A),&mat));
  PetscCall(MatSetSizes(mat,A->rmap->n,A->cmap->n,A->rmap->N,A->cmap->N));
  PetscCall(MatSetType(mat,((PetscObject)A)->type_name));
  a       = (Mat_MPIDense*)mat->data;

  mat->factortype   = A->factortype;
  mat->assembled    = PETSC_TRUE;
  mat->preallocated = PETSC_TRUE;

  mat->insertmode = NOT_SET_VALUES;
  a->donotstash   = oldmat->donotstash;

  PetscCall(PetscLayoutReference(A->rmap,&mat->rmap));
  PetscCall(PetscLayoutReference(A->cmap,&mat->cmap));

  PetscCall(MatDuplicate(oldmat->A,cpvalues,&a->A));
  PetscCall(PetscLogObjectParent((PetscObject)mat,(PetscObject)a->A));

  *newmat = mat;
  PetscFunctionReturn(0);
}

PetscErrorCode MatLoad_MPIDense(Mat newMat, PetscViewer viewer)
{
  PetscBool      isbinary;
#if defined(PETSC_HAVE_HDF5)
  PetscBool      ishdf5;
#endif

  PetscFunctionBegin;
  PetscValidHeaderSpecific(newMat,MAT_CLASSID,1);
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  /* force binary viewer to load .info file if it has not yet done so */
  PetscCall(PetscViewerSetUp(viewer));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERBINARY,&isbinary));
#if defined(PETSC_HAVE_HDF5)
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERHDF5,  &ishdf5));
#endif
  if (isbinary) {
    PetscCall(MatLoad_Dense_Binary(newMat,viewer));
#if defined(PETSC_HAVE_HDF5)
  } else if (ishdf5) {
    PetscCall(MatLoad_Dense_HDF5(newMat,viewer));
#endif
  } else SETERRQ(PetscObjectComm((PetscObject)newMat),PETSC_ERR_SUP,"Viewer type %s not yet supported for reading %s matrices",((PetscObject)viewer)->type_name,((PetscObject)newMat)->type_name);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatEqual_MPIDense(Mat A,Mat B,PetscBool *flag)
{
  Mat_MPIDense   *matB = (Mat_MPIDense*)B->data,*matA = (Mat_MPIDense*)A->data;
  Mat            a,b;

  PetscFunctionBegin;
  a    = matA->A;
  b    = matB->A;
  PetscCall(MatEqual(a,b,flag));
  PetscCall(MPIU_Allreduce(MPI_IN_PLACE,flag,1,MPIU_BOOL,MPI_LAND,PetscObjectComm((PetscObject)A)));
  PetscFunctionReturn(0);
}

PetscErrorCode MatDestroy_MatTransMatMult_MPIDense_MPIDense(void *data)
{
  Mat_TransMatMultDense *atb = (Mat_TransMatMultDense *)data;

  PetscFunctionBegin;
  PetscCall(PetscFree2(atb->sendbuf,atb->recvcounts));
  PetscCall(MatDestroy(&atb->atb));
  PetscCall(PetscFree(atb));
  PetscFunctionReturn(0);
}

PetscErrorCode MatDestroy_MatMatTransMult_MPIDense_MPIDense(void *data)
{
  Mat_MatTransMultDense *abt = (Mat_MatTransMultDense *)data;

  PetscFunctionBegin;
  PetscCall(PetscFree2(abt->buf[0],abt->buf[1]));
  PetscCall(PetscFree2(abt->recvcounts,abt->recvdispls));
  PetscCall(PetscFree(abt));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatTransposeMatMultNumeric_MPIDense_MPIDense(Mat A,Mat B,Mat C)
{
  Mat_MPIDense          *a=(Mat_MPIDense*)A->data, *b=(Mat_MPIDense*)B->data, *c=(Mat_MPIDense*)C->data;
  Mat_TransMatMultDense *atb;
  MPI_Comm              comm;
  PetscMPIInt           size,*recvcounts;
  PetscScalar           *carray,*sendbuf;
  const PetscScalar     *atbarray;
  PetscInt              i,cN=C->cmap->N,proc,k,j,lda;
  const PetscInt        *ranges;

  PetscFunctionBegin;
  MatCheckProduct(C,3);
  PetscCheck(C->product->data,PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Product data empty");
  atb = (Mat_TransMatMultDense *)C->product->data;
  recvcounts = atb->recvcounts;
  sendbuf = atb->sendbuf;

  PetscCall(PetscObjectGetComm((PetscObject)A,&comm));
  PetscCallMPI(MPI_Comm_size(comm,&size));

  /* compute atbarray = aseq^T * bseq */
  PetscCall(MatTransposeMatMult(a->A,b->A,atb->atb ? MAT_REUSE_MATRIX : MAT_INITIAL_MATRIX,PETSC_DEFAULT,&atb->atb));

  PetscCall(MatGetOwnershipRanges(C,&ranges));

  /* arrange atbarray into sendbuf */
  PetscCall(MatDenseGetArrayRead(atb->atb,&atbarray));
  PetscCall(MatDenseGetLDA(atb->atb,&lda));
  for (proc=0, k=0; proc<size; proc++) {
    for (j=0; j<cN; j++) {
      for (i=ranges[proc]; i<ranges[proc+1]; i++) sendbuf[k++] = atbarray[i+j*lda];
    }
  }
  PetscCall(MatDenseRestoreArrayRead(atb->atb,&atbarray));

  /* sum all atbarray to local values of C */
  PetscCall(MatDenseGetArrayWrite(c->A,&carray));
  PetscCallMPI(MPI_Reduce_scatter(sendbuf,carray,recvcounts,MPIU_SCALAR,MPIU_SUM,comm));
  PetscCall(MatDenseRestoreArrayWrite(c->A,&carray));
  PetscCall(MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatTransposeMatMultSymbolic_MPIDense_MPIDense(Mat A,Mat B,PetscReal fill,Mat C)
{
  MPI_Comm              comm;
  PetscMPIInt           size;
  PetscInt              cm=A->cmap->n,cM,cN=B->cmap->N;
  Mat_TransMatMultDense *atb;
  PetscBool             cisdense;
  PetscInt              i;
  const PetscInt        *ranges;

  PetscFunctionBegin;
  MatCheckProduct(C,3);
  PetscCheck(!C->product->data,PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Product data not empty");
  PetscCall(PetscObjectGetComm((PetscObject)A,&comm));
  if (A->rmap->rstart != B->rmap->rstart || A->rmap->rend != B->rmap->rend) {
    SETERRQ(comm,PETSC_ERR_ARG_SIZ,"Matrix local dimensions are incompatible, A (%" PetscInt_FMT ", %" PetscInt_FMT ") != B (%" PetscInt_FMT ",%" PetscInt_FMT ")",A->rmap->rstart,A->rmap->rend,B->rmap->rstart,B->rmap->rend);
  }

  /* create matrix product C */
  PetscCall(MatSetSizes(C,cm,B->cmap->n,A->cmap->N,B->cmap->N));
  PetscCall(PetscObjectTypeCompareAny((PetscObject)C,&cisdense,MATMPIDENSE,MATMPIDENSECUDA,""));
  if (!cisdense) {
    PetscCall(MatSetType(C,((PetscObject)A)->type_name));
  }
  PetscCall(MatSetUp(C));

  /* create data structure for reuse C */
  PetscCallMPI(MPI_Comm_size(comm,&size));
  PetscCall(PetscNew(&atb));
  cM   = C->rmap->N;
  PetscCall(PetscMalloc2(cM*cN,&atb->sendbuf,size,&atb->recvcounts));
  PetscCall(MatGetOwnershipRanges(C,&ranges));
  for (i=0; i<size; i++) atb->recvcounts[i] = (ranges[i+1] - ranges[i])*cN;

  C->product->data    = atb;
  C->product->destroy = MatDestroy_MatTransMatMult_MPIDense_MPIDense;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMatTransposeMultSymbolic_MPIDense_MPIDense(Mat A, Mat B, PetscReal fill, Mat C)
{
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
  PetscCheck(!C->product->data,PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Product data not empty");
  /* check local size of A and B */
  PetscCheck(A->cmap->n == B->cmap->n,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Matrix local column dimensions are incompatible, A (%" PetscInt_FMT ") != B (%" PetscInt_FMT ")",A->cmap->n,B->cmap->n);

  PetscCall(PetscStrcmp(product->alg,"allgatherv",&flg));
  alg  = flg ? 0 : 1;

  /* setup matrix product C */
  PetscCall(MatSetSizes(C,A->rmap->n,B->rmap->n,A->rmap->N,B->rmap->N));
  PetscCall(MatSetType(C,MATMPIDENSE));
  PetscCall(MatSetUp(C));
  PetscCall(PetscObjectGetNewTag((PetscObject)C,&tag));

  /* create data structure for reuse C */
  PetscCall(PetscObjectGetComm((PetscObject)C,&comm));
  PetscCallMPI(MPI_Comm_size(comm,&size));
  PetscCall(PetscNew(&abt));
  abt->tag = tag;
  abt->alg = alg;
  switch (alg) {
  case 1: /* alg: "cyclic" */
    for (maxRows = 0, i = 0; i < size; i++) maxRows = PetscMax(maxRows, (B->rmap->range[i + 1] - B->rmap->range[i]));
    bufsiz = A->cmap->N * maxRows;
    PetscCall(PetscMalloc2(bufsiz,&(abt->buf[0]),bufsiz,&(abt->buf[1])));
    break;
  default: /* alg: "allgatherv" */
    PetscCall(PetscMalloc2(B->rmap->n * B->cmap->N, &(abt->buf[0]), B->rmap->N * B->cmap->N, &(abt->buf[1])));
    PetscCall(PetscMalloc2(size,&(abt->recvcounts),size+1,&(abt->recvdispls)));
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
  PetscCheck(C->product->data,PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Product data empty");
  abt  = (Mat_MatTransMultDense*)C->product->data;
  PetscCall(PetscObjectGetComm((PetscObject)C,&comm));
  PetscCallMPI(MPI_Comm_rank(comm,&rank));
  PetscCallMPI(MPI_Comm_size(comm,&size));
  PetscCall(MatDenseGetArrayRead(a->A,&av));
  PetscCall(MatDenseGetArrayRead(b->A,&bv));
  PetscCall(MatDenseGetArrayWrite(c->A,&cv));
  PetscCall(MatDenseGetLDA(a->A,&i));
  PetscCall(PetscBLASIntCast(i,&alda));
  PetscCall(MatDenseGetLDA(b->A,&i));
  PetscCall(PetscBLASIntCast(i,&blda));
  PetscCall(MatDenseGetLDA(c->A,&i));
  PetscCall(PetscBLASIntCast(i,&clda));
  PetscCall(MatGetOwnershipRanges(B,&ranges));
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
  PetscCall(PetscBLASIntCast(cK,&ck));
  PetscCall(PetscBLASIntCast(c->A->rmap->n,&cm));
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
      PetscCallMPI(MPI_Isend(sendbuf, sendsiz, MPIU_SCALAR, sendto, abt->tag, comm, &reqs[0]));
      PetscCallMPI(MPI_Irecv(recvbuf, recvsiz, MPIU_SCALAR, recvfrom, abt->tag, comm, &reqs[1]));
    }

    /* local aseq * sendbuf^T */
    PetscCall(PetscBLASIntCast(ranges[recvisfrom + 1] - ranges[recvisfrom], &cn));
    if (cm && cn && ck) PetscStackCallBLAS("BLASgemm",BLASgemm_("N","T",&cm,&cn,&ck,&_DOne,av,&alda,sendbuf,&cn,&_DZero,cv + clda * ranges[recvisfrom],&clda));

    if (nextrecvisfrom != rank) {
      /* wait for the sends and receives to complete, swap sendbuf and recvbuf */
      PetscCallMPI(MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE));
    }
    bn = nextbn;
    recvisfrom = nextrecvisfrom;
    sendbuf = recvbuf;
  }
  PetscCall(MatDenseRestoreArrayRead(a->A,&av));
  PetscCall(MatDenseRestoreArrayRead(b->A,&bv));
  PetscCall(MatDenseRestoreArrayWrite(c->A,&cv));
  PetscCall(MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMatTransposeMultNumeric_MPIDense_MPIDense_Allgatherv(Mat A, Mat B, Mat C)
{
  Mat_MPIDense          *a=(Mat_MPIDense*)A->data, *b=(Mat_MPIDense*)B->data, *c=(Mat_MPIDense*)C->data;
  Mat_MatTransMultDense *abt;
  MPI_Comm              comm;
  PetscMPIInt           size;
  PetscScalar           *cv, *sendbuf, *recvbuf;
  const PetscScalar     *av,*bv;
  PetscInt              blda,i,cK=A->cmap->N,k,j,bn;
  PetscScalar           _DOne=1.0,_DZero=0.0;
  PetscBLASInt          cm, cn, ck, alda, clda;

  PetscFunctionBegin;
  MatCheckProduct(C,3);
  PetscCheck(C->product->data,PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Product data empty");
  abt  = (Mat_MatTransMultDense*)C->product->data;
  PetscCall(PetscObjectGetComm((PetscObject)A,&comm));
  PetscCallMPI(MPI_Comm_size(comm,&size));
  PetscCall(MatDenseGetArrayRead(a->A,&av));
  PetscCall(MatDenseGetArrayRead(b->A,&bv));
  PetscCall(MatDenseGetArrayWrite(c->A,&cv));
  PetscCall(MatDenseGetLDA(a->A,&i));
  PetscCall(PetscBLASIntCast(i,&alda));
  PetscCall(MatDenseGetLDA(b->A,&blda));
  PetscCall(MatDenseGetLDA(c->A,&i));
  PetscCall(PetscBLASIntCast(i,&clda));
  /* copy transpose of B into buf[0] */
  bn      = B->rmap->n;
  sendbuf = abt->buf[0];
  recvbuf = abt->buf[1];
  for (k = 0, j = 0; j < bn; j++) {
    for (i = 0; i < cK; i++, k++) {
      sendbuf[k] = bv[i * blda + j];
    }
  }
  PetscCall(MatDenseRestoreArrayRead(b->A,&bv));
  PetscCallMPI(MPI_Allgatherv(sendbuf, bn * cK, MPIU_SCALAR, recvbuf, abt->recvcounts, abt->recvdispls, MPIU_SCALAR, comm));
  PetscCall(PetscBLASIntCast(cK,&ck));
  PetscCall(PetscBLASIntCast(c->A->rmap->n,&cm));
  PetscCall(PetscBLASIntCast(c->A->cmap->n,&cn));
  if (cm && cn && ck) PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&cm,&cn,&ck,&_DOne,av,&alda,recvbuf,&ck,&_DZero,cv,&clda));
  PetscCall(MatDenseRestoreArrayRead(a->A,&av));
  PetscCall(MatDenseRestoreArrayRead(b->A,&bv));
  PetscCall(MatDenseRestoreArrayWrite(c->A,&cv));
  PetscCall(MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMatTransposeMultNumeric_MPIDense_MPIDense(Mat A, Mat B, Mat C)
{
  Mat_MatTransMultDense *abt;

  PetscFunctionBegin;
  MatCheckProduct(C,3);
  PetscCheck(C->product->data,PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Product data empty");
  abt = (Mat_MatTransMultDense*)C->product->data;
  switch (abt->alg) {
  case 1:
    PetscCall(MatMatTransposeMultNumeric_MPIDense_MPIDense_Cyclic(A, B, C));
    break;
  default:
    PetscCall(MatMatTransposeMultNumeric_MPIDense_MPIDense_Allgatherv(A, B, C));
    break;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatDestroy_MatMatMult_MPIDense_MPIDense(void *data)
{
  Mat_MatMultDense *ab = (Mat_MatMultDense*)data;

  PetscFunctionBegin;
  PetscCall(MatDestroy(&ab->Ce));
  PetscCall(MatDestroy(&ab->Ae));
  PetscCall(MatDestroy(&ab->Be));
  PetscCall(PetscFree(ab));
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_ELEMENTAL)
PetscErrorCode MatMatMultNumeric_MPIDense_MPIDense(Mat A,Mat B,Mat C)
{
  Mat_MatMultDense *ab;

  PetscFunctionBegin;
  MatCheckProduct(C,3);
  PetscCheck(C->product->data,PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Missing product data");
  ab   = (Mat_MatMultDense*)C->product->data;
  PetscCall(MatConvert_MPIDense_Elemental(A,MATELEMENTAL,MAT_REUSE_MATRIX, &ab->Ae));
  PetscCall(MatConvert_MPIDense_Elemental(B,MATELEMENTAL,MAT_REUSE_MATRIX, &ab->Be));
  PetscCall(MatMatMultNumeric_Elemental(ab->Ae,ab->Be,ab->Ce));
  PetscCall(MatConvert(ab->Ce,MATMPIDENSE,MAT_REUSE_MATRIX,&C));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMatMultSymbolic_MPIDense_MPIDense(Mat A,Mat B,PetscReal fill,Mat C)
{
  Mat              Ae,Be,Ce;
  Mat_MatMultDense *ab;

  PetscFunctionBegin;
  MatCheckProduct(C,4);
  PetscCheck(!C->product->data,PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Product data not empty");
  /* check local size of A and B */
  if (A->cmap->rstart != B->rmap->rstart || A->cmap->rend != B->rmap->rend) {
    SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_SIZ,"Matrix local dimensions are incompatible, A (%" PetscInt_FMT ", %" PetscInt_FMT ") != B (%" PetscInt_FMT ",%" PetscInt_FMT ")",A->rmap->rstart,A->rmap->rend,B->rmap->rstart,B->rmap->rend);
  }

  /* create elemental matrices Ae and Be */
  PetscCall(MatCreate(PetscObjectComm((PetscObject)A), &Ae));
  PetscCall(MatSetSizes(Ae,PETSC_DECIDE,PETSC_DECIDE,A->rmap->N,A->cmap->N));
  PetscCall(MatSetType(Ae,MATELEMENTAL));
  PetscCall(MatSetUp(Ae));
  PetscCall(MatSetOption(Ae,MAT_ROW_ORIENTED,PETSC_FALSE));

  PetscCall(MatCreate(PetscObjectComm((PetscObject)B), &Be));
  PetscCall(MatSetSizes(Be,PETSC_DECIDE,PETSC_DECIDE,B->rmap->N,B->cmap->N));
  PetscCall(MatSetType(Be,MATELEMENTAL));
  PetscCall(MatSetUp(Be));
  PetscCall(MatSetOption(Be,MAT_ROW_ORIENTED,PETSC_FALSE));

  /* compute symbolic Ce = Ae*Be */
  PetscCall(MatCreate(PetscObjectComm((PetscObject)C),&Ce));
  PetscCall(MatMatMultSymbolic_Elemental(Ae,Be,fill,Ce));

  /* setup C */
  PetscCall(MatSetSizes(C,A->rmap->n,B->cmap->n,PETSC_DECIDE,PETSC_DECIDE));
  PetscCall(MatSetType(C,MATDENSE));
  PetscCall(MatSetUp(C));

  /* create data structure for reuse Cdense */
  PetscCall(PetscNew(&ab));
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
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Matrix local dimensions are incompatible, (%" PetscInt_FMT ", %" PetscInt_FMT ") != (%" PetscInt_FMT ",%" PetscInt_FMT ")",A->rmap->rstart,A->rmap->rend,B->rmap->rstart,B->rmap->rend);
  C->ops->transposematmultsymbolic = MatTransposeMatMultSymbolic_MPIDense_MPIDense;
  C->ops->productsymbolic = MatProductSymbolic_AtB;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatProductSetFromOptions_MPIDense_ABt(Mat C)
{
  Mat_Product    *product = C->product;
  const char     *algTypes[2] = {"allgatherv","cyclic"};
  PetscInt       alg,nalg = 2;
  PetscBool      flg = PETSC_FALSE;

  PetscFunctionBegin;
  /* Set default algorithm */
  alg = 0; /* default is allgatherv */
  PetscCall(PetscStrcmp(product->alg,"default",&flg));
  if (flg) {
    PetscCall(MatProductSetAlgorithm(C,(MatProductAlgorithm)algTypes[alg]));
  }

  /* Get runtime option */
  if (product->api_user) {
    PetscOptionsBegin(PetscObjectComm((PetscObject)C),((PetscObject)C)->prefix,"MatMatTransposeMult","Mat");
    PetscCall(PetscOptionsEList("-matmattransmult_mpidense_mpidense_via","Algorithmic approach","MatMatTransposeMult",algTypes,nalg,algTypes[alg],&alg,&flg));
    PetscOptionsEnd();
  } else {
    PetscOptionsBegin(PetscObjectComm((PetscObject)C),((PetscObject)C)->prefix,"MatProduct_ABt","Mat");
    PetscCall(PetscOptionsEList("-mat_product_algorithm","Algorithmic approach","MatProduct_ABt",algTypes,nalg,algTypes[alg],&alg,&flg));
    PetscOptionsEnd();
  }
  if (flg) {
    PetscCall(MatProductSetAlgorithm(C,(MatProductAlgorithm)algTypes[alg]));
  }

  C->ops->mattransposemultsymbolic = MatMatTransposeMultSymbolic_MPIDense_MPIDense;
  C->ops->productsymbolic          = MatProductSymbolic_ABt;
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatProductSetFromOptions_MPIDense(Mat C)
{
  Mat_Product    *product = C->product;

  PetscFunctionBegin;
  switch (product->type) {
#if defined(PETSC_HAVE_ELEMENTAL)
  case MATPRODUCT_AB:
    PetscCall(MatProductSetFromOptions_MPIDense_AB(C));
    break;
#endif
  case MATPRODUCT_AtB:
    PetscCall(MatProductSetFromOptions_MPIDense_AtB(C));
    break;
  case MATPRODUCT_ABt:
    PetscCall(MatProductSetFromOptions_MPIDense_ABt(C));
    break;
  default:
    break;
  }
  PetscFunctionReturn(0);
}
