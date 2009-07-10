#ifndef _PETSC_COMPAT_MAT_H
#define _PETSC_COMPAT_MAT_H

#include "private/matimpl.h"

#define MATBLOCKMAT  "blockmat"
#define MATCOMPOSITE "composite"
#define MATSEQFFTW   "seqfftw"

#undef __FUNCT__
#define __FUNCT__ "MatSetUp_232"
static PETSC_UNUSED
PetscErrorCode PETSCMAT_DLLEXPORT MatSetUp_232(Mat A)
{
  PetscMPIInt    size;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_COOKIE,1);
  if (!((PetscObject)A)->type_name) {
    ierr = MPI_Comm_size(((PetscObject)A)->comm, &size);CHKERRQ(ierr);
    if (size == 1) { ierr = MatSetType(A, MATSEQAIJ);CHKERRQ(ierr); }
    else           { ierr = MatSetType(A, MATMPIAIJ);CHKERRQ(ierr); }
  }
  ierr = MatSetUpPreallocation(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#define MatSetUp MatSetUp_232

#define MAT_ROW_ORIENTED                  MAT_ROW_ORIENTED
#define MAT_NEW_NONZERO_LOCATIONS         MAT_YES_NEW_NONZERO_LOCATIONS
#define MAT_SYMMETRIC                     MAT_SYMMETRIC
#define MAT_STRUCTURALLY_SYMMETRIC        MAT_STRUCTURALLY_SYMMETRIC
#define MAT_NEW_DIAGONALS                 MAT_YES_NEW_DIAGONALS
#define MAT_IGNORE_OFF_PROC_ENTRIES       MAT_IGNORE_OFF_PROC_ENTRIES
#define MAT_NEW_NONZERO_LOCATION_ERR      MAT_NEW_NONZERO_LOCATION_ERR
#define MAT_NEW_NONZERO_ALLOCATION_ERR    MAT_NEW_NONZERO_ALLOCATION_ERR
#define MAT_USE_HASH_TABLE                MAT_USE_HASH_TABLE
#define MAT_KEEP_NONZERO_PATTERN          MAT_KEEP_ZEROED_ROWS
#define MAT_IGNORE_ZERO_ENTRIES           MAT_IGNORE_ZERO_ENTRIES
#define MAT_USE_INODES                    MAT_USE_INODES
#define MAT_HERMITIAN                     MAT_HERMITIAN
#define MAT_SYMMETRY_ETERNAL              MAT_SYMMETRY_ETERNAL
#define MAT_USE_COMPRESSEDROW             MAT_USE_COMPRESSEDROW
#define MAT_IGNORE_LOWER_TRIANGULAR       MAT_IGNORE_LOWER_TRIANGULAR
#define MAT_ERROR_LOWER_TRIANGULAR        MAT_ERROR_LOWER_TRIANGULAR
#define MAT_GETROW_UPPERTRIANGULAR        MAT_GETROW_UPPERTRIANGULAR

#undef __FUNCT__
#define __FUNCT__ "MatSetOption_232"
static PETSC_UNUSED
PetscErrorCode MatSetOption_232(Mat mat,MatOption op,PetscTruth flag) {
#define MAT_OPTION_INVALID ((MatOption)(100))
  PetscErrorCode ierr;
  MatOption o = MAT_OPTION_INVALID;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  switch (op) {

  case MAT_ROW_ORIENTED:
    o = (flag ? MAT_ROW_ORIENTED : MAT_COLUMN_ORIENTED); break;
  case MAT_NEW_NONZERO_LOCATIONS:
    o = (flag ? MAT_YES_NEW_NONZERO_LOCATIONS : MAT_NO_NEW_NONZERO_LOCATIONS); break;
  case MAT_SYMMETRIC:
    o = (flag ? MAT_SYMMETRIC : MAT_NOT_SYMMETRIC); break;
  case MAT_STRUCTURALLY_SYMMETRIC:
    o = (flag ? MAT_STRUCTURALLY_SYMMETRIC : MAT_NOT_STRUCTURALLY_SYMMETRIC); break;
  case MAT_NEW_DIAGONALS:
    o = (flag ? MAT_YES_NEW_DIAGONALS : MAT_NO_NEW_DIAGONALS); break;
  case MAT_IGNORE_OFF_PROC_ENTRIES:
    o = (flag ? MAT_IGNORE_OFF_PROC_ENTRIES : MAT_OPTION_INVALID); break;
  case MAT_NEW_NONZERO_LOCATION_ERR:
    o = (flag ? MAT_NEW_NONZERO_LOCATION_ERR : MAT_OPTION_INVALID); break;
  case MAT_NEW_NONZERO_ALLOCATION_ERR:
    o = (flag ? MAT_NEW_NONZERO_ALLOCATION_ERR : MAT_OPTION_INVALID); break;
  case MAT_USE_HASH_TABLE:
    o = (flag ? MAT_USE_HASH_TABLE : MAT_OPTION_INVALID); break;
  case MAT_KEEP_NONZERO_PATTERN:
    o = (flag ? MAT_KEEP_NONZERO_PATTERN : MAT_OPTION_INVALID); break;
  case MAT_IGNORE_ZERO_ENTRIES:
    o = (flag ? MAT_IGNORE_ZERO_ENTRIES : MAT_OPTION_INVALID); break;
  case MAT_USE_INODES:
    o = (flag ? MAT_USE_INODES : MAT_DO_NOT_USE_INODES); break;
  case MAT_HERMITIAN:
    o = (flag ? MAT_HERMITIAN: MAT_NOT_HERMITIAN); break;
  case MAT_SYMMETRY_ETERNAL:
    o = (flag ? MAT_SYMMETRY_ETERNAL : MAT_NOT_SYMMETRY_ETERNAL); break;
  case MAT_USE_COMPRESSEDROW:
    o = (flag ? MAT_USE_COMPRESSEDROW : MAT_DO_NOT_USE_COMPRESSEDROW); break;
  case MAT_IGNORE_LOWER_TRIANGULAR:
    o = (flag ? MAT_IGNORE_LOWER_TRIANGULAR : MAT_OPTION_INVALID); break;
  case MAT_ERROR_LOWER_TRIANGULAR:
    o = (flag ? MAT_ERROR_LOWER_TRIANGULAR : MAT_OPTION_INVALID); break;
  case MAT_GETROW_UPPERTRIANGULAR:
    o = (flag ? MAT_GETROW_UPPERTRIANGULAR : MAT_OPTION_INVALID); break;
  default:
    o = op; break;
  }
  if (o == MAT_OPTION_INVALID) {
    SETERRQ(1, "option and flag combination unsupported");
    PetscFunctionReturn(1);
  }
  ierr = MatSetOption(mat,o);CHKERRQ(ierr);
  PetscFunctionReturn(0);
#undef MAT_OPTION_INVALID
}
#define MatSetOption MatSetOption_232


#define MatGetRowIJ(mat,shift,symm,bc,n,ia,ja,done) \
	MatGetRowIJ(mat,shift,symm,n,ia,ja,done)
#define MatRestoreRowIJ(mat,shift,symm,bc,n,ia,ja,done) \
	MatRestoreRowIJ(mat,shift,symm,n,ia,ja,done)
#define MatGetColumnIJ(mat,shift,symm,bc,n,ia,ja,done) \
	MatGetColumnIJ(mat,shift,symm,n,ia,ja,done)
#define MatRestoreColumnIJ(mat,shift,symm,bc,n,ia,ja,done) \
	MatRestoreColumnIJ(mat,shift,symm,n,ia,ja,done)

#undef __FUNCT__
#define __FUNCT__ "MatIsHermitian_232"
static PETSC_UNUSED
PetscErrorCode MatIsHermitian_232(Mat A,PetscReal tol,PetscTruth *flg)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = MatIsHermitian(A,flg);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#define MatIsHermitian MatIsHermitian_232


#undef __FUNCT__
#define __FUNCT__ "MatSetOption_232"
static PETSC_UNUSED
PetscErrorCode MatTranspose_232(Mat mat,MatReuse reuse,Mat *B)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidPointer(B,3);
  if (mat == *B) { /* always in-place */
    ierr= MatTranspose(mat,PETSC_NULL);CHKERRQ(ierr);
    PetscFunctionReturn(ierr);
  } else {
    if (reuse == MAT_INITIAL_MATRIX) {
      ierr = MatTranspose(mat,B);CHKERRQ(ierr);
      PetscFunctionReturn(ierr);
    } else {
      SETERRQ(PETSC_ERR_SUP,"MAT_REUSE_MATRIX only supported for in-place transpose currently");
      PetscFunctionReturn(PETSC_ERR_SUP);
    }
  }
  PetscFunctionReturn(0);
}
#define MatTranspose MatTranspose_232

#undef __FUNCT__
#define __FUNCT__ "MatSetBlockSize_232"
static PETSC_UNUSED
PetscErrorCode MatSetBlockSize_232(Mat mat,PetscInt bs)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidType(mat,1);

  if (bs < 1) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"Invalid block size specified, must be positive but it is %D",bs);
  if (mat->ops->setblocksize) {
    mat->rmap.bs = mat->cmap.bs = bs;
    ierr = (*mat->ops->setblocksize)(mat,bs);CHKERRQ(ierr);
  } else if (mat->rmap.bs != bs || mat->cmap.bs != bs) {
    SETERRQ1(PETSC_ERR_ARG_INCOMP,"Cannot set/change the block size for matrix type %s",((PetscObject)mat)->type_name);
  }
  PetscFunctionReturn(0);
}
#define MatSetBlockSize MatSetBlockSize_232

PetscErrorCode PETSCMAT_DLLEXPORT MatGetOwnershipRanges_232(Mat mat,const PetscInt *ranges[])
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidType(mat,1);
  PetscValidPointer(ranges,2);
  ierr = MatPreallocated(mat);CHKERRQ(ierr);
  ierr = PetscMapGetGlobalRange(&mat->rmap,(PetscInt **)ranges);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#define MatGetOwnershipRanges MatGetOwnershipRanges_232

#undef __FUNCT__
#define __FUNCT__ "MatGetOwnershipRangeColumn_232"
static PETSC_UNUSED
PetscErrorCode MatGetOwnershipRangeColumn_232(Mat mat,PetscInt *m,PetscInt *n)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidType(mat,1);
  if (m) PetscValidIntPointer(m,2);
  if (n) PetscValidIntPointer(n,3);
  ierr = MatPreallocated(mat);CHKERRQ(ierr);
  if (m) *m = mat->cmap.rstart;
  if (n) *n = mat->cmap.rend;
  PetscFunctionReturn(0);
}
#define MatGetOwnershipRangeColumn MatGetOwnershipRangeColumn_232

#undef __FUNCT__
#define __FUNCT__ "MatGetOwnershipRangesColumn_232"
static PETSC_UNUSED
PetscErrorCode MatGetOwnershipRangesColumn_232(Mat mat,const PetscInt *ranges[])
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidType(mat,1);
  PetscValidPointer(ranges,2);
  ierr = MatPreallocated(mat);CHKERRQ(ierr);
  ierr = PetscMapGetGlobalRange(&mat->cmap,(PetscInt **)ranges);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#define MatGetOwnershipRangesColumn MatGetOwnershipRangesColumn_232

#undef __FUNCT__
#define __FUNCT__ "MatSetValuesBlocked_232"
static PETSC_UNUSED
PetscErrorCode MatSetValuesBlocked_232(Mat mat,PetscInt m,const PetscInt idxm[],PetscInt n,const PetscInt idxn[],const PetscScalar v[],InsertMode addv)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!m || !n) PetscFunctionReturn(0); /* no values to insert */
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidType(mat,1);
  PetscValidIntPointer(idxm,3);
  PetscValidIntPointer(idxn,5);
  PetscValidScalarPointer(v,6);
  ierr = MatPreallocated(mat);CHKERRQ(ierr);
  if (mat->insertmode == NOT_SET_VALUES) {
    mat->insertmode = addv;
  }
#if defined(PETSC_USE_DEBUG)
  else if (mat->insertmode != addv) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Cannot mix add values and insert values");
  }
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");
#endif

  if (mat->assembled) {
    mat->was_assembled = PETSC_TRUE;
    mat->assembled     = PETSC_FALSE;
  }
  ierr = PetscLogEventBegin(MAT_SetValues,mat,0,0,0);CHKERRQ(ierr);
  if (mat->ops->setvaluesblocked) {
    ierr = (*mat->ops->setvaluesblocked)(mat,m,idxm,n,idxn,v,addv);CHKERRQ(ierr);
  } else if (!mat->ops->setvalues) {
    SETERRQ1(PETSC_ERR_SUP,"Mat type %s",((PetscObject)mat)->type_name);
  } else {
    PetscInt i,j,bs = mat->rmap.bs;
    PetscInt *iidxm,*iidxn;
    PetscInt aux[4096],*itmpm=0,*itmpn=0;
    if ((m+n)*bs <= 4096) {
      iidxm = aux; iidxn = aux + m*bs;
    } else {
      ierr = PetscMalloc2(m*bs,PetscInt,&itmpm,n*bs,PetscInt,&itmpn);CHKERRQ(ierr);
      iidxm = itmpm; iidxn = itmpn;
    }
    for (i=0; i<m; i++) {
      for (j=0; j<bs; j++) {
	iidxm[i*bs+j] = bs*idxm[i] + j;
      }
    }
    for (i=0; i<n; i++) {
      for (j=0; j<bs; j++) {
	iidxn[i*bs+j] = bs*idxn[i] + j;
      }
    }
    ierr = MatSetValues(mat,bs*m,iidxm,bs*n,iidxn,v,addv);CHKERRQ(ierr);
    ierr = PetscFree2(itmpm,itmpn);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(MAT_SetValues,mat,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#define MatSetValuesBlocked MatSetValuesBlocked_232


#undef __FUNCT__
#define __FUNCT__ "MatSetValuesBlockedLocal_232"
static PETSC_UNUSED
PetscErrorCode MatSetValuesBlockedLocal_232(Mat mat,PetscInt m,const PetscInt irow[],PetscInt n,const PetscInt icol[],const PetscScalar v[],InsertMode addv)
{
  PetscErrorCode ierr;
  PetscInt       idxm[2048],idxn[2048];

  PetscFunctionBegin;
  if (!m || !n) PetscFunctionReturn(0); /* no values to insert */
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidType(mat,1);
  PetscValidIntPointer(irow,3);
  PetscValidIntPointer(icol,5);
  PetscValidScalarPointer(v,6);
  ierr = MatPreallocated(mat);CHKERRQ(ierr);
  if (mat->insertmode == NOT_SET_VALUES) {
    mat->insertmode = addv;
  }
#if defined(PETSC_USE_DEBUG)
  else if (mat->insertmode != addv) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Cannot mix add values and insert values");
  }
  if (!mat->bmapping) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Local to global never set with MatSetLocalToGlobalMappingBlock()");
  }
  if (m > 2048 || n > 2048) {
    SETERRQ2(PETSC_ERR_SUP,"Number column/row indices must be <= 2048: are %D %D",m,n);
  }
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");
#endif

  if (mat->assembled) {
    mat->was_assembled = PETSC_TRUE;
    mat->assembled     = PETSC_FALSE;
  }
  ierr = PetscLogEventBegin(MAT_SetValues,mat,0,0,0);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingApply(mat->bmapping,m,irow,idxm);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingApply(mat->bmapping,n,icol,idxn);CHKERRQ(ierr);
  if (mat->ops->setvaluesblocked) {
    ierr = (*mat->ops->setvaluesblocked)(mat,m,idxm,n,idxn,v,addv);CHKERRQ(ierr);
  } else if (!mat->ops->setvalues) {
    SETERRQ1(PETSC_ERR_SUP,"Mat type %s",((PetscObject)mat)->type_name);
  } else {
    PetscInt i,j,bs = mat->rmap.bs;
    PetscInt *iidxm,*iidxn;
    PetscInt aux[4096],*itmpm=0,*itmpn=0;
    if ((m+n)*bs <= 4096) {
      iidxm = aux; iidxn = aux + m*bs;
    } else {
      ierr = PetscMalloc2(m*bs,PetscInt,&itmpm,n*bs,PetscInt,&itmpn);CHKERRQ(ierr);
      iidxm = itmpm; iidxn = itmpn;
    }
    for (i=0; i<m; i++) {
      for (j=0; j<bs; j++) {
	iidxm[i*bs+j] = bs*idxm[i] + j;
      }
    }
    for (i=0; i<n; i++) {
      for (j=0; j<bs; j++) {
	iidxn[i*bs+j] = bs*idxn[i] + j;
      }
    }
    ierr = MatSetValues(mat,bs*m,iidxm,bs*n,iidxn,v,addv);CHKERRQ(ierr);
    ierr = PetscFree2(itmpm,itmpn);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(MAT_SetValues,mat,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#define MatSetValuesBlockedLocal MatSetValuesBlockedLocal_232

#undef __FUNCT__
#define __FUNCT__ "MatSeqBAIJSetPreallocationCSR_SeqBAIJ_232"
static PETSC_UNUSED
PetscErrorCode MatSeqBAIJSetPreallocationCSR_SeqBAIJ_232(Mat B,PetscInt bs,
							 const PetscInt Ii[],
							 const PetscInt Jj[],
							 const PetscScalar V[])
{
  PetscInt       i,m,nz,nz_max=0,*nnz;
  PetscScalar    *values=0;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  if (bs < 1) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"Invalid block size specified, must be positive but it is %D",bs);
  B->rmap.bs = bs;
  B->cmap.bs = bs;
  ierr = PetscMapInitialize(B->comm,&B->rmap);CHKERRQ(ierr);
  ierr = PetscMapInitialize(B->comm,&B->cmap);CHKERRQ(ierr);
  m = B->rmap.n/bs;

  if (Ii[0] != 0) { SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE, "I[0] must be 0 but it is %D",Ii[0]); }
  ierr = PetscMalloc((m+1) * sizeof(PetscInt), &nnz);CHKERRQ(ierr);
  for(i=0; i<m; i++) {
    nz = Ii[i+1]- Ii[i];
    if (nz < 0) { SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE, "Local row %D has a negative number of columns %D",i,nz); }
    nz_max = PetscMax(nz_max, nz);
    nnz[i] = nz;
  }
  ierr = MatSeqBAIJSetPreallocation(B,bs,0,nnz);CHKERRQ(ierr);
  ierr = PetscFree(nnz);CHKERRQ(ierr);

  values = (PetscScalar*)V;
  if (!values) {
    ierr = PetscMalloc(bs*bs*(nz_max+1)*sizeof(PetscScalar),&values);CHKERRQ(ierr);
    ierr = PetscMemzero(values,bs*bs*nz_max*sizeof(PetscScalar));CHKERRQ(ierr);
  }
  for (i=0; i<m; i++) {
    PetscInt          ncols  = Ii[i+1] - Ii[i];
    const PetscInt    *icols = Jj + Ii[i];
    const PetscScalar *svals = values + (V ? (bs*bs*Ii[i]) : 0);
    ierr = MatSetValuesBlocked(B,1,&i,ncols,icols,svals,INSERT_VALUES);CHKERRQ(ierr);
  }
  if (!V) { ierr = PetscFree(values);CHKERRQ(ierr); }
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
#define MatSeqBAIJSetPreallocationCSR_SeqBAIJ MatSeqBAIJSetPreallocationCSR_SeqBAIJ_232

#undef __FUNCT__
#define __FUNCT__ "MatSeqBAIJSetPreallocationCSR_232"
static PETSC_UNUSED
PetscErrorCode MatSeqBAIJSetPreallocationCSR_232(Mat B,PetscInt bs,
						 const PetscInt i[],
						 const PetscInt j[],
						 const PetscScalar v[])
{
  PetscErrorCode ierr,(*f)(Mat,PetscInt,const PetscInt[],const PetscInt[],const PetscScalar[]);
  PetscFunctionBegin;
  ierr = PetscObjectQueryFunction((PetscObject)B, "MatSeqBAIJSetPreallocation_C",
				  (void (**)(void))&f);CHKERRQ(ierr);
  if (f) { f = MatSeqBAIJSetPreallocationCSR_SeqBAIJ; }
  if (f) {
    ierr = (*f)(B,bs,i,j,v);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
#define MatSeqBAIJSetPreallocationCSR MatSeqBAIJSetPreallocationCSR_232

#undef __FUNCT__
#define __FUNCT__ "MatMPIBAIJSetPreallocationCSR_MPIBAIJ_232"
static PETSC_UNUSED
PetscErrorCode MatMPIBAIJSetPreallocationCSR_MPIBAIJ_232(Mat B, PetscInt bs,
							 const PetscInt Ii[],
							 const PetscInt Jj[],
							 const PetscScalar V[])
{
  PetscInt       m,rstart,cstart,cend;
  PetscInt       i,j,d,nz,nz_max=0,*d_nnz=0,*o_nnz=0;
  const PetscInt *jj;
  PetscScalar    *values=0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* XXX explain */
  if (bs < 1) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"Invalid block size specified, must be positive but it is %D",bs);
  B->rmap.bs = bs;
  B->cmap.bs = bs;
  ierr = PetscMapInitialize(B->comm,&B->rmap);CHKERRQ(ierr);
  ierr = PetscMapInitialize(B->comm,&B->cmap);CHKERRQ(ierr);
  m      = B->rmap.n/bs;
  rstart = B->rmap.rstart/bs;
  cstart = B->cmap.rstart/bs;
  cend   = B->cmap.rend/bs;
  /* XXX explain */
  if (Ii[0]) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"I[0] must be 0 but it is %D",Ii[0]);
  ierr  = PetscMalloc((2*m+1)*sizeof(PetscInt),&d_nnz);CHKERRQ(ierr);
  o_nnz = d_nnz + m;
  for (i=0; i<m; i++) {
    nz = Ii[i+1] - Ii[i];
    if (nz < 0) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Local row %D has a negative number of columns %D",i,nz);
    nz_max = PetscMax(nz_max,nz);
    jj  = Jj + Ii[i];
    for (j=0; j<nz; j++) {
      if (*jj >= cstart) break;
      jj++;
    }
    d = 0;
    for (; j<nz; j++) {
      if (*jj++ >= cend) break;
      d++;
    }
    d_nnz[i] = d;
    o_nnz[i] = nz - d;
  }
  ierr = MatMPIBAIJSetPreallocation(B,bs,0,d_nnz,0,o_nnz);CHKERRQ(ierr);
  ierr = PetscFree(d_nnz);CHKERRQ(ierr);
  /* XXX explain */
  values = (PetscScalar*)V;
  if (!values) {
    ierr = PetscMalloc(bs*bs*(nz_max+1)*sizeof(PetscScalar),&values);CHKERRQ(ierr);
    ierr = PetscMemzero(values,bs*bs*nz_max*sizeof(PetscScalar));CHKERRQ(ierr);
  }
  for (i=0; i<m; i++) {
    PetscInt          row    = i + rstart;
    PetscInt          ncols  = Ii[i+1] - Ii[i];
    const PetscInt    *icols = Jj + Ii[i];
    const PetscScalar *svals = values + (V ? (bs*bs*Ii[i]) : 0);
    ierr = MatSetValuesBlocked(B,1,&row,ncols,icols,svals,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (!V) { ierr = PetscFree(values);CHKERRQ(ierr); }
  PetscFunctionReturn(0);
}
#define MatMPIBAIJSetPreallocationCSR_MPIBAIJ MatMPIBAIJSetPreallocationCSR_MPIBAIJ_232

#undef __FUNCT__
#define __FUNCT__ "MatMPIBAIJSetPreallocationCSR_232"
static PETSC_UNUSED
PetscErrorCode MatMPIBAIJSetPreallocationCSR_232(Mat B,PetscInt bs,
						 const PetscInt i[],
						 const PetscInt j[],
						 const PetscScalar v[])
{
  PetscErrorCode ierr,(*f)(Mat,PetscInt,const PetscInt[],const PetscInt[],const PetscScalar[]);
  PetscFunctionBegin;
  ierr = PetscObjectQueryFunction((PetscObject)B, "MatMPIBAIJSetPreallocationCSR_C",
				  (void (**)(void))&f);CHKERRQ(ierr);
  if (f) { f = MatMPIBAIJSetPreallocationCSR_MPIBAIJ; }
  if (f) {
    ierr = (*f)(B,bs,i,j,v);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
#define MatMPIBAIJSetPreallocationCSR MatMPIBAIJSetPreallocationCSR_232

#undef __FUNCT__
#define __FUNCT__ "MatGetDiagonalBlock_232"
static PETSC_UNUSED
PetscErrorCode MatGetDiagonalBlock_232(Mat A,PetscTruth *iscopy,MatReuse reuse,Mat *a)
{
  PetscErrorCode ierr,(*f)(Mat,PetscTruth*,MatReuse,Mat*);
  PetscMPIInt    size;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_COOKIE,1);
  PetscValidPointer(iscopy,2);
  PetscValidPointer(a,3);
  PetscValidType(A,1);
  if (!A->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (A->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 
  ierr = MPI_Comm_size(((PetscObject)A)->comm,&size);CHKERRQ(ierr);
  ierr = PetscObjectQueryFunction((PetscObject)A,"MatGetDiagonalBlock_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(A,iscopy,reuse,a);CHKERRQ(ierr);
  } else if (size == 1) {
    *a = A;
    *iscopy = PETSC_FALSE;
  } else {
    SETERRQ(PETSC_ERR_SUP,"Cannot get diagonal part for this matrix");
  }
  PetscFunctionReturn(0);
}
#define MatGetDiagonalBlock MatGetDiagonalBlock_232

#undef __FUNCT__
#define __FUNCT__ "MatGetSubMatrix_232"
static PETSC_UNUSED
PetscErrorCode MatGetSubMatrix_232(Mat mat,IS isrow,IS iscol,MatReuse cll,Mat *newmat)
{
  MPI_Comm comm;
  PetscMPIInt size;
  IS iscolall = PETSC_NULL, iscoltmp = PETSC_NULL;
  PetscInt csize = PETSC_DECIDE;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidHeaderSpecific(isrow,IS_COOKIE,2);
  if (iscol) PetscValidHeaderSpecific(iscol,IS_COOKIE,3);
  PetscValidPointer(newmat,6);
  if (cll == MAT_REUSE_MATRIX) PetscValidHeaderSpecific(*newmat,MAT_COOKIE,6);
  PetscValidType(mat,1);

  ierr = PetscObjectGetComm((PetscObject)mat,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  if (!iscol) {
    PetscInt N;
    ierr = MatGetSize(mat, PETSC_NULL, &N);CHKERRQ(ierr);
    ierr = ISCreateStride(comm,N,0,1,&iscoltmp);CHKERRQ(ierr);
    iscol = iscoltmp;
  }
  if (iscol) {
    ierr = ISGetLocalSize(iscol,&csize);CHKERRQ(ierr);
    if (size == 1) {
      iscolall = iscol;
    } else if (cll == MAT_INITIAL_MATRIX) {
      ierr = ISAllGather(iscol, &iscolall);CHKERRQ(ierr);
    } else {
      ierr = PetscObjectQuery((PetscObject)*newmat,"ISAllGather",(PetscObject*)&iscolall);CHKERRQ(ierr);
      if (!iscolall) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Submatrix passed in was not used before, cannot reuse");
    }
  }
  ierr = MatGetSubMatrix(mat,isrow,iscolall,csize,cll,newmat); CHKERRQ(ierr);
  if (iscol && size > 1 && cll == MAT_INITIAL_MATRIX) {
    ierr = PetscObjectCompose((PetscObject)*newmat,"ISAllGather",(PetscObject)iscolall);CHKERRQ(ierr);
    ierr = ISDestroy(iscolall); CHKERRQ(ierr);
  }
  if (iscoltmp) { ierr = ISDestroy(iscoltmp); CHKERRQ(ierr); }

  PetscFunctionReturn(0);
}
#define MatGetSubMatrix MatGetSubMatrix_232

#endif /* _PETSC_COMPAT_MAT_H */
