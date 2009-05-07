#ifndef _PETSC_COMPAT_MAT_H
#define _PETSC_COMPAT_MAT_H

#include "private/matimpl.h"

#define MATBLOCKMAT  "blockmat"
#define MATCOMPOSITE "composite"
#define MATSEQFFTW   "seqfftw"

#undef __FUNCT__
#define __FUNCT__ "MatSetUp_233"
static PETSC_UNUSED
PetscErrorCode PETSCMAT_DLLEXPORT MatSetUp_233(Mat A)
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
#define MatSetUp MatSetUp_233


#define MAT_NEW_NONZERO_LOCATIONS MAT_YES_NEW_NONZERO_LOCATIONS
#define MAT_NEW_DIAGONALS         MAT_YES_NEW_DIAGONALS

#undef __FUNCT__
#define __FUNCT__ "MatSetOption_233"
static PETSC_UNUSED
PetscErrorCode MatSetOption_233(Mat A,MatOption op,PetscTruth flag)
{
#define MAT_OPTION_INVALID ((MatOption)(-1))
  MatOption o = MAT_OPTION_INVALID;
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

  case MAT_KEEP_ZEROED_ROWS:
    o = (flag ? MAT_KEEP_ZEROED_ROWS : MAT_OPTION_INVALID); break;

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
  return MatSetOption(A,o);
#undef MAT_OPTION_INVALID
}
#define MatSetOption MatSetOption_233

#undef __FUNCT__
#define __FUNCT__ "MatIsHermitian_233"
static PETSC_UNUSED
PetscErrorCode MatIsHermitian_233(Mat A,PetscReal tol,PetscTruth *flg)
{
  return MatIsHermitian(A,flg);
}
#define MatIsHermitian MatIsHermitian_233

#undef __FUNCT__
#define __FUNCT__ "MatSetOption_233"
static PETSC_UNUSED
PetscErrorCode MatTranspose_233(Mat mat,MatReuse reuse,Mat *B)
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
#define MatTranspose MatTranspose_233


#undef __FUNCT__
#define __FUNCT__ "MatSetBlockSize_233"
static PETSC_UNUSED
PetscErrorCode MatSetBlockSize_233(Mat mat,PetscInt bs)
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
#define MatSetBlockSize MatSetBlockSize_233

#undef __FUNCT__
#define __FUNCT__ "MatGetOwnershipRangeColumn_233"
static PETSC_UNUSED
PetscErrorCode MatGetOwnershipRangeColumn_233(Mat mat,PetscInt *m,PetscInt *n)
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
#define MatGetOwnershipRangeColumn MatGetOwnershipRangeColumn_233

#undef __FUNCT__
#define __FUNCT__ "MatGetOwnershipRangesColumn_233"
static PETSC_UNUSED
PetscErrorCode MatGetOwnershipRangesColumn_233(Mat mat,const PetscInt *ranges[])
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidType(mat,1);
  PetscValidPointer(ranges,2);
  ierr = MatPreallocated(mat);CHKERRQ(ierr);
  ierr = PetscMapGetGlobalRange(&mat->cmap,ranges);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#define MatGetOwnershipRangesColumn MatGetOwnershipRangesColumn_233

#undef __FUNCT__
#define __FUNCT__ "MatSetValuesBlocked_233"
static PETSC_UNUSED
PetscErrorCode MatSetValuesBlocked_233(Mat mat,PetscInt m,const PetscInt idxm[],PetscInt n,const PetscInt idxn[],const PetscScalar v[],InsertMode addv)
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
#define MatSetValuesBlocked MatSetValuesBlocked_233


#undef __FUNCT__
#define __FUNCT__ "MatSetValuesBlockedLocal_233"
static PETSC_UNUSED
PetscErrorCode MatSetValuesBlockedLocal_233(Mat mat,PetscInt m,const PetscInt irow[],PetscInt n,const PetscInt icol[],const PetscScalar v[],InsertMode addv)
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
#define MatSetValuesBlockedLocal MatSetValuesBlockedLocal_233

#undef __FUNCT__
#define __FUNCT__ "MatSeqBAIJSetPreallocationCSR_SeqBAIJ_233"
static PETSC_UNUSED
PetscErrorCode MatSeqBAIJSetPreallocationCSR_SeqBAIJ_233(Mat B,PetscInt bs,
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
  ierr = PetscMapSetUp(&B->rmap);CHKERRQ(ierr);
  ierr = PetscMapSetUp(&B->cmap);CHKERRQ(ierr);
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
#define MatSeqBAIJSetPreallocationCSR_SeqBAIJ MatSeqBAIJSetPreallocationCSR_SeqBAIJ_233

#undef __FUNCT__
#define __FUNCT__ "MatSeqBAIJSetPreallocationCSR_233"
static PETSC_UNUSED
PetscErrorCode MatSeqBAIJSetPreallocationCSR_233(Mat B,PetscInt bs,
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
#define MatSeqBAIJSetPreallocationCSR MatSeqBAIJSetPreallocationCSR_233

#undef __FUNCT__
#define __FUNCT__ "MatMPIBAIJSetPreallocationCSR_MPIBAIJ_233"
static PETSC_UNUSED
PetscErrorCode MatMPIBAIJSetPreallocationCSR_MPIBAIJ_233(Mat B, PetscInt bs,
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
  ierr = PetscMapSetUp(&B->rmap);CHKERRQ(ierr);
  ierr = PetscMapSetUp(&B->cmap);CHKERRQ(ierr);
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
#define MatMPIBAIJSetPreallocationCSR_MPIBAIJ MatMPIBAIJSetPreallocationCSR_MPIBAIJ_233

#undef __FUNCT__
#define __FUNCT__ "MatMPIBAIJSetPreallocationCSR_233"
static PETSC_UNUSED
PetscErrorCode MatMPIBAIJSetPreallocationCSR_233(Mat B,PetscInt bs,
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
#define MatMPIBAIJSetPreallocationCSR MatMPIBAIJSetPreallocationCSR_233

#undef __FUNCT__
#define __FUNCT__ "MatGetDiagonalBlock_233"
static PETSC_UNUSED
PetscErrorCode MatGetDiagonalBlock_233(Mat A,PetscTruth *iscopy,MatReuse reuse,Mat *a)
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
#define MatGetDiagonalBlock MatGetDiagonalBlock_233

#endif /* _PETSC_COMPAT_MAT_H */
