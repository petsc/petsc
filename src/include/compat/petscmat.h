#ifndef _COMPAT_PETSC_MAT_H
#define _COMPAT_PETSC_MAT_H

#include "private/matimpl.h"

#if (PETSC_VERSION_(3,1,0) || \
     PETSC_VERSION_(3,0,0))
#define MATSEQAIJPERM  MATSEQCSRPERM
#define MATMPIAIJPERM  MATMPICSRPERM
#define MATAIJPERM     MATCSRPERM
#define MATSEQAIJCRL   MATSEQCRL
#define MATMPIAIJCRL   MATMPICRL
#define MATAIJCRL      MATCRL
#define MATAIJCUDA     "aijcuda"
#define MATSEQAIJCUDA  "seqaijcuda"
#define MATMPIAIJCUDA  "mpiaijcuda"
#define MATSEQCUFFT    "seqcufft"
#define MATDD          "dd"
#define MATIM          "im"
#endif

#if (PETSC_VERSION_(3,1,0) || \
     PETSC_VERSION_(3,0,0))
#undef __FUNCT__
#define __FUNCT__ "MatZeroRows"
static PetscErrorCode MatZeroRows_Compat(Mat mat,PetscInt n,const PetscInt rows[],PetscScalar diag,Vec x,Vec b)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  if (x != PETSC_NULL)
    SETERRQ(PETSC_ERR_SUP,__FUNCT__"() not supported in this PETSc version");
  if (b != PETSC_NULL)
    SETERRQ(PETSC_ERR_SUP,__FUNCT__"() not supported in this PETSc version");
  ierr = MatZeroRows(mat,n,rows,diag);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#define MatZeroRows MatZeroRows_Compat
#undef __FUNCT__
#define __FUNCT__ "MatZeroRowsIS"
static PetscErrorCode MatZeroRowsIS_Compat(Mat mat,IS is,PetscScalar diag,Vec x,Vec b)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidHeaderSpecific(is,IS_COOKIE,2);
  if (x != PETSC_NULL)
    SETERRQ(PETSC_ERR_SUP,__FUNCT__"() not supported in this PETSc version");
  if (b != PETSC_NULL)
    SETERRQ(PETSC_ERR_SUP,__FUNCT__"() not supported in this PETSc version");
  ierr = MatZeroRowsIS(mat,is,diag);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#define MatZeroRowsIS MatZeroRowsIS_Compat
#undef __FUNCT__
#define __FUNCT__ "MatZeroRowsLocal"
static PetscErrorCode MatZeroRowsLocal_Compat(Mat mat,PetscInt n,const PetscInt rows[],PetscScalar diag,Vec x,Vec b)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  if (x != PETSC_NULL)
    SETERRQ(PETSC_ERR_SUP,__FUNCT__"() not supported in this PETSc version");
  if (b != PETSC_NULL)
    SETERRQ(PETSC_ERR_SUP,__FUNCT__"() not supported in this PETSc version");
  ierr = MatZeroRowsLocal(mat,n,rows,diag);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#define MatZeroRowsLocal MatZeroRowsLocal_Compat
#undef __FUNCT__
#define __FUNCT__ "MatZeroRowsLocalIS"
static PetscErrorCode MatZeroRowsLocalIS_Compat(Mat mat,IS is,PetscScalar diag,Vec x,Vec b)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidHeaderSpecific(is,IS_COOKIE,2);
  if (x != PETSC_NULL)
    SETERRQ(PETSC_ERR_SUP,__FUNCT__"() not supported in this PETSc version");
  if (b != PETSC_NULL)
    SETERRQ(PETSC_ERR_SUP,__FUNCT__"() not supported in this PETSc version");
  ierr = MatZeroRowsLocalIS(mat,is,diag);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#define MatZeroRowsLocalIS MatZeroRowsLocalIS_Compat
#endif

#if (PETSC_VERSION_(3,1,0) || \
     PETSC_VERSION_(3,0,0))
#undef __FUNCT__
#define __FUNCT__ "MatLoad"
static PetscErrorCode MatLoad_Compat(Mat mat,PetscViewer viewer)
{
  const MatType  type=0;
  PetscInt       m=-1,n=-1,M=-1,N=-1;
  Mat            newmat;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_COOKIE,2);
  ierr = MatGetType(mat,&type);CHKERRQ(ierr);
  ierr = MatGetSize(mat,&M,&N);CHKERRQ(ierr);
  ierr = MatGetLocalSize(mat,&m,&n);CHKERRQ(ierr);
  if (!type || (m<0 && n<0 && M<0 && N<0)) {
    if (m<0 && n<0 && M<0 && N<0) {
      ierr = MatSetSizes(mat,0,0,0,0);CHKERRQ(ierr);
      ierr = MatGetType(mat,&type);CHKERRQ(ierr);
    }
    if (!type) {
      ierr = MatSetType(mat,MATAIJ);CHKERRQ(ierr);
      ierr = MatGetType(mat,&type);CHKERRQ(ierr);
    }
  }
  ierr = MatLoad(viewer,type,&newmat);CHKERRQ(ierr);
  ierr = MatHeaderReplace(mat,newmat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#define MatLoad MatLoad_Compat
#endif

#if (PETSC_VERSION_(3,1,0) || \
     PETSC_VERSION_(3,0,0))
#define MATORDERINGNATURAL      MATORDERING_NATURAL
#define MATORDERINGND           MATORDERING_ND
#define MATORDERING1WD          MATORDERING_1WD
#define MATORDERINGRCM          MATORDERING_RCM
#define MATORDERINGQMD          MATORDERING_QMD
#define MATORDERINGROWLENGTH    MATORDERING_ROWLENGTH
#define MATORDERINGDSC_ND       MATORDERING_DSC_ND
#define MATORDERINGDSC_MMD      MATORDERING_DSC_MMD
#define MATORDERINGDSC_MDF      MATORDERING_DSC_MDF
#define MATORDERINGCONSTRAINED  MATORDERING_CONSTRAINED
#define MATORDERINGIDENTITY     MATORDERING_IDENTITY
#define MATORDERINGREVERSE      MATORDERING_REVERSE
#define MATORDERINGFLOW         MATORDERING_FLOW
#define MATORDERINGAMD          MATORDERING_AMD
#endif

#if (PETSC_VERSION_(3,1,0) || \
     PETSC_VERSION_(3,0,0))
#define MATCOLORINGNATURAL  MATCOLORING_NATURAL
#define MATCOLORINGSL	    MATCOLORING_SL
#define MATCOLORINGLF	    MATCOLORING_LF
#define MATCOLORINGID       MATCOLORING_ID
#endif

#if (PETSC_VERSION_(3,0,0))
#define MATHYPRESTRUCT  "hyprestruct"
#define MATHYPRESSTRUCT "hypresstruct"
#define MATSUBMATRIX    "submatrix"
#define MATORDERING_FLOW "flow"
#define MATORDERING_AMD  "amd"
#endif

#if (PETSC_VERSION_(3,0,0))
typedef PetscErrorCode MatNullSpaceFunction(Vec,void*);
#else
typedef PetscErrorCode MatNullSpaceFunction(MatNullSpace,Vec,void*);
#endif

#if (PETSC_VERSION_(3,0,0))
#undef __FUNCT__
#define __FUNCT__ "MatCreateSubMatrix"
static PetscErrorCode MatCreateSubMatrix_Compat(Mat A, IS r, IS c, Mat *B)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_COOKIE,1);
  PetscValidHeaderSpecific(r,IS_COOKIE,2);
  PetscValidHeaderSpecific(c,IS_COOKIE,3);
  PetscValidPointer(B,4);
  SETERRQ(PETSC_ERR_SUP,"MatCreateSubMatrix() "
	  "not available in this PETSc version");
  PetscFunctionReturn(0);
}
#define MatCreateSubMatrix MatCreateSubMatrix_Compat 
#endif

#if (PETSC_VERSION_(3,0,0))
#undef __FUNCT__
#define __FUNCT__ "MatGetDiagonalBlock"
static PetscErrorCode MatGetDiagonalBlock_Compat(Mat A,PetscTruth *iscopy,MatReuse reuse,Mat *a)
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
#define MatGetDiagonalBlock MatGetDiagonalBlock_Compat
#endif

#if (PETSC_VERSION_(3,0,0))
#undef __FUNCT__
#define __FUNCT__ "MatGetSubMatrix"
static PetscErrorCode MatGetSubMatrix_Compat(Mat mat,IS isrow,IS iscol,MatReuse cll,Mat *newmat)
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
#define MatGetSubMatrix MatGetSubMatrix_Compat
#endif

#if (PETSC_VERSION_(3,0,0))
#define MAT_KEEP_NONZERO_PATTERN MAT_KEEP_ZEROED_ROWS
#endif

#endif /* _COMPAT_PETSC_MAT_H */
