
#include <petscmat.h>
#include <../src/mat/order/order.h>
#define UF_long long long
#define UF_long_max LONG_LONG_MAX
#define UF_long_id "%lld"
#include <amd.h>

#if defined(PETSC_USE_64BIT_INDICES)
#  define amd_AMD_defaults amd_l_defaults
#  define amd_AMD_order    amd_l_order
#else
#  define amd_AMD_defaults amd_defaults
#  define amd_AMD_order    amd_order
#endif

EXTERN_C_BEGIN
/*
    MatGetOrdering_AMD - Find the Approximate Minimum Degree ordering

    This provides an interface to Tim Davis' AMD package (used by UMFPACK, CHOLMOD, MATLAB, etc).
*/
#undef __FUNCT__  
#define __FUNCT__ "MatGetOrdering_AMD"
PetscErrorCode  MatGetOrdering_AMD(Mat mat,const MatOrderingType type,IS *row,IS *col)
{
  PetscErrorCode ierr;
  PetscInt       nrow,*ia,*ja,*perm;
  int            status;
  PetscReal      val;
  double         Control[AMD_CONTROL],Info[AMD_INFO];
  PetscBool      tval,done;

  PetscFunctionBegin;
  /*
     AMD does not require that the matrix be symmetric (it does so internally,
     at least in so far as computing orderings for A+A^T.
  */
  ierr = MatGetRowIJ(mat,0,PETSC_FALSE,PETSC_TRUE,&nrow,&ia,&ja,&done);CHKERRQ(ierr);
  if (!done) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot get rows for matrix type %s",((PetscObject)mat)->type_name);

  amd_AMD_defaults(Control);
  ierr = PetscOptionsBegin(((PetscObject)mat)->comm,((PetscObject)mat)->prefix,"AMD Options","Mat");CHKERRQ(ierr);
  /*
    We have to use temporary values here because AMD always uses double, even though PetscReal may be single
  */
  val = (PetscReal)Control[AMD_DENSE];
  ierr = PetscOptionsReal("-mat_ordering_amd_dense","threshold for \"dense\" rows/columns","None",val,&val,PETSC_NULL);CHKERRQ(ierr);
  Control[AMD_DENSE] = (double)val;

  tval = (PetscBool)Control[AMD_AGGRESSIVE];
  ierr = PetscOptionsBool("-mat_ordering_amd_aggressive","use aggressive absorption","None",tval,&tval,PETSC_NULL);CHKERRQ(ierr);
  Control[AMD_AGGRESSIVE] = (double)tval;
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  ierr = PetscMalloc(nrow*sizeof(PetscInt),&perm);CHKERRQ(ierr);
  status = amd_AMD_order(nrow,ia,ja,perm,Control,Info);
  switch (status) {
    case AMD_OK: break;
    case AMD_OK_BUT_JUMBLED:
      /* The result is fine, but PETSc matrices are supposed to satisfy stricter preconditions, so PETSc considers a
      * matrix that triggers this error condition to be invalid.
      */
      SETERRQ(((PetscObject)mat)->comm,PETSC_ERR_PLIB,"According to AMD, the matrix has unsorted and/or duplicate row indices");
    case AMD_INVALID:
      amd_info(Info);
      SETERRQ(((PetscObject)mat)->comm,PETSC_ERR_PLIB,"According to AMD, the matrix is invalid");
    case AMD_OUT_OF_MEMORY:
      SETERRQ(((PetscObject)mat)->comm,PETSC_ERR_MEM,"AMD could not compute ordering");
    default:
      SETERRQ(((PetscObject)mat)->comm,PETSC_ERR_LIB,"Unexpected return value");
  }
  ierr = MatRestoreRowIJ(mat,0,PETSC_FALSE,PETSC_TRUE,&nrow,&ia,&ja,&done);CHKERRQ(ierr);

  ierr = ISCreateGeneral(PETSC_COMM_SELF,nrow,perm,PETSC_COPY_VALUES,row);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF,nrow,perm,PETSC_OWN_POINTER,col);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END
