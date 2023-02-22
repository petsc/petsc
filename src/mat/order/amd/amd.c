
#include <petscmat.h>
#include <petsc/private/matorderimpl.h>
#include <amd.h>

#if defined(PETSC_USE_64BIT_INDICES)
  #define amd_AMD_defaults amd_l_defaults
  /* the type casts are needed because PetscInt is long long while SuiteSparse_long is long and compilers warn even when they are identical */
  #define amd_AMD_order(a, b, c, d, e, f) amd_l_order((SuiteSparse_long)a, (SuiteSparse_long *)b, (SuiteSparse_long *)c, (SuiteSparse_long *)d, e, f)
#else
  #define amd_AMD_defaults amd_defaults
  #define amd_AMD_order    amd_order
#endif

/*
    MatGetOrdering_AMD - Find the Approximate Minimum Degree ordering

    This provides an interface to Tim Davis' AMD package (used by UMFPACK, CHOLMOD, MATLAB, etc).
*/
PETSC_INTERN PetscErrorCode MatGetOrdering_AMD(Mat mat, MatOrderingType type, IS *row, IS *col)
{
  PetscInt        nrow, *perm;
  const PetscInt *ia, *ja;
  int             status;
  PetscReal       val;
  double          Control[AMD_CONTROL], Info[AMD_INFO];
  PetscBool       tval, done;

  PetscFunctionBegin;
  /*
     AMD does not require that the matrix be symmetric (it does so internally,
     at least in so far as computing orderings for A+A^T.
  */
  PetscCall(MatGetRowIJ(mat, 0, PETSC_FALSE, PETSC_TRUE, &nrow, &ia, &ja, &done));
  PetscCheck(done, PETSC_COMM_SELF, PETSC_ERR_SUP, "Cannot get rows for matrix type %s", ((PetscObject)mat)->type_name);

  amd_AMD_defaults(Control);
  PetscOptionsBegin(PetscObjectComm((PetscObject)mat), ((PetscObject)mat)->prefix, "AMD Options", "Mat");
  /*
    We have to use temporary values here because AMD always uses double, even though PetscReal may be single
  */
  val = (PetscReal)Control[AMD_DENSE];
  PetscCall(PetscOptionsReal("-mat_ordering_amd_dense", "threshold for \"dense\" rows/columns", "None", val, &val, NULL));
  Control[AMD_DENSE] = (double)val;

  tval = (PetscBool)Control[AMD_AGGRESSIVE];
  PetscCall(PetscOptionsBool("-mat_ordering_amd_aggressive", "use aggressive absorption", "None", tval, &tval, NULL));
  Control[AMD_AGGRESSIVE] = (double)tval;

  PetscOptionsEnd();

  PetscCall(PetscMalloc1(nrow, &perm));
  status = amd_AMD_order(nrow, ia, ja, perm, Control, Info);
  switch (status) {
  case AMD_OK:
    break;
  case AMD_OK_BUT_JUMBLED:
    /* The result is fine, but PETSc matrices are supposed to satisfy stricter preconditions, so PETSc considers a
    * matrix that triggers this error condition to be invalid.
    */
    SETERRQ(PetscObjectComm((PetscObject)mat), PETSC_ERR_PLIB, "According to AMD, the matrix has unsorted and/or duplicate row indices");
  case AMD_INVALID:
    amd_info(Info);
    SETERRQ(PetscObjectComm((PetscObject)mat), PETSC_ERR_PLIB, "According to AMD, the matrix is invalid");
  case AMD_OUT_OF_MEMORY:
    SETERRQ(PetscObjectComm((PetscObject)mat), PETSC_ERR_MEM, "AMD could not compute ordering");
  default:
    SETERRQ(PetscObjectComm((PetscObject)mat), PETSC_ERR_LIB, "Unexpected return value");
  }
  PetscCall(MatRestoreRowIJ(mat, 0, PETSC_FALSE, PETSC_TRUE, NULL, &ia, &ja, &done));

  PetscCall(ISCreateGeneral(PETSC_COMM_SELF, nrow, perm, PETSC_COPY_VALUES, row));
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF, nrow, perm, PETSC_OWN_POINTER, col));
  PetscFunctionReturn(PETSC_SUCCESS);
}
