#if !defined(PETSC4PY_COMPAT_HTOOL_H)
#define PETSC4PY_COMPAT_HTOOL_H

#if !defined(PETSC_HAVE_HTOOL)

typedef PetscErrorCode MatHtoolKernelFn(PetscInt, PetscInt, PetscInt, const PetscInt *, const PetscInt *, PetscScalar *, void *);
typedef MatHtoolKernelFn *MatHtoolKernel;

#define PetscMatHtoolError do { \
    PetscFunctionBegin; \
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "%s() requires Htool", PETSC_FUNCTION_NAME); \
    PetscFunctionReturn(PETSC_ERR_SUP);} while (0)

PetscErrorCode MatCreateHtoolFromKernel(PETSC_UNUSED MPI_Comm a, PETSC_UNUSED PetscInt b, PETSC_UNUSED PetscInt c, PETSC_UNUSED PetscInt d, PETSC_UNUSED PetscInt e, PETSC_UNUSED PetscInt f, PETSC_UNUSED const PetscReal g[], PETSC_UNUSED const PetscReal h[], PETSC_UNUSED MatHtoolKernelFn *i, PETSC_UNUSED void *j, PETSC_UNUSED Mat *k) {PetscMatHtoolError;}
PetscErrorCode MatHtoolGetPermutationSource(PETSC_UNUSED Mat a, PETSC_UNUSED IS *b) {PetscMatHtoolError;}
PetscErrorCode MatHtoolGetPermutationTarget(PETSC_UNUSED Mat a, PETSC_UNUSED IS *b) {PetscMatHtoolError;}
PetscErrorCode MatHtoolUsePermutation(PETSC_UNUSED Mat a, PETSC_UNUSED PetscBool b) {PetscMatHtoolError;}
PetscErrorCode MatHtoolUseRecompression(PETSC_UNUSED Mat a, PETSC_UNUSED PetscBool b) {PetscMatHtoolError;}

#undef PetscMatHtoolError

#endif /* !PETSC_HAVE_HTOOL */

#endif /* PETSC4PY_COMPAT_HTOOL_H */
