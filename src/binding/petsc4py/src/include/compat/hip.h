#ifndef PETSC4PY_COMPAT_HIP_H
#define PETSC4PY_COMPAT_HIP_H

#if !defined(PETSC_HAVE_HIP)

#define PetscHIPError do { \
    PetscFunctionBegin; \
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"%s() requires HIP",PETSC_FUNCTION_NAME); \
    PetscFunctionReturn(PETSC_ERR_SUP);} while (0)

#undef PetscHIPError

#endif

#endif/*PETSC4PY_COMPAT_HIP_H*/
