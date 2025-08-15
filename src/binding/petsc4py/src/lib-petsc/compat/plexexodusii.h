#ifndef PETSC4PY_COMPAT_PLEXEXODUSII_H
#define PETSC4PY_COMPAT_PLEXEXODUSII_H

#if !defined(PETSC_HAVE_EXODUSII)

#define PetscPlexExodusIIError do { \
    PetscFunctionBegin; \
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"%s() requires ExodusII",PETSC_FUNCTION_NAME); \
    PetscFunctionReturn(PETSC_ERR_SUP);} while (0)

PetscErrorCode DMPlexCreateExodus(PETSC_UNUSED MPI_Comm comm, PETSC_UNUSED int n, PETSC_UNUSED PetscBool flg, PETSC_UNUSED DM *dm){PetscPlexExodusIIError;}

#undef PetscPlexExodusIIError

#endif

#endif/*PETSC4PY_COMPAT_PLEXEXODUSII_H*/
