#include <petsc/private/fortranimpl.h>
#include <petscdmswarm.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define dmswarmcreateglobalvectorfromfield_  DMSWARMCREATEGLOBALVECTORFROMFIELD
  #define dmswarmdestroyglobalvectorfromfield_ DMSWARMDESTROYGLOBALVECTORFROMFIELD
  #define dmswarmregisterpetscdatatypefield_   DMSWARMREGISTERPETSCDATATYPEFIELD
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
  #define dmswarmcreateglobalvectorfromfield_  dmswarmcreateglobalvectorfromfield
  #define dmswarmdestroyglobalvectorfromfield_ dmswarmdestroyglobalvectorfromfield
  #define dmswarmregisterpetscdatatypefield_   dmswarmregisterpetscdatatypefield
#endif

/* Definitions of Fortran Wrapper routines */

PETSC_EXTERN void dmswarmcreateglobalvectorfromfield_(DM *dm, char *name, Vec *vec, int *ierr, PETSC_FORTRAN_CHARLEN_T lenN)
{
  char *fieldname;

  FIXCHAR(name, lenN, fieldname);
  *ierr = DMSwarmCreateGlobalVectorFromField(*dm, fieldname, vec);
  FREECHAR(name, fieldname);
}

PETSC_EXTERN void dmswarmdestroyglobalvectorfromfield_(DM *dm, char *name, Vec *vec, int *ierr, PETSC_FORTRAN_CHARLEN_T lenN)
{
  char *fieldname;

  FIXCHAR(name, lenN, fieldname);
  *ierr = DMSwarmDestroyGlobalVectorFromField(*dm, fieldname, vec);
  FREECHAR(name, fieldname);
}

PETSC_EXTERN void dmswarmregisterpetscdatatypefield_(DM *dm, char *name, PetscInt *blocksize, PetscDataType *type, int *ierr, PETSC_FORTRAN_CHARLEN_T lenN)
{
  char *fieldname;

  FIXCHAR(name, lenN, fieldname);
  *ierr = DMSwarmRegisterPetscDatatypeField(*dm, fieldname, *blocksize, *type);
  FREECHAR(name, fieldname);
}
