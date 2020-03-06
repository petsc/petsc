#include <petsc/private/fortranimpl.h>
#include <petscdmplex.h>
#include <petsc/private/f90impl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define dmplexcreatesection_            DMPLEXCREATESECTION
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
#define dmplexcreatesection_            dmplexcreatesection
#endif

PETSC_EXTERN void dmplexcreatesection_(DM *dm, F90Array1d *ptrL, F90Array1d *ptrC, F90Array1d *ptrD, PetscInt *numBC, F90Array1d *ptrF, F90Array1d *ptrCp, F90Array1d *ptrP, IS *perm, PetscSection *section, int *ierr PETSC_F90_2PTR_PROTO(ptrLd) PETSC_F90_2PTR_PROTO(ptrCd) PETSC_F90_2PTR_PROTO(ptrDd) PETSC_F90_2PTR_PROTO(ptrFd) PETSC_F90_2PTR_PROTO(ptrCpd) PETSC_F90_2PTR_PROTO(ptrPd))
{
  DMLabel  *labels = NULL;
  PetscInt *numComp;
  PetscInt *numDof;
  PetscInt *bcField;
  IS       *bcComps;
  IS       *bcPoints;

  *ierr = F90Array1dAccess(ptrL, MPIU_FORTRANADDR, (void**) &labels  PETSC_F90_2PTR_PARAM(ptrLd));if (*ierr) return;
  *ierr = F90Array1dAccess(ptrC, MPIU_INT, (void**) &numComp PETSC_F90_2PTR_PARAM(ptrCd));if (*ierr) return;
  *ierr = F90Array1dAccess(ptrD, MPIU_INT, (void**) &numDof  PETSC_F90_2PTR_PARAM(ptrDd));if (*ierr) return;
  *ierr = F90Array1dAccess(ptrF, MPIU_INT, (void**) &bcField PETSC_F90_2PTR_PARAM(ptrFd));if (*ierr) return;
  *ierr = F90Array1dAccess(ptrCp, MPIU_FORTRANADDR, (void**) &bcComps PETSC_F90_2PTR_PARAM(ptrCpd));if (*ierr) return;
  *ierr = F90Array1dAccess(ptrP, MPIU_FORTRANADDR,  (void**) &bcPoints PETSC_F90_2PTR_PARAM(ptrPd));if (*ierr) return;
  *ierr = DMPlexCreateSection(*dm, labels, numComp, numDof, *numBC, bcField, bcComps, bcPoints, *perm, section);
}
