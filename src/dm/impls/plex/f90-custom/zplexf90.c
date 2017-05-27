#include <petsc/private/fortranimpl.h>
#include <petscdmplex.h>
#include <petsc/private/f90impl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define dmplexgetcone_                  DMPLEXGETCONE
#define dmplexrestorecone_              DMPLEXRESTORECONE
#define dmplexgetconeorientation_       DMPLEXGETCONEORIENTATION
#define dmplexrestoreconeorientation_   DMPLEXRESTORECONEORIENTATION
#define dmplexgetsupport_               DMPLEXGETSUPPORT
#define dmplexrestoresupport_           DMPLEXRESTORESUPPORT
#define dmplexgettransitiveclosure_     DMPLEXGETTRANSITIVECLOSURE
#define dmplexrestoretransitiveclosure_ DMPLEXRESTORETRANSITIVECLOSURE
#define dmplexvecgetclosure_            DMPLEXVECGETCLOSURE
#define dmplexvecrestoreclosure_        DMPLEXVECRESTORECLOSURE
#define dmplexvecsetclosure_            DMPLEXVECSETCLOSURE
#define dmplexmatsetclosure_            DMPLEXMATSETCLOSURE
#define dmplexgetjoin_                  DMPLEXGETJOIN
#define dmplexgetfulljoin_              DMPLEXGETFULLJOIN
#define dmplexrestorejoin_              DMPLEXRESTOREJOIN
#define dmplexgetmeet_                  DMPLEXGETMEET
#define dmplexgetfullmeet_              DMPLEXGETFULLMEET
#define dmplexrestoremeet_              DMPLEXRESTOREMEET
#define dmplexcreatesection_            DMPLEXCREATESECTION
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
#define dmplexgetcone_                  dmplexgetcone
#define dmplexrestorecone_              dmplexrestorecone
#define dmplexgetconeorientation_       dmplexgetconeorientation
#define dmplexrestoreconeorientation_   dmplexrestoreconeorientation
#define dmplexgetsupport_               dmplexgetsupport
#define dmplexrestoresupport_           dmplexrestoresupport
#define dmplexgettransitiveclosure_     dmplexgettransitiveclosure
#define dmplexrestoretransitiveclosure_ dmplexrestoretransitiveclosure
#define dmplexvecgetclosure_            dmplexvecgetclosure
#define dmplexvecrestoreclosure_        dmplexvecrestoreclosure
#define dmplexvecsetclosure_            dmplexvecsetclosure
#define dmplexmatsetclosure_            dmplexmatsetclosure
#define dmplexgetjoin_                  dmplexgetjoin
#define dmplexgetfulljoin_              dmplexgetfulljoin
#define dmplexrestorejoin_              dmplexrestorejoin
#define dmplexgetmeet_                  dmplexgetmeet
#define dmplexgetfullmeet_              dmplexgetfullmeet
#define dmplexrestoremeet_              dmplexrestoremeet
#define dmplexcreatesection_            dmplexcreatesection
#endif

/* Definitions of Fortran Wrapper routines */

PETSC_EXTERN void PETSC_STDCALL dmplexgetcone_(DM *dm, PetscInt *p, F90Array1d *ptr, int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  const PetscInt *v;
  PetscInt       n;

  *ierr = DMPlexGetConeSize(*dm, *p, &n);if (*ierr) return;
  *ierr = DMPlexGetCone(*dm, *p, &v);if (*ierr) return;
  *ierr = F90Array1dCreate((void*) v, PETSC_INT, 1, n, ptr PETSC_F90_2PTR_PARAM(ptrd));
}

PETSC_EXTERN void PETSC_STDCALL dmplexrestorecone_(DM *dm, PetscInt *p, F90Array1d *ptr, int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  *ierr = F90Array1dDestroy(ptr, PETSC_INT PETSC_F90_2PTR_PARAM(ptrd));if (*ierr) return;
}

PETSC_EXTERN void PETSC_STDCALL dmplexgetconeorientation_(DM *dm, PetscInt *p, F90Array1d *ptr, int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  const PetscInt *v;
  PetscInt       n;

  *ierr = DMPlexGetConeSize(*dm, *p, &n);if (*ierr) return;
  *ierr = DMPlexGetConeOrientation(*dm, *p, &v);if (*ierr) return;
  *ierr = F90Array1dCreate((void*) v, PETSC_INT, 1, n, ptr PETSC_F90_2PTR_PARAM(ptrd));
}

PETSC_EXTERN void PETSC_STDCALL dmplexrestoreconeorientation_(DM *dm, PetscInt *p, F90Array1d *ptr, int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  *ierr = F90Array1dDestroy(ptr, PETSC_INT PETSC_F90_2PTR_PARAM(ptrd));if (*ierr) return;
}

PETSC_EXTERN void PETSC_STDCALL dmplexgetsupport_(DM *dm, PetscInt *p, F90Array1d *ptr, int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  const PetscInt *v;
  PetscInt       n;

  *ierr = DMPlexGetSupportSize(*dm, *p, &n);if (*ierr) return;
  *ierr = DMPlexGetSupport(*dm, *p, &v);if (*ierr) return;
  *ierr = F90Array1dCreate((void*) v, PETSC_INT, 1, n, ptr PETSC_F90_2PTR_PARAM(ptrd));
}

PETSC_EXTERN void PETSC_STDCALL dmplexrestoresupport_(DM *dm, PetscInt *p, F90Array1d *ptr, int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  *ierr = F90Array1dDestroy(ptr, PETSC_INT PETSC_F90_2PTR_PARAM(ptrd));if (*ierr) return;
}

PETSC_EXTERN void PETSC_STDCALL dmplexgettransitiveclosure_(DM *dm, PetscInt *p, PetscBool *useCone, F90Array1d *ptr, int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscInt *v = NULL;
  PetscInt n;

  *ierr = DMPlexGetTransitiveClosure(*dm, *p, *useCone, &n, &v);if (*ierr) return;
  *ierr = F90Array1dCreate((void*) v, PETSC_INT, 1, n*2, ptr PETSC_F90_2PTR_PARAM(ptrd));
}

PETSC_EXTERN void PETSC_STDCALL dmplexrestoretransitiveclosure_(DM *dm, PetscInt *p, PetscBool *useCone, F90Array1d *ptr, int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscInt *array;

  *ierr = F90Array1dAccess(ptr, PETSC_INT, (void**) &array PETSC_F90_2PTR_PARAM(ptrd));if (*ierr) return;
  *ierr = DMPlexRestoreTransitiveClosure(*dm, *p, *useCone, NULL, &array);if (*ierr) return;
  *ierr = F90Array1dDestroy(ptr, PETSC_INT PETSC_F90_2PTR_PARAM(ptrd));if (*ierr) return;
}

PETSC_EXTERN void PETSC_STDCALL dmplexvecgetclosure_(DM *dm, PetscSection *section, Vec *x, PetscInt *point, F90Array1d *ptr, int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscScalar *v = NULL;
  PetscInt     n;

  CHKFORTRANNULLOBJECTDEREFERENCE(section);
  *ierr = DMPlexVecGetClosure(*dm, *section, *x, *point, &n, &v);if (*ierr) return;
  *ierr = F90Array1dCreate((void*) v, PETSC_SCALAR, 1, n, ptr PETSC_F90_2PTR_PARAM(ptrd));
}

PETSC_EXTERN void PETSC_STDCALL dmplexvecrestoreclosure_(DM *dm, PetscSection *section, Vec *v, PetscInt *point, F90Array1d *ptr, int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscScalar *array;

  CHKFORTRANNULLOBJECTDEREFERENCE(section);
  *ierr = F90Array1dAccess(ptr, PETSC_SCALAR, (void **) &array PETSC_F90_2PTR_PARAM(ptrd));if (*ierr) return;
  *ierr = DMPlexVecRestoreClosure(*dm, *section, *v, *point, NULL, &array);if (*ierr) return;
  *ierr = F90Array1dDestroy(ptr, PETSC_SCALAR PETSC_F90_2PTR_PARAM(ptrd));if (*ierr) return;
}

PETSC_EXTERN void PETSC_STDCALL dmplexvecsetclosure_(DM *dm, PetscSection *section, Vec *v, PetscInt *point, F90Array1d *ptr, InsertMode *mode, int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscScalar *array;

  CHKFORTRANNULLOBJECTDEREFERENCE(section);
  *ierr = F90Array1dAccess(ptr, PETSC_SCALAR, (void**) &array PETSC_F90_2PTR_PARAM(ptrd));if (*ierr) return;
  *ierr = DMPlexVecSetClosure(*dm,  *section, *v, *point, array, *mode);
}

PETSC_EXTERN void PETSC_STDCALL dmplexmatsetclosure_(DM *dm, PetscSection *section, PetscSection *globalSection, Mat *A, PetscInt *point, F90Array1d *ptr, InsertMode *mode, int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscScalar *array;

  CHKFORTRANNULLOBJECTDEREFERENCE(section);
  CHKFORTRANNULLOBJECTDEREFERENCE(globalSection);
  *ierr = F90Array1dAccess(ptr, PETSC_SCALAR, (void**) &array PETSC_F90_2PTR_PARAM(ptrd));if (*ierr) return;
  *ierr = DMPlexMatSetClosure(*dm, *section, *globalSection, *A, *point, array, *mode);
}

PETSC_EXTERN void PETSC_STDCALL dmplexgetjoin_(DM *dm, PetscInt *numPoints, F90Array1d *pptr, F90Array1d *cptr, int *ierr PETSC_F90_2PTR_PROTO(pptrd) PETSC_F90_2PTR_PROTO(cptrd))
{
  PetscInt       *points;
  const PetscInt *coveredPoints;
  PetscInt       numCoveredPoints;

  *ierr = F90Array1dAccess(pptr, PETSC_INT, (void**) &points PETSC_F90_2PTR_PARAM(pptrd));if (*ierr) return;
  *ierr = DMPlexGetJoin(*dm, *numPoints, points, &numCoveredPoints, &coveredPoints);if (*ierr) return;
  *ierr = F90Array1dCreate((void*) coveredPoints, PETSC_INT, 1, numCoveredPoints, cptr PETSC_F90_2PTR_PARAM(cptrd));
}

PETSC_EXTERN void PETSC_STDCALL dmplexgetfulljoin_(DM *dm, PetscInt *numPoints, F90Array1d *pptr, F90Array1d *cptr, int *ierr PETSC_F90_2PTR_PROTO(pptrd) PETSC_F90_2PTR_PROTO(cptrd))
{
  PetscInt       *points;
  const PetscInt *coveredPoints;
  PetscInt        numCoveredPoints;

  *ierr = F90Array1dAccess(pptr, PETSC_INT, (void**) &points PETSC_F90_2PTR_PARAM(pptrd));if (*ierr) return;
  *ierr = DMPlexGetFullJoin(*dm, *numPoints, points, &numCoveredPoints, &coveredPoints);if (*ierr) return;
  *ierr = F90Array1dCreate((void*) coveredPoints, PETSC_INT, 1, numCoveredPoints, cptr PETSC_F90_2PTR_PARAM(cptrd));
}

PETSC_EXTERN void PETSC_STDCALL dmplexrestorejoin_(DM *dm, PetscInt *numPoints, F90Array1d *pptr, F90Array1d *cptr, int *ierr PETSC_F90_2PTR_PROTO(pptrd) PETSC_F90_2PTR_PROTO(cptrd))
{
  PetscInt *coveredPoints;

  *ierr = F90Array1dAccess(cptr, PETSC_INT, (void**) &coveredPoints PETSC_F90_2PTR_PARAM(cptrd));if (*ierr) return;
  *ierr = DMPlexRestoreJoin(*dm, 0, NULL, NULL, (const PetscInt**) &coveredPoints);if (*ierr) return;
  *ierr = F90Array1dDestroy(cptr, PETSC_INT PETSC_F90_2PTR_PARAM(cptrd));if (*ierr) return;
}

PETSC_EXTERN void PETSC_STDCALL dmplexgetmeet_(DM *dm, PetscInt *numPoints, F90Array1d *pptr, F90Array1d *cptr, int *ierr PETSC_F90_2PTR_PROTO(pptrd) PETSC_F90_2PTR_PROTO(cptrd))
{
  PetscInt       *points;
  const PetscInt *coveredPoints;
  PetscInt       numCoveredPoints;

  *ierr = F90Array1dAccess(pptr, PETSC_INT, (void**) &points PETSC_F90_2PTR_PARAM(pptrd));if (*ierr) return;
  *ierr = DMPlexGetMeet(*dm, *numPoints, points, &numCoveredPoints, &coveredPoints);if (*ierr) return;
  *ierr = F90Array1dCreate((void*) coveredPoints, PETSC_INT, 1, numCoveredPoints, cptr PETSC_F90_2PTR_PARAM(cptrd));
}

PETSC_EXTERN void PETSC_STDCALL dmplexgetfullmeet_(DM *dm, PetscInt *numPoints, F90Array1d *pptr, F90Array1d *cptr, int *ierr PETSC_F90_2PTR_PROTO(pptrd) PETSC_F90_2PTR_PROTO(cptrd))
{
  PetscInt       *points;
  const PetscInt *coveredPoints;
  PetscInt       numCoveredPoints;

  *ierr = F90Array1dAccess(pptr, PETSC_INT, (void**) &points PETSC_F90_2PTR_PARAM(pptrd));if (*ierr) return;
  *ierr = DMPlexGetFullMeet(*dm, *numPoints, points, &numCoveredPoints, &coveredPoints);if (*ierr) return;
  *ierr = F90Array1dCreate((void*) coveredPoints, PETSC_INT, 1, numCoveredPoints, cptr PETSC_F90_2PTR_PARAM(cptrd));
}

PETSC_EXTERN void PETSC_STDCALL dmplexrestoremeet_(DM *dm, PetscInt *numPoints, F90Array1d *pptr, F90Array1d *cptr, int *ierr PETSC_F90_2PTR_PROTO(pptrd) PETSC_F90_2PTR_PROTO(cptrd))
{
  PetscInt *coveredPoints;

  *ierr = F90Array1dAccess(cptr, PETSC_INT, (void**) &coveredPoints PETSC_F90_2PTR_PARAM(cptrd));if (*ierr) return;
  *ierr = DMPlexRestoreMeet(*dm, 0, NULL, NULL, (const PetscInt**) &coveredPoints);if (*ierr) return;
  *ierr = F90Array1dDestroy(cptr, PETSC_INT PETSC_F90_2PTR_PARAM(cptrd));if (*ierr) return;
}

PETSC_EXTERN void PETSC_STDCALL dmplexcreatesection_(DM *dm, PetscInt *dim, PetscInt *numFields, F90Array1d *ptrC, F90Array1d *ptrD, PetscInt *numBC, F90Array1d *ptrF, F90Array1d *ptrCp, F90Array1d *ptrP, IS *perm, PetscSection *section, int *ierr PETSC_F90_2PTR_PROTO(ptrCd) PETSC_F90_2PTR_PROTO(ptrDd) PETSC_F90_2PTR_PROTO(ptrFd) PETSC_F90_2PTR_PROTO(ptrCpd) PETSC_F90_2PTR_PROTO(ptrPd))
{
  PetscInt *numComp;
  PetscInt *numDof;
  PetscInt *bcField;
  IS       *bcComps;
  IS       *bcPoints;

  CHKFORTRANNULLOBJECTDEREFERENCE(perm);
  *ierr = F90Array1dAccess(ptrC, PETSC_INT, (void**) &numComp PETSC_F90_2PTR_PARAM(ptrCd));if (*ierr) return;
  *ierr = F90Array1dAccess(ptrD, PETSC_INT, (void**) &numDof PETSC_F90_2PTR_PARAM(ptrDd));if (*ierr) return;
  *ierr = F90Array1dAccess(ptrF, PETSC_INT, (void**) &bcField PETSC_F90_2PTR_PARAM(ptrFd));if (*ierr) return;
  *ierr = F90Array1dAccess(ptrCp, PETSC_FORTRANADDR, (void**) &bcComps PETSC_F90_2PTR_PARAM(ptrCpd));if (*ierr) return;
  *ierr = F90Array1dAccess(ptrP, PETSC_FORTRANADDR, (void**) &bcPoints PETSC_F90_2PTR_PARAM(ptrPd));if (*ierr) return;
  *ierr = DMPlexCreateSection(*dm, *dim, *numFields, numComp, numDof, *numBC, bcField, bcComps, bcPoints, *perm, section);
}

