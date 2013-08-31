#include <petsc-private/fortranimpl.h>
#include <petscdmplex.h>
#include <../src/sys/f90-src/f90impl.h>

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
#define dmplexcomputecellgeometry_      DMPLEXCOMPUTECELLGEOMETRY
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
#define dmplexcomputecellgeometry_      dmplexcomputecellgeometry
#endif

/* Definitions of Fortran Wrapper routines */

PETSC_EXTERN void PETSC_STDCALL dmplexgetcone_(DM *dm, PetscInt *p, F90Array1d *ptr, int *__ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  const PetscInt *v;
  PetscInt       n;

  *__ierr = DMPlexGetConeSize(*dm, *p, &n);if (*__ierr) return;
  *__ierr = DMPlexGetCone(*dm, *p, &v);if (*__ierr) return;
  *__ierr = F90Array1dCreate((void*) v, PETSC_INT, 1, n, ptr PETSC_F90_2PTR_PARAM(ptrd));
}

PETSC_EXTERN void PETSC_STDCALL dmplexrestorecone_(DM *dm, PetscInt *p, F90Array1d *ptr, int *__ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  *__ierr = F90Array1dDestroy(ptr, PETSC_INT PETSC_F90_2PTR_PARAM(ptrd));if (*__ierr) return;
}

PETSC_EXTERN void PETSC_STDCALL dmplexgetconeorientation_(DM *dm, PetscInt *p, F90Array1d *ptr, int *__ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  const PetscInt *v;
  PetscInt       n;

  *__ierr = DMPlexGetConeSize(*dm, *p, &n);if (*__ierr) return;
  *__ierr = DMPlexGetConeOrientation(*dm, *p, &v);if (*__ierr) return;
  *__ierr = F90Array1dCreate((void*) v, PETSC_INT, 1, n, ptr PETSC_F90_2PTR_PARAM(ptrd));
}

PETSC_EXTERN void PETSC_STDCALL dmplexrestoreconeorientation_(DM *dm, PetscInt *p, F90Array1d *ptr, int *__ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  *__ierr = F90Array1dDestroy(ptr, PETSC_INT PETSC_F90_2PTR_PARAM(ptrd));if (*__ierr) return;
}

PETSC_EXTERN void PETSC_STDCALL dmplexgetsupport_(DM *dm, PetscInt *p, F90Array1d *ptr, int *__ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  const PetscInt *v;
  PetscInt       n;

  *__ierr = DMPlexGetSupportSize(*dm, *p, &n);if (*__ierr) return;
  *__ierr = DMPlexGetSupport(*dm, *p, &v);if (*__ierr) return;
  *__ierr = F90Array1dCreate((void*) v, PETSC_INT, 1, n, ptr PETSC_F90_2PTR_PARAM(ptrd));
}

PETSC_EXTERN void PETSC_STDCALL dmplexrestoresupport_(DM *dm, PetscInt *p, F90Array1d *ptr, int *__ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  *__ierr = F90Array1dDestroy(ptr, PETSC_INT PETSC_F90_2PTR_PARAM(ptrd));if (*__ierr) return;
}

PETSC_EXTERN void PETSC_STDCALL dmplexgettransitiveclosure_(DM *dm, PetscInt *p, PetscBool *useCone, F90Array1d *ptr, int *__ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscInt *v = NULL;
  PetscInt n;

  *__ierr = DMPlexGetTransitiveClosure(*dm, *p, *useCone, &n, &v);if (*__ierr) return;
  *__ierr = F90Array1dCreate((void*) v, PETSC_INT, 1, n*2, ptr PETSC_F90_2PTR_PARAM(ptrd));
}

PETSC_EXTERN void PETSC_STDCALL dmplexrestoretransitiveclosure_(DM *dm, PetscInt *p, PetscBool *useCone, F90Array1d *ptr, int *__ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscInt *array;

  *__ierr = F90Array1dAccess(ptr, PETSC_INT, (void**) &array PETSC_F90_2PTR_PARAM(ptrd));if (*__ierr) return;
  *__ierr = DMPlexRestoreTransitiveClosure(*dm, *p, *useCone, NULL, &array);if (*__ierr) return;
  *__ierr = F90Array1dDestroy(ptr, PETSC_INT PETSC_F90_2PTR_PARAM(ptrd));if (*__ierr) return;
}

PETSC_EXTERN void PETSC_STDCALL dmplexvecgetclosure_(DM *dm, PetscSection *section, Vec *x, PetscInt *point, F90Array1d *ptr, int *__ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscScalar *v = NULL;
  PetscInt     n;

  *__ierr = DMPlexVecGetClosure(*dm, *section, *x, *point, &n, &v);if (*__ierr) return;
  *__ierr = F90Array1dCreate((void*) v, PETSC_SCALAR, 1, n, ptr PETSC_F90_2PTR_PARAM(ptrd));
}

PETSC_EXTERN void PETSC_STDCALL dmplexvecrestoreclosure_(DM *dm, PetscSection *section, Vec *v, PetscInt *point, F90Array1d *ptr, int *__ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscScalar *array;

  *__ierr = F90Array1dAccess(ptr, PETSC_SCALAR, (void **) &array PETSC_F90_2PTR_PARAM(ptrd));if (*__ierr) return;
  *__ierr = DMPlexVecRestoreClosure(*dm, *section, *v, *point, NULL, &array);if (*__ierr) return;
  *__ierr = F90Array1dDestroy(ptr, PETSC_SCALAR PETSC_F90_2PTR_PARAM(ptrd));if (*__ierr) return;
}

PETSC_EXTERN void PETSC_STDCALL dmplexvecsetclosure_(DM *dm, PetscSection *section, Vec *v, PetscInt *point, F90Array1d *ptr, InsertMode *mode, int *__ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscScalar *array;

  *__ierr = F90Array1dAccess(ptr, PETSC_SCALAR, (void**) &array PETSC_F90_2PTR_PARAM(ptrd));if (*__ierr) return;
  *__ierr = DMPlexVecSetClosure(*dm, *section, *v, *point, array, *mode);
}

PETSC_EXTERN void PETSC_STDCALL dmplexmatsetclosure_(DM *dm, PetscSection *section, PetscSection *globalSection, Mat *A, PetscInt *point, F90Array1d *ptr, InsertMode *mode, int *__ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscScalar *array;

  *__ierr = F90Array1dAccess(ptr, PETSC_SCALAR, (void**) &array PETSC_F90_2PTR_PARAM(ptrd));if (*__ierr) return;
  *__ierr = DMPlexMatSetClosure(*dm, *section, *globalSection, *A, *point, array, *mode);
}

PETSC_EXTERN void PETSC_STDCALL dmplexgetjoin_(DM *dm, PetscInt *numPoints, F90Array1d *pptr, F90Array1d *cptr, int *__ierr PETSC_F90_2PTR_PROTO(pptrd) PETSC_F90_2PTR_PROTO(cptrd))
{
  PetscInt       *points;
  const PetscInt *coveredPoints;
  PetscInt       numCoveredPoints;

  *__ierr = F90Array1dAccess(pptr, PETSC_INT, (void**) &points PETSC_F90_2PTR_PARAM(pptrd));if (*__ierr) return;
  *__ierr = DMPlexGetJoin(*dm, *numPoints, points, &numCoveredPoints, &coveredPoints);if (*__ierr) return;
  *__ierr = F90Array1dCreate((void*) coveredPoints, PETSC_INT, 1, numCoveredPoints, cptr PETSC_F90_2PTR_PARAM(cptrd));
}

PETSC_EXTERN void PETSC_STDCALL dmplexgetfulljoin_(DM *dm, PetscInt *numPoints, F90Array1d *pptr, F90Array1d *cptr, int *__ierr PETSC_F90_2PTR_PROTO(pptrd) PETSC_F90_2PTR_PROTO(cptrd))
{
  PetscInt       *points;
  const PetscInt *coveredPoints;
  PetscInt        numCoveredPoints;

  *__ierr = F90Array1dAccess(pptr, PETSC_INT, (void**) &points PETSC_F90_2PTR_PARAM(pptrd));if (*__ierr) return;
  *__ierr = DMPlexGetFullJoin(*dm, *numPoints, points, &numCoveredPoints, &coveredPoints);if (*__ierr) return;
  *__ierr = F90Array1dCreate((void*) coveredPoints, PETSC_INT, 1, numCoveredPoints, cptr PETSC_F90_2PTR_PARAM(cptrd));
}

PETSC_EXTERN void PETSC_STDCALL dmplexrestorejoin_(DM *dm, PetscInt *numPoints, F90Array1d *pptr, F90Array1d *cptr, int *__ierr PETSC_F90_2PTR_PROTO(pptrd) PETSC_F90_2PTR_PROTO(cptrd))
{
  PetscInt *coveredPoints;

  *__ierr = F90Array1dAccess(cptr, PETSC_INT, (void**) &coveredPoints PETSC_F90_2PTR_PARAM(cptrd));if (*__ierr) return;
  *__ierr = DMPlexRestoreJoin(*dm, 0, NULL, NULL, (const PetscInt**) &coveredPoints);if (*__ierr) return;
  *__ierr = F90Array1dDestroy(cptr, PETSC_INT PETSC_F90_2PTR_PARAM(cptrd));if (*__ierr) return;
}

PETSC_EXTERN void PETSC_STDCALL dmplexgetmeet_(DM *dm, PetscInt *numPoints, F90Array1d *pptr, F90Array1d *cptr, int *__ierr PETSC_F90_2PTR_PROTO(pptrd) PETSC_F90_2PTR_PROTO(cptrd))
{
  PetscInt       *points;
  const PetscInt *coveredPoints;
  PetscInt       numCoveredPoints;

  *__ierr = F90Array1dAccess(pptr, PETSC_INT, (void**) &points PETSC_F90_2PTR_PARAM(pptrd));if (*__ierr) return;
  *__ierr = DMPlexGetMeet(*dm, *numPoints, points, &numCoveredPoints, &coveredPoints);if (*__ierr) return;
  *__ierr = F90Array1dCreate((void*) coveredPoints, PETSC_INT, 1, numCoveredPoints, cptr PETSC_F90_2PTR_PARAM(cptrd));
}

PETSC_EXTERN void PETSC_STDCALL dmplexgetfullmeet_(DM *dm, PetscInt *numPoints, F90Array1d *pptr, F90Array1d *cptr, int *__ierr PETSC_F90_2PTR_PROTO(pptrd) PETSC_F90_2PTR_PROTO(cptrd))
{
  PetscInt       *points;
  const PetscInt *coveredPoints;
  PetscInt       numCoveredPoints;

  *__ierr = F90Array1dAccess(pptr, PETSC_INT, (void**) &points PETSC_F90_2PTR_PARAM(pptrd));if (*__ierr) return;
  *__ierr = DMPlexGetFullMeet(*dm, *numPoints, points, &numCoveredPoints, &coveredPoints);if (*__ierr) return;
  *__ierr = F90Array1dCreate((void*) coveredPoints, PETSC_INT, 1, numCoveredPoints, cptr PETSC_F90_2PTR_PARAM(cptrd));
}

PETSC_EXTERN void PETSC_STDCALL dmplexrestoremeet_(DM *dm, PetscInt *numPoints, F90Array1d *pptr, F90Array1d *cptr, int *__ierr PETSC_F90_2PTR_PROTO(pptrd) PETSC_F90_2PTR_PROTO(cptrd))
{
  PetscInt *coveredPoints;

  *__ierr = F90Array1dAccess(cptr, PETSC_INT, (void**) &coveredPoints PETSC_F90_2PTR_PARAM(cptrd));if (*__ierr) return;
  *__ierr = DMPlexRestoreMeet(*dm, 0, NULL, NULL, (const PetscInt**) &coveredPoints);if (*__ierr) return;
  *__ierr = F90Array1dDestroy(cptr, PETSC_INT PETSC_F90_2PTR_PARAM(cptrd));if (*__ierr) return;
}

PETSC_EXTERN void PETSC_STDCALL dmplexcreatesection_(DM *dm, PetscInt *dim, PetscInt *numFields, F90Array1d *ptrC, F90Array1d *ptrD, PetscInt *numBC, F90Array1d *ptrF, F90Array1d *ptrP, PetscSection *section, int *__ierr PETSC_F90_2PTR_PROTO(ptrCd) PETSC_F90_2PTR_PROTO(ptrDd) PETSC_F90_2PTR_PROTO(ptrFd) PETSC_F90_2PTR_PROTO(ptrPd))
{
  PetscInt *numComp;
  PetscInt *numDof;
  PetscInt *bcField;
  IS       *bcPoints;

  *__ierr = F90Array1dAccess(ptrC, PETSC_INT, (void**) &numComp PETSC_F90_2PTR_PARAM(ptrCd));if (*__ierr) return;
  *__ierr = F90Array1dAccess(ptrD, PETSC_INT, (void**) &numDof PETSC_F90_2PTR_PARAM(ptrDd));if (*__ierr) return;
  *__ierr = F90Array1dAccess(ptrF, PETSC_INT, (void**) &bcField PETSC_F90_2PTR_PARAM(ptrFd));if (*__ierr) return;
  *__ierr = F90Array1dAccess(ptrP, PETSC_FORTRANADDR, (void**) &bcPoints PETSC_F90_2PTR_PARAM(ptrPd));if (*__ierr) return;
  *__ierr = DMPlexCreateSection(*dm, *dim, *numFields, numComp, numDof, *numBC, bcField, bcPoints, section);
}

PETSC_EXTERN void PETSC_STDCALL dmplexcomputecellgeometry_(DM *dm, PetscInt *cell, F90Array1d *ptrV, F90Array1d *ptrJ, F90Array1d *ptrIJ, PetscReal *detJ, int *__ierr PETSC_F90_2PTR_PROTO(ptrVd) PETSC_F90_2PTR_PROTO(ptrJd) PETSC_F90_2PTR_PROTO(ptrIJd))
{
  PetscReal *v0;
  PetscReal *J;
  PetscReal *invJ;

  *__ierr = F90Array1dAccess(ptrV,  PETSC_REAL, (void**) &v0 PETSC_F90_2PTR_PARAM(ptrVd));if (*__ierr) return;
  *__ierr = F90Array1dAccess(ptrJ,  PETSC_REAL, (void**) &J PETSC_F90_2PTR_PARAM(ptrJd));if (*__ierr) return;
  *__ierr = F90Array1dAccess(ptrIJ, PETSC_REAL, (void**) &invJ PETSC_F90_2PTR_PARAM(ptrIJd));if (*__ierr) return;
  *__ierr = DMPlexComputeCellGeometry(*dm, *cell, v0, J, invJ, detJ);
}

