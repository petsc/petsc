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
#endif

/* Definitions of Fortran Wrapper routines */

PETSC_EXTERN void dmplexgetcone_(DM *dm, PetscInt *p, F90Array1d *ptr, int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  const PetscInt *v;
  PetscInt        n;

  *ierr = DMPlexGetConeSize(*dm, *p, &n);
  if (*ierr) return;
  *ierr = DMPlexGetCone(*dm, *p, &v);
  if (*ierr) return;
  *ierr = F90Array1dCreate((void *)v, MPIU_INT, 1, n, ptr PETSC_F90_2PTR_PARAM(ptrd));
}

PETSC_EXTERN void dmplexrestorecone_(DM *dm, PetscInt *p, F90Array1d *ptr, int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  *ierr = F90Array1dDestroy(ptr, MPIU_INT PETSC_F90_2PTR_PARAM(ptrd));
  if (*ierr) return;
}

PETSC_EXTERN void dmplexgetconeorientation_(DM *dm, PetscInt *p, F90Array1d *ptr, int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  const PetscInt *v;
  PetscInt        n;

  *ierr = DMPlexGetConeSize(*dm, *p, &n);
  if (*ierr) return;
  *ierr = DMPlexGetConeOrientation(*dm, *p, &v);
  if (*ierr) return;
  *ierr = F90Array1dCreate((void *)v, MPIU_INT, 1, n, ptr PETSC_F90_2PTR_PARAM(ptrd));
}

PETSC_EXTERN void dmplexrestoreconeorientation_(DM *dm, PetscInt *p, F90Array1d *ptr, int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  *ierr = F90Array1dDestroy(ptr, MPIU_INT PETSC_F90_2PTR_PARAM(ptrd));
  if (*ierr) return;
}

PETSC_EXTERN void dmplexgetsupport_(DM *dm, PetscInt *p, F90Array1d *ptr, int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  const PetscInt *v;
  PetscInt        n;

  *ierr = DMPlexGetSupportSize(*dm, *p, &n);
  if (*ierr) return;
  *ierr = DMPlexGetSupport(*dm, *p, &v);
  if (*ierr) return;
  *ierr = F90Array1dCreate((void *)v, MPIU_INT, 1, n, ptr PETSC_F90_2PTR_PARAM(ptrd));
}

PETSC_EXTERN void dmplexrestoresupport_(DM *dm, PetscInt *p, F90Array1d *ptr, int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  *ierr = F90Array1dDestroy(ptr, MPIU_INT PETSC_F90_2PTR_PARAM(ptrd));
  if (*ierr) return;
}

PETSC_EXTERN void dmplexgettransitiveclosure_(DM *dm, PetscInt *p, PetscBool *useCone, F90Array1d *ptr, int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscInt *v = NULL;
  PetscInt  n;

  *ierr = DMPlexGetTransitiveClosure(*dm, *p, *useCone, &n, &v);
  if (*ierr) return;
  *ierr = F90Array1dCreate((void *)v, MPIU_INT, 1, n * 2, ptr PETSC_F90_2PTR_PARAM(ptrd));
}

PETSC_EXTERN void dmplexrestoretransitiveclosure_(DM *dm, PetscInt *p, PetscBool *useCone, F90Array1d *ptr, int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscInt *array;

  *ierr = F90Array1dAccess(ptr, MPIU_INT, (void **)&array PETSC_F90_2PTR_PARAM(ptrd));
  if (*ierr) return;
  *ierr = DMPlexRestoreTransitiveClosure(*dm, *p, *useCone, NULL, &array);
  if (*ierr) return;
  *ierr = F90Array1dDestroy(ptr, MPIU_INT PETSC_F90_2PTR_PARAM(ptrd));
  if (*ierr) return;
}

PETSC_EXTERN void dmplexvecgetclosure_(DM *dm, PetscSection *section, Vec *x, PetscInt *point, F90Array1d *ptr, int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscScalar *v = NULL;
  PetscInt     n;

  *ierr = DMPlexVecGetClosure(*dm, *section, *x, *point, &n, &v);
  if (*ierr) return;
  *ierr = F90Array1dCreate((void *)v, MPIU_SCALAR, 1, n, ptr PETSC_F90_2PTR_PARAM(ptrd));
}

PETSC_EXTERN void dmplexvecrestoreclosure_(DM *dm, PetscSection *section, Vec *v, PetscInt *point, F90Array1d *ptr, int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscScalar *array;

  *ierr = F90Array1dAccess(ptr, MPIU_SCALAR, (void **)&array PETSC_F90_2PTR_PARAM(ptrd));
  if (*ierr) return;
  *ierr = DMPlexVecRestoreClosure(*dm, *section, *v, *point, NULL, &array);
  if (*ierr) return;
  *ierr = F90Array1dDestroy(ptr, MPIU_SCALAR PETSC_F90_2PTR_PARAM(ptrd));
  if (*ierr) return;
}

PETSC_EXTERN void dmplexvecsetclosure_(DM *dm, PetscSection *section, Vec *v, PetscInt *point, F90Array1d *ptr, InsertMode *mode, int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscScalar *array;

  *ierr = F90Array1dAccess(ptr, MPIU_SCALAR, (void **)&array PETSC_F90_2PTR_PARAM(ptrd));
  if (*ierr) return;
  *ierr = DMPlexVecSetClosure(*dm, *section, *v, *point, array, *mode);
}

PETSC_EXTERN void dmplexmatsetclosure_(DM *dm, PetscSection *section, PetscSection *globalSection, Mat *A, PetscInt *point, F90Array1d *ptr, InsertMode *mode, int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscScalar *array;

  *ierr = F90Array1dAccess(ptr, MPIU_SCALAR, (void **)&array PETSC_F90_2PTR_PARAM(ptrd));
  if (*ierr) return;
  *ierr = DMPlexMatSetClosure(*dm, *section, *globalSection, *A, *point, array, *mode);
}

PETSC_EXTERN void dmplexgetjoin_(DM *dm, PetscInt *numPoints, F90Array1d *pptr, F90Array1d *cptr, int *ierr PETSC_F90_2PTR_PROTO(pptrd) PETSC_F90_2PTR_PROTO(cptrd))
{
  PetscInt       *points;
  const PetscInt *coveredPoints;
  PetscInt        numCoveredPoints;

  *ierr = F90Array1dAccess(pptr, MPIU_INT, (void **)&points PETSC_F90_2PTR_PARAM(pptrd));
  if (*ierr) return;
  *ierr = DMPlexGetJoin(*dm, *numPoints, points, &numCoveredPoints, &coveredPoints);
  if (*ierr) return;
  *ierr = F90Array1dCreate((void *)coveredPoints, MPIU_INT, 1, numCoveredPoints, cptr PETSC_F90_2PTR_PARAM(cptrd));
}

PETSC_EXTERN void dmplexgetfulljoin_(DM *dm, PetscInt *numPoints, F90Array1d *pptr, F90Array1d *cptr, int *ierr PETSC_F90_2PTR_PROTO(pptrd) PETSC_F90_2PTR_PROTO(cptrd))
{
  PetscInt       *points;
  const PetscInt *coveredPoints;
  PetscInt        numCoveredPoints;

  *ierr = F90Array1dAccess(pptr, MPIU_INT, (void **)&points PETSC_F90_2PTR_PARAM(pptrd));
  if (*ierr) return;
  *ierr = DMPlexGetFullJoin(*dm, *numPoints, points, &numCoveredPoints, &coveredPoints);
  if (*ierr) return;
  *ierr = F90Array1dCreate((void *)coveredPoints, MPIU_INT, 1, numCoveredPoints, cptr PETSC_F90_2PTR_PARAM(cptrd));
}

PETSC_EXTERN void dmplexrestorejoin_(DM *dm, PetscInt *numPoints, F90Array1d *pptr, F90Array1d *cptr, int *ierr PETSC_F90_2PTR_PROTO(pptrd) PETSC_F90_2PTR_PROTO(cptrd))
{
  PetscInt *coveredPoints;

  *ierr = F90Array1dAccess(cptr, MPIU_INT, (void **)&coveredPoints PETSC_F90_2PTR_PARAM(cptrd));
  if (*ierr) return;
  *ierr = DMPlexRestoreJoin(*dm, 0, NULL, NULL, (const PetscInt **)&coveredPoints);
  if (*ierr) return;
  *ierr = F90Array1dDestroy(cptr, MPIU_INT PETSC_F90_2PTR_PARAM(cptrd));
  if (*ierr) return;
}

PETSC_EXTERN void dmplexgetmeet_(DM *dm, PetscInt *numPoints, F90Array1d *pptr, F90Array1d *cptr, int *ierr PETSC_F90_2PTR_PROTO(pptrd) PETSC_F90_2PTR_PROTO(cptrd))
{
  PetscInt       *points;
  const PetscInt *coveredPoints;
  PetscInt        numCoveredPoints;

  *ierr = F90Array1dAccess(pptr, MPIU_INT, (void **)&points PETSC_F90_2PTR_PARAM(pptrd));
  if (*ierr) return;
  *ierr = DMPlexGetMeet(*dm, *numPoints, points, &numCoveredPoints, &coveredPoints);
  if (*ierr) return;
  *ierr = F90Array1dCreate((void *)coveredPoints, MPIU_INT, 1, numCoveredPoints, cptr PETSC_F90_2PTR_PARAM(cptrd));
}

PETSC_EXTERN void dmplexgetfullmeet_(DM *dm, PetscInt *numPoints, F90Array1d *pptr, F90Array1d *cptr, int *ierr PETSC_F90_2PTR_PROTO(pptrd) PETSC_F90_2PTR_PROTO(cptrd))
{
  PetscInt       *points;
  const PetscInt *coveredPoints;
  PetscInt        numCoveredPoints;

  *ierr = F90Array1dAccess(pptr, MPIU_INT, (void **)&points PETSC_F90_2PTR_PARAM(pptrd));
  if (*ierr) return;
  *ierr = DMPlexGetFullMeet(*dm, *numPoints, points, &numCoveredPoints, &coveredPoints);
  if (*ierr) return;
  *ierr = F90Array1dCreate((void *)coveredPoints, MPIU_INT, 1, numCoveredPoints, cptr PETSC_F90_2PTR_PARAM(cptrd));
}

PETSC_EXTERN void dmplexrestoremeet_(DM *dm, PetscInt *numPoints, F90Array1d *pptr, F90Array1d *cptr, int *ierr PETSC_F90_2PTR_PROTO(pptrd) PETSC_F90_2PTR_PROTO(cptrd))
{
  PetscInt *coveredPoints;

  *ierr = F90Array1dAccess(cptr, MPIU_INT, (void **)&coveredPoints PETSC_F90_2PTR_PARAM(cptrd));
  if (*ierr) return;
  *ierr = DMPlexRestoreMeet(*dm, 0, NULL, NULL, (const PetscInt **)&coveredPoints);
  if (*ierr) return;
  *ierr = F90Array1dDestroy(cptr, MPIU_INT PETSC_F90_2PTR_PARAM(cptrd));
  if (*ierr) return;
}
