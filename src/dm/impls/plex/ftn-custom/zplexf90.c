#include <petsc/private/ftnimpl.h>
#include <petscdmplex.h>

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
  #define dmplexgetclosureindices_        DMPLEXGETCLOSUREINDICES
  #define dmplexrestoreclosureindices_    DMPLEXRESTORECLOSUREINDICES
  #define dmplexgetjoin_                  DMPLEXGETJOIN
  #define dmplexgetfulljoin_              DMPLEXGETFULLJOIN
  #define dmplexrestorejoin_              DMPLEXRESTOREJOIN
  #define dmplexgetmeet_                  DMPLEXGETMEET
  #define dmplexgetfullmeet_              DMPLEXGETFULLMEET
  #define dmplexrestoremeet_              DMPLEXRESTOREMEET
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
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
  #define dmplexgetclosureindices_        dmplexgetclosureindices
  #define dmplexrestoreclosureindices_    dmplexrestoreclosureindices
  #define dmplexgetjoin_                  dmplexgetjoin
  #define dmplexgetfulljoin_              dmplexgetfulljoin
  #define dmplexrestorejoin_              dmplexrestorejoin
  #define dmplexgetmeet_                  dmplexgetmeet
  #define dmplexgetfullmeet_              dmplexgetfullmeet
  #define dmplexrestoremeet_              dmplexrestoremeet
#endif

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

PETSC_EXTERN void dmplexgettransitiveclosure_(DM *dm, PetscInt *p, PetscBool *useCone, PetscInt *N, F90Array1d *ptr, int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscInt *v = NULL;
  PetscInt  n;

  CHKFORTRANNULL(N);
  *ierr = DMPlexGetTransitiveClosure(*dm, *p, *useCone, &n, &v);
  if (*ierr) return;
  *ierr = F90Array1dCreate((void *)v, MPIU_INT, 1, n * 2, ptr PETSC_F90_2PTR_PARAM(ptrd));
  if (N) *N = n;
}

PETSC_EXTERN void dmplexrestoretransitiveclosure_(DM *dm, PetscInt *p, PetscBool *useCone, PetscInt *N, F90Array1d *ptr, int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscInt *array;

  *ierr = F90Array1dAccess(ptr, MPIU_INT, (void **)&array PETSC_F90_2PTR_PARAM(ptrd));
  if (*ierr) return;
  *ierr = DMPlexRestoreTransitiveClosure(*dm, *p, *useCone, NULL, &array);
  if (*ierr) return;
  *ierr = F90Array1dDestroy(ptr, MPIU_INT PETSC_F90_2PTR_PARAM(ptrd));
  if (*ierr) return;
}

PETSC_EXTERN void dmplexvecgetclosure_(DM *dm, PetscSection *section, Vec *x, PetscInt *point, PetscInt *N, F90Array1d *ptr, int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscScalar *v = NULL;
  PetscInt     n;

  CHKFORTRANNULL(N);
  *ierr = DMPlexVecGetClosure(*dm, *section, *x, *point, &n, &v);
  if (*ierr) return;
  *ierr = F90Array1dCreate((void *)v, MPIU_SCALAR, 1, n, ptr PETSC_F90_2PTR_PARAM(ptrd));
  if (N) *N = n;
}

PETSC_EXTERN void dmplexvecrestoreclosure_(DM *dm, PetscSection *section, Vec *v, PetscInt *point, PetscInt *N, F90Array1d *ptr, int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscScalar *array;

  *ierr = F90Array1dAccess(ptr, MPIU_SCALAR, (void **)&array PETSC_F90_2PTR_PARAM(ptrd));
  if (*ierr) return;
  *ierr = DMPlexVecRestoreClosure(*dm, *section, *v, *point, NULL, &array);
  if (*ierr) return;
  *ierr = F90Array1dDestroy(ptr, MPIU_SCALAR PETSC_F90_2PTR_PARAM(ptrd));
  if (*ierr) return;
}

PETSC_EXTERN void dmplexgetclosureindices_(DM *dm, PetscSection *section, PetscSection *idxSection, PetscInt *point, PetscBool *useClPerm, PetscInt *numIndices, F90Array1d *idxPtr, PetscInt *outOffsets, F90Array1d *valPtr, int *ierr PETSC_F90_2PTR_PROTO(idxPtrd) PETSC_F90_2PTR_PROTO(valPtrd))
{
  PetscInt    *indices;
  PetscScalar *values;

  CHKFORTRANNULL(outOffsets);
  if (FORTRANNULLSCALARPOINTER(valPtr)) *ierr = DMPlexGetClosureIndices(*dm, *section, *idxSection, *point, *useClPerm, numIndices, &indices, outOffsets, NULL);
  else *ierr = DMPlexGetClosureIndices(*dm, *section, *idxSection, *point, *useClPerm, numIndices, &indices, outOffsets, &values);
  if (*ierr) return;
  *ierr = F90Array1dCreate((void *)indices, MPIU_INT, 1, *numIndices, idxPtr PETSC_F90_2PTR_PARAM(idxPtrd));
  if (*ierr) return;
  if (FORTRANNULLSCALARPOINTER(valPtr)) *ierr = F90Array1dCreate((void *)values, MPIU_SCALAR, 1, *numIndices, valPtr PETSC_F90_2PTR_PARAM(valPtrd));
}

PETSC_EXTERN void dmplexrestoreclosureindices_(DM *dm, PetscSection *section, PetscSection *idxSection, PetscInt *point, PetscBool *useClPerm, PetscInt *numIndices, F90Array1d *idxPtr, PetscInt *outOffsets, F90Array1d *valPtr, int *ierr PETSC_F90_2PTR_PROTO(idxPtrd) PETSC_F90_2PTR_PROTO(valPtrd))
{
  PetscInt    *indices;
  PetscScalar *values = NULL;

  CHKFORTRANNULL(outOffsets);
  *ierr = F90Array1dAccess(idxPtr, MPIU_INT, (void **)&indices PETSC_F90_2PTR_PARAM(idxPtrd));
  if (*ierr) return;
  if (!FORTRANNULLSCALARPOINTER(valPtr)) {
    *ierr = F90Array1dAccess(valPtr, MPIU_SCALAR, (void **)&values PETSC_F90_2PTR_PARAM(valPtrd));
    if (*ierr) return;
    *ierr = DMPlexRestoreClosureIndices(*dm, *section, *idxSection, *point, *useClPerm, numIndices, &indices, outOffsets, &values);
  } else *ierr = DMPlexRestoreClosureIndices(*dm, *section, *idxSection, *point, *useClPerm, numIndices, &indices, outOffsets, NULL);
  if (*ierr) return;
  *ierr = F90Array1dDestroy(idxPtr, MPIU_INT PETSC_F90_2PTR_PARAM(idxPtrd));
  if (*ierr) return;
  if (!FORTRANNULLSCALARPOINTER(valPtr)) *ierr = F90Array1dDestroy(valPtr, MPIU_SCALAR PETSC_F90_2PTR_PARAM(valPtrd));
}

PETSC_EXTERN void dmplexgetjoin_(DM *dm, PetscInt *numPoints, PetscInt *points, PetscInt *N, F90Array1d *cptr, int *ierr PETSC_F90_2PTR_PROTO(cptrd))
{
  const PetscInt *coveredPoints;
  PetscInt        n;

  CHKFORTRANNULL(N);
  *ierr = DMPlexGetJoin(*dm, *numPoints, points, &n, &coveredPoints);
  if (*ierr) return;
  *ierr = F90Array1dCreate((void *)coveredPoints, MPIU_INT, 1, n, cptr PETSC_F90_2PTR_PARAM(cptrd));
  if (N) *N = n;
}

PETSC_EXTERN void dmplexgetfulljoin_(DM *dm, PetscInt *numPoints, PetscInt *points, PetscInt *N, F90Array1d *cptr, int *ierr PETSC_F90_2PTR_PROTO(cptrd))
{
  const PetscInt *coveredPoints;
  PetscInt        n;

  CHKFORTRANNULL(N);
  *ierr = DMPlexGetFullJoin(*dm, *numPoints, points, &n, &coveredPoints);
  if (*ierr) return;
  *ierr = F90Array1dCreate((void *)coveredPoints, MPIU_INT, 1, n, cptr PETSC_F90_2PTR_PARAM(cptrd));
  if (N) *N = n;
}

PETSC_EXTERN void dmplexrestorejoin_(DM *dm, PetscInt *numPoints, PetscInt *points, PetscInt *N, F90Array1d *cptr, int *ierr PETSC_F90_2PTR_PROTO(cptrd))
{
  PetscInt *coveredPoints;

  *ierr = F90Array1dAccess(cptr, MPIU_INT, (void **)&coveredPoints PETSC_F90_2PTR_PARAM(cptrd));
  if (*ierr) return;
  *ierr = DMPlexRestoreJoin(*dm, 0, NULL, NULL, (const PetscInt **)&coveredPoints);
  if (*ierr) return;
  *ierr = F90Array1dDestroy(cptr, MPIU_INT PETSC_F90_2PTR_PARAM(cptrd));
  if (*ierr) return;
}

PETSC_EXTERN void dmplexgetmeet_(DM *dm, PetscInt *numPoints, PetscInt *points, PetscInt *N, F90Array1d *cptr, int *ierr PETSC_F90_2PTR_PROTO(cptrd))
{
  const PetscInt *coveredPoints;
  PetscInt        n;

  CHKFORTRANNULL(N);
  *ierr = DMPlexGetMeet(*dm, *numPoints, points, &n, &coveredPoints);
  if (*ierr) return;
  *ierr = F90Array1dCreate((void *)coveredPoints, MPIU_INT, 1, n, cptr PETSC_F90_2PTR_PARAM(cptrd));
  if (N) *N = n;
}

PETSC_EXTERN void dmplexgetfullmeet_(DM *dm, PetscInt *numPoints, PetscInt *points, PetscInt *N, F90Array1d *cptr, int *ierr PETSC_F90_2PTR_PROTO(cptrd))
{
  const PetscInt *coveredPoints;
  PetscInt        n;

  CHKFORTRANNULL(N);
  if (*ierr) return;
  *ierr = DMPlexGetFullMeet(*dm, *numPoints, points, &n, &coveredPoints);
  if (*ierr) return;
  *ierr = F90Array1dCreate((void *)coveredPoints, MPIU_INT, 1, n, cptr PETSC_F90_2PTR_PARAM(cptrd));
  if (N) *N = n;
}

PETSC_EXTERN void dmplexrestoremeet_(DM *dm, PetscInt *numPoints, PetscInt *points, PetscInt *N, F90Array1d *cptr, int *ierr PETSC_F90_2PTR_PROTO(cptrd))
{
  PetscInt *coveredPoints;

  *ierr = F90Array1dAccess(cptr, MPIU_INT, (void **)&coveredPoints PETSC_F90_2PTR_PARAM(cptrd));
  if (*ierr) return;
  *ierr = DMPlexRestoreMeet(*dm, 0, NULL, NULL, (const PetscInt **)&coveredPoints);
  if (*ierr) return;
  *ierr = F90Array1dDestroy(cptr, MPIU_INT PETSC_F90_2PTR_PARAM(cptrd));
  if (*ierr) return;
}
