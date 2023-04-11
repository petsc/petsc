#include <petsc/private/fortranimpl.h>
#include <petscdmplex.h>
#include <petsc/private/f90impl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define dmplexcomputecellgeometryaffinefem_ DMPLEXCOMPUTECELLGEOMETRYAFFINEFEM
  #define dmplexcomputecellgeometryfvm_       DMPLEXCOMPUTECELLGEOMETRYFVM
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
  #define dmplexcomputecellgeometryaffinefem_ dmplexcomputecellgeometryaffinefem
  #define dmplexcomputecellgeometryfvm_       dmplexcomputecellgeometryfvm
#endif

PETSC_EXTERN void dmplexcomputecellgeometryaffinefem_(DM *dm, PetscInt *cell, F90Array1d *ptrV, F90Array1d *ptrJ, F90Array1d *ptrIJ, PetscReal *detJ, int *ierr PETSC_F90_2PTR_PROTO(ptrVd) PETSC_F90_2PTR_PROTO(ptrJd) PETSC_F90_2PTR_PROTO(ptrIJd))
{
  PetscReal *v0;
  PetscReal *J;
  PetscReal *invJ;

  *ierr = F90Array1dAccess(ptrV, MPIU_REAL, (void **)&v0 PETSC_F90_2PTR_PARAM(ptrVd));
  if (*ierr) return;
  *ierr = F90Array1dAccess(ptrJ, MPIU_REAL, (void **)&J PETSC_F90_2PTR_PARAM(ptrJd));
  if (*ierr) return;
  *ierr = F90Array1dAccess(ptrIJ, MPIU_REAL, (void **)&invJ PETSC_F90_2PTR_PARAM(ptrIJd));
  if (*ierr) return;
  *ierr = DMPlexComputeCellGeometryAffineFEM(*dm, *cell, v0, J, invJ, detJ);
}

PETSC_EXTERN void dmplexcomputecellgeometryfem_(DM *dm, PetscInt *cell, PetscQuadrature *quad, F90Array1d *ptrV, F90Array1d *ptrJ, F90Array1d *ptrIJ, PetscReal *detJ, int *ierr PETSC_F90_2PTR_PROTO(ptrVd) PETSC_F90_2PTR_PROTO(ptrJd) PETSC_F90_2PTR_PROTO(ptrIJd))
{
  PetscReal *v0;
  PetscReal *J;
  PetscReal *invJ;

  *ierr = F90Array1dAccess(ptrV, MPIU_REAL, (void **)&v0 PETSC_F90_2PTR_PARAM(ptrVd));
  if (*ierr) return;
  *ierr = F90Array1dAccess(ptrJ, MPIU_REAL, (void **)&J PETSC_F90_2PTR_PARAM(ptrJd));
  if (*ierr) return;
  *ierr = F90Array1dAccess(ptrIJ, MPIU_REAL, (void **)&invJ PETSC_F90_2PTR_PARAM(ptrIJd));
  if (*ierr) return;
  *ierr = DMPlexComputeCellGeometryFEM(*dm, *cell, *quad, v0, J, invJ, detJ);
}

PETSC_EXTERN void dmplexcomputecellgeometryfvm_(DM *dm, PetscInt *cell, PetscReal *vol, F90Array1d *ptrCentroid, F90Array1d *ptrNormal, int *ierr PETSC_F90_2PTR_PROTO(ptrCentroidd) PETSC_F90_2PTR_PROTO(ptrNormald))
{
  PetscReal *centroid;
  PetscReal *normal;

  *ierr = F90Array1dAccess(ptrCentroid, MPIU_REAL, (void **)&centroid PETSC_F90_2PTR_PARAM(ptrCentroidd));
  if (*ierr) return;
  *ierr = F90Array1dAccess(ptrNormal, MPIU_REAL, (void **)&normal PETSC_F90_2PTR_PARAM(ptrNormald));
  if (*ierr) return;
  *ierr = DMPlexComputeCellGeometryFVM(*dm, *cell, vol, centroid, normal);
}
