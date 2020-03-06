#include <petsc/private/fortranimpl.h>
#include <petscdmplex.h>
#include <petscds.h>
#include <petsc/private/f90impl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define dmplexgetcellfields_       DMPLEXGETCELLFIELDS
#define dmplexrestorecellfields_   DMPLEXRESTORECELLFIELDS
#define dmplexgetfacefields_       DMPLEXGETFACEFIELDS
#define dmplexrestorefacefields_   DMPLEXRESTOREFACEFIELDS
#define dmplexgetfacegeometry_     DMPLEXGETFACEGEOMETRY
#define dmplexrestorefacegeometry_ DMPLEXRESTOREFACEGEOMETRY
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
#define dmplexgetcellfields_       dmplexgetcellfields
#define dmplexrestorecellfields_   dmplexrestorecellfields
#define dmplexgetfacefields_       dmplexgetfacefields
#define dmplexrestorefacefields_   dmplexrestorefacefields
#define dmplexgetfacegeometry_     dmplexgetfacegeometry
#define dmplexrestorefacegeometry_ dmplexrestorefacegeometry
#endif

PETSC_EXTERN void dmplexgetcellfields_(DM *dm, IS *cellIS, Vec *locX, Vec *locX_t, Vec *locA, F90Array1d *uPtr, F90Array1d *utPtr, F90Array1d *aPtr, int *ierr PETSC_F90_2PTR_PROTO(uPtrd) PETSC_F90_2PTR_PROTO(utPtrd) PETSC_F90_2PTR_PROTO(aPtrd))
{
  PetscDS      prob;
  PetscScalar *u, *u_t, *a;
  PetscInt     numCells, totDim, totDimAux = 0;

  *ierr = ISGetLocalSize(*cellIS, &numCells);if (*ierr) return;
  *ierr = DMPlexGetCellFields(*dm, *cellIS, *locX, *locX_t, *locA, &u, &u_t, &a);if (*ierr) return;
  *ierr = DMGetDS(*dm, &prob);if (*ierr) return;
  *ierr = PetscDSGetTotalDimension(prob, &totDim);if (*ierr) return;
  if (locA) {
    DM      dmAux;
    PetscDS probAux;

    *ierr = VecGetDM(*locA, &dmAux);if (*ierr) return;
    *ierr = DMGetDS(dmAux, &probAux);if (*ierr) return;
    *ierr = PetscDSGetTotalDimension(probAux, &totDimAux);if (*ierr) return;
  }
  *ierr = F90Array1dCreate((void*) u,   MPIU_SCALAR, 1, numCells*totDim,               uPtr  PETSC_F90_2PTR_PARAM(uPtrd));if (*ierr) return;
  *ierr = F90Array1dCreate((void*) u_t, MPIU_SCALAR, 1, locX_t ? numCells*totDim : 0,  utPtr PETSC_F90_2PTR_PARAM(utPtrd));if (*ierr) return;
  *ierr = F90Array1dCreate((void*) a,   MPIU_SCALAR, 1, locA ? numCells*totDimAux : 0, aPtr  PETSC_F90_2PTR_PARAM(aPtrd));
}

PETSC_EXTERN void dmplexrestorecellfields_(DM *dm, IS *cellIS, Vec *locX, Vec *locX_t, Vec *locA, F90Array1d *uPtr, F90Array1d *utPtr, F90Array1d *aPtr, int *ierr PETSC_F90_2PTR_PROTO(uPtrd) PETSC_F90_2PTR_PROTO(utPtrd) PETSC_F90_2PTR_PROTO(aPtrd))
{
  PetscScalar *u, *u_t, *a;

  *ierr = F90Array1dAccess(uPtr,  MPIU_SCALAR, (void **) &u   PETSC_F90_2PTR_PARAM(uPtrd));if (*ierr) return;
  *ierr = F90Array1dAccess(utPtr, MPIU_SCALAR, (void **) &u_t PETSC_F90_2PTR_PARAM(utPtrd));if (*ierr) return;
  *ierr = F90Array1dAccess(aPtr,  MPIU_SCALAR, (void **) &a   PETSC_F90_2PTR_PARAM(aPtrd));if (*ierr) return;
  *ierr = DMPlexRestoreCellFields(*dm, *cellIS, *locX, NULL, NULL, &u, u_t ? &u_t : NULL, a ? &a : NULL);if (*ierr) return;
  *ierr = F90Array1dDestroy(uPtr,  MPIU_SCALAR PETSC_F90_2PTR_PARAM(uPtrd));if (*ierr) return;
  *ierr = F90Array1dDestroy(utPtr, MPIU_SCALAR PETSC_F90_2PTR_PARAM(utPtrd));if (*ierr) return;
  *ierr = F90Array1dDestroy(aPtr,  MPIU_SCALAR PETSC_F90_2PTR_PARAM(aPtrd));if (*ierr) return;
}

PETSC_EXTERN void dmplexgetfacefields_(DM *dm, PetscInt *fStart, PetscInt *fEnd, Vec *locX, Vec *locX_t, Vec *faceGeometry, Vec *cellGeometry, Vec *locGrad, PetscInt *Nface, F90Array1d *uLPtr, F90Array1d *uRPtr, int *ierr PETSC_F90_2PTR_PROTO(uLPtrd) PETSC_F90_2PTR_PROTO(uRPtrd))
{
  PetscDS      prob;
  PetscScalar *uL, *uR;
  PetscInt     numFaces = *fEnd - *fStart, totDim;

  *ierr = DMPlexGetFaceFields(*dm, *fStart, *fEnd, *locX, *locX_t, *faceGeometry, *cellGeometry, *locGrad, Nface, &uL, &uR);if (*ierr) return;
  *ierr = DMGetDS(*dm, &prob);if (*ierr) return;
  *ierr = PetscDSGetTotalDimension(prob, &totDim);if (*ierr) return;
  *ierr = F90Array1dCreate((void*) uL, MPIU_SCALAR, 1, numFaces*totDim, uLPtr PETSC_F90_2PTR_PARAM(uLPtrd));if (*ierr) return;
  *ierr = F90Array1dCreate((void*) uR, MPIU_SCALAR, 1, numFaces*totDim, uRPtr PETSC_F90_2PTR_PARAM(uRPtrd));if (*ierr) return;
}

PETSC_EXTERN void dmplexrestorefacefields_(DM *dm, PetscInt *fStart, PetscInt *fEnd, Vec *locX, Vec *locX_t, Vec *faceGeometry, Vec *cellGeometry, Vec *locGrad, PetscInt *Nface, F90Array1d *uLPtr, F90Array1d *uRPtr, int *ierr PETSC_F90_2PTR_PROTO(uLPtrd) PETSC_F90_2PTR_PROTO(uRPtrd))
{
  PetscScalar *uL, *uR;

  *ierr = F90Array1dAccess(uLPtr, MPIU_SCALAR, (void **) &uL PETSC_F90_2PTR_PARAM(uLPtrd));if (*ierr) return;
  *ierr = F90Array1dAccess(uRPtr, MPIU_SCALAR, (void **) &uR PETSC_F90_2PTR_PARAM(uRPtrd));if (*ierr) return;
  *ierr = DMPlexRestoreFaceFields(*dm, *fStart, *fEnd, *locX, NULL, *faceGeometry, *cellGeometry, NULL, Nface, &uL, &uR);if (*ierr) return;
  *ierr = F90Array1dDestroy(uLPtr, MPIU_SCALAR PETSC_F90_2PTR_PARAM(uLPtrd));if (*ierr) return;
  *ierr = F90Array1dDestroy(uRPtr, MPIU_SCALAR PETSC_F90_2PTR_PARAM(uRPtrd));if (*ierr) return;
}

PETSC_EXTERN void dmplexgetfacegeometry_(DM *dm, PetscInt *fStart, PetscInt *fEnd, Vec *faceGeometry, Vec *cellGeometry, PetscInt *Nface, F90Array1d *gPtr, F90Array1d *vPtr, int *ierr PETSC_F90_2PTR_PROTO(gPtrd) PETSC_F90_2PTR_PROTO(vPtrd))
{
  PetscFVFaceGeom *g;
  PetscReal       *v;
  PetscInt         numFaces = *fEnd - *fStart, structSize = sizeof(PetscFVFaceGeom)/sizeof(PetscScalar);

  *ierr = DMPlexGetFaceGeometry(*dm, *fStart, *fEnd, *faceGeometry, *cellGeometry, Nface, &g, &v);if (*ierr) return;
  *ierr = F90Array1dCreate((void*) g, MPIU_SCALAR, 1, numFaces*structSize, gPtr PETSC_F90_2PTR_PARAM(gPtrd));if (*ierr) return;
  *ierr = F90Array1dCreate((void*) v, MPIU_REAL,   1, numFaces*2,          vPtr PETSC_F90_2PTR_PARAM(vPtrd));if (*ierr) return;
}

PETSC_EXTERN void dmplexrestorefacegeometry_(DM *dm, PetscInt *fStart, PetscInt *fEnd, Vec *faceGeometry, Vec *cellGeometry, PetscInt *Nface, F90Array1d *gPtr, F90Array1d *vPtr, int *ierr PETSC_F90_2PTR_PROTO(gPtrd) PETSC_F90_2PTR_PROTO(vPtrd))
{
  PetscFVFaceGeom *g;
  PetscReal       *v;

  *ierr = F90Array1dAccess(gPtr, MPIU_SCALAR, (void **) &g PETSC_F90_2PTR_PARAM(gPtrd));if (*ierr) return;
  *ierr = F90Array1dAccess(vPtr, MPIU_REAL,   (void **) &v PETSC_F90_2PTR_PARAM(vPtrd));if (*ierr) return;
  *ierr = DMPlexRestoreFaceGeometry(*dm, *fStart, *fEnd, *faceGeometry, *cellGeometry, Nface, &g, &v);if (*ierr) return;
  *ierr = F90Array1dDestroy(gPtr, MPIU_SCALAR PETSC_F90_2PTR_PARAM(vPtrd));if (*ierr) return;
  *ierr = F90Array1dDestroy(vPtr, MPIU_REAL   PETSC_F90_2PTR_PARAM(gPtrd));if (*ierr) return;
}
