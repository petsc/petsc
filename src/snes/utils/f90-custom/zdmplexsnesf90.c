#include <petsc/private/fortranimpl.h>
#include <petscdmplex.h>
#include <petscsnes.h>
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

PETSC_EXTERN void PETSC_STDCALL dmplexgetcellfields_(DM *dm, PetscInt *cStart, PetscInt *cEnd, Vec *locX, Vec *locX_t, Vec *locA, F90Array1d *uPtr, F90Array1d *utPtr, F90Array1d *aPtr, int *ierr PETSC_F90_2PTR_PROTO(uPtrd) PETSC_F90_2PTR_PROTO(utPtrd) PETSC_F90_2PTR_PROTO(aPtrd))
{
  PetscDS      prob;
  PetscScalar *u, *u_t, *a;
  PetscInt     numCells = *cEnd - *cStart, totDim, totDimAux = 0;

  CHKFORTRANNULLOBJECTDEREFERENCE(locX_t);
  CHKFORTRANNULLOBJECTDEREFERENCE(locA);
  *ierr = DMPlexGetCellFields(*dm, *cStart, *cEnd, *locX, *locX_t, *locA, &u, &u_t, &a);if (*ierr) return;
  *ierr = DMGetDS(*dm, &prob);if (*ierr) return;
  *ierr = PetscDSGetTotalDimension(prob, &totDim);if (*ierr) return;
  if (locA) {
    DM      dmAux;
    PetscDS probAux;

    *ierr = VecGetDM(*locA, &dmAux);if (*ierr) return;
    *ierr = DMGetDS(dmAux, &probAux);if (*ierr) return;
    *ierr = PetscDSGetTotalDimension(probAux, &totDimAux);if (*ierr) return;
  }
  *ierr = F90Array1dCreate((void*) u,   PETSC_SCALAR, 1, numCells*totDim,               uPtr  PETSC_F90_2PTR_PARAM(uPtrd));if (*ierr) return;
  *ierr = F90Array1dCreate((void*) u_t, PETSC_SCALAR, 1, locX_t ? numCells*totDim : 0,  utPtr PETSC_F90_2PTR_PARAM(utPtrd));if (*ierr) return;
  *ierr = F90Array1dCreate((void*) a,   PETSC_SCALAR, 1, locA ? numCells*totDimAux : 0, aPtr  PETSC_F90_2PTR_PARAM(aPtrd));
}

PETSC_EXTERN void PETSC_STDCALL dmplexrestorecellfields_(DM *dm, PetscInt *cStart, PetscInt *cEnd, Vec *locX, Vec *locX_t, Vec *locA, F90Array1d *uPtr, F90Array1d *utPtr, F90Array1d *aPtr, int *ierr PETSC_F90_2PTR_PROTO(uPtrd) PETSC_F90_2PTR_PROTO(utPtrd) PETSC_F90_2PTR_PROTO(aPtrd))
{
  PetscScalar *u, *u_t, *a;

  *ierr = F90Array1dAccess(uPtr,  PETSC_SCALAR, (void **) &u   PETSC_F90_2PTR_PARAM(uPtrd));if (*ierr) return;
  *ierr = F90Array1dAccess(utPtr, PETSC_SCALAR, (void **) &u_t PETSC_F90_2PTR_PARAM(utPtrd));if (*ierr) return;
  *ierr = F90Array1dAccess(aPtr,  PETSC_SCALAR, (void **) &a   PETSC_F90_2PTR_PARAM(aPtrd));if (*ierr) return;
  *ierr = DMPlexRestoreCellFields(*dm, *cStart, *cEnd, *locX, NULL, NULL, &u, u_t ? &u_t : NULL, a ? &a : NULL);if (*ierr) return;
  *ierr = F90Array1dDestroy(uPtr,  PETSC_SCALAR PETSC_F90_2PTR_PARAM(uPtrd));if (*ierr) return;
  *ierr = F90Array1dDestroy(utPtr, PETSC_SCALAR PETSC_F90_2PTR_PARAM(utPtrd));if (*ierr) return;
  *ierr = F90Array1dDestroy(aPtr,  PETSC_SCALAR PETSC_F90_2PTR_PARAM(aPtrd));if (*ierr) return;
}

PETSC_EXTERN void PETSC_STDCALL dmplexgetfacefields_(DM *dm, PetscInt *fStart, PetscInt *fEnd, Vec *locX, Vec *locX_t, Vec *faceGeometry, Vec *cellGeometry, Vec *locGrad, PetscInt *Nface, F90Array1d *uLPtr, F90Array1d *uRPtr, int *ierr PETSC_F90_2PTR_PROTO(uLPtrd) PETSC_F90_2PTR_PROTO(uRPtrd))
{
  PetscDS      prob;
  PetscScalar *uL, *uR;
  PetscInt     numFaces = *fEnd - *fStart, totDim;

  CHKFORTRANNULLOBJECTDEREFERENCE(locX_t);
  CHKFORTRANNULLOBJECTDEREFERENCE(locGrad);
  *ierr = DMPlexGetFaceFields(*dm, *fStart, *fEnd, *locX, *locX_t, *faceGeometry, *cellGeometry, *locGrad, Nface, &uL, &uR);if (*ierr) return;
  *ierr = DMGetDS(*dm, &prob);if (*ierr) return;
  *ierr = PetscDSGetTotalDimension(prob, &totDim);if (*ierr) return;
  *ierr = F90Array1dCreate((void*) uL, PETSC_SCALAR, 1, numFaces*totDim, uLPtr PETSC_F90_2PTR_PARAM(uLPtrd));if (*ierr) return;
  *ierr = F90Array1dCreate((void*) uR, PETSC_SCALAR, 1, numFaces*totDim, uRPtr PETSC_F90_2PTR_PARAM(uRPtrd));if (*ierr) return;
}

PETSC_EXTERN void PETSC_STDCALL dmplexrestorefacefields_(DM *dm, PetscInt *fStart, PetscInt *fEnd, Vec *locX, Vec *locX_t, Vec *faceGeometry, Vec *cellGeometry, Vec *locGrad, PetscInt *Nface, F90Array1d *uLPtr, F90Array1d *uRPtr, int *ierr PETSC_F90_2PTR_PROTO(uLPtrd) PETSC_F90_2PTR_PROTO(uRPtrd))
{
  PetscScalar *uL, *uR;

  *ierr = F90Array1dAccess(uLPtr, PETSC_SCALAR, (void **) &uL PETSC_F90_2PTR_PARAM(uLPtrd));if (*ierr) return;
  *ierr = F90Array1dAccess(uRPtr, PETSC_SCALAR, (void **) &uR PETSC_F90_2PTR_PARAM(uRPtrd));if (*ierr) return;
  *ierr = DMPlexRestoreFaceFields(*dm, *fStart, *fEnd, *locX, NULL, *faceGeometry, *cellGeometry, NULL, Nface, &uL, &uR);if (*ierr) return;
  *ierr = F90Array1dDestroy(uLPtr, PETSC_SCALAR PETSC_F90_2PTR_PARAM(uLPtrd));if (*ierr) return;
  *ierr = F90Array1dDestroy(uRPtr, PETSC_SCALAR PETSC_F90_2PTR_PARAM(uRPtrd));if (*ierr) return;
}

PETSC_EXTERN void PETSC_STDCALL dmplexgetfacegeometry_(DM *dm, PetscInt *fStart, PetscInt *fEnd, Vec *faceGeometry, Vec *cellGeometry, PetscInt *Nface, F90Array1d *gPtr, F90Array1d *vPtr, int *ierr PETSC_F90_2PTR_PROTO(gPtrd) PETSC_F90_2PTR_PROTO(vPtrd))
{
  PetscFVFaceGeom *g;
  PetscReal       *v;
  PetscInt         numFaces = *fEnd - *fStart, structSize = sizeof(PetscFVFaceGeom)/sizeof(PetscScalar);

  *ierr = DMPlexGetFaceGeometry(*dm, *fStart, *fEnd, *faceGeometry, *cellGeometry, Nface, &g, &v);if (*ierr) return;
  *ierr = F90Array1dCreate((void*) g, PETSC_SCALAR, 1, numFaces*structSize, gPtr PETSC_F90_2PTR_PARAM(gPtrd));if (*ierr) return;
  *ierr = F90Array1dCreate((void*) v, PETSC_REAL,   1, numFaces*2,          vPtr PETSC_F90_2PTR_PARAM(vPtrd));if (*ierr) return;
}

PETSC_EXTERN void PETSC_STDCALL dmplexrestorefacegeometry_(DM *dm, PetscInt *fStart, PetscInt *fEnd, Vec *faceGeometry, Vec *cellGeometry, PetscInt *Nface, F90Array1d *gPtr, F90Array1d *vPtr, int *ierr PETSC_F90_2PTR_PROTO(gPtrd) PETSC_F90_2PTR_PROTO(vPtrd))
{
  PetscFVFaceGeom *g;
  PetscReal       *v;

  *ierr = F90Array1dAccess(gPtr, PETSC_SCALAR, (void **) &g PETSC_F90_2PTR_PARAM(gPtrd));if (*ierr) return;
  *ierr = F90Array1dAccess(vPtr, PETSC_REAL,   (void **) &v PETSC_F90_2PTR_PARAM(vPtrd));if (*ierr) return;
  *ierr = DMPlexRestoreFaceGeometry(*dm, *fStart, *fEnd, *faceGeometry, *cellGeometry, Nface, &g, &v);if (*ierr) return;
  *ierr = F90Array1dDestroy(gPtr, PETSC_SCALAR PETSC_F90_2PTR_PARAM(vPtrd));if (*ierr) return;
  *ierr = F90Array1dDestroy(vPtr, PETSC_REAL   PETSC_F90_2PTR_PARAM(gPtrd));if (*ierr) return;
}
