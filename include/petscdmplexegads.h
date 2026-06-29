#pragma once

#include <petscdmplex.h>

/* MANSEC = DM */
/* SUBMANSEC = DMPlex */

#if !defined(PETSC_HAVE_EGADS)
  #error "PETSc not configured for EGADS; reconfigrue --with-egads or --download-egads"
#endif

/* Declarations below provide an interface to the EGADS/EGADSlite libraries, that can be automatically installed
   by PETSc. These functions enable the creation, interrogation, and manipulation of CAD geometries attached to
   a DMPlex. */
#include <egads.h>
#include <egads_lite.h>

/*S
   PetscGeom - Handle for an EGADS geometric entity (body, shell, face, loop, edge, or node) attached to a `DMPLEX`

   Level: intermediate

   Note:
   `PetscGeom` is a typedef for the EGADS `ego` handle. It is used by the `DMPlexGetGeom...()` family of routines to navigate the CAD model that PETSc associates with a mesh.

.seealso: `DMPlexGetGeomModelBodies()`, `DMPlexGetGeomObject()`, `DMPlexFreeGeomObject()`, `DMPlexGeomDataAndGrads()`
S*/
typedef ego PetscGeom;

/*MC
  PetscCallEGADS - Calls an EGADS function and then checks the resulting error code, if it is
  non-zero it calls the error handler and returns from the current function with the error
  code `PETSC_ERR_LIB`.

  Synopsis:
  #include <petscdmplexegads.h>
  void PetscCallEGADS(func, args)

  Not Collective

  Input Parameters:
+ func - any EGADS function that returns an error code
- args - the arguments to the function

  Level: beginner

.seealso: `PetscCall()`, `SETERRQ()`, `PetscCheck()`, `PetscAssert()`, `PetscTraceBackErrorHandler()`, `PetscCallMPI()`,
          `PetscPushErrorHandler()`, `PetscError()`, `CHKMEMQ`, `CHKERRA()`,
          `CHKERRMPI()`, `PetscCallBack()`, `PetscCallAbort()`, `PetscCallVoid()`, `PetscCallNull()`
M*/
#define PetscCallEGADS(func, args) \
  do { \
    int _status; \
    PetscStackPushExternal(#func); \
    _status = func args; \
    PetscStackPop; \
    PetscCheck(_status >= 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "Error in EGADS call %s() Status %d", #func, (int)_status); \
  } while (0)

PETSC_EXTERN PetscErrorCode DMPlexGeomDataAndGrads(DM, PetscBool);
PETSC_EXTERN PetscErrorCode DMPlexModifyGeomModel(DM, MPI_Comm, PetscScalar[], PetscScalar[], PetscBool, PetscBool, const char[]);
PETSC_EXTERN PetscErrorCode DMPlexInflateToGeomModelUseXYZ(DM);
PETSC_EXTERN PetscErrorCode DMPlexGetGeomModelTUV(DM);
PETSC_EXTERN PetscErrorCode DMPlexInflateToGeomModelUseTUV(DM);
PETSC_EXTERN PetscErrorCode DMPlexGetGeomModelBodies(DM, PetscGeom **, PetscInt *);
PETSC_EXTERN PetscErrorCode DMPlexGetGeomModelBodyShells(DM, PetscGeom, PetscGeom **, PetscInt *);
PETSC_EXTERN PetscErrorCode DMPlexGetGeomModelBodyFaces(DM, PetscGeom, PetscGeom **, PetscInt *);
PETSC_EXTERN PetscErrorCode DMPlexGetGeomModelBodyLoops(DM, PetscGeom, PetscGeom **, PetscInt *);
PETSC_EXTERN PetscErrorCode DMPlexGetGeomModelBodyEdges(DM, PetscGeom, PetscGeom **, PetscInt *);
PETSC_EXTERN PetscErrorCode DMPlexGetGeomModelBodyNodes(DM, PetscGeom, PetscGeom **, PetscInt *);
PETSC_EXTERN PetscErrorCode DMPlexGetGeomModelShellFaces(DM, PetscGeom, PetscGeom, PetscGeom **, PetscInt *);
PETSC_EXTERN PetscErrorCode DMPlexGetGeomModelFaceLoops(DM, PetscGeom, PetscGeom, PetscGeom **, PetscInt *);
PETSC_EXTERN PetscErrorCode DMPlexGetGeomModelFaceEdges(DM, PetscGeom, PetscGeom, PetscGeom **, PetscInt *);
PETSC_EXTERN PetscErrorCode DMPlexGetGeomModelEdgeNodes(DM, PetscGeom, PetscGeom, PetscGeom **, PetscInt *);
PETSC_EXTERN PetscErrorCode DMPlexGetGeomBodyMassProperties(DM, PetscGeom, PetscScalar *, PetscScalar *, PetscScalar **, PetscInt *, PetscScalar **, PetscInt *);
PETSC_EXTERN PetscErrorCode DMPlexRestoreGeomBodyMassProperties(DM, PetscGeom, PetscScalar *, PetscScalar *, PetscScalar **, PetscInt *, PetscScalar **, PetscInt *);
PETSC_EXTERN PetscErrorCode DMPlexGetGeomID(DM, PetscGeom, PetscGeom, PetscInt *);
PETSC_EXTERN PetscErrorCode DMPlexGetGeomObject(DM, PetscGeom, PetscInt, PetscInt, PetscGeom *);
PETSC_EXTERN PetscErrorCode DMPlexGetGeomFaceNumOfControlPoints(DM, PetscGeom, PetscInt *);
PETSC_EXTERN PetscErrorCode DMPlexFreeGeomObject(DM, PetscGeom *);
PETSC_EXTERN PetscErrorCode DMPlexGetGeomCntrlPntAndWeightData(DM, PetscHMapI *, PetscInt *, PetscScalar **, PetscInt *, Mat *, PetscHMapI *, PetscInt *, PetscScalar **);
PETSC_EXTERN PetscErrorCode DMPlexRestoreGeomCntrlPntAndWeightData(DM, PetscHMapI *, PetscInt *, PetscScalar **, PetscInt *, Mat *, PetscHMapI *, PetscInt *, PetscScalar **);
PETSC_EXTERN PetscErrorCode DMPlexGetGeomGradData(DM, PetscHMapI *, Mat *, PetscInt *, PetscScalar **, PetscScalar **, PetscInt *, PetscScalar **, PetscScalar **);
PETSC_EXTERN PetscErrorCode DMPlexRestoreGeomGradData(DM, PetscHMapI *, Mat *, PetscInt *, PetscScalar **, PetscScalar **, PetscInt *, PetscScalar **, PetscScalar **);
PETSC_EXTERN PetscErrorCode DMPlexGetGeomCntrlPntMaps(DM, PetscInt *, PetscInt **, PetscInt **, PetscInt **, PetscInt **, PetscInt **, PetscInt **);
