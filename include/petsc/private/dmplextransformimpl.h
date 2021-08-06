#if !defined(_PLEXTRANSFORMIMPL_H)
#define _PLEXTRANSFORMIMPL_H

#include <petsc/private/dmpleximpl.h>
#include <petscdmplextransform.h>

typedef struct _p_DMPlexTransformOps *DMPlexTransformOps;
struct _p_DMPlexTransformOps {
  PetscErrorCode (*view)(DMPlexTransform, PetscViewer);
  PetscErrorCode (*setfromoptions)(PetscOptionItems *, DMPlexTransform);
  PetscErrorCode (*setup)(DMPlexTransform);
  PetscErrorCode (*destroy)(DMPlexTransform);
  PetscErrorCode (*celltransform)(DMPlexTransform,DMPolytopeType,PetscInt,PetscInt*,PetscInt*,DMPolytopeType*[],PetscInt*[],PetscInt*[],PetscInt*[]);
  PetscErrorCode (*getsubcellorientation)(DMPlexTransform,DMPolytopeType,PetscInt,PetscInt,DMPolytopeType,PetscInt,PetscInt,PetscInt*,PetscInt*);
  PetscErrorCode (*mapcoordinates)(DMPlexTransform, DMPolytopeType, DMPolytopeType, PetscInt, PetscInt, PetscInt, const PetscScalar[], PetscScalar[]);
};

struct _p_DMPlexTransform {
  PETSCHEADER(struct _p_DMPlexTransformOps);
  void *data;

  DM            dm;          /* This is the DM for which the transform has been computed */
  DMLabel       active;      /* If not NULL, indicates points that are participating in the transform */
  DMLabel       trType;      /* If not NULL, this holds the transformation type for each point */
  PetscInt      setupcalled; /* Flag to indicate the setup stage */
  PetscInt     *ctOrder;     /* [i] = ct: An array with cell types in depth order */
  PetscInt     *ctOrderInv;  /* [ct] = i: An array with the ordinal numbers for each cell type */
  PetscInt     *ctStart;     /* [ct]: The number for the first cell of each polytope type in the original mesh */
  PetscInt     *ctStartNew;  /* [ctNew]: The number for the first cell of each polytope type in the new mesh */
  PetscInt     *offset;      /* [ct/rt][ctNew]: The offset from ctStartNew[ctNew] in the new point numbering of a point of type ctNew produced from an old point of type ct or refine type rt */
  PetscInt     *trNv;        /* The number of transformed vertices in the closure of a cell of each type */
  PetscScalar **trVerts;     /* The transformed vertex coordinates in the closure of a cell of each type */
  PetscInt  ****trSubVerts;  /* The indices for vertices of subcell (rct, r) in a cell of each type */
  PetscFE      *coordFE;     /* Finite element for each cell type, used for localized coordinate interpolation */
  PetscFEGeom **refGeom;     /* Geometry of the reference cell for each cell type */
};

typedef struct {
  DMLabel label; /* This marks the points to be deleted/ignored */
} DMPlexTransform_Filter;

typedef struct {
  PetscInt dummy;
} DMPlexRefine_Regular;

typedef struct {
  PetscInt dummy;
} DMPlexRefine_ToBox;

typedef struct {
  PetscInt dummy;
} DMPlexRefine_Alfeld;

typedef struct {
  DMLabel      splitPoints; /* List of edges to be bisected (1) and cells to be divided (2) */
  PetscSection secEdgeLen;  /* Section for edge length field */
  PetscReal   *edgeLen;     /* Storage for edge length field */
  PetscInt    *splitArray;  /* Array for communication of split points label */
} DMPlexRefine_SBR;

typedef struct {
  PetscInt       n;         /* The number of divisions to produce, so n = 1 gives 2 new cells */
  PetscReal      r;         /* The factor increase for cell height */
  PetscScalar    *h;        /* The computed cell heights, based on r */
  PetscInt       *Nt;       /* The array of the number of target types */
  DMPolytopeType **target;  /* The array of target types */
  PetscInt       **size;    /* The array of the number of each target type */
  PetscInt       **cone;    /* The array of cones for each target cell */
  PetscInt       **ornt;    /* The array of orientation for each target cell */
} DMPlexRefine_BL;

PetscErrorCode DMPlexTransformMapCoordinatesBarycenter_Internal(DMPlexTransform, DMPolytopeType, DMPolytopeType, PetscInt, PetscInt, PetscInt, const PetscScalar[], PetscScalar[]);
PetscErrorCode DMPlexTransformGetSubcellOrientation_Regular(DMPlexTransform, DMPolytopeType, PetscInt, PetscInt, DMPolytopeType, PetscInt, PetscInt, PetscInt *, PetscInt *);
PetscErrorCode DMPlexTransformCellRefine_Regular(DMPlexTransform, DMPolytopeType, PetscInt, PetscInt *, PetscInt *, DMPolytopeType *[], PetscInt *[], PetscInt *[], PetscInt *[]);

#endif
