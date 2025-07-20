#pragma once

#include <petsc/private/dmpleximpl.h>
#include <petscdmplextransform.h>

PETSC_EXTERN PetscLogEvent DMPLEXTRANSFORM_SetUp;
PETSC_EXTERN PetscLogEvent DMPLEXTRANSFORM_Apply;
PETSC_EXTERN PetscLogEvent DMPLEXTRANSFORM_SetConeSizes;
PETSC_EXTERN PetscLogEvent DMPLEXTRANSFORM_SetCones;
PETSC_EXTERN PetscLogEvent DMPLEXTRANSFORM_CreateSF;
PETSC_EXTERN PetscLogEvent DMPLEXTRANSFORM_CreateLabels;
PETSC_EXTERN PetscLogEvent DMPLEXTRANSFORM_SetCoordinates;

typedef struct _p_DMPlexTransformOps *DMPlexTransformOps;
struct _p_DMPlexTransformOps {
  PetscErrorCode (*view)(DMPlexTransform, PetscViewer);
  PetscErrorCode (*setfromoptions)(DMPlexTransform, PetscOptionItems);
  PetscErrorCode (*setup)(DMPlexTransform);
  PetscErrorCode (*destroy)(DMPlexTransform);
  PetscErrorCode (*setdimensions)(DMPlexTransform, DM, DM);
  PetscErrorCode (*celltransform)(DMPlexTransform, DMPolytopeType, PetscInt, PetscInt *, PetscInt *, DMPolytopeType *[], PetscInt *[], PetscInt *[], PetscInt *[]);
  PetscErrorCode (*ordersupports)(DMPlexTransform, DM, DM);
  PetscErrorCode (*getsubcellorientation)(DMPlexTransform, DMPolytopeType, PetscInt, PetscInt, DMPolytopeType, PetscInt, PetscInt, PetscInt *, PetscInt *);
  PetscErrorCode (*mapcoordinates)(DMPlexTransform, DMPolytopeType, DMPolytopeType, PetscInt, PetscInt, PetscInt, PetscInt, const PetscScalar[], PetscScalar[]);
};

struct _p_DMPlexTransform {
  PETSCHEADER(struct _p_DMPlexTransformOps);
  void *data;

  DM            dm;            /* This is the DM for which the transform has been computed */
  DMLabel       active;        /* If not NULL, indicates points that are participating in the transform */
  DMLabel       trType;        /* If not NULL, this holds the transformation type for each point */
  PetscBool     setupcalled;   /* true if setup has been called */
  PetscInt     *ctOrderOld;    /* [i] = ct: An array with original cell types in depth order */
  PetscInt     *ctOrderInvOld; /* [ct] = i: An array with the ordinal numbers for each original cell type */
  PetscInt     *ctStart;       /* [ct]: The number for the first cell of each polytope type in the original mesh */
  PetscInt     *ctOrderNew;    /* [i] = ct: An array with produced cell types in depth order */
  PetscInt     *ctOrderInvNew; /* [ct] = i: An array with the ordinal numbers for each produced cell type */
  PetscInt     *ctStartNew;    /* [ctNew]: The number for the first cell of each polytope type in the new mesh */
  PetscInt     *offset;        /* [ct/rt][ctNew]: The offset from ctStartNew[ctNew] in the new point numbering of a point of type ctNew produced from an old point of type ct or refine type rt */
  PetscInt      depth;         /* The depth of the transformed mesh */
  PetscInt     *depthStart;    /* The starting point for each depth stratum */
  PetscInt     *depthEnd;      /* The starting point for the next depth stratum */
  PetscInt     *trNv;          /* The number of transformed vertices in the closure of a cell of each type */
  PetscScalar **trVerts;       /* The transformed vertex coordinates in the closure of a cell of each type */
  PetscInt  ****trSubVerts;    /* The indices for vertices of subcell (rct, r) in a cell of each type */
  PetscFE      *coordFE;       /* Finite element for each cell type, used for localized coordinate interpolation */
  PetscFEGeom **refGeom;       /* Geometry of the reference cell for each cell type */
  /* Label construction */
  PetscBool labelMatchStrata; /* Flag to restrict labeled points to the same cell type as parents */
  PetscInt  labelReplicaInc;  /* Multiplier to create new label values for replicas v = oldv + r * repInc */
};

typedef struct {
  PetscInt dummy;
} DMPlexTransform_Filter;

typedef enum {
  NORMAL_DEFAULT,
  NORMAL_INPUT,
  NORMAL_COMPUTE,
  NORMAL_COMPUTE_BD
} PlexNormalAlg;
PETSC_EXTERN const char *const PlexNormalAlgs[];

typedef struct {
  /* Inputs */
  PetscInt            dimEx;       /* The dimension of the extruded mesh */
  PetscInt            cdim;        /* The coordinate dimension of the input mesh */
  PetscInt            cdimEx;      /* The coordinate dimension of the extruded mesh */
  PetscInt            layers;      /* The number of extruded layers */
  PetscReal           thickness;   /* The total thickness of the extruded layers */
  PetscInt            Nth;         /* The number of specified thicknesses */
  PetscReal          *thicknesses; /* The input layer thicknesses */
  PetscBool           useTensor;   /* Flag to create tensor cells */
  PlexNormalAlg       normalAlg;   /* Algorithm to use for computing normal */
  PetscReal           normal[3];   /* Surface normal from input */
  DM                  dmNormal;    // DM for normal field
  Vec                 vecNormal;   // Normal at each vertex
  PetscSimplePointFn *normalFunc;  /* A function returning the normal at a given point */
  PetscBool           symmetric;   /* Extrude layers symmetrically about the surface */
  PetscBool           periodic;    /* Connect the extruded layer periodically to the beginning */
  /* Calculated quantities */
  PetscReal       *layerPos; /* The position of each layer relative to the original surface, along the local normal direction */
  PetscInt        *Nt;       /* The array of the number of target types */
  DMPolytopeType **target;   /* The array of target types */
  PetscInt       **size;     /* The array of the number of each target type */
  PetscInt       **cone;     /* The array of cones for each target cell */
  PetscInt       **ornt;     /* The array of orientation for each target cell */
  // Borrowed storage
  const PetscInt *degree; // The root degree of all points in the original mesh
} DMPlexTransform_Extrude;

typedef struct {
  PetscInt         debug;     // Debugging level
  PetscBool        useTensor; // Flag to create tensor cells
  PetscReal        width;     // The width of a cohesive cell
  PetscInt        *Nt;        // The array of the number of target types
  DMPolytopeType **target;    // The array of target types
  PetscInt       **size;      // The array of the number of each target type
  PetscInt       **cone;      // The array of cones for each target cell
  PetscInt       **ornt;      // The array of orientation for each target cell
} DMPlexTransform_Cohesive;

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
} DMPlexRefine_SBR;

typedef struct {
  PetscInt dummy;
} DMPlexRefine_1D;

typedef struct {
  PetscInt         n;      /* The number of divisions to produce, so n = 1 gives 2 new cells */
  PetscReal        r;      /* The factor increase for cell height */
  PetscScalar     *h;      /* The computed cell heights, based on r */
  PetscInt        *Nt;     /* The array of the number of target types */
  DMPolytopeType **target; /* The array of target types */
  PetscInt       **size;   /* The array of the number of each target type */
  PetscInt       **cone;   /* The array of cones for each target cell */
  PetscInt       **ornt;   /* The array of orientation for each target cell */
} DMPlexRefine_BL;

PetscErrorCode DMPlexTransformSetDimensions_Internal(DMPlexTransform, DM, DM);
PetscErrorCode DMPlexTransformMapCoordinatesBarycenter_Internal(DMPlexTransform, DMPolytopeType, DMPolytopeType, PetscInt, PetscInt, PetscInt, PetscInt, const PetscScalar[], PetscScalar[]);
PetscErrorCode DMPlexTransformGetSubcellOrientation_Regular(DMPlexTransform, DMPolytopeType, PetscInt, PetscInt, DMPolytopeType, PetscInt, PetscInt, PetscInt *, PetscInt *);
PetscErrorCode DMPlexTransformCellRefine_Regular(DMPlexTransform, DMPolytopeType, PetscInt, PetscInt *, PetscInt *, DMPolytopeType *[], PetscInt *[], PetscInt *[], PetscInt *[]);
