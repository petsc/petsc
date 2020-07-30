#if !defined(_PLEXIMPL_H)
#define _PLEXIMPL_H

#include <petscmat.h>       /*I      "petscmat.h"          I*/
#include <petscdmplex.h> /*I      "petscdmplex.h"    I*/
#include <petscbt.h>
#include <petscsf.h>
#include <petsc/private/dmimpl.h>

PETSC_EXTERN PetscLogEvent DMPLEX_Interpolate;
PETSC_EXTERN PetscLogEvent DMPLEX_Partition;
PETSC_EXTERN PetscLogEvent DMPLEX_PartSelf;
PETSC_EXTERN PetscLogEvent DMPLEX_PartLabelInvert;
PETSC_EXTERN PetscLogEvent DMPLEX_PartLabelCreateSF;
PETSC_EXTERN PetscLogEvent DMPLEX_PartStratSF;
PETSC_EXTERN PetscLogEvent DMPLEX_CreatePointSF;
PETSC_EXTERN PetscLogEvent DMPLEX_Distribute;
PETSC_EXTERN PetscLogEvent DMPLEX_DistributeCones;
PETSC_EXTERN PetscLogEvent DMPLEX_DistributeLabels;
PETSC_EXTERN PetscLogEvent DMPLEX_DistributeSF;
PETSC_EXTERN PetscLogEvent DMPLEX_DistributeOverlap;
PETSC_EXTERN PetscLogEvent DMPLEX_DistributeField;
PETSC_EXTERN PetscLogEvent DMPLEX_DistributeData;
PETSC_EXTERN PetscLogEvent DMPLEX_Migrate;
PETSC_EXTERN PetscLogEvent DMPLEX_InterpolateSF;
PETSC_EXTERN PetscLogEvent DMPLEX_GlobalToNaturalBegin;
PETSC_EXTERN PetscLogEvent DMPLEX_GlobalToNaturalEnd;
PETSC_EXTERN PetscLogEvent DMPLEX_NaturalToGlobalBegin;
PETSC_EXTERN PetscLogEvent DMPLEX_NaturalToGlobalEnd;
PETSC_EXTERN PetscLogEvent DMPLEX_Stratify;
PETSC_EXTERN PetscLogEvent DMPLEX_Symmetrize;
PETSC_EXTERN PetscLogEvent DMPLEX_Preallocate;
PETSC_EXTERN PetscLogEvent DMPLEX_ResidualFEM;
PETSC_EXTERN PetscLogEvent DMPLEX_JacobianFEM;
PETSC_EXTERN PetscLogEvent DMPLEX_InterpolatorFEM;
PETSC_EXTERN PetscLogEvent DMPLEX_InjectorFEM;
PETSC_EXTERN PetscLogEvent DMPLEX_IntegralFEM;
PETSC_EXTERN PetscLogEvent DMPLEX_CreateGmsh;
PETSC_EXTERN PetscLogEvent DMPLEX_RebalanceSharedPoints;
PETSC_EXTERN PetscLogEvent DMPLEX_CreateFromFile;
PETSC_EXTERN PetscLogEvent DMPLEX_CreateFromCellList;
PETSC_EXTERN PetscLogEvent DMPLEX_CreateFromCellList_Coordinates;
PETSC_EXTERN PetscLogEvent DMPLEX_LocatePoints;

typedef struct _DMPlexCellRefinerOps *DMPlexCellRefinerOps;
struct _DMPlexCellRefinerOps {
  PetscErrorCode (*refine)(DMPlexCellRefiner, DMPolytopeType, PetscInt *, DMPolytopeType *[], PetscInt *[], PetscInt *[], PetscInt *[]);
  PetscErrorCode (*mapsubcells)(DMPlexCellRefiner, DMPolytopeType, PetscInt, DMPolytopeType, PetscInt, PetscInt, PetscInt *, PetscInt *);
  PetscErrorCode (*getaffinetransforms)(DMPlexCellRefiner, DMPolytopeType, PetscInt *, PetscReal *[], PetscReal *[], PetscReal *[]);
  PetscErrorCode (*getaffinefacetransforms)(DMPlexCellRefiner, DMPolytopeType, PetscInt *, PetscReal *[], PetscReal *[], PetscReal *[], PetscReal *[]);
  PetscErrorCode (*getcellvertices)(DMPlexCellRefiner, DMPolytopeType, PetscInt *, PetscReal *[]);
  PetscErrorCode (*getsubcellvertices)(DMPlexCellRefiner, DMPolytopeType, DMPolytopeType, PetscInt, PetscInt *, PetscInt *[]);
  PetscErrorCode (*mapcoords)(DMPlexCellRefiner, DMPolytopeType, DMPolytopeType, PetscInt, PetscInt, PetscInt, const PetscScalar[], PetscScalar[]);
  PetscErrorCode (*setup)(DMPlexCellRefiner);
  PetscErrorCode (*destroy)(DMPlexCellRefiner);
};

struct _p_DMPlexCellRefiner {
  PETSCHEADER(struct _DMPlexCellRefinerOps);
  DM                    dm;          /* The original DM */
  PetscBool             setupcalled;
  DMPlexCellRefinerType type;
  PetscInt              *ctOrder;    /* [i] = ct: An array with cell types in depth order */
  PetscInt              *ctOrderInv; /* [ct] = i: An array with the ordinal numbers for each cell type */
  PetscInt              *ctStart;    /* The number for the first cell of each polytope type in the original mesh, indexed by cell type */
  PetscInt              *ctStartNew; /* The number for the first cell of each polytope type in the new mesh, indexed by cell type */
  PetscInt              *offset;     /* [ct][ctNew]: The offset in the new point numbering of a point of type ctNew produced from an old point of type ct */
  PetscFE               *coordFE;    /* Finite element for each cell type, used for localized coordinate interpolation */
  PetscFEGeom           **refGeom;   /* Geometry of the reference cell for each cell type */
  void                  *data;       /* refiner private data */
};

/* Utility struct to store the contents of a Fluent file in memory */
typedef struct {
  int          index;    /* Type of section */
  unsigned int zoneID;
  unsigned int first;
  unsigned int last;
  int          type;
  int          nd;       /* Either ND or element-type */
  void        *data;
} FluentSection;

struct _PetscGridHash {
  PetscInt     dim;
  PetscReal    lower[3];    /* The lower-left corner */
  PetscReal    upper[3];    /* The upper-right corner */
  PetscReal    extent[3];   /* The box size */
  PetscReal    h[3];        /* The subbox size */
  PetscInt     n[3];        /* The number of subboxes */
  PetscSection cellSection; /* Offsets for cells in each subbox*/
  IS           cells;       /* List of cells in each subbox */
  DMLabel      cellsSparse; /* Sparse storage for cell map */
};

/* Point Numbering in Plex:

   Points are numbered contiguously by stratum. Strate are organized as follows:

   First Stratum:  Cells [height 0]
   Second Stratum: Vertices [depth 0]
   Third Stratum:  Faces [height 1]
   Fourth Stratum: Edges [depth 1]

   We do this so that the numbering of a cell-vertex mesh does not change after interpolation. Within a given stratum,
   we allow additional segregation of by cell type.
*/
typedef struct {
  PetscInt             refct;

  PetscSection         coneSection;       /* Layout of cones (inedges for DAG) */
  PetscInt             maxConeSize;       /* Cached for fast lookup */
  PetscInt            *cones;             /* Cone for each point */
  PetscInt            *coneOrientations;  /* Orientation of each cone point, means cone traveral should start on point 'o', and if negative start on -(o+1) and go in reverse */
  PetscSection         supportSection;    /* Layout of cones (inedges for DAG) */
  PetscInt             maxSupportSize;    /* Cached for fast lookup */
  PetscInt            *supports;          /* Cone for each point */
  PetscBool            refinementUniform; /* Flag for uniform cell refinement */
  PetscReal            refinementLimit;   /* Maximum volume for refined cell */
  PetscErrorCode     (*refinementFunc)(const PetscReal [], PetscReal *); /* Function giving the maximum volume for refined cell */
  PetscInt             overlap;           /* Overlap of the partitions as passed to DMPlexDistribute() or DMPlexDistributeOverlap() */
  DMPlexInterpolatedFlag interpolated;
  DMPlexInterpolatedFlag interpolatedCollective;

  PetscInt            *facesTmp;          /* Work space for faces operation */

  /* Hierarchy */
  DMPlexCellRefinerType cellRefiner;       /* Strategy for refining cells */
  PetscBool             regularRefinement; /* This flag signals that we are a regular refinement of coarseMesh */

  /* Generation */
  char                *tetgenOpts;
  char                *triangleOpts;
  PetscPartitioner     partitioner;
  PetscBool            partitionBalance;  /* Evenly divide partition overlap when distributing */
  PetscBool            remeshBd;

  /* Submesh */
  DMLabel              subpointMap;       /* Label each original mesh point in the submesh with its depth, subpoint are the implicit numbering */
  IS                   subpointIS;        /* IS holding point number in the enclosing mesh of every point in the submesh chart */
  PetscObjectState     subpointState;     /* The state of subpointMap when the subpointIS was last created */

  /* Labels and numbering */
  PetscObjectState     depthState;        /* State of depth label, so that we can determine if a user changes it */
  PetscObjectState     celltypeState;     /* State of celltype label, so that we can determine if a user changes it */
  IS                   globalVertexNumbers;
  IS                   globalCellNumbers;

  /* Constraints */
  PetscSection         anchorSection;      /* maps constrained points to anchor points */
  IS                   anchorIS;           /* anchors indexed by the above section */
  PetscErrorCode     (*createanchors)(DM); /* automatically compute anchors (probably from tree constraints) */
  PetscErrorCode     (*computeanchormatrix)(DM,PetscSection,PetscSection,Mat);

  /* Tree: automatically construct constraints for hierarchically non-conforming meshes */
  PetscSection         parentSection;     /* dof == 1 if point has parent */
  PetscInt            *parents;           /* point to parent */
  PetscInt            *childIDs;          /* point to child ID */
  PetscSection         childSection;      /* inverse of parent section */
  PetscInt            *children;          /* point to children */
  DM                   referenceTree;     /* reference tree to which child ID's refer */
  PetscErrorCode      (*getchildsymmetry)(DM,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt*,PetscInt*);

  /* MATIS support */
  PetscSection         subdomainSection;

  /* Adjacency */
  PetscBool            useAnchors;        /* Replace constrained points with their anchors in adjacency lists */
  PetscErrorCode      (*useradjacency)(DM,PetscInt,PetscInt*,PetscInt[],void*); /* User callback for adjacency */
  void                *useradjacencyctx;  /* User context for callback */

  /* Projection */
  PetscInt             maxProjectionHeight; /* maximum height of cells used in DMPlexProject functions */
  PetscInt             activePoint;         /* current active point in iteration */

  /* Output */
  PetscInt             vtkCellHeight;            /* The height of cells for output, default is 0 */
  PetscReal            scale[NUM_PETSC_UNITS];   /* The scale for each SI unit */

  /* Geometry */
  PetscReal            minradius;         /* Minimum distance from cell centroid to face */
  PetscBool            useHashLocation;   /* Use grid hashing for point location */
  PetscGridHash        lbox;              /* Local box for searching */
  void               (*coordFunc)(PetscInt, PetscInt, PetscInt, /* Function used to remap newly introduced vertices */
                                  const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                  const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                  PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]);

  /* Neighbors */
  PetscMPIInt*         neighbors;

  /* Debugging */
  PetscBool            printSetValues;
  PetscInt             printFEM;
  PetscInt             printL2;
  PetscReal            printTol;
} DM_Plex;

PETSC_EXTERN PetscErrorCode DMPlexVTKWriteAll_VTU(DM,PetscViewer);
PETSC_EXTERN PetscErrorCode VecView_Plex_Local(Vec,PetscViewer);
PETSC_EXTERN PetscErrorCode VecView_Plex_Native(Vec,PetscViewer);
PETSC_EXTERN PetscErrorCode VecView_Plex(Vec,PetscViewer);
PETSC_EXTERN PetscErrorCode VecLoad_Plex_Local(Vec,PetscViewer);
PETSC_EXTERN PetscErrorCode VecLoad_Plex_Native(Vec,PetscViewer);
PETSC_EXTERN PetscErrorCode VecLoad_Plex(Vec,PetscViewer);
PETSC_INTERN PetscErrorCode DMPlexGetFieldType_Internal(DM, PetscSection, PetscInt, PetscInt *, PetscInt *, PetscViewerVTKFieldType *);
PETSC_INTERN PetscErrorCode DMPlexView_GLVis(DM,PetscViewer);
PETSC_INTERN PetscErrorCode DMSetUpGLVisViewer_Plex(PetscObject,PetscViewer);
#if defined(PETSC_HAVE_HDF5)
PETSC_EXTERN PetscErrorCode VecView_Plex_Local_HDF5(Vec, PetscViewer);
PETSC_EXTERN PetscErrorCode VecView_Plex_HDF5(Vec, PetscViewer);
PETSC_EXTERN PetscErrorCode VecLoad_Plex_HDF5(Vec, PetscViewer);
PETSC_EXTERN PetscErrorCode VecView_Plex_HDF5_Native(Vec, PetscViewer);
PETSC_EXTERN PetscErrorCode VecLoad_Plex_HDF5_Native(Vec, PetscViewer);
PETSC_EXTERN PetscErrorCode DMPlexView_HDF5(DM, PetscViewer);
PETSC_EXTERN PetscErrorCode DMPlexLoad_HDF5(DM, PetscViewer);
#endif

PETSC_INTERN PetscErrorCode DMPlexVecGetClosureAtDepth_Internal(DM, PetscSection, Vec, PetscInt, PetscInt, PetscInt *, PetscScalar *[]);
PETSC_INTERN PetscErrorCode DMPlexClosurePoints_Private(DM,PetscInt,const PetscInt[],IS*);
PETSC_INTERN PetscErrorCode DMSetFromOptions_NonRefinement_Plex(PetscOptionItems *, DM);
PETSC_INTERN PetscErrorCode DMCoarsen_Plex(DM, MPI_Comm, DM *);
PETSC_INTERN PetscErrorCode DMCoarsenHierarchy_Plex(DM, PetscInt, DM []);
PETSC_INTERN PetscErrorCode DMRefine_Plex(DM, MPI_Comm, DM *);
PETSC_INTERN PetscErrorCode DMRefineHierarchy_Plex(DM, PetscInt, DM []);
PETSC_INTERN PetscErrorCode DMAdaptLabel_Plex(DM, DMLabel, DM *);
PETSC_INTERN PetscErrorCode DMAdaptMetric_Plex(DM, Vec, DMLabel, DM *);
PETSC_INTERN PetscErrorCode DMPlexInsertBoundaryValues_Plex(DM, PetscBool, Vec, PetscReal, Vec, Vec, Vec);
PETSC_INTERN PetscErrorCode DMProjectFunctionLocal_Plex(DM,PetscReal,PetscErrorCode(**)(PetscInt,PetscReal,const PetscReal[],PetscInt,PetscScalar *,void *),void **,InsertMode,Vec);
PETSC_INTERN PetscErrorCode DMProjectFunctionLabelLocal_Plex(DM,PetscReal,DMLabel,PetscInt,const PetscInt[],PetscInt,const PetscInt[],PetscErrorCode(**)(PetscInt,PetscReal,const PetscReal[],PetscInt,PetscScalar *,void *),void **,InsertMode,Vec);
PETSC_INTERN PetscErrorCode DMProjectFieldLocal_Plex(DM,PetscReal,Vec,void (**)(PetscInt,PetscInt,PetscInt,const PetscInt[],const PetscInt[],const PetscScalar[],const PetscScalar[],const PetscScalar[],const PetscInt[],const PetscInt[],const PetscScalar[],const PetscScalar[],const PetscScalar[],PetscReal,const PetscReal[],PetscInt,const PetscScalar[],PetscScalar[]),InsertMode,Vec);
PETSC_INTERN PetscErrorCode DMProjectFieldLabelLocal_Plex(DM,PetscReal,DMLabel,PetscInt,const PetscInt[],PetscInt,const PetscInt[],Vec,void (**)(PetscInt,PetscInt,PetscInt,const PetscInt[],const PetscInt[],const PetscScalar[],const PetscScalar[],const PetscScalar[],const PetscInt[],const PetscInt[],const PetscScalar[],const PetscScalar[],const PetscScalar[],PetscReal,const PetscReal[],PetscInt,const PetscScalar[],PetscScalar[]),InsertMode,Vec);
PETSC_INTERN PetscErrorCode DMProjectBdFieldLabelLocal_Plex(DM,PetscReal,DMLabel,PetscInt,const PetscInt[],PetscInt,const PetscInt[],Vec,void (**)(PetscInt,PetscInt,PetscInt,const PetscInt[],const PetscInt[],const PetscScalar[],const PetscScalar[],const PetscScalar[],const PetscInt[],const PetscInt[],const PetscScalar[],const PetscScalar[],const PetscScalar[],PetscReal,const PetscReal[],const PetscReal[],PetscInt,const PetscScalar[],PetscScalar[]),InsertMode,Vec);
PETSC_INTERN PetscErrorCode DMComputeL2Diff_Plex(DM,PetscReal,PetscErrorCode(**)(PetscInt,PetscReal,const PetscReal[],PetscInt,PetscScalar *,void *),void **,Vec,PetscReal *);
PETSC_INTERN PetscErrorCode DMComputeL2GradientDiff_Plex(DM,PetscReal,PetscErrorCode(**)(PetscInt,PetscReal,const PetscReal[], const PetscReal[],PetscInt,PetscScalar *,void *),void **,Vec,const PetscReal [],PetscReal *);
PETSC_INTERN PetscErrorCode DMComputeL2FieldDiff_Plex(DM,PetscReal,PetscErrorCode(**)(PetscInt,PetscReal,const PetscReal[],PetscInt,PetscScalar *,void *),void **,Vec,PetscReal *);
PETSC_INTERN PetscErrorCode DMLocatePoints_Plex(DM, Vec, DMPointLocationType, PetscSF);

PETSC_INTERN PetscErrorCode DMPlexBuildFromCellList_Internal(DM, PetscInt, PetscInt, PetscInt, const PetscInt[], PetscBool);
PETSC_INTERN PetscErrorCode DMPlexBuildFromCellList_Parallel_Internal(DM, PetscInt, PetscInt, PetscInt, const PetscInt[], PetscBool, PetscSF *);
PETSC_INTERN PetscErrorCode DMPlexBuildCoordinates_Internal(DM, PetscInt, PetscInt, PetscInt, const PetscReal[]);
PETSC_INTERN PetscErrorCode DMPlexBuildCoordinates_Parallel_Internal(DM, PetscInt, PetscInt, PetscSF, const PetscReal[]);
PETSC_INTERN PetscErrorCode DMPlexLoadLabels_HDF5_Internal(DM, PetscViewer);
PETSC_INTERN PetscErrorCode DMPlexView_HDF5_Internal(DM, PetscViewer);
PETSC_INTERN PetscErrorCode DMPlexLoad_HDF5_Internal(DM, PetscViewer);
PETSC_INTERN PetscErrorCode DMPlexLoad_HDF5_Xdmf_Internal(DM, PetscViewer);
PETSC_INTERN PetscErrorCode VecView_Plex_HDF5_Internal(Vec, PetscViewer);
PETSC_INTERN PetscErrorCode VecView_Plex_HDF5_Native_Internal(Vec, PetscViewer);
PETSC_INTERN PetscErrorCode VecView_Plex_Local_HDF5_Internal(Vec, PetscViewer);
PETSC_INTERN PetscErrorCode VecLoad_Plex_HDF5_Internal(Vec, PetscViewer);
PETSC_INTERN PetscErrorCode VecLoad_Plex_HDF5_Native_Internal(Vec, PetscViewer);
/* TODO Make these INTERN */
PETSC_EXTERN PetscErrorCode DMPlexView_ExodusII_Internal(DM, int, PetscInt);
PETSC_EXTERN PetscErrorCode VecViewPlex_ExodusII_Nodal_Internal(Vec, int, int);
PETSC_EXTERN PetscErrorCode VecLoadPlex_ExodusII_Nodal_Internal(Vec, int, int);
PETSC_EXTERN PetscErrorCode VecViewPlex_ExodusII_Zonal_Internal(Vec, int, int);
PETSC_EXTERN PetscErrorCode VecLoadPlex_ExodusII_Zonal_Internal(Vec, int, int);
PETSC_INTERN PetscErrorCode DMPlexVTKGetCellType_Internal(DM,PetscInt,PetscInt,PetscInt*);
PETSC_INTERN PetscErrorCode DMPlexGetAdjacency_Internal(DM,PetscInt,PetscBool,PetscBool,PetscBool,PetscInt*,PetscInt*[]);
PETSC_INTERN PetscErrorCode DMPlexGetRawFaces_Internal(DM,DMPolytopeType,const PetscInt[],PetscInt*,const DMPolytopeType*[],const PetscInt*[],const PetscInt*[]);
PETSC_INTERN PetscErrorCode DMPlexRestoreRawFaces_Internal(DM,DMPolytopeType,const PetscInt[],PetscInt*,const DMPolytopeType*[],const PetscInt*[],const PetscInt*[]);
PETSC_INTERN PetscErrorCode CellRefinerInCellTest_Internal(DMPolytopeType, const PetscReal[], PetscBool *);
PETSC_INTERN PetscErrorCode DMPlexComputeCellType_Internal(DM, PetscInt, PetscInt, DMPolytopeType *);
PETSC_INTERN PetscErrorCode DMPlexCreateCellTypeOrder_Internal(DMPolytopeType, PetscInt *[], PetscInt *[]);
PETSC_INTERN PetscErrorCode DMPlexVecSetFieldClosure_Internal(DM, PetscSection, Vec, PetscBool[], PetscInt, PetscInt, const PetscInt[], DMLabel, PetscInt, const PetscScalar[], InsertMode);
PETSC_INTERN PetscErrorCode DMPlexProjectConstraints_Internal(DM, Vec, Vec);
PETSC_EXTERN PetscErrorCode DMPlexCreateReferenceTree_SetTree(DM, PetscSection, PetscInt[], PetscInt[]);
PETSC_EXTERN PetscErrorCode DMPlexCreateReferenceTree_Union(DM,DM,const char *,DM*);
PETSC_EXTERN PetscErrorCode DMPlexComputeInterpolatorTree(DM,DM,PetscSF,PetscInt *,Mat);
PETSC_EXTERN PetscErrorCode DMPlexComputeInjectorTree(DM,DM,PetscSF,PetscInt *,Mat);
PETSC_EXTERN PetscErrorCode DMPlexAnchorsModifyMat(DM,PetscSection,PetscInt,PetscInt,const PetscInt[],const PetscInt ***,const PetscScalar[],PetscInt*,PetscInt*,PetscInt*[],PetscScalar*[],PetscInt[],PetscBool);
PETSC_EXTERN PetscErrorCode indicesPoint_private(PetscSection,PetscInt,PetscInt,PetscInt *,PetscBool,PetscInt,PetscInt []);
PETSC_EXTERN PetscErrorCode indicesPointFields_private(PetscSection,PetscInt,PetscInt,PetscInt [],PetscBool,PetscInt,PetscInt []);
PETSC_INTERN PetscErrorCode DMPlexLocatePoint_Internal(DM,PetscInt,const PetscScalar [],PetscInt,PetscInt *);
/* these two are PETSC_EXTERN just because of src/dm/impls/plex/tests/ex18.c */
PETSC_EXTERN PetscErrorCode DMPlexOrientCell_Internal(DM,PetscInt,PetscInt,PetscBool);
PETSC_EXTERN PetscErrorCode DMPlexOrientInterface_Internal(DM);

/* Applications may use this function */
PETSC_EXTERN PetscErrorCode DMPlexCreateNumbering_Plex(DM, PetscInt, PetscInt, PetscInt, PetscInt *, PetscSF, IS *);

PETSC_INTERN PetscErrorCode DMPlexCreateCellNumbering_Internal(DM, PetscBool, IS *);
PETSC_INTERN PetscErrorCode DMPlexCreateVertexNumbering_Internal(DM, PetscBool, IS *);
PETSC_INTERN PetscErrorCode DMPlexRefine_Internal(DM, DMLabel, DM *);
PETSC_INTERN PetscErrorCode DMPlexCoarsen_Internal(DM, DMLabel, DM *);
PETSC_INTERN PetscErrorCode DMCreateMatrix_Plex(DM, Mat*);

PETSC_INTERN PetscErrorCode DMPlexGetOverlap_Plex(DM, PetscInt *);

/* invert dihedral symmetry: return a^-1,
 * using the representation described in
 * DMPlexGetConeOrientation() */
PETSC_STATIC_INLINE PetscInt DihedralInvert(PetscInt N, PetscInt a)
{
  return (a <= 0) ? a : (N - a);
}

/* invert dihedral symmetry: return b * a,
 * using the representation described in
 * DMPlexGetConeOrientation() */
PETSC_STATIC_INLINE PetscInt DihedralCompose(PetscInt N, PetscInt a, PetscInt b)
{
  if (!N) return 0;
  return  (a >= 0) ?
         ((b >= 0) ? ((a + b) % N) : -(((a - b - 1) % N) + 1)) :
         ((b >= 0) ? -(((N - b - a - 1) % N) + 1) : ((N + b - a) % N));
}

/* swap dihedral symmetries: return b * a^-1,
 * using the representation described in
 * DMPlexGetConeOrientation() */
PETSC_STATIC_INLINE PetscInt DihedralSwap(PetscInt N, PetscInt a, PetscInt b)
{
  return DihedralCompose(N,DihedralInvert(N,a),b);
}

PETSC_EXTERN PetscErrorCode DMPlexComputeResidual_Internal(DM, IS , PetscReal, Vec, Vec, PetscReal, Vec, void *);
PETSC_EXTERN PetscErrorCode DMPlexComputeResidual_Hybrid_Internal(DM, IS , PetscReal, Vec, Vec, PetscReal, Vec, void *);
PETSC_EXTERN PetscErrorCode DMPlexComputeJacobian_Internal(DM, IS, PetscReal, PetscReal, Vec, Vec, Mat, Mat, void *);
PETSC_EXTERN PetscErrorCode DMPlexComputeJacobian_Hybrid_Internal(DM, IS, PetscReal, PetscReal, Vec, Vec, Mat, Mat, void *);
PETSC_EXTERN PetscErrorCode DMPlexReconstructGradients_Internal(DM, PetscFV, PetscInt, PetscInt, Vec, Vec, Vec, Vec);

/* Matvec with A in row-major storage, x and y can be aliased */
PETSC_STATIC_INLINE void DMPlex_Mult2D_Internal(const PetscScalar A[], PetscInt ldx, const PetscScalar x[], PetscScalar y[])
{
  PetscScalar z[2];
  z[0] = x[0]; z[1] = x[ldx];
  y[0]   = A[0]*z[0] + A[1]*z[1];
  y[ldx] = A[2]*z[0] + A[3]*z[1];
  (void)PetscLogFlops(6.0);
}
PETSC_STATIC_INLINE void DMPlex_Mult3D_Internal(const PetscScalar A[], PetscInt ldx, const PetscScalar x[], PetscScalar y[])
{
  PetscScalar z[3];
  z[0] = x[0]; z[1] = x[ldx]; z[2] = x[ldx*2];
  y[0]     = A[0]*z[0] + A[1]*z[1] + A[2]*z[2];
  y[ldx]   = A[3]*z[0] + A[4]*z[1] + A[5]*z[2];
  y[ldx*2] = A[6]*z[0] + A[7]*z[1] + A[8]*z[2];
  (void)PetscLogFlops(15.0);
}
PETSC_STATIC_INLINE void DMPlex_MultTranspose2D_Internal(const PetscScalar A[], PetscInt ldx, const PetscScalar x[], PetscScalar y[])
{
  PetscScalar z[2];
  z[0] = x[0]; z[1] = x[ldx];
  y[0]   = A[0]*z[0] + A[2]*z[1];
  y[ldx] = A[1]*z[0] + A[3]*z[1];
  (void)PetscLogFlops(6.0);
}
PETSC_STATIC_INLINE void DMPlex_MultTranspose3D_Internal(const PetscScalar A[], PetscInt ldx, const PetscScalar x[], PetscScalar y[])
{
  PetscScalar z[3];
  z[0] = x[0]; z[1] = x[ldx]; z[2] = x[ldx*2];
  y[0]     = A[0]*z[0] + A[3]*z[1] + A[6]*z[2];
  y[ldx]   = A[1]*z[0] + A[4]*z[1] + A[7]*z[2];
  y[ldx*2] = A[2]*z[0] + A[5]*z[1] + A[8]*z[2];
  (void)PetscLogFlops(15.0);
}
PETSC_STATIC_INLINE void DMPlex_Mult2DReal_Internal(const PetscReal A[], PetscInt ldx, const PetscScalar x[], PetscScalar y[])
{
  PetscScalar z[2];
  z[0] = x[0]; z[1] = x[ldx];
  y[0]   = A[0]*z[0] + A[1]*z[1];
  y[ldx] = A[2]*z[0] + A[3]*z[1];
  (void)PetscLogFlops(6.0);
}
PETSC_STATIC_INLINE void DMPlex_Mult3DReal_Internal(const PetscReal A[], PetscInt ldx, const PetscScalar x[], PetscScalar y[])
{
  PetscScalar z[3];
  z[0] = x[0]; z[1] = x[ldx]; z[2] = x[ldx*2];
  y[0]     = A[0]*z[0] + A[1]*z[1] + A[2]*z[2];
  y[ldx]   = A[3]*z[0] + A[4]*z[1] + A[5]*z[2];
  y[ldx*2] = A[6]*z[0] + A[7]*z[1] + A[8]*z[2];
  (void)PetscLogFlops(15.0);
}
PETSC_STATIC_INLINE void DMPlex_MultTransposeReal_Internal(const PetscReal A[], PetscInt m, PetscInt n, PetscInt ldx, const PetscScalar x[], PetscScalar y[])
{
  PetscScalar z[3];
  PetscInt    i, j;
  for (i = 0; i < m; ++i) z[i] = x[i*ldx];
  for (j = 0; j < n; ++j) {
    const PetscInt l = j*ldx;
    y[l] = 0;
    for (i = 0; i < m; ++i) {
      y[l] += A[j*n+i]*z[i];
    }
  }
  (void)PetscLogFlops(2*m*n);
}
PETSC_STATIC_INLINE void DMPlex_MultTranspose2DReal_Internal(const PetscReal A[], PetscInt ldx, const PetscScalar x[], PetscScalar y[])
{
  PetscScalar z[2];
  z[0] = x[0]; z[1] = x[ldx];
  y[0]   = A[0]*z[0] + A[2]*z[1];
  y[ldx] = A[1]*z[0] + A[3]*z[1];
  (void)PetscLogFlops(6.0);
}
PETSC_STATIC_INLINE void DMPlex_MultTranspose3DReal_Internal(const PetscReal A[], PetscInt ldx, const PetscScalar x[], PetscScalar y[])
{
  PetscScalar z[3];
  z[0] = x[0]; z[1] = x[ldx]; z[2] = x[ldx*2];
  y[0]     = A[0]*z[0] + A[3]*z[1] + A[6]*z[2];
  y[ldx]   = A[1]*z[0] + A[4]*z[1] + A[7]*z[2];
  y[ldx*2] = A[2]*z[0] + A[5]*z[1] + A[8]*z[2];
  (void)PetscLogFlops(15.0);
}

PETSC_STATIC_INLINE void DMPlex_MatMult2D_Internal(const PetscScalar A[], PetscInt n, PetscInt ldb, const PetscScalar B[], PetscScalar C[])
{
  PetscInt j;
  for (j = 0; j < n; ++j) {
    PetscScalar z[2];
    z[0] = B[0+j]; z[1] = B[1*ldb+j];
    DMPlex_Mult2D_Internal(A, 1, z, z);
    C[0+j] = z[0]; C[1*ldb+j] = z[1];
  }
  (void)PetscLogFlops(8.0*n);
}
PETSC_STATIC_INLINE void DMPlex_MatMult3D_Internal(const PetscScalar A[], PetscInt n, PetscInt ldb, const PetscScalar B[], PetscScalar C[])
{
  PetscInt j;
  for (j = 0; j < n; ++j) {
    PetscScalar z[3];
    z[0] = B[0+j]; z[1] = B[1*ldb+j]; z[2] = B[2*ldb+j];
    DMPlex_Mult3D_Internal(A, 1, z, z);
    C[0+j] = z[0]; C[1*ldb+j] = z[1]; C[2*ldb+j] = z[2];
  }
  (void)PetscLogFlops(8.0*n);
}
PETSC_STATIC_INLINE void DMPlex_MatMultTranspose2D_Internal(const PetscScalar A[], PetscInt n, PetscInt ldb, const PetscScalar B[], PetscScalar C[])
{
  PetscInt j;
  for (j = 0; j < n; ++j) {
    PetscScalar z[2];
    z[0] = B[0+j]; z[1] = B[1*ldb+j];
    DMPlex_MultTranspose2D_Internal(A, 1, z, z);
    C[0+j] = z[0]; C[1*ldb+j] = z[1];
  }
  (void)PetscLogFlops(8.0*n);
}
PETSC_STATIC_INLINE void DMPlex_MatMultTranspose3D_Internal(const PetscScalar A[], PetscInt n, PetscInt ldb, const PetscScalar B[], PetscScalar C[])
{
  PetscInt j;
  for (j = 0; j < n; ++j) {
    PetscScalar z[3];
    z[0] = B[0+j]; z[1] = B[1*ldb+j]; z[2] = B[2*ldb+j];
    DMPlex_MultTranspose3D_Internal(A, 1, z, z);
    C[0+j] = z[0]; C[1*ldb+j] = z[1]; C[2*ldb+j] = z[2];
  }
  (void)PetscLogFlops(8.0*n);
}

PETSC_STATIC_INLINE void DMPlex_MatMultLeft2D_Internal(const PetscScalar A[], PetscInt m, PetscInt ldb, const PetscScalar B[], PetscScalar C[])
{
  PetscInt j;
  for (j = 0; j < m; ++j) {
    DMPlex_MultTranspose2D_Internal(A, 1, &B[j*ldb], &C[j*ldb]);
  }
  (void)PetscLogFlops(8.0*m);
}
PETSC_STATIC_INLINE void DMPlex_MatMultLeft3D_Internal(const PetscScalar A[], PetscInt m, PetscInt ldb, const PetscScalar B[], PetscScalar C[])
{
  PetscInt j;
  for (j = 0; j < m; ++j) {
    DMPlex_MultTranspose3D_Internal(A, 1, &B[j*ldb], &C[j*ldb]);
  }
  (void)PetscLogFlops(8.0*m);
}
PETSC_STATIC_INLINE void DMPlex_MatMultTransposeLeft2D_Internal(const PetscScalar A[], PetscInt m, PetscInt ldb, const PetscScalar B[], PetscScalar C[])
{
  PetscInt j;
  for (j = 0; j < m; ++j) {
    DMPlex_Mult2D_Internal(A, 1, &B[j*ldb], &C[j*ldb]);
  }
  (void)PetscLogFlops(8.0*m);
}
PETSC_STATIC_INLINE void DMPlex_MatMultTransposeLeft3D_Internal(const PetscScalar A[], PetscInt m, PetscInt ldb, const PetscScalar B[], PetscScalar C[])
{
  PetscInt j;
  for (j = 0; j < m; ++j) {
    DMPlex_Mult3D_Internal(A, 1, &B[j*ldb], &C[j*ldb]);
  }
  (void)PetscLogFlops(8.0*m);
}

PETSC_STATIC_INLINE void DMPlex_Transpose2D_Internal(PetscScalar A[])
{
  PetscScalar tmp;
  tmp = A[1]; A[1] = A[2]; A[2] = tmp;
}
PETSC_STATIC_INLINE void DMPlex_Transpose3D_Internal(PetscScalar A[])
{
  PetscScalar tmp;
  tmp = A[1]; A[1] = A[3]; A[3] = tmp;
  tmp = A[2]; A[2] = A[6]; A[6] = tmp;
  tmp = A[5]; A[5] = A[7]; A[7] = tmp;
}

PETSC_STATIC_INLINE void DMPlex_Invert2D_Internal(PetscReal invJ[], PetscReal J[], PetscReal detJ)
{
  const PetscReal invDet = 1.0/detJ;

  invJ[0] =  invDet*J[3];
  invJ[1] = -invDet*J[1];
  invJ[2] = -invDet*J[2];
  invJ[3] =  invDet*J[0];
  (void)PetscLogFlops(5.0);
}

PETSC_STATIC_INLINE void DMPlex_Invert3D_Internal(PetscReal invJ[], PetscReal J[], PetscReal detJ)
{
  const PetscReal invDet = 1.0/detJ;

  invJ[0*3+0] = invDet*(J[1*3+1]*J[2*3+2] - J[1*3+2]*J[2*3+1]);
  invJ[0*3+1] = invDet*(J[0*3+2]*J[2*3+1] - J[0*3+1]*J[2*3+2]);
  invJ[0*3+2] = invDet*(J[0*3+1]*J[1*3+2] - J[0*3+2]*J[1*3+1]);
  invJ[1*3+0] = invDet*(J[1*3+2]*J[2*3+0] - J[1*3+0]*J[2*3+2]);
  invJ[1*3+1] = invDet*(J[0*3+0]*J[2*3+2] - J[0*3+2]*J[2*3+0]);
  invJ[1*3+2] = invDet*(J[0*3+2]*J[1*3+0] - J[0*3+0]*J[1*3+2]);
  invJ[2*3+0] = invDet*(J[1*3+0]*J[2*3+1] - J[1*3+1]*J[2*3+0]);
  invJ[2*3+1] = invDet*(J[0*3+1]*J[2*3+0] - J[0*3+0]*J[2*3+1]);
  invJ[2*3+2] = invDet*(J[0*3+0]*J[1*3+1] - J[0*3+1]*J[1*3+0]);
  (void)PetscLogFlops(37.0);
}

PETSC_STATIC_INLINE void DMPlex_Det2D_Internal(PetscReal *detJ, const PetscReal J[])
{
  *detJ = J[0]*J[3] - J[1]*J[2];
  (void)PetscLogFlops(3.0);
}

PETSC_STATIC_INLINE void DMPlex_Det3D_Internal(PetscReal *detJ, const PetscReal J[])
{
  *detJ = (J[0*3+0]*(J[1*3+1]*J[2*3+2] - J[1*3+2]*J[2*3+1]) +
           J[0*3+1]*(J[1*3+2]*J[2*3+0] - J[1*3+0]*J[2*3+2]) +
           J[0*3+2]*(J[1*3+0]*J[2*3+1] - J[1*3+1]*J[2*3+0]));
  (void)PetscLogFlops(12.0);
}

PETSC_STATIC_INLINE void DMPlex_Det2D_Scalar_Internal(PetscReal *detJ, const PetscScalar J[])
{
  *detJ = PetscRealPart(J[0])*PetscRealPart(J[3]) - PetscRealPart(J[1])*PetscRealPart(J[2]);
  (void)PetscLogFlops(3.0);
}

PETSC_STATIC_INLINE void DMPlex_Det3D_Scalar_Internal(PetscReal *detJ, const PetscScalar J[])
{
  *detJ = (PetscRealPart(J[0*3+0])*(PetscRealPart(J[1*3+1])*PetscRealPart(J[2*3+2]) - PetscRealPart(J[1*3+2])*PetscRealPart(J[2*3+1])) +
           PetscRealPart(J[0*3+1])*(PetscRealPart(J[1*3+2])*PetscRealPart(J[2*3+0]) - PetscRealPart(J[1*3+0])*PetscRealPart(J[2*3+2])) +
           PetscRealPart(J[0*3+2])*(PetscRealPart(J[1*3+0])*PetscRealPart(J[2*3+1]) - PetscRealPart(J[1*3+1])*PetscRealPart(J[2*3+0])));
  (void)PetscLogFlops(12.0);
}

PETSC_STATIC_INLINE void DMPlex_WaxpyD_Internal(PetscInt dim, PetscReal a, const PetscReal *x, const PetscReal *y, PetscReal *w) {PetscInt d; for (d = 0; d < dim; ++d) w[d] = a*x[d] + y[d];}

PETSC_STATIC_INLINE PetscReal DMPlex_DotD_Internal(PetscInt dim, const PetscScalar *x, const PetscReal *y) {PetscReal sum = 0.0; PetscInt d; for (d = 0; d < dim; ++d) sum += PetscRealPart(x[d])*y[d]; return sum;}

PETSC_STATIC_INLINE PetscReal DMPlex_DotRealD_Internal(PetscInt dim, const PetscReal *x, const PetscReal *y) {PetscReal sum = 0.0; PetscInt d; for (d = 0; d < dim; ++d) sum += x[d]*y[d]; return sum;}

PETSC_STATIC_INLINE PetscReal DMPlex_NormD_Internal(PetscInt dim, const PetscReal *x) {PetscReal sum = 0.0; PetscInt d; for (d = 0; d < dim; ++d) sum += x[d]*x[d]; return PetscSqrtReal(sum);}

PETSC_STATIC_INLINE PetscErrorCode DMPlexFixFaceOrientations_Translate_Private(PetscInt ornt, PetscInt *start, PetscBool *reverse)
{
  PetscFunctionBegin;
  *reverse = (ornt < 0) ? PETSC_TRUE : PETSC_FALSE;
  *start = *reverse ? -(ornt+1) : ornt;
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode DMPlexFixFaceOrientations_Combine_Private(PetscInt coneSize, PetscInt origStart, PetscBool origReverse, PetscInt rotateStart, PetscBool rotateReverse, PetscInt *newStart, PetscBool *newReverse)
{
  PetscFunctionBegin;
  *newReverse = (origReverse == rotateReverse) ? PETSC_FALSE : PETSC_TRUE;
  *newStart = rotateReverse ? (coneSize + rotateStart - origStart) : (coneSize + origStart - rotateStart);
  *newStart %= coneSize;
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode DMPlexFixFaceOrientations_TranslateBack_Private(PetscInt coneSize, PetscInt start, PetscBool reverse, PetscInt *ornt)
{
  PetscFunctionBegin;
  if (coneSize < 3) {
    /* edges just get flipped if start == 1 regardless direction */
    *ornt = start ? -2 : 0;
  } else {
    *ornt = reverse ? -(start+1) : start;
  }
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode DMPlexFixFaceOrientations_Permute_Private(PetscInt n, const PetscInt arr[], PetscInt start, PetscBool reverse, PetscInt newarr[])
{
  PetscInt i;

  PetscFunctionBegin;
  if (reverse) {for (i=0; i<n; i++) newarr[i] = arr[(n+start-i)%n];}
  else         {for (i=0; i<n; i++) newarr[i] = arr[(start+i)%n];}
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode DMPlexGetPointDualSpaceFEM(DM,PetscInt,PetscInt,PetscDualSpace *);
PETSC_INTERN PetscErrorCode DMPlexGetIndicesPoint_Internal(PetscSection,PetscBool,PetscInt,PetscInt,PetscInt *,PetscBool,const PetscInt[],const PetscInt[],PetscInt[]);
PETSC_INTERN PetscErrorCode DMPlexGetIndicesPointFields_Internal(PetscSection,PetscBool,PetscInt,PetscInt,PetscInt[],PetscBool,const PetscInt***,PetscInt,const PetscInt[],PetscInt[]);
PETSC_INTERN PetscErrorCode DMPlexGetCompressedClosure(DM, PetscSection, PetscInt, PetscInt *, PetscInt **, PetscSection *, IS *, const PetscInt **);
PETSC_INTERN PetscErrorCode DMPlexRestoreCompressedClosure(DM, PetscSection, PetscInt, PetscInt *, PetscInt **, PetscSection *, IS *, const PetscInt **);

PETSC_EXTERN PetscErrorCode DMSNESGetFEGeom(DMField, IS, PetscQuadrature, PetscBool, PetscFEGeom **);
PETSC_EXTERN PetscErrorCode DMSNESRestoreFEGeom(DMField, IS, PetscQuadrature, PetscBool, PetscFEGeom **);
PETSC_EXTERN PetscErrorCode DMPlexComputeResidual_Patch_Internal(DM, PetscSection, IS, PetscReal, Vec, Vec, Vec, void *);
PETSC_EXTERN PetscErrorCode DMPlexComputeJacobian_Patch_Internal(DM, PetscSection, PetscSection, IS, PetscReal, PetscReal, Vec, Vec, Mat, Mat, void *);
PETSC_INTERN PetscErrorCode DMCreateSubDomainDM_Plex(DM,DMLabel,PetscInt,IS*,DM*);
PETSC_INTERN PetscErrorCode DMPlexBasisTransformPoint_Internal(DM, DM, Vec, PetscInt, PetscBool[], PetscBool, PetscScalar *);
PETSC_EXTERN PetscErrorCode DMPlexBasisTransformPointTensor_Internal(DM, DM, Vec, PetscInt, PetscBool, PetscInt, PetscScalar *);
PETSC_INTERN PetscErrorCode DMPlexBasisTransformApplyReal_Internal(DM, const PetscReal[], PetscBool, PetscInt, const PetscReal *, PetscReal *, void *);
PETSC_INTERN PetscErrorCode DMPlexBasisTransformApply_Internal(DM, const PetscReal[], PetscBool, PetscInt, const PetscScalar *, PetscScalar *, void *);
PETSC_INTERN PetscErrorCode DMCreateNeumannOverlap_Plex(DM, IS*, Mat*, PetscErrorCode (**)(Mat, PetscReal, Vec, Vec, PetscReal, IS, void*), void **);

#endif /* _PLEXIMPL_H */
