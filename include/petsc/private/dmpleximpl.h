#if !defined(_PLEXIMPL_H)
#define _PLEXIMPL_H

#include <petscmat.h>       /*I      "petscmat.h"          I*/
#include <petscdmplex.h> /*I      "petscdmplex.h"    I*/
#include <petscbt.h>
#include <petscsf.h>
#include <petsc/private/dmimpl.h>
#include <petsc/private/isimpl.h>     /* for inline access to atlasOff */
#include <../src/sys/utils/hash.h>

PETSC_EXTERN PetscLogEvent DMPLEX_Interpolate, PETSCPARTITIONER_Partition, DMPLEX_Distribute, DMPLEX_DistributeCones, DMPLEX_DistributeLabels, DMPLEX_DistributeSF, DMPLEX_DistributeOverlap, DMPLEX_DistributeField, DMPLEX_DistributeData, DMPLEX_Migrate, DMPLEX_GlobalToNaturalBegin, DMPLEX_GlobalToNaturalEnd, DMPLEX_NaturalToGlobalBegin, DMPLEX_NaturalToGlobalEnd, DMPLEX_Stratify, DMPLEX_Preallocate, DMPLEX_ResidualFEM, DMPLEX_JacobianFEM, DMPLEX_InterpolatorFEM, DMPLEX_InjectorFEM, DMPLEX_IntegralFEM, DMPLEX_CreateGmsh;

PETSC_EXTERN PetscBool      PetscPartitionerRegisterAllCalled;
PETSC_EXTERN PetscErrorCode PetscPartitionerRegisterAll(void);
PETSC_EXTERN PetscErrorCode PetscPartitionerSetTypeFromOptions_Internal(PetscPartitioner);

typedef enum {REFINER_NOOP = 0,
              REFINER_SIMPLEX_1D,
              REFINER_SIMPLEX_2D,
              REFINER_HYBRID_SIMPLEX_2D,
              REFINER_HEX_2D,
              REFINER_HYBRID_HEX_2D,
              REFINER_SIMPLEX_3D,
              REFINER_HYBRID_SIMPLEX_3D,
              REFINER_HEX_3D,
              REFINER_HYBRID_HEX_3D} CellRefiner;

typedef struct _PetscPartitionerOps *PetscPartitionerOps;
struct _PetscPartitionerOps {
  PetscErrorCode (*setfromoptions)(PetscPartitioner);
  PetscErrorCode (*setup)(PetscPartitioner);
  PetscErrorCode (*view)(PetscPartitioner,PetscViewer);
  PetscErrorCode (*destroy)(PetscPartitioner);
  PetscErrorCode (*partition)(PetscPartitioner, DM, PetscInt, PetscInt, PetscInt[], PetscInt[], PetscSection, IS *);
};

struct _p_PetscPartitioner {
  PETSCHEADER(struct _PetscPartitionerOps);
  void           *data;             /* Implementation object */
  PetscInt        height;           /* Height of points to partition into non-overlapping subsets */
};

typedef struct {
  PetscInt dummy;
} PetscPartitioner_Chaco;

typedef struct {
  PetscInt dummy;
} PetscPartitioner_ParMetis;

typedef struct {
  PetscSection section;   /* Sizes for each partition */
  IS           partition; /* Points in each partition */
} PetscPartitioner_Shell;

typedef struct {
  PetscInt dummy;
} PetscPartitioner_Simple;

typedef struct {
  PetscInt dummy;
} PetscPartitioner_Gather;

/* Utility struct to store the contents of a Gmsh file in memory */
typedef struct {
  PetscInt dim;      /* Entity dimension */
  PetscInt id;       /* Element number */
  PetscInt numNodes; /* Size of node array */
  int nodes[8];      /* Node array */
  PetscInt numTags;  /* Size of tag array */
  int tags[4];       /* Tag array */
} GmshElement;

/* Utility struct to store the contents of a Fluent file in memory */
typedef struct {
  int   index;    /* Type of section */
  int   zoneID;
  int   first;
  int   last;
  int   type;
  int   nd;       /* Either ND or element-type */
  void *data;
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

typedef struct {
  PetscInt             refct;

  /* Sieve */
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
  PetscInt             hybridPointMax[8]; /* Allow segregation of some points, each dimension has a divider (used in VTK output and refinement) */

  PetscInt            *facesTmp;          /* Work space for faces operation */

  /* Hierarchy */
  PetscBool            regularRefinement; /* This flag signals that we are a regular refinement of coarseMesh */

  /* Generation */
  char                *tetgenOpts;
  char                *triangleOpts;
  PetscPartitioner     partitioner;

  /* Submesh */
  DMLabel              subpointMap;       /* Label each original mesh point in the submesh with its depth, subpoint are the implicit numbering */

  /* Labels and numbering */
  PetscObjectState     depthState;        /* State of depth label, so that we can determine if a user changes it */
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

  /* Adjacency */
  PetscBool            useCone;           /* Use cone() first when defining adjacency */
  PetscBool            useClosure;        /* Use the transitive closure when defining adjacency */
  PetscBool            useAnchors;        /* Replace constrained points with their anchors in adjacency lists */

  /* Projection */
  PetscInt             maxProjectionHeight; /* maximum height of cells used in DMPlexProject functions */

  /* Output */
  PetscInt             vtkCellHeight;            /* The height of cells for output, default is 0 */
  PetscReal            scale[NUM_PETSC_UNITS];   /* The scale for each SI unit */

  /* Geometry */
  PetscReal            minradius;         /* Minimum distance from cell centroid to face */
  PetscBool            useHashLocation;   /* Use grid hashing for point location */
  PetscGridHash        lbox;              /* Local box for searching */

  /* Debugging */
  PetscBool            printSetValues;
  PetscInt             printFEM;
  PetscReal            printTol;
} DM_Plex;

PETSC_EXTERN PetscErrorCode DMPlexVTKWriteAll_VTU(DM,PetscViewer);
PETSC_EXTERN PetscErrorCode DMPlexVTKGetCellType(DM,PetscInt,PetscInt,PetscInt*);
PETSC_EXTERN PetscErrorCode VecView_Plex_Local(Vec,PetscViewer);
PETSC_EXTERN PetscErrorCode VecView_Plex_Native(Vec,PetscViewer);
PETSC_EXTERN PetscErrorCode VecView_Plex(Vec,PetscViewer);
PETSC_EXTERN PetscErrorCode VecLoad_Plex_Local(Vec,PetscViewer);
PETSC_EXTERN PetscErrorCode VecLoad_Plex_Native(Vec,PetscViewer);
PETSC_EXTERN PetscErrorCode VecLoad_Plex(Vec,PetscViewer);
PETSC_EXTERN PetscErrorCode DMPlexGetFieldType_Internal(DM, PetscSection, PetscInt, PetscInt *, PetscInt *, PetscViewerVTKFieldType *);
#if defined(PETSC_HAVE_HDF5)
PETSC_EXTERN PetscErrorCode VecView_Plex_Local_HDF5(Vec, PetscViewer);
PETSC_EXTERN PetscErrorCode VecView_Plex_HDF5(Vec, PetscViewer);
PETSC_EXTERN PetscErrorCode VecLoad_Plex_HDF5(Vec, PetscViewer);
PETSC_EXTERN PetscErrorCode VecView_Plex_HDF5_Native(Vec, PetscViewer);
PETSC_EXTERN PetscErrorCode VecLoad_Plex_HDF5_Native(Vec, PetscViewer);
PETSC_EXTERN PetscErrorCode DMPlexView_HDF5(DM, PetscViewer);
PETSC_EXTERN PetscErrorCode DMPlexLoad_HDF5(DM, PetscViewer);
#endif

PETSC_EXTERN PetscErrorCode DMPlexGetAdjacency_Internal(DM,PetscInt,PetscBool,PetscBool,PetscBool,PetscInt*,PetscInt*[]);
PETSC_EXTERN PetscErrorCode DMPlexGetFaces_Internal(DM,PetscInt,PetscInt,PetscInt*,PetscInt*,const PetscInt*[]);
PETSC_EXTERN PetscErrorCode DMPlexGetRawFaces_Internal(DM,PetscInt,PetscInt,const PetscInt[], PetscInt*,PetscInt*,const PetscInt*[]);
PETSC_EXTERN PetscErrorCode DMPlexRestoreFaces_Internal(DM,PetscInt,PetscInt,PetscInt*,PetscInt*,const PetscInt*[]);
PETSC_EXTERN PetscErrorCode DMPlexRefineUniform_Internal(DM,CellRefiner,DM*);
PETSC_EXTERN PetscErrorCode DMPlexGetCellRefiner_Internal(DM,CellRefiner*);
PETSC_EXTERN PetscErrorCode CellRefinerGetAffineTransforms_Internal(CellRefiner, PetscInt *, PetscReal *[], PetscReal *[], PetscReal *[]);
PETSC_EXTERN PetscErrorCode CellRefinerRestoreAffineTransforms_Internal(CellRefiner, PetscInt *, PetscReal *[], PetscReal *[], PetscReal *[]);
PETSC_EXTERN PetscErrorCode CellRefinerInCellTest_Internal(CellRefiner, const PetscReal[], PetscBool *);
PETSC_EXTERN PetscErrorCode DMPlexCreateGmsh_ReadElement(PetscViewer, PetscInt, PetscBool, PetscBool, GmshElement **);
PETSC_EXTERN PetscErrorCode DMPlexInvertCell_Internal(PetscInt, PetscInt, PetscInt[]);
PETSC_EXTERN PetscErrorCode DMPlexVecSetFieldClosure_Internal(DM, PetscSection, Vec, PetscBool[], PetscInt, const PetscScalar[], InsertMode);
PETSC_EXTERN PetscErrorCode DMPlexProjectConstraints_Internal(DM, Vec, Vec);
PETSC_EXTERN PetscErrorCode DMSetFromOptions_NonRefinement_Plex(PetscOptionItems*,DM);
PETSC_EXTERN PetscErrorCode DMPlexLabelComplete_Internal(DM,DMLabel,PetscBool);
PETSC_EXTERN PetscErrorCode DMPlexCreateReferenceTree_Union(DM,DM,const char *,DM*);
PETSC_EXTERN PetscErrorCode DMPlexComputeInterpolatorTree(DM,DM,PetscSF,PetscInt *,Mat);
PETSC_EXTERN PetscErrorCode DMPlexComputeInjectorTree(DM,DM,PetscSF,PetscInt *,Mat);
PETSC_EXTERN PetscErrorCode DMPlexAnchorsModifyMat(DM,PetscSection,PetscInt,PetscInt,const PetscInt*,const PetscScalar*,PetscInt*,PetscInt*,PetscInt**,PetscScalar**,PetscInt*,PetscBool);
PETSC_EXTERN PetscErrorCode indicesPoint_private(PetscSection,PetscInt,PetscInt,PetscInt *,PetscBool,PetscInt,PetscInt []);
PETSC_EXTERN PetscErrorCode indicesPointFields_private(PetscSection,PetscInt,PetscInt,PetscInt [],PetscBool,PetscInt,PetscInt []);
PETSC_EXTERN PetscErrorCode DMPlexLocatePoint_Internal(DM,PetscInt,const PetscScalar [],PetscInt,PetscInt *);

PETSC_INTERN PetscErrorCode DMPlexCreateCellNumbering_Internal(DM, PetscBool, IS *);
PETSC_INTERN PetscErrorCode DMPlexCreateVertexNumbering_Internal(DM, PetscBool, IS *);

#undef __FUNCT__
#define __FUNCT__ "DihedralInvert"
/* invert dihedral symmetry: return a^-1,
 * using the representation described in
 * DMPlexGetConeOrientation() */
PETSC_STATIC_INLINE PetscInt DihedralInvert(PetscInt N, PetscInt a)
{
  return (a <= 0) ? a : (N - a);
}

#undef __FUNCT__
#define __FUNCT__ "DihedralCompose"
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

#undef __FUNCT__
#define __FUNCT__ "DihedralSwap"
/* swap dihedral symmetries: return b * a^-1,
 * using the representation described in
 * DMPlexGetConeOrientation() */
PETSC_STATIC_INLINE PetscInt DihedralSwap(PetscInt N, PetscInt a, PetscInt b)
{
  return DihedralCompose(N,DihedralInvert(N,a),b);
}

PETSC_EXTERN PetscErrorCode DMPlexComputeResidual_Internal(DM, PetscInt, PetscInt, PetscReal, Vec, Vec, Vec, void *);
PETSC_EXTERN PetscErrorCode DMPlexComputeJacobian_Internal(DM, PetscInt, PetscInt, PetscReal, PetscReal, Vec, Vec, Mat, Mat,void *);

#undef __FUNCT__
#define __FUNCT__ "DMPlex_Invert2D_Internal"
PETSC_STATIC_INLINE void DMPlex_Invert2D_Internal(PetscReal invJ[], PetscReal J[], PetscReal detJ)
{
  const PetscReal invDet = 1.0/detJ;

  invJ[0] =  invDet*J[3];
  invJ[1] = -invDet*J[1];
  invJ[2] = -invDet*J[2];
  invJ[3] =  invDet*J[0];
  (void)PetscLogFlops(5.0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlex_Invert3D_Internal"
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

#undef __FUNCT__
#define __FUNCT__ "DMPlex_Det2D_Internal"
PETSC_STATIC_INLINE void DMPlex_Det2D_Internal(PetscReal *detJ, PetscReal J[])
{
  *detJ = J[0]*J[3] - J[1]*J[2];
  (void)PetscLogFlops(3.0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlex_Det3D_Internal"
PETSC_STATIC_INLINE void DMPlex_Det3D_Internal(PetscReal *detJ, PetscReal J[])
{
  *detJ = (J[0*3+0]*(J[1*3+1]*J[2*3+2] - J[1*3+2]*J[2*3+1]) +
           J[0*3+1]*(J[1*3+2]*J[2*3+0] - J[1*3+0]*J[2*3+2]) +
           J[0*3+2]*(J[1*3+0]*J[2*3+1] - J[1*3+1]*J[2*3+0]));
  (void)PetscLogFlops(12.0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlex_WaxpyD_Internal"
PETSC_STATIC_INLINE void DMPlex_WaxpyD_Internal(PetscInt dim, PetscReal a, const PetscReal *x, const PetscReal *y, PetscReal *w) {PetscInt d; for (d = 0; d < dim; ++d) w[d] = a*x[d] + y[d];}

#undef __FUNCT__
#define __FUNCT__ "DMPlex_DotD_Internal"
PETSC_STATIC_INLINE PetscReal DMPlex_DotD_Internal(PetscInt dim, const PetscScalar *x, const PetscReal *y) {PetscReal sum = 0.0; PetscInt d; for (d = 0; d < dim; ++d) sum += PetscRealPart(x[d])*y[d]; return sum;}

#undef __FUNCT__
#define __FUNCT__ "DMPlex_DotRealD_Internal"
PETSC_STATIC_INLINE PetscReal DMPlex_DotRealD_Internal(PetscInt dim, const PetscReal *x, const PetscReal *y) {PetscReal sum = 0.0; PetscInt d; for (d = 0; d < dim; ++d) sum += x[d]*y[d]; return sum;}

#undef __FUNCT__
#define __FUNCT__ "DMPlex_NormD_Internal"
PETSC_STATIC_INLINE PetscReal DMPlex_NormD_Internal(PetscInt dim, const PetscReal *x) {PetscReal sum = 0.0; PetscInt d; for (d = 0; d < dim; ++d) sum += x[d]*x[d]; return PetscSqrtReal(sum);}


#undef __FUNCT__
#define __FUNCT__ "DMPlexGetLocalOffset_Private"
PETSC_STATIC_INLINE PetscErrorCode DMPlexGetLocalOffset_Private(DM dm, PetscInt point, PetscInt *start, PetscInt *end)
{
  PetscFunctionBeginHot;
#if defined(PETSC_USE_DEBUG)
  {
    PetscInt       dof;
    PetscErrorCode ierr;
    *start = *end = 0; /* Silence overzealous compiler warning */
    if (!dm->defaultSection) SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONG, "DM must have a default Section, see DMSetDefaultSection()");
    ierr = PetscSectionGetOffset(dm->defaultSection, point, start);CHKERRQ(ierr);
    ierr = PetscSectionGetDof(dm->defaultSection, point, &dof);CHKERRQ(ierr);
    *end = *start + dof;
  }
#else
  {
    const PetscSection s = dm->defaultSection;
    *start = s->atlasOff[point - s->pStart];
    *end   = *start + s->atlasDof[point - s->pStart];
  }
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexGetLocalFieldOffset_Private"
PETSC_STATIC_INLINE PetscErrorCode DMPlexGetLocalFieldOffset_Private(DM dm, PetscInt point, PetscInt field, PetscInt *start, PetscInt *end)
{
  PetscFunctionBegin;
#if defined(PETSC_USE_DEBUG)
  {
    PetscInt       dof;
    PetscErrorCode ierr;
    *start = *end = 0; /* Silence overzealous compiler warning */
    if (!dm->defaultSection) SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONG, "DM must have a default Section, see DMSetDefaultSection()");
    ierr = PetscSectionGetFieldOffset(dm->defaultSection, point, field, start);CHKERRQ(ierr);
    ierr = PetscSectionGetFieldDof(dm->defaultSection, point, field, &dof);CHKERRQ(ierr);
    *end = *start + dof;
  }
#else
  {
    const PetscSection s = dm->defaultSection->field[field];
    *start = s->atlasOff[point - s->pStart];
    *end   = *start + s->atlasDof[point - s->pStart];
  }
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexGetGlobalOffset_Private"
PETSC_STATIC_INLINE PetscErrorCode DMPlexGetGlobalOffset_Private(DM dm, PetscInt point, PetscInt *start, PetscInt *end)
{
  PetscFunctionBegin;
#if defined(PETSC_USE_DEBUG)
  {
    PetscErrorCode ierr;
    PetscInt       dof,cdof;
    *start = *end = 0; /* Silence overzealous compiler warning */
    if (!dm->defaultSection) SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONG, "DM must have a default Section, see DMSetDefaultSection()");
    if (!dm->defaultGlobalSection) SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONG, "DM must have a default global Section. It will be created automatically by DMGetDefaultGlobalSection()");
    ierr = PetscSectionGetOffset(dm->defaultGlobalSection, point, start);CHKERRQ(ierr);
    ierr = PetscSectionGetDof(dm->defaultGlobalSection, point, &dof);CHKERRQ(ierr);
    ierr = PetscSectionGetConstraintDof(dm->defaultGlobalSection, point, &cdof);CHKERRQ(ierr);
    *end = *start + dof-cdof;
  }
#else
  {
    const PetscSection s    = dm->defaultGlobalSection;
    const PetscInt     dof  = s->atlasDof[point - s->pStart];
    const PetscInt     cdof = s->bc ? s->bc->atlasDof[point - s->bc->pStart] : 0;
    *start = s->atlasOff[point - s->pStart];
    *end   = *start + dof-cdof;
  }
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexGetGlobalFieldOffset_Private"
PETSC_STATIC_INLINE PetscErrorCode DMPlexGetGlobalFieldOffset_Private(DM dm, PetscInt point, PetscInt field, PetscInt *start, PetscInt *end)
{
  PetscFunctionBegin;
#if defined(PETSC_USE_DEBUG)
  {
    PetscInt       loff, lfoff, fdof, fcdof, ffcdof, f;
    PetscErrorCode ierr;
    *start = *end = 0; /* Silence overzealous compiler warning */
    if (!dm->defaultSection) SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONG, "DM must have a default Section, see DMSetDefaultSection()");
    if (!dm->defaultGlobalSection) SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONG, "DM must have a default global Section. It will be crated automatically by DMGetDefaultGlobalSection()");
    ierr = PetscSectionGetOffset(dm->defaultGlobalSection, point, start);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(dm->defaultSection, point, &loff);CHKERRQ(ierr);
    ierr = PetscSectionGetFieldOffset(dm->defaultSection, point, field, &lfoff);CHKERRQ(ierr);
    ierr = PetscSectionGetFieldDof(dm->defaultSection, point, field, &fdof);CHKERRQ(ierr);
    ierr = PetscSectionGetFieldConstraintDof(dm->defaultSection, point, field, &fcdof);CHKERRQ(ierr);
    *start = *start < 0 ? *start - (lfoff-loff) : *start + lfoff-loff;
    for (f = 0; f < field; ++f) {
      ierr = PetscSectionGetFieldConstraintDof(dm->defaultSection, point, f, &ffcdof);CHKERRQ(ierr);
      *start = *start < 0 ? *start + ffcdof : *start - ffcdof;
    }
    *end   = *start < 0 ? *start - (fdof-fcdof) : *start + fdof-fcdof;
  }
#else
  {
    const PetscSection s     = dm->defaultSection;
    const PetscSection fs    = dm->defaultSection->field[field];
    const PetscSection gs    = dm->defaultGlobalSection;
    const PetscInt     loff  = s->atlasOff[point - s->pStart];
    const PetscInt     goff  = gs->atlasOff[point - s->pStart];
    const PetscInt     lfoff = fs->atlasOff[point - s->pStart];
    const PetscInt     fdof  = fs->atlasDof[point - s->pStart];
    const PetscInt     fcdof = fs->bc ? fs->bc->atlasDof[point - fs->bc->pStart] : 0;
    PetscInt           ffcdof = 0, f;

    for (f = 0; f < field; ++f) {
      const PetscSection ffs = dm->defaultSection->field[f];
      ffcdof += ffs->bc ? ffs->bc->atlasDof[point - ffs->bc->pStart] : 0;
    }
    *start = goff + (goff < 0 ? loff-lfoff + ffcdof : lfoff-loff - ffcdof);
    *end   = *start < 0 ? *start - (fdof-fcdof) : *start + fdof-fcdof;
  }
#endif
  PetscFunctionReturn(0);
}

#endif /* _PLEXIMPL_H */
