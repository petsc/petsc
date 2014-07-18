#if !defined(_PLEXIMPL_H)
#define _PLEXIMPL_H

#include <petscmat.h>       /*I      "petscmat.h"          I*/
#include <petscdmplex.h> /*I      "petscdmplex.h"    I*/
#include <petscbt.h>
#include <petsc-private/dmimpl.h>
#include <../src/sys/utils/hash.h>

PETSC_EXTERN PetscLogEvent DMPLEX_Interpolate, DMPLEX_Partition, DMPLEX_Distribute, DMPLEX_DistributeCones, DMPLEX_DistributeLabels, DMPLEX_DistributeSF, DMPLEX_DistributeField, DMPLEX_DistributeData, DMPLEX_Stratify, DMPLEX_Preallocate, DMPLEX_ResidualFEM, DMPLEX_JacobianFEM, DMPLEX_InterpolatorFEM, DMPLEX_InjectorFEM;

/* This is an integer map, in addition it is also a container class
   Design points:
     - Low storage is the most important design point
     - We want flexible insertion and deletion
     - We can live with O(log) query, but we need O(1) iteration over strata
*/
struct _n_DMLabel {
  PetscInt    refct;
  char       *name;           /* Label name */
  PetscInt    numStrata;      /* Number of integer values */
  PetscInt   *stratumValues;  /* Value of each stratum */
  /* Basic sorted array storage */
  PetscBool   arrayValid;     /* The array storage is valid (no additions need to be merged in) */
  PetscInt   *stratumOffsets; /* Offset of each stratum */
  PetscInt   *stratumSizes;   /* Size of each stratum */
  PetscInt   *points;         /* Points for each stratum, always sorted */
  /* Hashtable for fast insertion */
  PetscHashI *ht;             /* Hash table for fast insertion */
  /* Index for fast search */
  PetscInt    pStart, pEnd;   /* Bounds for index lookup */
  PetscBT     bt;             /* A bit-wise index */
  DMLabel     next;           /* Linked list */
};


struct _n_Boundary {
  const char *name;
  const char *labelname;
  DMLabel     label;
  PetscBool   essential;
  PetscInt    field;
  void      (*func)();
  PetscInt    numids;
  PetscInt   *ids;
  void       *ctx;
  DMBoundary  next;
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
  PetscInt             hybridPointMax[8]; /* Allow segregation of some points, each dimension has a divider (used in VTK output and refinement) */

  PetscInt            *facesTmp;          /* Work space for faces operation */

  /* Hierarchy */
  DM                   coarseMesh;        /* This mesh was obtained from coarse mesh using DMRefineHierarchy() */

  /* Submesh */
  DMLabel              subpointMap;       /* Label each original mesh point in the submesh with its depth, subpoint are the implicit numbering */

  /* Labels and numbering */
  DMLabel              labels;            /* Linked list of labels */
  DMLabel              depthLabel;
  IS                   globalVertexNumbers;
  IS                   globalCellNumbers;

  /* Constraints */
  PetscSection         anchorSection;     /* maps constrained points to anchor points */
  IS                   anchorIS;          /* anchors indexed by the above section */
  PetscSection         constraintSection; /* maps points to rows in the constraint matrix below */
  Mat                  constraintMat;     /* (m x n), m = constrained dofs in the default section, n = dofs in the default section */

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
  PetscBool            useConstraints;    /* Replace constrained points with their anchors in adjacency lists */

  /* Output */
  PetscInt             vtkCellHeight;            /* The height of cells for output, default is 0 */
  PetscReal            scale[NUM_PETSC_UNITS];   /* The scale for each SI unit */

  /* Problem definition */
  DMBoundary           boundary;          /* List of boundary conditions */

  /* Debugging */
  PetscBool            printSetValues;
  PetscInt             printFEM;
  PetscReal            printTol;
} DM_Plex;

PETSC_EXTERN PetscErrorCode DMPlexVTKWriteAll_VTU(DM,PetscViewer);
PETSC_EXTERN PetscErrorCode DMPlexVTKGetCellType(DM,PetscInt,PetscInt,PetscInt*);
PETSC_EXTERN PetscErrorCode VecView_Plex_Local(Vec,PetscViewer);
PETSC_EXTERN PetscErrorCode VecView_Plex(Vec,PetscViewer);
PETSC_EXTERN PetscErrorCode VecLoad_Plex_Local(Vec,PetscViewer);
PETSC_EXTERN PetscErrorCode VecLoad_Plex(Vec,PetscViewer);
PETSC_EXTERN PetscErrorCode DMPlexGetFieldType_Internal(DM, PetscSection, PetscInt, PetscInt *, PetscInt *, PetscViewerVTKFieldType *);
#if defined(PETSC_HAVE_HDF5)
PETSC_EXTERN PetscErrorCode VecView_Plex_Local_HDF5(Vec, PetscViewer);
PETSC_EXTERN PetscErrorCode VecView_Plex_HDF5(Vec, PetscViewer);
PETSC_EXTERN PetscErrorCode VecLoad_Plex_HDF5(Vec, PetscViewer);
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
PETSC_EXTERN PetscErrorCode DMPlexCreateGmsh_ReadElement(FILE *, PetscInt *, int *, PetscInt *, int[], int[]);
PETSC_EXTERN PetscErrorCode DMPlexInvertCell_Internal(PetscInt, PetscInt, PetscInt[]);
PETSC_EXTERN PetscErrorCode DMPlexLocalizeCoordinate_Internal(DM, PetscInt, const PetscScalar[], const PetscScalar[], PetscScalar[]);
PETSC_EXTERN PetscErrorCode DMPlexLocalizeAddCoordinate_Internal(DM, PetscInt, const PetscScalar[], const PetscScalar[], PetscScalar[]);
PETSC_EXTERN PetscErrorCode DMPlexVecSetFieldClosure_Internal(DM, PetscSection, Vec, PetscBool[], PetscInt, const PetscScalar[], InsertMode);
PETSC_EXTERN PetscErrorCode DMPlexProjectConstraints_Internal(DM, Vec, Vec);

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

#undef __FUNCT__
#define __FUNCT__ "DMPlex_Invert2D_Internal"
PETSC_STATIC_INLINE void DMPlex_Invert2D_Internal(PetscReal invJ[], PetscReal J[], PetscReal detJ)
{
  const PetscReal invDet = 1.0/detJ;

  invJ[0] =  invDet*J[3];
  invJ[1] = -invDet*J[1];
  invJ[2] = -invDet*J[2];
  invJ[3] =  invDet*J[0];
  PetscLogFlops(5.0);
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
  PetscLogFlops(37.0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlex_Det2D_Internal"
PETSC_STATIC_INLINE void DMPlex_Det2D_Internal(PetscReal *detJ, PetscReal J[])
{
  *detJ = J[0]*J[3] - J[1]*J[2];
  PetscLogFlops(3.0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlex_Det3D_Internal"
PETSC_STATIC_INLINE void DMPlex_Det3D_Internal(PetscReal *detJ, PetscReal J[])
{
  *detJ = (J[0*3+0]*(J[1*3+1]*J[2*3+2] - J[1*3+2]*J[2*3+1]) +
           J[0*3+1]*(J[1*3+2]*J[2*3+0] - J[1*3+0]*J[2*3+2]) +
           J[0*3+2]*(J[1*3+0]*J[2*3+1] - J[1*3+1]*J[2*3+0]));
  PetscLogFlops(12.0);
}

#endif /* _PLEXIMPL_H */
