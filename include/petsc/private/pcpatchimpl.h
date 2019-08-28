/*
      Data structure used for Patch preconditioner.
*/
#if !defined(__PATCH_IMPL)
#define __PATCH_IMPL
#include <petsc/private/pcimpl.h>
#include <petsc/private/hashseti.h>
#include <petsc/private/hashmapi.h>
#include <petscksp.h>

typedef struct {
  /* Topology */
  PCPatchConstructType ctype;              /* Algorithm for patch construction */
  PetscErrorCode     (*patchconstructop)(void*, DM, PetscInt, PetscHSetI); /* patch construction */
  PetscErrorCode     (*userpatchconstructionop)(PC, PetscInt*, IS**, IS*, void* ctx);
  void                *userpatchconstructctx;
  IS                  *userIS;
  PetscInt             npatch;             /* Number of patches */
  PetscBool            user_patches;       /* Flag for user construction of patches */
  PetscInt             dim, codim;         /* Dimension or codimension of mesh points to loop over; only one of them can be set */
  PetscSection         cellCounts;         /* Maps patch -> # cells in patch */
  IS                   cells;              /* [patch][cell in patch]: Cell number */
  IS                   extFacets;
  IS                   intFacets;
  IS                   intFacetsToPatchCell; /* Support of interior facet in local patch point numbering: AKA which two cells touch the facet (in patch local numbering of cells) */
  PetscSection         intFacetCounts;
  PetscSection         extFacetCounts;
  PetscSection         cellNumbering;      /* Plex: NULL Firedrake: Numbering of cells in DM */
  PetscSection         pointCounts;        /* Maps patch -> # points with dofs in patch */
  IS                   points;             /* [patch][point in patch]: Point number */
  /* Dof layout */
  PetscBool            combined;           /* Use a combined space with all fields */
  PetscInt             nsubspaces;         /* Number of fields */
  PetscSF              sectionSF;          /* Combined SF mapping process local to global */
  PetscSection        *dofSection;         /* ?? For each field, patch -> # dofs in patch */
  PetscInt            *subspaceOffsets;    /* Plex: NULL Firedrake: offset of each field in concatenated process local numbering for mixed spaces */
  PetscInt           **cellNodeMap;        /* [field][cell][dof in cell]: global dofs in cell TODO Free this after its use in PCPatchCreateCellPatchDiscretisationInfo() */
  IS                   dofs;               /* [patch][cell in patch][dof in cell]: patch local dof */
  IS                   offs;               /* [patch][point in patch]: patch local offset (same layout as 'points', used for filling up patchSection) */
  IS                   dofsWithArtificial;
  IS                   offsWithArtificial;
  IS                   dofsWithAll;
  IS                   offsWithAll;
  PetscSection         patchSection;       /* Maps points -> patch local dofs */
  IS                   globalBcNodes;      /* Global dofs constrained by global Dirichlet conditions TODO Replace these with process local constrained dofs */
  IS                   ghostBcNodes;       /* Global dofs constrained by global Dirichlet conditions on this process and possibly others (patch overlaps boundary) */
  PetscSection         gtolCounts;         /* ?? Indices to extract from local to patch vectors */
  PetscSection    gtolCountsWithArtificial;/* ?? Indices to extract from local to patch vectors including those with artifical bcs*/
  PetscSection    gtolCountsWithAll;/* ?? Indices to extract from local to patch vectors including those in artificial or global bcs*/
  IS                   gtol;
  IS                   gtolWithArtificial;
  IS                   gtolWithAll;
  PetscInt            *bs;                 /* [field] block size per field (can come from global operators?) */
  PetscInt            *nodesPerCell;       /* [field] Dofs per cell TODO Change "node" to "dof" everywhere */
  PetscInt             totalDofsPerCell;   /* Dofs per cell counting all fields */
  PetscHSetI       subspaces_to_exclude;   /* If you don't want any other dofs from a particular subspace you can exclude them with this.
                                                Used for Vanka in Stokes, for example, to eliminate all pressure dofs not on the vertex
                                                you're building the patch around */
  PetscInt             vankadim;           /* In Vanka construction, should we eliminate any entities of a certain dimension on the initial patch? */
  PetscInt             ignoredim;          /* In Vanka construction, should we eliminate any entities of a certain dimension on the boundary? */
  PetscInt             pardecomp_overlap;  /* In parallel decomposition construction, how much overlap? */
  /* Patch system assembly */
  PetscErrorCode     (*usercomputeop)(PC, PetscInt, Vec, Mat, IS, PetscInt, const PetscInt *, const PetscInt *, void *);
  void                *usercomputeopctx;
  PetscErrorCode     (*usercomputef)(PC, PetscInt, Vec, Vec, IS, PetscInt, const PetscInt *, const PetscInt *, void *);
  void                *usercomputefctx;
  /* Interior facet integrals: Jacobian */
  PetscErrorCode     (*usercomputeopintfacet)(PC, PetscInt, Vec, Mat, IS, PetscInt, const PetscInt *, const PetscInt *, void *);
  void                *usercomputeopintfacetctx;
  /* Residual */
  PetscErrorCode     (*usercomputefintfacet)(PC, PetscInt, Vec, Vec, IS, PetscInt, const PetscInt *, const PetscInt *, void *);
  void                *usercomputefintfacetctx;
  IS                   cellIS;             /* Temporary IS for each cell patch */
  PetscBool            save_operators;     /* Save all operators (or create/destroy one at a time?) */
  PetscBool            precomputeElementTensors; /* Precompute all element tensors (each cell is assembled exactly once)? */
  IS                   allCells;                 /* Unique cells in union of all patches */
  IS                   allIntFacets;                 /* Unique interior facets in union of all patches */
  PetscBool            partition_of_unity; /* Weight updates by dof multiplicity? */
  PetscBool            multiplicative;     /* Gauss-Seidel instead of Jacobi?  */
  PCCompositeType      local_composition_type; /* locally additive or multiplicative? */
  /* Patch solves */
  Vec                  cellMats;           /* Cell element tensors */
  PetscInt            *precomputedTensorLocations; /* Locations of the precomputed tensors for each cell. */
  Vec                  intFacetMats;               /* interior facet element tensors */
  PetscInt            *precomputedIntFacetTensorLocations; /* Locations of the precomputed tensors for each interior facet. */
  Mat                 *mat;                /* System matrix for each patch */
  Mat                 *matWithArtificial;   /* System matrix including dofs with artificial bcs for each patch */
  MatType              sub_mat_type;       /* Matrix type for patch systems */
  Vec                 *patchRHS, *patchUpdate;  /* RHS and solution for each patch */
  IS                  *dofMappingWithoutToWithArtificial;
  IS                  *dofMappingWithoutToWithAll;
  Vec                 *patchRHSWithArtificial;    /* like patchRHS but extra entries to include dofs with artificial bcs*/
  Vec                 *patch_dof_weights;  /* Weighting for dof in each patch */
  Vec                  localRHS, localUpdate;     /* ??? */
  Vec                  dof_weights;        /* In how many patches does each dof lie? */
  PetscBool            symmetrise_sweep;   /* Should we sweep forwards->backwards, backwards->forwards? */
  PetscBool            optionsSet;         /* SetFromOptions was called on this PC */
  IS                   iterationSet;       /* Index set specifying how we iterate over patches */
  PetscInt             currentPatch;       /* The current patch number when iterating */
  PetscObject         *solver;             /* Solvers for each patch TODO Do we need a new KSP for each patch? */
  PetscBool            denseinverse;       /* Should the patch inverse by applied by computing the inverse and a matmult? (Skips KSP/PC etc...) */
  PetscErrorCode     (*setupsolver)(PC);
  PetscErrorCode     (*applysolver)(PC, PetscInt, Vec, Vec);
  PetscErrorCode     (*resetsolver)(PC);
  PetscErrorCode     (*destroysolver)(PC);
  PetscErrorCode     (*updatemultiplicative)(PC, PetscInt, PetscInt);
  /* Monitoring */
  PetscBool            viewPatches;        /* View information about patch construction */
  PetscBool            viewCells;          /* View cells for each patch */
  PetscViewer          viewerCells;        /*   Viewer for patch cells */
  PetscViewerFormat    formatCells;        /*   Format for patch cells */
  PetscBool            viewIntFacets;          /* View intFacets for each patch */
  PetscViewer          viewerIntFacets;        /*   Viewer for patch intFacets */
  PetscViewerFormat    formatIntFacets;        /*   Format for patch intFacets */
  PetscBool            viewExtFacets;          /* View extFacets for each patch */
  PetscViewer          viewerExtFacets;        /*   Viewer for patch extFacets */
  PetscViewerFormat    formatExtFacets;        /*   Format for patch extFacets */
  PetscBool            viewPoints;         /* View points for each patch */
  PetscViewer          viewerPoints;       /*   Viewer for patch points */
  PetscViewerFormat    formatPoints;       /*   Format for patch points */
  PetscBool            viewSection;        /* View global section for each patch */
  PetscViewer          viewerSection;      /*   Viewer for patch sections */
  PetscViewerFormat    formatSection;      /*   Format for patch sections */
  PetscBool            viewMatrix;         /* View matrix for each patch */
  PetscViewer          viewerMatrix;       /*   Viewer for patch matrix */
  PetscViewerFormat    formatMatrix;       /*   Format for patch matrix */
  /* Extra variables for SNESPATCH */
  Vec                 *patchState;         /* State vectors for patch solvers */
  Vec                 *patchStateWithAll;  /* State vectors for patch solvers with all boundary data */
  Vec                  localState;         /* Scatter vector for state */
  Vec                 *patchResidual;      /* Work vectors for patch residual evaluation*/
  const char          *classname;          /* "snes" or "pc" for options */
  PetscBool            isNonlinear;        /* we need to do some things differently in nonlinear mode */
} PC_PATCH;

PETSC_EXTERN PetscLogEvent PC_Patch_CreatePatches;
PETSC_EXTERN PetscLogEvent PC_Patch_ComputeOp;
PETSC_EXTERN PetscLogEvent PC_Patch_Solve;
PETSC_EXTERN PetscLogEvent PC_Patch_Scatter;
PETSC_EXTERN PetscLogEvent PC_Patch_Apply;
PETSC_EXTERN PetscLogEvent PC_Patch_Prealloc;

PETSC_EXTERN PetscErrorCode PCPatchComputeFunction_Internal(PC, Vec, Vec, PetscInt);
PETSC_EXTERN PetscErrorCode PCPatchComputeOperator_Internal(PC, Vec, Mat, PetscInt, PetscBool);
typedef enum {SCATTER_INTERIOR, SCATTER_WITHARTIFICIAL, SCATTER_WITHALL} PatchScatterType;
PETSC_EXTERN PetscErrorCode PCPatch_ScatterLocal_Private(PC, PetscInt, Vec, Vec, InsertMode, ScatterMode, PatchScatterType);

#endif
