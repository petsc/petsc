/*
      Data structure used for Patch preconditioner.
*/
#if !defined(__PATCH_IMPL)
#define __PATCH_IMPL
#include <petsc/private/pcimpl.h>
#include <petscksp.h>
#include <petsc/private/hash.h>

typedef struct {
  PetscSF          defaultSF;
  PetscSection    *dofSection;
  PetscSection     cellCounts;
  PetscSection     cellNumbering;   /* Numbering of cells in DM */
  PetscSection     gtolCounts;      /* Indices to extract from local to patch vectors */
  PetscInt         nsubspaces;      /* for mixed problems */
  PetscInt        *subspaceOffsets; /* offsets for calculating concatenated numbering for mixed spaces */
  PetscSection     bcCounts;
  IS               cells;
  IS               dofs;
  IS               ghostBcNodes;
  IS               globalBcNodes;
  IS               gtol;
  IS              *bcs;
  IS              *multBcs;            /* Only used for multiplicative smoothing to recalculate residual */

  PetscBool        save_operators;     /* Save all operators (or create/destroy one at a time?) */
  PetscBool        partition_of_unity; /* Weight updates by dof multiplicity? */
  PetscBool        multiplicative;     /* Gauss-Seidel or Jacobi? */
  PetscInt         npatch;             /* Number of patches */
  PetscInt        *bs;                 /* block size (can come from global operators?) */
  PetscInt        *nodesPerCell;
  PetscInt         totalDofsPerCell;
  const PetscInt **cellNodeMap;        /* Map from cells to nodes */

  KSP             *ksp;                /* Solvers for each patch */
  Vec              localX, localY;
  Vec              dof_weights;        /* In how many patches does each dof lie? */
  Vec             *patchX, *patchY;    /* Work vectors for patches */
  Vec             *patch_dof_weights;
  Mat             *mat;                /* Operators */
  Mat             *multMat;            /* Operators for multiplicative residual calculation */
  MatType          sub_mat_type;
  PetscErrorCode  (*usercomputeop)(PC, Mat, PetscInt, const PetscInt *, PetscInt, const PetscInt *, void *);
  void            *usercomputectx;

  PCPatchConstructType ctype;            /* Algorithm for patch construction */
  PetscErrorCode     (*patchconstructop)(void*, DM, PetscInt, PetscHashI); /* patch construction */
  PetscErrorCode     (*userpatchconstructionop)(PC, PetscInt*, IS**, IS*, void* ctx);
  void                *userpatchconstructctx;
  PetscBool            user_patches;
  PetscInt             codim;            /* dimension or codimension of mesh points to loop over; */
  PetscInt             dim;              /*   only one of them can be set */
  PetscInt             exclude_subspace; /* If you don't want any other dofs from a particular subspace you can exclude them with this.
                                             Used for Vanka in Stokes, for example, to eliminate all pressure dofs not on the vertex
                                             you're building the patch around */
  PetscInt             vankadim;         /* In Vanka construction, should we eliminate any entities of a certain dimension? */

  PetscBool            print_patches;    /* Should we print out information about patch construction? */
  PetscBool            symmetrise_sweep; /* Should we sweep forwards->backwards, backwards->forwards? */

  IS                  *userIS;
  IS                   iterationSet;     /* Index set specifying how we iterate over patches */
  PetscInt             nuserIS;          /* user-specified index sets to specify the patches */
} PC_PATCH;

PETSC_EXTERN PetscLogEvent PC_Patch_CreatePatches, PC_Patch_ComputeOp, PC_Patch_Solve, PC_Patch_Scatter, PC_Patch_Apply, PC_Patch_Prealloc;

#endif
