
#if !defined(__pcis_h)
#define __pcis_h

#include <petsc/private/pcimpl.h>
#include <../src/mat/impls/is/matis.h>
#include <petscksp.h>

/*
   Context (data structure) common for all Iterative Substructuring preconditioners.
*/

typedef struct {

  /* In naming the variables, we adopted the following convention: */
  /* * B - stands for interface nodes;                             */
  /* * I - stands for interior nodes;                              */
  /* * D - stands for Dirichlet (by extension, refers to interior  */
  /*       nodes) and                                              */
  /* * N - stands for Neumann (by extension, refers to all local   */
  /*       nodes, interior plus interface).                        */
  /* In some cases, I or D would apply equally well (e.g. vec1_D).  */

  PetscInt n;                /* number of nodes (interior+interface) in this subdomain */
  PetscInt n_B;              /* number of interface nodes in this subdomain */
  IS       is_B_local,       /* local (sequential) index sets for interface (B) and interior (I) nodes */
           is_I_local,
           is_B_global,
           is_I_global;

  Mat A_II, A_IB,            /* local (sequential) submatrices */
      A_BI, A_BB;
  Mat pA_II;
  Vec D;                     /* diagonal scaling "matrix" (stored as a vector, since it's diagonal) */
  KSP ksp_N,                /* linear solver contexts */
      ksp_D;
  Vec vec1_N,                /* local (sequential) work vectors */
      vec2_N,
      vec1_D,
      vec2_D,
      vec3_D,
      vec4_D,
      vec1_B,
      vec2_B,
      vec3_B,
      vec1_global;

  PetscScalar * work_N;
  VecScatter  N_to_D;             /* scattering context from all local nodes to local interior nodes */
  VecScatter  global_to_D;        /* scattering context from global to local interior nodes */
  VecScatter  N_to_B;             /* scattering context from all local nodes to local interface nodes */
  VecScatter  global_to_B;        /* scattering context from global to local interface nodes */
  PetscBool   pure_neumann;
  PetscScalar scaling_factor;
  PetscBool   use_stiffness_scaling;

  ISLocalToGlobalMapping mapping;
  PetscInt  n_neigh;     /* number of neighbours this subdomain has (INCLUDING the subdomain itself).       */
  PetscInt *neigh;       /* list of neighbouring subdomains                                                 */
  PetscInt *n_shared;    /* n_shared[j] is the number of nodes shared with subdomain neigh[j]               */
  PetscInt **shared;     /* shared[j][i] is the local index of the i-th node shared with subdomain neigh[j] */
  /* It is necessary some consistency in the                                                  */
  /* numbering of the shared edges from each side.                                            */
  /* For instance:                                                                            */
  /*                                                                                          */
  /* +-------+-------+                                                                        */
  /* |   k   |   l   | subdomains k and l are neighbours                                      */
  /* +-------+-------+                                                                        */
  /*                                                                                          */
  /* Let i and j be s.t. proc[k].neigh[i]==l and                                              */
  /*                     proc[l].neigh[j]==k.                                                 */
  /*                                                                                          */
  /* We need:                                                                                 */
  /* proc[k].loc_to_glob(proc[k].shared[i][m]) == proc[l].loc_to_glob(proc[l].shared[j][m])   */
  /* for all 0 <= m < proc[k].n_shared[i], or equiv'ly, for all 0 <= m < proc[l].n_shared[j]  */
  ISLocalToGlobalMapping BtoNmap;
  PetscBool reusesubmatrices;
} PC_IS;

PETSC_EXTERN PetscErrorCode PCISSetUp(PC,PetscBool,PetscBool);
PETSC_EXTERN PetscErrorCode PCISDestroy(PC);
PETSC_EXTERN PetscErrorCode PCISCreate(PC);
PETSC_EXTERN PetscErrorCode PCISApplySchur(PC,Vec,Vec,Vec,Vec,Vec);
PETSC_EXTERN PetscErrorCode PCISScatterArrayNToVecB(PetscScalar*,Vec,InsertMode,ScatterMode,PC);
PETSC_EXTERN PetscErrorCode PCISApplyInvSchur(PC,Vec,Vec,Vec,Vec);

#endif /* __pcis_h */
