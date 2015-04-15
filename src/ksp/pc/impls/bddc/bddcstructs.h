#if !defined(__pcbddc_structs_h)
#define __pcbddc_structs_h

#include <petscksp.h>
#include <petscbt.h>

/* special marks for interface graph: they cannot be enums, since special marks should in principle range from -4 to -max_int */
#define PCBDDCGRAPH_NEUMANN_MARK -1
#define PCBDDCGRAPH_DIRICHLET_MARK -2
#define PCBDDCGRAPH_LOCAL_PERIODIC_MARK -3
#define PCBDDCGRAPH_SPECIAL_MARK -4

/* Structure for local graph partitioning */
struct _PCBDDCGraph {
  ISLocalToGlobalMapping l2gmap;
  PetscInt               nvtxs;
  PetscBT                touched;
  PetscInt               *count;
  PetscInt               **neighbours_set;
  PetscInt               *subset;
  PetscInt               *which_dof;
  PetscInt               *cptr;
  PetscInt               *queue;
  PetscInt               *special_dof;
  PetscInt               *subset_ncc;
  PetscInt               *subset_ref_node;
  PetscInt               *mirrors;
  PetscInt               **mirrors_set;
  PetscInt               ncc;
  PetscInt               n_subsets;
  PetscInt               custom_minimal_size;
  PetscInt               nvtxs_csr;
  PetscInt               *xadj;
  PetscInt               *adjncy;
};
typedef struct _PCBDDCGraph *PCBDDCGraph;

/* Temporary wrap to MUMPS solver for interior variables in Schur complement mode */
/* It assumes that interior variables are a contiguous set starting from 0 */
struct _PCBDDCMumpsInterior {
  /* the factored matrix obtained from MatGetFactor(...,MAT_SOLVER_MUMPS...) */
  Mat F;
  /* placeholders for the solution and rhs on the whole set of dofs of A */
  Vec sol;
  Vec rhs;
  /* size of interior problem */
  PetscInt n;
};
typedef struct _PCBDDCMumpsInterior *PCBDDCMumpsInterior;

/* structure to handle Schur complements on subsets */
struct _PCBDDCSubSchurs {
  /* local Neumann matrix */
  Mat A;
  /* local Schur complement */
  Mat S;
  /* index sets */
  IS  is_I;
  IS  is_B;
  /* whether Schur complements are computed with MUMPS or not */
  PetscBool use_mumps;
  /* matrices cointained explicit schur complements cat together */
  /* note that AIJ format is used but the values are inserted as in column major ordering */
  Mat S_Ej_all;
  Mat sum_S_Ej_all;
  Mat sum_S_Ej_inv_all;
  Mat sum_S_Ej_tilda_all;
  IS  is_Ej_all;
  IS  is_Ej_com;
  /* IS */
  IS is_I_layer;
  /* l2g maps */
  ISLocalToGlobalMapping l2gmap;
  ISLocalToGlobalMapping BtoNmap;
  /* number of local subproblems */
  PetscInt n_subs;
  /* connected components */
  IS*      is_subs;
  PetscBT  is_edge;
  PetscBT  computed_Stilda_subs;
  /* mat flags */
  PetscBool is_hermitian;
  PetscBool is_posdef;
  /* shell PC to handle MUMPS interior solver */
  PC interior_solver;
};
typedef struct _PCBDDCSubSchurs *PCBDDCSubSchurs;

/* Structure for deluxe scaling */
struct _PCBDDCDeluxeScaling {
  /* simple scaling on selected dofs (i.e. primal vertices and nodes on interface dirichlet boundaries) */
  PetscInt        n_simple;
  PetscInt*       idx_simple_B;
  /* handle deluxe problems  */
  VecScatter      seq_scctx;
  Vec             seq_work1;
  Vec             seq_work2;
  Mat             seq_mat;
  KSP             seq_ksp;
};
typedef struct _PCBDDCDeluxeScaling *PCBDDCDeluxeScaling;

/* inexact solvers with nullspace correction */
struct _NullSpaceCorrection_ctx {
  Mat basis_mat;
  Mat Kbasis_mat;
  Mat Lbasis_mat;
  PC  local_pc;
  Vec work_small_1;
  Vec work_small_2;
  Vec work_full_1;
  Vec work_full_2;
};
typedef struct _NullSpaceCorrection_ctx *NullSpaceCorrection_ctx;

/* change of basis */
struct _PCBDDCChange_ctx {
  Mat original_mat;
  Mat global_change;
  Vec *work;
};
typedef struct _PCBDDCChange_ctx *PCBDDCChange_ctx;

/* feti-dp mat */
struct _FETIDPMat_ctx {
  PetscInt   n_lambda;
  Vec        lambda_local;
  Vec        temp_solution_B;
  Vec        temp_solution_D;
  Mat        B_delta;
  Mat        B_Ddelta;
  VecScatter l2g_lambda;
  PC         pc;
};
typedef struct _FETIDPMat_ctx *FETIDPMat_ctx;

/* feti-dp dirichlet preconditioner */
struct _FETIDPPC_ctx {
  Mat        S_j;
  Vec        lambda_local;
  Mat        B_Ddelta;
  VecScatter l2g_lambda;
  PC         pc;
};
typedef struct _FETIDPPC_ctx *FETIDPPC_ctx;

#endif
