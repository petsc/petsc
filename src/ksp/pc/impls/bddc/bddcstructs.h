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
  /* graph information */
  ISLocalToGlobalMapping l2gmap;
  PetscInt               nvtxs;
  PetscInt               nvtxs_global;
  PetscBT                touched;
  PetscInt               *count;
  PetscInt               **neighbours_set;
  PetscInt               *subset;
  PetscInt               *which_dof;
  PetscInt               *special_dof;
  PetscInt               custom_minimal_size;
  PetscBool              twodim;
  PetscBool              has_dirichlet;
  IS                     dirdofs;
  IS                     dirdofsB;
  /* data for connected components */
  PetscInt               ncc;
  PetscInt               *cptr;
  PetscInt               *queue;
  PetscBool              queue_sorted;
  /* data for interface subsets */
  PetscInt               n_subsets;
  PetscInt               *subsets_size;
  PetscInt               **subsets;
  PetscInt               *subset_ncc;
  PetscInt               *subset_ref_node;
  /* data for periodic dofs */
  PetscInt               *mirrors;
  PetscInt               **mirrors_set;
  /* placeholders for connectivity relation between dofs */
  PetscInt               nvtxs_csr;
  PetscInt               *xadj;
  PetscInt               *adjncy;
};
typedef struct _PCBDDCGraph *PCBDDCGraph;

/* Temporary wrap to MUMPS solver in Schur complement mode. Provides
   - standalone solver for interior variables
   - forward and backward substitutions for correction solver
*/
/* It assumes that interior variables are a contiguous set starting from 0 */
struct _PCBDDCReuseMumps {
  /* the factored matrix obtained from MatGetFactor(...,MAT_SOLVER_MUMPS...) */
  Mat        F;
  /* placeholders for the solution and rhs on the whole set of dofs of A (size local_dofs - local_vertices)*/
  Vec        sol;
  Vec        rhs;
  /* size of interior problem */
  PetscInt   n_I;
  /* shell PCs to handle MUMPS interior/correction solvers */
  PC         interior_solver;
  PC         correction_solver;
  IS         is_R;
  /* objects to hanlde Schur complement solution */
  Vec        rhs_B;
  Vec        sol_B;
  IS         is_B;
  VecScatter correction_scatter_B;
};
typedef struct _PCBDDCReuseMumps *PCBDDCReuseMumps;

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
  IS  is_vertices;
  IS  is_dir;
  /* l2g maps */
  ISLocalToGlobalMapping l2gmap;
  ISLocalToGlobalMapping BtoNmap;
  /* number of local subproblems */
  PetscInt n_subs;
  /* connected components */
  IS*      is_subs;
  PetscBT  is_edge;
  /* mat flags */
  PetscBool is_hermitian;
  PetscBool is_posdef;
  /* data structure to reuse MUMPS Schur solver */
  PCBDDCReuseMumps reuse_mumps;
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
