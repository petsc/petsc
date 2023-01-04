#ifndef PETSC_PCBDDCSTRUCTSIMPL_H
#define PETSC_PCBDDCSTRUCTSIMPL_H

#include <petscksp.h>
#include <petscbt.h>

/* special marks for interface graph: they cannot be enums
   since PCBDDCGRAPH_SPECIAL_MARK ranges from -4 to -max_int */
#define PCBDDCGRAPH_NEUMANN_MARK        -1
#define PCBDDCGRAPH_DIRICHLET_MARK      -2
#define PCBDDCGRAPH_LOCAL_PERIODIC_MARK -3
#define PCBDDCGRAPH_SPECIAL_MARK        -4

/* Structure for local graph partitioning */
struct _PCBDDCGraph {
  PetscBool setupcalled;
  /* graph information */
  ISLocalToGlobalMapping l2gmap;
  PetscInt               nvtxs;
  PetscInt               nvtxs_global;
  PetscBT                touched;
  PetscInt              *count;
  PetscInt             **neighbours_set;
  PetscInt              *subset;
  PetscInt              *which_dof;
  PetscInt              *special_dof;
  PetscInt               custom_minimal_size;
  PetscBool              twodim;
  PetscBool              twodimset;
  PetscBool              has_dirichlet;
  IS                     dirdofs;
  IS                     dirdofsB;
  PetscInt               commsizelimit;
  PetscInt               maxcount;
  /* data for connected components */
  PetscInt  ncc;
  PetscInt *cptr;
  PetscInt *queue;
  PetscBool queue_sorted;
  /* data for interface subsets */
  PetscInt   n_subsets;
  PetscInt  *subset_size;
  PetscInt **subset_idxs;
  PetscInt  *subset_ncc;
  PetscInt  *subset_ref_node;
  /* data for periodic dofs */
  PetscInt  *mirrors;
  PetscInt **mirrors_set;
  /* placeholders for connectivity relation between dofs */
  PetscInt  nvtxs_csr;
  PetscInt *xadj;
  PetscInt *adjncy;
  PetscBool freecsr;
  /* data for local subdomains (if any have been detected)
     these are not intended to be exposed */
  PetscInt  n_local_subs;
  PetscInt *local_subs;
  /* coordinates (for corner detection) */
  PetscBool  active_coords;
  PetscBool  cloc;
  PetscInt   cdim, cnloc;
  PetscReal *coords;
};
typedef struct _PCBDDCGraph *PCBDDCGraph;

struct _PCBDDCGraphCandidates {
  PetscInt nfc, nec;
  IS      *Faces, *Edges, Vertices;
};
typedef struct _PCBDDCGraphCandidates *PCBDDCGraphCandidates;

/* Wrap to MatFactor solver in Schur complement mode. Provides
   - standalone solver for interior variables
   - forward and backward substitutions for correction solver
*/
/* It assumes that interior variables are a contiguous set starting from 0 */
struct _PCBDDCReuseSolvers {
  /* the factored matrix obtained from MatGetFactor(...,solver_package,...) */
  Mat F;
  /* placeholders for the solution and rhs on the whole set of dofs of A (size local_dofs - local_vertices)*/
  Vec sol;
  Vec rhs;
  /* */
  PetscBool has_vertices;
  /* shell PCs to handle interior/correction solvers */
  PC interior_solver;
  PC correction_solver;
  IS is_R;
  /* objects to handle Schur complement solution */
  Vec        rhs_B;
  Vec        sol_B;
  IS         is_B;
  VecScatter correction_scatter_B;
  /* handle benign trick without change of basis on pressures */
  PetscInt     benign_n;
  IS          *benign_zerodiag_subs;
  PetscScalar *benign_save_vals;
  Mat          benign_csAIB;
  Mat          benign_AIIm1ones;
  Vec          benign_corr_work;
  Vec          benign_dummy_schur_vec;
};
typedef struct _PCBDDCReuseSolvers *PCBDDCReuseSolvers;

/* structure to handle Schur complements on subsets */
struct _PCBDDCSubSchurs {
  /* local Neumann matrix */
  Mat A;
  /* local Schur complement */
  Mat S;
  /* index sets */
  IS is_I;
  IS is_B;
  /* whether Schur complements are explicitly computed with or not */
  char      mat_solver_type[64];
  PetscBool schur_explicit;
  /* BDDC or GDSW */
  PetscBool gdsw;
  /* matrices contained explicit schur complements cat together */
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
  IS     *is_subs;
  PetscBT is_edge;
  /* mat flags */
  PetscBool is_symmetric;
  PetscBool is_hermitian;
  PetscBool is_posdef;
  /* data structure to reuse MatFactor with Schur solver */
  PCBDDCReuseSolvers reuse_solver;
  /* change of variables */
  KSP      *change;
  IS       *change_primal_sub;
  PetscBool change_with_qr;
  /* prefix */
  char *prefix;
  /* */
  PetscBool restrict_comm;
  /* debug */
  PetscBool debug;
};
typedef struct _PCBDDCSubSchurs *PCBDDCSubSchurs;

/* Structure for deluxe scaling */
struct _PCBDDCDeluxeScaling {
  /* simple scaling on selected dofs (i.e. primal vertices and nodes on interface dirichlet boundaries) */
  PetscInt  n_simple;
  PetscInt *idx_simple_B;
  /* handle deluxe problems  */
  PetscInt     seq_n;
  PetscScalar *workspace;
  VecScatter  *seq_scctx;
  Vec         *seq_work1;
  Vec         *seq_work2;
  Mat         *seq_mat;
  Mat         *seq_mat_inv_sum;
  KSP         *change;
  PetscBool    change_with_qr;
};
typedef struct _PCBDDCDeluxeScaling *PCBDDCDeluxeScaling;

/* inexact solvers with nullspace correction */
struct _NullSpaceCorrection_ctx {
  Mat           basis_mat;
  Mat           inv_smat;
  PC            local_pc;
  Vec          *fw;
  Vec          *sw;
  PetscScalar   scale;
  PetscLogEvent evapply;
  PetscBool     symm;
};
typedef struct _NullSpaceCorrection_ctx *NullSpaceCorrection_ctx;

/* MatShell context for benign mat mults */
struct _PCBDDCBenignMatMult_ctx {
  Mat          A;
  PetscInt     benign_n;
  IS          *benign_zerodiag_subs;
  PetscScalar *work;
  PetscBool    apply_left;
  PetscBool    apply_right;
  PetscBool    apply_p0;
  PetscBool    free;
};
typedef struct _PCBDDCBenignMatMult_ctx *PCBDDCBenignMatMult_ctx;

/* feti-dp mat */
struct _FETIDPMat_ctx {
  PetscInt   n;        /* local number of rows */
  PetscInt   N;        /* global number of rows */
  PetscInt   n_lambda; /* global number of multipliers */
  Vec        lambda_local;
  Vec        temp_solution_B;
  Vec        temp_solution_D;
  Mat        B_delta;
  Mat        B_Ddelta;
  PetscBool  deluxe_nonred;
  VecScatter l2g_lambda;
  PC         pc;
  PetscBool  fully_redundant;
  /* saddle point */
  VecScatter l2g_lambda_only;
  Mat        B_BB;
  Mat        B_BI;
  Mat        Bt_BB;
  Mat        Bt_BI;
  Mat        C;
  VecScatter l2g_p;
  VecScatter g2g_p;
  Vec        vP;
  Vec        xPg;
  Vec        yPg;
  Vec        rhs_flip;
  IS         pressure;
  IS         lagrange;
};
typedef struct _FETIDPMat_ctx *FETIDPMat_ctx;

/* feti-dp preconditioner */
struct _FETIDPPC_ctx {
  Mat        S_j;
  Vec        lambda_local;
  Mat        B_Ddelta;
  VecScatter l2g_lambda;
  PC         pc;
  /* saddle point */
  Vec xPg;
  Vec yPg;
};
typedef struct _FETIDPPC_ctx *FETIDPPC_ctx;

struct _BDdelta_DN {
  Mat BD;
  KSP kBD;
  Vec work;
};
typedef struct _BDdelta_DN *BDdelta_DN;

/* Schur interface preconditioner */
struct _BDDCIPC_ctx {
  VecScatter g2l;
  PC         bddc;
};
typedef struct _BDDCIPC_ctx *BDDCIPC_ctx;

#endif // PETSC_PCBDDCSTRUCTSIMPL_H
