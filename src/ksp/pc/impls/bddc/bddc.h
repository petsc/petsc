#if !defined(__pcbddc_h)
#define __pcbddc_h

#include <../src/ksp/pc/impls/is/pcis.h>
#include <../src/ksp/pc/impls/bddc/bddcstructs.h>

/* Private context (data structure) for the BDDC preconditioner.  */
typedef struct {
  /* First MUST come the folowing line, for the stuff that is common to FETI and Neumann-Neumann. */
  PC_IS         pcis;
  /* Coarse stuffs needed by BDDC application in KSP */
  Vec           coarse_vec;
  KSP           coarse_ksp;
  Mat           coarse_phi_B;
  Mat           coarse_phi_D;
  Mat           coarse_psi_B;
  Mat           coarse_psi_D;
  PetscInt      local_primal_size;
  PetscInt      coarse_size;
  PetscInt*     global_primal_indices;
  VecScatter    coarse_loc_to_glob;
  /* Local stuffs needed by BDDC application in KSP */
  Vec           vec1_P;
  Vec           vec1_C;
  Mat           local_auxmat1;
  Mat           local_auxmat2;
  Vec           vec1_R;
  Vec           vec2_R;
  IS            is_R_local;
  VecScatter    R_to_B;
  VecScatter    R_to_D;
  KSP           ksp_R;
  KSP           ksp_D;
  /* Quantities defining constraining details (local) of the preconditioner */
  /* These quantities define the preconditioner itself */
  PetscInt      n_vertices;
  Mat           ConstraintMatrix;
  PetscBool     new_primal_space;
  PetscBool     new_primal_space_local;
  PetscInt      *primal_indices_local_idxs;
  PetscInt      local_primal_size_cc;
  PetscInt      *local_primal_ref_node;
  PetscInt      *local_primal_ref_mult;
  PetscBool     use_change_of_basis;
  PetscBool     use_change_on_faces;
  Mat           ChangeOfBasisMatrix;
  Mat           user_ChangeOfBasisMatrix;
  Mat           new_global_mat;
  Vec           original_rhs;
  Vec           temp_solution;
  Mat           local_mat;
  PetscBool     use_exact_dirichlet_trick;
  PetscBool     ksp_guess_nonzero;
  PetscBool     rhs_change;
  /* Some defaults on selecting vertices and constraints*/
  PetscBool     use_local_adj;
  PetscBool     use_vertices;
  PetscBool     use_faces;
  PetscBool     use_edges;
  /* Some customization is possible */
  PetscBool           recompute_topography;
  PCBDDCGraph         mat_graph;
  MatNullSpace        onearnullspace;
  PetscObjectState    *onearnullvecs_state;
  MatNullSpace        NullSpace;
  IS                  user_primal_vertices;
  PetscBool           use_nnsp_true;
  PetscBool           use_qr_single;
  PetscBool           user_provided_isfordofs;
  PetscInt            n_ISForDofs;
  PetscInt            n_ISForDofsLocal;
  IS                  *ISForDofs;
  IS                  *ISForDofsLocal;
  IS                  NeumannBoundaries;
  IS                  NeumannBoundariesLocal;
  IS                  DirichletBoundaries;
  IS                  DirichletBoundariesLocal;
  PetscBool           switch_static;
  PetscInt            coarsening_ratio;
  PetscInt            coarse_adj_red;
  PetscInt            current_level;
  PetscInt            max_levels;
  PetscInt            redistribute_coarse;
  IS                  coarse_subassembling;
  IS                  coarse_subassembling_init;
  PetscBool           use_coarse_estimates;
  PetscBool           symmetric_primal;
  /* scaling */
  Vec                 work_scaling;
  PetscBool           use_deluxe_scaling;
  PCBDDCDeluxeScaling deluxe_ctx;
  PetscBool           faster_deluxe;

  /* schur complements on interface's subsets */
  PCBDDCSubSchurs sub_schurs;
  PetscBool       sub_schurs_rebuild;
  PetscInt        sub_schurs_layers;
  PetscBool       sub_schurs_use_useradj;
  PetscBool       computed_rowadj;

  /* adaptive selection of constraints */
  PetscBool    adaptive_selection;
  PetscReal    adaptive_threshold;
  PetscInt     adaptive_nmin;
  PetscInt     adaptive_nmax;
  PetscInt*    adaptive_constraints_n;
  PetscInt*    adaptive_constraints_idxs;
  PetscInt*    adaptive_constraints_idxs_ptr;
  PetscScalar* adaptive_constraints_data;
  PetscInt*    adaptive_constraints_data_ptr;

  /* For verbose output of some bddc data structures */
  PetscInt    dbg_flag;
  PetscViewer dbg_viewer;
} PC_BDDC;


#endif /* __pcbddc_h */
