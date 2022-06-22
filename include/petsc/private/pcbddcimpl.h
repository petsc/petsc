#if !defined(__pcbddc_h)
#define __pcbddc_h

#include <petsc/private/pcisimpl.h>
#include <petsc/private/pcbddcstructsimpl.h>

#if !defined(PETSC_PCBDDC_MAXLEVELS)
#define PETSC_PCBDDC_MAXLEVELS 8
#endif

PETSC_EXTERN PetscLogEvent PC_BDDC_Topology[PETSC_PCBDDC_MAXLEVELS];
PETSC_EXTERN PetscLogEvent PC_BDDC_LocalSolvers[PETSC_PCBDDC_MAXLEVELS];
PETSC_EXTERN PetscLogEvent PC_BDDC_LocalWork[PETSC_PCBDDC_MAXLEVELS];
PETSC_EXTERN PetscLogEvent PC_BDDC_CorrectionSetUp[PETSC_PCBDDC_MAXLEVELS];
PETSC_EXTERN PetscLogEvent PC_BDDC_CoarseSetUp[PETSC_PCBDDC_MAXLEVELS];
PETSC_EXTERN PetscLogEvent PC_BDDC_ApproxSetUp[PETSC_PCBDDC_MAXLEVELS];
PETSC_EXTERN PetscLogEvent PC_BDDC_ApproxApply[PETSC_PCBDDC_MAXLEVELS];
PETSC_EXTERN PetscLogEvent PC_BDDC_CoarseSolver[PETSC_PCBDDC_MAXLEVELS];
PETSC_EXTERN PetscLogEvent PC_BDDC_AdaptiveSetUp[PETSC_PCBDDC_MAXLEVELS];
PETSC_EXTERN PetscLogEvent PC_BDDC_Scaling[PETSC_PCBDDC_MAXLEVELS];
PETSC_EXTERN PetscLogEvent PC_BDDC_Schurs[PETSC_PCBDDC_MAXLEVELS];
PETSC_EXTERN PetscLogEvent PC_BDDC_Solves[PETSC_PCBDDC_MAXLEVELS][3];

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
  PetscBool     fake_change;
  Mat           ChangeOfBasisMatrix;
  Mat           user_ChangeOfBasisMatrix;
  PetscBool     change_interior;
  Mat           switch_static_change;
  Vec           work_change;
  Vec           original_rhs;
  Vec           temp_solution;
  Mat           local_mat;
  PetscBool     use_exact_dirichlet_trick;
  PetscBool     exact_dirichlet_trick_app;
  PetscBool     ksp_guess_nonzero;
  PetscBool     rhs_change;
  PetscBool     temp_solution_used;
  /* benign subspace trick */
  PetscBool     benign_saddle_point;
  PetscBool     benign_have_null;
  PetscBool     benign_skip_correction;
  PetscBool     benign_compute_correction;
  Mat           benign_change;
  Mat           benign_original_mat;
  IS            *benign_zerodiag_subs;
  Vec           benign_vec;
  Mat           benign_B0;
  PetscSF       benign_sf;
  PetscScalar   *benign_p0;
  PetscInt      benign_n;
  PetscInt      *benign_p0_lidx;
  PetscInt      *benign_p0_gidx;
  PetscBool     benign_null;
  PetscBool     benign_change_explicit;
  PetscBool     benign_apply_coarse_only;

  /* Some defaults on selecting vertices and constraints*/
  PetscBool     use_local_adj;
  PetscBool     use_vertices;
  PetscBool     use_faces;
  PetscBool     use_edges;

  /* Some customization is possible */
  PetscBool           corner_selection;
  PetscBool           corner_selected;
  PetscBool           recompute_topography;
  PetscBool           graphanalyzed;
  PCBDDCGraph         mat_graph;
  PetscInt            graphmaxcount;
  MatNullSpace        onearnullspace;
  PetscObjectState    *onearnullvecs_state;
  PetscBool           NullSpace_corr[4];
  IS                  user_primal_vertices;
  IS                  user_primal_vertices_local;
  PetscBool           use_nnsp;
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
  PetscBool           eliminate_dirdofs;
  PetscBool           switch_static;
  PetscInt            coarsening_ratio;
  PetscInt            coarse_adj_red;
  PetscInt            current_level;
  PetscInt            max_levels;
  PetscInt            coarse_eqs_per_proc;
  PetscInt            coarse_eqs_limit;
  IS                  coarse_subassembling;
  PetscBool           use_coarse_estimates;
  PetscBool           symmetric_primal;
  PetscInt            vertex_size;
  PCBDDCInterfaceExtType interface_extension;

  /* no-net-flux */
  PetscBool compute_nonetflux;
  Mat       divudotp;
  PetscBool divudotp_trans;
  IS        divudotp_vl2l;

  /* nedelec */
  Mat       discretegradient;
  PetscInt  nedorder;
  PetscBool conforming;
  PetscInt  nedfield;
  PetscBool nedglobal;
  Mat       nedcG;
  IS        nedclocal;

  /* local disconnected subdomains */
  PetscBool detect_disconnected;
  PetscBool detect_disconnected_filter;
  PetscInt  n_local_subs;
  IS        *local_subs;

  /* scaling */
  Vec                 work_scaling;
  PetscBool           use_deluxe_scaling;
  PCBDDCDeluxeScaling deluxe_ctx;
  PetscBool           deluxe_zerorows;
  PetscBool           deluxe_singlemat;

  /* schur complements on interface's subsets */
  PCBDDCSubSchurs sub_schurs;
  PetscBool       sub_schurs_rebuild;
  PetscBool       sub_schurs_exact_schur;
  PetscInt        sub_schurs_layers;
  PetscBool       sub_schurs_use_useradj;
  PetscBool       computed_rowadj;

  /* adaptive selection of constraints */
  PetscBool    adaptive_selection;
  PetscBool    adaptive_userdefined;
  PetscReal    adaptive_threshold[2];
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
