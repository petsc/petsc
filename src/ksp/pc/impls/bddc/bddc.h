
#if !defined(__pcbddc_h)
#define __pcbddc_h

#include <../src/ksp/pc/impls/is/pcis.h>

/* BDDC requires metis 5.0.1 for multilevel */
#include "metis.h"
#define MetisInt    idx_t
#define MetisScalar real_t


/* Structure for graph partitioning (adapted from Metis) */
struct _PCBDDCGraph {
  PetscInt nvtxs;
  PetscInt ncmps;
  PetscInt *xadj;
  PetscInt *adjncy;
  PetscInt *where;
  PetscInt *which_dof;
  PetscInt *queue;
  PetscInt *cptr;
  PetscInt *count;
  PetscBool *touched;
};

typedef enum {SCATTERS_BDDC,GATHERS_BDDC} CoarseCommunicationsType;
typedef struct _PCBDDCGraph *PCBDDCGraph;

/* Private context (data structure) for the BDDC preconditioner.  */
typedef struct {
  /* First MUST come the folowing line, for the stuff that is common to FETI and Neumann-Neumann. */
  PC_IS         pcis;
  /* Coarse stuffs needed by BDDC application in KSP */
  PetscInt      coarse_size;
  Mat           coarse_mat;
  Vec           coarse_vec;
  Vec           coarse_rhs;
  KSP           coarse_ksp;
  Mat           coarse_phi_B;
  Mat           coarse_phi_D;
  PetscMPIInt   local_primal_size;
  PetscMPIInt   *local_primal_indices;
  PetscMPIInt   *local_primal_displacements;
  PetscMPIInt   *local_primal_sizes;
  PetscMPIInt   replicated_primal_size;
  PetscMPIInt   *replicated_local_primal_indices;
  PetscScalar   *replicated_local_primal_values;
  VecScatter    coarse_loc_to_glob;
  /* Local stuffs needed by BDDC application in KSP */
  Vec           vec1_P;
  Vec           vec1_C;
  Mat           local_auxmat1;
  Mat           local_auxmat2;
  Vec           vec1_R;
  Vec           vec2_R;
  VecScatter    R_to_B;
  VecScatter    R_to_D;
  KSP           ksp_R;
  KSP           ksp_D;
  Vec           vec4_D;
  /* Quantities defining constraining details (local) of the preconditioner */
  /* These quantities define the preconditioner itself */
  PetscInt      n_constraints;
  PetscInt      n_vertices;
  PetscInt      *vertices;
  PetscInt      *sizes_of_constraint;
  PetscInt      **indices_to_constraint;
  PetscScalar   **quadrature_constraint;
  /* Some defaults on selecting vertices and constraints*/
  PetscBool     vertices_flag;
  PetscBool     constraints_flag;
  PetscBool     faces_flag;
  PetscBool     edges_flag;
  /* Some customization is possible */
  IS                         NeumannBoundaries;
  PetscBool                  prec_type;
  CoarseProblemType          coarse_problem_type;
  CoarseCommunicationsType   coarse_communications_type;
  PetscInt                   coarsening_ratio;
  PetscInt                   active_procs;
  /* For verbose output of some bddc data structures */
  PetscBool                  check_flag;
} PC_BDDC;

/* In case of multilevel BDDC, this is the minimum number of procs for which it will be allowed */
#define MIN_PROCS_FOR_BDDC 16

/* prototypes for functions contained in bddc.c */
static PetscErrorCode PCBDDCCoarseSetUp(PC);
static PetscErrorCode PCBDDCFindConnectedComponents(PCBDDCGraph,PetscInt,PetscInt* );
static PetscErrorCode PCBDDCSetupCoarseEnvironment(PC,PetscScalar*);
static PetscErrorCode PCBDDCManageLocalBoundaries(PC);
static PetscErrorCode PCBDDCApplyInterfacePreconditioner(PC,Vec);
static PetscErrorCode PCBDDCSolveSaddlePoint(PC);
static PetscErrorCode PCBDDCScatterCoarseDataBegin(PC,Vec,Vec,InsertMode,ScatterMode);
static PetscErrorCode PCBDDCScatterCoarseDataEnd(PC,Vec,Vec,InsertMode,ScatterMode);

#endif /* __pcbddc_h */
