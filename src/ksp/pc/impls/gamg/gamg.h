#if !defined(__GAMG_IMPL)
#define __GAMG_IMPL
#include <private/pcimpl.h>   /*I "petscpc.h" I*/
#include <../src/ksp/pc/impls/mg/mgimpl.h>                    /*I "petscpcmg.h" I*/
#include <../src/mat/impls/aij/seq/aij.h>
#include <../src/mat/impls/aij/mpi/mpiaij.h>
#include <assert.h>

/* Private context for the GAMG preconditioner */
typedef struct gamg_TAG {
  PetscInt       m_dim;
  PetscInt       m_Nlevels;
  PetscInt       m_data_sz;
  PetscInt       m_data_rows;
  PetscInt       m_data_cols;
  PetscInt       m_count;
  PetscInt       m_method; /* 0: geomg; 1: plain agg 'pa'; 2: smoothed agg 'sa' */
  PetscReal     *m_data; /* blocked vector of vertex data on fine grid (coordinates) */
  char           m_type[64];
  PetscBool      m_avoid_repart;
  PetscInt       m_min_eq_proc;
  PetscInt       m_coarse_eq_limit;
  PetscReal      m_threshold;
  PetscBool      m_verbose;
} PC_GAMG;

extern PetscErrorCode PCSetFromOptions_MG(PC);
extern PetscErrorCode PCReset_MG(PC);
extern PetscErrorCode createProlongation( const Mat, const PetscReal [], const PetscInt, 
                                          PetscInt*, Mat *, PetscReal **, PetscBool *, PetscReal *, const PC_GAMG* );
#if defined PETSC_USE_LOG
enum tag {SET1,SET2,GRAPH,GRAPH_MAT,GRAPH_FILTER,GRAPH_SQR,SET4,SET5,SET6,FIND_V,SET7,SET8,SET9,SET10,SET11,SET12,SET13,NUM_SET};
extern PetscLogEvent gamg_setup_events[NUM_SET];
#endif

#define PETSC_GAMG_SMOOTHER PCJACOBI

#endif

