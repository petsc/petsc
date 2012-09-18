#if !defined(__GAMG_IMPL)
#define __GAMG_IMPL
#include <petsc-private/pcimpl.h>   /*I "petscpc.h" I*/
#include <../src/ksp/pc/impls/mg/mgimpl.h>                    /*I "petscpcmg.h" I*/
#include <../src/mat/impls/aij/seq/aij.h>
#include <../src/mat/impls/aij/mpi/mpiaij.h>
#include <assert.h>

/* Private context for the GAMG preconditioner */
typedef struct gamg_TAG{
  PetscInt       Nlevels;
  PetscInt       setup_count;
  PetscBool      repart;
  PetscBool      use_aggs_in_gasm;
  PetscInt       min_eq_proc;
  PetscInt       coarse_eq_limit;
  PetscReal      threshold; /* common quatity to many AMG methods so keep it up here */
  PetscInt       verbose;
  PetscInt       emax_id; /* stashing places */
  /* these 4 are all related to the method data and should be in the subctx */
  PetscInt       data_sz; /* nloc*data_rows*data_cols */
  PetscInt       data_cell_rows;
  PetscInt       data_cell_cols;
  PetscInt       orig_data_cell_rows;
  PetscInt       orig_data_cell_cols;
  PetscReal      eigtarget[2];
  PetscReal     *data;      /* [data_sz] blocked vector of vertex data on fine grid (coordinates/nullspace) */
  PetscReal     *orig_data;      /* cache data */
  PetscErrorCode (*graph)( PC, const Mat, Mat * );
  PetscErrorCode (*coarsen)( PC, Mat *, PetscCoarsenData** );
  PetscErrorCode (*prolongator)( PC, const Mat, const Mat, PetscCoarsenData *, Mat* );
  PetscErrorCode (*optprol)( PC, const Mat, Mat* );
  PetscErrorCode (*formkktprol)( PC, const Mat, const Mat, Mat* );
  PetscErrorCode (*createdefaultdata)( PC, Mat ); /* for data methods that have a default (SA) */
  void          *subctx;
} PC_GAMG;

#define GAMGAGG "agg"
#define GAMGGEO "geo"

PetscErrorCode PCSetFromOptions_MG( PC );
PetscErrorCode PCReset_MG( PC );

/* hooks create derivied classes */
PetscErrorCode  PCCreateGAMG_GEO( PC pc );
PetscErrorCode  PCCreateGAMG_AGG( PC pc );

PetscErrorCode PCSetFromOptions_GAMG( PC pc );
PetscErrorCode PCDestroy_GAMG(PC pc);

/* helper methods */
PetscErrorCode PCGAMGCreateGraph( const Mat, Mat * );
PetscErrorCode PCGAMGFilterGraph( Mat *, const PetscReal, const PetscBool, const PetscInt );
PetscErrorCode PCGAMGGetDataWithGhosts( const Mat a_Gmat, const PetscInt a_data_sz, const PetscReal a_data_in[],
                                       PetscInt *a_stride, PetscReal **a_data_out );

#if defined PETSC_USE_LOG
/* #define PETSC_GAMG_USE_LOG */
enum tag {SET1,SET2,GRAPH,GRAPH_MAT,GRAPH_FILTER,GRAPH_SQR,SET4,SET5,SET6,FIND_V,SET7,SET8,SET9,SET10,SET11,SET12,SET13,SET14,SET15,SET16,NUM_SET};
#if defined PETSC_GAMG_USE_LOG
extern PetscLogEvent petsc_gamg_setup_events[NUM_SET];
#endif
extern PetscLogEvent PC_GAMGGgraph_AGG;
extern PetscLogEvent PC_GAMGGgraph_GEO;
extern PetscLogEvent PC_GAMGCoarsen_AGG;
extern PetscLogEvent PC_GAMGCoarsen_GEO;
extern PetscLogEvent PC_GAMGProlongator_AGG;
extern PetscLogEvent PC_GAMGProlongator_GEO;
extern PetscLogEvent PC_GAMGOptprol_AGG;
extern PetscLogEvent PC_GAMGKKTProl_AGG;
#endif

typedef struct _GAMGHashTable{
  PetscInt  *table;
  PetscInt  *data;
  PetscInt   size;
}GAMGHashTable;

extern PetscErrorCode GAMGTableCreate( PetscInt a_size, GAMGHashTable *a_tab );
extern PetscErrorCode GAMGTableDestroy( GAMGHashTable * );
extern PetscErrorCode GAMGTableAdd( GAMGHashTable *a_tab, PetscInt a_key, PetscInt a_data );
extern PetscErrorCode GAMGTableFind( GAMGHashTable *a_tab, PetscInt a_key, PetscInt *a_data );

#endif

