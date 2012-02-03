#if !defined(__GAMG_IMPL)
#define __GAMG_IMPL
#include <private/pcimpl.h>   /*I "petscpc.h" I*/
#include <../src/ksp/pc/impls/mg/mgimpl.h>                    /*I "petscpcmg.h" I*/
#include <../src/mat/impls/aij/seq/aij.h>
#include <../src/mat/impls/aij/mpi/mpiaij.h>
#include <assert.h>

/* Private context for the GAMG preconditioner */
typedef struct gamg_TAG{
  PetscInt       Nlevels;
  PetscInt       setup_count;
  PetscBool      repart;
  PetscInt       min_eq_proc;
  PetscInt       coarse_eq_limit;
  PetscReal      threshold; /* common quatity to many AMG methods so keep it up here */
  PetscInt       verbose;
  PetscInt       emax_id;
  PetscInt       col_bs_id;
  PetscInt       data_sz;   /* these 4 things are all related to the method data and should be in the subctx */
  PetscInt       data_rows;
  PetscInt       data_cols;
  PetscReal     *data; /* blocked vector of vertex data on fine grid (coordinates) */
  PetscErrorCode (*createprolongator)( PC, const Mat A, const PetscReal [], Mat *P, PetscReal** );
  PetscErrorCode (*createdefaultdata)( PC );

  void          *subctx;
} PC_GAMG;

#define PCGAMGType char*

#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
#  define PCGAMGRegisterDynamic(a,b,c,d)       PCGAMGRegister(a,b,c,0)
#else
#  define PCGAMGRegisterDynamic(a,b,c,d)       PCGAMGRegister(a,b,c,d)
#endif

PetscErrorCode PCGAMGRegister(const char *implname,const char *path,const char *fname,PetscErrorCode (*cfunc)(PC));

PetscErrorCode PCGAMGSetType( PC,const PCGAMGType );

#define GAMGAGG "agg"
#define GAMGGEO "geo"

/* Private context for the GAMG preconditioner */
typedef struct{
  PetscInt       lid;      /* local vertex index */
  PetscInt       degree;   /* vertex degree */
} GAMGNode;

PetscErrorCode PCSetFromOptions_MG(PC);
PetscErrorCode PCReset_MG(PC);

/* hooks create derivied classes */
PetscErrorCode  PCCreateGAMG_GEO( PC pc );
PetscErrorCode  PCCreateGAMG_AGG( PC pc );

PetscErrorCode PCSetFromOptions_GAMG( PC pc );
PetscErrorCode PCDestroy_GAMG(PC pc);

PetscErrorCode PCGAMGcreateProl_AGG( PC, const Mat, const PetscReal [], Mat *, PetscReal **);
PetscErrorCode PCGAMGcreateProl_GEO( PC, const Mat, const PetscReal [], Mat *, PetscReal **);
/* helper methods */
PetscErrorCode getDataWithGhosts( const Mat a_Gmat, const PetscInt a_data_sz, const PetscReal a_data_in[],
                                  PetscInt *a_stride, PetscReal **a_data_out );
PetscErrorCode maxIndSetAgg( const IS a_perm, const Mat a_Gmat, const Mat a_Auxmat,
			     const PetscBool a_strict_aggs, IS *a_selected, IS *a_locals_llist );
PetscErrorCode createGraph(PC a_pc, const Mat a_Amat, Mat *, Mat *, IS * );

#if defined PETSC_USE_LOG
enum tag {SET1,SET2,GRAPH,GRAPH_MAT,GRAPH_FILTER,GRAPH_SQR,SET4,SET5,SET6,FIND_V,SET7,SET8,SET9,SET10,SET11,SET12,SET13,SET14,SET15,SET16,NUM_SET};
extern PetscLogEvent gamg_setup_events[NUM_SET];
#endif

#endif

