#if !defined(__GAMG_IMPL)
#define __GAMG_IMPL
#include <private/pcimpl.h>   /*I "petscpc.h" I*/
#include <../src/ksp/pc/impls/mg/mgimpl.h>                    /*I "petscpcmg.h" I*/
#include <../src/mat/impls/aij/seq/aij.h>
#include <../src/mat/impls/aij/mpi/mpiaij.h>
#include <assert.h>

extern PetscErrorCode PCSetFromOptions_MG(PC);
extern PetscErrorCode PCReset_MG(PC);
extern PetscErrorCode createProlongation( const Mat, const PetscReal [], const PetscInt, 
                                          const PetscInt, const PetscInt, const PetscInt,
                                          const PetscReal,
                                          PetscInt*, Mat *, PetscReal **, PetscBool *, PetscReal *,
                                          const PetscBool);
#if defined PETSC_USE_LOG
enum tag {SET1,SET2,GRAPH,GRAPH_MAT,GRAPH_FILTER,GRAPH_SQR,SET4,SET5,SET6,FIND_V,SET7,SET8,SET9,SET10,SET11,SET12,SET13,NUM_SET};
extern PetscLogEvent gamg_setup_events[NUM_SET];
#endif

#define PETSC_GAMG_SMOOTHER PCJACOBI

#endif

