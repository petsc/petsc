#if !defined(__GAMG_IMPL)
#define __GAMG_IMPL
#include <private/pcimpl.h>   /*I "petscpc.h" I*/
#include <../src/ksp/pc/impls/mg/mgimpl.h>                    /*I "petscpcmg.h" I*/
#include <../src/mat/impls/aij/seq/aij.h>
#include <../src/mat/impls/aij/mpi/mpiaij.h>
#include <assert.h>

extern PetscErrorCode PCSetFromOptions_MG(PC);
extern PetscErrorCode PCReset_MG(PC);
extern PetscErrorCode createProlongation( Mat, PetscReal [], const PetscInt,
                                          Mat *, PetscReal **, PetscBool *a_isOK );
enum tag {SET1,SET2,SET3,SET4,SET5,SET6,FIND_V,SET7,NUM_SET};
extern PetscLogEvent gamg_setup_stages[NUM_SET];

#endif

