#if !defined(__GAMG_IMPL)
#define __GAMG_IMPL
#include <private/pcimpl.h>   /*I "petscpc.h" I*/
#include <../src/ksp/pc/impls/mg/mgimpl.h>                    /*I "petscpcmg.h" I*/
#include <../src/mat/impls/aij/seq/aij.h>
#include <../src/mat/impls/aij/mpi/mpiaij.h>
#include <assert.h>

extern PetscErrorCode PCSetFromOptions_MG(PC);
extern PetscErrorCode PCReset_MG(PC);
extern PetscErrorCode createProlongation( const Mat, const PetscReal [], const PetscInt, const PetscInt, 
                                          const PetscBool, const PetscInt, 
                                          PetscInt*, Mat *, PetscReal **, PetscBool *, PetscReal *);
enum tag {SET1,SET2,SET3,SET4,SET5,SET6,FIND_V,SET7,SET8,SET9,SET10,SET11,SET12,SET13,NUM_SET};
extern PetscLogEvent gamg_setup_stages[NUM_SET];

#define PETSC_GAMG_SMOOTHER PCJACOBI

#endif

