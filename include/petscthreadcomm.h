
#if !defined(__PETSCTHREADCOMM_H)
#define __PETSCTHREADCOMM_H
#include "petscsys.h"
PETSC_EXTERN_CXX_BEGIN

extern PetscClassId PETSCTHREADCOMM_CLASSID;

/* Function pointer cast for the kernel function */
typedef PetscErrorCode (*PetscThreadKernel)(PetscInt,...);

/* Max. number of arguments for kernel */
#define PETSC_KERNEL_NARGS_MAX 10
/*S
  ThreadComm - Abstract PETSc object that manages all thread communication models

  Level: developer

  Concepts: threads

.seealso: PetscThreadCommCreate(), PetscThreadCommDestroy
S*/
typedef struct _p_PetscThreadComm* PetscThreadComm;

/*MC
   PETSC_THREAD_COMM_WORLD - The basic thread communicator which uses all the threads that
                 PETSc knows about.

   Level: developer

   Notes: This is created during PetscInitialize() with the number of threads
          requested. All the other thread communicators use threads from
          PETSC_THREAD_COMM_WORLD for performing their operations.	  
M*/		 
extern PetscThreadComm PETSC_THREAD_COMM_WORLD;

extern PetscInt N_CORES; /* Number of available cores */

#define PetscThreadCommType char*
#define PTHREAD             "pthread"

extern PetscFList PetscThreadCommList;

extern PetscErrorCode PetscThreadCommInitializePackage(const char*);
extern PetscErrorCode PetscThreadCommFinalizePackage(void);
extern PetscErrorCode PetscThreadCommCreate(PetscThreadComm*);
extern PetscErrorCode PetscThreadCommDestroy(PetscThreadComm*);
extern PetscErrorCode PetscThreadCommReference(PetscThreadComm,PetscThreadComm*);
extern PetscErrorCode PetscThreadCommSetNThreads(PetscThreadComm,PetscInt);
extern PetscErrorCode PetscThreadCommGetNThreads(PetscThreadComm,PetscInt*);
extern PetscErrorCode PetscThreadCommSetAffinities(PetscThreadComm,const PetscInt[]);
extern PetscErrorCode PetscThreadCommGetAffinities(PetscThreadComm,PetscInt[]);
extern PetscErrorCode PetscThreadCommView(PetscThreadComm,PetscViewer);
extern PetscErrorCode PetscThreadCommSetType(PetscThreadComm,const PetscThreadCommType);
extern PetscErrorCode PetscThreadCommRunKernel(PetscThreadComm,PetscErrorCode (*)(PetscInt,...),PetscInt,...);

/* register thread communicator models */
extern PetscErrorCode PetscThreadCommRegister(const char[],const char[],const char[],PetscErrorCode(*)(PetscThreadComm));
extern PetscErrorCode PetscThreadCommRegisterAll(const char path[]);
extern PetscErrorCode PetscThreadCommRegisterDestroy(void);

#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
#define PetscThreadCommRegisterDynamic(a,b,c,d) PetscThreadCommRegister(a,b,c,0)
#else
#define PetscThreadCommRegisterDynamic(a,b,c,d) PetscThreadCommRegister(a,b,c,d)
#endif

PETSC_EXTERN_CXX_END
#endif
