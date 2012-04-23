
#ifndef __PTHREADIMPLH
#define __PTHREADIMPLH

#include <petsc-private/threadcommimpl.h>

#if defined(PETSC_HAVE_PTHREAD_H)
#include <pthread.h>
#elif defined(PETSC_HAVE_WINPTHREADS_H)
#include "winpthreads.h"       /* http://locklessinc.com/downloads/winpthreads.h */
#endif

/*
  PetscPThreadCommSynchronizationType - Type of thread synchronization for pthreads communicator.

$ PTHREADSYNC_LOCKFREE -  A lock-free variant.

*/
typedef enum {PTHREADSYNC_LOCKFREE} PetscPThreadCommSynchronizationType;
extern const char *const PetscPThreadCommSynchronizationTypes[];

/*
  PetscPThreadCommAffinityPolicy - Core affinity policy for pthreads

$ PTHREADAFFPOLICY_ALL     - threads can run on any core. OS decides thread scheduling
$ PTHREADAFFPOLICY_ONECORE - threads can run on only one core.
$ PTHREADAFFPOLICY_NONE    - No set affinity policy. OS decides thread scheduling
*/
typedef enum {PTHREADAFFPOLICY_ALL,PTHREADAFFPOLICY_ONECORE,PTHREADAFFPOLICY_NONE} PetscPThreadCommAffinityPolicyType;
extern const char *const PetscPTheadCommAffinityPolicyTypes[];

/*
   PetscThreadComm_PThread - The main data structure to manage the thread
   communicator using pthreads. This data structure is shared by NONTHREADED
   and PTHREAD threading models. For NONTHREADED threading model, no extra
   pthreads are created
*/
typedef struct {
  PetscInt    nthreads;            /* Number of threads created */
  pthread_t  *tid;                 /* thread ids */
  PetscBool  ismainworker;         /* Is the main thread also a work thread?*/
  PetscInt   *ranks;               /* Thread ranks - if main thread is a worker then main thread 
				      rank is 0 and ranks for other threads start from 1, 
				      otherwise the thread ranks start from 0 */
  PetscInt    thread_num_start;     /* index for the first created thread (= 1 if the main thread is a worker
                                       else 0) */
  PetscPThreadCommSynchronizationType sync; /* Synchronization type */
  PetscPThreadCommAffinityPolicyType  aff; /* affinity policy */
  PetscErrorCode (*initialize)(PetscThreadComm);
  PetscErrorCode (*finalize)(PetscThreadComm);
  PetscErrorCode (*runkernel)(PetscThreadComm,PetscErrorCode (*)(void*),void**);
}PetscThreadComm_PThread;

#if defined(PETSC_PTHREAD_LOCAL)
extern PETSC_PTHREAD_LOCAL PetscInt PetscPThreadRank; /* Rank of the calling thread ... thread local variable */
#else
extern pthread_key_t  PetscPThreadRankkey;
#endif

#define PetscReadOnce(type,val) (*(volatile type *)&val)

EXTERN_C_BEGIN
extern PetscErrorCode PetscThreadCommCreate_PThread(PetscThreadComm);
EXTERN_C_END

extern PetscErrorCode PetscPThreadCommInitialize_LockFree(PetscThreadComm);
extern PetscErrorCode PetscPThreadCommFinalize_LockFree(PetscThreadComm);
extern PetscErrorCode PetscPThreadCommRunKernel_LockFree(PetscThreadComm,PetscErrorCode (*)(void*),void**);

#if defined(PETSC_HAVE_SCHED_CPU_SET_T)
extern void PetscPThreadCommDoCoreAffinity();
#endif


#endif
