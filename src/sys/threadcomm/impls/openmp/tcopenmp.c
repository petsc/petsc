#define PETSC_DESIRE_FEATURE_TEST_MACROS
#include <../src/sys/threadcomm/impls/openmp/tcopenmpimpl.h>
#include <omp.h>

PetscErrorCode PetscThreadCommGetRank_OpenMP(PetscInt *trank)
{
  *trank =  omp_get_thread_num();
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommCreate_OpenMP"
PETSC_EXTERN PetscErrorCode PetscThreadCommCreate_OpenMP(PetscThreadComm tcomm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr                  = PetscStrcpy(tcomm->type,OPENMP);CHKERRQ(ierr);
  tcomm->ops->runkernel = PetscThreadCommRunKernel_OpenMP;
  tcomm->ops->getrank   = PetscThreadCommGetRank_OpenMP;
#pragma omp parallel num_threads(tcomm->nworkThreads) shared(tcomm)
  {
#if defined(PETSC_HAVE_SCHED_CPU_SET_T)
    cpu_set_t mset;
    PetscInt  ncores, icorr,trank;
    PetscGetNCores(&ncores);
    CPU_ZERO(&mset);
    trank = omp_get_thread_num();
    icorr = tcomm->affinities[trank]%ncores;
    CPU_SET(icorr,&mset);
    sched_setaffinity(0,sizeof(cpu_set_t),&mset);
#endif
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommRunKernel_OpenMP"
PetscErrorCode PetscThreadCommRunKernel_OpenMP(PetscThreadComm tcomm,PetscThreadCommJobCtx job)
{
  PetscInt        trank=0;

  PetscFunctionBegin;
#pragma omp parallel num_threads(tcomm->nworkThreads) shared(job) private(trank)
  {
    trank = omp_get_thread_num();
    PetscRunKernel(trank,job->nargs,job);
    job->job_status[trank] = THREAD_JOB_COMPLETED;
  }
  PetscFunctionReturn(0);
}
