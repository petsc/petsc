#define PETSC_DESIRE_FEATURE_TEST_MACROS
#include <../src/sys/threadcomm/impls/openmp/tcopenmpimpl.h>
#include <omp.h>

PetscErrorCode PetscThreadCommGetRank_OpenMP(PetscInt *trank)
{
  *trank =  omp_get_thread_num();
  return 0;
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommCreate_OpenMP"
PetscErrorCode PetscThreadCommCreate_OpenMP(PetscThreadComm tcomm)
{
  PetscErrorCode ierr;
  PetscInt       trank;

  PetscFunctionBegin;
  ierr = PetscStrcpy(tcomm->type,OPENMP);CHKERRQ(ierr);
  tcomm->ops->runkernel = PetscThreadCommRunKernel_OpenMP;
  tcomm->ops->getrank   = PetscThreadCommGetRank_OpenMP;
#pragma omp parallel num_threads(tcomm->nworkThreads) shared(tcomm) private(ierr,trank)
  {
#if defined(PETSC_HAVE_SCHED_CPU_SET_T)
    cpu_set_t mset;
    PetscInt ncores, icorr;
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
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommRunKernel_OpenMP"
PetscErrorCode PetscThreadCommRunKernel_OpenMP(MPI_Comm comm,PetscThreadCommJobCtx job)
{
  PetscErrorCode ierr;
  PetscThreadComm tcomm;
  PetscInt        trank=0;

  PetscFunctionBegin;
  ierr = PetscCommGetThreadComm(comm,&tcomm);CHKERRQ(ierr);
#pragma omp parallel num_threads(tcomm->nworkThreads) shared(comm,job) private(trank,ierr)
  {
    trank = omp_get_thread_num();
    PetscRunKernel(trank,job->nargs,job);
    job->job_status[trank] = THREAD_JOB_COMPLETED;
  }
  PetscFunctionReturn(0);
}
