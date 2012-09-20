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

  PetscFunctionBegin;
  ierr = PetscStrcpy(tcomm->type,OPENMP);CHKERRQ(ierr);
  tcomm->ops->runkernel = PetscThreadCommRunKernel_OpenMP;
  tcomm->ops->getrank   = PetscThreadCommGetRank_OpenMP;
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
