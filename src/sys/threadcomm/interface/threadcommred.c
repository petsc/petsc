/* This file contains code for threaded reductions */
#include <petsc-private/threadcommimpl.h>      /*I "petscthreadcomm.h" I*/

#undef __FUNCT__
#define __FUNCT__ "PetscThreadReductionBegin"
/*@C
   PetscThreadReductionBegin - Initiates a threaded reduction and returns a
                               reduction object to be passed to PetscThreadCommRunKernel

   Input Parameters:
+  comm - the MPI comm
.  op   - the reduction operation
.  type - the data type for reduction
.  red  - the reduction context

   Level: developer

   Notes:
   See include/petscthreadcomm.h for the available reduction operations

   To be called from the main thread before calling PetscThreadCommRunKernel

.seealso: PetscThreadCommReductionKernelBegin(), PetscThreadCommReductionKernelEnd(), PetscThreadCommReductionEnd()
@*/
PetscErrorCode PetscThreadReductionBegin(MPI_Comm comm,PetscThreadCommReductionOp op, PetscDataType type,PetscThreadCommRedCtx *red)
{
  PetscErrorCode ierr;
  PetscThreadComm tcomm;

  PetscFunctionBegin;
  ierr = PetscCommGetThreadComm(comm,&tcomm);CHKERRQ(ierr);
  tcomm->red->op = op;
  tcomm->red->type = type;
  tcomm->red->red_status = THREADCOMM_REDUCTION_NEW;
  tcomm->red->tcomm = tcomm;
  *red = tcomm->red;
  PetscFunctionReturn(0);
}
  
#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommReductionDestroy"
/* 
   PetscThreadCommReductionDestroy - Destroys the reduction context

   Input Parameters:
.  red - the reduction context

*/
PetscErrorCode PetscThreadCommReductionDestroy(PetscThreadCommRedCtx red)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if(!red) PetscFunctionReturn(0);

  ierr = PetscFree(red->thread_status);CHKERRQ(ierr);
  ierr = PetscFree(red->local_red);CHKERRQ(ierr);
  ierr = PetscFree(red);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadReductionKernelBegin"
/*
   PetscThreadReductionKernelBegin - Begins a threaded reduction operation

   Input Parameters:
+  trank   - Rank of the calling thread
.  red     - the reduction context 
.  lred    - local contribution from the thread

   Level: developer

   Notes:
   This routine posts the local reduction of each thread in the reduction context and
   updates its reduction status. 

   Must call PetscThreadReductionBegin before launching the kernel.
*/
PetscErrorCode PetscThreadReductionKernelBegin(PetscInt trank,PetscThreadCommRedCtx red,void* lred)
{
  if(PetscReadOnce(int,red->red_status) != THREADCOMM_REDUCTION_NEW) {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Did not call PetscThreadReductionBegin() before calling PetscThreadCommRunKernel()");
  }

  switch(red->type) {
  case PETSC_INT:
    ((PetscInt*)red->local_red)[trank] = *(PetscInt*)lred;
    break;
#if defined(PETSC_USE_COMPLEX)
  case PETSC_REAL:
    ((PetscReal*)red->local_red)[trank] = *(PetscReal*)lred;
    break;
#endif
  case PETSC_SCALAR:
    ((PetscScalar*)red->local_red)[trank] = *(PetscScalar*)lred;
    break;
  default:
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Unknown datatype provided for kernel reduction");
    break;
  }
  red->thread_status[trank] = THREADCOMM_THREAD_POSTED_LOCALRED;
  return 0;
}

/* Completes the given reduction */
#undef __FUNCT__
#define __FUNCT__ "PetscThreadReductionEnd_Private"
PetscErrorCode PetscThreadReductionEnd_Private(PetscThreadCommRedCtx red,void * outdata)
{
  /* Check whether all threads have posted their contributions */
  PetscBool wait=PETSC_TRUE;
  PetscInt  i;
  while(wait) {
    for(i=0;i < red->tcomm->nworkThreads;i++) { 
      if(PetscReadOnce(int,red->thread_status[i]) != THREADCOMM_THREAD_POSTED_LOCALRED) {
	wait = PETSC_TRUE;
	break;
      }
      wait = PETSC_FALSE;
    }
  }
  
  /* Apply the reduction operation */
  switch(red->op) {
  case THREADCOMM_SUM:
    if(red->type == PETSC_REAL) {
      PetscReal red_sum=*(PetscReal*)outdata;
      for(i=0; i < red->tcomm->nworkThreads;i++) {
	red_sum += ((PetscReal*)red->local_red)[i];
      }
      PetscMemcpy(outdata,&red_sum,sizeof(PetscReal));
      break;
    }
    if(red->type == PETSC_SCALAR) {
      PetscScalar red_sum=*(PetscScalar*)outdata;
      for(i=0; i < red->tcomm->nworkThreads;i++) {
	red_sum += ((PetscScalar*)red->local_red)[i];
      }
      PetscMemcpy(outdata,&red_sum,sizeof(PetscScalar));
      break;
    }
    if(red->type == PETSC_INT) {
      PetscInt red_sum=*(PetscInt*)outdata;
      for(i=0; i < red->tcomm->nworkThreads;i++) {
	red_sum += ((PetscInt*)red->local_red)[i];
      }
      PetscMemcpy(outdata,&red_sum,sizeof(PetscInt));
      break;
    }
    break;
  case THREADCOMM_PROD:
    if(red->type == PETSC_REAL) {
      PetscReal red_prod=*(PetscReal*)outdata;
      for(i=0; i < red->tcomm->nworkThreads;i++) {
	red_prod *= ((PetscReal*)red->local_red)[i];
      }
      PetscMemcpy(outdata,&red_prod,sizeof(PetscReal));
      break;
    }
    if(red->type == PETSC_SCALAR) {
      PetscScalar red_prod=*(PetscScalar*)outdata;
      for(i=0; i < red->tcomm->nworkThreads;i++) {
	red_prod *= ((PetscScalar*)red->local_red)[i];
      }
      PetscMemcpy(outdata,&red_prod,sizeof(PetscScalar));
      break;
    }
    if(red->type == PETSC_INT) {
      PetscInt red_prod=*(PetscInt*)outdata;
      for(i=0; i < red->tcomm->nworkThreads;i++) {
	red_prod *= ((PetscInt*)red->local_red)[i];
      }
      PetscMemcpy(outdata,&red_prod,sizeof(PetscInt));
      break;
    }
    break;
  default:
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Undefined thread reduction operation");
    break;
  }
  return 0;
}
  
#undef __FUNCT__
#define __FUNCT__ "PetscThreadReductionEnd"
/*@C 
   PetscThreadReductionEnd - Completes the given reduction

   Input Parameters:
+  red     - the reduction context
.  outdata - the reduction result

   Level: developer

   Notes:
   To be called by the main thread only
@*/
PetscErrorCode PetscThreadReductionEnd(PetscThreadCommRedCtx red,void *outdata)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscThreadReductionEnd_Private(red,outdata);CHKERRQ(ierr);
  red->red_status = THREADCOMM_REDUCTION_COMPLETE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadReductionKernelEnd"
/*
   PetscThreadReductionKernelBegin - Finishes a reduction operation

   Input Parameters:
+  trank   - Rank of the calling thread
.  red     - the reduction context
-  outdata - the reduction result 

   Level: developer

   Notes: This should be called only from kernels only if the reduction needs to 
   be completed while in the kernel for some future operation.

*/
PetscErrorCode PetscThreadReductionKernelEnd(PetscInt trank,PetscThreadCommRedCtx red,void *outdata)
{

  if(PetscReadOnce(int,red->tcomm->leader) == trank) {
    PetscThreadReductionEnd_Private(red,outdata);
    red->red_status = THREADCOMM_REDUCTION_COMPLETE;
  }

  /* Wait till the leader performs the reduction so that the other threads
     can also see the reduction result */
  while(PetscReadOnce(int,red->red_status) != THREADCOMM_REDUCTION_COMPLETE) 
    ;
  red->thread_status[trank] = THREADCOMM_THREAD_WAITING_FOR_NEWRED;
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommReductionCreate"
/*
   PetscThreadCommReductionCreate - Allocates the reduction context and
                                 initializes it

   Input Parameters:
+  tcomm - the thread communicator
.  red   - the reduction context

*/
PetscErrorCode PetscThreadCommReductionCreate(PetscThreadComm tcomm,PetscThreadCommRedCtx *red)
{
  PetscErrorCode        ierr;
  PetscThreadCommRedCtx redout;
  PetscInt              i;
  
  PetscFunctionBegin;
  ierr = PetscNew(struct _p_PetscThreadCommRedCtx,&redout);CHKERRQ(ierr);
  ierr = PetscMalloc(tcomm->nworkThreads*sizeof(PetscInt),&redout->thread_status);CHKERRQ(ierr);
  ierr = PetscMalloc(tcomm->nworkThreads*sizeof(PetscScalar),&redout->local_red);CHKERRQ(ierr);
  redout->nworkThreads = tcomm->nworkThreads;
  redout->red_status = THREADCOMM_REDUCTION_NONE;
  for(i=0;i<redout->nworkThreads;i++) redout->thread_status[i] = THREADCOMM_THREAD_WAITING_FOR_NEWRED;
  *red = redout;
  PetscFunctionReturn(0);
}

