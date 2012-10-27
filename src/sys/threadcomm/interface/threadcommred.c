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
-  nreds - Number of reductions

   Output Parameters:
.  redout  - the reduction context

   Level: developer

   Notes:
   See include/petscthreadcomm.h for the available reduction operations

   To be called from the main thread before calling PetscThreadCommRunKernel

.seealso: PetscThreadCommReductionKernelPost(), PetscThreadCommReductionKernelEnd(), PetscThreadCommReductionEnd()
@*/
PetscErrorCode PetscThreadReductionBegin(MPI_Comm comm,PetscThreadCommReductionOp op, PetscDataType type,PetscInt nreds,PetscThreadCommReduction *redout)
{
  PetscErrorCode ierr;
  PetscThreadComm tcomm;
  PetscInt        i;
  PetscThreadCommRedCtx redctx;
  PetscThreadCommReduction red;

  PetscFunctionBegin;
  ierr = PetscCommGetThreadComm(comm,&tcomm);CHKERRQ(ierr);
  red = tcomm->red;
  if(red->ctr+nreds > PETSC_REDUCTIONS_MAX) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Reductions in operation: %D Max. allowed: %D",red->ctr+nreds,PETSC_REDUCTIONS_MAX);
  for(i=red->ctr;i<red->ctr+nreds;i++) {
    redctx = red->redctx[i];
    redctx->op = op;
    redctx->type = type;
    redctx->red_status = THREADCOMM_REDUCTION_NEW;
    redctx->tcomm = tcomm;
  }
  red->nreds += nreds;
  red->ctr = red->ctr+nreds;
  *redout = red;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommReductionDestroy"
/*
   PetscThreadCommReductionDestroy - Destroys the reduction context

   Input Parameters:
.  red - the reduction context

*/
PetscErrorCode PetscThreadCommReductionDestroy(PetscThreadCommReduction red)
{
  PetscErrorCode        ierr;
  PetscThreadCommRedCtx redctx;
  PetscInt              i;

  PetscFunctionBegin;
  if (!red) PetscFunctionReturn(0);

  for(i=0;i<PETSC_REDUCTIONS_MAX;i++) {
    redctx = red->redctx[i];
    ierr = PetscFree(redctx->thread_status);CHKERRQ(ierr);
    ierr = PetscFree(redctx->local_red);CHKERRQ(ierr);
    ierr = PetscFree(redctx);CHKERRQ(ierr);
  }
  ierr = PetscFree(red->thread_ctr);CHKERRQ(ierr);
  ierr = PetscFree(red);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadReductionKernelPost"
/*
   PetscThreadReductionKernelPost - Begins a threaded reduction operation

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
PetscErrorCode PetscThreadReductionKernelPost(PetscInt trank,PetscThreadCommReduction red,void* lred)
{
  PetscThreadCommRedCtx redctx=red->redctx[red->thread_ctr[trank]];
  red->thread_ctr[trank] = (red->thread_ctr[trank]+1)%PETSC_REDUCTIONS_MAX;

  if (PetscReadOnce(int,redctx->red_status) != THREADCOMM_REDUCTION_NEW) {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Did not call PetscThreadReductionBegin() before calling PetscThreadCommRunKernel()");
  }

  if (redctx->op == THREADCOMM_MAXLOC || redctx->op == THREADCOMM_MINLOC) {
    switch(redctx->type) {
    case PETSC_INT:
      ((PetscInt*)redctx->local_red)[trank] = ((PetscInt*)lred)[0];
      ((PetscInt*)redctx->local_red)[redctx->tcomm->nworkThreads+trank] = ((PetscInt*)lred)[1];
      break;
#if defined(PETSC_USE_COMPLEX)
    case PETSC_REAL:
      ((PetscReal*)redctx->local_red)[trank] = ((PetscReal*)lred)[0];
      ((PetscReal*)redctx->local_red)[redctx->tcomm->nworkThreads+trank] = ((PetscReal*)lred)[1];
      break;
#endif
    case PETSC_SCALAR:
      ((PetscScalar*)redctx->local_red)[trank] = ((PetscScalar*)lred)[0];
      ((PetscScalar*)redctx->local_red)[redctx->tcomm->nworkThreads+trank] = ((PetscScalar*)lred)[1];
      break;
    default:
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Unknown datatype provided for kernel reduction");
      break;
    }
  } else {
    switch(redctx->type) {
    case PETSC_INT:
      ((PetscInt*)redctx->local_red)[trank] = *(PetscInt*)lred;
      break;
#if defined(PETSC_USE_COMPLEX)
    case PETSC_REAL:
      ((PetscReal*)redctx->local_red)[trank] = *(PetscReal*)lred;
      break;
#endif
    case PETSC_SCALAR:
      ((PetscScalar*)redctx->local_red)[trank] = *(PetscScalar*)lred;
      break;
    default:
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Unknown datatype provided for kernel reduction");
      break;
    }
  }
  redctx->thread_status[trank] = THREADCOMM_THREAD_POSTED_LOCALRED;
  return 0;
}

/* Completes the given reduction */
#undef __FUNCT__
#define __FUNCT__ "PetscThreadReductionEnd_Private"
PetscErrorCode PetscThreadReductionEnd_Private(PetscThreadCommRedCtx redctx,void * outdata)
{
  /* Check whether all threads have posted their contributions */
  PetscBool wait=PETSC_TRUE;
  PetscInt  i;
  while(wait) {
    for (i=0;i < redctx->tcomm->nworkThreads;i++) { 
      if (PetscReadOnce(int,redctx->thread_status[i]) != THREADCOMM_THREAD_POSTED_LOCALRED) {
	wait = PETSC_TRUE;
	break;
      }
      wait = PETSC_FALSE;
    }
  }

  /* Apply the reduction operation */
  switch(redctx->op) {
  case THREADCOMM_SUM:
    if (redctx->type == PETSC_REAL) {
      PetscReal red_sum=0.0;
      for (i=0; i < redctx->tcomm->nworkThreads;i++) {
	red_sum += ((PetscReal*)redctx->local_red)[i];
      }
      PetscMemcpy(outdata,&red_sum,sizeof(PetscReal));
      break;
    }
    if (redctx->type == PETSC_SCALAR) {
      PetscScalar red_sum=0.0;
      for (i=0; i < redctx->tcomm->nworkThreads;i++) {
	red_sum += ((PetscScalar*)redctx->local_red)[i];
      }
      PetscMemcpy(outdata,&red_sum,sizeof(PetscScalar));
      break;
    }
    if (redctx->type == PETSC_INT) {
      PetscInt red_sum=0;
      for (i=0; i < redctx->tcomm->nworkThreads;i++) {
	red_sum += ((PetscInt*)redctx->local_red)[i];
      }
      PetscMemcpy(outdata,&red_sum,sizeof(PetscInt));
    }
    break;
  case THREADCOMM_PROD:
    if (redctx->type == PETSC_REAL) {
      PetscReal red_prod=0.0;
      for (i=0; i < redctx->tcomm->nworkThreads;i++) {
	red_prod *= ((PetscReal*)redctx->local_red)[i];
      }
      PetscMemcpy(outdata,&red_prod,sizeof(PetscReal));
      break;
    }
    if (redctx->type == PETSC_SCALAR) {
      PetscScalar red_prod=0.0;
      for (i=0; i < redctx->tcomm->nworkThreads;i++) {
	red_prod *= ((PetscScalar*)redctx->local_red)[i];
      }
      PetscMemcpy(outdata,&red_prod,sizeof(PetscScalar));
      break;
    }
    if (redctx->type == PETSC_INT) {
      PetscInt red_prod=0.0;
      for (i=0; i < redctx->tcomm->nworkThreads;i++) {
	red_prod *= ((PetscInt*)redctx->local_red)[i];
      }
      PetscMemcpy(outdata,&red_prod,sizeof(PetscInt));
    }
    break;
  case THREADCOMM_MIN:
#if defined(PETSC_USE_COMPLEX)
    if (redctx->type == PETSC_REAL) {
      PetscReal min = ((PetscReal*)redctx->local_red)[0];
      for (i=1; i < redctx->tcomm->nworkThreads;i++) {
        if (((PetscReal*)redctx->local_red)[i] < min) min = ((PetscReal*)redctx->local_red)[i];
      }
      PetscMemcpy(outdata,&min,sizeof(PetscReal));
      break;
    }
#endif
    if (redctx->type == PETSC_SCALAR) {
      PetscScalar min = ((PetscScalar*)redctx->local_red)[0];
      for (i=1; i < redctx->tcomm->nworkThreads;i++) {
        if (PetscRealPart(((PetscScalar*)redctx->local_red)[i]) < PetscRealPart(min)) min = ((PetscScalar*)redctx->local_red)[i];
      }
      PetscMemcpy(outdata,&min,sizeof(PetscScalar));
      break;
    }
    if (redctx->type == PETSC_INT) {
      PetscInt min = ((PetscInt*)redctx->local_red)[0];
      for (i=1; i < redctx->tcomm->nworkThreads;i++) {
        if (((PetscInt*)redctx->local_red)[i] < min) min = ((PetscInt*)redctx->local_red)[i];
      }
      PetscMemcpy(outdata,&min,sizeof(PetscInt));
    }
    break;
  case THREADCOMM_MAX:
#if defined(PETSC_USE_COMPLEX)
    if (redctx->type == PETSC_REAL) {
      PetscReal max = ((PetscReal*)redctx->local_red)[0];
      for (i=1; i < redctx->tcomm->nworkThreads;i++) {
        if (((PetscReal*)redctx->local_red)[i] > max) max = ((PetscReal*)redctx->local_red)[i];
      }
      PetscMemcpy(outdata,&max,sizeof(PetscReal));
      break;
    }
#endif
    if (redctx->type == PETSC_SCALAR) {
      PetscScalar max = ((PetscScalar*)redctx->local_red)[0];
      for (i=1; i < redctx->tcomm->nworkThreads;i++) {
        if (PetscRealPart(((PetscScalar*)redctx->local_red)[i]) > PetscRealPart(max)) max = ((PetscScalar*)redctx->local_red)[i];
      }
      PetscMemcpy(outdata,&max,sizeof(PetscScalar));
      break;
    }
    if (redctx->type == PETSC_INT) {
      PetscInt max = ((PetscInt*)redctx->local_red)[0];
      for (i=1; i < redctx->tcomm->nworkThreads;i++) {
        if (((PetscInt*)redctx->local_red)[i] > max) max = ((PetscInt*)redctx->local_red)[i];
      }
      PetscMemcpy(outdata,&max,sizeof(PetscInt));
    }
    break;
  case THREADCOMM_MAXLOC:
#if defined(PETSC_USE_COMPLEX)
    if (redctx->type == PETSC_REAL) {
      PetscReal maxloc[2];
      maxloc[0] = ((PetscReal*)redctx->local_red)[0];
      maxloc[1] = ((PetscReal*)redctx->local_red)[redctx->tcomm->nworkThreads];
      for (i=1; i < redctx->tcomm->nworkThreads;i++) {
        if (((PetscReal*)redctx->local_red)[i] > maxloc[0]) { 
          maxloc[0] = ((PetscReal*)redctx->local_red)[i];
          maxloc[1] = ((PetscReal*)redctx->local_red)[redctx->tcomm->nworkThreads+i];
        }
      }
      PetscMemcpy(outdata,maxloc,2*sizeof(PetscReal));
      break;
    }
#endif
    if (redctx->type == PETSC_SCALAR) {
      PetscScalar maxloc[2];
      maxloc[0] = ((PetscScalar*)redctx->local_red)[0];
      maxloc[1] = ((PetscScalar*)redctx->local_red)[redctx->tcomm->nworkThreads];
      for (i=1; i < redctx->tcomm->nworkThreads;i++) {
        if (PetscRealPart(((PetscScalar*)redctx->local_red)[i]) > PetscRealPart(maxloc[0])) { 
          maxloc[0] = ((PetscScalar*)redctx->local_red)[i];
          maxloc[1] = ((PetscScalar*)redctx->local_red)[redctx->tcomm->nworkThreads+i];
        }
      }
      PetscMemcpy(outdata,maxloc,2*sizeof(PetscScalar));
      break;
    }
    if (redctx->type == PETSC_INT) {
      PetscInt maxloc[2];
      maxloc[0] = ((PetscInt*)redctx->local_red)[0];
      maxloc[1] = ((PetscInt*)redctx->local_red)[redctx->tcomm->nworkThreads];
      for (i=1; i < redctx->tcomm->nworkThreads;i++) {
        if (((PetscInt*)redctx->local_red)[i] > maxloc[0]) { 
          maxloc[0] = ((PetscInt*)redctx->local_red)[i];
          maxloc[1] = ((PetscInt*)redctx->local_red)[redctx->tcomm->nworkThreads+i];
        }
      }
      PetscMemcpy(outdata,maxloc,2*sizeof(PetscInt));
    }
    break;
  case THREADCOMM_MINLOC:
#if defined(PETSC_USE_COMPLEX)
    if (redctx->type == PETSC_REAL) {
      PetscReal minloc[2];
      minloc[0] = ((PetscReal*)redctx->local_red)[0];
      minloc[1] = ((PetscReal*)redctx->local_red)[redctx->tcomm->nworkThreads];
      for (i=1; i < redctx->tcomm->nworkThreads;i++) {
        if (((PetscReal*)redctx->local_red)[i] < minloc[0]) { 
          minloc[0] = ((PetscReal*)redctx->local_red)[i];
          minloc[1] = ((PetscReal*)redctx->local_red)[redctx->tcomm->nworkThreads+i];
        }
      }
      PetscMemcpy(outdata,minloc,2*sizeof(PetscReal));
      break;
    }
#endif
    if (redctx->type == PETSC_SCALAR) {
      PetscScalar minloc[2];
      minloc[0] = ((PetscScalar*)redctx->local_red)[0];
      minloc[1] = ((PetscScalar*)redctx->local_red)[redctx->tcomm->nworkThreads];
      for (i=1; i < redctx->tcomm->nworkThreads;i++) {
        if (PetscRealPart(((PetscScalar*)redctx->local_red)[i]) < PetscRealPart(minloc[0])) { 
          minloc[0] = ((PetscScalar*)redctx->local_red)[i];
          minloc[1] = ((PetscScalar*)redctx->local_red)[redctx->tcomm->nworkThreads+i];
        }
      }
      PetscMemcpy(outdata,minloc,2*sizeof(PetscScalar));
      break;
    }
    if (redctx->type == PETSC_INT) {
      PetscInt minloc[2];
      minloc[0] = ((PetscInt*)redctx->local_red)[0];
      minloc[1] = ((PetscInt*)redctx->local_red)[redctx->tcomm->nworkThreads];
      for (i=1; i < redctx->tcomm->nworkThreads;i++) {
        if (((PetscInt*)redctx->local_red)[i] < minloc[0]) { 
          minloc[0] = ((PetscInt*)redctx->local_red)[i];
          minloc[1] = ((PetscInt*)redctx->local_red)[redctx->tcomm->nworkThreads+i];
        }
      }
      PetscMemcpy(outdata,minloc,2*sizeof(PetscInt));
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
PetscErrorCode PetscThreadReductionEnd(PetscThreadCommReduction red,void *outdata)
{
  PetscErrorCode        ierr;
  PetscThreadCommRedCtx redctx;
  PetscInt              i;

  PetscFunctionBegin;
  redctx = red->redctx[red->ctr-red->nreds];
  ierr = PetscThreadReductionEnd_Private(redctx,outdata);CHKERRQ(ierr);
  redctx->red_status = THREADCOMM_REDUCTION_COMPLETE;
  red->nreds--;
  if(!red->nreds) {
    /* Reset the counters */
    red->ctr=0;
    for(i=0;i<redctx->tcomm->nworkThreads;i++) red->thread_ctr[i] = 0;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadReductionKernelEnd"
/*
   PetscThreadReductionKernelEnd - Finishes a reduction operation

   Input Parameters:
+  trank   - Rank of the calling thread
.  red     - the reduction context
-  outdata - the reduction result

   Level: developer

   Notes: This should be called only from kernels only if the reduction needs to
   be completed while in the kernel for some future operation.

*/
PetscErrorCode PetscThreadReductionKernelEnd(PetscInt trank,PetscThreadCommReduction red,void *outdata)
{
  PetscThreadCommRedCtx redctx=red->redctx[red->ctr];

  if (PetscReadOnce(int,redctx->tcomm->leader) == trank) {
    PetscThreadReductionEnd_Private(redctx,outdata);
    redctx->red_status = THREADCOMM_REDUCTION_COMPLETE;
    red->ctr++;
  }

  /* Wait till the leader performs the reduction so that the other threads
     can also see the reduction result */
  while(PetscReadOnce(int,redctx->red_status) != THREADCOMM_REDUCTION_COMPLETE) 
    ;
  redctx->thread_status[trank] = THREADCOMM_THREAD_WAITING_FOR_NEWRED;
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
PetscErrorCode PetscThreadCommReductionCreate(PetscThreadComm tcomm,PetscThreadCommReduction *newred)
{
  PetscErrorCode           ierr;
  PetscThreadCommReduction redout;
  PetscThreadCommRedCtx    redctx;
  PetscInt                 i,j;

  PetscFunctionBegin;
  ierr = PetscNew(struct _p_PetscThreadCommReduction,&redout);CHKERRQ(ierr);
  redout->nreds=0;
  redout->ctr = 0;
  for(i=0;i < PETSC_REDUCTIONS_MAX; i++) {
    ierr = PetscNew(struct _p_PetscThreadCommRedCtx,&redout->redctx[i]);CHKERRQ(ierr);
    redctx = redout->redctx[i];
    ierr = PetscMalloc(tcomm->nworkThreads*sizeof(PetscInt),&redctx->thread_status);CHKERRQ(ierr);
    /* Note that the size of local_red is twice the number of threads. The first half holds the local reductions
     from each thread while the second half is used only for maxloc and minloc operations to hold the local max and min locations
    */
    ierr = PetscMalloc(2*tcomm->nworkThreads*sizeof(PetscScalar),&redctx->local_red);CHKERRQ(ierr);
    redctx->red_status = THREADCOMM_REDUCTION_NONE;
    for(j=0;j<tcomm->nworkThreads;j++) redctx->thread_status[j] = THREADCOMM_THREAD_WAITING_FOR_NEWRED;
  }
  ierr = PetscMalloc(tcomm->nworkThreads*sizeof(PetscScalar),&redout->thread_ctr);CHKERRQ(ierr);
  for(j=0;j<tcomm->nworkThreads;j++) redout->thread_ctr[j] = 0;
  *newred = redout;
  PetscFunctionReturn(0);
}

