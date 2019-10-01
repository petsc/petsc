
/*
      Split phase global vector reductions with support for combining the
   communication portion of several operations. Using MPI-1.1 support only

      The idea for this and much of the initial code is contributed by
   Victor Eijkhout.

       Usage:
             VecDotBegin(Vec,Vec,PetscScalar *);
             VecNormBegin(Vec,NormType,PetscReal *);
             ....
             VecDotEnd(Vec,Vec,PetscScalar *);
             VecNormEnd(Vec,NormType,PetscReal *);

       Limitations:
         - The order of the xxxEnd() functions MUST be in the same order
           as the xxxBegin(). There is extensive error checking to try to
           insure that the user calls the routines in the correct order
*/

#include <petsc/private/vecimpl.h>    /*I   "petscvec.h"    I*/

static PetscErrorCode MPIPetsc_Iallreduce(void *sendbuf,void *recvbuf,PetscMPIInt count,MPI_Datatype datatype,MPI_Op op,MPI_Comm comm,MPI_Request *request)
{
  PETSC_UNUSED PetscErrorCode ierr;

  PetscFunctionBegin;
#if defined(PETSC_HAVE_MPI_IALLREDUCE)
  ierr = MPI_Iallreduce(sendbuf,recvbuf,count,datatype,op,comm,request);CHKERRQ(ierr);
#elif defined(PETSC_HAVE_MPIX_IALLREDUCE)
  ierr = MPIX_Iallreduce(sendbuf,recvbuf,count,datatype,op,comm,request);CHKERRQ(ierr);
#else
  ierr = MPIU_Allreduce(sendbuf,recvbuf,count,datatype,op,comm);CHKERRQ(ierr);
  *request = MPI_REQUEST_NULL;
#endif
  PetscFunctionReturn(0);
}

/*
   Note: the lvalues and gvalues are twice as long as maxops, this is to allow the second half of
the entries to have a flag indicating if they are PETSC_SR_REDUCE_SUM, PETSC_SR_REDUCE_MAX, or PETSC_SR_REDUCE_MIN these are used by
the custom reduction operation that replaces MPI_SUM, MPI_MAX, or MPI_MIN in the case when a reduction involves
some of each.
*/

static PetscErrorCode PetscSplitReductionApply(PetscSplitReduction*);

/*
   PetscSplitReductionCreate - Creates a data structure to contain the queued information.
*/
static PetscErrorCode  PetscSplitReductionCreate(MPI_Comm comm,PetscSplitReduction **sr)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr               = PetscNew(sr);CHKERRQ(ierr);
  (*sr)->numopsbegin = 0;
  (*sr)->numopsend   = 0;
  (*sr)->state       = STATE_BEGIN;
#define MAXOPS 32
  (*sr)->maxops      = MAXOPS;
  ierr               = PetscMalloc4(2*MAXOPS,&(*sr)->lvalues,2*MAXOPS,&(*sr)->gvalues,MAXOPS,&(*sr)->invecs,MAXOPS,&(*sr)->reducetype);CHKERRQ(ierr);
#undef MAXOPS
  (*sr)->comm        = comm;
  (*sr)->request     = MPI_REQUEST_NULL;
  (*sr)->async       = PETSC_FALSE;
#if defined(PETSC_HAVE_MPI_IALLREDUCE) || defined(PETSC_HAVE_MPIX_IALLREDUCE)
  (*sr)->async = PETSC_TRUE;    /* Enable by default */
#endif
  /* always check for option; so that tests that run on systems without support don't warn about unhandled options */
  ierr = PetscOptionsGetBool(NULL,NULL,"-splitreduction_async",&(*sr)->async,NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
       This function is the MPI reduction operation used when there is
   a combination of sums and max in the reduction. The call below to
   MPI_Op_create() converts the function PetscSplitReduction_Local() to the
   MPI operator PetscSplitReduction_Op.
*/
MPI_Op PetscSplitReduction_Op = 0;

PETSC_EXTERN void MPIAPI PetscSplitReduction_Local(void *in,void *out,PetscMPIInt *cnt,MPI_Datatype *datatype)
{
  PetscScalar *xin = (PetscScalar*)in,*xout = (PetscScalar*)out;
  PetscInt    i,count = (PetscInt)*cnt;

  PetscFunctionBegin;
  if (*datatype != MPIU_SCALAR) {
    (*PetscErrorPrintf)("Can only handle MPIU_SCALAR data types");
    PETSCABORT(MPI_COMM_SELF,PETSC_ERR_ARG_WRONG);
  }
  count = count/2;
  for (i=0; i<count; i++) {
     /* second half of xin[] is flags for reduction type */
    if      ((PetscInt)PetscRealPart(xin[count+i]) == PETSC_SR_REDUCE_SUM) xout[i] += xin[i];
    else if ((PetscInt)PetscRealPart(xin[count+i]) == PETSC_SR_REDUCE_MAX) xout[i] = PetscMax(*(PetscReal*)(xout+i),*(PetscReal*)(xin+i));
    else if ((PetscInt)PetscRealPart(xin[count+i]) == PETSC_SR_REDUCE_MIN) xout[i] = PetscMin(*(PetscReal*)(xout+i),*(PetscReal*)(xin+i));
    else {
      (*PetscErrorPrintf)("Reduction type input is not PETSC_SR_REDUCE_SUM, PETSC_SR_REDUCE_MAX, or PETSC_SR_REDUCE_MIN");
      PETSCABORT(MPI_COMM_SELF,PETSC_ERR_ARG_WRONG);
    }
  }
  PetscFunctionReturnVoid();
}

/*@
   PetscCommSplitReductionBegin - Begin an asynchronous split-mode reduction

   Collective but not synchronizing

   Input Arguments:
   comm - communicator on which split reduction has been queued

   Level: advanced

   Note:
   Calling this function is optional when using split-mode reduction. On supporting hardware, calling this after all
   VecXxxBegin() allows the reduction to make asynchronous progress before the result is needed (in VecXxxEnd()).

.seealso: VecNormBegin(), VecNormEnd(), VecDotBegin(), VecDotEnd(), VecTDotBegin(), VecTDotEnd(), VecMDotBegin(), VecMDotEnd(), VecMTDotBegin(), VecMTDotEnd()
@*/
PetscErrorCode PetscCommSplitReductionBegin(MPI_Comm comm)
{
  PetscErrorCode      ierr;
  PetscSplitReduction *sr;

  PetscFunctionBegin;
  ierr = PetscSplitReductionGet(comm,&sr);CHKERRQ(ierr);
  if (sr->numopsend > 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Cannot call this after VecxxxEnd() has been called");
  if (sr->async) {              /* Bad reuse, setup code copied from PetscSplitReductionApply(). */
    PetscInt       i,numops = sr->numopsbegin,*reducetype = sr->reducetype;
    PetscScalar    *lvalues = sr->lvalues,*gvalues = sr->gvalues;
    PetscInt       sum_flg = 0,max_flg = 0, min_flg = 0;
    MPI_Comm       comm = sr->comm;
    PetscMPIInt    size,cmul = sizeof(PetscScalar)/sizeof(PetscReal);;
    ierr = PetscLogEventBegin(VEC_ReduceBegin,0,0,0,0);CHKERRQ(ierr);
    ierr = MPI_Comm_size(sr->comm,&size);CHKERRQ(ierr);
    if (size == 1) {
      ierr = PetscArraycpy(gvalues,lvalues,numops);CHKERRQ(ierr);
    } else {
      /* determine if all reductions are sum, max, or min */
      for (i=0; i<numops; i++) {
        if      (reducetype[i] == PETSC_SR_REDUCE_MAX) max_flg = 1;
        else if (reducetype[i] == PETSC_SR_REDUCE_SUM) sum_flg = 1;
        else if (reducetype[i] == PETSC_SR_REDUCE_MIN) min_flg = 1;
        else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Error in PetscSplitReduction() data structure, probably memory corruption");
      }
      if (sum_flg + max_flg + min_flg > 1) {
        /*
         after all the entires in lvalues we store the reducetype flags to indicate
         to the reduction operations what are sums and what are max
         */
        for (i=0; i<numops; i++) lvalues[numops+i] = reducetype[i];

        ierr = MPIPetsc_Iallreduce(lvalues,gvalues,2*numops,MPIU_SCALAR,PetscSplitReduction_Op,comm,&sr->request);CHKERRQ(ierr);
      } else if (max_flg) {   /* Compute max of real and imag parts separately, presumably only the real part is used */
        ierr = MPIPetsc_Iallreduce((PetscReal*)lvalues,(PetscReal*)gvalues,cmul*numops,MPIU_REAL,MPIU_MAX,comm,&sr->request);CHKERRQ(ierr);
      } else if (min_flg) {
        ierr = MPIPetsc_Iallreduce((PetscReal*)lvalues,(PetscReal*)gvalues,cmul*numops,MPIU_REAL,MPIU_MIN,comm,&sr->request);CHKERRQ(ierr);
      } else {
        ierr = MPIPetsc_Iallreduce(lvalues,gvalues,numops,MPIU_SCALAR,MPIU_SUM,comm,&sr->request);CHKERRQ(ierr);
      }
    }
    sr->state     = STATE_PENDING;
    sr->numopsend = 0;
    ierr = PetscLogEventEnd(VEC_ReduceBegin,0,0,0,0);CHKERRQ(ierr);
  } else {
    ierr = PetscSplitReductionApply(sr);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscSplitReductionEnd(PetscSplitReduction *sr)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  switch (sr->state) {
  case STATE_BEGIN: /* We are doing synchronous communication and this is the first call to VecXxxEnd() so do the communication */
    ierr = PetscSplitReductionApply(sr);CHKERRQ(ierr);
    break;
  case STATE_PENDING:
    /* We are doing asynchronous-mode communication and this is the first VecXxxEnd() so wait for comm to complete */
    ierr = PetscLogEventBegin(VEC_ReduceEnd,0,0,0,0);CHKERRQ(ierr);
    if (sr->request != MPI_REQUEST_NULL) {
      ierr = MPI_Wait(&sr->request,MPI_STATUS_IGNORE);CHKERRQ(ierr);
    }
    sr->state = STATE_END;
    ierr = PetscLogEventEnd(VEC_ReduceEnd,0,0,0,0);CHKERRQ(ierr);
    break;
  default: break;            /* everything is already done */
  }
  PetscFunctionReturn(0);
}

/*
   PetscSplitReductionApply - Actually do the communication required for a split phase reduction
*/
static PetscErrorCode PetscSplitReductionApply(PetscSplitReduction *sr)
{
  PetscErrorCode ierr;
  PetscInt       i,numops = sr->numopsbegin,*reducetype = sr->reducetype;
  PetscScalar    *lvalues = sr->lvalues,*gvalues = sr->gvalues;
  PetscInt       sum_flg  = 0,max_flg = 0, min_flg = 0;
  MPI_Comm       comm     = sr->comm;
  PetscMPIInt    size,cmul = sizeof(PetscScalar)/sizeof(PetscReal);

  PetscFunctionBegin;
  if (sr->numopsend > 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Cannot call this after VecxxxEnd() has been called");
  ierr = PetscLogEventBegin(VEC_ReduceCommunication,0,0,0,0);CHKERRQ(ierr);
  ierr = MPI_Comm_size(sr->comm,&size);CHKERRQ(ierr);
  if (size == 1) {
    ierr = PetscArraycpy(gvalues,lvalues,numops);CHKERRQ(ierr);
  } else {
    /* determine if all reductions are sum, max, or min */
    for (i=0; i<numops; i++) {
      if      (reducetype[i] == PETSC_SR_REDUCE_MAX) max_flg = 1;
      else if (reducetype[i] == PETSC_SR_REDUCE_SUM) sum_flg = 1;
      else if (reducetype[i] == PETSC_SR_REDUCE_MIN) min_flg = 1;
      else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Error in PetscSplitReduction() data structure, probably memory corruption");
    }
    if (sum_flg + max_flg + min_flg > 1) {
      /*
         after all the entires in lvalues we store the reducetype flags to indicate
         to the reduction operations what are sums and what are max
      */
      for (i=0; i<numops; i++) lvalues[numops+i] = reducetype[i];
      ierr = MPIU_Allreduce(lvalues,gvalues,2*numops,MPIU_SCALAR,PetscSplitReduction_Op,comm);CHKERRQ(ierr);
    } else if (max_flg) {     /* Compute max of real and imag parts separately, presumably only the real part is used */
      ierr = MPIU_Allreduce((PetscReal*)lvalues,(PetscReal*)gvalues,cmul*numops,MPIU_REAL,MPIU_MAX,comm);CHKERRQ(ierr);
    } else if (min_flg) {
      ierr = MPIU_Allreduce((PetscReal*)lvalues,(PetscReal*)gvalues,cmul*numops,MPIU_REAL,MPIU_MIN,comm);CHKERRQ(ierr);
    } else {
      ierr = MPIU_Allreduce(lvalues,gvalues,numops,MPIU_SCALAR,MPIU_SUM,comm);CHKERRQ(ierr);
    }
  }
  sr->state     = STATE_END;
  sr->numopsend = 0;
  ierr = PetscLogEventEnd(VEC_ReduceCommunication,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   PetscSplitReductionExtend - Double the amount of space (slots) allocated for a split reduction object.
*/
PetscErrorCode  PetscSplitReductionExtend(PetscSplitReduction *sr)
{
  PetscErrorCode ierr;
  PetscInt       maxops   = sr->maxops,*reducetype = sr->reducetype;
  PetscScalar    *lvalues = sr->lvalues,*gvalues = sr->gvalues;
  void           **invecs = sr->invecs;

  PetscFunctionBegin;
  sr->maxops     = 2*maxops;
  ierr = PetscMalloc4(2*2*maxops,&sr->lvalues,2*2*maxops,&sr->gvalues,2*maxops,&sr->reducetype,2*maxops,&sr->invecs);CHKERRQ(ierr);
  ierr = PetscArraycpy(sr->lvalues,lvalues,maxops);CHKERRQ(ierr);
  ierr = PetscArraycpy(sr->gvalues,gvalues,maxops);CHKERRQ(ierr);
  ierr = PetscArraycpy(sr->reducetype,reducetype,maxops);CHKERRQ(ierr);
  ierr = PetscArraycpy(sr->invecs,invecs,maxops);CHKERRQ(ierr);
  ierr = PetscFree4(lvalues,gvalues,reducetype,invecs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode  PetscSplitReductionDestroy(PetscSplitReduction *sr)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree4(sr->lvalues,sr->gvalues,sr->reducetype,sr->invecs);CHKERRQ(ierr);
  ierr = PetscFree(sr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscMPIInt Petsc_Reduction_keyval = MPI_KEYVAL_INVALID;

/*
   Private routine to delete internal storage when a communicator is freed.
  This is called by MPI, not by users.

  The binding for the first argument changed from MPI 1.0 to 1.1; in 1.0
  it was MPI_Comm *comm.
*/
PETSC_EXTERN int MPIAPI Petsc_DelReduction(MPI_Comm comm,int keyval,void* attr_val,void* extra_state)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscInfo1(0,"Deleting reduction data in an MPI_Comm %ld\n",(long)comm);CHKERRMPI(ierr);
  ierr = PetscSplitReductionDestroy((PetscSplitReduction*)attr_val);CHKERRMPI(ierr);
  PetscFunctionReturn(0);
}

/*
     PetscSplitReductionGet - Gets the split reduction object from a
        PETSc vector, creates if it does not exit.

*/
PetscErrorCode PetscSplitReductionGet(MPI_Comm comm,PetscSplitReduction **sr)
{
  PetscErrorCode ierr;
  PetscMPIInt    flag;

  PetscFunctionBegin;
  if (Petsc_Reduction_keyval == MPI_KEYVAL_INVALID) {
    /*
       The calling sequence of the 2nd argument to this function changed
       between MPI Standard 1.0 and the revisions 1.1 Here we match the
       new standard, if you are using an MPI implementation that uses
       the older version you will get a warning message about the next line;
       it is only a warning message and should do no harm.
    */
    ierr = MPI_Comm_create_keyval(MPI_COMM_NULL_COPY_FN,Petsc_DelReduction,&Petsc_Reduction_keyval,0);CHKERRQ(ierr);
  }
  ierr = MPI_Comm_get_attr(comm,Petsc_Reduction_keyval,(void**)sr,&flag);CHKERRQ(ierr);
  if (!flag) {  /* doesn't exist yet so create it and put it in */
    ierr = PetscSplitReductionCreate(comm,sr);CHKERRQ(ierr);
    ierr = MPI_Comm_set_attr(comm,Petsc_Reduction_keyval,*sr);CHKERRQ(ierr);
    ierr = PetscInfo1(0,"Putting reduction data in an MPI_Comm %ld\n",(long)comm);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* ----------------------------------------------------------------------------------------------------*/

/*@
   VecDotBegin - Starts a split phase dot product computation.

   Input Parameters:
+   x - the first vector
.   y - the second vector
-   result - where the result will go (can be NULL)

   Level: advanced

   Notes:
   Each call to VecDotBegin() should be paired with a call to VecDotEnd().

seealso: VecDotEnd(), VecNormBegin(), VecNormEnd(), VecNorm(), VecDot(), VecMDot(),
         VecTDotBegin(), VecTDotEnd(), PetscCommSplitReductionBegin()
@*/
PetscErrorCode  VecDotBegin(Vec x,Vec y,PetscScalar *result)
{
  PetscErrorCode      ierr;
  PetscSplitReduction *sr;
  MPI_Comm            comm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_CLASSID,1);
  PetscValidHeaderSpecific(y,VEC_CLASSID,1);
  ierr = PetscObjectGetComm((PetscObject)x,&comm);CHKERRQ(ierr);
  ierr = PetscSplitReductionGet(comm,&sr);CHKERRQ(ierr);
  if (sr->state != STATE_BEGIN) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Called before all VecxxxEnd() called");
  if (sr->numopsbegin >= sr->maxops) {
    ierr = PetscSplitReductionExtend(sr);CHKERRQ(ierr);
  }
  sr->reducetype[sr->numopsbegin] = PETSC_SR_REDUCE_SUM;
  sr->invecs[sr->numopsbegin]     = (void*)x;
  if (!x->ops->dot_local) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Vector does not suppport local dots");
  ierr = PetscLogEventBegin(VEC_ReduceArithmetic,0,0,0,0);CHKERRQ(ierr);
  ierr = (*x->ops->dot_local)(x,y,sr->lvalues+sr->numopsbegin++);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(VEC_ReduceArithmetic,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   VecDotEnd - Ends a split phase dot product computation.

   Input Parameters:
+  x - the first vector (can be NULL)
.  y - the second vector (can be NULL)
-  result - where the result will go

   Level: advanced

   Notes:
   Each call to VecDotBegin() should be paired with a call to VecDotEnd().

.seealso: VecDotBegin(), VecNormBegin(), VecNormEnd(), VecNorm(), VecDot(), VecMDot(),
         VecTDotBegin(),VecTDotEnd(), PetscCommSplitReductionBegin()

@*/
PetscErrorCode  VecDotEnd(Vec x,Vec y,PetscScalar *result)
{
  PetscErrorCode      ierr;
  PetscSplitReduction *sr;
  MPI_Comm            comm;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)x,&comm);CHKERRQ(ierr);
  ierr = PetscSplitReductionGet(comm,&sr);CHKERRQ(ierr);
  ierr = PetscSplitReductionEnd(sr);CHKERRQ(ierr);

  if (sr->numopsend >= sr->numopsbegin) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Called VecxxxEnd() more times then VecxxxBegin()");
  if (x && (void*)x != sr->invecs[sr->numopsend]) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Called VecxxxEnd() in a different order or with a different vector than VecxxxBegin()");
  if (sr->reducetype[sr->numopsend] != PETSC_SR_REDUCE_SUM) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Called VecDotEnd() on a reduction started with VecNormBegin()");
  *result = sr->gvalues[sr->numopsend++];

  /*
     We are finished getting all the results so reset to no outstanding requests
  */
  if (sr->numopsend == sr->numopsbegin) {
    sr->state       = STATE_BEGIN;
    sr->numopsend   = 0;
    sr->numopsbegin = 0;
  }
  PetscFunctionReturn(0);
}

/*@
   VecTDotBegin - Starts a split phase transpose dot product computation.

   Input Parameters:
+  x - the first vector
.  y - the second vector
-  result - where the result will go (can be NULL)

   Level: advanced

   Notes:
   Each call to VecTDotBegin() should be paired with a call to VecTDotEnd().

.seealso: VecTDotEnd(), VecNormBegin(), VecNormEnd(), VecNorm(), VecDot(), VecMDot(),
         VecDotBegin(), VecDotEnd(), PetscCommSplitReductionBegin()

@*/
PetscErrorCode  VecTDotBegin(Vec x,Vec y,PetscScalar *result)
{
  PetscErrorCode      ierr;
  PetscSplitReduction *sr;
  MPI_Comm            comm;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)x,&comm);CHKERRQ(ierr);
  ierr = PetscSplitReductionGet(comm,&sr);CHKERRQ(ierr);
  if (sr->state != STATE_BEGIN) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Called before all VecxxxEnd() called");
  if (sr->numopsbegin >= sr->maxops) {
    ierr = PetscSplitReductionExtend(sr);CHKERRQ(ierr);
  }
  sr->reducetype[sr->numopsbegin] = PETSC_SR_REDUCE_SUM;
  sr->invecs[sr->numopsbegin]     = (void*)x;
  if (!x->ops->tdot_local) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Vector does not suppport local dots");
  ierr = PetscLogEventBegin(VEC_ReduceArithmetic,0,0,0,0);CHKERRQ(ierr);
  ierr = (*x->ops->tdot_local)(x,y,sr->lvalues+sr->numopsbegin++);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(VEC_ReduceArithmetic,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   VecTDotEnd - Ends a split phase transpose dot product computation.

   Input Parameters:
+  x - the first vector (can be NULL)
.  y - the second vector (can be NULL)
-  result - where the result will go

   Level: advanced

   Notes:
   Each call to VecTDotBegin() should be paired with a call to VecTDotEnd().

seealso: VecTDotBegin(), VecNormBegin(), VecNormEnd(), VecNorm(), VecDot(), VecMDot(),
         VecDotBegin(), VecDotEnd()
@*/
PetscErrorCode  VecTDotEnd(Vec x,Vec y,PetscScalar *result)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /*
      TDotEnd() is the same as DotEnd() so reuse the code
  */
  ierr = VecDotEnd(x,y,result);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------*/

/*@
   VecNormBegin - Starts a split phase norm computation.

   Input Parameters:
+  x - the first vector
.  ntype - norm type, one of NORM_1, NORM_2, NORM_MAX, NORM_1_AND_2
-  result - where the result will go (can be NULL)

   Level: advanced

   Notes:
   Each call to VecNormBegin() should be paired with a call to VecNormEnd().

.seealso: VecNormEnd(), VecNorm(), VecDot(), VecMDot(), VecDotBegin(), VecDotEnd(), PetscCommSplitReductionBegin()

@*/
PetscErrorCode  VecNormBegin(Vec x,NormType ntype,PetscReal *result)
{
  PetscErrorCode      ierr;
  PetscSplitReduction *sr;
  PetscReal           lresult[2];
  MPI_Comm            comm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_CLASSID,1);
  ierr = PetscObjectGetComm((PetscObject)x,&comm);CHKERRQ(ierr);
  ierr = PetscSplitReductionGet(comm,&sr);CHKERRQ(ierr);
  if (sr->state != STATE_BEGIN) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Called before all VecxxxEnd() called");
  if (sr->numopsbegin >= sr->maxops || (sr->numopsbegin == sr->maxops-1 && ntype == NORM_1_AND_2)) {
    ierr = PetscSplitReductionExtend(sr);CHKERRQ(ierr);
  }

  sr->invecs[sr->numopsbegin] = (void*)x;
  if (!x->ops->norm_local) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Vector does not support local norms");
  ierr = PetscLogEventBegin(VEC_ReduceArithmetic,0,0,0,0);CHKERRQ(ierr);
  ierr = (*x->ops->norm_local)(x,ntype,lresult);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(VEC_ReduceArithmetic,0,0,0,0);CHKERRQ(ierr);
  if (ntype == NORM_2)         lresult[0]                = lresult[0]*lresult[0];
  if (ntype == NORM_1_AND_2)   lresult[1]                = lresult[1]*lresult[1];
  if (ntype == NORM_MAX) sr->reducetype[sr->numopsbegin] = PETSC_SR_REDUCE_MAX;
  else                   sr->reducetype[sr->numopsbegin] = PETSC_SR_REDUCE_SUM;
  sr->lvalues[sr->numopsbegin++] = lresult[0];
  if (ntype == NORM_1_AND_2) {
    sr->reducetype[sr->numopsbegin] = PETSC_SR_REDUCE_SUM;
    sr->lvalues[sr->numopsbegin++]  = lresult[1];
  }
  PetscFunctionReturn(0);
}

/*@
   VecNormEnd - Ends a split phase norm computation.

   Input Parameters:
+  x - the first vector 
.  ntype - norm type, one of NORM_1, NORM_2, NORM_MAX, NORM_1_AND_2
-  result - where the result will go

   Level: advanced

   Notes:
   Each call to VecNormBegin() should be paired with a call to VecNormEnd().

   The x vector is not allowed to be NULL, otherwise the vector would not have its correctly cached norm value

.seealso: VecNormBegin(), VecNorm(), VecDot(), VecMDot(), VecDotBegin(), VecDotEnd(), PetscCommSplitReductionBegin()

@*/
PetscErrorCode  VecNormEnd(Vec x,NormType ntype,PetscReal *result)
{
  PetscErrorCode      ierr;
  PetscSplitReduction *sr;
  MPI_Comm            comm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_CLASSID,1);  
  ierr = PetscObjectGetComm((PetscObject)x,&comm);CHKERRQ(ierr);
  ierr = PetscSplitReductionGet(comm,&sr);CHKERRQ(ierr);
  ierr = PetscSplitReductionEnd(sr);CHKERRQ(ierr);

  if (sr->numopsend >= sr->numopsbegin) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Called VecxxxEnd() more times then VecxxxBegin()");
  if ((void*)x != sr->invecs[sr->numopsend]) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Called VecxxxEnd() in a different order or with a different vector than VecxxxBegin()");
  if (sr->reducetype[sr->numopsend] != PETSC_SR_REDUCE_MAX && ntype == NORM_MAX) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Called VecNormEnd(,NORM_MAX,) on a reduction started with VecDotBegin() or NORM_1 or NORM_2");
  result[0] = PetscRealPart(sr->gvalues[sr->numopsend++]);

  if (ntype == NORM_2) result[0] = PetscSqrtReal(result[0]);
  else if (ntype == NORM_1_AND_2) {
    result[1] = PetscRealPart(sr->gvalues[sr->numopsend++]);
    result[1] = PetscSqrtReal(result[1]);
  }
  if (ntype!=NORM_1_AND_2) {
    ierr = PetscObjectComposedDataSetReal((PetscObject)x,NormIds[ntype],result[0]);CHKERRQ(ierr);
  }

  if (sr->numopsend == sr->numopsbegin) {
    sr->state       = STATE_BEGIN;
    sr->numopsend   = 0;
    sr->numopsbegin = 0;
  }
  PetscFunctionReturn(0);
}

/*
   Possibly add

     PetscReductionSumBegin/End()
     PetscReductionMaxBegin/End()
     PetscReductionMinBegin/End()
   or have more like MPI with a single function with flag for Op? Like first better
*/

/*@
   VecMDotBegin - Starts a split phase multiple dot product computation.

   Input Parameters:
+   x - the first vector
.   nv - number of vectors
.   y - array of vectors
-   result - where the result will go (can be NULL)

   Level: advanced

   Notes:
   Each call to VecMDotBegin() should be paired with a call to VecMDotEnd().

.seealso: VecMDotEnd(), VecNormBegin(), VecNormEnd(), VecNorm(), VecDot(), VecMDot(),
         VecTDotBegin(), VecTDotEnd(), VecMTDotBegin(), VecMTDotEnd(), PetscCommSplitReductionBegin()
@*/
PetscErrorCode  VecMDotBegin(Vec x,PetscInt nv,const Vec y[],PetscScalar result[])
{
  PetscErrorCode      ierr;
  PetscSplitReduction *sr;
  MPI_Comm            comm;
  int                 i;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)x,&comm);CHKERRQ(ierr);
  ierr = PetscSplitReductionGet(comm,&sr);CHKERRQ(ierr);
  if (sr->state != STATE_BEGIN) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Called before all VecxxxEnd() called");
  for (i=0; i<nv; i++) {
    if (sr->numopsbegin+i >= sr->maxops) {
      ierr = PetscSplitReductionExtend(sr);CHKERRQ(ierr);
    }
    sr->reducetype[sr->numopsbegin+i] = PETSC_SR_REDUCE_SUM;
    sr->invecs[sr->numopsbegin+i]     = (void*)x;
  }
  if (!x->ops->mdot_local) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Vector does not suppport local mdots");
  ierr = PetscLogEventBegin(VEC_ReduceArithmetic,0,0,0,0);CHKERRQ(ierr);
  ierr = (*x->ops->mdot_local)(x,nv,y,sr->lvalues+sr->numopsbegin);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(VEC_ReduceArithmetic,0,0,0,0);CHKERRQ(ierr);
  sr->numopsbegin += nv;
  PetscFunctionReturn(0);
}

/*@
   VecMDotEnd - Ends a split phase multiple dot product computation.

   Input Parameters:
+   x - the first vector (can be NULL)
.   nv - number of vectors
-   y - array of vectors (can be NULL)

   Output Parameters:
.   result - where the result will go

   Level: advanced

   Notes:
   Each call to VecMDotBegin() should be paired with a call to VecMDotEnd().

.seealso: VecMDotBegin(), VecNormBegin(), VecNormEnd(), VecNorm(), VecDot(), VecMDot(),
         VecTDotBegin(),VecTDotEnd(), VecMTDotBegin(), VecMTDotEnd(), PetscCommSplitReductionBegin()

@*/
PetscErrorCode  VecMDotEnd(Vec x,PetscInt nv,const Vec y[],PetscScalar result[])
{
  PetscErrorCode      ierr;
  PetscSplitReduction *sr;
  MPI_Comm            comm;
  int                 i;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)x,&comm);CHKERRQ(ierr);
  ierr = PetscSplitReductionGet(comm,&sr);CHKERRQ(ierr);
  ierr = PetscSplitReductionEnd(sr);CHKERRQ(ierr);

  if (sr->numopsend >= sr->numopsbegin) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Called VecxxxEnd() more times then VecxxxBegin()");
  if (x && (void*)x != sr->invecs[sr->numopsend]) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Called VecxxxEnd() in a different order or with a different vector than VecxxxBegin()");
  if (sr->reducetype[sr->numopsend] != PETSC_SR_REDUCE_SUM) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Called VecDotEnd() on a reduction started with VecNormBegin()");
  for (i=0;i<nv;i++) result[i] = sr->gvalues[sr->numopsend++];

  /*
     We are finished getting all the results so reset to no outstanding requests
  */
  if (sr->numopsend == sr->numopsbegin) {
    sr->state       = STATE_BEGIN;
    sr->numopsend   = 0;
    sr->numopsbegin = 0;
  }
  PetscFunctionReturn(0);
}

/*@
   VecMTDotBegin - Starts a split phase transpose multiple dot product computation.

   Input Parameters:
+  x - the first vector
.  nv - number of vectors
.  y - array of  vectors
-  result - where the result will go (can be NULL)

   Level: advanced

   Notes:
   Each call to VecMTDotBegin() should be paired with a call to VecMTDotEnd().

.seealso: VecMTDotEnd(), VecNormBegin(), VecNormEnd(), VecNorm(), VecDot(), VecMDot(),
         VecDotBegin(), VecDotEnd(), VecMDotBegin(), VecMDotEnd(), PetscCommSplitReductionBegin()

@*/
PetscErrorCode  VecMTDotBegin(Vec x,PetscInt nv,const Vec y[],PetscScalar result[])
{
  PetscErrorCode      ierr;
  PetscSplitReduction *sr;
  MPI_Comm            comm;
  int                 i;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)x,&comm);CHKERRQ(ierr);
  ierr = PetscSplitReductionGet(comm,&sr);CHKERRQ(ierr);
  if (sr->state != STATE_BEGIN) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Called before all VecxxxEnd() called");
  for (i=0; i<nv; i++) {
    if (sr->numopsbegin+i >= sr->maxops) {
      ierr = PetscSplitReductionExtend(sr);CHKERRQ(ierr);
    }
    sr->reducetype[sr->numopsbegin+i] = PETSC_SR_REDUCE_SUM;
    sr->invecs[sr->numopsbegin+i]     = (void*)x;
  }
  if (!x->ops->mtdot_local) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Vector does not suppport local mdots");
  ierr = PetscLogEventBegin(VEC_ReduceArithmetic,0,0,0,0);CHKERRQ(ierr);
  ierr = (*x->ops->mdot_local)(x,nv,y,sr->lvalues+sr->numopsbegin);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(VEC_ReduceArithmetic,0,0,0,0);CHKERRQ(ierr);
  sr->numopsbegin += nv;
  PetscFunctionReturn(0);
}

/*@
   VecMTDotEnd - Ends a split phase transpose multiple dot product computation.

   Input Parameters:
+  x - the first vector (can be NULL)
.  nv - number of vectors
-  y - array of  vectors (can be NULL)

   Output Parameters
.  result - where the result will go

   Level: advanced

   Notes:
   Each call to VecTDotBegin() should be paired with a call to VecTDotEnd().

.seealso: VecMTDotBegin(), VecNormBegin(), VecNormEnd(), VecNorm(), VecDot(), VecMDot(),
         VecDotBegin(), VecDotEnd(), VecMDotBegin(), VecMDotEnd(), PetscCommSplitReductionBegin()
@*/
PetscErrorCode  VecMTDotEnd(Vec x,PetscInt nv,const Vec y[],PetscScalar result[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /*
      MTDotEnd() is the same as MDotEnd() so reuse the code
  */
  ierr = VecMDotEnd(x,nv,y,result);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
