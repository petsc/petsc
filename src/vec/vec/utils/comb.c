
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
#if defined(PETSC_HAVE_MPI_NONBLOCKING_COLLECTIVES)
  CHKERRMPI(MPI_Iallreduce(sendbuf,recvbuf,count,datatype,op,comm,request));
#else
  CHKERRMPI(MPIU_Allreduce(sendbuf,recvbuf,count,datatype,op,comm));
  *request = MPI_REQUEST_NULL;
#endif
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSplitReductionApply(PetscSplitReduction*);

/*
   PetscSplitReductionCreate - Creates a data structure to contain the queued information.
*/
static PetscErrorCode  PetscSplitReductionCreate(MPI_Comm comm,PetscSplitReduction **sr)
{
  PetscFunctionBegin;
  CHKERRQ(PetscNew(sr));
  (*sr)->numopsbegin = 0;
  (*sr)->numopsend   = 0;
  (*sr)->state       = STATE_BEGIN;
#define MAXOPS 32
  (*sr)->maxops      = MAXOPS;
  CHKERRQ(PetscMalloc6(MAXOPS,&(*sr)->lvalues,MAXOPS,&(*sr)->gvalues,MAXOPS,&(*sr)->invecs,MAXOPS,&(*sr)->reducetype,MAXOPS,&(*sr)->lvalues_mix,MAXOPS,&(*sr)->gvalues_mix));
#undef MAXOPS
  (*sr)->comm        = comm;
  (*sr)->request     = MPI_REQUEST_NULL;
  (*sr)->mix         = PETSC_FALSE;
  (*sr)->async       = PETSC_FALSE;
#if defined(PETSC_HAVE_MPI_NONBLOCKING_COLLECTIVES)
  (*sr)->async = PETSC_TRUE;    /* Enable by default */
#endif
  /* always check for option; so that tests that run on systems without support don't warn about unhandled options */
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-splitreduction_async",&(*sr)->async,NULL));
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
  struct PetscScalarInt { PetscScalar v; PetscInt i; };
  struct PetscScalarInt *xin = (struct PetscScalarInt*)in;
  struct PetscScalarInt *xout = (struct PetscScalarInt*)out;
  PetscInt              i,count = (PetscInt)*cnt;

  PetscFunctionBegin;
  if (*datatype != MPIU_SCALAR_INT) {
    (*PetscErrorPrintf)("Can only handle MPIU_SCALAR_INT data types");
    PETSCABORT(MPI_COMM_SELF,PETSC_ERR_ARG_WRONG);
  }
  for (i=0; i<count; i++) {
    if      (xin[i].i == PETSC_SR_REDUCE_SUM) xout[i].v += xin[i].v;
    else if (xin[i].i == PETSC_SR_REDUCE_MAX) xout[i].v = PetscMax(PetscRealPart(xout[i].v),PetscRealPart(xin[i].v));
    else if (xin[i].i == PETSC_SR_REDUCE_MIN) xout[i].v = PetscMin(PetscRealPart(xout[i].v),PetscRealPart(xin[i].v));
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

   Input Parameter:
   comm - communicator on which split reduction has been queued

   Level: advanced

   Note:
   Calling this function is optional when using split-mode reduction. On supporting hardware, calling this after all
   VecXxxBegin() allows the reduction to make asynchronous progress before the result is needed (in VecXxxEnd()).

.seealso: VecNormBegin(), VecNormEnd(), VecDotBegin(), VecDotEnd(), VecTDotBegin(), VecTDotEnd(), VecMDotBegin(), VecMDotEnd(), VecMTDotBegin(), VecMTDotEnd()
@*/
PetscErrorCode PetscCommSplitReductionBegin(MPI_Comm comm)
{
  PetscSplitReduction *sr;

  PetscFunctionBegin;
  CHKERRQ(PetscSplitReductionGet(comm,&sr));
  PetscCheckFalse(sr->numopsend > 0,PETSC_COMM_SELF,PETSC_ERR_ORDER,"Cannot call this after VecxxxEnd() has been called");
  if (sr->async) {              /* Bad reuse, setup code copied from PetscSplitReductionApply(). */
    PetscInt    i,numops = sr->numopsbegin,*reducetype = sr->reducetype;
    PetscScalar *lvalues = sr->lvalues,*gvalues = sr->gvalues;
    PetscInt    sum_flg = 0,max_flg = 0, min_flg = 0;
    MPI_Comm    comm = sr->comm;
    PetscMPIInt size,cmul = sizeof(PetscScalar)/sizeof(PetscReal);

    CHKERRQ(PetscLogEventBegin(VEC_ReduceBegin,0,0,0,0));
    CHKERRMPI(MPI_Comm_size(sr->comm,&size));
    if (size == 1) {
      CHKERRQ(PetscArraycpy(gvalues,lvalues,numops));
    } else {
      /* determine if all reductions are sum, max, or min */
      for (i=0; i<numops; i++) {
        if      (reducetype[i] == PETSC_SR_REDUCE_MAX) max_flg = 1;
        else if (reducetype[i] == PETSC_SR_REDUCE_SUM) sum_flg = 1;
        else if (reducetype[i] == PETSC_SR_REDUCE_MIN) min_flg = 1;
        else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Error in PetscSplitReduction() data structure, probably memory corruption");
      }
      PetscCheckFalse(sum_flg + max_flg + min_flg > 1 && sr->mix,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Error in PetscSplitReduction() data structure, probably memory corruption");
      if (sum_flg + max_flg + min_flg > 1) {
        sr->mix = PETSC_TRUE;
        for (i=0; i<numops; i++) { sr->lvalues_mix[i].v = lvalues[i]; sr->lvalues_mix[i].i = reducetype[i]; }
        CHKERRQ(MPIPetsc_Iallreduce(sr->lvalues_mix,sr->gvalues_mix,numops,MPIU_SCALAR_INT,PetscSplitReduction_Op,comm,&sr->request));
      } else if (max_flg) {   /* Compute max of real and imag parts separately, presumably only the real part is used */
        CHKERRQ(MPIPetsc_Iallreduce((PetscReal*)lvalues,(PetscReal*)gvalues,cmul*numops,MPIU_REAL,MPIU_MAX,comm,&sr->request));
      } else if (min_flg) {
        CHKERRQ(MPIPetsc_Iallreduce((PetscReal*)lvalues,(PetscReal*)gvalues,cmul*numops,MPIU_REAL,MPIU_MIN,comm,&sr->request));
      } else {
        CHKERRQ(MPIPetsc_Iallreduce(lvalues,gvalues,numops,MPIU_SCALAR,MPIU_SUM,comm,&sr->request));
      }
    }
    sr->state     = STATE_PENDING;
    sr->numopsend = 0;
    CHKERRQ(PetscLogEventEnd(VEC_ReduceBegin,0,0,0,0));
  } else {
    CHKERRQ(PetscSplitReductionApply(sr));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscSplitReductionEnd(PetscSplitReduction *sr)
{
  PetscFunctionBegin;
  switch (sr->state) {
  case STATE_BEGIN: /* We are doing synchronous communication and this is the first call to VecXxxEnd() so do the communication */
    CHKERRQ(PetscSplitReductionApply(sr));
    break;
  case STATE_PENDING:
    /* We are doing asynchronous-mode communication and this is the first VecXxxEnd() so wait for comm to complete */
    CHKERRQ(PetscLogEventBegin(VEC_ReduceEnd,0,0,0,0));
    if (sr->request != MPI_REQUEST_NULL) {
      CHKERRMPI(MPI_Wait(&sr->request,MPI_STATUS_IGNORE));
    }
    sr->state = STATE_END;
    if (sr->mix) {
      PetscInt i;
      for (i=0; i<sr->numopsbegin; i++) { sr->gvalues[i] = sr->gvalues_mix[i].v; }
      sr->mix = PETSC_FALSE;
    }
    CHKERRQ(PetscLogEventEnd(VEC_ReduceEnd,0,0,0,0));
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
  PetscInt       i,numops = sr->numopsbegin,*reducetype = sr->reducetype;
  PetscScalar    *lvalues = sr->lvalues,*gvalues = sr->gvalues;
  PetscInt       sum_flg  = 0,max_flg = 0, min_flg = 0;
  MPI_Comm       comm     = sr->comm;
  PetscMPIInt    size,cmul = sizeof(PetscScalar)/sizeof(PetscReal);

  PetscFunctionBegin;
  PetscCheckFalse(sr->numopsend > 0,PETSC_COMM_SELF,PETSC_ERR_ORDER,"Cannot call this after VecxxxEnd() has been called");
  CHKERRQ(PetscLogEventBegin(VEC_ReduceCommunication,0,0,0,0));
  CHKERRMPI(MPI_Comm_size(sr->comm,&size));
  if (size == 1) {
    CHKERRQ(PetscArraycpy(gvalues,lvalues,numops));
  } else {
    /* determine if all reductions are sum, max, or min */
    for (i=0; i<numops; i++) {
      if      (reducetype[i] == PETSC_SR_REDUCE_MAX) max_flg = 1;
      else if (reducetype[i] == PETSC_SR_REDUCE_SUM) sum_flg = 1;
      else if (reducetype[i] == PETSC_SR_REDUCE_MIN) min_flg = 1;
      else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Error in PetscSplitReduction() data structure, probably memory corruption");
    }
    if (sum_flg + max_flg + min_flg > 1) {
      PetscCheckFalse(sr->mix,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Error in PetscSplitReduction() data structure, probably memory corruption");
      for (i=0; i<numops; i++) { sr->lvalues_mix[i].v = lvalues[i]; sr->lvalues_mix[i].i = reducetype[i]; }
      CHKERRMPI(MPIU_Allreduce(sr->lvalues_mix,sr->gvalues_mix,numops,MPIU_SCALAR_INT,PetscSplitReduction_Op,comm));
      for (i=0; i<numops; i++) { sr->gvalues[i] = sr->gvalues_mix[i].v; }
    } else if (max_flg) {     /* Compute max of real and imag parts separately, presumably only the real part is used */
      CHKERRMPI(MPIU_Allreduce((PetscReal*)lvalues,(PetscReal*)gvalues,cmul*numops,MPIU_REAL,MPIU_MAX,comm));
    } else if (min_flg) {
      CHKERRMPI(MPIU_Allreduce((PetscReal*)lvalues,(PetscReal*)gvalues,cmul*numops,MPIU_REAL,MPIU_MIN,comm));
    } else {
      CHKERRMPI(MPIU_Allreduce(lvalues,gvalues,numops,MPIU_SCALAR,MPIU_SUM,comm));
    }
  }
  sr->state     = STATE_END;
  sr->numopsend = 0;
  CHKERRQ(PetscLogEventEnd(VEC_ReduceCommunication,0,0,0,0));
  PetscFunctionReturn(0);
}

/*
   PetscSplitReductionExtend - Double the amount of space (slots) allocated for a split reduction object.
*/
PetscErrorCode  PetscSplitReductionExtend(PetscSplitReduction *sr)
{
  struct PetscScalarInt { PetscScalar v; PetscInt i; };
  PetscInt              maxops   = sr->maxops,*reducetype = sr->reducetype;
  PetscScalar           *lvalues = sr->lvalues,*gvalues = sr->gvalues;
  struct PetscScalarInt *lvalues_mix = (struct PetscScalarInt*)sr->lvalues_mix;
  struct PetscScalarInt *gvalues_mix = (struct PetscScalarInt*)sr->gvalues_mix;
  void                  **invecs = sr->invecs;

  PetscFunctionBegin;
  sr->maxops = 2*maxops;
  CHKERRQ(PetscMalloc6(2*maxops,&sr->lvalues,2*maxops,&sr->gvalues,2*maxops,&sr->reducetype,2*maxops,&sr->invecs,2*maxops,&sr->lvalues_mix,2*maxops,&sr->gvalues_mix));
  CHKERRQ(PetscArraycpy(sr->lvalues,lvalues,maxops));
  CHKERRQ(PetscArraycpy(sr->gvalues,gvalues,maxops));
  CHKERRQ(PetscArraycpy(sr->reducetype,reducetype,maxops));
  CHKERRQ(PetscArraycpy(sr->invecs,invecs,maxops));
  CHKERRQ(PetscArraycpy(sr->lvalues_mix,lvalues_mix,maxops));
  CHKERRQ(PetscArraycpy(sr->gvalues_mix,gvalues_mix,maxops));
  CHKERRQ(PetscFree6(lvalues,gvalues,reducetype,invecs,lvalues_mix,gvalues_mix));
  PetscFunctionReturn(0);
}

PetscErrorCode  PetscSplitReductionDestroy(PetscSplitReduction *sr)
{
  PetscFunctionBegin;
  CHKERRQ(PetscFree6(sr->lvalues,sr->gvalues,sr->reducetype,sr->invecs,sr->lvalues_mix,sr->gvalues_mix));
  CHKERRQ(PetscFree(sr));
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
  PetscFunctionBegin;
  CHKERRMPI(PetscInfo(0,"Deleting reduction data in an MPI_Comm %ld\n",(long)comm));
  CHKERRMPI(PetscSplitReductionDestroy((PetscSplitReduction*)attr_val));
  PetscFunctionReturn(0);
}

/*
     PetscSplitReductionGet - Gets the split reduction object from a
        PETSc vector, creates if it does not exit.

*/
PetscErrorCode PetscSplitReductionGet(MPI_Comm comm,PetscSplitReduction **sr)
{
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
    CHKERRMPI(MPI_Comm_create_keyval(MPI_COMM_NULL_COPY_FN,Petsc_DelReduction,&Petsc_Reduction_keyval,NULL));
  }
  CHKERRMPI(MPI_Comm_get_attr(comm,Petsc_Reduction_keyval,(void**)sr,&flag));
  if (!flag) {  /* doesn't exist yet so create it and put it in */
    CHKERRQ(PetscSplitReductionCreate(comm,sr));
    CHKERRMPI(MPI_Comm_set_attr(comm,Petsc_Reduction_keyval,*sr));
    CHKERRQ(PetscInfo(0,"Putting reduction data in an MPI_Comm %ld\n",(long)comm));
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
  PetscSplitReduction *sr;
  MPI_Comm            comm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_CLASSID,1);
  PetscValidHeaderSpecific(y,VEC_CLASSID,2);
  CHKERRQ(PetscObjectGetComm((PetscObject)x,&comm));
  CHKERRQ(PetscSplitReductionGet(comm,&sr));
  PetscCheckFalse(sr->state != STATE_BEGIN,PETSC_COMM_SELF,PETSC_ERR_ORDER,"Called before all VecxxxEnd() called");
  if (sr->numopsbegin >= sr->maxops) {
    CHKERRQ(PetscSplitReductionExtend(sr));
  }
  sr->reducetype[sr->numopsbegin] = PETSC_SR_REDUCE_SUM;
  sr->invecs[sr->numopsbegin]     = (void*)x;
  PetscCheckFalse(!x->ops->dot_local,PETSC_COMM_SELF,PETSC_ERR_SUP,"Vector does not support local dots");
  CHKERRQ(PetscLogEventBegin(VEC_ReduceArithmetic,0,0,0,0));
  CHKERRQ((*x->ops->dot_local)(x,y,sr->lvalues+sr->numopsbegin++));
  CHKERRQ(PetscLogEventEnd(VEC_ReduceArithmetic,0,0,0,0));
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
  PetscSplitReduction *sr;
  MPI_Comm            comm;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject)x,&comm));
  CHKERRQ(PetscSplitReductionGet(comm,&sr));
  CHKERRQ(PetscSplitReductionEnd(sr));

  PetscCheckFalse(sr->numopsend >= sr->numopsbegin,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Called VecxxxEnd() more times then VecxxxBegin()");
  PetscCheckFalse(x && (void*)x != sr->invecs[sr->numopsend],PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Called VecxxxEnd() in a different order or with a different vector than VecxxxBegin()");
  PetscCheckFalse(sr->reducetype[sr->numopsend] != PETSC_SR_REDUCE_SUM,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Called VecDotEnd() on a reduction started with VecNormBegin()");
  *result = sr->gvalues[sr->numopsend++];

  /*
     We are finished getting all the results so reset to no outstanding requests
  */
  if (sr->numopsend == sr->numopsbegin) {
    sr->state       = STATE_BEGIN;
    sr->numopsend   = 0;
    sr->numopsbegin = 0;
    sr->mix         = PETSC_FALSE;
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
  PetscSplitReduction *sr;
  MPI_Comm            comm;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject)x,&comm));
  CHKERRQ(PetscSplitReductionGet(comm,&sr));
  PetscCheckFalse(sr->state != STATE_BEGIN,PETSC_COMM_SELF,PETSC_ERR_ORDER,"Called before all VecxxxEnd() called");
  if (sr->numopsbegin >= sr->maxops) {
    CHKERRQ(PetscSplitReductionExtend(sr));
  }
  sr->reducetype[sr->numopsbegin] = PETSC_SR_REDUCE_SUM;
  sr->invecs[sr->numopsbegin]     = (void*)x;
  PetscCheckFalse(!x->ops->tdot_local,PETSC_COMM_SELF,PETSC_ERR_SUP,"Vector does not support local dots");
  CHKERRQ(PetscLogEventBegin(VEC_ReduceArithmetic,0,0,0,0));
  CHKERRQ((*x->ops->tdot_local)(x,y,sr->lvalues+sr->numopsbegin++));
  CHKERRQ(PetscLogEventEnd(VEC_ReduceArithmetic,0,0,0,0));
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
  PetscFunctionBegin;
  /*
      TDotEnd() is the same as DotEnd() so reuse the code
  */
  CHKERRQ(VecDotEnd(x,y,result));
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
  PetscSplitReduction *sr;
  PetscReal           lresult[2];
  MPI_Comm            comm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_CLASSID,1);
  CHKERRQ(PetscObjectGetComm((PetscObject)x,&comm));
  CHKERRQ(PetscSplitReductionGet(comm,&sr));
  PetscCheckFalse(sr->state != STATE_BEGIN,PETSC_COMM_SELF,PETSC_ERR_ORDER,"Called before all VecxxxEnd() called");
  if (sr->numopsbegin >= sr->maxops || (sr->numopsbegin == sr->maxops-1 && ntype == NORM_1_AND_2)) {
    CHKERRQ(PetscSplitReductionExtend(sr));
  }

  sr->invecs[sr->numopsbegin] = (void*)x;
  PetscCheckFalse(!x->ops->norm_local,PETSC_COMM_SELF,PETSC_ERR_SUP,"Vector does not support local norms");
  CHKERRQ(PetscLogEventBegin(VEC_ReduceArithmetic,0,0,0,0));
  CHKERRQ((*x->ops->norm_local)(x,ntype,lresult));
  CHKERRQ(PetscLogEventEnd(VEC_ReduceArithmetic,0,0,0,0));
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
  PetscSplitReduction *sr;
  MPI_Comm            comm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_CLASSID,1);
  CHKERRQ(PetscObjectGetComm((PetscObject)x,&comm));
  CHKERRQ(PetscSplitReductionGet(comm,&sr));
  CHKERRQ(PetscSplitReductionEnd(sr));

  PetscCheckFalse(sr->numopsend >= sr->numopsbegin,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Called VecxxxEnd() more times then VecxxxBegin()");
  PetscCheckFalse((void*)x != sr->invecs[sr->numopsend],PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Called VecxxxEnd() in a different order or with a different vector than VecxxxBegin()");
  PetscCheckFalse(sr->reducetype[sr->numopsend] != PETSC_SR_REDUCE_MAX && ntype == NORM_MAX,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Called VecNormEnd(,NORM_MAX,) on a reduction started with VecDotBegin() or NORM_1 or NORM_2");
  result[0] = PetscRealPart(sr->gvalues[sr->numopsend++]);

  if (ntype == NORM_2) result[0] = PetscSqrtReal(result[0]);
  else if (ntype == NORM_1_AND_2) {
    result[1] = PetscRealPart(sr->gvalues[sr->numopsend++]);
    result[1] = PetscSqrtReal(result[1]);
  }
  if (ntype!=NORM_1_AND_2) {
    CHKERRQ(PetscObjectComposedDataSetReal((PetscObject)x,NormIds[ntype],result[0]));
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
  PetscSplitReduction *sr;
  MPI_Comm            comm;
  int                 i;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject)x,&comm));
  CHKERRQ(PetscSplitReductionGet(comm,&sr));
  PetscCheckFalse(sr->state != STATE_BEGIN,PETSC_COMM_SELF,PETSC_ERR_ORDER,"Called before all VecxxxEnd() called");
  for (i=0; i<nv; i++) {
    if (sr->numopsbegin+i >= sr->maxops) {
      CHKERRQ(PetscSplitReductionExtend(sr));
    }
    sr->reducetype[sr->numopsbegin+i] = PETSC_SR_REDUCE_SUM;
    sr->invecs[sr->numopsbegin+i]     = (void*)x;
  }
  PetscCheckFalse(!x->ops->mdot_local,PETSC_COMM_SELF,PETSC_ERR_SUP,"Vector does not support local mdots");
  CHKERRQ(PetscLogEventBegin(VEC_ReduceArithmetic,0,0,0,0));
  CHKERRQ((*x->ops->mdot_local)(x,nv,y,sr->lvalues+sr->numopsbegin));
  CHKERRQ(PetscLogEventEnd(VEC_ReduceArithmetic,0,0,0,0));
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
  PetscSplitReduction *sr;
  MPI_Comm            comm;
  int                 i;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject)x,&comm));
  CHKERRQ(PetscSplitReductionGet(comm,&sr));
  CHKERRQ(PetscSplitReductionEnd(sr));

  PetscCheckFalse(sr->numopsend >= sr->numopsbegin,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Called VecxxxEnd() more times then VecxxxBegin()");
  PetscCheckFalse(x && (void*)x != sr->invecs[sr->numopsend],PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Called VecxxxEnd() in a different order or with a different vector than VecxxxBegin()");
  PetscCheckFalse(sr->reducetype[sr->numopsend] != PETSC_SR_REDUCE_SUM,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Called VecDotEnd() on a reduction started with VecNormBegin()");
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
  PetscSplitReduction *sr;
  MPI_Comm            comm;
  int                 i;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject)x,&comm));
  CHKERRQ(PetscSplitReductionGet(comm,&sr));
  PetscCheckFalse(sr->state != STATE_BEGIN,PETSC_COMM_SELF,PETSC_ERR_ORDER,"Called before all VecxxxEnd() called");
  for (i=0; i<nv; i++) {
    if (sr->numopsbegin+i >= sr->maxops) {
      CHKERRQ(PetscSplitReductionExtend(sr));
    }
    sr->reducetype[sr->numopsbegin+i] = PETSC_SR_REDUCE_SUM;
    sr->invecs[sr->numopsbegin+i]     = (void*)x;
  }
  PetscCheckFalse(!x->ops->mtdot_local,PETSC_COMM_SELF,PETSC_ERR_SUP,"Vector does not support local mdots");
  CHKERRQ(PetscLogEventBegin(VEC_ReduceArithmetic,0,0,0,0));
  CHKERRQ((*x->ops->mdot_local)(x,nv,y,sr->lvalues+sr->numopsbegin));
  CHKERRQ(PetscLogEventEnd(VEC_ReduceArithmetic,0,0,0,0));
  sr->numopsbegin += nv;
  PetscFunctionReturn(0);
}

/*@
   VecMTDotEnd - Ends a split phase transpose multiple dot product computation.

   Input Parameters:
+  x - the first vector (can be NULL)
.  nv - number of vectors
-  y - array of  vectors (can be NULL)

   Output Parameters:
.  result - where the result will go

   Level: advanced

   Notes:
   Each call to VecTDotBegin() should be paired with a call to VecTDotEnd().

.seealso: VecMTDotBegin(), VecNormBegin(), VecNormEnd(), VecNorm(), VecDot(), VecMDot(),
         VecDotBegin(), VecDotEnd(), VecMDotBegin(), VecMDotEnd(), PetscCommSplitReductionBegin()
@*/
PetscErrorCode  VecMTDotEnd(Vec x,PetscInt nv,const Vec y[],PetscScalar result[])
{
  PetscFunctionBegin;
  /*
      MTDotEnd() is the same as MDotEnd() so reuse the code
  */
  CHKERRQ(VecMDotEnd(x,nv,y,result));
  PetscFunctionReturn(0);
}
