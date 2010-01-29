#define PETSCVEC_DLL
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

#include "private/vecimpl.h"                              /*I   "petscvec.h"   I*/

#define STATE_BEGIN 0
#define STATE_END   1

#define REDUCE_SUM  0
#define REDUCE_MAX  1
#define REDUCE_MIN  2

typedef struct {
  MPI_Comm     comm;
  PetscScalar  *lvalues;    /* this are the reduced values before call to MPI_Allreduce() */
  PetscScalar  *gvalues;    /* values after call to MPI_Allreduce() */
  void         **invecs;    /* for debugging only, vector/memory used with each op */
  PetscInt     *reducetype; /* is particular value to be summed or maxed? */
  PetscInt     state;       /* are we calling xxxBegin() or xxxEnd()? */
  PetscInt     maxops;      /* total amount of space we have for requests */
  PetscInt     numopsbegin; /* number of requests that have been queued in */
  PetscInt     numopsend;   /* number of requests that have been gotten by user */
} PetscSplitReduction;
/*
   Note: the lvalues and gvalues are twice as long as maxops, this is to allow the second half of
the entries to have a flag indicating if they are REDUCE_SUM, REDUCE_MAX, or REDUCE_MIN these are used by 
the custom reduction operation that replaces MPI_SUM, MPI_MAX, or MPI_MIN in the case when a reduction involves
some of each.
*/

#undef __FUNCT__
#define __FUNCT__ "PetscSplitReductionCreate"
/*
   PetscSplitReductionCreate - Creates a data structure to contain the queued information.
*/
PetscErrorCode PETSCVEC_DLLEXPORT PetscSplitReductionCreate(MPI_Comm comm,PetscSplitReduction **sr)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr               = PetscNew(PetscSplitReduction,sr);CHKERRQ(ierr);
  (*sr)->numopsbegin = 0;
  (*sr)->numopsend   = 0;
  (*sr)->state       = STATE_BEGIN;
  (*sr)->maxops      = 32;
  ierr               = PetscMalloc(2*32*sizeof(PetscScalar),&(*sr)->lvalues);CHKERRQ(ierr);
  ierr               = PetscMalloc(2*32*sizeof(PetscScalar),&(*sr)->gvalues);CHKERRQ(ierr);
  ierr               = PetscMalloc(32*sizeof(void*),&(*sr)->invecs);CHKERRQ(ierr);
  (*sr)->comm        = comm;
  ierr               = PetscMalloc(32*sizeof(PetscInt),&(*sr)->reducetype);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
       This function is the MPI reduction operation used when there is 
   a combination of sums and max in the reduction. The call below to 
   MPI_Op_create() converts the function PetscSplitReduction_Local() to the 
   MPI operator PetscSplitReduction_Op.
*/
MPI_Op PetscSplitReduction_Op = 0;

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "PetscSplitReduction_Local"
void PETSCVEC_DLLEXPORT MPIAPI PetscSplitReduction_Local(void *in,void *out,PetscMPIInt *cnt,MPI_Datatype *datatype)
{
  PetscScalar *xin = (PetscScalar *)in,*xout = (PetscScalar*)out;
  PetscInt    i,count = (PetscInt)*cnt;

  PetscFunctionBegin;
  if (*datatype != MPIU_REAL) {
    (*PetscErrorPrintf)("Can only handle MPIU_REAL data types");
    MPI_Abort(MPI_COMM_WORLD,1);
  }
#if defined(PETSC_USE_COMPLEX)
  count = count/2;
#endif
  count = count/2; 
  for (i=0; i<count; i++) {
    if (((int)PetscRealPart(xin[count+i])) == REDUCE_SUM) { /* second half of xin[] is flags for reduction type */
      xout[i] += xin[i]; 
    } else if ((PetscInt)PetscRealPart(xin[count+i]) == REDUCE_MAX) {
      xout[i] = PetscMax(*(PetscReal *)(xout+i),*(PetscReal *)(xin+i));
    } else if ((PetscInt)PetscRealPart(xin[count+i]) == REDUCE_MIN) {
      xout[i] = PetscMin(*(PetscReal *)(xout+i),*(PetscReal *)(xin+i));
    } else {
      (*PetscErrorPrintf)("Reduction type input is not REDUCE_SUM, REDUCE_MAX, or REDUCE_MIN");
      MPI_Abort(MPI_COMM_WORLD,1);
    }
  }
  PetscStackPop; /* since function returns void cannot use PetscFunctionReturn(); */
  return;
}
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "PetscSplitReductionApply"
/*
   PetscSplitReductionApply - Actually do the communication required for a split phase reduction
*/
PetscErrorCode PETSCVEC_DLLEXPORT PetscSplitReductionApply(PetscSplitReduction *sr)
{
  PetscErrorCode ierr;
  PetscInt       i,numops = sr->numopsbegin,*reducetype = sr->reducetype;
  PetscScalar    *lvalues = sr->lvalues,*gvalues = sr->gvalues;
  PetscInt       sum_flg = 0,max_flg = 0, min_flg = 0;
  MPI_Comm       comm = sr->comm;
  PetscMPIInt    size;

  PetscFunctionBegin;
  if (sr->numopsend > 0) {
    SETERRQ(PETSC_ERR_ORDER,"Cannot call this after VecxxxEnd() has been called");
  }

  ierr = PetscLogEventBarrierBegin(VEC_ReduceBarrier,0,0,0,0,comm);CHKERRQ(ierr);
  ierr  = MPI_Comm_size(sr->comm,&size);CHKERRQ(ierr); 
  if (size == 1) {
    ierr = PetscMemcpy(gvalues,lvalues,numops*sizeof(PetscScalar));CHKERRQ(ierr);
  } else {
    /* determine if all reductions are sum, max, or min */
    for (i=0; i<numops; i++) {
      if (reducetype[i] == REDUCE_MAX) {
        max_flg = 1;
      } else if (reducetype[i] == REDUCE_SUM) {
        sum_flg = 1;
      } else if (reducetype[i] == REDUCE_MIN) {
        min_flg = 1;
      } else {
        SETERRQ(PETSC_ERR_PLIB,"Error in PetscSplitReduction() data structure, probably memory corruption");
      }
    }
    if (sum_flg + max_flg + min_flg > 1) {
      /* 
         after all the entires in lvalues we store the reducetype flags to indicate
         to the reduction operations what are sums and what are max
      */
      for (i=0; i<numops; i++) {
        lvalues[numops+i] = reducetype[i];
      }
#if defined(PETSC_USE_COMPLEX)
      ierr = MPI_Allreduce(lvalues,gvalues,2*2*numops,MPIU_REAL,PetscSplitReduction_Op,comm);CHKERRQ(ierr);
#else
      ierr = MPI_Allreduce(lvalues,gvalues,2*numops,MPIU_REAL,PetscSplitReduction_Op,comm);CHKERRQ(ierr);
#endif
    } else if (max_flg) {
#if defined(PETSC_USE_COMPLEX)
      /* 
        complex case we max both the real and imaginary parts, the imaginary part
        is just ignored later
      */
      ierr = MPI_Allreduce(lvalues,gvalues,2*numops,MPIU_REAL,MPI_MAX,comm);CHKERRQ(ierr);
#else
      ierr = MPI_Allreduce(lvalues,gvalues,numops,MPIU_REAL,MPI_MAX,comm);CHKERRQ(ierr);
#endif
    } else if (min_flg) {
#if defined(PETSC_USE_COMPLEX)
      /* 
        complex case we min both the real and imaginary parts, the imaginary part
        is just ignored later
      */
      ierr = MPI_Allreduce(lvalues,gvalues,2*numops,MPIU_REAL,MPI_MIN,comm);CHKERRQ(ierr);
#else
      ierr = MPI_Allreduce(lvalues,gvalues,numops,MPIU_REAL,MPI_MIN,comm);CHKERRQ(ierr);
#endif
    } else {
      ierr = MPI_Allreduce(lvalues,gvalues,numops,MPIU_SCALAR,MPIU_SUM,comm);CHKERRQ(ierr);
    }
  }
  sr->state     = STATE_END;
  sr->numopsend = 0;
  ierr = PetscLogEventBarrierEnd(VEC_ReduceBarrier,0,0,0,0,comm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "PetscSplitReductionExtend"
/*
   PetscSplitReductionExtend - Double the amount of space (slots) allocated for a split reduction object.
*/
PetscErrorCode PETSCVEC_DLLEXPORT PetscSplitReductionExtend(PetscSplitReduction *sr)
{
  PetscErrorCode ierr;
  PetscInt         maxops = sr->maxops,*reducetype = sr->reducetype;
  PetscScalar *lvalues = sr->lvalues,*gvalues = sr->gvalues;
  void        *invecs = sr->invecs;

  PetscFunctionBegin;
  sr->maxops     = 2*maxops;
  ierr = PetscMalloc(2*2*maxops*sizeof(PetscScalar),&sr->lvalues);CHKERRQ(ierr);
  ierr = PetscMalloc(2*2*maxops*sizeof(PetscScalar),&sr->gvalues);CHKERRQ(ierr);
  ierr = PetscMalloc(2*maxops*sizeof(PetscInt),&sr->reducetype);CHKERRQ(ierr);
  ierr = PetscMalloc(2*maxops*sizeof(void*),&sr->invecs);CHKERRQ(ierr);
  ierr = PetscMemcpy(sr->lvalues,lvalues,maxops*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = PetscMemcpy(sr->gvalues,gvalues,maxops*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = PetscMemcpy(sr->reducetype,reducetype,maxops*sizeof(PetscInt));CHKERRQ(ierr);
  ierr = PetscMemcpy(sr->invecs,invecs,maxops*sizeof(void*));CHKERRQ(ierr);
  ierr = PetscFree(lvalues);CHKERRQ(ierr);
  ierr = PetscFree(gvalues);CHKERRQ(ierr);
  ierr = PetscFree(reducetype);CHKERRQ(ierr);
  ierr = PetscFree(invecs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSplitReductionDestroy"
PetscErrorCode PETSCVEC_DLLEXPORT PetscSplitReductionDestroy(PetscSplitReduction *sr)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(sr->lvalues);CHKERRQ(ierr);
  ierr = PetscFree(sr->gvalues);CHKERRQ(ierr);
  ierr = PetscFree(sr->reducetype);CHKERRQ(ierr);
  ierr = PetscFree(sr->invecs);CHKERRQ(ierr);
  ierr = PetscFree(sr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscMPIInt Petsc_Reduction_keyval = MPI_KEYVAL_INVALID;

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "Petsc_DelReduction" 
/*
   Private routine to delete internal storage when a communicator is freed.
  This is called by MPI, not by users.

  The binding for the first argument changed from MPI 1.0 to 1.1; in 1.0
  it was MPI_Comm *comm.  
*/
int PETSCVEC_DLLEXPORT MPIAPI Petsc_DelReduction(MPI_Comm comm,int keyval,void* attr_val,void* extra_state)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscInfo1(0,"Deleting reduction data in an MPI_Comm %ld\n",(long)comm);CHKERRQ(ierr);
  ierr = PetscSplitReductionDestroy((PetscSplitReduction *)attr_val);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

/*
     PetscSplitReductionGet - Gets the split reduction object from a 
        PETSc vector, creates if it does not exit.

*/
#undef __FUNCT__
#define __FUNCT__ "PetscSplitReductionGet"
PetscErrorCode PETSCVEC_DLLEXPORT PetscSplitReductionGet(MPI_Comm comm,PetscSplitReduction **sr)
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
    ierr = MPI_Keyval_create(MPI_NULL_COPY_FN,Petsc_DelReduction,&Petsc_Reduction_keyval,0);CHKERRQ(ierr);
  }
  ierr = MPI_Attr_get(comm,Petsc_Reduction_keyval,(void **)sr,&flag);CHKERRQ(ierr);
  if (!flag) {  /* doesn't exist yet so create it and put it in */
    ierr = PetscSplitReductionCreate(comm,sr);CHKERRQ(ierr);
    ierr = MPI_Attr_put(comm,Petsc_Reduction_keyval,*sr);CHKERRQ(ierr);
    ierr = PetscInfo1(0,"Putting reduction data in an MPI_Comm %ld\n",(long)comm);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

/* ----------------------------------------------------------------------------------------------------*/

#undef __FUNCT__
#define __FUNCT__ "VecDotBegin"
/*@
   VecDotBegin - Starts a split phase dot product computation.

   Input Parameters:
+   x - the first vector
.   y - the second vector
-   result - where the result will go (can be PETSC_NULL)

   Level: advanced

   Notes:
   Each call to VecDotBegin() should be paired with a call to VecDotEnd().

seealso: VecDotEnd(), VecNormBegin(), VecNormEnd(), VecNorm(), VecDot(), VecMDot(), 
         VecTDotBegin(), VecTDotEnd()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT VecDotBegin(Vec x,Vec y,PetscScalar *result) 
{
  PetscErrorCode      ierr;
  PetscSplitReduction *sr;
  MPI_Comm            comm;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)x,&comm);CHKERRQ(ierr);
  ierr = PetscSplitReductionGet(comm,&sr);CHKERRQ(ierr);
  if (sr->state == STATE_END) {
    SETERRQ(PETSC_ERR_ORDER,"Called before all VecxxxEnd() called");
  }
  if (sr->numopsbegin >= sr->maxops) {
    ierr = PetscSplitReductionExtend(sr);CHKERRQ(ierr);
  }
  sr->reducetype[sr->numopsbegin] = REDUCE_SUM;
  sr->invecs[sr->numopsbegin]     = (void*)x;
  if (!x->ops->dot_local) SETERRQ(PETSC_ERR_SUP,"Vector does not suppport local dots");
  ierr = PetscLogEventBegin(VEC_ReduceArithmetic,0,0,0,0);CHKERRQ(ierr);
  ierr = (*x->ops->dot_local)(x,y,sr->lvalues+sr->numopsbegin++);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(VEC_ReduceArithmetic,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecDotEnd"
/*@
   VecDotEnd - Ends a split phase dot product computation.

   Input Parameters:
+  x - the first vector (can be PETSC_NULL)
.  y - the second vector (can be PETSC_NULL)
-  result - where the result will go

   Level: advanced

   Notes:
   Each call to VecDotBegin() should be paired with a call to VecDotEnd().

seealso: VecDotBegin(), VecNormBegin(), VecNormEnd(), VecNorm(), VecDot(), VecMDot(), 
         VecTDotBegin(),VecTDotEnd()

@*/
PetscErrorCode PETSCVEC_DLLEXPORT VecDotEnd(Vec x,Vec y,PetscScalar *result) 
{
  PetscErrorCode      ierr;
  PetscSplitReduction *sr;
  MPI_Comm            comm;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)x,&comm);CHKERRQ(ierr);
  ierr = PetscSplitReductionGet(comm,&sr);CHKERRQ(ierr);
  
  if (sr->state != STATE_END) {
    /* this is the first call to VecxxxEnd() so do the communication */
    ierr = PetscSplitReductionApply(sr);CHKERRQ(ierr);
  }

  if (sr->numopsend >= sr->numopsbegin) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Called VecxxxEnd() more times then VecxxxBegin()");
  }
  if (x && (void*) x != sr->invecs[sr->numopsend]) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Called VecxxxEnd() in a different order or with a different vector than VecxxxBegin()");
  }
  if (sr->reducetype[sr->numopsend] != REDUCE_SUM) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Called VecDotEnd() on a reduction started with VecNormBegin()");
  }
  *result = sr->gvalues[sr->numopsend++];

  /*
     We are finished getting all the results so reset to no outstanding requests
  */
  if (sr->numopsend == sr->numopsbegin) {
    sr->state        = STATE_BEGIN;
    sr->numopsend    = 0;
    sr->numopsbegin  = 0;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecTDotBegin"
/*@
   VecTDotBegin - Starts a split phase transpose dot product computation.

   Input Parameters:
+  x - the first vector
.  y - the second vector
-  result - where the result will go (can be PETSC_NULL)

   Level: advanced

   Notes:
   Each call to VecTDotBegin() should be paired with a call to VecTDotEnd().

seealso: VecTDotEnd(), VecNormBegin(), VecNormEnd(), VecNorm(), VecDot(), VecMDot(), 
         VecDotBegin(), VecDotEnd()

@*/
PetscErrorCode PETSCVEC_DLLEXPORT VecTDotBegin(Vec x,Vec y,PetscScalar *result) 
{
  PetscErrorCode      ierr;
  PetscSplitReduction *sr;
  MPI_Comm            comm;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)x,&comm);CHKERRQ(ierr);
  ierr = PetscSplitReductionGet(comm,&sr);CHKERRQ(ierr);
  if (sr->state == STATE_END) {
    SETERRQ(PETSC_ERR_ORDER,"Called before all VecxxxEnd() called");
  }
  if (sr->numopsbegin >= sr->maxops) {
    ierr = PetscSplitReductionExtend(sr);CHKERRQ(ierr);
  }
  sr->reducetype[sr->numopsbegin] = REDUCE_SUM;
  sr->invecs[sr->numopsbegin]     = (void*)x;
  if (!x->ops->tdot_local) SETERRQ(PETSC_ERR_SUP,"Vector does not suppport local dots");
  ierr = PetscLogEventBegin(VEC_ReduceArithmetic,0,0,0,0);CHKERRQ(ierr);
  ierr = (*x->ops->dot_local)(x,y,sr->lvalues+sr->numopsbegin++);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(VEC_ReduceArithmetic,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecTDotEnd"
/*@
   VecTDotEnd - Ends a split phase transpose dot product computation.

   Input Parameters:
+  x - the first vector (can be PETSC_NULL)
.  y - the second vector (can be PETSC_NULL)
-  result - where the result will go

   Level: advanced

   Notes:
   Each call to VecTDotBegin() should be paired with a call to VecTDotEnd().

seealso: VecTDotBegin(), VecNormBegin(), VecNormEnd(), VecNorm(), VecDot(), VecMDot(), 
         VecDotBegin(), VecDotEnd()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT VecTDotEnd(Vec x,Vec y,PetscScalar *result) 
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

#undef __FUNCT__
#define __FUNCT__ "VecNormBegin"
/*@
   VecNormBegin - Starts a split phase norm computation.

   Input Parameters:
+  x - the first vector
.  ntype - norm type, one of NORM_1, NORM_2, NORM_MAX, NORM_1_AND_2
-  result - where the result will go (can be PETSC_NULL)

   Level: advanced

   Notes:
   Each call to VecNormBegin() should be paired with a call to VecNormEnd().

.seealso: VecNormEnd(), VecNorm(), VecDot(), VecMDot(), VecDotBegin(), VecDotEnd()

@*/
PetscErrorCode PETSCVEC_DLLEXPORT VecNormBegin(Vec x,NormType ntype,PetscReal *result) 
{
  PetscErrorCode      ierr;
  PetscSplitReduction *sr;
  PetscReal           lresult[2];
  MPI_Comm            comm;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)x,&comm);CHKERRQ(ierr);
  ierr = PetscSplitReductionGet(comm,&sr);CHKERRQ(ierr);
  if (sr->state == STATE_END) {
    SETERRQ(PETSC_ERR_ORDER,"Called before all VecxxxEnd() called");
  }
  if (sr->numopsbegin >= sr->maxops || (sr->numopsbegin == sr->maxops-1 && ntype == NORM_1_AND_2)) {
    ierr = PetscSplitReductionExtend(sr);CHKERRQ(ierr);
  }
  
  sr->invecs[sr->numopsbegin]     = (void*)x;
  if (!x->ops->norm_local) SETERRQ(PETSC_ERR_SUP,"Vector does not support local norms");
  ierr = PetscLogEventBegin(VEC_ReduceArithmetic,0,0,0,0);CHKERRQ(ierr);
  ierr = (*x->ops->norm_local)(x,ntype,lresult);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(VEC_ReduceArithmetic,0,0,0,0);CHKERRQ(ierr);
  if (ntype == NORM_2)         lresult[0]                = lresult[0]*lresult[0];
  if (ntype == NORM_1_AND_2)   lresult[1]                = lresult[1]*lresult[1];
  if (ntype == NORM_MAX) sr->reducetype[sr->numopsbegin] = REDUCE_MAX;
  else                   sr->reducetype[sr->numopsbegin] = REDUCE_SUM;
  sr->lvalues[sr->numopsbegin++] = lresult[0];
  if (ntype == NORM_1_AND_2) {
    sr->reducetype[sr->numopsbegin] = REDUCE_SUM;
    sr->lvalues[sr->numopsbegin++]  = lresult[1]; 
  }   
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecNormEnd"
/*@
   VecNormEnd - Ends a split phase norm computation.

   Input Parameters:
+  x - the first vector (can be PETSC_NULL)
.  ntype - norm type, one of NORM_1, NORM_2, NORM_MAX, NORM_1_AND_2
-  result - where the result will go

   Level: advanced

   Notes:
   Each call to VecNormBegin() should be paired with a call to VecNormEnd().

.seealso: VecNormBegin(), VecNorm(), VecDot(), VecMDot(), VecDotBegin(), VecDotEnd()

@*/
PetscErrorCode PETSCVEC_DLLEXPORT VecNormEnd(Vec x,NormType ntype,PetscReal *result) 
{
  PetscErrorCode      ierr;
  PetscSplitReduction *sr;
  MPI_Comm            comm;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)x,&comm);CHKERRQ(ierr);
  ierr = PetscSplitReductionGet(comm,&sr);CHKERRQ(ierr);
  
  if (sr->state != STATE_END) {
    /* this is the first call to VecxxxEnd() so do the communication */
    ierr = PetscSplitReductionApply(sr);CHKERRQ(ierr);
  }

  if (sr->numopsend >= sr->numopsbegin) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Called VecxxxEnd() more times then VecxxxBegin()");
  }
  if (x && (void*)x != sr->invecs[sr->numopsend]) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Called VecxxxEnd() in a different order or with a different vector than VecxxxBegin()");
  }
  if (sr->reducetype[sr->numopsend] != REDUCE_MAX && ntype == NORM_MAX) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Called VecNormEnd(,NORM_MAX,) on a reduction started with VecDotBegin() or NORM_1 or NORM_2");
  }
  result[0] = PetscRealPart(sr->gvalues[sr->numopsend++]);

  if (ntype == NORM_2) {
    result[0] = sqrt(result[0]);
  } else if (ntype == NORM_1_AND_2) {
    result[1] = PetscRealPart(sr->gvalues[sr->numopsend++]);
    result[1] = sqrt(result[1]);
  }
  if (ntype!=NORM_1_AND_2) {
    ierr = PetscObjectComposedDataSetReal((PetscObject)x,NormIds[ntype],result[0]);CHKERRQ(ierr);
  }

  if (sr->numopsend == sr->numopsbegin) {
    sr->state        = STATE_BEGIN;
    sr->numopsend    = 0;
    sr->numopsbegin  = 0;
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

#undef __FUNCT__
#define __FUNCT__ "VecMDotBegin"
/*@
   VecMDotBegin - Starts a split phase multiple dot product computation.

   Input Parameters:
+   x - the first vector
.   nv - number of vectors
.   y - array of vectors
-   result - where the result will go (can be PETSC_NULL)

   Level: advanced

   Notes:
   Each call to VecMDotBegin() should be paired with a call to VecMDotEnd().

seealso: VecMDotEnd(), VecNormBegin(), VecNormEnd(), VecNorm(), VecDot(), VecMDot(), 
         VecTDotBegin(), VecTDotEnd(), VecMTDotBegin(), VecMTDotEnd()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT VecMDotBegin(Vec x,PetscInt nv,const Vec y[],PetscScalar result[]) 
{
  PetscErrorCode      ierr;
  PetscSplitReduction *sr;
  MPI_Comm            comm;
  int                 i;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)x,&comm);CHKERRQ(ierr);
  ierr = PetscSplitReductionGet(comm,&sr);CHKERRQ(ierr);
  if (sr->state == STATE_END) {
    SETERRQ(PETSC_ERR_ORDER,"Called before all VecxxxEnd() called");
  }
  for (i=0;i<nv;i++) {
    if (sr->numopsbegin+i >= sr->maxops) {
      ierr = PetscSplitReductionExtend(sr);CHKERRQ(ierr);
    }
    sr->reducetype[sr->numopsbegin+i] = REDUCE_SUM;
    sr->invecs[sr->numopsbegin+i]     = (void*)x;
  }
  if (!x->ops->mdot_local) SETERRQ(PETSC_ERR_SUP,"Vector does not suppport local mdots");
  ierr = PetscLogEventBegin(VEC_ReduceArithmetic,0,0,0,0);CHKERRQ(ierr);
  ierr = (*x->ops->mdot_local)(x,nv,y,sr->lvalues+sr->numopsbegin);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(VEC_ReduceArithmetic,0,0,0,0);CHKERRQ(ierr);
  sr->numopsbegin += nv;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecMDotEnd"
/*@
   VecMDotEnd - Ends a split phase multiple dot product computation.

   Input Parameters:
+   x - the first vector (can be PETSC_NULL)
.   nv - number of vectors
-   y - array of vectors (can be PETSC_NULL)

   Output Parameters:
.   result - where the result will go

   Level: advanced

   Notes:
   Each call to VecMDotBegin() should be paired with a call to VecMDotEnd().

seealso: VecMDotBegin(), VecNormBegin(), VecNormEnd(), VecNorm(), VecDot(), VecMDot(), 
         VecTDotBegin(),VecTDotEnd(), VecMTDotBegin(), VecMTDotEnd()

@*/
PetscErrorCode PETSCVEC_DLLEXPORT VecMDotEnd(Vec x,PetscInt nv,const Vec y[],PetscScalar result[]) 
{
  PetscErrorCode      ierr;
  PetscSplitReduction *sr;
  MPI_Comm            comm;
  int                 i;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)x,&comm);CHKERRQ(ierr);
  ierr = PetscSplitReductionGet(comm,&sr);CHKERRQ(ierr);
  
  if (sr->state != STATE_END) {
    /* this is the first call to VecxxxEnd() so do the communication */
    ierr = PetscSplitReductionApply(sr);CHKERRQ(ierr);
  }

  if (sr->numopsend >= sr->numopsbegin) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Called VecxxxEnd() more times then VecxxxBegin()");
  }
  if (x && (void*) x != sr->invecs[sr->numopsend]) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Called VecxxxEnd() in a different order or with a different vector than VecxxxBegin()");
  }
  if (sr->reducetype[sr->numopsend] != REDUCE_SUM) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Called VecDotEnd() on a reduction started with VecNormBegin()");
  }
  for (i=0;i<nv;i++) {
    result[i] = sr->gvalues[sr->numopsend++];
  }
  
  /*
     We are finished getting all the results so reset to no outstanding requests
  */
  if (sr->numopsend == sr->numopsbegin) {
    sr->state        = STATE_BEGIN;
    sr->numopsend    = 0;
    sr->numopsbegin  = 0;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecMTDotBegin"
/*@
   VecMTDotBegin - Starts a split phase transpose multiple dot product computation.

   Input Parameters:
+  x - the first vector
.  nv - number of vectors
.  y - array of  vectors
-  result - where the result will go (can be PETSC_NULL)

   Level: advanced

   Notes:
   Each call to VecMTDotBegin() should be paired with a call to VecMTDotEnd().

seealso: VecMTDotEnd(), VecNormBegin(), VecNormEnd(), VecNorm(), VecDot(), VecMDot(), 
         VecDotBegin(), VecDotEnd(), VecMDotBegin(), VecMDotEnd()

@*/
PetscErrorCode PETSCVEC_DLLEXPORT VecMTDotBegin(Vec x,PetscInt nv,const Vec y[],PetscScalar result[]) 
{
  PetscErrorCode      ierr;
  PetscSplitReduction *sr;
  MPI_Comm            comm;
  int                 i;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)x,&comm);CHKERRQ(ierr);
  ierr = PetscSplitReductionGet(comm,&sr);CHKERRQ(ierr);
  if (sr->state == STATE_END) {
    SETERRQ(PETSC_ERR_ORDER,"Called before all VecxxxEnd() called");
  }
  for (i=0;i<nv;i++) {
    if (sr->numopsbegin+i >= sr->maxops) {
      ierr = PetscSplitReductionExtend(sr);CHKERRQ(ierr);
    }
    sr->reducetype[sr->numopsbegin+i] = REDUCE_SUM;
    sr->invecs[sr->numopsbegin+i]     = (void*)x;
  }
  if (!x->ops->mtdot_local) SETERRQ(PETSC_ERR_SUP,"Vector does not suppport local mdots");
  ierr = PetscLogEventBegin(VEC_ReduceArithmetic,0,0,0,0);CHKERRQ(ierr);
  ierr = (*x->ops->mdot_local)(x,nv,y,sr->lvalues+sr->numopsbegin);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(VEC_ReduceArithmetic,0,0,0,0);CHKERRQ(ierr);
  sr->numopsbegin += nv;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecMTDotEnd"
/*@
   VecMTDotEnd - Ends a split phase transpose multiple dot product computation.

   Input Parameters:
+  x - the first vector (can be PETSC_NULL)
.  nv - number of vectors
-  y - array of  vectors (can be PETSC_NULL)

   Output Parameters
.  result - where the result will go

   Level: advanced

   Notes:
   Each call to VecTDotBegin() should be paired with a call to VecTDotEnd().

seealso: VecMTDotBegin(), VecNormBegin(), VecNormEnd(), VecNorm(), VecDot(), VecMDot(), 
         VecDotBegin(), VecDotEnd(), VecMDotBegin(), VecMdotEnd()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT VecMTDotEnd(Vec x,PetscInt nv,const Vec y[],PetscScalar result[]) 
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /*
      MTDotEnd() is the same as MDotEnd() so reuse the code
  */
  ierr = VecMDotEnd(x,nv,y,result);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
