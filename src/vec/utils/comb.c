#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: comb.c,v 1.12 1999/04/06 03:02:31 bsmith Exp bsmith $";
#endif

/*
      Split phase global vector reductions with support for combining the
   communication portion of several operations. Using MPI-1.1 support only

      The idea for this and much of the initial code is contributed by 
   Victor Eijkhout.

       Usage:
             VecDotBegin(Vec,Vec,Scalar *);
             VecNormBegin(Vec,NormType,double *);
             ....
             VecDotEnd(Vec,Vec,Scalar *);
             VecNormEnd(Vec,NormType,double *);

       Limitations: 
         - currently only works for PETSc seq and mpi vectors
         - The order of the xxxEnd() functions MUST be in the same order
           as the xxxBegin(). There is extensive error checking to try to 
           insure that the user calls the routines in the correct order
*/

#include "src/vec/vecimpl.h"                              /*I   "vec.h"   I*/

#define STATE_BEGIN 0
#define STATE_END   1

#define REDUCE_SUM  0
#define REDUCE_MAX  1

typedef struct {
  MPI_Comm comm;
  Scalar   *lvalues;    /* this are the reduced values before call to MPI_Allreduce() */
  Scalar   *gvalues;    /* values after call to MPI_Allreduce() */
  Vec      *invecs;     /* for debugging only, vector used with each op */
  int      *reducetype; /* is particular value to be summed or maxed? */
  int      state;       /* are we calling xxxBegin() or xxxEnd()? */
  int      maxops;      /* total amount of space we have for requests */
  int      numopsbegin; /* number of requests that have been queued in */
  int      numopsend;   /* number of requests that have been gotten by user */
} VecSplitReduction;
/*
   Note: the lvalues and gvalues are twice as long as maxops this is to allow the second half of
the entries to have a flag indicating if they are REDUCE_SUM or REDUCE_MAX, these are used by 
the custom reduction operation that replaces MPI_SUM or MPI_MAX in the case when a reduction involves
some of each.
*/

#undef __FUNC__
#define __FUNC__ "VecSplitReductionCreate"
/*
   VecSplitReductionCreate - Creates a data structure to contain the queued 
                             information.
*/
int VecSplitReductionCreate(MPI_Comm comm,VecSplitReduction **sr)
{
  PetscFunctionBegin;
  (*sr)              = PetscNew(VecSplitReduction); CHKPTRQ((*sr));
  (*sr)->numopsbegin = 0;
  (*sr)->numopsend   = 0;
  (*sr)->state       = STATE_BEGIN;
  (*sr)->maxops      = 32;
  (*sr)->lvalues     = (Scalar *) PetscMalloc(2*32*sizeof(Scalar));CHKPTRQ((*sr)->lvalues);
  (*sr)->gvalues     = (Scalar *) PetscMalloc(2*32*sizeof(Scalar));CHKPTRQ((*sr)->gvalues);
  (*sr)->invecs      = (Vec *)    PetscMalloc(32*sizeof(Vec));CHKPTRQ((*sr)->invecs);
  (*sr)->comm        = comm;
  (*sr)->reducetype  = (int *) PetscMalloc(32*sizeof(int));CHKPTRQ((*sr)->reducetype);
  PetscFunctionReturn(0);
}

/*
       This function is the MPI reduction operation used when there is 
   a combination of sums and max in the reduction. The call below to 
   MPI_Op_create() converts the function VecSplitReduction_Local() to the 
   MPI operator VecSplitReduction_Op.
*/
MPI_Op VecSplitReduction_Op = 0;

#undef __FUNC__
#define __FUNC__ "VecSplitReduction_Local"
void VecSplitReduction_Local(void *in, void *out,int *cnt,MPI_Datatype *datatype)
{
  Scalar *xin = (Scalar *)in, *xout = (Scalar *) out;
  int    i, count = *cnt;

  if (*datatype != MPI_DOUBLE) {
    (*PetscErrorPrintf)("Can only handle MPI_DOUBLE data types");
    MPI_Abort(MPI_COMM_WORLD,1);
  }
#if defined(USE_PETSC_COMPLEX)
  count = count/2;
#endif
  count = count/2; 
  for ( i=0; i<count; i++ ) {
    if ((int) PetscReal(xin[count+i]) == REDUCE_SUM) { /* second half of xin[] is flags for reduction type */
      xout[i] += xin[i]; 
    } else if (xin[count+i] == REDUCE_MAX) {
      xout[i] = PetscMax(*(double *)(xout+i),*(double *)(xin+i));
    } else {
      (*PetscErrorPrintf)("Reduction type input is not REDUCE_SUM or REDUCE_MAX");
      MPI_Abort(MPI_COMM_WORLD,1);
    }
  }
  return;
}

#undef __FUNC__
#define __FUNC__ "VecSplitReductionApply"
/*
   VecSplitReductionApply - Actually do the communication required for a split phase reduction
*/
int VecSplitReductionApply(VecSplitReduction *sr)
{
  int        size,ierr,i,numops = sr->numopsbegin, *reducetype = sr->reducetype;
  Scalar     *lvalues = sr->lvalues,*gvalues = sr->gvalues;
  PetscTruth sum_flg = PETSC_FALSE, max_flg = PETSC_FALSE;
  MPI_Comm   comm = sr->comm;

  PetscFunctionBegin;
  if (sr->numopsend > 0) {
    SETERRQ(1,1,"Cannot call this after VecxxxEnd() has been called");
  }

  PLogEventBegin(VEC_ReduceCommunication,0,0,0,0);
  PLogEventBarrierBegin(VEC_ReduceBarrier,0,0,0,0,comm);
  MPI_Comm_size(sr->comm,&size);
  if (size == 1) {
    PetscMemcpy(gvalues,lvalues,numops*sizeof(Scalar));
  } else {
    /* determine if all reductions are sum, or if some involve max */
    for ( i=0; i<numops; i++ ) {
      if (reducetype[i] == REDUCE_MAX) {
        max_flg = PETSC_TRUE;
      } else if (reducetype[i] == REDUCE_SUM) {
        sum_flg = PETSC_TRUE;
      } else {
        SETERRQ(1,1,"Error in VecSplitReduction data structure, probably memory corruption");
      }
    }
    if (sum_flg && max_flg) {
      /* 
         after all the entires in lvalues we store the reducetype flags to indicate
         to the reduction operations what are sums and what are max
      */
      for ( i=0; i<numops; i++ ) {
        lvalues[numops+i] = reducetype[i];
      }
#if defined(USE_PETSC_COMPLEX)
      ierr = MPI_Allreduce(lvalues,gvalues,2*2*numops,MPI_DOUBLE,VecSplitReduction_Op,comm);CHKERRQ(ierr);
#else
      ierr = MPI_Allreduce(lvalues,gvalues,2*numops,MPI_DOUBLE,VecSplitReduction_Op,comm);CHKERRQ(ierr);
#endif
    } else if (max_flg) {
#if defined(USE_PETSC_COMPLEX)
      /* 
        complex case we max both the real and imaginary parts, the imaginary part
        is just ignored later
      */
      ierr = MPI_Allreduce(lvalues,gvalues,2*numops,MPI_DOUBLE,MPI_MAX,comm);CHKERRQ(ierr);
#else
      ierr = MPI_Allreduce(lvalues,gvalues,numops,MPI_DOUBLE,MPI_MAX,comm);CHKERRQ(ierr);
#endif
    } else {
#if defined(USE_PETSC_COMPLEX)
      ierr = MPI_Allreduce(lvalues,gvalues,2*numops,MPI_DOUBLE,MPI_SUM,comm);CHKERRQ(ierr);
#else
      ierr = MPI_Allreduce(lvalues,gvalues,numops,MPI_DOUBLE,MPI_SUM,comm);CHKERRQ(ierr);
#endif
    }
  }
  sr->state     = STATE_END;
  sr->numopsend = 0;
  PLogEventBarrierEnd(VEC_ReduceBarrier,0,0,0,0,comm);
  PLogEventEnd(VEC_ReduceCommunication,0,0,0,0);
  PetscFunctionReturn(0);
}


#undef __FUNC__
#define __FUNC__ "VecSplitReductionExtend"
/*
   VecSplitReductionExtend - Double the amount of space (slots) allocated for 
                             a split reduction object.
*/
int VecSplitReductionExtend(VecSplitReduction *sr)
{
  int    maxops = sr->maxops, *reducetype = sr->reducetype;
  Scalar *lvalues = sr->lvalues,*gvalues = sr->gvalues;
  Vec    *invecs = sr->invecs;

  PetscFunctionBegin;
  sr->maxops     = 2*maxops;
  sr->lvalues    = (Scalar *) PetscMalloc(2*2*maxops*sizeof(Scalar));CHKPTRQ(sr->lvalues);
  sr->gvalues    = (Scalar *) PetscMalloc(2*2*maxops*sizeof(Scalar));CHKPTRQ(sr->gvalues);
  sr->reducetype = (int *) PetscMalloc(2*maxops*sizeof(int));CHKPTRQ(sr->reducetype);
  sr->invecs     = (Vec *) PetscMalloc(2*maxops*sizeof(Vec));CHKPTRQ(sr->invecs);
  PetscMemcpy(sr->lvalues,lvalues,maxops*sizeof(Scalar));
  PetscMemcpy(sr->gvalues,gvalues,maxops*sizeof(Scalar));
  PetscMemcpy(sr->reducetype,reducetype,maxops*sizeof(int));
  PetscMemcpy(sr->invecs,invecs,maxops*sizeof(Vec));
  PetscFree(lvalues);
  PetscFree(gvalues);
  PetscFree(reducetype);
  PetscFree(invecs);
  PetscFunctionReturn(0);
}

#undef __FUNC__
#define __FUNC__ "VecSplitReductionDestroy"
int VecSplitReductionDestroy(VecSplitReduction *sr)
{
  PetscFunctionBegin;
  PetscFree(sr->lvalues); 
  PetscFree(sr->gvalues); 
  PetscFree(sr->reducetype); 
  PetscFree(sr->invecs); 
  PetscFree(sr);
  PetscFunctionReturn(0);
}

static int Petsc_Reduction_keyval = MPI_KEYVAL_INVALID;

#undef __FUNC__  
#define __FUNC__ "Petsc_DelReduction" 
/*
   Private routine to delete internal storage when a communicator is freed.
  This is called by MPI, not by users.

  The binding for the first argument changed from MPI 1.0 to 1.1; in 1.0
  it was MPI_Comm *comm.  
*/
int Petsc_DelReduction(MPI_Comm comm,int keyval,void* attr_val,void* extra_state )
{
  int ierr;

  PetscFunctionBegin;
  PLogInfo(0,"Petsc_DelReduction:Deleting reduction data in an MPI_Comm %d\n",(int) comm);
  ierr = VecSplitReductionDestroy((VecSplitReduction *)attr_val);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
     VecSplitReductionGet - Gets the split reduction object from a 
        PETSc vector, creates if it does not exit.

*/
#undef __FUNC__
#define __FUNC__ "VecSplitReductionGet"
int VecSplitReductionGet(Vec x,VecSplitReduction **sr)
{
  MPI_Comm comm;
  int      ierr,flag;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)x,&comm);CHKERRQ(ierr);
  if (Petsc_Reduction_keyval == MPI_KEYVAL_INVALID) {
    /* 
       The calling sequence of the 2nd argument to this function changed
       between MPI Standard 1.0 and the revisions 1.1 Here we match the 
       new standard, if you are using an MPI implementation that uses 
       the older version you will get a warning message about the next line;
       it is only a warning message and should do no harm.
    */
    ierr = MPI_Keyval_create(MPI_NULL_COPY_FN,Petsc_DelReduction,&Petsc_Reduction_keyval,0);CHKERRQ(ierr);
    /*
         Also create the special MPI reduction operation that may be needed 
    */
    ierr = MPI_Op_create(VecSplitReduction_Local,1,&VecSplitReduction_Op);CHKERRQ(ierr);
  }
  ierr = MPI_Attr_get(comm,Petsc_Reduction_keyval,(void **)sr,&flag);CHKERRQ(ierr);
  if (!flag) {  /* doesn't exist yet so create it and put it in */
    ierr = VecSplitReductionCreate(comm,sr);CHKERRQ(ierr);
    ierr = MPI_Attr_put(comm,Petsc_Reduction_keyval,*sr);CHKERRQ(ierr);
    PLogInfo(0,"VecSplitReductionGet:Putting reduction data in an MPI_Comm %d\n",(int) comm);
  }

  PetscFunctionReturn(0);
}

/* ----------------------------------------------------------------------------------------------------*/

#undef __FUNC__
#define __FUNC__ "VecDotBegin"
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
int VecDotBegin(Vec x, Vec y,Scalar *result) 
{
  int               ierr;
  VecSplitReduction *sr;

  PetscFunctionBegin;
  ierr = VecSplitReductionGet(x,&sr);CHKERRQ(ierr);
  if (sr->state == STATE_END) {
    SETERRQ(1,1,"Called before all VecxxxEnd() called");
  }
  if (sr->numopsbegin >= sr->maxops) {
    ierr = VecSplitReductionExtend(sr);CHKERRQ(ierr);
  }
  sr->reducetype[sr->numopsbegin] = REDUCE_SUM;
  sr->invecs[sr->numopsbegin]     = x;
  if (!x->ops->dot_local) SETERRQ(1,1,"Vector does not suppport local dots");
  PLogEventBegin(VEC_ReduceArithmetic,0,0,0,0);
  ierr = (*x->ops->dot_local)(x,y,sr->lvalues+sr->numopsbegin++);CHKERRQ(ierr);
  PLogEventEnd(VEC_ReduceArithmetic,0,0,0,0);
  PetscFunctionReturn(0);
}

#undef __FUNC__
#define __FUNC__ "VecDotEnd"
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
         VecTDotBegin(), VecTDotEnd()

@*/
int VecDotEnd(Vec x, Vec y,Scalar *result) 
{
  int               ierr;
  VecSplitReduction *sr;

  PetscFunctionBegin;
  ierr = VecSplitReductionGet(x,&sr);CHKERRQ(ierr);
  
  if (sr->state != STATE_END) {
    /* this is the first call to VecxxxEnd() so do the communication */
    ierr = VecSplitReductionApply(sr);CHKERRQ(ierr);
  }

  if (sr->numopsend >= sr->numopsbegin) {
    SETERRQ(1,1,"Called VecxxxEnd() more times then VecxxxBegin()");
  }
  if (x && x != sr->invecs[sr->numopsend]) {
    SETERRQ(1,1,"Called VecxxxEnd() in a different order or with a different vector than VecxxxBegin()");
  }
  if (sr->reducetype[sr->numopsend] != REDUCE_SUM) {
    SETERRQ(1,1,"Called VecDotEnd() on a reduction started with VecNormBegin()");
  }
  *result = sr->lvalues[sr->numopsend++];

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

#undef __FUNC__
#define __FUNC__ "VecTDotBegin"
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
int VecTDotBegin(Vec x, Vec y,Scalar *result) 
{
  int               ierr;
  VecSplitReduction *sr;

  PetscFunctionBegin;
  ierr = VecSplitReductionGet(x,&sr);CHKERRQ(ierr);
  if (sr->state == STATE_END) {
    SETERRQ(1,1,"Called before all VecxxxEnd() called");
  }
  if (sr->numopsbegin >= sr->maxops) {
    ierr = VecSplitReductionExtend(sr);CHKERRQ(ierr);
  }
  sr->reducetype[sr->numopsbegin] = REDUCE_SUM;
  sr->invecs[sr->numopsbegin]     = x;
  if (!x->ops->tdot_local) SETERRQ(1,1,"Vector does not suppport local dots");
  PLogEventBegin(VEC_ReduceArithmetic,0,0,0,0);
  ierr = (*x->ops->dot_local)(x,y,sr->lvalues+sr->numopsbegin++);CHKERRQ(ierr);
  PLogEventEnd(VEC_ReduceArithmetic,0,0,0,0);
  PetscFunctionReturn(0);
}

#undef __FUNC__
#define __FUNC__ "VecTDotEnd"
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
int VecTDotEnd(Vec x, Vec y,Scalar *result) 
{
  int               ierr;

  PetscFunctionBegin;
  /*
      TDotEnd() is the same as DotEnd() so reuse the code
  */
  ierr = VecDotEnd(x,y,result);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------*/

#undef __FUNC__
#define __FUNC__ "VecNormBegin"
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
int VecNormBegin(Vec x, NormType ntype, double *result) 
{
  int               ierr;
  VecSplitReduction *sr;
  double            lresult[2];

  PetscFunctionBegin;
  ierr = VecSplitReductionGet(x,&sr);CHKERRQ(ierr);
  if (sr->state == STATE_END) {
    SETERRQ(1,1,"Called before all VecxxxEnd() called");
  }
  if (sr->numopsbegin >= sr->maxops || (sr->numopsbegin == sr->maxops-1 && ntype == NORM_1_AND_2)) {
    ierr = VecSplitReductionExtend(sr);CHKERRQ(ierr);
  }
  
  sr->invecs[sr->numopsbegin]     = x;
  if (!x->ops->norm_local) SETERRQ(1,1,"Vector does not support local norms");
  PLogEventBegin(VEC_ReduceArithmetic,0,0,0,0);
  ierr = (*x->ops->norm_local)(x,ntype,lresult);CHKERRQ(ierr);
  PLogEventEnd(VEC_ReduceArithmetic,0,0,0,0);
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

#undef __FUNC__
#define __FUNC__ "VecNormBegin"
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
int VecNormEnd(Vec x, NormType ntype,double *result) 
{
  int               ierr;
  VecSplitReduction *sr;

  PetscFunctionBegin;
  ierr = VecSplitReductionGet(x,&sr);CHKERRQ(ierr);
  
  if (sr->state != STATE_END) {
    /* this is the first call to VecxxxEnd() so do the communication */
    ierr = VecSplitReductionApply(sr);CHKERRQ(ierr);
  }

  if (sr->numopsend >= sr->numopsbegin) {
    SETERRQ(1,1,"Called VecxxxEnd() more times then VecxxxBegin()");
  }
  if (x && x != sr->invecs[sr->numopsend]) {
    SETERRQ(1,1,"Called VecxxxEnd() in a different order or with a different vector than VecxxxBegin()");
  }
  if (sr->reducetype[sr->numopsend] != REDUCE_MAX && ntype == NORM_MAX) {
    SETERRQ(1,1,"Called VecNormEnd(,NORM_MAX,) on a reduction started with VecDotBegin() or NORM_1 or NORM_2");
  }
  result[0] = PetscReal(sr->lvalues[sr->numopsend++]);
  if (ntype == NORM_2) {
    result[0] = sqrt(result[0]);
  } else if (ntype == NORM_1_AND_2) {
    result[1] = PetscReal(sr->lvalues[sr->numopsend++]);
    result[1] = sqrt(result[1]);
  }

  if (sr->numopsend == sr->numopsbegin) {
    sr->state        = STATE_BEGIN;
    sr->numopsend    = 0;
    sr->numopsbegin  = 0;
  }
  PetscFunctionReturn(0);
}



