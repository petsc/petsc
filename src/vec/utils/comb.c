
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
           as the xxxBegin()
*/

#include "src/vec/impls/dvecimpl.h"   /*I   "vec.h"   I*/

#define STATE_BEGIN 0
#define STATE_END   1

#define REDUCE_SUM  0
#define REDUCE_ABS  1

typedef struct {
  MPI_Comm comm;
  Scalar   *lvalues;    /* this are the reduced values before call to MPI_Allreduce() */
  Scalar   *gvalues;    /* values after call to MPI_Allreduce() */
  Vec      *invecs;     /* for debugging only, vector used with each op */
  int      *reducetype; /* is particular value to be summed or absed? */
  int      state;       /* are we still calling xxxBegin() or xxxEnd()? */
  int      maxops;      /* total amount of space we have for requests */
  int      numopsbegin; /* number of requests that have been queued in */
  int      numopsend;   /* number of requests that have been gotten by user */
} VecSplitReduction;



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
  (*sr)->maxops      = 32;
  (*sr)->lvalues     = (Scalar *) PetscMalloc(2*32*sizeof(Scalar));CHKPTRQ((*sr)->lvalues);
  (*sr)->gvalues     = (Scalar *) PetscMalloc(2*32*sizeof(Scalar));CHKPTRQ((*sr)->gvalues);
  (*sr)->invecs      = (Vec *)    PetscMalloc(32*sizeof(Vec));CHKPTRQ((*sr)->invecs);
  (*sr)->comm        = comm;
  (*sr)->reducetype  = (int *) PetscMalloc(32*sizeof(int));CHKPTRQ((*sr)->reducetype);
  PetscFunctionReturn(0);
}

#undef __FUNC__
#define __FUNC__ "VecSplitReductionApply"
/*
   VecSplitReductionApply - Actually do the communication required for a split phase reduction
*/
int VecSplitReductionApply(VecSplitReduction *sr)
{
  int        ierr,i,numops = sr->numopsbegin, *reducetype = sr->reducetype;
  Scalar     *lvalues = sr->lvalues,*gvalues = sr->gvalues;
  PetscTruth sum_flg = PETSC_FALSE, abs_flg = PETSC_FALSE;
  MPI_Comm   comm = sr->comm;

  PetscFunctionBegin;
  /* determine if all reductions are sum, or if some involve abs() */
  for ( i=0; i<numops; i++ ) {
    if (reducetype[i] == REDUCE_ABS) {
      abs_flg = PETSC_TRUE;
    } else {
      sum_flg = PETSC_TRUE;
    }
  }
  if (sum_flg && abs_flg) {
    SETERRQ(1,1,"Cannot yet handled mixed sum and abs in reductions");
  } else if (abs_flg) {
    ;
  } else {
    ierr = MPI_Allreduce(lvalues,gvalues,numops,MPI_DOUBLE,MPI_SUM,comm);CHKERRQ(ierr);
  }
  sr->state     = STATE_END;
  sr->numopsend = 0;
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
  sr->reducetype = (int *) PetscMalloc(maxops*sizeof(int));CHKPTRQ(sr->reducetype);
  sr->invecs     = (Vec *) PetscMalloc(maxops*sizeof(Vec));CHKPTRQ(sr->invecs);
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
  PLogInfo(0,"Deleting reduction data in an MPI_Comm %d\n",(int) comm);
  ierr = VecSplitReductionDestroy((VecSplitReduction *)attr_val);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
     VecSplitReductionGet - Gets the split reduction object from a 
        PETSc vector, creates if it does not exit.

*/
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
    ierr = MPI_Keyval_create(MPI_NULL_COPY_FN,Petsc_DelReduction,&Petsc_Reduction_keyval,(void*)0);CHKERRQ(ierr);
  }
  ierr = MPI_Attr_get(comm,Petsc_Reduction_keyval,(void **)sr,&flag);CHKERRQ(ierr);
  if (!flag) {  /* doesn't exist yet so create it and put it in */
    ierr = VecSplitReductionCreate(comm,sr);CHKERRQ(ierr);
    ierr = MPI_Attr_put(comm,Petsc_Reduction_keyval,sr);CHKERRQ(ierr);
    PLogInfo(0,"Putting reduction data in an MPI_Comm %d\n",(int) comm);
  }

  PetscFunctionReturn(0);
}

/*@
     VecDotBegin - Starts a split phase dot product

  Input Parameters:
+   x - the first vector
.   y - the second vector
-   result - where the result will go (can be PETSC_NULL)

@*/
int VecDotBegin(Vec x, Vec y,Scalar *result) 
{
  int               ierr;
  VecSplitReduction *sr;

  PetscFunctionBegin;
  ierr = VecSplitReductionGet(x,&sr);CHKERRQ(ierr);
  if (sr->state == STATE_END) {
    SETERRQ(1,1,"Vectors communicator involved in a reduction operation that was completed\n\
                 but not yet read");
  }
  if (sr->numopsbegin >= sr->maxops) {
    ierr = VecSplitReductionExtend(sr);CHKERRQ(ierr);
  }
  sr->reducetype[sr->numopsbegin] = REDUCE_SUM;
  sr->invecs[sr->numopsbegin]     = x;
  ierr = VecDot_Seq(x,y,sr->lvalues+sr->numopsbegin++);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
     VecDotEnd - Ends a split phase dot product

  Input Parameters:
+   x - the first vector (can be PETSC_NULL)
.   y - the second vector (can be PETSC_NULL)
-   result - where the result will go

@*/
int VecDotEnd(Vec x, Vec y,Scalar *result) 
{
  int               ierr;
  VecSplitReduction *sr;

  PetscFunctionBegin;
  ierr = VecSplitReductionGet(x,&sr);CHKERRQ(ierr);
  if (sr->numopsend >= sr->numopsbegin) 
    SETERRQ(1,1,"Vectors communicator involved in a reduction operation that was completed\n\
                 but not yet read");

  if (sr->numopsbegin >= sr->maxops) {
    ierr = VecSplitReductionExtend(sr);CHKERRQ(ierr);
  }
  sr->reducetype[sr->numopsbegin] = REDUCE_SUM;
  sr->invecs[sr->numopsbegin]     = x;
  ierr = VecDot_Seq(x,y,sr->lvalues+sr->numopsbegin++);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

