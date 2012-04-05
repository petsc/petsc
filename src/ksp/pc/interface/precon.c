
/*
    The PC (preconditioner) interface routines, callable by users.
*/
#include <petsc-private/pcimpl.h>            /*I "petscksp.h" I*/

/* Logging support */
PetscClassId   PC_CLASSID;
PetscLogEvent  PC_SetUp, PC_SetUpOnBlocks, PC_Apply, PC_ApplyCoarse, PC_ApplyMultiple, PC_ApplySymmetricLeft;
PetscLogEvent  PC_ApplySymmetricRight, PC_ModifySubMatrices, PC_ApplyOnBlocks, PC_ApplyTransposeOnBlocks, PC_ApplyOnMproc;

#undef __FUNCT__  
#define __FUNCT__ "PCGetDefaultType_Private"
PetscErrorCode PCGetDefaultType_Private(PC pc,const char* type[])
{
  PetscErrorCode ierr;
  PetscMPIInt    size;
  PetscBool      flg1,flg2,set,flg3;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(((PetscObject)pc)->comm,&size);CHKERRQ(ierr);
  if (pc->pmat) {
    PetscErrorCode (*f)(Mat,MatReuse,Mat*);
    ierr = PetscObjectQueryFunction((PetscObject)pc->pmat,"MatGetDiagonalBlock_C",(void (**)(void))&f);CHKERRQ(ierr);
    if (size == 1) {
      ierr = MatGetFactorAvailable(pc->pmat,"petsc",MAT_FACTOR_ICC,&flg1);CHKERRQ(ierr);
      ierr = MatGetFactorAvailable(pc->pmat,"petsc",MAT_FACTOR_ILU,&flg2);CHKERRQ(ierr);
      ierr = MatIsSymmetricKnown(pc->pmat,&set,&flg3);CHKERRQ(ierr);
      if (flg1 && (!flg2 || (set && flg3))) {
	*type = PCICC;
      } else if (flg2) {
	*type = PCILU;
      } else if (f) { /* likely is a parallel matrix run on one processor */
	*type = PCBJACOBI;
      } else {
	*type = PCNONE;
      }
    } else {
       if (f) {
	*type = PCBJACOBI;
      } else {
	*type = PCNONE;
      }
    }
  } else {
    if (size == 1) {
      *type = PCILU;
    } else {
      *type = PCBJACOBI;
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCReset"
/*@
   PCReset - Resets a PC context to the pcsetupcalled = 0 state and removes any allocated Vecs and Mats

   Collective on PC

   Input Parameter:
.  pc - the preconditioner context

   Level: developer

   Notes: This allows a PC to be reused for a different sized linear system but using the same options that have been previously set in the PC

.keywords: PC, destroy

.seealso: PCCreate(), PCSetUp()
@*/
PetscErrorCode  PCReset(PC pc)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  if (pc->ops->reset) {
    ierr = (*pc->ops->reset)(pc);CHKERRQ(ierr);
  }
  ierr = VecDestroy(&pc->diagonalscaleright);CHKERRQ(ierr);
  ierr = VecDestroy(&pc->diagonalscaleleft);CHKERRQ(ierr);
  ierr = MatDestroy(&pc->pmat);CHKERRQ(ierr);
  ierr = MatDestroy(&pc->mat);CHKERRQ(ierr);
  pc->setupcalled = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCDestroy"
/*@
   PCDestroy - Destroys PC context that was created with PCCreate().

   Collective on PC

   Input Parameter:
.  pc - the preconditioner context

   Level: developer

.keywords: PC, destroy

.seealso: PCCreate(), PCSetUp()
@*/
PetscErrorCode  PCDestroy(PC *pc)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*pc) PetscFunctionReturn(0);
  PetscValidHeaderSpecific((*pc),PC_CLASSID,1);
  if (--((PetscObject)(*pc))->refct > 0) {*pc = 0; PetscFunctionReturn(0);}

  ierr = PCReset(*pc);CHKERRQ(ierr);

  /* if memory was published with AMS then destroy it */
  ierr = PetscObjectDepublish((*pc));CHKERRQ(ierr);
  if ((*pc)->ops->destroy) {ierr = (*(*pc)->ops->destroy)((*pc));CHKERRQ(ierr);}
  ierr = DMDestroy(&(*pc)->dm);CHKERRQ(ierr);
  ierr = PetscHeaderDestroy(pc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCGetDiagonalScale"
/*@C
   PCGetDiagonalScale - Indicates if the preconditioner applies an additional left and right
      scaling as needed by certain time-stepping codes.

   Logically Collective on PC

   Input Parameter:
.  pc - the preconditioner context

   Output Parameter:
.  flag - PETSC_TRUE if it applies the scaling

   Level: developer

   Notes: If this returns PETSC_TRUE then the system solved via the Krylov method is
$           D M A D^{-1} y = D M b  for left preconditioning or
$           D A M D^{-1} z = D b for right preconditioning

.keywords: PC

.seealso: PCCreate(), PCSetUp(), PCDiagonalScaleLeft(), PCDiagonalScaleRight(), PCSetDiagonalScale()
@*/
PetscErrorCode  PCGetDiagonalScale(PC pc,PetscBool  *flag)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidPointer(flag,2);
  *flag = pc->diagonalscale;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCSetDiagonalScale"
/*@
   PCSetDiagonalScale - Indicates the left scaling to use to apply an additional left and right
      scaling as needed by certain time-stepping codes.

   Logically Collective on PC

   Input Parameters:
+  pc - the preconditioner context
-  s - scaling vector

   Level: intermediate

   Notes: The system solved via the Krylov method is
$           D M A D^{-1} y = D M b  for left preconditioning or
$           D A M D^{-1} z = D b for right preconditioning

   PCDiagonalScaleLeft() scales a vector by D. PCDiagonalScaleRight() scales a vector by D^{-1}.

.keywords: PC

.seealso: PCCreate(), PCSetUp(), PCDiagonalScaleLeft(), PCDiagonalScaleRight(), PCGetDiagonalScale()
@*/
PetscErrorCode  PCSetDiagonalScale(PC pc,Vec s)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidHeaderSpecific(s,VEC_CLASSID,2);
  pc->diagonalscale     = PETSC_TRUE;
  ierr = PetscObjectReference((PetscObject)s);CHKERRQ(ierr);
  ierr = VecDestroy(&pc->diagonalscaleleft);CHKERRQ(ierr);
  pc->diagonalscaleleft = s;
  ierr = VecDuplicate(s,&pc->diagonalscaleright);CHKERRQ(ierr);
  ierr = VecCopy(s,pc->diagonalscaleright);CHKERRQ(ierr);
  ierr = VecReciprocal(pc->diagonalscaleright);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCDiagonalScaleLeft"
/*@
   PCDiagonalScaleLeft - Scales a vector by the left scaling as needed by certain time-stepping codes.

   Logically Collective on PC

   Input Parameters:
+  pc - the preconditioner context
.  in - input vector
+  out - scaled vector (maybe the same as in)

   Level: intermediate

   Notes: The system solved via the Krylov method is
$           D M A D^{-1} y = D M b  for left preconditioning or
$           D A M D^{-1} z = D b for right preconditioning

   PCDiagonalScaleLeft() scales a vector by D. PCDiagonalScaleRight() scales a vector by D^{-1}.

   If diagonal scaling is turned off and in is not out then in is copied to out

.keywords: PC

.seealso: PCCreate(), PCSetUp(), PCDiagonalScaleSet(), PCDiagonalScaleRight(), PCDiagonalScale()
@*/
PetscErrorCode  PCDiagonalScaleLeft(PC pc,Vec in,Vec out)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidHeaderSpecific(in,VEC_CLASSID,2);
  PetscValidHeaderSpecific(out,VEC_CLASSID,3);
  if (pc->diagonalscale) {
    ierr = VecPointwiseMult(out,pc->diagonalscaleleft,in);CHKERRQ(ierr);
  } else if (in != out) {
    ierr = VecCopy(in,out);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCDiagonalScaleRight"
/*@
   PCDiagonalScaleRight - Scales a vector by the right scaling as needed by certain time-stepping codes.

   Logically Collective on PC

   Input Parameters:
+  pc - the preconditioner context
.  in - input vector
+  out - scaled vector (maybe the same as in)

   Level: intermediate

   Notes: The system solved via the Krylov method is
$           D M A D^{-1} y = D M b  for left preconditioning or
$           D A M D^{-1} z = D b for right preconditioning

   PCDiagonalScaleLeft() scales a vector by D. PCDiagonalScaleRight() scales a vector by D^{-1}.

   If diagonal scaling is turned off and in is not out then in is copied to out

.keywords: PC

.seealso: PCCreate(), PCSetUp(), PCDiagonalScaleLeft(), PCDiagonalScaleSet(), PCDiagonalScale()
@*/
PetscErrorCode  PCDiagonalScaleRight(PC pc,Vec in,Vec out)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidHeaderSpecific(in,VEC_CLASSID,2);
  PetscValidHeaderSpecific(out,VEC_CLASSID,3);
  if (pc->diagonalscale) {
    ierr = VecPointwiseMult(out,pc->diagonalscaleright,in);CHKERRQ(ierr);
  } else if (in != out) {
    ierr = VecCopy(in,out);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#if 0
#undef __FUNCT__  
#define __FUNCT__ "PCPublish_Petsc"
static PetscErrorCode PCPublish_Petsc(PetscObject obj)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}
#endif

#undef __FUNCT__  
#define __FUNCT__ "PCCreate"
/*@
   PCCreate - Creates a preconditioner context.

   Collective on MPI_Comm

   Input Parameter:
.  comm - MPI communicator 

   Output Parameter:
.  pc - location to put the preconditioner context

   Notes:
   The default preconditioner for sparse matrices is PCILU or PCICC with 0 fill on one process and block Jacobi with PCILU or ICC 
   in parallel. For dense matrices it is always PCNONE.

   Level: developer

.keywords: PC, create, context

.seealso: PCSetUp(), PCApply(), PCDestroy()
@*/
PetscErrorCode  PCCreate(MPI_Comm comm,PC *newpc)
{
  PC             pc;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(newpc,1);
  *newpc = 0;
#ifndef PETSC_USE_DYNAMIC_LIBRARIES
  ierr = PCInitializePackage(PETSC_NULL);CHKERRQ(ierr);
#endif

  ierr = PetscHeaderCreate(pc,_p_PC,struct _PCOps,PC_CLASSID,-1,"PC","Preconditioner","PC",comm,PCDestroy,PCView);CHKERRQ(ierr);

  pc->mat                  = 0;
  pc->pmat                 = 0;
  pc->setupcalled          = 0;
  pc->setfromoptionscalled = 0;
  pc->data                 = 0;
  pc->diagonalscale        = PETSC_FALSE;
  pc->diagonalscaleleft    = 0;
  pc->diagonalscaleright   = 0;
  pc->reuse                = 0;

  pc->modifysubmatrices   = 0;
  pc->modifysubmatricesP  = 0;
  *newpc = pc;
  PetscFunctionReturn(0);

}

/* -------------------------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "PCApply"
/*@
   PCApply - Applies the preconditioner to a vector.

   Collective on PC and Vec

   Input Parameters:
+  pc - the preconditioner context
-  x - input vector

   Output Parameter:
.  y - output vector

   Level: developer

.keywords: PC, apply

.seealso: PCApplyTranspose(), PCApplyBAorAB()
@*/
PetscErrorCode  PCApply(PC pc,Vec x,Vec y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,2);
  PetscValidHeaderSpecific(y,VEC_CLASSID,3);
  if (x == y) SETERRQ(((PetscObject)pc)->comm,PETSC_ERR_ARG_IDN,"x and y must be different vectors");
  VecValidValues(x,2,PETSC_TRUE);
  if (pc->setupcalled < 2) {
    ierr = PCSetUp(pc);CHKERRQ(ierr);
  }
  if (!pc->ops->apply) SETERRQ(((PetscObject)pc)->comm,PETSC_ERR_SUP,"PC does not have apply");
  ierr = PetscLogEventBegin(PC_Apply,pc,x,y,0);CHKERRQ(ierr);
  ierr = (*pc->ops->apply)(pc,x,y);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(PC_Apply,pc,x,y,0);CHKERRQ(ierr);
  VecValidValues(y,3,PETSC_FALSE);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCApplySymmetricLeft"
/*@
   PCApplySymmetricLeft - Applies the left part of a symmetric preconditioner to a vector.

   Collective on PC and Vec

   Input Parameters:
+  pc - the preconditioner context
-  x - input vector

   Output Parameter:
.  y - output vector

   Notes:
   Currently, this routine is implemented only for PCICC and PCJACOBI preconditioners.

   Level: developer

.keywords: PC, apply, symmetric, left

.seealso: PCApply(), PCApplySymmetricRight()
@*/
PetscErrorCode  PCApplySymmetricLeft(PC pc,Vec x,Vec y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,2);
  PetscValidHeaderSpecific(y,VEC_CLASSID,3);
  if (x == y) SETERRQ(((PetscObject)pc)->comm,PETSC_ERR_ARG_IDN,"x and y must be different vectors");
  VecValidValues(x,2,PETSC_TRUE);
  if (pc->setupcalled < 2) {
    ierr = PCSetUp(pc);CHKERRQ(ierr);
  }
  if (!pc->ops->applysymmetricleft) SETERRQ(((PetscObject)pc)->comm,PETSC_ERR_SUP,"PC does not have left symmetric apply");
  ierr = PetscLogEventBegin(PC_ApplySymmetricLeft,pc,x,y,0);CHKERRQ(ierr);
  ierr = (*pc->ops->applysymmetricleft)(pc,x,y);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(PC_ApplySymmetricLeft,pc,x,y,0);CHKERRQ(ierr);
  VecValidValues(y,3,PETSC_FALSE);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCApplySymmetricRight"
/*@
   PCApplySymmetricRight - Applies the right part of a symmetric preconditioner to a vector.

   Collective on PC and Vec

   Input Parameters:
+  pc - the preconditioner context
-  x - input vector

   Output Parameter:
.  y - output vector

   Level: developer

   Notes:
   Currently, this routine is implemented only for PCICC and PCJACOBI preconditioners.

.keywords: PC, apply, symmetric, right

.seealso: PCApply(), PCApplySymmetricLeft()
@*/
PetscErrorCode  PCApplySymmetricRight(PC pc,Vec x,Vec y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,2);
  PetscValidHeaderSpecific(y,VEC_CLASSID,3);
  if (x == y) SETERRQ(((PetscObject)pc)->comm,PETSC_ERR_ARG_IDN,"x and y must be different vectors");
  VecValidValues(x,2,PETSC_TRUE);
  if (pc->setupcalled < 2) {
    ierr = PCSetUp(pc);CHKERRQ(ierr);
  }
  if (!pc->ops->applysymmetricright) SETERRQ(((PetscObject)pc)->comm,PETSC_ERR_SUP,"PC does not have left symmetric apply");
  ierr = PetscLogEventBegin(PC_ApplySymmetricRight,pc,x,y,0);CHKERRQ(ierr);
  ierr = (*pc->ops->applysymmetricright)(pc,x,y);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(PC_ApplySymmetricRight,pc,x,y,0);CHKERRQ(ierr);
  VecValidValues(y,3,PETSC_FALSE);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCApplyTranspose"
/*@
   PCApplyTranspose - Applies the transpose of preconditioner to a vector.

   Collective on PC and Vec

   Input Parameters:
+  pc - the preconditioner context
-  x - input vector

   Output Parameter:
.  y - output vector

   Notes: For complex numbers this applies the non-Hermitian transpose.

   Developer Notes: We need to implement a PCApplyHermitianTranspose()

   Level: developer

.keywords: PC, apply, transpose

.seealso: PCApply(), PCApplyBAorAB(), PCApplyBAorABTranspose(), PCApplyTransposeExists()
@*/
PetscErrorCode  PCApplyTranspose(PC pc,Vec x,Vec y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,2);
  PetscValidHeaderSpecific(y,VEC_CLASSID,3);
  if (x == y) SETERRQ(((PetscObject)pc)->comm,PETSC_ERR_ARG_IDN,"x and y must be different vectors");
  VecValidValues(x,2,PETSC_TRUE);
  if (pc->setupcalled < 2) {
    ierr = PCSetUp(pc);CHKERRQ(ierr);
  }
  if (!pc->ops->applytranspose) SETERRQ(((PetscObject)pc)->comm,PETSC_ERR_SUP,"PC does not have apply transpose");
  ierr = PetscLogEventBegin(PC_Apply,pc,x,y,0);CHKERRQ(ierr);
  ierr = (*pc->ops->applytranspose)(pc,x,y);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(PC_Apply,pc,x,y,0);CHKERRQ(ierr);
  VecValidValues(y,3,PETSC_FALSE);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCApplyTransposeExists"
/*@
   PCApplyTransposeExists - Test whether the preconditioner has a transpose apply operation

   Collective on PC and Vec

   Input Parameters:
.  pc - the preconditioner context

   Output Parameter:
.  flg - PETSC_TRUE if a transpose operation is defined

   Level: developer

.keywords: PC, apply, transpose

.seealso: PCApplyTranspose()
@*/
PetscErrorCode  PCApplyTransposeExists(PC pc,PetscBool  *flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidPointer(flg,2);
  if (pc->ops->applytranspose) *flg = PETSC_TRUE;
  else                         *flg = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCApplyBAorAB"
/*@
   PCApplyBAorAB - Applies the preconditioner and operator to a vector. y = B*A*x or y = A*B*x.

   Collective on PC and Vec

   Input Parameters:
+  pc - the preconditioner context
.  side - indicates the preconditioner side, one of PC_LEFT, PC_RIGHT, or PC_SYMMETRIC
.  x - input vector
-  work - work vector

   Output Parameter:
.  y - output vector

   Level: developer

   Notes: If the PC has had PCSetDiagonalScale() set then D M A D^{-1} for left preconditioning or  D A M D^{-1} is actually applied. Note that the 
   specific KSPSolve() method must also be written to handle the post-solve "correction" for the diagonal scaling.

.keywords: PC, apply, operator

.seealso: PCApply(), PCApplyTranspose(), PCApplyBAorABTranspose()
@*/
PetscErrorCode  PCApplyBAorAB(PC pc,PCSide side,Vec x,Vec y,Vec work)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,3);
  PetscValidHeaderSpecific(y,VEC_CLASSID,4);
  PetscValidHeaderSpecific(work,VEC_CLASSID,5);
  if (x == y) SETERRQ(((PetscObject)pc)->comm,PETSC_ERR_ARG_IDN,"x and y must be different vectors");
  VecValidValues(x,3,PETSC_TRUE);
  if (side != PC_LEFT && side != PC_SYMMETRIC && side != PC_RIGHT) SETERRQ(((PetscObject)pc)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Side must be right, left, or symmetric");
  if (pc->diagonalscale && side == PC_SYMMETRIC) SETERRQ(((PetscObject)pc)->comm,PETSC_ERR_SUP,"Cannot include diagonal scaling with symmetric preconditioner application");

  if (pc->setupcalled < 2) {
    ierr = PCSetUp(pc);CHKERRQ(ierr);
  }

  if (pc->diagonalscale) {
    if (pc->ops->applyBA) {
      Vec work2; /* this is expensive, but to fix requires a second work vector argument to PCApplyBAorAB() */
      ierr = VecDuplicate(x,&work2);CHKERRQ(ierr);
      ierr = PCDiagonalScaleRight(pc,x,work2);CHKERRQ(ierr);
      ierr = (*pc->ops->applyBA)(pc,side,work2,y,work);CHKERRQ(ierr);
      ierr = PCDiagonalScaleLeft(pc,y,y);CHKERRQ(ierr);
      ierr = VecDestroy(&work2);CHKERRQ(ierr);
    } else if (side == PC_RIGHT) {
      ierr = PCDiagonalScaleRight(pc,x,y);CHKERRQ(ierr);
      ierr = PCApply(pc,y,work);CHKERRQ(ierr);
      ierr = MatMult(pc->mat,work,y);CHKERRQ(ierr);
      ierr = PCDiagonalScaleLeft(pc,y,y);CHKERRQ(ierr);
    } else if (side == PC_LEFT) {
      ierr = PCDiagonalScaleRight(pc,x,y);CHKERRQ(ierr);
      ierr = MatMult(pc->mat,y,work);CHKERRQ(ierr);
      ierr = PCApply(pc,work,y);CHKERRQ(ierr);
      ierr = PCDiagonalScaleLeft(pc,y,y);CHKERRQ(ierr);
    } else if (side == PC_SYMMETRIC) SETERRQ(((PetscObject)pc)->comm,PETSC_ERR_SUP,"Cannot provide diagonal scaling with symmetric application of preconditioner");
  } else {
    if (pc->ops->applyBA) {
      ierr = (*pc->ops->applyBA)(pc,side,x,y,work);CHKERRQ(ierr);
    } else if (side == PC_RIGHT) {
      ierr = PCApply(pc,x,work);CHKERRQ(ierr);
      ierr = MatMult(pc->mat,work,y);CHKERRQ(ierr);
    } else if (side == PC_LEFT) {
      ierr = MatMult(pc->mat,x,work);CHKERRQ(ierr);
      ierr = PCApply(pc,work,y);CHKERRQ(ierr);
    } else if (side == PC_SYMMETRIC) {
      /* There's an extra copy here; maybe should provide 2 work vectors instead? */
      ierr = PCApplySymmetricRight(pc,x,work);CHKERRQ(ierr);
      ierr = MatMult(pc->mat,work,y);CHKERRQ(ierr);
      ierr = VecCopy(y,work);CHKERRQ(ierr);
      ierr = PCApplySymmetricLeft(pc,work,y);CHKERRQ(ierr);
    }
  }
  VecValidValues(y,4,PETSC_FALSE);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCApplyBAorABTranspose"
/*@ 
   PCApplyBAorABTranspose - Applies the transpose of the preconditioner
   and operator to a vector. That is, applies tr(B) * tr(A) with left preconditioning,
   NOT tr(B*A) = tr(A)*tr(B).

   Collective on PC and Vec

   Input Parameters:
+  pc - the preconditioner context
.  side - indicates the preconditioner side, one of PC_LEFT, PC_RIGHT, or PC_SYMMETRIC
.  x - input vector
-  work - work vector

   Output Parameter:
.  y - output vector


   Notes: this routine is used internally so that the same Krylov code can be used to solve A x = b and A' x = b, with a preconditioner
      defined by B'. This is why this has the funny form that it computes tr(B) * tr(A) 
          
    Level: developer

.keywords: PC, apply, operator, transpose

.seealso: PCApply(), PCApplyTranspose(), PCApplyBAorAB()
@*/
PetscErrorCode  PCApplyBAorABTranspose(PC pc,PCSide side,Vec x,Vec y,Vec work)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,3);
  PetscValidHeaderSpecific(y,VEC_CLASSID,4);
  PetscValidHeaderSpecific(work,VEC_CLASSID,5);
  if (x == y) SETERRQ(((PetscObject)pc)->comm,PETSC_ERR_ARG_IDN,"x and y must be different vectors");
  VecValidValues(x,3,PETSC_TRUE);
  if (pc->ops->applyBAtranspose) {
    ierr = (*pc->ops->applyBAtranspose)(pc,side,x,y,work);CHKERRQ(ierr);
    VecValidValues(y,4,PETSC_FALSE);
    PetscFunctionReturn(0);
  }
  if (side != PC_LEFT && side != PC_RIGHT) SETERRQ(((PetscObject)pc)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Side must be right or left");

  if (pc->setupcalled < 2) {
    ierr = PCSetUp(pc);CHKERRQ(ierr);
  }

  if (side == PC_RIGHT) {
    ierr = PCApplyTranspose(pc,x,work);CHKERRQ(ierr);
    ierr = MatMultTranspose(pc->mat,work,y);CHKERRQ(ierr);
  } else if (side == PC_LEFT) {
    ierr = MatMultTranspose(pc->mat,x,work);CHKERRQ(ierr);
    ierr = PCApplyTranspose(pc,work,y);CHKERRQ(ierr);
  }
  /* add support for PC_SYMMETRIC */
  VecValidValues(y,4,PETSC_FALSE);
  PetscFunctionReturn(0); 
}

/* -------------------------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "PCApplyRichardsonExists"
/*@
   PCApplyRichardsonExists - Determines whether a particular preconditioner has a 
   built-in fast application of Richardson's method.

   Not Collective

   Input Parameter:
.  pc - the preconditioner

   Output Parameter:
.  exists - PETSC_TRUE or PETSC_FALSE

   Level: developer

.keywords: PC, apply, Richardson, exists

.seealso: PCApplyRichardson()
@*/
PetscErrorCode  PCApplyRichardsonExists(PC pc,PetscBool  *exists)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidIntPointer(exists,2);
  if (pc->ops->applyrichardson) *exists = PETSC_TRUE; 
  else                          *exists = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCApplyRichardson"
/*@
   PCApplyRichardson - Applies several steps of Richardson iteration with 
   the particular preconditioner. This routine is usually used by the 
   Krylov solvers and not the application code directly.

   Collective on PC

   Input Parameters:
+  pc  - the preconditioner context
.  b   - the right hand side
.  w   - one work vector
.  rtol - relative decrease in residual norm convergence criteria
.  abstol - absolute residual norm convergence criteria
.  dtol - divergence residual norm increase criteria
.  its - the number of iterations to apply.
-  guesszero - if the input x contains nonzero initial guess

   Output Parameter:
+  outits - number of iterations actually used (for SOR this always equals its)
.  reason - the reason the apply terminated
-  y - the solution (also contains initial guess if guesszero is PETSC_FALSE

   Notes: 
   Most preconditioners do not support this function. Use the command
   PCApplyRichardsonExists() to determine if one does.

   Except for the multigrid PC this routine ignores the convergence tolerances
   and always runs for the number of iterations
 
   Level: developer

.keywords: PC, apply, Richardson

.seealso: PCApplyRichardsonExists()
@*/
PetscErrorCode  PCApplyRichardson(PC pc,Vec b,Vec y,Vec w,PetscReal rtol,PetscReal abstol, PetscReal dtol,PetscInt its,PetscBool  guesszero,PetscInt *outits,PCRichardsonConvergedReason *reason)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidHeaderSpecific(b,VEC_CLASSID,2);
  PetscValidHeaderSpecific(y,VEC_CLASSID,3);
  PetscValidHeaderSpecific(w,VEC_CLASSID,4);
  if (b == y) SETERRQ(((PetscObject)pc)->comm,PETSC_ERR_ARG_IDN,"b and y must be different vectors");
  if (pc->setupcalled < 2) {
    ierr = PCSetUp(pc);CHKERRQ(ierr);
  }
  if (!pc->ops->applyrichardson) SETERRQ(((PetscObject)pc)->comm,PETSC_ERR_SUP,"PC does not have apply richardson");
  ierr = (*pc->ops->applyrichardson)(pc,b,y,w,rtol,abstol,dtol,its,guesszero,outits,reason);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* 
      a setupcall of 0 indicates never setup, 
                     1 needs to be resetup,
                     2 does not need any changes.
*/
#undef __FUNCT__  
#define __FUNCT__ "PCSetUp"
/*@
   PCSetUp - Prepares for the use of a preconditioner.

   Collective on PC

   Input Parameter:
.  pc - the preconditioner context

   Level: developer

.keywords: PC, setup

.seealso: PCCreate(), PCApply(), PCDestroy()
@*/
PetscErrorCode  PCSetUp(PC pc)
{
  PetscErrorCode ierr;
  const char     *def;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  if (!pc->mat) SETERRQ(((PetscObject)pc)->comm,PETSC_ERR_ARG_WRONGSTATE,"Matrix must be set first");

  if (pc->setupcalled > 1) {
    ierr = PetscInfo(pc,"Setting PC with identical preconditioner\n");CHKERRQ(ierr);
    PetscFunctionReturn(0);
  } else if (!pc->setupcalled) {
    ierr = PetscInfo(pc,"Setting up new PC\n");CHKERRQ(ierr);
  } else if (pc->flag == SAME_NONZERO_PATTERN) {
    ierr = PetscInfo(pc,"Setting up PC with same nonzero pattern\n");CHKERRQ(ierr);
  } else {
    ierr = PetscInfo(pc,"Setting up PC with different nonzero pattern\n");CHKERRQ(ierr);
  }

  if (!((PetscObject)pc)->type_name) {
    ierr = PCGetDefaultType_Private(pc,&def);CHKERRQ(ierr);
    ierr = PCSetType(pc,def);CHKERRQ(ierr);
  }

  ierr = PetscLogEventBegin(PC_SetUp,pc,0,0,0);CHKERRQ(ierr);
  if (pc->ops->setup) {
    ierr = (*pc->ops->setup)(pc);CHKERRQ(ierr);
  }
  pc->setupcalled = 2;
  ierr = PetscLogEventEnd(PC_SetUp,pc,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCSetUpOnBlocks"
/*@
   PCSetUpOnBlocks - Sets up the preconditioner for each block in
   the block Jacobi, block Gauss-Seidel, and overlapping Schwarz 
   methods.

   Collective on PC

   Input Parameters:
.  pc - the preconditioner context

   Level: developer

.keywords: PC, setup, blocks

.seealso: PCCreate(), PCApply(), PCDestroy(), PCSetUp()
@*/
PetscErrorCode  PCSetUpOnBlocks(PC pc)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  if (!pc->ops->setuponblocks) PetscFunctionReturn(0);
  ierr = PetscLogEventBegin(PC_SetUpOnBlocks,pc,0,0,0);CHKERRQ(ierr);
  ierr = (*pc->ops->setuponblocks)(pc);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(PC_SetUpOnBlocks,pc,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCSetModifySubMatrices"
/*@C
   PCSetModifySubMatrices - Sets a user-defined routine for modifying the
   submatrices that arise within certain subdomain-based preconditioners.
   The basic submatrices are extracted from the preconditioner matrix as
   usual; the user can then alter these (for example, to set different boundary
   conditions for each submatrix) before they are used for the local solves.

   Logically Collective on PC

   Input Parameters:
+  pc - the preconditioner context
.  func - routine for modifying the submatrices
-  ctx - optional user-defined context (may be null)

   Calling sequence of func:
$     func (PC pc,PetscInt nsub,IS *row,IS *col,Mat *submat,void *ctx);

.  row - an array of index sets that contain the global row numbers
         that comprise each local submatrix
.  col - an array of index sets that contain the global column numbers
         that comprise each local submatrix
.  submat - array of local submatrices
-  ctx - optional user-defined context for private data for the 
         user-defined func routine (may be null)

   Notes:
   PCSetModifySubMatrices() MUST be called before KSPSetUp() and
   KSPSolve().

   A routine set by PCSetModifySubMatrices() is currently called within
   the block Jacobi (PCBJACOBI) and additive Schwarz (PCASM)
   preconditioners.  All other preconditioners ignore this routine.

   Level: advanced

.keywords: PC, set, modify, submatrices

.seealso: PCModifySubMatrices(), PCASMGetSubMatrices()
@*/
PetscErrorCode  PCSetModifySubMatrices(PC pc,PetscErrorCode (*func)(PC,PetscInt,const IS[],const IS[],Mat[],void*),void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  pc->modifysubmatrices  = func;
  pc->modifysubmatricesP = ctx;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCModifySubMatrices"
/*@C
   PCModifySubMatrices - Calls an optional user-defined routine within 
   certain preconditioners if one has been set with PCSetModifySubMarices().

   Collective on PC

   Input Parameters:
+  pc - the preconditioner context
.  nsub - the number of local submatrices
.  row - an array of index sets that contain the global row numbers
         that comprise each local submatrix
.  col - an array of index sets that contain the global column numbers
         that comprise each local submatrix
.  submat - array of local submatrices
-  ctx - optional user-defined context for private data for the 
         user-defined routine (may be null)

   Output Parameter:
.  submat - array of local submatrices (the entries of which may
            have been modified)

   Notes:
   The user should NOT generally call this routine, as it will
   automatically be called within certain preconditioners (currently
   block Jacobi, additive Schwarz) if set.

   The basic submatrices are extracted from the preconditioner matrix
   as usual; the user can then alter these (for example, to set different
   boundary conditions for each submatrix) before they are used for the
   local solves.

   Level: developer

.keywords: PC, modify, submatrices

.seealso: PCSetModifySubMatrices()
@*/
PetscErrorCode  PCModifySubMatrices(PC pc,PetscInt nsub,const IS row[],const IS col[],Mat submat[],void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  if (!pc->modifysubmatrices) PetscFunctionReturn(0);
  ierr = PetscLogEventBegin(PC_ModifySubMatrices,pc,0,0,0);CHKERRQ(ierr);
  ierr = (*pc->modifysubmatrices)(pc,nsub,row,col,submat,ctx);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(PC_ModifySubMatrices,pc,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCSetOperators"
/*@
   PCSetOperators - Sets the matrix associated with the linear system and 
   a (possibly) different one associated with the preconditioner.

   Logically Collective on PC and Mat

   Input Parameters:
+  pc - the preconditioner context
.  Amat - the matrix associated with the linear system
.  Pmat - the matrix to be used in constructing the preconditioner, usually
          the same as Amat. 
-  flag - flag indicating information about the preconditioner matrix structure
   during successive linear solves.  This flag is ignored the first time a
   linear system is solved, and thus is irrelevant when solving just one linear
   system.

   Notes: 
   The flag can be used to eliminate unnecessary work in the preconditioner 
   during the repeated solution of linear systems of the same size.  The 
   available options are
+    SAME_PRECONDITIONER -
       Pmat is identical during successive linear solves.
       This option is intended for folks who are using
       different Amat and Pmat matrices and wish to reuse the
       same preconditioner matrix.  For example, this option
       saves work by not recomputing incomplete factorization
       for ILU/ICC preconditioners.
.     SAME_NONZERO_PATTERN -
       Pmat has the same nonzero structure during
       successive linear solves. 
-     DIFFERENT_NONZERO_PATTERN -
       Pmat does not have the same nonzero structure.

    Passing a PETSC_NULL for Amat or Pmat removes the matrix that is currently used.

    If you wish to replace either Amat or Pmat but leave the other one untouched then
    first call KSPGetOperators() to get the one you wish to keep, call PetscObjectReference()
    on it and then pass it back in in your call to KSPSetOperators().

   Caution:
   If you specify SAME_NONZERO_PATTERN, PETSc believes your assertion
   and does not check the structure of the matrix.  If you erroneously
   claim that the structure is the same when it actually is not, the new
   preconditioner will not function correctly.  Thus, use this optimization
   feature carefully!

   If in doubt about whether your preconditioner matrix has changed
   structure or not, use the flag DIFFERENT_NONZERO_PATTERN.

   More Notes about Repeated Solution of Linear Systems:
   PETSc does NOT reset the matrix entries of either Amat or Pmat
   to zero after a linear solve; the user is completely responsible for
   matrix assembly.  See the routine MatZeroEntries() if desiring to
   zero all elements of a matrix.

   Level: intermediate

.keywords: PC, set, operators, matrix, linear system

.seealso: PCGetOperators(), MatZeroEntries()
 @*/
PetscErrorCode  PCSetOperators(PC pc,Mat Amat,Mat Pmat,MatStructure flag)
{
  PetscErrorCode ierr;
  PetscInt       m1,n1,m2,n2;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  if (Amat) PetscValidHeaderSpecific(Amat,MAT_CLASSID,2);
  if (Pmat) PetscValidHeaderSpecific(Pmat,MAT_CLASSID,3);
  if (Amat) PetscCheckSameComm(pc,1,Amat,2);
  if (Pmat) PetscCheckSameComm(pc,1,Pmat,3);
  if (pc->setupcalled && Amat && Pmat) {
    ierr = MatGetLocalSize(Amat,&m1,&n1);CHKERRQ(ierr);
    ierr = MatGetLocalSize(pc->mat,&m2,&n2);CHKERRQ(ierr);
    if (m1 != m2 || n1 != n2) SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Cannot change local size of Amat after use old sizes %D %D new sizes %D %D",m2,n2,m1,n1);
    ierr = MatGetLocalSize(Pmat,&m1,&n1);CHKERRQ(ierr);
    ierr = MatGetLocalSize(pc->pmat,&m2,&n2);CHKERRQ(ierr);
    if (m1 != m2 || n1 != n2) SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Cannot change local size of Pmat after use old sizes %D %D new sizes %D %D",m2,n2,m1,n1);
  }

  /* reference first in case the matrices are the same */
  if (Amat) {ierr = PetscObjectReference((PetscObject)Amat);CHKERRQ(ierr);}
  ierr = MatDestroy(&pc->mat);CHKERRQ(ierr);
  if (Pmat) {ierr = PetscObjectReference((PetscObject)Pmat);CHKERRQ(ierr);}
  ierr = MatDestroy(&pc->pmat);CHKERRQ(ierr);
  pc->mat  = Amat;
  pc->pmat = Pmat;

  if (pc->setupcalled == 2 && flag != SAME_PRECONDITIONER) {
    pc->setupcalled = 1;
  }
  pc->flag = flag;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCGetOperators"
/*@C
   PCGetOperators - Gets the matrix associated with the linear system and
   possibly a different one associated with the preconditioner.

   Not collective, though parallel Mats are returned if the PC is parallel

   Input Parameter:
.  pc - the preconditioner context

   Output Parameters:
+  mat - the matrix associated with the linear system
.  pmat - matrix associated with the preconditioner, usually the same
          as mat. 
-  flag - flag indicating information about the preconditioner
          matrix structure.  See PCSetOperators() for details.

   Level: intermediate

   Notes: Does not increase the reference count of the matrices, so you should not destroy them

   Alternative usage: If the operators have NOT been set with KSP/PCSetOperators() then the operators
      are created in PC and returned to the user. In this case, if both operators
      mat and pmat are requested, two DIFFERENT operators will be returned. If
      only one is requested both operators in the PC will be the same (i.e. as
      if one had called KSP/PCSetOperators() with the same argument for both Mats).
      The user must set the sizes of the returned matrices and their type etc just
      as if the user created them with MatCreate(). For example,

$         KSP/PCGetOperators(ksp/pc,&mat,PETSC_NULL,PETSC_NULL); is equivalent to
$           set size, type, etc of mat

$         MatCreate(comm,&mat);
$         KSP/PCSetOperators(ksp/pc,mat,mat,SAME_NONZERO_PATTERN);
$         PetscObjectDereference((PetscObject)mat);
$           set size, type, etc of mat

     and

$         KSP/PCGetOperators(ksp/pc,&mat,&pmat,PETSC_NULL); is equivalent to
$           set size, type, etc of mat and pmat

$         MatCreate(comm,&mat);
$         MatCreate(comm,&pmat);
$         KSP/PCSetOperators(ksp/pc,mat,pmat,SAME_NONZERO_PATTERN);
$         PetscObjectDereference((PetscObject)mat);
$         PetscObjectDereference((PetscObject)pmat);
$           set size, type, etc of mat and pmat

    The rational for this support is so that when creating a TS, SNES, or KSP the hierarchy
    of underlying objects (i.e. SNES, KSP, PC, Mat) and their livespans can be completely 
    managed by the top most level object (i.e. the TS, SNES, or KSP). Another way to look
    at this is when you create a SNES you do not NEED to create a KSP and attach it to 
    the SNES object (the SNES object manages it for you). Similarly when you create a KSP
    you do not need to attach a PC to it (the KSP object manages the PC object for you).
    Thus, why should YOU have to create the Mat and attach it to the SNES/KSP/PC, when
    it can be created for you?
     

.keywords: PC, get, operators, matrix, linear system

.seealso: PCSetOperators(), KSPGetOperators(), KSPSetOperators(), PCGetOperatorsSet()
@*/
PetscErrorCode  PCGetOperators(PC pc,Mat *mat,Mat *pmat,MatStructure *flag)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  if (mat) {
    if (!pc->mat) {
      if (pc->pmat && !pmat) {  /* pmat has been set, but user did not request it, so use for mat */
        pc->mat = pc->pmat;
        ierr = PetscObjectReference((PetscObject)pc->mat);CHKERRQ(ierr);
      } else {                  /* both mat and pmat are empty */
        ierr = MatCreate(((PetscObject)pc)->comm,&pc->mat);CHKERRQ(ierr);
        if (!pmat) { /* user did NOT request pmat, so make same as mat */
          pc->pmat = pc->mat;
          ierr     = PetscObjectReference((PetscObject)pc->pmat);CHKERRQ(ierr);
        }
      }
    }
    *mat  = pc->mat;
  }  
  if (pmat) {
    if (!pc->pmat) {
      if (pc->mat && !mat) {    /* mat has been set but was not requested, so use for pmat */
        pc->pmat = pc->mat;
        ierr    = PetscObjectReference((PetscObject)pc->pmat);CHKERRQ(ierr);
      } else {
        ierr = MatCreate(((PetscObject)pc)->comm,&pc->pmat);CHKERRQ(ierr);
        if (!mat) { /* user did NOT request mat, so make same as pmat */
          pc->mat = pc->pmat;
          ierr    = PetscObjectReference((PetscObject)pc->mat);CHKERRQ(ierr);
        }
      }
    }
    *pmat = pc->pmat;
  }
  if (flag) *flag = pc->flag;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCGetOperatorsSet"
/*@C
   PCGetOperatorsSet - Determines if the matrix associated with the linear system and
   possibly a different one associated with the preconditioner have been set in the PC.

   Not collective, though the results on all processes should be the same

   Input Parameter:
.  pc - the preconditioner context

   Output Parameters:
+  mat - the matrix associated with the linear system was set
-  pmat - matrix associated with the preconditioner was set, usually the same

   Level: intermediate

.keywords: PC, get, operators, matrix, linear system

.seealso: PCSetOperators(), KSPGetOperators(), KSPSetOperators(), PCGetOperators()
@*/
PetscErrorCode  PCGetOperatorsSet(PC pc,PetscBool  *mat,PetscBool  *pmat)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  if (mat)  *mat  = (pc->mat)  ? PETSC_TRUE : PETSC_FALSE;
  if (pmat) *pmat = (pc->pmat) ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCFactorGetMatrix"
/*@
   PCFactorGetMatrix - Gets the factored matrix from the
   preconditioner context.  This routine is valid only for the LU, 
   incomplete LU, Cholesky, and incomplete Cholesky methods.

   Not Collective on PC though Mat is parallel if PC is parallel

   Input Parameters:
.  pc - the preconditioner context

   Output parameters:
.  mat - the factored matrix

   Level: advanced

   Notes: Does not increase the reference count for the matrix so DO NOT destroy it

.keywords: PC, get, factored, matrix
@*/
PetscErrorCode  PCFactorGetMatrix(PC pc,Mat *mat)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidPointer(mat,2);
  if (pc->ops->getfactoredmatrix) {
    ierr = (*pc->ops->getfactoredmatrix)(pc,mat);CHKERRQ(ierr);
  } else {
    SETERRQ(((PetscObject)pc)->comm,PETSC_ERR_SUP,"PC type does not support getting factor matrix");
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCSetOptionsPrefix"
/*@C
   PCSetOptionsPrefix - Sets the prefix used for searching for all 
   PC options in the database.

   Logically Collective on PC

   Input Parameters:
+  pc - the preconditioner context
-  prefix - the prefix string to prepend to all PC option requests

   Notes:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the
   hyphen.

   Level: advanced

.keywords: PC, set, options, prefix, database

.seealso: PCAppendOptionsPrefix(), PCGetOptionsPrefix()
@*/
PetscErrorCode  PCSetOptionsPrefix(PC pc,const char prefix[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)pc,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCAppendOptionsPrefix"
/*@C
   PCAppendOptionsPrefix - Appends to the prefix used for searching for all 
   PC options in the database.

   Logically Collective on PC

   Input Parameters:
+  pc - the preconditioner context
-  prefix - the prefix string to prepend to all PC option requests

   Notes:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the
   hyphen.

   Level: advanced

.keywords: PC, append, options, prefix, database

.seealso: PCSetOptionsPrefix(), PCGetOptionsPrefix()
@*/
PetscErrorCode  PCAppendOptionsPrefix(PC pc,const char prefix[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscObjectAppendOptionsPrefix((PetscObject)pc,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCGetOptionsPrefix"
/*@C
   PCGetOptionsPrefix - Gets the prefix used for searching for all 
   PC options in the database.

   Not Collective

   Input Parameters:
.  pc - the preconditioner context

   Output Parameters:
.  prefix - pointer to the prefix string used, is returned

   Notes: On the fortran side, the user should pass in a string 'prifix' of
   sufficient length to hold the prefix.

   Level: advanced

.keywords: PC, get, options, prefix, database

.seealso: PCSetOptionsPrefix(), PCAppendOptionsPrefix()
@*/
PetscErrorCode  PCGetOptionsPrefix(PC pc,const char *prefix[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidPointer(prefix,2);
  ierr = PetscObjectGetOptionsPrefix((PetscObject)pc,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCPreSolve"
/*@
   PCPreSolve - Optional pre-solve phase, intended for any
   preconditioner-specific actions that must be performed before 
   the iterative solve itself.

   Collective on PC

   Input Parameters:
+  pc - the preconditioner context
-  ksp - the Krylov subspace context

   Level: developer

   Sample of Usage:
.vb
    PCPreSolve(pc,ksp);
    KSPSolve(ksp,b,x);
    PCPostSolve(pc,ksp);
.ve

   Notes:
   The pre-solve phase is distinct from the PCSetUp() phase.

   KSPSolve() calls this directly, so is rarely called by the user.

.keywords: PC, pre-solve

.seealso: PCPostSolve()
@*/
PetscErrorCode  PCPreSolve(PC pc,KSP ksp)
{
  PetscErrorCode ierr;
  Vec            x,rhs;
  Mat            A,B;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,2);
  ierr = KSPGetSolution(ksp,&x);CHKERRQ(ierr);
  ierr = KSPGetRhs(ksp,&rhs);CHKERRQ(ierr);
  /*
      Scale the system and have the matrices use the scaled form
    only if the two matrices are actually the same (and hence
    have the same scaling
  */  
  ierr = PCGetOperators(pc,&A,&B,PETSC_NULL);CHKERRQ(ierr);
  if (A == B) {
    ierr = MatScaleSystem(pc->mat,rhs,x);CHKERRQ(ierr);
    ierr = MatUseScaledForm(pc->mat,PETSC_TRUE);CHKERRQ(ierr);
  }

  if (pc->ops->presolve) {
    ierr = (*pc->ops->presolve)(pc,ksp,rhs,x);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCPostSolve"
/*@
   PCPostSolve - Optional post-solve phase, intended for any
   preconditioner-specific actions that must be performed after
   the iterative solve itself.

   Collective on PC

   Input Parameters:
+  pc - the preconditioner context
-  ksp - the Krylov subspace context

   Sample of Usage:
.vb
    PCPreSolve(pc,ksp);
    KSPSolve(ksp,b,x);
    PCPostSolve(pc,ksp);
.ve

   Note:
   KSPSolve() calls this routine directly, so it is rarely called by the user.

   Level: developer

.keywords: PC, post-solve

.seealso: PCPreSolve(), KSPSolve()
@*/
PetscErrorCode  PCPostSolve(PC pc,KSP ksp)
{
  PetscErrorCode ierr;
  Vec            x,rhs;
  Mat            A,B;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,2);
  ierr = KSPGetSolution(ksp,&x);CHKERRQ(ierr);
  ierr = KSPGetRhs(ksp,&rhs);CHKERRQ(ierr);
  if (pc->ops->postsolve) {
    ierr =  (*pc->ops->postsolve)(pc,ksp,rhs,x);CHKERRQ(ierr);
  }
  /*
      Scale the system and have the matrices use the scaled form
    only if the two matrices are actually the same (and hence
    have the same scaling
  */  
  ierr = PCGetOperators(pc,&A,&B,PETSC_NULL);CHKERRQ(ierr);
  if (A == B) {
    ierr = MatUnScaleSystem(pc->mat,rhs,x);CHKERRQ(ierr);
    ierr = MatUseScaledForm(pc->mat,PETSC_FALSE);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCView"
/*@C
   PCView - Prints the PC data structure.

   Collective on PC

   Input Parameters:
+  PC - the PC context
-  viewer - optional visualization context

   Note:
   The available visualization contexts include
+     PETSC_VIEWER_STDOUT_SELF - standard output (default)
-     PETSC_VIEWER_STDOUT_WORLD - synchronized standard
         output where only the first processor opens
         the file.  All other processors send their 
         data to the first processor to print. 

   The user can open an alternative visualization contexts with
   PetscViewerASCIIOpen() (output to a specified file).

   Level: developer

.keywords: PC, view

.seealso: KSPView(), PetscViewerASCIIOpen()
@*/
PetscErrorCode  PCView(PC pc,PetscViewer viewer)
{
  const PCType      cstr;
  PetscErrorCode    ierr;
  PetscBool         iascii,isstring;
  PetscViewerFormat format;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  if (!viewer) {
    ierr = PetscViewerASCIIGetStdout(((PetscObject)pc)->comm,&viewer);CHKERRQ(ierr);
  }
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2); 
  PetscCheckSameComm(pc,1,viewer,2);

  ierr = PetscTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSCVIEWERSTRING,&isstring);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
    ierr = PetscObjectPrintClassNamePrefixType((PetscObject)pc,viewer,"PC Object");CHKERRQ(ierr);
    if (pc->ops->view) {
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = (*pc->ops->view)(pc,viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    }
    if (pc->mat) {
      ierr = PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_INFO);CHKERRQ(ierr);
      if (pc->pmat == pc->mat) {
        ierr = PetscViewerASCIIPrintf(viewer,"  linear system matrix = precond matrix:\n");CHKERRQ(ierr);
        ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
        ierr = MatView(pc->mat,viewer);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
      } else {
        if (pc->pmat) {
          ierr = PetscViewerASCIIPrintf(viewer,"  linear system matrix followed by preconditioner matrix:\n");CHKERRQ(ierr);
        } else {
          ierr = PetscViewerASCIIPrintf(viewer,"  linear system matrix:\n");CHKERRQ(ierr);
        }
        ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
        ierr = MatView(pc->mat,viewer);CHKERRQ(ierr);
        if (pc->pmat) {ierr = MatView(pc->pmat,viewer);CHKERRQ(ierr);}
        ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
      }
      ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
    }
  } else if (isstring) {
    ierr = PCGetType(pc,&cstr);CHKERRQ(ierr);
    ierr = PetscViewerStringSPrintf(viewer," %-7.7s",cstr);CHKERRQ(ierr);
    if (pc->ops->view) {ierr = (*pc->ops->view)(pc,viewer);CHKERRQ(ierr);}
  } else {
    SETERRQ1(((PetscObject)pc)->comm,PETSC_ERR_SUP,"Viewer type %s not supported by PC",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "PCSetInitialGuessNonzero"
/*@
   PCSetInitialGuessNonzero - Tells the iterative solver that the 
   initial guess is nonzero; otherwise PC assumes the initial guess
   is to be zero (and thus zeros it out before solving).

   Logically Collective on PC

   Input Parameters:
+  pc - iterative context obtained from PCCreate()
-  flg - PETSC_TRUE indicates the guess is non-zero, PETSC_FALSE indicates the guess is zero

   Level: Developer

   Notes:
    This is a weird function. Since PC's are linear operators on the right hand side they
    CANNOT use an initial guess. This function is for the "pass-through" preconditioners
    PCKSP, PCREDUNDANT and PCHMPI and causes the inner KSP object to use the nonzero
    initial guess. Not currently working for PCREDUNDANT, that has to be rewritten to use KSP.


.keywords: PC, set, initial guess, nonzero

.seealso: PCGetInitialGuessNonzero(), PCSetInitialGuessKnoll(), PCGetInitialGuessKnoll()
@*/
PetscErrorCode  PCSetInitialGuessNonzero(PC pc,PetscBool  flg)
{
  PetscFunctionBegin;
  PetscValidLogicalCollectiveBool(pc,flg,2);
  pc->nonzero_guess   = flg;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCRegister"
/*@C
  PCRegister - See PCRegisterDynamic()

  Level: advanced
@*/
PetscErrorCode  PCRegister(const char sname[],const char path[],const char name[],PetscErrorCode (*function)(PC))
{
  PetscErrorCode ierr;
  char           fullname[PETSC_MAX_PATH_LEN];

  PetscFunctionBegin;

  ierr = PetscFListConcat(path,name,fullname);CHKERRQ(ierr);
  ierr = PetscFListAdd(&PCList,sname,fullname,(void (*)(void))function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCComputeExplicitOperator"
/*@
    PCComputeExplicitOperator - Computes the explicit preconditioned operator.  

    Collective on PC

    Input Parameter:
.   pc - the preconditioner object

    Output Parameter:
.   mat - the explict preconditioned operator

    Notes:
    This computation is done by applying the operators to columns of the 
    identity matrix.

    Currently, this routine uses a dense matrix format when 1 processor
    is used and a sparse format otherwise.  This routine is costly in general,
    and is recommended for use only with relatively small systems.

    Level: advanced
   
.keywords: PC, compute, explicit, operator

.seealso: KSPComputeExplicitOperator()

@*/
PetscErrorCode  PCComputeExplicitOperator(PC pc,Mat *mat)
{
  Vec            in,out;
  PetscErrorCode ierr;
  PetscInt       i,M,m,*rows,start,end;
  PetscMPIInt    size;
  MPI_Comm       comm;
  PetscScalar    *array,one = 1.0;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidPointer(mat,2);

  comm = ((PetscObject)pc)->comm;
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);

  if (!pc->pmat) SETERRQ(((PetscObject)pc)->comm,PETSC_ERR_ORDER,"You must call KSPSetOperators() or PCSetOperators() before this call");
  ierr = MatGetVecs(pc->pmat,&in,0);CHKERRQ(ierr);
  ierr = VecDuplicate(in,&out);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(in,&start,&end);CHKERRQ(ierr);
  ierr = VecGetSize(in,&M);CHKERRQ(ierr);
  ierr = VecGetLocalSize(in,&m);CHKERRQ(ierr);
  ierr = PetscMalloc((m+1)*sizeof(PetscInt),&rows);CHKERRQ(ierr);
  for (i=0; i<m; i++) {rows[i] = start + i;}

  ierr = MatCreate(comm,mat);CHKERRQ(ierr);
  ierr = MatSetSizes(*mat,m,m,M,M);CHKERRQ(ierr);
  if (size == 1) {
    ierr = MatSetType(*mat,MATSEQDENSE);CHKERRQ(ierr);
    ierr = MatSeqDenseSetPreallocation(*mat,PETSC_NULL);CHKERRQ(ierr);
  } else {
    ierr = MatSetType(*mat,MATMPIAIJ);CHKERRQ(ierr);
    ierr = MatMPIAIJSetPreallocation(*mat,0,PETSC_NULL,0,PETSC_NULL);CHKERRQ(ierr);
  }

  for (i=0; i<M; i++) {

    ierr = VecSet(in,0.0);CHKERRQ(ierr);
    ierr = VecSetValues(in,1,&i,&one,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(in);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(in);CHKERRQ(ierr);

    /* should fix, allowing user to choose side */
    ierr = PCApply(pc,in,out);CHKERRQ(ierr);
    
    ierr = VecGetArray(out,&array);CHKERRQ(ierr);
    ierr = MatSetValues(*mat,m,rows,1,&i,array,INSERT_VALUES);CHKERRQ(ierr); 
    ierr = VecRestoreArray(out,&array);CHKERRQ(ierr);

  }
  ierr = PetscFree(rows);CHKERRQ(ierr);
  ierr = VecDestroy(&out);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(*mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCSetCoordinates"
/*@
   PCSetCoordinates - sets the coordinates of all the nodes on the local process

   Collective on PC

   Input Parameters:
+  pc - the solver context
.  dim - the dimension of the coordinates 1, 2, or 3
-  coords - the coordinates

   Level: intermediate

   Notes: coords is an array of the 3D coordinates for the nodes on
   the local processor.  So if there are 108 equation on a processor
   for a displacement finite element discretization of elasticity (so
   that there are 36 = 108/3 nodes) then the array must have 108
   double precision values (ie, 3 * 36).  These x y z coordinates
   should be ordered for nodes 0 to N-1 like so: [ 0.x, 0.y, 0.z, 1.x,
   ... , N-1.z ].

.seealso: MatSetNearNullSpace
@*/
PetscErrorCode PCSetCoordinates(PC pc, PetscInt dim, PetscInt nloc, PetscReal *coords )
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscTryMethod(pc,"PCSetCoordinates_C",(PC,PetscInt,PetscInt,PetscReal*),(pc,dim,nloc,coords));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
