#define PETSCKSP_DLL

/*
    The PC (preconditioner) interface routines, callable by users.
*/
#include "src/ksp/pc/pcimpl.h"            /*I "petscksp.h" I*/

/* Logging support */
PetscCookie PC_COOKIE = 0;
PetscEvent  PC_SetUp = 0, PC_SetUpOnBlocks = 0, PC_Apply = 0, PC_ApplyCoarse = 0, PC_ApplyMultiple = 0, PC_ApplySymmetricLeft = 0;
PetscEvent  PC_ApplySymmetricRight = 0, PC_ModifySubMatrices = 0;

#undef __FUNCT__  
#define __FUNCT__ "PCGetDefaultType_Private"
PetscErrorCode PCGetDefaultType_Private(PC pc,const char* type[])
{
  PetscErrorCode ierr;
  PetscMPIInt    size;
  PetscTruth     flg1,flg2,set,flg3;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(pc->comm,&size);CHKERRQ(ierr);
  if (pc->pmat) {
    PetscErrorCode (*f)(Mat,PetscTruth*,MatReuse,Mat*);
    ierr = PetscObjectQueryFunction((PetscObject)pc->pmat,"MatGetDiagonalBlock_C",(void (**)(void))&f);CHKERRQ(ierr);
    if (size == 1) {
      ierr = MatHasOperation(pc->pmat,MATOP_ICCFACTOR_SYMBOLIC,&flg1);CHKERRQ(ierr);
      ierr = MatHasOperation(pc->pmat,MATOP_ILUFACTOR_SYMBOLIC,&flg2);CHKERRQ(ierr);
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
#define __FUNCT__ "PCDestroy"
/*@C
   PCDestroy - Destroys PC context that was created with PCCreate().

   Collective on PC

   Input Parameter:
.  pc - the preconditioner context

   Level: developer

.keywords: PC, destroy

.seealso: PCCreate(), PCSetUp()
@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCDestroy(PC pc)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  if (--pc->refct > 0) PetscFunctionReturn(0);

  /* if memory was published with AMS then destroy it */
  ierr = PetscObjectDepublish(pc);CHKERRQ(ierr);

  if (pc->ops->destroy)       {ierr =  (*pc->ops->destroy)(pc);CHKERRQ(ierr);}
  if (pc->diagonalscaleright) {ierr = VecDestroy(pc->diagonalscaleright);CHKERRQ(ierr);}
  if (pc->diagonalscaleleft)  {ierr = VecDestroy(pc->diagonalscaleleft);CHKERRQ(ierr);}

  ierr = PetscHeaderDestroy(pc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCDiagonalScale"
/*@C
   PCDiagonalScale - Indicates if the preconditioner applies an additional left and right
      scaling as needed by certain time-stepping codes.

   Collective on PC

   Input Parameter:
.  pc - the preconditioner context

   Output Parameter:
.  flag - PETSC_TRUE if it applies the scaling

   Level: developer

   Notes: If this returns PETSC_TRUE then the system solved via the Krylov method is
$           D M A D^{-1} y = D M b  for left preconditioning or
$           D A M D^{-1} z = D b for right preconditioning

.keywords: PC

.seealso: PCCreate(), PCSetUp(), PCDiagonalScaleLeft(), PCDiagonalScaleRight(), PCDiagonalScaleSet()
@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCDiagonalScale(PC pc,PetscTruth *flag)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  PetscValidPointer(flag,2);
  *flag = pc->diagonalscale;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCDiagonalScaleSet"
/*@
   PCDiagonalScaleSet - Indicates the left scaling to use to apply an additional left and right
      scaling as needed by certain time-stepping codes.

   Collective on PC

   Input Parameters:
+  pc - the preconditioner context
-  s - scaling vector

   Level: intermediate

   Notes: The system solved via the Krylov method is
$           D M A D^{-1} y = D M b  for left preconditioning or
$           D A M D^{-1} z = D b for right preconditioning

   PCDiagonalScaleLeft() scales a vector by D. PCDiagonalScaleRight() scales a vector by D^{-1}.

.keywords: PC

.seealso: PCCreate(), PCSetUp(), PCDiagonalScaleLeft(), PCDiagonalScaleRight(), PCDiagonalScale()
@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCDiagonalScaleSet(PC pc,Vec s)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  PetscValidHeaderSpecific(s,VEC_COOKIE,2);
  pc->diagonalscale     = PETSC_TRUE;
  if (pc->diagonalscaleleft) {
    ierr = VecDestroy(pc->diagonalscaleleft);CHKERRQ(ierr);
  }
  pc->diagonalscaleleft = s;
  ierr                  = PetscObjectReference((PetscObject)s);CHKERRQ(ierr);
  if (!pc->diagonalscaleright) {
    ierr = VecDuplicate(s,&pc->diagonalscaleright);CHKERRQ(ierr);
  }
  ierr = VecCopy(s,pc->diagonalscaleright);CHKERRQ(ierr);
  ierr = VecReciprocal(pc->diagonalscaleright);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCDiagonalScaleLeft"
/*@C
   PCDiagonalScaleLeft - Indicates the left scaling to use to apply an additional left and right
      scaling as needed by certain time-stepping codes.

   Collective on PC

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
PetscErrorCode PETSCKSP_DLLEXPORT PCDiagonalScaleLeft(PC pc,Vec in,Vec out)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  PetscValidHeaderSpecific(in,VEC_COOKIE,2);
  PetscValidHeaderSpecific(out,VEC_COOKIE,3);
  if (pc->diagonalscale) {
    ierr = VecPointwiseMult(pc->diagonalscaleleft,in,out);CHKERRQ(ierr);
  } else if (in != out) {
    ierr = VecCopy(in,out);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCDiagonalScaleRight"
/*@C
   PCDiagonalScaleRight - Scales a vector by the right scaling as needed by certain time-stepping codes.

   Collective on PC

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
PetscErrorCode PETSCKSP_DLLEXPORT PCDiagonalScaleRight(PC pc,Vec in,Vec out)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  PetscValidHeaderSpecific(in,VEC_COOKIE,2);
  PetscValidHeaderSpecific(out,VEC_COOKIE,3);
  if (pc->diagonalscale) {
    ierr = VecPointwiseMult(pc->diagonalscaleright,in,out);CHKERRQ(ierr);
  } else if (in != out) {
    ierr = VecCopy(in,out);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCPublish_Petsc"
static PetscErrorCode PCPublish_Petsc(PetscObject obj)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCCreate"
/*@C
   PCCreate - Creates a preconditioner context.

   Collective on MPI_Comm

   Input Parameter:
.  comm - MPI communicator 

   Output Parameter:
.  pc - location to put the preconditioner context

   Notes:
   The default preconditioner on one processor is PCILU with 0 fill on more 
   then one it is PCBJACOBI with ILU() on each processor.

   Level: developer

.keywords: PC, create, context

.seealso: PCSetUp(), PCApply(), PCDestroy()
@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCCreate(MPI_Comm comm,PC *newpc)
{
  PC             pc;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(newpc,1)
  *newpc = 0;
#ifndef PETSC_USE_DYNAMIC_LIBRARIES
  ierr = PCInitializePackage(PETSC_NULL);CHKERRQ(ierr);
#endif

  ierr = PetscHeaderCreate(pc,_p_PC,struct _PCOps,PC_COOKIE,-1,"PC",comm,PCDestroy,PCView);CHKERRQ(ierr);
  pc->bops->publish      = PCPublish_Petsc;
  pc->mat                = 0;
  pc->pmat               = 0;
  pc->setupcalled        = 0;
  pc->data               = 0;
  pc->diagonalscale      = PETSC_FALSE;
  pc->diagonalscaleleft  = 0;
  pc->diagonalscaleright = 0;

  pc->ops->destroy             = 0;
  pc->ops->apply               = 0;
  pc->ops->applytranspose      = 0;
  pc->ops->applyBA             = 0;
  pc->ops->applyBAtranspose    = 0;
  pc->ops->applyrichardson     = 0;
  pc->ops->view                = 0;
  pc->ops->getfactoredmatrix   = 0;
  pc->ops->applysymmetricright = 0;
  pc->ops->applysymmetricleft  = 0;
  pc->ops->setuponblocks       = 0;

  pc->modifysubmatrices   = 0;
  pc->modifysubmatricesP  = 0;
  *newpc                  = pc;
  ierr = PetscPublishAll(pc);CHKERRQ(ierr);
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
PetscErrorCode PETSCKSP_DLLEXPORT PCApply(PC pc,Vec x,Vec y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  PetscValidHeaderSpecific(x,VEC_COOKIE,2);
  PetscValidHeaderSpecific(y,VEC_COOKIE,3);
  if (x == y) SETERRQ(PETSC_ERR_ARG_IDN,"x and y must be different vectors");

  if (pc->setupcalled < 2) {
    ierr = PCSetUp(pc);CHKERRQ(ierr);
  }
  ierr = PetscLogEventBegin(PC_Apply,pc,x,y,0);CHKERRQ(ierr);
  ierr = (*pc->ops->apply)(pc,x,y);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(PC_Apply,pc,x,y,0);CHKERRQ(ierr);
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
PetscErrorCode PETSCKSP_DLLEXPORT PCApplySymmetricLeft(PC pc,Vec x,Vec y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  PetscValidHeaderSpecific(x,VEC_COOKIE,2);
  PetscValidHeaderSpecific(y,VEC_COOKIE,3);
  if (!pc->ops->applysymmetricleft) SETERRQ(PETSC_ERR_SUP,"PC does not have left symmetric apply");

  if (pc->setupcalled < 2) {
    ierr = PCSetUp(pc);CHKERRQ(ierr);
  }

  ierr = PetscLogEventBegin(PC_ApplySymmetricLeft,pc,x,y,0);CHKERRQ(ierr);
  ierr = (*pc->ops->applysymmetricleft)(pc,x,y);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(PC_ApplySymmetricLeft,pc,x,y,0);CHKERRQ(ierr);
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
PetscErrorCode PETSCKSP_DLLEXPORT PCApplySymmetricRight(PC pc,Vec x,Vec y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  PetscValidHeaderSpecific(x,VEC_COOKIE,2);
  PetscValidHeaderSpecific(y,VEC_COOKIE,3);
  if (!pc->ops->applysymmetricright) SETERRQ(PETSC_ERR_SUP,"PC does not have left symmetric apply");

  if (pc->setupcalled < 2) {
    ierr = PCSetUp(pc);CHKERRQ(ierr);
  }

  ierr = PetscLogEventBegin(PC_ApplySymmetricRight,pc,x,y,0);CHKERRQ(ierr);
  ierr = (*pc->ops->applysymmetricright)(pc,x,y);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(PC_ApplySymmetricRight,pc,x,y,0);CHKERRQ(ierr);
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

   Level: developer

.keywords: PC, apply, transpose

.seealso: PCApply(), PCApplyBAorAB(), PCApplyBAorABTranspose(), PCHasApplyTranspose()
@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCApplyTranspose(PC pc,Vec x,Vec y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  PetscValidHeaderSpecific(x,VEC_COOKIE,2);
  PetscValidHeaderSpecific(y,VEC_COOKIE,3);
  if (x == y) SETERRQ(PETSC_ERR_ARG_IDN,"x and y must be different vectors");
  if (!pc->ops->applytranspose) SETERRQ(PETSC_ERR_SUP," ");

  if (pc->setupcalled < 2) {
    ierr = PCSetUp(pc);CHKERRQ(ierr);
  }

  ierr = PetscLogEventBegin(PC_Apply,pc,x,y,0);CHKERRQ(ierr);
  ierr = (*pc->ops->applytranspose)(pc,x,y);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(PC_Apply,pc,x,y,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCHasApplyTranspose"
/*@
   PCHasApplyTranspose - Test whether the preconditioner has a transpose apply operation

   Collective on PC and Vec

   Input Parameters:
.  pc - the preconditioner context

   Output Parameter:
.  flg - PETSC_TRUE if a transpose operation is defined

   Level: developer

.keywords: PC, apply, transpose

.seealso: PCApplyTranspose()
@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCHasApplyTranspose(PC pc,PetscTruth *flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  PetscValidPointer(flg,2);
  *flg = (PetscTruth) (pc->ops->applytranspose == 0);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCApplyBAorAB"
/*@
   PCApplyBAorAB - Applies the preconditioner and operator to a vector. 

   Collective on PC and Vec

   Input Parameters:
+  pc - the preconditioner context
.  side - indicates the preconditioner side, one of PC_LEFT, PC_RIGHT, or PC_SYMMETRIC
.  x - input vector
-  work - work vector

   Output Parameter:
.  y - output vector

   Level: developer

.keywords: PC, apply, operator

.seealso: PCApply(), PCApplyTranspose(), PCApplyBAorABTranspose()
@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCApplyBAorAB(PC pc,PCSide side,Vec x,Vec y,Vec work)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  PetscValidHeaderSpecific(x,VEC_COOKIE,3);
  PetscValidHeaderSpecific(y,VEC_COOKIE,4);
  PetscValidHeaderSpecific(work,VEC_COOKIE,5);
  if (x == y) SETERRQ(PETSC_ERR_ARG_IDN,"x and y must be different vectors");
  if (side != PC_LEFT && side != PC_SYMMETRIC && side != PC_RIGHT) {
    SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Side must be right, left, or symmetric");
  }
  if (pc->diagonalscale && side == PC_SYMMETRIC) {
    SETERRQ(PETSC_ERR_SUP,"Cannot include diagonal scaling with symmetric preconditioner application");
  }

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
      ierr = VecDestroy(work2);CHKERRQ(ierr);
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
    } else if (side == PC_SYMMETRIC) {
      SETERRQ(PETSC_ERR_SUP,"Cannot provide diagonal scaling with symmetric application of preconditioner");
    }
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
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCApplyBAorABTranspose"
/*@ 
   PCApplyBAorABTranspose - Applies the transpose of the preconditioner
   and operator to a vector. That is, applies tr(B) * tr(A) with left preconditioning,
   not tr(B*A) = tr(A)*tr(B).

   Collective on PC and Vec

   Input Parameters:
+  pc - the preconditioner context
.  side - indicates the preconditioner side, one of PC_LEFT, PC_RIGHT, or PC_SYMMETRIC
.  x - input vector
-  work - work vector

   Output Parameter:
.  y - output vector

   Level: developer

.keywords: PC, apply, operator, transpose

.seealso: PCApply(), PCApplyTranspose(), PCApplyBAorAB()
@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCApplyBAorABTranspose(PC pc,PCSide side,Vec x,Vec y,Vec work)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  PetscValidHeaderSpecific(x,VEC_COOKIE,3);
  PetscValidHeaderSpecific(y,VEC_COOKIE,4);
  PetscValidHeaderSpecific(work,VEC_COOKIE,5);
  if (x == y) SETERRQ(PETSC_ERR_ARG_IDN,"x and y must be different vectors");
  if (pc->ops->applyBAtranspose) {
    ierr = (*pc->ops->applyBAtranspose)(pc,side,x,y,work);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  if (side != PC_LEFT && side != PC_RIGHT) {
    SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Side must be right or left");
  }

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
  PetscFunctionReturn(0); /* actually will never get here */
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
PetscErrorCode PETSCKSP_DLLEXPORT PCApplyRichardsonExists(PC pc,PetscTruth *exists)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  PetscValidIntPointer(exists,2);
  if (pc->ops->applyrichardson) *exists = PETSC_TRUE; 
  else                    *exists = PETSC_FALSE;
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
.  x   - the initial guess 
.  w   - one work vector
.  rtol - relative decrease in residual norm convergence criteria
.  abstol - absolute residual norm convergence criteria
.  dtol - divergence residual norm increase criteria
-  its - the number of iterations to apply.

   Output Parameter:
.  y - the solution

   Notes: 
   Most preconditioners do not support this function. Use the command
   PCApplyRichardsonExists() to determine if one does.

   Except for the multigrid PC this routine ignores the convergence tolerances
   and always runs for the number of iterations
 
   Level: developer

.keywords: PC, apply, Richardson

.seealso: PCApplyRichardsonExists()
@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCApplyRichardson(PC pc,Vec x,Vec y,Vec w,PetscReal rtol,PetscReal abstol, PetscReal dtol,PetscInt its)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  PetscValidHeaderSpecific(x,VEC_COOKIE,2);
  PetscValidHeaderSpecific(y,VEC_COOKIE,3);
  PetscValidHeaderSpecific(w,VEC_COOKIE,4);
  if (!pc->ops->applyrichardson) SETERRQ(PETSC_ERR_SUP," ");

  if (pc->setupcalled < 2) {
    ierr = PCSetUp(pc);CHKERRQ(ierr);
  }

  ierr = (*pc->ops->applyrichardson)(pc,x,y,w,rtol,abstol,dtol,its);CHKERRQ(ierr);
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
PetscErrorCode PETSCKSP_DLLEXPORT PCSetUp(PC pc)
{
  PetscErrorCode ierr;
  const char     *def;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);

  if (pc->setupcalled > 1) {
    ierr = PetscLogInfo((pc,"PCSetUp:Setting PC with identical preconditioner\n"));CHKERRQ(ierr);
    PetscFunctionReturn(0);
  } else if (!pc->setupcalled) {
    ierr = PetscLogInfo((pc,"PCSetUp:Setting up new PC\n"));CHKERRQ(ierr);
  } else if (pc->flag == SAME_NONZERO_PATTERN) {
    ierr = PetscLogInfo((pc,"PCSetUp:Setting up PC with same nonzero pattern\n"));CHKERRQ(ierr);
  } else {
    ierr = PetscLogInfo((pc,"PCSetUp:Setting up PC with different nonzero pattern\n"));CHKERRQ(ierr);
  }

  ierr = PetscLogEventBegin(PC_SetUp,pc,0,0,0);CHKERRQ(ierr);
  if (!pc->mat) {SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Matrix must be set first");}

  if (!pc->type_name) {
    ierr = PCGetDefaultType_Private(pc,&def);CHKERRQ(ierr);
    ierr = PCSetType(pc,def);CHKERRQ(ierr);
  }

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
PetscErrorCode PETSCKSP_DLLEXPORT PCSetUpOnBlocks(PC pc)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
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

   Collective on PC

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

.seealso: PCModifySubMatrices()
@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCSetModifySubMatrices(PC pc,PetscErrorCode (*func)(PC,PetscInt,const IS[],const IS[],Mat[],void*),void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
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
PetscErrorCode PETSCKSP_DLLEXPORT PCModifySubMatrices(PC pc,PetscInt nsub,const IS row[],const IS col[],Mat submat[],void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
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

   Collective on PC and Mat

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
PetscErrorCode PETSCKSP_DLLEXPORT PCSetOperators(PC pc,Mat Amat,Mat Pmat,MatStructure flag)
{
  PetscErrorCode ierr;
  PetscTruth     isbjacobi,isrowbs;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  if (Amat) PetscValidHeaderSpecific(Amat,MAT_COOKIE,2);
  if (Pmat) PetscValidHeaderSpecific(Pmat,MAT_COOKIE,3);

  /*
      BlockSolve95 cannot use default BJacobi preconditioning
  */
  if (Amat) {
    ierr = PetscTypeCompare((PetscObject)Amat,MATMPIROWBS,&isrowbs);CHKERRQ(ierr);
    if (isrowbs) {
      ierr = PetscTypeCompare((PetscObject)pc,PCBJACOBI,&isbjacobi);CHKERRQ(ierr);
      if (isbjacobi) {
        ierr = PCSetType(pc,PCILU);CHKERRQ(ierr);
        ierr = PetscLogInfo((pc,"PCSetOperators:Switching default PC to PCILU since BS95 doesn't support PCBJACOBI\n"));CHKERRQ(ierr);
      }
    }
  }

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

.keywords: PC, get, operators, matrix, linear system

.seealso: PCSetOperators()
@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCGetOperators(PC pc,Mat *mat,Mat *pmat,MatStructure *flag)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  if (mat)  *mat  = pc->mat;
  if (pmat) *pmat = pc->pmat;
  if (flag) *flag = pc->flag;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCGetFactoredMatrix"
/*@C 
   PCGetFactoredMatrix - Gets the factored matrix from the
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
PetscErrorCode PETSCKSP_DLLEXPORT PCGetFactoredMatrix(PC pc,Mat *mat)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  PetscValidPointer(mat,2);
  if (pc->ops->getfactoredmatrix) {
    ierr = (*pc->ops->getfactoredmatrix)(pc,mat);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCSetOptionsPrefix"
/*@C
   PCSetOptionsPrefix - Sets the prefix used for searching for all 
   PC options in the database.

   Collective on PC

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
PetscErrorCode PETSCKSP_DLLEXPORT PCSetOptionsPrefix(PC pc,const char prefix[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)pc,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCAppendOptionsPrefix"
/*@C
   PCAppendOptionsPrefix - Appends to the prefix used for searching for all 
   PC options in the database.

   Collective on PC

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
PetscErrorCode PETSCKSP_DLLEXPORT PCAppendOptionsPrefix(PC pc,const char prefix[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
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
PetscErrorCode PETSCKSP_DLLEXPORT PCGetOptionsPrefix(PC pc,const char *prefix[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
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
PetscErrorCode PETSCKSP_DLLEXPORT PCPreSolve(PC pc,KSP ksp)
{
  PetscErrorCode ierr;
  Vec            x,rhs;
  Mat            A,B;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  PetscValidHeaderSpecific(ksp,KSP_COOKIE,2);
  ierr = KSPGetSolution(ksp,&x);CHKERRQ(ierr);
  ierr = KSPGetRhs(ksp,&rhs);CHKERRQ(ierr);
  /*
      Scale the system and have the matrices use the scaled form
    only if the two matrices are actually the same (and hence
    have the same scaling
  */  
  ierr = PCGetOperators(pc,&A,&B,PETSC_NULL);CHKERRQ(ierr);
  if (A == B) {
    ierr = MatScaleSystem(pc->mat,x,rhs);CHKERRQ(ierr);
    ierr = MatUseScaledForm(pc->mat,PETSC_TRUE);CHKERRQ(ierr);
  }

  if (pc->ops->presolve) {
    ierr = (*pc->ops->presolve)(pc,ksp,x,rhs);CHKERRQ(ierr);
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
PetscErrorCode PETSCKSP_DLLEXPORT PCPostSolve(PC pc,KSP ksp)
{
  PetscErrorCode ierr;
  Vec            x,rhs;
  Mat            A,B;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  PetscValidHeaderSpecific(ksp,KSP_COOKIE,2);
  ierr = KSPGetSolution(ksp,&x);CHKERRQ(ierr);
  ierr = KSPGetRhs(ksp,&rhs);CHKERRQ(ierr);
  if (pc->ops->postsolve) {
    ierr =  (*pc->ops->postsolve)(pc,ksp,x,rhs);CHKERRQ(ierr);
  }

  /*
      Scale the system and have the matrices use the scaled form
    only if the two matrices are actually the same (and hence
    have the same scaling
  */  
  ierr = PCGetOperators(pc,&A,&B,PETSC_NULL);CHKERRQ(ierr);
  if (A == B) {
    ierr = MatUnScaleSystem(pc->mat,x,rhs);CHKERRQ(ierr);
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
PetscErrorCode PETSCKSP_DLLEXPORT PCView(PC pc,PetscViewer viewer)
{
  PCType            cstr;
  PetscErrorCode    ierr;
  PetscTruth        mat_exists,iascii,isstring;
  PetscViewerFormat format;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  if (!viewer) viewer = PETSC_VIEWER_STDOUT_(pc->comm);
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_COOKIE,2); 
  PetscCheckSameComm(pc,1,viewer,2);

  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&iascii);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_STRING,&isstring);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
    if (pc->prefix) {
      ierr = PetscViewerASCIIPrintf(viewer,"PC Object:(%s)\n",pc->prefix);CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"PC Object:\n");CHKERRQ(ierr);
    }
    ierr = PCGetType(pc,&cstr);CHKERRQ(ierr);
    if (cstr) {
      ierr = PetscViewerASCIIPrintf(viewer,"  type: %s\n",cstr);CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"  type: not yet set\n");CHKERRQ(ierr);
    }
    if (pc->ops->view) {
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = (*pc->ops->view)(pc,viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    }
    ierr = PetscObjectExists((PetscObject)pc->mat,&mat_exists);CHKERRQ(ierr);
    if (mat_exists) {
      ierr = PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_INFO);CHKERRQ(ierr);
      if (pc->pmat == pc->mat) {
        ierr = PetscViewerASCIIPrintf(viewer,"  linear system matrix = precond matrix:\n");CHKERRQ(ierr);
        ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
        ierr = MatView(pc->mat,viewer);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
      } else {
        ierr = PetscObjectExists((PetscObject)pc->pmat,&mat_exists);CHKERRQ(ierr);
        if (mat_exists) {
          ierr = PetscViewerASCIIPrintf(viewer,"  linear system matrix followed by preconditioner matrix:\n");CHKERRQ(ierr);
        } else {
          ierr = PetscViewerASCIIPrintf(viewer,"  linear system matrix:\n");CHKERRQ(ierr);
        }
        ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
        ierr = MatView(pc->mat,viewer);CHKERRQ(ierr);
        if (mat_exists) {ierr = MatView(pc->pmat,viewer);CHKERRQ(ierr);}
        ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
      }
      ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
    }
  } else if (isstring) {
    ierr = PCGetType(pc,&cstr);CHKERRQ(ierr);
    ierr = PetscViewerStringSPrintf(viewer," %-7.7s",cstr);CHKERRQ(ierr);
    if (pc->ops->view) {ierr = (*pc->ops->view)(pc,viewer);CHKERRQ(ierr);}
  } else {
    SETERRQ1(PETSC_ERR_SUP,"Viewer type %s not supported by PC",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCRegister"
/*@C
  PCRegister - See PCRegisterDynamic()

  Level: advanced
@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCRegister(const char sname[],const char path[],const char name[],PetscErrorCode (*function)(PC))
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

@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCComputeExplicitOperator(PC pc,Mat *mat)
{
  Vec            in,out;
  PetscErrorCode ierr;
  PetscInt       i,M,m,*rows,start,end;
  PetscMPIInt    size;
  MPI_Comm       comm;
  PetscScalar    *array,zero = 0.0,one = 1.0;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  PetscValidPointer(mat,2);

  comm = pc->comm;
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);

  if (!pc->pmat) SETERRQ(PETSC_ERR_ORDER,"You must call KSPSetOperators() or PCSetOperators() before this call");
  ierr = MatGetVecs(pc->pmat,&in,0);CHKERRQ(ierr);
  ierr = VecDuplicate(in,&out);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(in,&start,&end);CHKERRQ(ierr);
  ierr = VecGetSize(in,&M);CHKERRQ(ierr);
  ierr = VecGetLocalSize(in,&m);CHKERRQ(ierr);
  ierr = PetscMalloc((m+1)*sizeof(PetscInt),&rows);CHKERRQ(ierr);
  for (i=0; i<m; i++) {rows[i] = start + i;}

  ierr = MatCreate(comm,m,m,M,M,mat);CHKERRQ(ierr);
  if (size == 1) {
    ierr = MatSetType(*mat,MATSEQDENSE);CHKERRQ(ierr);
    ierr = MatSeqDenseSetPreallocation(*mat,PETSC_NULL);CHKERRQ(ierr);
  } else {
    ierr = MatSetType(*mat,MATMPIAIJ);CHKERRQ(ierr);
    ierr = MatMPIAIJSetPreallocation(*mat,0,PETSC_NULL,0,PETSC_NULL);CHKERRQ(ierr);
  }

  for (i=0; i<M; i++) {

    ierr = VecSet(&zero,in);CHKERRQ(ierr);
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
  ierr = VecDestroy(out);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(*mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

