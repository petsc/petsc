
/*  
   Defines the abstract operations on index sets 
*/
#include "pcimpl.h"      /*I "pc.h" I*/

/*@
    PCPrintHelp - Prints help message for PC

  Input Parameter:
.  pc - a preconditioner context
@*/
int PCPrintHelp(PC pc)
{
  fprintf(stderr,"PC options\n");
  PCPrintMethods(pc->namemethod);
  fprintf(stderr,"Run program with -pcmethod method -help for help on ");
  fprintf(stderr,"a particular method\n");
  if (pc->printhelp) (*pc->printhelp)(pc);
  return 0;
}

/*@
    PCDestroy - Destroy an preconditioner context.

  Input Parameters:
.  pc - the preconditioner context.

@*/
int PCDestroy(PC pc)
{
  int ierr = 0;
  VALIDHEADER(pc,PC_COOKIE);
  if (pc->destroy) ierr =  (*pc->destroy)((PetscObject)pc);
  FREE(pc);
  return ierr;
}

/*@
    PCCreate - create a preconditioner context

  Output Parameters:
.  pc - the preconditioner context.

@*/
int PCCreate(PC *newpc)
{
  PC pc;
  *newpc          = 0;
  CREATEHEADER(pc,_PC);
  pc->cookie      = PC_COOKIE;
  pc->type        = 0;
  pc->vec         = 0;
  pc->mat         = 0;
  pc->setupcalled = 0;
  pc->data        = 0;
  pc->namemethod  = "-pcmethod";
  pc->apply       = 0;
  pc->applyrich   = 0;
  *newpc          = pc;
  /* this violates rule about seperating abstract from implementions*/
  return PCSetMethod(pc,PCJACOBI);
}

/*
      Applies preconditioner to a vector 
*/
int PCApply(void *pcin,Vec x,Vec y)
{
  PC pc = (PC) pcin;
  return (*pc->apply)(pc,x,y);
}

int PCApplyRichardsonExists(PC pc)
{
  if (pc->applyrich) return 1; else return 0;
}

int PCApplyRichardson(void *pcin,Vec x,Vec y,Vec w,int its)
{
  PC pc = (PC) pcin;
  return (*pc->applyrich)(pc,x,y,w,its);
}

/*@
    PCSetUp - prepares for the use of a preconditioner.

  Input parameters:
.   pc - the preconditioner context
@*/
int PCSetUp(PC pc)
{
  if (pc->setupcalled) return 0;
  if (!pc->vec) {SETERR(1,"Vector must be set before calling PCSetUp");}
  if (!pc->mat) {SETERR(1,"Matrix must be set before calling PCSetUp");}
  pc->setupcalled = 1;
  if (pc->setup) return (*pc->setup)(pc);
  else           return 0;
}

/*@
    PCSetMatrix - set the matrix associated with the preconditioner.

  Input Parameters:
.  pc - the preconditioner context
.  mat - the matrix
@*/
int PCSetMatrix(PC pc,Mat mat)
{
  VALIDHEADER(pc,PC_COOKIE);
  pc->mat = mat;
  return 0;
}
/*@
    PCGetMatrix - Gets the matrix associated with the preconditioner.

  Input Parameters:
.  pc - the preconditioner context

  Output Parameter:
.  mat - the matrix
@*/
int PCGetMatrix(PC pc,Mat *mat)
{
  VALIDHEADER(pc,PC_COOKIE);
  *mat = pc->mat;
  return 0;
}

/*@
    PCSetVector - set a vector associated with the preconditioner.

  Input Parameters:
.  pc - the preconditioner context
.  vec - the vector
@*/
int PCSetVector(PC pc,Vec vec)
{
  VALIDHEADER(pc,PC_COOKIE);
  pc->vec = vec;
  return 0;
}

/*@
     PCGetMethodFromContext - gets the preconditioner method from an 
            active preconditioner context.

  Input Parameters:
.  pc - the preconditioner context

  Output parameters:
.  method - the method id
@*/
int PCGetMethodFromContext(PC pc,PCMETHOD *method)
{
  VALIDHEADER(pc,PC_COOKIE);
  *method = (PCMETHOD) pc->type;
  return 0;
}
