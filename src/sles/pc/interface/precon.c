#ifndef lint
static char vcid[] = "$Id: precon.c,v 1.52 1995/10/13 00:57:11 curfman Exp bsmith $";
#endif

#include "pcimpl.h"            /*I "pc.h" I*/
#include "pinclude/pviewer.h"

extern int PCPrintMethods_Private(char*,char*);
/*@
   PCPrintHelp - Prints all the options for the PC component.

   Input Parameter:
.  pc - the preconditioner context

   Options Database Keys:
$  -help, -h

.keywords: PC, help

.seealso: PCSetFromOptions()
@*/
int PCPrintHelp(PC pc)
{
  char *p; 
  if (pc->prefix) p = pc->prefix; else p = "-";
  MPIU_printf(pc->comm,"PC options ----------------------------------------\n");
  PCPrintMethods_Private(p,"pc_method");
  MPIU_printf(pc->comm,"Run program with %spc_method method -help for help on ",p);
  MPIU_printf(pc->comm,"a particular method\n");
  if (pc->printhelp) (*pc->printhelp)(pc);
  return 0;
}

/*@C
   PCDestroy - Destroys PC context that was created with PCCreate().

   Input Parameter:
.  pc - the preconditioner context

.keywords: PC, destroy

.seealso: PCCreate(), PCSetUp()
@*/
int PCDestroy(PC pc)
{
  int ierr = 0;
  PETSCVALIDHEADERSPECIFIC(pc,PC_COOKIE);
  if (pc->destroy) ierr =  (*pc->destroy)((PetscObject)pc);
  else {if (pc->data) PetscFree(pc->data);}
  PLogObjectDestroy(pc);
  PetscHeaderDestroy(pc);
  return ierr;
}

/*@C
   PCCreate - Creates a preconditioner context.

   Input Parameter:
.   comm - MPI communicator 

   Output Parameter:
.  pc - location to put the preconditioner context

   Notes:
   The default preconditioner is PCJACOBI.

.keywords: PC, create, context

.seealso: PCSetUp(), PCApply(), PCDestroy()
@*/
int PCCreate(MPI_Comm comm,PC *newpc)
{
  PC pc;
  *newpc          = 0;
  PetscHeaderCreate(pc,_PC,PC_COOKIE,PCJACOBI,comm);
  PLogObjectCreate(pc);
  pc->vec         = 0;
  pc->mat         = 0;
  pc->setupcalled = 0;
  pc->destroy     = 0;
  pc->data        = 0;
  pc->apply       = 0;
  pc->applytrans  = 0;
  pc->applyBA     = 0;
  pc->applyBAtrans= 0;
  pc->applyrich   = 0;
  pc->prefix      = 0;
  pc->view        = 0;
  pc->getfactmat  = 0;
  *newpc          = pc;
  /* this violates rule about seperating abstract from implementions*/
  return PCSetMethod(pc,PCJACOBI);
}

/*@
   PCApply - Applies the preconditioner to a vector.

   Input Parameters:
.  pc - the preconditioner context
.  x - input vector

   Output Parameter:
.  y - output vector

.keywords: PC, apply

.seealso: PCApplyTrans(), PCApplyBAorAB()
@*/
int PCApply(PC pc,Vec x,Vec y)
{
  int ierr;
  PETSCVALIDHEADERSPECIFIC(pc,PC_COOKIE);
  PLogEventBegin(PC_Apply,pc,x,y,0);
  ierr = (*pc->apply)(pc,x,y); CHKERRQ(ierr);
  PLogEventEnd(PC_Apply,pc,x,y,0);
  return 0;
}
/*@
   PCApplyTrans - Applies the transpose of preconditioner to a vector.

   Input Parameters:
.  pc - the preconditioner context
.  x - input vector

   Output Parameter:
.  y - output vector

.keywords: PC, apply, transpose

.seealso: PCApplyTrans(), PCApplyBAorAB(), PCApplyBAorABTrans()
@*/
int PCApplyTrans(PC pc,Vec x,Vec y)
{
  PETSCVALIDHEADERSPECIFIC(pc,PC_COOKIE);
  if (pc->applytrans) return (*pc->applytrans)(pc,x,y);
  SETERRQ(PETSC_ERR_SUP,"PCApplyTrans");
}

/*@
   PCApplyBAorAB - Applies the preconditioner and operator to a vector. 

   Input Parameters:
.  pc - the preconditioner context
.  right - indicates right or left preconditioner
.  x - input vector
.  work - work vector

   Output Parameter:
.  y - output vector

.keywords: PC, apply, operator

.seealso: PCApply(), PCApplyTrans(), PCApplyBAorABTrans()
@*/
int PCApplyBAorAB(PC pc,int right,Vec x,Vec y,Vec work)
{
  int ierr;
  PETSCVALIDHEADERSPECIFIC(pc,PC_COOKIE);
  if (pc->applyBA)  return (*pc->applyBA)(pc,right,x,y,work);
  if (right) {
    ierr = PCApply(pc,x,work); CHKERRQ(ierr);
    return MatMult(pc->mat,work,y); 
  }
  ierr = MatMult(pc->mat,x,work); CHKERRQ(ierr);
  return PCApply(pc,work,y);
}
/*@ 
   PCApplyBAorABTrans - Applies the transpose of the preconditioner
   and operator to a vector.

   Input Parameters:
.  pc - the preconditioner context
.  right - indicates right or left preconditioner
.  x - input vector
.  work - work vector

   Output Parameter:
.  y - output vector

.keywords: PC, apply, operator, transpose

.seealso: PCApply(), PCApplyTrans(), PCApplyBAorAB()
@*/
int PCApplyBAorABTrans(PC pc,int right,Vec x,Vec y,Vec work)
{
  int ierr;
  PETSCVALIDHEADERSPECIFIC(pc,PC_COOKIE);
  if (pc->applyBAtrans)  return (*pc->applyBAtrans)(pc,right,x,y,work);
  if (right) {
    ierr = MatMultTrans(pc->mat,x,work); CHKERRQ(ierr);
    return PCApplyTrans(pc,work,y);
  }
  ierr = PCApplyTrans(pc,x,work); CHKERRQ(ierr);
  return MatMultTrans(pc->mat,work,y); 
}

/*@
   PCApplyRichardsonExists - Determines if a particular preconditioner has a 
   built-in fast application of Richardson's method.

   Input Parameter:
.  pc - the preconditioner

.keywords: PC, apply, Richardson, exists

.seealso: PCApplyRichardson()
@*/
int PCApplyRichardsonExists(PC pc)
{
  if (pc->applyrich) return 1; else return 0;
}

/*@
   PCApplyRichardson - Applies several steps of Richardson iteration with 
   the particular preconditioner. This routine is usually used by the 
   Krylov solvers and not the application code directly.

   Input Parameters:
.  pc  - the preconditioner context
.  x   - the initial guess 
.  w   - one work vector
.  its - the number of iterations to apply.

   Output Parameter:
.  y - the solution

   Notes: 
   Most preconditioners do not support this function. Use the command
   PCApplyRichardsonExists() to determine if one does.

.keywords: PC, apply, Richardson

.seealso: PCApplyRichardsonExists()
@*/
int PCApplyRichardson(PC pc,Vec x,Vec y,Vec w,int its)
{
  PETSCVALIDHEADERSPECIFIC(pc,PC_COOKIE);
  if (!pc->applyrich) SETERRQ(PETSC_ERR_SUP,"PCApplyRichardson");
  return (*pc->applyrich)(pc,x,y,w,its);
}

/* 
      a setupcall of 0 indicates never setup, 
                     1 needs to be resetup,
                     2 does not need any changes.
*/
/*@
   PCSetUp - Prepares for the use of a preconditioner.

   Input parameters:
.  pc - the preconditioner context

.keywords: PC, setup

.seealso: PCCreate(), PCApply(), PCDestroy()
@*/
int PCSetUp(PC pc)
{
  int ierr;
  if (pc->setupcalled > 1) return 0;
  PLogEventBegin(PC_SetUp,pc,0,0,0);
  if (!pc->vec) {SETERRQ(1,"PCSetUp:Vector must be set first");}
  if (!pc->mat) {SETERRQ(1,"PCSetUp:Matrix must be set be set first");}
  if (pc->setup) { ierr = (*pc->setup)(pc); CHKERRQ(ierr);}
  pc->setupcalled = 2;
  PLogEventEnd(PC_SetUp,pc,0,0,0);
  return 0;
}

/*@
   PCSetOperators - Sets the matrix associated with the linear system and 
   a (possibly) different one associated with the preconditioner.

   Input Parameters:
.  pc - the preconditioner context
.  Amat - the matrix associated with the linear system
.  Pmat - matrix to be used in constructing preconditioner, usually the same
          as Amat.  If Pmat is 0 for repeated linear solves, the old 
          preconditioner is used.
.  flag - flag indicating information about matrix structure.  When solving
   just one linear system, this flag is NOT used and can thus be set to 0.

   Notes: 
   The flag can be used to eliminate unnecessary work in the repeated
   solution of linear systems of the same size.  The available options are
$    MAT_SAME_NONZERO_PATTERN - 
$       Amat has the same nonzero structure 
$       during successive linear solves
$    PMAT_SAME_NONZERO_PATTERN -
$       Pmat has the same nonzero structure 
$       during successive linear solves
$    ALLMAT_SAME_NONZERO_PATTERN -
$       Both Amat and Pmat have the same nonzero
$       structure during successive linear solves
$    ALLMAT_DIFFERENT_NONZERO_PATTERN -
$       Neither Amat nor Pmat has same nonzero structure

.keywords: PC, set, operators, matrix, linear system

.seealso: PCGetOperators()
 @*/
int PCSetOperators(PC pc,Mat Amat,Mat Pmat,MatStructure flag)
{
  PETSCVALIDHEADERSPECIFIC(pc,PC_COOKIE);
  pc->mat         = Amat;
  if (pc->setupcalled == 0 && !Pmat) {
    pc->pmat = Amat;
  }
  else if (pc->setupcalled && Pmat) {
    pc->pmat        = Pmat;
    pc->setupcalled = 1;  
  }
  else if (pc->setupcalled == 0) {
    pc->pmat = Pmat;
  }
  if (Pmat==Amat && (flag==MAT_SAME_NONZERO_PATTERN || flag==PMAT_SAME_NONZERO_PATTERN))
    pc->flag = ALLMAT_SAME_NONZERO_PATTERN;
  else pc->flag = flag;

  return 0;
}
/*@C
   PCGetOperators - Gets the matrix associated with the linear system and
   possibly a different one associated with the preconditioner.

   Input Parameter:
.  pc - the preconditioner context

   Output Parameters:
.  mat - the matrix associated with the linear system
.  pmat - matrix associated with the preconditioner, usually the same
          as mat.  If pmat is 0, the old preconditioner is used.
.  flag - flag indicating information about matrix structure

.keywords: PC, get, operators, matrix, linear system

.seealso: PCSetOperators()
@*/
int PCGetOperators(PC pc,Mat *mat,Mat *pmat,MatStructure *flag)
{
  PETSCVALIDHEADERSPECIFIC(pc,PC_COOKIE);
  if (!mat || !pmat || !flag) SETERRQ(PETSC_ERR_ARG,"PCGetOperators");
  *mat  = pc->mat;
  *pmat = pc->pmat;
  *flag = pc->flag;
  return 0;
}

/*@
   PCSetVector - Set a vector associated with the preconditioner.

   Input Parameters:
.  pc - the preconditioner context
.  vec - the vector

   Notes:
   The vector must be set so that the preconditioner knows what type
   of vector to allocate if necessary.

.keywords: PC, set, vector
@*/
int PCSetVector(PC pc,Vec vec)
{
  PETSCVALIDHEADERSPECIFIC(pc,PC_COOKIE);
  pc->vec = vec;
  return 0;
}

/*@C
   PCGetMethodFromContext - Gets the preconditioner method from an 
   active preconditioner context.

   Input Parameters:
.  pc - the preconditioner context

   Output parameters:
.  method - the method ID

.keywords: PC, get, method, context, type

.seealso: PCGetMethodName()
@*/
int PCGetMethodFromContext(PC pc,PCMethod *method)
{
  PETSCVALIDHEADERSPECIFIC(pc,PC_COOKIE);
  *method = (PCMethod) pc->type;
  return 0;
}

/*@C 
   PCGetFactoredMatrix - Gets the factored matrix from the
   preconditioner context.  This routine is valid only for the LU, 
   Incomplete LU, Cholesky and Incomplete Cholesky methods.

   Input Parameters:
.  pc - the preconditioner context

   Output parameters:
.  mat - the factored matrix

.keywords: PC, get, factored, matrix
@*/
int PCGetFactoredMatrix(PC pc,Mat *mat)
{
  PETSCVALIDHEADERSPECIFIC(pc,PC_COOKIE);
  if (pc->getfactmat) return (*pc->getfactmat)(pc,mat);
  return 0;
}

/*@C
   PCSetOptionsPrefix - Sets the prefix used for searching for all 
   PC options in the database.

   Input Parameters:
.  pc - the preconditioner context
.  prefix - the prefix string to prepend to all PC option requests

.keywords: PC, set, options, prefix, database
@*/
int PCSetOptionsPrefix(PC pc,char *prefix)
{
  pc->prefix = prefix;
  return 0;
}

int PCPreSolve(PC pc,KSP ksp)
{
  if (pc->presolve) return (*pc->presolve)(pc,ksp);
  else return 0;
}

int PCPostSolve(PC pc,KSP ksp)
{
  if (pc->postsolve) return (*pc->postsolve)(pc,ksp);
  else return 0;
}

/*@ 
   PCView - Prints the PC data structure.

   Input Parameters:
.  PC - the PC context
.  viewer - optional visualization context

   Note:
   The available visualization contexts include
$     STDOUT_VIEWER_SELF - standard output (default)
$     STDOUT_VIEWER_WORLD - synchronized standard
$       output where only the first processor opens
$       the file.  All other processors send their 
$       data to the first processor to print. 

   The user can open alternative vistualization contexts with
$    ViewerFileOpenASCII() - output to a specified file

.keywords: PC, view

.seealso: KSPView(), ViewerFileOpenASCII()
@*/
int PCView(PC pc,Viewer viewer)
{
  PetscObject vobj = (PetscObject) viewer;
  FILE *fd;
  char *cstring;
  int  fmt, ierr, mat_exists;
  PETSCVALIDHEADERSPECIFIC(pc,PC_COOKIE);
  if ((vobj->type == ASCII_FILE_VIEWER || vobj->type == ASCII_FILES_VIEWER) &&
     vobj->cookie == VIEWER_COOKIE) {
    ierr = ViewerFileGetPointer_Private(viewer,&fd); CHKERRQ(ierr);
    ierr = ViewerFileGetFormat_Private(viewer,&fmt); CHKERRQ(ierr);
    MPIU_fprintf(pc->comm,fd,"PC Object:\n");
    PCGetMethodName((PCMethod)pc->type,&cstring);
    MPIU_fprintf(pc->comm,fd,"  method: %s\n",cstring);
    if (pc->view) (*pc->view)((PetscObject)pc,viewer);
    PetscObjectExists((PetscObject)pc->mat,&mat_exists);
    ViewerFileSetFormat(viewer,FILE_FORMAT_INFO,0);
    if (mat_exists) {
      if (pc->pmat == pc->mat) {
        MPIU_fprintf(pc->comm,fd,"  linear system matrix = precond matrix:\n");
        ierr = MatView(pc->mat,viewer); CHKERRQ(ierr);
      } else {
        MPIU_fprintf(pc->comm,fd,"  linear system matrix:\n");
        ierr = MatView(pc->mat,viewer); CHKERRQ(ierr);
        PetscObjectExists((PetscObject)pc->pmat,&mat_exists);
        if (mat_exists) {
          MPIU_fprintf(pc->comm,fd,"  preconditioner matrix:\n");
          ierr = MatView(pc->mat,viewer); CHKERRQ(ierr);
        }
      }
    }
  }
  return 0;
}

