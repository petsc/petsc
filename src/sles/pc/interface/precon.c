#ifndef lint
static char vcid[] = "$Id: precon.c,v 1.85 1996/07/02 18:05:37 bsmith Exp bsmith $";
#endif
/*
    The PC (preconditioner) interface routines, callable by users.
*/
#include "pcimpl.h"            /*I "pc.h" I*/
#include "pinclude/pviewer.h"

extern int PCPrintTypes_Private(MPI_Comm,char*,char*);
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
  char p[64]; 
  PetscStrcpy(p,"-");
  if (pc->prefix) PetscStrcat(p,pc->prefix);
  PetscPrintf(pc->comm,"PC options ----------------------------------------\n");
  PCPrintTypes_Private(pc->comm,p,"pc_type");
  PetscPrintf(pc->comm,"Run program with %spc_type method -help for help on ",p);
  PetscPrintf(pc->comm,"a particular method\n");
  if (pc->printhelp) (*pc->printhelp)(pc,p);
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
  PetscValidHeaderSpecific(pc,PC_COOKIE);
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
  pc->type        = -1;
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
  pc->view        = 0;
  pc->getfactoredmatrix  = 0;
  *newpc          = pc;
  /* this violates rule about seperating abstract from implementions*/
  return PCSetType(pc,PCJACOBI);
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
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  if (x == y) SETERRQ(1,"PCApply:x and y must be different vectors");
  PLogEventBegin(PC_Apply,pc,x,y,0);
  ierr = (*pc->apply)(pc,x,y); CHKERRQ(ierr);
  PLogEventEnd(PC_Apply,pc,x,y,0);
  return 0;
}

/*@
   PCApplySymmetricLeft - Applies the left part of a symmetric preconditioner to a vector.

   Input Parameters:
.  pc - the preconditioner context
.  x - input vector

   Output Parameter:
.  y - output vector

   Notes:
   Currently, this routine is implemented only for PCICC and PCJACOBI preconditioners.

.keywords: PC, apply

.seealso: PCApply(), PCApplySymmetricRight()
@*/
int PCApplySymmetricLeft(PC pc,Vec x,Vec y)
{
  int ierr;
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  PLogEventBegin(PC_ApplySymmetricLeft,pc,x,y,0);
  ierr = (*pc->applysymmetricleft)(pc,x,y); CHKERRQ(ierr);
  PLogEventEnd(PC_ApplySymmetricLeft,pc,x,y,0);
  return 0;
}

/*@
   PCApplySymmetricRight - Applies the right part of a symmetric preconditioner to a vector.

   Input Parameters:
.  pc - the preconditioner context
.  x - input vector

   Output Parameter:
.  y - output vector

   Notes:
   Currently, this routine is implemented only for PCICC and PCJACOBI preconditioners.

.keywords: PC, apply

.seealso: PCApply(), PCApplySymmetricLeft()
@*/
int PCApplySymmetricRight(PC pc,Vec x,Vec y)
{
  int ierr;
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  PLogEventBegin(PC_ApplySymmetricRight,pc,x,y,0);
  ierr = (*pc->applysymmetricright)(pc,x,y); CHKERRQ(ierr);
  PLogEventEnd(PC_ApplySymmetricRight,pc,x,y,0);
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
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  if (x == y) SETERRQ(1,"PCApplyTrans:x and y must be different vectors");
  if (pc->applytrans) return (*pc->applytrans)(pc,x,y);
  SETERRQ(PETSC_ERR_SUP,"PCApplyTrans");
}

/*@
   PCApplyBAorAB - Applies the preconditioner and operator to a vector. 

   Input Parameters:
.  pc - the preconditioner context
.  side - indicates the preconditioner side, one of
$   PC_LEFT, PC_RIGHT, or PC_SYMMETRIC
.  x - input vector
.  work - work vector

   Output Parameter:
.  y - output vector

.keywords: PC, apply, operator

.seealso: PCApply(), PCApplyTrans(), PCApplyBAorABTrans()
@*/
int PCApplyBAorAB(PC pc, PCSide side,Vec x,Vec y,Vec work)
{
  int ierr;
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  if (x == y) SETERRQ(1,"PCApplyBAorAB:x and y must be different vectors");
  if (pc->applyBA)  return (*pc->applyBA)(pc,side,x,y,work);
  if (side == PC_RIGHT) {
    ierr = PCApply(pc,x,work); CHKERRQ(ierr);
    return MatMult(pc->mat,work,y); 
  }
  else if (side == PC_LEFT) {
    ierr = MatMult(pc->mat,x,work); CHKERRQ(ierr);
    return PCApply(pc,work,y);
  }
  else if (side == PC_SYMMETRIC) {
    /* There's an extra copy here; maybe should provide 2 work vectors instead? */
    ierr = PCApplySymmetricRight(pc,x,work); CHKERRQ(ierr);
    ierr = MatMult(pc->mat,work,y); CHKERRQ(ierr);
    ierr = VecCopy(y,work); CHKERRQ(ierr);
    return PCApplySymmetricLeft(pc,work,y);
  }
  else SETERRQ(1,"PCApplyBAorAB: Preconditioner side must be right, left, or symmetric");
}

/*@ 
   PCApplyBAorABTrans - Applies the transpose of the preconditioner
   and operator to a vector.

   Input Parameters:
.  pc - the preconditioner context
.  side - indicates the preconditioner side, one of
$   PC_LEFT, PC_RIGHT, or PC_SYMMETRIC
.  x - input vector
.  work - work vector

   Output Parameter:
.  y - output vector

.keywords: PC, apply, operator, transpose

.seealso: PCApply(), PCApplyTrans(), PCApplyBAorAB()
@*/
int PCApplyBAorABTrans(PC pc,PCSide side,Vec x,Vec y,Vec work)
{
  int ierr;
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  if (x == y) SETERRQ(1,"PCApplyBAorABTrans:x and y must be different vectors");
  if (pc->applyBAtrans)  return (*pc->applyBAtrans)(pc,side,x,y,work);
  if (side == PC_RIGHT) {
    ierr = MatMultTrans(pc->mat,x,work); CHKERRQ(ierr);
    return PCApplyTrans(pc,work,y);
  }
  else if (side == PC_LEFT) {
    ierr = PCApplyTrans(pc,x,work); CHKERRQ(ierr);
    return MatMultTrans(pc->mat,work,y); 
  }
  /* add support for PC_SYMMETRIC */
  else 
   SETERRQ(1,"PCApplyBAorABTrans: Only right and left preconditioning are currently supported");
}

/*@
   PCApplyRichardsonExists - Determines if a particular preconditioner has a 
   built-in fast application of Richardson's method.

   Input Parameter:
.  pc - the preconditioner

   Output Parameter:
.  exists - PETSC_TRUE or PETSC_FALSE

.keywords: PC, apply, Richardson, exists

.seealso: PCApplyRichardson()
@*/
int PCApplyRichardsonExists(PC pc, PetscTruth *exists)
{
  if (pc->applyrich) *exists = PETSC_TRUE; 
  else               *exists = PETSC_FALSE;
  return 0;
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
  PetscValidHeaderSpecific(pc,PC_COOKIE);
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
  if (pc->setupcalled > 1) {
    PLogInfo(pc,"Setting PC with identical preconditioner\n");
  }
  else if (pc->setupcalled == 0) {
    PLogInfo(pc,"Setting up new PC\n");
  }
  else if (pc->flag == SAME_NONZERO_PATTERN) {
    PLogInfo(pc,"Setting up PC with same nonzero pattern\n");
  }
  else {
    PLogInfo(pc,"Setting up PC with different nonzero pattern\n");
  }
  if (pc->setupcalled > 1) return 0;
  PLogEventBegin(PC_SetUp,pc,0,0,0);
  if (!pc->vec) {SETERRQ(1,"PCSetUp:Vector must be set first");}
  if (!pc->mat) {SETERRQ(1,"PCSetUp:Matrix must be set first");}
  if (pc->setup) { ierr = (*pc->setup)(pc); CHKERRQ(ierr);}
  pc->setupcalled = 2;
  PLogEventEnd(PC_SetUp,pc,0,0,0);
  return 0;
}

/*@
   PCSetUpOnBlocks - For block Jacobi, Gauss-Seidel and overlapping Schwarz 
        block methods sets up the preconditioner for each block.

   Input parameters:
.  pc - the preconditioner context

.keywords: PC, setup

.seealso: PCCreate(), PCApply(), PCDestroy(), PCSetUp()
@*/
int PCSetUpOnBlocks(PC pc)
{
  int ierr;
  if (!pc->setuponblocks) return 0;
  PLogEventBegin(PC_SetUpOnBlocks,pc,0,0,0);
  ierr = (*pc->setuponblocks)(pc); CHKERRQ(ierr);
  PLogEventEnd(PC_SetUpOnBlocks,pc,0,0,0);
  return 0;
}

/*@
   PCSetOperators - Sets the matrix associated with the linear system and 
   a (possibly) different one associated with the preconditioner.

   Input Parameters:
.  pc - the preconditioner context
.  Amat - the matrix associated with the linear system
.  Pmat - matrix to be used in constructing preconditioner, usually the same
          as Amat. 
.  flag - flag indicating information about the preconditioner matrix structure
   during successive linear solves. When solving just one linear system, this
   flag is ignored.

   Notes: 
   The flag can be used to eliminate unnecessary work in the preconditioner 
   during the repeated solution of linear systems of the same size.  The
   available options are
$    SAME_PRECONDITIONER -
$      Pmat is identical during successive linear solves.
$      This option is intended for folks who are using
$      different Amat and Pmat matrices and want to reuse the
$      same preconditioner matrix.  For example, this option
$      saves work by not recomputing incomplete factorization
$      for ILU/ICC preconditioners.
$    SAME_NONZERO_PATTERN -
$      Pmat has the same nonzero structure during
$      successive linear solves. 
$    DIFFERENT_NONZERO_PATTERN -
$      Pmat does not have the same nonzero structure.

    If in doubt about whether your preconditioner matrix has changed
    structure or not, use the flag DIFFERENT_NONZERO_PATTERN.

.keywords: PC, set, operators, matrix, linear system

.seealso: PCGetOperators()
 @*/
int PCSetOperators(PC pc,Mat Amat,Mat Pmat,MatStructure flag)
{
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  PetscValidHeaderSpecific(Amat,MAT_COOKIE);
  PetscValidHeaderSpecific(Pmat,MAT_COOKIE);

  pc->mat  = Amat;
  pc->pmat = Pmat;
  if (pc->setupcalled == 2 && flag != SAME_PRECONDITIONER) {
    pc->setupcalled = 1;
  }
  pc->flag = flag;
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
          as mat. 
.  flag - flag indicating information about the preconditioner
          matrix structure.  See PCSetOperators() for details.

.keywords: PC, get, operators, matrix, linear system

.seealso: PCSetOperators()
@*/
int PCGetOperators(PC pc,Mat *mat,Mat *pmat,MatStructure *flag)
{
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  if (mat) *mat  = pc->mat;
  if (pmat) *pmat = pc->pmat;
  if (flag) *flag = pc->flag;
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
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  pc->vec = vec;
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
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  if (pc->getfactoredmatrix) return (*pc->getfactoredmatrix)(pc,mat);
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
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  return PetscObjectSetPrefix((PetscObject)pc, prefix);
}
/*@C
   PCAppendOptionsPrefix - Adds to the prefix used for searching for all 
   PC options in the database.

   Input Parameters:
.  pc - the preconditioner context
.  prefix - the prefix string to prepend to all PC option requests

.keywords: PC, append, options, prefix, database
@*/
int PCAppendOptionsPrefix(PC pc,char *prefix)
{
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  return PetscObjectAppendPrefix((PetscObject)pc, prefix);
}

/*@
   PCGetOptionsPrefix - Gets the prefix used for searching for all 
   PC options in the database.

   Input Parameters:
.  pc - the preconditioner context

   Output Parameters:
.  prefix - pointer to the prefix string used, is returned

.keywords: PC, get, options, prefix, database
@*/
int PCGetOptionsPrefix(PC pc,char **prefix)
{
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  return PetscObjectGetPrefix((PetscObject)pc, prefix);
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
$     VIEWER_STDOUT_SELF - standard output (default)
$     VIEWER_STDOUT_WORLD - synchronized standard
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
  FILE        *fd;
  char        *cstr;
  int         fmt, ierr, mat_exists;
  ViewerType  vtype;

  PetscValidHeaderSpecific(pc,PC_COOKIE);
  ViewerGetType(viewer,&vtype);
  if (vtype  == ASCII_FILE_VIEWER || vtype == ASCII_FILES_VIEWER) {
    ierr = ViewerASCIIGetPointer(viewer,&fd); CHKERRQ(ierr);
    ierr = ViewerGetFormat(viewer,&fmt); CHKERRQ(ierr);
    PetscFPrintf(pc->comm,fd,"PC Object:\n");
    PCGetType(pc,PETSC_NULL,&cstr);
    PetscFPrintf(pc->comm,fd,"  method: %s\n",cstr);
    if (pc->view) (*pc->view)((PetscObject)pc,viewer);
    PetscObjectExists((PetscObject)pc->mat,&mat_exists);
    if (mat_exists) {
      int viewer_format;
      ierr = ViewerGetFormat(viewer,&viewer_format);
      ViewerSetFormat(viewer,ASCII_FORMAT_INFO,0);
      if (pc->pmat == pc->mat) {
        PetscFPrintf(pc->comm,fd,"  linear system matrix = precond matrix:\n");
        ierr = MatView(pc->mat,viewer); CHKERRQ(ierr);
      } else {
        PetscObjectExists((PetscObject)pc->pmat,&mat_exists);
        if (mat_exists) {
          PetscFPrintf(pc->comm,fd,"  linear system matrix followed by preconditioner matrix:\n");
        }
        else {
          PetscFPrintf(pc->comm,fd,"  linear system matrix:\n");
        }
        ierr = MatView(pc->mat,viewer); CHKERRQ(ierr);
        if (mat_exists) {ierr = MatView(pc->pmat,viewer); CHKERRQ(ierr);}
      }
      ViewerSetFormat(viewer,viewer_format,0);
    }
  }
  else if (vtype == STRING_VIEWER) {
    PCGetType(pc,PETSC_NULL,&cstr);
    ViewerStringSPrintf(viewer," %-7.7s",cstr);
    if (pc->view) (*pc->view)((PetscObject)pc,viewer);
  }
  return 0;
}


