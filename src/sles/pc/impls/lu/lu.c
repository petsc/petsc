#ifndef lint
static char vcid[] = "$Id: lu.c,v 1.42 1995/08/28 18:35:50 curfman Exp bsmith $";
#endif
/*
   Defines a direct factorization preconditioner for any Mat implementation
   Note: this need not be consided a preconditioner since it supplies
         a direct solver.
*/
#include "pcimpl.h"
#include "pinclude/pviewer.h"
#if defined(HAVE_STRING_H)
#include <string.h>
#endif

typedef struct {
  Mat         fact;       /* factored matrix */
  int         inplace;    /* flag indicating in-place factorization */
  IS          row, col;   /* index sets used for reordering */
} PC_LU;


/*@
   PCLUSetUseInplace - Tells the system to do an in-place factorization.
   For some implementations, for instance, dense matrices, this enables the 
   solution of much larger problems. 

   Input Parameters:
.  pc - the preconditioner context

   Options Database Key:
$  -pc_lu_in_place

   Note:
   PCLUSetUseInplace() can only be used with the KSP method KSPPREONLY.
   This is because the Krylov space methods require an application of the 
   matrix multiplication, which is not possible here because the matrix has 
   been factored in-place, replacing the original matrix.

.keywords: PC, set, factorization, direct, inplace, in-place, LU, Cholesky

@*/
int PCLUSetUseInplace(PC pc)
{
  PC_LU *dir;
  PETSCVALIDHEADERSPECIFIC(pc,PC_COOKIE);
  dir = (PC_LU *) pc->data;
  if (pc->type != PCLU) return 0;
  dir->inplace = 1;
  return 0;
}

static int PCSetFromOptions_LU(PC pc)
{
  if (OptionsHasName(pc->prefix,"-pc_lu_in_place")) {
    PCLUSetUseInplace(pc);
  }
  return 0;
}

static int PCPrintHelp_LU(PC pc)
{
  char *p;
  if (pc->prefix) p = pc->prefix; else p = "-";
  MPIU_printf(pc->comm," Options for PCLU preconditioner:\n");
  MPIU_printf(pc->comm," %spc_lu_in_place: do factorization in place\n",p);
  MPIU_printf(pc->comm," -mat_order name: ordering to reduce fill",p);
  MPIU_printf(pc->comm," (nd,natural,1wd,rcm,qmd)\n");
  return 0;
}

static int PCView_LU(PetscObject obj,Viewer viewer)
{
  PC    pc = (PC)obj;
  FILE  *fd;
  PC_LU *lu = (PC_LU *) pc->data;
  int   ierr;
  ierr = ViewerFileGetPointer_Private(viewer,&fd); CHKERRQ(ierr);
  if (lu->inplace) MPIU_fprintf(pc->comm,fd,"  LU: in-place factorization\n");
  return 0;
}

static int PCGetFactoredMatrix_LU(PC pc,Mat *mat)
{
  PC_LU *dir = (PC_LU *) pc->data;
  *mat = dir->fact;
  return 0;
}

static int PCSetUp_LU(PC pc)
{
  int       ierr;
  PC_LU *dir = (PC_LU *) pc->data;
  MatType   type;

  ierr = MatGetType(pc->pmat,&type); CHKERRQ(ierr);
  if (dir->inplace) {
    ierr = MatGetReordering(pc->pmat,ORDER_ND,&dir->row,&dir->col); CHKERRQ(ierr);
    if (dir->row) {PLogObjectParent(pc,dir->row); PLogObjectParent(pc,dir->col);}

    /* this uses an arbritrary 5.0 as the fill factor! We should
       allow the user to set this!*/
    ierr = MatLUFactor(pc->pmat,dir->row,dir->col,5.0); CHKERRQ(ierr);
  }
  else {
    if (!pc->setupcalled) {
      ierr = MatGetReordering(pc->pmat,ORDER_ND,&dir->row,&dir->col); CHKERRQ(ierr);
      if (dir->row) {PLogObjectParent(pc,dir->row);PLogObjectParent(pc,dir->col);}
      ierr = MatLUFactorSymbolic(pc->pmat,dir->row,dir->col,5.0,&dir->fact); CHKERRQ(ierr);
      PLogObjectParent(pc,dir->fact);
    }
    else if (!(pc->flag & PMAT_SAME_NONZERO_PATTERN)) { 
      ierr = MatDestroy(dir->fact); CHKERRQ(ierr);
      ierr = MatGetReordering(pc->pmat,ORDER_ND,&dir->row,&dir->col); CHKERRQ(ierr);
      if (dir->row) {PLogObjectParent(pc,dir->row);PLogObjectParent(pc,dir->col);}
      ierr = MatLUFactorSymbolic(pc->pmat,dir->row,dir->col,5.0,&dir->fact); CHKERRQ(ierr);
      PLogObjectParent(pc,dir->fact);
    }
    ierr = MatLUFactorNumeric(pc->pmat,&dir->fact); CHKERRQ(ierr);
  }
  return 0;
}

static int PCDestroy_LU(PetscObject obj)
{
  PC        pc   = (PC) obj;
  PC_LU *dir = (PC_LU*) pc->data;

  if (!dir->inplace) MatDestroy(dir->fact);
  if (dir->row && dir->col && dir->row != dir->col) ISDestroy(dir->row);
  if (dir->col) ISDestroy(dir->col);
  PETSCFREE(dir); 
  return 0;
}

static int PCApply_LU(PC pc,Vec x,Vec y)
{
  PC_LU *dir = (PC_LU *) pc->data;
  if (dir->inplace) return MatSolve(pc->pmat,x,y);
  else  return MatSolve(dir->fact,x,y);
}

int PCCreate_LU(PC pc)
{
  PC_LU *dir = PETSCNEW(PC_LU); CHKPTRQ(dir);
  dir->fact      = 0;
  dir->inplace   = 0;
  dir->col       = 0;
  dir->row       = 0;
  pc->destroy    = PCDestroy_LU;
  pc->apply      = PCApply_LU;
  pc->setup      = PCSetUp_LU;
  pc->type       = PCLU;
  pc->data       = (void *) dir;
  pc->setfrom    = PCSetFromOptions_LU;
  pc->printhelp  = PCPrintHelp_LU;
  pc->view       = PCView_LU;
  pc->getfactmat = PCGetFactoredMatrix_LU;
  return 0;
}
