#ifndef lint
static char vcid[] = "$Id: ilu.c,v 1.37 1995/09/12 14:09:52 curfman Exp curfman $";
#endif
/*
   Defines a ILU factorization preconditioner for any Mat implementation
*/
#include "pcimpl.h"
#include "ilu.h"
#include "matimpl.h"
#include "pinclude/pviewer.h"
#if defined(HAVE_STRING_H)
#include <string.h>
#endif



/*@
   PCILUSetLevels - Sets the number of levels of fill to use.

   Input Parameters:
.  pc - the preconditioner context
.  levels - number of levels of fill

   Options Database Key:
$  -pc_ilu_levels  levels

.keywords: PC, levels, fill, factorization, incomplete, ILU

@*/
int PCILUSetLevels(PC pc,int levels)
{
  PC_ILU *ilu;
  PETSCVALIDHEADERSPECIFIC(pc,PC_COOKIE);
  if (pc->type != PCILU) return 0;
  if (levels < 0) SETERRQ(1,"PCILUSetLevels: levels cannot be negative");
  ilu = (PC_ILU *) pc->data;
  if (pc->type != PCILU) return 0;
  ilu->levels = levels;
  return 0;
}

static int PCSetFromOptions_ILU(PC pc)
{
  int         levels;
  if (OptionsGetInt(pc->prefix,"-pc_ilu_levels",&levels)) {
    PCILUSetLevels(pc,levels);
  }
  return 0;
}

static int PCPrintHelp_ILU(PC pc)
{
  char *p;
  if (pc->prefix) p = pc->prefix; else p = "-";
  MPIU_printf(pc->comm," Options for PCILU preconditioner:\n");
  MPIU_printf(pc->comm," mat_order name: ordering to reduce fill",p);
  MPIU_printf(pc->comm," (nd,natural,1wd,rcm,qmd)\n");
  MPIU_printf(pc->comm," %spc_ilu_levels levels: levels of fill\n",p);
  return 0;
}

static int PCView_ILU(PetscObject obj,Viewer viewer)
{
  PC     pc = (PC)obj;
  FILE   *fd;
  PC_ILU *lu = (PC_ILU *) pc->data;
  int    ierr;

  ierr = ViewerFileGetPointer_Private(viewer,&fd); CHKERRQ(ierr);
  if (lu->levels == 1)
    MPIU_fprintf(pc->comm,fd,"    ILU: %d level of fill\n",
    lu->levels);
  else
    MPIU_fprintf(pc->comm,fd,"    ILU: %d levels of fill\n",
    lu->levels);
  return 0;
}

#if defined(HAVE_BLOCKSOLVE) && !defined(__cplusplus)
extern int PCImplCreate_ILU_MPIRowbs(PC pc);
extern int PCImplDestroy_ILU_MPIRowbs(PC pc);
#endif

static int PCSetUp_ILU(PC pc)
{
  int    ierr;
  double f;
  PC_ILU *ilu = (PC_ILU *) pc->data;

  ierr = MatGetReordering(pc->pmat,ORDER_NATURAL,&ilu->row,&ilu->col); CHKERRQ(ierr);
  if (ilu->row) {PLogObjectParent(pc,ilu->row); PLogObjectParent(pc,ilu->col);}
  if (!pc->setupcalled) {
#if defined(HAVE_BLOCKSOLVE) && !defined(__cplusplus)
    if (pc->pmat->type == MATMPIROWBS) {
      ilu->ImplCreate = PCImplCreate_ILU_MPIRowbs;
    }
#endif
    /* this is a heuristic guess for how much fill there will be */
    if (ilu->ImplCreate) {ierr = (*ilu->ImplCreate)(pc); CHKERRQ(ierr);}
    f = 1.0 + .5*ilu->levels;
    ierr = MatILUFactorSymbolic(pc->pmat,ilu->row,ilu->col,f,ilu->levels,
           &ilu->fact); CHKERRQ(ierr);
    PLogObjectParent(pc,ilu->fact);
  }
  else if (!(pc->flag & PMAT_SAME_NONZERO_PATTERN)) { 
    ierr = MatDestroy(ilu->fact); CHKERRQ(ierr);
    ierr = MatILUFactorSymbolic(pc->pmat,ilu->row,ilu->col,2.0,ilu->levels,
           &ilu->fact); CHKERRQ(ierr);
    PLogObjectParent(pc,ilu->fact);
  }
  ierr = MatLUFactorNumeric(pc->pmat,&ilu->fact); CHKERRQ(ierr);
  return 0;
}

static int PCDestroy_ILU(PetscObject obj)
{
  PC     pc   = (PC) obj;
  PC_ILU *ilu = (PC_ILU*) pc->data;
  int    ierr;

  MatDestroy(ilu->fact);
  if (ilu->ImplDestroy) {ierr = (*ilu->ImplDestroy)(pc); CHKERRQ(ierr);}
  if (ilu->row && ilu->col && ilu->row != ilu->col) ISDestroy(ilu->row);
  if (ilu->col) ISDestroy(ilu->col);
  PETSCFREE(ilu); 
  return 0;
}

static int PCApply_ILU(PC pc,Vec x,Vec y)
{
  PC_ILU *ilu = (PC_ILU *) pc->data;
  return MatSolve(ilu->fact,x,y);
}

static int PCGetFactoredMatrix_ILU(PC pc,Mat *mat)
{
  PC_ILU *ilu = (PC_ILU *) pc->data;
  *mat = ilu->fact;
  return 0;
}

int PCCreate_ILU(PC pc)
{
  PC_ILU    *ilu = PETSCNEW(PC_ILU); CHKPTRQ(ilu);
  ilu->fact      = 0;
  ilu->levels    = 0;
  ilu->col       = 0;
  ilu->row       = 0;
  ilu->ImplCreate  = 0;
  ilu->ImplDestroy = 0;
  pc->destroy    = PCDestroy_ILU;
  pc->apply      = PCApply_ILU;
  pc->setup      = PCSetUp_ILU;
  pc->type       = PCILU;
  pc->data       = (void *) ilu;
  pc->setfrom    = PCSetFromOptions_ILU;
  pc->printhelp  = PCPrintHelp_ILU;
  pc->getfactmat = PCGetFactoredMatrix_ILU;
  pc->view       = PCView_ILU;
  return 0;
}
