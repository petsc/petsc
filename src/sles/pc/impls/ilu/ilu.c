#ifndef lint
static char vcid[] = "$Id: ilu.c,v 1.32 1995/08/24 22:27:52 bsmith Exp curfman $";
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
  int         levels;     /* levels of fill */
  IS          row, col;   /* index sets used for reordering */
} PC_ILU;

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
  PC_ILU *dir;
  PETSCVALIDHEADERSPECIFIC(pc,PC_COOKIE);
  if (pc->type != PCILU) return 0;
  if (levels < 0) SETERRQ(1,"PCILUSetLevels: levels cannot be negative");
  dir = (PC_ILU *) pc->data;
  if (pc->type != PCILU) return 0;
  dir->levels = levels;
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
  FILE   *fd = ViewerFileGetPointer_Private(viewer);
  PC_ILU *lu = (PC_ILU *) pc->data;
  if (lu->levels == 1)
    MPIU_fprintf(pc->comm,fd,"    ILU: %d level of fill\n",
    lu->levels);
  else
    MPIU_fprintf(pc->comm,fd,"    ILU: %d levels of fill\n",
    lu->levels);
  return 0;
}

static int PCSetUp_ILU(PC pc)
{
  int    ierr;
  double f;

  PC_ILU *dir = (PC_ILU *) pc->data;
  ierr = MatGetReordering(pc->pmat,ORDER_NATURAL,&dir->row,&dir->col); CHKERRQ(ierr);
  if (dir->row) {PLogObjectParent(pc,dir->row); PLogObjectParent(pc,dir->col);}
  if (!pc->setupcalled) {
    /* this is a heuristic guess for how much fill there will be */
    f = 1.0 + .5*dir->levels;
    ierr = MatILUFactorSymbolic(pc->pmat,dir->row,dir->col,f,dir->levels,
           &dir->fact); CHKERRQ(ierr);
    PLogObjectParent(pc,dir->fact);
  }
  else if (!(pc->flag & PMAT_SAME_NONZERO_PATTERN)) { 
    ierr = MatDestroy(dir->fact); CHKERRQ(ierr);
    ierr = MatILUFactorSymbolic(pc->pmat,dir->row,dir->col,2.0,dir->levels,
           &dir->fact); CHKERRQ(ierr);
    PLogObjectParent(pc,dir->fact);
  }
  ierr = MatLUFactorNumeric(pc->pmat,&dir->fact); CHKERRQ(ierr);
  return 0;
}

static int PCDestroy_ILU(PetscObject obj)
{
  PC        pc   = (PC) obj;
  PC_ILU *dir = (PC_ILU*) pc->data;

  MatDestroy(dir->fact);
  if (dir->row && dir->col && dir->row != dir->col) ISDestroy(dir->row);
  if (dir->col) ISDestroy(dir->col);
  PETSCFREE(dir); 
  return 0;
}

static int PCApply_ILU(PC pc,Vec x,Vec y)
{
  PC_ILU *dir = (PC_ILU *) pc->data;
  return MatSolve(dir->fact,x,y);
}

static int PCGetFactoredMatrix_ILU(PC pc,Mat *mat)
{
  PC_ILU *dir = (PC_ILU *) pc->data;
  *mat = dir->fact;
  return 0;
}

int PCCreate_ILU(PC pc)
{
  PC_ILU *dir = PETSCNEW(PC_ILU); CHKPTRQ(dir);
  dir->fact      = 0;
  dir->levels    = 0;
  dir->col       = 0;
  dir->row       = 0;
  pc->destroy    = PCDestroy_ILU;
  pc->apply      = PCApply_ILU;
  pc->setup      = PCSetUp_ILU;
  pc->type       = PCILU;
  pc->data       = (void *) dir;
  pc->setfrom    = PCSetFromOptions_ILU;
  pc->printhelp  = PCPrintHelp_ILU;
  pc->getfactmat = PCGetFactoredMatrix_ILU;
  pc->view       = PCView_ILU;
  return 0;
}
