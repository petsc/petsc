#ifndef lint
static char vcid[] = "$Id: ilu.c,v 1.17 1995/06/08 03:08:46 bsmith Exp bsmith $";
#endif
/*
   Defines a direct factorization preconditioner for any Mat implementation
   Note: this need not be consided a preconditioner since it supplies
         a direct solver.
*/
#include "pcimpl.h"

typedef struct {
  Mat         fact;
  MatOrdering ordering;
  int         levels;
} PC_ILU;

/*@
   PCILUSetLevels - Sets the number of levels of fill to use.

   Input Parameters:
.  pc - the preconditioner context
.  levels - number of levels of fill

   Options Database Key:
$  -pc_ilu_levels  levels

.keywords: PC, levels, fill, factorization, incomplete, ILU

.seealso: PCILUSetOrdering()
@*/
int PCILUSetLevels(PC pc,int levels)
{
  PC_ILU *dir;
  VALIDHEADER(pc,PC_COOKIE);
  if (levels < 0) SETERRQ(1,"PCILUSetLevels: levels cannot be negative");
  dir = (PC_ILU *) pc->data;
  if (pc->type != PCILU) return 0;
  dir->levels = levels;
  return 0;
}

/*@
   PCILUSetOrdering - Sets the ordering to use for a direct factorization.

  Input Parameters:
.   pc - the preconditioner context
.   ordering - the type of ordering to use, one of:
$     ORDER_NATURAL - Natural 
$     ORDER_ND - Nested Dissection
$     ORDER_1WD - One-way Dissection
$     ORDER_RCM - Reverse Cuthill-McGee
$     ORDER_QMD - Quotient Minimum Degree

  Options Database Key:
$ -pc_ilu_ordering  <name>, where <name> is one of the following:
$     natural, nd, 1wd, rcm, qmd

.keywords: PC, ordering, reordering, factorization, incomplete, ILU, fill

.seealso: PCILUSetLevels()
@*/
int PCILUSetOrdering(PC pc,MatOrdering ordering)
{
  PC_ILU *dir;
  VALIDHEADER(pc,PC_COOKIE);
  dir = (PC_ILU *) pc->data;
  if (pc->type != PCILU) return 0;
  dir->ordering = ordering;
  return 0;
}

static int PCSetFromOptions_ILU(PC pc)
{
  char        name[10];
  MatOrdering ordering = ORDER_ND;
  int         levels;
  if (OptionsGetString(pc->prefix,"-pc_ilu_ordering",name,10)) {
    if (!strcmp(name,"nd")) ordering = ORDER_ND;
    else if (!strcmp(name,"natural")) ordering = ORDER_NATURAL;
    else if (!strcmp(name,"1wd")) ordering = ORDER_1WD;
    else if (!strcmp(name,"rcm")) ordering = ORDER_RCM;
    else if (!strcmp(name,"qmd")) ordering = ORDER_QMD;
    else fprintf(stderr,"Unknown order: %s\n",name);
    PCILUSetOrdering(pc,ordering);
  }
  if (OptionsGetInt(pc->prefix,"-pc_ilu_levels",&levels)) {
    PCILUSetLevels(pc,levels);
  }
  return 0;
}

static int PCPrintHelp_ILU(PC pc)
{
  char *p;
  if (pc->prefix) p = pc->prefix; else p = "-";
  fprintf(stderr," %spc_ilu_ordering name: ordering to reduce fill",p);
  fprintf(stderr," (nd,natural,1wd,rcm,qmd)\n");
  fprintf(stderr," %spc_ilu_levels levels: levels of fill",p);
  return 0;
}

static int PCSetUp_ILU(PC pc)
{
  IS     row,col;
  int    ierr;
  PC_ILU *dir = (PC_ILU *) pc->data;
  ierr = MatGetReordering(pc->pmat,dir->ordering,&row,&col); CHKERRQ(ierr);
  if (!pc->setupcalled) {
    ierr = MatILUFactorSymbolic(pc->pmat,row,col,dir->levels,&dir->fact); CHKERRQ(ierr);
    PLogObjectParent(pc,dir->fact);
  }
  else if (!(pc->flag & PMAT_SAME_NONZERO_PATTERN)) { 
    ierr = MatDestroy(dir->fact); CHKERRQ(ierr);
    ierr = MatILUFactorSymbolic(pc->pmat,row,col,dir->levels,&dir->fact); CHKERRQ(ierr);
  }
  ierr = MatLUFactorNumeric(pc->pmat,&dir->fact); CHKERRQ(ierr);
  return 0;
}

static int PCDestroy_ILU(PetscObject obj)
{
  PC        pc   = (PC) obj;
  PC_ILU *dir = (PC_ILU*) pc->data;

  MatDestroy(dir->fact);
  PETSCFREE(dir); 
  PLogObjectDestroy(pc);
  PETSCHEADERDESTROY(pc);
  return 0;
}

static int PCApply_ILU(PC pc,Vec x,Vec y)
{
  PC_ILU *dir = (PC_ILU *) pc->data;
  return MatSolve(dir->fact,x,y);
}

int PCCreate_ILU(PC pc)
{
  PC_ILU *dir = PETSCNEW(PC_ILU); CHKPTRQ(dir);
  dir->fact     = 0;
  dir->ordering = ORDER_ND;
  dir->levels   = 0;
  pc->destroy   = PCDestroy_ILU;
  pc->apply     = PCApply_ILU;
  pc->setup     = PCSetUp_ILU;
  pc->type      = PCILU;
  pc->data      = (void *) dir;
  pc->setfrom   = PCSetFromOptions_ILU;
  pc->printhelp = PCPrintHelp_ILU;
  return 0;
}
