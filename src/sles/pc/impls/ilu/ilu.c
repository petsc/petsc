#ifndef lint
static char vcid[] = "$Id: ilu.c,v 1.4 1995/03/25 01:26:33 bsmith Exp bsmith $";
#endif
/*
   Defines a direct factorization preconditioner for any Mat implementation
   Note: this need not be consided a preconditioner since it supplies
         a direct solver.
*/
#include "pcimpl.h"
#include "options.h"

typedef struct {
  Mat fact;
  int ordering,levels;
} PC_ILU;

/*@
      PCILUSetLevels - Sets the number of levels of fill to use.

  Input Parameters:
.   pc - the preconditioner context
.   levels - number of levels 
@*/
int PCILUSetLevels(PC pc,int levels)
{
  PC_ILU *dir;
  VALIDHEADER(pc,PC_COOKIE);
  if (levels < 0) SETERR(1,"Number of levels may not be negative");
  dir = (PC_ILU *) pc->data;
  if (pc->type != PCILU) return 0;
  dir->levels = levels;
  return 0;
}

/*@
      PCILUSetOrdering - Sets the ordering to use for a direct 
              factorization.

  Input Parameters:
.   pc - the preconditioner context
.   ordering - the type of ordering to use, one of 
$      ORDER_NATURAL - Natural 
$      ORDER_ND - Nested Dissection
$      ORDER_1WD - One-way Dissection
$      ORDER_RCM - Reverse Cuthill-McGee
$      ORDER_QMD - Quotient Minimum Degree
@*/
int PCILUSetOrdering(PC pc,int ordering)
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
  char      name[10];
  int       ordering = ORDER_ND, levels;
  if (OptionsGetString(0,pc->prefix,"-ilu_ordering",name,10)) {
    if (!strcmp(name,"nd")) ordering = ORDER_ND;
    else if (!strcmp(name,"natural")) ordering = ORDER_NATURAL;
    else if (!strcmp(name,"1wd")) ordering = ORDER_1WD;
    else if (!strcmp(name,"rcm")) ordering = ORDER_RCM;
    else if (!strcmp(name,"qmd")) ordering = ORDER_QMD;
    else fprintf(stderr,"Unknown order: %s\n",name);
    PCILUSetOrdering(pc,ordering);
  }
  if (OptionsGetInt(0,pc->prefix,"-ilu_levels",&levels)) {
    PCILUSetLevels(pc,levels);
  }
  return 0;
}

static int PCPrintHelp_ILU(PC pc)
{
  char *p;
  if (pc->prefix) p = pc->prefix; else p = "-";
  fprintf(stderr,"%silu_ordering name: ordering to reduce fill",p);
  fprintf(stderr," (nd,natural,1wd,rcm,qmd)\n");
  fprintf(stderr,"%silu_levels levels: levels of fill",p);
  return 0;
}

static int PCSetUp_ILU(PC pc)
{
  IS     row,col;
  int    ierr;
  PC_ILU *dir = (PC_ILU *) pc->data;
  ierr = MatGetReordering(pc->pmat,dir->ordering,&row,&col); CHKERR(ierr);
  if (!pc->setupcalled) {
    ierr = MatILUFactorSymbolic(pc->pmat,row,col,dir->levels,&dir->fact); CHKERR(ierr);
    PLogObjectParent(pc,dir->fact);
  }
  else if (!(pc->flag & MAT_SAME_NONZERO_PATTERN)) { 
    ierr = MatDestroy(dir->fact); CHKERR(ierr);
    ierr = MatILUFactorSymbolic(pc->pmat,row,col,dir->levels,&dir->fact); CHKERR(ierr);
  }
  ierr = MatLUFactorNumeric(pc->pmat,&dir->fact); CHKERR(ierr);
  return 0;
}

static int PCDestroy_ILU(PetscObject obj)
{
  PC        pc   = (PC) obj;
  PC_ILU *dir = (PC_ILU*) pc->data;

  MatDestroy(dir->fact);
  FREE(dir); 
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
  PC_ILU *dir = NEW(PC_ILU); CHKPTR(dir);
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
