#ifndef lint
static char vcid[] = "$Id: ilu.c,v 1.1 1995/03/10 20:16:43 bsmith Exp bsmith $";
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
} PCiILU;

/*@
      PCILUSetLevels - Sets the number of levels of fill to use.

  Input Parameters:
.   pc - the preconditioner context
.   levels - number of levels 
@*/
int PCILUSetLevels(PC pc,int levels)
{
  PCiILU *dir;
  VALIDHEADER(pc,PC_COOKIE);
  if (levels < 0) SETERR(1,"Number of levels may not be negative");
  dir = (PCiILU *) pc->data;
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
  PCiILU *dir;
  VALIDHEADER(pc,PC_COOKIE);
  dir = (PCiILU *) pc->data;
  if (pc->type != PCILU) return 0;
  dir->ordering = ordering;
  return 0;
}

static int PCisetfrom(PC pc)
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

static int PCiprinthelp(PC pc)
{
  char *p;
  if (pc->prefix) p = pc->prefix; else p = "-";
  fprintf(stderr,"%silu_ordering name: ordering to reduce fill",p);
  fprintf(stderr," (nd,natural,1wd,rcm,qmd)\n");
  fprintf(stderr,"%silu_levels levels: levels of fill",p);
  return 0;
}

static int PCiILUSetup(PC pc)
{
  IS     row,col;
  int    ierr;
  PCiILU *dir = (PCiILU *) pc->data;
  ierr = MatGetReordering(pc->pmat,dir->ordering,&row,&col); CHKERR(ierr);
  if (!pc->setupcalled) {
    ierr = MatILUFactorSymbolic(pc->pmat,row,col,dir->levels,&dir->fact); CHKERR(ierr);
  }
  else if (!(pc->flag & MAT_SAME_NONZERO_PATTERN)) { 
    ierr = MatDestroy(dir->fact); CHKERR(ierr);
    ierr = MatILUFactorSymbolic(pc->pmat,row,col,dir->levels,&dir->fact); CHKERR(ierr);
  }
  ierr = MatLUFactorNumeric(pc->pmat,&dir->fact); CHKERR(ierr);
  return 0;
}

static int PCiILUDestroy(PetscObject obj)
{
  PC        pc   = (PC) obj;
  PCiILU *dir = (PCiILU*) pc->data;

  MatDestroy(dir->fact);
  FREE(dir); 
  PETSCHEADERDESTROY(pc);
  return 0;
}

static int PCiILUApply(PC pc,Vec x,Vec y)
{
  PCiILU *dir = (PCiILU *) pc->data;
  return MatSolve(dir->fact,x,y);
}

int PCiILUCreate(PC pc)
{
  PCiILU *dir = NEW(PCiILU); CHKPTR(dir);
  dir->fact     = 0;
  dir->ordering = ORDER_ND;
  dir->levels   = 0;
  pc->destroy   = PCiILUDestroy;
  pc->apply     = PCiILUApply;
  pc->setup     = PCiILUSetup;
  pc->type      = PCILU;
  pc->data      = (void *) dir;
  pc->setfrom   = PCisetfrom;
  pc->printhelp = PCiprinthelp;
  return 0;
}
