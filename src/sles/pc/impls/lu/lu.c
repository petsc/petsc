#ifndef lint
static char vcid[] = "$Id: direct.c,v 1.16 1995/04/17 02:25:27 bsmith Exp curfman $";
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
  int ordering,inplace;
} PC_Direct;

/*@
   PCDirectSetOrdering - Sets the ordering to use for a direct 
   factorization.

   Input Parameters:
.  pc - the preconditioner context
.  ordering - the type of ordering to use, one of the following:
$      ORDER_NATURAL - Natural 
$      ORDER_ND - Nested Dissection
$      ORDER_1WD - One-way Dissection
$      ORDER_RCM - Reverse Cuthill-McGee
$      ORDER_QMD - Quotient Minimum Degree

   Options Database Key:
$  -pc_direct_ordering <name>, where <name> is one of the following:
$      natural, nd, 1wd, rcm, qmd

.keywords: PC, set, ordering, factorization, direct, LU, Cholesky, fill
.keywords: natural, Nested Dissection, One-way Dissection
.keywords: Reverse Cuthill-McGee, Quotient Minimum Degree

.seealso: PCSetDirectUseInplace()
@*/
int PCDirectSetOrdering(PC pc,int ordering)
{
  PC_Direct *dir;
  VALIDHEADER(pc,PC_COOKIE);
  dir = (PC_Direct *) pc->data;
  if (pc->type != PCDIRECT) return 0;
  dir->ordering = ordering;
  return 0;
}
/*@
   PCDirectSetUseInplace - Tells the system to do an in-place factorization.
   For some implementations, for instance, dense matrices, this enables the 
   solution of much larger problems. 

   Input Parameters:
.  pc - the preconditioner context

   Options Database Key:
$  -pc_direct_in_place

   Note:
   PCDirectSetUseInplace() can only be used with the KSP method KSPPREONLY.
   This is because the Krylov space methods require an application of the 
   matrix multiplication, which is not possible here because the matrix has 
   been factored in-place, replacing the original matrix.

.keywords: PC, set, factorization, direct, inplace, in-place, LU, Cholesky

.seealso: PCDirectSetOrdering()
@*/
int PCDirectSetUseInplace(PC pc)
{
  PC_Direct *dir;
  VALIDHEADER(pc,PC_COOKIE);
  dir = (PC_Direct *) pc->data;
  if (pc->type != PCDIRECT) return 0;
  dir->inplace = 1;
  return 0;
}
static int PCSetFromOptions_Direct(PC pc)
{
  char      name[10];
  int       ordering = ORDER_ND;
  if (OptionsHasName(0,pc->prefix,"-pc_direct_in_place")) {
    PCDirectSetUseInplace(pc);
  }
  if (OptionsGetString(0,pc->prefix,"-pc_direct_ordering",name,10)) {
    if (!strcmp(name,"nd")) ordering = ORDER_ND;
    else if (!strcmp(name,"natural")) ordering = ORDER_NATURAL;
    else if (!strcmp(name,"1wd")) ordering = ORDER_1WD;
    else if (!strcmp(name,"rcm")) ordering = ORDER_RCM;
    else if (!strcmp(name,"qmd")) ordering = ORDER_QMD;
    else fprintf(stderr,"Unknown order: %s\n",name);
    PCDirectSetOrdering(pc,ordering);
  }
  return 0;
}

static int PCPrintHelp_Direct(PC pc)
{
  char *p;
  if (pc->prefix) p = pc->prefix; else p = "-";
  fprintf(stderr," %spc_direct_in_place: do factorization in place\n",p);
  fprintf(stderr," %spc_direct_ordering name: ordering to reduce fill",p);
  fprintf(stderr," (nd,natural,1wd,rcm,qmd)\n");
  return 0;
}

static int PCSetUp_Direct(PC pc)
{
  IS        row,col;
  int       ierr;
  PC_Direct *dir = (PC_Direct *) pc->data;
  if (dir->inplace) {
    ierr = MatGetReordering(pc->pmat,dir->ordering,&row,&col); CHKERR(ierr);
    PLogObjectParent(pc,row);PLogObjectParent(pc,col);
    if ((ierr = MatLUFactor(pc->pmat,row,col))) SETERR(ierr,0);
  }
  else {
    if (!pc->setupcalled) {
      ierr = MatGetReordering(pc->pmat,dir->ordering,&row,&col); CHKERR(ierr);
      PLogObjectParent(pc,row);PLogObjectParent(pc,col);
      ierr = MatLUFactorSymbolic(pc->pmat,row,col,&dir->fact); CHKERR(ierr);
      PLogObjectParent(pc,dir->fact);
    }
    else if (!(pc->flag & MAT_SAME_NONZERO_PATTERN)) { 
      ierr = MatDestroy(dir->fact); CHKERR(ierr);
      ierr = MatGetReordering(pc->pmat,dir->ordering,&row,&col); CHKERR(ierr);
      PLogObjectParent(pc,row);PLogObjectParent(pc,col);
      ierr = MatLUFactorSymbolic(pc->pmat,row,col,&dir->fact); CHKERR(ierr);
      PLogObjectParent(pc,dir->fact);
    }
    ierr = MatLUFactorNumeric(pc->pmat,&dir->fact); CHKERR(ierr);
  }
  return 0;
}

static int PCDestroy_Direct(PetscObject obj)
{
  PC        pc   = (PC) obj;
  PC_Direct *dir = (PC_Direct*) pc->data;

  if (!dir->inplace) MatDestroy(dir->fact);
  FREE(dir); 
  PLogObjectDestroy(pc);
  PETSCHEADERDESTROY(pc);
  return 0;
}

static int PCApply_Direct(PC pc,Vec x,Vec y)
{
  PC_Direct *dir = (PC_Direct *) pc->data;
  if (dir->inplace) return MatSolve(pc->pmat,x,y);
  else  return MatSolve(dir->fact,x,y);
}

int PCCreate_Direct(PC pc)
{
  PC_Direct *dir = NEW(PC_Direct); CHKPTR(dir);
  dir->fact     = 0;
  dir->ordering = ORDER_ND;
  dir->inplace  = 0;
  pc->destroy   = PCDestroy_Direct;
  pc->apply     = PCApply_Direct;
  pc->setup     = PCSetUp_Direct;
  pc->type      = PCDIRECT;
  pc->data      = (void *) dir;
  pc->setfrom   = PCSetFromOptions_Direct;
  pc->printhelp = PCPrintHelp_Direct;
  return 0;
}
