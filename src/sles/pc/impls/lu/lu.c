#ifndef lint
static char vcid[] = "$Id: $";
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
} PCiDirect;

/*@
      PCDirectSetOrdering - Sets the ordering to use for a direct 
              factorization.

  Input Parameters:
.   pc - the preconditioner context
.   ordering - the type of ordering to use, one of ORDER_NATURAL,
.              ORDER_ND, ORDER_RCM, ORDER_1WD, ORDER_QMD.
@*/
int PCDirectSetOrdering(PC pc,int ordering)
{
  PCiDirect *dir;
  VALIDHEADER(pc,PC_COOKIE);
  dir = (PCiDirect *) pc->data;
  if (pc->type != PCDIRECT) return 0;
  dir->ordering = ordering;
  return 0;
}
/*@
      PCDirectSetUseInplace - tells system to do an inplace factorization.
              For some implementations, for instance, dense matrices,
              this can allow one to do much larger problems. This can 
              only be used with the KSP method, KSPPREONLY, because
              the Krylov space methods require an application of the 
              matrix multiply, which is not possible here because the 
              matrix has been factored inplace of the original matrix.

  Input Parameters:
.   pc - the preconditioner context
@*/
int PCDirectSetUseInplace(PC pc)
{
  PCiDirect *dir;
  VALIDHEADER(pc,PC_COOKIE);
  dir = (PCiDirect *) pc->data;
  if (pc->type != PCDIRECT) return 0;
  dir->inplace = 1;
  return 0;
}
static int PCisetfrom(PC pc)
{
  char      name[10];
  int       ordering = ORDER_ND;
  if (OptionsHasName(0,pc->prefix,"-direct_in_place")) {
    PCDirectSetUseInplace(pc);
  }
  if (OptionsGetString(0,pc->prefix,"-direct_ordering",name,10)) {
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

static int PCiprinthelp(PC pc)
{
  char *p;
  if (pc->prefix) p = pc->prefix; else p = "-";
  fprintf(stderr,"%sdirect_in_place: do factorization in place\n",p);
  fprintf(stderr,"%sdirect_ordering name: ordering to reduce fill",p);
  fprintf(stderr," (nd,natural,1wd,rcm,qmd)\n");
  return 0;
}

static int PCiDirectSetup(PC pc)
{
  IS        row,col;
  int       ierr;
  PCiDirect *dir = (PCiDirect *) pc->data;
  ierr = MatGetReordering(pc->mat,dir->ordering,&row,&col); CHKERR(ierr);
  if (dir->inplace) {
    if ((ierr = MatLUFactor(pc->mat,row,col))) SETERR(ierr,0);
  }
  else {
    if ((ierr = MatLUFactorSymbolic(pc->mat,row,col,&dir->fact)))
      SETERR(ierr,0);
    if ((ierr = MatLUFactorNumeric(pc->mat,&dir->fact))) SETERR(ierr,0);
  }
  return 0;
}

static int PCiDirectDestroy(PetscObject obj)
{
  PC        pc   = (PC) obj;
  PCiDirect *dir = (PCiDirect*) pc->data;

  if (!dir->inplace) MatDestroy(dir->fact);
  FREE(dir); FREE(pc);
  return 0;
}

static int PCiDirectApply(PC pc,Vec x,Vec y)
{
  PCiDirect *dir = (PCiDirect *) pc->data;
  if (dir->inplace) return MatSolve(pc->mat,x,y);
  else  return MatSolve(dir->fact,x,y);
}

int PCiDirectCreate(PC pc)
{
  PCiDirect *dir = NEW(PCiDirect); CHKPTR(dir);
  dir->fact     = 0;
  dir->ordering = ORDER_ND;
  dir->inplace  = 0;
  pc->destroy   = PCiDirectDestroy;
  pc->apply     = PCiDirectApply;
  pc->setup     = PCiDirectSetup;
  pc->type      = PCDIRECT;
  pc->data      = (void *) dir;
  pc->setfrom   = PCisetfrom;
  pc->printhelp = PCiprinthelp;
  return 0;
}
