#ifndef lint
static char vcid[] = "$Id: ilu.c,v 1.51 1996/01/02 20:15:29 bsmith Exp bsmith $";
#endif
/*
   Defines a ILU factorization preconditioner for any Mat implementation
*/
#include "pcimpl.h"                 /*I "pc.h"  I*/
#include "ilu.h"
#include "matimpl.h"
#include "pinclude/pviewer.h"

extern int PCSetUp_ILU_MPIRowbs(PC);

static int (*setups[])(PC) = {0,
                              0,
                              0,
                              0,
                              0,
                              0,
#if defined(HAVE_BLOCKSOLVE) && !defined(__cplusplus)
                              PCSetUp_ILU_MPIRowbs,
#else
                              0,
#endif
                              0,   
                              0,
                              0,0,0,0,0};

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
  if (levels < 0) SETERRQ(1,"PCILUSetLevels:negative levels");
  ilu = (PC_ILU *) pc->data;
  if (pc->type != PCILU) return 0;
  ilu->levels = levels;
  return 0;
}

/*@
   PCILUSetUseInPlace - Tells the system to do an in-place incomplete factorization.

   Input Parameters:
.  pc - the preconditioner context

   Options Database Key:
$  -pc_ilu_in_place

   Note:
   PCILUSetUseInPlace() is intended for use with matrix-free variants of
   Krylov methods, or when a different matrices are employed for the linear
   system and preconditioner.  Do NOT use this option if the linear system
   matrix also serves as the preconditioning matrix, since the factored
   matrix overwrites the original matrix. 

.keywords: PC, set, factorization, inplace, in-place, ILU

.seealso:  PCLUSetUseInPlace()
@*/
int PCILUSetUseInPlace(PC pc)
{
  PC_ILU *dir;
  PETSCVALIDHEADERSPECIFIC(pc,PC_COOKIE);
  dir = (PC_ILU *) pc->data;
  if (pc->type != PCILU) return 0;
  dir->inplace = 1;
  return 0;
}

static int PCSetFromOptions_ILU(PC pc)
{
  int         levels;
  if (OptionsGetInt(pc->prefix,"-pc_ilu_levels",&levels)) {
    PCILUSetLevels(pc,levels);
  }
  if (OptionsHasName(pc->prefix,"-pc_ilu_in_place")) {
    PCILUSetUseInPlace(pc);
  }
  return 0;
}

static int PCPrintHelp_ILU(PC pc,char *p)
{
  MPIU_printf(pc->comm," Options for PCILU preconditioner:\n");
  MPIU_printf(pc->comm," -mat_order name: ordering to reduce fill",p);
  MPIU_printf(pc->comm," (nd,natural,1wd,rcm,qmd)\n");
  MPIU_printf(pc->comm," %spc_ilu_levels levels: levels of fill\n",p);
  MPIU_printf(pc->comm," %spc_ilu_in_place: do factorization in place\n",p);
  MPIU_printf(pc->comm," %spc_ilu_factorpointwise: DO NOT use block factorization\n");
  MPIU_printf(pc->comm,"    (note this only applies to MatCreateMPIRowBS, all others\n");
  MPIU_printf(pc->comm,"    currently only support point factorization.\n");
  return 0;
}

static int PCView_ILU(PetscObject obj,Viewer viewer)
{
  PC     pc = (PC)obj;
  FILE   *fd;
  PC_ILU *ilu = (PC_ILU *) pc->data;
  int    ierr;
  char   *order;

  ierr = ViewerFileGetPointer_Private(viewer,&fd); CHKERRQ(ierr);
  if (ilu->levels == 1)
    MPIU_fprintf(pc->comm,fd,"    ILU: %d level of fill\n",ilu->levels);
  else
    MPIU_fprintf(pc->comm,fd,"    ILU: %d levels of fill\n",ilu->levels);
  if (ilu->inplace) MPIU_fprintf(pc->comm,fd,"         in-place factorization\n");
  else MPIU_fprintf(pc->comm,fd,"         out-of-place factorization\n");
  if (ilu->ordering == ORDER_NATURAL)  order = "Natural";
  else if (ilu->ordering == ORDER_ND)  order = "Nested Dissection";
  else if (ilu->ordering == ORDER_1WD) order = "One-way Dissection";
  else if (ilu->ordering == ORDER_RCM) order = "Reverse Cuthill-McGee";
  else if (ilu->ordering == ORDER_QMD) order = "Quotient Minimum Degree";
  else                                order = "unknown";
  MPIU_fprintf(pc->comm,fd,"         matrix ordering: %s\n",order);
  return 0;
}

static int PCSetUp_ILU(PC pc)
{
  int         ierr;
  double      f;
  PC_ILU      *ilu = (PC_ILU *) pc->data;

  if (ilu->inplace) {
    ierr = MatGetReorderingTypeFromOptions(0,&ilu->ordering); CHKERRQ(ierr);
    ierr = MatGetReordering(pc->pmat,ilu->ordering,&ilu->row,&ilu->col); CHKERRQ(ierr);
    if (ilu->row) {PLogObjectParent(pc,ilu->row); PLogObjectParent(pc,ilu->col);}
    if (setups[pc->pmat->type]) {
      ierr = (*setups[pc->pmat->type])(pc);
    }
    f = 1.0 + .5*ilu->levels;
    /* this uses an arbritrary 5.0 as the fill factor! User may set
       with the option -mat_ilu_fill */
    ierr = MatILUFactor(pc->pmat,ilu->row,ilu->col,f,ilu->levels); CHKERRQ(ierr);
    ilu->fact = pc->pmat;
  }
  else {
    if (!pc->setupcalled) {
      if (ilu->row) {ISDestroy(ilu->row); ISDestroy(ilu->col);}
      ierr = MatGetReorderingTypeFromOptions(0,&ilu->ordering); CHKERRQ(ierr);
      ierr = MatGetReordering(pc->pmat,ilu->ordering,&ilu->row,&ilu->col); CHKERRQ(ierr);
      if (ilu->row) {PLogObjectParent(pc,ilu->row); PLogObjectParent(pc,ilu->col);}
      if (setups[pc->pmat->type]) {
        ierr = (*setups[pc->pmat->type])(pc);
      }
      /* this uses an arbritrary 5.0 as the fill factor! User may set
         with the option -mat_ilu_fill */
      f = 1.0 + .5*ilu->levels;
      ierr = MatILUFactorSymbolic(pc->pmat,ilu->row,ilu->col,f,ilu->levels,
                                &ilu->fact); CHKERRQ(ierr);
      PLogObjectParent(pc,ilu->fact);
    }
    else if (!(pc->flag & PMAT_SAME_NONZERO_PATTERN)) { 
      if (ilu->row) {ISDestroy(ilu->row); ISDestroy(ilu->col);}
      ierr = MatGetReordering(pc->pmat,ilu->ordering,&ilu->row,&ilu->col); CHKERRQ(ierr);
      if (ilu->row) {PLogObjectParent(pc,ilu->row); PLogObjectParent(pc,ilu->col);}
      ierr = MatDestroy(ilu->fact); CHKERRQ(ierr);
      ierr = MatILUFactorSymbolic(pc->pmat,ilu->row,ilu->col,2.0,ilu->levels,
                                  &ilu->fact); CHKERRQ(ierr);
      PLogObjectParent(pc,ilu->fact);
    }
    ierr = MatLUFactorNumeric(pc->pmat,&ilu->fact); CHKERRQ(ierr);
  }
  return 0;
}

static int PCDestroy_ILU(PetscObject obj)
{
  PC     pc   = (PC) obj;
  PC_ILU *ilu = (PC_ILU*) pc->data;

  if (!ilu->inplace) MatDestroy(ilu->fact);
  if (ilu->row && ilu->col && ilu->row != ilu->col) ISDestroy(ilu->row);
  if (ilu->col) ISDestroy(ilu->col);
  PetscFree(ilu); 
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
  PC_ILU    *ilu = PetscNew(PC_ILU); CHKPTRQ(ilu);
  ilu->fact      = 0;
  ilu->levels    = 0;
  ilu->col       = 0;
  ilu->row       = 0;
  ilu->inplace   = 0;
  ilu->ordering  = ORDER_NATURAL;
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
