#ifndef lint
static char vcid[] = "$Id: bjacobi.c,v 1.8 1995/03/21 23:18:56 bsmith Exp bsmith $";
#endif
/*
   Defines a block Jacobi preconditioner.

   This is far from object oriented. We include the matrix 
  details in the code. This is because otherwise we would have to 
  include knowledge of SLES in the matrix code. In other words,
  it is a loop of objects, not a tree, so inheritence is a bad joke.
*/
#include "src/mat/matimpl.h"
#include "pcimpl.h"
#include "options.h"
#include "bjacobi.h"

int PCSetUp_BJacobiMPIAIJ(PC);

static int PCSetUp_BJacobi(PC pc)
{
  Mat        mat = pc->mat;
  if (mat->type == MATMPIAIJ) {
    return PCSetUp_BJacobiMPIAIJ(pc);
  }
  SETERR(1,"Cannot use block Jacobi on this matrix type\n");
}

int PCApply_BJacobiMPIAIJ(PC,Vec,Vec);

/* default destroy, if it has never been setup */
static int PCDestroy_BJacobi(PetscObject obj)
{
  PC pc = (PC) obj;
  PC_BJacobi *jac = (PC_BJacobi *) pc->data;
  FREE(jac);
  PLogObjectDestroy(pc);
  PETSCHEADERDESTROY(pc);
  return 0;
}

static int PCSetFromOptions_BJacobi(PC pc)
{
  int        blocks;

  if (OptionsGetInt(0,pc->prefix,"-bjacobi_blocks",&blocks)) {
    PCBJacobiSetBlocks(pc,blocks);
  }
  if (OptionsHasName(0,pc->prefix,"-bjacobi_truelocal")) {
    PCBJacobiSetUseTrueLocal(pc);
  }
  return 0;
}

/*@
      PCBJacobiSetUseTrueLocal - If the preconditioner is different from 
         the matrix, then if the block problem is solved iteratively
         this determines if the block problem is the block from the matrix
         or from the preconditioner.

  Input Parameters:
.  pc - the preconditioner context
@*/
int PCBJacobiSetUseTrueLocal(PC pc)
{
  PC_BJacobi   *jac;
  VALIDHEADER(pc,PC_COOKIE);
  if (pc->type != PCBJACOBI) return 0;
  jac = (PC_BJacobi *) pc->data;
  jac->usetruelocal = 1;
  return 0;
}
  
int PCPrintHelp_BJacobi(PC pc)
{
  char *p;
  if (pc->prefix) p = pc->prefix; else p = "-";
  fprintf(stderr,"%sbjacobi_blocks blks: number of blocks in preconditioner\n",
                 p);
  fprintf(stderr,"%sbjacobi_truelocal: use local blocks in local it. solve\n",
                 p);
  return 0;
}

int PCCreate_BJacobi(PC pc)
{
  PC_BJacobi   *jac = NEW(PC_BJacobi); CHKPTR(jac);
  pc->apply         = 0;
  pc->setup         = PCSetUp_BJacobi;
  pc->destroy       = PCDestroy_BJacobi;
  pc->setfrom       = PCSetFromOptions_BJacobi;
  pc->printhelp     = PCPrintHelp_BJacobi;
  pc->type          = PCBJACOBI;
  pc->data          = (void *) jac;
  jac->n            = 0;
  jac->usetruelocal = 0;
  return 0;
}
/*@
     PCBJacobiSetBlocks - Sets the number of blocks for block Jacobi.


  Input Parameters:
.   pc - the preconditioner context
.   blocks - the number of blocks
@*/
int PCBJacobiSetBlocks(PC pc, int blocks)
{
  PC_BJacobi *jac = (PC_BJacobi *) pc->data; 
  VALIDHEADER(pc,PC_COOKIE);
  jac->n = blocks;
  return 0;
}
