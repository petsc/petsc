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

int PCiBJacobiMPIAIJSetup(PC);

static int PCiBJacobiSetup(PC pc)
{
  int        ierr;
  PCiBJacobi *jac = (PCiBJacobi *) pc->data;
  Mat        mat = pc->mat;
  if (mat->type == MATAIJMPI) {
    return PCiBJacobiMPIAIJSetup(pc);
  }
  else {
    SETERR(1,"Cannot use block Jacobi on this matrix type\n");
  }
  return 0;
}

int PCiBJacobiMPIAIJApply(PC,Vec,Vec);

/* default destroy, if it has never been setup */
static int PCiBJacobiDestroy(PetscObject obj)
{
  PC pc = (PC) obj;
  PCiBJacobi *jac = (PCiBJacobi *) pc->data;
  FREE(jac);
  return 0;
}

static int PCisetfrom(PC pc)
{
  PCiBJacobi *jac = (PCiBJacobi *) pc->data;
  int        blocks;

  if (OptionsGetInt(0,pc->prefix,"-bjacobi_blocks",&blocks)) {
    PCBJacobiSetBlocks(pc,blocks);
  }
  return 0;
}

int PCiprinthelp(PC pc)
{
  char *p;
  if (pc->prefix) p = pc->prefix; else p = "-";
  fprintf(stderr,"%sbjacobi_blocks blks: number of blocks in preconditioner\n",
                 p);
  return 0;
}

int PCiBJacobiCreate(PC pc)
{
  PCiBJacobi *jac = NEW(PCiBJacobi); CHKPTR(jac);
  pc->apply     = 0;
  pc->setup     = PCiBJacobiSetup;
  pc->destroy   = PCiBJacobiDestroy;
  pc->setfrom   = PCisetfrom;
  pc->printhelp = PCiprinthelp;
  pc->type      = PCBJACOBI;
  pc->data      = (void *) jac;
  jac->n        = 0;
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
  PCiBJacobi *jac = (PCiBJacobi *) pc->data; 
  VALIDHEADER(pc,PC_COOKIE);
  jac->n = blocks;
  return 0;
}
