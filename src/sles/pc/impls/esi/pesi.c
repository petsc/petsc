/*$Id: jacobi.c,v 1.75 2001/08/07 03:03:32 balay Exp $*/


#include "src/sles/pc/pcimpl.h"          /*I "petscpc.h" I*/
#include "esi/petsc/preconditioner.h"

/* 
   Private context (data structure) for the ESI
*/
typedef struct {
  esi::Preconditioner<double,int>  *epc;
} PC_ESI;

#undef __FUNCT__  
#define __FUNCT__ "PCESISetPreconditioner"
/*@C
  PCESISetPreconditioner - Takes a PETSc PC sets it to type ESI and 
  provides the ESI preconditioner that it wraps to look like a PETSc PC.

  Input Parameter:
. xin - The Petsc PC

  Output Parameter:
. v   - The ESI preconditioner

  Level: advanced

.keywords: PC, ESI
@*/
int PCESISetPreconditioner(PC xin,esi::Preconditioner<double,int> *v)
{
  PC_ESI     *x = (PC_ESI*)xin->data;
  PetscTruth tesi;
  int        ierr;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)xin,0,&tesi);CHKERRQ(ierr);
  if (tesi) {
    ierr = PCSetType(xin,PCESI);CHKERRQ(ierr);
  }
  ierr = PetscTypeCompare((PetscObject)xin,PCESI,&tesi);CHKERRQ(ierr);
  if (tesi) {
    x->epc  = v;
    v->addReference();
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCSetUp_ESI"
static int PCSetUp_ESI(PC pc)
{
  PC_ESI     *jac = (PC_ESI*)pc->data;
  int        ierr;

  PetscFunctionBegin;
  ierr = jac->epc->setup();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCApply_ESI"
static int PCApply_ESI(PC pc,Vec x,Vec y)
{
  PC_ESI                  *jac = (PC_ESI*)pc->data;
  esi::Vector<double,int> *xx,*yy;
  int                     ierr;

  PetscFunctionBegin;
  ierr = VecESIWrap(x,&xx);CHKERRQ(ierr);
  ierr = VecESIWrap(y,&yy);CHKERRQ(ierr);
  ierr = jac->epc->solve(*xx,*yy);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCApplySymmetricLeft_ESI"
static int PCApplySymmetricLeft_ESI(PC pc,Vec x,Vec y)
{
  int                     ierr;
  PC_ESI                  *jac = (PC_ESI*)pc->data;
  esi::Vector<double,int> *xx,*yy;

  PetscFunctionBegin;
  ierr = VecESIWrap(x,&xx);CHKERRQ(ierr);
  ierr = VecESIWrap(y,&yy);CHKERRQ(ierr);
  ierr = jac->epc->solveLeft(*xx,*yy);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCApplySymmetricRight_ESI"
static int PCApplySymmetricRight_ESI(PC pc,Vec x,Vec y)
{
  int                     ierr;
  PC_ESI                  *jac = (PC_ESI*)pc->data;
  esi::Vector<double,int> *xx,*yy;

  PetscFunctionBegin;
  ierr = VecESIWrap(x,&xx);CHKERRQ(ierr);
  ierr = VecESIWrap(y,&yy);CHKERRQ(ierr);
  ierr = jac->epc->solveRight(*xx,*yy);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCDestroy_ESI"
static int PCDestroy_ESI(PC pc)
{
  PC_ESI *jac = (PC_ESI*)pc->data;
  int    ierr;

  PetscFunctionBegin;
  ierr = jac->epc->deleteReference();
  /*
      Free the private data structure that was hanging off the PC
  */
  ierr = PetscFree(jac);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCSetFromOptions_ESI"
static int PCSetFromOptions_ESI(PC pc)
{
  PC_ESI  *jac = (PC_ESI*)pc->data;
  int     ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("ESI options");CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCCreate_ESI"
int PCCreate_ESI(PC pc)
{
  PC_ESI *jac;
  int    ierr;

  PetscFunctionBegin;

  /*
     Creates the private data structure for this preconditioner and
     attach it to the PC object.
  */
  ierr      = PetscNew(PC_ESI,&jac);CHKERRQ(ierr);
  pc->data  = (void*)jac;

  /*
     Logs the memory usage; this is not needed but allows PETSc to 
     monitor how much memory is being used for various purposes.
  */
  PetscLogObjectMemory(pc,sizeof(PC_ESI));

  /*
      Set the pointers for the functions that are provided above.
      Now when the user-level routines (such as PCApply(), PCDestroy(), etc.)
      are called, they will automatically call these functions.  Note we
      choose not to provide a couple of these functions since they are
      not needed.
  */
  pc->ops->apply               = PCApply_ESI;
  pc->ops->applytranspose      = 0;
  pc->ops->setup               = PCSetUp_ESI;
  pc->ops->destroy             = PCDestroy_ESI;
  pc->ops->setfromoptions      = PCSetFromOptions_ESI;
  pc->ops->view                = 0;
  pc->ops->applyrichardson     = 0;
  pc->ops->applysymmetricleft  = PCApplySymmetricLeft_ESI;
  pc->ops->applysymmetricright = PCApplySymmetricRight_ESI;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCCreate_PetscESI"
int PCCreate_PetscESI(PC V)
{
  int                                    ierr;
  PC                                     v;
  esi::petsc::Preconditioner<double,int> *ve;

  PetscFunctionBegin;
  V->ops->destroy = 0;  /* since this is called from MatSetType() we have to make sure it doesn't get destroyed twice */
  ierr = PCSetType(V,PCESI);CHKERRQ(ierr);
  ierr = PCCreate(V->comm,&v);CHKERRQ(ierr);
  ierr = PCSetType(v,PCNONE);CHKERRQ(ierr);
  ve   = new esi::petsc::Preconditioner<double,int>(v);
  ierr = PCESISetPreconditioner(V,ve);CHKERRQ(ierr);
  ierr = ve->deleteReference();CHKERRQ(ierr);
  ierr = PetscObjectDereference((PetscObject)v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END
