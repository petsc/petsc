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

#include "esi/petsc/matrix.h"

#undef __FUNCT__  
#define __FUNCT__ "PCSetUp_ESI"
static int PCSetUp_ESI(PC pc)
{
  PC_ESI                      *jac = (PC_ESI*)pc->data;
  int                         ierr;
  ::esi::Operator<double,int> *em;

  PetscFunctionBegin;
  ierr = MatESIWrap(pc->mat,&em);CHKERRQ(ierr);
  ierr = jac->epc->setOperator(*em);CHKERRQ(ierr);
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
  /* PC_ESI  *jac = (PC_ESI*)pc->data; */
  int     ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("ESI options");CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

extern PetscFList CCAList;

#undef __FUNCT__  
#define __FUNCT__ "PCESISetType"
/*@C
    PCESISetType - Given a PETSc matrix of type ESI loads the ESI constructor
          by name and wraps the ESI operator to look like a PETSc matrix.

   Collective on PC

   Input Parameters:
+   V - preconditioner object
-   name - name of the ESI constructor

   Level: intermediate

@*/
int PCESISetType(PC V,char *name)
{
  int                                        ierr;
  ::esi::Preconditioner<double,int>          *ve;
  ::esi::Preconditioner<double,int>::Factory *f,*(*r)(void);

  PetscFunctionBegin;
  ierr = PetscFListFind(V->comm,CCAList,name,(void(**)(void))&r);CHKERRQ(ierr);
  if (!r) SETERRQ1(1,"Unable to load esi::PreconditionerFactory constructor %s",name);
  f    = (*r)();

  ierr = f->create("MPI",(void*)&V->comm,ve);CHKERRQ(ierr);
  delete f;
  ierr = PCESISetPreconditioner(V,ve);CHKERRQ(ierr);
  ierr = ve->deleteReference();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCESISetFromOptions"
int PCESISetFromOptions(PC V)
{
  char       string[PETSC_MAX_PATH_LEN];
  PetscTruth flg;
  int        ierr;
 
  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)V,PCESI,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscOptionsGetString(V->prefix,"-pc_esi_type",string,1024,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PCESISetType(V,string);CHKERRQ(ierr);
    }
  }
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
  V->ops->destroy = 0;  /* since this is called from PCSetType() we have to make sure it doesn't get destroyed twice */
  ierr = PCSetType(V,PCESI);CHKERRQ(ierr);
  ierr = PCCreate(V->comm,&v);CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)v,"esi_");CHKERRQ(ierr);
  ierr = PCSetFromOptions(v);CHKERRQ(ierr);
  ve   = new esi::petsc::Preconditioner<double,int>(v);
  ierr = PCESISetPreconditioner(V,ve);CHKERRQ(ierr);
  ierr = ve->deleteReference();CHKERRQ(ierr);
  ierr = PetscObjectDereference((PetscObject)v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END
