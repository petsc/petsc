/*

*/
#include "src/ksp/pc/pcimpl.h"     /*I "petscpc.h" I*/

typedef struct _PC_FieldSplitLink *PC_FieldSplitLink;
struct _PC_FieldSplitLink {
  PC                pc;
  Vec               x,y;
  PetscInt          nfields;
  PetscInt          *fields;
  PC_FieldSplitLink next;
};

typedef struct {
  PetscInt          bs;
  PetscInt          nsplits;
  PC_FieldSplitLink head;
  Vec               *x,*y;
} PC_FieldSplit;

#undef __FUNCT__  
#define __FUNCT__ "PCView_FieldSplit"
static PetscErrorCode PCView_FieldSplit(PC pc,PetscViewer viewer)
{
  PC_FieldSplit     *jac = (PC_FieldSplit*)pc->data;
  PetscErrorCode    ierr;
  PetscTruth        iascii;
  PetscInt          i,j;
  PC_FieldSplitLink link = jac->head;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  FieldSplit: total splits = %D",jac->nsplits);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  Solver info for each split is in the following PC objects:\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    for (i=0; i<jac->nsplits; i++) {
      ierr = PetscViewerASCIIPrintf(viewer,"Split number %D Fields ",i);CHKERRQ(ierr);
      for (j=0; j<link->nfields; j++) {
        ierr = PetscViewerASCIIPrintf(viewer,"%D \n",link->fields[j]);CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
      ierr = PCView(link->pc,viewer);CHKERRQ(ierr);
      link = link->next;
    }
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  } else {
    SETERRQ1(PETSC_ERR_SUP,"Viewer type %s not supported for PCFieldSplit",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCSetUp_FieldSplit"
static PetscErrorCode PCSetUp_FieldSplit(PC pc)
{
  PC_FieldSplit     *jac  = (PC_FieldSplit*)pc->data;
  PetscErrorCode    ierr;
  PC_FieldSplitLink link = jac->head;

  PetscFunctionBegin;
  while (link) {
    ierr = PCSetUp(link->pc);CHKERRQ(ierr);
    link = link->next;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCApply_FieldSplit"
static PetscErrorCode PCApply_FieldSplit(PC pc,Vec x,Vec y)
{
  PC_FieldSplit     *jac = (PC_FieldSplit*)pc->data;
  PetscErrorCode    ierr;
  PC_FieldSplitLink link = jac->head;

  PetscFunctionBegin;
  ierr = VecStrideGatherAll(x,jac->x,INSERT_VALUES);CHKERRQ(ierr);
  while (link) {
    ierr = PCApply(link->pc,link->x,link->y);CHKERRQ(ierr);
    link = link->next;
  }
  ierr = VecStrideScatterAll(jac->y,y,INSERT_VALUES);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCDestroy_FieldSplit"
static PetscErrorCode PCDestroy_FieldSplit(PC pc)
{
  PC_FieldSplit     *jac = (PC_FieldSplit*)pc->data;
  PetscErrorCode    ierr;
  PC_FieldSplitLink link = jac->head;

  PetscFunctionBegin;
  while (link) {
    ierr = PCDestroy(link->pc);CHKERRQ(ierr);
    ierr = PetscFree2(link,link->fields);CHKERRQ(ierr);
  }
  if (jac->x) {ierr = PetscFree2(jac->x,jac->y);CHKERRQ(ierr);}
  ierr = PetscFree(jac);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCSetFromOptions_FieldSplit"
static PetscErrorCode PCSetFromOptions_FieldSplit(PC pc)
{
  PC_FieldSplit     *jac = (PC_FieldSplit*)pc->data;
  PetscErrorCode    ierr;
  PC_FieldSplitLink link = jac->head;

  PetscFunctionBegin;
  if (!link) { /* user never set fields, so set them ourselves */
    link = jac->head;
  }
  while (link) {
    ierr = PCApply(link->pc,link->x,link->y);CHKERRQ(ierr);
    link = link->next;
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------------------------------*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCFieldSplitSetFields_FieldSplit"
PetscErrorCode PCFieldSplitSetFields_FieldSplit(PC pc,PetscInt n,PetscInt *fields)
{
  PC_FieldSplit     *jac;
  PetscErrorCode    ierr;
  PetscInt          *myfields;
  PC_FieldSplitLink link,next = jac->head;

  PetscFunctionBegin;
  if (n <= 0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Negative number of fields requested");
  ierr = PetscMalloc2(1,struct _PC_FieldSplitLink,&link,n,PetscInt,&myfields);CHKERRQ(ierr);
  ierr = PetscMemcpy(myfields,fields,n*sizeof(PetscInt));CHKERRQ(ierr);
  link->fields  = myfields;
  link->nfields = n;

  if (!next) {
    jac->head = link;
  } else {
    while (next->next) {
      next = next->next;
    }
    next->next = link;
  }
  jac->nsplits++;
  PetscFunctionReturn(0);
}
EXTERN_C_END


EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCFieldSplitGetSubPC_FieldSplit"
PetscErrorCode PCFieldSplitGetSubPC_FieldSplit(PC pc,PetscInt *n,PC **subpc)
{
  PC_FieldSplit     *jac = (PC_FieldSplit*)pc->data;
  PetscErrorCode    ierr;
  PetscInt          cnt = 0;
  PC_FieldSplitLink link;

  PetscFunctionBegin;
  ierr = PetscMalloc(jac->nsplits*sizeof(PC*),subpc);CHKERRQ(ierr);
  while (link) {
    (*subpc)[cnt++] = link->pc;
    link = link->next;
  }
  if (cnt != jac->nsplits) SETERRQ2(PETSC_ERR_PLIB,"Corrupt PCFIELDSPLIT object: number splits in linked list %D in object %D",cnt,jac->nsplits);
  *n = jac->nsplits;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "PCFieldSplitSetFields"
/*@
    PCFieldSplitSetFields - Sets the fields for one particular split in the field split preconditioner

    Collective on PC

    Input Parameters:
+   pc  - the preconditioner context
.   n - the number of fields in this split
.   fields - the fields in this split

    Level: intermediate

.seealso: PCFieldSplitGetSubKSP(), PCFIELDSPLIT

@*/
PetscErrorCode PCFieldSplitSetFields(PC pc,PetscInt n, PetscInt *fields)
{
  PetscErrorCode ierr,(*f)(PC,PetscInt,PetscInt *);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCFieldSplitSetFields_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,n,fields);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCFieldSplitGetSubPC"
/*@C
   PCFieldSplitGetSubPC - Gets the PC contexts for all splits
   
   Collective on PC

   Input Parameter:
.  pc - the preconditioner context

   Output Parameters:
+  n - the number of split
-  pc - the array of PC contexts

   Note:  
   After PCFieldSplitGetSubPC() the array of PCs IS to be freed

   You must call KSPSetUp() before calling PCFieldSplitGetSubPC().

   Level: advanced

.keywords: PC, 

.seealso: PCFIELDSPLIT
@*/
PetscErrorCode PCFieldSplitGetSubPC(PC pc,PetscInt *n,PC *subpc[])
{
  PetscErrorCode ierr,(*f)(PC,PetscInt*,PC **);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  PetscValidIntPointer(n,2);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCFieldSplitGetSubPC_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,n,subpc);CHKERRQ(ierr);
  } else {
    SETERRQ(PETSC_ERR_ARG_WRONG,"Cannot get subpc for this type of PC");
  }

 PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------------------*/
/*MC
   PCFieldSplit - Preconditioner created by combining seperate preconditioners for individual
                  fields or groups of fields


     To set options on the solvers for each block append -sub_ to all the PC
        options database keys. For example, -sub_pc_type ilu -sub_pc_ilu_levels 1
        
     To set the options on the solvers seperate for each block call PCFieldSplitGetSubPC()
         and set the options directly on the resulting PC object

   Level: intermediate

   Concepts: physics based preconditioners

.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PC,
           PCFieldSplitGetSubPC(), PCFieldSplitSetFields()
M*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCCreate_FieldSplit"
PetscErrorCode PCCreate_FieldSplit(PC pc)
{
  PetscErrorCode ierr;
  PC_FieldSplit  *jac;

  PetscFunctionBegin;
  ierr = PetscNew(PC_FieldSplit,&jac);CHKERRQ(ierr);
  PetscLogObjectMemory(pc,sizeof(PC_FieldSplit));
  ierr = PetscMemzero(jac,sizeof(PC_FieldSplit));CHKERRQ(ierr);
  jac->bs      = -1;
  jac->nsplits = 0;
  pc->data     = (void*)jac;

  pc->ops->apply             = PCApply_FieldSplit;
  pc->ops->setup             = PCSetUp_FieldSplit;
  pc->ops->destroy           = PCDestroy_FieldSplit;
  pc->ops->setfromoptions    = PCSetFromOptions_FieldSplit;
  pc->ops->view              = PCView_FieldSplit;
  pc->ops->applyrichardson   = 0;

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFieldSplitGetSubPC_C","PCFieldSplitGetSubPC_FieldSplit",
                    PCFieldSplitGetSubPC_FieldSplit);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFieldSplitSetFields_C","PCFieldSplitSetFields_FieldSplit",
                    PCFieldSplitSetFields_FieldSplit);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END


