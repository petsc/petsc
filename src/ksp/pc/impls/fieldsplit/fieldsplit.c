/*

*/
#include "src/ksp/pc/pcimpl.h"     /*I "petscpc.h" I*/

typedef struct _PC_FieldSplitLink *PC_FieldSplitLink;
struct _PC_FieldSplitLink {
  KSP               ksp;
  Vec               x,y;
  PetscInt          nfields;
  PetscInt          *fields;
  PC_FieldSplitLink next;
};

typedef struct {
  PetscTruth        defaultsplit;
  PetscInt          bs;
  PetscInt          nsplits;
  Vec               *x,*y;
  Mat               *pmat;
  IS                *is;
  PC_FieldSplitLink head;
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
    ierr = PetscViewerASCIIPrintf(viewer,"  Solver info for each split is in the following KSP objects:\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    for (i=0; i<jac->nsplits; i++) {
      ierr = PetscViewerASCIIPrintf(viewer,"Split number %D Fields ",i);CHKERRQ(ierr);
      for (j=0; j<link->nfields; j++) {
        ierr = PetscViewerASCIIPrintf(viewer,"%D \n",link->fields[j]);CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
      ierr = KSPView(link->ksp,viewer);CHKERRQ(ierr);
      link = link->next;
    }
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  } else {
    SETERRQ1(PETSC_ERR_SUP,"Viewer type %s not supported for PCFieldSplit",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCFieldSplitSetDefaults"
static PetscErrorCode PCFieldSplitSetDefaults(PC pc)
{
  PC_FieldSplit     *jac  = (PC_FieldSplit*)pc->data;
  PetscErrorCode    ierr;
  PC_FieldSplitLink link = jac->head;
  PetscInt          i;

  PetscFunctionBegin;
  /* user has not split fields so use default */
  if (!link) { 
    if (jac->bs <= 0) {
      ierr   = MatGetBlockSize(pc->pmat,&jac->bs);CHKERRQ(ierr);
    }
    for (i=0; i<jac->bs; i++) {
      ierr = PCFieldSplitSetFields(pc,1,&i);CHKERRQ(ierr);
    }
    link              = jac->head;
    jac->defaultsplit = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "PCSetUp_FieldSplit"
static PetscErrorCode PCSetUp_FieldSplit(PC pc)
{
  PC_FieldSplit     *jac = (PC_FieldSplit*)pc->data;
  PetscErrorCode    ierr;
  PC_FieldSplitLink link;
  PetscInt          i,nsplit;
  MatStructure      flag = pc->flag;

  PetscFunctionBegin;
  ierr   = PCFieldSplitSetDefaults(pc);CHKERRQ(ierr);
  nsplit = jac->nsplits;
  link   = jac->head;

  /* get the matrices for each split */
  if (!jac->is) {
    if (jac->defaultsplit) {
      PetscInt rstart,rend,bs = nsplit;

      ierr = MatGetOwnershipRange(pc->pmat,&rstart,&rend);CHKERRQ(ierr);
      ierr = PetscMalloc(bs*sizeof(IS),&jac->is);CHKERRQ(ierr);
      for (i=0; i<bs; i++) {
	ierr = ISCreateStride(pc->comm,(rend-rstart)/bs,rstart+i,bs,&jac->is[i]);CHKERRQ(ierr);
      }
    } else {
      SETERRQ(PETSC_ERR_SUP,"Do not yet support nontrivial split");
    }
  }
  if (!jac->pmat) {
    ierr = MatGetSubMatrices(pc->pmat,nsplit,jac->is,jac->is,MAT_INITIAL_MATRIX,&jac->pmat);CHKERRQ(ierr);
  } else {
    ierr = MatGetSubMatrices(pc->pmat,nsplit,jac->is,jac->is,MAT_REUSE_MATRIX,&jac->pmat);CHKERRQ(ierr);
  }

  /* set up the individual PCs */
  i = 0;
  while (link) {
    ierr = KSPSetOperators(link->ksp,jac->pmat[i],jac->pmat[i],flag);CHKERRQ(ierr);
    ierr = KSPSetFromOptions(link->ksp);CHKERRQ(ierr);
    ierr = KSPSetUp(link->ksp);CHKERRQ(ierr);
    i++;
    link = link->next;
  }
  
  /* create work vectors for each split */
  if (jac->defaultsplit && !jac->x) {
    ierr = PetscMalloc2(nsplit,Vec,&jac->x,nsplit,Vec,&jac->y);CHKERRQ(ierr);
    link = jac->head;
    for (i=0; i<nsplit; i++) {
      Mat A;
      ierr      = KSPGetOperators(link->ksp,PETSC_NULL,&A,PETSC_NULL);CHKERRQ(ierr);
      ierr      = MatGetVecs(A,&link->x,&link->y);CHKERRQ(ierr);
      jac->x[i] = link->x;
      jac->y[i] = link->y;
      link      = link->next;
    }
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
    ierr = KSPSolve(link->ksp,link->x,link->y);CHKERRQ(ierr);
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
  PC_FieldSplitLink link = jac->head,next;

  PetscFunctionBegin;
  while (link) {
    ierr = KSPDestroy(link->ksp);CHKERRQ(ierr);
    if (link->x) {ierr = VecDestroy(link->x);CHKERRQ(ierr);}
    if (link->y) {ierr = VecDestroy(link->y);CHKERRQ(ierr);}
    next = link->next;
    ierr = PetscFree2(link,link->fields);CHKERRQ(ierr);
    link = next;
  }
  if (jac->x) {ierr = PetscFree2(jac->x,jac->y);CHKERRQ(ierr);}
  if (jac->pmat) {ierr = MatDestroyMatrices(jac->nsplits,&jac->pmat);CHKERRQ(ierr);}
  if (jac->is) {
    PetscInt i;
    for (i=0; i<jac->nsplits; i++) {ierr = ISDestroy(jac->is[i]);CHKERRQ(ierr);}
    ierr = PetscFree(jac->is);CHKERRQ(ierr);
  }
  ierr = PetscFree(jac);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCSetFromOptions_FieldSplit"
static PetscErrorCode PCSetFromOptions_FieldSplit(PC pc)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------------------------------*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCFieldSplitSetFields_FieldSplit"
PetscErrorCode PCFieldSplitSetFields_FieldSplit(PC pc,PetscInt n,PetscInt *fields)
{
  PC_FieldSplit     *jac = (PC_FieldSplit*)pc->data;
  PetscErrorCode    ierr;
  PC_FieldSplitLink link,next = jac->head;
  char              prefix[128];

  PetscFunctionBegin;
  if (n <= 0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Negative number of fields requested");
  ierr = PetscMalloc2(1,struct _PC_FieldSplitLink,&link,n,PetscInt,&link->fields);CHKERRQ(ierr);
  ierr = PetscMemcpy(link->fields,fields,n*sizeof(PetscInt));CHKERRQ(ierr);
  link->nfields = n;
  link->next    = PETSC_NULL;
  ierr          = KSPCreate(pc->comm,&link->ksp);CHKERRQ(ierr);
  ierr          = KSPSetType(link->ksp,KSPPREONLY);CHKERRQ(ierr);

  if (pc->prefix) {
    sprintf(prefix,"%s_fieldsplit_%d_",pc->prefix,jac->nsplits);
  } else {
    sprintf(prefix,"fieldsplit_%d_",jac->nsplits);
  }
  ierr = KSPSetOptionsPrefix(link->ksp,prefix);CHKERRQ(ierr);

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
#define __FUNCT__ "PCFieldSplitGetSubKSP_FieldSplit"
PetscErrorCode PCFieldSplitGetSubKSP_FieldSplit(PC pc,PetscInt *n,KSP **subksp)
{
  PC_FieldSplit     *jac = (PC_FieldSplit*)pc->data;
  PetscErrorCode    ierr;
  PetscInt          cnt = 0;
  PC_FieldSplitLink link;

  PetscFunctionBegin;
  ierr = PetscMalloc(jac->nsplits*sizeof(KSP*),subksp);CHKERRQ(ierr);
  while (link) {
    (*subksp)[cnt++] = link->ksp;
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
#define __FUNCT__ "PCFieldSplitGetSubKSP"
/*@C
   PCFieldSplitGetSubKSP - Gets the KSP contexts for all splits
   
   Collective on KSP

   Input Parameter:
.  pc - the preconditioner context

   Output Parameters:
+  n - the number of split
-  pc - the array of KSP contexts

   Note:  
   After PCFieldSplitGetSubKSP() the array of KSPs IS to be freed

   You must call KSPSetUp() before calling PCFieldSplitGetSubKSP().

   Level: advanced

.seealso: PCFIELDSPLIT
@*/
PetscErrorCode PCFieldSplitGetSubKSP(PC pc,PetscInt *n,KSP *subksp[])
{
  PetscErrorCode ierr,(*f)(PC,PetscInt*,KSP **);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  PetscValidIntPointer(n,2);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCFieldSplitGetSubKSP_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,n,subksp);CHKERRQ(ierr);
  } else {
    SETERRQ(PETSC_ERR_ARG_WRONG,"Cannot get subksp for this type of PC");
  }
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------------------*/
/*MC
   PCFieldSplit - Preconditioner created by combining seperate preconditioners for individual
                  fields or groups of fields


     To set options on the solvers for each block append -sub_ to all the PC
        options database keys. For example, -sub_pc_type ilu -sub_pc_ilu_levels 1
        
     To set the options on the solvers seperate for each block call PCFieldSplitGetSubKSP()
         and set the options directly on the resulting KSP object

   Level: intermediate

   Concepts: physics based preconditioners

.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PC,
           PCFieldSplitGetSubKSP(), PCFieldSplitSetFields()
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

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFieldSplitGetSubKSP_C","PCFieldSplitGetSubKSP_FieldSplit",
                    PCFieldSplitGetSubKSP_FieldSplit);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFieldSplitSetFields_C","PCFieldSplitSetFields_FieldSplit",
                    PCFieldSplitSetFields_FieldSplit);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END


