#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: redundant.c,v 1.11 1999/10/13 20:38:02 bsmith Exp bsmith $";
#endif
/*
  This file defines a "solve the problem redundantly on each processor" preconditioner.

*/
#include "src/sles/pc/pcimpl.h"     /*I "pc.h" I*/
#include "sles.h"

typedef struct {
  PC         pc;                    /* actual preconditioner used on each processor */
  Vec        x,b;                   /* sequential vectors to hold parallel vectors */
  Mat        *mats,*pmats;          /* matrix and optional preconditioner matrix */
  VecScatter scatterin,scatterout;  /* scatter used to move all values to each processor */
  PetscTruth useparallelmat;
} PC_Redundant;

#undef __FUNC__  
#define __FUNC__ "PCView_Redundant"
static int PCView_Redundant(PC pc,Viewer viewer)
{
  PC_Redundant *red = (PC_Redundant *) pc->data;
  int          ierr;
  PetscTruth   isascii,isstring;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,ASCII_VIEWER,&isascii);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,STRING_VIEWER,&isstring);CHKERRQ(ierr);
  if (isascii) {
    ierr = ViewerASCIIPrintf(viewer,"  Redundant solver preconditioner: Actual PC follows\n");CHKERRQ(ierr);
    ierr = ViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    ierr = PCView(red->pc,viewer);CHKERRQ(ierr);
    ierr = ViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  } else if (isstring) {
    ierr = ViewerStringSPrintf(viewer," Redundant solver preconditioner");CHKERRQ(ierr);
    ierr = PCView(red->pc,viewer);CHKERRQ(ierr);
  } else {
    SETERRQ1(1,1,"Viewer type %s not supported for PC redundant",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCSetUp_Redundant"
static int PCSetUp_Redundant(PC pc)
{
  PC_Redundant   *red  = (PC_Redundant *) pc->data;
  int            ierr,mstart,mlocal,m,size;
  IS             isl;
  MatReuse       reuse = MAT_INITIAL_MATRIX;
  MatStructure   str   = DIFFERENT_NONZERO_PATTERN;
  MPI_Comm       comm;

  PetscFunctionBegin;
  ierr = VecGetSize(pc->vec,&m);CHKERRQ(ierr);
  if (pc->setupcalled == 0) {
    ierr = VecGetLocalSize(pc->vec,&mlocal);CHKERRQ(ierr);
    ierr = VecCreateSeq(PETSC_COMM_SELF,m,&red->x);CHKERRQ(ierr);
    ierr = VecDuplicate(red->x,&red->b);CHKERRQ(ierr);
    ierr = PCSetVector(red->pc,red->x);CHKERRQ(ierr);
    if (!red->scatterin) {

      /*
         Create the vectors and vector scatter to get the entire vector onto each processor
      */
      ierr = VecGetOwnershipRange(pc->vec,&mstart,PETSC_NULL);CHKERRQ(ierr);
      ierr = VecScatterCreate(pc->vec,0,red->x,0,&red->scatterin);CHKERRQ(ierr);
      ierr = ISCreateStride(pc->comm,mlocal,mstart,1,&isl);CHKERRQ(ierr);
      ierr = VecScatterCreate(red->x,isl,pc->vec,isl,&red->scatterout);CHKERRQ(ierr);
      ierr = ISDestroy(isl);CHKERRQ(ierr);
    }
  }

  /* if pmatrix set by user is sequential then we do not need to gather the parallel matrix*/

  ierr = PetscObjectGetComm((PetscObject)pc->pmat,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);CHKERRQ(ierr);
  if (size == 1) {
    red->useparallelmat = PETSC_FALSE;
  }

  if (red->useparallelmat) {
    if (pc->setupcalled == 1 && pc->flag == DIFFERENT_NONZERO_PATTERN) {
      /* destroy old matrices */
      if (red->pmats && red->pmats != red->mats) {
        ierr = MatDestroyMatrices(1,&red->pmats);CHKERRQ(ierr);
      }
      if (red->mats) {
        ierr = MatDestroyMatrices(1,&red->mats);CHKERRQ(ierr);
      }   
    } else if (pc->setupcalled == 1) {
      reuse = MAT_REUSE_MATRIX;
      str   = SAME_NONZERO_PATTERN;
    }
        
    /* 
       grab the parallel matrix and put it on each processor
    */
    ierr = ISCreateStride(PETSC_COMM_SELF,m,0,1,&isl);CHKERRQ(ierr);
    ierr = MatGetSubMatrices(pc->mat,1,&isl,&isl,reuse,&red->mats);CHKERRQ(ierr);
    if (pc->pmat != pc->mat) {
      ierr = MatGetSubMatrices(pc->pmat,1,&isl,&isl,reuse,&red->pmats);CHKERRQ(ierr);
    } else {
      red->pmats = red->mats;
    }
    ierr = ISDestroy(isl);CHKERRQ(ierr);

    /* tell sequential PC its operators */
    ierr = PCSetOperators(red->pc,red->mats[0],red->pmats[0],str);CHKERRQ(ierr);
  } else {
    ierr = PCSetOperators(red->pc,pc->mat,pc->pmat,pc->flag);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}


#undef __FUNC__  
#define __FUNC__ "PCApply_Redundant"
static int PCApply_Redundant(PC pc,Vec x,Vec y)
{
  PC_Redundant      *red = (PC_Redundant *) pc->data;
  int               ierr;

  PetscFunctionBegin;
  /* move all values to each processor */
  ierr = VecScatterBegin(x,red->b,INSERT_VALUES,SCATTER_FORWARD,red->scatterin);CHKERRQ(ierr);
  ierr = VecScatterEnd(x,red->b,INSERT_VALUES,SCATTER_FORWARD,red->scatterin);CHKERRQ(ierr);

  /* apply preconditioner on each processor */
  ierr = PCApply(red->pc,red->b,red->x);CHKERRQ(ierr);

  /* move local part of values into y vector */
  ierr = VecScatterBegin(red->x,y,INSERT_VALUES,SCATTER_FORWARD,red->scatterout);CHKERRQ(ierr);
  ierr = VecScatterEnd(red->x,y,INSERT_VALUES,SCATTER_FORWARD,red->scatterout);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNC__  
#define __FUNC__ "PCDestroy_Redundant"
static int PCDestroy_Redundant(PC pc)
{
  PC_Redundant *red = (PC_Redundant *) pc->data;
  int          ierr;

  PetscFunctionBegin;
  if (red->scatterin)  {ierr = VecScatterDestroy(red->scatterin);CHKERRQ(ierr);}
  if (red->scatterout) {ierr = VecScatterDestroy(red->scatterout);CHKERRQ(ierr);}
  if (red->x)          {ierr = VecDestroy(red->x);CHKERRQ(ierr);}
  if (red->b)          {ierr = VecDestroy(red->b);CHKERRQ(ierr);}
  if (red->pmats && red->pmats != red->mats) {
    ierr = MatDestroyMatrices(1,&red->pmats);CHKERRQ(ierr);
  }
  if (red->mats) {
    ierr = MatDestroyMatrices(1,&red->mats);CHKERRQ(ierr);
  }
  ierr = PCDestroy(red->pc);CHKERRQ(ierr);
  ierr = PetscFree(red);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCPrintHelp_Redundant"
static int PCPrintHelp_Redundant(PC pc,char *p)
{
  int ierr;

  PetscFunctionBegin;
  ierr = (*PetscHelpPrintf)(pc->comm," Options for PCRedundant preconditioner:\n");CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(pc->comm," %sredundant : prefix to control options for redundant PC.\
  Add before the \n      usual PC option names (e.g., %sredundant_pc_type\
  <type>)\n",p,p);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCSetFromOptions_Redundant"
static int PCSetFromOptions_Redundant(PC pc)
{
  int          ierr;
  PC_Redundant *red = (PC_Redundant *) pc->data;

  PetscFunctionBegin;
  ierr = PCSetFromOptions(red->pc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "PCRedundantSetScatter_Redundant"
int PCRedundantSetScatter_Redundant(PC pc,VecScatter in,VecScatter out)
{
  PC_Redundant *red;

  PetscFunctionBegin;
  red                 = (PC_Redundant *) pc->data;
  red->scatterin  = in; PetscObjectReference((PetscObject)in);
  red->scatterout = out;PetscObjectReference((PetscObject)out);
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNC__  
#define __FUNC__ "PCRedundantSetScatter"
/*@
   PCRedundantSetScatter - Sets the scatter used to copy values into the
     redundant local solve and the scatter to move them back into the global
     vector.

   Collective on PC

   Input Parameters:
+  pc - the preconditioner context
.  in - the scatter to move the values in
-  out - the scatter to move them out

   Level: advanced

.keywords: PC, redundant solve
@*/
int PCRedundantSetScatter(PC pc,VecScatter in,VecScatter out)
{
  int ierr, (*f)(PC,VecScatter,VecScatter);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCRedundantSetScatter_C",(void **)&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,in,out);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}  

/* -------------------------------------------------------------------------------------*/
EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "PCCreate_Redundant"
int PCCreate_Redundant(PC pc)
{
  int          ierr;
  PC_Redundant *red;
  char         *prefix;

  PetscFunctionBegin;
  red = PetscNew(PC_Redundant);CHKPTRQ(red);
  PLogObjectMemory(pc,sizeof(PC_Redundant));
  ierr = PetscMemzero(red,sizeof(PC_Redundant));CHKERRQ(ierr);
  red->useparallelmat   = PETSC_TRUE;

  /* create the sequential PC that each processor has copy of */
  ierr = PCCreate(PETSC_COMM_SELF,&red->pc);CHKERRQ(ierr);
  ierr = PCGetOptionsPrefix(pc,&prefix);CHKERRQ(ierr);
  ierr = PCSetOptionsPrefix(red->pc,prefix);CHKERRQ(ierr);
  ierr = PCAppendOptionsPrefix(red->pc,"redundant_");CHKERRQ(ierr);

  pc->ops->apply             = PCApply_Redundant;
  pc->ops->applytrans        = 0;
  pc->ops->setup             = PCSetUp_Redundant;
  pc->ops->destroy           = PCDestroy_Redundant;
  pc->ops->printhelp         = PCPrintHelp_Redundant;
  pc->ops->setfromoptions    = PCSetFromOptions_Redundant;
  pc->ops->setuponblocks     = 0;
  pc->ops->view              = PCView_Redundant;
  pc->ops->applyrichardson   = 0;

  pc->data              = (void *) red;

  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCRedundantSetScatter_C","PCRedundantSetScatter_Redundant",
                    (void*)PCRedundantSetScatter_Redundant);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
EXTERN_C_END
