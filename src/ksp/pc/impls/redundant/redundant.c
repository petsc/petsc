#define PETSCKSP_DLL

/*
  This file defines a "solve the problem redundantly on each subgroup of processor" preconditioner.
*/
#include "private/pcimpl.h"     /*I "petscpc.h" I*/
#include "petscksp.h"

typedef struct {
  PC           pc;                   /* actual preconditioner used on each processor */
  Vec          xsub,ysub;            /* vectors of a subcommunicator to hold parallel vectors of pc->comm */
  Vec          xdup,ydup;            /* parallel vector that congregates xsub or ysub facilitating vector scattering */
  Mat          pmats;                /* matrix and optional preconditioner matrix belong to a subcommunicator */
  VecScatter   scatterin,scatterout; /* scatter used to move all values to each processor group (subcommunicator) */
  PetscTruth   useparallelmat;
  PetscSubcomm *psubcomm;          
  PetscInt     nsubcomm;           /* num of data structure PetscSubcomm */
} PC_Redundant;

#undef __FUNCT__  
#define __FUNCT__ "PCView_Redundant"
static PetscErrorCode PCView_Redundant(PC pc,PetscViewer viewer)
{
  PC_Redundant   *red = (PC_Redundant*)pc->data;
  PetscErrorCode ierr;
  PetscMPIInt    rank;
  PetscTruth     iascii,isstring;
  PetscViewer    sviewer,subviewer;
  PetscInt       color = red->psubcomm->color;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(pc->comm,&rank);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&iascii);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_STRING,&isstring);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  Redundant solver preconditioner: First PC (color=0) follows\n");CHKERRQ(ierr);
    ierr = PetscViewerGetSubcomm(viewer,red->pc->comm,&subviewer);CHKERRQ(ierr);
    if (!color) { /* only view first redundant pc */
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = PCView(red->pc,subviewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    }
    ierr = PetscViewerRestoreSubcomm(viewer,red->pc->comm,&subviewer);CHKERRQ(ierr);
  } else if (isstring) { /* not test it yet! */
    ierr = PetscViewerStringSPrintf(viewer," Redundant solver preconditioner");CHKERRQ(ierr);
    ierr = PetscViewerGetSingleton(viewer,&sviewer);CHKERRQ(ierr);
    if (!rank) {
      ierr = PCView(red->pc,sviewer);CHKERRQ(ierr);
    }
    ierr = PetscViewerRestoreSingleton(viewer,&sviewer);CHKERRQ(ierr);
  } else {
    SETERRQ1(PETSC_ERR_SUP,"Viewer type %s not supported for PC redundant",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

#include "include/private/matimpl.h"        /*I "petscmat.h" I*/
#undef __FUNCT__  
#define __FUNCT__ "PCSetUp_Redundant"
static PetscErrorCode PCSetUp_Redundant(PC pc)
{
  PC_Redundant   *red = (PC_Redundant*)pc->data;
  PetscErrorCode ierr;
  PetscInt       mstart,mend,mlocal,m;
  PetscMPIInt    size;
  MatReuse       reuse = MAT_INITIAL_MATRIX;
  MatStructure   str   = DIFFERENT_NONZERO_PATTERN;
  MPI_Comm       comm = pc->comm,subcomm;
  Vec            vec;
  PetscInt       mlocal_sub;
  PetscMPIInt    subsize,subrank;
  PetscInt       rstart_sub,rend_sub,mloc_sub;
  const char     *prefix;

  PetscFunctionBegin;
  ierr = MatGetVecs(pc->pmat,&vec,0);CHKERRQ(ierr);
  ierr = VecGetSize(vec,&m);CHKERRQ(ierr);

  if (!pc->setupcalled) {
    ierr = PetscSubcommCreate(comm,red->nsubcomm,&red->psubcomm);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory(pc,sizeof(PetscSubcomm));CHKERRQ(ierr);

    /* create a new PC that processors in each subcomm have copy of */
    subcomm = red->psubcomm->comm;
    ierr = PCCreate(subcomm,&red->pc);CHKERRQ(ierr);
    ierr = PCSetType(red->pc,PCLU);CHKERRQ(ierr);
    ierr = PCGetOptionsPrefix(pc,&prefix);CHKERRQ(ierr);
    ierr = PCSetOptionsPrefix(red->pc,prefix);CHKERRQ(ierr);
    ierr = PCAppendOptionsPrefix(red->pc,"redundant_");CHKERRQ(ierr);
    ierr = PCSetFromOptions(red->pc);CHKERRQ(ierr);

    /* create working vectors xsub/ysub and xdup/ydup */
    ierr = VecGetLocalSize(vec,&mlocal);CHKERRQ(ierr);  
    ierr = VecGetOwnershipRange(vec,&mstart,&mend);CHKERRQ(ierr);

    /* get local size of xsub/ysub */    
    ierr = MPI_Comm_size(subcomm,&subsize);CHKERRQ(ierr);
    ierr = MPI_Comm_rank(subcomm,&subrank);CHKERRQ(ierr);
    rstart_sub = pc->pmat->rmap.range[red->psubcomm->n*subrank]; /* rstart in xsub/ysub */    
    if (subrank+1 < subsize){
      rend_sub = pc->pmat->rmap.range[red->psubcomm->n*(subrank+1)];
    } else {
      rend_sub = m; 
    }
    mloc_sub = rend_sub - rstart_sub;
    ierr = VecCreateMPI(subcomm,mloc_sub,PETSC_DECIDE,&red->ysub);CHKERRQ(ierr);
    /* create xsub with empty local arrays, because xdup's arrays will be placed into it */
    ierr = VecCreateMPIWithArray(subcomm,mloc_sub,PETSC_DECIDE,PETSC_NULL,&red->xsub);CHKERRQ(ierr);

    /* create xdup and ydup. ydup has empty local arrays because ysub's arrays will be place into it. 
       Note: we use communicator dupcomm, not pc->comm! */      
    ierr = VecCreateMPI(red->psubcomm->dupparent,mloc_sub,PETSC_DECIDE,&red->xdup);CHKERRQ(ierr);
    ierr = VecCreateMPIWithArray(red->psubcomm->dupparent,mloc_sub,PETSC_DECIDE,PETSC_NULL,&red->ydup);CHKERRQ(ierr);
  
    /* create vec scatters */
    if (!red->scatterin){
      IS       is1,is2;
      PetscInt *idx1,*idx2,i,j,k; 
      ierr = PetscMalloc(2*red->psubcomm->n*mlocal*sizeof(PetscInt),&idx1);CHKERRQ(ierr);
      idx2 = idx1 + red->psubcomm->n*mlocal;
      j = 0;
      for (k=0; k<red->psubcomm->n; k++){
        for (i=mstart; i<mend; i++){
          idx1[j]   = i;
          idx2[j++] = i + m*k;
        }
      }
      ierr = ISCreateGeneral(comm,red->psubcomm->n*mlocal,idx1,&is1);CHKERRQ(ierr);
      ierr = ISCreateGeneral(comm,red->psubcomm->n*mlocal,idx2,&is2);CHKERRQ(ierr);      
      ierr = VecScatterCreate(vec,is1,red->xdup,is2,&red->scatterin);CHKERRQ(ierr);
      ierr = ISDestroy(is1);CHKERRQ(ierr);
      ierr = ISDestroy(is2);CHKERRQ(ierr);

      ierr = ISCreateStride(comm,mlocal,mstart+ red->psubcomm->color*m,1,&is1);CHKERRQ(ierr);
      ierr = ISCreateStride(comm,mlocal,mstart,1,&is2);CHKERRQ(ierr);
      ierr = VecScatterCreate(red->xdup,is1,vec,is2,&red->scatterout);CHKERRQ(ierr);      
      ierr = ISDestroy(is1);CHKERRQ(ierr);
      ierr = ISDestroy(is2);CHKERRQ(ierr);
      ierr = PetscFree(idx1);CHKERRQ(ierr);
    }
  }
  ierr = VecDestroy(vec);CHKERRQ(ierr);

  /* if pmatrix set by user is sequential then we do not need to gather the parallel matrix */
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  if (size == 1) {
    red->useparallelmat = PETSC_FALSE;
  }

  if (red->useparallelmat) {
    if (pc->setupcalled == 1 && pc->flag == DIFFERENT_NONZERO_PATTERN) {
      /* destroy old matrices */
      if (red->pmats) {
        ierr = MatDestroy(red->pmats);CHKERRQ(ierr);
      }
    } else if (pc->setupcalled == 1) {
      reuse = MAT_REUSE_MATRIX;
      str   = SAME_NONZERO_PATTERN;
    }
       
    /* grab the parallel matrix and put it into processors of a subcomminicator */ 
    /*--------------------------------------------------------------------------*/
    ierr = VecGetLocalSize(red->ysub,&mlocal_sub);CHKERRQ(ierr);  
    ierr = MatGetRedundantMatrix(pc->pmat,red->psubcomm->n,red->psubcomm->comm,mlocal_sub,reuse,&red->pmats);CHKERRQ(ierr);
   
    /* tell PC of the subcommunicator its operators */
    ierr = PCSetOperators(red->pc,red->pmats,red->pmats,str);CHKERRQ(ierr);
  } else {
    ierr = PCSetOperators(red->pc,pc->mat,pc->pmat,pc->flag);CHKERRQ(ierr);
  }
  ierr = PCSetFromOptions(red->pc);CHKERRQ(ierr);
  ierr = PCSetUp(red->pc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCApply_Redundant"
static PetscErrorCode PCApply_Redundant(PC pc,Vec x,Vec y)
{
  PC_Redundant   *red = (PC_Redundant*)pc->data;
  PetscErrorCode ierr;
  PetscScalar    *array;

  PetscFunctionBegin;
  /* scatter x to xdup */
  ierr = VecScatterBegin(x,red->xdup,INSERT_VALUES,SCATTER_FORWARD,red->scatterin);CHKERRQ(ierr);
  ierr = VecScatterEnd(x,red->xdup,INSERT_VALUES,SCATTER_FORWARD,red->scatterin);CHKERRQ(ierr);
  
  /* place xdup's local array into xsub */
  ierr = VecGetArray(red->xdup,&array);CHKERRQ(ierr);
  ierr = VecPlaceArray(red->xsub,(const PetscScalar*)array);CHKERRQ(ierr);

  /* apply preconditioner on each processor */
  ierr = PCApply(red->pc,red->xsub,red->ysub);CHKERRQ(ierr);
  ierr = VecResetArray(red->xsub);CHKERRQ(ierr);
  ierr = VecRestoreArray(red->xdup,&array);CHKERRQ(ierr);
 
  /* place ysub's local array into ydup */
  ierr = VecGetArray(red->ysub,&array);CHKERRQ(ierr);
  ierr = VecPlaceArray(red->ydup,(const PetscScalar*)array);CHKERRQ(ierr);

  /* scatter ydup to y */
  ierr = VecScatterBegin(red->ydup,y,INSERT_VALUES,SCATTER_FORWARD,red->scatterout);CHKERRQ(ierr);
  ierr = VecScatterEnd(red->ydup,y,INSERT_VALUES,SCATTER_FORWARD,red->scatterout);CHKERRQ(ierr);
  ierr = VecResetArray(red->ydup);CHKERRQ(ierr);
  ierr = VecRestoreArray(red->ysub,&array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCDestroy_Redundant"
static PetscErrorCode PCDestroy_Redundant(PC pc)
{
  PC_Redundant   *red = (PC_Redundant*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (red->scatterin)  {ierr = VecScatterDestroy(red->scatterin);CHKERRQ(ierr);}
  if (red->scatterout) {ierr = VecScatterDestroy(red->scatterout);CHKERRQ(ierr);}
  if (red->ysub)       {ierr = VecDestroy(red->ysub);CHKERRQ(ierr);}
  if (red->xsub)       {ierr = VecDestroy(red->xsub);CHKERRQ(ierr);}
  if (red->xdup)       {ierr = VecDestroy(red->xdup);CHKERRQ(ierr);}
  if (red->ydup)       {ierr = VecDestroy(red->ydup);CHKERRQ(ierr);}
  if (red->pmats) {
    ierr = MatDestroy(red->pmats);CHKERRQ(ierr);
  }
  if (red->psubcomm) {ierr = PetscSubcommDestroy(red->psubcomm);CHKERRQ(ierr);}
  if (red->pc) {ierr = PCDestroy(red->pc);CHKERRQ(ierr);}
  ierr = PetscFree(red);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCSetFromOptions_Redundant"
static PetscErrorCode PCSetFromOptions_Redundant(PC pc)
{
  PetscErrorCode ierr;
  PC_Redundant   *red = (PC_Redundant*)pc->data;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("Redundant options");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-pc_redundant_number","Number of redundant pc","PCRedundantSetNumber",red->nsubcomm,&red->nsubcomm,0);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCRedundantSetNumber_Redundant"
PetscErrorCode PETSCKSP_DLLEXPORT PCRedundantSetNumber_Redundant(PC pc,PetscInt nreds)
{
  PC_Redundant   *red = (PC_Redundant*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  red->nsubcomm = nreds; 
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "PCRedundantSetNumber"
/*@
   PCRedundantSetNumber - Sets the number of redundant preconditioner contexts.

   Collective on PC

   Input Parameters:
+  pc - the preconditioner context
-  nredundant - number of redundant preconditioner contexts

   Level: advanced

.keywords: PC, redundant solve
@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCRedundantSetNumber(PC pc,PetscInt nredundant)
{
  PetscErrorCode ierr,(*f)(PC,PetscInt);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  if (nredundant <= 0) SETERRQ1(PETSC_ERR_ARG_WRONG, "num of redundant pc %D must be positive",nredundant);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCRedundantSetNumber_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,nredundant);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}  

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCRedundantSetScatter_Redundant"
PetscErrorCode PETSCKSP_DLLEXPORT PCRedundantSetScatter_Redundant(PC pc,VecScatter in,VecScatter out)
{
  PC_Redundant   *red = (PC_Redundant*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  red->scatterin  = in; 
  red->scatterout = out;
  ierr = PetscObjectReference((PetscObject)in);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)out);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "PCRedundantSetScatter"
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
PetscErrorCode PETSCKSP_DLLEXPORT PCRedundantSetScatter(PC pc,VecScatter in,VecScatter out)
{
  PetscErrorCode ierr,(*f)(PC,VecScatter,VecScatter);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  PetscValidHeaderSpecific(in,VEC_SCATTER_COOKIE,2);
  PetscValidHeaderSpecific(out,VEC_SCATTER_COOKIE,3);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCRedundantSetScatter_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,in,out);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}  

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCRedundantGetPC_Redundant"
PetscErrorCode PETSCKSP_DLLEXPORT PCRedundantGetPC_Redundant(PC pc,PC *innerpc)
{
  PC_Redundant *red = (PC_Redundant*)pc->data;

  PetscFunctionBegin;
  *innerpc = red->pc;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "PCRedundantGetPC"
/*@
   PCRedundantGetPC - Gets the sequential PC created by the redundant PC.

   Not Collective

   Input Parameter:
.  pc - the preconditioner context

   Output Parameter:
.  innerpc - the sequential PC 

   Level: advanced

.keywords: PC, redundant solve
@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCRedundantGetPC(PC pc,PC *innerpc)
{
  PetscErrorCode ierr,(*f)(PC,PC*);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  PetscValidPointer(innerpc,2);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCRedundantGetPC_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,innerpc);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}  

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCRedundantGetOperators_Redundant"
PetscErrorCode PETSCKSP_DLLEXPORT PCRedundantGetOperators_Redundant(PC pc,Mat *mat,Mat *pmat)
{
  PC_Redundant *red = (PC_Redundant*)pc->data;

  PetscFunctionBegin;
  if (mat)  *mat  = red->pmats;
  if (pmat) *pmat = red->pmats;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "PCRedundantGetOperators"
/*@
   PCRedundantGetOperators - gets the sequential matrix and preconditioner matrix

   Not Collective

   Input Parameter:
.  pc - the preconditioner context

   Output Parameters:
+  mat - the matrix
-  pmat - the (possibly different) preconditioner matrix

   Level: advanced

.keywords: PC, redundant solve
@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCRedundantGetOperators(PC pc,Mat *mat,Mat *pmat)
{
  PetscErrorCode ierr,(*f)(PC,Mat*,Mat*);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  if (mat)  PetscValidPointer(mat,2);
  if (pmat) PetscValidPointer(pmat,3); 
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCRedundantGetOperators_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,mat,pmat);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}  

/* -------------------------------------------------------------------------------------*/
/*MC
     PCREDUNDANT - Runs a preconditioner for the entire problem on subgroups of processors

     Options for the redundant preconditioners can be set with -redundant_pc_xxx

  Options Database:
.  -pc_redundant_number_comm - number of sub communicators to use

   Level: intermediate

.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PCRedundantSetScatter(),
           PCRedundantGetPC(), PCRedundantGetOperators()
M*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCCreate_Redundant"
PetscErrorCode PETSCKSP_DLLEXPORT PCCreate_Redundant(PC pc)
{
  PetscErrorCode ierr;
  PC_Redundant   *red;
  PetscMPIInt    size;
  
  PetscFunctionBegin;
  ierr = PetscNew(PC_Redundant,&red);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory(pc,sizeof(PC_Redundant));CHKERRQ(ierr);
  ierr = MPI_Comm_size(pc->comm,&size);CHKERRQ(ierr);
  red->nsubcomm       = size;
  red->useparallelmat = PETSC_TRUE;
  pc->data            = (void*)red; 

  pc->ops->apply           = PCApply_Redundant;
  pc->ops->applytranspose  = 0;
  pc->ops->setup           = PCSetUp_Redundant;
  pc->ops->destroy         = PCDestroy_Redundant;
  pc->ops->setfromoptions  = PCSetFromOptions_Redundant;
  pc->ops->view            = PCView_Redundant;    
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCRedundantSetScatter_C","PCRedundantSetScatter_Redundant",
                    PCRedundantSetScatter_Redundant);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCRedundantSetNumber_C","PCRedundantSetNumber_Redundant",
                    PCRedundantSetNumber_Redundant);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCRedundantGetPC_C","PCRedundantGetPC_Redundant",
                    PCRedundantGetPC_Redundant);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCRedundantGetOperators_C","PCRedundantGetOperators_Redundant",
                    PCRedundantGetOperators_Redundant);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END
