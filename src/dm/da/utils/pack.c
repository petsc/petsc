#define PETSCDM_DLL
 
#include "petscda.h"             /*I      "petscda.h"     I*/
#include "private/dmimpl.h"    
#include "petscmat.h"    

typedef struct _DMCompositeOps *DMCompositeOps;
struct _DMCompositeOps {
  DMOPS(DMComposite)
};

/*
   rstart is where an array/subvector starts in the global parallel vector, so arrays
   rstarts are meaningless (and set to the previous one) except on the processor where the array lives
*/

typedef enum {DMCOMPOSITE_ARRAY, DMCOMPOSITE_DM} DMCompositeLinkType;

struct DMCompositeLink {
  DMCompositeLinkType    type;
  struct DMCompositeLink *next;
  PetscInt               n,rstart;      /* rstart is relative to this process */
  PetscInt               grstart;       /* grstart is relative to all processes */

  /* only used for DMCOMPOSITE_DM */
  PetscInt               *grstarts;     /* global row for first unknown of this DM on each process */
  DM                     dm;

  /* only used for DMCOMPOSITE_ARRAY */
  PetscMPIInt            rank;          /* process where array unknowns live */
};

struct _p_DMComposite {
  PETSCHEADER(struct _DMCompositeOps);
  DMHEADER
  PetscInt               n,N,rstart;           /* rstart is relative to all processors, n unknowns owned by this process, N is total unknowns */
  PetscInt               nghost;               /* number of all local entries include DA ghost points and any shared redundant arrays */
  PetscInt               nDM,nredundant,nmine; /* how many DM's and seperate redundant arrays used to build DMComposite (nmine is ones on this process) */
  PetscTruth             setup;                /* after this is set, cannot add new links to the DMComposite */
  struct DMCompositeLink *next;

  PetscErrorCode (*FormCoupleLocations)(DMComposite,Mat,PetscInt*,PetscInt*,PetscInt,PetscInt,PetscInt,PetscInt);
  void                   *ctx;                 /* place for user to set information they may need in FormCoupleLocation */
};

#undef __FUNCT__  
#define __FUNCT__ "DMCompositeSetCoupling"
/*@C
    DMCompositeSetCoupling - Sets user provided routines that compute the coupling between the 
      seperate components (DA's and arrays) in a DMComposite to build the correct matrix nonzero structure.


    Collective on MPI_Comm

    Input Parameter:
+   dmcomposite - the composite object
-   formcouplelocations - routine to set the nonzero locations in the matrix

    Level: advanced

    Notes: See DMCompositeSetContext() and DMCompositeGetContext() for how to get user information into
        this routine

.seealso DMCompositeDestroy(), DMCompositeAddArray(), DMCompositeAddDM(), DMCompositeScatter(),
         DMCompositeGather(), DMCompositeCreateGlobalVector(), DMCompositeGetGlobalIndices(), DMCompositeGetAccess()
         DMCompositeGetLocalVectors(), DMCompositeRestoreLocalVectors(), DMCompositeGetEntries(), DMCompositeSetContext(),
         DMCompositeGetContext()

@*/
PetscErrorCode PETSCDM_DLLEXPORT DMCompositeSetCoupling(DMComposite dmcomposite,PetscErrorCode (*FormCoupleLocations)(DMComposite,Mat,PetscInt*,PetscInt*,PetscInt,PetscInt,PetscInt,PetscInt))
{
  PetscFunctionBegin;
  dmcomposite->FormCoupleLocations = FormCoupleLocations;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMCompositeSetContext"
/*@
    DMCompositeSetContext - Allows user to stash data they may need within the form coupling routine they 
      set with DMCompositeSetCoupling()


    Not Collective

    Input Parameter:
+   dmcomposite - the composite object
-   ctx - the user supplied context

    Level: advanced

    Notes: Use DMCompositeGetContext() to retrieve the context when needed.

.seealso DMCompositeDestroy(), DMCompositeAddArray(), DMCompositeAddDM(), DMCompositeScatter(),
         DMCompositeGather(), DMCompositeCreateGlobalVector(), DMCompositeGetGlobalIndices(), DMCompositeGetAccess()
         DMCompositeGetLocalVectors(), DMCompositeRestoreLocalVectors(), DMCompositeGetEntries(), DMCompositeSetCoupling(),
         DMCompositeGetContext()

@*/
PetscErrorCode PETSCDM_DLLEXPORT DMCompositeSetContext(DMComposite dmcomposite,void *ctx)
{
  PetscFunctionBegin;
  dmcomposite->ctx = ctx;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMCompositeGetContext"
/*@
    DMCompositeGetContext - Access the context set with DMCompositeSetContext()


    Not Collective

    Input Parameter:
.   dmcomposite - the composite object

    Output Parameter:
.    ctx - the user supplied context

    Level: advanced

    Notes: Use DMCompositeGetContext() to retrieve the context when needed.

.seealso DMCompositeDestroy(), DMCompositeAddArray(), DMCompositeAddDM(), DMCompositeScatter(),
         DMCompositeGather(), DMCompositeCreateGlobalVector(), DMCompositeGetGlobalIndices(), DMCompositeGetAccess()
         DMCompositeGetLocalVectors(), DMCompositeRestoreLocalVectors(), DMCompositeGetEntries(), DMCompositeSetCoupling(),
         DMCompositeSetContext()

@*/
PetscErrorCode PETSCDM_DLLEXPORT DMCompositeGetContext(DMComposite dmcomposite,void **ctx)
{
  PetscFunctionBegin;
  *ctx = dmcomposite->ctx;
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "DMCompositeCreate"
/*@C
    DMCompositeCreate - Creates a vector packer, used to generate "composite"
      vectors made up of several subvectors.

    Collective on MPI_Comm

    Input Parameter:
.   comm - the processors that will share the global vector

    Output Parameters:
.   packer - the packer object

    Level: advanced

.seealso DMCompositeDestroy(), DMCompositeAddArray(), DMCompositeAddDM(), DMCompositeScatter(),
         DMCompositeGather(), DMCompositeCreateGlobalVector(), DMCompositeGetGlobalIndices(), DMCompositeGetAccess()
         DMCompositeGetLocalVectors(), DMCompositeRestoreLocalVectors(), DMCompositeGetEntries()

@*/
PetscErrorCode PETSCDM_DLLEXPORT DMCompositeCreate(MPI_Comm comm,DMComposite *packer)
{
  PetscErrorCode ierr;
  DMComposite    p;

  PetscFunctionBegin;
  PetscValidPointer(packer,2);
  *packer = PETSC_NULL;
#ifndef PETSC_USE_DYNAMIC_LIBRARIES
  ierr = DMInitializePackage(PETSC_NULL);CHKERRQ(ierr);
#endif

  ierr = PetscHeaderCreate(p,_p_DMComposite,struct _DMCompositeOps,DM_COOKIE,0,"DM",comm,DMCompositeDestroy,DMCompositeView);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)p,"DMComposite");CHKERRQ(ierr);
  p->n            = 0;
  p->next         = PETSC_NULL;
  p->nredundant   = 0;
  p->nDM          = 0;

  p->ops->createglobalvector = DMCompositeCreateGlobalVector;
  p->ops->createlocalvector  = DMCompositeCreateLocalVector;
  p->ops->refine             = DMCompositeRefine;
  p->ops->getinterpolation   = DMCompositeGetInterpolation;
  p->ops->getmatrix          = DMCompositeGetMatrix;
  p->ops->getcoloring        = DMCompositeGetColoring;
  p->ops->globaltolocalbegin = DMCompositeGlobalToLocalBegin;
  p->ops->globaltolocalend   = DMCompositeGlobalToLocalEnd;
  p->ops->destroy            = DMCompositeDestroy;

  *packer = p;
  PetscFunctionReturn(0);
}

extern PetscErrorCode DMDestroy_Private(DM,PetscTruth*);

#undef __FUNCT__  
#define __FUNCT__ "DMCompositeDestroy"
/*@C
    DMCompositeDestroy - Destroys a vector packer.

    Collective on DMComposite

    Input Parameter:
.   packer - the packer object

    Level: advanced

.seealso DMCompositeCreate(), DMCompositeAddArray(), DMCompositeAddDM(), DMCompositeScatter(),DMCompositeGetEntries()
         DMCompositeGather(), DMCompositeCreateGlobalVector(), DMCompositeGetGlobalIndices(), DMCompositeGetAccess()

@*/
PetscErrorCode PETSCDM_DLLEXPORT DMCompositeDestroy(DMComposite packer)
{
  PetscErrorCode         ierr;
  struct DMCompositeLink *next, *prev;
  PetscTruth             done;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(packer,DM_COOKIE,1);
  ierr = DMDestroy_Private((DM)packer,&done);CHKERRQ(ierr);
  if (!done) PetscFunctionReturn(0);

  next = packer->next;
  while (next) {
    prev = next;
    next = next->next;
    if (prev->type == DMCOMPOSITE_DM) {
      ierr = DMDestroy(prev->dm);CHKERRQ(ierr);
    }
    if (prev->grstarts) {
      ierr = PetscFree(prev->grstarts);CHKERRQ(ierr);
    }
    ierr = PetscFree(prev);CHKERRQ(ierr);
  }
  ierr = PetscHeaderDestroy(packer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMCompositeView"
/*@
    DMCompositeView - Views a composite DM

    Collective on DMComposite

    Input Parameter:
+   packer - the DMComposite object to view
-   v - the viewer

    Level: intermediate

.seealso DMCompositeCreate()

@*/
PetscErrorCode PETSCDM_DLLEXPORT DMCompositeView(DMComposite packer,PetscViewer v)
{
  PetscErrorCode ierr;
  PetscTruth     iascii;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)v,PETSC_VIEWER_ASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    struct DMCompositeLink *lnk = packer->next;
    PetscInt i;

    ierr = PetscViewerASCIIPrintf(v,"DMComposite (%s)\n",((PetscObject)packer)->prefix?((PetscObject)packer)->prefix:"no prefix");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(v,"  contains %d DMs and %d redundant arrays\n",packer->nDM,packer->nredundant);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(v);CHKERRQ(ierr);
    for (i=0; lnk; lnk=lnk->next,i++) {
      if (lnk->dm) {
        ierr = PetscViewerASCIIPrintf(v,"Link %d: DM of type %s\n",i,((PetscObject)lnk->dm)->type_name);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPushTab(v);CHKERRQ(ierr);
        ierr = DMView(lnk->dm,v);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPopTab(v);CHKERRQ(ierr);
      } else {
        ierr = PetscViewerASCIIPrintf(v,"Link %d: Redundant array of size %d owned by rank %d\n",i,lnk->n,lnk->rank);CHKERRQ(ierr);
      }
    }
    ierr = PetscViewerASCIIPopTab(v);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "DMCompositeSetUp"
PetscErrorCode PETSCDM_DLLEXPORT DMCompositeSetUp(DMComposite packer)
{
  PetscErrorCode         ierr;
  PetscInt               nprev = 0;
  PetscMPIInt            rank,size;
  struct DMCompositeLink *next = packer->next;
  PetscLayout            map;

  PetscFunctionBegin;
  if (packer->setup) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Packer has already been setup");
  ierr = PetscLayoutCreate(((PetscObject)packer)->comm,&map);CHKERRQ(ierr);
  ierr = PetscLayoutSetLocalSize(map,packer->n);CHKERRQ(ierr);
  ierr = PetscLayoutSetSize(map,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = PetscLayoutSetBlockSize(map,1);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(map);CHKERRQ(ierr);
  ierr = PetscLayoutGetSize(map,&packer->N);CHKERRQ(ierr);
  ierr = PetscLayoutGetRange(map,&packer->rstart,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscLayoutDestroy(map);CHKERRQ(ierr);
    
  /* now set the rstart for each linked array/vector */
  ierr = MPI_Comm_rank(((PetscObject)packer)->comm,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(((PetscObject)packer)->comm,&size);CHKERRQ(ierr);
  while (next) {
    next->rstart = nprev; 
    if ((rank == next->rank) || next->type != DMCOMPOSITE_ARRAY) nprev += next->n;
    next->grstart = packer->rstart + next->rstart;
    if (next->type == DMCOMPOSITE_ARRAY) {
      ierr = MPI_Bcast(&next->grstart,1,MPIU_INT,next->rank,((PetscObject)packer)->comm);CHKERRQ(ierr);
    } else {
      ierr = PetscMalloc(size*sizeof(PetscInt),&next->grstarts);CHKERRQ(ierr);
      ierr = MPI_Allgather(&next->grstart,1,MPIU_INT,next->grstarts,1,MPIU_INT,((PetscObject)packer)->comm);CHKERRQ(ierr);
    }
    next = next->next;
  }
  packer->setup = PETSC_TRUE;
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "DMCompositeGetAccess_Array"
PetscErrorCode DMCompositeGetAccess_Array(DMComposite packer,struct DMCompositeLink *mine,Vec vec,PetscScalar **array)
{
  PetscErrorCode ierr;
  PetscScalar    *varray;
  PetscMPIInt    rank;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(((PetscObject)packer)->comm,&rank);CHKERRQ(ierr);
  if (array) {
    if (rank == mine->rank) {
      ierr    = VecGetArray(vec,&varray);CHKERRQ(ierr);
      *array  = varray + mine->rstart;
      ierr    = VecRestoreArray(vec,&varray);CHKERRQ(ierr);
    } else {
      *array = 0;
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMCompositeGetAccess_DM"
PetscErrorCode DMCompositeGetAccess_DM(DMComposite packer,struct DMCompositeLink *mine,Vec vec,Vec *global)
{
  PetscErrorCode ierr;
  PetscScalar    *array;

  PetscFunctionBegin;
  if (global) {
    ierr    = DMGetGlobalVector(mine->dm,global);CHKERRQ(ierr);
    ierr    = VecGetArray(vec,&array);CHKERRQ(ierr);
    ierr    = VecPlaceArray(*global,array+mine->rstart);CHKERRQ(ierr);
    ierr    = VecRestoreArray(vec,&array);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMCompositeRestoreAccess_Array"
PetscErrorCode DMCompositeRestoreAccess_Array(DMComposite packer,struct DMCompositeLink *mine,Vec vec,PetscScalar **array)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMCompositeRestoreAccess_DM"
PetscErrorCode DMCompositeRestoreAccess_DM(DMComposite packer,struct DMCompositeLink *mine,Vec vec,Vec *global)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (global) {
    ierr = VecResetArray(*global);CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(mine->dm,global);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMCompositeScatter_Array"
PetscErrorCode DMCompositeScatter_Array(DMComposite packer,struct DMCompositeLink *mine,Vec vec,PetscScalar *array)
{
  PetscErrorCode ierr;
  PetscScalar    *varray;
  PetscMPIInt    rank;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(((PetscObject)packer)->comm,&rank);CHKERRQ(ierr);
  if (rank == mine->rank) {
    ierr    = VecGetArray(vec,&varray);CHKERRQ(ierr);
    ierr    = PetscMemcpy(array,varray+mine->rstart,mine->n*sizeof(PetscScalar));CHKERRQ(ierr);
    ierr    = VecRestoreArray(vec,&varray);CHKERRQ(ierr);
  }
  ierr    = MPI_Bcast(array,mine->n,MPIU_SCALAR,mine->rank,((PetscObject)packer)->comm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMCompositeScatter_DM"
PetscErrorCode DMCompositeScatter_DM(DMComposite packer,struct DMCompositeLink *mine,Vec vec,Vec local)
{
  PetscErrorCode ierr;
  PetscScalar    *array;
  Vec            global;

  PetscFunctionBegin;
  ierr = DMGetGlobalVector(mine->dm,&global);CHKERRQ(ierr);
  ierr = VecGetArray(vec,&array);CHKERRQ(ierr);
  ierr = VecPlaceArray(global,array+mine->rstart);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(mine->dm,global,INSERT_VALUES,local);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(mine->dm,global,INSERT_VALUES,local);CHKERRQ(ierr);
  ierr = VecRestoreArray(vec,&array);CHKERRQ(ierr);
  ierr = VecResetArray(global);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(mine->dm,&global);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMCompositeGather_Array"
PetscErrorCode DMCompositeGather_Array(DMComposite packer,struct DMCompositeLink *mine,Vec vec,PetscScalar *array)
{
  PetscErrorCode ierr;
  PetscScalar    *varray;
  PetscMPIInt    rank;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(((PetscObject)packer)->comm,&rank);CHKERRQ(ierr);
  if (rank == mine->rank) {
    ierr    = VecGetArray(vec,&varray);CHKERRQ(ierr);
    if (varray+mine->rstart == array) SETERRQ(PETSC_ERR_ARG_WRONG,"You need not DMCompositeGather() into objects obtained via DMCompositeGetAccess()");
    ierr    = PetscMemcpy(varray+mine->rstart,array,mine->n*sizeof(PetscScalar));CHKERRQ(ierr);
    ierr    = VecRestoreArray(vec,&varray);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMCompositeGather_DM"
PetscErrorCode DMCompositeGather_DM(DMComposite packer,struct DMCompositeLink *mine,Vec vec,Vec local)
{
  PetscErrorCode ierr;
  PetscScalar    *array;
  Vec            global;

  PetscFunctionBegin;
  ierr = DMGetGlobalVector(mine->dm,&global);CHKERRQ(ierr);
  ierr = VecGetArray(vec,&array);CHKERRQ(ierr);
  ierr = VecPlaceArray(global,array+mine->rstart);CHKERRQ(ierr);
  ierr = DMLocalToGlobal(mine->dm,local,INSERT_VALUES,global);CHKERRQ(ierr);
  ierr = VecRestoreArray(vec,&array);CHKERRQ(ierr);
  ierr = VecResetArray(global);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(mine->dm,&global);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ----------------------------------------------------------------------------------*/

#include <stdarg.h>

#undef __FUNCT__  
#define __FUNCT__ "DMCompositeGetNumberDM"
/*@C
    DMCompositeGetNumberDM - Get's the number of DM objects in the DMComposite
       representation.

    Collective on DMComposite

    Input Parameter:
.    packer - the packer object

    Output Parameter:
.     nDM - the number of DMs

    Level: beginner

.seealso DMCompositeDestroy(), DMCompositeAddArray(), DMCompositeAddDM(), DMCompositeCreateGlobalVector(),
         DMCompositeGather(), DMCompositeCreate(), DMCompositeGetGlobalIndices(), DMCompositeScatter(),
         DMCompositeRestoreAccess(), DMCompositeGetLocalVectors(), DMCompositeRestoreLocalVectors(),
         DMCompositeGetEntries()

@*/
PetscErrorCode PETSCDM_DLLEXPORT DMCompositeGetNumberDM(DMComposite packer,PetscInt *nDM)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(packer,DM_COOKIE,1);
  *nDM = packer->nDM;
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "DMCompositeGetAccess"
/*@C
    DMCompositeGetAccess - Allows one to access the individual packed vectors in their global
       representation.

    Collective on DMComposite

    Input Parameter:
+    packer - the packer object
.    gvec - the global vector
-    ... - the individual sequential or parallel objects (arrays or vectors)

    Notes: Use DMCompositeRestoreAccess() to return the vectors when you no longer need them
 
    Level: advanced

.seealso DMCompositeDestroy(), DMCompositeAddArray(), DMCompositeAddDM(), DMCompositeCreateGlobalVector(),
         DMCompositeGather(), DMCompositeCreate(), DMCompositeGetGlobalIndices(), DMCompositeScatter(),
         DMCompositeRestoreAccess(), DMCompositeGetLocalVectors(), DMCompositeRestoreLocalVectors(),
         DMCompositeGetEntries()

@*/
PetscErrorCode PETSCDM_DLLEXPORT DMCompositeGetAccess(DMComposite packer,Vec gvec,...)
{
  va_list                Argp;
  PetscErrorCode         ierr;
  struct DMCompositeLink *next;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(packer,DM_COOKIE,1);
  PetscValidHeaderSpecific(gvec,VEC_COOKIE,2);
  next = packer->next;
  if (!packer->setup) {
    ierr = DMCompositeSetUp(packer);CHKERRQ(ierr);
  }

  /* loop over packed objects, handling one at at time */
  va_start(Argp,gvec);
  while (next) {
    if (next->type == DMCOMPOSITE_ARRAY) {
      PetscScalar **array;
      array = va_arg(Argp, PetscScalar**);
      ierr  = DMCompositeGetAccess_Array(packer,next,gvec,array);CHKERRQ(ierr);
    } else if (next->type == DMCOMPOSITE_DM) {
      Vec *vec;
      vec  = va_arg(Argp, Vec*);
      ierr = DMCompositeGetAccess_DM(packer,next,gvec,vec);CHKERRQ(ierr);
    } else {
      SETERRQ(PETSC_ERR_SUP,"Cannot handle that object type yet");
    }
    next = next->next;
  }
  va_end(Argp);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMCompositeRestoreAccess"
/*@C
    DMCompositeRestoreAccess - Returns the vectors obtained with DACompositeGetAccess()
       representation.

    Collective on DMComposite

    Input Parameter:
+    packer - the packer object
.    gvec - the global vector
-    ... - the individual sequential or parallel objects (arrays or vectors)
 
    Level: advanced

.seealso DMCompositeDestroy(), DMCompositeAddArray(), DMCompositeAddDM(), DMCompositeCreateGlobalVector(),
         DMCompositeGather(), DMCompositeCreate(), DMCompositeGetGlobalIndices(), DMCompositeScatter(),
         DMCompositeRestoreAccess(), DACompositeGetAccess()

@*/
PetscErrorCode PETSCDM_DLLEXPORT DMCompositeRestoreAccess(DMComposite packer,Vec gvec,...)
{
  va_list                Argp;
  PetscErrorCode         ierr;
  struct DMCompositeLink *next;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(packer,DM_COOKIE,1);
  PetscValidHeaderSpecific(gvec,VEC_COOKIE,2);
  next = packer->next;
  if (!packer->setup) {
    ierr = DMCompositeSetUp(packer);CHKERRQ(ierr);
  }

  /* loop over packed objects, handling one at at time */
  va_start(Argp,gvec);
  while (next) {
    if (next->type == DMCOMPOSITE_ARRAY) {
      PetscScalar **array;
      array = va_arg(Argp, PetscScalar**);
      ierr  = DMCompositeRestoreAccess_Array(packer,next,gvec,array);CHKERRQ(ierr);
    } else if (next->type == DMCOMPOSITE_DM) {
      Vec *vec;
      vec  = va_arg(Argp, Vec*);
      ierr = DMCompositeRestoreAccess_DM(packer,next,gvec,vec);CHKERRQ(ierr);
    } else {
      SETERRQ(PETSC_ERR_SUP,"Cannot handle that object type yet");
    }
    next = next->next;
  }
  va_end(Argp);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMCompositeScatter"
/*@C
    DMCompositeScatter - Scatters from a global packed vector into its individual local vectors

    Collective on DMComposite

    Input Parameter:
+    packer - the packer object
.    gvec - the global vector
-    ... - the individual sequential objects (arrays or vectors)
 
    Level: advanced

.seealso DMCompositeDestroy(), DMCompositeAddArray(), DMCompositeAddDM(), DMCompositeCreateGlobalVector(),
         DMCompositeGather(), DMCompositeCreate(), DMCompositeGetGlobalIndices(), DMCompositeGetAccess(),
         DMCompositeGetLocalVectors(), DMCompositeRestoreLocalVectors(), DMCompositeGetEntries()

@*/
PetscErrorCode PETSCDM_DLLEXPORT DMCompositeScatter(DMComposite packer,Vec gvec,...)
{
  va_list                Argp;
  PetscErrorCode         ierr;
  struct DMCompositeLink *next;
  PetscInt               cnt = 3;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(packer,DM_COOKIE,1);
  PetscValidHeaderSpecific(gvec,VEC_COOKIE,2);
  next = packer->next;
  if (!packer->setup) {
    ierr = DMCompositeSetUp(packer);CHKERRQ(ierr);
  }

  /* loop over packed objects, handling one at at time */
  va_start(Argp,gvec);
  while (next) {
    if (next->type == DMCOMPOSITE_ARRAY) {
      PetscScalar *array;
      array = va_arg(Argp, PetscScalar*);
      ierr = DMCompositeScatter_Array(packer,next,gvec,array);CHKERRQ(ierr);
    } else if (next->type == DMCOMPOSITE_DM) {
      Vec vec;
      vec = va_arg(Argp, Vec);
      PetscValidHeaderSpecific(vec,VEC_COOKIE,cnt);
      ierr = DMCompositeScatter_DM(packer,next,gvec,vec);CHKERRQ(ierr);
    } else {
      SETERRQ(PETSC_ERR_SUP,"Cannot handle that object type yet");
    }
    cnt++;
    next = next->next;
  }
  va_end(Argp);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMCompositeGather"
/*@C
    DMCompositeGather - Gathers into a global packed vector from its individual local vectors

    Collective on DMComposite

    Input Parameter:
+    packer - the packer object
.    gvec - the global vector
-    ... - the individual sequential objects (arrays or vectors)
 
    Level: advanced

.seealso DMCompositeDestroy(), DMCompositeAddArray(), DMCompositeAddDM(), DMCompositeCreateGlobalVector(),
         DMCompositeScatter(), DMCompositeCreate(), DMCompositeGetGlobalIndices(), DMCompositeGetAccess(),
         DMCompositeGetLocalVectors(), DMCompositeRestoreLocalVectors(), DMCompositeGetEntries()

@*/
PetscErrorCode PETSCDM_DLLEXPORT DMCompositeGather(DMComposite packer,Vec gvec,...)
{
  va_list                Argp;
  PetscErrorCode         ierr;
  struct DMCompositeLink *next;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(packer,DM_COOKIE,1);
  PetscValidHeaderSpecific(gvec,VEC_COOKIE,2);
  next = packer->next;
  if (!packer->setup) {
    ierr = DMCompositeSetUp(packer);CHKERRQ(ierr);
  }

  /* loop over packed objects, handling one at at time */
  va_start(Argp,gvec);
  while (next) {
    if (next->type == DMCOMPOSITE_ARRAY) {
      PetscScalar *array;
      array = va_arg(Argp, PetscScalar*);
      ierr  = DMCompositeGather_Array(packer,next,gvec,array);CHKERRQ(ierr);
    } else if (next->type == DMCOMPOSITE_DM) {
      Vec vec;
      vec = va_arg(Argp, Vec);
      PetscValidHeaderSpecific(vec,VEC_COOKIE,3);
      ierr = DMCompositeGather_DM(packer,next,gvec,vec);CHKERRQ(ierr);
    } else {
      SETERRQ(PETSC_ERR_SUP,"Cannot handle that object type yet");
    }
    next = next->next;
  }
  va_end(Argp);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMCompositeAddArray"
/*@C
    DMCompositeAddArray - adds an "redundant" array to a DMComposite. The array values will 
       be stored in part of the array on process orank.

    Collective on DMComposite

    Input Parameter:
+    packer - the packer object
.    orank - the process on which the array entries officially live, this number must be
             the same on all processes.
-    n - the length of the array
 
    Level: advanced

.seealso DMCompositeDestroy(), DMCompositeGather(), DMCompositeAddDM(), DMCompositeCreateGlobalVector(),
         DMCompositeScatter(), DMCompositeCreate(), DMCompositeGetGlobalIndices(), DMCompositeGetAccess(),
         DMCompositeGetLocalVectors(), DMCompositeRestoreLocalVectors(), DMCompositeGetEntries()

@*/
PetscErrorCode PETSCDM_DLLEXPORT DMCompositeAddArray(DMComposite packer,PetscMPIInt orank,PetscInt n)
{
  struct DMCompositeLink *mine,*next;
  PetscErrorCode         ierr;
  PetscMPIInt            rank;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(packer,DM_COOKIE,1);
  next = packer->next;
  if (packer->setup) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Cannot add an array once you have used the DMComposite");
  }
#if defined(PETSC_USE_DEBUG)
  {
    PetscMPIInt        orankmax;
    ierr = MPI_Allreduce(&orank,&orankmax,1,MPI_INT,MPI_MAX,((PetscObject)packer)->comm);CHKERRQ(ierr);
    if (orank != orankmax) SETERRQ2(PETSC_ERR_ARG_INCOMP,"orank %d must be equal on all processes, another process has value %d",orank,orankmax);
  }
#endif

  ierr = MPI_Comm_rank(((PetscObject)packer)->comm,&rank);CHKERRQ(ierr);
  /* create new link */
  ierr                = PetscNew(struct DMCompositeLink,&mine);CHKERRQ(ierr);
  mine->n             = n;
  mine->rank          = orank;
  mine->dm            = PETSC_NULL;
  mine->type          = DMCOMPOSITE_ARRAY;
  mine->next          = PETSC_NULL;
  if (rank == mine->rank) {packer->n += n;packer->nmine++;}

  /* add to end of list */
  if (!next) {
    packer->next = mine;
  } else {
    while (next->next) next = next->next;
    next->next = mine;
  }
  packer->nredundant++;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMCompositeAddDM"
/*@C
    DMCompositeAddDM - adds a DM (includes DA) vector to a DMComposite

    Collective on DMComposite

    Input Parameter:
+    packer - the packer object
-    dm - the DM object, if the DM is a da you will need to caste it with a (DM)
 
    Level: advanced

.seealso DMCompositeDestroy(), DMCompositeGather(), DMCompositeAddDM(), DMCompositeCreateGlobalVector(),
         DMCompositeScatter(), DMCompositeCreate(), DMCompositeGetGlobalIndices(), DMCompositeGetAccess(),
         DMCompositeGetLocalVectors(), DMCompositeRestoreLocalVectors(), DMCompositeGetEntries()

@*/
PetscErrorCode PETSCDM_DLLEXPORT DMCompositeAddDM(DMComposite packer,DM dm)
{
  PetscErrorCode         ierr;
  PetscInt               n;
  struct DMCompositeLink *mine,*next;
  Vec                    global;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(packer,DM_COOKIE,1);
  PetscValidHeaderSpecific(dm,DM_COOKIE,2);
  next = packer->next;
  if (packer->setup) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Cannot add a DA once you have used the DMComposite");
  }

  /* create new link */
  ierr = PetscNew(struct DMCompositeLink,&mine);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)dm);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(dm,&global);CHKERRQ(ierr);
  ierr = VecGetLocalSize(global,&n);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dm,&global);CHKERRQ(ierr);
  mine->n      = n;
  mine->dm     = dm;  
  mine->type   = DMCOMPOSITE_DM;
  mine->next   = PETSC_NULL;
  packer->n   += n;

  /* add to end of list */
  if (!next) {
    packer->next = mine;
  } else {
    while (next->next) next = next->next;
    next->next = mine;
  }
  packer->nDM++;
  packer->nmine++;
  PetscFunctionReturn(0);
}

extern PetscErrorCode PETSCDM_DLLEXPORT VecView_MPI(Vec,PetscViewer);
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "VecView_DMComposite"
PetscErrorCode PETSCDM_DLLEXPORT VecView_DMComposite(Vec gvec,PetscViewer viewer)
{
  DMComposite            packer;
  PetscErrorCode         ierr;
  struct DMCompositeLink *next;
  PetscTruth             isdraw;

  PetscFunctionBegin;
  ierr = PetscObjectQuery((PetscObject)gvec,"DMComposite",(PetscObject*)&packer);CHKERRQ(ierr);
  if (!packer) SETERRQ(PETSC_ERR_ARG_WRONG,"Vector not generated from a DMComposite");
  next = packer->next;

  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_DRAW,&isdraw);CHKERRQ(ierr);
  if (!isdraw) {
    /* do I really want to call this? */
    ierr = VecView_MPI(gvec,viewer);CHKERRQ(ierr);
  } else {
    PetscInt cnt = 0;

    /* loop over packed objects, handling one at at time */
    while (next) {
      if (next->type == DMCOMPOSITE_ARRAY) {
	PetscScalar *array;
	ierr  = DMCompositeGetAccess_Array(packer,next,gvec,&array);CHKERRQ(ierr);

	/*skip it for now */
      } else if (next->type == DMCOMPOSITE_DM) {
	Vec      vec;
        PetscInt bs;

	ierr = DMCompositeGetAccess_DM(packer,next,gvec,&vec);CHKERRQ(ierr);
	ierr = VecView(vec,viewer);CHKERRQ(ierr);
        ierr = VecGetBlockSize(vec,&bs);CHKERRQ(ierr);
	ierr = DMCompositeRestoreAccess_DM(packer,next,gvec,&vec);CHKERRQ(ierr);
        ierr = PetscViewerDrawBaseAdd(viewer,bs);CHKERRQ(ierr);
        cnt += bs;
      } else {
	SETERRQ(PETSC_ERR_SUP,"Cannot handle that object type yet");
      }
      next = next->next;
    }
    ierr = PetscViewerDrawBaseAdd(viewer,-cnt);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END


#undef __FUNCT__  
#define __FUNCT__ "DMCompositeCreateGlobalVector"
/*@C
    DMCompositeCreateGlobalVector - Creates a vector of the correct size to be gathered into 
        by the packer.

    Collective on DMComposite

    Input Parameter:
.    packer - the packer object

    Output Parameters:
.   gvec - the global vector

    Level: advanced

    Notes: Once this has been created you cannot add additional arrays or vectors to be packed.

.seealso DMCompositeDestroy(), DMCompositeAddArray(), DMCompositeAddDM(), DMCompositeScatter(),
         DMCompositeGather(), DMCompositeCreate(), DMCompositeGetGlobalIndices(), DMCompositeGetAccess(),
         DMCompositeGetLocalVectors(), DMCompositeRestoreLocalVectors(), DMCompositeGetEntries(),
         DMCompositeCreateLocalVector()

@*/
PetscErrorCode PETSCDM_DLLEXPORT DMCompositeCreateGlobalVector(DMComposite packer,Vec *gvec)
{
  PetscErrorCode         ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(packer,DM_COOKIE,1);
  if (!packer->setup) {
    ierr = DMCompositeSetUp(packer);CHKERRQ(ierr);
  }
  ierr = VecCreateMPI(((PetscObject)packer)->comm,packer->n,packer->N,gvec);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)*gvec,"DMComposite",(PetscObject)packer);CHKERRQ(ierr);
  ierr = VecSetOperation(*gvec,VECOP_VIEW,(void(*)(void))VecView_DMComposite);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMCompositeCreateLocalVector"
/*@C
    DMCompositeCreateLocalVector - Creates a vector of the correct size to contain all ghost points
        and redundant arrays.

    Collective on DMComposite

    Input Parameter:
.    packer - the packer object

    Output Parameters:
.   lvec - the local vector

    Level: advanced

    Notes: Once this has been created you cannot add additional arrays or vectors to be packed.

.seealso DMCompositeDestroy(), DMCompositeAddArray(), DMCompositeAddDM(), DMCompositeScatter(),
         DMCompositeGather(), DMCompositeCreate(), DMCompositeGetGlobalIndices(), DMCompositeGetAccess(),
         DMCompositeGetLocalVectors(), DMCompositeRestoreLocalVectors(), DMCompositeGetEntries(),
         DMCompositeCreateGlobalVector()

@*/
PetscErrorCode PETSCDM_DLLEXPORT DMCompositeCreateLocalVector(DMComposite packer,Vec *lvec)
{
  PetscErrorCode         ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(packer,DM_COOKIE,1);
  if (!packer->setup) {
    ierr = DMCompositeSetUp(packer);CHKERRQ(ierr);
  }
  ierr = VecCreateSeq(((PetscObject)packer)->comm,packer->nghost,lvec);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)*lvec,"DMComposite",(PetscObject)packer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMCompositeGetLocalISs"
/*@C
    DMCompositeGetLocalISs - gets an IS for each DM/array in the DMComposite, include ghost points

    Collective on DMComposite

    Input Parameter:
.    packer - the packer object

    Output Parameters:
.    is - the individual indices for each packed vector/array. Note that this includes
           all the ghost points that individual ghosted DA's may have. Also each process has an 
           is for EACH redundant array (not just the local redundant arrays).
 
    Level: advanced

    Notes:
       The is entries should be destroyed with ISDestroy(), the is array should be freed with PetscFree()

       Use DMCompositeGetGlobalISs() for non-ghosted ISs.

.seealso DMCompositeDestroy(), DMCompositeAddArray(), DMCompositeAddDM(), DMCompositeCreateGlobalVector(),
         DMCompositeGather(), DMCompositeCreate(), DMCompositeGetAccess(), DMCompositeScatter(),
         DMCompositeGetLocalVectors(), DMCompositeRestoreLocalVectors(),DMCompositeGetEntries()

@*/
PetscErrorCode PETSCDM_DLLEXPORT DMCompositeGetGlobalIndices(DMComposite packer,IS *is[])
{
  PetscErrorCode         ierr;
  PetscInt               i,*idx,n,cnt;
  struct DMCompositeLink *next;
  Vec                    global,dglobal;
  PF                     pf;
  PetscScalar            *array;
  PetscMPIInt            rank;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(packer,DM_COOKIE,1);
  ierr = PetscMalloc(packer->nmine*sizeof(IS),is);CHKERRQ(ierr);
  next = packer->next;
  ierr = DMCompositeCreateGlobalVector(packer,&global);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(((PetscObject)packer)->comm,&rank);CHKERRQ(ierr);

  /* put 0 to N-1 into the global vector */
  ierr = PFCreate(PETSC_COMM_WORLD,1,1,&pf);CHKERRQ(ierr);
  ierr = PFSetType(pf,PFIDENTITY,PETSC_NULL);CHKERRQ(ierr);
  ierr = PFApplyVec(pf,PETSC_NULL,global);CHKERRQ(ierr);
  ierr = PFDestroy(pf);CHKERRQ(ierr);

  /* loop over packed objects, handling one at at time */
  cnt = 0;
  while (next) {

    if (next->type == DMCOMPOSITE_ARRAY) {
      
      ierr = PetscMalloc(next->n*sizeof(PetscInt),&idx);CHKERRQ(ierr);
      if (rank == next->rank) {
        ierr   = VecGetArray(global,&array);CHKERRQ(ierr);
        array += next->rstart;
        for (i=0; i<next->n; i++) idx[i] = (PetscInt)PetscRealPart(array[i]);
        array -= next->rstart;
        ierr   = VecRestoreArray(global,&array);CHKERRQ(ierr);
      }
      ierr = MPI_Bcast(idx,next->n,MPIU_INT,next->rank,((PetscObject)packer)->comm);CHKERRQ(ierr);
      ierr = ISCreateGeneral(((PetscObject)packer)->comm,next->n,idx,&(*is)[cnt]);CHKERRQ(ierr);
      ierr = PetscFree(idx);CHKERRQ(ierr);
    } else if (next->type == DMCOMPOSITE_DM) {
      Vec local;

      ierr   = DMCreateLocalVector(next->dm,&local);CHKERRQ(ierr);
      ierr   = VecGetArray(global,&array);CHKERRQ(ierr);
      array += next->rstart;
      ierr   = DMGetGlobalVector(next->dm,&dglobal);CHKERRQ(ierr);
      ierr   = VecPlaceArray(dglobal,array);CHKERRQ(ierr);
      ierr   = DMGlobalToLocalBegin(next->dm,dglobal,INSERT_VALUES,local);CHKERRQ(ierr);
      ierr   = DMGlobalToLocalEnd(next->dm,dglobal,INSERT_VALUES,local);CHKERRQ(ierr);
      array -= next->rstart;
      ierr   = VecRestoreArray(global,&array);CHKERRQ(ierr);
      ierr   = VecResetArray(dglobal);CHKERRQ(ierr);
      ierr   = DMRestoreGlobalVector(next->dm,&dglobal);CHKERRQ(ierr);

      ierr   = VecGetArray(local,&array);CHKERRQ(ierr);
      ierr   = VecGetSize(local,&n);CHKERRQ(ierr);
      ierr   = PetscMalloc(n*sizeof(PetscInt),&idx);CHKERRQ(ierr);
      for (i=0; i<n; i++) idx[i] = (PetscInt)PetscRealPart(array[i]);
      ierr    = VecRestoreArray(local,&array);CHKERRQ(ierr);
      ierr    = VecDestroy(local);CHKERRQ(ierr);
      ierr    = ISCreateGeneral(((PetscObject)packer)->comm,next->n,idx,&(*is)[cnt]);CHKERRQ(ierr);
      ierr    = PetscFree(idx);CHKERRQ(ierr);

    } else {
      SETERRQ(PETSC_ERR_SUP,"Cannot handle that object type yet");
    }
    next = next->next;
    cnt++;
  }
  ierr = VecDestroy(global);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMCompositeGetGlobalISs"
/*@C
    DMCompositeGetGlobalISs - Gets the index sets for each composed object

    Collective on DMComposite

    Input Parameter:
.    packer - the packer object

    Output Parameters:
.    is - the array of index sets
 
    Level: advanced

    Notes:
       The is entries should be destroyed with ISDestroy(), the is array should be freed with PetscFree()

       The number of IS on each process will/may be different when redundant arrays are used

       These could be used to extract a subset of vector entries for a "multi-physics" preconditioner

       Use DMCompositeGetLocalISs() for index sets that include ghost points

.seealso DMCompositeDestroy(), DMCompositeAddArray(), DMCompositeAddDM(), DMCompositeCreateGlobalVector(),
         DMCompositeGather(), DMCompositeCreate(), DMCompositeGetAccess(), DMCompositeScatter(),
         DMCompositeGetLocalVectors(), DMCompositeRestoreLocalVectors(),DMCompositeGetEntries()

@*/
PetscErrorCode PETSCDM_DLLEXPORT DMCompositeGetGlobalISs(DMComposite packer,IS *is[])
{
  PetscErrorCode         ierr;
  PetscInt               cnt = 0;
  struct DMCompositeLink *next;
  PetscMPIInt            rank;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(packer,DM_COOKIE,1);
  ierr = PetscMalloc(packer->nmine*sizeof(IS),is);CHKERRQ(ierr);
  next = packer->next;
  ierr = MPI_Comm_rank(((PetscObject)packer)->comm,&rank);CHKERRQ(ierr);

  /* loop over packed objects, handling one at at time */
  while (next) {

    if (next->type == DMCOMPOSITE_ARRAY) {
      
      if (rank == next->rank) {
        ierr = ISCreateBlock(((PetscObject)packer)->comm,next->n,1,&next->grstart,&(*is)[cnt]);CHKERRQ(ierr);
        cnt++;
      }

    } else if (next->type == DMCOMPOSITE_DM) {

      ierr = ISCreateBlock(((PetscObject)packer)->comm,next->n,1,&next->grstart,&(*is)[cnt]);CHKERRQ(ierr);
      cnt++;

    } else {
      SETERRQ(PETSC_ERR_SUP,"Cannot handle that object type yet");
    }
    next = next->next;
  }
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "DMCompositeGetLocalVectors_Array"
PetscErrorCode DMCompositeGetLocalVectors_Array(DMComposite packer,struct DMCompositeLink *mine,PetscScalar **array)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (array) {
    ierr = PetscMalloc(mine->n*sizeof(PetscScalar),array);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMCompositeGetLocalVectors_DM"
PetscErrorCode DMCompositeGetLocalVectors_DM(DMComposite packer,struct DMCompositeLink *mine,Vec *local)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (local) {
    ierr = DMGetLocalVector(mine->dm,local);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMCompositeRestoreLocalVectors_Array"
PetscErrorCode DMCompositeRestoreLocalVectors_Array(DMComposite packer,struct DMCompositeLink *mine,PetscScalar **array)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (array) {
    ierr = PetscFree(*array);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMCompositeRestoreLocalVectors_DM"
PetscErrorCode DMCompositeRestoreLocalVectors_DM(DMComposite packer,struct DMCompositeLink *mine,Vec *local)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (local) {
    ierr = DMRestoreLocalVector(mine->dm,local);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMCompositeGetLocalVectors"
/*@C
    DMCompositeGetLocalVectors - Gets local vectors and arrays for each part of a DMComposite.'
       Use DMCompositeRestoreLocalVectors() to return them.

    Collective on DMComposite

    Input Parameter:
.    packer - the packer object
 
    Output Parameter:
.    ... - the individual sequential objects (arrays or vectors)
 
    Level: advanced

.seealso DMCompositeDestroy(), DMCompositeAddArray(), DMCompositeAddDM(), DMCompositeCreateGlobalVector(),
         DMCompositeGather(), DMCompositeCreate(), DMCompositeGetGlobalIndices(), DMCompositeGetAccess(), 
         DMCompositeRestoreLocalVectors(), DMCompositeScatter(), DMCompositeGetEntries()

@*/
PetscErrorCode PETSCDM_DLLEXPORT DMCompositeGetLocalVectors(DMComposite packer,...)
{
  va_list                Argp;
  PetscErrorCode         ierr;
  struct DMCompositeLink *next;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(packer,DM_COOKIE,1);
  next = packer->next;
  /* loop over packed objects, handling one at at time */
  va_start(Argp,packer);
  while (next) {
    if (next->type == DMCOMPOSITE_ARRAY) {
      PetscScalar **array;
      array = va_arg(Argp, PetscScalar**);
      ierr = DMCompositeGetLocalVectors_Array(packer,next,array);CHKERRQ(ierr);
    } else if (next->type == DMCOMPOSITE_DM) {
      Vec *vec;
      vec = va_arg(Argp, Vec*);
      ierr = DMCompositeGetLocalVectors_DM(packer,next,vec);CHKERRQ(ierr);
    } else {
      SETERRQ(PETSC_ERR_SUP,"Cannot handle that object type yet");
    }
    next = next->next;
  }
  va_end(Argp);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMCompositeRestoreLocalVectors"
/*@C
    DMCompositeRestoreLocalVectors - Restores local vectors and arrays for each part of a DMComposite.'
       Use VecPakcRestoreLocalVectors() to return them.

    Collective on DMComposite

    Input Parameter:
.    packer - the packer object
 
    Output Parameter:
.    ... - the individual sequential objects (arrays or vectors)
 
    Level: advanced

.seealso DMCompositeDestroy(), DMCompositeAddArray(), DMCompositeAddDM(), DMCompositeCreateGlobalVector(),
         DMCompositeGather(), DMCompositeCreate(), DMCompositeGetGlobalIndices(), DMCompositeGetAccess(), 
         DMCompositeGetLocalVectors(), DMCompositeScatter(), DMCompositeGetEntries()

@*/
PetscErrorCode PETSCDM_DLLEXPORT DMCompositeRestoreLocalVectors(DMComposite packer,...)
{
  va_list                Argp;
  PetscErrorCode         ierr;
  struct DMCompositeLink *next;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(packer,DM_COOKIE,1);
  next = packer->next;
  /* loop over packed objects, handling one at at time */
  va_start(Argp,packer);
  while (next) {
    if (next->type == DMCOMPOSITE_ARRAY) {
      PetscScalar **array;
      array = va_arg(Argp, PetscScalar**);
      ierr = DMCompositeRestoreLocalVectors_Array(packer,next,array);CHKERRQ(ierr);
    } else if (next->type == DMCOMPOSITE_DM) {
      Vec *vec;
      vec = va_arg(Argp, Vec*);
      ierr = DMCompositeRestoreLocalVectors_DM(packer,next,vec);CHKERRQ(ierr);
    } else {
      SETERRQ(PETSC_ERR_SUP,"Cannot handle that object type yet");
    }
    next = next->next;
  }
  va_end(Argp);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "DMCompositeGetEntries_Array"
PetscErrorCode DMCompositeGetEntries_Array(DMComposite packer,struct DMCompositeLink *mine,PetscInt *n)
{
  PetscFunctionBegin;
  if (n) *n = mine->n;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMCompositeGetEntries_DM"
PetscErrorCode DMCompositeGetEntries_DM(DMComposite packer,struct DMCompositeLink *mine,DM *dm)
{
  PetscFunctionBegin;
  if (dm) *dm = mine->dm;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMCompositeGetEntries"
/*@C
    DMCompositeGetEntries - Gets the DA, redundant size, etc for each entry in a DMComposite.

    Collective on DMComposite

    Input Parameter:
.    packer - the packer object
 
    Output Parameter:
.    ... - the individual entries, DAs or integer sizes)
 
    Level: advanced

.seealso DMCompositeDestroy(), DMCompositeAddArray(), DMCompositeAddDM(), DMCompositeCreateGlobalVector(),
         DMCompositeGather(), DMCompositeCreate(), DMCompositeGetGlobalIndices(), DMCompositeGetAccess(), 
         DMCompositeRestoreLocalVectors(), DMCompositeGetLocalVectors(),  DMCompositeScatter(),
         DMCompositeGetLocalVectors(), DMCompositeRestoreLocalVectors()

@*/
PetscErrorCode PETSCDM_DLLEXPORT DMCompositeGetEntries(DMComposite packer,...)
{
  va_list                Argp;
  PetscErrorCode         ierr;
  struct DMCompositeLink *next;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(packer,DM_COOKIE,1);
  next = packer->next;
  /* loop over packed objects, handling one at at time */
  va_start(Argp,packer);
  while (next) {
    if (next->type == DMCOMPOSITE_ARRAY) {
      PetscInt *n;
      n = va_arg(Argp, PetscInt*);
      ierr = DMCompositeGetEntries_Array(packer,next,n);CHKERRQ(ierr);
    } else if (next->type == DMCOMPOSITE_DM) {
      DM *dm;
      dm = va_arg(Argp, DM*);
      ierr = DMCompositeGetEntries_DM(packer,next,dm);CHKERRQ(ierr);
    } else {
      SETERRQ(PETSC_ERR_SUP,"Cannot handle that object type yet");
    }
    next = next->next;
  }
  va_end(Argp);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMCompositeRefine"
/*@C
    DMCompositeRefine - Refines a DMComposite by refining all of its DAs

    Collective on DMComposite

    Input Parameters:
+    packer - the packer object
-    comm - communicator to contain the new DM object, usually PETSC_NULL

    Output Parameter:
.    fine - new packer
 
    Level: advanced

.seealso DMCompositeDestroy(), DMCompositeAddArray(), DMCompositeAddDM(), DMCompositeCreateGlobalVector(),
         DMCompositeGather(), DMCompositeCreate(), DMCompositeGetGlobalIndices(), DMCompositeGetAccess(),
         DMCompositeGetLocalVectors(), DMCompositeRestoreLocalVectors(),  DMCompositeScatter(),
         DMCompositeGetEntries()

@*/
PetscErrorCode PETSCDM_DLLEXPORT DMCompositeRefine(DMComposite packer,MPI_Comm comm,DMComposite *fine)
{
  PetscErrorCode         ierr;
  struct DMCompositeLink *next;
  DM                     dm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(packer,DM_COOKIE,1);
  next = packer->next;
  ierr = DMCompositeCreate(comm,fine);CHKERRQ(ierr);

  /* loop over packed objects, handling one at at time */
  while (next) {
    if (next->type == DMCOMPOSITE_ARRAY) {
      ierr = DMCompositeAddArray(*fine,next->rank,next->n);CHKERRQ(ierr);
    } else if (next->type == DMCOMPOSITE_DM) {
      ierr = DMRefine(next->dm,comm,&dm);CHKERRQ(ierr);
      ierr = DMCompositeAddDM(*fine,dm);CHKERRQ(ierr);
      ierr = PetscObjectDereference((PetscObject)dm);CHKERRQ(ierr);
    } else {
      SETERRQ(PETSC_ERR_SUP,"Cannot handle that object type yet");
    }
    next = next->next;
  }
  PetscFunctionReturn(0);
}

#include "petscmat.h"

struct MatPackLink {
  Mat                A;
  struct MatPackLink *next;
};

struct MatPack {
  DMComposite            right,left;
  struct MatPackLink *next;
};

#undef __FUNCT__  
#define __FUNCT__ "MatMultBoth_Shell_Pack"
PetscErrorCode MatMultBoth_Shell_Pack(Mat A,Vec x,Vec y,PetscTruth add)
{
  struct MatPack         *mpack;
  struct DMCompositeLink *xnext,*ynext;
  struct MatPackLink     *anext;
  PetscScalar            *xarray,*yarray;
  PetscErrorCode         ierr;
  PetscInt               i;
  Vec                    xglobal,yglobal;
  PetscMPIInt            rank;

  PetscFunctionBegin;
  ierr = MatShellGetContext(A,(void**)&mpack);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(((PetscObject)mpack->right)->comm,&rank);CHKERRQ(ierr);
  xnext = mpack->right->next;
  ynext = mpack->left->next;
  anext = mpack->next;

  while (xnext) {
    if (xnext->type == DMCOMPOSITE_ARRAY) {
      if (rank == xnext->rank) {
        ierr    = VecGetArray(x,&xarray);CHKERRQ(ierr);
        ierr    = VecGetArray(y,&yarray);CHKERRQ(ierr);
        if (add) {
          for (i=0; i<xnext->n; i++) {
            yarray[ynext->rstart+i] += xarray[xnext->rstart+i];
          }
        } else {
          ierr    = PetscMemcpy(yarray+ynext->rstart,xarray+xnext->rstart,xnext->n*sizeof(PetscScalar));CHKERRQ(ierr);
        }
        ierr    = VecRestoreArray(x,&xarray);CHKERRQ(ierr);
        ierr    = VecRestoreArray(y,&yarray);CHKERRQ(ierr);
      }
    } else if (xnext->type == DMCOMPOSITE_DM) {
      ierr  = VecGetArray(x,&xarray);CHKERRQ(ierr);
      ierr  = VecGetArray(y,&yarray);CHKERRQ(ierr);
      ierr  = DMGetGlobalVector(xnext->dm,&xglobal);CHKERRQ(ierr);
      ierr  = DMGetGlobalVector(ynext->dm,&yglobal);CHKERRQ(ierr);
      ierr  = VecPlaceArray(xglobal,xarray+xnext->rstart);CHKERRQ(ierr);
      ierr  = VecPlaceArray(yglobal,yarray+ynext->rstart);CHKERRQ(ierr);
      if (add) {
        ierr  = MatMultAdd(anext->A,xglobal,yglobal,yglobal);CHKERRQ(ierr);
      } else {
        ierr  = MatMult(anext->A,xglobal,yglobal);CHKERRQ(ierr);
      }
      ierr  = VecRestoreArray(x,&xarray);CHKERRQ(ierr);
      ierr  = VecRestoreArray(y,&yarray);CHKERRQ(ierr);
      ierr  = VecResetArray(xglobal);CHKERRQ(ierr);
      ierr  = VecResetArray(yglobal);CHKERRQ(ierr);
      ierr  = DMRestoreGlobalVector(xnext->dm,&xglobal);CHKERRQ(ierr);
      ierr  = DMRestoreGlobalVector(ynext->dm,&yglobal);CHKERRQ(ierr);
      anext = anext->next;
    } else {
      SETERRQ(PETSC_ERR_SUP,"Cannot handle that object type yet");
    }
    xnext = xnext->next;
    ynext = ynext->next;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultAdd_Shell_Pack"
PetscErrorCode MatMultAdd_Shell_Pack(Mat A,Vec x,Vec y,Vec z)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (z != y) SETERRQ(PETSC_ERR_SUP,"Handles y == z only");
  ierr = MatMultBoth_Shell_Pack(A,x,y,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMult_Shell_Pack"
PetscErrorCode MatMult_Shell_Pack(Mat A,Vec x,Vec y)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = MatMultBoth_Shell_Pack(A,x,y,PETSC_FALSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultTranspose_Shell_Pack"
PetscErrorCode MatMultTranspose_Shell_Pack(Mat A,Vec x,Vec y)
{
  struct MatPack         *mpack;
  struct DMCompositeLink *xnext,*ynext;
  struct MatPackLink     *anext;
  PetscScalar            *xarray,*yarray;
  PetscErrorCode         ierr;
  Vec                    xglobal,yglobal;
  PetscMPIInt            rank;

  PetscFunctionBegin;
  ierr  = MatShellGetContext(A,(void**)&mpack);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(((PetscObject)mpack->right)->comm,&rank);CHKERRQ(ierr);
  xnext = mpack->left->next;
  ynext = mpack->right->next;
  anext = mpack->next;

  while (xnext) {
    if (xnext->type == DMCOMPOSITE_ARRAY) {
      if (rank == ynext->rank) {
        ierr    = VecGetArray(x,&xarray);CHKERRQ(ierr);
        ierr    = VecGetArray(y,&yarray);CHKERRQ(ierr);
        ierr    = PetscMemcpy(yarray+ynext->rstart,xarray+xnext->rstart,xnext->n*sizeof(PetscScalar));CHKERRQ(ierr);
        ierr    = VecRestoreArray(x,&xarray);CHKERRQ(ierr);
        ierr    = VecRestoreArray(y,&yarray);CHKERRQ(ierr);
      }
    } else if (xnext->type == DMCOMPOSITE_DM) {
      ierr  = VecGetArray(x,&xarray);CHKERRQ(ierr);
      ierr  = VecGetArray(y,&yarray);CHKERRQ(ierr);
      ierr  = DMGetGlobalVector(xnext->dm,&xglobal);CHKERRQ(ierr);
      ierr  = DMGetGlobalVector(ynext->dm,&yglobal);CHKERRQ(ierr);
      ierr  = VecPlaceArray(xglobal,xarray+xnext->rstart);CHKERRQ(ierr);
      ierr  = VecPlaceArray(yglobal,yarray+ynext->rstart);CHKERRQ(ierr);
      ierr  = MatMultTranspose(anext->A,xglobal,yglobal);CHKERRQ(ierr);
      ierr  = VecRestoreArray(x,&xarray);CHKERRQ(ierr);
      ierr  = VecRestoreArray(y,&yarray);CHKERRQ(ierr);
      ierr  = VecResetArray(xglobal);CHKERRQ(ierr);
      ierr  = VecResetArray(yglobal);CHKERRQ(ierr);
      ierr  = DMRestoreGlobalVector(xnext->dm,&xglobal);CHKERRQ(ierr);
      ierr  = DMRestoreGlobalVector(ynext->dm,&yglobal);CHKERRQ(ierr);
      anext = anext->next;
    } else {
      SETERRQ(PETSC_ERR_SUP,"Cannot handle that object type yet");
    }
    xnext = xnext->next;
    ynext = ynext->next;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatDestroy_Shell_Pack"
PetscErrorCode MatDestroy_Shell_Pack(Mat A)
{
  struct MatPack     *mpack;
  struct MatPackLink *anext,*oldanext;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr  = MatShellGetContext(A,(void**)&mpack);CHKERRQ(ierr);
  anext = mpack->next;

  while (anext) {
    ierr     = MatDestroy(anext->A);CHKERRQ(ierr);
    oldanext = anext;
    anext    = anext->next;
    ierr     = PetscFree(oldanext);CHKERRQ(ierr);
  }
  ierr = PetscFree(mpack);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)A,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMCompositeGetInterpolation"
/*@C
    DMCompositeGetInterpolation - GetInterpolations a DMComposite by refining all of its DAs

    Collective on DMComposite

    Input Parameters:
+    coarse - coarse grid packer
-    fine - fine grid packer

    Output Parameter:
+    A - interpolation matrix
-    v - scaling vector
 
    Level: advanced

.seealso DMCompositeDestroy(), DMCompositeAddArray(), DMCompositeAddDM(), DMCompositeCreateGlobalVector(),
         DMCompositeGather(), DMCompositeCreate(), DMCompositeGetGlobalIndices(), DMCompositeGetAccess(),
         DMCompositeGetLocalVectors(), DMCompositeRestoreLocalVectors(),  DMCompositeScatter(),DMCompositeGetEntries()

@*/
PetscErrorCode PETSCDM_DLLEXPORT DMCompositeGetInterpolation(DMComposite coarse,DMComposite fine,Mat *A,Vec *v)
{
  PetscErrorCode         ierr;
  PetscInt               m,n,M,N;
  struct DMCompositeLink *nextc;
  struct DMCompositeLink *nextf;
  struct MatPackLink     *nextmat,*pnextmat = 0;
  struct MatPack         *mpack;
  Vec                    gcoarse,gfine;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(coarse,DM_COOKIE,1);
  PetscValidHeaderSpecific(fine,DM_COOKIE,2);
  nextc = coarse->next;
  nextf = fine->next;
  /* use global vectors only for determining matrix layout */
  ierr = DMCompositeCreateGlobalVector(coarse,&gcoarse);CHKERRQ(ierr);
  ierr = DMCompositeCreateGlobalVector(fine,&gfine);CHKERRQ(ierr);
  ierr = VecGetLocalSize(gcoarse,&n);CHKERRQ(ierr);
  ierr = VecGetLocalSize(gfine,&m);CHKERRQ(ierr);
  ierr = VecGetSize(gcoarse,&N);CHKERRQ(ierr);
  ierr = VecGetSize(gfine,&M);CHKERRQ(ierr);
  ierr = VecDestroy(gcoarse);CHKERRQ(ierr);
  ierr = VecDestroy(gfine);CHKERRQ(ierr);

  ierr         = PetscNew(struct MatPack,&mpack);CHKERRQ(ierr);
  mpack->right = coarse;
  mpack->left  = fine;
  ierr  = MatCreate(((PetscObject)fine)->comm,A);CHKERRQ(ierr);
  ierr  = MatSetSizes(*A,m,n,M,N);CHKERRQ(ierr);
  ierr  = MatSetType(*A,MATSHELL);CHKERRQ(ierr);
  ierr  = MatShellSetContext(*A,mpack);CHKERRQ(ierr);
  ierr  = MatShellSetOperation(*A,MATOP_MULT,(void(*)(void))MatMult_Shell_Pack);CHKERRQ(ierr);
  ierr  = MatShellSetOperation(*A,MATOP_MULT_TRANSPOSE,(void(*)(void))MatMultTranspose_Shell_Pack);CHKERRQ(ierr);
  ierr  = MatShellSetOperation(*A,MATOP_MULT_ADD,(void(*)(void))MatMultAdd_Shell_Pack);CHKERRQ(ierr);
  ierr  = MatShellSetOperation(*A,MATOP_DESTROY,(void(*)(void))MatDestroy_Shell_Pack);CHKERRQ(ierr);

  /* loop over packed objects, handling one at at time */
  while (nextc) {
    if (nextc->type != nextf->type) SETERRQ(PETSC_ERR_ARG_INCOMP,"Two DMComposite have different layout");

    if (nextc->type == DMCOMPOSITE_ARRAY) {
      ;
    } else if (nextc->type == DMCOMPOSITE_DM) {
      ierr          = PetscNew(struct MatPackLink,&nextmat);CHKERRQ(ierr);
      nextmat->next = 0;
      if (pnextmat) {
        pnextmat->next = nextmat;
        pnextmat       = nextmat;
      } else {
        pnextmat    = nextmat;
        mpack->next = nextmat;
      }
      ierr = DMGetInterpolation(nextc->dm,nextf->dm,&nextmat->A,PETSC_NULL);CHKERRQ(ierr);
    } else {
      SETERRQ(PETSC_ERR_SUP,"Cannot handle that object type yet");
    }
    nextc = nextc->next;
    nextf = nextf->next;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMCompositeGetMatrix" 
/*@C
    DMCompositeGetMatrix - Creates a matrix with the correct parallel layout and nonzero structure required for 
      computing the Jacobian on a function defined using the stencils set in the DA's and coupling in the array variables

    Collective on DA

    Input Parameter:
+   packer - the distributed array
-   mtype - Supported types are MATSEQAIJ, MATMPIAIJ

    Output Parameters:
.   J  - matrix with the correct nonzero structure
        (obviously without the correct Jacobian values)

    Level: advanced

    Notes: This properly preallocates the number of nonzeros in the sparse matrix so you 
       do not need to do it yourself. 


.seealso DAGetMatrix(), DMCompositeCreate()

@*/
PetscErrorCode PETSCDM_DLLEXPORT DMCompositeGetMatrix(DMComposite packer, const MatType mtype,Mat *J)
{
  PetscErrorCode         ierr;
  struct DMCompositeLink *next = packer->next;
  PetscInt               m,*dnz,*onz,i,j,mA;
  Mat                    Atmp;
  PetscMPIInt            rank;
  PetscScalar            zero = 0.0;
  PetscTruth             dense = PETSC_FALSE;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(packer,DM_COOKIE,1);

  /* use global vector to determine layout needed for matrix */
  m = packer->n;
  ierr = MPI_Comm_rank(((PetscObject)packer)->comm,&rank);CHKERRQ(ierr);
  ierr = MatCreate(((PetscObject)packer)->comm,J);CHKERRQ(ierr);
  ierr = MatSetSizes(*J,m,m,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatSetType(*J,MATAIJ);CHKERRQ(ierr);

  /*
     Extremely inefficient but will compute entire Jacobian for testing
  */
  ierr = PetscOptionsGetTruth(PETSC_NULL,"-dmcomposite_dense_jacobian",&dense,PETSC_NULL);CHKERRQ(ierr);
  if (dense) {
    PetscInt    rstart,rend,*indices;
    PetscScalar *values;

    mA    = packer->N;
    ierr = MatMPIAIJSetPreallocation(*J,mA,PETSC_NULL,mA-m,PETSC_NULL);CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(*J,mA,PETSC_NULL);CHKERRQ(ierr);

    ierr = MatGetOwnershipRange(*J,&rstart,&rend);CHKERRQ(ierr);
    ierr = PetscMalloc2(mA,PetscScalar,&values,mA,PetscInt,&indices);CHKERRQ(ierr);
    ierr = PetscMemzero(values,mA*sizeof(PetscScalar));CHKERRQ(ierr);
    for (i=0; i<mA; i++) indices[i] = i;
    for (i=rstart; i<rend; i++) {
      ierr = MatSetValues(*J,1,&i,mA,indices,values,INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = PetscFree2(values,indices);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);  
    ierr = MatAssemblyEnd(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);  
    PetscFunctionReturn(0);
  }

  ierr = MatPreallocateInitialize(((PetscObject)packer)->comm,m,m,dnz,onz);CHKERRQ(ierr);
  /* loop over packed objects, handling one at at time */
  next = packer->next;
  while (next) {
    if (next->type == DMCOMPOSITE_ARRAY) {
      if (rank == next->rank) {  /* zero the "little" block */
        for (j=packer->rstart+next->rstart; j<packer->rstart+next->rstart+next->n; j++) {
          for (i=packer->rstart+next->rstart; i<packer->rstart+next->rstart+next->n; i++) {
            ierr = MatPreallocateSet(j,1,&i,dnz,onz);CHKERRQ(ierr);
          }
        }
      }
    } else if (next->type == DMCOMPOSITE_DM) {
      PetscInt       nc,rstart,*ccols,maxnc;
      const PetscInt *cols,*rstarts;
      PetscMPIInt    proc;

      ierr = DMGetMatrix(next->dm,mtype,&Atmp);CHKERRQ(ierr);
      ierr = MatGetOwnershipRange(Atmp,&rstart,PETSC_NULL);CHKERRQ(ierr);
      ierr = MatGetOwnershipRanges(Atmp,&rstarts);CHKERRQ(ierr);
      ierr = MatGetLocalSize(Atmp,&mA,PETSC_NULL);CHKERRQ(ierr);

      maxnc = 0;
      for (i=0; i<mA; i++) {
        ierr  = MatGetRow(Atmp,rstart+i,&nc,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
        ierr  = MatRestoreRow(Atmp,rstart+i,&nc,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
        maxnc = PetscMax(nc,maxnc);
      }
      ierr = PetscMalloc(maxnc*sizeof(PetscInt),&ccols);CHKERRQ(ierr);
      for (i=0; i<mA; i++) {
        ierr = MatGetRow(Atmp,rstart+i,&nc,&cols,PETSC_NULL);CHKERRQ(ierr);
        /* remap the columns taking into how much they are shifted on each process */
        for (j=0; j<nc; j++) {
          proc = 0;
          while (cols[j] >= rstarts[proc+1]) proc++;
          ccols[j] = cols[j] + next->grstarts[proc] - rstarts[proc];
        } 
        ierr = MatPreallocateSet(packer->rstart+next->rstart+i,nc,ccols,dnz,onz);CHKERRQ(ierr);
        ierr = MatRestoreRow(Atmp,rstart+i,&nc,&cols,PETSC_NULL);CHKERRQ(ierr);
      }
      ierr = PetscFree(ccols);CHKERRQ(ierr);
      ierr = MatDestroy(Atmp);CHKERRQ(ierr);
    } else {
      SETERRQ(PETSC_ERR_SUP,"Cannot handle that object type yet");
    }
    next = next->next;
  }
  if (packer->FormCoupleLocations) {
    ierr = (*packer->FormCoupleLocations)(packer,PETSC_NULL,dnz,onz,__rstart,__nrows,__start,__end);CHKERRQ(ierr);
  }
  ierr = MatMPIAIJSetPreallocation(*J,0,dnz,0,onz);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(*J,0,dnz);CHKERRQ(ierr);
  ierr = MatPreallocateFinalize(dnz,onz);CHKERRQ(ierr);

  next = packer->next;
  while (next) {
    if (next->type == DMCOMPOSITE_ARRAY) {
      if (rank == next->rank) {
        for (j=packer->rstart+next->rstart; j<packer->rstart+next->rstart+next->n; j++) {
          for (i=packer->rstart+next->rstart; i<packer->rstart+next->rstart+next->n; i++) {
            ierr = MatSetValues(*J,1,&j,1,&i,&zero,INSERT_VALUES);CHKERRQ(ierr);
          }
        }
      }
    } else if (next->type == DMCOMPOSITE_DM) {
      PetscInt          nc,rstart,row,maxnc,*ccols;
      const PetscInt    *cols,*rstarts;
      const PetscScalar *values;
      PetscMPIInt       proc;

      ierr = DMGetMatrix(next->dm,mtype,&Atmp);CHKERRQ(ierr);
      ierr = MatGetOwnershipRange(Atmp,&rstart,PETSC_NULL);CHKERRQ(ierr);
      ierr = MatGetOwnershipRanges(Atmp,&rstarts);CHKERRQ(ierr);
      ierr = MatGetLocalSize(Atmp,&mA,PETSC_NULL);CHKERRQ(ierr);
      maxnc = 0;
      for (i=0; i<mA; i++) {
        ierr  = MatGetRow(Atmp,rstart+i,&nc,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
        ierr  = MatRestoreRow(Atmp,rstart+i,&nc,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
        maxnc = PetscMax(nc,maxnc);
      }
      ierr = PetscMalloc(maxnc*sizeof(PetscInt),&ccols);CHKERRQ(ierr);
      for (i=0; i<mA; i++) {
        ierr = MatGetRow(Atmp,rstart+i,&nc,(const PetscInt **)&cols,&values);CHKERRQ(ierr);
        for (j=0; j<nc; j++) {
          proc = 0;
          while (cols[j] >= rstarts[proc+1]) proc++;
          ccols[j] = cols[j] + next->grstarts[proc] - rstarts[proc];
        } 
        row  = packer->rstart+next->rstart+i;
        ierr = MatSetValues(*J,1,&row,nc,ccols,values,INSERT_VALUES);CHKERRQ(ierr);
        ierr = MatRestoreRow(Atmp,rstart+i,&nc,(const PetscInt **)&cols,&values);CHKERRQ(ierr);
      }
      ierr = PetscFree(ccols);CHKERRQ(ierr);
      ierr = MatDestroy(Atmp);CHKERRQ(ierr);
    } else {
      SETERRQ(PETSC_ERR_SUP,"Cannot handle that object type yet");
    }
    next = next->next;
  }
  if (packer->FormCoupleLocations) {
    PetscInt __rstart;
    ierr = MatGetOwnershipRange(*J,&__rstart,PETSC_NULL);CHKERRQ(ierr);
    ierr = (*packer->FormCoupleLocations)(packer,*J,PETSC_NULL,PETSC_NULL,__rstart,0,0,0);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);  
  ierr = MatAssemblyEnd(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);  
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMCompositeGetColoring" 
/*@
    DMCompositeGetColoring - Gets the coloring required for computing the Jacobian via
    finite differences on a function defined using a DMComposite "grid"

    Collective on DA

    Input Parameter:
+   dmcomposite - the DMComposite object
.   ctype - IS_COLORING_GLOBAL or IS_COLORING_GHOSTED
-   mtype - MATAIJ or MATBAIJ

    Output Parameters:
.   coloring - matrix coloring for use in computing Jacobians (or PETSC_NULL if not needed)

    Level: advanced

    Notes: This colors each diagonal block (associated with a single DM) with a different set of colors;
      this it will compute the diagonal blocks of the Jacobian correctly. The off diagonal blocks are
      not computed, hence the Jacobian computed is not the entire Jacobian. If -dmcomposite_dense_jacobian
      is used then each column of the Jacobian is given a different color so the full Jacobian is computed
      correctly.

    Notes: These compute the graph coloring of the graph of A^{T}A. The coloring used 
   for efficient (parallel or thread based) triangular solves etc is NOT yet 
   available. 


.seealso ISColoringView(), ISColoringGetIS(), MatFDColoringCreate(), ISColoringType, ISColoring, DAGetColoring()

@*/
PetscErrorCode PETSCDM_DLLEXPORT DMCompositeGetColoring(DMComposite dmcomposite,ISColoringType ctype,const MatType mtype,ISColoring *coloring)
{
  PetscErrorCode         ierr;
  PetscInt               n,i,cnt;
  ISColoringValue        *colors;
  PetscTruth             dense = PETSC_FALSE;
  ISColoringValue        maxcol = 0;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dmcomposite,DM_COOKIE,1);
  if (ctype == IS_COLORING_GHOSTED) {
    SETERRQ(PETSC_ERR_SUP,"Currently you must use -dmmg_iscoloring_type global" );
  } else if (ctype == IS_COLORING_GLOBAL) {
    n = dmcomposite->n;
  } else SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Unknown ISColoringType");
  ierr = PetscMalloc(n*sizeof(ISColoringValue),&colors);CHKERRQ(ierr); /* freed in ISColoringDestroy() */

  ierr = PetscOptionsGetTruth(PETSC_NULL,"-dmcomposite_dense_jacobian",&dense,PETSC_NULL);CHKERRQ(ierr);
  if (dense) {
    for (i=0; i<n; i++) {
      colors[i] = (ISColoringValue)(dmcomposite->rstart + i);
    }
    maxcol = dmcomposite->N;
  } else {
    struct DMCompositeLink *next = dmcomposite->next;
    PetscMPIInt            rank;
  
    ierr = MPI_Comm_rank(((PetscObject)dmcomposite)->comm,&rank);CHKERRQ(ierr);
    cnt  = 0;
    while (next) {
      if (next->type == DMCOMPOSITE_ARRAY) {
        if (rank == next->rank) {  /* each column gets is own color */
          for (i=dmcomposite->rstart+next->rstart; i<dmcomposite->rstart+next->rstart+next->n; i++) {
            colors[cnt++] = maxcol++;
          }
        }
        ierr = MPI_Bcast(&maxcol,1,MPIU_COLORING_VALUE,next->rank,((PetscObject)dmcomposite)->comm);CHKERRQ(ierr);
      } else if (next->type == DMCOMPOSITE_DM) {
        ISColoring     lcoloring;

        ierr = DMGetColoring(next->dm,IS_COLORING_GLOBAL,mtype,&lcoloring);CHKERRQ(ierr);
        for (i=0; i<lcoloring->N; i++) {
          colors[cnt++] = maxcol + lcoloring->colors[i];
        }
        maxcol += lcoloring->n;
        ierr = ISColoringDestroy(lcoloring);CHKERRQ(ierr);
      } else {
        SETERRQ(PETSC_ERR_SUP,"Cannot handle that object type yet");
      }
      next = next->next;
    }
  }
  ierr = ISColoringCreate(((PetscObject)dmcomposite)->comm,maxcol,n,colors,coloring);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMCompositeGlobalToLocalBegin"
/*@C
    DMCompositeGlobalToLocalBegin - begin update of single local vector from global vector

    Collective on DMComposite

    Input Parameter:
+    packer - the packer object
.    gvec - the global vector
-    lvec - single local vector
 
    Level: advanced

.seealso DMCompositeDestroy(), DMCompositeAddArray(), DMCompositeAddDM(), DMCompositeCreateGlobalVector(),
         DMCompositeGather(), DMCompositeCreate(), DMCompositeGetGlobalIndices(), DMCompositeGetAccess(),
         DMCompositeGetLocalVectors(), DMCompositeRestoreLocalVectors(), DMCompositeGetEntries()

@*/
PetscErrorCode PETSCDM_DLLEXPORT DMCompositeGlobalToLocalBegin(DMComposite packer,Vec gvec,InsertMode mode,Vec lvec)
{
  PetscErrorCode         ierr;
  struct DMCompositeLink *next;
  PetscInt               cnt = 3;
  PetscMPIInt            rank;
  PetscScalar            *garray,*larray;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(packer,DM_COOKIE,1);
  PetscValidHeaderSpecific(gvec,VEC_COOKIE,2);
  next = packer->next;
  if (!packer->setup) {
    ierr = DMCompositeSetUp(packer);CHKERRQ(ierr);
  }
  ierr = MPI_Comm_rank(((PetscObject)packer)->comm,&rank);CHKERRQ(ierr);
  ierr = VecGetArray(gvec,&garray);CHKERRQ(ierr);
  ierr = VecGetArray(lvec,&larray);CHKERRQ(ierr);

  /* loop over packed objects, handling one at at time */
  while (next) {
    if (next->type == DMCOMPOSITE_ARRAY) {
      if (rank == next->rank) {
        ierr    = PetscMemcpy(larray,garray,next->n*sizeof(PetscScalar));CHKERRQ(ierr);
        garray += next->n;
      }
      /* does not handle ADD_VALUES */
      ierr = MPI_Bcast(larray,next->n,MPIU_SCALAR,next->rank,((PetscObject)packer)->comm);CHKERRQ(ierr);
      larray += next->n;
    } else if (next->type == DMCOMPOSITE_DM) {
      Vec      local,global;
      PetscInt N;

      ierr = DMGetGlobalVector(next->dm,&global);CHKERRQ(ierr);
      ierr = VecGetLocalSize(global,&N);CHKERRQ(ierr);
      ierr = VecPlaceArray(global,garray);CHKERRQ(ierr);
      ierr = DMGetLocalVector(next->dm,&local);CHKERRQ(ierr);
      ierr = VecPlaceArray(local,larray);CHKERRQ(ierr);
      ierr = DMGlobalToLocalBegin(next->dm,global,mode,local);CHKERRQ(ierr);
      ierr = DMGlobalToLocalEnd(next->dm,global,mode,local);CHKERRQ(ierr);
      ierr = VecResetArray(global);CHKERRQ(ierr);
      ierr = VecResetArray(local);CHKERRQ(ierr);
      ierr = DMRestoreGlobalVector(next->dm,&global);CHKERRQ(ierr)
      ierr = DMRestoreGlobalVector(next->dm,&local);CHKERRQ(ierr);
      larray += next->n;
    } else {
      SETERRQ(PETSC_ERR_SUP,"Cannot handle that object type yet");
    }
    cnt++;
    next = next->next;
  }

  ierr = VecRestoreArray(gvec,PETSC_NULL);CHKERRQ(ierr);
  ierr = VecRestoreArray(lvec,PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMCompositeGlobalToLocalEnd"
/*@C
    DMCompositeGlobalToLocalEnd - All communication is handled in the Begin phase

    Collective on DMComposite

    Input Parameter:
+    packer - the packer object
.    gvec - the global vector
-    lvec - single local vector
 
    Level: advanced

.seealso DMCompositeDestroy(), DMCompositeAddArray(), DMCompositeAddDM(), DMCompositeCreateGlobalVector(),
         DMCompositeGather(), DMCompositeCreate(), DMCompositeGetGlobalIndices(), DMCompositeGetAccess(),
         DMCompositeGetLocalVectors(), DMCompositeRestoreLocalVectors(), DMCompositeGetEntries()

@*/
PetscErrorCode PETSCDM_DLLEXPORT DMCompositeGlobalToLocalEnd(DMComposite packer,Vec gvec,InsertMode mode,Vec lvec)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}
