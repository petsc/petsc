
#include "packimpl.h" /*I   "petscdmcomposite.h"   I*/

#undef __FUNCT__  
#define __FUNCT__ "DMCompositeSetCoupling"
/*@C
    DMCompositeSetCoupling - Sets user provided routines that compute the coupling between the 
      seperate components (DMDA's and arrays) in a DMto build the correct matrix nonzero structure.


    Logically Collective on MPI_Comm

    Input Parameter:
+   dm - the composite object
-   formcouplelocations - routine to set the nonzero locations in the matrix

    Level: advanced

    Notes: See DMCompositeSetContext() and DMCompositeGetContext() for how to get user information into
        this routine

@*/
PetscErrorCode  DMCompositeSetCoupling(DM dm,PetscErrorCode (*FormCoupleLocations)(DM,Mat,PetscInt*,PetscInt*,PetscInt,PetscInt,PetscInt,PetscInt))
{
  DM_Composite *com = (DM_Composite*)dm->data;

  PetscFunctionBegin;
  com->FormCoupleLocations = FormCoupleLocations;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMCompositeSetContext"
/*@
    DMCompositeSetContext - Allows user to stash data they may need within the form coupling routine they 
      set with DMCompositeSetCoupling()


    Not Collective

    Input Parameter:
+   dm - the composite object
-   ctx - the user supplied context

    Level: advanced

    Notes: Use DMCompositeGetContext() to retrieve the context when needed.

@*/
PetscErrorCode  DMCompositeSetContext(DM dm,void *ctx)
{
  PetscFunctionBegin;
  dm->ctx = ctx;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMCompositeGetContext"
/*@
    DMCompositeGetContext - Access the context set with DMCompositeSetContext()


    Not Collective

    Input Parameter:
.   dm - the composite object

    Output Parameter:
.    ctx - the user supplied context

    Level: advanced

    Notes: Use DMCompositeGetContext() to retrieve the context when needed.

@*/
PetscErrorCode  DMCompositeGetContext(DM dm,void **ctx)
{
  PetscFunctionBegin;
  *ctx = dm->ctx;
  PetscFunctionReturn(0);
}



extern PetscErrorCode DMDestroy_Private(DM,PetscBool *);

#undef __FUNCT__  
#define __FUNCT__ "DMDestroy_Composite"
PetscErrorCode  DMDestroy_Composite(DM dm)
{
  PetscErrorCode         ierr;
  struct DMCompositeLink *next, *prev;
  PetscBool              done;
  DM_Composite           *com = (DM_Composite *)dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMDestroy_Private((DM)dm,&done);CHKERRQ(ierr);
  if (!done) PetscFunctionReturn(0);

  next = com->next;
  while (next) {
    prev = next;
    next = next->next;
    if (prev->type == DMCOMPOSITE_DM) {
      ierr = DMDestroy(prev->dm);CHKERRQ(ierr);
    }
    ierr = PetscFree(prev->grstarts);CHKERRQ(ierr);
    ierr = PetscFree(prev);CHKERRQ(ierr);
  }
  ierr = PetscFree(com);CHKERRQ(ierr);
  ierr = PetscHeaderDestroy(dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMView_Composite"
PetscErrorCode  DMView_Composite(DM dm,PetscViewer v)
{
  PetscErrorCode ierr;
  PetscBool      iascii;
  DM_Composite   *com = (DM_Composite *)dm->data;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)v,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    struct DMCompositeLink *lnk = com->next;
    PetscInt               i;

    ierr = PetscViewerASCIIPrintf(v,"DM (%s)\n",((PetscObject)dm)->prefix?((PetscObject)dm)->prefix:"no prefix");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(v,"  contains %d DMs and %d redundant arrays\n",com->nDM,com->nredundant);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(v);CHKERRQ(ierr);
    for (i=0; lnk; lnk=lnk->next,i++) {
      if (lnk->dm) {
        ierr = PetscViewerASCIIPrintf(v,"Link %d: DM of type %s\n",i,((PetscObject)lnk->dm)->type_name);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPushTab(v);CHKERRQ(ierr);
        ierr = DMView(lnk->dm,v);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPopTab(v);CHKERRQ(ierr);
      } else {
        ierr = PetscViewerASCIIPrintf(v,"Link %d: Redundant array of size %d owned by rank %d\n",i,lnk->nlocal,lnk->rank);CHKERRQ(ierr);
      }
    }
    ierr = PetscViewerASCIIPopTab(v);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "DMSetUp_Composite"
PetscErrorCode  DMSetUp_Composite(DM dm)
{
  PetscErrorCode         ierr;
  PetscInt               nprev = 0;
  PetscMPIInt            rank,size;
  DM_Composite           *com = (DM_Composite*)dm->data;
  struct DMCompositeLink *next = com->next;
  PetscLayout            map;

  PetscFunctionBegin;
  if (com->setup) SETERRQ(((PetscObject)dm)->comm,PETSC_ERR_ARG_WRONGSTATE,"Packer has already been setup");
  ierr = PetscLayoutCreate(((PetscObject)dm)->comm,&map);CHKERRQ(ierr);
  ierr = PetscLayoutSetLocalSize(map,com->n);CHKERRQ(ierr);
  ierr = PetscLayoutSetSize(map,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = PetscLayoutSetBlockSize(map,1);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(map);CHKERRQ(ierr);
  ierr = PetscLayoutGetSize(map,&com->N);CHKERRQ(ierr);
  ierr = PetscLayoutGetRange(map,&com->rstart,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscLayoutDestroy(map);CHKERRQ(ierr);
    
  /* now set the rstart for each linked array/vector */
  ierr = MPI_Comm_rank(((PetscObject)dm)->comm,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(((PetscObject)dm)->comm,&size);CHKERRQ(ierr);
  while (next) {
    next->rstart = nprev; 
    nprev += next->n;
    next->grstart = com->rstart + next->rstart;
    if (next->type == DMCOMPOSITE_ARRAY) {
      ierr = MPI_Bcast(&next->grstart,1,MPIU_INT,next->rank,((PetscObject)dm)->comm);CHKERRQ(ierr);
    } else {
      ierr = PetscMalloc(size*sizeof(PetscInt),&next->grstarts);CHKERRQ(ierr);
      ierr = MPI_Allgather(&next->grstart,1,MPIU_INT,next->grstarts,1,MPIU_INT,((PetscObject)dm)->comm);CHKERRQ(ierr);
    }
    next = next->next;
  }
  com->setup = PETSC_TRUE;
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "DMCompositeGetAccess_Array"
PetscErrorCode DMCompositeGetAccess_Array(DM dm,struct DMCompositeLink *mine,Vec vec,PetscScalar **array)
{
  PetscErrorCode ierr;
  PetscScalar    *varray;
  PetscMPIInt    rank;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(((PetscObject)dm)->comm,&rank);CHKERRQ(ierr);
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
PetscErrorCode DMCompositeGetAccess_DM(DM dm,struct DMCompositeLink *mine,Vec vec,Vec *global)
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
PetscErrorCode DMCompositeRestoreAccess_Array(DM dm,struct DMCompositeLink *mine,Vec vec,PetscScalar **array)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMCompositeRestoreAccess_DM"
PetscErrorCode DMCompositeRestoreAccess_DM(DM dm,struct DMCompositeLink *mine,Vec vec,Vec *global)
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
PetscErrorCode DMCompositeScatter_Array(DM dm,struct DMCompositeLink *mine,Vec vec,PetscScalar *array)
{
  PetscErrorCode ierr;
  PetscScalar    *varray;
  PetscMPIInt    rank;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(((PetscObject)dm)->comm,&rank);CHKERRQ(ierr);
  if (rank == mine->rank) {
    ierr    = VecGetArray(vec,&varray);CHKERRQ(ierr);
    ierr    = PetscMemcpy(array,varray+mine->rstart,mine->n*sizeof(PetscScalar));CHKERRQ(ierr);
    ierr    = VecRestoreArray(vec,&varray);CHKERRQ(ierr);
  }
  ierr    = MPI_Bcast(array,mine->nlocal,MPIU_SCALAR,mine->rank,((PetscObject)dm)->comm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMCompositeScatter_DM"
PetscErrorCode DMCompositeScatter_DM(DM dm,struct DMCompositeLink *mine,Vec vec,Vec local)
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
PetscErrorCode DMCompositeGather_Array(DM dm,struct DMCompositeLink *mine,Vec vec,InsertMode imode,const PetscScalar *array)
{
  PetscErrorCode ierr;
  PetscScalar    *varray;
  PetscMPIInt    rank;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(((PetscObject)dm)->comm,&rank);CHKERRQ(ierr);
  if (rank == mine->rank) {
    ierr = VecGetArray(vec,&varray);CHKERRQ(ierr);
    if (varray+mine->rstart == array) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"You need not DMCompositeGather() into objects obtained via DMCompositeGetAccess()");
  }
  switch (imode) {
  case INSERT_VALUES:
    if (rank == mine->rank) {
      ierr = PetscMemcpy(varray+mine->rstart,array,mine->n*sizeof(PetscScalar));CHKERRQ(ierr);
    }
    break;
  case ADD_VALUES: {
    PetscInt          i;
    void             *source;
    PetscScalar       *buffer,*dest;
    if (rank == mine->rank) {
      dest = &varray[mine->rstart];
#if defined(PETSC_HAVE_MPI_IN_PLACE)
      buffer = dest;
      source = MPI_IN_PLACE;
#else
      ierr = PetscMalloc(mine->nlocal*sizeof(PetscScalar),&buffer);CHKERRQ(ierr);
      source = (void *) buffer;
#endif
      for (i=0; i<mine->nlocal; i++) buffer[i] = varray[mine->rstart+i] + array[i];
    } else {
      source = (void *) array;
      dest   = PETSC_NULL;
    }
    ierr = MPI_Reduce(source,dest,mine->nlocal,MPIU_SCALAR,MPI_SUM,mine->rank,((PetscObject)dm)->comm);CHKERRQ(ierr);
#if !defined(PETSC_HAVE_MPI_IN_PLACE)
    if (rank == mine->rank) {ierr = PetscFree(source);CHKERRQ(ierr);}
#endif
  } break;
  default: SETERRQ(((PetscObject)dm)->comm,PETSC_ERR_SUP,"imode");
  }
  if (rank == mine->rank) {ierr = VecRestoreArray(vec,&varray);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMCompositeGather_DM"
PetscErrorCode DMCompositeGather_DM(DM dm,struct DMCompositeLink *mine,Vec vec,InsertMode imode,Vec local)
{
  PetscErrorCode ierr;
  PetscScalar    *array;
  Vec            global;

  PetscFunctionBegin;
  ierr = DMGetGlobalVector(mine->dm,&global);CHKERRQ(ierr);
  ierr = VecGetArray(vec,&array);CHKERRQ(ierr);
  ierr = VecPlaceArray(global,array+mine->rstart);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(mine->dm,local,imode,global);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(mine->dm,local,imode,global);CHKERRQ(ierr);
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

    Not Collective

    Input Parameter:
.    dm - the packer object

    Output Parameter:
.     nDM - the number of DMs

    Level: beginner

@*/
PetscErrorCode  DMCompositeGetNumberDM(DM dm,PetscInt *nDM)
{
  DM_Composite *com = (DM_Composite*)dm->data;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  *nDM = com->nDM;
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "DMCompositeGetAccess"
/*@C
    DMCompositeGetAccess - Allows one to access the individual packed vectors in their global
       representation.

    Collective on DMComposite

    Input Parameter:
+    dm - the packer object
.    gvec - the global vector
-    ... - the individual sequential or parallel objects (arrays or vectors)

    Notes: Use DMCompositeRestoreAccess() to return the vectors when you no longer need them
 
    Level: advanced

@*/
PetscErrorCode  DMCompositeGetAccess(DM dm,Vec gvec,...)
{
  va_list                Argp;
  PetscErrorCode         ierr;
  struct DMCompositeLink *next;
  DM_Composite           *com = (DM_Composite*)dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidHeaderSpecific(gvec,VEC_CLASSID,2);
  next = com->next;
  if (!com->setup) {
    ierr = DMSetUp(dm);CHKERRQ(ierr);
  }

  /* loop over packed objects, handling one at at time */
  va_start(Argp,gvec);
  while (next) {
    if (next->type == DMCOMPOSITE_ARRAY) {
      PetscScalar **array;
      array = va_arg(Argp, PetscScalar**);
      ierr  = DMCompositeGetAccess_Array(dm,next,gvec,array);CHKERRQ(ierr);
    } else if (next->type == DMCOMPOSITE_DM) {
      Vec *vec;
      vec  = va_arg(Argp, Vec*);
      ierr = DMCompositeGetAccess_DM(dm,next,gvec,vec);CHKERRQ(ierr);
    } else {
      SETERRQ(((PetscObject)dm)->comm,PETSC_ERR_SUP,"Cannot handle that object type yet");
    }
    next = next->next;
  }
  va_end(Argp);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMCompositeRestoreAccess"
/*@C
    DMCompositeRestoreAccess - Returns the vectors obtained with DMCompositeGetAccess()
       representation.

    Collective on DMComposite

    Input Parameter:
+    dm - the packer object
.    gvec - the global vector
-    ... - the individual sequential or parallel objects (arrays or vectors)
 
    Level: advanced

.seealso  DMCompositeAddArray(), DMCompositeAddDM(), DMCreateGlobalVector(),
         DMCompositeGather(), DMCompositeCreate(), DMCompositeGetISLocalToGlobalMappings(), DMCompositeScatter(),
         DMCompositeRestoreAccess(), DMCompositeGetAccess()

@*/
PetscErrorCode  DMCompositeRestoreAccess(DM dm,Vec gvec,...)
{
  va_list                Argp;
  PetscErrorCode         ierr;
  struct DMCompositeLink *next;
  DM_Composite           *com = (DM_Composite*)dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidHeaderSpecific(gvec,VEC_CLASSID,2);
  next = com->next;
  if (!com->setup) {
    ierr = DMSetUp(dm);CHKERRQ(ierr);
  }

  /* loop over packed objects, handling one at at time */
  va_start(Argp,gvec);
  while (next) {
    if (next->type == DMCOMPOSITE_ARRAY) {
      PetscScalar **array;
      array = va_arg(Argp, PetscScalar**);
      ierr  = DMCompositeRestoreAccess_Array(dm,next,gvec,array);CHKERRQ(ierr);
    } else if (next->type == DMCOMPOSITE_DM) {
      Vec *vec;
      vec  = va_arg(Argp, Vec*);
      ierr = DMCompositeRestoreAccess_DM(dm,next,gvec,vec);CHKERRQ(ierr);
    } else {
      SETERRQ(((PetscObject)dm)->comm,PETSC_ERR_SUP,"Cannot handle that object type yet");
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
+    dm - the packer object
.    gvec - the global vector
-    ... - the individual sequential objects (arrays or vectors), PETSC_NULL for those that are not needed

    Level: advanced

.seealso DMDestroy(), DMCompositeAddArray(), DMCompositeAddDM(), DMCreateGlobalVector(),
         DMCompositeGather(), DMCompositeCreate(), DMCompositeGetISLocalToGlobalMappings(), DMCompositeGetAccess(),
         DMCompositeGetLocalVectors(), DMCompositeRestoreLocalVectors(), DMCompositeGetEntries()

@*/
PetscErrorCode  DMCompositeScatter(DM dm,Vec gvec,...)
{
  va_list                Argp;
  PetscErrorCode         ierr;
  struct DMCompositeLink *next;
  PetscInt               cnt;
  DM_Composite           *com = (DM_Composite*)dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidHeaderSpecific(gvec,VEC_CLASSID,2);
  if (!com->setup) {
    ierr = DMSetUp(dm);CHKERRQ(ierr);
  }

  /* loop over packed objects, handling one at at time */
  va_start(Argp,gvec);
  for (cnt=3,next=com->next; next; cnt++,next=next->next) {
    if (next->type == DMCOMPOSITE_ARRAY) {
      PetscScalar *array;
      array = va_arg(Argp, PetscScalar*);
      if (array) PetscValidScalarPointer(array,cnt);
      PetscValidLogicalCollectiveBool(dm,!!array,cnt);
      if (!array) continue;
      ierr = DMCompositeScatter_Array(dm,next,gvec,array);CHKERRQ(ierr);
    } else if (next->type == DMCOMPOSITE_DM) {
      Vec vec;
      vec = va_arg(Argp, Vec);
      if (!vec) continue;
      PetscValidHeaderSpecific(vec,VEC_CLASSID,cnt);
      ierr = DMCompositeScatter_DM(dm,next,gvec,vec);CHKERRQ(ierr);
    } else {
      SETERRQ(((PetscObject)dm)->comm,PETSC_ERR_SUP,"Cannot handle that object type yet");
    }
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
+    dm - the packer object
.    gvec - the global vector
-    ... - the individual sequential objects (arrays or vectors), PETSC_NULL for any that are not needed

    Level: advanced

.seealso DMDestroy(), DMCompositeAddArray(), DMCompositeAddDM(), DMCreateGlobalVector(),
         DMCompositeScatter(), DMCompositeCreate(), DMCompositeGetISLocalToGlobalMappings(), DMCompositeGetAccess(),
         DMCompositeGetLocalVectors(), DMCompositeRestoreLocalVectors(), DMCompositeGetEntries()

@*/
PetscErrorCode  DMCompositeGather(DM dm,Vec gvec,InsertMode imode,...)
{
  va_list                Argp;
  PetscErrorCode         ierr;
  struct DMCompositeLink *next;
  DM_Composite           *com = (DM_Composite*)dm->data;
  PetscInt               cnt;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidHeaderSpecific(gvec,VEC_CLASSID,2);
  if (!com->setup) {
    ierr = DMSetUp(dm);CHKERRQ(ierr);
  }

  /* loop over packed objects, handling one at at time */
  va_start(Argp,imode);
  for (cnt=3,next=com->next; next; cnt++,next=next->next) {
    if (next->type == DMCOMPOSITE_ARRAY) {
      PetscScalar *array;
      array = va_arg(Argp, PetscScalar*);
      if (!array) continue;
      PetscValidScalarPointer(array,cnt);
      ierr  = DMCompositeGather_Array(dm,next,gvec,imode,array);CHKERRQ(ierr);
    } else if (next->type == DMCOMPOSITE_DM) {
      Vec vec;
      vec = va_arg(Argp, Vec);
      if (!vec) continue;
      PetscValidHeaderSpecific(vec,VEC_CLASSID,cnt);
      ierr = DMCompositeGather_DM(dm,next,gvec,imode,vec);CHKERRQ(ierr);
    } else {
      SETERRQ(((PetscObject)dm)->comm,PETSC_ERR_SUP,"Cannot handle that object type yet");
    }
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
+    dm - the packer object
.    orank - the process on which the array entries officially live, this number must be
             the same on all processes.
-    n - the length of the array
 
    Level: advanced

.seealso DMDestroy(), DMCompositeGather(), DMCompositeAddDM(), DMCreateGlobalVector(),
         DMCompositeScatter(), DMCompositeCreate(), DMCompositeGetISLocalToGlobalMappings(), DMCompositeGetAccess(),
         DMCompositeGetLocalVectors(), DMCompositeRestoreLocalVectors(), DMCompositeGetEntries()

@*/
PetscErrorCode  DMCompositeAddArray(DM dm,PetscMPIInt orank,PetscInt n)
{
  struct DMCompositeLink *mine,*next;
  PetscErrorCode         ierr;
  PetscMPIInt            rank;
  DM_Composite           *com = (DM_Composite*)dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  next = com->next;
  if (com->setup) {
    SETERRQ(((PetscObject)dm)->comm,PETSC_ERR_ARG_WRONGSTATE,"Cannot add an array once you have used the DMComposite");
  }
#if defined(PETSC_USE_DEBUG)
  {
    PetscMPIInt orankmax;
    ierr = MPI_Allreduce(&orank,&orankmax,1,MPI_INT,MPI_MAX,((PetscObject)dm)->comm);CHKERRQ(ierr);
    if (orank != orankmax) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"orank %d must be equal on all processes, another process has value %d",orank,orankmax);
  }
#endif

  ierr = MPI_Comm_rank(((PetscObject)dm)->comm,&rank);CHKERRQ(ierr);
  /* create new link */
  ierr                = PetscNew(struct DMCompositeLink,&mine);CHKERRQ(ierr);
  mine->nlocal        = n;
  mine->n             = (rank == orank) ? n : 0;
  mine->rank          = orank;
  mine->dm            = PETSC_NULL;
  mine->type          = DMCOMPOSITE_ARRAY;
  mine->next          = PETSC_NULL;
  if (rank == mine->rank) {com->n += n;com->nmine++;}

  /* add to end of list */
  if (!next) {
    com->next = mine;
  } else {
    while (next->next) next = next->next;
    next->next = mine;
  }
  com->nredundant++;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMCompositeAddDM"
/*@C
    DMCompositeAddDM - adds a DM  vector to a DMComposite

    Collective on DMComposite

    Input Parameter:
+    dm - the packer object
-    dm - the DM object, if the DM is a da you will need to caste it with a (DM)
 
    Level: advanced

.seealso DMDestroy(), DMCompositeGather(), DMCompositeAddDM(), DMCreateGlobalVector(),
         DMCompositeScatter(), DMCompositeCreate(), DMCompositeGetISLocalToGlobalMappings(), DMCompositeGetAccess(),
         DMCompositeGetLocalVectors(), DMCompositeRestoreLocalVectors(), DMCompositeGetEntries()

@*/
PetscErrorCode  DMCompositeAddDM(DM dmc,DM dm)
{
  PetscErrorCode         ierr;
  PetscInt               n,nlocal;
  struct DMCompositeLink *mine,*next;
  Vec                    global,local;
  DM_Composite           *com = (DM_Composite*)dmc->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dmc,DM_CLASSID,1);
  PetscValidHeaderSpecific(dm,DM_CLASSID,2);
  next = com->next;
  if (com->setup) SETERRQ(((PetscObject)dmc)->comm,PETSC_ERR_ARG_WRONGSTATE,"Cannot add a DM once you have used the DMComposite");

  /* create new link */
  ierr = PetscNew(struct DMCompositeLink,&mine);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)dm);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(dm,&global);CHKERRQ(ierr);
  ierr = VecGetLocalSize(global,&n);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dm,&global);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm,&local);CHKERRQ(ierr);
  ierr = VecGetSize(local,&nlocal);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&local);CHKERRQ(ierr);
  mine->n      = n;
  mine->nlocal = nlocal;
  mine->dm     = dm;  
  mine->type   = DMCOMPOSITE_DM;
  mine->next   = PETSC_NULL;
  com->n       += n;

  /* add to end of list */
  if (!next) {
    com->next = mine;
  } else {
    while (next->next) next = next->next;
    next->next = mine;
  }
  com->nDM++;
  com->nmine++;
  PetscFunctionReturn(0);
}

extern PetscErrorCode  VecView_MPI(Vec,PetscViewer);
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "VecView_DMComposite"
PetscErrorCode  VecView_DMComposite(Vec gvec,PetscViewer viewer)
{
  DM                     dm;
  PetscErrorCode         ierr;
  struct DMCompositeLink *next;
  PetscBool              isdraw;
  DM_Composite           *com;

  PetscFunctionBegin;
  ierr = PetscObjectQuery((PetscObject)gvec,"DMComposite",(PetscObject*)&dm);CHKERRQ(ierr);
  if (!dm) SETERRQ(((PetscObject)gvec)->comm,PETSC_ERR_ARG_WRONG,"Vector not generated from a DMComposite");
  com = (DM_Composite*)dm->data;
  next = com->next;

  ierr = PetscTypeCompare((PetscObject)viewer,PETSCVIEWERDRAW,&isdraw);CHKERRQ(ierr);
  if (!isdraw) {
    /* do I really want to call this? */
    ierr = VecView_MPI(gvec,viewer);CHKERRQ(ierr);
  } else {
    PetscInt cnt = 0;

    /* loop over packed objects, handling one at at time */
    while (next) {
      if (next->type == DMCOMPOSITE_ARRAY) {
	PetscScalar *array;
	ierr  = DMCompositeGetAccess_Array(dm,next,gvec,&array);CHKERRQ(ierr);

	/*skip it for now */
      } else if (next->type == DMCOMPOSITE_DM) {
	Vec      vec;
        PetscInt bs;

	ierr = DMCompositeGetAccess_DM(dm,next,gvec,&vec);CHKERRQ(ierr);
	ierr = VecView(vec,viewer);CHKERRQ(ierr);
        ierr = VecGetBlockSize(vec,&bs);CHKERRQ(ierr);
	ierr = DMCompositeRestoreAccess_DM(dm,next,gvec,&vec);CHKERRQ(ierr);
        ierr = PetscViewerDrawBaseAdd(viewer,bs);CHKERRQ(ierr);
        cnt += bs;
      } else {
	SETERRQ(((PetscObject)dm)->comm,PETSC_ERR_SUP,"Cannot handle that object type yet");
      }
      next = next->next;
    }
    ierr = PetscViewerDrawBaseAdd(viewer,-cnt);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END


#undef __FUNCT__  
#define __FUNCT__ "DMCreateGlobalVector_Composite"
PetscErrorCode  DMCreateGlobalVector_Composite(DM dm,Vec *gvec)
{
  PetscErrorCode         ierr;
  DM_Composite           *com = (DM_Composite*)dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  if (!com->setup) {
    ierr = DMSetUp(dm);CHKERRQ(ierr);
  }
  ierr = VecCreateMPI(((PetscObject)dm)->comm,com->n,com->N,gvec);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)*gvec,"DMComposite",(PetscObject)dm);CHKERRQ(ierr);
  ierr = VecSetOperation(*gvec,VECOP_VIEW,(void(*)(void))VecView_DMComposite);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMCreateLocalVector_Composite"
PetscErrorCode  DMCreateLocalVector_Composite(DM dm,Vec *lvec)
{
  PetscErrorCode         ierr;
  DM_Composite           *com = (DM_Composite*)dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  if (!com->setup) {
    ierr = DMSetUp(dm);CHKERRQ(ierr);
  }
  ierr = VecCreateSeq(((PetscObject)dm)->comm,com->nghost,lvec);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)*lvec,"DMComposite",(PetscObject)dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMCompositeGetISLocalToGlobalMappings"
/*@C
    DMCompositeGetISLocalToGlobalMappings - gets an ISLocalToGlobalMapping for each DM/array in the DMComposite, maps to the composite global space

    Collective on DM

    Input Parameter:
.    dm - the packer object

    Output Parameters:
.    ltogs - the individual mappings for each packed vector/array. Note that this includes
           all the ghost points that individual ghosted DMDA's may have. Also each process has an
           mapping for EACH redundant array (not just the local redundant arrays).

    Level: advanced

    Notes:
       Each entry of ltogs should be destroyed with ISLocalToGlobalMappingDestroy(), the ltogs array should be freed with PetscFree().

.seealso DMDestroy(), DMCompositeAddArray(), DMCompositeAddDM(), DMCreateGlobalVector(),
         DMCompositeGather(), DMCompositeCreate(), DMCompositeGetAccess(), DMCompositeScatter(),
         DMCompositeGetLocalVectors(), DMCompositeRestoreLocalVectors(),DMCompositeGetEntries()

@*/
PetscErrorCode  DMCompositeGetISLocalToGlobalMappings(DM dm,ISLocalToGlobalMapping **ltogs)
{
  PetscErrorCode         ierr;
  PetscInt               i,*idx,n,cnt;
  struct DMCompositeLink *next;
  PetscMPIInt            rank;
  DM_Composite           *com = (DM_Composite*)dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = PetscMalloc((com->nDM+com->nredundant)*sizeof(ISLocalToGlobalMapping),ltogs);CHKERRQ(ierr);
  next = com->next;
  ierr = MPI_Comm_rank(((PetscObject)dm)->comm,&rank);CHKERRQ(ierr);

  /* loop over packed objects, handling one at at time */
  cnt = 0;
  while (next) {
    if (next->type == DMCOMPOSITE_ARRAY) {
      ierr = PetscMalloc(next->nlocal*sizeof(PetscInt),&idx);CHKERRQ(ierr);
      if (rank == next->rank) {
        for (i=0; i<next->nlocal; i++) idx[i] = next->grstart + i;
      }
      ierr = MPI_Bcast(idx,next->nlocal,MPIU_INT,next->rank,((PetscObject)dm)->comm);CHKERRQ(ierr);
      ierr = ISLocalToGlobalMappingCreate(((PetscObject)dm)->comm,next->nlocal,idx,PETSC_OWN_POINTER,&(*ltogs)[cnt]);CHKERRQ(ierr);
    } else if (next->type == DMCOMPOSITE_DM) {
      ISLocalToGlobalMapping ltog;
      PetscMPIInt            size;
      const PetscInt         *suboff,*indices;
      Vec                    global;

      /* Get sub-DM global indices for each local dof */
      ierr = DMGetLocalToGlobalMapping(next->dm,&ltog);CHKERRQ(ierr);
      ierr = ISLocalToGlobalMappingGetSize(ltog,&n);CHKERRQ(ierr);
      ierr = ISLocalToGlobalMappingGetIndices(ltog,&indices);CHKERRQ(ierr);
      ierr = PetscMalloc(n*sizeof(PetscInt),&idx);CHKERRQ(ierr);

      /* Get the offsets for the sub-DM global vector */
      ierr = DMGetGlobalVector(next->dm,&global);CHKERRQ(ierr);
      ierr = VecGetOwnershipRanges(global,&suboff);CHKERRQ(ierr);
      ierr = MPI_Comm_size(((PetscObject)global)->comm,&size);CHKERRQ(ierr);

      /* Shift the sub-DM definition of the global space to the composite global space */
      for (i=0; i<n; i++) {
        PetscInt subi = indices[i],lo = 0,hi = size,t;
        /* Binary search to find which rank owns subi */
        while (hi-lo > 1) {
          t = lo + (hi-lo)/2;
          if (suboff[t] > subi) hi = t;
          else                  lo = t;
        }
        idx[i] = subi - suboff[lo] + next->grstarts[lo];
      }
      ierr = ISLocalToGlobalMappingRestoreIndices(ltog,&indices);CHKERRQ(ierr);
      ierr = ISLocalToGlobalMappingCreate(((PetscObject)dm)->comm,n,idx,PETSC_OWN_POINTER,&(*ltogs)[cnt]);CHKERRQ(ierr);
      ierr = DMRestoreGlobalVector(next->dm,&global);CHKERRQ(ierr);
    } else SETERRQ(((PetscObject)dm)->comm,PETSC_ERR_SUP,"Cannot handle that object type yet");
    next = next->next;
    cnt++;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMCompositeGetLocalISs"
/*@C
   DMCompositeGetLocalISs - Gets index sets for each DM/array component of a composite local vector

   Not Collective

   Input Arguments:
. dm - composite DM

   Output Arguments:
. is - array of serial index sets for each each component of the DMComposite

   Level: intermediate

   Notes:
   At present, a composite local vector does not normally exist.  This function is used to provide index sets for
   MatGetLocalSubMatrix().  In the future, the scatters for each entry in the DMComposite may be be merged into a single
   scatter to a composite local vector.

   To get the composite global indices at all local points (including ghosts), use DMCompositeGetISLocalToGlobalMappings().

   To get index sets for pieces of the composite global vector, use DMCompositeGetGlobalISs().

   Each returned IS should be destroyed with ISDestroy(), the array should be freed with PetscFree().

.seealso: DMCompositeGetGlobalISs(), DMCompositeGetISLocalToGlobalMappings(), MatGetLocalSubMatrix(), MatCreateLocalRef()
@*/
PetscErrorCode  DMCompositeGetLocalISs(DM dm,IS **is)
{
  PetscErrorCode         ierr;
  DM_Composite           *com = (DM_Composite*)dm->data;
  struct DMCompositeLink *link;
  PetscInt cnt,start;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidPointer(is,2);
  ierr = PetscMalloc(com->nmine*sizeof(IS),is);CHKERRQ(ierr);
  for (cnt=0,start=0,link=com->next; link; start+=link->nlocal,cnt++,link=link->next) {
    ierr = ISCreateStride(PETSC_COMM_SELF,link->nlocal,start,1,&(*is)[cnt]);CHKERRQ(ierr);
    if (link->type == DMCOMPOSITE_DM) {
      PetscInt bs;
      ierr = DMGetBlockSize(link->dm,&bs);CHKERRQ(ierr);
      ierr = ISSetBlockSize((*is)[cnt],bs);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMCompositeGetGlobalISs"
/*@C
    DMCompositeGetGlobalISs - Gets the index sets for each composed object

    Collective on DMComposite

    Input Parameter:
.    dm - the packer object

    Output Parameters:
.    is - the array of index sets

    Level: advanced

    Notes:
       The is entries should be destroyed with ISDestroy(), the is array should be freed with PetscFree()

       These could be used to extract a subset of vector entries for a "multi-physics" preconditioner

       Use DMCompositeGetLocalISs() for index sets in the packed local numbering, and
       DMCompositeGetISLocalToGlobalMappings() for to map local sub-DM (including ghost) indices to packed global
       indices.

.seealso DMDestroy(), DMCompositeAddArray(), DMCompositeAddDM(), DMCreateGlobalVector(),
         DMCompositeGather(), DMCompositeCreate(), DMCompositeGetAccess(), DMCompositeScatter(),
         DMCompositeGetLocalVectors(), DMCompositeRestoreLocalVectors(),DMCompositeGetEntries()

@*/

PetscErrorCode  DMCompositeGetGlobalISs(DM dm,IS *is[])
{
  PetscErrorCode         ierr;
  PetscInt               cnt = 0,*idx,i;
  struct DMCompositeLink *next;
  PetscMPIInt            rank;
  DM_Composite           *com = (DM_Composite*)dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = PetscMalloc((com->nDM+com->nredundant)*sizeof(IS),is);CHKERRQ(ierr);
  next = com->next;
  ierr = MPI_Comm_rank(((PetscObject)dm)->comm,&rank);CHKERRQ(ierr);

  /* loop over packed objects, handling one at at time */
  while (next) {
    ierr = PetscMalloc(next->n*sizeof(PetscInt),&idx);CHKERRQ(ierr);
    for (i=0; i<next->n; i++) idx[i] = next->grstart + i;
    ierr = ISCreateGeneral(((PetscObject)dm)->comm,next->n,idx,PETSC_OWN_POINTER,&(*is)[cnt]);CHKERRQ(ierr);
    cnt++;
    next = next->next;
  }
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "DMCompositeGetLocalVectors_Array"
PetscErrorCode DMCompositeGetLocalVectors_Array(DM dm,struct DMCompositeLink *mine,PetscScalar **array)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (array) {
    ierr = PetscMalloc(mine->nlocal*sizeof(PetscScalar),array);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMCompositeGetLocalVectors_DM"
PetscErrorCode DMCompositeGetLocalVectors_DM(DM dm,struct DMCompositeLink *mine,Vec *local)
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
PetscErrorCode DMCompositeRestoreLocalVectors_Array(DM dm,struct DMCompositeLink *mine,PetscScalar **array)
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
PetscErrorCode DMCompositeRestoreLocalVectors_DM(DM dm,struct DMCompositeLink *mine,Vec *local)
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
    DMCompositeGetLocalVectors - Gets local vectors and arrays for each part of a DMComposite.
       Use DMCompositeRestoreLocalVectors() to return them.

    Not Collective

    Input Parameter:
.    dm - the packer object
 
    Output Parameter:
.    ... - the individual sequential objects (arrays or vectors)
 
    Level: advanced

.seealso DMDestroy(), DMCompositeAddArray(), DMCompositeAddDM(), DMCreateGlobalVector(),
         DMCompositeGather(), DMCompositeCreate(), DMCompositeGetISLocalToGlobalMappings(), DMCompositeGetAccess(),
         DMCompositeRestoreLocalVectors(), DMCompositeScatter(), DMCompositeGetEntries()

@*/
PetscErrorCode  DMCompositeGetLocalVectors(DM dm,...)
{
  va_list                Argp;
  PetscErrorCode         ierr;
  struct DMCompositeLink *next;
  DM_Composite           *com = (DM_Composite*)dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  next = com->next;
  /* loop over packed objects, handling one at at time */
  va_start(Argp,dm);
  while (next) {
    if (next->type == DMCOMPOSITE_ARRAY) {
      PetscScalar **array;
      array = va_arg(Argp, PetscScalar**);
      ierr = DMCompositeGetLocalVectors_Array(dm,next,array);CHKERRQ(ierr);
    } else if (next->type == DMCOMPOSITE_DM) {
      Vec *vec;
      vec = va_arg(Argp, Vec*);
      ierr = DMCompositeGetLocalVectors_DM(dm,next,vec);CHKERRQ(ierr);
    } else {
      SETERRQ(((PetscObject)dm)->comm,PETSC_ERR_SUP,"Cannot handle that object type yet");
    }
    next = next->next;
  }
  va_end(Argp);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMCompositeRestoreLocalVectors"
/*@C
    DMCompositeRestoreLocalVectors - Restores local vectors and arrays for each part of a DMComposite.

    Not Collective

    Input Parameter:
.    dm - the packer object
 
    Output Parameter:
.    ... - the individual sequential objects (arrays or vectors)
 
    Level: advanced

.seealso DMDestroy(), DMCompositeAddArray(), DMCompositeAddDM(), DMCreateGlobalVector(),
         DMCompositeGather(), DMCompositeCreate(), DMCompositeGetISLocalToGlobalMappings(), DMCompositeGetAccess(),
         DMCompositeGetLocalVectors(), DMCompositeScatter(), DMCompositeGetEntries()

@*/
PetscErrorCode  DMCompositeRestoreLocalVectors(DM dm,...)
{
  va_list                Argp;
  PetscErrorCode         ierr;
  struct DMCompositeLink *next;
  DM_Composite           *com = (DM_Composite*)dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  next = com->next;
  /* loop over packed objects, handling one at at time */
  va_start(Argp,dm);
  while (next) {
    if (next->type == DMCOMPOSITE_ARRAY) {
      PetscScalar **array;
      array = va_arg(Argp, PetscScalar**);
      ierr = DMCompositeRestoreLocalVectors_Array(dm,next,array);CHKERRQ(ierr);
    } else if (next->type == DMCOMPOSITE_DM) {
      Vec *vec;
      vec = va_arg(Argp, Vec*);
      ierr = DMCompositeRestoreLocalVectors_DM(dm,next,vec);CHKERRQ(ierr);
    } else {
      SETERRQ(((PetscObject)dm)->comm,PETSC_ERR_SUP,"Cannot handle that object type yet");
    }
    next = next->next;
  }
  va_end(Argp);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "DMCompositeGetEntries_Array"
PetscErrorCode DMCompositeGetEntries_Array(DM dm,struct DMCompositeLink *mine,PetscInt *n)
{
  PetscFunctionBegin;
  if (n) *n = mine->nlocal;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMCompositeGetEntries_DM"
PetscErrorCode DMCompositeGetEntries_DM(DM dmi,struct DMCompositeLink *mine,DM *dm)
{
  PetscFunctionBegin;
  if (dm) *dm = mine->dm;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMCompositeGetEntries"
/*@C
    DMCompositeGetEntries - Gets the DM, redundant size, etc for each entry in a DMComposite.

    Not Collective

    Input Parameter:
.    dm - the packer object
 
    Output Parameter:
.    ... - the individual entries, DMs or integer sizes)
 
    Level: advanced

.seealso DMDestroy(), DMCompositeAddArray(), DMCompositeAddDM(), DMCreateGlobalVector(),
         DMCompositeGather(), DMCompositeCreate(), DMCompositeGetISLocalToGlobalMappings(), DMCompositeGetAccess(),
         DMCompositeRestoreLocalVectors(), DMCompositeGetLocalVectors(),  DMCompositeScatter(),
         DMCompositeGetLocalVectors(), DMCompositeRestoreLocalVectors()

@*/
PetscErrorCode  DMCompositeGetEntries(DM dm,...)
{
  va_list                Argp;
  PetscErrorCode         ierr;
  struct DMCompositeLink *next;
  DM_Composite           *com = (DM_Composite*)dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  next = com->next;
  /* loop over packed objects, handling one at at time */
  va_start(Argp,dm);
  while (next) {
    if (next->type == DMCOMPOSITE_ARRAY) {
      PetscInt *n;
      n = va_arg(Argp, PetscInt*);
      ierr = DMCompositeGetEntries_Array(dm,next,n);CHKERRQ(ierr);
    } else if (next->type == DMCOMPOSITE_DM) {
      DM *dmn;
      dmn = va_arg(Argp, DM*);
      ierr = DMCompositeGetEntries_DM(dm,next,dmn);CHKERRQ(ierr);
    } else {
      SETERRQ(((PetscObject)dm)->comm,PETSC_ERR_SUP,"Cannot handle that object type yet");
    }
    next = next->next;
  }
  va_end(Argp);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMRefine_Composite"
PetscErrorCode  DMRefine_Composite(DM dmi,MPI_Comm comm,DM *fine)
{
  PetscErrorCode         ierr;
  struct DMCompositeLink *next;
  DM_Composite           *com = (DM_Composite*)dmi->data;
  DM                     dm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dmi,DM_CLASSID,1);
  next = com->next;
  ierr = DMCompositeCreate(comm,fine);CHKERRQ(ierr);

  /* loop over packed objects, handling one at at time */
  while (next) {
    if (next->type == DMCOMPOSITE_ARRAY) {
      ierr = DMCompositeAddArray(*fine,next->rank,next->nlocal);CHKERRQ(ierr);
    } else if (next->type == DMCOMPOSITE_DM) {
      ierr = DMRefine(next->dm,comm,&dm);CHKERRQ(ierr);
      ierr = DMCompositeAddDM(*fine,dm);CHKERRQ(ierr);
      ierr = PetscObjectDereference((PetscObject)dm);CHKERRQ(ierr);
    } else {
      SETERRQ(((PetscObject)dm)->comm,PETSC_ERR_SUP,"Cannot handle that object type yet");
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
  DM                 right,left;
  struct MatPackLink *next;
};

#undef __FUNCT__  
#define __FUNCT__ "MatMultBoth_Shell_Pack"
PetscErrorCode MatMultBoth_Shell_Pack(Mat A,Vec x,Vec y,PetscBool  add)
{
  struct MatPack         *mpack;
  struct DMCompositeLink *xnext,*ynext;
  struct MatPackLink     *anext;
  PetscScalar            *xarray,*yarray;
  PetscErrorCode         ierr;
  PetscInt               i;
  Vec                    xglobal,yglobal;
  PetscMPIInt            rank;
  DM_Composite           *comright;
  DM_Composite           *comleft;

  PetscFunctionBegin;
  ierr = MatShellGetContext(A,(void**)&mpack);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(((PetscObject)mpack->right)->comm,&rank);CHKERRQ(ierr);
  comright = (DM_Composite*)mpack->right->data;
  comleft = (DM_Composite*)mpack->left->data;
  xnext = comright->next;
  ynext = comleft->next;
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
      SETERRQ(((PetscObject)A)->comm,PETSC_ERR_SUP,"Cannot handle that object type yet");
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
  if (z != y) SETERRQ(((PetscObject)A)->comm,PETSC_ERR_SUP,"Handles y == z only");
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
  DM_Composite           *comright;
  DM_Composite           *comleft;

  PetscFunctionBegin;
  ierr  = MatShellGetContext(A,(void**)&mpack);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(((PetscObject)mpack->right)->comm,&rank);CHKERRQ(ierr);
  comright = (DM_Composite*)mpack->right->data;
  comleft = (DM_Composite*)mpack->left->data;
  ynext = comright->next;
  xnext = comleft->next;
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
      SETERRQ(((PetscObject)A)->comm,PETSC_ERR_SUP,"Cannot handle that object type yet");
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
#define __FUNCT__ "DMGetInterpolation_Composite"
PetscErrorCode  DMGetInterpolation_Composite(DM coarse,DM fine,Mat *A,Vec *v)
{
  PetscErrorCode         ierr;
  PetscInt               m,n,M,N;
  struct DMCompositeLink *nextc;
  struct DMCompositeLink *nextf;
  struct MatPackLink     *nextmat,*pnextmat = 0;
  struct MatPack         *mpack;
  Vec                    gcoarse,gfine;
  DM_Composite           *comcoarse = (DM_Composite*)coarse->data;
  DM_Composite           *comfine = (DM_Composite*)fine->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(coarse,DM_CLASSID,1);
  PetscValidHeaderSpecific(fine,DM_CLASSID,2);
  nextc = comcoarse->next;
  nextf = comfine->next;
  /* use global vectors only for determining matrix layout */
  ierr = DMCreateGlobalVector(coarse,&gcoarse);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(fine,&gfine);CHKERRQ(ierr);
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
    if (nextc->type != nextf->type) SETERRQ(((PetscObject)fine)->comm,PETSC_ERR_ARG_INCOMP,"Two DM have different layout");

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
      SETERRQ(((PetscObject)fine)->comm,PETSC_ERR_SUP,"Cannot handle that object type yet");
    }
    nextc = nextc->next;
    nextf = nextf->next;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMCreateLocalToGlobalMapping_Composite"
static PetscErrorCode DMCreateLocalToGlobalMapping_Composite(DM dm)
{
  DM_Composite           *com = (DM_Composite*)dm->data;
  ISLocalToGlobalMapping *ltogs;
  PetscInt               i,cnt,m,*idx;
  PetscErrorCode         ierr;

  PetscFunctionBegin;
  /* Set the ISLocalToGlobalMapping on the new matrix */
  ierr = DMCompositeGetISLocalToGlobalMappings(dm,&ltogs);CHKERRQ(ierr);
  for (cnt=0,i=0; i<(com->nDM+com->nredundant); i++) {
    ierr = ISLocalToGlobalMappingGetSize(ltogs[i],&m);CHKERRQ(ierr);
    cnt += m;
  }
  ierr = PetscMalloc(cnt*sizeof(PetscInt),&idx);CHKERRQ(ierr);
  for (cnt=0,i=0; i<(com->nDM+com->nredundant); i++) {
    const PetscInt *subidx;
    ierr = ISLocalToGlobalMappingGetSize(ltogs[i],&m);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingGetIndices(ltogs[i],&subidx);CHKERRQ(ierr);
    ierr = PetscMemcpy(&idx[cnt],subidx,m*sizeof(PetscInt));CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingRestoreIndices(ltogs[i],&subidx);CHKERRQ(ierr);
    cnt += m;
  }
  ierr = ISLocalToGlobalMappingCreate(((PetscObject)dm)->comm,cnt,idx,PETSC_OWN_POINTER,&dm->ltogmap);CHKERRQ(ierr);
  for (i=0; i<com->nDM+com->nredundant; i++) {ierr = ISLocalToGlobalMappingDestroy(ltogs[i]);CHKERRQ(ierr);}
  ierr = PetscFree(ltogs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "DMGetColoring_Composite" 
PetscErrorCode  DMGetColoring_Composite(DM dm,ISColoringType ctype,const MatType mtype,ISColoring *coloring)
{
  PetscErrorCode         ierr;
  PetscInt               n,i,cnt;
  ISColoringValue        *colors;
  PetscBool              dense = PETSC_FALSE;
  ISColoringValue        maxcol = 0;
  DM_Composite           *com = (DM_Composite*)dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  if (ctype == IS_COLORING_GHOSTED) {
    SETERRQ(((PetscObject)dm)->comm,PETSC_ERR_SUP,"Currently you must use -dmmg_iscoloring_type global" );
  } else if (ctype == IS_COLORING_GLOBAL) {
    n = com->n;
  } else SETERRQ(((PetscObject)dm)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Unknown ISColoringType");
  ierr = PetscMalloc(n*sizeof(ISColoringValue),&colors);CHKERRQ(ierr); /* freed in ISColoringDestroy() */

  ierr = PetscOptionsGetBool(PETSC_NULL,"-dmcomposite_dense_jacobian",&dense,PETSC_NULL);CHKERRQ(ierr);
  if (dense) {
    for (i=0; i<n; i++) {
      colors[i] = (ISColoringValue)(com->rstart + i);
    }
    maxcol = com->N;
  } else {
    struct DMCompositeLink *next = com->next;
    PetscMPIInt            rank;
  
    ierr = MPI_Comm_rank(((PetscObject)dm)->comm,&rank);CHKERRQ(ierr);
    cnt  = 0;
    while (next) {
      if (next->type == DMCOMPOSITE_ARRAY) {
        if (rank == next->rank) {  /* each column gets is own color */
          for (i=com->rstart+next->rstart; i<com->rstart+next->rstart+next->n; i++) {
            colors[cnt++] = maxcol++;
          }
        }
        ierr = MPI_Bcast(&maxcol,1,MPIU_COLORING_VALUE,next->rank,((PetscObject)dm)->comm);CHKERRQ(ierr);
      } else if (next->type == DMCOMPOSITE_DM) {
        ISColoring     lcoloring;

        ierr = DMGetColoring(next->dm,IS_COLORING_GLOBAL,mtype,&lcoloring);CHKERRQ(ierr);
        for (i=0; i<lcoloring->N; i++) {
          colors[cnt++] = maxcol + lcoloring->colors[i];
        }
        maxcol += lcoloring->n;
        ierr = ISColoringDestroy(lcoloring);CHKERRQ(ierr);
      } else {
        SETERRQ(((PetscObject)dm)->comm,PETSC_ERR_SUP,"Cannot handle that object type yet");
      }
      next = next->next;
    }
  }
  ierr = ISColoringCreate(((PetscObject)dm)->comm,maxcol,n,colors,coloring);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMGlobalToLocalBegin_Composite"
PetscErrorCode  DMGlobalToLocalBegin_Composite(DM dm,Vec gvec,InsertMode mode,Vec lvec)
{
  PetscErrorCode         ierr;
  struct DMCompositeLink *next;
  PetscInt               cnt = 3;
  PetscMPIInt            rank;
  PetscScalar            *garray,*larray;
  DM_Composite           *com = (DM_Composite*)dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidHeaderSpecific(gvec,VEC_CLASSID,2);
  next = com->next;
  if (!com->setup) {
    ierr = DMSetUp(dm);CHKERRQ(ierr);
  }
  ierr = MPI_Comm_rank(((PetscObject)dm)->comm,&rank);CHKERRQ(ierr);
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
      ierr = MPI_Bcast(larray,next->nlocal,MPIU_SCALAR,next->rank,((PetscObject)dm)->comm);CHKERRQ(ierr);
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
      ierr = DMRestoreGlobalVector(next->dm,&global);CHKERRQ(ierr);
      ierr = DMRestoreGlobalVector(next->dm,&local);CHKERRQ(ierr);
    } else {
      SETERRQ(((PetscObject)dm)->comm,PETSC_ERR_SUP,"Cannot handle that object type yet");
    }
    cnt++;
    larray += next->nlocal;
    next    = next->next;
  }

  ierr = VecRestoreArray(gvec,PETSC_NULL);CHKERRQ(ierr);
  ierr = VecRestoreArray(lvec,PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMGlobalToLocalEnd_Composite"
PetscErrorCode  DMGlobalToLocalEnd_Composite(DM dm,Vec gvec,InsertMode mode,Vec lvec)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "DMCreate_Composite"
PetscErrorCode  DMCreate_Composite(DM p)
{
  PetscErrorCode ierr;
  DM_Composite   *com;

  PetscFunctionBegin;
  ierr = PetscNewLog(p,DM_Composite,&com);CHKERRQ(ierr);
  p->data = com;
  ierr = PetscObjectChangeTypeName((PetscObject)p,"DMComposite");CHKERRQ(ierr);
  com->n            = 0;
  com->next         = PETSC_NULL;
  com->nredundant   = 0;
  com->nDM          = 0;

  p->ops->createglobalvector              = DMCreateGlobalVector_Composite;
  p->ops->createlocalvector               = DMCreateLocalVector_Composite;
  p->ops->createlocaltoglobalmapping      = DMCreateLocalToGlobalMapping_Composite;
  p->ops->createlocaltoglobalmappingblock = 0;
  p->ops->refine                          = DMRefine_Composite;
  p->ops->getinterpolation                = DMGetInterpolation_Composite;
  p->ops->getmatrix                       = DMGetMatrix_Composite;
  p->ops->getcoloring                     = DMGetColoring_Composite;
  p->ops->globaltolocalbegin              = DMGlobalToLocalBegin_Composite;
  p->ops->globaltolocalend                = DMGlobalToLocalEnd_Composite;
  p->ops->destroy                         = DMDestroy_Composite;
  p->ops->view                            = DMView_Composite;
  p->ops->setup                           = DMSetUp_Composite;
  PetscFunctionReturn(0);
}
EXTERN_C_END

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

.seealso DMDestroy(), DMCompositeAddArray(), DMCompositeAddDM(), DMCompositeScatter(),
         DMCompositeGather(), DMCreateGlobalVector(), DMCompositeGetISLocalToGlobalMappings(), DMCompositeGetAccess()
         DMCompositeGetLocalVectors(), DMCompositeRestoreLocalVectors(), DMCompositeGetEntries()

@*/
PetscErrorCode  DMCompositeCreate(MPI_Comm comm,DM *packer)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(packer,2);
  ierr = DMCreate(comm,packer);CHKERRQ(ierr);
  ierr = DMSetType(*packer,DMCOMPOSITE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
