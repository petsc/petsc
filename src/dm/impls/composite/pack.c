
#include <../src/dm/impls/composite/packimpl.h>       /*I  "petscdmcomposite.h"  I*/
#include <petsc/private/isimpl.h>
#include <petsc/private/glvisviewerimpl.h>
#include <petscds.h>

/*@C
    DMCompositeSetCoupling - Sets user provided routines that compute the coupling between the
      separate components (DMs) in a DMto build the correct matrix nonzero structure.

    Logically Collective

    Input Parameters:
+   dm - the composite object
-   formcouplelocations - routine to set the nonzero locations in the matrix

    Level: advanced

    Not available from Fortran

    Notes:
    See DMSetApplicationContext() and DMGetApplicationContext() for how to get user information into
        this routine

@*/
PetscErrorCode  DMCompositeSetCoupling(DM dm,PetscErrorCode (*FormCoupleLocations)(DM,Mat,PetscInt*,PetscInt*,PetscInt,PetscInt,PetscInt,PetscInt))
{
  DM_Composite   *com = (DM_Composite*)dm->data;
  PetscBool      flg;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectTypeCompare((PetscObject)dm,DMCOMPOSITE,&flg));
  PetscCheck(flg,PetscObjectComm((PetscObject)dm),PETSC_ERR_USER,"Not for type %s",((PetscObject)dm)->type_name);
  com->FormCoupleLocations = FormCoupleLocations;
  PetscFunctionReturn(0);
}

PetscErrorCode  DMDestroy_Composite(DM dm)
{
  struct DMCompositeLink *next, *prev;
  DM_Composite           *com = (DM_Composite*)dm->data;

  PetscFunctionBegin;
  next = com->next;
  while (next) {
    prev = next;
    next = next->next;
    CHKERRQ(DMDestroy(&prev->dm));
    CHKERRQ(PetscFree(prev->grstarts));
    CHKERRQ(PetscFree(prev));
  }
  CHKERRQ(PetscObjectComposeFunction((PetscObject)dm,"DMSetUpGLVisViewer_C",NULL));
  /* This was originally freed in DMDestroy(), but that prevents reference counting of backend objects */
  CHKERRQ(PetscFree(com));
  PetscFunctionReturn(0);
}

PetscErrorCode  DMView_Composite(DM dm,PetscViewer v)
{
  PetscBool      iascii;
  DM_Composite   *com = (DM_Composite*)dm->data;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectTypeCompare((PetscObject)v,PETSCVIEWERASCII,&iascii));
  if (iascii) {
    struct DMCompositeLink *lnk = com->next;
    PetscInt               i;

    CHKERRQ(PetscViewerASCIIPrintf(v,"DM (%s)\n",((PetscObject)dm)->prefix ? ((PetscObject)dm)->prefix : "no prefix"));
    CHKERRQ(PetscViewerASCIIPrintf(v,"  contains %D DMs\n",com->nDM));
    CHKERRQ(PetscViewerASCIIPushTab(v));
    for (i=0; lnk; lnk=lnk->next,i++) {
      CHKERRQ(PetscViewerASCIIPrintf(v,"Link %D: DM of type %s\n",i,((PetscObject)lnk->dm)->type_name));
      CHKERRQ(PetscViewerASCIIPushTab(v));
      CHKERRQ(DMView(lnk->dm,v));
      CHKERRQ(PetscViewerASCIIPopTab(v));
    }
    CHKERRQ(PetscViewerASCIIPopTab(v));
  }
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------------------------*/
PetscErrorCode  DMSetUp_Composite(DM dm)
{
  PetscInt               nprev = 0;
  PetscMPIInt            rank,size;
  DM_Composite           *com  = (DM_Composite*)dm->data;
  struct DMCompositeLink *next = com->next;
  PetscLayout            map;

  PetscFunctionBegin;
  PetscCheck(!com->setup,PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Packer has already been setup");
  CHKERRQ(PetscLayoutCreate(PetscObjectComm((PetscObject)dm),&map));
  CHKERRQ(PetscLayoutSetLocalSize(map,com->n));
  CHKERRQ(PetscLayoutSetSize(map,PETSC_DETERMINE));
  CHKERRQ(PetscLayoutSetBlockSize(map,1));
  CHKERRQ(PetscLayoutSetUp(map));
  CHKERRQ(PetscLayoutGetSize(map,&com->N));
  CHKERRQ(PetscLayoutGetRange(map,&com->rstart,NULL));
  CHKERRQ(PetscLayoutDestroy(&map));

  /* now set the rstart for each linked vector */
  CHKERRMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)dm),&rank));
  CHKERRMPI(MPI_Comm_size(PetscObjectComm((PetscObject)dm),&size));
  while (next) {
    next->rstart  = nprev;
    nprev        += next->n;
    next->grstart = com->rstart + next->rstart;
    CHKERRQ(PetscMalloc1(size,&next->grstarts));
    CHKERRMPI(MPI_Allgather(&next->grstart,1,MPIU_INT,next->grstarts,1,MPIU_INT,PetscObjectComm((PetscObject)dm)));
    next          = next->next;
  }
  com->setup = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/* ----------------------------------------------------------------------------------*/

/*@
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
  DM_Composite   *com = (DM_Composite*)dm->data;
  PetscBool      flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  CHKERRQ(PetscObjectTypeCompare((PetscObject)dm,DMCOMPOSITE,&flg));
  PetscCheck(flg,PetscObjectComm((PetscObject)dm),PETSC_ERR_USER,"Not for type %s",((PetscObject)dm)->type_name);
  *nDM = com->nDM;
  PetscFunctionReturn(0);
}

/*@C
    DMCompositeGetAccess - Allows one to access the individual packed vectors in their global
       representation.

    Collective on dm

    Input Parameters:
+    dm - the packer object
-    gvec - the global vector

    Output Parameters:
.    Vec* ... - the packed parallel vectors, NULL for those that are not needed

    Notes:
    Use DMCompositeRestoreAccess() to return the vectors when you no longer need them

    Fortran Notes:

    Fortran callers must use numbered versions of this routine, e.g., DMCompositeGetAccess4(dm,gvec,vec1,vec2,vec3,vec4)
    or use the alternative interface DMCompositeGetAccessArray().

    Level: advanced

.seealso: DMCompositeGetEntries(), DMCompositeScatter()
@*/
PetscErrorCode  DMCompositeGetAccess(DM dm,Vec gvec,...)
{
  va_list                Argp;
  struct DMCompositeLink *next;
  DM_Composite           *com = (DM_Composite*)dm->data;
  PetscInt               readonly;
  PetscBool              flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidHeaderSpecific(gvec,VEC_CLASSID,2);
  CHKERRQ(PetscObjectTypeCompare((PetscObject)dm,DMCOMPOSITE,&flg));
  PetscCheck(flg,PetscObjectComm((PetscObject)dm),PETSC_ERR_USER,"Not for type %s",((PetscObject)dm)->type_name);
  next = com->next;
  if (!com->setup) {
    CHKERRQ(DMSetUp(dm));
  }

  CHKERRQ(VecLockGet(gvec,&readonly));
  /* loop over packed objects, handling one at at time */
  va_start(Argp,gvec);
  while (next) {
    Vec *vec;
    vec = va_arg(Argp, Vec*);
    if (vec) {
      CHKERRQ(DMGetGlobalVector(next->dm,vec));
      if (readonly) {
        const PetscScalar *array;
        CHKERRQ(VecGetArrayRead(gvec,&array));
        CHKERRQ(VecPlaceArray(*vec,array+next->rstart));
        CHKERRQ(VecLockReadPush(*vec));
        CHKERRQ(VecRestoreArrayRead(gvec,&array));
      } else {
        PetscScalar *array;
        CHKERRQ(VecGetArray(gvec,&array));
        CHKERRQ(VecPlaceArray(*vec,array+next->rstart));
        CHKERRQ(VecRestoreArray(gvec,&array));
      }
    }
    next = next->next;
  }
  va_end(Argp);
  PetscFunctionReturn(0);
}

/*@C
    DMCompositeGetAccessArray - Allows one to access the individual packed vectors in their global
       representation.

    Collective on dm

    Input Parameters:
+    dm - the packer object
.    pvec - packed vector
.    nwanted - number of vectors wanted
-    wanted - sorted array of vectors wanted, or NULL to get all vectors

    Output Parameters:
.    vecs - array of requested global vectors (must be allocated)

    Notes:
    Use DMCompositeRestoreAccessArray() to return the vectors when you no longer need them

    Level: advanced

.seealso: DMCompositeGetAccess(), DMCompositeGetEntries(), DMCompositeScatter(), DMCompositeGather()
@*/
PetscErrorCode  DMCompositeGetAccessArray(DM dm,Vec pvec,PetscInt nwanted,const PetscInt *wanted,Vec *vecs)
{
  struct DMCompositeLink *link;
  PetscInt               i,wnum;
  DM_Composite           *com = (DM_Composite*)dm->data;
  PetscInt               readonly;
  PetscBool              flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidHeaderSpecific(pvec,VEC_CLASSID,2);
  CHKERRQ(PetscObjectTypeCompare((PetscObject)dm,DMCOMPOSITE,&flg));
  PetscCheck(flg,PetscObjectComm((PetscObject)dm),PETSC_ERR_USER,"Not for type %s",((PetscObject)dm)->type_name);
  if (!com->setup) {
    CHKERRQ(DMSetUp(dm));
  }

  CHKERRQ(VecLockGet(pvec,&readonly));
  for (i=0,wnum=0,link=com->next; link && wnum<nwanted; i++,link=link->next) {
    if (!wanted || i == wanted[wnum]) {
      Vec v;
      CHKERRQ(DMGetGlobalVector(link->dm,&v));
      if (readonly) {
        const PetscScalar *array;
        CHKERRQ(VecGetArrayRead(pvec,&array));
        CHKERRQ(VecPlaceArray(v,array+link->rstart));
        CHKERRQ(VecLockReadPush(v));
        CHKERRQ(VecRestoreArrayRead(pvec,&array));
      } else {
        PetscScalar *array;
        CHKERRQ(VecGetArray(pvec,&array));
        CHKERRQ(VecPlaceArray(v,array+link->rstart));
        CHKERRQ(VecRestoreArray(pvec,&array));
      }
      vecs[wnum++] = v;
    }
  }
  PetscFunctionReturn(0);
}

/*@C
    DMCompositeGetLocalAccessArray - Allows one to access the individual
    packed vectors in their local representation.

    Collective on dm.

    Input Parameters:
+    dm - the packer object
.    pvec - packed vector
.    nwanted - number of vectors wanted
-    wanted - sorted array of vectors wanted, or NULL to get all vectors

    Output Parameters:
.    vecs - array of requested local vectors (must be allocated)

    Notes:
    Use DMCompositeRestoreLocalAccessArray() to return the vectors
    when you no longer need them.

    Level: advanced

.seealso: DMCompositeRestoreLocalAccessArray(), DMCompositeGetAccess(),
DMCompositeGetEntries(), DMCompositeScatter(), DMCompositeGather()
@*/
PetscErrorCode  DMCompositeGetLocalAccessArray(DM dm,Vec pvec,PetscInt nwanted,const PetscInt *wanted,Vec *vecs)
{
  struct DMCompositeLink *link;
  PetscInt               i,wnum;
  DM_Composite           *com = (DM_Composite*)dm->data;
  PetscInt               readonly;
  PetscInt               nlocal = 0;
  PetscBool              flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidHeaderSpecific(pvec,VEC_CLASSID,2);
  CHKERRQ(PetscObjectTypeCompare((PetscObject)dm,DMCOMPOSITE,&flg));
  PetscCheck(flg,PetscObjectComm((PetscObject)dm),PETSC_ERR_USER,"Not for type %s",((PetscObject)dm)->type_name);
  if (!com->setup) {
    CHKERRQ(DMSetUp(dm));
  }

  CHKERRQ(VecLockGet(pvec,&readonly));
  for (i=0,wnum=0,link=com->next; link && wnum<nwanted; i++,link=link->next) {
    if (!wanted || i == wanted[wnum]) {
      Vec v;
      CHKERRQ(DMGetLocalVector(link->dm,&v));
      if (readonly) {
        const PetscScalar *array;
        CHKERRQ(VecGetArrayRead(pvec,&array));
        CHKERRQ(VecPlaceArray(v,array+nlocal));
        // this method does not make sense. The local vectors are not updated with a global-to-local and the user can not do it because it is locked
        CHKERRQ(VecLockReadPush(v));
        CHKERRQ(VecRestoreArrayRead(pvec,&array));
      } else {
        PetscScalar *array;
        CHKERRQ(VecGetArray(pvec,&array));
        CHKERRQ(VecPlaceArray(v,array+nlocal));
        CHKERRQ(VecRestoreArray(pvec,&array));
      }
      vecs[wnum++] = v;
    }

    nlocal += link->nlocal;
  }

  PetscFunctionReturn(0);
}

/*@C
    DMCompositeRestoreAccess - Returns the vectors obtained with DMCompositeGetAccess()
       representation.

    Collective on dm

    Input Parameters:
+    dm - the packer object
.    gvec - the global vector
-    Vec* ... - the individual parallel vectors, NULL for those that are not needed

    Level: advanced

.seealso  DMCompositeAddDM(), DMCreateGlobalVector(),
         DMCompositeGather(), DMCompositeCreate(), DMCompositeGetISLocalToGlobalMappings(), DMCompositeScatter(),
         DMCompositeRestoreAccess(), DMCompositeGetAccess()

@*/
PetscErrorCode  DMCompositeRestoreAccess(DM dm,Vec gvec,...)
{
  va_list                Argp;
  struct DMCompositeLink *next;
  DM_Composite           *com = (DM_Composite*)dm->data;
  PetscInt               readonly;
  PetscBool              flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidHeaderSpecific(gvec,VEC_CLASSID,2);
  CHKERRQ(PetscObjectTypeCompare((PetscObject)dm,DMCOMPOSITE,&flg));
  PetscCheck(flg,PetscObjectComm((PetscObject)dm),PETSC_ERR_USER,"Not for type %s",((PetscObject)dm)->type_name);
  next = com->next;
  if (!com->setup) {
    CHKERRQ(DMSetUp(dm));
  }

  CHKERRQ(VecLockGet(gvec,&readonly));
  /* loop over packed objects, handling one at at time */
  va_start(Argp,gvec);
  while (next) {
    Vec *vec;
    vec = va_arg(Argp, Vec*);
    if (vec) {
      CHKERRQ(VecResetArray(*vec));
      if (readonly) {
        CHKERRQ(VecLockReadPop(*vec));
      }
      CHKERRQ(DMRestoreGlobalVector(next->dm,vec));
    }
    next = next->next;
  }
  va_end(Argp);
  PetscFunctionReturn(0);
}

/*@C
    DMCompositeRestoreAccessArray - Returns the vectors obtained with DMCompositeGetAccessArray()

    Collective on dm

    Input Parameters:
+    dm - the packer object
.    pvec - packed vector
.    nwanted - number of vectors wanted
.    wanted - sorted array of vectors wanted, or NULL to get all vectors
-    vecs - array of global vectors to return

    Level: advanced

.seealso: DMCompositeRestoreAccess(), DMCompositeRestoreEntries(), DMCompositeScatter(), DMCompositeGather()
@*/
PetscErrorCode  DMCompositeRestoreAccessArray(DM dm,Vec pvec,PetscInt nwanted,const PetscInt *wanted,Vec *vecs)
{
  struct DMCompositeLink *link;
  PetscInt               i,wnum;
  DM_Composite           *com = (DM_Composite*)dm->data;
  PetscInt               readonly;
  PetscBool              flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidHeaderSpecific(pvec,VEC_CLASSID,2);
  CHKERRQ(PetscObjectTypeCompare((PetscObject)dm,DMCOMPOSITE,&flg));
  PetscCheck(flg,PetscObjectComm((PetscObject)dm),PETSC_ERR_USER,"Not for type %s",((PetscObject)dm)->type_name);
  if (!com->setup) {
    CHKERRQ(DMSetUp(dm));
  }

  CHKERRQ(VecLockGet(pvec,&readonly));
  for (i=0,wnum=0,link=com->next; link && wnum<nwanted; i++,link=link->next) {
    if (!wanted || i == wanted[wnum]) {
      CHKERRQ(VecResetArray(vecs[wnum]));
      if (readonly) {
        CHKERRQ(VecLockReadPop(vecs[wnum]));
      }
      CHKERRQ(DMRestoreGlobalVector(link->dm,&vecs[wnum]));
      wnum++;
    }
  }
  PetscFunctionReturn(0);
}

/*@C
    DMCompositeRestoreLocalAccessArray - Returns the vectors obtained with DMCompositeGetLocalAccessArray().

    Collective on dm.

    Input Parameters:
+    dm - the packer object
.    pvec - packed vector
.    nwanted - number of vectors wanted
.    wanted - sorted array of vectors wanted, or NULL to restore all vectors
-    vecs - array of local vectors to return

    Level: advanced

    Notes:
    nwanted and wanted must match the values given to DMCompositeGetLocalAccessArray()
    otherwise the call will fail.

.seealso: DMCompositeGetLocalAccessArray(), DMCompositeRestoreAccessArray(),
DMCompositeRestoreAccess(), DMCompositeRestoreEntries(),
DMCompositeScatter(), DMCompositeGather()
@*/
PetscErrorCode  DMCompositeRestoreLocalAccessArray(DM dm,Vec pvec,PetscInt nwanted,const PetscInt *wanted,Vec *vecs)
{
  struct DMCompositeLink *link;
  PetscInt               i,wnum;
  DM_Composite           *com = (DM_Composite*)dm->data;
  PetscInt               readonly;
  PetscBool              flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidHeaderSpecific(pvec,VEC_CLASSID,2);
  CHKERRQ(PetscObjectTypeCompare((PetscObject)dm,DMCOMPOSITE,&flg));
  PetscCheck(flg,PetscObjectComm((PetscObject)dm),PETSC_ERR_USER,"Not for type %s",((PetscObject)dm)->type_name);
  if (!com->setup) {
    CHKERRQ(DMSetUp(dm));
  }

  CHKERRQ(VecLockGet(pvec,&readonly));
  for (i=0,wnum=0,link=com->next; link && wnum<nwanted; i++,link=link->next) {
    if (!wanted || i == wanted[wnum]) {
      CHKERRQ(VecResetArray(vecs[wnum]));
      if (readonly) {
        CHKERRQ(VecLockReadPop(vecs[wnum]));
      }
      CHKERRQ(DMRestoreLocalVector(link->dm,&vecs[wnum]));
      wnum++;
    }
  }
  PetscFunctionReturn(0);
}

/*@C
    DMCompositeScatter - Scatters from a global packed vector into its individual local vectors

    Collective on dm

    Input Parameters:
+    dm - the packer object
.    gvec - the global vector
-    Vec ... - the individual sequential vectors, NULL for those that are not needed

    Level: advanced

    Notes:
    DMCompositeScatterArray() is a non-variadic alternative that is often more convenient for library callers and is
    accessible from Fortran.

.seealso DMDestroy(), DMCompositeAddDM(), DMCreateGlobalVector(),
         DMCompositeGather(), DMCompositeCreate(), DMCompositeGetISLocalToGlobalMappings(), DMCompositeGetAccess(),
         DMCompositeGetLocalVectors(), DMCompositeRestoreLocalVectors(), DMCompositeGetEntries()
         DMCompositeScatterArray()

@*/
PetscErrorCode  DMCompositeScatter(DM dm,Vec gvec,...)
{
  va_list                Argp;
  struct DMCompositeLink *next;
  PetscInt               cnt;
  DM_Composite           *com = (DM_Composite*)dm->data;
  PetscBool              flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidHeaderSpecific(gvec,VEC_CLASSID,2);
  CHKERRQ(PetscObjectTypeCompare((PetscObject)dm,DMCOMPOSITE,&flg));
  PetscCheck(flg,PetscObjectComm((PetscObject)dm),PETSC_ERR_USER,"Not for type %s",((PetscObject)dm)->type_name);
  if (!com->setup) {
    CHKERRQ(DMSetUp(dm));
  }

  /* loop over packed objects, handling one at at time */
  va_start(Argp,gvec);
  for (cnt=3,next=com->next; next; cnt++,next=next->next) {
    Vec local;
    local = va_arg(Argp, Vec);
    if (local) {
      Vec               global;
      const PetscScalar *array;
      PetscDisableStaticAnalyzerForExpressionUnderstandingThatThisIsDangerousAndBugprone(PetscValidHeaderSpecific(local,VEC_CLASSID,cnt));
      CHKERRQ(DMGetGlobalVector(next->dm,&global));
      CHKERRQ(VecGetArrayRead(gvec,&array));
      CHKERRQ(VecPlaceArray(global,array+next->rstart));
      CHKERRQ(DMGlobalToLocalBegin(next->dm,global,INSERT_VALUES,local));
      CHKERRQ(DMGlobalToLocalEnd(next->dm,global,INSERT_VALUES,local));
      CHKERRQ(VecRestoreArrayRead(gvec,&array));
      CHKERRQ(VecResetArray(global));
      CHKERRQ(DMRestoreGlobalVector(next->dm,&global));
    }
  }
  va_end(Argp);
  PetscFunctionReturn(0);
}

/*@
    DMCompositeScatterArray - Scatters from a global packed vector into its individual local vectors

    Collective on dm

    Input Parameters:
+    dm - the packer object
.    gvec - the global vector
-    lvecs - array of local vectors, NULL for any that are not needed

    Level: advanced

    Note:
    This is a non-variadic alternative to DMCompositeScatter()

.seealso DMDestroy(), DMCompositeAddDM(), DMCreateGlobalVector()
         DMCompositeGather(), DMCompositeCreate(), DMCompositeGetISLocalToGlobalMappings(), DMCompositeGetAccess(),
         DMCompositeGetLocalVectors(), DMCompositeRestoreLocalVectors(), DMCompositeGetEntries()

@*/
PetscErrorCode  DMCompositeScatterArray(DM dm,Vec gvec,Vec *lvecs)
{
  struct DMCompositeLink *next;
  PetscInt               i;
  DM_Composite           *com = (DM_Composite*)dm->data;
  PetscBool              flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidHeaderSpecific(gvec,VEC_CLASSID,2);
  CHKERRQ(PetscObjectTypeCompare((PetscObject)dm,DMCOMPOSITE,&flg));
  PetscCheck(flg,PetscObjectComm((PetscObject)dm),PETSC_ERR_USER,"Not for type %s",((PetscObject)dm)->type_name);
  if (!com->setup) {
    CHKERRQ(DMSetUp(dm));
  }

  /* loop over packed objects, handling one at at time */
  for (i=0,next=com->next; next; next=next->next,i++) {
    if (lvecs[i]) {
      Vec         global;
      const PetscScalar *array;
      PetscValidHeaderSpecific(lvecs[i],VEC_CLASSID,3);
      CHKERRQ(DMGetGlobalVector(next->dm,&global));
      CHKERRQ(VecGetArrayRead(gvec,&array));
      CHKERRQ(VecPlaceArray(global,(PetscScalar*)array+next->rstart));
      CHKERRQ(DMGlobalToLocalBegin(next->dm,global,INSERT_VALUES,lvecs[i]));
      CHKERRQ(DMGlobalToLocalEnd(next->dm,global,INSERT_VALUES,lvecs[i]));
      CHKERRQ(VecRestoreArrayRead(gvec,&array));
      CHKERRQ(VecResetArray(global));
      CHKERRQ(DMRestoreGlobalVector(next->dm,&global));
    }
  }
  PetscFunctionReturn(0);
}

/*@C
    DMCompositeGather - Gathers into a global packed vector from its individual local vectors

    Collective on dm

    Input Parameters:
+    dm - the packer object
.    gvec - the global vector
.    imode - INSERT_VALUES or ADD_VALUES
-    Vec ... - the individual sequential vectors, NULL for any that are not needed

    Level: advanced

    Not available from Fortran, Fortran users can use DMCompositeGatherArray()

.seealso DMDestroy(), DMCompositeAddDM(), DMCreateGlobalVector(),
         DMCompositeScatter(), DMCompositeCreate(), DMCompositeGetISLocalToGlobalMappings(), DMCompositeGetAccess(),
         DMCompositeGetLocalVectors(), DMCompositeRestoreLocalVectors(), DMCompositeGetEntries()

@*/
PetscErrorCode  DMCompositeGather(DM dm,InsertMode imode,Vec gvec,...)
{
  va_list                Argp;
  struct DMCompositeLink *next;
  DM_Composite           *com = (DM_Composite*)dm->data;
  PetscInt               cnt;
  PetscBool              flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidHeaderSpecific(gvec,VEC_CLASSID,3);
  CHKERRQ(PetscObjectTypeCompare((PetscObject)dm,DMCOMPOSITE,&flg));
  PetscCheck(flg,PetscObjectComm((PetscObject)dm),PETSC_ERR_USER,"Not for type %s",((PetscObject)dm)->type_name);
  if (!com->setup) {
    CHKERRQ(DMSetUp(dm));
  }

  /* loop over packed objects, handling one at at time */
  va_start(Argp,gvec);
  for (cnt=3,next=com->next; next; cnt++,next=next->next) {
    Vec local;
    local = va_arg(Argp, Vec);
    if (local) {
      PetscScalar *array;
      Vec         global;
      PetscDisableStaticAnalyzerForExpressionUnderstandingThatThisIsDangerousAndBugprone(PetscValidHeaderSpecific(local,VEC_CLASSID,cnt));
      CHKERRQ(DMGetGlobalVector(next->dm,&global));
      CHKERRQ(VecGetArray(gvec,&array));
      CHKERRQ(VecPlaceArray(global,array+next->rstart));
      CHKERRQ(DMLocalToGlobalBegin(next->dm,local,imode,global));
      CHKERRQ(DMLocalToGlobalEnd(next->dm,local,imode,global));
      CHKERRQ(VecRestoreArray(gvec,&array));
      CHKERRQ(VecResetArray(global));
      CHKERRQ(DMRestoreGlobalVector(next->dm,&global));
    }
  }
  va_end(Argp);
  PetscFunctionReturn(0);
}

/*@
    DMCompositeGatherArray - Gathers into a global packed vector from its individual local vectors

    Collective on dm

    Input Parameters:
+    dm - the packer object
.    gvec - the global vector
.    imode - INSERT_VALUES or ADD_VALUES
-    lvecs - the individual sequential vectors, NULL for any that are not needed

    Level: advanced

    Notes:
    This is a non-variadic alternative to DMCompositeGather().

.seealso DMDestroy(), DMCompositeAddDM(), DMCreateGlobalVector(),
         DMCompositeScatter(), DMCompositeCreate(), DMCompositeGetISLocalToGlobalMappings(), DMCompositeGetAccess(),
         DMCompositeGetLocalVectors(), DMCompositeRestoreLocalVectors(), DMCompositeGetEntries(),
@*/
PetscErrorCode  DMCompositeGatherArray(DM dm,InsertMode imode,Vec gvec,Vec *lvecs)
{
  struct DMCompositeLink *next;
  DM_Composite           *com = (DM_Composite*)dm->data;
  PetscInt               i;
  PetscBool              flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidHeaderSpecific(gvec,VEC_CLASSID,3);
  CHKERRQ(PetscObjectTypeCompare((PetscObject)dm,DMCOMPOSITE,&flg));
  PetscCheck(flg,PetscObjectComm((PetscObject)dm),PETSC_ERR_USER,"Not for type %s",((PetscObject)dm)->type_name);
  if (!com->setup) {
    CHKERRQ(DMSetUp(dm));
  }

  /* loop over packed objects, handling one at at time */
  for (next=com->next,i=0; next; next=next->next,i++) {
    if (lvecs[i]) {
      PetscScalar *array;
      Vec         global;
      PetscValidHeaderSpecific(lvecs[i],VEC_CLASSID,4);
      CHKERRQ(DMGetGlobalVector(next->dm,&global));
      CHKERRQ(VecGetArray(gvec,&array));
      CHKERRQ(VecPlaceArray(global,array+next->rstart));
      CHKERRQ(DMLocalToGlobalBegin(next->dm,lvecs[i],imode,global));
      CHKERRQ(DMLocalToGlobalEnd(next->dm,lvecs[i],imode,global));
      CHKERRQ(VecRestoreArray(gvec,&array));
      CHKERRQ(VecResetArray(global));
      CHKERRQ(DMRestoreGlobalVector(next->dm,&global));
    }
  }
  PetscFunctionReturn(0);
}

/*@
    DMCompositeAddDM - adds a DM vector to a DMComposite

    Collective on dm

    Input Parameters:
+    dmc - the DMComposite (packer) object
-    dm - the DM object

    Level: advanced

.seealso DMDestroy(), DMCompositeGather(), DMCompositeAddDM(), DMCreateGlobalVector(),
         DMCompositeScatter(), DMCompositeCreate(), DMCompositeGetISLocalToGlobalMappings(), DMCompositeGetAccess(),
         DMCompositeGetLocalVectors(), DMCompositeRestoreLocalVectors(), DMCompositeGetEntries()

@*/
PetscErrorCode  DMCompositeAddDM(DM dmc,DM dm)
{
  PetscInt               n,nlocal;
  struct DMCompositeLink *mine,*next;
  Vec                    global,local;
  DM_Composite           *com = (DM_Composite*)dmc->data;
  PetscBool              flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dmc,DM_CLASSID,1);
  PetscValidHeaderSpecific(dm,DM_CLASSID,2);
  CHKERRQ(PetscObjectTypeCompare((PetscObject)dmc,DMCOMPOSITE,&flg));
  PetscCheck(flg,PetscObjectComm((PetscObject)dm),PETSC_ERR_USER,"Not for type %s",((PetscObject)dm)->type_name);
  next = com->next;
  PetscCheck(!com->setup,PetscObjectComm((PetscObject)dmc),PETSC_ERR_ARG_WRONGSTATE,"Cannot add a DM once you have used the DMComposite");

  /* create new link */
  CHKERRQ(PetscNew(&mine));
  CHKERRQ(PetscObjectReference((PetscObject)dm));
  CHKERRQ(DMGetGlobalVector(dm,&global));
  CHKERRQ(VecGetLocalSize(global,&n));
  CHKERRQ(DMRestoreGlobalVector(dm,&global));
  CHKERRQ(DMGetLocalVector(dm,&local));
  CHKERRQ(VecGetSize(local,&nlocal));
  CHKERRQ(DMRestoreLocalVector(dm,&local));

  mine->n      = n;
  mine->nlocal = nlocal;
  mine->dm     = dm;
  mine->next   = NULL;
  com->n      += n;
  com->nghost += nlocal;

  /* add to end of list */
  if (!next) com->next = mine;
  else {
    while (next->next) next = next->next;
    next->next = mine;
  }
  com->nDM++;
  com->nmine++;
  PetscFunctionReturn(0);
}

#include <petscdraw.h>
PETSC_EXTERN PetscErrorCode  VecView_MPI(Vec,PetscViewer);
PetscErrorCode  VecView_DMComposite(Vec gvec,PetscViewer viewer)
{
  DM                     dm;
  struct DMCompositeLink *next;
  PetscBool              isdraw;
  DM_Composite           *com;

  PetscFunctionBegin;
  CHKERRQ(VecGetDM(gvec, &dm));
  PetscCheck(dm,PetscObjectComm((PetscObject)gvec),PETSC_ERR_ARG_WRONG,"Vector not generated from a DMComposite");
  com  = (DM_Composite*)dm->data;
  next = com->next;

  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERDRAW,&isdraw));
  if (!isdraw) {
    /* do I really want to call this? */
    CHKERRQ(VecView_MPI(gvec,viewer));
  } else {
    PetscInt cnt = 0;

    /* loop over packed objects, handling one at at time */
    while (next) {
      Vec               vec;
      const PetscScalar *array;
      PetscInt          bs;

      /* Should use VecGetSubVector() eventually, but would need to forward the DM for that to work */
      CHKERRQ(DMGetGlobalVector(next->dm,&vec));
      CHKERRQ(VecGetArrayRead(gvec,&array));
      CHKERRQ(VecPlaceArray(vec,(PetscScalar*)array+next->rstart));
      CHKERRQ(VecRestoreArrayRead(gvec,&array));
      CHKERRQ(VecView(vec,viewer));
      CHKERRQ(VecResetArray(vec));
      CHKERRQ(VecGetBlockSize(vec,&bs));
      CHKERRQ(DMRestoreGlobalVector(next->dm,&vec));
      CHKERRQ(PetscViewerDrawBaseAdd(viewer,bs));
      cnt += bs;
      next = next->next;
    }
    CHKERRQ(PetscViewerDrawBaseAdd(viewer,-cnt));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode  DMCreateGlobalVector_Composite(DM dm,Vec *gvec)
{
  DM_Composite   *com = (DM_Composite*)dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  CHKERRQ(DMSetFromOptions(dm));
  CHKERRQ(DMSetUp(dm));
  CHKERRQ(VecCreate(PetscObjectComm((PetscObject)dm),gvec));
  CHKERRQ(VecSetType(*gvec,dm->vectype));
  CHKERRQ(VecSetSizes(*gvec,com->n,com->N));
  CHKERRQ(VecSetDM(*gvec, dm));
  CHKERRQ(VecSetOperation(*gvec,VECOP_VIEW,(void (*)(void))VecView_DMComposite));
  PetscFunctionReturn(0);
}

PetscErrorCode  DMCreateLocalVector_Composite(DM dm,Vec *lvec)
{
  DM_Composite   *com = (DM_Composite*)dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  if (!com->setup) {
    CHKERRQ(DMSetFromOptions(dm));
    CHKERRQ(DMSetUp(dm));
  }
  CHKERRQ(VecCreate(PETSC_COMM_SELF,lvec));
  CHKERRQ(VecSetType(*lvec,dm->vectype));
  CHKERRQ(VecSetSizes(*lvec,com->nghost,PETSC_DECIDE));
  CHKERRQ(VecSetDM(*lvec, dm));
  PetscFunctionReturn(0);
}

/*@C
    DMCompositeGetISLocalToGlobalMappings - gets an ISLocalToGlobalMapping for each DM in the DMComposite, maps to the composite global space

    Collective on DM

    Input Parameter:
.    dm - the packer object

    Output Parameters:
.    ltogs - the individual mappings for each packed vector. Note that this includes
           all the ghost points that individual ghosted DMDA's may have.

    Level: advanced

    Notes:
       Each entry of ltogs should be destroyed with ISLocalToGlobalMappingDestroy(), the ltogs array should be freed with PetscFree().

    Not available from Fortran

.seealso DMDestroy(), DMCompositeAddDM(), DMCreateGlobalVector(),
         DMCompositeGather(), DMCompositeCreate(), DMCompositeGetAccess(), DMCompositeScatter(),
         DMCompositeGetLocalVectors(), DMCompositeRestoreLocalVectors(),DMCompositeGetEntries()

@*/
PetscErrorCode  DMCompositeGetISLocalToGlobalMappings(DM dm,ISLocalToGlobalMapping **ltogs)
{
  PetscInt               i,*idx,n,cnt;
  struct DMCompositeLink *next;
  PetscMPIInt            rank;
  DM_Composite           *com = (DM_Composite*)dm->data;
  PetscBool              flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  CHKERRQ(PetscObjectTypeCompare((PetscObject)dm,DMCOMPOSITE,&flg));
  PetscCheck(flg,PetscObjectComm((PetscObject)dm),PETSC_ERR_USER,"Not for type %s",((PetscObject)dm)->type_name);
  CHKERRQ(DMSetUp(dm));
  CHKERRQ(PetscMalloc1(com->nDM,ltogs));
  next = com->next;
  CHKERRMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)dm),&rank));

  /* loop over packed objects, handling one at at time */
  cnt = 0;
  while (next) {
    ISLocalToGlobalMapping ltog;
    PetscMPIInt            size;
    const PetscInt         *suboff,*indices;
    Vec                    global;

    /* Get sub-DM global indices for each local dof */
    CHKERRQ(DMGetLocalToGlobalMapping(next->dm,&ltog));
    CHKERRQ(ISLocalToGlobalMappingGetSize(ltog,&n));
    CHKERRQ(ISLocalToGlobalMappingGetIndices(ltog,&indices));
    CHKERRQ(PetscMalloc1(n,&idx));

    /* Get the offsets for the sub-DM global vector */
    CHKERRQ(DMGetGlobalVector(next->dm,&global));
    CHKERRQ(VecGetOwnershipRanges(global,&suboff));
    CHKERRMPI(MPI_Comm_size(PetscObjectComm((PetscObject)global),&size));

    /* Shift the sub-DM definition of the global space to the composite global space */
    for (i=0; i<n; i++) {
      PetscInt subi = indices[i],lo = 0,hi = size,t;
      /* There's no consensus on what a negative index means,
         except for skipping when setting the values in vectors and matrices */
      if (subi < 0) { idx[i] = subi - next->grstarts[rank]; continue; }
      /* Binary search to find which rank owns subi */
      while (hi-lo > 1) {
        t = lo + (hi-lo)/2;
        if (suboff[t] > subi) hi = t;
        else                  lo = t;
      }
      idx[i] = subi - suboff[lo] + next->grstarts[lo];
    }
    CHKERRQ(ISLocalToGlobalMappingRestoreIndices(ltog,&indices));
    CHKERRQ(ISLocalToGlobalMappingCreate(PetscObjectComm((PetscObject)dm),1,n,idx,PETSC_OWN_POINTER,&(*ltogs)[cnt]));
    CHKERRQ(DMRestoreGlobalVector(next->dm,&global));
    next = next->next;
    cnt++;
  }
  PetscFunctionReturn(0);
}

/*@C
   DMCompositeGetLocalISs - Gets index sets for each component of a composite local vector

   Not Collective

   Input Parameter:
. dm - composite DM

   Output Parameter:
. is - array of serial index sets for each each component of the DMComposite

   Level: intermediate

   Notes:
   At present, a composite local vector does not normally exist.  This function is used to provide index sets for
   MatGetLocalSubMatrix().  In the future, the scatters for each entry in the DMComposite may be be merged into a single
   scatter to a composite local vector.  The user should not typically need to know which is being done.

   To get the composite global indices at all local points (including ghosts), use DMCompositeGetISLocalToGlobalMappings().

   To get index sets for pieces of the composite global vector, use DMCompositeGetGlobalISs().

   Each returned IS should be destroyed with ISDestroy(), the array should be freed with PetscFree().

   Not available from Fortran

.seealso: DMCompositeGetGlobalISs(), DMCompositeGetISLocalToGlobalMappings(), MatGetLocalSubMatrix(), MatCreateLocalRef()
@*/
PetscErrorCode  DMCompositeGetLocalISs(DM dm,IS **is)
{
  DM_Composite           *com = (DM_Composite*)dm->data;
  struct DMCompositeLink *link;
  PetscInt               cnt,start;
  PetscBool              flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidPointer(is,2);
  CHKERRQ(PetscObjectTypeCompare((PetscObject)dm,DMCOMPOSITE,&flg));
  PetscCheck(flg,PetscObjectComm((PetscObject)dm),PETSC_ERR_USER,"Not for type %s",((PetscObject)dm)->type_name);
  CHKERRQ(PetscMalloc1(com->nmine,is));
  for (cnt=0,start=0,link=com->next; link; start+=link->nlocal,cnt++,link=link->next) {
    PetscInt bs;
    CHKERRQ(ISCreateStride(PETSC_COMM_SELF,link->nlocal,start,1,&(*is)[cnt]));
    CHKERRQ(DMGetBlockSize(link->dm,&bs));
    CHKERRQ(ISSetBlockSize((*is)[cnt],bs));
  }
  PetscFunctionReturn(0);
}

/*@C
    DMCompositeGetGlobalISs - Gets the index sets for each composed object

    Collective on dm

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

    Fortran Notes:

       The output argument 'is' must be an allocated array of sufficient length, which can be learned using DMCompositeGetNumberDM().

.seealso DMDestroy(), DMCompositeAddDM(), DMCreateGlobalVector(),
         DMCompositeGather(), DMCompositeCreate(), DMCompositeGetAccess(), DMCompositeScatter(),
         DMCompositeGetLocalVectors(), DMCompositeRestoreLocalVectors(),DMCompositeGetEntries()

@*/
PetscErrorCode  DMCompositeGetGlobalISs(DM dm,IS *is[])
{
  PetscInt               cnt = 0;
  struct DMCompositeLink *next;
  PetscMPIInt            rank;
  DM_Composite           *com = (DM_Composite*)dm->data;
  PetscBool              flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  CHKERRQ(PetscObjectTypeCompare((PetscObject)dm,DMCOMPOSITE,&flg));
  PetscCheck(flg,PetscObjectComm((PetscObject)dm),PETSC_ERR_USER,"Not for type %s",((PetscObject)dm)->type_name);
  PetscCheck(dm->setupcalled,PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Must call DMSetUp() before");
  CHKERRQ(PetscMalloc1(com->nDM,is));
  next = com->next;
  CHKERRMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)dm),&rank));

  /* loop over packed objects, handling one at at time */
  while (next) {
    PetscDS prob;

    CHKERRQ(ISCreateStride(PetscObjectComm((PetscObject)dm),next->n,next->grstart,1,&(*is)[cnt]));
    CHKERRQ(DMGetDS(dm, &prob));
    if (prob) {
      MatNullSpace space;
      Mat          pmat;
      PetscObject  disc;
      PetscInt     Nf;

      CHKERRQ(PetscDSGetNumFields(prob, &Nf));
      if (cnt < Nf) {
        CHKERRQ(PetscDSGetDiscretization(prob, cnt, &disc));
        CHKERRQ(PetscObjectQuery(disc, "nullspace", (PetscObject*) &space));
        if (space) CHKERRQ(PetscObjectCompose((PetscObject) (*is)[cnt], "nullspace", (PetscObject) space));
        CHKERRQ(PetscObjectQuery(disc, "nearnullspace", (PetscObject*) &space));
        if (space) CHKERRQ(PetscObjectCompose((PetscObject) (*is)[cnt], "nearnullspace", (PetscObject) space));
        CHKERRQ(PetscObjectQuery(disc, "pmat", (PetscObject*) &pmat));
        if (pmat) CHKERRQ(PetscObjectCompose((PetscObject) (*is)[cnt], "pmat", (PetscObject) pmat));
      }
    }
    cnt++;
    next = next->next;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMCreateFieldIS_Composite(DM dm, PetscInt *numFields,char ***fieldNames, IS **fields)
{
  PetscInt       nDM;
  DM             *dms;
  PetscInt       i;

  PetscFunctionBegin;
  CHKERRQ(DMCompositeGetNumberDM(dm, &nDM));
  if (numFields) *numFields = nDM;
  CHKERRQ(DMCompositeGetGlobalISs(dm, fields));
  if (fieldNames) {
    CHKERRQ(PetscMalloc1(nDM, &dms));
    CHKERRQ(PetscMalloc1(nDM, fieldNames));
    CHKERRQ(DMCompositeGetEntriesArray(dm, dms));
    for (i=0; i<nDM; i++) {
      char       buf[256];
      const char *splitname;

      /* Split naming precedence: object name, prefix, number */
      splitname = ((PetscObject) dm)->name;
      if (!splitname) {
        CHKERRQ(PetscObjectGetOptionsPrefix((PetscObject)dms[i],&splitname));
        if (splitname) {
          size_t len;
          CHKERRQ(PetscStrncpy(buf,splitname,sizeof(buf)));
          buf[sizeof(buf) - 1] = 0;
          CHKERRQ(PetscStrlen(buf,&len));
          if (buf[len-1] == '_') buf[len-1] = 0; /* Remove trailing underscore if it was used */
          splitname = buf;
        }
      }
      if (!splitname) {
        CHKERRQ(PetscSNPrintf(buf,sizeof(buf),"%D",i));
        splitname = buf;
      }
      CHKERRQ(PetscStrallocpy(splitname,&(*fieldNames)[i]));
    }
    CHKERRQ(PetscFree(dms));
  }
  PetscFunctionReturn(0);
}

/*
 This could take over from DMCreateFieldIS(), as it is more general,
 making DMCreateFieldIS() a special case -- calling with dmlist == NULL;
 At this point it's probably best to be less intrusive, however.
 */
PetscErrorCode DMCreateFieldDecomposition_Composite(DM dm, PetscInt *len,char ***namelist, IS **islist, DM **dmlist)
{
  PetscInt       nDM;
  PetscInt       i;

  PetscFunctionBegin;
  CHKERRQ(DMCreateFieldIS_Composite(dm, len, namelist, islist));
  if (dmlist) {
    CHKERRQ(DMCompositeGetNumberDM(dm, &nDM));
    CHKERRQ(PetscMalloc1(nDM, dmlist));
    CHKERRQ(DMCompositeGetEntriesArray(dm, *dmlist));
    for (i=0; i<nDM; i++) {
      CHKERRQ(PetscObjectReference((PetscObject)((*dmlist)[i])));
    }
  }
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------------------*/
/*@C
    DMCompositeGetLocalVectors - Gets local vectors for each part of a DMComposite.
       Use DMCompositeRestoreLocalVectors() to return them.

    Not Collective

    Input Parameter:
.    dm - the packer object

    Output Parameter:
.   Vec ... - the individual sequential Vecs

    Level: advanced

    Not available from Fortran

.seealso DMDestroy(), DMCompositeAddDM(), DMCreateGlobalVector(),
         DMCompositeGather(), DMCompositeCreate(), DMCompositeGetISLocalToGlobalMappings(), DMCompositeGetAccess(),
         DMCompositeRestoreLocalVectors(), DMCompositeScatter(), DMCompositeGetEntries()

@*/
PetscErrorCode  DMCompositeGetLocalVectors(DM dm,...)
{
  va_list                Argp;
  struct DMCompositeLink *next;
  DM_Composite           *com = (DM_Composite*)dm->data;
  PetscBool              flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  CHKERRQ(PetscObjectTypeCompare((PetscObject)dm,DMCOMPOSITE,&flg));
  PetscCheck(flg,PetscObjectComm((PetscObject)dm),PETSC_ERR_USER,"Not for type %s",((PetscObject)dm)->type_name);
  next = com->next;
  /* loop over packed objects, handling one at at time */
  va_start(Argp,dm);
  while (next) {
    Vec *vec;
    vec = va_arg(Argp, Vec*);
    if (vec) CHKERRQ(DMGetLocalVector(next->dm,vec));
    next = next->next;
  }
  va_end(Argp);
  PetscFunctionReturn(0);
}

/*@C
    DMCompositeRestoreLocalVectors - Restores local vectors for each part of a DMComposite.

    Not Collective

    Input Parameter:
.    dm - the packer object

    Output Parameter:
.   Vec ... - the individual sequential Vecs

    Level: advanced

    Not available from Fortran

.seealso DMDestroy(), DMCompositeAddDM(), DMCreateGlobalVector(),
         DMCompositeGather(), DMCompositeCreate(), DMCompositeGetISLocalToGlobalMappings(), DMCompositeGetAccess(),
         DMCompositeGetLocalVectors(), DMCompositeScatter(), DMCompositeGetEntries()

@*/
PetscErrorCode  DMCompositeRestoreLocalVectors(DM dm,...)
{
  va_list                Argp;
  struct DMCompositeLink *next;
  DM_Composite           *com = (DM_Composite*)dm->data;
  PetscBool              flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  CHKERRQ(PetscObjectTypeCompare((PetscObject)dm,DMCOMPOSITE,&flg));
  PetscCheck(flg,PetscObjectComm((PetscObject)dm),PETSC_ERR_USER,"Not for type %s",((PetscObject)dm)->type_name);
  next = com->next;
  /* loop over packed objects, handling one at at time */
  va_start(Argp,dm);
  while (next) {
    Vec *vec;
    vec = va_arg(Argp, Vec*);
    if (vec) CHKERRQ(DMRestoreLocalVector(next->dm,vec));
    next = next->next;
  }
  va_end(Argp);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------------------*/
/*@C
    DMCompositeGetEntries - Gets the DM for each entry in a DMComposite.

    Not Collective

    Input Parameter:
.    dm - the packer object

    Output Parameter:
.   DM ... - the individual entries (DMs)

    Level: advanced

    Fortran Notes:
    Available as DMCompositeGetEntries() for one output DM, DMCompositeGetEntries2() for 2, etc

.seealso DMDestroy(), DMCompositeAddDM(), DMCreateGlobalVector(), DMCompositeGetEntriesArray()
         DMCompositeGather(), DMCompositeCreate(), DMCompositeGetISLocalToGlobalMappings(), DMCompositeGetAccess(),
         DMCompositeRestoreLocalVectors(), DMCompositeGetLocalVectors(),  DMCompositeScatter(),
         DMCompositeGetLocalVectors(), DMCompositeRestoreLocalVectors()

@*/
PetscErrorCode  DMCompositeGetEntries(DM dm,...)
{
  va_list                Argp;
  struct DMCompositeLink *next;
  DM_Composite           *com = (DM_Composite*)dm->data;
  PetscBool              flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  CHKERRQ(PetscObjectTypeCompare((PetscObject)dm,DMCOMPOSITE,&flg));
  PetscCheck(flg,PetscObjectComm((PetscObject)dm),PETSC_ERR_USER,"Not for type %s",((PetscObject)dm)->type_name);
  next = com->next;
  /* loop over packed objects, handling one at at time */
  va_start(Argp,dm);
  while (next) {
    DM *dmn;
    dmn = va_arg(Argp, DM*);
    if (dmn) *dmn = next->dm;
    next = next->next;
  }
  va_end(Argp);
  PetscFunctionReturn(0);
}

/*@C
    DMCompositeGetEntriesArray - Gets the DM for each entry in a DMComposite.

    Not Collective

    Input Parameter:
.    dm - the packer object

    Output Parameter:
.    dms - array of sufficient length (see DMCompositeGetNumberDM()) to hold the individual DMs

    Level: advanced

.seealso DMDestroy(), DMCompositeAddDM(), DMCreateGlobalVector(), DMCompositeGetEntries()
         DMCompositeGather(), DMCompositeCreate(), DMCompositeGetISLocalToGlobalMappings(), DMCompositeGetAccess(),
         DMCompositeRestoreLocalVectors(), DMCompositeGetLocalVectors(),  DMCompositeScatter(),
         DMCompositeGetLocalVectors(), DMCompositeRestoreLocalVectors()

@*/
PetscErrorCode DMCompositeGetEntriesArray(DM dm,DM dms[])
{
  struct DMCompositeLink *next;
  DM_Composite           *com = (DM_Composite*)dm->data;
  PetscInt               i;
  PetscBool              flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  CHKERRQ(PetscObjectTypeCompare((PetscObject)dm,DMCOMPOSITE,&flg));
  PetscCheck(flg,PetscObjectComm((PetscObject)dm),PETSC_ERR_USER,"Not for type %s",((PetscObject)dm)->type_name);
  /* loop over packed objects, handling one at at time */
  for (next=com->next,i=0; next; next=next->next,i++) dms[i] = next->dm;
  PetscFunctionReturn(0);
}

typedef struct {
  DM          dm;
  PetscViewer *subv;
  Vec         *vecs;
} GLVisViewerCtx;

static PetscErrorCode  DestroyGLVisViewerCtx_Private(void *vctx)
{
  GLVisViewerCtx *ctx = (GLVisViewerCtx*)vctx;
  PetscInt       i,n;

  PetscFunctionBegin;
  CHKERRQ(DMCompositeGetNumberDM(ctx->dm,&n));
  for (i = 0; i < n; i++) {
    CHKERRQ(PetscViewerDestroy(&ctx->subv[i]));
  }
  CHKERRQ(PetscFree2(ctx->subv,ctx->vecs));
  CHKERRQ(DMDestroy(&ctx->dm));
  CHKERRQ(PetscFree(ctx));
  PetscFunctionReturn(0);
}

static PetscErrorCode  DMCompositeSampleGLVisFields_Private(PetscObject oX, PetscInt nf, PetscObject oXfield[], void *vctx)
{
  Vec            X = (Vec)oX;
  GLVisViewerCtx *ctx = (GLVisViewerCtx*)vctx;
  PetscInt       i,n,cumf;

  PetscFunctionBegin;
  CHKERRQ(DMCompositeGetNumberDM(ctx->dm,&n));
  CHKERRQ(DMCompositeGetAccessArray(ctx->dm,X,n,NULL,ctx->vecs));
  for (i = 0, cumf = 0; i < n; i++) {
    PetscErrorCode (*g2l)(PetscObject,PetscInt,PetscObject[],void*);
    void           *fctx;
    PetscInt       nfi;

    CHKERRQ(PetscViewerGLVisGetFields_Private(ctx->subv[i],&nfi,NULL,NULL,&g2l,NULL,&fctx));
    if (!nfi) continue;
    if (g2l) {
      CHKERRQ((*g2l)((PetscObject)ctx->vecs[i],nfi,oXfield+cumf,fctx));
    } else {
      CHKERRQ(VecCopy(ctx->vecs[i],(Vec)(oXfield[cumf])));
    }
    cumf += nfi;
  }
  CHKERRQ(DMCompositeRestoreAccessArray(ctx->dm,X,n,NULL,ctx->vecs));
  PetscFunctionReturn(0);
}

static PetscErrorCode  DMSetUpGLVisViewer_Composite(PetscObject odm, PetscViewer viewer)
{
  DM             dm = (DM)odm, *dms;
  Vec            *Ufds;
  GLVisViewerCtx *ctx;
  PetscInt       i,n,tnf,*sdim;
  char           **fecs;

  PetscFunctionBegin;
  CHKERRQ(PetscNew(&ctx));
  CHKERRQ(PetscObjectReference((PetscObject)dm));
  ctx->dm = dm;
  CHKERRQ(DMCompositeGetNumberDM(dm,&n));
  CHKERRQ(PetscMalloc1(n,&dms));
  CHKERRQ(DMCompositeGetEntriesArray(dm,dms));
  CHKERRQ(PetscMalloc2(n,&ctx->subv,n,&ctx->vecs));
  for (i = 0, tnf = 0; i < n; i++) {
    PetscInt nf;

    CHKERRQ(PetscViewerCreate(PetscObjectComm(odm),&ctx->subv[i]));
    CHKERRQ(PetscViewerSetType(ctx->subv[i],PETSCVIEWERGLVIS));
    CHKERRQ(PetscViewerGLVisSetDM_Private(ctx->subv[i],(PetscObject)dms[i]));
    CHKERRQ(PetscViewerGLVisGetFields_Private(ctx->subv[i],&nf,NULL,NULL,NULL,NULL,NULL));
    tnf += nf;
  }
  CHKERRQ(PetscFree(dms));
  CHKERRQ(PetscMalloc3(tnf,&fecs,tnf,&sdim,tnf,&Ufds));
  for (i = 0, tnf = 0; i < n; i++) {
    PetscInt   *sd,nf,f;
    const char **fec;
    Vec        *Uf;

    CHKERRQ(PetscViewerGLVisGetFields_Private(ctx->subv[i],&nf,&fec,&sd,NULL,(PetscObject**)&Uf,NULL));
    for (f = 0; f < nf; f++) {
      CHKERRQ(PetscStrallocpy(fec[f],&fecs[tnf+f]));
      Ufds[tnf+f] = Uf[f];
      sdim[tnf+f] = sd[f];
    }
    tnf += nf;
  }
  CHKERRQ(PetscViewerGLVisSetFields(viewer,tnf,(const char**)fecs,sdim,DMCompositeSampleGLVisFields_Private,(PetscObject*)Ufds,ctx,DestroyGLVisViewerCtx_Private));
  for (i = 0; i < tnf; i++) {
    CHKERRQ(PetscFree(fecs[i]));
  }
  CHKERRQ(PetscFree3(fecs,sdim,Ufds));
  PetscFunctionReturn(0);
}

PetscErrorCode  DMRefine_Composite(DM dmi,MPI_Comm comm,DM *fine)
{
  struct DMCompositeLink *next;
  DM_Composite           *com = (DM_Composite*)dmi->data;
  DM                     dm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dmi,DM_CLASSID,1);
  if (comm == MPI_COMM_NULL) {
    CHKERRQ(PetscObjectGetComm((PetscObject)dmi,&comm));
  }
  CHKERRQ(DMSetUp(dmi));
  next = com->next;
  CHKERRQ(DMCompositeCreate(comm,fine));

  /* loop over packed objects, handling one at at time */
  while (next) {
    CHKERRQ(DMRefine(next->dm,comm,&dm));
    CHKERRQ(DMCompositeAddDM(*fine,dm));
    CHKERRQ(PetscObjectDereference((PetscObject)dm));
    next = next->next;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode  DMCoarsen_Composite(DM dmi,MPI_Comm comm,DM *fine)
{
  struct DMCompositeLink *next;
  DM_Composite           *com = (DM_Composite*)dmi->data;
  DM                     dm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dmi,DM_CLASSID,1);
  CHKERRQ(DMSetUp(dmi));
  if (comm == MPI_COMM_NULL) {
    CHKERRQ(PetscObjectGetComm((PetscObject)dmi,&comm));
  }
  next = com->next;
  CHKERRQ(DMCompositeCreate(comm,fine));

  /* loop over packed objects, handling one at at time */
  while (next) {
    CHKERRQ(DMCoarsen(next->dm,comm,&dm));
    CHKERRQ(DMCompositeAddDM(*fine,dm));
    CHKERRQ(PetscObjectDereference((PetscObject)dm));
    next = next->next;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode  DMCreateInterpolation_Composite(DM coarse,DM fine,Mat *A,Vec *v)
{
  PetscInt               m,n,M,N,nDM,i;
  struct DMCompositeLink *nextc;
  struct DMCompositeLink *nextf;
  Vec                    gcoarse,gfine,*vecs;
  DM_Composite           *comcoarse = (DM_Composite*)coarse->data;
  DM_Composite           *comfine   = (DM_Composite*)fine->data;
  Mat                    *mats;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(coarse,DM_CLASSID,1);
  PetscValidHeaderSpecific(fine,DM_CLASSID,2);
  CHKERRQ(DMSetUp(coarse));
  CHKERRQ(DMSetUp(fine));
  /* use global vectors only for determining matrix layout */
  CHKERRQ(DMGetGlobalVector(coarse,&gcoarse));
  CHKERRQ(DMGetGlobalVector(fine,&gfine));
  CHKERRQ(VecGetLocalSize(gcoarse,&n));
  CHKERRQ(VecGetLocalSize(gfine,&m));
  CHKERRQ(VecGetSize(gcoarse,&N));
  CHKERRQ(VecGetSize(gfine,&M));
  CHKERRQ(DMRestoreGlobalVector(coarse,&gcoarse));
  CHKERRQ(DMRestoreGlobalVector(fine,&gfine));

  nDM = comfine->nDM;
  PetscCheckFalse(nDM != comcoarse->nDM,PetscObjectComm((PetscObject)fine),PETSC_ERR_ARG_INCOMP,"Fine DMComposite has %D entries, but coarse has %D",nDM,comcoarse->nDM);
  CHKERRQ(PetscCalloc1(nDM*nDM,&mats));
  if (v) {
    CHKERRQ(PetscCalloc1(nDM,&vecs));
  }

  /* loop over packed objects, handling one at at time */
  for (nextc=comcoarse->next,nextf=comfine->next,i=0; nextc; nextc=nextc->next,nextf=nextf->next,i++) {
    if (!v) {
      CHKERRQ(DMCreateInterpolation(nextc->dm,nextf->dm,&mats[i*nDM+i],NULL));
    } else {
      CHKERRQ(DMCreateInterpolation(nextc->dm,nextf->dm,&mats[i*nDM+i],&vecs[i]));
    }
  }
  CHKERRQ(MatCreateNest(PetscObjectComm((PetscObject)fine),nDM,NULL,nDM,NULL,mats,A));
  if (v) {
    CHKERRQ(VecCreateNest(PetscObjectComm((PetscObject)fine),nDM,NULL,vecs,v));
  }
  for (i=0; i<nDM*nDM; i++) CHKERRQ(MatDestroy(&mats[i]));
  CHKERRQ(PetscFree(mats));
  if (v) {
    for (i=0; i<nDM; i++) CHKERRQ(VecDestroy(&vecs[i]));
    CHKERRQ(PetscFree(vecs));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMGetLocalToGlobalMapping_Composite(DM dm)
{
  DM_Composite           *com = (DM_Composite*)dm->data;
  ISLocalToGlobalMapping *ltogs;
  PetscInt               i;

  PetscFunctionBegin;
  /* Set the ISLocalToGlobalMapping on the new matrix */
  CHKERRQ(DMCompositeGetISLocalToGlobalMappings(dm,&ltogs));
  CHKERRQ(ISLocalToGlobalMappingConcatenate(PetscObjectComm((PetscObject)dm),com->nDM,ltogs,&dm->ltogmap));
  for (i=0; i<com->nDM; i++) CHKERRQ(ISLocalToGlobalMappingDestroy(&ltogs[i]));
  CHKERRQ(PetscFree(ltogs));
  PetscFunctionReturn(0);
}

PetscErrorCode  DMCreateColoring_Composite(DM dm,ISColoringType ctype,ISColoring *coloring)
{
  PetscInt        n,i,cnt;
  ISColoringValue *colors;
  PetscBool       dense  = PETSC_FALSE;
  ISColoringValue maxcol = 0;
  DM_Composite    *com   = (DM_Composite*)dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscCheckFalse(ctype == IS_COLORING_LOCAL,PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Only global coloring supported");
  else if (ctype == IS_COLORING_GLOBAL) {
    n = com->n;
  } else SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_OUTOFRANGE,"Unknown ISColoringType");
  CHKERRQ(PetscMalloc1(n,&colors)); /* freed in ISColoringDestroy() */

  CHKERRQ(PetscOptionsGetBool(((PetscObject)dm)->options,((PetscObject)dm)->prefix,"-dmcomposite_dense_jacobian",&dense,NULL));
  if (dense) {
    for (i=0; i<n; i++) {
      colors[i] = (ISColoringValue)(com->rstart + i);
    }
    maxcol = com->N;
  } else {
    struct DMCompositeLink *next = com->next;
    PetscMPIInt            rank;

    CHKERRMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)dm),&rank));
    cnt  = 0;
    while (next) {
      ISColoring lcoloring;

      CHKERRQ(DMCreateColoring(next->dm,IS_COLORING_GLOBAL,&lcoloring));
      for (i=0; i<lcoloring->N; i++) {
        colors[cnt++] = maxcol + lcoloring->colors[i];
      }
      maxcol += lcoloring->n;
      CHKERRQ(ISColoringDestroy(&lcoloring));
      next    = next->next;
    }
  }
  CHKERRQ(ISColoringCreate(PetscObjectComm((PetscObject)dm),maxcol,n,colors,PETSC_OWN_POINTER,coloring));
  PetscFunctionReturn(0);
}

PetscErrorCode  DMGlobalToLocalBegin_Composite(DM dm,Vec gvec,InsertMode mode,Vec lvec)
{
  struct DMCompositeLink *next;
  PetscScalar            *garray,*larray;
  DM_Composite           *com = (DM_Composite*)dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidHeaderSpecific(gvec,VEC_CLASSID,2);

  if (!com->setup) {
    CHKERRQ(DMSetUp(dm));
  }

  CHKERRQ(VecGetArray(gvec,&garray));
  CHKERRQ(VecGetArray(lvec,&larray));

  /* loop over packed objects, handling one at at time */
  next = com->next;
  while (next) {
    Vec      local,global;
    PetscInt N;

    CHKERRQ(DMGetGlobalVector(next->dm,&global));
    CHKERRQ(VecGetLocalSize(global,&N));
    CHKERRQ(VecPlaceArray(global,garray));
    CHKERRQ(DMGetLocalVector(next->dm,&local));
    CHKERRQ(VecPlaceArray(local,larray));
    CHKERRQ(DMGlobalToLocalBegin(next->dm,global,mode,local));
    CHKERRQ(DMGlobalToLocalEnd(next->dm,global,mode,local));
    CHKERRQ(VecResetArray(global));
    CHKERRQ(VecResetArray(local));
    CHKERRQ(DMRestoreGlobalVector(next->dm,&global));
    CHKERRQ(DMRestoreLocalVector(next->dm,&local));

    larray += next->nlocal;
    garray += next->n;
    next    = next->next;
  }

  CHKERRQ(VecRestoreArray(gvec,NULL));
  CHKERRQ(VecRestoreArray(lvec,NULL));
  PetscFunctionReturn(0);
}

PetscErrorCode  DMGlobalToLocalEnd_Composite(DM dm,Vec gvec,InsertMode mode,Vec lvec)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidHeaderSpecific(gvec,VEC_CLASSID,2);
  PetscValidHeaderSpecific(lvec,VEC_CLASSID,4);
  PetscFunctionReturn(0);
}

PetscErrorCode  DMLocalToGlobalBegin_Composite(DM dm,Vec lvec,InsertMode mode,Vec gvec)
{
  struct DMCompositeLink *next;
  PetscScalar            *larray,*garray;
  DM_Composite           *com = (DM_Composite*)dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidHeaderSpecific(lvec,VEC_CLASSID,2);
  PetscValidHeaderSpecific(gvec,VEC_CLASSID,4);

  if (!com->setup) {
    CHKERRQ(DMSetUp(dm));
  }

  CHKERRQ(VecGetArray(lvec,&larray));
  CHKERRQ(VecGetArray(gvec,&garray));

  /* loop over packed objects, handling one at at time */
  next = com->next;
  while (next) {
    Vec      global,local;

    CHKERRQ(DMGetLocalVector(next->dm,&local));
    CHKERRQ(VecPlaceArray(local,larray));
    CHKERRQ(DMGetGlobalVector(next->dm,&global));
    CHKERRQ(VecPlaceArray(global,garray));
    CHKERRQ(DMLocalToGlobalBegin(next->dm,local,mode,global));
    CHKERRQ(DMLocalToGlobalEnd(next->dm,local,mode,global));
    CHKERRQ(VecResetArray(local));
    CHKERRQ(VecResetArray(global));
    CHKERRQ(DMRestoreGlobalVector(next->dm,&global));
    CHKERRQ(DMRestoreLocalVector(next->dm,&local));

    garray += next->n;
    larray += next->nlocal;
    next    = next->next;
  }

  CHKERRQ(VecRestoreArray(gvec,NULL));
  CHKERRQ(VecRestoreArray(lvec,NULL));

  PetscFunctionReturn(0);
}

PetscErrorCode  DMLocalToGlobalEnd_Composite(DM dm,Vec lvec,InsertMode mode,Vec gvec)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidHeaderSpecific(lvec,VEC_CLASSID,2);
  PetscValidHeaderSpecific(gvec,VEC_CLASSID,4);
  PetscFunctionReturn(0);
}

PetscErrorCode  DMLocalToLocalBegin_Composite(DM dm,Vec vec1,InsertMode mode,Vec vec2)
{
  struct DMCompositeLink *next;
  PetscScalar            *array1,*array2;
  DM_Composite           *com = (DM_Composite*)dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidHeaderSpecific(vec1,VEC_CLASSID,2);
  PetscValidHeaderSpecific(vec2,VEC_CLASSID,4);

  if (!com->setup) {
    CHKERRQ(DMSetUp(dm));
  }

  CHKERRQ(VecGetArray(vec1,&array1));
  CHKERRQ(VecGetArray(vec2,&array2));

  /* loop over packed objects, handling one at at time */
  next = com->next;
  while (next) {
    Vec      local1,local2;

    CHKERRQ(DMGetLocalVector(next->dm,&local1));
    CHKERRQ(VecPlaceArray(local1,array1));
    CHKERRQ(DMGetLocalVector(next->dm,&local2));
    CHKERRQ(VecPlaceArray(local2,array2));
    CHKERRQ(DMLocalToLocalBegin(next->dm,local1,mode,local2));
    CHKERRQ(DMLocalToLocalEnd(next->dm,local1,mode,local2));
    CHKERRQ(VecResetArray(local2));
    CHKERRQ(DMRestoreLocalVector(next->dm,&local2));
    CHKERRQ(VecResetArray(local1));
    CHKERRQ(DMRestoreLocalVector(next->dm,&local1));

    array1 += next->nlocal;
    array2 += next->nlocal;
    next    = next->next;
  }

  CHKERRQ(VecRestoreArray(vec1,NULL));
  CHKERRQ(VecRestoreArray(vec2,NULL));

  PetscFunctionReturn(0);
}

PetscErrorCode  DMLocalToLocalEnd_Composite(DM dm,Vec lvec,InsertMode mode,Vec gvec)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidHeaderSpecific(lvec,VEC_CLASSID,2);
  PetscValidHeaderSpecific(gvec,VEC_CLASSID,4);
  PetscFunctionReturn(0);
}

/*MC
   DMCOMPOSITE = "composite" - A DM object that is used to manage data for a collection of DMs

  Level: intermediate

.seealso: DMType, DM, DMDACreate(), DMCreate(), DMSetType(), DMCompositeCreate()
M*/

PETSC_EXTERN PetscErrorCode DMCreate_Composite(DM p)
{
  DM_Composite   *com;

  PetscFunctionBegin;
  CHKERRQ(PetscNewLog(p,&com));
  p->data       = com;
  com->n        = 0;
  com->nghost   = 0;
  com->next     = NULL;
  com->nDM      = 0;

  p->ops->createglobalvector              = DMCreateGlobalVector_Composite;
  p->ops->createlocalvector               = DMCreateLocalVector_Composite;
  p->ops->getlocaltoglobalmapping         = DMGetLocalToGlobalMapping_Composite;
  p->ops->createfieldis                   = DMCreateFieldIS_Composite;
  p->ops->createfielddecomposition        = DMCreateFieldDecomposition_Composite;
  p->ops->refine                          = DMRefine_Composite;
  p->ops->coarsen                         = DMCoarsen_Composite;
  p->ops->createinterpolation             = DMCreateInterpolation_Composite;
  p->ops->creatematrix                    = DMCreateMatrix_Composite;
  p->ops->getcoloring                     = DMCreateColoring_Composite;
  p->ops->globaltolocalbegin              = DMGlobalToLocalBegin_Composite;
  p->ops->globaltolocalend                = DMGlobalToLocalEnd_Composite;
  p->ops->localtoglobalbegin              = DMLocalToGlobalBegin_Composite;
  p->ops->localtoglobalend                = DMLocalToGlobalEnd_Composite;
  p->ops->localtolocalbegin               = DMLocalToLocalBegin_Composite;
  p->ops->localtolocalend                 = DMLocalToLocalEnd_Composite;
  p->ops->destroy                         = DMDestroy_Composite;
  p->ops->view                            = DMView_Composite;
  p->ops->setup                           = DMSetUp_Composite;

  CHKERRQ(PetscObjectComposeFunction((PetscObject)p,"DMSetUpGLVisViewer_C",DMSetUpGLVisViewer_Composite));
  PetscFunctionReturn(0);
}

/*@
    DMCompositeCreate - Creates a vector packer, used to generate "composite"
      vectors made up of several subvectors.

    Collective

    Input Parameter:
.   comm - the processors that will share the global vector

    Output Parameters:
.   packer - the packer object

    Level: advanced

.seealso DMDestroy(), DMCompositeAddDM(), DMCompositeScatter(), DMCOMPOSITE,DMCreate()
         DMCompositeGather(), DMCreateGlobalVector(), DMCompositeGetISLocalToGlobalMappings(), DMCompositeGetAccess()
         DMCompositeGetLocalVectors(), DMCompositeRestoreLocalVectors(), DMCompositeGetEntries()

@*/
PetscErrorCode  DMCompositeCreate(MPI_Comm comm,DM *packer)
{
  PetscFunctionBegin;
  PetscValidPointer(packer,2);
  CHKERRQ(DMCreate(comm,packer));
  CHKERRQ(DMSetType(*packer,DMCOMPOSITE));
  PetscFunctionReturn(0);
}
