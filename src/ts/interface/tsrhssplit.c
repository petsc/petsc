#include <petsc/private/tsimpl.h>        /*I "petscts.h"  I*/

static PetscErrorCode TSRHSSplitGetRHSSplit(TS ts,const char splitname[],TS_RHSSplitLink *isplit)
{
  PetscBool       found = PETSC_FALSE;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  *isplit = ts->tsrhssplit;
  /* look up the split */
  while (*isplit) {
    ierr = PetscStrcmp((*isplit)->splitname,splitname,&found);CHKERRQ(ierr);
    if (found) break;
    *isplit = (*isplit)->next;
  }
  PetscFunctionReturn(0);
}

/*@C
   TSRHSSplitSetIS - Set the index set for the specified split

   Logically Collective on TS

   Input Parameters:
+  ts        - the TS context obtained from TSCreate()
.  splitname - name of this split, if NULL the number of the split is used
-  is        - the index set for part of the solution vector

   Level: intermediate

.seealso: TSRHSSplitGetIS()

@*/
PetscErrorCode TSRHSSplitSetIS(TS ts,const char splitname[],IS is)
{
  TS_RHSSplitLink newsplit,next = ts->tsrhssplit;
  char            prefix[128];
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(is,IS_CLASSID,3);

  ierr = PetscNew(&newsplit);CHKERRQ(ierr);
  if (splitname) {
    ierr = PetscStrallocpy(splitname,&newsplit->splitname);CHKERRQ(ierr);
  } else {
    ierr = PetscMalloc1(8,&newsplit->splitname);CHKERRQ(ierr);
    ierr = PetscSNPrintf(newsplit->splitname,7,"%D",ts->num_rhs_splits);CHKERRQ(ierr);
  }
  ierr = PetscObjectReference((PetscObject)is);CHKERRQ(ierr);
  newsplit->is = is;
  ierr = TSCreate(PetscObjectComm((PetscObject)ts),&newsplit->ts);CHKERRQ(ierr);

  ierr = PetscObjectIncrementTabLevel((PetscObject)newsplit->ts,(PetscObject)ts,1);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)ts,(PetscObject)newsplit->ts);CHKERRQ(ierr);
  ierr = PetscSNPrintf(prefix,sizeof(prefix),"%srhsplit_%s_",((PetscObject)ts)->prefix ? ((PetscObject)ts)->prefix : "",newsplit->splitname);CHKERRQ(ierr);
  ierr = TSSetOptionsPrefix(newsplit->ts,prefix);CHKERRQ(ierr);
  if (!next) ts->tsrhssplit = newsplit;
  else {
    while (next->next) next = next->next;
    next->next = newsplit;
  }
  ts->num_rhs_splits++;
  PetscFunctionReturn(0);
}

/*@C
   TSRHSSplitGetIS - Retrieves the elements for a split as an IS

   Logically Collective on TS

   Input Parameters:
+  ts        - the TS context obtained from TSCreate()
-  splitname - name of this split

   Output Parameters:
-  is        - the index set for part of the solution vector

   Level: intermediate

.seealso: TSRHSSplitSetIS()

@*/
PetscErrorCode TSRHSSplitGetIS(TS ts,const char splitname[],IS *is)
{
  TS_RHSSplitLink isplit;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  *is = NULL;
  /* look up the split */
  ierr = TSRHSSplitGetRHSSplit(ts,splitname,&isplit);CHKERRQ(ierr);
  if (isplit) *is = isplit->is;
  PetscFunctionReturn(0);
}

/*@C
   TSRHSSplitSetRHSFunction - Set the split right-hand-side functions.

   Logically Collective on TS

   Input Parameters:
+  ts        - the TS context obtained from TSCreate()
.  splitname - name of this split
.  r         - vector to hold the residual (or NULL to have it created internally)
.  rhsfunc   - the RHS function evaluation routine
-  ctx       - user-defined context for private data for the split function evaluation routine (may be NULL)

 Calling sequence of fun:
$  rhsfunc(TS ts,PetscReal t,Vec u,Vec f,ctx);

+  t    - time at step/stage being solved
.  u    - state vector
.  f    - function vector
-  ctx  - [optional] user-defined context for matrix evaluation routine (may be NULL)

 Level: beginner

@*/
PetscErrorCode TSRHSSplitSetRHSFunction(TS ts,const char splitname[],Vec r,TSRHSFunction rhsfunc,void *ctx)
{
  TS_RHSSplitLink isplit;
  Vec             subvec,ralloc = NULL;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (r) PetscValidHeaderSpecific(r,VEC_CLASSID,3);

  /* look up the split */
  ierr = TSRHSSplitGetRHSSplit(ts,splitname,&isplit);CHKERRQ(ierr);
  if (!isplit) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"The split %s is not created, check the split name or call TSRHSSplitSetIS() to create one",splitname);

  if (!r && ts->vec_sol) {
    ierr = VecGetSubVector(ts->vec_sol,isplit->is,&subvec);CHKERRQ(ierr);
    ierr = VecDuplicate(subvec,&ralloc);CHKERRQ(ierr);
    r    = ralloc;
    ierr = VecRestoreSubVector(ts->vec_sol,isplit->is,&subvec);CHKERRQ(ierr);
  }
  ierr = TSSetRHSFunction(isplit->ts,r,rhsfunc,ctx);CHKERRQ(ierr);
  ierr = VecDestroy(&ralloc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   TSRHSSplitGetSubTS - Get the sub-TS by split name.

   Logically Collective on TS

   Output Parameters:
+  splitname - the number of the split
-  subts - the array of TS contexts

   Level: advanced

.seealso: TSGetRHSSplitFunction()
@*/
PetscErrorCode TSRHSSplitGetSubTS(TS ts,const char splitname[],TS *subts)
{
  TS_RHSSplitLink isplit;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidPointer(subts,3);
  *subts = NULL;
  /* look up the split */
  ierr = TSRHSSplitGetRHSSplit(ts,splitname,&isplit);CHKERRQ(ierr);
  if (isplit) *subts = isplit->ts;
  PetscFunctionReturn(0);
}

/*@C
   TSRHSSplitGetSubTSs - Get an array of all sub-TS contexts.

   Logically Collective on TS

   Output Parameters:
+  n - the number of splits
-  subksp - the array of TS contexts

   Note:
   After TSRHSSplitGetSubTS() the array of TSs is to be freed by the user with PetscFree()
   (not the TS just the array that contains them).

   Level: advanced

.seealso: TSGetRHSSplitFunction()
@*/
PetscErrorCode TSRHSSplitGetSubTSs(TS ts,PetscInt *n,TS *subts[])
{
  TS_RHSSplitLink ilink = ts->tsrhssplit;
  PetscInt        i = 0;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (subts) {
    ierr = PetscMalloc1(ts->num_rhs_splits,subts);CHKERRQ(ierr);
    while (ilink) {
      (*subts)[i++] = ilink->ts;
      ilink = ilink->next;
    }
  }
  if (n) *n = ts->num_rhs_splits;
  PetscFunctionReturn(0);
}
