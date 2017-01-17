#include <petsc/private/vecimpl.h> /*I "petscvec.h" I*/

/*@C
   VecTaggerCreate - create a Vec tagger context

   Not Collective

   Input Arguments:
.  comm - communicator on which the vec tagger will operate

   Output Arguments:
.  tagger - new Vec tagger context

   Level: advanced

.seealso: VecTaggerSetGraph(), VecTaggerDestroy()
@*/
PetscErrorCode VecTaggerCreate(MPI_Comm comm,VecTagger *tagger)
{
  PetscErrorCode ierr;
  VecTagger      b;

  PetscFunctionBegin;
  PetscValidPointer(tagger,2);
  ierr = VecTaggerInitializePackage();CHKERRQ(ierr);

  ierr = PetscHeaderCreate(b,VEC_TAGGER_CLASSID,"VecTagger","Vec Tagger","Vec",comm,VecTaggerDestroy,VecTaggerView);CHKERRQ(ierr);

  b->invert = PETSC_FALSE;

  *tagger = b;
  PetscFunctionReturn(0);
}

/*@C
   VecTaggerSetType - set the Vec tagger implementation

   Collective on VecTagger

   Input Parameters:
+  tagger - the VecTagger context
-  type - a known method

   Options Database Key:
.  -vec_tagger_type <type> - Sets the method; use -help for a list
   of available methods (for instance, interval, relative, cumulate, or, and)

   Notes:
   See "include/petscvec.h" for available methods (for instance)
+    VECTAGGERINTERVAL   - tag based on an interval of values
.    VECTAGGERRELATIVE   - tag based on an interval relative to the range of values present in the vector
.    VECTAGGERCUMULATIVE - tag based on an interval in the cumulate distribution of values present in the vector
.    VECTAGGEROR         - tag based on the union of a set of VecTagger contexts
.    VECTAGGERAND        - tag based on the intersection of a set of other VecTagger contexts

  Level: advanced

.keywords: VecTagger, set, type

.seealso: VecTaggerType, VecTaggerCreate()
@*/
PetscErrorCode VecTaggerSetType(VecTagger tagger,VecTaggerType type)
{
  PetscErrorCode ierr,(*r)(VecTagger);
  PetscBool      match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tagger,VEC_TAGGER_CLASSID,1);
  PetscValidCharPointer(type,2);

  ierr = PetscObjectTypeCompare((PetscObject)tagger,type,&match);CHKERRQ(ierr);
  if (match) PetscFunctionReturn(0);

  ierr = PetscFunctionListFind(VecTaggerList,type,&r);CHKERRQ(ierr);
  if (!r) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Unable to find requested VecTagger type %s",type);
  /* Destroy the previous private VecTagger context */
  if (tagger->ops->destroy) {
    ierr = (*(tagger)->ops->destroy)(tagger);CHKERRQ(ierr);
  }
  ierr = PetscMemzero(tagger->ops,sizeof(*tagger->ops));CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)tagger,type);CHKERRQ(ierr);
  ierr = (*r)(tagger);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  VecTaggerGetType - Gets the Vec tagger type name (as a string) from the VecTagger.

  Not Collective

  Input Parameter:
. tagger  - The Vec tagger context

  Output Parameter:
. type - The VecTagger type name

  Level: advanced

.keywords: VecTagger, get, type, name
.seealso: VecTaggerSetType(), VecTaggerCreate()
@*/
PetscErrorCode  VecTaggerGetType(VecTagger tagger, VecTaggerType *type)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tagger, VEC_TAGGER_CLASSID,1);
  PetscValidPointer(type,2);
  ierr = VecTaggerRegisterAll();CHKERRQ(ierr);
  *type = ((PetscObject)tagger)->type_name;
  PetscFunctionReturn(0);
}

/*@
   VecTaggerDestroy - destroy tagger

   Collective

   Input Arguments:
.  tagger - address of tagger

   Level: advanced

.seealso: VecTaggerCreate()
@*/
PetscErrorCode VecTaggerDestroy(VecTagger *tagger)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*tagger) PetscFunctionReturn(0);
  PetscValidHeaderSpecific((*tagger),VEC_TAGGER_CLASSID,1);
  if (--((PetscObject)(*tagger))->refct > 0) {*tagger = 0; PetscFunctionReturn(0);}
  if ((*tagger)->ops->destroy) {ierr = (*(*tagger)->ops->destroy)(*tagger);CHKERRQ(ierr);}
  ierr = PetscHeaderDestroy(tagger);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   VecTaggerSetUp - set up tagger

   Collective

   Input Arguments:
.  tagger - Vec tagger object

   Level: advanced

.seealso: VecTaggerSetFromOptions(), VecTaggerSetType()
@*/
PetscErrorCode VecTaggerSetUp(VecTagger tagger)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (tagger->setupcalled) PetscFunctionReturn(0);
  if (!((PetscObject)tagger)->type_name) {ierr = VecTaggerSetType(tagger,VECTAGGERINTERVAL);CHKERRQ(ierr);}
  if (tagger->ops->setup) {ierr = (*tagger->ops->setup)(tagger);CHKERRQ(ierr);}
  tagger->setupcalled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*@C
   VecTaggerSetFromOptions - set VecTagger options using the options database

   Logically Collective

   Input Arguments:
.  tagger - vec tagger

   Options Database Keys:
+  -vec_tagger_type   - implementation type, see VecTaggerSetType()
-  -vec_tagger_invert - invert the index set returned by VecTaggerComputeIS()

   Level: advanced

.keywords: KSP, set, from, options, database
@*/
PetscErrorCode VecTaggerSetFromOptions(VecTagger tagger)
{
  VecTaggerType  deft;
  char           type[256];
  PetscErrorCode ierr;
  PetscBool      flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tagger,VEC_TAGGER_CLASSID,1);
  ierr = PetscObjectOptionsBegin((PetscObject)tagger);CHKERRQ(ierr);
  deft = ((PetscObject)tagger)->type_name ? ((PetscObject)tagger)->type_name : VECTAGGERINTERVAL;;
  ierr = PetscOptionsFList("-vec_tagger_type","VecTagger implementation type","VecTaggerSetType",VecTaggerList,deft,type,256,&flg);CHKERRQ(ierr);
  ierr = VecTaggerSetType(tagger,flg ? type : deft);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-vec_tagger_invert","invert the set of indices returned by VecTaggerComputeIS()","VecTaggerSetInvert",tagger->invert,&tagger->invert,NULL);CHKERRQ(ierr);
  if (tagger->ops->setfromoptions) {ierr = (*tagger->ops->setfromoptions)(PetscOptionsObject,tagger);CHKERRQ(ierr);}
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   VecTaggerSetInvert - invert the set of indices returned by VecTaggerComputeIS()

   Logically Collective

   Input Arguments:
+  tagger - vec tagger
-  flg - PETSC_TRUE to invert, PETSC_FALSE to use the indices as is

   Level: advanced

.seealso: VecTaggerComputeIS(), VecTaggerGetInvert()
@*/
PetscErrorCode VecTaggerSetInvert(VecTagger tagger, PetscBool invert)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tagger,VEC_TAGGER_CLASSID,1);
  PetscValidLogicalCollectiveBool(tagger,invert,2);
  tagger->invert = invert;
  PetscFunctionReturn(0);
}

/*@C
   VecTaggerGetInvert - get whether the set of indices returned by VecTaggerComputeIS() are inverted

   Logically Collective

   Input Arguments:
+  tagger - vec tagger
-  flg - PETSC_TRUE to invert, PETSC_FALSE to use the indices as is

   Level: advanced

.seealso: VecTaggerComputeIS(), VecTaggerSetInvert()
@*/
PetscErrorCode VecTaggerGetInvert(VecTagger tagger, PetscBool *invert)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tagger,VEC_TAGGER_CLASSID,1);
  PetscValidPointer(invert,2);
  *invert = tagger->invert;
  PetscFunctionReturn(0);
}

/*@C
   VecTaggerView - view a vec tagger

   Collective

   Input Arguments:
+  tagger - vec tagger
-  viewer - viewer to display tagger, for example PETSC_VIEWER_STDOUT_WORLD

   Level: advanced

.seealso: VecTaggerCreate()
@*/
PetscErrorCode VecTaggerView(VecTagger tagger,PetscViewer viewer)
{
  PetscErrorCode    ierr;
  PetscBool         iascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tagger,VEC_TAGGER_CLASSID,1);
  if (!viewer) {ierr = PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)tagger),&viewer);CHKERRQ(ierr);}
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(tagger,1,viewer,2);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscObjectPrintClassNamePrefixType((PetscObject)tagger,viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    if (tagger->ops->view) {ierr = (*tagger->ops->view)(tagger,viewer);CHKERRQ(ierr);}
    if (tagger->invert) {ierr = PetscViewerASCIIPrintf(viewer,"Inverting ISs.");CHKERRQ(ierr);}
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@C
   VecTaggerComputeIntervals - If the tag can be summarized as a list of intervals, returns that list

   Collective on VecTagger

   Input Aguments:
+  tagger - the VecTagger context
-  vec - the vec to tag

   Output Arguments:
+  numIntervals - the number of intervals in the tag definition
-  intervals - a newly allocated list of intervals, given by (min,max) pairs.  It is up to the user to free this list with PetscFree().

   Notes:
.  A value is tagged if it is in any of the intervals, unles the tagger has been inverted (see VecTaggerSetInvert()/VecTaggerGetInvert()), in which case a value is tagged if it is in none of the intervals.

.seealso: VecTaggerComputeIS()
@*/
PetscErrorCode VecTaggerComputeIntervals(VecTagger tagger,Vec vec,PetscInt *numIntervals,PetscScalar (**intervals)[2])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tagger,VEC_TAGGER_CLASSID,1);
  PetscValidHeaderSpecific(vec,VEC_CLASSID,2);
  PetscValidIntPointer(numIntervals,3);
  PetscValidPointer(intervals,4);
  if (tagger->ops->computeintervals) {ierr = (*tagger->ops->computeintervals) (tagger,vec,numIntervals,intervals);CHKERRQ(ierr);}
  else {
    const char *type;
    ierr = PetscObjectGetType ((PetscObject)tagger,&type);CHKERRQ(ierr);
    SETERRQ1(PetscObjectComm((PetscObject)tagger),PETSC_ERR_SUP,"VecTagger type %s does not compute value intervals",type);
  }
  PetscFunctionReturn(0);
}

/*@C
   VecTaggerComputeIS - Use a VecTagger context to tag a set of indices based on a vector's values

   Collective on VecTagger

   Input Aguments:
+  tagger - the VecTagger context
-  vec - the vec to tag

   Output Arguments:
.  IS - a list of the local indices tagged by the tagger

.seealso: VecTaggerComputeIntervals()
@*/
PetscErrorCode VecTaggerComputeIS(VecTagger tagger,Vec vec,IS *is)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tagger,VEC_TAGGER_CLASSID,1);
  PetscValidHeaderSpecific(vec,VEC_CLASSID,2);
  PetscValidPointer(is,3);
  if (tagger->ops->computeis) {ierr = (*tagger->ops->computeis) (tagger,vec,is);CHKERRQ(ierr);}
  else {
    SETERRQ(PetscObjectComm((PetscObject)tagger),PETSC_ERR_SUP,"VecTagger type does not compute ISs");
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecTaggerComputeIS_FromIntervals(VecTagger tagger, Vec vec, IS *is)
{ PetscInt       numIntervals;
  PetscScalar    (*intervals)[2];
  PetscInt       numTagged, offset;
  PetscInt       *tagged;
  PetscInt       i, j, k, n;
  PetscBool      invert;
  const PetscScalar *vecArray;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecTaggerComputeIntervals(tagger,vec,&numIntervals,&intervals);CHKERRQ(ierr);
  ierr = VecGetArrayRead (vec, &vecArray);CHKERRQ(ierr);
  ierr = VecGetLocalSize (vec, &n);CHKERRQ(ierr);
  invert = tagger->invert;
  numTagged = 0;
  offset = 0;
  tagged = NULL;
  for (i = 0; i < 2; i++) {
    if (i) {
      ierr = PetscMalloc1(numTagged,&tagged);CHKERRQ(ierr);
    }
    for (j = 0; j < n; j++) {
      PetscScalar val = vecArray[j];

      for (k = 0; k < numIntervals; k++) {
        PetscScalar interval[2] = {intervals[k][0], intervals[k][1]};
        PetscBool   in;
#if !defined(PETSC_USE_COMPLEX)
        in = (interval[0] <= val) && (val <= interval[1]);
#else
        in = (PetscRealPart   (interval[0]) <= PetscRealPart   (val)        )&&
             (PetscComplexPart(interval[0]) <= PetscComplexPart(val)        )&&
             (PetscRealPart   (val)         <= PetscRealPart   (interval[1]))&&
             (PetscComplexPart(val)         <= PetscComplexPart(interval[1]));
#endif
        if (in) break;
      }
      if ((k < numIntervals) ^ invert) {
        if (!i) numTagged++;
        else    tagged[offset++] = j;
      }
    }
  }
  ierr = VecRestoreArrayRead (vec, &vecArray);CHKERRQ(ierr);
  ierr = ISCreateGeneral (PetscObjectComm((PetscObject)vec),numTagged,tagged,PETSC_OWN_POINTER,is);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
