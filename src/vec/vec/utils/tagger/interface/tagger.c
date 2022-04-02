#include <petsc/private/vecimpl.h> /*I "petscvec.h" I*/

/*@C
   VecTaggerCreate - create a Vec tagger context.  This object is used to control the tagging/selection of index sets
   based on the values in a vector.  This is used, for example, in adaptive simulations when aspects are selected for
   refinement or coarsening.  The primary intent is that the selected index sets are based purely on the values in the
   vector, though implementations that do not follow this intent are possible.

   Once a VecTagger is created (VecTaggerCreate()), optionally modified by options (VecTaggerSetFromOptions()), and
   set up (VecTaggerSetUp()), it is applied to vectors with VecTaggerComputeIS() to comute the selected index sets.

   In many cases, the selection criteria for an index is whether the corresponding value falls within a collection of
   boxes: for this common case, VecTaggerCreateBoxes() can also be used to determine those boxes.

   Provided implementations support tagging based on a box/interval of values (VECTAGGERABSOLUTE), based on a box of
   values of relative to the range of values present in the vector (VECTAGGERRELATIVE), based on where values fall in
   the cumulative distribution of values in the vector (VECTAGGERCDF), and based on unions (VECTAGGEROR) or
   intersections (VECTAGGERAND) of other criteria.

   Collective

   Input Parameter:
.  comm - communicator on which the vec tagger will operate

   Output Parameter:
.  tagger - new Vec tagger context

   Level: advanced

.seealso: VecTaggerSetBlockSize(), VecTaggerSetFromOptions(), VecTaggerSetUp(), VecTaggerComputeIS(), VecTaggerComputeBoxes(), VecTaggerDestroy()
@*/
PetscErrorCode VecTaggerCreate(MPI_Comm comm,VecTagger *tagger)
{
  VecTagger      b;

  PetscFunctionBegin;
  PetscValidPointer(tagger,2);
  PetscCall(VecTaggerInitializePackage());

  PetscCall(PetscHeaderCreate(b,VEC_TAGGER_CLASSID,"VecTagger","Vec Tagger","Vec",comm,VecTaggerDestroy,VecTaggerView));

  b->blocksize   = 1;
  b->invert      = PETSC_FALSE;
  b->setupcalled = PETSC_FALSE;

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
   of available methods (for instance, absolute, relative, cdf, or, and)

   Notes:
   See "include/petscvec.h" for available methods (for instance)
+    VECTAGGERABSOLUTE - tag based on a box of values
.    VECTAGGERRELATIVE - tag based on a box relative to the range of values present in the vector
.    VECTAGGERCDF      - tag based on a box in the cumulative distribution of values present in the vector
.    VECTAGGEROR       - tag based on the union of a set of VecTagger contexts
-    VECTAGGERAND      - tag based on the intersection of a set of other VecTagger contexts

  Level: advanced

.seealso: VecTaggerType, VecTaggerCreate(), VecTagger
@*/
PetscErrorCode VecTaggerSetType(VecTagger tagger,VecTaggerType type)
{
  PetscBool      match;
  PetscErrorCode (*r)(VecTagger);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tagger,VEC_TAGGER_CLASSID,1);
  PetscValidCharPointer(type,2);

  PetscCall(PetscObjectTypeCompare((PetscObject)tagger,type,&match));
  if (match) PetscFunctionReturn(0);

  PetscCall(PetscFunctionListFind(VecTaggerList,type,&r));
  PetscCheck(r,PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Unable to find requested VecTagger type %s",type);
  /* Destroy the previous private VecTagger context */
  if (tagger->ops->destroy) PetscCall((*(tagger)->ops->destroy)(tagger));
  PetscCall(PetscMemzero(tagger->ops,sizeof(*tagger->ops)));
  PetscCall(PetscObjectChangeTypeName((PetscObject)tagger,type));
  tagger->ops->create = r;
  PetscCall((*r)(tagger));
  PetscFunctionReturn(0);
}

/*@C
  VecTaggerGetType - Gets the VecTagger type name (as a string) from the VecTagger.

  Not Collective

  Input Parameter:
. tagger  - The Vec tagger context

  Output Parameter:
. type - The VecTagger type name

  Level: advanced

.seealso: VecTaggerSetType(), VecTaggerCreate(), VecTaggerSetFromOptions(), VecTagger, VecTaggerType
@*/
PetscErrorCode  VecTaggerGetType(VecTagger tagger, VecTaggerType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tagger, VEC_TAGGER_CLASSID,1);
  PetscValidPointer(type,2);
  PetscCall(VecTaggerRegisterAll());
  *type = ((PetscObject)tagger)->type_name;
  PetscFunctionReturn(0);
}

/*@
   VecTaggerDestroy - destroy a VecTagger context

   Collective

   Input Parameter:
.  tagger - address of tagger

   Level: advanced

.seealso: VecTaggerCreate(), VecTaggerSetType(), VecTagger
@*/
PetscErrorCode VecTaggerDestroy(VecTagger *tagger)
{
  PetscFunctionBegin;
  if (!*tagger) PetscFunctionReturn(0);
  PetscValidHeaderSpecific((*tagger),VEC_TAGGER_CLASSID,1);
  if (--((PetscObject)(*tagger))->refct > 0) {*tagger = NULL; PetscFunctionReturn(0);}
  if ((*tagger)->ops->destroy) PetscCall((*(*tagger)->ops->destroy)(*tagger));
  PetscCall(PetscHeaderDestroy(tagger));
  PetscFunctionReturn(0);
}

/*@
   VecTaggerSetUp - set up a VecTagger context

   Collective

   Input Parameter:
.  tagger - Vec tagger object

   Level: advanced

.seealso: VecTaggerSetFromOptions(), VecTaggerSetType(), VecTagger, VecTaggerCreate(), VecTaggerSetUp()
@*/
PetscErrorCode VecTaggerSetUp(VecTagger tagger)
{
  PetscFunctionBegin;
  if (tagger->setupcalled) PetscFunctionReturn(0);
  if (!((PetscObject)tagger)->type_name) PetscCall(VecTaggerSetType(tagger,VECTAGGERABSOLUTE));
  if (tagger->ops->setup) PetscCall((*tagger->ops->setup)(tagger));
  tagger->setupcalled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*@C
   VecTaggerSetFromOptions - set VecTagger options using the options database

   Logically Collective

   Input Parameter:
.  tagger - vec tagger

   Options Database Keys:
+  -vec_tagger_type       - implementation type, see VecTaggerSetType()
.  -vec_tagger_block_size - set the block size, see VecTaggerSetBlockSize()
-  -vec_tagger_invert     - invert the index set returned by VecTaggerComputeIS()

   Level: advanced

.seealso: VecTagger, VecTaggerCreate(), VecTaggerSetUp()

@*/
PetscErrorCode VecTaggerSetFromOptions(VecTagger tagger)
{
  VecTaggerType  deft;
  char           type[256];
  PetscBool      flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tagger,VEC_TAGGER_CLASSID,1);
  PetscObjectOptionsBegin((PetscObject)tagger);
  deft = ((PetscObject)tagger)->type_name ? ((PetscObject)tagger)->type_name : VECTAGGERABSOLUTE;
  PetscCall(PetscOptionsFList("-vec_tagger_type","VecTagger implementation type","VecTaggerSetType",VecTaggerList,deft,type,256,&flg));
  PetscCall(VecTaggerSetType(tagger,flg ? type : deft));
  PetscCall(PetscOptionsInt("-vec_tagger_block_size","block size of the vectors the tagger operates on","VecTaggerSetBlockSize",tagger->blocksize,&tagger->blocksize,NULL));
  PetscCall(PetscOptionsBool("-vec_tagger_invert","invert the set of indices returned by VecTaggerComputeIS()","VecTaggerSetInvert",tagger->invert,&tagger->invert,NULL));
  if (tagger->ops->setfromoptions) PetscCall((*tagger->ops->setfromoptions)(PetscOptionsObject,tagger));
  PetscOptionsEnd();
  PetscFunctionReturn(0);
}

/*@C
   VecTaggerSetBlockSize - block size of the set of indices returned by VecTaggerComputeIS().  Values greater than one
   are useful when there are multiple criteria for determining which indices to include in the set.  For example,
   consider adaptive mesh refinement in a multiphysics problem, with metrics of solution quality for multiple fields
   measure on each cell.  The size of the vector will be [numCells * numFields]; the VecTagger block size should be
   numFields; VecTaggerComputeIS() will return indices in the range [0,numCells), i.e., one index is given for each
   block of values.

   Note that the block size of the vector does not have to match.

   Note also that the index set created in VecTaggerComputeIS() has block size: it is an index set over the list of
   items that the vector refers to, not to the vector itself.

   Logically Collective

   Input Parameters:
+  tagger - vec tagger
-  blocksize - block size of the criteria used to tagger vectors

   Level: advanced

.seealso: VecTaggerComputeIS(), VecTaggerGetBlockSize(), VecSetBlockSize(), VecGetBlockSize(), VecTagger, VecTaggerCreate()
@*/
PetscErrorCode VecTaggerSetBlockSize(VecTagger tagger, PetscInt blocksize)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tagger,VEC_TAGGER_CLASSID,1);
  PetscValidLogicalCollectiveInt(tagger,blocksize,2);
  tagger->blocksize = blocksize;
  PetscFunctionReturn(0);
}

/*@C
   VecTaggerGetBlockSize - get the block size of the indices created by VecTaggerComputeIS().

   Logically Collective

   Input Parameter:
.  tagger - vec tagger

   Output Parameter:
.  blocksize - block size of the vectors the tagger operates on

   Level: advanced

.seealso: VecTaggerComputeIS(), VecTaggerSetBlockSize(), VecTagger, VecTaggerCreate()
@*/
PetscErrorCode VecTaggerGetBlockSize(VecTagger tagger, PetscInt *blocksize)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tagger,VEC_TAGGER_CLASSID,1);
  PetscValidIntPointer(blocksize,2);
  *blocksize = tagger->blocksize;
  PetscFunctionReturn(0);
}

/*@C
   VecTaggerSetInvert - If the tagged index sets are based on boxes that can be returned by VecTaggerComputeBoxes(),
   then this option inverts values used to compute the IS, i.e., from being in the union of the boxes to being in the
   intersection of their exteriors.

   Logically Collective

   Input Parameters:
+  tagger - vec tagger
-  invert - PETSC_TRUE to invert, PETSC_FALSE to use the indices as is

   Level: advanced

.seealso: VecTaggerComputeIS(), VecTaggerGetInvert(), VecTagger, VecTaggerCreate()
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

   Input Parameter:
.  tagger - vec tagger

   Output Parameter:
.  invert - PETSC_TRUE to invert, PETSC_FALSE to use the indices as is

   Level: advanced

.seealso: VecTaggerComputeIS(), VecTaggerSetInvert(), VecTagger, VecTaggerCreate()
@*/
PetscErrorCode VecTaggerGetInvert(VecTagger tagger, PetscBool *invert)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tagger,VEC_TAGGER_CLASSID,1);
  PetscValidBoolPointer(invert,2);
  *invert = tagger->invert;
  PetscFunctionReturn(0);
}

/*@C
   VecTaggerView - view a VecTagger context

   Collective

   Input Parameters:
+  tagger - vec tagger
-  viewer - viewer to display tagger, for example PETSC_VIEWER_STDOUT_WORLD

   Level: advanced

.seealso: VecTaggerCreate(), VecTagger, VecTaggerCreate()
@*/
PetscErrorCode VecTaggerView(VecTagger tagger,PetscViewer viewer)
{
  PetscBool         iascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tagger,VEC_TAGGER_CLASSID,1);
  if (!viewer) PetscCall(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)tagger),&viewer));
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(tagger,1,viewer,2);
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  if (iascii) {
    PetscCall(PetscObjectPrintClassNamePrefixType((PetscObject)tagger,viewer));
    PetscCall(PetscViewerASCIIPushTab(viewer));
    PetscCall(PetscViewerASCIIPrintf(viewer,"Block size: %" PetscInt_FMT "\n",tagger->blocksize));
    if (tagger->ops->view) PetscCall((*tagger->ops->view)(tagger,viewer));
    if (tagger->invert) PetscCall(PetscViewerASCIIPrintf(viewer,"Inverting ISs.\n"));
    PetscCall(PetscViewerASCIIPopTab(viewer));
  }
  PetscFunctionReturn(0);
}

/*@C
   VecTaggerComputeBoxes - If the tagged index set can be summarized as a list of boxes of values, returns that list, otherwise returns
         in listed PETSC_FALSE

   Collective on VecTagger

   Input Parameters:
+  tagger - the VecTagger context
-  vec - the vec to tag

   Output Parameters:
+  numBoxes - the number of boxes in the tag definition
.  boxes - a newly allocated list of boxes.  This is a flat array of (BlockSize * numBoxes) pairs that the user can free with PetscFree().
-  listed - PETSC_TRUE if a list was created, pass in NULL if not needed

   Notes:
     A value is tagged if it is in any of the boxes, unless the tagger has been inverted (see VecTaggerSetInvert()/VecTaggerGetInvert()), in which case a value is tagged if it is in none of the boxes.

   Level: advanced

.seealso: VecTaggerComputeIS(), VecTagger, VecTaggerCreate()
@*/
PetscErrorCode VecTaggerComputeBoxes(VecTagger tagger,Vec vec,PetscInt *numBoxes,VecTaggerBox **boxes,PetscBool *listed)
{
  PetscInt       vls, tbs;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tagger,VEC_TAGGER_CLASSID,1);
  PetscValidHeaderSpecific(vec,VEC_CLASSID,2);
  PetscValidIntPointer(numBoxes,3);
  PetscValidPointer(boxes,4);
  PetscCall(VecGetLocalSize(vec,&vls));
  PetscCall(VecTaggerGetBlockSize(tagger,&tbs));
  PetscCheck(vls % tbs == 0,PetscObjectComm((PetscObject)tagger),PETSC_ERR_ARG_INCOMP,"vec local size %" PetscInt_FMT " is not a multiple of tagger block size %" PetscInt_FMT,vls,tbs);
  if (tagger->ops->computeboxes) {
    *listed = PETSC_TRUE;
    PetscCall((*tagger->ops->computeboxes) (tagger,vec,numBoxes,boxes,listed));
  }  else *listed = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@C
   VecTaggerComputeIS - Use a VecTagger context to tag a set of indices based on a vector's values

   Collective on VecTagger

   Input Parameters:
+  tagger - the VecTagger context
-  vec - the vec to tag

   Output Parameters:
+  IS - a list of the indices tagged by the tagger, i.e., if the number of local indices will be n / bs, where n is VecGetLocalSize() and bs is VecTaggerGetBlockSize().
-  listed - routine was able to compute the IS, pass in NULL if not needed

   Level: advanced

.seealso: VecTaggerComputeBoxes(), VecTagger, VecTaggerCreate()
@*/
PetscErrorCode VecTaggerComputeIS(VecTagger tagger,Vec vec,IS *is,PetscBool *listed)
{
  PetscInt       vls, tbs;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tagger,VEC_TAGGER_CLASSID,1);
  PetscValidHeaderSpecific(vec,VEC_CLASSID,2);
  PetscValidPointer(is,3);
  PetscCall(VecGetLocalSize(vec,&vls));
  PetscCall(VecTaggerGetBlockSize(tagger,&tbs));
  PetscCheck(vls % tbs == 0,PetscObjectComm((PetscObject)tagger),PETSC_ERR_ARG_INCOMP,"vec local size %" PetscInt_FMT " is not a multiple of tagger block size %" PetscInt_FMT,vls,tbs);
  if (tagger->ops->computeis) {
    PetscCall((*tagger->ops->computeis) (tagger,vec,is,listed));
  } else if (listed) *listed = PETSC_FALSE;
  PetscFunctionReturn(0);
}

PetscErrorCode VecTaggerComputeIS_FromBoxes(VecTagger tagger, Vec vec, IS *is,PetscBool *listed)
{
  PetscInt          numBoxes;
  VecTaggerBox      *boxes;
  PetscInt          numTagged, offset;
  PetscInt          *tagged;
  PetscInt          bs, b, i, j, k, n;
  PetscBool         invert;
  const PetscScalar *vecArray;
  PetscBool         boxlisted;

  PetscFunctionBegin;
  PetscCall(VecTaggerGetBlockSize(tagger,&bs));
  PetscCall(VecTaggerComputeBoxes(tagger,vec,&numBoxes,&boxes,&boxlisted));
  if (!boxlisted) {
    if (listed) *listed = PETSC_FALSE;
    PetscFunctionReturn(0);
  }
  PetscCall(VecGetArrayRead(vec, &vecArray));
  PetscCall(VecGetLocalSize(vec, &n));
  invert = tagger->invert;
  numTagged = 0;
  offset = 0;
  tagged = NULL;
  PetscCheck(n % bs == 0,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"blocksize %" PetscInt_FMT " does not divide vector length %" PetscInt_FMT, bs, n);
  n /= bs;
  for (i = 0; i < 2; i++) {
    if (i) {
      PetscCall(PetscMalloc1(numTagged,&tagged));
    }
    for (j = 0; j < n; j++) {

      for (k = 0; k < numBoxes; k++) {
        for (b = 0; b < bs; b++) {
          PetscScalar  val = vecArray[j * bs + b];
          PetscInt     l = k * bs + b;
          VecTaggerBox box;
          PetscBool    in;

          box = boxes[l];
#if !defined(PETSC_USE_COMPLEX)
          in = (PetscBool) ((box.min <= val) && (val <= box.max));
#else
          in = (PetscBool) ((PetscRealPart     (box.min) <= PetscRealPart     (val)) &&
                            (PetscImaginaryPart(box.min) <= PetscImaginaryPart(val)) &&
                            (PetscRealPart     (val)     <= PetscRealPart     (box.max)) &&
                            (PetscImaginaryPart(val)     <= PetscImaginaryPart(box.max)));
#endif
          if (!in) break;
        }
        if (b == bs) break;
      }
      if ((PetscBool)(k < numBoxes) ^ invert) {
        if (!i) numTagged++;
        else    tagged[offset++] = j;
      }
    }
  }
  PetscCall(VecRestoreArrayRead(vec, &vecArray));
  PetscCall(PetscFree(boxes));
  PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)vec),numTagged,tagged,PETSC_OWN_POINTER,is));
  PetscCall(ISSort(*is));
  if (listed) *listed = PETSC_TRUE;
  PetscFunctionReturn(0);
}
