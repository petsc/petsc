#include <petsc/private/matimpl.h>      /*I "petscmat.h"  I*/

PetscFunctionList MatColoringList              = NULL;
PetscBool         MatColoringRegisterAllCalled = PETSC_FALSE;
const char *const MatColoringWeightTypes[] = {"RANDOM","LEXICAL","LF","SL","MatColoringWeightType","MAT_COLORING_WEIGHT_",NULL};

/*@C
   MatColoringRegister - Adds a new sparse matrix coloring to the  matrix package.

   Not Collective

   Input Parameters:
+  sname - name of Coloring (for example MATCOLORINGSL)
-  function - function pointer that creates the coloring

   Level: developer

   Sample usage:
.vb
   MatColoringRegister("my_color",MyColor);
.ve

   Then, your partitioner can be chosen with the procedural interface via
$     MatColoringSetType(part,"my_color")
   or at runtime via the option
$     -mat_coloring_type my_color

.seealso: `MatColoringRegisterDestroy()`, `MatColoringRegisterAll()`
@*/
PetscErrorCode  MatColoringRegister(const char sname[],PetscErrorCode (*function)(MatColoring))
{
  PetscFunctionBegin;
  PetscCall(MatInitializePackage());
  PetscCall(PetscFunctionListAdd(&MatColoringList,sname,function));
  PetscFunctionReturn(0);
}

/*@
   MatColoringCreate - Creates a matrix coloring context.

   Collective on MatColoring

   Input Parameters:
.  comm - MPI communicator

   Output Parameter:
.  mcptr - the new MatColoring context

   Options Database Keys:
+   -mat_coloring_type - the type of coloring algorithm used. See MatColoringType.
.   -mat_coloring_maxcolors - the maximum number of relevant colors, all nodes not in a color are in maxcolors+1
.   -mat_coloring_distance - compute a distance 1,2,... coloring.
.   -mat_coloring_view - print information about the coloring and the produced index sets
.   -mat_coloring_test - debugging option that prints all coloring incompatibilities
-   -mat_is_coloring_test - debugging option that throws an error if MatColoringApply() generates an incorrect iscoloring

   Level: beginner

   Notes:
    A distance one coloring is useful, for example, multi-color SOR. A distance two coloring is for the finite difference computation of Jacobians
          (see MatFDColoringCreate()).

       Coloring of matrices can be computed directly from the sparse matrix nonzero structure via the MatColoring object or from the mesh from which the
       matrix comes from with DMCreateColoring(). In general using the mesh produces a more optimal coloring (fewer colors).

          Some coloring types only support distance two colorings

.seealso: `MatColoring`, `MatColoringApply()`, `MatFDColoringCreate()`, `DMCreateColoring()`, `MatColoringType`
@*/
PetscErrorCode MatColoringCreate(Mat m,MatColoring *mcptr)
{
  MatColoring    mc;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,MAT_CLASSID,1);
  PetscValidPointer(mcptr,2);
  *mcptr = NULL;

  PetscCall(MatInitializePackage());
  PetscCall(PetscHeaderCreate(mc, MAT_COLORING_CLASSID,"MatColoring","Matrix coloring", "MatColoring",PetscObjectComm((PetscObject)m),MatColoringDestroy, MatColoringView));
  PetscCall(PetscObjectReference((PetscObject)m));
  mc->mat       = m;
  mc->dist      = 2; /* default to Jacobian computation case */
  mc->maxcolors = IS_COLORING_MAX;
  *mcptr        = mc;
  mc->valid     = PETSC_FALSE;
  mc->weight_type = MAT_COLORING_WEIGHT_RANDOM;
  mc->user_weights = NULL;
  mc->user_lperm = NULL;
  PetscFunctionReturn(0);
}

/*@
   MatColoringDestroy - Destroys the matrix coloring context

   Collective on MatColoring

   Input Parameter:
.  mc - the MatColoring context

   Level: beginner

.seealso: `MatColoringCreate()`, `MatColoringApply()`
@*/
PetscErrorCode MatColoringDestroy(MatColoring *mc)
{
  PetscFunctionBegin;
  if (--((PetscObject)(*mc))->refct > 0) {*mc = NULL; PetscFunctionReturn(0);}
  PetscCall(MatDestroy(&(*mc)->mat));
  if ((*mc)->ops->destroy) PetscCall((*((*mc)->ops->destroy))(*mc));
  if ((*mc)->user_weights) PetscCall(PetscFree((*mc)->user_weights));
  if ((*mc)->user_lperm) PetscCall(PetscFree((*mc)->user_lperm));
  PetscCall(PetscHeaderDestroy(mc));
  PetscFunctionReturn(0);
}

/*@C
   MatColoringSetType - Sets the type of coloring algorithm used

   Collective on MatColoring

   Input Parameters:
+  mc - the MatColoring context
-  type - the type of coloring

   Level: beginner

   Notes:
    Possible types include the sequential types MATCOLORINGLF,
   MATCOLORINGSL, and MATCOLORINGID from the MINPACK package as well
   as a parallel MATCOLORINGMIS algorithm.

.seealso: `MatColoringCreate()`, `MatColoringApply()`
@*/
PetscErrorCode MatColoringSetType(MatColoring mc,MatColoringType type)
{
  PetscBool      match;
  PetscErrorCode (*r)(MatColoring);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mc,MAT_COLORING_CLASSID,1);
  PetscValidCharPointer(type,2);
  PetscCall(PetscObjectTypeCompare((PetscObject)mc,type,&match));
  if (match) PetscFunctionReturn(0);
  PetscCall(PetscFunctionListFind(MatColoringList,type,&r));
  PetscCheck(r,PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Unable to find requested MatColoring type %s",type);
  if (mc->ops->destroy) {
    PetscCall((*(mc)->ops->destroy)(mc));
    mc->ops->destroy = NULL;
  }
  mc->ops->apply            = NULL;
  mc->ops->view             = NULL;
  mc->ops->setfromoptions   = NULL;
  mc->ops->destroy          = NULL;

  PetscCall(PetscObjectChangeTypeName((PetscObject)mc,type));
  PetscCall((*r)(mc));
  PetscFunctionReturn(0);
}

/*@
   MatColoringSetFromOptions - Sets MatColoring options from user parameters

   Collective on MatColoring

   Input Parameter:
.  mc - MatColoring context

   Options Database Keys:
+   -mat_coloring_type - the type of coloring algorithm used. See MatColoringType.
.   -mat_coloring_maxcolors - the maximum number of relevant colors, all nodes not in a color are in maxcolors+1
.   -mat_coloring_distance - compute a distance 1,2,... coloring.
.   -mat_coloring_view - print information about the coloring and the produced index sets
.   -snes_fd_color - instruct SNES to using coloring and then MatFDColoring to compute the Jacobians
-   -snes_fd_color_use_mat - instruct SNES to color the matrix directly instead of the DM from which the matrix comes (the default)

   Level: beginner

.seealso: `MatColoring`, `MatColoringApply()`, `MatColoringSetDistance()`, `SNESComputeJacobianDefaultColor()`, `MatColoringType`
@*/
PetscErrorCode MatColoringSetFromOptions(MatColoring mc)
{
  PetscBool      flg;
  MatColoringType deft = MATCOLORINGSL;
  char           type[256];
  PetscInt       dist,maxcolors;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mc,MAT_COLORING_CLASSID,1);
  PetscCall(MatColoringGetDistance(mc,&dist));
  if (dist == 2) deft = MATCOLORINGSL;
  else           deft = MATCOLORINGGREEDY;
  PetscCall(MatColoringGetMaxColors(mc,&maxcolors));
  PetscCall(MatColoringRegisterAll());
  PetscObjectOptionsBegin((PetscObject)mc);
  if (((PetscObject)mc)->type_name) deft = ((PetscObject)mc)->type_name;
  PetscCall(PetscOptionsFList("-mat_coloring_type","The coloring method used","MatColoringSetType",MatColoringList,deft,type,256,&flg));
  if (flg) {
    PetscCall(MatColoringSetType(mc,type));
  } else if (!((PetscObject)mc)->type_name) {
    PetscCall(MatColoringSetType(mc,deft));
  }
  PetscCall(PetscOptionsInt("-mat_coloring_distance","Distance of the coloring","MatColoringSetDistance",dist,&dist,&flg));
  if (flg) PetscCall(MatColoringSetDistance(mc,dist));
  PetscCall(PetscOptionsInt("-mat_coloring_maxcolors","Maximum colors returned at the end. 1 returns an independent set","MatColoringSetMaxColors",maxcolors,&maxcolors,&flg));
  if (flg) PetscCall(MatColoringSetMaxColors(mc,maxcolors));
  if (mc->ops->setfromoptions) {
    PetscCall((*mc->ops->setfromoptions)(PetscOptionsObject,mc));
  }
  PetscCall(PetscOptionsBool("-mat_coloring_test","Check that a valid coloring has been produced","",mc->valid,&mc->valid,NULL));
  PetscCall(PetscOptionsBool("-mat_is_coloring_test","Check that a valid iscoloring has been produced","",mc->valid_iscoloring,&mc->valid_iscoloring,NULL));
  PetscCall(PetscOptionsEnum("-mat_coloring_weight_type","Sets the type of vertex weighting used","MatColoringSetWeightType",MatColoringWeightTypes,(PetscEnum)mc->weight_type,(PetscEnum*)&mc->weight_type,NULL));
  PetscCall(PetscObjectProcessOptionsHandlers(PetscOptionsObject,(PetscObject)mc));
  PetscOptionsEnd();
  PetscFunctionReturn(0);
}

/*@
   MatColoringSetDistance - Sets the distance of the coloring

   Logically Collective on MatColoring

   Input Parameters:
+  mc - the MatColoring context
-  dist - the distance the coloring should compute

   Level: beginner

   Notes:
    The distance of the coloring denotes the minimum number
   of edges in the graph induced by the matrix any two vertices
   of the same color are.  Distance-1 colorings are the classical
   coloring, where no two vertices of the same color are adjacent.
   distance-2 colorings are useful for the computation of Jacobians.

.seealso: `MatColoringGetDistance()`, `MatColoringApply()`
@*/
PetscErrorCode MatColoringSetDistance(MatColoring mc,PetscInt dist)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mc,MAT_COLORING_CLASSID,1);
  mc->dist = dist;
  PetscFunctionReturn(0);
}

/*@
   MatColoringGetDistance - Gets the distance of the coloring

   Logically Collective on MatColoring

   Input Parameter:
.  mc - the MatColoring context

   Output Parameter:
.  dist - the current distance being used for the coloring.

   Level: beginner

.seealso: `MatColoringSetDistance()`, `MatColoringApply()`
@*/
PetscErrorCode MatColoringGetDistance(MatColoring mc,PetscInt *dist)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mc,MAT_COLORING_CLASSID,1);
  if (dist) *dist = mc->dist;
  PetscFunctionReturn(0);
}

/*@
   MatColoringSetMaxColors - Sets the maximum number of colors

   Logically Collective on MatColoring

   Input Parameters:
+  mc - the MatColoring context
-  maxcolors - the maximum number of colors to produce

   Level: beginner

   Notes:
    This may be used to compute a certain number of
   independent sets from the graph.  For instance, while using
   MATCOLORINGMIS and maxcolors = 1, one gets out an MIS.  Vertices
   not in a color are set to have color maxcolors+1, which is not
   a valid color as they may be adjacent.

.seealso: `MatColoringGetMaxColors()`, `MatColoringApply()`
@*/
PetscErrorCode MatColoringSetMaxColors(MatColoring mc,PetscInt maxcolors)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mc,MAT_COLORING_CLASSID,1);
  mc->maxcolors = maxcolors;
  PetscFunctionReturn(0);
}

/*@
   MatColoringGetMaxColors - Gets the maximum number of colors

   Logically Collective on MatColoring

   Input Parameter:
.  mc - the MatColoring context

   Output Parameter:
.  maxcolors - the current maximum number of colors to produce

   Level: beginner

.seealso: `MatColoringSetMaxColors()`, `MatColoringApply()`
@*/
PetscErrorCode MatColoringGetMaxColors(MatColoring mc,PetscInt *maxcolors)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mc,MAT_COLORING_CLASSID,1);
  if (maxcolors) *maxcolors = mc->maxcolors;
  PetscFunctionReturn(0);
}

/*@
   MatColoringApply - Apply the coloring to the matrix, producing index
   sets corresponding to a number of independent sets in the induced
   graph.

   Collective on MatColoring

   Input Parameters:
.  mc - the MatColoring context

   Output Parameter:
.  coloring - the ISColoring instance containing the coloring

   Level: beginner

.seealso: `MatColoring`, `MatColoringCreate()`
@*/
PetscErrorCode MatColoringApply(MatColoring mc,ISColoring *coloring)
{
  PetscBool         flg;
  PetscViewerFormat format;
  PetscViewer       viewer;
  PetscInt          nc,ncolors;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mc,MAT_COLORING_CLASSID,1);
  PetscValidPointer(coloring,2);
  PetscCall(PetscLogEventBegin(MATCOLORING_Apply,mc,0,0,0));
  PetscCall((*mc->ops->apply)(mc,coloring));
  PetscCall(PetscLogEventEnd(MATCOLORING_Apply,mc,0,0,0));

  /* valid */
  if (mc->valid) {
    PetscCall(MatColoringTest(mc,*coloring));
  }
  if (mc->valid_iscoloring) {
    PetscCall(MatISColoringTest(mc->mat,*coloring));
  }

  /* view */
  PetscCall(PetscOptionsGetViewer(PetscObjectComm((PetscObject)mc),((PetscObject)mc)->options,((PetscObject)mc)->prefix,"-mat_coloring_view",&viewer,&format,&flg));
  if (flg && !PetscPreLoadingOn) {
    PetscCall(PetscViewerPushFormat(viewer,format));
    PetscCall(MatColoringView(mc,viewer));
    PetscCall(MatGetSize(mc->mat,NULL,&nc));
    PetscCall(ISColoringGetIS(*coloring,PETSC_USE_POINTER,&ncolors,NULL));
    PetscCall(PetscViewerASCIIPrintf(viewer,"  Number of colors %" PetscInt_FMT "\n",ncolors));
    PetscCall(PetscViewerASCIIPrintf(viewer,"  Number of total columns %" PetscInt_FMT "\n",nc));
    if (nc <= 1000) PetscCall(ISColoringView(*coloring,viewer));
    PetscCall(PetscViewerPopFormat(viewer));
    PetscCall(PetscViewerDestroy(&viewer));
  }
  PetscFunctionReturn(0);
}

/*@
   MatColoringView - Output details about the MatColoring.

   Collective on MatColoring

   Input Parameters:
-  mc - the MatColoring context
+  viewer - the Viewer context

   Level: beginner

.seealso: `MatColoring`, `MatColoringApply()`
@*/
PetscErrorCode MatColoringView(MatColoring mc,PetscViewer viewer)
{
  PetscBool      iascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mc,MAT_COLORING_CLASSID,1);
  if (!viewer) {
    PetscCall(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)mc),&viewer));
  }
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(mc,1,viewer,2);

  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  if (iascii) {
    PetscCall(PetscObjectPrintClassNamePrefixType((PetscObject)mc,viewer));
    PetscCall(PetscViewerASCIIPrintf(viewer,"  Weight type: %s\n",MatColoringWeightTypes[mc->weight_type]));
    if (mc->maxcolors > 0) {
      PetscCall(PetscViewerASCIIPrintf(viewer,"  Distance %" PetscInt_FMT ", Max. Colors %" PetscInt_FMT "\n",mc->dist,mc->maxcolors));
    } else {
      PetscCall(PetscViewerASCIIPrintf(viewer,"  Distance %" PetscInt_FMT "\n",mc->dist));
    }
  }
  PetscFunctionReturn(0);
}

/*@
   MatColoringSetWeightType - Set the type of weight computation used.

   Logically collective on MatColoring

   Input Parameters:
-  mc - the MatColoring context
+  wt - the weight type

   Level: beginner

.seealso: `MatColoring`, `MatColoringWeightType`
@*/
PetscErrorCode MatColoringSetWeightType(MatColoring mc,MatColoringWeightType wt)
{
  PetscFunctionBegin;
  mc->weight_type = wt;
  PetscFunctionReturn(0);

}
