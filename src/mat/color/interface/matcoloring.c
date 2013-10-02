#include <petsc-private/matimpl.h>      /*I "petscmat.h"  I*/

PetscFunctionList MatColoringList              = 0;
PetscBool         MatColoringRegisterAllCalled = PETSC_FALSE;

#undef __FUNCT__
#define __FUNCT__ "MatColoringRegister"
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

.keywords: matrix, Coloring, register

.seealso: MatColoringRegisterDestroy(), MatColoringRegisterAll()
@*/
PetscErrorCode  MatColoringRegister(const char sname[],PetscErrorCode (*function)(MatColoring))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFunctionListAdd(&MatColoringList,sname,function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatColoringCreate"
PetscErrorCode MatColoringCreate(Mat m,MatColoring *mcptr)
{
  MatColoring    mc;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  *mcptr = 0;

#if !defined(PETSC_USE_DYNAMIC_LIBRARIES)
  ierr = MatInitializePackage();CHKERRQ(ierr);
#endif
  ierr = PetscHeaderCreate(mc,_p_MatColoring, struct _MatColoringOps, MAT_COLORING_CLASSID,"MatColoring","Matrix coloring",
                           "MatColoring",PetscObjectComm((PetscObject)m),MatColoringDestroy, MatColoringView);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)m);CHKERRQ(ierr);
  mc->mat       = m;
  mc->dist      = 1;
  mc->maxcolors = 0; /* no maximum */
  *mcptr        = mc;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "MatColoringDestroy"
PetscErrorCode MatColoringDestroy(MatColoring *mc)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (--((PetscObject)(*mc))->refct > 0) {*mc = 0; PetscFunctionReturn(0);}
  ierr = MatDestroy(&(*mc)->mat);CHKERRQ(ierr);
  ierr = PetscHeaderDestroy(mc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatColoringSetType"
PetscErrorCode MatColoringSetType(MatColoring mc,MatColoringType type)
{
  PetscBool      match;
  PetscErrorCode ierr,(*r)(MatColoring);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mc,MAT_COLORING_CLASSID,1);
  PetscValidCharPointer(type,2);
  ierr = PetscObjectTypeCompare((PetscObject)mc,type,&match);CHKERRQ(ierr);
  if (match) PetscFunctionReturn(0);
  ierr =  PetscFunctionListFind(MatColoringList,type,&r);CHKERRQ(ierr);
  if (!r) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Unable to find requested MatColoring type %s",type);
  if (mc->ops->destroy) {
    ierr             = (*(mc)->ops->destroy)(mc);CHKERRQ(ierr);
    mc->ops->destroy = NULL;
  }
  /* Reinitialize function pointers in SNESOps structure */
  mc->ops->apply            = 0;
  mc->ops->view             = 0;
  mc->ops->setfromoptions   = 0;
  mc->ops->destroy          = 0;

  ierr = PetscObjectChangeTypeName((PetscObject)mc,type);CHKERRQ(ierr);
  ierr = (*r)(mc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatColoringSetFromOptions"
PetscErrorCode MatColoringSetFromOptions(MatColoring mc)
{
  PetscBool      flg;
  MatColoringType deft        = MATCOLORINGSL;
  char           type[256];
  PetscErrorCode ierr;
  PetscInt       dist,maxcolors;

  PetscFunctionBegin;

  PetscValidHeaderSpecific(mc,MAT_COLORING_CLASSID,1);
  ierr = MatColoringGetDistance(mc,&dist);CHKERRQ(ierr);
  ierr = MatColoringGetMaxColors(mc,&maxcolors);CHKERRQ(ierr);
  if (!MatColoringRegisterAllCalled) {ierr = MatColoringRegisterAll();CHKERRQ(ierr);}
  ierr = PetscObjectOptionsBegin((PetscObject)mc);CHKERRQ(ierr);
  if (((PetscObject)mc)->type_name) deft = ((PetscObject)mc)->type_name;
  ierr = PetscOptionsList("-mat_coloring_type","The coloring method used","MatColoringSetType",MatColoringList,deft,type,256,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = MatColoringSetType(mc,type);CHKERRQ(ierr);
  } else if (!((PetscObject)mc)->type_name) {
    ierr = MatColoringSetType(mc,deft);CHKERRQ(ierr);
  }
  ierr = PetscOptionsName("-mat_coloring_view","Print detailed information on solver used","MatColoringView",0);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-mat_coloring_distance","Distance of the coloring","MatColoringSetDistance",dist,&dist,&flg);CHKERRQ(ierr);
  if (flg) {ierr = MatColoringSetDistance(mc,dist);CHKERRQ(ierr);}
  ierr = PetscOptionsInt("-mat_coloring_maxcolors","Maximum colors returned at the end. 1 returns an independent set","SNESSetTolerances",maxcolors,&maxcolors,&flg);CHKERRQ(ierr);
  if (flg) {ierr = MatColoringSetMaxColors(mc,maxcolors);CHKERRQ(ierr);}
  if (mc->ops->setfromoptions) {
    ierr = (*mc->ops->setfromoptions)(mc);CHKERRQ(ierr);
  }
  ierr = PetscObjectProcessOptionsHandlers((PetscObject)mc);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatColoringSetDistance"
PetscErrorCode MatColoringSetDistance(MatColoring mc,PetscInt dist)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mc,MAT_COLORING_CLASSID,1);
  mc->dist = dist;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatColoringGetDistance"
PetscErrorCode MatColoringGetDistance(MatColoring mc,PetscInt *dist)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mc,MAT_COLORING_CLASSID,1);
  if (dist) *dist = mc->dist;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatColoringSetMaxColors"
PetscErrorCode MatColoringSetMaxColors(MatColoring mc,PetscInt maxcolors)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mc,MAT_COLORING_CLASSID,1);
  mc->maxcolors = maxcolors;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatColoringGetMaxColors"
PetscErrorCode MatColoringGetMaxColors(MatColoring mc,PetscInt *maxcolors)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mc,MAT_COLORING_CLASSID,1);
  if (maxcolors) *maxcolors = mc->maxcolors;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatColoringApply"
PetscErrorCode MatColoringApply(MatColoring mc,ISColoring *coloring)
{
  PetscErrorCode    ierr;
  PetscBool         flg;
  PetscViewerFormat format;
  PetscViewer       viewer;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mc,MAT_COLORING_CLASSID,1);
  ierr = (*mc->ops->apply)(mc,coloring);CHKERRQ(ierr);
  /* view */
  ierr = PetscOptionsGetViewer(PetscObjectComm((PetscObject)mc),((PetscObject)mc)->prefix,"-mat_color_view",&viewer,&format,&flg);CHKERRQ(ierr);
  if (flg && !PetscPreLoadingOn) {
    ierr = PetscViewerPushFormat(viewer,format);CHKERRQ(ierr);
    ierr = MatColoringView(mc,viewer);CHKERRQ(ierr);
    ierr = ISColoringView(*coloring,viewer);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatColoringView"
PetscErrorCode MatColoringView(MatColoring mc,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscBool      iascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mc,MAT_COLORING_CLASSID,1);
  if (!viewer) {
    ierr = PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)mc),&viewer);CHKERRQ(ierr);
  }
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(mc,1,viewer,2);

  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscObjectPrintClassNamePrefixType((PetscObject)mc,viewer);CHKERRQ(ierr);
    if (mc->maxcolors > 0) {
      ierr = PetscViewerASCIIPrintf(viewer,"  MatColoring: %d distance, %d max colors\n",mc->dist,mc->maxcolors);CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"  MatColoring: %d distance\n",mc->dist);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}
