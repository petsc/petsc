/*$Id: pf.c,v 1.1 2000/01/22 23:00:14 bsmith Exp bsmith $*/
/*
    The PF mathematical functions interface routines, callable by users.
*/
#include "src/pf/pfimpl.h"            /*I "pf.h" I*/


#undef __FUNC__  
#define __FUNC__ "PFDestroy"
/*@C
   PFDestroy - Destroys PF context that was created with PFCreate().

   Collective on PF

   Input Parameter:
.  pf - the preconditioner context

   Level: developer

.keywords: PF, destroy

.seealso: PFCreate(), PFSetUp()
@*/
int PFDestroy(PF pf)
{
  int ierr = 0;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pf,PF_COOKIE);
  if (--pf->refct > 0) PetscFunctionReturn(0);

  /* if memory was published with AMS then destroy it */
  ierr = PetscObjectDepublish(pf);CHKERRQ(ierr);

  if (pf->ops->destroy) {ierr =  (*pf->ops->destroy)(pf);CHKERRQ(ierr);}
  if (pf->nullsp) {ierr = PFNullSpaceDestroy(pf->nullsp);CHKERRQ(ierr);}
  PLogObjectDestroy(pf);
  PetscHeaderDestroy(pf);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PFPublish_Petsc"
static int PFPublish_Petsc(PetscObject obj)
{
#if defined(PETSC_HAVE_AMS)
  PF          v = (PF) obj;
  int         ierr;
#endif

  PetscFunctionBegin;

#if defined(PETSC_HAVE_AMS)
  /* if it is already published then return */
  if (v->amem >=0) PetscFunctionReturn(0);

  ierr = PetscObjectPublishBaseBegin(obj);CHKERRQ(ierr);
  ierr = PetscObjectPublishBaseEnd(obj);CHKERRQ(ierr);
#endif

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PFCreate"
/*@C
   PFCreate - Creates a preconditioner context.

   Collective on MPI_Comm

   Input Parameter:
.  comm - MPI communicator 

   Output Parameter:
.  pf - location to put the preconditioner context

   Notes:
   The default preconditioner on one processor is PFILU with 0 fill on more 
   then one it is PFBJACOBI with ILU() on each processor.

   Level: developer

.keywords: PF, create, context

.seealso: PFSetUp(), PFApply(), PFDestroy()
@*/
int PFCreate(MPI_Comm comm,PF *newpf)
{
  PF     pf;

  PetscFunctionBegin;
  *newpf          = 0;

  PetscHeaderCreate(pf,_p_PF,struct _PFOps,PF_COOKIE,-1,"PF",comm,PFDestroy,PFView);
  PLogObjectCreate(pf);
  pf->bops->publish      = PFPublish_Petsc;
  pf->vec                = 0;
  pf->mat                = 0;
  pf->setupfalled        = 0;
  pf->nullsp             = 0;
  pf->data               = 0;

  pf->ops->destroy             = 0;
  pf->ops->apply               = 0;
  pf->ops->applytranspose      = 0;
  pf->ops->applyBA             = 0;
  pf->ops->applyBAtranspose    = 0;
  pf->ops->applyrichardson     = 0;
  pf->ops->view                = 0;
  pf->ops->getfactoredmatrix   = 0;
  pf->ops->applysymmetricright = 0;
  pf->ops->applysymmetricleft  = 0;
  pf->ops->setuponblocks       = 0;

  pf->modifysubmatrices   = 0;
  pf->modifysubmatricesP  = 0;
  *newpf                  = pf;
  PetscPublishAll(pf);
  PetscFunctionReturn(0);

}

/* -------------------------------------------------------------------------------*/

#undef __FUNC__  
#define __FUNC__ "PFApply"
/*@
   PFApply - Applies the preconditioner to a vector.

   Collective on PF and Vec

   Input Parameters:
+  pf - the preconditioner context
-  x - input vector

   Output Parameter:
.  y - output vector

   Level: developer

.keywords: PF, apply

.seealso: PFApplyTranspose(), PFApplyBAorAB()
@*/
int PFApply(PF pf,Vec x,Vec y)
{
  int        ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pf,PF_COOKIE);
  PetscValidHeaderSpecific(x,VEC_COOKIE);
  PetscValidHeaderSpecific(y,VEC_COOKIE);
  if (x == y) SETERRQ(PETSC_ERR_ARG_IDN,0,"x and y must be different vectors");

  if (pf->setupfalled < 2) {
    ierr = PFSetUp(pf);CHKERRQ(ierr);
  }

  PLogEventBegin(PF_Apply,pf,x,y,0);
  ierr = (*pf->ops->apply)(pf,x,y);CHKERRQ(ierr);

  /* Remove null space from preconditioned vector y */
  if (pf->nullsp) {
    ierr = PFNullSpaceRemove(pf->nullsp,y);CHKERRQ(ierr);
  }

  PLogEventEnd(PF_Apply,pf,x,y,0);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PFView"
/*@ 
   PFView - Prints the PF data structure.

   Collective on PF unless Viewer is VIEWER_STDOUT_SELF  

   Input Parameters:
+  PF - the PF context
-  viewer - optional visualization context

   Note:
   The available visualization contexts include
+     VIEWER_STDOUT_SELF - standard output (default)
-     VIEWER_STDOUT_WORLD - synchronized standard
         output where only the first processor opens
         the file.  All other processors send their 
         data to the first processor to print. 

   The user can open an alternative visualization contexts with
   ViewerASCIIOpen() (output to a specified file).

   Level: developer

.keywords: PF, view

.seealso: KSPView(), ViewerASCIIOpen()
@*/
int PFView(PF pf,Viewer viewer)
{
  PFType      cstr;
  int         fmt,ierr;
  PetscTruth  mat_exists,isascii,isstring;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pf,PF_COOKIE);
  if (!viewer) viewer = VIEWER_STDOUT_SELF;
  PetscValidHeaderSpecific(viewer,VIEWER_COOKIE); 
  PetscCheckSameComm(pf,viewer);

  ierr = PetscTypeCompare((PetscObject)viewer,ASCII_VIEWER,&isascii);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,STRING_VIEWER,&isstring);CHKERRQ(ierr);
  if (isascii) {
    ierr = ViewerGetFormat(viewer,&fmt);CHKERRQ(ierr);
    ierr = ViewerASCIIPrintf(viewer,"PF Object:\n");CHKERRQ(ierr);
    ierr = PFGetType(pf,&cstr);CHKERRQ(ierr);
    if (cstr) {
      ierr = ViewerASCIIPrintf(viewer,"  type: %s\n",cstr);CHKERRQ(ierr);
    } else {
      ierr = ViewerASCIIPrintf(viewer,"  type: not yet set\n");CHKERRQ(ierr);
    }
    if (pf->ops->view) {
      ierr = ViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = (*pf->ops->view)(pf,viewer);CHKERRQ(ierr);
      ierr = ViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    }
    ierr = PetscObjectExists((PetscObject)pf->mat,&mat_exists);CHKERRQ(ierr);
    if (mat_exists) {
      ierr = ViewerPushFormat(viewer,VIEWER_FORMAT_ASCII_INFO,0);CHKERRQ(ierr);
      if (pf->pmat == pf->mat) {
        ierr = ViewerASCIIPrintf(viewer,"  linear system matrix = precond matrix:\n");CHKERRQ(ierr);
        ierr = ViewerASCIIPushTab(viewer);CHKERRQ(ierr);
        ierr = MatView(pf->mat,viewer);CHKERRQ(ierr);
        ierr = ViewerASCIIPopTab(viewer);CHKERRQ(ierr);
      } else {
        ierr = PetscObjectExists((PetscObject)pf->pmat,&mat_exists);CHKERRQ(ierr);
        if (mat_exists) {
          ierr = ViewerASCIIPrintf(viewer,"  linear system matrix followed by preconditioner matrix:\n");CHKERRQ(ierr);
        } else {
          ierr = ViewerASCIIPrintf(viewer,"  linear system matrix:\n");CHKERRQ(ierr);
        }
        ierr = ViewerASCIIPushTab(viewer);CHKERRQ(ierr);
        ierr = MatView(pf->mat,viewer);CHKERRQ(ierr);
        if (mat_exists) {ierr = MatView(pf->pmat,viewer);CHKERRQ(ierr);}
        ierr = ViewerASCIIPopTab(viewer);CHKERRQ(ierr);
      }
      ierr = ViewerPopFormat(viewer);CHKERRQ(ierr);
    }
  } else if (isstring) {
    ierr = PFGetType(pf,&cstr);CHKERRQ(ierr);
    ierr = ViewerStringSPrintf(viewer," %-7.7s",cstr);CHKERRQ(ierr);
    if (pf->ops->view) {ierr = (*pf->ops->view)(pf,viewer);CHKERRQ(ierr);}
  } else {
    SETERRQ1(1,1,"Viewer type %s not supported by PF",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

/*MC
   PFRegisterDynamic - Adds a method to the preconditioner package.

   Synopsis:
   PFRegisterDynamic(char *name_solver,char *path,char *name_create,int (*routine_create)(PF))

   Not collective

   Input Parameters:
+  name_solver - name of a new user-defined solver
.  path - path (either absolute or relative) the library containing this solver
.  name_create - name of routine to create method context
-  routine_create - routine to create method context

   Notes:
   PFRegisterDynamic() may be called multiple times to add several user-defined preconditioners.

   If dynamic libraries are used, then the fourth input argument (routine_create)
   is ignored.

   Sample usage:
.vb
   PFRegisterDynamic("my_solver","/home/username/my_lib/lib/libO/solaris/mylib",
              "MySolverCreate",MySolverCreate);
.ve

   Then, your solver can be chosen with the procedural interface via
$     PFSetType(pf,"my_solver")
   or at runtime via the option
$     -pf_type my_solver

   Level: advanced

   ${PETSC_ARCH}, ${PETSC_DIR}, ${PETSC_LDIR}, ${BOPT}, or ${any environmental variable}
 occuring in pathname will be replaced with appropriate values.

.keywords: PF, register

.seealso: PFRegisterAll(), PFRegisterDestroy(), PFRegister()
M*/

#undef __FUNC__  
#define __FUNC__ "PFRegister"
int PFRegister(char *sname,char *path,char *name,int (*function)(PF))
{
  int  ierr;
  char fullname[256];

  PetscFunctionBegin;

  ierr = FListConcat(path,name,fullname);CHKERRQ(ierr);
  ierr = FListAdd(&PFList,sname,fullname,(int (*)(void*))function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
