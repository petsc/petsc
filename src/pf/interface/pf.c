/*$Id: pf.c,v 1.2 2000/01/24 04:08:15 bsmith Exp bsmith $*/
/*
    The PF mathematical functions interface routines, callable by users.
*/
#include "src/pf/pfimpl.h"            /*I "pf.h" I*/

FList PFList = 0; /* list of all registered PD functions */

#undef __FUNC__  
#define __FUNC__ "PFSet"
/*@C
   PFSet - Sets the C/C++/Fortran functions to be used by the PF function

   Collective on PF

   Input Parameter:
+  pf - the function context
.  apply - function to apply to an array
.  applyvec - function to apply to a Vec
.  view - function that prints information about the PF
.  destroy - function to free the private function context
-  ctx - private function context

   Level: beginner

.keywords: PF, setting

.seealso: PFCreate(), PFDestroy(), PFSetType(), PFApply(), PFApplyVec()
@*/
int PFSet(PF pf,int(*apply)(void*,int,Scalar*,Scalar*),int(*applyvec)(void*,Vec,Vec),int(*view)(void*,Viewer),int(*destroy)(void*),void*ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pf,PF_COOKIE);
  pf->data             = ctx;

  pf->ops->destroy     = destroy;
  pf->ops->apply       = apply;
  pf->ops->applyvec    = applyvec;
  pf->ops->view        = view;

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PFDestroy"
/*@C
   PFDestroy - Destroys PF context that was created with PFCreate().

   Collective on PF

   Input Parameter:
.  pf - the function context

   Level: beginner

.keywords: PF, destroy

.seealso: PFCreate(), PFSet(), PFSetType()
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
   PFCreate - Creates a mathematical function context.

   Collective on MPI_Comm

   Input Parameter:
+  comm - MPI communicator 
.  dimin - dimension of the space you are mapping from
-  dimout - dimension of the space you are mapping to

   Output Parameter:
.  pf - the function context


   Level: developer

.keywords: PF, create, context

.seealso: PFSetUp(), PFApply(), PFDestroy()
@*/
int PFCreate(MPI_Comm comm,int dimin,int dimout,PF *pf)
{
  PF     newpf;

  PetscFunctionBegin;
  *pf          = 0;

  PetscHeaderCreate(newpf,_p_PF,struct _PFOps,PF_COOKIE,-1,"PF",comm,PFDestroy,PFView);
  PLogObjectCreate(newpf);
  newpf->bops->publish    = PFPublish_Petsc;
  newpf->data             = 0;

  newpf->ops->destroy     = 0;
  newpf->ops->apply       = 0;
  newpf->ops->applyvec    = 0;
  newpf->ops->view        = 0;
  newpf->dimin            = dimin;
  newpf->dimout           = dimout;

  *pf                     = newpf;
  PetscPublishAll(pf);
  PetscFunctionReturn(0);

}

/* -------------------------------------------------------------------------------*/

#undef __FUNC__  
#define __FUNC__ "PFApplyVec"
/*@
   PFApplyVec - Applies the mathematical function to a vector

   Collective on PF

   Input Parameters:
+  pf - the preconditioner context
-  x - input vector

   Output Parameter:
.  y - output vector

   Level: beginner

.keywords: PF, apply

.seealso: PFApply(), PFCreate(), PFDestroy(), PFSetType(), PFSet()
@*/
int PFApplyVec(PF pf,Vec x,Vec y)
{
  int        ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pf,PF_COOKIE);
  PetscValidHeaderSpecific(x,VEC_COOKIE);
  PetscValidHeaderSpecific(y,VEC_COOKIE);
  if (x == y) SETERRQ(PETSC_ERR_ARG_IDN,0,"x and y must be different vectors");

  if (pf->ops->applyvec) {
    ierr = (*pf->ops->applyvec)(pf->data,x,y);CHKERRQ(ierr);
  } else {
    Scalar *xx,*yy;
    int    n;

    ierr = VecGetLocalSize(x,&n);CHKERRQ(ierr);
    n    = n/pf->dimin;
    ierr = VecGetArray(x,&xx);CHKERRQ(ierr);
    ierr = VecGetArray(y,&yy);CHKERRQ(ierr);
    ierr = (*pf->ops->apply)(pf->data,n,xx,yy);CHKERRQ(ierr);
    ierr = VecRestoreArray(x,&xx);CHKERRQ(ierr);
    ierr = VecRestoreArray(y,&yy);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PFApply"
/*@
   PFApply - Applies the mathematical function to an array of values.

   Collective on PF

   Input Parameters:
+  pf - the preconditioner context
.  n - number of entries in input array
-  x - input array

   Output Parameter:
.  y - output array

   Level: beginner

.keywords: PF, apply

.seealso: PFApplyVec(), PFCreate(), PFDestroy(), PFSetType(), PFSet()
@*/
int PFApply(PF pf,int n,Scalar* x,Scalar* y)
{
  int        ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pf,PF_COOKIE);
  if (x == y) SETERRQ(PETSC_ERR_ARG_IDN,0,"x and y must be different arrays");

  ierr = (*pf->ops->apply)(pf->data,n,x,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PFView"
/*@ 
   PFView - Prints information about a mathematical function

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
  PetscTruth  isascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pf,PF_COOKIE);
  if (!viewer) viewer = VIEWER_STDOUT_SELF;
  PetscValidHeaderSpecific(viewer,VIEWER_COOKIE); 
  PetscCheckSameComm(pf,viewer);

  ierr = PetscTypeCompare((PetscObject)viewer,ASCII_VIEWER,&isascii);CHKERRQ(ierr);
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
      ierr = (*pf->ops->view)(pf->data,viewer);CHKERRQ(ierr);
      ierr = ViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    }
  } else {
    SETERRQ1(1,1,"Viewer type %s not supported by PF",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

/*MC
   PFRegisterDynamic - Adds a method to the mathematical function package.

   Synopsis:
   PFRegisterDynamic(char *name_solver,char *path,char *name_create,int (*routine_create)(PF))

   Not collective

   Input Parameters:
+  name_solver - name of a new user-defined solver
.  path - path (either absolute or relative) the library containing this solver
.  name_create - name of routine to create method context
-  routine_create - routine to create method context

   Notes:
   PFRegisterDynamic() may be called multiple times to add several user-defined functions

   If dynamic libraries are used, then the fourth input argument (routine_create)
   is ignored.

   Sample usage:
.vb
   PFRegisterDynamic("my_function","/home/username/my_lib/lib/libO/solaris/mylib",
              "MyFunctionCreate",MyFunctionSetCreate);
.ve

   Then, your solver can be chosen with the procedural interface via
$     PFSetType(pf,"my_function")
   or at runtime via the option
$     -pf_type my_function

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

#undef __FUNC__  
#define __FUNC__ "PFGetType"
/*@C
   PFGetType - Gets the PF method type and name (as a string) from the PF
   context.

   Not Collective

   Input Parameter:
.  pf - the preconditioner context

   Output Parameter:
.  name - name of preconditioner 

   Level: intermediate

.keywords: PF, get, method, name, type

.seealso: PFSetType()

@*/
int PFGetType(PF pf,PFType *meth)
{
  int ierr;

  PetscFunctionBegin;
  *meth = (PFType) pf->type_name;
  PetscFunctionReturn(0);
}
