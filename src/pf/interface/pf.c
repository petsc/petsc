/*$Id: pf.c,v 1.10 2000/05/05 22:20:09 balay Exp bsmith $*/
/*
    The PF mathematical functions interface routines, callable by users.
*/
#include "src/pf/pfimpl.h"            /*I "petscpf.h" I*/

FList      PFList = 0; /* list of all registered PD functions */
PetscTruth PFRegisterAllCalled = PETSC_FALSE;

#undef __FUNC__  
#define __FUNC__ /*<a name="PFSet"></a>*/"PFSet"
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
#define __FUNC__ /*<a name="PFDestroy"></a>*/"PFDestroy"
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
  int        ierr;
  PetscTruth flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pf,PF_COOKIE);
  if (--pf->refct > 0) PetscFunctionReturn(0);

  ierr = OptionsHasName(pf->prefix,"-pf_view",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PFView(pf,VIEWER_STDOUT_(pf->comm));CHKERRQ(ierr);
  }

  /* if memory was published with AMS then destroy it */
  ierr = PetscObjectDepublish(pf);CHKERRQ(ierr);

  if (pf->ops->destroy) {ierr =  (*pf->ops->destroy)(pf->data);CHKERRQ(ierr);}
  PLogObjectDestroy(pf);
  PetscHeaderDestroy(pf);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name="PFPublish_Petsc"></a>*/"PFPublish_Petsc"
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
#define __FUNC__ /*<a name="PFCreate"></a>*/"PFCreate"
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

.seealso: PFSetUp(), PFApply(), PFDestroy(), PFApplyVec()
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
#define __FUNC__ /*<a name="PFApplyVec"></a>*/"PFApplyVec"
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
    if (!pf->ops->apply) SETERRQ(1,1,"No function has been provided for this PF");
    ierr = (*pf->ops->apply)(pf->data,n,xx,yy);CHKERRQ(ierr);
    ierr = VecRestoreArray(x,&xx);CHKERRQ(ierr);
    ierr = VecRestoreArray(y,&yy);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name="PFApply"></a>*/"PFApply"
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
  if (!pf->ops->apply) SETERRQ(1,1,"No function has been provided for this PF");

  ierr = (*pf->ops->apply)(pf->data,n,x,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name="PFView"></a>*/"PFView"
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

.seealso: ViewerCreate(), ViewerASCIIOpen()
@*/
int PFView(PF pf,Viewer viewer)
{
  PFType      cstr;
  int         fmt,ierr;
  PetscTruth  isascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pf,PF_COOKIE);
  if (!viewer) viewer = VIEWER_STDOUT_(pf->comm);
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
#define __FUNC__ /*<a name="PFRegister"></a>*/"PFRegister"
int PFRegister(char *sname,char *path,char *name,int (*function)(PF,void*))
{
  int  ierr;
  char fullname[256];

  PetscFunctionBegin;

  ierr = FListConcat(path,name,fullname);CHKERRQ(ierr);
  ierr = FListAdd(&PFList,sname,fullname,(int (*)(void*))function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}



#undef __FUNC__  
#define __FUNC__ /*<a name="PFGetType"></a>*/"PFGetType"
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
  PetscFunctionBegin;
  *meth = (PFType) pf->type_name;
  PetscFunctionReturn(0);
}


#undef __FUNC__  
#define __FUNC__ /*<a name="PFSetType"></a>*/"PFSetType"
/*@C
   PFSetType - Builds PF for a particular function

   Collective on PF

   Input Parameter:
+  pf - the preconditioner context.
.  type - a known method
-  ctx - optional type dependent context

   Options Database Key:
.  -pf_type <type> - Sets PF type


  Notes:
  See "petsc/include/petscpf.h" for available methods (for instance,
  PFCONSTANT)

  Level: intermediate

.keywords: PF, set, method, type

.seealso: PFSet(), PFRegisterDynamic(), PFCreate(), DACreatePF()

@*/
int PFSetType(PF pf,PFType type,void *ctx)
{
  int        ierr,(*r)(PF,void*);
  PetscTruth match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pf,PF_COOKIE);
  PetscValidCharPointer(type);

  ierr = PetscTypeCompare((PetscObject)pf,type,&match);CHKERRQ(ierr);
  if (match) PetscFunctionReturn(0);

  if (pf->ops->destroy) {ierr =  (*pf->ops->destroy)(pf);CHKERRQ(ierr);}
  pf->data        = 0;

  /* Get the function pointers for the method requested */
  if (!PFRegisterAllCalled) {ierr = PFRegisterAll(0);CHKERRQ(ierr);}

  /* Determine the PFCreateXXX routine for a particular function */
  ierr =  FListFind(pf->comm,PFList,type,(int (**)(void *)) &r);CHKERRQ(ierr);
  if (!r) SETERRQ1(1,1,"Unable to find requested PF type %s",type);

  pf->ops->destroy             = 0;
  pf->ops->view                = 0;
  pf->ops->apply               = 0;
  pf->ops->applyvec            = 0;

  /* Call the PFCreateXXX routine for this particular function */
  ierr = (*r)(pf,ctx);CHKERRQ(ierr);

  ierr = PetscObjectChangeTypeName((PetscObject)pf,type);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name="PFSetFromOptions"></a>*/"PFSetFromOptions"
/*@
   PFSetFromOptions - Sets PF options from the options database.

   Collective on PF

   Input Parameters:
.  pf - the mathematical function context

   Options Database Keys:

   Notes:  
   To see all options, run your program with the -help option
   or consult the users manual.

   Level: intermediate

.keywords: PF, set, from, options, database

.seealso: PFPrintHelp()
@*/
int PFSetFromOptions(PF pf)
{
  int        ierr;
  char       type[256];
  PetscTruth flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pf,PF_COOKIE);

  ierr = OptionsGetString(pf->prefix,"-pf_type",type,256,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PFSetType(pf,type,PETSC_NULL);CHKERRQ(ierr);
  }

  if (pf->ops->setfromoptions) {
    ierr = (*pf->ops->setfromoptions)(pf);CHKERRQ(ierr);
  }
  
  ierr = OptionsHasName(PETSC_NULL,"-help",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PFPrintHelp(pf);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name="PFPrintHelp"></a>*/"PFPrintHelp"
/*@
   PFPrintHelp - Prints all the options for the PF component.

   Collective on PF

   Input Parameter:
.  pf - the mathematical function context

   Options Database Keys:
+  -help - Prints PF options
-  -h - Prints PF options

   Level: developer

.keywords: PF, help

.seealso: PFSetFromOptions()

@*/
int PFPrintHelp(PF pf)
{
  char p[64]; 
  int  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pf,PF_COOKIE);
  ierr = PetscStrcpy(p,"-");CHKERRQ(ierr);
  if (pf->prefix) {ierr = PetscStrcat(p,pf->prefix);CHKERRQ(ierr);}
  if (!PFRegisterAllCalled) {ierr = PFRegisterAll(0);CHKERRQ(ierr);}
  ierr = (*PetscHelpPrintf)(pf->comm,"PF options --------------------------------------------------\n");CHKERRQ(ierr);
  ierr = FListPrintTypes(pf->comm,stdout,pf->prefix,"pf_type",PFList);CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(pf->comm,"Run program with -help %spf_type <type> for help on ",p);CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(pf->comm,"a particular function\n");CHKERRQ(ierr);
  if (pf->ops->printhelp) {
    ierr = (*pf->ops->printhelp)(pf,p);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
