#define PETSCVEC_DLL
/*
    The PF mathematical functions interface routines, callable by users.
*/
#include "../src/vec/pf/pfimpl.h"            /*I "petscpf.h" I*/

/* Logging support */
PetscCookie PF_COOKIE = 0;

PetscFList PFList         = PETSC_NULL; /* list of all registered PD functions */
PetscTruth PFRegisterAllCalled = PETSC_FALSE;

#undef __FUNCT__  
#define __FUNCT__ "PFSet"
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
PetscErrorCode PETSCVEC_DLLEXPORT PFSet(PF pf,PetscErrorCode (*apply)(void*,PetscInt,PetscScalar*,PetscScalar*),PetscErrorCode (*applyvec)(void*,Vec,Vec),PetscErrorCode (*view)(void*,PetscViewer),PetscErrorCode (*destroy)(void*),void*ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pf,PF_COOKIE,1);
  pf->data             = ctx;

  pf->ops->destroy     = destroy;
  pf->ops->apply       = apply;
  pf->ops->applyvec    = applyvec;
  pf->ops->view        = view;

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PFDestroy"
/*@C
   PFDestroy - Destroys PF context that was created with PFCreate().

   Collective on PF

   Input Parameter:
.  pf - the function context

   Level: beginner

.keywords: PF, destroy

.seealso: PFCreate(), PFSet(), PFSetType()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT PFDestroy(PF pf)
{
  PetscErrorCode ierr;
  PetscTruth     flg = PETSC_FALSE;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pf,PF_COOKIE,1);
  if (--((PetscObject)pf)->refct > 0) PetscFunctionReturn(0);

  ierr = PetscOptionsGetTruth(((PetscObject)pf)->prefix,"-pf_view",&flg,PETSC_NULL);CHKERRQ(ierr);
  if (flg) {
    PetscViewer viewer;
    ierr = PetscViewerASCIIGetStdout(((PetscObject)pf)->comm,&viewer);CHKERRQ(ierr);
    ierr = PFView(pf,viewer);CHKERRQ(ierr);
  }

  /* if memory was published with AMS then destroy it */
  ierr = PetscObjectDepublish(pf);CHKERRQ(ierr);

  if (pf->ops->destroy) {ierr =  (*pf->ops->destroy)(pf->data);CHKERRQ(ierr);}
  ierr = PetscHeaderDestroy(pf);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PFCreate"
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
PetscErrorCode PETSCVEC_DLLEXPORT PFCreate(MPI_Comm comm,PetscInt dimin,PetscInt dimout,PF *pf)
{
  PF             newpf;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(pf,1);
  *pf = PETSC_NULL;
#ifndef PETSC_USE_DYNAMIC_LIBRARIES
  ierr = PFInitializePackage(PETSC_NULL);CHKERRQ(ierr);   
#endif

  ierr = PetscHeaderCreate(newpf,_p_PF,struct _PFOps,PF_COOKIE,-1,"PF",comm,PFDestroy,PFView);CHKERRQ(ierr);
  newpf->data             = 0;

  newpf->ops->destroy     = 0;
  newpf->ops->apply       = 0;
  newpf->ops->applyvec    = 0;
  newpf->ops->view        = 0;
  newpf->dimin            = dimin;
  newpf->dimout           = dimout;

  *pf                     = newpf;
  ierr = PetscPublishAll(pf);CHKERRQ(ierr);
  PetscFunctionReturn(0);

}

/* -------------------------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "PFApplyVec"
/*@
   PFApplyVec - Applies the mathematical function to a vector

   Collective on PF

   Input Parameters:
+  pf - the function context
-  x - input vector (or PETSC_NULL for the vector (0,1, .... N-1)

   Output Parameter:
.  y - output vector

   Level: beginner

.keywords: PF, apply

.seealso: PFApply(), PFCreate(), PFDestroy(), PFSetType(), PFSet()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT PFApplyVec(PF pf,Vec x,Vec y)
{
  PetscErrorCode ierr;
  PetscInt       i,rstart,rend,n,p;
  PetscTruth     nox = PETSC_FALSE;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pf,PF_COOKIE,1);
  PetscValidHeaderSpecific(y,VEC_COOKIE,3);
  if (x) {
    PetscValidHeaderSpecific(x,VEC_COOKIE,2);
    if (x == y) SETERRQ(PETSC_ERR_ARG_IDN,"x and y must be different vectors");
  } else {
    PetscScalar *xx;

    ierr = VecDuplicate(y,&x);CHKERRQ(ierr);
    nox  = PETSC_TRUE;
    ierr = VecGetOwnershipRange(x,&rstart,&rend);CHKERRQ(ierr);
    ierr = VecGetArray(x,&xx);CHKERRQ(ierr);
    for (i=rstart; i<rend; i++) {
      xx[i-rstart] = (PetscScalar)i;
    }
    ierr = VecRestoreArray(x,&xx);CHKERRQ(ierr);
  }

  ierr = VecGetLocalSize(x,&n);CHKERRQ(ierr);
  ierr = VecGetLocalSize(y,&p);CHKERRQ(ierr);
  if ((pf->dimin*(n/pf->dimin)) != n) SETERRQ2(PETSC_ERR_ARG_SIZ,"Local input vector length %D not divisible by dimin %D of function",n,pf->dimin);
  if ((pf->dimout*(p/pf->dimout)) != p) SETERRQ2(PETSC_ERR_ARG_SIZ,"Local output vector length %D not divisible by dimout %D of function",p,pf->dimout);
  if ((n/pf->dimin) != (p/pf->dimout)) SETERRQ4(PETSC_ERR_ARG_SIZ,"Local vector lengths %D %D are wrong for dimin and dimout %D %D of function",n,p,pf->dimin,pf->dimout);

  if (pf->ops->applyvec) {
    ierr = (*pf->ops->applyvec)(pf->data,x,y);CHKERRQ(ierr);
  } else {
    PetscScalar *xx,*yy;

    ierr = VecGetLocalSize(x,&n);CHKERRQ(ierr);
    n    = n/pf->dimin;
    ierr = VecGetArray(x,&xx);CHKERRQ(ierr);
    ierr = VecGetArray(y,&yy);CHKERRQ(ierr);
    if (!pf->ops->apply) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"No function has been provided for this PF");
    ierr = (*pf->ops->apply)(pf->data,n,xx,yy);CHKERRQ(ierr);
    ierr = VecRestoreArray(x,&xx);CHKERRQ(ierr);
    ierr = VecRestoreArray(y,&yy);CHKERRQ(ierr);
  }
  if (nox) {
    ierr = VecDestroy(x);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PFApply"
/*@
   PFApply - Applies the mathematical function to an array of values.

   Collective on PF

   Input Parameters:
+  pf - the function context
.  n - number of pointwise function evaluations to perform, each pointwise function evaluation
       is a function of dimin variables and computes dimout variables where dimin and dimout are defined
       in the call to PFCreate()
-  x - input array

   Output Parameter:
.  y - output array

   Level: beginner

   Notes: 

.keywords: PF, apply

.seealso: PFApplyVec(), PFCreate(), PFDestroy(), PFSetType(), PFSet()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT PFApply(PF pf,PetscInt n,PetscScalar* x,PetscScalar* y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pf,PF_COOKIE,1);
  PetscValidScalarPointer(x,2);
  PetscValidScalarPointer(y,3);
  if (x == y) SETERRQ(PETSC_ERR_ARG_IDN,"x and y must be different arrays");
  if (!pf->ops->apply) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"No function has been provided for this PF");

  ierr = (*pf->ops->apply)(pf->data,n,x,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PFView"
/*@ 
   PFView - Prints information about a mathematical function

   Collective on PF unless PetscViewer is PETSC_VIEWER_STDOUT_SELF  

   Input Parameters:
+  PF - the PF context
-  viewer - optional visualization context

   Note:
   The available visualization contexts include
+     PETSC_VIEWER_STDOUT_SELF - standard output (default)
-     PETSC_VIEWER_STDOUT_WORLD - synchronized standard
         output where only the first processor opens
         the file.  All other processors send their 
         data to the first processor to print. 

   The user can open an alternative visualization contexts with
   PetscViewerASCIIOpen() (output to a specified file).

   Level: developer

.keywords: PF, view

.seealso: PetscViewerCreate(), PetscViewerASCIIOpen()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT PFView(PF pf,PetscViewer viewer)
{
  const PFType      cstr;
  PetscErrorCode    ierr;
  PetscTruth        iascii;
  PetscViewerFormat format;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pf,PF_COOKIE,1);
  if (!viewer) {
    ierr = PetscViewerASCIIGetStdout(((PetscObject)pf)->comm,&viewer);CHKERRQ(ierr);
  }
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_COOKIE,2); 
  PetscCheckSameComm(pf,1,viewer,2);

  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"PF Object:\n");CHKERRQ(ierr);
    ierr = PFGetType(pf,&cstr);CHKERRQ(ierr);
    if (cstr) {
      ierr = PetscViewerASCIIPrintf(viewer,"  type: %s\n",cstr);CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"  type: not yet set\n");CHKERRQ(ierr);
    }
    if (pf->ops->view) {
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = (*pf->ops->view)(pf->data,viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    }
  } else {
    SETERRQ1(PETSC_ERR_SUP,"Viewer type %s not supported by PF",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

/*MC
   PFRegisterDynamic - Adds a method to the mathematical function package.

   Synopsis:
   PetscErrorCode PFRegisterDynamic(char *name_solver,char *path,char *name_create,PetscErrorCode (*routine_create)(PF))

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

   ${PETSC_ARCH}, ${PETSC_DIR}, ${PETSC_LIB_DIR}, or ${any environmental variable}
 occuring in pathname will be replaced with appropriate values.

.keywords: PF, register

.seealso: PFRegisterAll(), PFRegisterDestroy(), PFRegister()
M*/

#undef __FUNCT__  
#define __FUNCT__ "PFRegister"
PetscErrorCode PETSCVEC_DLLEXPORT PFRegister(const char sname[],const char path[],const char name[],PetscErrorCode (*function)(PF,void*))
{
  PetscErrorCode ierr;
  char           fullname[PETSC_MAX_PATH_LEN];

  PetscFunctionBegin;
  ierr = PetscFListConcat(path,name,fullname);CHKERRQ(ierr);
  ierr = PetscFListAdd(&PFList,sname,fullname,(void (*)(void))function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PFGetType"
/*@C
   PFGetType - Gets the PF method type and name (as a string) from the PF
   context.

   Not Collective

   Input Parameter:
.  pf - the function context

   Output Parameter:
.  type - name of function 

   Level: intermediate

.keywords: PF, get, method, name, type

.seealso: PFSetType()

@*/
PetscErrorCode PETSCVEC_DLLEXPORT PFGetType(PF pf,const PFType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pf,PF_COOKIE,1);
  PetscValidPointer(type,2);
  *type = ((PetscObject)pf)->type_name;
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "PFSetType"
/*@C
   PFSetType - Builds PF for a particular function

   Collective on PF

   Input Parameter:
+  pf - the function context.
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
PetscErrorCode PETSCVEC_DLLEXPORT PFSetType(PF pf,const PFType type,void *ctx)
{
  PetscErrorCode ierr,(*r)(PF,void*);
  PetscTruth     match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pf,PF_COOKIE,1);
  PetscValidCharPointer(type,2);

  ierr = PetscTypeCompare((PetscObject)pf,type,&match);CHKERRQ(ierr);
  if (match) PetscFunctionReturn(0);

  if (pf->ops->destroy) {ierr =  (*pf->ops->destroy)(pf);CHKERRQ(ierr);}
  pf->data        = 0;

  /* Determine the PFCreateXXX routine for a particular function */
  ierr =  PetscFListFind(PFList,((PetscObject)pf)->comm,type,(void (**)(void)) &r);CHKERRQ(ierr);
  if (!r) SETERRQ1(PETSC_ERR_ARG_UNKNOWN_TYPE,"Unable to find requested PF type %s",type);
  pf->ops->destroy             = 0;
  pf->ops->view                = 0;
  pf->ops->apply               = 0;
  pf->ops->applyvec            = 0;

  /* Call the PFCreateXXX routine for this particular function */
  ierr = (*r)(pf,ctx);CHKERRQ(ierr);

  ierr = PetscObjectChangeTypeName((PetscObject)pf,type);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PFSetFromOptions"
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

.seealso:
@*/
PetscErrorCode PETSCVEC_DLLEXPORT PFSetFromOptions(PF pf)
{
  PetscErrorCode ierr;
  char           type[256];
  PetscTruth     flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pf,PF_COOKIE,1);

  ierr = PetscOptionsBegin(((PetscObject)pf)->comm,((PetscObject)pf)->prefix,"Mathematical functions options","Vec");CHKERRQ(ierr);
    ierr = PetscOptionsList("-pf_type","Type of function","PFSetType",PFList,0,type,256,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PFSetType(pf,type,PETSC_NULL);CHKERRQ(ierr);
    }
    if (pf->ops->setfromoptions) {
      ierr = (*pf->ops->setfromoptions)(pf);CHKERRQ(ierr);
    }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscTruth PFPackageInitialized = PETSC_FALSE;
#undef __FUNCT__  
#define __FUNCT__ "PFFinalizePackage"
/*@C
  PFFinalizePackage - This function destroys everything in the Petsc interface to Mathematica. It is
  called from PetscFinalize().

  Level: developer

.keywords: Petsc, destroy, package, mathematica
.seealso: PetscFinalize()
@*/
PetscErrorCode PETSC_DLLEXPORT PFFinalizePackage(void) 
{
  PetscFunctionBegin;
  PFPackageInitialized = PETSC_FALSE;
  PFList               = PETSC_NULL;
  PFRegisterAllCalled  = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PFInitializePackage"
/*@C
  PFInitializePackage - This function initializes everything in the PF package. It is called
  from PetscDLLibraryRegister() when using dynamic libraries, and on the first call to PFCreate()
  when using static libraries.

  Input Parameter:
. path - The dynamic library path, or PETSC_NULL

  Level: developer

.keywords: Vec, initialize, package
.seealso: PetscInitialize()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT PFInitializePackage(const char path[]) 
{
  char              logList[256];
  char              *className;
  PetscTruth        opt;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (PFPackageInitialized) PetscFunctionReturn(0);
  PFPackageInitialized = PETSC_TRUE;
  /* Register Classes */
  ierr = PetscCookieRegister("PointFunction",&PF_COOKIE);CHKERRQ(ierr);
  /* Register Constructors */
  ierr = PFRegisterAll(path);CHKERRQ(ierr);
  /* Process info exclusions */
  ierr = PetscOptionsGetString(PETSC_NULL, "-info_exclude", logList, 256, &opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrstr(logList, "pf", &className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscInfoDeactivateClass(PF_COOKIE);CHKERRQ(ierr);
    }
  }
  /* Process summary exclusions */
  ierr = PetscOptionsGetString(PETSC_NULL, "-log_summary_exclude", logList, 256, &opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrstr(logList, "pf", &className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscLogEventDeactivateClass(PF_COOKIE);CHKERRQ(ierr);
    }
  }
  ierr = PetscRegisterFinalize(PFFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}









