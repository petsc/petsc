/*
    The PF mathematical functions interface routines, callable by users.
*/
#include <../src/vec/pf/pfimpl.h>            /*I "petscpf.h" I*/

PetscClassId      PF_CLASSID          = 0;
PetscFunctionList PFList              = NULL;   /* list of all registered PD functions */
PetscBool         PFRegisterAllCalled = PETSC_FALSE;

/*@C
   PFSet - Sets the C/C++/Fortran functions to be used by the PF function

   Collective on PF

   Input Parameters:
+  pf - the function context
.  apply - function to apply to an array
.  applyvec - function to apply to a Vec
.  view - function that prints information about the PF
.  destroy - function to free the private function context
-  ctx - private function context

   Level: beginner

.seealso: PFCreate(), PFDestroy(), PFSetType(), PFApply(), PFApplyVec()
@*/
PetscErrorCode  PFSet(PF pf,PetscErrorCode (*apply)(void*,PetscInt,const PetscScalar*,PetscScalar*),PetscErrorCode (*applyvec)(void*,Vec,Vec),PetscErrorCode (*view)(void*,PetscViewer),PetscErrorCode (*destroy)(void*),void*ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pf,PF_CLASSID,1);
  pf->data          = ctx;
  pf->ops->destroy  = destroy;
  pf->ops->apply    = apply;
  pf->ops->applyvec = applyvec;
  pf->ops->view     = view;
  PetscFunctionReturn(0);
}

/*@C
   PFDestroy - Destroys PF context that was created with PFCreate().

   Collective on PF

   Input Parameter:
.  pf - the function context

   Level: beginner

.seealso: PFCreate(), PFSet(), PFSetType()
@*/
PetscErrorCode  PFDestroy(PF *pf)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*pf) PetscFunctionReturn(0);
  PetscValidHeaderSpecific((*pf),PF_CLASSID,1);
  if (--((PetscObject)(*pf))->refct > 0) PetscFunctionReturn(0);

  ierr = PFViewFromOptions(*pf,NULL,"-pf_view");CHKERRQ(ierr);
  /* if memory was published with SAWs then destroy it */
  ierr = PetscObjectSAWsViewOff((PetscObject)*pf);CHKERRQ(ierr);

  if ((*pf)->ops->destroy) {ierr =  (*(*pf)->ops->destroy)((*pf)->data);CHKERRQ(ierr);}
  ierr = PetscHeaderDestroy(pf);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PFCreate - Creates a mathematical function context.

   Collective

   Input Parameters:
+  comm - MPI communicator
.  dimin - dimension of the space you are mapping from
-  dimout - dimension of the space you are mapping to

   Output Parameter:
.  pf - the function context

   Level: developer

.seealso: PFSet(), PFApply(), PFDestroy(), PFApplyVec()
@*/
PetscErrorCode  PFCreate(MPI_Comm comm,PetscInt dimin,PetscInt dimout,PF *pf)
{
  PF             newpf;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(pf,4);
  *pf = NULL;
  ierr = PFInitializePackage();CHKERRQ(ierr);

  ierr = PetscHeaderCreate(newpf,PF_CLASSID,"PF","Mathematical functions","Vec",comm,PFDestroy,PFView);CHKERRQ(ierr);
  newpf->data          = NULL;
  newpf->ops->destroy  = NULL;
  newpf->ops->apply    = NULL;
  newpf->ops->applyvec = NULL;
  newpf->ops->view     = NULL;
  newpf->dimin         = dimin;
  newpf->dimout        = dimout;

  *pf                  = newpf;
  PetscFunctionReturn(0);

}

/* -------------------------------------------------------------------------------*/

/*@
   PFApplyVec - Applies the mathematical function to a vector

   Collective on PF

   Input Parameters:
+  pf - the function context
-  x - input vector (or NULL for the vector (0,1, .... N-1)

   Output Parameter:
.  y - output vector

   Level: beginner

.seealso: PFApply(), PFCreate(), PFDestroy(), PFSetType(), PFSet()
@*/
PetscErrorCode  PFApplyVec(PF pf,Vec x,Vec y)
{
  PetscErrorCode ierr;
  PetscInt       i,rstart,rend,n,p;
  PetscBool      nox = PETSC_FALSE;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pf,PF_CLASSID,1);
  PetscValidHeaderSpecific(y,VEC_CLASSID,3);
  if (x) {
    PetscValidHeaderSpecific(x,VEC_CLASSID,2);
    if (x == y) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_IDN,"x and y must be different vectors");
  } else {
    PetscScalar *xx;
    PetscInt    lsize;

    ierr  = VecGetLocalSize(y,&lsize);CHKERRQ(ierr);
    lsize = pf->dimin*lsize/pf->dimout;
    ierr  = VecCreateMPI(PetscObjectComm((PetscObject)y),lsize,PETSC_DETERMINE,&x);CHKERRQ(ierr);
    nox   = PETSC_TRUE;
    ierr  = VecGetOwnershipRange(x,&rstart,&rend);CHKERRQ(ierr);
    ierr  = VecGetArray(x,&xx);CHKERRQ(ierr);
    for (i=rstart; i<rend; i++) xx[i-rstart] = (PetscScalar)i;
    ierr = VecRestoreArray(x,&xx);CHKERRQ(ierr);
  }

  ierr = VecGetLocalSize(x,&n);CHKERRQ(ierr);
  ierr = VecGetLocalSize(y,&p);CHKERRQ(ierr);
  if ((pf->dimin*(n/pf->dimin)) != n) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Local input vector length %" PetscInt_FMT " not divisible by dimin %" PetscInt_FMT " of function",n,pf->dimin);
  if ((pf->dimout*(p/pf->dimout)) != p) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Local output vector length %" PetscInt_FMT " not divisible by dimout %" PetscInt_FMT " of function",p,pf->dimout);
  if ((n/pf->dimin) != (p/pf->dimout)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Local vector lengths %" PetscInt_FMT " %" PetscInt_FMT " are wrong for dimin and dimout %" PetscInt_FMT " %" PetscInt_FMT " of function",n,p,pf->dimin,pf->dimout);

  if (pf->ops->applyvec) {
    ierr = (*pf->ops->applyvec)(pf->data,x,y);CHKERRQ(ierr);
  } else {
    PetscScalar *xx,*yy;

    ierr = VecGetLocalSize(x,&n);CHKERRQ(ierr);
    n    = n/pf->dimin;
    ierr = VecGetArray(x,&xx);CHKERRQ(ierr);
    ierr = VecGetArray(y,&yy);CHKERRQ(ierr);
    if (!pf->ops->apply) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"No function has been provided for this PF");
    ierr = (*pf->ops->apply)(pf->data,n,xx,yy);CHKERRQ(ierr);
    ierr = VecRestoreArray(x,&xx);CHKERRQ(ierr);
    ierr = VecRestoreArray(y,&yy);CHKERRQ(ierr);
  }
  if (nox) {
    ierr = VecDestroy(&x);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

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

.seealso: PFApplyVec(), PFCreate(), PFDestroy(), PFSetType(), PFSet()
@*/
PetscErrorCode  PFApply(PF pf,PetscInt n,const PetscScalar *x,PetscScalar *y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pf,PF_CLASSID,1);
  PetscValidScalarPointer(x,3);
  PetscValidScalarPointer(y,4);
  if (x == y) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_IDN,"x and y must be different arrays");
  if (!pf->ops->apply) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"No function has been provided for this PF");

  ierr = (*pf->ops->apply)(pf->data,n,x,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PFViewFromOptions - View from Options

   Collective on PF

   Input Parameters:
+  A - the PF context
.  obj - Optional object
-  name - command line option

   Level: intermediate
.seealso:  PF, PFView, PetscObjectViewFromOptions(), PFCreate()
@*/
PetscErrorCode  PFViewFromOptions(PF A,PetscObject obj,const char name[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,PF_CLASSID,1);
  ierr = PetscObjectViewFromOptions((PetscObject)A,obj,name);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

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

.seealso: PetscViewerCreate(), PetscViewerASCIIOpen()
@*/
PetscErrorCode  PFView(PF pf,PetscViewer viewer)
{
  PetscErrorCode    ierr;
  PetscBool         iascii;
  PetscViewerFormat format;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pf,PF_CLASSID,1);
  if (!viewer) {
    ierr = PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)pf),&viewer);CHKERRQ(ierr);
  }
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(pf,1,viewer,2);

  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
    ierr = PetscObjectPrintClassNamePrefixType((PetscObject)pf,viewer);CHKERRQ(ierr);
    if (pf->ops->view) {
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = (*pf->ops->view)(pf->data,viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/*@C
   PFRegister - Adds a method to the mathematical function package.

   Not collective

   Input Parameters:
+  name_solver - name of a new user-defined solver
-  routine_create - routine to create method context

   Notes:
   PFRegister() may be called multiple times to add several user-defined functions

   Sample usage:
.vb
   PFRegister("my_function",MyFunctionSetCreate);
.ve

   Then, your solver can be chosen with the procedural interface via
$     PFSetType(pf,"my_function")
   or at runtime via the option
$     -pf_type my_function

   Level: advanced

.seealso: PFRegisterAll(), PFRegisterDestroy(), PFRegister()
@*/
PetscErrorCode  PFRegister(const char sname[],PetscErrorCode (*function)(PF,void*))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PFInitializePackage();CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&PFList,sname,function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PFGetType - Gets the PF method type and name (as a string) from the PF
   context.

   Not Collective

   Input Parameter:
.  pf - the function context

   Output Parameter:
.  type - name of function

   Level: intermediate

.seealso: PFSetType()

@*/
PetscErrorCode  PFGetType(PF pf,PFType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pf,PF_CLASSID,1);
  PetscValidPointer(type,2);
  *type = ((PetscObject)pf)->type_name;
  PetscFunctionReturn(0);
}

/*@C
   PFSetType - Builds PF for a particular function

   Collective on PF

   Input Parameters:
+  pf - the function context.
.  type - a known method
-  ctx - optional type dependent context

   Options Database Key:
.  -pf_type <type> - Sets PF type

  Notes:
  See "petsc/include/petscpf.h" for available methods (for instance,
  PFCONSTANT)

  Level: intermediate

.seealso: PFSet(), PFRegister(), PFCreate(), DMDACreatePF()

@*/
PetscErrorCode  PFSetType(PF pf,PFType type,void *ctx)
{
  PetscErrorCode ierr,(*r)(PF,void*);
  PetscBool      match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pf,PF_CLASSID,1);
  PetscValidCharPointer(type,2);

  ierr = PetscObjectTypeCompare((PetscObject)pf,type,&match);CHKERRQ(ierr);
  if (match) PetscFunctionReturn(0);

  if (pf->ops->destroy) {ierr =  (*pf->ops->destroy)(pf);CHKERRQ(ierr);}
  pf->data = NULL;

  /* Determine the PFCreateXXX routine for a particular function */
  ierr = PetscFunctionListFind(PFList,type,&r);CHKERRQ(ierr);
  if (!r) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Unable to find requested PF type %s",type);
  pf->ops->destroy  = NULL;
  pf->ops->view     = NULL;
  pf->ops->apply    = NULL;
  pf->ops->applyvec = NULL;

  /* Call the PFCreateXXX routine for this particular function */
  ierr = (*r)(pf,ctx);CHKERRQ(ierr);

  ierr = PetscObjectChangeTypeName((PetscObject)pf,type);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

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

.seealso:
@*/
PetscErrorCode  PFSetFromOptions(PF pf)
{
  PetscErrorCode ierr;
  char           type[256];
  PetscBool      flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pf,PF_CLASSID,1);

  ierr = PetscObjectOptionsBegin((PetscObject)pf);CHKERRQ(ierr);
  ierr = PetscOptionsFList("-pf_type","Type of function","PFSetType",PFList,NULL,type,256,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PFSetType(pf,type,NULL);CHKERRQ(ierr);
  }
  if (pf->ops->setfromoptions) {
    ierr = (*pf->ops->setfromoptions)(PetscOptionsObject,pf);CHKERRQ(ierr);
  }

  /* process any options handlers added with PetscObjectAddOptionsHandler() */
  ierr = PetscObjectProcessOptionsHandlers(PetscOptionsObject,(PetscObject)pf);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscBool PFPackageInitialized = PETSC_FALSE;
/*@C
  PFFinalizePackage - This function destroys everything in the Petsc interface to Mathematica. It is
  called from PetscFinalize().

  Level: developer

.seealso: PetscFinalize()
@*/
PetscErrorCode  PFFinalizePackage(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFunctionListDestroy(&PFList);CHKERRQ(ierr);
  PFPackageInitialized = PETSC_FALSE;
  PFRegisterAllCalled  = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@C
  PFInitializePackage - This function initializes everything in the PF package. It is called
  from PetscDLLibraryRegister_petscvec() when using dynamic libraries, and on the first call to PFCreate()
  when using shared or static libraries.

  Level: developer

.seealso: PetscInitialize()
@*/
PetscErrorCode  PFInitializePackage(void)
{
  char           logList[256];
  PetscBool      opt,pkg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PFPackageInitialized) PetscFunctionReturn(0);
  PFPackageInitialized = PETSC_TRUE;
  /* Register Classes */
  ierr = PetscClassIdRegister("PointFunction",&PF_CLASSID);CHKERRQ(ierr);
  /* Register Constructors */
  ierr = PFRegisterAll();CHKERRQ(ierr);
  /* Process Info */
  {
    PetscClassId  classids[1];

    classids[0] = PF_CLASSID;
    ierr = PetscInfoProcessClass("pf", 1, classids);CHKERRQ(ierr);
  }
  /* Process summary exclusions */
  ierr = PetscOptionsGetString(NULL,NULL,"-log_exclude",logList,sizeof(logList),&opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrInList("pf",logList,',',&pkg);CHKERRQ(ierr);
    if (pkg) {ierr = PetscLogEventExcludeClass(PF_CLASSID);CHKERRQ(ierr);}
  }
  /* Register package finalizer */
  ierr = PetscRegisterFinalize(PFFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

