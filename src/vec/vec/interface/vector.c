/*
     Provides the interface functions for vector operations that do NOT have PetscScalar/PetscReal in the signature
   These are the vector functions the user calls.
*/
#include <petsc/private/vecimpl.h>    /*I  "petscvec.h"   I*/

/* Logging support */
PetscClassId  VEC_CLASSID;
PetscLogEvent VEC_View, VEC_Max, VEC_Min, VEC_Dot, VEC_MDot, VEC_TDot;
PetscLogEvent VEC_Norm, VEC_Normalize, VEC_Scale, VEC_Copy, VEC_Set, VEC_AXPY, VEC_AYPX, VEC_WAXPY;
PetscLogEvent VEC_MTDot, VEC_MAXPY, VEC_Swap, VEC_AssemblyBegin, VEC_ScatterBegin, VEC_ScatterEnd;
PetscLogEvent VEC_AssemblyEnd, VEC_PointwiseMult, VEC_SetValues, VEC_Load;
PetscLogEvent VEC_SetRandom, VEC_ReduceArithmetic, VEC_ReduceCommunication,VEC_ReduceBegin,VEC_ReduceEnd,VEC_Ops;
PetscLogEvent VEC_DotNorm2, VEC_AXPBYPCZ;
PetscLogEvent VEC_ViennaCLCopyFromGPU, VEC_ViennaCLCopyToGPU;
PetscLogEvent VEC_CUDACopyFromGPU, VEC_CUDACopyToGPU;
PetscLogEvent VEC_CUDACopyFromGPUSome, VEC_CUDACopyToGPUSome;
PetscLogEvent VEC_HIPCopyFromGPU, VEC_HIPCopyToGPU;
PetscLogEvent VEC_HIPCopyFromGPUSome, VEC_HIPCopyToGPUSome;

/*@
   VecStashGetInfo - Gets how many values are currently in the vector stash, i.e. need
       to be communicated to other processors during the VecAssemblyBegin/End() process

    Not collective

   Input Parameter:
.   vec - the vector

   Output Parameters:
+   nstash   - the size of the stash
.   reallocs - the number of additional mallocs incurred.
.   bnstash   - the size of the block stash
-   breallocs - the number of additional mallocs incurred.in the block stash

   Level: advanced

.seealso: `VecAssemblyBegin()`, `VecAssemblyEnd()`, `Vec`, `VecStashSetInitialSize()`, `VecStashView()`

@*/
PetscErrorCode  VecStashGetInfo(Vec vec,PetscInt *nstash,PetscInt *reallocs,PetscInt *bnstash,PetscInt *breallocs)
{
  PetscFunctionBegin;
  PetscCall(VecStashGetInfo_Private(&vec->stash,nstash,reallocs));
  PetscCall(VecStashGetInfo_Private(&vec->bstash,bnstash,breallocs));
  PetscFunctionReturn(0);
}

/*@
   VecSetLocalToGlobalMapping - Sets a local numbering to global numbering used
   by the routine VecSetValuesLocal() to allow users to insert vector entries
   using a local (per-processor) numbering.

   Logically Collective on Vec

   Input Parameters:
+  x - vector
-  mapping - mapping created with ISLocalToGlobalMappingCreate() or ISLocalToGlobalMappingCreateIS()

   Notes:
   All vectors obtained with VecDuplicate() from this vector inherit the same mapping.

   Level: intermediate

seealso:  VecAssemblyBegin(), VecAssemblyEnd(), VecSetValues(), VecSetValuesLocal(),
           VecSetLocalToGlobalMapping(), VecSetValuesBlockedLocal()
@*/
PetscErrorCode  VecSetLocalToGlobalMapping(Vec x,ISLocalToGlobalMapping mapping)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_CLASSID,1);
  if (mapping) PetscValidHeaderSpecific(mapping,IS_LTOGM_CLASSID,2);
  if (x->ops->setlocaltoglobalmapping) {
    PetscCall((*x->ops->setlocaltoglobalmapping)(x,mapping));
  } else {
    PetscCall(PetscLayoutSetISLocalToGlobalMapping(x->map,mapping));
  }
  PetscFunctionReturn(0);
}

/*@
   VecGetLocalToGlobalMapping - Gets the local-to-global numbering set by VecSetLocalToGlobalMapping()

   Not Collective

   Input Parameter:
.  X - the vector

   Output Parameter:
.  mapping - the mapping

   Level: advanced

.seealso: `VecSetValuesLocal()`
@*/
PetscErrorCode VecGetLocalToGlobalMapping(Vec X,ISLocalToGlobalMapping *mapping)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(X,VEC_CLASSID,1);
  PetscValidType(X,1);
  PetscValidPointer(mapping,2);
  *mapping = X->map->mapping;
  PetscFunctionReturn(0);
}

/*@
   VecAssemblyBegin - Begins assembling the vector.  This routine should
   be called after completing all calls to VecSetValues().

   Collective on Vec

   Input Parameter:
.  vec - the vector

   Level: beginner

.seealso: `VecAssemblyEnd()`, `VecSetValues()`
@*/
PetscErrorCode  VecAssemblyBegin(Vec vec)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(vec,VEC_CLASSID,1);
  PetscValidType(vec,1);
  PetscCall(VecStashViewFromOptions(vec,NULL,"-vec_view_stash"));
  PetscCall(PetscLogEventBegin(VEC_AssemblyBegin,vec,0,0,0));
  if (vec->ops->assemblybegin) {
    PetscCall((*vec->ops->assemblybegin)(vec));
  }
  PetscCall(PetscLogEventEnd(VEC_AssemblyBegin,vec,0,0,0));
  PetscCall(PetscObjectStateIncrease((PetscObject)vec));
  PetscFunctionReturn(0);
}

/*@
   VecAssemblyEnd - Completes assembling the vector.  This routine should
   be called after VecAssemblyBegin().

   Collective on Vec

   Input Parameter:
.  vec - the vector

   Options Database Keys:
+  -vec_view - Prints vector in ASCII format
.  -vec_view ::ascii_matlab - Prints vector in ASCII MATLAB format to stdout
.  -vec_view matlab:filename - Prints vector in MATLAB format to matlaboutput.mat
.  -vec_view draw - Activates vector viewing using drawing tools
.  -display <name> - Sets display name (default is host)
.  -draw_pause <sec> - Sets number of seconds to pause after display
-  -vec_view socket - Activates vector viewing using a socket

   Level: beginner

.seealso: `VecAssemblyBegin()`, `VecSetValues()`
@*/
PetscErrorCode  VecAssemblyEnd(Vec vec)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(vec,VEC_CLASSID,1);
  PetscCall(PetscLogEventBegin(VEC_AssemblyEnd,vec,0,0,0));
  PetscValidType(vec,1);
  if (vec->ops->assemblyend) {
    PetscCall((*vec->ops->assemblyend)(vec));
  }
  PetscCall(PetscLogEventEnd(VEC_AssemblyEnd,vec,0,0,0));
  PetscCall(VecViewFromOptions(vec,NULL,"-vec_view"));
  PetscFunctionReturn(0);
}

/*@
   VecPointwiseMax - Computes the componentwise maximum w_i = max(x_i, y_i).

   Logically Collective on Vec

   Input Parameters:
.  x, y  - the vectors

   Output Parameter:
.  w - the result

   Level: advanced

   Notes:
    any subset of the x, y, and w may be the same vector.
          For complex numbers compares only the real part

.seealso: `VecPointwiseDivide()`, `VecPointwiseMult()`, `VecPointwiseMin()`, `VecPointwiseMaxAbs()`, `VecMaxPointwiseDivide()`
@*/
PetscErrorCode  VecPointwiseMax(Vec w,Vec x,Vec y)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(w,VEC_CLASSID,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,2);
  PetscValidHeaderSpecific(y,VEC_CLASSID,3);
  PetscValidType(w,1);
  PetscValidType(x,2);
  PetscValidType(y,3);
  PetscCheckSameTypeAndComm(x,2,y,3);
  PetscCheckSameTypeAndComm(y,3,w,1);
  VecCheckSameSize(w,1,x,2);
  VecCheckSameSize(w,1,y,3);
  PetscCall(VecSetErrorIfLocked(w,1));
  PetscCall((*w->ops->pointwisemax)(w,x,y));
  PetscCall(PetscObjectStateIncrease((PetscObject)w));
  PetscFunctionReturn(0);
}

/*@
   VecPointwiseMin - Computes the componentwise minimum w_i = min(x_i, y_i).

   Logically Collective on Vec

   Input Parameters:
.  x, y  - the vectors

   Output Parameter:
.  w - the result

   Level: advanced

   Notes:
    any subset of the x, y, and w may be the same vector.
          For complex numbers compares only the real part

.seealso: `VecPointwiseDivide()`, `VecPointwiseMult()`, `VecPointwiseMin()`, `VecPointwiseMaxAbs()`, `VecMaxPointwiseDivide()`
@*/
PetscErrorCode  VecPointwiseMin(Vec w,Vec x,Vec y)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(w,VEC_CLASSID,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,2);
  PetscValidHeaderSpecific(y,VEC_CLASSID,3);
  PetscValidType(w,1);
  PetscValidType(x,2);
  PetscValidType(y,3);
  PetscCheckSameTypeAndComm(x,2,y,3);
  PetscCheckSameTypeAndComm(y,3,w,1);
  VecCheckSameSize(w,1,x,2);
  VecCheckSameSize(w,1,y,3);
  PetscCall(VecSetErrorIfLocked(w,1));
  PetscCall((*w->ops->pointwisemin)(w,x,y));
  PetscCall(PetscObjectStateIncrease((PetscObject)w));
  PetscFunctionReturn(0);
}

/*@
   VecPointwiseMaxAbs - Computes the componentwise maximum of the absolute values w_i = max(abs(x_i), abs(y_i)).

   Logically Collective on Vec

   Input Parameters:
.  x, y  - the vectors

   Output Parameter:
.  w - the result

   Level: advanced

   Notes:
    any subset of the x, y, and w may be the same vector.

.seealso: `VecPointwiseDivide()`, `VecPointwiseMult()`, `VecPointwiseMin()`, `VecPointwiseMax()`, `VecMaxPointwiseDivide()`
@*/
PetscErrorCode  VecPointwiseMaxAbs(Vec w,Vec x,Vec y)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(w,VEC_CLASSID,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,2);
  PetscValidHeaderSpecific(y,VEC_CLASSID,3);
  PetscValidType(w,1);
  PetscValidType(x,2);
  PetscValidType(y,3);
  PetscCheckSameTypeAndComm(x,2,y,3);
  PetscCheckSameTypeAndComm(y,3,w,1);
  VecCheckSameSize(w,1,x,2);
  VecCheckSameSize(w,1,y,3);
  PetscCall(VecSetErrorIfLocked(w,1));
  PetscCall((*w->ops->pointwisemaxabs)(w,x,y));
  PetscCall(PetscObjectStateIncrease((PetscObject)w));
  PetscFunctionReturn(0);
}

/*@
   VecPointwiseDivide - Computes the componentwise division w = x/y.

   Logically Collective on Vec

   Input Parameters:
.  x, y  - the vectors

   Output Parameter:
.  w - the result

   Level: advanced

   Notes:
    any subset of the x, y, and w may be the same vector.

.seealso: `VecPointwiseMult()`, `VecPointwiseMax()`, `VecPointwiseMin()`, `VecPointwiseMaxAbs()`, `VecMaxPointwiseDivide()`
@*/
PetscErrorCode  VecPointwiseDivide(Vec w,Vec x,Vec y)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(w,VEC_CLASSID,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,2);
  PetscValidHeaderSpecific(y,VEC_CLASSID,3);
  PetscValidType(w,1);
  PetscValidType(x,2);
  PetscValidType(y,3);
  PetscCheckSameTypeAndComm(x,2,y,3);
  PetscCheckSameTypeAndComm(y,3,w,1);
  VecCheckSameSize(w,1,x,2);
  VecCheckSameSize(w,1,y,3);
  PetscCall(VecSetErrorIfLocked(w,1));
  PetscCall((*w->ops->pointwisedivide)(w,x,y));
  PetscCall(PetscObjectStateIncrease((PetscObject)w));
  PetscFunctionReturn(0);
}

/*@
   VecDuplicate - Creates a new vector of the same type as an existing vector.

   Collective on Vec

   Input Parameters:
.  v - a vector to mimic

   Output Parameter:
.  newv - location to put new vector

   Notes:
   VecDuplicate() DOES NOT COPY the vector entries, but rather allocates storage
   for the new vector.  Use VecCopy() to copy a vector.

   Use VecDestroy() to free the space. Use VecDuplicateVecs() to get several
   vectors.

   Level: beginner

.seealso: `VecDestroy()`, `VecDuplicateVecs()`, `VecCreate()`, `VecCopy()`
@*/
PetscErrorCode  VecDuplicate(Vec v,Vec *newv)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,1);
  PetscValidPointer(newv,2);
  PetscValidType(v,1);
  PetscCall((*v->ops->duplicate)(v,newv));
#if defined(PETSC_HAVE_VIENNACL) || defined(PETSC_HAVE_CUDA)
  if (v->boundtocpu && v->bindingpropagates) {
    PetscCall(VecSetBindingPropagates(*newv,PETSC_TRUE));
    PetscCall(VecBindToCPU(*newv,PETSC_TRUE));
  }
#endif
  PetscCall(PetscObjectStateIncrease((PetscObject)*newv));
  PetscFunctionReturn(0);
}

/*@C
   VecDestroy - Destroys a vector.

   Collective on Vec

   Input Parameters:
.  v  - the vector

   Level: beginner

.seealso: `VecDuplicate()`, `VecDestroyVecs()`
@*/
PetscErrorCode  VecDestroy(Vec *v)
{
  PetscFunctionBegin;
  if (!*v) PetscFunctionReturn(0);
  PetscValidHeaderSpecific((*v),VEC_CLASSID,1);
  if (--((PetscObject)(*v))->refct > 0) {*v = NULL; PetscFunctionReturn(0);}

  PetscCall(PetscObjectSAWsViewOff((PetscObject)*v));
  /* destroy the internal part */
  if ((*v)->ops->destroy) {
    PetscCall((*(*v)->ops->destroy)(*v));
  }
  PetscCall(PetscFree((*v)->defaultrandtype));
  /* destroy the external/common part */
  PetscCall(PetscLayoutDestroy(&(*v)->map));
  PetscCall(PetscHeaderDestroy(v));
  PetscFunctionReturn(0);
}

/*@C
   VecDuplicateVecs - Creates several vectors of the same type as an existing vector.

   Collective on Vec

   Input Parameters:
+  m - the number of vectors to obtain
-  v - a vector to mimic

   Output Parameter:
.  V - location to put pointer to array of vectors

   Notes:
   Use VecDestroyVecs() to free the space. Use VecDuplicate() to form a single
   vector.

   Fortran Note:
   The Fortran interface is slightly different from that given below, it
   requires one to pass in V a Vec (integer) array of size at least m.
   See the Fortran chapter of the users manual and petsc/src/vec/vec/examples for details.

   Level: intermediate

.seealso: `VecDestroyVecs()`, `VecDuplicate()`, `VecCreate()`, `VecDuplicateVecsF90()`
@*/
PetscErrorCode  VecDuplicateVecs(Vec v,PetscInt m,Vec *V[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,1);
  PetscValidPointer(V,3);
  PetscValidType(v,1);
  PetscCall((*v->ops->duplicatevecs)(v,m,V));
#if defined(PETSC_HAVE_VIENNACL) || defined(PETSC_HAVE_CUDA)
  if (v->boundtocpu && v->bindingpropagates) {
    PetscInt i;

    for (i=0; i<m; i++) {
      /* Since ops->duplicatevecs might itself propagate the value of boundtocpu,
       * avoid unnecessary overhead by only calling VecBindToCPU() if the vector isn't already bound. */
      if (!(*V)[i]->boundtocpu) {
        PetscCall(VecSetBindingPropagates((*V)[i],PETSC_TRUE));
        PetscCall(VecBindToCPU((*V)[i],PETSC_TRUE));
      }
    }
  }
#endif
  PetscFunctionReturn(0);
}

/*@C
   VecDestroyVecs - Frees a block of vectors obtained with VecDuplicateVecs().

   Collective on Vec

   Input Parameters:
+  vv - pointer to pointer to array of vector pointers, if NULL no vectors are destroyed
-  m - the number of vectors previously obtained, if zero no vectors are destroyed

   Fortran Note:
   The Fortran interface is slightly different from that given below.
   See the Fortran chapter of the users manual

   Level: intermediate

.seealso: `VecDuplicateVecs()`, `VecDestroyVecsf90()`
@*/
PetscErrorCode  VecDestroyVecs(PetscInt m,Vec *vv[])
{
  PetscFunctionBegin;
  PetscValidPointer(vv,2);
  PetscCheck(m >= 0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Trying to destroy negative number of vectors %" PetscInt_FMT,m);
  if (!m || !*vv) {*vv  = NULL; PetscFunctionReturn(0);}
  PetscValidHeaderSpecific(**vv,VEC_CLASSID,2);
  PetscValidType(**vv,2);
  PetscCall((*(**vv)->ops->destroyvecs)(m,*vv));
  *vv  = NULL;
  PetscFunctionReturn(0);
}

/*@C
   VecViewFromOptions - View from Options

   Collective on Vec

   Input Parameters:
+  A - the vector
.  obj - Optional object
-  name - command line option

   Level: intermediate
.seealso: `Vec`, `VecView`, `PetscObjectViewFromOptions()`, `VecCreate()`
@*/
PetscErrorCode  VecViewFromOptions(Vec A,PetscObject obj,const char name[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,VEC_CLASSID,1);
  PetscCall(PetscObjectViewFromOptions((PetscObject)A,obj,name));
  PetscFunctionReturn(0);
}

/*@C
   VecView - Views a vector object.

   Collective on Vec

   Input Parameters:
+  vec - the vector
-  viewer - an optional visualization context

   Notes:
   The available visualization contexts include
+     PETSC_VIEWER_STDOUT_SELF - for sequential vectors
.     PETSC_VIEWER_STDOUT_WORLD - for parallel vectors created on PETSC_COMM_WORLD
-     PETSC_VIEWER_STDOUT_(comm) - for parallel vectors created on MPI communicator comm

   You can change the format the vector is printed using the
   option PetscViewerPushFormat().

   The user can open alternative viewers with
+    PetscViewerASCIIOpen() - Outputs vector to a specified file
.    PetscViewerBinaryOpen() - Outputs vector in binary to a
         specified file; corresponding input uses VecLoad()
.    PetscViewerDrawOpen() - Outputs vector to an X window display
.    PetscViewerSocketOpen() - Outputs vector to Socket viewer
-    PetscViewerHDF5Open() - Outputs vector to HDF5 file viewer

   The user can call PetscViewerPushFormat() to specify the output
   format of ASCII printed objects (when using PETSC_VIEWER_STDOUT_SELF,
   PETSC_VIEWER_STDOUT_WORLD and PetscViewerASCIIOpen).  Available formats include
+    PETSC_VIEWER_DEFAULT - default, prints vector contents
.    PETSC_VIEWER_ASCII_MATLAB - prints vector contents in MATLAB format
.    PETSC_VIEWER_ASCII_INDEX - prints vector contents, including indices of vector elements
-    PETSC_VIEWER_ASCII_COMMON - prints vector contents, using a
         format common among all vector types

   Notes:
    You can pass any number of vector objects, or other PETSc objects to the same viewer.

    In the debugger you can do "call VecView(v,0)" to display the vector. (The same holds for any PETSc object viewer).

   Notes for binary viewer:
     If you pass multiple vectors to a binary viewer you can read them back in in the same order
     with VecLoad().

     If the blocksize of the vector is greater than one then you must provide a unique prefix to
     the vector with PetscObjectSetOptionsPrefix((PetscObject)vec,"uniqueprefix"); BEFORE calling VecView() on the
     vector to be stored and then set that same unique prefix on the vector that you pass to VecLoad(). The blocksize
     information is stored in an ASCII file with the same name as the binary file plus a ".info" appended to the
     filename. If you copy the binary file, make sure you copy the associated .info file with it.

     See the manual page for VecLoad() on the exact format the binary viewer stores
     the values in the file.

   Notes for HDF5 Viewer:
     The name of the Vec (given with PetscObjectSetName() is the name that is used
     for the object in the HDF5 file. If you wish to store the same Vec into multiple
     datasets in the same file (typically with different values), you must change its
     name each time before calling the VecView(). To load the same vector,
     the name of the Vec object passed to VecLoad() must be the same.

     If the block size of the vector is greater than 1 then it is used as the first dimension in the HDF5 array.
     If the function PetscViewerHDF5SetBaseDimension2()is called then even if the block size is one it will
     be used as the first dimension in the HDF5 array (that is the HDF5 array will always be two dimensional)
     See also PetscViewerHDF5SetTimestep() which adds an additional complication to reading and writing Vecs
     with the HDF5 viewer.

   Level: beginner

.seealso: `PetscViewerASCIIOpen()`, `PetscViewerDrawOpen()`, `PetscDrawLGCreate()`,
          `PetscViewerSocketOpen()`, `PetscViewerBinaryOpen()`, `VecLoad()`, `PetscViewerCreate()`,
          `PetscRealView()`, `PetscScalarView()`, `PetscIntView()`, `PetscViewerHDF5SetTimestep()`
@*/
PetscErrorCode  VecView(Vec vec,PetscViewer viewer)
{
  PetscBool         iascii;
  PetscViewerFormat format;
  PetscMPIInt       size;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(vec,VEC_CLASSID,1);
  PetscValidType(vec,1);
  if (!viewer) PetscCall(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)vec),&viewer));
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCall(PetscViewerGetFormat(viewer,&format));
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)vec),&size));
  if (size == 1 && format == PETSC_VIEWER_LOAD_BALANCE) PetscFunctionReturn(0);

  PetscCheck(!vec->stash.n && !vec->bstash.n,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call VecAssemblyBegin/End() before viewing this vector");

  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  if (iascii) {
    PetscInt rows,bs;

    PetscCall(PetscObjectPrintClassNamePrefixType((PetscObject)vec,viewer));
    if (format == PETSC_VIEWER_ASCII_INFO || format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
      PetscCall(PetscViewerASCIIPushTab(viewer));
      PetscCall(VecGetSize(vec,&rows));
      PetscCall(VecGetBlockSize(vec,&bs));
      if (bs != 1) {
        PetscCall(PetscViewerASCIIPrintf(viewer,"length=%" PetscInt_FMT ", bs=%" PetscInt_FMT "\n",rows,bs));
      } else {
        PetscCall(PetscViewerASCIIPrintf(viewer,"length=%" PetscInt_FMT "\n",rows));
      }
      PetscCall(PetscViewerASCIIPopTab(viewer));
    }
  }
  PetscCall(VecLockReadPush(vec));
  PetscCall(PetscLogEventBegin(VEC_View,vec,viewer,0,0));
  if ((format == PETSC_VIEWER_NATIVE || format == PETSC_VIEWER_LOAD_BALANCE) && vec->ops->viewnative) {
    PetscCall((*vec->ops->viewnative)(vec,viewer));
  } else {
    PetscCall((*vec->ops->view)(vec,viewer));
  }
  PetscCall(VecLockReadPop(vec));
  PetscCall(PetscLogEventEnd(VEC_View,vec,viewer,0,0));
  PetscFunctionReturn(0);
}

#if defined(PETSC_USE_DEBUG)
#include <../src/sys/totalview/tv_data_display.h>
PETSC_UNUSED static int TV_display_type(const struct _p_Vec *v)
{
  const PetscScalar *values;
  char              type[32];

  TV_add_row("Local rows", "int", &v->map->n);
  TV_add_row("Global rows", "int", &v->map->N);
  TV_add_row("Typename", TV_ascii_string_type, ((PetscObject)v)->type_name);
  PetscCall(VecGetArrayRead((Vec)v,&values));
  PetscCall(PetscSNPrintf(type,32,"double[%" PetscInt_FMT "]",v->map->n));
  TV_add_row("values",type, values);
  PetscCall(VecRestoreArrayRead((Vec)v,&values));
  return TV_format_OK;
}
#endif

/*@C
   VecViewNative - Views a vector object with the original type specific viewer

   Collective on Vec

   Input Parameters:
+  vec - the vector
-  viewer - an optional visualization context

   Level: developer

.seealso: `PetscViewerASCIIOpen()`, `PetscViewerDrawOpen()`, `PetscDrawLGCreate()`, `VecView()`
          `PetscViewerSocketOpen()`, `PetscViewerBinaryOpen()`, `VecLoad()`, `PetscViewerCreate()`,
          `PetscRealView()`, `PetscScalarView()`, `PetscIntView()`, `PetscViewerHDF5SetTimestep()`
@*/
PetscErrorCode  VecViewNative(Vec vec,PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(vec,VEC_CLASSID,1);
  PetscValidType(vec,1);
  if (!viewer) PetscCall(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)vec),&viewer));
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCall((*vec->ops->viewnative)(vec,viewer));
  PetscFunctionReturn(0);
}

/*@
   VecGetSize - Returns the global number of elements of the vector.

   Not Collective

   Input Parameter:
.  x - the vector

   Output Parameters:
.  size - the global length of the vector

   Level: beginner

.seealso: `VecGetLocalSize()`
@*/
PetscErrorCode  VecGetSize(Vec x,PetscInt *size)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_CLASSID,1);
  PetscValidIntPointer(size,2);
  PetscValidType(x,1);
  PetscCall((*x->ops->getsize)(x,size));
  PetscFunctionReturn(0);
}

/*@
   VecGetLocalSize - Returns the number of elements of the vector stored
   in local memory.

   Not Collective

   Input Parameter:
.  x - the vector

   Output Parameter:
.  size - the length of the local piece of the vector

   Level: beginner

.seealso: `VecGetSize()`
@*/
PetscErrorCode  VecGetLocalSize(Vec x,PetscInt *size)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_CLASSID,1);
  PetscValidIntPointer(size,2);
  PetscValidType(x,1);
  PetscCall((*x->ops->getlocalsize)(x,size));
  PetscFunctionReturn(0);
}

/*@C
   VecGetOwnershipRange - Returns the range of indices owned by
   this processor, assuming that the vectors are laid out with the
   first n1 elements on the first processor, next n2 elements on the
   second, etc.  For certain parallel layouts this range may not be
   well defined.

   Not Collective

   Input Parameter:
.  x - the vector

   Output Parameters:
+  low - the first local element, pass in NULL if not interested
-  high - one more than the last local element, pass in NULL if not interested

   Note:
   The high argument is one more than the last element stored locally.

   Fortran: PETSC_NULL_INTEGER should be used instead of NULL

   Level: beginner

.seealso: `MatGetOwnershipRange()`, `MatGetOwnershipRanges()`, `VecGetOwnershipRanges()`
@*/
PetscErrorCode  VecGetOwnershipRange(Vec x,PetscInt *low,PetscInt *high)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_CLASSID,1);
  PetscValidType(x,1);
  if (low) PetscValidIntPointer(low,2);
  if (high) PetscValidIntPointer(high,3);
  if (low)  *low  = x->map->rstart;
  if (high) *high = x->map->rend;
  PetscFunctionReturn(0);
}

/*@C
   VecGetOwnershipRanges - Returns the range of indices owned by EACH processor,
   assuming that the vectors are laid out with the
   first n1 elements on the first processor, next n2 elements on the
   second, etc.  For certain parallel layouts this range may not be
   well defined.

   Not Collective

   Input Parameter:
.  x - the vector

   Output Parameters:
.  range - array of length size+1 with the start and end+1 for each process

   Note:
   The high argument is one more than the last element stored locally.

   Fortran: You must PASS in an array of length size+1

   If the ranges are used after all vectors that share the ranges has been destroyed then the program will crash accessing ranges[].

   Level: beginner

.seealso: `MatGetOwnershipRange()`, `MatGetOwnershipRanges()`, `VecGetOwnershipRange()`
@*/
PetscErrorCode  VecGetOwnershipRanges(Vec x,const PetscInt *ranges[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_CLASSID,1);
  PetscValidType(x,1);
  PetscCall(PetscLayoutGetRanges(x->map,ranges));
  PetscFunctionReturn(0);
}

/*@
   VecSetOption - Sets an option for controling a vector's behavior.

   Collective on Vec

   Input Parameters:
+  x - the vector
.  op - the option
-  flag - turn the option on or off

   Supported Options:
+     VEC_IGNORE_OFF_PROC_ENTRIES, which causes VecSetValues() to ignore
          entries destined to be stored on a separate processor. This can be used
          to eliminate the global reduction in the VecAssemblyXXXX() if you know
          that you have only used VecSetValues() to set local elements
.     VEC_IGNORE_NEGATIVE_INDICES, which means you can pass negative indices
          in ix in calls to VecSetValues() or VecGetValues(). These rows are simply
          ignored.
-     VEC_SUBSET_OFF_PROC_ENTRIES, which causes VecAssemblyBegin() to assume that the off-process
          entries will always be a subset (possibly equal) of the off-process entries set on the
          first assembly which had a true VEC_SUBSET_OFF_PROC_ENTRIES and the vector has not
          changed this flag afterwards. If this assembly is not such first assembly, then this
          assembly can reuse the communication pattern setup in that first assembly, thus avoiding
          a global reduction. Subsequent assemblies setting off-process values should use the same
          InsertMode as the first assembly.

   Developer Note:
   The InsertMode restriction could be removed by packing the stash messages out of place.

   Level: intermediate

@*/
PetscErrorCode  VecSetOption(Vec x,VecOption op,PetscBool flag)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_CLASSID,1);
  PetscValidType(x,1);
  if (x->ops->setoption) {
    PetscCall((*x->ops->setoption)(x,op,flag));
  }
  PetscFunctionReturn(0);
}

/* Default routines for obtaining and releasing; */
/* may be used by any implementation */
PetscErrorCode VecDuplicateVecs_Default(Vec w,PetscInt m,Vec *V[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(w,VEC_CLASSID,1);
  PetscValidPointer(V,3);
  PetscCheck(m > 0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"m must be > 0: m = %" PetscInt_FMT,m);
  PetscCall(PetscMalloc1(m,V));
  for (PetscInt i=0; i<m; i++) PetscCall(VecDuplicate(w,*V+i));
  PetscFunctionReturn(0);
}

PetscErrorCode VecDestroyVecs_Default(PetscInt m,Vec v[])
{
  PetscInt       i;

  PetscFunctionBegin;
  PetscValidPointer(v,2);
  for (i=0; i<m; i++) PetscCall(VecDestroy(&v[i]));
  PetscCall(PetscFree(v));
  PetscFunctionReturn(0);
}

/*@
   VecResetArray - Resets a vector to use its default memory. Call this
   after the use of VecPlaceArray().

   Not Collective

   Input Parameters:
.  vec - the vector

   Level: developer

.seealso: `VecGetArray()`, `VecRestoreArray()`, `VecReplaceArray()`, `VecPlaceArray()`

@*/
PetscErrorCode  VecResetArray(Vec vec)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(vec,VEC_CLASSID,1);
  PetscValidType(vec,1);
  if (vec->ops->resetarray) {
    PetscCall((*vec->ops->resetarray)(vec));
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot reset array in this type of vector");
  PetscCall(PetscObjectStateIncrease((PetscObject)vec));
  PetscFunctionReturn(0);
}

/*@C
  VecLoad - Loads a vector that has been stored in binary or HDF5 format
  with VecView().

  Collective on PetscViewer

  Input Parameters:
+ vec - the newly loaded vector, this needs to have been created with VecCreate() or
           some related function before a call to VecLoad().
- viewer - binary file viewer, obtained from PetscViewerBinaryOpen() or
           HDF5 file viewer, obtained from PetscViewerHDF5Open()

   Level: intermediate

  Notes:
  Defaults to the standard Seq or MPI Vec, if you want some other type of Vec call VecSetFromOptions()
  before calling this.

  The input file must contain the full global vector, as
  written by the routine VecView().

  If the type or size of vec is not set before a call to VecLoad, PETSc
  sets the type and the local and global sizes. If type and/or
  sizes are already set, then the same are used.

  If using the binary viewer and the blocksize of the vector is greater than one then you must provide a unique prefix to
  the vector with PetscObjectSetOptionsPrefix((PetscObject)vec,"uniqueprefix"); BEFORE calling VecView() on the
  vector to be stored and then set that same unique prefix on the vector that you pass to VecLoad(). The blocksize
  information is stored in an ASCII file with the same name as the binary file plus a ".info" appended to the
  filename. If you copy the binary file, make sure you copy the associated .info file with it.

  If using HDF5, you must assign the Vec the same name as was used in the Vec
  that was stored in the file using PetscObjectSetName(). Otherwise you will
  get the error message: "Cannot H5DOpen2() with Vec name NAMEOFOBJECT".

  If the HDF5 file contains a two dimensional array the first dimension is treated as the block size
  in loading the vector. Hence, for example, using Matlab notation h5create('vector.dat','/Test_Vec',[27 1]);
  will load a vector of size 27 and block size 27 thus resulting in all 27 entries being on the first process of
  vectors communicator and the rest of the processes having zero entries

  Notes for advanced users when using the binary viewer:
  Most users should not need to know the details of the binary storage
  format, since VecLoad() and VecView() completely hide these details.
  But for anyone who's interested, the standard binary vector storage
  format is
.vb
     PetscInt    VEC_FILE_CLASSID
     PetscInt    number of rows
     PetscScalar *values of all entries
.ve

   In addition, PETSc automatically uses byte swapping to work on all machines; the files
   are written ALWAYS using big-endian ordering. On small-endian machines the numbers
   are converted to the small-endian format when they are read in from the file.
   See PetscBinaryRead() and PetscBinaryWrite() to see how this may be done.

.seealso: `PetscViewerBinaryOpen()`, `VecView()`, `MatLoad()`, `VecLoad()`
@*/
PetscErrorCode  VecLoad(Vec vec, PetscViewer viewer)
{
  PetscBool         isbinary,ishdf5,isadios,isexodusii;
  PetscViewerFormat format;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(vec,VEC_CLASSID,1);
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(vec,1,viewer,2);
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERBINARY,&isbinary));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERHDF5,&ishdf5));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERADIOS,&isadios));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWEREXODUSII,&isexodusii));
  PetscCheck(isbinary || ishdf5 || isadios || isexodusii,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Invalid viewer; open viewer with PetscViewerBinaryOpen()");

  PetscCall(VecSetErrorIfLocked(vec,1));
  if (!((PetscObject)vec)->type_name && !vec->ops->create) PetscCall(VecSetType(vec, VECSTANDARD));
  PetscCall(PetscLogEventBegin(VEC_Load,viewer,0,0,0));
  PetscCall(PetscViewerGetFormat(viewer,&format));
  if (format == PETSC_VIEWER_NATIVE && vec->ops->loadnative) {
    PetscCall((*vec->ops->loadnative)(vec,viewer));
  } else {
    PetscCall((*vec->ops->load)(vec,viewer));
  }
  PetscCall(PetscLogEventEnd(VEC_Load,viewer,0,0,0));
  PetscFunctionReturn(0);
}

/*@
   VecReciprocal - Replaces each component of a vector by its reciprocal.

   Logically Collective on Vec

   Input Parameter:
.  vec - the vector

   Output Parameter:
.  vec - the vector reciprocal

   Level: intermediate

.seealso: `VecLog()`, `VecExp()`, `VecSqrtAbs()`

@*/
PetscErrorCode  VecReciprocal(Vec vec)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(vec,VEC_CLASSID,1);
  PetscValidType(vec,1);
  PetscCheck(vec->stash.insertmode == NOT_SET_VALUES,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled vector");
  PetscCheck(vec->ops->reciprocal,PETSC_COMM_SELF,PETSC_ERR_SUP,"Vector does not support reciprocal operation");
  PetscCall(VecSetErrorIfLocked(vec,1));
  PetscCall((*vec->ops->reciprocal)(vec));
  PetscCall(PetscObjectStateIncrease((PetscObject)vec));
  PetscFunctionReturn(0);
}

/*@C
    VecSetOperation - Allows user to set a vector operation.

   Logically Collective on Vec

    Input Parameters:
+   vec - the vector
.   op - the name of the operation
-   f - the function that provides the operation.

   Level: advanced

    Usage:
$      PetscErrorCode userview(Vec,PetscViewer);
$      PetscCall(VecCreateMPI(comm,m,M,&x));
$      PetscCall(VecSetOperation(x,VECOP_VIEW,(void(*)(void))userview));

    Notes:
    See the file include/petscvec.h for a complete list of matrix
    operations, which all have the form VECOP_<OPERATION>, where
    <OPERATION> is the name (in all capital letters) of the
    user interface routine (e.g., VecView() -> VECOP_VIEW).

    This function is not currently available from Fortran.

.seealso: `VecCreate()`, `MatShellSetOperation()`
@*/
PetscErrorCode VecSetOperation(Vec vec,VecOperation op, void (*f)(void))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(vec,VEC_CLASSID,1);
  if (op == VECOP_VIEW && !vec->ops->viewnative) {
    vec->ops->viewnative = vec->ops->view;
  } else if (op == VECOP_LOAD && !vec->ops->loadnative) {
    vec->ops->loadnative = vec->ops->load;
  }
  (((void(**)(void))vec->ops)[(int)op]) = f;
  PetscFunctionReturn(0);
}

/*@
   VecStashSetInitialSize - sets the sizes of the vec-stash, that is
   used during the assembly process to store values that belong to
   other processors.

   Not Collective, different processes can have different size stashes

   Input Parameters:
+  vec   - the vector
.  size  - the initial size of the stash.
-  bsize - the initial size of the block-stash(if used).

   Options Database Keys:
+   -vecstash_initial_size <size> or <size0,size1,...sizep-1>
-   -vecstash_block_initial_size <bsize> or <bsize0,bsize1,...bsizep-1>

   Level: intermediate

   Notes:
     The block-stash is used for values set with VecSetValuesBlocked() while
     the stash is used for values set with VecSetValues()

     Run with the option -info and look for output of the form
     VecAssemblyBegin_MPIXXX:Stash has MM entries, uses nn mallocs.
     to determine the appropriate value, MM, to use for size and
     VecAssemblyBegin_MPIXXX:Block-Stash has BMM entries, uses nn mallocs.
     to determine the value, BMM to use for bsize

.seealso: `VecSetBlockSize()`, `VecSetValues()`, `VecSetValuesBlocked()`, `VecStashView()`

@*/
PetscErrorCode  VecStashSetInitialSize(Vec vec,PetscInt size,PetscInt bsize)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(vec,VEC_CLASSID,1);
  PetscCall(VecStashSetInitialSize_Private(&vec->stash,size));
  PetscCall(VecStashSetInitialSize_Private(&vec->bstash,bsize));
  PetscFunctionReturn(0);
}

/*@
   VecConjugate - Conjugates a vector.

   Logically Collective on Vec

   Input Parameters:
.  x - the vector

   Level: intermediate

@*/
PetscErrorCode  VecConjugate(Vec x)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_CLASSID,1);
  PetscValidType(x,1);
  PetscCheck(x->stash.insertmode == NOT_SET_VALUES,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled vector");
  if (PetscDefined(USE_COMPLEX)) {
    PetscCall(VecSetErrorIfLocked(x,1));
    PetscCall((*x->ops->conjugate)(x));
    /* we need to copy norms here */
    PetscCall(PetscObjectStateIncrease((PetscObject)x));
  }
  PetscFunctionReturn(0);
}

/*@
   VecPointwiseMult - Computes the componentwise multiplication w = x*y.

   Logically Collective on Vec

   Input Parameters:
.  x, y  - the vectors

   Output Parameter:
.  w - the result

   Level: advanced

   Notes:
    any subset of the x, y, and w may be the same vector.

.seealso: `VecPointwiseDivide()`, `VecPointwiseMax()`, `VecPointwiseMin()`, `VecPointwiseMaxAbs()`, `VecMaxPointwiseDivide()`
@*/
PetscErrorCode  VecPointwiseMult(Vec w,Vec x,Vec y)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(w,VEC_CLASSID,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,2);
  PetscValidHeaderSpecific(y,VEC_CLASSID,3);
  PetscValidType(w,1);
  PetscValidType(x,2);
  PetscValidType(y,3);
  PetscCheckSameTypeAndComm(x,2,y,3);
  PetscCheckSameTypeAndComm(y,3,w,1);
  VecCheckSameSize(w,1,x,2);
  VecCheckSameSize(w,2,y,3);
  PetscCall(VecSetErrorIfLocked(w,1));
  PetscCall(PetscLogEventBegin(VEC_PointwiseMult,x,y,w,0));
  PetscCall((*w->ops->pointwisemult)(w,x,y));
  PetscCall(PetscLogEventEnd(VEC_PointwiseMult,x,y,w,0));
  PetscCall(PetscObjectStateIncrease((PetscObject)w));
  PetscFunctionReturn(0);
}

/*@
   VecSetRandom - Sets all components of a vector to random numbers.

   Logically Collective on Vec

   Input Parameters:
+  x  - the vector
-  rctx - the random number context, formed by PetscRandomCreate(), or NULL and
          it will create one internally.

   Output Parameter:
.  x  - the vector

   Example of Usage:
.vb
     PetscRandomCreate(PETSC_COMM_WORLD,&rctx);
     VecSetRandom(x,rctx);
     PetscRandomDestroy(&rctx);
.ve

   Level: intermediate

.seealso: `VecSet()`, `VecSetValues()`, `PetscRandomCreate()`, `PetscRandomDestroy()`
@*/
PetscErrorCode  VecSetRandom(Vec x,PetscRandom rctx)
{
  PetscRandom randObj = NULL;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_CLASSID,1);
  if (rctx) PetscValidHeaderSpecific(rctx,PETSC_RANDOM_CLASSID,2);
  PetscValidType(x,1);
  PetscCheck(x->stash.insertmode == NOT_SET_VALUES,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled vector");
  PetscCall(VecSetErrorIfLocked(x,1));

  if (!rctx) {
    PetscCall(PetscRandomCreate(PetscObjectComm((PetscObject)x),&randObj));
    PetscCall(PetscRandomSetType(randObj,x->defaultrandtype));
    PetscCall(PetscRandomSetFromOptions(randObj));
    rctx = randObj;
  }

  PetscCall(PetscLogEventBegin(VEC_SetRandom,x,rctx,0,0));
  PetscCall((*x->ops->setrandom)(x,rctx));
  PetscCall(PetscLogEventEnd(VEC_SetRandom,x,rctx,0,0));

  PetscCall(PetscRandomDestroy(&randObj));
  PetscCall(PetscObjectStateIncrease((PetscObject)x));
  PetscFunctionReturn(0);
}

/*@
  VecZeroEntries - puts a 0.0 in each element of a vector

  Logically Collective on Vec

  Input Parameter:
. vec - The vector

  Level: beginner

.seealso: `VecCreate()`, `VecSetOptionsPrefix()`, `VecSet()`, `VecSetValues()`
@*/
PetscErrorCode  VecZeroEntries(Vec vec)
{
  PetscFunctionBegin;
  PetscCall(VecSet(vec,0));
  PetscFunctionReturn(0);
}

/*
  VecSetTypeFromOptions_Private - Sets the type of vector from user options. Defaults to a PETSc sequential vector on one
  processor and a PETSc MPI vector on more than one processor.

  Collective on Vec

  Input Parameter:
. vec - The vector

  Level: intermediate

.seealso: `VecSetFromOptions()`, `VecSetType()`
*/
static PetscErrorCode VecSetTypeFromOptions_Private(PetscOptionItems *PetscOptionsObject,Vec vec)
{
  PetscBool      opt;
  VecType        defaultType;
  char           typeName[256];
  PetscMPIInt    size;

  PetscFunctionBegin;
  if (((PetscObject)vec)->type_name) defaultType = ((PetscObject)vec)->type_name;
  else {
    PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)vec), &size));
    if (size > 1) defaultType = VECMPI;
    else defaultType = VECSEQ;
  }

  PetscCall(VecRegisterAll());
  PetscCall(PetscOptionsFList("-vec_type","Vector type","VecSetType",VecList,defaultType,typeName,256,&opt));
  if (opt) {
    PetscCall(VecSetType(vec, typeName));
  } else {
    PetscCall(VecSetType(vec, defaultType));
  }
  PetscFunctionReturn(0);
}

/*@
  VecSetFromOptions - Configures the vector from the options database.

  Collective on Vec

  Input Parameter:
. vec - The vector

  Notes:
    To see all options, run your program with the -help option, or consult the users manual.
          Must be called after VecCreate() but before the vector is used.

  Level: beginner

.seealso: `VecCreate()`, `VecSetOptionsPrefix()`
@*/
PetscErrorCode  VecSetFromOptions(Vec vec)
{
  PetscBool      flg;
  PetscInt       bind_below = 0;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(vec,VEC_CLASSID,1);

  PetscObjectOptionsBegin((PetscObject)vec);
  /* Handle vector type options */
  PetscCall(VecSetTypeFromOptions_Private(PetscOptionsObject,vec));

  /* Handle specific vector options */
  if (vec->ops->setfromoptions) PetscCall((*vec->ops->setfromoptions)(PetscOptionsObject,vec));

  /* Bind to CPU if below a user-specified size threshold.
   * This perhaps belongs in the options for the GPU Vec types, but VecBindToCPU() does nothing when called on non-GPU types,
   * and putting it here makes is more maintainable than duplicating this for all. */
  PetscCall(PetscOptionsInt("-vec_bind_below","Set the size threshold (in local entries) below which the Vec is bound to the CPU","VecBindToCPU",bind_below,&bind_below,&flg));
  if (flg && vec->map->n < bind_below) PetscCall(VecBindToCPU(vec,PETSC_TRUE));

  /* process any options handlers added with PetscObjectAddOptionsHandler() */
  PetscCall(PetscObjectProcessOptionsHandlers(PetscOptionsObject,(PetscObject)vec));
  PetscOptionsEnd();
  PetscFunctionReturn(0);
}

/*@
  VecSetSizes - Sets the local and global sizes, and checks to determine compatibility

  Collective on Vec

  Input Parameters:
+ v - the vector
. n - the local size (or PETSC_DECIDE to have it set)
- N - the global size (or PETSC_DECIDE)

  Notes:
  n and N cannot be both PETSC_DECIDE
  If one processor calls this with N of PETSC_DECIDE then all processors must, otherwise the program will hang.

  Level: intermediate

.seealso: `VecGetSize()`, `PetscSplitOwnership()`
@*/
PetscErrorCode  VecSetSizes(Vec v, PetscInt n, PetscInt N)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(v, VEC_CLASSID,1);
  if (N >= 0) {
    PetscValidLogicalCollectiveInt(v,N,3);
    PetscCheck(n <= N,PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Local size %" PetscInt_FMT " cannot be larger than global size %" PetscInt_FMT,n,N);
  }
  PetscCheck(!(v->map->n >= 0 || v->map->N >= 0) || !(v->map->n != n || v->map->N != N),PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot change/reset vector sizes to %" PetscInt_FMT " local %" PetscInt_FMT " global after previously setting them to %" PetscInt_FMT " local %" PetscInt_FMT " global",n,N,v->map->n,v->map->N);
  v->map->n = n;
  v->map->N = N;
  if (v->ops->create) {
    PetscCall((*v->ops->create)(v));
    v->ops->create = NULL;
  }
  PetscFunctionReturn(0);
}

/*@
   VecSetBlockSize - Sets the blocksize for future calls to VecSetValuesBlocked()
   and VecSetValuesBlockedLocal().

   Logically Collective on Vec

   Input Parameters:
+  v - the vector
-  bs - the blocksize

   Notes:
   All vectors obtained by VecDuplicate() inherit the same blocksize.

   Level: advanced

.seealso: `VecSetValuesBlocked()`, `VecSetLocalToGlobalMapping()`, `VecGetBlockSize()`

@*/
PetscErrorCode  VecSetBlockSize(Vec v,PetscInt bs)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,1);
  PetscValidLogicalCollectiveInt(v,bs,2);
  PetscCall(PetscLayoutSetBlockSize(v->map,bs));
  v->bstash.bs = bs; /* use the same blocksize for the vec's block-stash */
  PetscFunctionReturn(0);
}

/*@
   VecGetBlockSize - Gets the blocksize for the vector, i.e. what is used for VecSetValuesBlocked()
   and VecSetValuesBlockedLocal().

   Not Collective

   Input Parameter:
.  v - the vector

   Output Parameter:
.  bs - the blocksize

   Notes:
   All vectors obtained by VecDuplicate() inherit the same blocksize.

   Level: advanced

.seealso: `VecSetValuesBlocked()`, `VecSetLocalToGlobalMapping()`, `VecSetBlockSize()`

@*/
PetscErrorCode  VecGetBlockSize(Vec v,PetscInt *bs)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,1);
  PetscValidIntPointer(bs,2);
  PetscCall(PetscLayoutGetBlockSize(v->map,bs));
  PetscFunctionReturn(0);
}

/*@C
   VecSetOptionsPrefix - Sets the prefix used for searching for all
   Vec options in the database.

   Logically Collective on Vec

   Input Parameters:
+  v - the Vec context
-  prefix - the prefix to prepend to all option names

   Notes:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the hyphen.

   Level: advanced

.seealso: `VecSetFromOptions()`
@*/
PetscErrorCode  VecSetOptionsPrefix(Vec v,const char prefix[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,1);
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)v,prefix));
  PetscFunctionReturn(0);
}

/*@C
   VecAppendOptionsPrefix - Appends to the prefix used for searching for all
   Vec options in the database.

   Logically Collective on Vec

   Input Parameters:
+  v - the Vec context
-  prefix - the prefix to prepend to all option names

   Notes:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the hyphen.

   Level: advanced

.seealso: `VecGetOptionsPrefix()`
@*/
PetscErrorCode  VecAppendOptionsPrefix(Vec v,const char prefix[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,1);
  PetscCall(PetscObjectAppendOptionsPrefix((PetscObject)v,prefix));
  PetscFunctionReturn(0);
}

/*@C
   VecGetOptionsPrefix - Sets the prefix used for searching for all
   Vec options in the database.

   Not Collective

   Input Parameter:
.  v - the Vec context

   Output Parameter:
.  prefix - pointer to the prefix string used

   Notes:
    On the fortran side, the user should pass in a string 'prefix' of
   sufficient length to hold the prefix.

   Level: advanced

.seealso: `VecAppendOptionsPrefix()`
@*/
PetscErrorCode  VecGetOptionsPrefix(Vec v,const char *prefix[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,1);
  PetscCall(PetscObjectGetOptionsPrefix((PetscObject)v,prefix));
  PetscFunctionReturn(0);
}

/*@
   VecSetUp - Sets up the internal vector data structures for the later use.

   Collective on Vec

   Input Parameters:
.  v - the Vec context

   Notes:
   For basic use of the Vec classes the user need not explicitly call
   VecSetUp(), since these actions will happen automatically.

   Level: advanced

.seealso: `VecCreate()`, `VecDestroy()`
@*/
PetscErrorCode  VecSetUp(Vec v)
{
  PetscMPIInt    size;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,1);
  PetscCheck(v->map->n >= 0 || v->map->N >= 0,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Sizes not set");
  if (!((PetscObject)v)->type_name) {
    PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)v), &size));
    if (size == 1) {
      PetscCall(VecSetType(v, VECSEQ));
    } else {
      PetscCall(VecSetType(v, VECMPI));
    }
  }
  PetscFunctionReturn(0);
}

/*
    These currently expose the PetscScalar/PetscReal in updating the
    cached norm. If we push those down into the implementation these
    will become independent of PetscScalar/PetscReal
*/

/*@
   VecCopy - Copies a vector. y <- x

   Logically Collective on Vec

   Input Parameter:
.  x - the vector

   Output Parameter:
.  y - the copy

   Notes:
   For default parallel PETSc vectors, both x and y must be distributed in
   the same manner; local copies are done.

   Developer Notes:
   PetscCheckSameTypeAndComm(x,1,y,2) is not used on these vectors because we allow one
   of the vectors to be sequential and one to be parallel so long as both have the same
   local sizes. This is used in some internal functions in PETSc.

   Level: beginner

.seealso: `VecDuplicate()`
@*/
PetscErrorCode  VecCopy(Vec x,Vec y)
{
  PetscBool flgs[4];
  PetscReal norms[4] = {0.0,0.0,0.0,0.0};

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_CLASSID,1);
  PetscValidHeaderSpecific(y,VEC_CLASSID,2);
  PetscValidType(x,1);
  PetscValidType(y,2);
  if (x == y) PetscFunctionReturn(0);
  VecCheckSameLocalSize(x,1,y,2);
  PetscCheck(x->stash.insertmode == NOT_SET_VALUES,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled vector");
  PetscCall(VecSetErrorIfLocked(y,2));

#if !defined(PETSC_USE_MIXED_PRECISION)
  for (PetscInt i=0; i<4; i++) PetscCall(PetscObjectComposedDataGetReal((PetscObject)x,NormIds[i],norms[i],flgs[i]));
#endif

  PetscCall(PetscLogEventBegin(VEC_Copy,x,y,0,0));
#if defined(PETSC_USE_MIXED_PRECISION)
  extern PetscErrorCode VecGetArray(Vec,double**);
  extern PetscErrorCode VecRestoreArray(Vec,double**);
  extern PetscErrorCode VecGetArray(Vec,float**);
  extern PetscErrorCode VecRestoreArray(Vec,float**);
  extern PetscErrorCode VecGetArrayRead(Vec,const double**);
  extern PetscErrorCode VecRestoreArrayRead(Vec,const double**);
  extern PetscErrorCode VecGetArrayRead(Vec,const float**);
  extern PetscErrorCode VecRestoreArrayRead(Vec,const float**);
  if ((((PetscObject)x)->precision == PETSC_PRECISION_SINGLE) && (((PetscObject)y)->precision == PETSC_PRECISION_DOUBLE)) {
    PetscInt    i,n;
    const float *xx;
    double      *yy;
    PetscCall(VecGetArrayRead(x,&xx));
    PetscCall(VecGetArray(y,&yy));
    PetscCall(VecGetLocalSize(x,&n));
    for (i=0; i<n; i++) yy[i] = xx[i];
    PetscCall(VecRestoreArrayRead(x,&xx));
    PetscCall(VecRestoreArray(y,&yy));
  } else if ((((PetscObject)x)->precision == PETSC_PRECISION_DOUBLE) && (((PetscObject)y)->precision == PETSC_PRECISION_SINGLE)) {
    PetscInt     i,n;
    float        *yy;
    const double *xx;
    PetscCall(VecGetArrayRead(x,&xx));
    PetscCall(VecGetArray(y,&yy));
    PetscCall(VecGetLocalSize(x,&n));
    for (i=0; i<n; i++) yy[i] = (float) xx[i];
    PetscCall(VecRestoreArrayRead(x,&xx));
    PetscCall(VecRestoreArray(y,&yy));
  } else {
    PetscCall((*x->ops->copy)(x,y));
  }
#else
  PetscCall((*x->ops->copy)(x,y));
#endif

  PetscCall(PetscObjectStateIncrease((PetscObject)y));
#if !defined(PETSC_USE_MIXED_PRECISION)
  for (PetscInt i=0; i<4; i++) {
    if (flgs[i]) PetscCall(PetscObjectComposedDataSetReal((PetscObject)y,NormIds[i],norms[i]));
  }
#endif

  PetscCall(PetscLogEventEnd(VEC_Copy,x,y,0,0));
  PetscFunctionReturn(0);
}

/*@
   VecSwap - Swaps the vectors x and y.

   Logically Collective on Vec

   Input Parameters:
.  x, y  - the vectors

   Level: advanced

@*/
PetscErrorCode  VecSwap(Vec x,Vec y)
{
  PetscReal normxs[4] = {0.0,0.0,0.0,0.0},normys[4]={0.0,0.0,0.0,0.0};
  PetscBool flgxs[4],flgys[4];

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_CLASSID,1);
  PetscValidHeaderSpecific(y,VEC_CLASSID,2);
  PetscValidType(x,1);
  PetscValidType(y,2);
  PetscCheckSameTypeAndComm(x,1,y,2);
  VecCheckSameSize(x,1,y,2);
  PetscCheck(x->stash.insertmode == NOT_SET_VALUES,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled vector");
  PetscCheck(y->stash.insertmode == NOT_SET_VALUES,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled vector");
  PetscCall(VecSetErrorIfLocked(x,1));
  PetscCall(VecSetErrorIfLocked(y,2));

  PetscCall(PetscLogEventBegin(VEC_Swap,x,y,0,0));
  for (PetscInt i=0; i<4; i++) {
    PetscCall(PetscObjectComposedDataGetReal((PetscObject)x,NormIds[i],normxs[i],flgxs[i]));
    PetscCall(PetscObjectComposedDataGetReal((PetscObject)y,NormIds[i],normys[i],flgys[i]));
  }
  PetscCall((*x->ops->swap)(x,y));
  PetscCall(PetscObjectStateIncrease((PetscObject)x));
  PetscCall(PetscObjectStateIncrease((PetscObject)y));
  for (PetscInt i=0; i<4; i++) {
    if (flgxs[i]) PetscCall(PetscObjectComposedDataSetReal((PetscObject)y,NormIds[i],normxs[i]));
    if (flgys[i]) PetscCall(PetscObjectComposedDataSetReal((PetscObject)x,NormIds[i],normys[i]));
  }
  PetscCall(PetscLogEventEnd(VEC_Swap,x,y,0,0));
  PetscFunctionReturn(0);
}

/*
  VecStashViewFromOptions - Processes command line options to determine if/how an VecStash object is to be viewed.

  Collective on VecStash

  Input Parameters:
+ obj   - the VecStash object
. bobj - optional other object that provides the prefix
- optionname - option to activate viewing

  Level: intermediate

  Developer Note: This cannot use PetscObjectViewFromOptions() because it takes a Vec as an argument but does not use VecView

*/
PetscErrorCode VecStashViewFromOptions(Vec obj,PetscObject bobj,const char optionname[])
{
  PetscViewer       viewer;
  PetscBool         flg;
  PetscViewerFormat format;
  char              *prefix;

  PetscFunctionBegin;
  prefix = bobj ? bobj->prefix : ((PetscObject)obj)->prefix;
  PetscCall(PetscOptionsGetViewer(PetscObjectComm((PetscObject)obj),((PetscObject)obj)->options,prefix,optionname,&viewer,&format,&flg));
  if (flg) {
    PetscCall(PetscViewerPushFormat(viewer,format));
    PetscCall(VecStashView(obj,viewer));
    PetscCall(PetscViewerPopFormat(viewer));
    PetscCall(PetscViewerDestroy(&viewer));
  }
  PetscFunctionReturn(0);
}

/*@
   VecStashView - Prints the entries in the vector stash and block stash.

   Collective on Vec

   Input Parameters:
+  v - the vector
-  viewer - the viewer

   Level: advanced

.seealso: `VecSetBlockSize()`, `VecSetValues()`, `VecSetValuesBlocked()`

@*/
PetscErrorCode  VecStashView(Vec v,PetscViewer viewer)
{
  PetscMPIInt  rank;
  PetscInt     i,j;
  PetscBool    match;
  VecStash    *s;
  PetscScalar  val;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,1);
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(v,1,viewer,2);

  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&match));
  PetscCheck(match,PETSC_COMM_SELF,PETSC_ERR_SUP,"Stash viewer only works with ASCII viewer not %s",((PetscObject)v)->type_name);
  PetscCall(PetscViewerASCIIUseTabs(viewer,PETSC_FALSE));
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)v),&rank));
  s = &v->bstash;

  /* print block stash */
  PetscCall(PetscViewerASCIIPushSynchronized(viewer));
  PetscCall(PetscViewerASCIISynchronizedPrintf(viewer,"[%d]Vector Block stash size %" PetscInt_FMT " block size %" PetscInt_FMT "\n",rank,s->n,s->bs));
  for (i=0; i<s->n; i++) {
    PetscCall(PetscViewerASCIISynchronizedPrintf(viewer,"[%d] Element %" PetscInt_FMT " ",rank,s->idx[i]));
    for (j=0; j<s->bs; j++) {
      val = s->array[i*s->bs+j];
#if defined(PETSC_USE_COMPLEX)
      PetscCall(PetscViewerASCIISynchronizedPrintf(viewer,"(%18.16e %18.16e) ",(double)PetscRealPart(val),(double)PetscImaginaryPart(val)));
#else
      PetscCall(PetscViewerASCIISynchronizedPrintf(viewer,"%18.16e ",(double)val));
#endif
    }
    PetscCall(PetscViewerASCIISynchronizedPrintf(viewer,"\n"));
  }
  PetscCall(PetscViewerFlush(viewer));

  s = &v->stash;

  /* print basic stash */
  PetscCall(PetscViewerASCIISynchronizedPrintf(viewer,"[%d]Vector stash size %" PetscInt_FMT "\n",rank,s->n));
  for (i=0; i<s->n; i++) {
    val = s->array[i];
#if defined(PETSC_USE_COMPLEX)
    PetscCall(PetscViewerASCIISynchronizedPrintf(viewer,"[%d] Element %" PetscInt_FMT " (%18.16e %18.16e) ",rank,s->idx[i],(double)PetscRealPart(val),(double)PetscImaginaryPart(val)));
#else
    PetscCall(PetscViewerASCIISynchronizedPrintf(viewer,"[%d] Element %" PetscInt_FMT " %18.16e\n",rank,s->idx[i],(double)val));
#endif
  }
  PetscCall(PetscViewerFlush(viewer));
  PetscCall(PetscViewerASCIIPopSynchronized(viewer));
  PetscCall(PetscViewerASCIIUseTabs(viewer,PETSC_TRUE));
  PetscFunctionReturn(0);
}

PetscErrorCode PetscOptionsGetVec(PetscOptions options,const char prefix[],const char key[],Vec v,PetscBool *set)
{
  PetscInt       i,N,rstart,rend;
  PetscScalar    *xx;
  PetscReal      *xreal;
  PetscBool      iset;

  PetscFunctionBegin;
  PetscCall(VecGetOwnershipRange(v,&rstart,&rend));
  PetscCall(VecGetSize(v,&N));
  PetscCall(PetscCalloc1(N,&xreal));
  PetscCall(PetscOptionsGetRealArray(options,prefix,key,xreal,&N,&iset));
  if (iset) {
    PetscCall(VecGetArray(v,&xx));
    for (i=rstart; i<rend; i++) xx[i-rstart] = xreal[i];
    PetscCall(VecRestoreArray(v,&xx));
  }
  PetscCall(PetscFree(xreal));
  if (set) *set = iset;
  PetscFunctionReturn(0);
}

/*@
   VecGetLayout - get PetscLayout describing vector layout

   Not Collective

   Input Parameter:
.  x - the vector

   Output Parameter:
.  map - the layout

   Level: developer

.seealso: `VecGetSizes()`, `VecGetOwnershipRange()`, `VecGetOwnershipRanges()`
@*/
PetscErrorCode VecGetLayout(Vec x,PetscLayout *map)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_CLASSID,1);
  PetscValidPointer(map,2);
  *map = x->map;
  PetscFunctionReturn(0);
}

/*@
   VecSetLayout - set PetscLayout describing vector layout

   Not Collective

   Input Parameters:
+  x - the vector
-  map - the layout

   Notes:
   It is normally only valid to replace the layout with a layout known to be equivalent.

   Level: developer

.seealso: `VecGetLayout()`, `VecGetSizes()`, `VecGetOwnershipRange()`, `VecGetOwnershipRanges()`
@*/
PetscErrorCode VecSetLayout(Vec x,PetscLayout map)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_CLASSID,1);
  PetscCall(PetscLayoutReference(map,&x->map));
  PetscFunctionReturn(0);
}

PetscErrorCode VecSetInf(Vec xin)
{
  PetscInt       i,n = xin->map->n;
  PetscScalar    *xx;
  PetscScalar    zero=0.0,one=1.0,inf=one/zero;

  PetscFunctionBegin;
  if (xin->ops->set) { /* can be called by a subset of processes, do not use collective routines */
    PetscCall((*xin->ops->set)(xin,inf));
  } else {
    PetscCall(VecGetArrayWrite(xin,&xx));
    for (i=0; i<n; i++) xx[i] = inf;
    PetscCall(VecRestoreArrayWrite(xin,&xx));
  }
  PetscFunctionReturn(0);
}

/*@
     VecBindToCPU - marks a vector to temporarily stay on the CPU and perform computations on the CPU

  Logically collective on Vec

   Input Parameters:
+   v - the vector
-   flg - bind to the CPU if value of PETSC_TRUE

   Level: intermediate
@*/
PetscErrorCode VecBindToCPU(Vec v,PetscBool flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,1);
  PetscValidLogicalCollectiveBool(v,flg,2);
#if defined(PETSC_HAVE_DEVICE)
  if (v->boundtocpu == flg) PetscFunctionReturn(0);
  v->boundtocpu = flg;
  if (v->ops->bindtocpu) {
    PetscCall((*v->ops->bindtocpu)(v,flg));
  }
#endif
  PetscFunctionReturn(0);
}

/*@
     VecBoundToCPU - query if a vector is bound to the CPU

  Not collective

   Input Parameter:
.   v - the vector

   Output Parameter:
.   flg - the logical flag

   Level: intermediate

.seealso: `VecBindToCPU()`
@*/
PetscErrorCode VecBoundToCPU(Vec v,PetscBool *flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,1);
  PetscValidBoolPointer(flg,2);
#if defined(PETSC_HAVE_DEVICE)
  *flg = v->boundtocpu;
#else
  *flg = PETSC_TRUE;
#endif
  PetscFunctionReturn(0);
}

/*@
   VecSetBindingPropagates - Sets whether the state of being bound to the CPU for a GPU vector type propagates to child and some other associated objects

   Input Parameters:
+  v - the vector
-  flg - flag indicating whether the boundtocpu flag should be propagated

   Level: developer

   Notes:
   If the value of flg is set to true, then VecDuplicate() and VecDuplicateVecs() will bind created vectors to GPU if the input vector is bound to the CPU.
   The created vectors will also have their bindingpropagates flag set to true.

   Developer Notes:
   If a DMDA has the -dm_bind_below option set to true, then vectors created by DMCreateGlobalVector() will have VecSetBindingPropagates() called on them to
   set their bindingpropagates flag to true.

.seealso: `MatSetBindingPropagates()`, `VecGetBindingPropagates()`
@*/
PetscErrorCode VecSetBindingPropagates(Vec v,PetscBool flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,1);
#if defined(PETSC_HAVE_VIENNACL) || defined(PETSC_HAVE_CUDA)
  v->bindingpropagates = flg;
#endif
  PetscFunctionReturn(0);
}

/*@
   VecGetBindingPropagates - Gets whether the state of being bound to the CPU for a GPU vector type propagates to child and some other associated objects

   Input Parameter:
.  v - the vector

   Output Parameter:
.  flg - flag indicating whether the boundtocpu flag will be propagated

   Level: developer

.seealso: `VecSetBindingPropagates()`
@*/
PetscErrorCode VecGetBindingPropagates(Vec v,PetscBool *flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,1);
  PetscValidBoolPointer(flg,2);
#if defined(PETSC_HAVE_VIENNACL) || defined(PETSC_HAVE_CUDA)
  *flg = v->bindingpropagates;
#else
  *flg = PETSC_FALSE;
#endif
  PetscFunctionReturn(0);
}

/*@C
  VecSetPinnedMemoryMin - Set the minimum data size for which pinned memory will be used for host (CPU) allocations.

  Logically Collective on Vec

  Input Parameters:
+  v    - the vector
-  mbytes - minimum data size in bytes

  Options Database Keys:

. -vec_pinned_memory_min <size> - minimum size (in bytes) for an allocation to use pinned memory on host.
                                  Note that this takes a PetscScalar, to accommodate large values;
                                  specifying -1 ensures that pinned memory will never be used.

  Level: developer

.seealso: `VecGetPinnedMemoryMin()`
@*/
PetscErrorCode VecSetPinnedMemoryMin(Vec v,size_t mbytes)
{
#if defined(PETSC_HAVE_VIENNACL) || defined(PETSC_HAVE_CUDA) || defined(PETSC_HAVE_HIP)
  PetscFunctionBegin;
  v->minimum_bytes_pinned_memory = mbytes;
  PetscFunctionReturn(0);
#else
  return 0;
#endif
}

/*@C
  VecGetPinnedMemoryMin - Get the minimum data size for which pinned memory will be used for host (CPU) allocations.

  Logically Collective on Vec

  Input Parameters:
.  v    - the vector

  Output Parameters:
.  mbytes - minimum data size in bytes

  Level: developer

.seealso: `VecSetPinnedMemoryMin()`
@*/
PetscErrorCode VecGetPinnedMemoryMin(Vec v,size_t *mbytes)
{
#if defined(PETSC_HAVE_VIENNACL) || defined(PETSC_HAVE_CUDA) || defined(PETSC_HAVE_HIP)
  PetscFunctionBegin;
  *mbytes = v->minimum_bytes_pinned_memory;
  PetscFunctionReturn(0);
#else
  return 0;
#endif
}

/*@
  VecGetOffloadMask - Get the offload mask of a Vec.

  Not Collective

  Input Parameters:
.   v - the vector

  Output Parameters:
.   mask - corresponding PetscOffloadMask enum value.

   Level: intermediate

.seealso: `VecCreateSeqCUDA()`, `VecCreateSeqViennaCL()`, `VecGetArray()`, `VecGetType()`
@*/
PetscErrorCode VecGetOffloadMask(Vec v,PetscOffloadMask* mask)
{
  PetscFunctionBegin;
  *mask = v->offloadmask;
  PetscFunctionReturn(0);
}

#if !defined(PETSC_HAVE_VIENNACL)
PETSC_EXTERN PetscErrorCode VecViennaCLGetCLContext(Vec v,PETSC_UINTPTR_T* ctx)
{
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"PETSc must be configured with --with-opencl to get a Vec's cl_context");
}

PETSC_EXTERN PetscErrorCode VecViennaCLGetCLQueue(Vec v,PETSC_UINTPTR_T* queue)
{
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"PETSc must be configured with --with-opencl to get a Vec's cl_command_queue");
}

PETSC_EXTERN PetscErrorCode VecViennaCLGetCLMem(Vec v,PETSC_UINTPTR_T* queue)
{
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"PETSc must be configured with --with-opencl to get a Vec's cl_mem");
}

PETSC_EXTERN PetscErrorCode VecViennaCLGetCLMemRead(Vec v,PETSC_UINTPTR_T* queue)
{
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"PETSc must be configured with --with-opencl to get a Vec's cl_mem");
}

PETSC_EXTERN PetscErrorCode VecViennaCLGetCLMemWrite(Vec v,PETSC_UINTPTR_T* queue)
{
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"PETSc must be configured with --with-opencl to get a Vec's cl_mem");
}

PETSC_EXTERN PetscErrorCode VecViennaCLRestoreCLMemWrite(Vec v)
{
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"PETSc must be configured with --with-opencl to restore a Vec's cl_mem");
}
#endif
