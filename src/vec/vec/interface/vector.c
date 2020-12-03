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

.seealso: VecAssemblyBegin(), VecAssemblyEnd(), Vec, VecStashSetInitialSize(), VecStashView()

@*/
PetscErrorCode  VecStashGetInfo(Vec vec,PetscInt *nstash,PetscInt *reallocs,PetscInt *bnstash,PetscInt *breallocs)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecStashGetInfo_Private(&vec->stash,nstash,reallocs);CHKERRQ(ierr);
  ierr = VecStashGetInfo_Private(&vec->bstash,bnstash,breallocs);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_CLASSID,1);
  PetscValidHeaderSpecific(mapping,IS_LTOGM_CLASSID,2);

  if (x->ops->setlocaltoglobalmapping) {
    ierr = (*x->ops->setlocaltoglobalmapping)(x,mapping);CHKERRQ(ierr);
  } else {
    ierr = PetscLayoutSetISLocalToGlobalMapping(x->map,mapping);CHKERRQ(ierr);
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


.seealso:  VecSetValuesLocal()
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

.seealso: VecAssemblyEnd(), VecSetValues()
@*/
PetscErrorCode  VecAssemblyBegin(Vec vec)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(vec,VEC_CLASSID,1);
  PetscValidType(vec,1);
  ierr = VecStashViewFromOptions(vec,NULL,"-vec_view_stash");CHKERRQ(ierr);
  ierr = PetscLogEventBegin(VEC_AssemblyBegin,vec,0,0,0);CHKERRQ(ierr);
  if (vec->ops->assemblybegin) {
    ierr = (*vec->ops->assemblybegin)(vec);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(VEC_AssemblyBegin,vec,0,0,0);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)vec);CHKERRQ(ierr);
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

.seealso: VecAssemblyBegin(), VecSetValues()
@*/
PetscErrorCode  VecAssemblyEnd(Vec vec)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(vec,VEC_CLASSID,1);
  ierr = PetscLogEventBegin(VEC_AssemblyEnd,vec,0,0,0);CHKERRQ(ierr);
  PetscValidType(vec,1);
  if (vec->ops->assemblyend) {
    ierr = (*vec->ops->assemblyend)(vec);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(VEC_AssemblyEnd,vec,0,0,0);CHKERRQ(ierr);
  ierr = VecViewFromOptions(vec,NULL,"-vec_view");CHKERRQ(ierr);
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

.seealso: VecPointwiseDivide(), VecPointwiseMult(), VecPointwiseMin(), VecPointwiseMaxAbs(), VecMaxPointwiseDivide()
@*/
PetscErrorCode  VecPointwiseMax(Vec w,Vec x,Vec y)
{
  PetscErrorCode ierr;

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
  ierr = VecSetErrorIfLocked(w,1);CHKERRQ(ierr);
  ierr = (*w->ops->pointwisemax)(w,x,y);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)w);CHKERRQ(ierr);
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

.seealso: VecPointwiseDivide(), VecPointwiseMult(), VecPointwiseMin(), VecPointwiseMaxAbs(), VecMaxPointwiseDivide()
@*/
PetscErrorCode  VecPointwiseMin(Vec w,Vec x,Vec y)
{
  PetscErrorCode ierr;

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
  ierr = VecSetErrorIfLocked(w,1);CHKERRQ(ierr);
  ierr = (*w->ops->pointwisemin)(w,x,y);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)w);CHKERRQ(ierr);
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

.seealso: VecPointwiseDivide(), VecPointwiseMult(), VecPointwiseMin(), VecPointwiseMax(), VecMaxPointwiseDivide()
@*/
PetscErrorCode  VecPointwiseMaxAbs(Vec w,Vec x,Vec y)
{
  PetscErrorCode ierr;

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
  ierr = VecSetErrorIfLocked(w,1);CHKERRQ(ierr);
  ierr = (*w->ops->pointwisemaxabs)(w,x,y);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)w);CHKERRQ(ierr);
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

.seealso: VecPointwiseMult(), VecPointwiseMax(), VecPointwiseMin(), VecPointwiseMaxAbs(), VecMaxPointwiseDivide()
@*/
PetscErrorCode  VecPointwiseDivide(Vec w,Vec x,Vec y)
{
  PetscErrorCode ierr;

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
  ierr = VecSetErrorIfLocked(w,1);CHKERRQ(ierr);
  ierr = (*w->ops->pointwisedivide)(w,x,y);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)w);CHKERRQ(ierr);
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

.seealso: VecDestroy(), VecDuplicateVecs(), VecCreate(), VecCopy()
@*/
PetscErrorCode  VecDuplicate(Vec v,Vec *newv)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,1);
  PetscValidPointer(newv,2);
  PetscValidType(v,1);
  ierr = (*v->ops->duplicate)(v,newv);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)*newv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   VecDestroy - Destroys a vector.

   Collective on Vec

   Input Parameters:
.  v  - the vector

   Level: beginner

.seealso: VecDuplicate(), VecDestroyVecs()
@*/
PetscErrorCode  VecDestroy(Vec *v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*v) PetscFunctionReturn(0);
  PetscValidHeaderSpecific((*v),VEC_CLASSID,1);
  if (--((PetscObject)(*v))->refct > 0) {*v = NULL; PetscFunctionReturn(0);}

  ierr = PetscObjectSAWsViewOff((PetscObject)*v);CHKERRQ(ierr);
  /* destroy the internal part */
  if ((*v)->ops->destroy) {
    ierr = (*(*v)->ops->destroy)(*v);CHKERRQ(ierr);
  }
  /* destroy the external/common part */
  ierr = PetscLayoutDestroy(&(*v)->map);CHKERRQ(ierr);
  ierr = PetscHeaderDestroy(v);CHKERRQ(ierr);
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

.seealso:  VecDestroyVecs(), VecDuplicate(), VecCreate(), VecDuplicateVecsF90()
@*/
PetscErrorCode  VecDuplicateVecs(Vec v,PetscInt m,Vec *V[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,1);
  PetscValidPointer(V,3);
  PetscValidType(v,1);
  ierr = (*v->ops->duplicatevecs)(v,m,V);CHKERRQ(ierr);
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

.seealso: VecDuplicateVecs(), VecDestroyVecsf90()
@*/
PetscErrorCode  VecDestroyVecs(PetscInt m,Vec *vv[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(vv,1);
  if (m < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Trying to destroy negative number of vectors %D",m);
  if (!m || !*vv) {*vv  = NULL; PetscFunctionReturn(0);}
  PetscValidHeaderSpecific(**vv,VEC_CLASSID,1);
  PetscValidType(**vv,1);
  ierr = (*(**vv)->ops->destroyvecs)(m,*vv);CHKERRQ(ierr);
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
.seealso:  Vec, VecView, PetscObjectViewFromOptions(), VecCreate()
@*/
PetscErrorCode  VecViewFromOptions(Vec A,PetscObject obj,const char name[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,VEC_CLASSID,1);
  ierr = PetscObjectViewFromOptions((PetscObject)A,obj,name);CHKERRQ(ierr);
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


.seealso: PetscViewerASCIIOpen(), PetscViewerDrawOpen(), PetscDrawLGCreate(),
          PetscViewerSocketOpen(), PetscViewerBinaryOpen(), VecLoad(), PetscViewerCreate(),
          PetscRealView(), PetscScalarView(), PetscIntView(), PetscViewerHDF5SetTimestep()
@*/
PetscErrorCode  VecView(Vec vec,PetscViewer viewer)
{
  PetscErrorCode    ierr;
  PetscBool         iascii;
  PetscViewerFormat format;
  PetscMPIInt       size;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(vec,VEC_CLASSID,1);
  PetscValidType(vec,1);
  if (!viewer) {ierr = PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)vec),&viewer);CHKERRQ(ierr);}
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)vec),&size);CHKERRQ(ierr);
  if (size == 1 && format == PETSC_VIEWER_LOAD_BALANCE) PetscFunctionReturn(0);

  if (vec->stash.n || vec->bstash.n) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call VecAssemblyBegin/End() before viewing this vector");

  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    PetscInt rows,bs;

    ierr = PetscObjectPrintClassNamePrefixType((PetscObject)vec,viewer);CHKERRQ(ierr);
    if (format == PETSC_VIEWER_ASCII_INFO || format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = VecGetSize(vec,&rows);CHKERRQ(ierr);
      ierr = VecGetBlockSize(vec,&bs);CHKERRQ(ierr);
      if (bs != 1) {
        ierr = PetscViewerASCIIPrintf(viewer,"length=%D, bs=%D\n",rows,bs);CHKERRQ(ierr);
      } else {
        ierr = PetscViewerASCIIPrintf(viewer,"length=%D\n",rows);CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    }
  }
  ierr = VecLockReadPush(vec);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(VEC_View,vec,viewer,0,0);CHKERRQ(ierr);
  if ((format == PETSC_VIEWER_NATIVE || format == PETSC_VIEWER_LOAD_BALANCE) && vec->ops->viewnative) {
    ierr = (*vec->ops->viewnative)(vec,viewer);CHKERRQ(ierr);
  } else {
    ierr = (*vec->ops->view)(vec,viewer);CHKERRQ(ierr);
  }
  ierr = VecLockReadPop(vec);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(VEC_View,vec,viewer,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#if defined(PETSC_USE_DEBUG)
#include <../src/sys/totalview/tv_data_display.h>
PETSC_UNUSED static int TV_display_type(const struct _p_Vec *v)
{
  const PetscScalar *values;
  char              type[32];
  PetscErrorCode    ierr;


  TV_add_row("Local rows", "int", &v->map->n);
  TV_add_row("Global rows", "int", &v->map->N);
  TV_add_row("Typename", TV_ascii_string_type, ((PetscObject)v)->type_name);
  ierr = VecGetArrayRead((Vec)v,&values);CHKERRQ(ierr);
  ierr = PetscSNPrintf(type,32,"double[%d]",v->map->n);CHKERRQ(ierr);
  TV_add_row("values",type, values);
  ierr = VecRestoreArrayRead((Vec)v,&values);CHKERRQ(ierr);
  return TV_format_OK;
}
#endif

/*@
   VecGetSize - Returns the global number of elements of the vector.

   Not Collective

   Input Parameter:
.  x - the vector

   Output Parameters:
.  size - the global length of the vector

   Level: beginner

.seealso: VecGetLocalSize()
@*/
PetscErrorCode  VecGetSize(Vec x,PetscInt *size)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_CLASSID,1);
  PetscValidIntPointer(size,2);
  PetscValidType(x,1);
  ierr = (*x->ops->getsize)(x,size);CHKERRQ(ierr);
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

.seealso: VecGetSize()
@*/
PetscErrorCode  VecGetLocalSize(Vec x,PetscInt *size)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_CLASSID,1);
  PetscValidIntPointer(size,2);
  PetscValidType(x,1);
  ierr = (*x->ops->getlocalsize)(x,size);CHKERRQ(ierr);
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


.seealso:   MatGetOwnershipRange(), MatGetOwnershipRanges(), VecGetOwnershipRanges()
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

   Level: beginner


.seealso:   MatGetOwnershipRange(), MatGetOwnershipRanges(), VecGetOwnershipRange()
@*/
PetscErrorCode  VecGetOwnershipRanges(Vec x,const PetscInt *ranges[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_CLASSID,1);
  PetscValidType(x,1);
  ierr = PetscLayoutGetRanges(x->map,ranges);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   VecSetOption - Sets an option for controling a vector's behavior.

   Collective on Vec

   Input Parameter:
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_CLASSID,1);
  PetscValidType(x,1);
  if (x->ops->setoption) {
    ierr = (*x->ops->setoption)(x,op,flag);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* Default routines for obtaining and releasing; */
/* may be used by any implementation */
PetscErrorCode VecDuplicateVecs_Default(Vec w,PetscInt m,Vec *V[])
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(w,VEC_CLASSID,1);
  PetscValidPointer(V,3);
  if (m <= 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"m must be > 0: m = %D",m);
  ierr = PetscMalloc1(m,V);CHKERRQ(ierr);
  for (i=0; i<m; i++) {ierr = VecDuplicate(w,*V+i);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

PetscErrorCode VecDestroyVecs_Default(PetscInt m,Vec v[])
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  PetscValidPointer(v,1);
  for (i=0; i<m; i++) {ierr = VecDestroy(&v[i]);CHKERRQ(ierr);}
  ierr = PetscFree(v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   VecResetArray - Resets a vector to use its default memory. Call this
   after the use of VecPlaceArray().

   Not Collective

   Input Parameters:
.  vec - the vector

   Level: developer

.seealso: VecGetArray(), VecRestoreArray(), VecReplaceArray(), VecPlaceArray()

@*/
PetscErrorCode  VecResetArray(Vec vec)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(vec,VEC_CLASSID,1);
  PetscValidType(vec,1);
  if (vec->ops->resetarray) {
    ierr = (*vec->ops->resetarray)(vec);CHKERRQ(ierr);
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot reset array in this type of vector");
  ierr = PetscObjectStateIncrease((PetscObject)vec);CHKERRQ(ierr);
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

.seealso: PetscViewerBinaryOpen(), VecView(), MatLoad(), VecLoad()
@*/
PetscErrorCode  VecLoad(Vec vec, PetscViewer viewer)
{
  PetscErrorCode    ierr;
  PetscBool         isbinary,ishdf5,isadios,isadios2;
  PetscViewerFormat format;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(vec,VEC_CLASSID,1);
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(vec,1,viewer,2);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERBINARY,&isbinary);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERHDF5,&ishdf5);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERADIOS,&isadios);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERADIOS,&isadios2);CHKERRQ(ierr);
  if (!isbinary && !ishdf5 && !isadios && !isadios2) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Invalid viewer; open viewer with PetscViewerBinaryOpen()");

  ierr = VecSetErrorIfLocked(vec,1);CHKERRQ(ierr);
  if (!((PetscObject)vec)->type_name && !vec->ops->create) {
    ierr = VecSetType(vec, VECSTANDARD);CHKERRQ(ierr);
  }
  ierr = PetscLogEventBegin(VEC_Load,viewer,0,0,0);CHKERRQ(ierr);
  ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
  if (format == PETSC_VIEWER_NATIVE && vec->ops->loadnative) {
    ierr = (*vec->ops->loadnative)(vec,viewer);CHKERRQ(ierr);
  } else {
    ierr = (*vec->ops->load)(vec,viewer);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(VEC_Load,viewer,0,0,0);CHKERRQ(ierr);
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

.seealso: VecLog(), VecExp(), VecSqrtAbs()

@*/
PetscErrorCode  VecReciprocal(Vec vec)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(vec,VEC_CLASSID,1);
  PetscValidType(vec,1);
  if (vec->stash.insertmode != NOT_SET_VALUES) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled vector");
  if (!vec->ops->reciprocal) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Vector does not support reciprocal operation");
  ierr = VecSetErrorIfLocked(vec,1);CHKERRQ(ierr);
  ierr = (*vec->ops->reciprocal)(vec);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)vec);CHKERRQ(ierr);
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
$      ierr = VecCreateMPI(comm,m,M,&x);CHKERRQ(ierr);
$      ierr = VecSetOperation(x,VECOP_VIEW,(void(*)(void))userview);CHKERRQ(ierr);

    Notes:
    See the file include/petscvec.h for a complete list of matrix
    operations, which all have the form VECOP_<OPERATION>, where
    <OPERATION> is the name (in all capital letters) of the
    user interface routine (e.g., VecView() -> VECOP_VIEW).

    This function is not currently available from Fortran.

.seealso: VecCreate(), MatShellSetOperation()
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


.seealso: VecSetBlockSize(), VecSetValues(), VecSetValuesBlocked(), VecStashView()

@*/
PetscErrorCode  VecStashSetInitialSize(Vec vec,PetscInt size,PetscInt bsize)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(vec,VEC_CLASSID,1);
  ierr = VecStashSetInitialSize_Private(&vec->stash,size);CHKERRQ(ierr);
  ierr = VecStashSetInitialSize_Private(&vec->bstash,bsize);CHKERRQ(ierr);
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
#if defined(PETSC_USE_COMPLEX)
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_CLASSID,1);
  PetscValidType(x,1);
  if (x->stash.insertmode != NOT_SET_VALUES) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled vector");
  ierr = VecSetErrorIfLocked(x,1);CHKERRQ(ierr);
  ierr = (*x->ops->conjugate)(x);CHKERRQ(ierr);
  /* we need to copy norms here */
  ierr = PetscObjectStateIncrease((PetscObject)x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
#else
  return(0);
#endif
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

.seealso: VecPointwiseDivide(), VecPointwiseMax(), VecPointwiseMin(), VecPointwiseMaxAbs(), VecMaxPointwiseDivide()
@*/
PetscErrorCode  VecPointwiseMult(Vec w, Vec x,Vec y)
{
  PetscErrorCode ierr;

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
  ierr = VecSetErrorIfLocked(w,1);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(VEC_PointwiseMult,x,y,w,0);CHKERRQ(ierr);
  ierr = (*w->ops->pointwisemult)(w,x,y);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(VEC_PointwiseMult,x,y,w,0);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)w);CHKERRQ(ierr);
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


.seealso: VecSet(), VecSetValues(), PetscRandomCreate(), PetscRandomDestroy()
@*/
PetscErrorCode  VecSetRandom(Vec x,PetscRandom rctx)
{
  PetscErrorCode ierr;
  PetscRandom    randObj = NULL;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_CLASSID,1);
  if (rctx) PetscValidHeaderSpecific(rctx,PETSC_RANDOM_CLASSID,2);
  PetscValidType(x,1);
  if (x->stash.insertmode != NOT_SET_VALUES) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled vector");
  ierr = VecSetErrorIfLocked(x,1);CHKERRQ(ierr);

  if (!rctx) {
    MPI_Comm comm;
    ierr = PetscObjectGetComm((PetscObject)x,&comm);CHKERRQ(ierr);
    ierr = PetscRandomCreate(comm,&randObj);CHKERRQ(ierr);
    ierr = PetscRandomSetFromOptions(randObj);CHKERRQ(ierr);
    rctx = randObj;
  }

  ierr = PetscLogEventBegin(VEC_SetRandom,x,rctx,0,0);CHKERRQ(ierr);
  ierr = (*x->ops->setrandom)(x,rctx);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(VEC_SetRandom,x,rctx,0,0);CHKERRQ(ierr);

  ierr = PetscRandomDestroy(&randObj);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  VecZeroEntries - puts a 0.0 in each element of a vector

  Logically Collective on Vec

  Input Parameter:
. vec - The vector

  Level: beginner

.seealso: VecCreate(),  VecSetOptionsPrefix(), VecSet(), VecSetValues()
@*/
PetscErrorCode  VecZeroEntries(Vec vec)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecSet(vec,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  VecSetTypeFromOptions_Private - Sets the type of vector from user options. Defaults to a PETSc sequential vector on one
  processor and a PETSc MPI vector on more than one processor.

  Collective on Vec

  Input Parameter:
. vec - The vector

  Level: intermediate

.seealso: VecSetFromOptions(), VecSetType()
*/
static PetscErrorCode VecSetTypeFromOptions_Private(PetscOptionItems *PetscOptionsObject,Vec vec)
{
  PetscBool      opt;
  VecType        defaultType;
  char           typeName[256];
  PetscMPIInt    size;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (((PetscObject)vec)->type_name) defaultType = ((PetscObject)vec)->type_name;
  else {
    ierr = MPI_Comm_size(PetscObjectComm((PetscObject)vec), &size);CHKERRQ(ierr);
    if (size > 1) defaultType = VECMPI;
    else defaultType = VECSEQ;
  }

  ierr = VecRegisterAll();CHKERRQ(ierr);
  ierr = PetscOptionsFList("-vec_type","Vector type","VecSetType",VecList,defaultType,typeName,256,&opt);CHKERRQ(ierr);
  if (opt) {
    ierr = VecSetType(vec, typeName);CHKERRQ(ierr);
  } else {
    ierr = VecSetType(vec, defaultType);CHKERRQ(ierr);
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


.seealso: VecCreate(), VecSetOptionsPrefix()
@*/
PetscErrorCode  VecSetFromOptions(Vec vec)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(vec,VEC_CLASSID,1);

  ierr = PetscObjectOptionsBegin((PetscObject)vec);CHKERRQ(ierr);
  /* Handle vector type options */
  ierr = VecSetTypeFromOptions_Private(PetscOptionsObject,vec);CHKERRQ(ierr);

  /* Handle specific vector options */
  if (vec->ops->setfromoptions) {
    ierr = (*vec->ops->setfromoptions)(PetscOptionsObject,vec);CHKERRQ(ierr);
  }

  /* process any options handlers added with PetscObjectAddOptionsHandler() */
  ierr = PetscObjectProcessOptionsHandlers(PetscOptionsObject,(PetscObject)vec);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
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

.seealso: VecGetSize(), PetscSplitOwnership()
@*/
PetscErrorCode  VecSetSizes(Vec v, PetscInt n, PetscInt N)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v, VEC_CLASSID,1);
  if (N >= 0) PetscValidLogicalCollectiveInt(v,N,3);
  if (N >= 0 && n > N) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Local size %D cannot be larger than global size %D",n,N);
  if ((v->map->n >= 0 || v->map->N >= 0) && (v->map->n != n || v->map->N != N)) SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot change/reset vector sizes to %D local %D global after previously setting them to %D local %D global",n,N,v->map->n,v->map->N);
  v->map->n = n;
  v->map->N = N;
  if (v->ops->create) {
    ierr = (*v->ops->create)(v);CHKERRQ(ierr);
    v->ops->create = NULL;
  }
  PetscFunctionReturn(0);
}

/*@
   VecSetBlockSize - Sets the blocksize for future calls to VecSetValuesBlocked()
   and VecSetValuesBlockedLocal().

   Logically Collective on Vec

   Input Parameter:
+  v - the vector
-  bs - the blocksize

   Notes:
   All vectors obtained by VecDuplicate() inherit the same blocksize.

   Level: advanced

.seealso: VecSetValuesBlocked(), VecSetLocalToGlobalMapping(), VecGetBlockSize()

@*/
PetscErrorCode  VecSetBlockSize(Vec v,PetscInt bs)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,1);
  PetscValidLogicalCollectiveInt(v,bs,2);
  ierr = PetscLayoutSetBlockSize(v->map,bs);CHKERRQ(ierr);
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

.seealso: VecSetValuesBlocked(), VecSetLocalToGlobalMapping(), VecSetBlockSize()


@*/
PetscErrorCode  VecGetBlockSize(Vec v,PetscInt *bs)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,1);
  PetscValidIntPointer(bs,2);
  ierr = PetscLayoutGetBlockSize(v->map,bs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   VecSetOptionsPrefix - Sets the prefix used for searching for all
   Vec options in the database.

   Logically Collective on Vec

   Input Parameter:
+  v - the Vec context
-  prefix - the prefix to prepend to all option names

   Notes:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the hyphen.

   Level: advanced

.seealso: VecSetFromOptions()
@*/
PetscErrorCode  VecSetOptionsPrefix(Vec v,const char prefix[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,1);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)v,prefix);CHKERRQ(ierr);
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

.seealso: VecGetOptionsPrefix()
@*/
PetscErrorCode  VecAppendOptionsPrefix(Vec v,const char prefix[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,1);
  ierr = PetscObjectAppendOptionsPrefix((PetscObject)v,prefix);CHKERRQ(ierr);
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

.seealso: VecAppendOptionsPrefix()
@*/
PetscErrorCode  VecGetOptionsPrefix(Vec v,const char *prefix[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,1);
  ierr = PetscObjectGetOptionsPrefix((PetscObject)v,prefix);CHKERRQ(ierr);
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

.seealso: VecCreate(), VecDestroy()
@*/
PetscErrorCode  VecSetUp(Vec v)
{
  PetscMPIInt    size;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,1);
  if (v->map->n < 0 && v->map->N < 0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Sizes not set");
  if (!((PetscObject)v)->type_name) {
    ierr = MPI_Comm_size(PetscObjectComm((PetscObject)v), &size);CHKERRQ(ierr);
    if (size == 1) {
      ierr = VecSetType(v, VECSEQ);CHKERRQ(ierr);
    } else {
      ierr = VecSetType(v, VECMPI);CHKERRQ(ierr);
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

.seealso: VecDuplicate()
@*/
PetscErrorCode  VecCopy(Vec x,Vec y)
{
  PetscBool      flgs[4];
  PetscReal      norms[4] = {0.0,0.0,0.0,0.0};
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_CLASSID,1);
  PetscValidHeaderSpecific(y,VEC_CLASSID,2);
  PetscValidType(x,1);
  PetscValidType(y,2);
  if (x == y) PetscFunctionReturn(0);
  VecCheckSameLocalSize(x,1,y,2);
  if (x->stash.insertmode != NOT_SET_VALUES) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled vector");
  ierr = VecSetErrorIfLocked(y,2);CHKERRQ(ierr);

#if !defined(PETSC_USE_MIXED_PRECISION)
  for (i=0; i<4; i++) {
    ierr = PetscObjectComposedDataGetReal((PetscObject)x,NormIds[i],norms[i],flgs[i]);CHKERRQ(ierr);
  }
#endif

  ierr = PetscLogEventBegin(VEC_Copy,x,y,0,0);CHKERRQ(ierr);
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
    ierr = VecGetArrayRead(x,&xx);CHKERRQ(ierr);
    ierr = VecGetArray(y,&yy);CHKERRQ(ierr);
    ierr = VecGetLocalSize(x,&n);CHKERRQ(ierr);
    for (i=0; i<n; i++) yy[i] = xx[i];
    ierr = VecRestoreArrayRead(x,&xx);CHKERRQ(ierr);
    ierr = VecRestoreArray(y,&yy);CHKERRQ(ierr);
  } else if ((((PetscObject)x)->precision == PETSC_PRECISION_DOUBLE) && (((PetscObject)y)->precision == PETSC_PRECISION_SINGLE)) {
    PetscInt     i,n;
    float        *yy;
    const double *xx;
    ierr = VecGetArrayRead(x,&xx);CHKERRQ(ierr);
    ierr = VecGetArray(y,&yy);CHKERRQ(ierr);
    ierr = VecGetLocalSize(x,&n);CHKERRQ(ierr);
    for (i=0; i<n; i++) yy[i] = (float) xx[i];
    ierr = VecRestoreArrayRead(x,&xx);CHKERRQ(ierr);
    ierr = VecRestoreArray(y,&yy);CHKERRQ(ierr);
  } else {
    ierr = (*x->ops->copy)(x,y);CHKERRQ(ierr);
  }
#else
  ierr = (*x->ops->copy)(x,y);CHKERRQ(ierr);
#endif

  ierr = PetscObjectStateIncrease((PetscObject)y);CHKERRQ(ierr);
#if !defined(PETSC_USE_MIXED_PRECISION)
  for (i=0; i<4; i++) {
    if (flgs[i]) {
      ierr = PetscObjectComposedDataSetReal((PetscObject)y,NormIds[i],norms[i]);CHKERRQ(ierr);
    }
  }
#endif

  ierr = PetscLogEventEnd(VEC_Copy,x,y,0,0);CHKERRQ(ierr);
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
  PetscReal      normxs[4]={0.0,0.0,0.0,0.0},normys[4]={0.0,0.0,0.0,0.0};
  PetscBool      flgxs[4],flgys[4];
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_CLASSID,1);
  PetscValidHeaderSpecific(y,VEC_CLASSID,2);
  PetscValidType(x,1);
  PetscValidType(y,2);
  PetscCheckSameTypeAndComm(x,1,y,2);
  VecCheckSameSize(x,1,y,2);
  if (x->stash.insertmode != NOT_SET_VALUES) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled vector");
  if (y->stash.insertmode != NOT_SET_VALUES) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled vector");
  ierr = VecSetErrorIfLocked(x,1);CHKERRQ(ierr);
  ierr = VecSetErrorIfLocked(y,2);CHKERRQ(ierr);

  ierr = PetscLogEventBegin(VEC_Swap,x,y,0,0);CHKERRQ(ierr);
  for (i=0; i<4; i++) {
    ierr = PetscObjectComposedDataGetReal((PetscObject)x,NormIds[i],normxs[i],flgxs[i]);CHKERRQ(ierr);
    ierr = PetscObjectComposedDataGetReal((PetscObject)y,NormIds[i],normys[i],flgys[i]);CHKERRQ(ierr);
  }
  ierr = (*x->ops->swap)(x,y);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)x);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)y);CHKERRQ(ierr);
  for (i=0; i<4; i++) {
    if (flgxs[i]) {
      ierr = PetscObjectComposedDataSetReal((PetscObject)y,NormIds[i],normxs[i]);CHKERRQ(ierr);
    }
    if (flgys[i]) {
      ierr = PetscObjectComposedDataSetReal((PetscObject)x,NormIds[i],normys[i]);CHKERRQ(ierr);
    }
  }
  ierr = PetscLogEventEnd(VEC_Swap,x,y,0,0);CHKERRQ(ierr);
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
  PetscErrorCode    ierr;
  PetscViewer       viewer;
  PetscBool         flg;
  PetscViewerFormat format;
  char              *prefix;

  PetscFunctionBegin;
  prefix = bobj ? bobj->prefix : ((PetscObject)obj)->prefix;
  ierr   = PetscOptionsGetViewer(PetscObjectComm((PetscObject)obj),((PetscObject)obj)->options,prefix,optionname,&viewer,&format,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscViewerPushFormat(viewer,format);CHKERRQ(ierr);
    ierr = VecStashView(obj,viewer);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
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


.seealso: VecSetBlockSize(), VecSetValues(), VecSetValuesBlocked()

@*/
PetscErrorCode  VecStashView(Vec v,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank;
  PetscInt       i,j;
  PetscBool      match;
  VecStash       *s;
  PetscScalar    val;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,1);
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(v,1,viewer,2);

  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&match);CHKERRQ(ierr);
  if (!match) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Stash viewer only works with ASCII viewer not %s\n",((PetscObject)v)->type_name);
  ierr = PetscViewerASCIIUseTabs(viewer,PETSC_FALSE);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)v),&rank);CHKERRQ(ierr);
  s    = &v->bstash;

  /* print block stash */
  ierr = PetscViewerASCIIPushSynchronized(viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d]Vector Block stash size %D block size %D\n",rank,s->n,s->bs);CHKERRQ(ierr);
  for (i=0; i<s->n; i++) {
    ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] Element %D ",rank,s->idx[i]);CHKERRQ(ierr);
    for (j=0; j<s->bs; j++) {
      val = s->array[i*s->bs+j];
#if defined(PETSC_USE_COMPLEX)
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"(%18.16e %18.16e) ",PetscRealPart(val),PetscImaginaryPart(val));CHKERRQ(ierr);
#else
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"%18.16e ",val);CHKERRQ(ierr);
#endif
    }
    ierr = PetscViewerASCIISynchronizedPrintf(viewer,"\n");CHKERRQ(ierr);
  }
  ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);

  s = &v->stash;

  /* print basic stash */
  ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d]Vector stash size %D\n",rank,s->n);CHKERRQ(ierr);
  for (i=0; i<s->n; i++) {
    val = s->array[i];
#if defined(PETSC_USE_COMPLEX)
    ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] Element %D (%18.16e %18.16e) ",rank,s->idx[i],PetscRealPart(val),PetscImaginaryPart(val));CHKERRQ(ierr);
#else
    ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] Element %D %18.16e\n",rank,s->idx[i],val);CHKERRQ(ierr);
#endif
  }
  ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPopSynchronized(viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIUseTabs(viewer,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscOptionsGetVec(PetscOptions options,const char prefix[],const char key[],Vec v,PetscBool *set)
{
  PetscInt       i,N,rstart,rend;
  PetscErrorCode ierr;
  PetscScalar    *xx;
  PetscReal      *xreal;
  PetscBool      iset;

  PetscFunctionBegin;
  ierr = VecGetOwnershipRange(v,&rstart,&rend);CHKERRQ(ierr);
  ierr = VecGetSize(v,&N);CHKERRQ(ierr);
  ierr = PetscCalloc1(N,&xreal);CHKERRQ(ierr);
  ierr = PetscOptionsGetRealArray(options,prefix,key,xreal,&N,&iset);CHKERRQ(ierr);
  if (iset) {
    ierr = VecGetArray(v,&xx);CHKERRQ(ierr);
    for (i=rstart; i<rend; i++) xx[i-rstart] = xreal[i];
    ierr = VecRestoreArray(v,&xx);CHKERRQ(ierr);
  }
  ierr = PetscFree(xreal);CHKERRQ(ierr);
  if (set) *set = iset;
  PetscFunctionReturn(0);
}

/*@
   VecGetLayout - get PetscLayout describing vector layout

   Not Collective

   Input Arguments:
.  x - the vector

   Output Arguments:
.  map - the layout

   Level: developer

.seealso: VecGetSizes(), VecGetOwnershipRange(), VecGetOwnershipRanges()
@*/
PetscErrorCode VecGetLayout(Vec x,PetscLayout *map)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_CLASSID,1);
  *map = x->map;
  PetscFunctionReturn(0);
}

/*@
   VecSetLayout - set PetscLayout describing vector layout

   Not Collective

   Input Arguments:
+  x - the vector
-  map - the layout

   Notes:
   It is normally only valid to replace the layout with a layout known to be equivalent.

   Level: developer

.seealso: VecGetLayout(), VecGetSizes(), VecGetOwnershipRange(), VecGetOwnershipRanges()
@*/
PetscErrorCode VecSetLayout(Vec x,PetscLayout map)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_CLASSID,1);
  ierr = PetscLayoutReference(map,&x->map);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecSetInf(Vec xin)
{
  PetscInt       i,n = xin->map->n;
  PetscScalar    *xx;
  PetscScalar    zero=0.0,one=1.0,inf=one/zero;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecGetArrayWrite(xin,&xx);CHKERRQ(ierr);
  for (i=0; i<n; i++) xx[i] = inf;
  ierr = VecRestoreArrayWrite(xin,&xx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
     VecBindToCPU - marks a vector to temporarily stay on the CPU and perform computations on the CPU

   Input Parameters:
+   v - the vector
-   flg - bind to the CPU if value of PETSC_TRUE

   Level: intermediate
@*/
PetscErrorCode VecBindToCPU(Vec v,PetscBool flg)
{
#if defined(PETSC_HAVE_VIENNACL) || defined(PETSC_HAVE_CUDA)
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (v->boundtocpu == flg) PetscFunctionReturn(0);
  v->boundtocpu = flg;
  if (v->ops->bindtocpu) {
    ierr = (*v->ops->bindtocpu)(v,flg);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
#else
  return 0;
#endif
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

.seealso: VecGetPinnedMemoryMin()
@*/
PetscErrorCode VecSetPinnedMemoryMin(Vec v,size_t mbytes)
{
#if defined(PETSC_HAVE_VIENNACL) || defined(PETSC_HAVE_CUDA)
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

.seealso: VecSetPinnedMemoryMin()
@*/
PetscErrorCode VecGetPinnedMemoryMin(Vec v,size_t *mbytes)
{
#if defined(PETSC_HAVE_VIENNACL) || defined(PETSC_HAVE_CUDA)
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

.seealso: VecCreateSeqCUDA(), VecCreateSeqViennaCL(), VecGetArray(), VecGetType()
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
