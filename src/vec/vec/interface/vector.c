#define PETSCVEC_DLL

/*
     Provides the interface functions for vector operations that do NOT have PetscScalar/PetscReal in the signature
   These are the vector functions the user calls.
*/
#include "private/vecimpl.h"    /*I "petscvec.h" I*/

/* Logging support */
PetscCookie PETSCVEC_DLLEXPORT VEC_COOKIE;
PetscLogEvent  VEC_View, VEC_Max, VEC_Min, VEC_DotBarrier, VEC_Dot, VEC_MDotBarrier, VEC_MDot, VEC_TDot;
PetscLogEvent  VEC_Norm, VEC_Normalize, VEC_Scale, VEC_Copy, VEC_Set, VEC_AXPY, VEC_AYPX, VEC_WAXPY;
PetscLogEvent  VEC_MTDot, VEC_NormBarrier, VEC_MAXPY, VEC_Swap, VEC_AssemblyBegin, VEC_ScatterBegin, VEC_ScatterEnd;
PetscLogEvent  VEC_AssemblyEnd, VEC_PointwiseMult, VEC_SetValues, VEC_Load, VEC_ScatterBarrier;
PetscLogEvent  VEC_SetRandom, VEC_ReduceArithmetic, VEC_ReduceBarrier, VEC_ReduceCommunication,VEC_Ops;
PetscLogEvent  VEC_DotNormBarrier, VEC_DotNorm, VEC_AXPBYPCZ;

EXTERN PetscErrorCode VecStashGetInfo_Private(VecStash*,PetscInt*,PetscInt*);
#undef __FUNCT__
#define __FUNCT__ "VecStashGetInfo"
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
PetscErrorCode PETSCVEC_DLLEXPORT VecStashGetInfo(Vec vec,PetscInt *nstash,PetscInt *reallocs,PetscInt *bnstash,PetscInt *breallocs)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = VecStashGetInfo_Private(&vec->stash,nstash,reallocs);CHKERRQ(ierr);
  ierr = VecStashGetInfo_Private(&vec->bstash,bnstash,breallocs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecSetLocalToGlobalMapping"
/*@
   VecSetLocalToGlobalMapping - Sets a local numbering to global numbering used
   by the routine VecSetValuesLocal() to allow users to insert vector entries
   using a local (per-processor) numbering.

   Collective on Vec

   Input Parameters:
+  x - vector
-  mapping - mapping created with ISLocalToGlobalMappingCreate() or ISLocalToGlobalMappingCreateIS()

   Notes:
   All vectors obtained with VecDuplicate() from this vector inherit the same mapping.

   Level: intermediate

   Concepts: vector^setting values with local numbering

seealso:  VecAssemblyBegin(), VecAssemblyEnd(), VecSetValues(), VecSetValuesLocal(),
           VecSetLocalToGlobalMappingBlock(), VecSetValuesBlockedLocal()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT VecSetLocalToGlobalMapping(Vec x,ISLocalToGlobalMapping mapping)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE,1);
  PetscValidHeaderSpecific(mapping,IS_LTOGM_COOKIE,2);

  if (x->mapping) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Mapping already set for vector");
  }

  if (x->ops->setlocaltoglobalmapping) {
    ierr = (*x->ops->setlocaltoglobalmapping)(x,mapping);CHKERRQ(ierr);
  } else {
    ierr = PetscObjectReference((PetscObject)mapping);CHKERRQ(ierr);
    if (x->mapping) { ierr = ISLocalToGlobalMappingDestroy(x->mapping);CHKERRQ(ierr); }
    x->mapping = mapping;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecSetLocalToGlobalMappingBlock"
/*@
   VecSetLocalToGlobalMappingBlock - Sets a local numbering to global numbering used
   by the routine VecSetValuesBlockedLocal() to allow users to insert vector entries
   using a local (per-processor) numbering.

   Collective on Vec

   Input Parameters:
+  x - vector
-  mapping - mapping created with ISLocalToGlobalMappingCreate() or ISLocalToGlobalMappingCreateIS()

   Notes:
   All vectors obtained with VecDuplicate() from this vector inherit the same mapping.

   Level: intermediate

   Concepts: vector^setting values blocked with local numbering

.seealso:  VecAssemblyBegin(), VecAssemblyEnd(), VecSetValues(), VecSetValuesLocal(),
           VecSetLocalToGlobalMapping(), VecSetValuesBlockedLocal()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT VecSetLocalToGlobalMappingBlock(Vec x,ISLocalToGlobalMapping mapping)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE,1);
  PetscValidHeaderSpecific(mapping,IS_LTOGM_COOKIE,2);

  if (x->bmapping) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Mapping already set for vector");
  }
  ierr = PetscObjectReference((PetscObject)mapping);CHKERRQ(ierr);
  if (x->bmapping) { ierr = ISLocalToGlobalMappingDestroy(x->bmapping);CHKERRQ(ierr); }
  x->bmapping = mapping;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecAssemblyBegin"
/*@
   VecAssemblyBegin - Begins assembling the vector.  This routine should
   be called after completing all calls to VecSetValues().

   Collective on Vec

   Input Parameter:
.  vec - the vector

   Level: beginner

   Concepts: assembly^vectors

.seealso: VecAssemblyEnd(), VecSetValues()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT VecAssemblyBegin(Vec vec)
{
  PetscErrorCode ierr;
  PetscTruth     flg = PETSC_FALSE;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(vec,VEC_COOKIE,1);
  PetscValidType(vec,1);

  ierr = PetscOptionsGetTruth(((PetscObject)vec)->prefix,"-vec_view_stash",&flg,PETSC_NULL);CHKERRQ(ierr);
  if (flg) {
    PetscViewer viewer;
    ierr = PetscViewerASCIIGetStdout(((PetscObject)vec)->comm,&viewer);CHKERRQ(ierr);
    ierr = VecStashView(vec,viewer);CHKERRQ(ierr);
  }

  ierr = PetscLogEventBegin(VEC_AssemblyBegin,vec,0,0,0);CHKERRQ(ierr);
  if (vec->ops->assemblybegin) {
    ierr = (*vec->ops->assemblybegin)(vec);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(VEC_AssemblyBegin,vec,0,0,0);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)vec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecView_Private"
/*
  Processes command line options to determine if/how a matrix
  is to be viewed. Called by VecAssemblyEnd().

.seealso: MatView_Private()

*/
PetscErrorCode PETSCVEC_DLLEXPORT VecView_Private(Vec vec)
{
  PetscErrorCode ierr;
  PetscTruth     flg = PETSC_FALSE;

  PetscFunctionBegin;
  ierr = PetscOptionsBegin(((PetscObject)vec)->comm,((PetscObject)vec)->prefix,"Vector Options","Vec");CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-vec_view","Print vector to stdout","VecView",flg,&flg,PETSC_NULL);CHKERRQ(ierr);
    if (flg) {
      PetscViewer viewer;
      ierr = PetscViewerASCIIGetStdout(((PetscObject)vec)->comm,&viewer);CHKERRQ(ierr);
      ierr = VecView(vec,viewer);CHKERRQ(ierr);
    }
    flg  = PETSC_FALSE;
    ierr = PetscOptionsTruth("-vec_view_matlab","Print vector to stdout in a format Matlab can read","VecView",flg,&flg,PETSC_NULL);CHKERRQ(ierr);
    if (flg) {
      PetscViewer viewer;
      ierr = PetscViewerASCIIGetStdout(((PetscObject)vec)->comm,&viewer);CHKERRQ(ierr);
      ierr = PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_MATLAB);CHKERRQ(ierr);
      ierr = VecView(vec,viewer);CHKERRQ(ierr);
      ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
    }
#if defined(PETSC_HAVE_MATLAB_ENGINE)
    flg  = PETSC_FALSE;
    ierr = PetscOptionsTruth("-vec_view_matlab_file","Print vector to matlaboutput.mat format Matlab can read","VecView",flg,&flg,PETSC_NULL);CHKERRQ(ierr);
    if (flg) {
      ierr = VecView(vec,PETSC_VIEWER_MATLAB_(((PetscObject)vec)->comm));CHKERRQ(ierr);
    }
#endif
#if defined(PETSC_USE_SOCKET_VIEWER)
    flg  = PETSC_FALSE;
    ierr = PetscOptionsTruth("-vec_view_socket","Send vector to socket (can be read from matlab)","VecView",flg,&flg,PETSC_NULL);CHKERRQ(ierr);
    if (flg) {
      ierr = VecView(vec,PETSC_VIEWER_SOCKET_(((PetscObject)vec)->comm));CHKERRQ(ierr);
      ierr = PetscViewerFlush(PETSC_VIEWER_SOCKET_(((PetscObject)vec)->comm));CHKERRQ(ierr);
    }
#endif
    flg  = PETSC_FALSE;
    ierr = PetscOptionsTruth("-vec_view_binary","Save vector to file in binary format","VecView",flg,&flg,PETSC_NULL);CHKERRQ(ierr);
    if (flg) {
      ierr = VecView(vec,PETSC_VIEWER_BINARY_(((PetscObject)vec)->comm));CHKERRQ(ierr);
      ierr = PetscViewerFlush(PETSC_VIEWER_BINARY_(((PetscObject)vec)->comm));CHKERRQ(ierr);
    }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  /* These invoke PetscDrawGetDraw which invokes PetscOptionsBegin/End, */
  /* hence they should not be inside the above PetscOptionsBegin/End block. */
  flg  = PETSC_FALSE;
  ierr = PetscOptionsGetTruth(((PetscObject)vec)->prefix,"-vec_view_draw",&flg,PETSC_NULL);CHKERRQ(ierr);
  if (flg) {
    ierr = VecView(vec,PETSC_VIEWER_DRAW_(((PetscObject)vec)->comm));CHKERRQ(ierr);
    ierr = PetscViewerFlush(PETSC_VIEWER_DRAW_(((PetscObject)vec)->comm));CHKERRQ(ierr);
  }
  flg  = PETSC_FALSE;
  ierr = PetscOptionsGetTruth(((PetscObject)vec)->prefix,"-vec_view_draw_lg",&flg,PETSC_NULL);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscViewerSetFormat(PETSC_VIEWER_DRAW_(((PetscObject)vec)->comm),PETSC_VIEWER_DRAW_LG);CHKERRQ(ierr);
    ierr = VecView(vec,PETSC_VIEWER_DRAW_(((PetscObject)vec)->comm));CHKERRQ(ierr);
    ierr = PetscViewerFlush(PETSC_VIEWER_DRAW_(((PetscObject)vec)->comm));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecAssemblyEnd"
/*@
   VecAssemblyEnd - Completes assembling the vector.  This routine should
   be called after VecAssemblyBegin().

   Collective on Vec

   Input Parameter:
.  vec - the vector

   Options Database Keys:
+  -vec_view - Prints vector in ASCII format
.  -vec_view_matlab - Prints vector in ASCII Matlab format to stdout
.  -vec_view_matlab_file - Prints vector in Matlab format to matlaboutput.mat
.  -vec_view_draw - Activates vector viewing using drawing tools
.  -display <name> - Sets display name (default is host)
.  -draw_pause <sec> - Sets number of seconds to pause after display
-  -vec_view_socket - Activates vector viewing using a socket

   Level: beginner

.seealso: VecAssemblyBegin(), VecSetValues()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT VecAssemblyEnd(Vec vec)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(vec,VEC_COOKIE,1);
  ierr = PetscLogEventBegin(VEC_AssemblyEnd,vec,0,0,0);CHKERRQ(ierr);
  PetscValidType(vec,1);
  if (vec->ops->assemblyend) {
    ierr = (*vec->ops->assemblyend)(vec);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(VEC_AssemblyEnd,vec,0,0,0);CHKERRQ(ierr);
  ierr = VecView_Private(vec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecPointwiseMax"
/*@
   VecPointwiseMax - Computes the componentwise maximum w_i = max(x_i, y_i).

   Collective on Vec

   Input Parameters:
.  x, y  - the vectors

   Output Parameter:
.  w - the result

   Level: advanced

   Notes: any subset of the x, y, and w may be the same vector.
          For complex numbers compares only the real part

   Concepts: vector^pointwise multiply

.seealso: VecPointwiseDivide(), VecPointwiseMult(), VecPointwiseMin(), VecPointwiseMaxAbs(), VecMaxPointwiseDivide()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT VecPointwiseMax(Vec w,Vec x,Vec y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(w,VEC_COOKIE,1);
  PetscValidHeaderSpecific(x,VEC_COOKIE,2);
  PetscValidHeaderSpecific(y,VEC_COOKIE,3);
  PetscValidType(w,1);
  PetscValidType(x,2);
  PetscValidType(y,3);
  PetscCheckSameTypeAndComm(x,2,y,3);
  PetscCheckSameTypeAndComm(y,3,w,1);
  if (x->map->N != y->map->N || x->map->N != w->map->N) SETERRQ(PETSC_ERR_ARG_INCOMP,"Incompatible vector global lengths");
  if (x->map->n != y->map->n || x->map->n != w->map->n) SETERRQ(PETSC_ERR_ARG_INCOMP,"Incompatible vector local lengths");

  ierr = (*w->ops->pointwisemax)(w,x,y);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)w);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "VecPointwiseMin"
/*@
   VecPointwiseMin - Computes the componentwise minimum w_i = min(x_i, y_i).

   Collective on Vec

   Input Parameters:
.  x, y  - the vectors

   Output Parameter:
.  w - the result

   Level: advanced

   Notes: any subset of the x, y, and w may be the same vector.
          For complex numbers compares only the real part

   Concepts: vector^pointwise multiply

.seealso: VecPointwiseDivide(), VecPointwiseMult(), VecPointwiseMin(), VecPointwiseMaxAbs(), VecMaxPointwiseDivide()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT VecPointwiseMin(Vec w,Vec x,Vec y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(w,VEC_COOKIE,1);
  PetscValidHeaderSpecific(x,VEC_COOKIE,2);
  PetscValidHeaderSpecific(y,VEC_COOKIE,3);
  PetscValidType(w,1);
  PetscValidType(x,2);
  PetscValidType(y,3);
  PetscCheckSameTypeAndComm(x,2,y,3);
  PetscCheckSameTypeAndComm(y,3,w,1);
  if (x->map->N != y->map->N || x->map->N != w->map->N) SETERRQ(PETSC_ERR_ARG_INCOMP,"Incompatible vector global lengths");
  if (x->map->n != y->map->n || x->map->n != w->map->n) SETERRQ(PETSC_ERR_ARG_INCOMP,"Incompatible vector local lengths");

  ierr = (*w->ops->pointwisemin)(w,x,y);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)w);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecPointwiseMaxAbs"
/*@
   VecPointwiseMaxAbs - Computes the componentwise maximum of the absolute values w_i = max(abs(x_i), abs(y_i)).

   Collective on Vec

   Input Parameters:
.  x, y  - the vectors

   Output Parameter:
.  w - the result

   Level: advanced

   Notes: any subset of the x, y, and w may be the same vector.

   Concepts: vector^pointwise multiply

.seealso: VecPointwiseDivide(), VecPointwiseMult(), VecPointwiseMin(), VecPointwiseMax(), VecMaxPointwiseDivide()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT VecPointwiseMaxAbs(Vec w,Vec x,Vec y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(w,VEC_COOKIE,1);
  PetscValidHeaderSpecific(x,VEC_COOKIE,2);
  PetscValidHeaderSpecific(y,VEC_COOKIE,3);
  PetscValidType(w,1);
  PetscValidType(x,2);
  PetscValidType(y,3);
  PetscCheckSameTypeAndComm(x,2,y,3);
  PetscCheckSameTypeAndComm(y,3,w,1);
  if (x->map->N != y->map->N || x->map->N != w->map->N) SETERRQ(PETSC_ERR_ARG_INCOMP,"Incompatible vector global lengths");
  if (x->map->n != y->map->n || x->map->n != w->map->n) SETERRQ(PETSC_ERR_ARG_INCOMP,"Incompatible vector local lengths");

  ierr = (*w->ops->pointwisemaxabs)(w,x,y);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)w);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecPointwiseDivide"
/*@
   VecPointwiseDivide - Computes the componentwise division w = x/y.

   Collective on Vec

   Input Parameters:
.  x, y  - the vectors

   Output Parameter:
.  w - the result

   Level: advanced

   Notes: any subset of the x, y, and w may be the same vector.

   Concepts: vector^pointwise divide

.seealso: VecPointwiseMult(), VecPointwiseMax(), VecPointwiseMin(), VecPointwiseMaxAbs(), VecMaxPointwiseDivide()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT VecPointwiseDivide(Vec w,Vec x,Vec y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(w,VEC_COOKIE,1);
  PetscValidHeaderSpecific(x,VEC_COOKIE,2);
  PetscValidHeaderSpecific(y,VEC_COOKIE,3);
  PetscValidType(w,1);
  PetscValidType(x,2);
  PetscValidType(y,3);
  PetscCheckSameTypeAndComm(x,2,y,3);
  PetscCheckSameTypeAndComm(y,3,w,1);
  if (x->map->N != y->map->N || x->map->N != w->map->N) SETERRQ(PETSC_ERR_ARG_INCOMP,"Incompatible vector global lengths");
  if (x->map->n != y->map->n || x->map->n != w->map->n) SETERRQ(PETSC_ERR_ARG_INCOMP,"Incompatible vector local lengths");

  ierr = (*w->ops->pointwisedivide)(w,x,y);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)w);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "VecDuplicate"
/*@
   VecDuplicate - Creates a new vector of the same type as an existing vector.

   Collective on Vec

   Input Parameters:
.  v - a vector to mimic

   Output Parameter:
.  newv - location to put new vector

   Notes:
   VecDuplicate() does not copy the vector, but rather allocates storage
   for the new vector.  Use VecCopy() to copy a vector.

   Use VecDestroy() to free the space. Use VecDuplicateVecs() to get several
   vectors.

   Level: beginner

.seealso: VecDestroy(), VecDuplicateVecs(), VecCreate(), VecCopy()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT VecDuplicate(Vec v,Vec *newv)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_COOKIE,1);
  PetscValidPointer(newv,2);
  PetscValidType(v,1);
  ierr = (*v->ops->duplicate)(v,newv);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)*newv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecDestroy"
/*@
   VecDestroy - Destroys a vector.

   Collective on Vec

   Input Parameters:
.  v  - the vector

   Level: beginner

.seealso: VecDuplicate(), VecDestroyVecs()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT VecDestroy(Vec v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_COOKIE,1);
  if (--((PetscObject)v)->refct > 0) PetscFunctionReturn(0);
  /* destroy the internal part */
  if (v->ops->destroy) {
    ierr = (*v->ops->destroy)(v);CHKERRQ(ierr);
  }
  /* destroy the external/common part */
  if (v->mapping) {
    ierr = ISLocalToGlobalMappingDestroy(v->mapping);CHKERRQ(ierr);
  }
  if (v->bmapping) {
    ierr = ISLocalToGlobalMappingDestroy(v->bmapping);CHKERRQ(ierr);
  }
  ierr = PetscLayoutDestroy(v->map);CHKERRQ(ierr);
  ierr = PetscHeaderDestroy(v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecDuplicateVecs"
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
PetscErrorCode PETSCVEC_DLLEXPORT VecDuplicateVecs(Vec v,PetscInt m,Vec *V[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_COOKIE,1);
  PetscValidPointer(V,3);
  PetscValidType(v,1);
  ierr = (*v->ops->duplicatevecs)(v, m,V);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecDestroyVecs"
/*@C
   VecDestroyVecs - Frees a block of vectors obtained with VecDuplicateVecs().

   Collective on Vec

   Input Parameters:
+  vv - pointer to array of vector pointers
-  m - the number of vectors previously obtained

   Fortran Note:
   The Fortran interface is slightly different from that given below.
   See the Fortran chapter of the users manual and
   petsc/src/vec/examples for details.

   Level: intermediate

.seealso: VecDuplicateVecs(), VecDestroyVecsf90()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT VecDestroyVecs(Vec vv[],PetscInt m)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(vv,1);
  PetscValidHeaderSpecific(*vv,VEC_COOKIE,1);
  PetscValidType(*vv,1);
  ierr = (*(*vv)->ops->destroyvecs)(vv,m);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "VecViewFromOptions"
/*@
  VecViewFromOptions - This function visualizes the vector based upon user options.

  Collective on Vec

  Input Parameters:
. vec   - The vector
. title - The title (currently ignored)

  Level: intermediate

.keywords: Vec, view, options, database
.seealso: VecSetFromOptions(), VecView()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT VecViewFromOptions(Vec vec, const char *title)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecView_Private(vec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecView"
/*@C
   VecView - Views a vector object.

   Collective on Vec

   Input Parameters:
+  vec - the vector
-  viewer - an optional visualization context

   Notes:
   The available visualization contexts include
+     PETSC_VIEWER_STDOUT_SELF - standard output (default)
-     PETSC_VIEWER_STDOUT_WORLD - synchronized standard
         output where only the first processor opens
         the file.  All other processors send their
         data to the first processor to print.

   You can change the format the vector is printed using the
   option PetscViewerSetFormat().

   The user can open alternative visualization contexts with
+    PetscViewerASCIIOpen() - Outputs vector to a specified file
.    PetscViewerBinaryOpen() - Outputs vector in binary to a
         specified file; corresponding input uses VecLoad()
.    PetscViewerDrawOpen() - Outputs vector to an X window display
-    PetscViewerSocketOpen() - Outputs vector to Socket viewer

   The user can call PetscViewerSetFormat() to specify the output
   format of ASCII printed objects (when using PETSC_VIEWER_STDOUT_SELF,
   PETSC_VIEWER_STDOUT_WORLD and PetscViewerASCIIOpen).  Available formats include
+    PETSC_VIEWER_DEFAULT - default, prints vector contents
.    PETSC_VIEWER_ASCII_MATLAB - prints vector contents in Matlab format
.    PETSC_VIEWER_ASCII_INDEX - prints vector contents, including indices of vector elements
-    PETSC_VIEWER_ASCII_COMMON - prints vector contents, using a
         format common among all vector types

   Notes for HDF5 Viewer: the name of the Vec (given with PetscObjectSetName() is the name that is used
   for the object in the HDF5 file. If you wish to store the same vector to the HDF5 viewer (with different values,
   obviously) several times, you must change its name each time before calling the VecView(). The name you use
   here should equal the name that you use in the Vec object that you use with VecLoadIntoVector().

   See the manual page for VecLoad() on the exact format the binary viewer stores
   the values in the file.

   Level: beginner

   Concepts: vector^printing
   Concepts: vector^saving to disk

.seealso: PetscViewerASCIIOpen(), PetscViewerDrawOpen(), PetscDrawLGCreate(),
          PetscViewerSocketOpen(), PetscViewerBinaryOpen(), VecLoad(), PetscViewerCreate(),
          PetscRealView(), PetscScalarView(), PetscIntView()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT VecView(Vec vec,PetscViewer viewer)
{
  PetscErrorCode    ierr;
  PetscViewerFormat format;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(vec,VEC_COOKIE,1);
  PetscValidType(vec,1);
  if (!viewer) {
    ierr = PetscViewerASCIIGetStdout(((PetscObject)vec)->comm,&viewer);CHKERRQ(ierr);
  }
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_COOKIE,2);
  PetscCheckSameComm(vec,1,viewer,2);
  if (vec->stash.n || vec->bstash.n) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Must call VecAssemblyBegin/End() before viewing this vector");

  ierr = PetscLogEventBegin(VEC_View,vec,viewer,0,0);CHKERRQ(ierr);
  /*
     Check if default viewer has been overridden, but user request it anyways
  */
  ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
  if (vec->ops->viewnative && format == PETSC_VIEWER_NATIVE) {
    ierr   = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
    ierr = (*vec->ops->viewnative)(vec,viewer);CHKERRQ(ierr);
    ierr   = PetscViewerPushFormat(viewer,PETSC_VIEWER_NATIVE);CHKERRQ(ierr);
  } else {
    ierr = (*vec->ops->view)(vec,viewer);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(VEC_View,vec,viewer,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecGetSize"
/*@
   VecGetSize - Returns the global number of elements of the vector.

   Not Collective

   Input Parameter:
.  x - the vector

   Output Parameters:
.  size - the global length of the vector

   Level: beginner

   Concepts: vector^local size

.seealso: VecGetLocalSize()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT VecGetSize(Vec x,PetscInt *size)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE,1);
  PetscValidIntPointer(size,2);
  PetscValidType(x,1);
  ierr = (*x->ops->getsize)(x,size);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecGetLocalSize"
/*@
   VecGetLocalSize - Returns the number of elements of the vector stored
   in local memory. This routine may be implementation dependent, so use
   with care.

   Not Collective

   Input Parameter:
.  x - the vector

   Output Parameter:
.  size - the length of the local piece of the vector

   Level: beginner

   Concepts: vector^size

.seealso: VecGetSize()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT VecGetLocalSize(Vec x,PetscInt *size)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE,1);
  PetscValidIntPointer(size,2);
  PetscValidType(x,1);
  ierr = (*x->ops->getlocalsize)(x,size);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecGetOwnershipRange"
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
+  low - the first local element, pass in PETSC_NULL if not interested
-  high - one more than the last local element, pass in PETSC_NULL if not interested

   Note:
   The high argument is one more than the last element stored locally.

   Fortran: PETSC_NULL_INTEGER should be used instead of PETSC_NULL

   Level: beginner

   Concepts: ownership^of vectors
   Concepts: vector^ownership of elements

.seealso:   MatGetOwnershipRange(), MatGetOwnershipRanges(), VecGetOwnershipRanges()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT VecGetOwnershipRange(Vec x,PetscInt *low,PetscInt *high)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE,1);
  PetscValidType(x,1);
  if (low) PetscValidIntPointer(low,2);
  if (high) PetscValidIntPointer(high,3);
  if (low)  *low  = x->map->rstart;
  if (high) *high = x->map->rend;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecGetOwnershipRanges"
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

   Concepts: ownership^of vectors
   Concepts: vector^ownership of elements

.seealso:   MatGetOwnershipRange(), MatGetOwnershipRanges(), VecGetOwnershipRange()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT VecGetOwnershipRanges(Vec x,const PetscInt *ranges[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE,1);
  PetscValidType(x,1);
  ierr = PetscLayoutGetRanges(x->map,ranges);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecSetOption"
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
          in ix in calls to VecSetValues or VecGetValues. These rows are simply
          ignored.

   Level: intermediate

@*/
PetscErrorCode PETSCVEC_DLLEXPORT VecSetOption(Vec x,VecOption op,PetscTruth flag)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE,1);
  PetscValidType(x,1);
  if (x->ops->setoption) {
    ierr = (*x->ops->setoption)(x,op,flag);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecDuplicateVecs_Default"
/* Default routines for obtaining and releasing; */
/* may be used by any implementation */
PetscErrorCode VecDuplicateVecs_Default(Vec w,PetscInt m,Vec *V[])
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(w,VEC_COOKIE,1);
  PetscValidPointer(V,3);
  if (m <= 0) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"m must be > 0: m = %D",m);
  ierr = PetscMalloc(m*sizeof(Vec*),V);CHKERRQ(ierr);
  for (i=0; i<m; i++) {ierr = VecDuplicate(w,*V+i);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecDestroyVecs_Default"
PetscErrorCode VecDestroyVecs_Default(Vec v[], PetscInt m)
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  PetscValidPointer(v,1);
  if (m <= 0) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"m must be > 0: m = %D",m);
  for (i=0; i<m; i++) {ierr = VecDestroy(v[i]);CHKERRQ(ierr);}
  ierr = PetscFree(v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecResetArray"
/*@
   VecResetArray - Resets a vector to use its default memory. Call this
   after the use of VecPlaceArray().

   Not Collective

   Input Parameters:
.  vec - the vector

   Level: developer

.seealso: VecGetArray(), VecRestoreArray(), VecReplaceArray(), VecPlaceArray()

@*/
PetscErrorCode PETSCVEC_DLLEXPORT VecResetArray(Vec vec)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(vec,VEC_COOKIE,1);
  PetscValidType(vec,1);
  if (vec->ops->resetarray) {
    ierr = (*vec->ops->resetarray)(vec);CHKERRQ(ierr);
  } else {
    SETERRQ(PETSC_ERR_SUP,"Cannot reset array in this type of vector");
  }
  ierr = PetscObjectStateIncrease((PetscObject)vec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecLoadIntoVector"
/*@C
  VecLoadIntoVector - Loads a vector that has been stored in binary (or HDF5) format
  with VecView().

  Collective on PetscViewer

  Input Parameters:
+ viewer - binary file viewer, obtained from PetscViewerBinaryOpen()
- vec - vector to contain files values (must be of correct length)

  Level: intermediate

  Notes:
  The input file must contain the full global vector, as
  written by the routine VecView().

  Use VecLoad() to create the vector as the values are read in

  If using HDF5, you must assign the Vec the same name as was used in the Vec that was stored
  in the file using PetscObjectSetName(). Otherwise you will get the error message
$     Cannot H5Dopen2() with Vec named NAMEOFOBJECT

  Notes for advanced users:
  Most users should not need to know the details of the binary storage
  format, since VecLoad() and VecView() completely hide these details.
  But for anyone who's interested, the standard binary matrix storage
  format is
.vb
     int    VEC_FILE_COOKIE
     int    number of rows
     PetscScalar *values of all nonzeros
.ve

   In addition, PETSc automatically does the byte swapping for
machines that store the bytes reversed, e.g.  DEC alpha, freebsd,
linux, Windows and the paragon; thus if you write your own binary
read/write routines you have to swap the bytes; see PetscBinaryRead()
and PetscBinaryWrite() to see how this may be done.

   Concepts: vector^loading from file

.seealso: PetscViewerBinaryOpen(), VecView(), MatLoad(), VecLoad()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT VecLoadIntoVector(PetscViewer viewer,Vec vec)
{
  PetscErrorCode    ierr;
  PetscViewerFormat format;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_COOKIE,1);
  PetscValidHeaderSpecific(vec,VEC_COOKIE,2);
  PetscValidType(vec,2);
  if (!vec->ops->loadintovector) {
    SETERRQ(PETSC_ERR_SUP,"Vector does not support load");
  }
  ierr = PetscLogEventBegin(VEC_Load,viewer,0,0,0);CHKERRQ(ierr);
  /*
     Check if default loader has been overridden, but user request it anyways
  */
  ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
  if (vec->ops->loadintovectornative && format == PETSC_VIEWER_NATIVE) {
    ierr   = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
    ierr = (*vec->ops->loadintovectornative)(viewer,vec);CHKERRQ(ierr);
    ierr   = PetscViewerPushFormat(viewer,PETSC_VIEWER_NATIVE);CHKERRQ(ierr);
  } else {
    ierr = (*vec->ops->loadintovector)(viewer,vec);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(VEC_Load,viewer,0,0,0);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)vec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecReciprocal"
/*@
   VecReciprocal - Replaces each component of a vector by its reciprocal.

   Collective on Vec

   Input Parameter:
.  vec - the vector

   Output Parameter:
.  vec - the vector reciprocal

   Level: intermediate

   Concepts: vector^reciprocal

.seealso: VecLog(), VecExp(), VecSqrt()

@*/
PetscErrorCode PETSCVEC_DLLEXPORT VecReciprocal(Vec vec)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(vec,VEC_COOKIE,1);
  PetscValidType(vec,1);
  if (vec->stash.insertmode != NOT_SET_VALUES) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled vector");
  if (!vec->ops->reciprocal) {
    SETERRQ(PETSC_ERR_SUP,"Vector does not support reciprocal operation");
  }
  ierr = (*vec->ops->reciprocal)(vec);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)vec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecSetOperation"
PetscErrorCode PETSCVEC_DLLEXPORT VecSetOperation(Vec vec,VecOperation op, void (*f)(void))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(vec,VEC_COOKIE,1);
  /* save the native version of the viewer */
  if (op == VECOP_VIEW && !vec->ops->viewnative) {
    vec->ops->viewnative = vec->ops->view;
  } else if (op == VECOP_LOADINTOVECTOR && !vec->ops->loadintovectornative) {
    vec->ops->loadintovectornative = vec->ops->loadintovector;
  }
  (((void(**)(void))vec->ops)[(int)op]) = f;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "VecStashSetInitialSize"
/*@
   VecStashSetInitialSize - sets the sizes of the vec-stash, that is
   used during the assembly process to store values that belong to
   other processors.

   Collective on Vec

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

   Concepts: vector^stash
   Concepts: stash^vector

.seealso: VecSetBlockSize(), VecSetValues(), VecSetValuesBlocked(), VecStashView()

@*/
PetscErrorCode PETSCVEC_DLLEXPORT VecStashSetInitialSize(Vec vec,PetscInt size,PetscInt bsize)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(vec,VEC_COOKIE,1);
  ierr = VecStashSetInitialSize_Private(&vec->stash,size);CHKERRQ(ierr);
  ierr = VecStashSetInitialSize_Private(&vec->bstash,bsize);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecConjugate"
/*@
   VecConjugate - Conjugates a vector.

   Collective on Vec

   Input Parameters:
.  x - the vector

   Level: intermediate

   Concepts: vector^conjugate

@*/
PetscErrorCode PETSCVEC_DLLEXPORT VecConjugate(Vec x)
{
#ifdef PETSC_USE_COMPLEX
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE,1);
  PetscValidType(x,1);
  if (x->stash.insertmode != NOT_SET_VALUES) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled vector");
  ierr = (*x->ops->conjugate)(x);CHKERRQ(ierr);
  /* we need to copy norms here */
  ierr = PetscObjectStateIncrease((PetscObject)x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
#else
  return(0);
#endif
}

#undef __FUNCT__
#define __FUNCT__ "VecPointwiseMult"
/*@
   VecPointwiseMult - Computes the componentwise multiplication w = x*y.

   Collective on Vec

   Input Parameters:
.  x, y  - the vectors

   Output Parameter:
.  w - the result

   Level: advanced

   Notes: any subset of the x, y, and w may be the same vector.

   Concepts: vector^pointwise multiply

.seealso: VecPointwiseDivide(), VecPointwiseMax(), VecPointwiseMin(), VecPointwiseMaxAbs(), VecMaxPointwiseDivide()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT VecPointwiseMult(Vec w, Vec x,Vec y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(w,VEC_COOKIE,1);
  PetscValidHeaderSpecific(x,VEC_COOKIE,2);
  PetscValidHeaderSpecific(y,VEC_COOKIE,3);
  PetscValidType(w,1);
  PetscValidType(x,2);
  PetscValidType(y,3);
  PetscCheckSameTypeAndComm(x,2,y,3);
  PetscCheckSameTypeAndComm(y,3,w,1);
  if (x->map->n != y->map->n || x->map->n != w->map->n) SETERRQ(PETSC_ERR_ARG_INCOMP,"Incompatible vector local lengths");

  ierr = PetscLogEventBegin(VEC_PointwiseMult,x,y,w,0);CHKERRQ(ierr);
  ierr = (*w->ops->pointwisemult)(w,x,y);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(VEC_PointwiseMult,x,y,w,0);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)w);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecSetRandom"
/*@
   VecSetRandom - Sets all components of a vector to random numbers.

   Collective on Vec

   Input Parameters:
+  x  - the vector
-  rctx - the random number context, formed by PetscRandomCreate(), or PETSC_NULL and
          it will create one internally.

   Output Parameter:
.  x  - the vector

   Example of Usage:
.vb
     PetscRandomCreate(PETSC_COMM_WORLD,&rctx);
     VecSetRandom(x,rctx);
     PetscRandomDestroy(rctx);
.ve

   Level: intermediate

   Concepts: vector^setting to random
   Concepts: random^vector

.seealso: VecSet(), VecSetValues(), PetscRandomCreate(), PetscRandomDestroy()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT VecSetRandom(Vec x,PetscRandom rctx)
{
  PetscErrorCode ierr;
  PetscRandom    randObj = PETSC_NULL;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE,1);
  if (rctx) PetscValidHeaderSpecific(rctx,PETSC_RANDOM_COOKIE,2);
  PetscValidType(x,1);
  if (x->stash.insertmode != NOT_SET_VALUES) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled vector");

  if (!rctx) {
    MPI_Comm    comm;
    ierr = PetscObjectGetComm((PetscObject)x,&comm);CHKERRQ(ierr);
    ierr = PetscRandomCreate(comm,&randObj);CHKERRQ(ierr);
    ierr = PetscRandomSetFromOptions(randObj);CHKERRQ(ierr);
    rctx = randObj;
  }

  ierr = PetscLogEventBegin(VEC_SetRandom,x,rctx,0,0);CHKERRQ(ierr);
  ierr = (*x->ops->setrandom)(x,rctx);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(VEC_SetRandom,x,rctx,0,0);CHKERRQ(ierr);

  if (randObj) {
    ierr = PetscRandomDestroy(randObj);CHKERRQ(ierr);
  }
  ierr = PetscObjectStateIncrease((PetscObject)x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecZeroEntries"
/*@
  VecZeroEntries - puts a 0.0 in each element of a vector

  Collective on Vec

  Input Parameter:
. vec - The vector

  Level: beginner

.keywords: Vec, set, options, database
.seealso: VecCreate(),  VecSetOptionsPrefix(), VecSet(), VecSetValues()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT VecZeroEntries(Vec vec)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = VecSet(vec,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecSetTypeFromOptions_Private"
/*
  VecSetTypeFromOptions_Private - Sets the type of vector from user options. Defaults to a PETSc sequential vector on one
  processor and a PETSc MPI vector on more than one processor.

  Collective on Vec

  Input Parameter:
. vec - The vector

  Level: intermediate

.keywords: Vec, set, options, database, type
.seealso: VecSetFromOptions(), VecSetType()
*/
static PetscErrorCode VecSetTypeFromOptions_Private(Vec vec)
{
  PetscTruth     opt;
  const VecType  defaultType;
  char           typeName[256];
  PetscMPIInt    size;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (((PetscObject)vec)->type_name) {
    defaultType = ((PetscObject)vec)->type_name;
  } else {
    ierr = MPI_Comm_size(((PetscObject)vec)->comm, &size);CHKERRQ(ierr);
    if (size > 1) {
      defaultType = VECMPI;
    } else {
      defaultType = VECSEQ;
    }
  }

  if (!VecRegisterAllCalled) {ierr = VecRegisterAll(PETSC_NULL);CHKERRQ(ierr);}
  ierr = PetscOptionsList("-vec_type","Vector type","VecSetType",VecList,defaultType,typeName,256,&opt);CHKERRQ(ierr);
  if (opt) {
    ierr = VecSetType(vec, typeName);CHKERRQ(ierr);
  } else {
    ierr = VecSetType(vec, defaultType);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecSetFromOptions"
/*@
  VecSetFromOptions - Configures the vector from the options database.

  Collective on Vec

  Input Parameter:
. vec - The vector

  Notes:  To see all options, run your program with the -help option, or consult the users manual.
          Must be called after VecCreate() but before the vector is used.

  Level: beginner

  Concepts: vectors^setting options
  Concepts: vectors^setting type

.keywords: Vec, set, options, database
.seealso: VecCreate(), VecSetOptionsPrefix()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT VecSetFromOptions(Vec vec)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(vec,VEC_COOKIE,1);

  ierr = PetscOptionsBegin(((PetscObject)vec)->comm, ((PetscObject)vec)->prefix, "Vector options", "Vec");CHKERRQ(ierr);
    /* Handle vector type options */
    ierr = VecSetTypeFromOptions_Private(vec);CHKERRQ(ierr);

    /* Handle specific vector options */
    if (vec->ops->setfromoptions) {
      ierr = (*vec->ops->setfromoptions)(vec);CHKERRQ(ierr);
    }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  ierr = VecViewFromOptions(vec, ((PetscObject)vec)->name);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecSetSizes"
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
PetscErrorCode PETSCVEC_DLLEXPORT VecSetSizes(Vec v, PetscInt n, PetscInt N)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v, VEC_COOKIE,1);
  if (N > 0 && n > N) SETERRQ2(PETSC_ERR_ARG_INCOMP,"Local size %D cannot be larger than global size %D",n,N);
  if ((v->map->n >= 0 || v->map->N >= 0) && (v->map->n != n || v->map->N != N)) SETERRQ4(PETSC_ERR_SUP,"Cannot change/reset vector sizes to %D local %D global after previously setting them to %D local %D global",n,N,v->map->n,v->map->N);
  v->map->n = n;
  v->map->N = N;
  if (v->ops->create) {
    ierr = (*v->ops->create)(v);CHKERRQ(ierr);
    v->ops->create = 0;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecSetBlockSize"
/*@
   VecSetBlockSize - Sets the blocksize for future calls to VecSetValuesBlocked()
   and VecSetValuesBlockedLocal().

   Collective on Vec

   Input Parameter:
+  v - the vector
-  bs - the blocksize

   Notes:
   All vectors obtained by VecDuplicate() inherit the same blocksize.

   Level: advanced

.seealso: VecSetValuesBlocked(), VecSetLocalToGlobalMappingBlock(), VecGetBlockSize()

  Concepts: block size^vectors
@*/
PetscErrorCode PETSCVEC_DLLEXPORT VecSetBlockSize(Vec v,PetscInt bs)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_COOKIE,1);
  if (bs <= 0) bs = 1;
  if (bs == v->map->bs) PetscFunctionReturn(0);
  if (v->map->N != -1 && v->map->N % bs) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Vector length not divisible by blocksize %D %D",v->map->N,bs);
  if (v->map->n != -1 && v->map->n % bs) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Local vector length not divisible by blocksize %D %D\n\
   Try setting blocksize before setting the vector type",v->map->n,bs);

  v->map->bs   = bs;
  v->bstash.bs = bs; /* use the same blocksize for the vec's block-stash */
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecGetBlockSize"
/*@
   VecGetBlockSize - Gets the blocksize for the vector, i.e. what is used for VecSetValuesBlocked()
   and VecSetValuesBlockedLocal().

   Collective on Vec

   Input Parameter:
.  v - the vector

   Output Parameter:
.  bs - the blocksize

   Notes:
   All vectors obtained by VecDuplicate() inherit the same blocksize.

   Level: advanced

.seealso: VecSetValuesBlocked(), VecSetLocalToGlobalMappingBlock(), VecSetBlockSize()

   Concepts: vector^block size
   Concepts: block^vector

@*/
PetscErrorCode PETSCVEC_DLLEXPORT VecGetBlockSize(Vec v,PetscInt *bs)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_COOKIE,1);
  PetscValidIntPointer(bs,2);
  *bs = v->map->bs;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecValid"
/*@
   VecValid - Checks whether a vector object is valid.

   Not Collective

   Input Parameter:
.  v - the object to check

   Output Parameter:
.  flg - flag indicating vector status, either
   PETSC_TRUE if vector is valid, or PETSC_FALSE otherwise.

   Level: developer

@*/
PetscErrorCode PETSCVEC_DLLEXPORT VecValid(Vec v,PetscTruth *flg)
{
  PetscFunctionBegin;
  PetscValidIntPointer(flg,2);
  if (!v)                                          *flg = PETSC_FALSE;
  else if (((PetscObject)v)->cookie != VEC_COOKIE) *flg = PETSC_FALSE;
  else                                             *flg = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecSetOptionsPrefix"
/*@C
   VecSetOptionsPrefix - Sets the prefix used for searching for all
   Vec options in the database.

   Collective on Vec

   Input Parameter:
+  v - the Vec context
-  prefix - the prefix to prepend to all option names

   Notes:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the hyphen.

   Level: advanced

.keywords: Vec, set, options, prefix, database

.seealso: VecSetFromOptions()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT VecSetOptionsPrefix(Vec v,const char prefix[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_COOKIE,1);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)v,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecAppendOptionsPrefix"
/*@C
   VecAppendOptionsPrefix - Appends to the prefix used for searching for all
   Vec options in the database.

   Collective on Vec

   Input Parameters:
+  v - the Vec context
-  prefix - the prefix to prepend to all option names

   Notes:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the hyphen.

   Level: advanced

.keywords: Vec, append, options, prefix, database

.seealso: VecGetOptionsPrefix()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT VecAppendOptionsPrefix(Vec v,const char prefix[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_COOKIE,1);
  ierr = PetscObjectAppendOptionsPrefix((PetscObject)v,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecGetOptionsPrefix"
/*@C
   VecGetOptionsPrefix - Sets the prefix used for searching for all
   Vec options in the database.

   Not Collective

   Input Parameter:
.  v - the Vec context

   Output Parameter:
.  prefix - pointer to the prefix string used

   Notes: On the fortran side, the user should pass in a string 'prefix' of
   sufficient length to hold the prefix.

   Level: advanced

.keywords: Vec, get, options, prefix, database

.seealso: VecAppendOptionsPrefix()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT VecGetOptionsPrefix(Vec v,const char *prefix[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_COOKIE,1);
  ierr = PetscObjectGetOptionsPrefix((PetscObject)v,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecSetUp"
/*@
   VecSetUp - Sets up the internal vector data structures for the later use.

   Collective on Vec

   Input Parameters:
.  v - the Vec context

   Notes:
   For basic use of the Vec classes the user need not explicitly call
   VecSetUp(), since these actions will happen automatically.

   Level: advanced

.keywords: Vec, setup

.seealso: VecCreate(), VecDestroy()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT VecSetUp(Vec v)
{
  PetscMPIInt    size;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_COOKIE,1);
  if (!((PetscObject)v)->type_name) {
    ierr = MPI_Comm_size(((PetscObject)v)->comm, &size);CHKERRQ(ierr);
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

#undef __FUNCT__
#define __FUNCT__ "VecCopy"
/*@
   VecCopy - Copies a vector. y <- x

   Collective on Vec

   Input Parameter:
.  x - the vector

   Output Parameter:
.  y - the copy

   Notes:
   For default parallel PETSc vectors, both x and y must be distributed in
   the same manner; local copies are done.

   Level: beginner

.seealso: VecDuplicate()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT VecCopy(Vec x,Vec y)
{
  PetscTruth     flgs[4];
  PetscReal      norms[4] = {0.0,0.0,0.0,0.0};
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE,1);
  PetscValidHeaderSpecific(y,VEC_COOKIE,2);
  PetscValidType(x,1);
  PetscValidType(y,2);
  if (x == y) PetscFunctionReturn(0);
  if (x->stash.insertmode != NOT_SET_VALUES) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled vector");
  if (x->map->n != y->map->n) SETERRQ(PETSC_ERR_ARG_INCOMP,"Incompatible vector local lengths");

  ierr = PetscLogEventBegin(VEC_Copy,x,y,0,0);CHKERRQ(ierr);
  ierr = (*x->ops->copy)(x,y);CHKERRQ(ierr);


  /*
   * Update cached data
  */
  /* in general we consider this object touched */
  ierr = PetscObjectStateIncrease((PetscObject)y);CHKERRQ(ierr);

  for (i=0; i<4; i++) {
    ierr = PetscObjectComposedDataGetReal((PetscObject)x,NormIds[i],norms[i],flgs[i]);CHKERRQ(ierr);
  }
  for (i=0; i<4; i++) {
    if (flgs[i]) {
      ierr = PetscObjectComposedDataSetReal((PetscObject)y,NormIds[i],norms[i]);CHKERRQ(ierr);
    }
  }

  ierr = PetscLogEventEnd(VEC_Copy,x,y,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecSwap"
/*@
   VecSwap - Swaps the vectors x and y.

   Collective on Vec

   Input Parameters:
.  x, y  - the vectors

   Level: advanced

   Concepts: vector^swapping values

@*/
PetscErrorCode PETSCVEC_DLLEXPORT VecSwap(Vec x,Vec y)
{
  PetscReal      normxs[4]={0.0,0.0,0.0,0.0},normys[4]={0.0,0.0,0.0,0.0};
  PetscTruth     flgxs[4],flgys[4];
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE,1);
  PetscValidHeaderSpecific(y,VEC_COOKIE,2);
  PetscValidType(x,1);
  PetscValidType(y,2);
  PetscCheckSameTypeAndComm(x,1,y,2);
  if (x->stash.insertmode != NOT_SET_VALUES) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled vector");
  if (y->stash.insertmode != NOT_SET_VALUES) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled vector");
  if (x->map->N != y->map->N) SETERRQ(PETSC_ERR_ARG_INCOMP,"Incompatible vector global lengths");
  if (x->map->n != y->map->n) SETERRQ(PETSC_ERR_ARG_INCOMP,"Incompatible vector local lengths");

  ierr = PetscLogEventBegin(VEC_Swap,x,y,0,0);CHKERRQ(ierr);
  ierr = (*x->ops->swap)(x,y);CHKERRQ(ierr);

  /* See if we have cached norms */
  for (i=0; i<4; i++) {
    ierr = PetscObjectComposedDataGetReal((PetscObject)x,NormIds[i],normxs[i],flgxs[i]);CHKERRQ(ierr);
    ierr = PetscObjectComposedDataGetReal((PetscObject)y,NormIds[i],normys[i],flgys[i]);CHKERRQ(ierr);
  }
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

#undef __FUNCT__
#define __FUNCT__ "VecStashView"
/*@
   VecStashView - Prints the entries in the vector stash and block stash.

   Collective on Vec

   Input Parameters:
+  v - the vector
-  viewer - the viewer

   Level: advanced

   Concepts: vector^stash
   Concepts: stash^vector

.seealso: VecSetBlockSize(), VecSetValues(), VecSetValuesBlocked()

@*/
PetscErrorCode PETSCVEC_DLLEXPORT VecStashView(Vec v,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank;
  PetscInt       i,j;
  PetscTruth     match;
  VecStash       *s;
  PetscScalar    val;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_COOKIE,1);
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_COOKIE,2);
  PetscCheckSameComm(v,1,viewer,2);

  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&match);CHKERRQ(ierr);
  if (!match) SETERRQ1(PETSC_ERR_SUP,"Stash viewer only works with ASCII viewer not %s\n",((PetscObject)v)->type_name);
  ierr = PetscViewerASCIIUseTabs(viewer,PETSC_FALSE);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(((PetscObject)v)->comm,&rank);CHKERRQ(ierr);
  s = &v->bstash;

  /* print block stash */
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

  ierr = PetscViewerASCIIUseTabs(viewer,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

