
/*
     Provides the interface functions for vector operations that do NOT have PetscScalar/PetscReal in the signature
   These are the vector functions the user calls.
*/
#include <petsc-private/vecimpl.h>    /*I "petscvec.h" I*/

/* Logging support */
PetscClassId  VEC_CLASSID;
PetscLogEvent  VEC_View, VEC_Max, VEC_Min, VEC_DotBarrier, VEC_Dot, VEC_MDotBarrier, VEC_MDot, VEC_TDot;
PetscLogEvent  VEC_Norm, VEC_Normalize, VEC_Scale, VEC_Copy, VEC_Set, VEC_AXPY, VEC_AYPX, VEC_WAXPY;
PetscLogEvent  VEC_MTDot, VEC_NormBarrier, VEC_MAXPY, VEC_Swap, VEC_AssemblyBegin, VEC_ScatterBegin, VEC_ScatterEnd;
PetscLogEvent  VEC_AssemblyEnd, VEC_PointwiseMult, VEC_SetValues, VEC_Load, VEC_ScatterBarrier;
PetscLogEvent  VEC_SetRandom, VEC_ReduceArithmetic, VEC_ReduceBarrier, VEC_ReduceCommunication,VEC_ReduceBegin,VEC_ReduceEnd,VEC_Ops;
PetscLogEvent  VEC_DotNormBarrier, VEC_DotNorm, VEC_AXPBYPCZ, VEC_CUSPCopyFromGPU, VEC_CUSPCopyToGPU;
PetscLogEvent  VEC_CUSPCopyFromGPUSome, VEC_CUSPCopyToGPUSome;

extern PetscErrorCode VecStashGetInfo_Private(VecStash*,PetscInt*,PetscInt*);
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
PetscErrorCode  VecStashGetInfo(Vec vec,PetscInt *nstash,PetscInt *reallocs,PetscInt *bnstash,PetscInt *breallocs)
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

   Logically Collective on Vec

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

#undef __FUNCT__
#define __FUNCT__ "VecSetLocalToGlobalMappingBlock"
/*@
   VecSetLocalToGlobalMappingBlock - Sets a local numbering to global numbering used
   by the routine VecSetValuesBlockedLocal() to allow users to insert vector entries
   using a local (per-processor) numbering.

   Logically Collective on Vec

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
PetscErrorCode  VecSetLocalToGlobalMappingBlock(Vec x,ISLocalToGlobalMapping mapping)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_CLASSID,1);
  PetscValidHeaderSpecific(mapping,IS_LTOGM_CLASSID,2);

  ierr = PetscLayoutSetISLocalToGlobalMappingBlock(x->map,mapping);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecGetLocalToGlobalMapping"
/*@
   VecGetLocalToGlobalMapping - Gets the local-to-global numbering set by VecSetLocalToGlobalMapping()

   Not Collective

   Input Parameter:
.  X - the vector

   Output Parameter:
.  mapping - the mapping

   Level: advanced

   Concepts: vectors^local to global mapping
   Concepts: local to global mapping^for vectors

.seealso:  VecSetValuesLocal(), VecGetLocalToGlobalMappingBlock()
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

#undef __FUNCT__  
#define __FUNCT__ "VecGetLocalToGlobalMappingBlock"
/*@
   VecGetLocalToGlobalMappingBlock - Gets the local-to-global numbering set by VecSetLocalToGlobalMappingBlock()

   Not Collective

   Input Parameters:
.  X - the vector

   Output Parameters:
.  mapping - the mapping

   Level: advanced

   Concepts: vectors^local to global mapping blocked
   Concepts: local to global mapping^for vectors, blocked

.seealso:  VecSetValuesBlockedLocal(), VecGetLocalToGlobalMapping()
@*/
PetscErrorCode VecGetLocalToGlobalMappingBlock(Vec X,ISLocalToGlobalMapping *mapping)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(X,VEC_CLASSID,1);
  PetscValidType(X,1);
  PetscValidPointer(mapping,2);
  *mapping = X->map->bmapping;
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
PetscErrorCode  VecAssemblyBegin(Vec vec)
{
  PetscErrorCode ierr;
  PetscBool      flg = PETSC_FALSE;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(vec,VEC_CLASSID,1);
  PetscValidType(vec,1);

  ierr = PetscOptionsGetBool(((PetscObject)vec)->prefix,"-vec_view_stash",&flg,PETSC_NULL);CHKERRQ(ierr);
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
PetscErrorCode  VecView_Private(Vec vec)
{
  PetscErrorCode ierr;
  PetscBool      flg = PETSC_FALSE;

  PetscFunctionBegin;
  ierr = PetscObjectOptionsBegin((PetscObject)vec);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-vec_view_info","Information on vector size","VecView",flg,&flg,PETSC_NULL);CHKERRQ(ierr);
    if (flg) {
      PetscViewer viewer;

      ierr = PetscViewerASCIIGetStdout(((PetscObject)vec)->comm,&viewer);CHKERRQ(ierr);
      ierr = PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_INFO);CHKERRQ(ierr);
      ierr = VecView(vec,viewer);CHKERRQ(ierr);
      ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
    }
    flg  = PETSC_FALSE;
    ierr = PetscOptionsBool("-vec_view","Print vector to stdout","VecView",flg,&flg,PETSC_NULL);CHKERRQ(ierr);
    if (flg) {
      PetscViewer viewer;
      ierr = PetscViewerASCIIGetStdout(((PetscObject)vec)->comm,&viewer);CHKERRQ(ierr);
      ierr = VecView(vec,viewer);CHKERRQ(ierr);
    }
    flg  = PETSC_FALSE;
    ierr = PetscOptionsBool("-vec_view_matlab","Print vector to stdout in a format MATLAB can read","VecView",flg,&flg,PETSC_NULL);CHKERRQ(ierr);
    if (flg) {
      PetscViewer viewer;
      ierr = PetscViewerASCIIGetStdout(((PetscObject)vec)->comm,&viewer);CHKERRQ(ierr);
      ierr = PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_MATLAB);CHKERRQ(ierr);
      ierr = VecView(vec,viewer);CHKERRQ(ierr);
      ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
    }
#if defined(PETSC_HAVE_MATLAB_ENGINE)
    flg  = PETSC_FALSE;
    ierr = PetscOptionsBool("-vec_view_matlab_file","Print vector to matlaboutput.mat format MATLAB can read","VecView",flg,&flg,PETSC_NULL);CHKERRQ(ierr);
    if (flg) {
      ierr = VecView(vec,PETSC_VIEWER_MATLAB_(((PetscObject)vec)->comm));CHKERRQ(ierr);
    }
#endif
#if defined(PETSC_USE_SOCKET_VIEWER)
    flg  = PETSC_FALSE;
    ierr = PetscOptionsBool("-vec_view_socket","Send vector to socket (can be read from matlab)","VecView",flg,&flg,PETSC_NULL);CHKERRQ(ierr);
    if (flg) {
      ierr = VecView(vec,PETSC_VIEWER_SOCKET_(((PetscObject)vec)->comm));CHKERRQ(ierr);
      ierr = PetscViewerFlush(PETSC_VIEWER_SOCKET_(((PetscObject)vec)->comm));CHKERRQ(ierr);
    }
#endif
    flg  = PETSC_FALSE;
    ierr = PetscOptionsBool("-vec_view_binary","Save vector to file in binary format","VecView",flg,&flg,PETSC_NULL);CHKERRQ(ierr);
    if (flg) {
      ierr = VecView(vec,PETSC_VIEWER_BINARY_(((PetscObject)vec)->comm));CHKERRQ(ierr);
      ierr = PetscViewerFlush(PETSC_VIEWER_BINARY_(((PetscObject)vec)->comm));CHKERRQ(ierr);
    }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  /* These invoke PetscDrawGetDraw which invokes PetscOptionsBegin/End, */
  /* hence they should not be inside the above PetscOptionsBegin/End block. */
  flg  = PETSC_FALSE;
  ierr = PetscOptionsGetBool(((PetscObject)vec)->prefix,"-vec_view_draw",&flg,PETSC_NULL);CHKERRQ(ierr);
  if (flg) {
    ierr = VecView(vec,PETSC_VIEWER_DRAW_(((PetscObject)vec)->comm));CHKERRQ(ierr);
    ierr = PetscViewerFlush(PETSC_VIEWER_DRAW_(((PetscObject)vec)->comm));CHKERRQ(ierr);
  }
  flg  = PETSC_FALSE;
  ierr = PetscOptionsGetBool(((PetscObject)vec)->prefix,"-vec_view_draw_lg",&flg,PETSC_NULL);CHKERRQ(ierr);
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
.  -vec_view_matlab - Prints vector in ASCII MATLAB format to stdout
.  -vec_view_matlab_file - Prints vector in MATLAB format to matlaboutput.mat
.  -vec_view_draw - Activates vector viewing using drawing tools
.  -display <name> - Sets display name (default is host)
.  -draw_pause <sec> - Sets number of seconds to pause after display
-  -vec_view_socket - Activates vector viewing using a socket

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
  ierr = VecView_Private(vec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecPointwiseMax"
/*@
   VecPointwiseMax - Computes the componentwise maximum w_i = max(x_i, y_i).

   Logically Collective on Vec

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
  if (x->map->N != y->map->N || x->map->N != w->map->N) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Incompatible vector global lengths");
  if (x->map->n != y->map->n || x->map->n != w->map->n) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Incompatible vector local lengths");

  ierr = (*w->ops->pointwisemax)(w,x,y);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)w);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "VecPointwiseMin"
/*@
   VecPointwiseMin - Computes the componentwise minimum w_i = min(x_i, y_i).

   Logically Collective on Vec

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
  if (x->map->N != y->map->N || x->map->N != w->map->N) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Incompatible vector global lengths");
  if (x->map->n != y->map->n || x->map->n != w->map->n) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Incompatible vector local lengths");

  ierr = (*w->ops->pointwisemin)(w,x,y);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)w);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecPointwiseMaxAbs"
/*@
   VecPointwiseMaxAbs - Computes the componentwise maximum of the absolute values w_i = max(abs(x_i), abs(y_i)).

   Logically Collective on Vec

   Input Parameters:
.  x, y  - the vectors

   Output Parameter:
.  w - the result

   Level: advanced

   Notes: any subset of the x, y, and w may be the same vector.

   Concepts: vector^pointwise multiply

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
  if (x->map->N != y->map->N || x->map->N != w->map->N) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Incompatible vector global lengths");
  if (x->map->n != y->map->n || x->map->n != w->map->n) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Incompatible vector local lengths");

  ierr = (*w->ops->pointwisemaxabs)(w,x,y);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)w);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecPointwiseDivide"
/*@
   VecPointwiseDivide - Computes the componentwise division w = x/y.

   Logically Collective on Vec

   Input Parameters:
.  x, y  - the vectors

   Output Parameter:
.  w - the result

   Level: advanced

   Notes: any subset of the x, y, and w may be the same vector.

   Concepts: vector^pointwise divide

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
  if (x->map->N != y->map->N || x->map->N != w->map->N) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Incompatible vector global lengths");
  if (x->map->n != y->map->n || x->map->n != w->map->n) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Incompatible vector local lengths");

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
PetscErrorCode  VecDestroy(Vec *v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*v) PetscFunctionReturn(0);
  PetscValidHeaderSpecific((*v),VEC_CLASSID,1);
  if (--((PetscObject)(*v))->refct > 0) {*v = 0; PetscFunctionReturn(0);}
  /* destroy the internal part */
  if ((*v)->ops->destroy) {
    ierr = (*(*v)->ops->destroy)(*v);CHKERRQ(ierr);
  }
  /* destroy the external/common part */
  ierr = PetscLayoutDestroy(&(*v)->map);CHKERRQ(ierr);
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
PetscErrorCode  VecDuplicateVecs(Vec v,PetscInt m,Vec *V[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,1);
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
+  vv - pointer to pointer to array of vector pointers
-  m - the number of vectors previously obtained

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
  if (!*vv) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(**vv,VEC_CLASSID,1);
  PetscValidType(**vv,1);
  if (m < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Trying to destroy negative number of vectors %D",m);
  ierr = (*(**vv)->ops->destroyvecs)(m,*vv);CHKERRQ(ierr);
  *vv = 0;
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
PetscErrorCode  VecViewFromOptions(Vec vec, const char *title)
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
.    PETSC_VIEWER_ASCII_MATLAB - prints vector contents in MATLAB format
.    PETSC_VIEWER_ASCII_INDEX - prints vector contents, including indices of vector elements
-    PETSC_VIEWER_ASCII_COMMON - prints vector contents, using a
         format common among all vector types

   Notes for HDF5 Viewer: the name of the Vec (given with PetscObjectSetName() is the name that is used
   for the object in the HDF5 file. If you wish to store the same vector to the HDF5 viewer (with different values,
   obviously) several times, you must change its name each time before calling the VecView(). The name you use
   here should equal the name that you use in the Vec object that you use with VecLoad().

   See the manual page for VecLoad() on the exact format the binary viewer stores
   the values in the file.

   Level: beginner

   Concepts: vector^printing
   Concepts: vector^saving to disk

.seealso: PetscViewerASCIIOpen(), PetscViewerDrawOpen(), PetscDrawLGCreate(),
          PetscViewerSocketOpen(), PetscViewerBinaryOpen(), VecLoad(), PetscViewerCreate(),
          PetscRealView(), PetscScalarView(), PetscIntView()
@*/
PetscErrorCode  VecView(Vec vec,PetscViewer viewer)
{
  PetscErrorCode    ierr;
  PetscBool         iascii;
  PetscViewerFormat format;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(vec,VEC_CLASSID,1);
  PetscValidType(vec,1);
  if (!viewer) {
    ierr = PetscViewerASCIIGetStdout(((PetscObject)vec)->comm,&viewer);CHKERRQ(ierr);
  }
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(vec,1,viewer,2);
  if (vec->stash.n || vec->bstash.n) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call VecAssemblyBegin/End() before viewing this vector");

  ierr = PetscLogEventBegin(VEC_View,vec,viewer,0,0);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    PetscInt rows,bs;

    ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);  
    if (format == PETSC_VIEWER_ASCII_INFO || format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
      ierr = PetscObjectPrintClassNamePrefixType((PetscObject)vec,viewer,"Vector Object");CHKERRQ(ierr);
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
  ierr = (*vec->ops->view)(vec,viewer);CHKERRQ(ierr);
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
  TV_add_row("Typename", TV_ascii_string_type , ((PetscObject)v)->type_name);
  ierr = VecGetArrayRead((Vec)v,&values);CHKERRQ(ierr);
  ierr = PetscSNPrintf(type,32,"double[%d]",v->map->n);CHKERRQ(ierr);
  TV_add_row("values",type, values);
  ierr = VecRestoreArrayRead((Vec)v,&values);CHKERRQ(ierr);
  return TV_format_OK;
}
#endif

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
PetscErrorCode  VecGetOwnershipRanges(Vec x,const PetscInt *ranges[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_CLASSID,1);
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
          in ix in calls to VecSetValues() or VecGetValues(). These rows are simply
          ignored.

   Level: intermediate

@*/
PetscErrorCode  VecSetOption(Vec x,VecOption op,PetscBool  flag)
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

#undef __FUNCT__
#define __FUNCT__ "VecDuplicateVecs_Default"
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
  ierr = PetscMalloc(m*sizeof(Vec*),V);CHKERRQ(ierr);
  for (i=0; i<m; i++) {ierr = VecDuplicate(w,*V+i);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecDestroyVecs_Default"
PetscErrorCode VecDestroyVecs_Default(PetscInt m,Vec v[])
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  PetscValidPointer(v,1);
  if (m <= 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"m must be > 0: m = %D",m);
  for (i=0; i<m; i++) {ierr = VecDestroy(&v[i]);CHKERRQ(ierr);}
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

#undef __FUNCT__
#define __FUNCT__ "VecLoad"
/*@C
  VecLoad - Loads a vector that has been stored in binary or HDF5 format
  with VecView().

  Collective on PetscViewer 

  Input Parameters:
+ newvec - the newly loaded vector, this needs to have been created with VecCreate() or
           some related function before a call to VecLoad(). 
- viewer - binary file viewer, obtained from PetscViewerBinaryOpen() or
           HDF5 file viewer, obtained from PetscViewerHDF5Open()

   Level: intermediate

  Notes:
  Defaults to the standard Seq or MPI Vec, if you want some other type of Vec call VecSetFromOptions()
  before calling this.

  The input file must contain the full global vector, as
  written by the routine VecView().

  If the type or size of newvec is not set before a call to VecLoad, PETSc 
  sets the type and the local and global sizes.If type and/or 
  sizes are already set, then the same are used.

  IF using HDF5, you must assign the Vec the same name as was used in the Vec
  that was stored in the file using PetscObjectSetName(). Otherwise you will
  get the error message: "Cannot H5DOpen2() with Vec name NAMEOFOBJECT"

  Notes for advanced users:
  Most users should not need to know the details of the binary storage
  format, since VecLoad() and VecView() completely hide these details.
  But for anyone who's interested, the standard binary matrix storage
  format is
.vb
     int    VEC_FILE_CLASSID
     int    number of rows
     PetscScalar *values of all entries
.ve

   In addition, PETSc automatically does the byte swapping for
machines that store the bytes reversed, e.g.  DEC alpha, freebsd,
linux, Windows and the paragon; thus if you write your own binary
read/write routines you have to swap the bytes; see PetscBinaryRead()
and PetscBinaryWrite() to see how this may be done.

  Concepts: vector^loading from file

.seealso: PetscViewerBinaryOpen(), VecView(), MatLoad(), VecLoad() 
@*/  
PetscErrorCode  VecLoad(Vec newvec, PetscViewer viewer)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(newvec,VEC_CLASSID,1);
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);

  ierr = PetscLogEventBegin(VEC_Load,viewer,0,0,0);CHKERRQ(ierr);
  if (!((PetscObject)newvec)->type_name && !newvec->ops->create) {
    ierr = VecSetType(newvec, VECSTANDARD);CHKERRQ(ierr);
  }
  ierr = (*newvec->ops->load)(newvec,viewer);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(VEC_Load,viewer,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "VecReciprocal"
/*@
   VecReciprocal - Replaces each component of a vector by its reciprocal.

   Logically Collective on Vec

   Input Parameter:
.  vec - the vector

   Output Parameter:
.  vec - the vector reciprocal

   Level: intermediate

   Concepts: vector^reciprocal

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
  ierr = (*vec->ops->reciprocal)(vec);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)vec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecSetOperation"
PetscErrorCode  VecSetOperation(Vec vec,VecOperation op, void (*f)(void))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(vec,VEC_CLASSID,1);
  (((void(**)(void))vec->ops)[(int)op]) = f;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "VecStashSetInitialSize"
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

   Concepts: vector^stash
   Concepts: stash^vector

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

#undef __FUNCT__
#define __FUNCT__ "VecConjugate"
/*@
   VecConjugate - Conjugates a vector.

   Logically Collective on Vec

   Input Parameters:
.  x - the vector

   Level: intermediate

   Concepts: vector^conjugate

@*/
PetscErrorCode  VecConjugate(Vec x)
{
#ifdef PETSC_USE_COMPLEX
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_CLASSID,1);
  PetscValidType(x,1);
  if (x->stash.insertmode != NOT_SET_VALUES) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled vector");
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

   Logically Collective on Vec

   Input Parameters:
.  x, y  - the vectors

   Output Parameter:
.  w - the result

   Level: advanced

   Notes: any subset of the x, y, and w may be the same vector.

   Concepts: vector^pointwise multiply

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
  if (x->map->n != y->map->n || x->map->n != w->map->n) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Incompatible vector local lengths");

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

   Logically Collective on Vec

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
PetscErrorCode  VecSetRandom(Vec x,PetscRandom rctx)
{
  PetscErrorCode ierr;
  PetscRandom    randObj = PETSC_NULL;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_CLASSID,1);
  if (rctx) PetscValidHeaderSpecific(rctx,PETSC_RANDOM_CLASSID,2);
  PetscValidType(x,1);
  if (x->stash.insertmode != NOT_SET_VALUES) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled vector");

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

  ierr = PetscRandomDestroy(&randObj);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecZeroEntries"
/*@
  VecZeroEntries - puts a 0.0 in each element of a vector

  Logically Collective on Vec

  Input Parameter:
. vec - The vector

  Level: beginner

  Developer Note: This routine does not need to exist since the exact functionality is obtained with 
     VecSet(vec,0);  I guess someone added it to mirror the functionality of MatZeroEntries() but Mat is nothing
     like a Vec (one is an operator and one is an element of a vector space, yeah yeah dual blah blah blah) so 
     this routine should not exist.

.keywords: Vec, set, options, database
.seealso: VecCreate(),  VecSetOptionsPrefix(), VecSet(), VecSetValues()
@*/
PetscErrorCode  VecZeroEntries(Vec vec)
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
  PetscBool      opt;
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
PetscErrorCode  VecSetFromOptions(Vec vec)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(vec,VEC_CLASSID,1);

  ierr = PetscObjectOptionsBegin((PetscObject)vec);CHKERRQ(ierr);
    /* Handle vector type options */
    ierr = VecSetTypeFromOptions_Private(vec);CHKERRQ(ierr);

    /* Handle specific vector options */
    if (vec->ops->setfromoptions) {
      ierr = (*vec->ops->setfromoptions)(vec);CHKERRQ(ierr);
    }

    /* process any options handlers added with PetscObjectAddOptionsHandler() */
    ierr = PetscObjectProcessOptionsHandlers((PetscObject)vec);CHKERRQ(ierr);
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
PetscErrorCode  VecSetSizes(Vec v, PetscInt n, PetscInt N)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v, VEC_CLASSID,1);
  if (N > 0 && n > N) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Local size %D cannot be larger than global size %D",n,N);
  if ((v->map->n >= 0 || v->map->N >= 0) && (v->map->n != n || v->map->N != N)) SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot change/reset vector sizes to %D local %D global after previously setting them to %D local %D global",n,N,v->map->n,v->map->N);
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

   Logically Collective on Vec

   Input Parameter:
+  v - the vector
-  bs - the blocksize

   Notes:
   All vectors obtained by VecDuplicate() inherit the same blocksize.

   Level: advanced

.seealso: VecSetValuesBlocked(), VecSetLocalToGlobalMappingBlock(), VecGetBlockSize()

  Concepts: block size^vectors
@*/
PetscErrorCode  VecSetBlockSize(Vec v,PetscInt bs)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,1);
  if (bs == v->map->bs) PetscFunctionReturn(0);
  ierr = PetscLayoutSetBlockSize(v->map,bs);CHKERRQ(ierr);
  v->bstash.bs = bs; /* use the same blocksize for the vec's block-stash */
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecGetBlockSize"
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

.seealso: VecSetValuesBlocked(), VecSetLocalToGlobalMappingBlock(), VecSetBlockSize()

   Concepts: vector^block size
   Concepts: block^vector

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

#undef __FUNCT__
#define __FUNCT__ "VecSetOptionsPrefix"
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

.keywords: Vec, set, options, prefix, database

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

#undef __FUNCT__
#define __FUNCT__ "VecAppendOptionsPrefix"
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

.keywords: Vec, append, options, prefix, database

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
PetscErrorCode  VecGetOptionsPrefix(Vec v,const char *prefix[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,1);
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
PetscErrorCode  VecSetUp(Vec v)
{
  PetscMPIInt    size;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,1);
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

   Logically Collective on Vec

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
  if (x->stash.insertmode != NOT_SET_VALUES) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled vector");
  if (x->map->n != y->map->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Incompatible vector local lengths %d != %d", x->map->n, y->map->n);

#if !defined(PETSC_USE_MIXED_PRECISION)
  for (i=0; i<4; i++) {
    ierr = PetscObjectComposedDataGetReal((PetscObject)x,NormIds[i],norms[i],flgs[i]);CHKERRQ(ierr);
  }
#endif

  ierr = PetscLogEventBegin(VEC_Copy,x,y,0,0);CHKERRQ(ierr);
#if defined(PETSC_USE_MIXED_PRECISION)
  extern PetscErrorCode VecGetArray(Vec,double **);
  extern PetscErrorCode VecRestoreArray(Vec,double **);
  extern PetscErrorCode VecGetArray(Vec,float **);
  extern PetscErrorCode VecRestoreArray(Vec,float **);
  extern PetscErrorCode VecGetArrayRead(Vec,const double **);
  extern PetscErrorCode VecRestoreArrayRead(Vec,const double **);
  extern PetscErrorCode VecGetArrayRead(Vec,const float **);
  extern PetscErrorCode VecRestoreArrayRead(Vec,const float **);
  if ((((PetscObject)x)->precision == PETSC_PRECISION_SINGLE) && (((PetscObject)y)->precision == PETSC_PRECISION_DOUBLE)) {
    PetscInt    i,n;
    const float *xx;
    double      *yy;
    ierr = VecGetArrayRead(x,&xx);CHKERRQ(ierr);
    ierr = VecGetArray(y,&yy);CHKERRQ(ierr);
    ierr = VecGetLocalSize(x,&n);CHKERRQ(ierr);
    for (i=0; i<n; i++) {
      yy[i] = xx[i];
    }
    ierr = VecRestoreArrayRead(x,&xx);CHKERRQ(ierr);
    ierr = VecRestoreArray(y,&yy);CHKERRQ(ierr);
  } else if ((((PetscObject)x)->precision == PETSC_PRECISION_DOUBLE) && (((PetscObject)y)->precision == PETSC_PRECISION_SINGLE)) {
    PetscInt     i,n;
    float        *yy;
    const double *xx;
    ierr = VecGetArrayRead(x,&xx);CHKERRQ(ierr);
    ierr = VecGetArray(y,&yy);CHKERRQ(ierr);
    ierr = VecGetLocalSize(x,&n);CHKERRQ(ierr);
    for (i=0; i<n; i++) {
      yy[i] = (float) xx[i];
    }
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

#undef __FUNCT__
#define __FUNCT__ "VecSwap"
/*@
   VecSwap - Swaps the vectors x and y.

   Logically Collective on Vec

   Input Parameters:
.  x, y  - the vectors

   Level: advanced

   Concepts: vector^swapping values

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
  if (x->stash.insertmode != NOT_SET_VALUES) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled vector");
  if (y->stash.insertmode != NOT_SET_VALUES) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled vector");
  if (x->map->N != y->map->N) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Incompatible vector global lengths");
  if (x->map->n != y->map->n) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Incompatible vector local lengths");

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
  ierr = MPI_Comm_rank(((PetscObject)v)->comm,&rank);CHKERRQ(ierr);
  s = &v->bstash;

  /* print block stash */
  ierr = PetscViewerASCIISynchronizedAllow(viewer,PETSC_TRUE);CHKERRQ(ierr);      
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
  ierr = PetscViewerASCIISynchronizedAllow(viewer,PETSC_FALSE);CHKERRQ(ierr);      

  ierr = PetscViewerASCIIUseTabs(viewer,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

