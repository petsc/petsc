
/*
    Routines to project vectors out of null spaces.
*/

#include <petsc/private/matimpl.h>      /*I "petscmat.h" I*/

PetscClassId MAT_NULLSPACE_CLASSID;

/*@C
   MatNullSpaceSetFunction - set a function that removes a null space from a vector
   out of null spaces.

   Logically Collective on MatNullSpace

   Input Parameters:
+  sp - the null space object
.  rem - the function that removes the null space
-  ctx - context for the remove function

   Level: advanced

.seealso: MatNullSpaceDestroy(), MatNullSpaceRemove(), MatSetNullSpace(), MatNullSpace, MatNullSpaceCreate()
@*/
PetscErrorCode  MatNullSpaceSetFunction(MatNullSpace sp, PetscErrorCode (*rem)(MatNullSpace,Vec,void*),void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp,MAT_NULLSPACE_CLASSID,1);
  sp->remove = rem;
  sp->rmctx  = ctx;
  PetscFunctionReturn(0);
}

/*@C
   MatNullSpaceGetVecs - get vectors defining the null space

   Not Collective

   Input Parameter:
.  sp - null space object

   Output Parameters:
+  has_cnst - PETSC_TRUE if the null space contains the constant vector, otherwise PETSC_FALSE
.  n - number of vectors (excluding constant vector) in null space
-  vecs - orthonormal vectors that span the null space (excluding the constant vector)

   Level: developer

   Notes:
      These vectors and the array are owned by the MatNullSpace and should not be destroyed or freeded by the caller

.seealso: MatNullSpaceCreate(), MatGetNullSpace(), MatGetNearNullSpace()
@*/
PetscErrorCode MatNullSpaceGetVecs(MatNullSpace sp,PetscBool *has_const,PetscInt *n,const Vec **vecs)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp,MAT_NULLSPACE_CLASSID,1);
  if (has_const) *has_const = sp->has_cnst;
  if (n) *n = sp->n;
  if (vecs) *vecs = sp->vecs;
  PetscFunctionReturn(0);
}

/*@
   MatNullSpaceCreateRigidBody - create rigid body modes from coordinates

   Collective on Vec

   Input Parameter:
.  coords - block of coordinates of each node, must have block size set

   Output Parameter:
.  sp - the null space

   Level: advanced

   Notes:
     If you are solving an elasticity problem you should likely use this, in conjunction with MatSetNearNullspace(), to provide information that
     the PCGAMG preconditioner can use to construct a much more efficient preconditioner.

     If you are solving an elasticity problem with pure Neumann boundary conditions you can use this in conjunction with MatSetNullspace() to
     provide this information to the linear solver so it can handle the null space appropriately in the linear solution.

.seealso: MatNullSpaceCreate(), MatSetNearNullspace(), MatSetNullspace()
@*/
PetscErrorCode MatNullSpaceCreateRigidBody(Vec coords,MatNullSpace *sp)
{
  const PetscScalar *x;
  PetscScalar       *v[6],dots[5];
  Vec               vec[6];
  PetscInt          n,N,dim,nmodes,i,j;
  PetscReal         sN;

  PetscFunctionBegin;
  PetscCall(VecGetBlockSize(coords,&dim));
  PetscCall(VecGetLocalSize(coords,&n));
  PetscCall(VecGetSize(coords,&N));
  n   /= dim;
  N   /= dim;
  sN = 1./PetscSqrtReal((PetscReal)N);
  switch (dim) {
  case 1:
    PetscCall(MatNullSpaceCreate(PetscObjectComm((PetscObject)coords),PETSC_TRUE,0,NULL,sp));
    break;
  case 2:
  case 3:
    nmodes = (dim == 2) ? 3 : 6;
    PetscCall(VecCreate(PetscObjectComm((PetscObject)coords),&vec[0]));
    PetscCall(VecSetSizes(vec[0],dim*n,dim*N));
    PetscCall(VecSetBlockSize(vec[0],dim));
    PetscCall(VecSetUp(vec[0]));
    for (i=1; i<nmodes; i++) PetscCall(VecDuplicate(vec[0],&vec[i]));
    for (i=0; i<nmodes; i++) PetscCall(VecGetArray(vec[i],&v[i]));
    PetscCall(VecGetArrayRead(coords,&x));
    for (i=0; i<n; i++) {
      if (dim == 2) {
        v[0][i*2+0] = sN;
        v[0][i*2+1] = 0.;
        v[1][i*2+0] = 0.;
        v[1][i*2+1] = sN;
        /* Rotations */
        v[2][i*2+0] = -x[i*2+1];
        v[2][i*2+1] = x[i*2+0];
      } else {
        v[0][i*3+0] = sN;
        v[0][i*3+1] = 0.;
        v[0][i*3+2] = 0.;
        v[1][i*3+0] = 0.;
        v[1][i*3+1] = sN;
        v[1][i*3+2] = 0.;
        v[2][i*3+0] = 0.;
        v[2][i*3+1] = 0.;
        v[2][i*3+2] = sN;

        v[3][i*3+0] = x[i*3+1];
        v[3][i*3+1] = -x[i*3+0];
        v[3][i*3+2] = 0.;
        v[4][i*3+0] = 0.;
        v[4][i*3+1] = -x[i*3+2];
        v[4][i*3+2] = x[i*3+1];
        v[5][i*3+0] = x[i*3+2];
        v[5][i*3+1] = 0.;
        v[5][i*3+2] = -x[i*3+0];
      }
    }
    for (i=0; i<nmodes; i++) PetscCall(VecRestoreArray(vec[i],&v[i]));
    PetscCall(VecRestoreArrayRead(coords,&x));
    for (i=dim; i<nmodes; i++) {
      /* Orthonormalize vec[i] against vec[0:i-1] */
      PetscCall(VecMDot(vec[i],i,vec,dots));
      for (j=0; j<i; j++) dots[j] *= -1.;
      PetscCall(VecMAXPY(vec[i],i,dots,vec));
      PetscCall(VecNormalize(vec[i],NULL));
    }
    PetscCall(MatNullSpaceCreate(PetscObjectComm((PetscObject)coords),PETSC_FALSE,nmodes,vec,sp));
    for (i=0; i<nmodes; i++) PetscCall(VecDestroy(&vec[i]));
  }
  PetscFunctionReturn(0);
}

/*@C
   MatNullSpaceView - Visualizes a null space object.

   Collective on MatNullSpace

   Input Parameters:
+  matnull - the null space
-  viewer - visualization context

   Level: advanced

   Fortran Note:
   This routine is not supported in Fortran.

.seealso: MatNullSpaceCreate(), PetscViewerASCIIOpen()
@*/
PetscErrorCode MatNullSpaceView(MatNullSpace sp,PetscViewer viewer)
{
  PetscBool      iascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp,MAT_NULLSPACE_CLASSID,1);
  if (!viewer) {
    PetscCall(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)sp),&viewer));
  }
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(sp,1,viewer,2);

  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  if (iascii) {
    PetscViewerFormat format;
    PetscInt          i;
    PetscCall(PetscViewerGetFormat(viewer,&format));
    PetscCall(PetscObjectPrintClassNamePrefixType((PetscObject)sp,viewer));
    PetscCall(PetscViewerASCIIPushTab(viewer));
    PetscCall(PetscViewerASCIIPrintf(viewer,"Contains %" PetscInt_FMT " vector%s%s\n",sp->n,sp->n==1 ? "" : "s",sp->has_cnst ? " and the constant" : ""));
    if (sp->remove) PetscCall(PetscViewerASCIIPrintf(viewer,"Has user-provided removal function\n"));
    if (!(format == PETSC_VIEWER_ASCII_INFO || format == PETSC_VIEWER_ASCII_INFO_DETAIL)) {
      for (i=0; i<sp->n; i++) {
        PetscCall(VecView(sp->vecs[i],viewer));
      }
    }
    PetscCall(PetscViewerASCIIPopTab(viewer));
  }
  PetscFunctionReturn(0);
}

/*@C
   MatNullSpaceCreate - Creates a data structure used to project vectors
   out of null spaces.

   Collective

   Input Parameters:
+  comm - the MPI communicator associated with the object
.  has_cnst - PETSC_TRUE if the null space contains the constant vector; otherwise PETSC_FALSE
.  n - number of vectors (excluding constant vector) in null space
-  vecs - the vectors that span the null space (excluding the constant vector);
          these vectors must be orthonormal. These vectors are NOT copied, so do not change them
          after this call. You should free the array that you pass in and destroy the vectors (this will reduce the reference count
          for them by one).

   Output Parameter:
.  SP - the null space context

   Level: advanced

   Notes:
    See MatNullSpaceSetFunction() as an alternative way of providing the null space information instead of setting vecs.

    If has_cnst is PETSC_TRUE you do not need to pass a constant vector in as a fourth argument to this routine, nor do you
    need to pass in a function that eliminates the constant function into MatNullSpaceSetFunction().

.seealso: MatNullSpaceDestroy(), MatNullSpaceRemove(), MatSetNullSpace(), MatNullSpace, MatNullSpaceSetFunction()
@*/
PetscErrorCode  MatNullSpaceCreate(MPI_Comm comm,PetscBool has_cnst,PetscInt n,const Vec vecs[],MatNullSpace *SP)
{
  MatNullSpace   sp;
  PetscInt       i;

  PetscFunctionBegin;
  PetscCheck(n >= 0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Number of vectors (given %" PetscInt_FMT ") cannot be negative",n);
  if (n) PetscValidPointer(vecs,4);
  for (i=0; i<n; i++) PetscValidHeaderSpecific(vecs[i],VEC_CLASSID,4);
  PetscValidPointer(SP,5);
  if (n) {
    for (i=0; i<n; i++) {
      /* prevent the user from changes values in the vector */
      PetscCall(VecLockReadPush(vecs[i]));
    }
  }
  if (PetscUnlikelyDebug(n)) {
    PetscScalar *dots;
    for (i=0; i<n; i++) {
      PetscReal norm;
      PetscCall(VecNorm(vecs[i],NORM_2,&norm));
      PetscCheck(PetscAbsReal(norm - 1) <= PETSC_SQRT_MACHINE_EPSILON,PetscObjectComm((PetscObject)vecs[i]),PETSC_ERR_ARG_WRONG,"Vector %" PetscInt_FMT " must have 2-norm of 1.0, it is %g",i,(double)norm);
    }
    if (has_cnst) {
      for (i=0; i<n; i++) {
        PetscScalar sum;
        PetscCall(VecSum(vecs[i],&sum));
        PetscCheck(PetscAbsScalar(sum) <= PETSC_SQRT_MACHINE_EPSILON,PetscObjectComm((PetscObject)vecs[i]),PETSC_ERR_ARG_WRONG,"Vector %" PetscInt_FMT " must be orthogonal to constant vector, inner product is %g",i,(double)PetscAbsScalar(sum));
      }
    }
    PetscCall(PetscMalloc1(n-1,&dots));
    for (i=0; i<n-1; i++) {
      PetscInt j;
      PetscCall(VecMDot(vecs[i],n-i-1,vecs+i+1,dots));
      for (j=0;j<n-i-1;j++) {
        PetscCheck(PetscAbsScalar(dots[j]) <= PETSC_SQRT_MACHINE_EPSILON,PetscObjectComm((PetscObject)vecs[i]),PETSC_ERR_ARG_WRONG,"Vector %" PetscInt_FMT " must be orthogonal to vector %" PetscInt_FMT ", inner product is %g",i,i+j+1,(double)PetscAbsScalar(dots[j]));
      }
    }
    PetscCall(PetscFree(dots));
  }

  *SP = NULL;
  PetscCall(MatInitializePackage());

  PetscCall(PetscHeaderCreate(sp,MAT_NULLSPACE_CLASSID,"MatNullSpace","Null space","Mat",comm,MatNullSpaceDestroy,MatNullSpaceView));

  sp->has_cnst = has_cnst;
  sp->n        = n;
  sp->vecs     = NULL;
  sp->alpha    = NULL;
  sp->remove   = NULL;
  sp->rmctx    = NULL;

  if (n) {
    PetscCall(PetscMalloc1(n,&sp->vecs));
    PetscCall(PetscMalloc1(n,&sp->alpha));
    PetscCall(PetscLogObjectMemory((PetscObject)sp,n*(sizeof(Vec)+sizeof(PetscScalar))));
    for (i=0; i<n; i++) {
      PetscCall(PetscObjectReference((PetscObject)vecs[i]));
      sp->vecs[i] = vecs[i];
    }
  }

  *SP = sp;
  PetscFunctionReturn(0);
}

/*@
   MatNullSpaceDestroy - Destroys a data structure used to project vectors
   out of null spaces.

   Collective on MatNullSpace

   Input Parameter:
.  sp - the null space context to be destroyed

   Level: advanced

.seealso: MatNullSpaceCreate(), MatNullSpaceRemove(), MatNullSpaceSetFunction()
@*/
PetscErrorCode  MatNullSpaceDestroy(MatNullSpace *sp)
{
  PetscInt       i;

  PetscFunctionBegin;
  if (!*sp) PetscFunctionReturn(0);
  PetscValidHeaderSpecific((*sp),MAT_NULLSPACE_CLASSID,1);
  if (--((PetscObject)(*sp))->refct > 0) {*sp = NULL; PetscFunctionReturn(0);}

  for (i=0; i < (*sp)->n; i++) {
    PetscCall(VecLockReadPop((*sp)->vecs[i]));
  }

  PetscCall(VecDestroyVecs((*sp)->n,&(*sp)->vecs));
  PetscCall(PetscFree((*sp)->alpha));
  PetscCall(PetscHeaderDestroy(sp));
  PetscFunctionReturn(0);
}

/*@C
   MatNullSpaceRemove - Removes all the components of a null space from a vector.

   Collective on MatNullSpace

   Input Parameters:
+  sp - the null space context (if this is NULL then no null space is removed)
-  vec - the vector from which the null space is to be removed

   Level: advanced

.seealso: MatNullSpaceCreate(), MatNullSpaceDestroy(), MatNullSpaceSetFunction()
@*/
PetscErrorCode  MatNullSpaceRemove(MatNullSpace sp,Vec vec)
{
  PetscScalar    sum;
  PetscInt       i,N;

  PetscFunctionBegin;
  if (!sp) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(sp,MAT_NULLSPACE_CLASSID,1);
  PetscValidHeaderSpecific(vec,VEC_CLASSID,2);

  if (sp->has_cnst) {
    PetscCall(VecGetSize(vec,&N));
    if (N > 0) {
      PetscCall(VecSum(vec,&sum));
      sum  = sum/((PetscScalar)(-1.0*N));
      PetscCall(VecShift(vec,sum));
    }
  }

  if (sp->n) {
    PetscCall(VecMDot(vec,sp->n,sp->vecs,sp->alpha));
    for (i=0; i<sp->n; i++) sp->alpha[i] = -sp->alpha[i];
    PetscCall(VecMAXPY(vec,sp->n,sp->alpha,sp->vecs));
  }

  if (sp->remove) {
    PetscCall((*sp->remove)(sp,vec,sp->rmctx));
  }
  PetscFunctionReturn(0);
}

/*@
   MatNullSpaceTest  - Tests if the claimed null space is really a
     null space of a matrix

   Collective on MatNullSpace

   Input Parameters:
+  sp - the null space context
-  mat - the matrix

   Output Parameters:
.  isNull - PETSC_TRUE if the nullspace is valid for this matrix

   Level: advanced

.seealso: MatNullSpaceCreate(), MatNullSpaceDestroy(), MatNullSpaceSetFunction()
@*/
PetscErrorCode  MatNullSpaceTest(MatNullSpace sp,Mat mat,PetscBool  *isNull)
{
  PetscScalar    sum;
  PetscReal      nrm,tol = 10. * PETSC_SQRT_MACHINE_EPSILON;
  PetscInt       j,n,N;
  Vec            l,r;
  PetscBool      flg1 = PETSC_FALSE,flg2 = PETSC_FALSE,consistent = PETSC_TRUE;
  PetscViewer    viewer;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp,MAT_NULLSPACE_CLASSID,1);
  PetscValidHeaderSpecific(mat,MAT_CLASSID,2);
  n    = sp->n;
  PetscCall(PetscOptionsGetBool(((PetscObject)sp)->options,((PetscObject)mat)->prefix,"-mat_null_space_test_view",&flg1,NULL));
  PetscCall(PetscOptionsGetBool(((PetscObject)sp)->options,((PetscObject)mat)->prefix,"-mat_null_space_test_view_draw",&flg2,NULL));

  if (n) {
    PetscCall(VecDuplicate(sp->vecs[0],&l));
  } else {
    PetscCall(MatCreateVecs(mat,&l,NULL));
  }

  PetscCall(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)sp),&viewer));
  if (sp->has_cnst) {
    PetscCall(VecDuplicate(l,&r));
    PetscCall(VecGetSize(l,&N));
    sum  = 1.0/PetscSqrtReal(N);
    PetscCall(VecSet(l,sum));
    PetscCall(MatMult(mat,l,r));
    PetscCall(VecNorm(r,NORM_2,&nrm));
    if (nrm >= tol) consistent = PETSC_FALSE;
    if (flg1) {
      if (consistent) {
        PetscCall(PetscPrintf(PetscObjectComm((PetscObject)sp),"Constants are likely null vector"));
      } else {
        PetscCall(PetscPrintf(PetscObjectComm((PetscObject)sp),"Constants are unlikely null vector "));
      }
      PetscCall(PetscPrintf(PetscObjectComm((PetscObject)sp),"|| A * 1/N || = %g\n",(double)nrm));
    }
    if (!consistent && flg1) PetscCall(VecView(r,viewer));
    if (!consistent && flg2) PetscCall(VecView(r,viewer));
    PetscCall(VecDestroy(&r));
  }

  for (j=0; j<n; j++) {
    PetscCall((*mat->ops->mult)(mat,sp->vecs[j],l));
    PetscCall(VecNorm(l,NORM_2,&nrm));
    if (nrm >= tol) consistent = PETSC_FALSE;
    if (flg1) {
      if (consistent) {
        PetscCall(PetscPrintf(PetscObjectComm((PetscObject)sp),"Null vector %" PetscInt_FMT " is likely null vector",j));
      } else {
        PetscCall(PetscPrintf(PetscObjectComm((PetscObject)sp),"Null vector %" PetscInt_FMT " unlikely null vector ",j));
        consistent = PETSC_FALSE;
      }
      PetscCall(PetscPrintf(PetscObjectComm((PetscObject)sp),"|| A * v[%" PetscInt_FMT "] || = %g\n",j,(double)nrm));
    }
    if (!consistent && flg1) PetscCall(VecView(l,viewer));
    if (!consistent && flg2) PetscCall(VecView(l,viewer));
  }

  PetscCheck(!sp->remove,PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Cannot test a null space provided as a function with MatNullSpaceSetFunction()");
  PetscCall(VecDestroy(&l));
  if (isNull) *isNull = consistent;
  PetscFunctionReturn(0);
}
