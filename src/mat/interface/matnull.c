
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
  CHKERRQ(VecGetBlockSize(coords,&dim));
  CHKERRQ(VecGetLocalSize(coords,&n));
  CHKERRQ(VecGetSize(coords,&N));
  n   /= dim;
  N   /= dim;
  sN = 1./PetscSqrtReal((PetscReal)N);
  switch (dim) {
  case 1:
    CHKERRQ(MatNullSpaceCreate(PetscObjectComm((PetscObject)coords),PETSC_TRUE,0,NULL,sp));
    break;
  case 2:
  case 3:
    nmodes = (dim == 2) ? 3 : 6;
    CHKERRQ(VecCreate(PetscObjectComm((PetscObject)coords),&vec[0]));
    CHKERRQ(VecSetSizes(vec[0],dim*n,dim*N));
    CHKERRQ(VecSetBlockSize(vec[0],dim));
    CHKERRQ(VecSetUp(vec[0]));
    for (i=1; i<nmodes; i++) CHKERRQ(VecDuplicate(vec[0],&vec[i]));
    for (i=0; i<nmodes; i++) CHKERRQ(VecGetArray(vec[i],&v[i]));
    CHKERRQ(VecGetArrayRead(coords,&x));
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
    for (i=0; i<nmodes; i++) CHKERRQ(VecRestoreArray(vec[i],&v[i]));
    CHKERRQ(VecRestoreArrayRead(coords,&x));
    for (i=dim; i<nmodes; i++) {
      /* Orthonormalize vec[i] against vec[0:i-1] */
      CHKERRQ(VecMDot(vec[i],i,vec,dots));
      for (j=0; j<i; j++) dots[j] *= -1.;
      CHKERRQ(VecMAXPY(vec[i],i,dots,vec));
      CHKERRQ(VecNormalize(vec[i],NULL));
    }
    CHKERRQ(MatNullSpaceCreate(PetscObjectComm((PetscObject)coords),PETSC_FALSE,nmodes,vec,sp));
    for (i=0; i<nmodes; i++) CHKERRQ(VecDestroy(&vec[i]));
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
    CHKERRQ(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)sp),&viewer));
  }
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(sp,1,viewer,2);

  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  if (iascii) {
    PetscViewerFormat format;
    PetscInt          i;
    CHKERRQ(PetscViewerGetFormat(viewer,&format));
    CHKERRQ(PetscObjectPrintClassNamePrefixType((PetscObject)sp,viewer));
    CHKERRQ(PetscViewerASCIIPushTab(viewer));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"Contains %" PetscInt_FMT " vector%s%s\n",sp->n,sp->n==1 ? "" : "s",sp->has_cnst ? " and the constant" : ""));
    if (sp->remove) CHKERRQ(PetscViewerASCIIPrintf(viewer,"Has user-provided removal function\n"));
    if (!(format == PETSC_VIEWER_ASCII_INFO || format == PETSC_VIEWER_ASCII_INFO_DETAIL)) {
      for (i=0; i<sp->n; i++) {
        CHKERRQ(VecView(sp->vecs[i],viewer));
      }
    }
    CHKERRQ(PetscViewerASCIIPopTab(viewer));
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
  PetscCheckFalse(n < 0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Number of vectors (given %" PetscInt_FMT ") cannot be negative",n);
  if (n) PetscValidPointer(vecs,4);
  for (i=0; i<n; i++) PetscValidHeaderSpecific(vecs[i],VEC_CLASSID,4);
  PetscValidPointer(SP,5);
  if (n) {
    for (i=0; i<n; i++) {
      /* prevent the user from changes values in the vector */
      CHKERRQ(VecLockReadPush(vecs[i]));
    }
  }
  if (PetscUnlikelyDebug(n)) {
    PetscScalar *dots;
    for (i=0; i<n; i++) {
      PetscReal norm;
      CHKERRQ(VecNorm(vecs[i],NORM_2,&norm));
      PetscCheckFalse(PetscAbsReal(norm - 1) > PETSC_SQRT_MACHINE_EPSILON,PetscObjectComm((PetscObject)vecs[i]),PETSC_ERR_ARG_WRONG,"Vector %" PetscInt_FMT " must have 2-norm of 1.0, it is %g",i,(double)norm);
    }
    if (has_cnst) {
      for (i=0; i<n; i++) {
        PetscScalar sum;
        CHKERRQ(VecSum(vecs[i],&sum));
        PetscCheckFalse(PetscAbsScalar(sum) > PETSC_SQRT_MACHINE_EPSILON,PetscObjectComm((PetscObject)vecs[i]),PETSC_ERR_ARG_WRONG,"Vector %" PetscInt_FMT " must be orthogonal to constant vector, inner product is %g",i,(double)PetscAbsScalar(sum));
      }
    }
    CHKERRQ(PetscMalloc1(n-1,&dots));
    for (i=0; i<n-1; i++) {
      PetscInt j;
      CHKERRQ(VecMDot(vecs[i],n-i-1,vecs+i+1,dots));
      for (j=0;j<n-i-1;j++) {
        PetscCheckFalse(PetscAbsScalar(dots[j]) > PETSC_SQRT_MACHINE_EPSILON,PetscObjectComm((PetscObject)vecs[i]),PETSC_ERR_ARG_WRONG,"Vector %" PetscInt_FMT " must be orthogonal to vector %" PetscInt_FMT ", inner product is %g",i,i+j+1,(double)PetscAbsScalar(dots[j]));
      }
    }
    CHKERRQ(PetscFree(dots));
  }

  *SP = NULL;
  CHKERRQ(MatInitializePackage());

  CHKERRQ(PetscHeaderCreate(sp,MAT_NULLSPACE_CLASSID,"MatNullSpace","Null space","Mat",comm,MatNullSpaceDestroy,MatNullSpaceView));

  sp->has_cnst = has_cnst;
  sp->n        = n;
  sp->vecs     = NULL;
  sp->alpha    = NULL;
  sp->remove   = NULL;
  sp->rmctx    = NULL;

  if (n) {
    CHKERRQ(PetscMalloc1(n,&sp->vecs));
    CHKERRQ(PetscMalloc1(n,&sp->alpha));
    CHKERRQ(PetscLogObjectMemory((PetscObject)sp,n*(sizeof(Vec)+sizeof(PetscScalar))));
    for (i=0; i<n; i++) {
      CHKERRQ(PetscObjectReference((PetscObject)vecs[i]));
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
    CHKERRQ(VecLockReadPop((*sp)->vecs[i]));
  }

  CHKERRQ(VecDestroyVecs((*sp)->n,&(*sp)->vecs));
  CHKERRQ(PetscFree((*sp)->alpha));
  CHKERRQ(PetscHeaderDestroy(sp));
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
    CHKERRQ(VecGetSize(vec,&N));
    if (N > 0) {
      CHKERRQ(VecSum(vec,&sum));
      sum  = sum/((PetscScalar)(-1.0*N));
      CHKERRQ(VecShift(vec,sum));
    }
  }

  if (sp->n) {
    CHKERRQ(VecMDot(vec,sp->n,sp->vecs,sp->alpha));
    for (i=0; i<sp->n; i++) sp->alpha[i] = -sp->alpha[i];
    CHKERRQ(VecMAXPY(vec,sp->n,sp->alpha,sp->vecs));
  }

  if (sp->remove) {
    CHKERRQ((*sp->remove)(sp,vec,sp->rmctx));
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
  CHKERRQ(PetscOptionsGetBool(((PetscObject)sp)->options,((PetscObject)mat)->prefix,"-mat_null_space_test_view",&flg1,NULL));
  CHKERRQ(PetscOptionsGetBool(((PetscObject)sp)->options,((PetscObject)mat)->prefix,"-mat_null_space_test_view_draw",&flg2,NULL));

  if (n) {
    CHKERRQ(VecDuplicate(sp->vecs[0],&l));
  } else {
    CHKERRQ(MatCreateVecs(mat,&l,NULL));
  }

  CHKERRQ(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)sp),&viewer));
  if (sp->has_cnst) {
    CHKERRQ(VecDuplicate(l,&r));
    CHKERRQ(VecGetSize(l,&N));
    sum  = 1.0/PetscSqrtReal(N);
    CHKERRQ(VecSet(l,sum));
    CHKERRQ(MatMult(mat,l,r));
    CHKERRQ(VecNorm(r,NORM_2,&nrm));
    if (nrm >= tol) consistent = PETSC_FALSE;
    if (flg1) {
      if (consistent) {
        CHKERRQ(PetscPrintf(PetscObjectComm((PetscObject)sp),"Constants are likely null vector"));
      } else {
        CHKERRQ(PetscPrintf(PetscObjectComm((PetscObject)sp),"Constants are unlikely null vector "));
      }
      CHKERRQ(PetscPrintf(PetscObjectComm((PetscObject)sp),"|| A * 1/N || = %g\n",(double)nrm));
    }
    if (!consistent && flg1) CHKERRQ(VecView(r,viewer));
    if (!consistent && flg2) CHKERRQ(VecView(r,viewer));
    CHKERRQ(VecDestroy(&r));
  }

  for (j=0; j<n; j++) {
    CHKERRQ((*mat->ops->mult)(mat,sp->vecs[j],l));
    CHKERRQ(VecNorm(l,NORM_2,&nrm));
    if (nrm >= tol) consistent = PETSC_FALSE;
    if (flg1) {
      if (consistent) {
        CHKERRQ(PetscPrintf(PetscObjectComm((PetscObject)sp),"Null vector %" PetscInt_FMT " is likely null vector",j));
      } else {
        CHKERRQ(PetscPrintf(PetscObjectComm((PetscObject)sp),"Null vector %" PetscInt_FMT " unlikely null vector ",j));
        consistent = PETSC_FALSE;
      }
      CHKERRQ(PetscPrintf(PetscObjectComm((PetscObject)sp),"|| A * v[%" PetscInt_FMT "] || = %g\n",j,(double)nrm));
    }
    if (!consistent && flg1) CHKERRQ(VecView(l,viewer));
    if (!consistent && flg2) CHKERRQ(VecView(l,viewer));
  }

  PetscCheck(!sp->remove,PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Cannot test a null space provided as a function with MatNullSpaceSetFunction()");
  CHKERRQ(VecDestroy(&l));
  if (isNull) *isNull = consistent;
  PetscFunctionReturn(0);
}
