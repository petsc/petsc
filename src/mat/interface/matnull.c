
/*
    Routines to project vectors out of null spaces.
*/

#include <petsc-private/matimpl.h>      /*I "petscmat.h" I*/

PetscClassId  MAT_NULLSPACE_CLASSID;

#undef __FUNCT__  
#define __FUNCT__ "MatNullSpaceSetFunction"
/*@C
   MatNullSpaceSetFunction - set a function that removes a null space from a vector
   out of null spaces.

   Logically Collective on MatNullSpace

   Input Parameters:
+  sp - the null space object
.  rem - the function that removes the null space
-  ctx - context for the remove function

   Level: advanced

.keywords: PC, null space, create

.seealso: MatNullSpaceDestroy(), MatNullSpaceRemove(), KSPSetNullSpace(), MatNullSpace, MatNullSpaceCreate()
@*/
PetscErrorCode  MatNullSpaceSetFunction(MatNullSpace sp, PetscErrorCode (*rem)(MatNullSpace,Vec,void*),void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp,MAT_NULLSPACE_CLASSID,1);
  sp->remove = rem;
  sp->rmctx  = ctx;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatNullSpaceGetVecs"
/*@C
   MatNullSpaceGetVecs - get vectors defining the null space

   Not Collective

   Input Arguments:
.  sp - null space object

   Output Arguments:
+  has_cnst - PETSC_TRUE if the null space contains the constant vector, otherwise PETSC_FALSE
.  n - number of vectors (excluding constant vector) in null space
-  vecs - orthonormal vectors that span the null space (excluding the constant vector)

   Level: developer

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

#undef __FUNCT__
#define __FUNCT__ "MatNullSpaceCreateRigidBody"
/*@
   MatNullSpaceCreateRigidBody - create rigid body modes from coordinates

   Collective on Vec

   Input Argument:
.  coords - block of coordinates of each node, must have block size set

   Output Argument:
.  sp - the null space

   Level: advanced

.seealso: MatNullSpaceCreate()
@*/
PetscErrorCode MatNullSpaceCreateRigidBody(Vec coords,MatNullSpace *sp)
{
  PetscErrorCode ierr;
  const PetscScalar *x;
  PetscScalar *v[6],dots[3];
  Vec vec[6];
  PetscInt n,N,dim,nmodes,i,j;

  PetscFunctionBegin;
  ierr = VecGetBlockSize(coords,&dim);CHKERRQ(ierr);
  ierr = VecGetLocalSize(coords,&n);CHKERRQ(ierr);
  ierr = VecGetSize(coords,&N);CHKERRQ(ierr);
  n /= dim;
  N /= dim;
  switch (dim) {
  case 1:
    ierr = MatNullSpaceCreate(((PetscObject)coords)->comm,PETSC_TRUE,0,PETSC_NULL,sp);CHKERRQ(ierr);
    break;
  case 2:
  case 3:
    nmodes = (dim == 2) ? 3 : 6;
    ierr = VecCreate(((PetscObject)coords)->comm,&vec[0]);CHKERRQ(ierr);
    ierr = VecSetSizes(vec[0],dim*n,dim*N);CHKERRQ(ierr);
    ierr = VecSetBlockSize(vec[0],dim);CHKERRQ(ierr);
    ierr = VecSetUp(vec[0]);CHKERRQ(ierr);
    for (i=1; i<nmodes; i++) {ierr = VecDuplicate(vec[0],&vec[i]);CHKERRQ(ierr);}
    for (i=0; i<nmodes; i++) {ierr = VecGetArray(vec[i],&v[i]);CHKERRQ(ierr);}
    ierr = VecGetArrayRead(coords,&x);CHKERRQ(ierr);
    for (i=0; i<n; i++) {
      if (dim == 2) {
        v[0][i*2+0] = 1./N;
        v[0][i*2+1] = 0.;
        v[1][i*2+0] = 0.;
        v[1][i*2+1] = 1./N;
        /* Rotations */
        v[2][i*2+0] = -x[i*2+1];
        v[2][i*2+1] = x[i*2+0];
      } else {
        v[0][i*3+0] = 1./N;
        v[0][i*3+1] = 0.;
        v[0][i*3+2] = 0.;
        v[1][i*3+0] = 0.;
        v[1][i*3+1] = 1./N;
        v[1][i*3+2] = 0.;
        v[2][i*3+0] = 0.;
        v[2][i*3+1] = 0.;
        v[2][i*3+2] = 1./N;

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
    for (i=0; i<nmodes; i++) {ierr = VecRestoreArray(vec[i],&v[i]);CHKERRQ(ierr);}
    ierr = VecRestoreArrayRead(coords,&x);CHKERRQ(ierr);
    for (i=dim; i<nmodes; i++) {
      /* Orthonormalize vec[i] against vec[0:dim] */
      ierr = VecMDot(vec[i],i,vec,dots);CHKERRQ(ierr);
      for (j=0; j<i; j++) dots[j] *= -1.;
      ierr = VecMAXPY(vec[i],i,dots,vec);CHKERRQ(ierr);
      ierr = VecNormalize(vec[i],PETSC_NULL);CHKERRQ(ierr);
    }
    ierr = MatNullSpaceCreate(((PetscObject)coords)->comm,PETSC_FALSE,nmodes,vec,sp);CHKERRQ(ierr);
    for (i=0; i<nmodes; i++) {ierr = VecDestroy(&vec[i]);CHKERRQ(ierr);}
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatNullSpaceView"
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
  PetscErrorCode ierr;
  PetscBool      iascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp,MAT_NULLSPACE_CLASSID,1);
  if (!viewer) viewer = PETSC_VIEWER_STDOUT_(((PetscObject)sp)->comm);
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(sp,1,viewer,2);

  ierr = PetscTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    PetscViewerFormat format;
    PetscInt i;
    ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
    ierr = PetscObjectPrintClassNamePrefixType((PetscObject)sp,viewer,"MatNullSpace Object");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"Contains %D vector%s%s\n",sp->n,sp->n==1?"":"s",sp->has_cnst?" and the constant":"");CHKERRQ(ierr);
    if (sp->remove) {ierr = PetscViewerASCIIPrintf(viewer,"Has user-provided removal function\n");CHKERRQ(ierr);}
    if (!(format == PETSC_VIEWER_ASCII_INFO || format == PETSC_VIEWER_ASCII_INFO_DETAIL)) {
      for (i=0; i<sp->n; i++) {
        ierr = VecView(sp->vecs[i],viewer);CHKERRQ(ierr);
      }
    }
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatNullSpaceCreate"
/*@
   MatNullSpaceCreate - Creates a data structure used to project vectors 
   out of null spaces.

   Collective on MPI_Comm

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

   Notes: See MatNullSpaceSetFunction() as an alternative way of providing the null space information instead of setting vecs.

      If has_cnst is PETSC_TRUE you do not need to pass a constant vector in as a fourth argument to this routine, nor do you 
       need to pass in a function that eliminates the constant function into MatNullSpaceSetFunction().

  Users manual sections:
.   sec_singular

.keywords: PC, null space, create

.seealso: MatNullSpaceDestroy(), MatNullSpaceRemove(), KSPSetNullSpace(), MatNullSpace, MatNullSpaceSetFunction()
@*/
PetscErrorCode  MatNullSpaceCreate(MPI_Comm comm,PetscBool  has_cnst,PetscInt n,const Vec vecs[],MatNullSpace *SP)
{
  MatNullSpace   sp;
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  if (n < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Number of vectors (given %D) cannot be negative",n);
  if (n) PetscValidPointer(vecs,4);
  for (i=0; i<n; i++) PetscValidHeaderSpecific(vecs[i],VEC_CLASSID,4);
  PetscValidPointer(SP,5); 
 
  *SP = PETSC_NULL; 
#ifndef PETSC_USE_DYNAMIC_LIBRARIES 
  ierr = MatInitializePackage(PETSC_NULL);CHKERRQ(ierr); 
#endif 

  ierr = PetscHeaderCreate(sp,_p_MatNullSpace,int,MAT_NULLSPACE_CLASSID,0,"MatNullSpace","Null space","Mat",comm,MatNullSpaceDestroy,MatNullSpaceView);CHKERRQ(ierr);

  sp->has_cnst = has_cnst;
  sp->n        = n;
  sp->vecs     = 0;
  sp->alpha    = 0;
  sp->vec      = 0;
  sp->remove   = 0;
  sp->rmctx    = 0;

  if (n) {
    ierr = PetscMalloc(n*sizeof(Vec),&sp->vecs);CHKERRQ(ierr);
    ierr = PetscMalloc(n*sizeof(PetscScalar),&sp->alpha);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory(sp,n*(sizeof(Vec)+sizeof(PetscScalar)));CHKERRQ(ierr);
    for (i=0; i<n; i++) {
      ierr = PetscObjectReference((PetscObject)vecs[i]);CHKERRQ(ierr);
      sp->vecs[i] = vecs[i];
    }
  }

  *SP          = sp;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatNullSpaceDestroy"
/*@
   MatNullSpaceDestroy - Destroys a data structure used to project vectors 
   out of null spaces.

   Collective on MatNullSpace

   Input Parameter:
.  sp - the null space context to be destroyed

   Level: advanced

.keywords: PC, null space, destroy

.seealso: MatNullSpaceCreate(), MatNullSpaceRemove(), MatNullSpaceSetFunction()
@*/
PetscErrorCode  MatNullSpaceDestroy(MatNullSpace *sp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*sp) PetscFunctionReturn(0);
  PetscValidHeaderSpecific((*sp),MAT_NULLSPACE_CLASSID,1); 
  if (--((PetscObject)(*sp))->refct > 0) {*sp = 0; PetscFunctionReturn(0);}

  ierr = VecDestroy(&(*sp)->vec);CHKERRQ(ierr); 
  ierr = VecDestroyVecs((*sp)->n,&(*sp)->vecs);CHKERRQ(ierr); 
  ierr = PetscFree((*sp)->alpha);CHKERRQ(ierr);
  ierr = PetscHeaderDestroy(sp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatNullSpaceRemove"
/*@C
   MatNullSpaceRemove - Removes all the components of a null space from a vector.

   Collective on MatNullSpace

   Input Parameters:
+  sp - the null space context
.  vec - the vector from which the null space is to be removed 
-  out - if this is requested (not PETSC_NULL) then this is a vector with the null space removed otherwise
         the removal is done in-place (in vec)

   Note: The user is not responsible for the vector returned and should not destroy it.

   Level: advanced

.keywords: PC, null space, remove

.seealso: MatNullSpaceCreate(), MatNullSpaceDestroy(), MatNullSpaceSetFunction()
@*/
PetscErrorCode  MatNullSpaceRemove(MatNullSpace sp,Vec vec,Vec *out)
{
  PetscScalar    sum;
  PetscInt       i,N;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp,MAT_NULLSPACE_CLASSID,1); 
  PetscValidHeaderSpecific(vec,VEC_CLASSID,2); 

  if (out) {
    PetscValidPointer(out,3); 
    if (!sp->vec) { 
      ierr = VecDuplicate(vec,&sp->vec);CHKERRQ(ierr); 
      ierr = PetscLogObjectParent(sp,sp->vec);CHKERRQ(ierr);
    }
    ierr = VecCopy(vec,sp->vec);CHKERRQ(ierr);
    vec = *out = sp->vec;
  }
  
  if (sp->has_cnst) {
    ierr = VecGetSize(vec,&N);CHKERRQ(ierr);
    if (N > 0) {
      ierr = VecSum(vec,&sum);CHKERRQ(ierr);
      sum  = sum/((PetscScalar)(-1.0*N));
      ierr = VecShift(vec,sum);CHKERRQ(ierr);
    }
  }
  
  if (sp->n) {
    ierr = VecMDot(vec,sp->n,sp->vecs,sp->alpha);CHKERRQ(ierr);
    for (i=0; i<sp->n; i++) sp->alpha[i] = -sp->alpha[i];
    ierr = VecMAXPY(vec,sp->n,sp->alpha,sp->vecs);CHKERRQ(ierr);
  }

  if (sp->remove){
    ierr = (*sp->remove)(sp,vec,sp->rmctx);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatNullSpaceTest"
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

.keywords: PC, null space, remove

.seealso: MatNullSpaceCreate(), MatNullSpaceDestroy(), MatNullSpaceSetFunction()
@*/
PetscErrorCode  MatNullSpaceTest(MatNullSpace sp,Mat mat,PetscBool  *isNull)
{
  PetscScalar    sum;
  PetscReal      nrm;
  PetscInt       j,n,N;
  PetscErrorCode ierr;
  Vec            l,r;
  PetscBool      flg1 = PETSC_FALSE,flg2 = PETSC_FALSE,consistent = PETSC_TRUE;
  PetscViewer    viewer;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp,MAT_NULLSPACE_CLASSID,1);
  PetscValidHeaderSpecific(mat,MAT_CLASSID,2);
  n = sp->n;
  ierr = PetscOptionsGetBool(PETSC_NULL,"-mat_null_space_test_view",&flg1,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(PETSC_NULL,"-mat_null_space_test_view_draw",&flg2,PETSC_NULL);CHKERRQ(ierr);

  if (!sp->vec) {
    if (n) {
      ierr = VecDuplicate(sp->vecs[0],&sp->vec);CHKERRQ(ierr);
    } else {
      ierr = MatGetVecs(mat,&sp->vec,PETSC_NULL);CHKERRQ(ierr);
    }
  }
  l    = sp->vec;

  ierr = PetscViewerASCIIGetStdout(((PetscObject)sp)->comm,&viewer);CHKERRQ(ierr);
  if (sp->has_cnst) {
    ierr = VecDuplicate(l,&r);CHKERRQ(ierr);
    ierr = VecGetSize(l,&N);CHKERRQ(ierr);
    sum  = 1.0/N;
    ierr = VecSet(l,sum);CHKERRQ(ierr);
    ierr = MatMult(mat,l,r);CHKERRQ(ierr);
    ierr = VecNorm(r,NORM_2,&nrm);CHKERRQ(ierr);
    if (flg1) {
      if (nrm < 1.e-7) {
        ierr = PetscPrintf(((PetscObject)sp)->comm,"Constants are likely null vector");CHKERRQ(ierr);
      } else {
        ierr = PetscPrintf(((PetscObject)sp)->comm,"Constants are unlikely null vector ");CHKERRQ(ierr);
        consistent = PETSC_FALSE;
      }
      ierr = PetscPrintf(((PetscObject)sp)->comm,"|| A * 1/N || = %G\n",nrm);CHKERRQ(ierr);
    }
    if (nrm > 1.e-7 && flg1) {ierr = VecView(r,viewer);CHKERRQ(ierr);}
    if (nrm > 1.e-7 && flg2) {ierr = VecView(r,viewer);CHKERRQ(ierr);}
    ierr = VecDestroy(&r);CHKERRQ(ierr);
  }

  for (j=0; j<n; j++) {
    ierr = (*mat->ops->mult)(mat,sp->vecs[j],l);CHKERRQ(ierr);
    ierr = VecNorm(l,NORM_2,&nrm);CHKERRQ(ierr);
    if (flg1) {
      if (nrm < 1.e-7) {
        ierr = PetscPrintf(((PetscObject)sp)->comm,"Null vector %D is likely null vector",j);CHKERRQ(ierr);
      } else {
        ierr = PetscPrintf(((PetscObject)sp)->comm,"Null vector %D unlikely null vector ",j);CHKERRQ(ierr);
        consistent = PETSC_FALSE;
      }
      ierr = PetscPrintf(((PetscObject)sp)->comm,"|| A * v[%D] || = %G\n",j,nrm);CHKERRQ(ierr);
    }
    if (nrm > 1.e-7 && flg1) {ierr = VecView(l,viewer);CHKERRQ(ierr);}
    if (nrm > 1.e-7 && flg2) {ierr = VecView(l,viewer);CHKERRQ(ierr);}
  }

  if (sp->remove) SETERRQ(((PetscObject)mat)->comm,PETSC_ERR_SUP,"Cannot test a null space provided as a function with MatNullSpaceSetFunction()");
  if (isNull) *isNull = consistent;
  PetscFunctionReturn(0);
}

