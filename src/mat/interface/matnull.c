#define PETSCMAT_DLL

/*
    Routines to project vectors out of null spaces.
*/

#include "private/matimpl.h"      /*I "petscmat.h" I*/

PetscCookie PETSCMAT_DLLEXPORT MAT_NULLSPACE_COOKIE;

#undef __FUNCT__  
#define __FUNCT__ "MatNullSpaceSetFunction"
/*@C
   MatNullSpaceSetFunction - set a function that removes a null space from a vector
   out of null spaces.

   Collective on MatNullSpace

   Input Parameters:
+  sp - the null space object
.  rem - the function that removes the null space
-  ctx - context for the remove function

   Level: advanced

.keywords: PC, null space, create

.seealso: MatNullSpaceDestroy(), MatNullSpaceRemove(), KSPSetNullSpace(), MatNullSpace, MatNullSpaceCreate()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatNullSpaceSetFunction(MatNullSpace sp, PetscErrorCode (*rem)(MatNullSpace,Vec,void*),void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp,MAT_NULLSPACE_COOKIE,1);
  sp->remove = rem;
  sp->rmctx  = ctx;
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
PetscErrorCode PETSCMAT_DLLEXPORT MatNullSpaceCreate(MPI_Comm comm,PetscTruth has_cnst,PetscInt n,const Vec vecs[],MatNullSpace *SP)
{
  MatNullSpace   sp;
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  if (n < 0) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"Number of vectors (given %D) cannot be negative",n);
  if (n) PetscValidPointer(vecs,4);
  for (i=0; i<n; i++) PetscValidHeaderSpecific(vecs[i],VEC_COOKIE,4);
  PetscValidPointer(SP,5); 
 
  *SP = PETSC_NULL; 
#ifndef PETSC_USE_DYNAMIC_LIBRARIES 
  ierr = MatInitializePackage(PETSC_NULL);CHKERRQ(ierr); 
#endif 

  ierr = PetscHeaderCreate(sp,_p_MatNullSpace,int,MAT_NULLSPACE_COOKIE,0,"MatNullSpace",comm,MatNullSpaceDestroy,0);CHKERRQ(ierr);

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
PetscErrorCode PETSCMAT_DLLEXPORT MatNullSpaceDestroy(MatNullSpace sp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp,MAT_NULLSPACE_COOKIE,1); 
  if (--((PetscObject)sp)->refct > 0) PetscFunctionReturn(0);

  if (sp->vec)  { ierr = VecDestroy(sp->vec);CHKERRQ(ierr); }
  if (sp->vecs) { ierr = VecDestroyVecs(sp->vecs,sp->n);CHKERRQ(ierr); }
  ierr = PetscFree(sp->alpha);CHKERRQ(ierr);
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
PetscErrorCode PETSCMAT_DLLEXPORT MatNullSpaceRemove(MatNullSpace sp,Vec vec,Vec *out)
{
  PetscScalar    sum;
  PetscInt       i,N;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp,MAT_NULLSPACE_COOKIE,1); 
  PetscValidHeaderSpecific(vec,VEC_COOKIE,2); 

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
      sum  = sum/(-1.0*N);
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
PetscErrorCode PETSCMAT_DLLEXPORT MatNullSpaceTest(MatNullSpace sp,Mat mat,PetscTruth *isNull)
{
  PetscScalar    sum;
  PetscReal      nrm;
  PetscInt       j,n,N;
  PetscErrorCode ierr;
  Vec            l,r;
  PetscTruth     flg1 = PETSC_FALSE,flg2 = PETSC_FALSE,consistent = PETSC_TRUE;
  PetscViewer    viewer;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp,MAT_NULLSPACE_COOKIE,1);
  PetscValidHeaderSpecific(mat,MAT_COOKIE,2);
  n = sp->n;
  ierr = PetscOptionsGetTruth(PETSC_NULL,"-mat_null_space_test_view",&flg1,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetTruth(PETSC_NULL,"-mat_null_space_test_view_draw",&flg2,PETSC_NULL);CHKERRQ(ierr);

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
    if (nrm < 1.e-7) {
      ierr = PetscPrintf(((PetscObject)sp)->comm,"Constants are likely null vector");CHKERRQ(ierr);
    } else {
      ierr = PetscPrintf(((PetscObject)sp)->comm,"Constants are unlikely null vector ");CHKERRQ(ierr);
      consistent = PETSC_FALSE;
    }
    ierr = PetscPrintf(((PetscObject)sp)->comm,"|| A * 1 || = %G\n",nrm);CHKERRQ(ierr);
    if (nrm > 1.e-7 && flg1) {ierr = VecView(r,viewer);CHKERRQ(ierr);}
    if (nrm > 1.e-7 && flg2) {ierr = VecView(r,viewer);CHKERRQ(ierr);}
    ierr = VecDestroy(r);CHKERRQ(ierr);
  }

  for (j=0; j<n; j++) {
    ierr = (*mat->ops->mult)(mat,sp->vecs[j],l);CHKERRQ(ierr);
    ierr = VecNorm(l,NORM_2,&nrm);CHKERRQ(ierr);
    if (nrm < 1.e-7) {
      ierr = PetscPrintf(((PetscObject)sp)->comm,"Null vector %D is likely null vector",j);CHKERRQ(ierr);
    } else {
      ierr = PetscPrintf(((PetscObject)sp)->comm,"Null vector %D unlikely null vector ",j);CHKERRQ(ierr);
      consistent = PETSC_FALSE;
    }
    ierr = PetscPrintf(((PetscObject)sp)->comm,"|| A * v[%D] || = %G\n",j,nrm);CHKERRQ(ierr);
    if (nrm > 1.e-7 && flg1) {ierr = VecView(l,viewer);CHKERRQ(ierr);}
    if (nrm > 1.e-7 && flg2) {ierr = VecView(l,viewer);CHKERRQ(ierr);}
  }

  if (sp->remove){
    SETERRQ(PETSC_ERR_SUP,"Cannot test a null space provided as a function with MatNullSpaceSetFunction()");
  }
  *isNull = consistent;
  PetscFunctionReturn(0);
}

