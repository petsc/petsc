#define PETSCMAT_DLL

/*
    Routines to project vectors out of null spaces.
*/

#include "src/mat/matimpl.h"      /*I "petscmat.h" I*/
#include "petscsys.h"

PetscCookie PETSCMAT_DLLEXPORT MAT_NULLSPACE_COOKIE = 0;

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
          after this call. You should free the array that you pass in.

   Output Parameter:
.  SP - the null space context

   Level: advanced

  Users manual sections:
.   sec_singular

.keywords: PC, null space, create

.seealso: MatNullSpaceDestroy(), MatNullSpaceRemove(), KSPSetNullSpace(), MatNullSpace
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatNullSpaceCreate(MPI_Comm comm,PetscTruth has_cnst,PetscInt n,const Vec vecs[],MatNullSpace *SP)
{
  MatNullSpace   sp;
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  if (n) PetscValidPointer(vecs,4); 
  for (i=0; i<n; i++) PetscValidHeaderSpecific(vecs[i],VEC_COOKIE,4); 
  PetscValidPointer(SP,5); 
 
  *SP = PETSC_NULL; 
#ifndef PETSC_USE_DYNAMIC_LIBRARIES 
  ierr = MatInitializePackage(PETSC_NULL);CHKERRQ(ierr); 
#endif 

  ierr = PetscHeaderCreate(sp,_p_MatNullSpace,int,MAT_NULLSPACE_COOKIE,0,"MatNullSpace",comm,MatNullSpaceDestroy,0);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory(sp,sizeof(struct _p_MatNullSpace));CHKERRQ(ierr);

  sp->has_cnst = has_cnst; 
  sp->n        = n;
  sp->vec      = PETSC_NULL;
  if (n) {
    ierr = PetscMalloc(n*sizeof(Vec),&sp->vecs);CHKERRQ(ierr);
    for (i=0; i<n; i++) sp->vecs[i] = vecs[i];
  } else {
    sp->vecs = 0;
  }

  for (i=0; i<n; i++) {
    ierr = PetscObjectReference((PetscObject)sp->vecs[i]);CHKERRQ(ierr);
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

.seealso: MatNullSpaceCreate(), MatNullSpaceRemove()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatNullSpaceDestroy(MatNullSpace sp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (--sp->refct > 0) PetscFunctionReturn(0);

  if (sp->vec) {ierr = VecDestroy(sp->vec);CHKERRQ(ierr);}
  if (sp->vecs) {
    ierr = VecDestroyVecs(sp->vecs,sp->n);CHKERRQ(ierr);
  }
  ierr = PetscHeaderDestroy(sp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatNullSpaceRemove"
/*@
   MatNullSpaceRemove - Removes all the components of a null space from a vector.

   Collective on MatNullSpace

   Input Parameters:
+  sp - the null space context
.  vec - the vector from which the null space is to be removed 
-  out - if this is requested (not PETSC_NULL) then this is a vector with the null space removed otherwise
         the removal is done in-place (in vec)

   Note: The user is responsible for the vector returned and should destroy it.

   Level: advanced

.keywords: PC, null space, remove

.seealso: MatNullSpaceCreate(), MatNullSpaceDestroy()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatNullSpaceRemove(MatNullSpace sp,Vec vec,Vec *out)
{
  PetscScalar    sum;
  PetscErrorCode ierr;
  PetscInt       j,n = sp->n,N;
  Vec            l = vec;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp,MAT_NULLSPACE_COOKIE,1); 
  PetscValidHeaderSpecific(vec,VEC_COOKIE,2); 

  if (out) {
    PetscValidPointer(out,3); 
    if (!sp->vec) {
      ierr = VecDuplicate(vec,&sp->vec);CHKERRQ(ierr);
    }
    *out = sp->vec;
    ierr = VecCopy(vec,*out);CHKERRQ(ierr);
    l    = *out;
    ierr = PetscObjectReference((PetscObject) l);CHKERRQ(ierr);
  }

  if (sp->has_cnst) {
    ierr = VecSum(l,&sum);CHKERRQ(ierr);
    ierr = VecGetSize(l,&N);CHKERRQ(ierr);
    sum  = sum/(-1.0*N);
    ierr = VecShift(l,sum);CHKERRQ(ierr);
  }

  for (j=0; j<n; j++) {
    ierr = VecDot(l,sp->vecs[j],&sum);CHKERRQ(ierr);
    ierr = VecAXPY(l,-sum,sp->vecs[j]);CHKERRQ(ierr);
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

   Level: advanced

.keywords: PC, null space, remove

.seealso: MatNullSpaceCreate(), MatNullSpaceDestroy()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatNullSpaceTest(MatNullSpace sp,Mat mat)
{
  PetscScalar    sum;
  PetscReal      nrm;
  PetscInt       j,n = sp->n,N,m;
  PetscErrorCode ierr;
  Vec            l,r;
  MPI_Comm       comm = sp->comm;
  PetscTruth     flg1,flg2;

  PetscFunctionBegin;
  ierr = PetscOptionsHasName(PETSC_NULL,"-mat_null_space_test_view",&flg1);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(PETSC_NULL,"-mat_null_space_test_view_draw",&flg2);CHKERRQ(ierr);

  if (!sp->vec) {
    if (n) {
      ierr = VecDuplicate(sp->vecs[0],&sp->vec);CHKERRQ(ierr);
    } else {
      ierr = MatGetLocalSize(mat,&m,PETSC_NULL);CHKERRQ(ierr);
      ierr = VecCreateMPI(sp->comm,m,PETSC_DETERMINE,&sp->vec);CHKERRQ(ierr);
    }
  }
  l    = sp->vec;

  if (sp->has_cnst) {
    ierr = VecDuplicate(l,&r);CHKERRQ(ierr);
    ierr = VecGetSize(l,&N);CHKERRQ(ierr);
    sum  = 1.0/N;
    ierr = VecSet(l,sum);CHKERRQ(ierr);
    ierr = MatMult(mat,l,r);CHKERRQ(ierr);
    ierr = VecNorm(r,NORM_2,&nrm);CHKERRQ(ierr);
    if (nrm < 1.e-7) {ierr = PetscPrintf(comm,"Constants are likely null vector");CHKERRQ(ierr);}
    else {ierr = PetscPrintf(comm,"Constants are unlikely null vector ");CHKERRQ(ierr);}
    ierr = PetscPrintf(comm,"|| A * 1 || = %g\n",nrm);CHKERRQ(ierr);
    if (nrm > 1.e-7 && flg1) {ierr = VecView(r,PETSC_VIEWER_STDOUT_(comm));CHKERRQ(ierr);}
    if (nrm > 1.e-7 && flg2) {ierr = VecView(r,PETSC_VIEWER_DRAW_(comm));CHKERRQ(ierr);}
    ierr = VecDestroy(r);CHKERRQ(ierr);
  }

  for (j=0; j<n; j++) {
    ierr = (*mat->ops->mult)(mat,sp->vecs[j],l);CHKERRQ(ierr);
    ierr = VecNorm(l,NORM_2,&nrm);CHKERRQ(ierr);
    if (nrm < 1.e-7) {ierr = PetscPrintf(comm,"Null vector %D is likely null vector",j);CHKERRQ(ierr);}
    else {ierr = PetscPrintf(comm,"Null vector %D unlikely null vector ",j);CHKERRQ(ierr);}
    ierr = PetscPrintf(comm,"|| A * v[%D] || = %g\n",j,nrm);CHKERRQ(ierr);
    if (nrm > 1.e-7 && flg1) {ierr = VecView(l,PETSC_VIEWER_STDOUT_(comm));CHKERRQ(ierr);}
    if (nrm > 1.e-7 && flg2) {ierr = VecView(l,PETSC_VIEWER_DRAW_(comm));CHKERRQ(ierr);}
  }
  
  PetscFunctionReturn(0);
}

