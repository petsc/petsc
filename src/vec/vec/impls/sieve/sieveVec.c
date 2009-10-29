#define PETSCVEC_DLL
/*
  Implementation of PETSc Vec using Sieve fields
*/
#include <Mesh.hh>
#include "private/vecimpl.h" /*I  "petscvec.h"  I*/
#include "private/petscaxpy.h"
#include "petscblaslapack.h"

#undef __FUNCT__  
#define __FUNCT__ "VecSet_Sieve"
PetscErrorCode VecSet_Sieve(Vec v, PetscScalar alpha)
{
  ALE::Mesh::field_type *field = (ALE::Mesh::field_type *) v->data;
  PetscInt               i,n     = v->map->n;
  PetscScalar           *xx    = (PetscScalar *) field->restrict(*field->getPatches()->begin());
  PetscErrorCode         ierr;

  PetscFunctionBegin;
  if (alpha == 0.0) {
    ierr = PetscMemzero(xx, n*sizeof(PetscScalar));CHKERRQ(ierr);
  } else {
    for (i=0; i<n; i++) xx[i] = alpha;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecScale_Sieve"
PetscErrorCode VecScale_Sieve(Vec v, PetscScalar alpha)
{
  ALE::Mesh::field_type *field = (ALE::Mesh::field_type *) v->data;
  PetscErrorCode         ierr;
  PetscBLASInt           one = 1,bn    = PetscBLASIntCast(v->map->n);

  PetscFunctionBegin;
  if (alpha == 0.0) {
    ierr = VecSet_Sieve(v, alpha);CHKERRQ(ierr);
  } else if (alpha != 1.0) {
    PetscScalar a = alpha;

    BLASscal_(&bn, &a, (PetscScalar *) field->restrict(*field->getPatches()->begin()), &one);
    ierr = PetscLogFlops(v->map->n);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecCopy_Sieve"
PetscErrorCode VecCopy_Sieve(Vec x, Vec y)
{
  ALE::Mesh::field_type *field = (ALE::Mesh::field_type *) x->data;
  PetscScalar           *yy;
  PetscErrorCode         ierr;

  PetscFunctionBegin;
  if (x != y) {
    ierr = VecGetArray(y, &yy);CHKERRQ(ierr);
    ierr = PetscMemcpy(yy, (PetscScalar *) field->restrict(*field->getPatches()->begin()), x->map->n*sizeof(PetscScalar));CHKERRQ(ierr);
    ierr = VecRestoreArray(y, &yy);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecAXPY_Sieve"
PetscErrorCode VecAXPY_Sieve(Vec y, PetscScalar alpha, Vec x)
{
  ALE::Mesh::field_type *field = (ALE::Mesh::field_type *) y->data;
  PetscScalar           *xarray;
  PetscErrorCode         ierr;
  PetscBLASInt           one = 1,bn = PetscBLASIntCast(y->map->n);

  PetscFunctionBegin;
  /* assume that the BLAS handles alpha == 1.0 efficiently since we have no fast code for it */
  if (alpha != 0.0) {
    PetscScalar oalpha = alpha;

    ierr = VecGetArray(x, &xarray);CHKERRQ(ierr);
    BLASaxpy_(&bn, &oalpha, xarray, &one, (PetscScalar *) field->restrict(*field->getPatches()->begin()), &one);
    ierr = VecRestoreArray(x, &xarray);CHKERRQ(ierr);
    ierr = PetscLogFlops(2.0*y->map->n);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
#include "../src/vec/vec/impls/seq/ftn-kernels/faypx.h"
#undef __FUNCT__  
#define __FUNCT__ "VecAYPX_Sieve"
PetscErrorCode VecAYPX_Sieve(Vec y, PetscScalar alpha, Vec x)
{
  ALE::Mesh::field_type *field = (ALE::Mesh::field_type *) y->data;
  PetscScalar           *yy    = (PetscScalar *) field->restrict(*field->getPatches()->begin());
  PetscInt               n     = y->map->n;
  PetscScalar           *xx;
  PetscErrorCode         ierr;

  PetscFunctionBegin;
  if (alpha == 0.0) {
    ierr = VecCopy_Sieve(x, y);CHKERRQ(ierr);
  } else if (alpha == 1.0) {
    ierr = VecAXPY_Sieve(y, alpha, x);CHKERRQ(ierr);
  } else {
    ierr = VecGetArray(x, &xx);CHKERRQ(ierr);
#if defined(PETSC_USE_FORTRAN_KERNEL_AYPX)
    {
      PetscScalar oalpha = alpha;
      fortranaypx_(&n, &oalpha, xx, yy);
    }
#else
    {
      PetscInt i;
      for (i=0; i<n; i++) {
        yy[i] = xx[i] + alpha*yy[i];
      }
    }
#endif
    ierr = VecRestoreArray(x, &xx);CHKERRQ(ierr);
    ierr = PetscLogFlops(2.0*n);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecAXPBY_Sieve"
PetscErrorCode VecAXPBY_Sieve(Vec y, PetscScalar alpha, PetscScalar beta, Vec x)
{
  ALE::Mesh::field_type *field = (ALE::Mesh::field_type *) y->data;
  PetscScalar           *yy = (PetscScalar *) field->restrict(*field->getPatches()->begin());
  PetscScalar           *xx ,a = alpha,b = beta;
  PetscInt               n = y->map->n, i;
  PetscErrorCode         ierr;

  PetscFunctionBegin;
  if (a == 0.0) {
    ierr = VecScale_Sieve(y, beta);CHKERRQ(ierr);
  } else if (b == 1.0) {
    ierr = VecAXPY_Sieve(y, alpha, x);CHKERRQ(ierr);
  } else if (a == 1.0) {
    ierr = VecAYPX_Sieve(y, beta, x);CHKERRQ(ierr);
  } else if (b == 0.0) {
    ierr = VecGetArray(x, &xx);CHKERRQ(ierr);
    for(i = 0; i < n; i++) {
      yy[i] = a*xx[i];
    }
    ierr = VecRestoreArray(x, &xx);CHKERRQ(ierr);
    ierr = PetscLogFlops(x->map->n);CHKERRQ(ierr);
  } else {
    ierr = VecGetArray(x, &xx);CHKERRQ(ierr);
    for(i = 0; i < n; i++) {
      yy[i] = a*xx[i] + b*yy[i];
    }
    ierr = VecRestoreArray(x, &xx);CHKERRQ(ierr);
    ierr = PetscLogFlops(3.0*x->map->n);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecMAXPY_Sieve"
PetscErrorCode VecMAXPY_Sieve(Vec x, PetscInt nv, const PetscScalar *alpha, Vec *y)
{
  ALE::Mesh::field_type *field = (ALE::Mesh::field_type *) x->data;
  PetscInt               n     = x->map->n,j,j_rem;
  PetscScalar           *xx    = (PetscScalar *) field->restrict(*field->getPatches()->begin());
  PetscScalar           *yy0,*yy1,*yy2,*yy3,alpha0,alpha1,alpha2,alpha3;
  PetscErrorCode         ierr;

#if defined(PETSC_HAVE_PRAGMA_DISJOINT)
#pragma disjoint(*xx,*yy0,*yy1,*yy2,*yy3,*alpha)
#endif

  PetscFunctionBegin;
  ierr = PetscLogFlops(nv*2.0*n);CHKERRQ(ierr);

  switch (j_rem=nv&0x3) {
  case 3: 
    ierr = VecGetArray(y[0],&yy0);CHKERRQ(ierr);
    ierr = VecGetArray(y[1],&yy1);CHKERRQ(ierr);
    ierr = VecGetArray(y[2],&yy2);CHKERRQ(ierr);
    alpha0 = alpha[0]; 
    alpha1 = alpha[1]; 
    alpha2 = alpha[2]; 
    alpha += 3;
    PetscAXPY3(xx,alpha0,alpha1,alpha2,yy0,yy1,yy2,n);
    ierr = VecRestoreArray(y[0],&yy0);CHKERRQ(ierr);
    ierr = VecRestoreArray(y[1],&yy1);CHKERRQ(ierr);
    ierr = VecRestoreArray(y[2],&yy2);CHKERRQ(ierr);
    y     += 3;
    break;
  case 2: 
    ierr = VecGetArray(y[0],&yy0);CHKERRQ(ierr);
    ierr = VecGetArray(y[1],&yy1);CHKERRQ(ierr);
    alpha0 = alpha[0]; 
    alpha1 = alpha[1]; 
    alpha +=2;
    PetscAXPY2(xx,alpha0,alpha1,yy0,yy1,n);
    ierr = VecRestoreArray(y[0],&yy0);CHKERRQ(ierr);
    ierr = VecRestoreArray(y[1],&yy1);CHKERRQ(ierr);
    y     +=2;
    break;
  case 1: 
    ierr = VecGetArray(y[0],&yy0);CHKERRQ(ierr);
    alpha0 = *alpha++; 
    PetscAXPY(xx,alpha0,yy0,n);
    ierr = VecRestoreArray(y[0],&yy0);CHKERRQ(ierr);
    y     +=1;
    break;
  }
  for (j=j_rem; j<nv; j+=4) {
    ierr = VecGetArray(y[0],&yy0);CHKERRQ(ierr);
    ierr = VecGetArray(y[1],&yy1);CHKERRQ(ierr);
    ierr = VecGetArray(y[2],&yy2);CHKERRQ(ierr);
    ierr = VecGetArray(y[3],&yy3);CHKERRQ(ierr);
    alpha0 = alpha[0];
    alpha1 = alpha[1];
    alpha2 = alpha[2];
    alpha3 = alpha[3];
    alpha  += 4;

    PetscAXPY4(xx,alpha0,alpha1,alpha2,alpha3,yy0,yy1,yy2,yy3,n);
    ierr = VecRestoreArray(y[0],&yy0);CHKERRQ(ierr);
    ierr = VecRestoreArray(y[1],&yy1);CHKERRQ(ierr);
    ierr = VecRestoreArray(y[2],&yy2);CHKERRQ(ierr);
    ierr = VecRestoreArray(y[3],&yy3);CHKERRQ(ierr);
    y      += 4;
  }
  PetscFunctionReturn(0);
} 

#include "../src/vec/vec/impls/seq/ftn-kernels/fwaxpy.h"
#undef __FUNCT__  
#define __FUNCT__ "VecWAXPY_Sieve"
PetscErrorCode VecWAXPY_Sieve(Vec w, PetscScalar alpha, Vec x, Vec y)
{
  ALE::Mesh::field_type *field = (ALE::Mesh::field_type *) w->data;
  PetscScalar           *ww    = (PetscScalar *) field->restrict(*field->getPatches()->begin());
  PetscInt               n     = w->map->n, i;
  PetscScalar           *yy, *xx;
  PetscErrorCode         ierr;

  PetscFunctionBegin;
  ierr = VecGetArray(y, &yy);CHKERRQ(ierr);
  ierr = VecGetArray(x, &xx);CHKERRQ(ierr);
  if (alpha == 1.0) {
    ierr = PetscLogFlops(n);CHKERRQ(ierr);
    /* could call BLAS axpy after call to memcopy, but may be slower */
    for (i=0; i<n; i++) ww[i] = yy[i] + xx[i];
  } else if (alpha == -1.0) {
    ierr = PetscLogFlops(n);CHKERRQ(ierr);
    for (i=0; i<n; i++) ww[i] = yy[i] - xx[i];
  } else if (alpha == 0.0) {
    ierr = PetscMemcpy(ww,yy,n*sizeof(PetscScalar));CHKERRQ(ierr);
  } else {
    PetscScalar oalpha = alpha;
#if defined(PETSC_USE_FORTRAN_KERNEL_WAXPY)
    fortranwaxpy_(&n,&oalpha,xx,yy,ww);
#else
    for (i=0; i<n; i++) ww[i] = yy[i] + oalpha * xx[i];
#endif
    ierr = PetscLogFlops(2.0*n);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(y, &yy);CHKERRQ(ierr);
  ierr = VecRestoreArray(x, &xx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecPointwiseMult_Sieve"
PetscErrorCode VecPointwiseMult_Sieve(Vec w, Vec x, Vec y)
{
  ALE::Mesh::field_type *field = (ALE::Mesh::field_type *) w->data;
  PetscScalar           *ww    = (PetscScalar *) field->restrict(*field->getPatches()->begin());
  PetscInt               n     = w->map->n, i;
  PetscScalar           *xx, *yy;
  PetscErrorCode         ierr;

  PetscFunctionBegin;
  ierr = VecGetArray(x, &xx);CHKERRQ(ierr);
  if (x != y) {
    ierr = VecGetArray(y, &yy);CHKERRQ(ierr);
  } else {
    yy = xx;
  }

  if (ww == xx) {
    for (i=0; i<n; i++) ww[i] *= yy[i];
  } else if (ww == yy) {
    for (i=0; i<n; i++) ww[i] *= xx[i];
  } else {
#if defined(PETSC_USE_FORTRAN_KERNEL_XTIMESY)
    fortranxtimesy_(xx,yy,ww,&n);
#else
    for (i=0; i<n; i++) ww[i] = xx[i] * yy[i];
#endif
  }
  ierr = VecRestoreArray(x, &xx);CHKERRQ(ierr);
  if (x != y) {
    ierr = VecRestoreArray(y, &yy);CHKERRQ(ierr);
  }
  ierr = PetscLogFlops(n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecPointwiseDivide_Sieve"
PetscErrorCode VecPointwiseDivide_Sieve(Vec w, Vec x, Vec y)
{
  ALE::Mesh::field_type *field = (ALE::Mesh::field_type *) w->data;
  PetscScalar           *ww    = (PetscScalar *) field->restrict(*field->getPatches()->begin());
  PetscInt               n     = w->map->n, i;
  PetscScalar           *xx, *yy;
  PetscErrorCode         ierr;

  PetscFunctionBegin;
  ierr = VecGetArray(x, &xx);CHKERRQ(ierr);
  if (x != y) {
    ierr = VecGetArray(y, &yy);CHKERRQ(ierr);
  } else {
    yy = xx;
  }
  for (i=0; i<n; i++) {
    ww[i] = xx[i] / yy[i];
  }
  ierr = VecRestoreArray(x, &xx);CHKERRQ(ierr);
  if (x != y) {
    ierr = VecRestoreArray(y, &yy);CHKERRQ(ierr);
  }
  ierr = PetscLogFlops(n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecGetArray_Sieve"
PetscErrorCode VecGetArray_Sieve(Vec v, PetscScalar *a[])
{
  ALE::Mesh::field_type *field = (ALE::Mesh::field_type *) v->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (v->array_gotten) {
    SETERRQ(PETSC_ERR_ORDER,"Array has already been gotten for this vector,you may\n\
    have forgotten a call to VecRestoreArray()");
  }
  v->array_gotten = PETSC_TRUE;
  *a = (PetscScalar *) field->restrict(*field->getPatches()->begin());
  ierr = PetscObjectTakeAccess(v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecRestoreArray_Sieve"
PetscErrorCode VecRestoreArray_Sieve(Vec v, PetscScalar *a[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!v->array_gotten) {
    SETERRQ(PETSC_ERR_ORDER,"Array has not been gotten for this vector, you may\n\
    have forgotten a call to VecGetArray()");
  }
  v->array_gotten = PETSC_FALSE;
  if (a) *a = PETSC_NULL; /* now user cannot accidently use it again */
  ierr = PetscObjectGrantAccess(v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static struct _VecOps DvOps = {
  0, /* 1 */
  VecDuplicateVecs_Default,
  VecDestroyVecs_Default,
  0,
  0,
  0,
  0,
  0,
  VecScale_Sieve,
  VecCopy_Sieve, /* 10 */
  VecSet_Sieve,
  0,
  VecAXPY_Sieve,
  VecAXPBY_Sieve,
  VecMAXPY_Sieve,
  VecAYPX_Sieve,
  VecWAXPY_Sieve,
  VecPointwiseMult_Sieve,
  VecPointwiseDivide_Sieve, 
  0, /* 20 */
  0,
  0,
  VecGetArray_Sieve,
  0,
  0,
  VecRestoreArray_Sieve,
  0,
  0,
  0,
  0, /* 30 */
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0, /* 40 */
  0,
  0,
  0, /* VecViewNative... */
  0,
  0,
  0,
  0,
  0,
  0,
  0, /* 50 */
  0,
  0,
  0,
  0};

#undef __FUNCT__  
#define __FUNCT__ "VecCreate_Sieve_Private"
PetscErrorCode VecCreate_Sieve_Private(Vec v, ALE::Mesh::field_type *field)
{
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr           = PetscLogObjectMemory(v, sizeof(ALE::Mesh::field_type) + v->map->n*sizeof(double));CHKERRQ(ierr);
  ierr           = PetscMemcpy(v->ops, &DvOps, sizeof(DvOps));CHKERRQ(ierr);
  v->data        = (void *) field;
  v->mapping     = PETSC_NULL;
  v->bmapping    = PETSC_NULL;
  v->petscnative = PETSC_FALSE;

  if (v->map->bs == -1) v->map->bs = 1;
  ierr = PetscLayoutSetUp(v->map);CHKERRQ(ierr);
  v->stash.insertmode = NOT_SET_VALUES;
                                                        
  ierr = PetscObjectChangeTypeName((PetscObject) v, VECSIEVE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
   VECSIEVE - VECSIEVE = "sieve" - The parallel vector based upon Sieve fields

   Options Database Keys:
. -vec_type sieve - sets the vector type to VECSIEVE during a call to VecSetFromOptions()

  Level: beginner

.seealso: VecCreate(), VecSetType(), VecSetFromOptions(), VECSIEVE, VecType
M*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "VecCreate_Sieve"
PetscErrorCode PETSCVEC_DLLEXPORT VecCreate_Sieve(Vec v)
{
  ALE::Mesh::field_type *field = new ALE::Mesh::field_type(((PetscObject)v)->comm, 0);
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCreate_Sieve_Private(v, field);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "VecCreateSieve"
/*@C
   VecCreateSieve - Creates a parallel vector implemented by the Sieve Field.

   Collective on MPI_Comm

   Input Parameter:
.  field - The Sieve Field

   Output Parameter:
.  v - The vector

   Level: intermediate

   Concepts: vectors^creating with array

.seealso: VecCreate(), VecDuplicate(), VecDuplicateVecs()
@*/ 
PetscErrorCode PETSCVEC_DLLEXPORT VecCreateSieve(ALE::Mesh::field_type *field, Vec *v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCreate(field->comm(), v);CHKERRQ(ierr);
  (*v)->map->n = field->getSize();
  (*v)->map->N = field->getGlobalOffsets()[field->commSize()];
  ierr = VecCreate_Sieve_Private(*v, field);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
