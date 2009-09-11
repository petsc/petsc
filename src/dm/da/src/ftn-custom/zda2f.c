
#include "private/fortranimpl.h"
#include "private/daimpl.h"

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define dasetlocaljacobian_          DASETLOCALJACOBIAN
#define dasetlocalfunction_          DASETLOCALFUNCTION
#define dacreate2d_                  DACREATE2D
#define dagetownershipranges_        DAGETOWNERSHIPRANGES
#define dagetneighbors_              DAGETNEIGHBORS
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define dasetlocaljacobian_          dasetlocaljacobian
#define dasetlocalfunction_          dasetlocalfunction
#define dacreate2d_                  dacreate2d
#define dagetownershipranges_        dagetownershipranges
#define dagetneighbors_              dagetneighbors
#endif

EXTERN_C_BEGIN
void PETSC_STDCALL dagetneighbors_(DA *da,PetscMPIInt *ranks,PetscErrorCode *ierr)
{
  const PetscMPIInt *r;
  PetscInt       n;

  *ierr = DAGetNeighbors(*da,&r);if (*ierr) return;
  if ((*da)->dim == 2) n = 9; else n = 27;
  *ierr = PetscMemcpy(ranks,r,n*sizeof(PetscMPIInt));  
}


/************************************************/
static PetscErrorCode ourlj1d(DALocalInfo *info,PetscScalar *in,Mat m,void *ptr)
{
  PetscErrorCode ierr = 0;
  (*(void (PETSC_STDCALL *)(DALocalInfo*,PetscScalar*,Mat*,void*,PetscErrorCode*))(((PetscObject)info->da)->fortran_func_pointers[0]))(info,&in[info->dof*info->gxs],&m,ptr,&ierr);CHKERRQ(ierr);
  return 0;
}

static PetscErrorCode ourlj2d(DALocalInfo *info,PetscScalar **in,Mat m,void *ptr)
{
  PetscErrorCode ierr = 0;
  (*(void (PETSC_STDCALL *)(DALocalInfo*,PetscScalar*,Mat*,void*,PetscErrorCode*))(((PetscObject)info->da)->fortran_func_pointers[1]))(info,&in[info->gys][info->dof*info->gxs],&m,ptr,&ierr);CHKERRQ(ierr);
  return 0;
}

static PetscErrorCode ourlj3d(DALocalInfo *info,PetscScalar ***in,Mat m,void *ptr)
{
  PetscErrorCode ierr = 0;
  (*(void (PETSC_STDCALL *)(DALocalInfo*,PetscScalar*,Mat*,void*,PetscErrorCode*))(((PetscObject)info->da)->fortran_func_pointers[2]))(info,&in[info->gzs][info->gys][info->dof*info->gxs],&m,ptr,&ierr);CHKERRQ(ierr);
  return 0;
}

void PETSC_STDCALL dasetlocaljacobian_(DA *da,void (PETSC_STDCALL *jac)(DALocalInfo*,void*,void*,void*,PetscErrorCode*),PetscErrorCode *ierr)
{
  PetscInt dim;

  PetscObjectAllocateFortranPointers(*da,6);
  *ierr = DAGetInfo(*da,&dim,0,0,0,0,0,0,0,0,0,0); if (*ierr) return;
  if (dim == 2) {
    ((PetscObject)*da)->fortran_func_pointers[1] = (PetscVoidFunction)jac;
    *ierr = DASetLocalJacobian(*da,(DALocalFunction1)ourlj2d);
  } else if (dim == 3) {
    ((PetscObject)*da)->fortran_func_pointers[2] = (PetscVoidFunction)jac;
    *ierr = DASetLocalJacobian(*da,(DALocalFunction1)ourlj3d);
  } else if (dim == 1) {
    ((PetscObject)*da)->fortran_func_pointers[0] = (PetscVoidFunction)jac;
    *ierr = DASetLocalJacobian(*da,(DALocalFunction1)ourlj1d);
  } else *ierr = 1;
}

/************************************************/

static PetscErrorCode ourlf1d(DALocalInfo *info,PetscScalar *in,PetscScalar *out,void *ptr)
{
  PetscErrorCode ierr = 0;
  (*(void (PETSC_STDCALL *)(DALocalInfo*,PetscScalar*,PetscScalar*,void*,PetscErrorCode*))(((PetscObject)info->da)->fortran_func_pointers[3]))(info,&in[info->dof*info->gxs],&out[info->dof*info->xs],ptr,&ierr);CHKERRQ(ierr);
  return 0;
}

static PetscErrorCode ourlf2d(DALocalInfo *info,PetscScalar **in,PetscScalar **out,void *ptr)
{
  PetscErrorCode ierr = 0;
  (*(void (PETSC_STDCALL *)(DALocalInfo*,PetscScalar*,PetscScalar*,void*,PetscErrorCode*))(((PetscObject)info->da)->fortran_func_pointers[4]))(info,&in[info->gys][info->dof*info->gxs],&out[info->ys][info->dof*info->xs],ptr,&ierr);CHKERRQ(ierr);
  return 0;
}

static PetscErrorCode ourlf3d(DALocalInfo *info,PetscScalar ***in,PetscScalar ***out,void *ptr)
{
  PetscErrorCode ierr = 0;
  (*(void (PETSC_STDCALL *)(DALocalInfo*,PetscScalar*,PetscScalar*,void*,PetscErrorCode*))(((PetscObject)info->da)->fortran_func_pointers[5]))(info,&in[info->gzs][info->gys][info->dof*info->gxs],&out[info->zs][info->ys][info->dof*info->xs],ptr,&ierr);CHKERRQ(ierr);
  return 0;
}

void PETSC_STDCALL dasetlocalfunction_(DA *da,void (PETSC_STDCALL *func)(DALocalInfo*,void*,void*,void*,PetscErrorCode*),PetscErrorCode *ierr)
{
  PetscInt dim;

  PetscObjectAllocateFortranPointers(*da,6);
  *ierr = DAGetInfo(*da,&dim,0,0,0,0,0,0,0,0,0,0); if (*ierr) return;
  if (dim == 2) {
    ((PetscObject)*da)->fortran_func_pointers[4] = (PetscVoidFunction)func;
    *ierr = DASetLocalFunction(*da,(DALocalFunction1)ourlf2d);
  } else if (dim == 3) {
    ((PetscObject)*da)->fortran_func_pointers[5] = (PetscVoidFunction)func;
    *ierr = DASetLocalFunction(*da,(DALocalFunction1)ourlf3d);
  } else if (dim == 1) {
    ((PetscObject)*da)->fortran_func_pointers[3] = (PetscVoidFunction)func;
    *ierr = DASetLocalFunction(*da,(DALocalFunction1)ourlf1d);
  } else *ierr = 1;
}

/************************************************/

void PETSC_STDCALL dacreate2d_(MPI_Comm *comm,DAPeriodicType *wrap,DAStencilType
                  *stencil_type,PetscInt *M,PetscInt *N,PetscInt *m,PetscInt *n,PetscInt *w,
                  PetscInt *s,PetscInt *lx,PetscInt *ly,DA *inra,PetscErrorCode *ierr)
{
  CHKFORTRANNULLINTEGER(lx);
  CHKFORTRANNULLINTEGER(ly);
  *ierr = DACreate2d(MPI_Comm_f2c(*(MPI_Fint *)&*comm),*wrap,*stencil_type,*M,*N,*m,*n,*w,*s,lx,ly,inra);
}

void PETSC_STDCALL dagetownershipranges_(DA *da,PetscInt lx[],PetscInt ly[],PetscInt lz[],PetscErrorCode *ierr)
{
  const PetscInt *gx,*gy,*gz;
  PetscInt       M,N,P,i;
  
  CHKFORTRANNULLINTEGER(lx);
  CHKFORTRANNULLINTEGER(ly);
  CHKFORTRANNULLINTEGER(lz);
  *ierr = DAGetInfo(*da,0,0,0,0,&M,&N,&P,0,0,0,0);if (*ierr) return;
  *ierr = DAGetOwnershipRanges(*da,&gx,&gy,&gz);if (*ierr) return;
  if (lx) {for (i=0; i<M; i++) {lx[i] = gx[i];}}
  if (ly) {for (i=0; i<N; i++) {ly[i] = gy[i];}}
  if (lz) {for (i=0; i<P; i++) {lz[i] = gz[i];}}
}


EXTERN_C_END
