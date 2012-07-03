
#include <petsc-private/fortranimpl.h>
#include <petsc-private/daimpl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define dmdasetlocaljacobian_          DMDASETLOCALJACOBIAN
#define dmdasetlocalfunction_          DMDASETLOCALFUNCTION
#define dmdacreate2d_                  DMDACREATE2D
#define dmdagetownershipranges_        DMDAGETOWNERSHIPRANGES
#define dmdagetneighbors_              DMDAGETNEIGHBORS
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define dmdasetlocaljacobian_          dmdasetlocaljacobian
#define dmdasetlocalfunction_          dmdasetlocalfunction
#define dmdacreate2d_                  dmdacreate2d
#define dmdagetownershipranges_        dmdagetownershipranges
#define dmdagetneighbors_              dmdagetneighbors
#endif

EXTERN_C_BEGIN
void PETSC_STDCALL dmdagetneighbors_(DM *da,PetscMPIInt *ranks,PetscErrorCode *ierr)
{
  const PetscMPIInt *r;
  PetscInt          n;
  DM_DA            *dd = (DM_DA*)(*da)->data;

  *ierr = DMDAGetNeighbors(*da,&r);if (*ierr) return;
  if (dd->dim == 2) n = 9; else n = 27;
  *ierr = PetscMemcpy(ranks,r,n*sizeof(PetscMPIInt));  
}


/************************************************/
static PetscErrorCode ourlj1d(DMDALocalInfo *info,PetscScalar *in,Mat m,void *ptr)
{
  PetscErrorCode ierr = 0;
  (*(void (PETSC_STDCALL *)(DMDALocalInfo*,PetscScalar*,Mat*,void*,PetscErrorCode*))(((PetscObject)info->da)->fortran_func_pointers[0]))(info,&in[info->dof*info->gxs],&m,ptr,&ierr);CHKERRQ(ierr);
  return 0;
}

static PetscErrorCode ourlj2d(DMDALocalInfo *info,PetscScalar **in,Mat m,void *ptr)
{
  PetscErrorCode ierr = 0;
  (*(void (PETSC_STDCALL *)(DMDALocalInfo*,PetscScalar*,Mat*,void*,PetscErrorCode*))(((PetscObject)info->da)->fortran_func_pointers[1]))(info,&in[info->gys][info->dof*info->gxs],&m,ptr,&ierr);CHKERRQ(ierr);
  return 0;
}

static PetscErrorCode ourlj3d(DMDALocalInfo *info,PetscScalar ***in,Mat m,void *ptr)
{
  PetscErrorCode ierr = 0;
  (*(void (PETSC_STDCALL *)(DMDALocalInfo*,PetscScalar*,Mat*,void*,PetscErrorCode*))(((PetscObject)info->da)->fortran_func_pointers[2]))(info,&in[info->gzs][info->gys][info->dof*info->gxs],&m,ptr,&ierr);CHKERRQ(ierr);
  return 0;
}

void PETSC_STDCALL dmdasetlocaljacobian_(DM *da,void (PETSC_STDCALL *jac)(DMDALocalInfo*,void*,void*,void*,PetscErrorCode*),PetscErrorCode *ierr)
{
  PetscInt dim;

  PetscObjectAllocateFortranPointers(*da,6);
  *ierr = DMDAGetInfo(*da,&dim,0,0,0,0,0,0,0,0,0,0,0,0); if (*ierr) return;
  if (dim == 2) {
    ((PetscObject)*da)->fortran_func_pointers[1] = (PetscVoidFunction)jac;
    *ierr = DMDASetLocalJacobian(*da,(DMDALocalFunction1)ourlj2d);
  } else if (dim == 3) {
    ((PetscObject)*da)->fortran_func_pointers[2] = (PetscVoidFunction)jac;
    *ierr = DMDASetLocalJacobian(*da,(DMDALocalFunction1)ourlj3d);
  } else if (dim == 1) {
    ((PetscObject)*da)->fortran_func_pointers[0] = (PetscVoidFunction)jac;
    *ierr = DMDASetLocalJacobian(*da,(DMDALocalFunction1)ourlj1d);
  } else *ierr = 1;
}

/************************************************/

static PetscErrorCode ourlf1d(DMDALocalInfo *info,PetscScalar *in,PetscScalar *out,void *ptr)
{
  PetscErrorCode ierr = 0;
  (*(void (PETSC_STDCALL *)(DMDALocalInfo*,PetscScalar*,PetscScalar*,void*,PetscErrorCode*))(((PetscObject)info->da)->fortran_func_pointers[3]))(info,&in[info->dof*info->gxs],&out[info->dof*info->xs],ptr,&ierr);CHKERRQ(ierr);
  return 0;
}

static PetscErrorCode ourlf2d(DMDALocalInfo *info,PetscScalar **in,PetscScalar **out,void *ptr)
{
  PetscErrorCode ierr = 0;
  (*(void (PETSC_STDCALL *)(DMDALocalInfo*,PetscScalar*,PetscScalar*,void*,PetscErrorCode*))(((PetscObject)info->da)->fortran_func_pointers[4]))(info,&in[info->gys][info->dof*info->gxs],&out[info->ys][info->dof*info->xs],ptr,&ierr);CHKERRQ(ierr);
  return 0;
}

static PetscErrorCode ourlf3d(DMDALocalInfo *info,PetscScalar ***in,PetscScalar ***out,void *ptr)
{
  PetscErrorCode ierr = 0;
  (*(void (PETSC_STDCALL *)(DMDALocalInfo*,PetscScalar*,PetscScalar*,void*,PetscErrorCode*))(((PetscObject)info->da)->fortran_func_pointers[5]))(info,&in[info->gzs][info->gys][info->dof*info->gxs],&out[info->zs][info->ys][info->dof*info->xs],ptr,&ierr);CHKERRQ(ierr);
  return 0;
}

void PETSC_STDCALL dmdasetlocalfunction_(DM *da,void (PETSC_STDCALL *func)(DMDALocalInfo*,void*,void*,void*,PetscErrorCode*),PetscErrorCode *ierr)
{
  PetscInt dim;

  PetscObjectAllocateFortranPointers(*da,6);
  *ierr = DMDAGetInfo(*da,&dim,0,0,0,0,0,0,0,0,0,0,0,0); if (*ierr) return;
  if (dim == 2) {
    ((PetscObject)*da)->fortran_func_pointers[4] = (PetscVoidFunction)func;
    *ierr = DMDASetLocalFunction(*da,(DMDALocalFunction1)ourlf2d);
  } else if (dim == 3) {
    ((PetscObject)*da)->fortran_func_pointers[5] = (PetscVoidFunction)func;
    *ierr = DMDASetLocalFunction(*da,(DMDALocalFunction1)ourlf3d);
  } else if (dim == 1) {
    ((PetscObject)*da)->fortran_func_pointers[3] = (PetscVoidFunction)func;
    *ierr = DMDASetLocalFunction(*da,(DMDALocalFunction1)ourlf1d);
  } else *ierr = 1;
}

/************************************************/

void PETSC_STDCALL dmdacreate2d_(MPI_Comm *comm,DMDABoundaryType *bx,DMDABoundaryType *by,DMDAStencilType
                  *stencil_type,PetscInt *M,PetscInt *N,PetscInt *m,PetscInt *n,PetscInt *w,
                  PetscInt *s,PetscInt *lx,PetscInt *ly,DM *inra,PetscErrorCode *ierr)
{
  CHKFORTRANNULLINTEGER(lx);
  CHKFORTRANNULLINTEGER(ly);
  *ierr = DMDACreate2d(MPI_Comm_f2c(*(MPI_Fint *)&*comm),*bx,*by,*stencil_type,*M,*N,*m,*n,*w,*s,lx,ly,inra);
}

void PETSC_STDCALL dmdagetownershipranges_(DM *da,PetscInt lx[],PetscInt ly[],PetscInt lz[],PetscErrorCode *ierr)
{
  const PetscInt *gx,*gy,*gz;
  PetscInt       M,N,P,i;
  
  CHKFORTRANNULLINTEGER(lx);
  CHKFORTRANNULLINTEGER(ly);
  CHKFORTRANNULLINTEGER(lz);
  *ierr = DMDAGetInfo(*da,0,0,0,0,&M,&N,&P,0,0,0,0,0,0);if (*ierr) return;
  *ierr = DMDAGetOwnershipRanges(*da,&gx,&gy,&gz);if (*ierr) return;
  if (lx) {for (i=0; i<M; i++) {lx[i] = gx[i];}}
  if (ly) {for (i=0; i<N; i++) {ly[i] = gy[i];}}
  if (lz) {for (i=0; i<P; i++) {lz[i] = gz[i];}}
}


EXTERN_C_END
