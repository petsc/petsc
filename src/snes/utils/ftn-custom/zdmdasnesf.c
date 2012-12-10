#include <petsc-private/fortranimpl.h>
#include <petsc-private/daimpl.h>
#include <petscsnes.h>
#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define dmdasnessetjacobianlocal_      DMDASNESSETJACOBIANLOCAL
#define dmdasnessetfunctionlocal_      DMDASNESSETFUNCTIONLOCAL
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define dmdasnessetjacobianlocal_      dmdasnessetjacobianlocal
#define dmdasnessetfunctionlocal_      dmdasnessetfunctionlocal
#endif

EXTERN_C_BEGIN
/************************************************/
static PetscErrorCode sourlj1d(DMDALocalInfo *info,PetscScalar *in,Mat A,Mat m,MatStructure *str,void *ptr)
{
  PetscErrorCode ierr = 0;
  (*(void (PETSC_STDCALL *)(DMDALocalInfo*,PetscScalar*,Mat*,Mat*,MatStructure*,void*,PetscErrorCode*))(((PetscObject)info->da)->fortran_func_pointers[0]))(info,&in[info->dof*info->gxs],&A,&m,str,ptr,&ierr);CHKERRQ(ierr);
  return 0;
}

static PetscErrorCode sourlj2d(DMDALocalInfo *info,PetscScalar **in,Mat A,Mat m,MatStructure *str,void *ptr)
{
  PetscErrorCode ierr = 0;
  (*(void (PETSC_STDCALL *)(DMDALocalInfo*,PetscScalar*,Mat*,Mat*,MatStructure*,void*,PetscErrorCode*))(((PetscObject)info->da)->fortran_func_pointers[1]))(info,&in[info->gys][info->dof*info->gxs],&A,&m,str,ptr,&ierr);CHKERRQ(ierr);
  return 0;
}

static PetscErrorCode sourlj3d(DMDALocalInfo *info,PetscScalar ***in,Mat A,Mat m,MatStructure *str,void *ptr)
{
  PetscErrorCode ierr = 0;
  (*(void (PETSC_STDCALL *)(DMDALocalInfo*,PetscScalar*,Mat*,Mat*,MatStructure*,void*,PetscErrorCode*))(((PetscObject)info->da)->fortran_func_pointers[2]))(info,&in[info->gzs][info->gys][info->dof*info->gxs],&A,&m,str,ptr,&ierr);CHKERRQ(ierr);
  return 0;
}

/*
   This is buggy, the function pointers should really be attached to the DMSNES object
*/
void PETSC_STDCALL dmdasnessetjacobianlocal_(DM *da,void (PETSC_STDCALL *jac)(DMDALocalInfo*,void*,void*,void*,void*,void*,PetscErrorCode*),void *ctx,PetscErrorCode *ierr)
{
  PetscInt dim;

  PetscObjectAllocateFortranPointers(*da,6);
  *ierr = DMDAGetInfo(*da,&dim,0,0,0,0,0,0,0,0,0,0,0,0); if (*ierr) return;
  if (dim == 2) {
    ((PetscObject)*da)->fortran_func_pointers[1] = (PetscVoidFunction)jac;
    *ierr = DMDASNESSetJacobianLocal(*da,(PetscErrorCode (*)(DMDALocalInfo*,void*,Mat,Mat,MatStructure*,void*))sourlj2d,ctx);
  } else if (dim == 3) {
    ((PetscObject)*da)->fortran_func_pointers[2] = (PetscVoidFunction)jac;
    *ierr = DMDASNESSetJacobianLocal(*da,(PetscErrorCode (*)(DMDALocalInfo*,void*,Mat,Mat,MatStructure*,void*))sourlj3d,ctx);
  } else if (dim == 1) {
    ((PetscObject)*da)->fortran_func_pointers[0] = (PetscVoidFunction)jac;
    *ierr = DMDASNESSetJacobianLocal(*da,(PetscErrorCode (*)(DMDALocalInfo*,void*,Mat,Mat,MatStructure*,void*))sourlj1d,ctx);
  } else *ierr = 1;
}

/************************************************/

static PetscErrorCode sourlf1d(DMDALocalInfo *info,PetscScalar *in,PetscScalar *out,void *ptr)
{
  PetscErrorCode ierr = 0;
  (*(void (PETSC_STDCALL *)(DMDALocalInfo*,PetscScalar*,PetscScalar*,void*,PetscErrorCode*))(((PetscObject)info->da)->fortran_func_pointers[3]))(info,&in[info->dof*info->gxs],&out[info->dof*info->xs],ptr,&ierr);CHKERRQ(ierr);
  return 0;
}

static PetscErrorCode sourlf2d(DMDALocalInfo *info,PetscScalar **in,PetscScalar **out,void *ptr)
{
  PetscErrorCode ierr = 0;
  (*(void (PETSC_STDCALL *)(DMDALocalInfo*,PetscScalar*,PetscScalar*,void*,PetscErrorCode*))(((PetscObject)info->da)->fortran_func_pointers[4]))(info,&in[info->gys][info->dof*info->gxs],&out[info->ys][info->dof*info->xs],ptr,&ierr);CHKERRQ(ierr);
  return 0;
}

static PetscErrorCode sourlf3d(DMDALocalInfo *info,PetscScalar ***in,PetscScalar ***out,void *ptr)
{
  PetscErrorCode ierr = 0;
  (*(void (PETSC_STDCALL *)(DMDALocalInfo*,PetscScalar*,PetscScalar*,void*,PetscErrorCode*))(((PetscObject)info->da)->fortran_func_pointers[5]))(info,&in[info->gzs][info->gys][info->dof*info->gxs],&out[info->zs][info->ys][info->dof*info->xs],ptr,&ierr);CHKERRQ(ierr);
  return 0;
}

/*
   This is buggy, the function pointers should really be attached to the DMSNES object
*/
void PETSC_STDCALL dmdasnessetfunctionlocal_(DM *da,InsertMode *mode,void (PETSC_STDCALL *func)(DMDALocalInfo*,void*,void*,void*,PetscErrorCode*),void *ctx,PetscErrorCode *ierr)
{
  PetscInt dim;

  PetscObjectAllocateFortranPointers(*da,6);
  *ierr = DMDAGetInfo(*da,&dim,0,0,0,0,0,0,0,0,0,0,0,0); if (*ierr) return;
  if (dim == 2) {
    ((PetscObject)*da)->fortran_func_pointers[4] = (PetscVoidFunction)func;
    *ierr = DMDASNESSetFunctionLocal(*da,*mode,(PetscErrorCode (*)(DMDALocalInfo*,void*,void*,void*))sourlf2d,ctx);
  } else if (dim == 3) {
    ((PetscObject)*da)->fortran_func_pointers[5] = (PetscVoidFunction)func;
    *ierr = DMDASNESSetFunctionLocal(*da,*mode,(PetscErrorCode (*)(DMDALocalInfo*,void*,void*,void*))sourlf3d,ctx);
  } else if (dim == 1) {
    ((PetscObject)*da)->fortran_func_pointers[3] = (PetscVoidFunction)func;
    *ierr = DMDASNESSetFunctionLocal(*da,*mode,(PetscErrorCode (*)(DMDALocalInfo*,void*,void*,void*))sourlf1d,ctx);
  } else *ierr = 1;
}

EXTERN_C_END
