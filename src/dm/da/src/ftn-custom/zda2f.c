
#include "zpetsc.h"
#include "petscda.h"

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define dasetlocaljacobian_          DASETLOCALJACOBIAN
#define dasetlocalfunction_          DASETLOCALFUNCTION
#define dacreate2d_                  DACREATE2D
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define dasetlocaljacobian_          dasetlocaljacobian
#define dasetlocalfunction_          dasetlocalfunction
#define dacreate2d_                  dacreate2d
#endif

EXTERN_C_BEGIN

/************************************************/
static void (PETSC_STDCALL *j1d)(DALocalInfo*,void*,void*,void*,PetscErrorCode*);
static PetscErrorCode ourlj1d(DALocalInfo *info,PetscScalar *in,Mat m,void *ptr)
{
  PetscErrorCode ierr = 0;
  (*j1d)(info,&in[info->dof*info->gxs],&m,ptr,&ierr);CHKERRQ(ierr);
  return 0;
}

static void (PETSC_STDCALL *j2d)(DALocalInfo*,void*,void*,void*,PetscErrorCode*);
static PetscErrorCode ourlj2d(DALocalInfo *info,PetscScalar **in,Mat m,void *ptr)
{
  PetscErrorCode ierr = 0;
  (*j2d)(info,&in[info->gys][info->dof*info->gxs],&m,ptr,&ierr);CHKERRQ(ierr);
  return 0;
}

static void (PETSC_STDCALL *j3d)(DALocalInfo*,void*,void*,void*,PetscErrorCode*);
static PetscErrorCode ourlj3d(DALocalInfo *info,PetscScalar ***in,Mat m,void *ptr)
{
  PetscErrorCode ierr = 0;
  (*j3d)(info,&in[info->gzs][info->gys][info->dof*info->gxs],&m,ptr,&ierr);CHKERRQ(ierr);
  return 0;
}

void PETSC_STDCALL dasetlocaljacobian_(DA *da,void (PETSC_STDCALL *jac)(DALocalInfo*,void*,void*,void*,PetscErrorCode*),PetscErrorCode *ierr)
{
  PetscInt dim;

  *ierr = DAGetInfo(*da,&dim,0,0,0,0,0,0,0,0,0,0); if (*ierr) return;
  if (dim == 2) {
     j2d    = (void (PETSC_STDCALL *)(DALocalInfo*,void*,void*,void*,PetscErrorCode*))jac; 
    *ierr = DASetLocalJacobian(*da,(DALocalFunction1)ourlj2d);
  } else if (dim == 3) {
     j3d    = (void (PETSC_STDCALL *)(DALocalInfo*,void*,void*,void*,PetscErrorCode*))jac;
    *ierr = DASetLocalJacobian(*da,(DALocalFunction1)ourlj3d);
  } else if (dim == 1) {
     j1d    = (void (PETSC_STDCALL *)(DALocalInfo*,void*,void*,void*,PetscErrorCode*))jac; 
    *ierr = DASetLocalJacobian(*da,(DALocalFunction1)ourlj1d);
  } else *ierr = 1;
}

/************************************************/
static void (PETSC_STDCALL *f1d)(DALocalInfo*,void*,void*,void*,PetscErrorCode*);
static PetscErrorCode ourlf1d(DALocalInfo *info,PetscScalar *in,PetscScalar *out,void *ptr)
{
  PetscErrorCode ierr = 0;
  (*f1d)(info,&in[info->dof*info->gxs],&out[info->dof*info->xs],ptr,&ierr);CHKERRQ(ierr);
  return 0;
}

static void (PETSC_STDCALL *f2d)(DALocalInfo*,void*,void*,void*,PetscErrorCode*);
static PetscErrorCode ourlf2d(DALocalInfo *info,PetscScalar **in,PetscScalar **out,void *ptr)
{
  PetscErrorCode ierr = 0;
  (*f2d)(info,&in[info->gys][info->dof*info->gxs],&out[info->ys][info->dof*info->xs],ptr,&ierr);CHKERRQ(ierr);
  return 0;
}

static void (PETSC_STDCALL *f3d)(DALocalInfo*,void*,void*,void*,PetscErrorCode*);
static PetscErrorCode ourlf3d(DALocalInfo *info,PetscScalar ***in,PetscScalar ***out,void *ptr)
{
  PetscErrorCode ierr = 0;
  (*f3d)(info,&in[info->gzs][info->gys][info->dof*info->gxs],&out[info->zs][info->ys][info->dof*info->xs],ptr,&ierr);CHKERRQ(ierr);
  return 0;
}

void PETSC_STDCALL dasetlocalfunction_(DA *da,void (PETSC_STDCALL *func)(DALocalInfo*,void*,void*,void*,PetscErrorCode*),PetscErrorCode *ierr)
{
  PetscInt dim;

  *ierr = DAGetInfo(*da,&dim,0,0,0,0,0,0,0,0,0,0); if (*ierr) return;
  if (dim == 2) {
     f2d    = (void (PETSC_STDCALL *)(DALocalInfo*,void*,void*,void*,PetscErrorCode*))func; 
    *ierr = DASetLocalFunction(*da,(DALocalFunction1)ourlf2d);
  } else if (dim == 3) {
     f3d    = (void (PETSC_STDCALL *)(DALocalInfo*,void*,void*,void*,PetscErrorCode*))func; 
    *ierr = DASetLocalFunction(*da,(DALocalFunction1)ourlf3d);
  } else if (dim == 1) {
     f1d    = (void (PETSC_STDCALL *)(DALocalInfo*,void*,void*,void*,PetscErrorCode*))func; 
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
  *ierr = DACreate2d((MPI_Comm)PetscToPointerComm(*comm),*wrap,
                       *stencil_type,*M,*N,*m,*n,*w,*s,lx,ly,inra);
}


EXTERN_C_END
