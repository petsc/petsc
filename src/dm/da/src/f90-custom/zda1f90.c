#include "../src/sys/f90-src/f90impl.h"
#include "petscda.h"

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define dagetlocalinfof90_           DAGETLOCALINFOF90
#define davecgetarrayf901_           DAVECGETARRAYF901
#define davecrestorearrayf901_       DAVECRESTOREARRAYF901
#define davecrestorearrayf90user1_   DAVECRESTOREARRAYF90USER1
#define davecgetarrayf90user1_       DAVECGETARRAYF90USER1
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define dagetlocalinfof90_           dagetlocalinfof90
#define davecgetarrayf901_           davecgetarrayf901
#define davecrestorearrayf901_       davecrestorearrayf901
#define davecrestorearrayf90user1_   davecrestorearrayf90user1
#define davecgetarrayf90user1_       davecgetarrayf90user1
#endif

EXTERN_C_BEGIN


void PETSC_STDCALL dagetlocalinfof90_(DA *da,DALocalInfo *info,PetscErrorCode *ierr)
{
  *ierr = DAGetLocalInfo(*da,info);
}

void PETSC_STDCALL davecgetarrayf901_(DA *da,Vec *v,F90Array1d *a,PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscInt    xs,ys,zs,xm,ym,zm,gxs,gys,gzs,gxm,gym,gzm,N,dim,dof;
  PetscScalar *aa;
  
  PetscFunctionBegin;
  *ierr = DAGetCorners(*da,&xs,&ys,&zs,&xm,&ym,&zm);if (*ierr) return;
  *ierr = DAGetGhostCorners(*da,&gxs,&gys,&gzs,&gxm,&gym,&gzm);if (*ierr) return;
  *ierr = DAGetInfo(*da,&dim,0,0,0,0,0,0,&dof,0,0,0);if (*ierr) return;

  /* Handle case where user passes in global vector as opposed to local */
  *ierr = VecGetLocalSize(*v,&N);if (*ierr) return;
  if (N == xm*ym*zm*dof) {
    gxm = xm;
    gym = ym;
    gzm = zm;
    gxs = xs;
    gys = ys;
    gzs = zs;
  } else if (N != gxm*gym*gzm*dof) {
    *ierr = PETSC_ERR_ARG_INCOMP;
  }
  *ierr = VecGetArray(*v,&aa);if (*ierr) return;
  *ierr = F90Array1dCreate(aa,PETSC_SCALAR,gxs,gxs+gxm-1,a PETSC_F90_2PTR_PARAM(ptrd));if (*ierr) return;
}

void PETSC_STDCALL davecrestorearrayf901_(DA *da,Vec *v,F90Array1d *a,PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  *ierr = VecRestoreArray(*v,0);if (*ierr) return;
  *ierr = F90Array1dDestroy(a,PETSC_SCALAR PETSC_F90_2PTR_PARAM(ptrd));
}

void PETSC_STDCALL davecgetarrayf90user1_(DA *da,Vec *v,F90Array1d *a,PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscInt    xs,ys,zs,xm,ym,zm,gxs,gys,gzs,gxm,gym,gzm,N,dim,dof;
  PetscScalar *aa;
  
  PetscFunctionBegin;
  *ierr = DAGetCorners(*da,&xs,&ys,&zs,&xm,&ym,&zm);if (*ierr) return;
  *ierr = DAGetGhostCorners(*da,&gxs,&gys,&gzs,&gxm,&gym,&gzm);if (*ierr) return;
  *ierr = DAGetInfo(*da,&dim,0,0,0,0,0,0,&dof,0,0,0);if (*ierr) return;

  /* Handle case where user passes in global vector as opposed to local */
  *ierr = VecGetLocalSize(*v,&N);if (*ierr) return;
  if (N == xm*ym*zm*dof) {
    gxm = xm;
    gym = ym;
    gzm = zm;
    gxs = xs;
    gys = ys;
    gzs = zs;
  } else if (N != gxm*gym*gzm*dof) {
    *ierr = PETSC_ERR_ARG_INCOMP;
  }
  *ierr = VecGetArray(*v,&aa);if (*ierr) return;
  if (dof > 1) {
    *ierr = F90Array1dCreate(aa,(PetscDataType)(-1*dof*sizeof(PetscScalar)),gxs,gxs+gxm-1,a PETSC_F90_2PTR_PARAM(ptrd));
  } else {
    *ierr = F90Array1dCreate(aa,PETSC_SCALAR,gxs,gxs+gxm-1,a PETSC_F90_2PTR_PARAM(ptrd));
  }
}

void PETSC_STDCALL davecrestorearrayf90user1_(DA *da,Vec *v,F90Array1d *a,PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  *ierr = VecRestoreArray(*v,0);if (*ierr) return;
  *ierr = F90Array1dDestroy(a,PETSC_SCALAR PETSC_F90_2PTR_PARAM(ptrd));
}

EXTERN_C_END
