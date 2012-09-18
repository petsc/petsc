#include <../src/sys/f90-src/f90impl.h>
#include <petscdmda.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define dmdagetlocalinfof90_           DMDAGETLOCALINFOF90
#define dmdavecgetarrayf901_           DMDAVECGETARRAYF901
#define dmdavecrestorearrayf901_       DMDAVECRESTOREARRAYF901
#define dmdavecgetarrayf902_           DMDAVECGETARRAYF902
#define dmdavecrestorearrayf902_       DMDAVECRESTOREARRAYF902
#define dmdavecgetarrayf903_           DMDAVECGETARRAYF903
#define dmdavecrestorearrayf903_       DMDAVECRESTOREARRAYF903
#define dmdavecgetarrayf904_           DMDAVECGETARRAYF904
#define dmdavecrestorearrayf904_       DMDAVECRESTOREARRAYF904
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define dmdagetlocalinfof90_           dmdagetlocalinfof90
#define dmdavecgetarrayf901_           dmdavecgetarrayf901
#define dmdavecrestorearrayf901_       dmdavecrestorearrayf901
#define dmdavecgetarrayf902_           dmdavecgetarrayf902
#define dmdavecrestorearrayf902_       dmdavecrestorearrayf902
#define dmdavecgetarrayf903_           dmdavecgetarrayf903
#define dmdavecrestorearrayf903_       dmdavecrestorearrayf903
#define dmdavecgetarrayf904_           dmdavecgetarrayf904
#define dmdavecrestorearrayf904_       dmdavecrestorearrayf904
#endif

EXTERN_C_BEGIN


void PETSC_STDCALL dmdagetlocalinfof90_(DM *da,DMDALocalInfo *info,PetscErrorCode *ierr)
{
  *ierr = DMDAGetLocalInfo(*da,info);
}

void PETSC_STDCALL dmdavecgetarrayf901_(DM *da,Vec *v,F90Array1d *a,PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscInt    xs,ys,zs,xm,ym,zm,gxs,gys,gzs,gxm,gym,gzm,N,dim,dof;
  PetscScalar *aa;

  *ierr = DMDAGetCorners(*da,&xs,&ys,&zs,&xm,&ym,&zm);if (*ierr) return;
  *ierr = DMDAGetGhostCorners(*da,&gxs,&gys,&gzs,&gxm,&gym,&gzm);if (*ierr) return;
  *ierr = DMDAGetInfo(*da,&dim,0,0,0,0,0,0,&dof,0,0,0,0,0);if (*ierr) return;

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
  *ierr = F90Array1dCreate(aa,PETSC_SCALAR,gxs,gxm,a PETSC_F90_2PTR_PARAM(ptrd));if (*ierr) return;
}

void PETSC_STDCALL dmdavecrestorearrayf901_(DM *da,Vec *v,F90Array1d *a,PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscScalar *fa;
  *ierr = F90Array1dAccess(a,PETSC_SCALAR,(void**)&fa PETSC_F90_2PTR_PARAM(ptrd));
  *ierr = VecRestoreArray(*v,&fa);if (*ierr) return;
  *ierr = F90Array1dDestroy(&a,PETSC_SCALAR PETSC_F90_2PTR_PARAM(ptrd));
}

void PETSC_STDCALL dmdavecgetarrayf902_(DM *da,Vec *v,F90Array2d *a,PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscInt    xs,ys,zs,xm,ym,zm,gxs,gys,gzs,gxm,gym,gzm,N,dim,dof;
  PetscScalar *aa;

  *ierr = DMDAGetCorners(*da,&xs,&ys,&zs,&xm,&ym,&zm);if (*ierr) return;
  *ierr = DMDAGetGhostCorners(*da,&gxs,&gys,&gzs,&gxm,&gym,&gzm);if (*ierr) return;
  *ierr = DMDAGetInfo(*da,&dim,0,0,0,0,0,0,&dof,0,0,0,0,0);if (*ierr) return;

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
  if (dim == 1) {
    gys = gxs;
    gym = gxm;
    gxs = 0;
    gxm = dof;
  }
  *ierr = VecGetArray(*v,&aa);if (*ierr) return;
  *ierr = F90Array2dCreate(aa,PETSC_SCALAR,gxs,gxm,gys,gym,a PETSC_F90_2PTR_PARAM(ptrd));if (*ierr) return;
}

void PETSC_STDCALL dmdavecrestorearrayf902_(DM *da,Vec *v,F90Array2d *a,PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscScalar *fa;
  *ierr = F90Array2dAccess(a,PETSC_SCALAR,(void**)&fa PETSC_F90_2PTR_PARAM(ptrd));
  *ierr = VecRestoreArray(*v,&fa);if (*ierr) return;
  *ierr = F90Array2dDestroy(&a,PETSC_SCALAR PETSC_F90_2PTR_PARAM(ptrd));
}

void PETSC_STDCALL dmdavecgetarrayf903_(DM *da,Vec *v,F90Array3d *a,PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscInt    xs,ys,zs,xm,ym,zm,gxs,gys,gzs,gxm,gym,gzm,N,dim,dof;
  PetscScalar *aa;

  *ierr = DMDAGetCorners(*da,&xs,&ys,&zs,&xm,&ym,&zm);if (*ierr) return;
  *ierr = DMDAGetGhostCorners(*da,&gxs,&gys,&gzs,&gxm,&gym,&gzm);if (*ierr) return;
  *ierr = DMDAGetInfo(*da,&dim,0,0,0,0,0,0,&dof,0,0,0,0,0);if (*ierr) return;

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
  if (dim == 2) {
    gzs = gys;
    gzm = gym;
    gys = gxs;
    gym = gxm;
    gxs = 0;
    gxm = dof;
  }
  *ierr = VecGetArray(*v,&aa);if (*ierr) return;
  *ierr = F90Array3dCreate(aa,PETSC_SCALAR,gxs,gxm,gys,gym,gzs,gzm,a PETSC_F90_2PTR_PARAM(ptrd));if (*ierr) return;
}

void PETSC_STDCALL dmdavecrestorearrayf903_(DM *da,Vec *v,F90Array3d *a,PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscScalar *fa;
  *ierr = F90Array3dAccess(a,PETSC_SCALAR,(void**)&fa PETSC_F90_2PTR_PARAM(ptrd));
  *ierr = VecRestoreArray(*v,&fa);if (*ierr) return;
  *ierr = F90Array3dDestroy(&a,PETSC_SCALAR PETSC_F90_2PTR_PARAM(ptrd));
}

void PETSC_STDCALL dmdavecgetarrayf904_(DM *da,Vec *v,F90Array4d *a,PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscInt    xs,ys,zs,xm,ym,zm,gxs,gys,gzs,gxm,gym,gzm,N,dim,dof,zero = 0;
  PetscScalar *aa;

  *ierr = DMDAGetCorners(*da,&xs,&ys,&zs,&xm,&ym,&zm);if (*ierr) return;
  *ierr = DMDAGetGhostCorners(*da,&gxs,&gys,&gzs,&gxm,&gym,&gzm);if (*ierr) return;
  *ierr = DMDAGetInfo(*da,&dim,0,0,0,0,0,0,&dof,0,0,0,0,0);if (*ierr) return;

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
  *ierr = F90Array4dCreate(aa,PETSC_SCALAR,zero,dof,gxs,gxm,gys,gym,gzs,gzm,a PETSC_F90_2PTR_PARAM(ptrd));if (*ierr) return;
}

void PETSC_STDCALL dmdavecrestorearrayf904_(DM *da,Vec *v,F90Array4d *a,PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscScalar *fa;
  /*
    F90Array4dAccess is not implemented, so the following call would fail
  */
  *ierr = F90Array4dAccess(a,PETSC_SCALAR,(void**)&fa PETSC_F90_2PTR_PARAM(ptrd));
  *ierr = VecRestoreArray(*v,&fa);if (*ierr) return;
  *ierr = F90Array4dDestroy(&a,PETSC_SCALAR PETSC_F90_2PTR_PARAM(ptrd));
}

EXTERN_C_END
