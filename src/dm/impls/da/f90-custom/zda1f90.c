#include <petsc/private/f90impl.h>
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
#define dmdavecgetarrayreadf901_       DMDAVECGETARRAYREADF901
#define dmdavecrestorearrayreadf901_   DMDAVECRESTOREARRAYREADF901
#define dmdavecgetarrayreadf902_       DMDAVECGETARRAYREADF902
#define dmdavecrestorearrayreadf902_   DMDAVECRESTOREARRAYREADF902
#define dmdavecgetarrayreadf903_       DMDAVECGETARRAYREADF903
#define dmdavecrestorearrayreadf903_   DMDAVECRESTOREARRAYREADF903
#define dmdavecgetarrayreadf904_       DMDAVECGETARRAYREADF904
#define dmdavecrestorearrayreadf904_   DMDAVECRESTOREARRAYREADF904
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
#define dmdavecgetarrayreadf901_       dmdavecgetarrayreadf901
#define dmdavecrestorearrayreadf901_   dmdavecrestorearrayreadf901
#define dmdavecgetarrayreadf902_       dmdavecgetarrayreadf902
#define dmdavecrestorearrayreadf902_   dmdavecrestorearrayreadf902
#define dmdavecgetarrayreadf903_       dmdavecgetarrayreadf903
#define dmdavecrestorearrayreadf903_   dmdavecrestorearrayreadf903
#define dmdavecgetarrayreadf904_       dmdavecgetarrayreadf904
#define dmdavecrestorearrayreadf904_   dmdavecrestorearrayreadf904
#endif

PETSC_EXTERN void dmdagetlocalinfof90_(DM *da,DMDALocalInfo *info,PetscErrorCode *ierr)
{
  *ierr = DMDAGetLocalInfo(*da,info);
}

PETSC_EXTERN void dmdavecgetarrayf901_(DM *da,Vec *v,F90Array1d *a,PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(ptrd))
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
    *ierr = PETSC_ERR_ARG_INCOMP; return;
  }
  *ierr = VecGetArray(*v,&aa);if (*ierr) return;
  *ierr = F90Array1dCreate(aa,MPIU_SCALAR,gxs,gxm,a PETSC_F90_2PTR_PARAM(ptrd));if (*ierr) return;
}

PETSC_EXTERN void dmdavecrestorearrayf901_(DM *da,Vec *v,F90Array1d *a,PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscScalar *fa;
  *ierr = F90Array1dAccess(a,MPIU_SCALAR,(void**)&fa PETSC_F90_2PTR_PARAM(ptrd));
  *ierr = VecRestoreArray(*v,&fa);if (*ierr) return;
  *ierr = F90Array1dDestroy(a,MPIU_SCALAR PETSC_F90_2PTR_PARAM(ptrd));
}

PETSC_EXTERN void dmdavecgetarrayf902_(DM *da,Vec *v,F90Array2d *a,PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(ptrd))
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
    *ierr = PETSC_ERR_ARG_INCOMP; return;
  }
  if (dim == 1) {
    gys = gxs;
    gym = gxm;
    gxs = 0;
    gxm = dof;
  }
  *ierr = VecGetArray(*v,&aa);if (*ierr) return;
  *ierr = F90Array2dCreate(aa,MPIU_SCALAR,gxs,gxm,gys,gym,a PETSC_F90_2PTR_PARAM(ptrd));if (*ierr) return;
}

PETSC_EXTERN void dmdavecrestorearrayf902_(DM *da,Vec *v,F90Array2d *a,PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscScalar *fa;
  *ierr = F90Array2dAccess(a,MPIU_SCALAR,(void**)&fa PETSC_F90_2PTR_PARAM(ptrd));
  *ierr = VecRestoreArray(*v,&fa);if (*ierr) return;
  *ierr = F90Array2dDestroy(a,MPIU_SCALAR PETSC_F90_2PTR_PARAM(ptrd));
}

PETSC_EXTERN void dmdavecgetarrayf903_(DM *da,Vec *v,F90Array3d *a,PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(ptrd))
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
    *ierr = PETSC_ERR_ARG_INCOMP; return;
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
  *ierr = F90Array3dCreate(aa,MPIU_SCALAR,gxs,gxm,gys,gym,gzs,gzm,a PETSC_F90_2PTR_PARAM(ptrd));if (*ierr) return;
}

PETSC_EXTERN void dmdavecrestorearrayf903_(DM *da,Vec *v,F90Array3d *a,PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscScalar *fa;
  *ierr = F90Array3dAccess(a,MPIU_SCALAR,(void**)&fa PETSC_F90_2PTR_PARAM(ptrd));
  *ierr = VecRestoreArray(*v,&fa);if (*ierr) return;
  *ierr = F90Array3dDestroy(a,MPIU_SCALAR PETSC_F90_2PTR_PARAM(ptrd));
}

PETSC_EXTERN void dmdavecgetarrayf904_(DM *da,Vec *v,F90Array4d *a,PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(ptrd))
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
    *ierr = PETSC_ERR_ARG_INCOMP; return;
  }
  *ierr = VecGetArray(*v,&aa);if (*ierr) return;
  *ierr = F90Array4dCreate(aa,MPIU_SCALAR,zero,dof,gxs,gxm,gys,gym,gzs,gzm,a PETSC_F90_2PTR_PARAM(ptrd));if (*ierr) return;
}

PETSC_EXTERN void dmdavecrestorearrayf904_(DM *da,Vec *v,F90Array4d *a,PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscScalar *fa;
  /*
    F90Array4dAccess is not implemented, so the following call would fail
  */
  *ierr = F90Array4dAccess(a,MPIU_SCALAR,(void**)&fa PETSC_F90_2PTR_PARAM(ptrd));
  *ierr = VecRestoreArray(*v,&fa);if (*ierr) return;
  *ierr = F90Array4dDestroy(a,MPIU_SCALAR PETSC_F90_2PTR_PARAM(ptrd));
}

PETSC_EXTERN void dmdavecgetarrayreadf901_(DM *da,Vec *v,F90Array1d *a,PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscInt          xs,ys,zs,xm,ym,zm,gxs,gys,gzs,gxm,gym,gzm,N,dim,dof;
  const PetscScalar *aa;

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
    *ierr = PETSC_ERR_ARG_INCOMP; return;
  }
  *ierr = VecGetArrayRead(*v,&aa);if (*ierr) return;
  *ierr = F90Array1dCreate((void*)aa,MPIU_SCALAR,gxs,gxm,a PETSC_F90_2PTR_PARAM(ptrd));if (*ierr) return;
}

PETSC_EXTERN void dmdavecrestorearrayreadf901_(DM *da,Vec *v,F90Array1d *a,PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  const PetscScalar *fa;
  *ierr = F90Array1dAccess(a,MPIU_SCALAR,(void**)&fa PETSC_F90_2PTR_PARAM(ptrd));
  *ierr = VecRestoreArrayRead(*v,&fa);if (*ierr) return;
  *ierr = F90Array1dDestroy(a,MPIU_SCALAR PETSC_F90_2PTR_PARAM(ptrd));
}

PETSC_EXTERN void dmdavecgetarrayreadf902_(DM *da,Vec *v,F90Array2d *a,PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscInt          xs,ys,zs,xm,ym,zm,gxs,gys,gzs,gxm,gym,gzm,N,dim,dof;
  const PetscScalar *aa;

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
    *ierr = PETSC_ERR_ARG_INCOMP; return;
  }
  if (dim == 1) {
    gys = gxs;
    gym = gxm;
    gxs = 0;
    gxm = dof;
  }
  *ierr = VecGetArrayRead(*v,&aa);if (*ierr) return;
  *ierr = F90Array2dCreate((void*)aa,MPIU_SCALAR,gxs,gxm,gys,gym,a PETSC_F90_2PTR_PARAM(ptrd));if (*ierr) return;
}

PETSC_EXTERN void dmdavecrestorearrayreadf902_(DM *da,Vec *v,F90Array2d *a,PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  const PetscScalar *fa;
  *ierr = F90Array2dAccess(a,MPIU_SCALAR,(void**)&fa PETSC_F90_2PTR_PARAM(ptrd));
  *ierr = VecRestoreArrayRead(*v,&fa);if (*ierr) return;
  *ierr = F90Array2dDestroy(a,MPIU_SCALAR PETSC_F90_2PTR_PARAM(ptrd));
}

PETSC_EXTERN void dmdavecgetarrayreadf903_(DM *da,Vec *v,F90Array3d *a,PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscInt          xs,ys,zs,xm,ym,zm,gxs,gys,gzs,gxm,gym,gzm,N,dim,dof;
  const PetscScalar *aa;

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
    *ierr = PETSC_ERR_ARG_INCOMP; return;
  }
  if (dim == 2) {
    gzs = gys;
    gzm = gym;
    gys = gxs;
    gym = gxm;
    gxs = 0;
    gxm = dof;
  }
  *ierr = VecGetArrayRead(*v,&aa);if (*ierr) return;
  *ierr = F90Array3dCreate((void*)aa,MPIU_SCALAR,gxs,gxm,gys,gym,gzs,gzm,a PETSC_F90_2PTR_PARAM(ptrd));if (*ierr) return;
}

PETSC_EXTERN void dmdavecrestorearrayreadf903_(DM *da,Vec *v,F90Array3d *a,PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  const PetscScalar *fa;
  *ierr = F90Array3dAccess(a,MPIU_SCALAR,(void**)&fa PETSC_F90_2PTR_PARAM(ptrd));
  *ierr = VecRestoreArrayRead(*v,&fa);if (*ierr) return;
  *ierr = F90Array3dDestroy(a,MPIU_SCALAR PETSC_F90_2PTR_PARAM(ptrd));
}

PETSC_EXTERN void dmdavecgetarrayreadf904_(DM *da,Vec *v,F90Array4d *a,PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscInt          xs,ys,zs,xm,ym,zm,gxs,gys,gzs,gxm,gym,gzm,N,dim,dof,zero = 0;
  const PetscScalar *aa;

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
    *ierr = PETSC_ERR_ARG_INCOMP; return;
  }
  *ierr = VecGetArrayRead(*v,&aa);if (*ierr) return;
  *ierr = F90Array4dCreate((void*)aa,MPIU_SCALAR,zero,dof,gxs,gxm,gys,gym,gzs,gzm,a PETSC_F90_2PTR_PARAM(ptrd));if (*ierr) return;
}

PETSC_EXTERN void dmdavecrestorearrayreadf904_(DM *da,Vec *v,F90Array4d *a,PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  const PetscScalar *fa;
  /*
    F90Array4dAccess is not implemented, so the following call would fail
  */
  *ierr = F90Array4dAccess(a,MPIU_SCALAR,(void**)&fa PETSC_F90_2PTR_PARAM(ptrd));
  *ierr = VecRestoreArrayRead(*v,&fa);if (*ierr) return;
  *ierr = F90Array4dDestroy(a,MPIU_SCALAR PETSC_F90_2PTR_PARAM(ptrd));
}
