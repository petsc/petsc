#include <petsc/private/vecimpl_kokkos.hpp>
#include <petsc/private/dmdaimpl.h>
#include <petscdmda_kokkos.hpp>

/* Use macro instead of inlined function just to avoid annoying warnings like: 'dof' may be used uninitialized in this function [-Wmaybe-uninitialized] */
#define DMDA_VEC_GET_SHAPE(da,vec,xs,ys,zs,xm,ym,zm,gxs,gys,gzs,gxm,gym,gzm,N,dim,dof) \
do { \
  PetscErrorCode ierr; \
  ierr = DMDAGetCorners(da,&xs,&ys,&zs,&xm,&ym,&zm);CHKERRQ(ierr); \
  ierr = DMDAGetGhostCorners(da,&gxs,&gys,&gzs,&gxm,&gym,&gzm);CHKERRQ(ierr); \
  ierr = DMDAGetInfo(da,&dim,NULL,NULL,NULL,NULL,NULL,NULL,&dof,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr); \
  /* Handle case where user passes in global vector as opposed to local */ \
  ierr = VecGetLocalSize(vec,&N);CHKERRQ(ierr); \
  if (N == xm*ym*zm*dof) { \
    gxm = xm; gym = ym; gzm = zm; \
    gxs = xs; gys = ys; gzs = zs; \
  } else if (N != gxm*gym*gzm*dof) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Vector local size %D is not compatible with DMDA local sizes %D %D",N,xm*ym*zm*dof,gxm*gym*gzm*dof); \
  if (dim != 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,"KokkosOffsetView is 1D but DMDA is %DD",dim); \
} while (0)

template<class MemorySpace>
PetscErrorCode DMDAVecGetKokkosOffsetView_Private(DM da,Vec vec,PetscScalarKokkosOffsetView1DType<MemorySpace> *ov,PetscBool overwrite)
{
  PetscErrorCode                               ierr;
  PetscInt                                     xs,ys,zs,xm,ym,zm,gxs,gys,gzs,gxm,gym,gzm,N,dim,dof;
  PetscScalarKokkosViewType<MemorySpace>       kv;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(da,DM_CLASSID,1,DMDA);
  PetscValidHeaderSpecific(vec,VEC_CLASSID,2);
  PetscValidPointer(ov,3);
  DMDA_VEC_GET_SHAPE(da,vec,xs,ys,zs,xm,ym,zm,gxs,gys,gzs,gxm,gym,gzm,N,dim,dof);
  if (overwrite) {ierr = VecGetKokkosViewWrite(vec,&kv);CHKERRQ(ierr);}
  else {ierr = VecGetKokkosView(vec,&kv);CHKERRQ(ierr);}
  *ov  = PetscScalarKokkosOffsetView1DType<MemorySpace>(kv,{gxs*dof}); /* View to OffsetView by giving the start. The extent is already known. */
  PetscFunctionReturn(0);
}

template<class MemorySpace>
PetscErrorCode DMDAVecRestoreKokkosOffsetView_Private(DM da,Vec vec,PetscScalarKokkosOffsetView1DType<MemorySpace> *ov,PetscBool overwrite)
{
  PetscErrorCode                               ierr;
  PetscScalarKokkosViewType<MemorySpace>       kv;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(da,DM_CLASSID,1,DMDA);
  PetscValidHeaderSpecific(vec,VEC_CLASSID,2);
  PetscValidPointer(ov,3);
  kv   = ov->view(); /* OffsetView to View */
  if (overwrite) {ierr = VecRestoreKokkosViewWrite(vec,&kv);CHKERRQ(ierr);}
  else {ierr = VecRestoreKokkosView(vec,&kv);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

template<class MemorySpace>
PetscErrorCode DMDAVecGetKokkosOffsetView(DM da,Vec vec,ConstPetscScalarKokkosOffsetView1DType<MemorySpace> *ov)
{
  PetscErrorCode                               ierr;
  PetscInt                                     xs,ys,zs,xm,ym,zm,gxs,gys,gzs,gxm,gym,gzm,N,dim,dof;
  ConstPetscScalarKokkosViewType<MemorySpace>  kv;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(da,DM_CLASSID,1,DMDA);
  PetscValidHeaderSpecific(vec,VEC_CLASSID,2);
  PetscValidPointer(ov,3);
  DMDA_VEC_GET_SHAPE(da,vec,xs,ys,zs,xm,ym,zm,gxs,gys,gzs,gxm,gym,gzm,N,dim,dof);
  ierr = VecGetKokkosView(vec,&kv);CHKERRQ(ierr);
  *ov  = ConstPetscScalarKokkosOffsetView1DType<MemorySpace>(kv,{gxs*dof}); /* View to OffsetView */
  PetscFunctionReturn(0);
}

template<class MemorySpace>
PetscErrorCode DMDAVecRestoreKokkosOffsetView(DM da,Vec vec,ConstPetscScalarKokkosOffsetView1DType<MemorySpace> *ov)
{
  PetscErrorCode                               ierr;
  ConstPetscScalarKokkosViewType<MemorySpace>  kv;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(da,DM_CLASSID,1,DMDA);
  PetscValidHeaderSpecific(vec,VEC_CLASSID,2);
  PetscValidPointer(ov,3);
  kv   = ov->view();
  ierr = VecRestoreKokkosView(vec,&kv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

template<class MemorySpace>
PetscErrorCode DMDAVecGetKokkosOffsetView_Private(DM da,Vec vec,PetscScalarKokkosOffsetView2DType<MemorySpace> *ov,PetscBool overwrite)
{
  PetscErrorCode                               ierr;
  PetscInt                                     xs,ys,zs,xm,ym,zm,gxs,gys,gzs,gxm,gym,gzm,N,dim,dof;
  PetscScalarKokkosViewType<MemorySpace>       kv;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(da,DM_CLASSID,1,DMDA);
  PetscValidHeaderSpecific(vec,VEC_CLASSID,2);
  PetscValidPointer(ov,3);
  DMDA_VEC_GET_SHAPE(da,vec,xs,ys,zs,xm,ym,zm,gxs,gys,gzs,gxm,gym,gzm,N,dim,dof);
  if (overwrite) {ierr = VecGetKokkosViewWrite(vec,&kv);CHKERRQ(ierr);}
  else {ierr = VecGetKokkosView(vec,&kv);CHKERRQ(ierr);}
  *ov  = PetscScalarKokkosOffsetView2DType<MemorySpace>(kv.data(), {gxs*dof,(gxs+gxm)*dof}, {gys*dof,(gys+gym)*dof}); /* View to OffsetView */
  PetscFunctionReturn(0);
}

template<class MemorySpace>
PetscErrorCode DMDAVecRestoreKokkosOffsetView_Private(DM da,Vec vec,PetscScalarKokkosOffsetView2DType<MemorySpace> *ov,PetscBool overwrite)
{
  PetscErrorCode                             ierr;
  PetscScalarKokkosViewType<MemorySpace>     kv;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(da,DM_CLASSID,1,DMDA);
  PetscValidHeaderSpecific(vec,VEC_CLASSID,2);
  PetscValidPointer(ov,3);
  // kv   = ov->view(); /* 2D OffsetView => 2D View => 1D View. Why does it not work? */
  kv   = PetscScalarKokkosViewType<MemorySpace>(ov->data(),ov->extent(0)*ov->extent(1));
  if (overwrite) {ierr = VecRestoreKokkosViewWrite(vec,&kv);CHKERRQ(ierr);}
  else {ierr = VecRestoreKokkosView(vec,&kv);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

template<class MemorySpace>
PetscErrorCode DMDAVecGetKokkosOffsetView(DM da,Vec vec,ConstPetscScalarKokkosOffsetView2DType<MemorySpace> *ov)
{
  PetscErrorCode                               ierr;
  PetscInt                                     xs,ys,zs,xm,ym,zm,gxs,gys,gzs,gxm,gym,gzm,N,dim,dof;
  ConstPetscScalarKokkosViewType<MemorySpace>  kv;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(da,DM_CLASSID,1,DMDA);
  PetscValidHeaderSpecific(vec,VEC_CLASSID,2);
  PetscValidPointer(ov,3);
  DMDA_VEC_GET_SHAPE(da,vec,xs,ys,zs,xm,ym,zm,gxs,gys,gzs,gxm,gym,gzm,N,dim,dof);
  ierr = VecGetKokkosView(vec,&kv);CHKERRQ(ierr);
  *ov  = ConstPetscScalarKokkosOffsetView2DType<MemorySpace>(kv.data(), {gxs*dof,(gxs+gxm)*dof}, {gys*dof,(gys+gym)*dof}); /* View to OffsetView */
  PetscFunctionReturn(0);
}

template<class MemorySpace>
PetscErrorCode DMDAVecRestoreKokkosOffsetView(DM da,Vec vec,ConstPetscScalarKokkosOffsetView2DType<MemorySpace> *ov)
{
  PetscErrorCode                               ierr;
  ConstPetscScalarKokkosViewType<MemorySpace>  kv;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(da,DM_CLASSID,1,DMDA);
  PetscValidHeaderSpecific(vec,VEC_CLASSID,2);
  PetscValidPointer(ov,3);
  kv   = ConstPetscScalarKokkosViewType<MemorySpace>(ov->data(),ov->extent(0)*ov->extent(1));
  ierr = VecRestoreKokkosView(vec,&kv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Function template explicit instantiation */
template   PETSC_VISIBILITY_PUBLIC PetscErrorCode DMDAVecGetKokkosOffsetView         (DM,Vec,ConstPetscScalarKokkosOffsetView1D*);
template   PETSC_VISIBILITY_PUBLIC PetscErrorCode DMDAVecRestoreKokkosOffsetView     (DM,Vec,ConstPetscScalarKokkosOffsetView1D*);
template<> PETSC_VISIBILITY_PUBLIC PetscErrorCode DMDAVecGetKokkosOffsetView         (DM da,Vec vec,PetscScalarKokkosOffsetView1D* ov) {return DMDAVecGetKokkosOffsetView_Private(da,vec,ov,PETSC_FALSE);}
template<> PETSC_VISIBILITY_PUBLIC PetscErrorCode DMDAVecRestoreKokkosOffsetView     (DM da,Vec vec,PetscScalarKokkosOffsetView1D* ov) {return DMDAVecRestoreKokkosOffsetView_Private(da,vec,ov,PETSC_FALSE);}
template<> PETSC_VISIBILITY_PUBLIC PetscErrorCode DMDAVecGetKokkosOffsetViewWrite    (DM da,Vec vec,PetscScalarKokkosOffsetView1D* ov) {return DMDAVecGetKokkosOffsetView_Private(da,vec,ov,PETSC_TRUE);}
template<> PETSC_VISIBILITY_PUBLIC PetscErrorCode DMDAVecRestoreKokkosOffsetViewWrite(DM da,Vec vec,PetscScalarKokkosOffsetView1D* ov) {return DMDAVecRestoreKokkosOffsetView_Private(da,vec,ov,PETSC_TRUE);}

template   PETSC_VISIBILITY_PUBLIC PetscErrorCode DMDAVecGetKokkosOffsetView         (DM,Vec,ConstPetscScalarKokkosOffsetView2D*);
template   PETSC_VISIBILITY_PUBLIC PetscErrorCode DMDAVecRestoreKokkosOffsetView     (DM,Vec,ConstPetscScalarKokkosOffsetView2D*);
template<> PETSC_VISIBILITY_PUBLIC PetscErrorCode DMDAVecGetKokkosOffsetView         (DM da,Vec vec,PetscScalarKokkosOffsetView2D* ov) {return DMDAVecGetKokkosOffsetView_Private(da,vec,ov,PETSC_FALSE);}
template<> PETSC_VISIBILITY_PUBLIC PetscErrorCode DMDAVecRestoreKokkosOffsetView     (DM da,Vec vec,PetscScalarKokkosOffsetView2D* ov) {return DMDAVecRestoreKokkosOffsetView_Private(da,vec,ov,PETSC_FALSE);}
template<> PETSC_VISIBILITY_PUBLIC PetscErrorCode DMDAVecGetKokkosOffsetViewWrite    (DM da,Vec vec,PetscScalarKokkosOffsetView2D* ov) {return DMDAVecGetKokkosOffsetView_Private(da,vec,ov,PETSC_TRUE);}
template<> PETSC_VISIBILITY_PUBLIC PetscErrorCode DMDAVecRestoreKokkosOffsetViewWrite(DM da,Vec vec,PetscScalarKokkosOffsetView2D* ov) {return DMDAVecRestoreKokkosOffsetView_Private(da,vec,ov,PETSC_TRUE);}

