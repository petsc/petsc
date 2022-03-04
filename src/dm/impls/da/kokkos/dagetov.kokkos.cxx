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
  } else PetscCheck(N == gxm*gym*gzm*dof,PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Vector local size %D is not compatible with DMDA local sizes %D %D",N,xm*ym*zm*dof,gxm*gym*gzm*dof); \
} while (0)

/* -------------------- 1D ---------------- */
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
  PetscCheck(dim == 1,PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,"KokkosOffsetView is 1D but DMDA is %dD",(int)dim);
  if (overwrite) {ierr = VecGetKokkosViewWrite(vec,&kv);CHKERRQ(ierr);}
  else {ierr = VecGetKokkosView(vec,&kv);CHKERRQ(ierr);}
  /* Construct the unmanaged OffsetView with {begin0,begin1,begins2},{end0,end1,end2} */
  *ov  = PetscScalarKokkosOffsetView1DType<MemorySpace>(kv.data(),{gxs*dof},{(gxs+gxm)*dof});
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
  PetscCheck(dim == 1,PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,"KokkosOffsetView is 1D but DMDA is %dD",(int)dim);
  ierr = VecGetKokkosView(vec,&kv);CHKERRQ(ierr);
  *ov  = ConstPetscScalarKokkosOffsetView1DType<MemorySpace>(kv.data(),{gxs*dof},{(gxs+gxm)*dof});
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

/* ============================== 2D ================================= */
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
  PetscCheck(dim == 2,PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,"KokkosOffsetView is 2D but DMDA is %dD",(int)dim);
  if (overwrite) {ierr = VecGetKokkosViewWrite(vec,&kv);CHKERRQ(ierr);}
  else {ierr = VecGetKokkosView(vec,&kv);CHKERRQ(ierr);}
  *ov  = PetscScalarKokkosOffsetView2DType<MemorySpace>(kv.data(),{gys*dof,gxs*dof},{(gys+gym)*dof,(gxs+gxm)*dof});
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
  PetscCheck(dim == 2,PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,"KokkosOffsetView is 2D but DMDA is %dD",(int)dim);
  ierr = VecGetKokkosView(vec,&kv);CHKERRQ(ierr);
  *ov  = ConstPetscScalarKokkosOffsetView2DType<MemorySpace>(kv.data(),{gys*dof,gxs*dof},{(gys+gym)*dof,(gxs+gxm)*dof});
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

/* ============================== 3D ================================= */
template<class MemorySpace>
PetscErrorCode DMDAVecGetKokkosOffsetView_Private(DM da,Vec vec,PetscScalarKokkosOffsetView3DType<MemorySpace> *ov,PetscBool overwrite)
{
  PetscErrorCode                               ierr;
  PetscInt                                     xs,ys,zs,xm,ym,zm,gxs,gys,gzs,gxm,gym,gzm,N,dim,dof;
  PetscScalarKokkosViewType<MemorySpace>       kv;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(da,DM_CLASSID,1,DMDA);
  PetscValidHeaderSpecific(vec,VEC_CLASSID,2);
  PetscValidPointer(ov,3);
  DMDA_VEC_GET_SHAPE(da,vec,xs,ys,zs,xm,ym,zm,gxs,gys,gzs,gxm,gym,gzm,N,dim,dof);
  PetscCheck(dim == 3,PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,"KokkosOffsetView is 3D but DMDA is %dD",(int)dim);
  if (overwrite) {ierr = VecGetKokkosViewWrite(vec,&kv);CHKERRQ(ierr);}
  else {ierr = VecGetKokkosView(vec,&kv);CHKERRQ(ierr);}
  *ov  = PetscScalarKokkosOffsetView3DType<MemorySpace>(kv.data(),{gzs*dof,gys*dof,gxs*dof},{(gzs+gzm)*dof,(gys+gym)*dof,(gxs+gxm)*dof});
  PetscFunctionReturn(0);
}

template<class MemorySpace>
PetscErrorCode DMDAVecRestoreKokkosOffsetView_Private(DM da,Vec vec,PetscScalarKokkosOffsetView3DType<MemorySpace> *ov,PetscBool overwrite)
{
  PetscErrorCode                             ierr;
  PetscScalarKokkosViewType<MemorySpace>     kv;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(da,DM_CLASSID,1,DMDA);
  PetscValidHeaderSpecific(vec,VEC_CLASSID,2);
  PetscValidPointer(ov,3);
  kv   = PetscScalarKokkosViewType<MemorySpace>(ov->data(),ov->extent(0)*ov->extent(1)*ov->extent(2));
  if (overwrite) {ierr = VecRestoreKokkosViewWrite(vec,&kv);CHKERRQ(ierr);}
  else {ierr = VecRestoreKokkosView(vec,&kv);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

template<class MemorySpace>
PetscErrorCode DMDAVecGetKokkosOffsetView(DM da,Vec vec,ConstPetscScalarKokkosOffsetView3DType<MemorySpace> *ov)
{
  PetscErrorCode                               ierr;
  PetscInt                                     xs,ys,zs,xm,ym,zm,gxs,gys,gzs,gxm,gym,gzm,N,dim,dof;
  ConstPetscScalarKokkosViewType<MemorySpace>  kv;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(da,DM_CLASSID,1,DMDA);
  PetscValidHeaderSpecific(vec,VEC_CLASSID,2);
  PetscValidPointer(ov,3);
  DMDA_VEC_GET_SHAPE(da,vec,xs,ys,zs,xm,ym,zm,gxs,gys,gzs,gxm,gym,gzm,N,dim,dof);
  PetscCheck(dim == 3,PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,"KokkosOffsetView is 3D but DMDA is %dD",(int)dim);
  ierr = VecGetKokkosView(vec,&kv);CHKERRQ(ierr);
  *ov  = ConstPetscScalarKokkosOffsetView3DType<MemorySpace>(kv.data(),{gzs*dof,gys*dof,gxs*dof},{(gzs+gzm)*dof,(gys+gym)*dof,(gxs+gxm)*dof});
  PetscFunctionReturn(0);
}

template<class MemorySpace>
PetscErrorCode DMDAVecRestoreKokkosOffsetView(DM da,Vec vec,ConstPetscScalarKokkosOffsetView3DType<MemorySpace> *ov)
{
  PetscErrorCode                               ierr;
  ConstPetscScalarKokkosViewType<MemorySpace>  kv;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(da,DM_CLASSID,1,DMDA);
  PetscValidHeaderSpecific(vec,VEC_CLASSID,2);
  PetscValidPointer(ov,3);
  kv   = ConstPetscScalarKokkosViewType<MemorySpace>(ov->data(),ov->extent(0)*ov->extent(1)*ov->extent(2));
  ierr = VecRestoreKokkosView(vec,&kv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Function template explicit instantiation */
template   PETSC_VISIBILITY_PUBLIC PetscErrorCode DMDAVecGetKokkosOffsetView         (DM,Vec,  ConstPetscScalarKokkosOffsetView1D*);
template   PETSC_VISIBILITY_PUBLIC PetscErrorCode DMDAVecRestoreKokkosOffsetView     (DM,Vec,  ConstPetscScalarKokkosOffsetView1D*);
template<> PETSC_VISIBILITY_PUBLIC PetscErrorCode DMDAVecGetKokkosOffsetView         (DM da,Vec vec,PetscScalarKokkosOffsetView1D* ov) {return DMDAVecGetKokkosOffsetView_Private(da,vec,ov,PETSC_FALSE);}
template<> PETSC_VISIBILITY_PUBLIC PetscErrorCode DMDAVecRestoreKokkosOffsetView     (DM da,Vec vec,PetscScalarKokkosOffsetView1D* ov) {return DMDAVecRestoreKokkosOffsetView_Private(da,vec,ov,PETSC_FALSE);}
template<> PETSC_VISIBILITY_PUBLIC PetscErrorCode DMDAVecGetKokkosOffsetViewWrite    (DM da,Vec vec,PetscScalarKokkosOffsetView1D* ov) {return DMDAVecGetKokkosOffsetView_Private(da,vec,ov,PETSC_TRUE);}
template<> PETSC_VISIBILITY_PUBLIC PetscErrorCode DMDAVecRestoreKokkosOffsetViewWrite(DM da,Vec vec,PetscScalarKokkosOffsetView1D* ov) {return DMDAVecRestoreKokkosOffsetView_Private(da,vec,ov,PETSC_TRUE);}

template   PETSC_VISIBILITY_PUBLIC PetscErrorCode DMDAVecGetKokkosOffsetView         (DM,Vec,  ConstPetscScalarKokkosOffsetView2D*);
template   PETSC_VISIBILITY_PUBLIC PetscErrorCode DMDAVecRestoreKokkosOffsetView     (DM,Vec,  ConstPetscScalarKokkosOffsetView2D*);
template<> PETSC_VISIBILITY_PUBLIC PetscErrorCode DMDAVecGetKokkosOffsetView         (DM da,Vec vec,PetscScalarKokkosOffsetView2D* ov) {return DMDAVecGetKokkosOffsetView_Private(da,vec,ov,PETSC_FALSE);}
template<> PETSC_VISIBILITY_PUBLIC PetscErrorCode DMDAVecRestoreKokkosOffsetView     (DM da,Vec vec,PetscScalarKokkosOffsetView2D* ov) {return DMDAVecRestoreKokkosOffsetView_Private(da,vec,ov,PETSC_FALSE);}
template<> PETSC_VISIBILITY_PUBLIC PetscErrorCode DMDAVecGetKokkosOffsetViewWrite    (DM da,Vec vec,PetscScalarKokkosOffsetView2D* ov) {return DMDAVecGetKokkosOffsetView_Private(da,vec,ov,PETSC_TRUE);}
template<> PETSC_VISIBILITY_PUBLIC PetscErrorCode DMDAVecRestoreKokkosOffsetViewWrite(DM da,Vec vec,PetscScalarKokkosOffsetView2D* ov) {return DMDAVecRestoreKokkosOffsetView_Private(da,vec,ov,PETSC_TRUE);}

template   PETSC_VISIBILITY_PUBLIC PetscErrorCode DMDAVecGetKokkosOffsetView         (DM,Vec,  ConstPetscScalarKokkosOffsetView3D*);
template   PETSC_VISIBILITY_PUBLIC PetscErrorCode DMDAVecRestoreKokkosOffsetView     (DM,Vec,  ConstPetscScalarKokkosOffsetView3D*);
template<> PETSC_VISIBILITY_PUBLIC PetscErrorCode DMDAVecGetKokkosOffsetView         (DM da,Vec vec,PetscScalarKokkosOffsetView3D* ov) {return DMDAVecGetKokkosOffsetView_Private(da,vec,ov,PETSC_FALSE);}
template<> PETSC_VISIBILITY_PUBLIC PetscErrorCode DMDAVecRestoreKokkosOffsetView     (DM da,Vec vec,PetscScalarKokkosOffsetView3D* ov) {return DMDAVecRestoreKokkosOffsetView_Private(da,vec,ov,PETSC_FALSE);}
template<> PETSC_VISIBILITY_PUBLIC PetscErrorCode DMDAVecGetKokkosOffsetViewWrite    (DM da,Vec vec,PetscScalarKokkosOffsetView3D* ov) {return DMDAVecGetKokkosOffsetView_Private(da,vec,ov,PETSC_TRUE);}
template<> PETSC_VISIBILITY_PUBLIC PetscErrorCode DMDAVecRestoreKokkosOffsetViewWrite(DM da,Vec vec,PetscScalarKokkosOffsetView3D* ov) {return DMDAVecRestoreKokkosOffsetView_Private(da,vec,ov,PETSC_TRUE);}

#if !defined(KOKKOS_ENABLE_DEFAULT_DEVICE_TYPE_HOST) /* Get host views if the default memory space is not host space */
template   PETSC_VISIBILITY_PUBLIC PetscErrorCode DMDAVecGetKokkosOffsetView         (DM,Vec,  ConstPetscScalarKokkosOffsetView1DHost*);
template   PETSC_VISIBILITY_PUBLIC PetscErrorCode DMDAVecRestoreKokkosOffsetView     (DM,Vec,  ConstPetscScalarKokkosOffsetView1DHost*);
template<> PETSC_VISIBILITY_PUBLIC PetscErrorCode DMDAVecGetKokkosOffsetView         (DM da,Vec vec,PetscScalarKokkosOffsetView1DHost* ov) {return DMDAVecGetKokkosOffsetView_Private(da,vec,ov,PETSC_FALSE);}
template<> PETSC_VISIBILITY_PUBLIC PetscErrorCode DMDAVecRestoreKokkosOffsetView     (DM da,Vec vec,PetscScalarKokkosOffsetView1DHost* ov) {return DMDAVecRestoreKokkosOffsetView_Private(da,vec,ov,PETSC_FALSE);}
template<> PETSC_VISIBILITY_PUBLIC PetscErrorCode DMDAVecGetKokkosOffsetViewWrite    (DM da,Vec vec,PetscScalarKokkosOffsetView1DHost* ov) {return DMDAVecGetKokkosOffsetView_Private(da,vec,ov,PETSC_TRUE);}
template<> PETSC_VISIBILITY_PUBLIC PetscErrorCode DMDAVecRestoreKokkosOffsetViewWrite(DM da,Vec vec,PetscScalarKokkosOffsetView1DHost* ov) {return DMDAVecRestoreKokkosOffsetView_Private(da,vec,ov,PETSC_TRUE);}

template   PETSC_VISIBILITY_PUBLIC PetscErrorCode DMDAVecGetKokkosOffsetView         (DM,Vec,  ConstPetscScalarKokkosOffsetView2DHost*);
template   PETSC_VISIBILITY_PUBLIC PetscErrorCode DMDAVecRestoreKokkosOffsetView     (DM,Vec,  ConstPetscScalarKokkosOffsetView2DHost*);
template<> PETSC_VISIBILITY_PUBLIC PetscErrorCode DMDAVecGetKokkosOffsetView         (DM da,Vec vec,PetscScalarKokkosOffsetView2DHost* ov) {return DMDAVecGetKokkosOffsetView_Private(da,vec,ov,PETSC_FALSE);}
template<> PETSC_VISIBILITY_PUBLIC PetscErrorCode DMDAVecRestoreKokkosOffsetView     (DM da,Vec vec,PetscScalarKokkosOffsetView2DHost* ov) {return DMDAVecRestoreKokkosOffsetView_Private(da,vec,ov,PETSC_FALSE);}
template<> PETSC_VISIBILITY_PUBLIC PetscErrorCode DMDAVecGetKokkosOffsetViewWrite    (DM da,Vec vec,PetscScalarKokkosOffsetView2DHost* ov) {return DMDAVecGetKokkosOffsetView_Private(da,vec,ov,PETSC_TRUE);}
template<> PETSC_VISIBILITY_PUBLIC PetscErrorCode DMDAVecRestoreKokkosOffsetViewWrite(DM da,Vec vec,PetscScalarKokkosOffsetView2DHost* ov) {return DMDAVecRestoreKokkosOffsetView_Private(da,vec,ov,PETSC_TRUE);}

template   PETSC_VISIBILITY_PUBLIC PetscErrorCode DMDAVecGetKokkosOffsetView         (DM,Vec,  ConstPetscScalarKokkosOffsetView3DHost*);
template   PETSC_VISIBILITY_PUBLIC PetscErrorCode DMDAVecRestoreKokkosOffsetView     (DM,Vec,  ConstPetscScalarKokkosOffsetView3DHost*);
template<> PETSC_VISIBILITY_PUBLIC PetscErrorCode DMDAVecGetKokkosOffsetView         (DM da,Vec vec,PetscScalarKokkosOffsetView3DHost* ov) {return DMDAVecGetKokkosOffsetView_Private(da,vec,ov,PETSC_FALSE);}
template<> PETSC_VISIBILITY_PUBLIC PetscErrorCode DMDAVecRestoreKokkosOffsetView     (DM da,Vec vec,PetscScalarKokkosOffsetView3DHost* ov) {return DMDAVecRestoreKokkosOffsetView_Private(da,vec,ov,PETSC_FALSE);}
template<> PETSC_VISIBILITY_PUBLIC PetscErrorCode DMDAVecGetKokkosOffsetViewWrite    (DM da,Vec vec,PetscScalarKokkosOffsetView3DHost* ov) {return DMDAVecGetKokkosOffsetView_Private(da,vec,ov,PETSC_TRUE);}
template<> PETSC_VISIBILITY_PUBLIC PetscErrorCode DMDAVecRestoreKokkosOffsetViewWrite(DM da,Vec vec,PetscScalarKokkosOffsetView3DHost* ov) {return DMDAVecRestoreKokkosOffsetView_Private(da,vec,ov,PETSC_TRUE);}
#endif

/* ============================== 2D including DOF ================================= */
template<class MemorySpace>
PetscErrorCode DMDAVecGetKokkosOffsetViewDOF_Private(DM da,Vec vec,PetscScalarKokkosOffsetView2DType<MemorySpace> *ov,PetscBool overwrite)
{
  PetscErrorCode                               ierr;
  PetscInt                                     xs,ys,zs,xm,ym,zm,gxs,gys,gzs,gxm,gym,gzm,N,dim,dof;
  PetscScalarKokkosViewType<MemorySpace>       kv;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(da,DM_CLASSID,1,DMDA);
  PetscValidHeaderSpecific(vec,VEC_CLASSID,2);
  PetscValidPointer(ov,3);
  DMDA_VEC_GET_SHAPE(da,vec,xs,ys,zs,xm,ym,zm,gxs,gys,gzs,gxm,gym,gzm,N,dim,dof);
  PetscCheck(dim == 1,PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,"KokkosOffsetView is 2D but DMDA is %dD",(int)dim);
  if (overwrite) {ierr = VecGetKokkosViewWrite(vec,&kv);CHKERRQ(ierr);}
  else {ierr = VecGetKokkosView(vec,&kv);CHKERRQ(ierr);}
  *ov  = PetscScalarKokkosOffsetView2DType<MemorySpace>(kv.data(),{gxs,0},{gxs+gxm,dof});
  PetscFunctionReturn(0);
}

template<class MemorySpace>
PetscErrorCode DMDAVecRestoreKokkosOffsetViewDOF_Private(DM da,Vec vec,PetscScalarKokkosOffsetView2DType<MemorySpace> *ov,PetscBool overwrite)
{
  PetscErrorCode                             ierr;
  PetscScalarKokkosViewType<MemorySpace>     kv;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(da,DM_CLASSID,1,DMDA);
  PetscValidHeaderSpecific(vec,VEC_CLASSID,2);
  PetscValidPointer(ov,3);
  kv   = PetscScalarKokkosViewType<MemorySpace>(ov->data(),ov->extent(0)*ov->extent(1));
  if (overwrite) {ierr = VecRestoreKokkosViewWrite(vec,&kv);CHKERRQ(ierr);}
  else {ierr = VecRestoreKokkosView(vec,&kv);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

template<class MemorySpace>
PetscErrorCode DMDAVecGetKokkosOffsetViewDOF(DM da,Vec vec,ConstPetscScalarKokkosOffsetView2DType<MemorySpace> *ov)
{
  PetscErrorCode                               ierr;
  PetscInt                                     xs,ys,zs,xm,ym,zm,gxs,gys,gzs,gxm,gym,gzm,N,dim,dof;
  ConstPetscScalarKokkosViewType<MemorySpace>  kv;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(da,DM_CLASSID,1,DMDA);
  PetscValidHeaderSpecific(vec,VEC_CLASSID,2);
  PetscValidPointer(ov,3);
  DMDA_VEC_GET_SHAPE(da,vec,xs,ys,zs,xm,ym,zm,gxs,gys,gzs,gxm,gym,gzm,N,dim,dof);
  PetscCheck(dim == 1,PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,"KokkosOffsetView is 2D but DMDA is %dD",(int)dim);
  ierr = VecGetKokkosView(vec,&kv);CHKERRQ(ierr);
  *ov  = ConstPetscScalarKokkosOffsetView2DType<MemorySpace>(kv.data(),{gxs,0},{gxs+gxm,dof});
  PetscFunctionReturn(0);
}

template<class MemorySpace>
PetscErrorCode DMDAVecRestoreKokkosOffsetViewDOF(DM da,Vec vec,ConstPetscScalarKokkosOffsetView2DType<MemorySpace> *ov)
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

/* ============================== 3D including DOF ================================= */
template<class MemorySpace>
PetscErrorCode DMDAVecGetKokkosOffsetViewDOF_Private(DM da,Vec vec,PetscScalarKokkosOffsetView3DType<MemorySpace> *ov,PetscBool overwrite)
{
  PetscErrorCode                               ierr;
  PetscInt                                     xs,ys,zs,xm,ym,zm,gxs,gys,gzs,gxm,gym,gzm,N,dim,dof;
  PetscScalarKokkosViewType<MemorySpace>       kv;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(da,DM_CLASSID,1,DMDA);
  PetscValidHeaderSpecific(vec,VEC_CLASSID,2);
  PetscValidPointer(ov,3);
  DMDA_VEC_GET_SHAPE(da,vec,xs,ys,zs,xm,ym,zm,gxs,gys,gzs,gxm,gym,gzm,N,dim,dof);
  PetscCheck(dim == 2,PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,"KokkosOffsetView is 3D but DMDA is %dD",(int)dim);
  if (overwrite) {ierr = VecGetKokkosViewWrite(vec,&kv);CHKERRQ(ierr);}
  else {ierr = VecGetKokkosView(vec,&kv);CHKERRQ(ierr);}
  *ov  = PetscScalarKokkosOffsetView3DType<MemorySpace>(kv.data(),{gys,gxs,0},{gys+gym,gxs+gxm,dof});
  PetscFunctionReturn(0);
}

template<class MemorySpace>
PetscErrorCode DMDAVecRestoreKokkosOffsetViewDOF_Private(DM da,Vec vec,PetscScalarKokkosOffsetView3DType<MemorySpace> *ov,PetscBool overwrite)
{
  PetscErrorCode                             ierr;
  PetscScalarKokkosViewType<MemorySpace>     kv;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(da,DM_CLASSID,1,DMDA);
  PetscValidHeaderSpecific(vec,VEC_CLASSID,2);
  PetscValidPointer(ov,3);
  kv   = PetscScalarKokkosViewType<MemorySpace>(ov->data(),ov->extent(0)*ov->extent(1)*ov->extent(2));
  if (overwrite) {ierr = VecRestoreKokkosViewWrite(vec,&kv);CHKERRQ(ierr);}
  else {ierr = VecRestoreKokkosView(vec,&kv);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

template<class MemorySpace>
PetscErrorCode DMDAVecGetKokkosOffsetViewDOF(DM da,Vec vec,ConstPetscScalarKokkosOffsetView3DType<MemorySpace> *ov)
{
  PetscErrorCode                               ierr;
  PetscInt                                     xs,ys,zs,xm,ym,zm,gxs,gys,gzs,gxm,gym,gzm,N,dim,dof;
  ConstPetscScalarKokkosViewType<MemorySpace>  kv;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(da,DM_CLASSID,1,DMDA);
  PetscValidHeaderSpecific(vec,VEC_CLASSID,2);
  PetscValidPointer(ov,3);
  DMDA_VEC_GET_SHAPE(da,vec,xs,ys,zs,xm,ym,zm,gxs,gys,gzs,gxm,gym,gzm,N,dim,dof);
  PetscCheck(dim == 2,PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,"KokkosOffsetView is 3D but DMDA is %dD",(int)dim);
  ierr = VecGetKokkosView(vec,&kv);CHKERRQ(ierr);
  *ov  = ConstPetscScalarKokkosOffsetView3DType<MemorySpace>(kv.data(),{gys,gxs,0},{gys+gym,gxs+gxm,dof});
  PetscFunctionReturn(0);
}

template<class MemorySpace>
PetscErrorCode DMDAVecRestoreKokkosOffsetViewDOF(DM da,Vec vec,ConstPetscScalarKokkosOffsetView3DType<MemorySpace> *ov)
{
  PetscErrorCode                               ierr;
  ConstPetscScalarKokkosViewType<MemorySpace>  kv;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(da,DM_CLASSID,1,DMDA);
  PetscValidHeaderSpecific(vec,VEC_CLASSID,2);
  PetscValidPointer(ov,3);
  kv   = ConstPetscScalarKokkosViewType<MemorySpace>(ov->data(),ov->extent(0)*ov->extent(1)*ov->extent(2));
  ierr = VecRestoreKokkosView(vec,&kv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ============================== 4D including DOF ================================= */
template<class MemorySpace>
PetscErrorCode DMDAVecGetKokkosOffsetViewDOF_Private(DM da,Vec vec,PetscScalarKokkosOffsetView4DType<MemorySpace> *ov,PetscBool overwrite)
{
  PetscErrorCode                               ierr;
  PetscInt                                     xs,ys,zs,xm,ym,zm,gxs,gys,gzs,gxm,gym,gzm,N,dim,dof;
  PetscScalarKokkosViewType<MemorySpace>       kv;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(da,DM_CLASSID,1,DMDA);
  PetscValidHeaderSpecific(vec,VEC_CLASSID,2);
  PetscValidPointer(ov,3);
  DMDA_VEC_GET_SHAPE(da,vec,xs,ys,zs,xm,ym,zm,gxs,gys,gzs,gxm,gym,gzm,N,dim,dof);
  PetscCheck(dim == 3,PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,"KokkosOffsetView is 4D but DMDA is %dD",(int)dim);
  if (overwrite) {ierr = VecGetKokkosViewWrite(vec,&kv);CHKERRQ(ierr);}
  else {ierr = VecGetKokkosView(vec,&kv);CHKERRQ(ierr);}
  *ov  = PetscScalarKokkosOffsetView4DType<MemorySpace>(kv.data(),{gzs,gys,gxs,0},{gzs+gzm,gys+gym,gxs+gxm,dof});
  PetscFunctionReturn(0);
}

template<class MemorySpace>
PetscErrorCode DMDAVecRestoreKokkosOffsetViewDOF_Private(DM da,Vec vec,PetscScalarKokkosOffsetView4DType<MemorySpace> *ov,PetscBool overwrite)
{
  PetscErrorCode                             ierr;
  PetscScalarKokkosViewType<MemorySpace>     kv;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(da,DM_CLASSID,1,DMDA);
  PetscValidHeaderSpecific(vec,VEC_CLASSID,2);
  PetscValidPointer(ov,3);
  kv   = PetscScalarKokkosViewType<MemorySpace>(ov->data(),ov->extent(0)*ov->extent(1)*ov->extent(2)*ov->extent(3));
  if (overwrite) {ierr = VecRestoreKokkosViewWrite(vec,&kv);CHKERRQ(ierr);}
  else {ierr = VecRestoreKokkosView(vec,&kv);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

template<class MemorySpace>
PetscErrorCode DMDAVecGetKokkosOffsetViewDOF(DM da,Vec vec,ConstPetscScalarKokkosOffsetView4DType<MemorySpace> *ov)
{
  PetscErrorCode                               ierr;
  PetscInt                                     xs,ys,zs,xm,ym,zm,gxs,gys,gzs,gxm,gym,gzm,N,dim,dof;
  ConstPetscScalarKokkosViewType<MemorySpace>  kv;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(da,DM_CLASSID,1,DMDA);
  PetscValidHeaderSpecific(vec,VEC_CLASSID,2);
  PetscValidPointer(ov,3);
  DMDA_VEC_GET_SHAPE(da,vec,xs,ys,zs,xm,ym,zm,gxs,gys,gzs,gxm,gym,gzm,N,dim,dof);
  PetscCheck(dim == 3,PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,"KokkosOffsetView is 4D but DMDA is %dD",(int)dim);
  ierr = VecGetKokkosView(vec,&kv);CHKERRQ(ierr);
  *ov  = ConstPetscScalarKokkosOffsetView4DType<MemorySpace>(kv.data(),{gzs,gys,gxs,0},{gzs+gzm,gys+gym,gxs+gxm,dof});
  PetscFunctionReturn(0);
}

template<class MemorySpace>
PetscErrorCode DMDAVecRestoreKokkosOffsetViewDOF(DM da,Vec vec,ConstPetscScalarKokkosOffsetView4DType<MemorySpace> *ov)
{
  PetscErrorCode                               ierr;
  ConstPetscScalarKokkosViewType<MemorySpace>  kv;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(da,DM_CLASSID,1,DMDA);
  PetscValidHeaderSpecific(vec,VEC_CLASSID,2);
  PetscValidPointer(ov,3);
  kv   = ConstPetscScalarKokkosViewType<MemorySpace>(ov->data(),ov->extent(0)*ov->extent(1)*ov->extent(2)*ov->extent(3));
  ierr = VecRestoreKokkosView(vec,&kv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

template   PETSC_VISIBILITY_PUBLIC PetscErrorCode DMDAVecGetKokkosOffsetViewDOF         (DM,Vec,  ConstPetscScalarKokkosOffsetView2D*);
template   PETSC_VISIBILITY_PUBLIC PetscErrorCode DMDAVecRestoreKokkosOffsetViewDOF     (DM,Vec,  ConstPetscScalarKokkosOffsetView2D*);
template<> PETSC_VISIBILITY_PUBLIC PetscErrorCode DMDAVecGetKokkosOffsetViewDOF         (DM da,Vec vec,PetscScalarKokkosOffsetView2D* ov) {return DMDAVecGetKokkosOffsetViewDOF_Private(da,vec,ov,PETSC_FALSE);}
template<> PETSC_VISIBILITY_PUBLIC PetscErrorCode DMDAVecRestoreKokkosOffsetViewDOF     (DM da,Vec vec,PetscScalarKokkosOffsetView2D* ov) {return DMDAVecRestoreKokkosOffsetViewDOF_Private(da,vec,ov,PETSC_FALSE);}
template<> PETSC_VISIBILITY_PUBLIC PetscErrorCode DMDAVecGetKokkosOffsetViewDOFWrite    (DM da,Vec vec,PetscScalarKokkosOffsetView2D* ov) {return DMDAVecGetKokkosOffsetViewDOF_Private(da,vec,ov,PETSC_TRUE);}
template<> PETSC_VISIBILITY_PUBLIC PetscErrorCode DMDAVecRestoreKokkosOffsetViewDOFWrite(DM da,Vec vec,PetscScalarKokkosOffsetView2D* ov) {return DMDAVecRestoreKokkosOffsetViewDOF_Private(da,vec,ov,PETSC_TRUE);}

template   PETSC_VISIBILITY_PUBLIC PetscErrorCode DMDAVecGetKokkosOffsetViewDOF         (DM,Vec,  ConstPetscScalarKokkosOffsetView3D*);
template   PETSC_VISIBILITY_PUBLIC PetscErrorCode DMDAVecRestoreKokkosOffsetViewDOF     (DM,Vec,  ConstPetscScalarKokkosOffsetView3D*);
template<> PETSC_VISIBILITY_PUBLIC PetscErrorCode DMDAVecGetKokkosOffsetViewDOF         (DM da,Vec vec,PetscScalarKokkosOffsetView3D* ov) {return DMDAVecGetKokkosOffsetViewDOF_Private(da,vec,ov,PETSC_FALSE);}
template<> PETSC_VISIBILITY_PUBLIC PetscErrorCode DMDAVecRestoreKokkosOffsetViewDOF     (DM da,Vec vec,PetscScalarKokkosOffsetView3D* ov) {return DMDAVecRestoreKokkosOffsetViewDOF_Private(da,vec,ov,PETSC_FALSE);}
template<> PETSC_VISIBILITY_PUBLIC PetscErrorCode DMDAVecGetKokkosOffsetViewDOFWrite    (DM da,Vec vec,PetscScalarKokkosOffsetView3D* ov) {return DMDAVecGetKokkosOffsetViewDOF_Private(da,vec,ov,PETSC_TRUE);}
template<> PETSC_VISIBILITY_PUBLIC PetscErrorCode DMDAVecRestoreKokkosOffsetViewDOFWrite(DM da,Vec vec,PetscScalarKokkosOffsetView3D* ov) {return DMDAVecRestoreKokkosOffsetViewDOF_Private(da,vec,ov,PETSC_TRUE);}

template   PETSC_VISIBILITY_PUBLIC PetscErrorCode DMDAVecGetKokkosOffsetViewDOF         (DM,Vec,  ConstPetscScalarKokkosOffsetView4D*);
template   PETSC_VISIBILITY_PUBLIC PetscErrorCode DMDAVecRestoreKokkosOffsetViewDOF     (DM,Vec,  ConstPetscScalarKokkosOffsetView4D*);
template<> PETSC_VISIBILITY_PUBLIC PetscErrorCode DMDAVecGetKokkosOffsetViewDOF         (DM da,Vec vec,PetscScalarKokkosOffsetView4D* ov) {return DMDAVecGetKokkosOffsetViewDOF_Private(da,vec,ov,PETSC_FALSE);}
template<> PETSC_VISIBILITY_PUBLIC PetscErrorCode DMDAVecRestoreKokkosOffsetViewDOF     (DM da,Vec vec,PetscScalarKokkosOffsetView4D* ov) {return DMDAVecRestoreKokkosOffsetViewDOF_Private(da,vec,ov,PETSC_FALSE);}
template<> PETSC_VISIBILITY_PUBLIC PetscErrorCode DMDAVecGetKokkosOffsetViewDOFWrite    (DM da,Vec vec,PetscScalarKokkosOffsetView4D* ov) {return DMDAVecGetKokkosOffsetViewDOF_Private(da,vec,ov,PETSC_TRUE);}
template<> PETSC_VISIBILITY_PUBLIC PetscErrorCode DMDAVecRestoreKokkosOffsetViewDOFWrite(DM da,Vec vec,PetscScalarKokkosOffsetView4D* ov) {return DMDAVecRestoreKokkosOffsetViewDOF_Private(da,vec,ov,PETSC_TRUE);}

#if !defined(KOKKOS_ENABLE_DEFAULT_DEVICE_TYPE_HOST) /* Get host views if the default memory space is not host space */
template   PETSC_VISIBILITY_PUBLIC PetscErrorCode DMDAVecGetKokkosOffsetViewDOF         (DM,Vec,  ConstPetscScalarKokkosOffsetView2DHost*);
template   PETSC_VISIBILITY_PUBLIC PetscErrorCode DMDAVecRestoreKokkosOffsetViewDOF     (DM,Vec,  ConstPetscScalarKokkosOffsetView2DHost*);
template<> PETSC_VISIBILITY_PUBLIC PetscErrorCode DMDAVecGetKokkosOffsetViewDOF         (DM da,Vec vec,PetscScalarKokkosOffsetView2DHost* ov) {return DMDAVecGetKokkosOffsetViewDOF_Private(da,vec,ov,PETSC_FALSE);}
template<> PETSC_VISIBILITY_PUBLIC PetscErrorCode DMDAVecRestoreKokkosOffsetViewDOF     (DM da,Vec vec,PetscScalarKokkosOffsetView2DHost* ov) {return DMDAVecRestoreKokkosOffsetViewDOF_Private(da,vec,ov,PETSC_FALSE);}
template<> PETSC_VISIBILITY_PUBLIC PetscErrorCode DMDAVecGetKokkosOffsetViewDOFWrite    (DM da,Vec vec,PetscScalarKokkosOffsetView2DHost* ov) {return DMDAVecGetKokkosOffsetViewDOF_Private(da,vec,ov,PETSC_TRUE);}
template<> PETSC_VISIBILITY_PUBLIC PetscErrorCode DMDAVecRestoreKokkosOffsetViewDOFWrite(DM da,Vec vec,PetscScalarKokkosOffsetView2DHost* ov) {return DMDAVecRestoreKokkosOffsetViewDOF_Private(da,vec,ov,PETSC_TRUE);}

template   PETSC_VISIBILITY_PUBLIC PetscErrorCode DMDAVecGetKokkosOffsetViewDOF         (DM,Vec,  ConstPetscScalarKokkosOffsetView3DHost*);
template   PETSC_VISIBILITY_PUBLIC PetscErrorCode DMDAVecRestoreKokkosOffsetViewDOF     (DM,Vec,  ConstPetscScalarKokkosOffsetView3DHost*);
template<> PETSC_VISIBILITY_PUBLIC PetscErrorCode DMDAVecGetKokkosOffsetViewDOF         (DM da,Vec vec,PetscScalarKokkosOffsetView3DHost* ov) {return DMDAVecGetKokkosOffsetViewDOF_Private(da,vec,ov,PETSC_FALSE);}
template<> PETSC_VISIBILITY_PUBLIC PetscErrorCode DMDAVecRestoreKokkosOffsetViewDOF     (DM da,Vec vec,PetscScalarKokkosOffsetView3DHost* ov) {return DMDAVecRestoreKokkosOffsetViewDOF_Private(da,vec,ov,PETSC_FALSE);}
template<> PETSC_VISIBILITY_PUBLIC PetscErrorCode DMDAVecGetKokkosOffsetViewDOFWrite    (DM da,Vec vec,PetscScalarKokkosOffsetView3DHost* ov) {return DMDAVecGetKokkosOffsetViewDOF_Private(da,vec,ov,PETSC_TRUE);}
template<> PETSC_VISIBILITY_PUBLIC PetscErrorCode DMDAVecRestoreKokkosOffsetViewDOFWrite(DM da,Vec vec,PetscScalarKokkosOffsetView3DHost* ov) {return DMDAVecRestoreKokkosOffsetViewDOF_Private(da,vec,ov,PETSC_TRUE);}

template   PETSC_VISIBILITY_PUBLIC PetscErrorCode DMDAVecGetKokkosOffsetViewDOF         (DM,Vec,  ConstPetscScalarKokkosOffsetView4DHost*);
template   PETSC_VISIBILITY_PUBLIC PetscErrorCode DMDAVecRestoreKokkosOffsetViewDOF     (DM,Vec,  ConstPetscScalarKokkosOffsetView4DHost*);
template<> PETSC_VISIBILITY_PUBLIC PetscErrorCode DMDAVecGetKokkosOffsetViewDOF         (DM da,Vec vec,PetscScalarKokkosOffsetView4DHost* ov) {return DMDAVecGetKokkosOffsetViewDOF_Private(da,vec,ov,PETSC_FALSE);}
template<> PETSC_VISIBILITY_PUBLIC PetscErrorCode DMDAVecRestoreKokkosOffsetViewDOF     (DM da,Vec vec,PetscScalarKokkosOffsetView4DHost* ov) {return DMDAVecRestoreKokkosOffsetViewDOF_Private(da,vec,ov,PETSC_FALSE);}
template<> PETSC_VISIBILITY_PUBLIC PetscErrorCode DMDAVecGetKokkosOffsetViewDOFWrite    (DM da,Vec vec,PetscScalarKokkosOffsetView4DHost* ov) {return DMDAVecGetKokkosOffsetViewDOF_Private(da,vec,ov,PETSC_TRUE);}
template<> PETSC_VISIBILITY_PUBLIC PetscErrorCode DMDAVecRestoreKokkosOffsetViewDOFWrite(DM da,Vec vec,PetscScalarKokkosOffsetView4DHost* ov) {return DMDAVecRestoreKokkosOffsetViewDOF_Private(da,vec,ov,PETSC_TRUE);}
#endif
