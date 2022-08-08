#if !defined(PETSCDMDA_KOKKOS_HPP)
#define PETSCDMDA_KOKKOS_HPP

#include <petscvec_kokkos.hpp>
#include <petscdmda.h>

/* SUBMANSEC = DMDA */

#if defined(PETSC_HAVE_KOKKOS)
#include <Kokkos_Core.hpp>
#include <Kokkos_OffsetView.hpp>

/*@C
   DMDAVecGetKokkosOffsetView - Gets a Kokkos OffsetView that contains up-to-date data of a vector in the given memory space.

   Synopsis:
   #include <petscdmda_kokkos.hpp>
   PetscErrorCode DMDAVecGetKokkosOffsetView(DM da,Vec v,Kokkos::Experimental::OffsetView<const PetscScalar*,MemorySpace>* kv);
   PetscErrorCode DMDAVecGetKokkosOffsetView(DM da,Vec v,Kokkos::Experimental::OffsetView<PetscScalar*,MemorySpace>* kv);
   PetscErrorCode DMDAVecGetKokkosOffsetViewWrite(DM da,Vec v,Kokkos::Experimental::OffsetView<PetscScalar*,MemorySpace>* kv);
   PetscErrorCode DMDAVecGetKokkosOffsetView(DM da,Vec v,Kokkos::Experimental::OffsetView<const PetscScalar**,Kokkos::LayoutRight,MemorySpace>* kv);
   PetscErrorCode DMDAVecGetKokkosOffsetView(DM da,Vec v,Kokkos::Experimental::OffsetView<PetscScalar**,Kokkos::LayoutRight,MemorySpace>* kv);
   PetscErrorCode DMDAVecGetKokkosOffsetViewWrite(DM da,Vec v,Kokkos::Experimental::OffsetView<PetscScalar**,Kokkos::LayoutRight,MemorySpace>* kv);
   PetscErrorCode DMDAVecGetKokkosOffsetView(DM da,Vec v,Kokkos::Experimental::OffsetView<const PetscScalar***,Kokkos::LayoutRight,MemorySpace>* kv);
   PetscErrorCode DMDAVecGetKokkosOffsetView(DM da,Vec v,Kokkos::Experimental::OffsetView<PetscScalar***,Kokkos::LayoutRight,MemorySpace>* kv);
   PetscErrorCode DMDAVecGetKokkosOffsetViewWrite(DM da,Vec v,Kokkos::Experimental::OffsetView<PetscScalar***,Kokkos::LayoutRight,MemorySpace>* kv);

   Logically collective on da

   Input Parameters:
+  da - the distributed array
-  v - the vector, either a vector the same size as one obtained with DMCreateGlobalVector() or DMCreateLocalVector()

   Output Parameter:
.  kv - the Kokkos OffsetView with a user-specified template parameter MemorySpace

   Notes:
    Call DMDAVecRestoreKokkosOffsetView() or DMDAVecRestoreKokkosOffsetViewWrite() once you have finished accessing the OffsetView.

    If the vector is not a Kokkos vector, an error will be raised.

    If the vector is a local vector (obtained with DMCreateLocalVector() etc) then the ghost point locations are accessible. If it is
    a global vector then the ghost points are not accessible. Of course with the local vector you will have to do the
    appropriate DMGlobalToLocalBegin() and DMGlobalToLocalEnd() to have correct values in the ghost locations.

    These routines are similar to DMDAVecGetArray() and friends. One can read-only, write-only or read/write access the returned
    Kokkos OffsetView.  Note that passing in a constant OffsetView enables read-only access.
    Currently, only two memory spaces are supported: Kokkos::HostSpace and Kokkos::DefaultExecutionSpace::memory_space.
    If needed, a memory copy will be internally called to copy the latest vector data to the specified memory space.

    In C, to access the returned array of DMDAVecGetArray(), the indexing is "backwards", i.e., array[k][j][i] (instead of array[i][j][k]),
    where i, j, k are loop variables for the x, y, z dimensions respectively specified in DMDACreate3d(), for example.

    To give users the same experience as DMDAVecGetArray(), we mandate the returned OffsetView always has Kokkos::LayoutRight (that is, rightest
    subscript has a stride 1, as in C multi-dimensional arrays), regardless of whether the memory space is host or device. Thus it is important
    to use Iterate::Right as IterateInner if one uses Kokkos::MDRangePolicy to access the OffsetView.

    Note that the OffsetView kv's first dimension (i.e., the leftest, dim 0) corresponds to DA's z direction, and its last dimension
    (rightest) corresponds to DA's x direction.

    If the vector is a global vector, we have
.vb
      kv.extent(0) = zm*dof,  kv.begin(0) = zs*dof, kv.end(0) = (zs+zm)*dof
      kv.extent(1) = ym*dof,  kv.begin(1) = ys*dof, kv.end(1) = (ys+ym)*dof
      kv.extent(2) = xm*dof,  kv.begin(2) = xs*dof, kv.end(2) = (xs+xm)*dof
.ve
    If the vector is a local vector, we have
.vb
      kv.extent(0) = gzm*dof,  kv.begin(0) = gzs*dof, kv.end(0) = (gzs+gzm)*dof
      kv.extent(1) = gym*dof,  kv.begin(1) = gys*dof, kv.end(1) = (gys+gym)*dof
      kv.extent(2) = gxm*dof,  kv.begin(2) = gxs*dof, kv.end(2) = (gxs+gxm)*dof
.ve

    The starts and widths above are obtained by
.vb
     DMDAGetCorners(da,&xs,&ys,&zs,&xm,&ym,&zm);
     DMDAGetGhostCorners(da,&gxs,&gys,&gzs,&gxm,&gym,&gzm);
.ve

    For example, to initialize a grid,

.vb
    typedef Kokkos::Experimental::OffsetView<const PetscScalar***,Kokkos::LayoutRight,MemorySpace> PetscScalarKokkosOffsetView3D;

    PetscScalarKokkosOffsetView3D kv;
    DMDAVecGetKokkosOffsetViewWrite(da,v,&kv); // v is a global vector and we assume dof is 1
    DMDAGetCorners(da,&xs,&ys,&zs,&xm,&ym,&zm);

    parallel_for ("Label",MDRangePolicy <Rank<3, Iterate::Right, Iterate::Right>>(
      {zs,ys,xs},{zs+zm,ys+ym,xs+xm}), KOKKOS_LAMBDA (PetscInt k,PetscInt j,PetscInt i) {
      kv(k,j,i) = ...;
    });
    DMDAVecRestoreKokkosOffsetViewWrite(da,v,&kv);
.ve

    For a multi-component problem, one could cast the returned OffsetView to a user's type. But one has also to shrink
    the OffsetView's extent accordingly. For example,
.vb
    typedef struct {
      PetscScalar omega,temperature;
    } Node;

    using NodeKokkosOffsetView3D = Kokkos::Experimental::OffsetView<const Node***,Kokkos::LayoutRight,MemorySpace>;
    DMDAVecGetKokkosOffsetViewWrite(da,v,&tv);
    NodeKokkosOffsetView3D kv(reinterpret_cast<Node*>(tv.data()),{tv.begin(0)/dof,tv.begin(1)/dof,tv.begin(2)/dof}, {tv.end(0)/dof,tv.end(1)/dof,tv.end(2)/dof});

    parallel_for ("Label",MDRangePolicy<Rank<3, Iterate::Right, Iterate::Right>>(
      {zs,ys,xs},{zs+zm,ys+ym,xs+xm}), KOKKOS_LAMBDA (PetscInt k,PetscInt j,PetscInt i) {
      kv(k,j,i).omega = ...;
    });
    DMDAVecRestoreKokkosOffsetViewWrite(da,v,&tv);
.ve

  Level: intermediate

.seealso: `DMDAVecRestoreKokkosOffsetView()`, `DMDAGetGhostCorners()`, `DMDAGetCorners()`, `VecGetArray()`, `VecRestoreArray()`, `DMDAVecRestoreArray()`, `DMDAVecRestoreArrayDOF()`
          `DMDAVecGetArrayDOF()`, `DMDAVecGetArrayWrite()`, `DMDAVecRestoreArrayWrite()`, `DMDAVecGetArrayRead()`, `DMDAVecRestoreArrayRead()`,
          `DMStagVecGetArray()`
@*/
template<class MemorySpace> PetscErrorCode DMDAVecGetKokkosOffsetView         (DM,Vec,Kokkos::Experimental::OffsetView<const PetscScalar*,MemorySpace>*);
template<class MemorySpace> PetscErrorCode DMDAVecGetKokkosOffsetView         (DM,Vec,Kokkos::Experimental::OffsetView<      PetscScalar*,MemorySpace>*);
template<class MemorySpace> PetscErrorCode DMDAVecGetKokkosOffsetViewWrite    (DM,Vec,Kokkos::Experimental::OffsetView<      PetscScalar*,MemorySpace>*);

template<class MemorySpace> PetscErrorCode DMDAVecGetKokkosOffsetView         (DM,Vec,Kokkos::Experimental::OffsetView<const PetscScalar**, Kokkos::LayoutRight,MemorySpace>*);
template<class MemorySpace> PetscErrorCode DMDAVecGetKokkosOffsetView         (DM,Vec,Kokkos::Experimental::OffsetView<      PetscScalar**, Kokkos::LayoutRight,MemorySpace>*);
template<class MemorySpace> PetscErrorCode DMDAVecGetKokkosOffsetViewWrite    (DM,Vec,Kokkos::Experimental::OffsetView<      PetscScalar**, Kokkos::LayoutRight,MemorySpace>*);

template<class MemorySpace> PetscErrorCode DMDAVecGetKokkosOffsetView         (DM,Vec,Kokkos::Experimental::OffsetView<const PetscScalar***,Kokkos::LayoutRight,MemorySpace>*);
template<class MemorySpace> PetscErrorCode DMDAVecGetKokkosOffsetView         (DM,Vec,Kokkos::Experimental::OffsetView<      PetscScalar***,Kokkos::LayoutRight,MemorySpace>*);
template<class MemorySpace> PetscErrorCode DMDAVecGetKokkosOffsetViewWrite    (DM,Vec,Kokkos::Experimental::OffsetView<      PetscScalar***,Kokkos::LayoutRight,MemorySpace>*);

/*@C
   DMDAVecRestoreKokkosOffsetView - Returns the Kokkos OffsetView that gotten from DMDAVecGetKokkosOffsetView()

   Synopsis:
   #include <petscdmda_kokkos.hpp>
   PetscErrorCode DMDAVecRestoreKokkosOffsetView(DM da,Vec v,Kokkos::Experimental::OffsetView<const PetscScalar*,MemorySpace>* kv);
   PetscErrorCode DMDAVecRestoreKokkosOffsetView(DM da,Vec v,Kokkos::Experimental::OffsetView<PetscScalar*,MemorySpace>* kv);
   PetscErrorCode DMDAVecRestoreKokkosOffsetViewWrite(DM da,Vec v,Kokkos::Experimental::OffsetView<PetscScalar*,MemorySpace>* kv);
   PetscErrorCode DMDAVecRestoreKokkosOffsetView(DM da,Vec v,Kokkos::Experimental::OffsetView<const PetscScalar**,Kokkos::LayoutRight,MemorySpace>* kv);
   PetscErrorCode DMDAVecRestoreKokkosOffsetView(DM da,Vec v,Kokkos::Experimental::OffsetView<PetscScalar**,Kokkos::LayoutRight,MemorySpace>* kv);
   PetscErrorCode DMDAVecRestoreKokkosOffsetViewWrite(DM da,Vec v,Kokkos::Experimental::OffsetView<PetscScalar**,Kokkos::LayoutRight,MemorySpace>* kv);
   PetscErrorCode DMDAVecRestoreKokkosOffsetView(DM da,Vec v,Kokkos::Experimental::OffsetView<const PetscScalar***,Kokkos::LayoutRight,MemorySpace>* kv);
   PetscErrorCode DMDAVecRestoreKokkosOffsetView(DM da,Vec v,Kokkos::Experimental::OffsetView<PetscScalar***,Kokkos::LayoutRight,MemorySpace>* kv);
   PetscErrorCode DMDAVecRestoreKokkosOffsetViewWrite(DM da,Vec v,Kokkos::Experimental::OffsetView<PetscScalar***,Kokkos::LayoutRight,MemorySpace>* kv);

   Logically collective on da

   Input Parameters:
+  da - the distributed array
.  v - the vector, either a vector the same size as one obtained with DMCreateGlobalVector() or DMCreateLocalVector()
-  kv - the Kokkos OffsetView with a user-specified template parameter MemorySpace

   Notes:
    If the vector is not of type VECKOKKOS, an error will be raised.

  Level: intermediate

.seealso: `DMDAVecGetKokkosOffsetView()`, `DMDAGetGhostCorners()`, `DMDAGetCorners()`, `VecGetArray()`, `VecRestoreArray()`, `DMDAVecRestoreArray()`, `DMDAVecRestoreArrayDOF()`
          `DMDAVecGetArrayDOF()`, `DMDAVecGetArrayWrite()`, `DMDAVecRestoreArrayWrite()`, `DMDAVecGetArrayRead()`, `DMDAVecRestoreArrayRead()`,
          `DMStagVecGetArray()`
@*/
template<class MemorySpace> PetscErrorCode DMDAVecRestoreKokkosOffsetView     (DM,Vec,Kokkos::Experimental::OffsetView<const PetscScalar*,MemorySpace>*);
template<class MemorySpace> PetscErrorCode DMDAVecRestoreKokkosOffsetView     (DM,Vec,Kokkos::Experimental::OffsetView<      PetscScalar*,MemorySpace>*);
template<class MemorySpace> PetscErrorCode DMDAVecRestoreKokkosOffsetViewWrite(DM,Vec,Kokkos::Experimental::OffsetView<      PetscScalar*,MemorySpace>*);

template<class MemorySpace> PetscErrorCode DMDAVecRestoreKokkosOffsetView     (DM,Vec,Kokkos::Experimental::OffsetView<const PetscScalar**, Kokkos::LayoutRight,MemorySpace>*);
template<class MemorySpace> PetscErrorCode DMDAVecRestoreKokkosOffsetView     (DM,Vec,Kokkos::Experimental::OffsetView<      PetscScalar**, Kokkos::LayoutRight,MemorySpace>*);
template<class MemorySpace> PetscErrorCode DMDAVecRestoreKokkosOffsetViewWrite(DM,Vec,Kokkos::Experimental::OffsetView<      PetscScalar**, Kokkos::LayoutRight,MemorySpace>*);

template<class MemorySpace> PetscErrorCode DMDAVecRestoreKokkosOffsetView     (DM,Vec,Kokkos::Experimental::OffsetView<const PetscScalar***,Kokkos::LayoutRight,MemorySpace>*);
template<class MemorySpace> PetscErrorCode DMDAVecRestoreKokkosOffsetView     (DM,Vec,Kokkos::Experimental::OffsetView<      PetscScalar***,Kokkos::LayoutRight,MemorySpace>*);
template<class MemorySpace> PetscErrorCode DMDAVecRestoreKokkosOffsetViewWrite(DM,Vec,Kokkos::Experimental::OffsetView<      PetscScalar***,Kokkos::LayoutRight,MemorySpace>*);

/*@C
   DMDAVecGetKokkosOffsetViewDOF - Gets a Kokkos OffsetView that contains up-to-date data of a vector in the given memory space, with DOF as the rightest dimension of the OffsetView

   Synopsis:
   #include <petscdmda_kokkos.hpp>
   PetscErrorCode DMDAVecGetKokkosOffsetViewDOF(DM da,Vec v,Kokkos::Experimental::OffsetView<const PetscScalar**,Kokkos::LayoutRight,MemorySpace>* kv);
   PetscErrorCode DMDAVecGetKokkosOffsetViewDOF(DM da,Vec v,Kokkos::Experimental::OffsetView<PetscScalar**,Kokkos::LayoutRight,MemorySpace>* kv);
   PetscErrorCode DMDAVecGetKokkosOffsetViewDOFWrite(DM da,Vec v,Kokkos::Experimental::OffsetView<PetscScalar**,Kokkos::LayoutRight,MemorySpace>* kv);
   PetscErrorCode DMDAVecGetKokkosOffsetViewDOF(DM da,Vec v,Kokkos::Experimental::OffsetView<const PetscScalar***,Kokkos::LayoutRight,MemorySpace>* kv);
   PetscErrorCode DMDAVecGetKokkosOffsetViewDOF(DM da,Vec v,Kokkos::Experimental::OffsetView<PetscScalar***,Kokkos::LayoutRight,MemorySpace>* kv);
   PetscErrorCode DMDAVecGetKokkosOffsetViewDOFWrite(DM da,Vec v,Kokkos::Experimental::OffsetView<PetscScalar***,Kokkos::LayoutRight,MemorySpace>* kv);
   PetscErrorCode DMDAVecGetKokkosOffsetViewDOF(DM da,Vec v,Kokkos::Experimental::OffsetView<const PetscScalar****,Kokkos::LayoutRight,MemorySpace>* kv);
   PetscErrorCode DMDAVecGetKokkosOffsetViewDOF(DM da,Vec v,Kokkos::Experimental::OffsetView<PetscScalar****,Kokkos::LayoutRight,MemorySpace>* kv);
   PetscErrorCode DMDAVecGetKokkosOffsetViewDOFWrite(DM da,Vec v,Kokkos::Experimental::OffsetView<PetscScalar****,Kokkos::LayoutRight,MemorySpace>* kv);

   Logically collective on da

   Input Parameters:
+  da - the distributed array
-  v - the vector, either a vector the same size as one obtained with DMCreateGlobalVector() or DMCreateLocalVector()

   Output Parameter:
.  kv - the Kokkos OffsetView with a user-specified template parameter MemorySpace

   Notes:
    Call DMDAVecRestoreKokkosOffsetViewDOF() or DMDAVecRestoreKokkosOffsetViewDOFWrite() once you have finished accessing the OffsetView.

    If the vector is not a Kokkos vector, an error will be raised.

    If the vector is a local vector (obtained with DMCreateLocalVector() etc) then the ghost point locations are accessible. If it is
    a global vector then the ghost points are not accessible. Of course with the local vector you will have to do the
    appropriate DMGlobalToLocalBegin() and DMGlobalToLocalEnd() to have correct values in the ghost locations.

    These routines are similar to DMDAVecGetArrayDOF() and friends. One can read-only, write-only or read/write access the returned
    Kokkos OffsetView.  Note that passing in a constant OffsetView enables read-only access.
    Currently, only two memory spaces are supported: Kokkos::HostSpace and Kokkos::DefaultExecutionSpace::memory_space.
    If needed, a memory copy will be internally called to copy the latest vector data to the given memory space.

    In C, to access the returned array of DMDAVecGetArrayDOF(), the indexing is "backwards", i.e., array[k][j][i][c] (instead of array[c][i][j][k]),
    where i, j, k are loop variables for the x, y, z dimensions respectively, and c is the loop variable for DOFs, as specified in DMDACreate3d(),
    for example.

    To give users the same experience as DMDAVecGetArrayDOF(), we mandate the returned OffsetView always has Kokkos::LayoutRight (that is, rightest
    subscript has a stride 1, as in C multi-dimensional arrays), regardless of whether the memory space is host or device. Thus it is important
    to use Iterate::Right as IterateInner if one uses Kokkos::MDRangePolicy to access the OffsetView.

    Note that for a 3D DA, the OffsetView kv's first dimension (i.e., the leftest, dim 0) corresponds to DA's z direction, and its second-to-last dimension
    (rightest) corresponds to DA's x direction.

    If the vector is a global vector, we have
.vb
      kv.extent(0) = zm,  kv.begin(0) = zs, kv.end(0) = zs+zm
      kv.extent(1) = ym,  kv.begin(1) = ys, kv.end(1) = ys+ym
      kv.extent(2) = xm,  kv.begin(2) = xs, kv.end(2) = xs+xm
      kv.extent(3) = dof, kv.begin(3) = 0,  kv.end(3) = dof
.ve
    If the vector is a local vector, we have
.vb
      kv.extent(0) = gzm, kv.begin(0) = gzs, kv.end(0) = gzs+gzm
      kv.extent(1) = gym, kv.begin(1) = gys, kv.end(1) = gys+gym
      kv.extent(2) = gxm, kv.begin(2) = gxs, kv.end(2) = gxs+gxm
      kv.extent(3) = dof, kv.begin(3) = 0,   kv.end(3) = dof
.ve

    The starts and widths above are obtained by
.vb
     DMDAGetCorners(da,&xs,&ys,&zs,&xm,&ym,&zm);
     DMDAGetGhostCorners(da,&gxs,&gys,&gzs,&gxm,&gym,&gzm);
.ve

    For example, to initialize a grid,
.vb
    typedef Kokkos::Experimental::OffsetView<const PetscScalar****,Kokkos::LayoutRight,MemorySpace> PetscScalarKokkosOffsetView4D;

    PetscScalarKokkosOffsetView4D kv;
    DMDAVecGetKokkosOffsetViewDOFWrite(da,v,&kv); // v is a global vector
    DMDAGetCorners(da,&xs,&ys,&zs,&xm,&ym,&zm);

    parallel_for ("Label",MDRangePolicy <Rank<4, Iterate::Right, Iterate::Right>>(
      {zs,ys,xs,0},{zs+zm,ys+ym,xs+xm,dof}), KOKKOS_LAMBDA (PetscInt k,PetscInt j,PetscInt i,PetscInt c) {
      kv(k,j,i,c) = ...;
    });
    DMDAVecRestoreKokkosOffsetViewDOFWrite(da,v,&kv);
.ve

  Level: intermediate

.seealso: `DMDAVecRestoreKokkosOffsetViewDOF()`, `DMDAGetGhostCorners()`, `DMDAGetCorners()`, `VecGetArray()`, `VecRestoreArray()`, `DMDAVecRestoreArray()`, `DMDAVecRestoreArrayDOF()`
          `DMDAVecGetArrayDOF()`, `DMDAVecGetArrayWrite()`, `DMDAVecRestoreArrayWrite()`, `DMDAVecGetArrayRead()`, `DMDAVecRestoreArrayRead()`,
          `DMStagVecGetArray()`
@*/
template<class MemorySpace> PetscErrorCode DMDAVecGetKokkosOffsetViewDOF         (DM,Vec,Kokkos::Experimental::OffsetView<const PetscScalar**, Kokkos::LayoutRight,MemorySpace>*);
template<class MemorySpace> PetscErrorCode DMDAVecGetKokkosOffsetViewDOF         (DM,Vec,Kokkos::Experimental::OffsetView<      PetscScalar**, Kokkos::LayoutRight,MemorySpace>*);
template<class MemorySpace> PetscErrorCode DMDAVecGetKokkosOffsetViewDOFWrite    (DM,Vec,Kokkos::Experimental::OffsetView<      PetscScalar**, Kokkos::LayoutRight,MemorySpace>*);

template<class MemorySpace> PetscErrorCode DMDAVecGetKokkosOffsetViewDOF         (DM,Vec,Kokkos::Experimental::OffsetView<const PetscScalar***, Kokkos::LayoutRight,MemorySpace>*);
template<class MemorySpace> PetscErrorCode DMDAVecGetKokkosOffsetViewDOF         (DM,Vec,Kokkos::Experimental::OffsetView<      PetscScalar***, Kokkos::LayoutRight,MemorySpace>*);
template<class MemorySpace> PetscErrorCode DMDAVecGetKokkosOffsetViewDOFWrite    (DM,Vec,Kokkos::Experimental::OffsetView<      PetscScalar***, Kokkos::LayoutRight,MemorySpace>*);

template<class MemorySpace> PetscErrorCode DMDAVecGetKokkosOffsetViewDOF         (DM,Vec,Kokkos::Experimental::OffsetView<const PetscScalar****, Kokkos::LayoutRight,MemorySpace>*);
template<class MemorySpace> PetscErrorCode DMDAVecGetKokkosOffsetViewDOF         (DM,Vec,Kokkos::Experimental::OffsetView<      PetscScalar****, Kokkos::LayoutRight,MemorySpace>*);
template<class MemorySpace> PetscErrorCode DMDAVecGetKokkosOffsetViewDOFWrite    (DM,Vec,Kokkos::Experimental::OffsetView<      PetscScalar****, Kokkos::LayoutRight,MemorySpace>*);

/*@C
   DMDAVecRestoreKokkosOffsetViewDOF - Returns the Kokkos OffsetView that gotten from DMDAVecGetKokkosOffsetViewDOF()

   Synopsis:
   #include <petscdmda_kokkos.hpp>
   PetscErrorCode DMDAVecRestoreKokkosOffsetViewDOF(DM da,Vec v,Kokkos::Experimental::OffsetView<const PetscScalar**,Kokkos::LayoutRight,MemorySpace>* kv);
   PetscErrorCode DMDAVecRestoreKokkosOffsetViewDOF(DM da,Vec v,Kokkos::Experimental::OffsetView<PetscScalar**,Kokkos::LayoutRight,MemorySpace>* kv);
   PetscErrorCode DMDAVecRestoreKokkosOffsetViewDOFWrite(DM da,Vec v,Kokkos::Experimental::OffsetView<PetscScalar**,Kokkos::LayoutRight,MemorySpace>* kv);
   PetscErrorCode DMDAVecRestoreKokkosOffsetViewDOF(DM da,Vec v,Kokkos::Experimental::OffsetView<const PetscScalar***,Kokkos::LayoutRight,MemorySpace>* kv);
   PetscErrorCode DMDAVecRestoreKokkosOffsetViewDOF(DM da,Vec v,Kokkos::Experimental::OffsetView<PetscScalar***,Kokkos::LayoutRight,MemorySpace>* kv);
   PetscErrorCode DMDAVecRestoreKokkosOffsetViewDOFWrite(DM da,Vec v,Kokkos::Experimental::OffsetView<PetscScalar***,Kokkos::LayoutRight,MemorySpace>* kv);
   PetscErrorCode DMDAVecRestoreKokkosOffsetViewDOF(DM da,Vec v,Kokkos::Experimental::OffsetView<const PetscScalar****,Kokkos::LayoutRight,MemorySpace>* kv);
   PetscErrorCode DMDAVecRestoreKokkosOffsetViewDOF(DM da,Vec v,Kokkos::Experimental::OffsetView<PetscScalar****,Kokkos::LayoutRight,MemorySpace>* kv);
   PetscErrorCode DMDAVecRestoreKokkosOffsetViewDOFWrite(DM da,Vec v,Kokkos::Experimental::OffsetView<PetscScalar****,Kokkos::LayoutRight,MemorySpace>* kv);

   Logically collective on da

   Input Parameters:
+  da - the distributed array
.  v - the vector, either a vector the same size as one obtained with DMCreateGlobalVector() or DMCreateLocalVector()
-  kv - the Kokkos OffsetView with a user-specified template parameter MemorySpace

   Notes:
    If the vector is not of type VECKOKKOS, an error will be raised.

  Level: intermediate

.seealso: `DMDAVecGetKokkosOffsetViewDOF()`, `DMDAVecGetKokkosOffsetView()`, `DMDAGetGhostCorners()`, `DMDAGetCorners()`, `VecGetArray()`, `VecRestoreArray()`, `DMDAVecRestoreArray()`, `DMDAVecRestoreArrayDOF()`
          `DMDAVecGetArrayDOF()`, `DMDAVecGetArrayWrite()`, `DMDAVecRestoreArrayWrite()`, `DMDAVecGetArrayRead()`, `DMDAVecRestoreArrayRead()`,
          `DMStagVecGetArray()`
@*/
template<class MemorySpace> PetscErrorCode DMDAVecRestoreKokkosOffsetViewDOF     (DM,Vec,Kokkos::Experimental::OffsetView<const PetscScalar**, Kokkos::LayoutRight,MemorySpace>*);
template<class MemorySpace> PetscErrorCode DMDAVecRestoreKokkosOffsetViewDOF     (DM,Vec,Kokkos::Experimental::OffsetView<      PetscScalar**, Kokkos::LayoutRight,MemorySpace>*);
template<class MemorySpace> PetscErrorCode DMDAVecRestoreKokkosOffsetViewDOFWrite(DM,Vec,Kokkos::Experimental::OffsetView<      PetscScalar**, Kokkos::LayoutRight,MemorySpace>*);

template<class MemorySpace> PetscErrorCode DMDAVecRestoreKokkosOffsetViewDOF     (DM,Vec,Kokkos::Experimental::OffsetView<const PetscScalar***, Kokkos::LayoutRight,MemorySpace>*);
template<class MemorySpace> PetscErrorCode DMDAVecRestoreKokkosOffsetViewDOF     (DM,Vec,Kokkos::Experimental::OffsetView<      PetscScalar***, Kokkos::LayoutRight,MemorySpace>*);
template<class MemorySpace> PetscErrorCode DMDAVecRestoreKokkosOffsetViewDOFWrite(DM,Vec,Kokkos::Experimental::OffsetView<      PetscScalar***, Kokkos::LayoutRight,MemorySpace>*);

template<class MemorySpace> PetscErrorCode DMDAVecRestoreKokkosOffsetViewDOF     (DM,Vec,Kokkos::Experimental::OffsetView<const PetscScalar****, Kokkos::LayoutRight,MemorySpace>*);
template<class MemorySpace> PetscErrorCode DMDAVecRestoreKokkosOffsetViewDOF     (DM,Vec,Kokkos::Experimental::OffsetView<      PetscScalar****, Kokkos::LayoutRight,MemorySpace>*);
template<class MemorySpace> PetscErrorCode DMDAVecRestoreKokkosOffsetViewDOFWrite(DM,Vec,Kokkos::Experimental::OffsetView<      PetscScalar****, Kokkos::LayoutRight,MemorySpace>*);
#endif

#endif
