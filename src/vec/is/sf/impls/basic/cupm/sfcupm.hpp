#pragma once
#include <../src/vec/is/sf/impls/basic/sfpack.h>
#include <petsc/private/cupminterface.hpp>
#include <petsc/private/cupmatomics.hpp>
#include <petsc/private/deviceimpl.h>

namespace Petsc
{

namespace sf
{

namespace cupm
{

namespace impl
{

template <device::cupm::DeviceType T>
struct PETSC_SINGLE_LIBRARY_VISIBILITY_INTERNAL SfInterface : device::cupm::impl::Interface<T> {
  PETSC_CUPM_INHERIT_INTERFACE_TYPEDEFS_USING(T);

private:
  template <typename Type, PetscInt BS, PetscInt EQ>
  static PetscErrorCode Pack(PetscSFLink, PetscInt, PetscInt, PetscSFPackOpt, const PetscInt *, const void *, void *) noexcept;

  template <typename Type, class Op, PetscInt BS, PetscInt EQ>
  static PetscErrorCode UnpackAndOp(PetscSFLink, PetscInt, PetscInt, PetscSFPackOpt, const PetscInt *, void *, const void *) noexcept;

  template <typename Type, class Op, PetscInt BS, PetscInt EQ>
  static PetscErrorCode FetchAndOp(PetscSFLink, PetscInt, PetscInt, PetscSFPackOpt, const PetscInt *, void *, void *) noexcept;

  template <typename Type, class Op, PetscInt BS, PetscInt EQ>
  static PetscErrorCode ScatterAndOp(PetscSFLink, PetscInt, PetscInt, PetscSFPackOpt, const PetscInt *, const void *, PetscInt, PetscSFPackOpt, const PetscInt *, void *) noexcept;

  template <typename Type, PetscInt BS, PetscInt EQ>
  static PetscErrorCode ScatterAndInsert(PetscSFLink, PetscInt, PetscInt, PetscSFPackOpt, const PetscInt *, const void *, PetscInt, PetscSFPackOpt, const PetscInt *, void *) noexcept;

  template <typename Type, class Op, PetscInt BS, PetscInt EQ>
  static PetscErrorCode FetchAndOpLocal(PetscSFLink, PetscInt, PetscInt, PetscSFPackOpt, const PetscInt *, void *, PetscInt, PetscSFPackOpt, const PetscInt *, const void *, void *) noexcept;

  template <typename Type, PetscInt BS, PetscInt EQ>
  static void PackInit_RealType(PetscSFLink) noexcept;

  template <typename Type, PetscInt BS, PetscInt EQ, PetscInt size /*sizeof(Type)*/>
  struct PackInit_IntegerType_Atomic;

  template <typename Type, PetscInt BS, PetscInt EQ>
  static void PackInit_IntegerType(PetscSFLink link) noexcept;

#if PetscDefined(HAVE_COMPLEX)
  template <typename Type, PetscInt BS, PetscInt EQ>
  static void PackInit_ComplexType(PetscSFLink link) noexcept;
#endif

  template <typename Type>
  static void PackInit_PairType(PetscSFLink link) noexcept;

  template <typename Type, PetscInt BS, PetscInt EQ>
  static void PackInit_DumbType(PetscSFLink link) noexcept;

  static PetscErrorCode LinkSyncDevice(PetscSFLink) noexcept;
  static PetscErrorCode LinkSyncStream(PetscSFLink) noexcept;
  static PetscErrorCode LinkMemcpy(PetscSFLink, PetscMemType, void *, PetscMemType, const void *, size_t) noexcept;
  static PetscErrorCode LinkDestroy_MPI(PetscSF, PetscSFLink) noexcept;

public:
  static PetscErrorCode Malloc(PetscMemType, size_t, void **) noexcept;
  static PetscErrorCode Free(PetscMemType, void *) noexcept;
  static PetscErrorCode LinkSetUp(PetscSF, PetscSFLink, MPI_Datatype) noexcept;
};

} // namespace impl

} // namespace cupm

} // namespace sf

} // namespace Petsc
