#ifndef PETSCDEVICETESTCOMMON_H
#define PETSCDEVICETESTCOMMON_H

/* this file needs to be the one to include petsc/private/deviceimpl.h since it needs to define
 * a special macro to ensure that the error checking macros stay defined even in optimized
 * builds
 */
#if defined(PETSCDEVICEIMPL_H)
  #error "must #include this file before petsc/private/deviceimpl.h"
#endif

#if !defined(PETSC_DEVICE_KEEP_ERROR_CHECKING_MACROS)
  #define PETSC_DEVICE_KEEP_ERROR_CHECKING_MACROS 1
#endif
#include <petsc/private/deviceimpl.h>

static inline PetscErrorCode AssertDeviceExists(PetscDevice device)
{
  PetscFunctionBegin;
  PetscValidDevice(device, 1);
  PetscFunctionReturn(0);
}

static inline PetscErrorCode AssertDeviceDoesNotExist(PetscDevice device)
{
  PetscFunctionBegin;
  PetscCheck(!device, PETSC_COMM_SELF, PETSC_ERR_PLIB, "PetscDevice was not destroyed for type %s", PetscDeviceTypes[device->type]);
  PetscFunctionReturn(0);
}

static inline PetscErrorCode AssertDeviceContextExists(PetscDeviceContext dctx)
{
  PetscFunctionBegin;
  PetscValidDeviceContext(dctx, 1);
  PetscFunctionReturn(0);
}

static inline PetscErrorCode AssertDeviceContextDoesNotExist(PetscDeviceContext dctx)
{
  PetscFunctionBegin;
  PetscCheck(!dctx, PETSC_COMM_SELF, PETSC_ERR_PLIB, "PetscDeviceContext was not destroyed");
  PetscFunctionReturn(0);
}

static inline PetscErrorCode AssertPetscStreamTypesValidAndEqual(PetscStreamType left, PetscStreamType right, const char *errStr)
{
  PetscFunctionBegin;
  PetscValidStreamType(left, 1);
  PetscValidStreamType(right, 2);
  PetscCheck(left == right, PETSC_COMM_SELF, PETSC_ERR_ARG_CORRUPT, errStr, PetscStreamTypes[left], PetscStreamTypes[right]);
  PetscFunctionReturn(0);
}

static inline PetscErrorCode AssertPetscDeviceTypesValidAndEqual(PetscDeviceType left, PetscDeviceType right, const char *errStr)
{
  PetscFunctionBegin;
  PetscValidDeviceType(left, 1);
  PetscValidDeviceType(right, 2);
  PetscCheck(left == right, PETSC_COMM_SELF, PETSC_ERR_ARG_CORRUPT, errStr, PetscDeviceTypes[left], PetscDeviceTypes[right]);
  PetscFunctionReturn(0);
}

static inline PetscErrorCode AssertPetscDevicesValidAndEqual(PetscDevice left, PetscDevice right, const char *errStr)
{
  PetscFunctionBegin;
  PetscCheckCompatibleDevices(left, 1, right, 2);
  PetscCheck(left == right, PETSC_COMM_SELF, PETSC_ERR_ARG_CORRUPT, "%s", errStr);
  PetscFunctionReturn(0);
}

static inline PetscErrorCode AssertPetscDeviceContextsValidAndEqual(PetscDeviceContext left, PetscDeviceContext right, const char *errStr)
{
  PetscFunctionBegin;
  PetscCheckCompatibleDeviceContexts(left, 1, right, 2);
  PetscCheck(left == right, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "%s", errStr);
  PetscFunctionReturn(0);
}
#endif /* PETSCDEVICETESTCOMMON_H */
