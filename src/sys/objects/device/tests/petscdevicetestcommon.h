#ifndef PETSCDEVICE_H
#error "included this file before petscdevice.h, this file must be included last to ensure that public petsc headers are well formed"
#endif
#ifndef PETSCDEVICETESTCOMMON_H
#define PETSCDEVICETESTCOMMON_H

/* all of the error checking macros are undefined and redefined verbatim so that they are also
 * defined for optimized builds.
 */

#undef  PetscValidDeviceType
#define PetscValidDeviceType(_p_dev_type__,_p_arg__) do {               \
    if (PetscUnlikely(((_p_dev_type__) < PETSC_DEVICE_INVALID) ||       \
                      ((_p_dev_type__) > PETSC_DEVICE_MAX))) {          \
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,              \
               "Unknown PetscDeviceType '%d': Argument #%d",            \
               (_p_dev_type__),(_p_arg__));                             \
    } else if (PetscUnlikely(!PetscDeviceConfiguredFor_Internal(_p_dev_type__))) { \
      switch(_p_dev_type__) {                                           \
      case PETSC_DEVICE_INVALID:                                        \
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,                         \
                 "Invalid PetscDeviceType '%s': Argument #%d;"          \
                 " PETSc is not configured with device support",        \
                 PetscDeviceTypes[_p_dev_type__],(_p_arg__));           \
        break;                                                          \
      case PETSC_DEVICE_MAX:                                            \
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,                  \
                 "Invalid PetscDeviceType '%s': Argument #%d",          \
                 PetscDeviceTypes[_p_dev_type__],(_p_arg__));           \
        break;                                                          \
      default:                                                          \
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,                         \
                 "Not configured for PetscDeviceType '%s': Argument #%d;" \
                 " run configure --help %s for available options",      \
                 PetscDeviceTypes[_p_dev_type__],(_p_arg__),            \
                 PetscDeviceTypes[_p_dev_type__]);                      \
        break;                                                          \
      }                                                                 \
    }                                                                   \
  } while (0)

#undef  PetscValidDevice
#define PetscValidDevice(_p_dev__,_p_arg__)          do {       \
    PetscValidPointer(_p_dev__,_p_arg__);                       \
    PetscValidDeviceType((_p_dev__)->type,_p_arg__);            \
    if (PetscUnlikely((_p_dev__)->id < 0)) {                    \
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,                  \
               "Invalid PetscDevice: Argument #%d; "            \
               "id %" PetscInt_FMT " < 0",                      \
               (_p_arg__),(_p_dev__)->id);                      \
    } else if (PetscUnlikely((_p_dev__)->refcnt < 0)) {         \
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,                  \
               "Invalid PetscDevice: Argument #%d; "            \
               "negative reference count %" PetscInt_FMT,       \
               (_p_arg__),(_p_dev__)->refcnt);                  \
    }                                                           \
  } while (0)

/* for now just checks strict equality, but this can be changed as some devices
   (i.e. kokkos and any cupm should be compatible once implemented) */
#undef  PetscCheckCompatibleDevices
#define PetscCheckCompatibleDevices(_p_dev1__,_p_arg1__,_p_dev2__,_p_arg2__) \
  do {                                                                  \
    PetscValidDevice(_p_dev1__,_p_arg1__);                              \
    PetscValidDevice(_p_dev2__,_p_arg2__);                              \
    if (PetscUnlikely((_p_dev1__)->type != (_p_dev2__)->type)) {        \
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,                    \
               "PetscDevices are incompatible: Arguments #%d and #%d",  \
               (_p_arg1__),(_p_arg2__));                                \
    }                                                                   \
 } while (0)

#undef  PetscValidStreamType
#define PetscValidStreamType(_p_strm_type__,_p_arg__)  do {             \
    if (PetscUnlikely(((_p_strm_type__) < 0) ||                         \
                      ((_p_strm_type__) > PETSC_STREAM_MAX))) {         \
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,              \
               "Unknown PetscStreamType '%d': Argument #%d",            \
               (_p_strm_type__),(_p_arg__));                            \
    } else if (PetscUnlikely((_p_strm_type__) == PETSC_STREAM_MAX)) {   \
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,                    \
               "Invalid PetscStreamType '%s': Argument #%d",            \
               PetscStreamTypes[_p_strm_type__],(_p_arg__));            \
    }                                                                   \
  } while (0)

#undef  PetscValidDeviceContext
#define PetscValidDeviceContext(_p_dev_ctx__,_p_arg__) do {             \
    PetscValidPointer(_p_dev_ctx__,_p_arg__);                           \
    PetscValidStreamType((_p_dev_ctx__)->streamType,_p_arg__);          \
    if ((_p_dev_ctx__)->device) {                                       \
      PetscValidDevice((_p_dev_ctx__)->device,_p_arg__);                \
    } else if (PetscUnlikely((_p_dev_ctx__)->setup)) {                  \
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,                \
               "Invalid PetscDeviceContext: Argument #%d; "             \
               "PetscDeviceContext is setup but has no PetscDevice",    \
               (_p_arg__));                                             \
    }                                                                   \
    if (PetscUnlikely((_p_dev_ctx__)->id < 1)) {                        \
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,                          \
               "Invalid PetscDeviceContext: Argument #%d; "             \
               "id %" PetscInt_FMT " < 1",                              \
               (_p_arg__),(_p_dev_ctx__)->id);                          \
    } else if (PetscUnlikely((_p_dev_ctx__)->numChildren      >         \
                             (_p_dev_ctx__)->maxNumChildren)) {         \
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,                   \
               "Invalid PetscDeviceContext: Argument #%d; "             \
               "number of children %" PetscInt_FMT " > "                \
               "max number of children %" PetscInt_FMT,                 \
               (_p_arg__),(_p_dev_ctx__)->numChildren,                  \
               (_p_dev_ctx__)->maxNumChildren);                         \
    }                                                                   \
  } while (0)

#undef  PetscCheckCompatibleDeviceContexts
#define PetscCheckCompatibleDeviceContexts(_p_dev_ctx1__,_p_arg1__,_p_dev_ctx2__,_p_arg2__) \
  do {                                                                  \
    PetscValidDeviceContext(_p_dev_ctx1__,_p_arg1__);                   \
    PetscValidDeviceContext(_p_dev_ctx2__,_p_arg2__);                   \
    PetscCheckCompatibleDevices((_p_dev_ctx1__)->device,_p_arg1__,      \
                                (_p_dev_ctx2__)->device,_p_arg2__);     \
  } while (0)

/*  This header file should NEVER #include another file and should be the last thing included
 *  in the test file. This is to guard against ill-formed PetscDevice header files!
 */
PETSC_STATIC_INLINE PetscErrorCode AssertDeviceExists(PetscDevice device)
{
  PetscFunctionBegin;
  PetscValidDevice(device,1);
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode AssertDeviceDoesNotExist(PetscDevice device)
{
  PetscFunctionBegin;
  if (device) {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"PetscDevice was not destroyed for type %s",PetscDeviceTypes[device->type]);
  }
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode AssertDeviceContextExists(PetscDeviceContext dctx)
{
  PetscFunctionBegin;
  PetscValidDeviceContext(dctx,1);
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode AssertDeviceContextDoesNotExist(PetscDeviceContext dctx)
{
  PetscFunctionBegin;
  if (dctx) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"PetscDeviceContext was not destroyed");
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode AssertPetscStreamTypesValidAndEqual(PetscStreamType left, PetscStreamType right, const char *errStr)
{
  PetscFunctionBegin;
  PetscValidStreamType(left,1);
  PetscValidStreamType(right,2);
  if (left != right) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,errStr,PetscStreamTypes[left],PetscStreamTypes[right]);
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode AssertPetscDevicesValidAndEqual(PetscDevice left, PetscDevice right, const char *errStr)
{
  PetscFunctionBegin;
  PetscCheckCompatibleDevices(left,1,right,2);
  if (left != right) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,"%s",errStr);
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode AssertPetscDeviceContextsValidAndEqual(PetscDeviceContext left, PetscDeviceContext right, const char *errStr)
{
  PetscFunctionBegin;
  PetscCheckCompatibleDeviceContexts(left,1,right,2);
  if (left != right) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"%s",errStr);
  PetscFunctionReturn(0);
}
#endif /* PETSCDEVICETESTCOMMON_H */
