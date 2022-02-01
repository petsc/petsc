#ifndef PETSCDEVICEIMPL_H
#define PETSCDEVICEIMPL_H

#include <petsc/private/petscimpl.h>
#include <petscdevice.h>

#if defined(PETSC_CLANG_STATIC_ANALYZER)
template <typename T> void PetscValidDeviceType(T,int);
template <typename T> void PetscValidDevice(T,int);
template <typename T> void PetscCheckCompatibleDevices(T,int,T,int);
template <typename T> void PetscValidStreamType(T,int);
template <typename T> void PetscValidDeviceContext(T,int);
template <typename T> void PetscCheckCompatibleDeviceContexts(T,int,T,int);
#elif PetscDefined(USE_DEBUG)
/* note any changes to these macros must be mirrored in
 * src/sys/objects/device/test/petscdevicecommon.h! */
#define PetscValidDeviceType(_p_dev_type__,_p_arg__) do {               \
    if (PetscUnlikely(((_p_dev_type__) < PETSC_DEVICE_INVALID) ||       \
                      ((_p_dev_type__) > PETSC_DEVICE_MAX))) {          \
      SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,              \
               "Unknown PetscDeviceType '%d': Argument #%d",            \
               (_p_dev_type__),(_p_arg__));                             \
    } else if (PetscUnlikely(!PetscDeviceConfiguredFor_Internal(_p_dev_type__))) { \
      switch(_p_dev_type__) {                                           \
      case PETSC_DEVICE_INVALID:                                        \
        SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_SUP,                         \
                 "Invalid PetscDeviceType '%s': Argument #%d;"          \
                 " PETSc is not configured with device support",        \
                 PetscDeviceTypes[_p_dev_type__],(_p_arg__));           \
        break;                                                          \
      case PETSC_DEVICE_MAX:                                            \
        SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,                  \
                 "Invalid PetscDeviceType '%s': Argument #%d",          \
                 PetscDeviceTypes[_p_dev_type__],(_p_arg__));           \
        break;                                                          \
      default:                                                          \
        SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_SUP,                         \
                 "Not configured for PetscDeviceType '%s': Argument #%d;" \
                 " run configure --help %s for available options",      \
                 PetscDeviceTypes[_p_dev_type__],(_p_arg__),            \
                 PetscDeviceTypes[_p_dev_type__]);                      \
        break;                                                          \
      }                                                                 \
    }                                                                   \
  } while (0)

#define PetscValidDevice(_p_dev__,_p_arg__)          do {       \
    PetscValidPointer(_p_dev__,_p_arg__);                       \
    PetscValidDeviceType((_p_dev__)->type,_p_arg__);            \
    if (PetscUnlikely((_p_dev__)->id < 0)) {                    \
      SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,                  \
               "Invalid PetscDevice: Argument #%d; "            \
               "id %" PetscInt_FMT " < 0",                      \
               (_p_arg__),(_p_dev__)->id);                      \
    } else if (PetscUnlikely((_p_dev__)->refcnt < 0)) {         \
      SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,                  \
               "Invalid PetscDevice: Argument #%d; "            \
               "negative reference count %" PetscInt_FMT,       \
               (_p_arg__),(_p_dev__)->refcnt);                  \
    }                                                           \
  } while (0)

/* for now just checks strict equality, but this can be changed as some devices (i.e. kokkos
 * and any cupm should be compatible once implemented) */
#define PetscCheckCompatibleDevices(_p_dev1__,_p_arg1__,_p_dev2__,_p_arg2__) \
  do {                                                                  \
    PetscValidDevice(_p_dev1__,_p_arg1__);                              \
    PetscValidDevice(_p_dev2__,_p_arg2__);                              \
    if (PetscUnlikely((_p_dev1__)->type != (_p_dev2__)->type)) {        \
      SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,                    \
               "PetscDevices are incompatible: Arguments #%d and #%d",  \
               (_p_arg1__),(_p_arg2__));                                \
    }                                                                   \
 } while (0)

#define PetscValidStreamType(_p_strm_type__,_p_arg__)  do {             \
    if (PetscUnlikely(((_p_strm_type__) < 0) ||                         \
                      ((_p_strm_type__) > PETSC_STREAM_MAX))) {         \
      SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,              \
               "Unknown PetscStreamType '%d': Argument #%d",            \
               (_p_strm_type__),(_p_arg__));                            \
    } else if (PetscUnlikely((_p_strm_type__) == PETSC_STREAM_MAX)) {   \
      SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,                    \
               "Invalid PetscStreamType '%s': Argument #%d",            \
               PetscStreamTypes[_p_strm_type__],(_p_arg__));            \
    }                                                                   \
  } while (0)

#define PetscValidDeviceContext(_p_dev_ctx__,_p_arg__) do {             \
    PetscValidPointer(_p_dev_ctx__,_p_arg__);                           \
    PetscValidStreamType((_p_dev_ctx__)->streamType,_p_arg__);          \
    if ((_p_dev_ctx__)->device) {                                       \
      PetscValidDevice((_p_dev_ctx__)->device,_p_arg__);                \
    } else if (PetscUnlikely((_p_dev_ctx__)->setup)) {                  \
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,                \
               "Invalid PetscDeviceContext: Argument #%d; "             \
               "PetscDeviceContext is setup but has no PetscDevice",    \
               (_p_arg__));                                             \
    }                                                                   \
    if (PetscUnlikely((_p_dev_ctx__)->id < 1)) {                        \
      SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,                          \
               "Invalid PetscDeviceContext: Argument #%d; "             \
               "id %" PetscInt_FMT " < 1",                              \
               (_p_arg__),(_p_dev_ctx__)->id);                          \
    } else if (PetscUnlikely((_p_dev_ctx__)->numChildren      >         \
                             (_p_dev_ctx__)->maxNumChildren)) {         \
      SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,                   \
               "Invalid PetscDeviceContext: Argument #%d; "             \
               "number of children %" PetscInt_FMT " > "                \
               "max number of children %" PetscInt_FMT,                 \
               (_p_arg__),(_p_dev_ctx__)->numChildren,                  \
               (_p_dev_ctx__)->maxNumChildren);                         \
    }                                                                   \
  } while (0)

#define PetscCheckCompatibleDeviceContexts(_p_dev_ctx1__,_p_arg1__,_p_dev_ctx2__,_p_arg2__) \
  do {                                                                  \
    PetscValidDeviceContext(_p_dev_ctx1__,_p_arg1__);                   \
    PetscValidDeviceContext(_p_dev_ctx2__,_p_arg2__);                   \
    PetscCheckCompatibleDevices((_p_dev_ctx1__)->device,_p_arg1__,      \
                                (_p_dev_ctx2__)->device,_p_arg2__);     \
  } while (0)

#else /* PetscDefined(USE_DEBUG) */
#define PetscValidDeviceType(_p_dev_type__,_p_arg__)
#define PetscValidDevice(_p_dev__,_p_arg__)
#define PetscCheckCompatibleDevices(_p_dev1__,_p_arg1__,_p_dev2__,_p_arg2__)
#define PetscValidStreamType(_p_strm_type__,_p_arg__)
#define PetscValidDeviceContext(_p_dev_ctx__,_p_arg__)
#define PetscCheckCompatibleDeviceContexts(_p_dev_ctx1__,_p_arg1__,_p_dev_ctx2__,_p_arg2__)
#endif /* PetscDefined(USE_DEBUG) */

/* if someone is ready to rock with more than 128 GPUs on hand then we're in real trouble */
#define PETSC_DEVICE_MAX_DEVICES 128

typedef struct _DeviceOps *DeviceOps;
struct _DeviceOps {
  /* the creation routine for the corresponding PetscDeviceContext, this is NOT intended to be
   * called by the PetscDevice itself */
  PetscErrorCode (*createcontext)(PetscDeviceContext);
  PetscErrorCode (*configure)(PetscDevice);
  PetscErrorCode (*view)(PetscDevice,PetscViewer);
};

struct _n_PetscDevice {
  struct _DeviceOps ops[1];
  PetscInt          refcnt;   /* reference count for the device */
  PetscInt          id;       /* unique id per created PetscDevice */
  PetscInt          deviceId; /* the id of the underlying device, i.e. the return of
                               * cudaGetDevice() for example */
  PetscDeviceType   type;     /* type of device */
  void             *data;     /* placeholder */
};

typedef struct _DeviceContextOps *DeviceContextOps;
struct _DeviceContextOps {
  PetscErrorCode (*destroy)(PetscDeviceContext);
  PetscErrorCode (*changestreamtype)(PetscDeviceContext,PetscStreamType);
  PetscErrorCode (*setup)(PetscDeviceContext);
  PetscErrorCode (*query)(PetscDeviceContext,PetscBool*);
  PetscErrorCode (*waitforcontext)(PetscDeviceContext,PetscDeviceContext);
  PetscErrorCode (*synchronize)(PetscDeviceContext);
  PetscErrorCode (*getblashandle)(PetscDeviceContext,void*);
  PetscErrorCode (*getsolverhandle)(PetscDeviceContext,void*);
  PetscErrorCode (*getstreamhandle)(PetscDeviceContext,void*);
  PetscErrorCode (*begintimer)(PetscDeviceContext);
  PetscErrorCode (*endtimer)(PetscDeviceContext,PetscLogDouble*);
};

struct _n_PetscDeviceContext {
  struct _DeviceContextOps  ops[1];
  PetscDevice               device;         /* the device this context stems from */
  void                     *data;           /* solver contexts, event, stream */
  PetscInt                  id;             /* unique id per created context */
  PetscInt                 *childIDs;       /* array containing ids of contexts currently forked from this one */
  PetscInt                  numChildren;    /* how many children does this context expect to destroy */
  PetscInt                  maxNumChildren; /* how many children can this context have room for without realloc'ing */
  PetscStreamType           streamType;     /* how should this contexts stream behave around other streams? */
  PetscBool                 idle;           /* does this context think it has work? this value non-binding in debug mode */
  PetscBool                 setup;
};

/* PetscDevice Internal Functions */
#if PetscDefined(HAVE_CXX_DIALECT_CXX11)
PETSC_INTERN PetscErrorCode PetscDeviceInitializeFromOptions_Internal(MPI_Comm);
PETSC_SINGLE_LIBRARY_INTERN PetscErrorCode PetscDeviceInitializeDefaultDevice_Internal(PetscDeviceType,PetscInt);
PETSC_SINGLE_LIBRARY_INTERN PetscErrorCode PetscDeviceGetDefaultForType_Internal(PetscDeviceType,PetscDevice*);

PETSC_STATIC_INLINE PETSC_CONSTEXPR_14 PetscBool PetscDeviceConfiguredFor_Internal(PetscDeviceType type)
{
  switch(type) {
  case PETSC_DEVICE_INVALID: return PETSC_FALSE;
    /* casts are needed in C++ */
  case PETSC_DEVICE_CUDA:    return (PetscBool)PetscDefined(HAVE_CUDA);
  case PETSC_DEVICE_HIP:     return (PetscBool)PetscDefined(HAVE_HIP);
  case PETSC_DEVICE_SYCL:    return (PetscBool)PetscDefined(HAVE_SYCL);
  case PETSC_DEVICE_MAX:     return PETSC_FALSE;
    /* Do not add default case! Will make compiler warn on new additions to PetscDeviceType! */
  }
  PetscUnreachable();
  return PETSC_FALSE;
}

/* More general form of PetscDeviceDefaultType_Internal(), as it calls the former using the
 * automatically selected default PetscDeviceType */
#define PetscDeviceGetDefault_Internal(device)                          \
  PetscDeviceGetDefaultForType_Internal(PETSC_DEVICE_DEFAULT,device)

PETSC_STATIC_INLINE PetscErrorCode PetscDeviceCheckDeviceCount_Internal(PetscInt count)
{
  PetscFunctionBegin;
  if (PetscUnlikelyDebug(count >= PETSC_DEVICE_MAX_DEVICES)) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Detected %" PetscInt_FMT " devices, which is larger than maximum supported number of devices %d",count,PETSC_DEVICE_MAX_DEVICES);
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode  PetscDeviceReference_Internal(PetscDevice device)
{
  PetscFunctionBegin;
  ++(device->refcnt);
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscDeviceDereference_Internal(PetscDevice device)
{
  PetscFunctionBegin;
  --(device->refcnt);
  if (PetscUnlikelyDebug(device->refcnt < 0)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,"PetscDevice has negative reference count %" PetscInt_FMT,device->refcnt);
  PetscFunctionReturn(0);
}
#else /* PetscDefined(HAVE_CXX_DIALECT_CXX11) */
#define PetscDeviceInitializeFromOptions_Internal(comm)      0
#define PetscDeviceInitializeDefaultDevice_Internal(type,id) 0
#define PetscDeviceConfiguredFor_Internal(type)              PETSC_FALSE
#define PetscDeviceGetDefaultForType_Internal(Type,device)   0
#define PetscDeviceGetDefault_Internal(device)               0
#define PetscDeviceCheckDeviceCount_Internal(count)          0
#define PetscDeviceReference_Internal(device)                0
#define PetscDeviceDereference_Internal(device)              0
#endif /* PetscDefined(HAVE_CXX_DIALECT_CXX11) */

/* PetscDeviceContext Internal Functions */
#if PetscDefined(HAVE_CXX_DIALECT_CXX11)
PETSC_INTERN PetscErrorCode PetscDeviceContextSetRootDeviceType_Internal(PetscDeviceType);

/* Called in debug-mode when a context claims it is idle to check that it isn't lying. A no-op
 * in performance-mode */
PETSC_STATIC_INLINE PetscErrorCode PetscDeviceContextValidateIdle_Internal(PetscDeviceContext dctx)
{
  PetscFunctionBegin;
  if (PetscDefined(USE_DEBUG)) {
    const PetscBool idleBefore = dctx->idle;
    PetscBool       idle;
    PetscErrorCode  ierr;

    PetscValidDeviceContext(dctx,1);
    ierr = (*dctx->ops->query)(dctx,&idle);CHKERRQ(ierr);
    if (PetscUnlikely(idleBefore && !idle)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"PetscDeviceContext cache corrupted, context %" PetscInt_FMT " thought it was idle when it still had work",dctx->id);
  }
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscDeviceContextSetDefaultDeviceForType_Internal(PetscDeviceContext dctx, PetscDeviceType type)
{
  PetscDevice    device;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscDeviceGetDefaultForType_Internal(type,&device);CHKERRQ(ierr);
  ierr = PetscDeviceContextSetDevice(dctx,device);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#define PetscDeviceContextSetDefaultDevice_Internal(dctx)               \
  PetscDeviceContextSetDefaultDeviceForType_Internal(dctx,PETSC_DEVICE_DEFAULT)

/* note, only does assertion checking in debug mode */
PETSC_STATIC_INLINE PetscErrorCode PetscDeviceContextGetCurrentContextAssertType_Internal(PetscDeviceContext *dctx, PetscDeviceType type)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(dctx,1);
  PetscValidDeviceType(type,2);
  ierr = PetscDeviceContextGetCurrentContext(dctx);CHKERRQ(ierr);
  if (PetscUnlikelyDebug((*dctx)->device->type != type)) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Expected current global PetscDeviceContext (id %" PetscInt_FMT ") to have PetscDeviceType '%s' but has '%s' instead",(*dctx)->id,PetscDeviceTypes[type],PetscDeviceTypes[(*dctx)->device->type]);
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscDeviceContextGetBLASHandle_Internal(PetscDeviceContext dctx, void *handle)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* we do error checking here as this routine is an entry-point */
  PetscValidDeviceContext(dctx,1);
  PetscValidPointer(handle,2);
  ierr = (*dctx->ops->getblashandle)(dctx,handle);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscDeviceContextGetSOLVERHandle_Internal(PetscDeviceContext dctx, void *handle)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* we do error checking here as this routine is an entry-point */
  PetscValidDeviceContext(dctx,1);
  PetscValidPointer(handle,2);
  ierr = (*dctx->ops->getsolverhandle)(dctx,handle);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscDeviceContextGetStreamHandle_Internal(PetscDeviceContext dctx, void *handle)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* we do error checking here as this routine is an entry-point */
  PetscValidDeviceContext(dctx,1);
  PetscValidPointer(handle,2);
  ierr = (*dctx->ops->getstreamhandle)(dctx,handle);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscDeviceContextBeginTimer_Internal(PetscDeviceContext dctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* we do error checking here as this routine is an entry-point */
  PetscValidDeviceContext(dctx,1);
  ierr = (*dctx->ops->begintimer)(dctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscDeviceContextEndTimer_Internal(PetscDeviceContext dctx, PetscLogDouble *elapsed)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* we do error checking here as this routine is an entry-point */
  PetscValidDeviceContext(dctx,1);
  PetscValidRealPointer(elapsed,2);
  ierr = (*dctx->ops->endtimer)(dctx,elapsed);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#else /* PetscDefined(HAVE_CXX_DIALECT_CXX11) */
#define PetscDeviceContextSetRootDeviceType_Internal(type)                0
#define PetscDeviceContextValidateIdle_Internal(dctx)                     0
#define PetscDeviceContextSetDefaultDeviceForType_Internal(dctx,type)     0
#define PetscDeviceContextSetDefaultDevice_Internal(dctx)                 0
#define PetscDeviceContextGetCurrentContextAssertType_Internal(dctx,type) 0
#define PetscDeviceContextGetBLASHandle_Internal(dctx,handle)             0
#define PetscDeviceContextGetSOLVERHandle_Internal(dctx,handle)           0
#define PetscDeviceContextGetStreamHandle_Internal(dctx,handle)           0
#define PetscDeviceContextBeginTimer_Internal(dctx)                       0
#define PetscDeviceContextEndTimer_Internal(dctx,elapsed)                 0
#endif /* PetscDefined(HAVE_CXX_DIALECT_CXX11) */

#if PetscDefined(HAVE_CUDA)
PETSC_INTERN PetscErrorCode PetscDeviceContextCreate_CUDA(PetscDeviceContext);
#endif
#if PetscDefined(HAVE_HIP)
PETSC_INTERN PetscErrorCode PetscDeviceContextCreate_HIP(PetscDeviceContext);
#endif
#if PetscDefined(HAVE_SYCL)
PETSC_INTERN PetscErrorCode PetscDeviceContextCreate_SYCL(PetscDeviceContext);
#endif

#endif /* PETSCDEVICEIMPL_H */
