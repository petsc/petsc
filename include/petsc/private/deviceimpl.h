#if !defined(PETSC_DEVICEIMPL_H)
#define PETSC_DEVICEIMPL_H

#include <petsc/private/petscimpl.h>
#include <petscdevice.h>

#if defined(PETSC_CLANG_STATIC_ANALYZER)
void PetscValidDeviceKind(int,int);
template <typename T>
void PetscValidDevice(T,int);
template <typename T>
void PetscCheckCompatibleDevices(T,int,T,int);
void PetscValidStreamType(int,int);
template <typename T>
void PetscValidDeviceContext(T,int);
template <typename T>
void PetscCheckCompatibleDeviceContexts(T,int,T,int);
#else /* PETSC_CLANG_STATIC_ANALYZER */
#if PetscDefined(USE_DEBUG)
#define PetscValidDeviceKind(_p_dev_kind__,_p_arg__)                    \
  do {                                                                  \
    if (PetscUnlikely(((_p_dev_kind__) < PETSC_DEVICE_INVALID) ||       \
                      ((_p_dev_kind__) > PETSC_DEVICE_MAX))) {          \
      SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,              \
               "Unknown PetscDeviceKind '%d': Argument #%d",            \
               (_p_dev_kind__),(_p_arg__));                             \
    } else if (PetscUnlikely(((_p_dev_kind__) == PETSC_DEVICE_INVALID) || \
                             ((_p_dev_kind__) == PETSC_DEVICE_MAX))) {  \
      SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,                    \
               "Invalid PetscDeviceKind '%s': Argument #%d",            \
               PetscDeviceKinds[_p_dev_kind__],(_p_arg__));             \
    }                                                                   \
  } while (0)

#define PetscValidDevice(_p_dev__,_p_arg__)                             \
  do {                                                                  \
    PetscValidPointer(_p_dev__,_p_arg__);                               \
    PetscValidDeviceKind((_p_dev__)->kind,_p_arg__);                    \
  } while (0)

/* for now just checks strict equality, but this can be changed as some devices
   (i.e. kokkos and any cupm should be compatible once implemented) */
#define PetscCheckCompatibleDevices(_p_dev1__,_p_arg1__,_p_dev2__,_p_arg2__) \
  do {                                                                  \
    PetscValidDevice(_p_dev1__,_p_arg1__);                              \
    PetscValidDevice(_p_dev2__,_p_arg2__);                              \
    if (PetscUnlikely((_p_dev1__)->kind != (_p_dev2__)->kind)) {        \
      SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,                    \
               "PetscDevices are incompatible: Arguments #%d and #%d",  \
               (_p_arg1__),(_p_arg2__));                                \
    }                                                                   \
 } while (0)

#define PetscValidStreamType(_p_strm_type__,_p_arg__)                   \
  do {                                                                  \
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

#define PetscValidDeviceContext(_p_dev_ctx__,_p_arg__)              \
  do {                                                              \
    PetscValidPointer(_p_dev_ctx__,_p_arg__);                       \
    if ((_p_dev_ctx__)->device) {                                   \
      PetscValidDevice((_p_dev_ctx__)->device,_p_arg__);            \
    }                                                               \
    PetscValidStreamType((_p_dev_ctx__)->streamType,_p_arg__);      \
  } while (0)

#define PetscCheckCompatibleDeviceContexts(_p_dev_ctx1__,_p_arg1__,_p_dev_ctx2__,_p_arg2__) \
  do {                                                                  \
    PetscValidDeviceContext(_p_dev_ctx1__,_p_arg1__);                   \
    PetscValidDeviceContext(_p_dev_ctx2__,_p_arg2__);                   \
    PetscCheckCompatibleDevices((_p_dev_ctx1__)->device,_p_arg1__,      \
                                (_p_dev_ctx2__)->device,_p_arg2__);     \
  } while (0)

#else /* PETSC_USE_DEBUG */
#define PetscValidDeviceKind(_p_dev_kind__,_p_arg__)
#define PetscValidDevice(_p_dev__,_p_arg__)
#define PetscCheckCompatibleDevices(_p_dev1__,_p_arg1__,_p_dev2__,_p_arg2__)
#define PetscValidStreamType(_p_strm_type__,_p_arg__)
#define PetscValidDeviceContext(_p_dev_ctx__,_p_arg__)
#define PetscCheckCompatibleDeviceContexts(_p_dev_ctx1__,_p_arg1__,_p_dev_ctx2__,_p_arg2__)
#endif /* PETSC_USE_DEBUG */
#endif /* PETSC_CLANG_STATIC_ANALYZER */

typedef struct _DeviceOps *DeviceOps;
struct _DeviceOps {
  /* the creation routine for the corresponding PetscDeviceContext, this is NOT intended
     to be called by the PetscDevice itself */
  PetscErrorCode (*createcontext)(PetscDeviceContext);
};

struct _n_PetscDevice {
  struct _DeviceOps ops[1];
  PetscInt          refcnt;   /* reference count for the device */
  PetscInt          id;       /* unique id per created PetscDevice */
  PetscDeviceKind   kind;     /* kind of device */
  int               deviceId; /* the id of the underlying device, i.e. the return of
                                 cudaGetDevice() for example */
  void             *data;     /* placeholder */
};

typedef struct _DeviceContextOps *DeviceContextOps;
struct _DeviceContextOps {
  PetscErrorCode (*destroy)(PetscDeviceContext);
  PetscErrorCode (*changestreamtype)(PetscDeviceContext,PetscStreamType);
  PetscErrorCode (*setup)(PetscDeviceContext);
  PetscErrorCode (*query)(PetscDeviceContext,PetscBool*);
  PetscErrorCode (*waitforctx)(PetscDeviceContext,PetscDeviceContext);
  PetscErrorCode (*synchronize)(PetscDeviceContext);
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
PETSC_INTERN PetscErrorCode PetscDeviceInitializeDefaultDevices_Internal(void);
PETSC_INTERN PetscDevice    PetscDeviceDefaultKind_Internal(PetscDeviceKind);

PETSC_STATIC_INLINE PetscDevice PetscDeviceReference(PetscDevice device)
{
  PetscFunctionBegin;
  ++(device->refcnt);
  PetscFunctionReturn(device);
}

/* More general form of PetscDeviceDefaultKind_Internal(), as it calls the former using
   the automatically selected default PetscDeviceKind */
PETSC_STATIC_INLINE PetscDevice PetscDeviceDefault_Internal(void)
{
  return PetscDeviceDefaultKind_Internal(PETSC_DEVICE_DEFAULT);
}

/* PetscDeviceContext Internal Functions */
PETSC_INTERN PetscErrorCode PetscDeviceContextInitializeRootContext_Internal(MPI_Comm,const char[]);
/* Called in debug-mode when a context claims it is idle to check that it isn't lying. A
   no-op when debugging is disabled */
PETSC_STATIC_INLINE PetscErrorCode PetscDeviceContextValidateIdle_Internal(PetscDeviceContext dctx)
{
  PetscFunctionBegin;
  if (PetscDefined(USE_DEBUG)) {
    PetscBool      idle;
    PetscErrorCode ierr;

    ierr = (*dctx->ops->query)(dctx,&idle);CHKERRQ(ierr);
    if (PetscUnlikely(dctx->idle && !idle)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"PetscDeviceContext cache corrupted, context %D thought it was idle when it still had work",dctx->id);
  }
  PetscFunctionReturn(0);
}

#if PetscDefined(HAVE_CUDA)
PETSC_INTERN PetscErrorCode PetscDeviceContextCreate_CUDA(PetscDeviceContext);
#endif
#if PetscDefined(HAVE_HIP)
PETSC_INTERN PetscErrorCode PetscDeviceContextCreate_HIP(PetscDeviceContext);
#endif

#endif /* PETSC_DEVICEIMPL_H */
