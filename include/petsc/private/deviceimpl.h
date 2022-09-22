#ifndef PETSCDEVICEIMPL_H
#define PETSCDEVICEIMPL_H

#include <petscdevice.h>
#include <petsc/private/petscimpl.h>

/* logging support */
PETSC_INTERN PetscLogEvent CUBLAS_HANDLE_CREATE;
PETSC_INTERN PetscLogEvent CUSOLVER_HANDLE_CREATE;
PETSC_INTERN PetscLogEvent HIPSOLVER_HANDLE_CREATE;
PETSC_INTERN PetscLogEvent HIPBLAS_HANDLE_CREATE;

PETSC_INTERN PetscLogEvent DCONTEXT_Create;
PETSC_INTERN PetscLogEvent DCONTEXT_Destroy;
PETSC_INTERN PetscLogEvent DCONTEXT_ChangeStream;
PETSC_INTERN PetscLogEvent DCONTEXT_SetDevice;
PETSC_INTERN PetscLogEvent DCONTEXT_SetUp;
PETSC_INTERN PetscLogEvent DCONTEXT_Duplicate;
PETSC_INTERN PetscLogEvent DCONTEXT_QueryIdle;
PETSC_INTERN PetscLogEvent DCONTEXT_WaitForCtx;
PETSC_INTERN PetscLogEvent DCONTEXT_Fork;
PETSC_INTERN PetscLogEvent DCONTEXT_Join;
PETSC_INTERN PetscLogEvent DCONTEXT_Sync;
PETSC_INTERN PetscLogEvent DCONTEXT_Mark;

/* type cast macros for some additional type-safety in C++ land */
#if defined(__cplusplus)
  #define PetscStreamTypeCast(...)     static_cast<PetscStreamType>(__VA_ARGS__)
  #define PetscDeviceTypeCast(...)     static_cast<PetscDeviceType>(__VA_ARGS__)
  #define PetscDeviceInitTypeCast(...) static_cast<PetscDeviceInitType>(__VA_ARGS__)
#else
  #define PetscStreamTypeCast(...)     ((PetscStreamType)(__VA_ARGS__))
  #define PetscDeviceTypeCast(...)     ((PetscDeviceType)(__VA_ARGS__))
  #define PetscDeviceInitTypeCast(...) ((PetscDeviceInitType)(__VA_ARGS__))
#endif

#if defined(PETSC_CLANG_STATIC_ANALYZER)
template <typename T>
void PetscValidDeviceType(T, int);
template <typename T, typename U>
void PetscCheckCompatibleDeviceTypes(T, int, U, int);
template <typename T>
void PetscValidDevice(T, int);
template <typename T>
void PetscValidDeviceAttribute(T, int);
template <typename T, typename U>
void PetscCheckCompatibleDevices(T, int, U, int);
template <typename T>
void PetscValidStreamType(T, int);
template <typename T>
void PetscValidDeviceContext(T, int);
template <typename T, typename U>
void PetscCheckCompatibleDeviceContexts(T, int, U, int);
#elif PetscDefined(HAVE_CXX) && (PetscDefined(USE_DEBUG) || PetscDefined(DEVICE_KEEP_ERROR_CHECKING_MACROS))
  #define PetscValidDeviceType(dtype, argno) \
    do { \
      PetscDeviceType pvdt_dtype_ = PetscDeviceTypeCast(dtype); \
      int             pvdt_argno_ = (int)(argno); \
      PetscCheck(((int)pvdt_dtype_ >= (int)PETSC_DEVICE_HOST) && ((int)pvdt_dtype_ <= (int)PETSC_DEVICE_MAX), PETSC_COMM_SELF, PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown PetscDeviceType '%d': Argument #%d", pvdt_dtype_, pvdt_argno_); \
      if (PetscUnlikely(!PetscDeviceConfiguredFor_Internal(pvdt_dtype_))) { \
        PetscCheck((int)pvdt_dtype_ != (int)PETSC_DEVICE_MAX, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Invalid PetscDeviceType '%s': Argument #%d", PetscDeviceTypes[pvdt_dtype_], pvdt_argno_); \
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, \
                "Not configured for PetscDeviceType '%s': Argument #%d;" \
                " run configure --help %s for available options", \
                PetscDeviceTypes[pvdt_dtype_], pvdt_argno_, PetscDeviceTypes[pvdt_dtype_]); \
      } \
    } while (0)

  #define PetscCheckCompatibleDeviceTypes(dtype1, argno1, dtype2, argno2) \
    do { \
      PetscDeviceType pccdt_dtype1_ = PetscDeviceTypeCast(dtype1); \
      PetscDeviceType pccdt_dtype2_ = PetscDeviceTypeCast(dtype2); \
      PetscValidDeviceType(pccdt_dtype1_, 1); \
      PetscValidDeviceType(pccdt_dtype2_, 2); \
      PetscCheck(pccdt_dtype1_ == pccdt_dtype2_, PETSC_COMM_SELF, PETSC_ERR_ARG_NOTSAMETYPE, "PetscDeviceTypes are incompatible: Arguments #%d and #%d. Expected PetscDeviceType '%s' but have '%s' instead", argno1, argno2, PetscDeviceTypes[pccdt_dtype1_], PetscDeviceTypes[pccdt_dtype2_]); \
    } while (0)

  #define PetscValidDevice(dev, argno) \
    do { \
      PetscDevice pvd_dev_   = dev; \
      int         pvd_argno_ = (int)(argno); \
      PetscValidPointer(pvd_dev_, pvd_argno_); \
      PetscValidDeviceType(pvd_dev_->type, pvd_argno_); \
      PetscCheck(pvd_dev_->id >= 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid PetscDevice: Argument #%d; id %" PetscInt_FMT " < 0", pvd_argno_, pvd_dev_->id); \
      PetscCheck(pvd_dev_->refcnt >= 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid PetscDevice: Argument #%d; negative reference count %" PetscInt_FMT, pvd_argno_, pvd_dev_->refcnt); \
    } while (0)

  #define PetscValidDeviceAttribute(dattr, argno) \
    do { \
      PetscDeviceAttribute pvda_attr_  = (dattr); \
      int                  pvda_argno_ = (int)(argno); \
      PetscCheck((((int)pvda_attr_) >= 0) && (pvda_attr_ <= PETSC_DEVICE_ATTR_MAX), PETSC_COMM_SELF, PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown PetscDeviceAttribute '%d': Argument #%d", (int)pvda_attr_, pvda_argno_); \
      PetscCheck(pvda_attr_ != PETSC_DEVICE_ATTR_MAX, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Invalid PetscDeviceAttribute '%s': Argument #%d", PetscDeviceAttributes[pvda_attr_], pvda_argno_); \
    } while (0)

  /*
  for now just checks strict equality, but this can be changed as some devices (i.e. kokkos and
  any cupm should be compatible once implemented)
*/
  #define PetscCheckCompatibleDevices(dev1, argno1, dev2, argno2) \
    do { \
      PetscDevice pccd_dev1_ = (dev1), pccd_dev2_ = (dev2); \
      int         pccd_argno1_ = (int)(argno1), pccd_argno2_ = (int)(argno2); \
      PetscValidDevice(pccd_dev1_, pccd_argno1_); \
      PetscValidDevice(pccd_dev2_, pccd_argno2_); \
      PetscCheckCompatibleDeviceTypes(pccd_dev1_->type, pccd_argno1_, pccd_dev2_->type, pccd_argno2_); \
    } while (0)

  #define PetscValidStreamType(stype, argno) \
    do { \
      PetscStreamType pvst_stype_ = PetscStreamTypeCast(stype); \
      int             pvst_argno_ = (int)(argno); \
      PetscCheck(((int)pvst_stype_ >= 0) && ((int)pvst_stype_ <= (int)PETSC_STREAM_MAX), PETSC_COMM_SELF, PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown PetscStreamType '%d': Argument #%d", pvst_stype_, pvst_argno_); \
      PetscCheck((int)pvst_stype_ != (int)PETSC_STREAM_MAX, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Invalid PetscStreamType '%s': Argument #%d", PetscStreamTypes[pvst_stype_], pvst_argno_); \
    } while (0)

  #define PetscValidDeviceContext(dctx, argno) \
    do { \
      PetscDeviceContext pvdc_dctx_  = dctx; \
      int                pvdc_argno_ = (int)(argno); \
      PetscValidHeaderSpecific(pvdc_dctx_, PETSC_DEVICE_CONTEXT_CLASSID, pvdc_argno_); \
      PetscValidStreamType(pvdc_dctx_->streamType, pvdc_argno_); \
      if (pvdc_dctx_->device) { \
        PetscValidDevice(pvdc_dctx_->device, pvdc_argno_); \
      } else { \
        PetscCheck(!pvdc_dctx_->setup, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, \
                   "Invalid PetscDeviceContext: Argument #%d; " \
                   "PetscDeviceContext is setup but has no PetscDevice", \
                   pvdc_argno_); \
      } \
      PetscCheck(((PetscObject)pvdc_dctx_)->id >= 1, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid PetscDeviceContext: Argument #%d; id %" PetscInt64_FMT " < 1", pvdc_argno_, ((PetscObject)pvdc_dctx_)->id); \
      PetscCheck(pvdc_dctx_->numChildren <= pvdc_dctx_->maxNumChildren, PETSC_COMM_SELF, PETSC_ERR_ARG_CORRUPT, "Invalid PetscDeviceContext: Argument #%d; number of children %" PetscInt_FMT " > max number of children %" PetscInt_FMT, pvdc_argno_, \
                 pvdc_dctx_->numChildren, pvdc_dctx_->maxNumChildren); \
    } while (0)

  #define PetscCheckCompatibleDeviceContexts(dctx1, argno1, dctx2, argno2) \
    do { \
      PetscDeviceContext pccdc_dctx1_ = (dctx1), pccdc_dctx2_ = (dctx2); \
      int                pccdc_argno1_ = (int)(argno1), pccdc_argno2_ = (int)(argno2); \
      PetscValidDeviceContext(pccdc_dctx1_, pccdc_argno1_); \
      PetscValidDeviceContext(pccdc_dctx2_, pccdc_argno2_); \
      if (pccdc_dctx1_->device && pccdc_dctx2_->device) PetscCheckCompatibleDevices(pccdc_dctx1_->device, pccdc_argno1_, pccdc_dctx2_->device, pccdc_argno2_); \
    } while (0)
#else /* PetscDefined(USE_DEBUG) */
  #define PetscValidDeviceType(dtype, argno)
  #define PetscCheckCompatibleDeviceTypes(dtype1, argno1, dtype2, argno2)
  #define PetscValidDeviceAttribute(dattr, argno)
  #define PetscValidDevice(dev, argno)
  #define PetscCheckCompatibleDevices(dev1, argno1, dev2, argno2)
  #define PetscValidStreamType(stype, argno)
  #define PetscValidDeviceContext(dctx, argno)
  #define PetscCheckCompatibleDeviceContexts(dctx1, argno1, dctx2, argno2)
#endif /* PetscDefined(USE_DEBUG) */

/* if someone is ready to rock with more than 128 GPUs on hand then we're in real trouble */
#define PETSC_DEVICE_MAX_DEVICES 128

/*
  the configure-time default device type, used as the initial the value of
  PETSC_DEVICE_DEFAULT() as well as what it is restored to during PetscFinalize()
*/
#if PetscDefined(HAVE_HIP)
  #define PETSC_DEVICE_HARDWARE_DEFAULT_TYPE PETSC_DEVICE_HIP
#elif PetscDefined(HAVE_CUDA)
  #define PETSC_DEVICE_HARDWARE_DEFAULT_TYPE PETSC_DEVICE_CUDA
#elif PetscDefined(HAVE_SYCL)
  #define PETSC_DEVICE_HARDWARE_DEFAULT_TYPE PETSC_DEVICE_SYCL
#else
  #define PETSC_DEVICE_HARDWARE_DEFAULT_TYPE PETSC_DEVICE_HOST
#endif

#define PETSC_DEVICE_CONTEXT_DEFAULT_DEVICE_TYPE PETSC_DEVICE_HARDWARE_DEFAULT_TYPE
// REMOVE ME (change)
#define PETSC_DEVICE_CONTEXT_DEFAULT_STREAM_TYPE PETSC_STREAM_GLOBAL_BLOCKING

typedef struct _DeviceOps *DeviceOps;
struct _DeviceOps {
  /* the creation routine for the corresponding PetscDeviceContext, this is NOT intended
   * to be called by the PetscDevice itself */
  PetscErrorCode (*createcontext)(PetscDeviceContext);
  PetscErrorCode (*configure)(PetscDevice);
  PetscErrorCode (*view)(PetscDevice, PetscViewer);
  PetscErrorCode (*getattribute)(PetscDevice, PetscDeviceAttribute, void *);
};

struct _n_PetscDevice {
  struct _DeviceOps ops[1];
  void             *data;     /* placeholder */
  PetscInt          refcnt;   /* reference count for the device */
  PetscInt          id;       /* unique id per created PetscDevice */
  PetscInt          deviceId; /* the id of the underlying device, i.e. the return of
                               * cudaGetDevice() for example */
  PetscDeviceType   type;     /* type of device */
};

typedef struct _n_PetscEvent *PetscEvent;
struct _n_PetscEvent {
  PetscDeviceType  dtype;      // this cannot change for the lifetime of the event
  PetscObjectId    dctx_id;    // id of last dctx to record this event
  PetscObjectState dctx_state; // state of last dctx to record this event
  void            *data;       // event handle
  PetscErrorCode (*destroy)(PetscEvent);
};

typedef struct _DeviceContextOps *DeviceContextOps;
struct _DeviceContextOps {
  PetscErrorCode (*destroy)(PetscDeviceContext);
  PetscErrorCode (*changestreamtype)(PetscDeviceContext, PetscStreamType);
  PetscErrorCode (*setup)(PetscDeviceContext);
  PetscErrorCode (*query)(PetscDeviceContext, PetscBool *);
  PetscErrorCode (*waitforcontext)(PetscDeviceContext, PetscDeviceContext);
  PetscErrorCode (*synchronize)(PetscDeviceContext);
  PetscErrorCode (*getblashandle)(PetscDeviceContext, void *);
  PetscErrorCode (*getsolverhandle)(PetscDeviceContext, void *);
  PetscErrorCode (*getstreamhandle)(PetscDeviceContext, void *);
  PetscErrorCode (*begintimer)(PetscDeviceContext);
  PetscErrorCode (*endtimer)(PetscDeviceContext, PetscLogDouble *);
  PetscErrorCode (*memalloc)(PetscDeviceContext, PetscBool, PetscMemType, size_t, size_t, void **);                             // optional
  PetscErrorCode (*memfree)(PetscDeviceContext, PetscMemType, void **);                                                         // optional
  PetscErrorCode (*memcopy)(PetscDeviceContext, void *PETSC_RESTRICT, const void *PETSC_RESTRICT, size_t, PetscDeviceCopyMode); // optional
  PetscErrorCode (*memset)(PetscDeviceContext, PetscMemType, void *, PetscInt, size_t);                                         // optional
  PetscErrorCode (*createevent)(PetscDeviceContext, PetscEvent);                                                                // optional
  PetscErrorCode (*recordevent)(PetscDeviceContext, PetscEvent);                                                                // optional
  PetscErrorCode (*waitforevent)(PetscDeviceContext, PetscEvent);                                                               // optional
};

struct _p_PetscDeviceContext {
  PETSCHEADER(struct _DeviceContextOps);
  PetscDevice     device;         /* the device this context stems from */
  void           *data;           /* solver contexts, event, stream */
  PetscObjectId  *childIDs;       /* array containing ids of contexts currently forked from this one */
  PetscInt        numChildren;    /* how many children does this context expect to destroy */
  PetscInt        maxNumChildren; /* how many children can this context have room for without realloc'ing */
  PetscStreamType streamType;     /* how should this contexts stream behave around other streams? */
  PetscBool       setup;
  PetscBool       usersetdevice;
};

// ===================================================================================
//                            PetscDevice Internal Functions
// ===================================================================================
#if PetscDefined(HAVE_CXX)
PETSC_INTERN PetscErrorCode                PetscDeviceInitializeFromOptions_Internal(MPI_Comm);
PETSC_SINGLE_LIBRARY_INTERN PetscErrorCode PetscDeviceGetDefaultForType_Internal(PetscDeviceType, PetscDevice *);

static inline PetscErrorCode PetscDeviceReference_Internal(PetscDevice device)
{
  PetscFunctionBegin;
  ++(device->refcnt);
  PetscFunctionReturn(0);
}

static inline PetscErrorCode PetscDeviceDereference_Internal(PetscDevice device)
{
  PetscFunctionBegin;
  --(device->refcnt);
  PetscAssert(device->refcnt >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_CORRUPT, "PetscDevice has negative reference count %" PetscInt_FMT, device->refcnt);
  PetscFunctionReturn(0);
}
#else /* PETSC_HAVE_CXX for PetscDevice Internal Functions */
  #define PetscDeviceInitializeFromOptions_Internal(comm)     0
  #define PetscDeviceGetDefaultForType_Internal(Type, device) 0
  #define PetscDeviceReference_Internal(device)               0
  #define PetscDeviceDereference_Internal(device)             0
#endif /* PETSC_HAVE_CXX for PetscDevice Internal Functions */

static inline PetscErrorCode PetscDeviceCheckDeviceCount_Internal(PetscInt count)
{
  PetscFunctionBegin;
  PetscAssert(count < PETSC_DEVICE_MAX_DEVICES, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Detected %" PetscInt_FMT " devices, which is larger than maximum supported number of devices %d", count, PETSC_DEVICE_MAX_DEVICES);
  PetscFunctionReturn(0);
}

/* More general form of PetscDeviceDefaultType_Internal(), as it calls the former using
 * the automatically selected default PetscDeviceType */
#define PetscDeviceGetDefault_Internal(device) PetscDeviceGetDefaultForType_Internal(PETSC_DEVICE_DEFAULT(), device)

static inline PETSC_CONSTEXPR_14 PetscBool PetscDeviceConfiguredFor_Internal(PetscDeviceType type)
{
  switch (type) {
  case PETSC_DEVICE_HOST:
    return PETSC_TRUE;
    /* casts are needed in C++ */
  case PETSC_DEVICE_CUDA:
    return (PetscBool)PetscDefined(HAVE_CUDA);
  case PETSC_DEVICE_HIP:
    return (PetscBool)PetscDefined(HAVE_HIP);
  case PETSC_DEVICE_SYCL:
    return (PetscBool)PetscDefined(HAVE_SYCL);
  case PETSC_DEVICE_MAX:
    return PETSC_FALSE;
    /* Do not add default case! Will make compiler warn on new additions to PetscDeviceType! */
  }
  PetscUnreachable();
  return PETSC_FALSE;
}

// ===================================================================================
//                     PetscDeviceContext Internal Functions
// ===================================================================================
#if PetscDefined(HAVE_CXX)
PETSC_SINGLE_LIBRARY_INTERN PetscErrorCode PetscDeviceContextGetNullContext_Internal(PetscDeviceContext *);

static inline PetscErrorCode PetscDeviceContextGetHandle_Private(PetscDeviceContext dctx, void *handle, PetscErrorCode (*gethandle_op)(PetscDeviceContext, void *))
{
  PetscFunctionBegin;
  PetscValidPointer(handle, 2);
  PetscValidFunction(gethandle_op, 3);
  PetscCall((*gethandle_op)(dctx, handle));
  PetscFunctionReturn(0);
}

static inline PetscErrorCode PetscDeviceContextGetBLASHandle_Internal(PetscDeviceContext dctx, void *handle)
{
  PetscFunctionBegin;
  /* we do error checking here as this routine is an entry-point */
  PetscValidDeviceContext(dctx, 1);
  PetscCall(PetscDeviceContextGetHandle_Private(dctx, handle, dctx->ops->getblashandle));
  PetscFunctionReturn(0);
}

static inline PetscErrorCode PetscDeviceContextGetSOLVERHandle_Internal(PetscDeviceContext dctx, void *handle)
{
  PetscFunctionBegin;
  /* we do error checking here as this routine is an entry-point */
  PetscValidDeviceContext(dctx, 1);
  PetscCall(PetscDeviceContextGetHandle_Private(dctx, handle, dctx->ops->getsolverhandle));
  PetscFunctionReturn(0);
}

static inline PetscErrorCode PetscDeviceContextGetStreamHandle_Internal(PetscDeviceContext dctx, void *handle)
{
  PetscFunctionBegin;
  /* we do error checking here as this routine is an entry-point */
  PetscValidDeviceContext(dctx, 1);
  PetscCall(PetscDeviceContextGetHandle_Private(dctx, handle, dctx->ops->getstreamhandle));
  PetscFunctionReturn(0);
}

static inline PetscErrorCode PetscDeviceContextBeginTimer_Internal(PetscDeviceContext dctx)
{
  PetscFunctionBegin;
  /* we do error checking here as this routine is an entry-point */
  PetscValidDeviceContext(dctx, 1);
  PetscUseTypeMethod(dctx, begintimer);
  PetscFunctionReturn(0);
}

static inline PetscErrorCode PetscDeviceContextEndTimer_Internal(PetscDeviceContext dctx, PetscLogDouble *elapsed)
{
  PetscFunctionBegin;
  /* we do error checking here as this routine is an entry-point */
  PetscValidDeviceContext(dctx, 1);
  PetscValidRealPointer(elapsed, 2);
  PetscUseTypeMethod(dctx, endtimer, elapsed);
  PetscFunctionReturn(0);
}
#else /* PETSC_HAVE_CXX for PetscDeviceContext Internal Functions */
  #define PetscDeviceContextGetNullContext_Internal(dctx)          (*(dctx) = PETSC_NULLPTR, 0)
  #define PetscDeviceContextGetBLASHandle_Internal(dctx, handle)   (*(handle) = PETSC_NULLPTR, 0)
  #define PetscDeviceContextGetSOLVERHandle_Internal(dctx, handle) (*(handle) = PETSC_NULLPTR, 0)
  #define PetscDeviceContextGetStreamHandle_Internal(dctx, handle) (*(handle) = PETSC_NULLPTR, 0)
  #define PetscDeviceContextBeginTimer_Internal(dctx)              0
  #define PetscDeviceContextEndTimer_Internal(dctx, elapsed)       0
#endif /* PETSC_HAVE_CXX for PetscDeviceContext Internal Functions */

/* note, only does assertion checking in debug mode */
static inline PetscErrorCode PetscDeviceContextGetCurrentContextAssertType_Internal(PetscDeviceContext *dctx, PetscDeviceType type)
{
  PetscFunctionBegin;
  PetscCall(PetscDeviceContextGetCurrentContext(dctx));
  if (PetscDefined(USE_DEBUG)) {
    PetscDeviceType dtype;

    PetscValidDeviceType(type, 2);
    PetscCall(PetscDeviceContextGetDeviceType(*dctx, &dtype));
    PetscCheckCompatibleDeviceTypes(dtype, 1, type, 2);
  }
  PetscFunctionReturn(0);
}

static inline PetscErrorCode PetscDeviceContextGetOptionalNullContext_Internal(PetscDeviceContext *dctx)
{
  PetscFunctionBegin;
  PetscValidPointer(dctx, 1);
  if (!*dctx) PetscCall(PetscDeviceContextGetNullContext_Internal(dctx));
  PetscValidDeviceContext(*dctx, 1);
  PetscFunctionReturn(0);
}

/* Experimental API -- it will eventually become public */
#if PetscDefined(HAVE_CXX)
PETSC_EXTERN PetscErrorCode PetscDeviceRegisterMemory(const void *PETSC_RESTRICT, PetscMemType, size_t);
PETSC_EXTERN PetscErrorCode PetscDeviceGetAttribute(PetscDevice, PetscDeviceAttribute, void *);
PETSC_EXTERN PetscErrorCode PetscDeviceContextMarkIntentFromID(PetscDeviceContext, PetscObjectId, PetscMemoryAccessMode, const char name[]);
  #if defined(__cplusplus)
namespace
{

PETSC_NODISCARD inline PetscErrorCode PetscDeviceContextMarkIntentFromID(PetscDeviceContext dctx, PetscObject obj, PetscMemoryAccessMode mode, const char name[])
{
  PetscFunctionBegin;
  PetscCall(PetscDeviceContextMarkIntentFromID(dctx, obj->id, mode, name));
  PetscFunctionReturn(0);
}

} // anonymous namespace
  #endif // __cplusplus
#else
  #define PetscDeviceRegisterMemory(void_ptr, PetscMemType, size)                                           0
  #define PetscDeviceGetAttribute(PetscDevice, PetscDeviceAttribute, void_star)                             ((*((int *)(void_star)) = 0), 0)
  #define PetscDeviceContextMarkIntentFromID(PetscDeviceContext, PetscObjectId, PetscMemoryAccessMode, ptr) 0
#endif

PETSC_INTERN PetscErrorCode PetscDeviceContextCreate_HOST(PetscDeviceContext);
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
