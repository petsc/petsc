/* A template file for the CUDA Programming Model (CUPM) initialization, to be included in init.c. CUPM is either CUDA or HIP. */

PetscBool PetscCUPMSynchronize     = PETSC_FALSE;
PetscBool PetscCUPMInitialized     = PETSC_FALSE;

static PetscBool PetscNotUseCUPM   = PETSC_FALSE; /* Assert the code will not use this type of devices */

/* Device validation after it is lazily initialized */
static PetscErrorCode PetscCUPMValidate(void)
{
  PetscBool  mpi_gpu_awareness;

  PetscFunctionBegin;
  if (use_gpu_aware_mpi) {
#if defined(PETSC_HAVE_OMPI_MAJOR_VERSION) && defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
    /* Trust OpenMPI's compile time gpu query interface */
    mpi_gpu_awareness = PETSC_TRUE;
#else
    /* For other MPI implementations without gpu query API, we do a GPU MPI call to see if it segfaults.
      Note that Spectrum MPI sets OMPI_MAJOR_VERSION and is CUDA-aware, but does not have MPIX_CUDA_AWARE_SUPPORT.
    */
    mpi_gpu_awareness = PetscMPICUPMAwarenessCheck();
#endif
    if (!mpi_gpu_awareness) {
      (*PetscErrorPrintf)("PETSc is configured with GPU support, but your MPI is not GPU-aware. For better performance, please use a GPU-aware MPI.\n");
      (*PetscErrorPrintf)("If you do not care, add option -use_gpu_aware_mpi 0. To not see the message again, add the option to your .petscrc, OR add it to the env var PETSC_OPTIONS.\n");
      (*PetscErrorPrintf)("If you do care, for IBM Spectrum MPI on OLCF Summit, you may need jsrun --smpiargs=-gpu.\n");
      (*PetscErrorPrintf)("For OpenMPI, you need to configure it --with-cuda (https://www.open-mpi.org/faq/?category=buildcuda)\n");
      (*PetscErrorPrintf)("For MVAPICH2-GDR, you need to set MV2_USE_CUDA=1 (http://mvapich.cse.ohio-state.edu/userguide/gdr/)\n");
      (*PetscErrorPrintf)("For Cray-MPICH, you need to set MPICH_RDMA_ENABLED_CUDA=1 (https://www.olcf.ornl.gov/tutorials/gpudirect-mpich-enabled-cuda/)\n");
      PETSCABORT(PETSC_COMM_SELF,PETSC_ERR_LIB);
    }
  }
  PetscFunctionReturn(0);
}

/*@C
     PetscCUDAInitializeCheck - Check if CUDA is initialized. If not, initialize it.

  Logically collective

  Level: beginner

  Notes:
    In PETSc lazy device initialization, PETSc calls this function right before creating the first CUDA/HIP object.
    It can be used by application developers who want to lazily initialize CUDA/HIP when they start to use it (which may before a PETSc CUDA/HIP object is created.)

  .seealso: PetscCUDAInitialize(), PetscHIPInitialize(), PetscHIPInitializeCheck()
@*/
PETSC_EXTERN PetscErrorCode PetscCUDAInitializeCheck(void);


/*@C
     PetscHIPInitializeCheck - Check if HIP is initialized. If not, initialize it.

  Logically collective

  Level: beginner

  Notes:
    See notes of PetscCUDAInitializeCheck() for details.

  .seealso: PetscHIPInitialize(), PetscCUDAInitialize(), PetscCUDAInitializeCheck()
@*/
PETSC_EXTERN PetscErrorCode PetscCUDAInitializeCheck(void);

PetscErrorCode PetscCUPMInitializeCheck(void)
{
  PetscErrorCode        ierr;
  cupmError_t           cerr;
  int                   devId,devCount;
  PetscMPIInt           rank;
  static PetscBool      cupmValdidateChecked = PETSC_FALSE;

  PetscFunctionBegin;
  if (PetscNotUseCUPM) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"You asserted the code wouldn't use devices with -device_set none, but now trying to create a device object. Remove this option or see manpage of PetscCUPMInitialize().");
  if (!PetscCUPMInitialized) {
    cerr = cupmGetDeviceCount(&devCount);CHKERRCUPM(cerr);
    if (devCount > 1) {
      cerr = cupmSetDeviceFlags(cupmDeviceMapHost);
      cupmGetLastError(); /* Reset the last error */
      if (cerr == cupmSuccess) { /* It implies device runtime has not been initialized? */
        ierr  = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
        devId = rank % devCount;
        for (int i=0; i<3; i++) {
          cerr  = cupmSetDevice(devId);
          if (cerr == cupmSuccess) break;
          if (cerr != cupmErrorMemoryAllocation && cerr != cupmErrorLaunchOutOfResources) CHKERRCUPM(cerr);
          if (i < 2) {ierr = PetscSleep(3);CHKERRQ(ierr);}
        }
        if (cerr) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_GPU_RESOURCE,"Unable to initialize the GPU");
      } else if (cerr == cupmErrorSetOnActiveProcess) {
        /* It implies user has initialized device runtime outside of petsc. We do nothing to respect the device choice. */
      }
    }
    ierr = PetscCUPMBLASInitializeHandle();CHKERRQ(ierr);
    ierr = PetscCUPMSOLVERDnInitializeHandle();CHKERRQ(ierr);
    PetscCUPMInitialized = PETSC_TRUE;
#if defined(PETSC_HAVE_KOKKOS)
    ierr = PetscKokkosInitialize_Private();CHKERRQ(ierr);
    PetscBeganKokkos = PETSC_TRUE;
#endif
  }

  if (!cupmValdidateChecked) {
    ierr = PetscCUPMValidate();CHKERRQ(ierr);
    cupmValdidateChecked = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

/*@C
     PetscCUDAInitialize - Initializes CUDA (eagerly in PetscInitialize() or soon after PetscInitialize()) and cuBLAS/cuSPARSE libraries on the device

     Logically collective

  Input Parameter:
+ comm   - the MPI communicator that will utilize the devices
- device - the device assigned to current MPI process. Special values like PETSC_DECIDE or PETSC_DEFAULT have special meanings (see details below)

  Options Database:
+  -cuda_device <device> - the device assigned to current MPI rank. <device> is case-insensitive and can be:
       NONE (or none, or -3) : the code will not use any device, otherwise it will error out;
       PETSC_DEFAULT(or DEFAULT, or -2) : do not explicitly set device, i.e., use whatever device already set by user (probably before PetscInitialize()). Init device runtime etc;
       PETSC_DECIDE (or DECIDE, or -1) : assign MPI ranks in comm to available devices in round-robin, and init device runtime etc on the selected device;
       >= 0 integer  : assign the device with this id to current MPI process. Error out if <device> is invalid. Init device runtime etc on this device;
     With PETSC_{DECIDE, DEFAULT}, if there are actually no devices, the code can still run, but it will error out when trying to create device objects.
.  -cuda_view              - view information about the devices.
.  -cuda_synchronize       - wait at the end of asynchronize device calls so that their time gets credited to the current event. With -log_view, the default is true, otherwise false.
.  -log_view               - logging, however if alone or combined with `-cuda_set_device DEFAULT | DECIDE | >=0 int`, will init device; if combined with `-cuda_set_device none`, won't init device.
-  -use_gpu_aware_mpi      - assume the MPI is device/GPU-aware when communicating data on devices. Default true.

  Level: beginner

  Notes:
    Unless the input parameter <device> = -3, this routine initializes the CUDA device. It also initializes the cuBLAS/cuSPARSE libraries, which
    takes a lot of time. Initializing them early helps avoid skewing timings in -log_view.

    If this routine is triggered by command line options, it is called in PetscInitialize(). If users want to directly call it, they should call it immediately after PetscInitialize().

    If this is not called then the CUDA initialization is delayed until first creation of a CUDA object and this can affect the timing since they happen asynchronously on different nodes and take a lot of time.

   .seealso: PetscCUDAInitializeCheck(), PetscHIPInitialize(), PetscHIPInitializeCheck()
@*/
PETSC_EXTERN PetscErrorCode PetscCUDAInitialize(MPI_Comm comm,PetscInt device);

/*@C
     PetscHIPInitialize - Initializes HIP (eagerly in PetscInitialize() or soon after PetscInitialize()) and hipBLAS/hipSPARSE libraries on the device

     Logically collective

  Input Parameter:
   (see notes)

  Options Database:
   (see notes)

  Level: beginner

  Notes:
    The functionality, parameters and options database of this routine are similar to that of PetscCUDAInitialize(), except that the option names
    are -hip_device, -hip_view, -hip_synchronize instead. See manpage of PetscCUDAInitialize() for details.

  .seealso: PetscHIPInitializeCheck(), PetscCUDAInitialize(), PetscCUDAInitializeCheck()
@*/
PETSC_EXTERN PetscErrorCode PetscHIPInitialize(MPI_Comm comm,PetscInt device);

PetscErrorCode PetscCUPMInitialize(MPI_Comm comm,PetscInt device)
{
  PetscErrorCode        ierr;
  cupmError_t           cerr;
  int                   devId,devCount=0;
  const PetscInt        PETSC_NONE=-3; /* Unlike PETSC_DECIDE, we don't have a macro PETSC_NONE in petsc headers */
  PetscMPIInt           rank;

  PetscFunctionBegin;
  if (!PetscCUPMInitialized) {
    cerr = cupmGetDeviceCount(&devCount);
    cupmGetLastError(); /* Reset the last error */
    if (cerr != cupmSuccess) devCount = 0;
    if (device >= 0) { /* User wants to use this specific device */
      cerr = cupmSetDeviceFlags(cupmDeviceMapHost); /* Allow it to fail since user might have already initialized the device. */
      cupmGetLastError(); /* Reset the last error */
      cerr = cupmSetDevice((int)device);CHKERRCUPM(cerr);
    } else if (device == PETSC_DECIDE) { /* Assign MPI ranks to available devices in round-robin */
      if (devCount > 0) { /* Allow no device as long as user does not use devices */
        /* Set the device flags so that it can map host memory */
        cerr  = cupmSetDeviceFlags(cupmDeviceMapHost);CHKERRCUPM(cerr);
        ierr  = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
        devId = rank % devCount;
        cerr  = cupmSetDevice(devId);CHKERRCUPM(cerr);
      }
    } else if (device == PETSC_DEFAULT) {
      /* Do nothing, i.e., use whatever device set by user before PetscInitialize() */
    } else if (device == PETSC_NONE) {
      PetscNotUseCUPM = PETSC_TRUE; /* Assert the code won't use devices even there are */
    } else SETERRQ1(comm,PETSC_ERR_ARG_OUTOFRANGE,"Wrong device (%D) passed to -device_set <dev>. Must be NONE(-3),PETSC_DEFAULT(-2),PETSC_DECIDE(-1) or a non-negative integer.",device);

    if (devCount > 0 && device != PETSC_NONE) {
      /* Do costly device handles initialization here to not to distort petsc logging later */
      ierr = PetscCUPMBLASInitializeHandle();CHKERRQ(ierr);
      ierr = PetscCUPMSOLVERDnInitializeHandle();CHKERRQ(ierr);
      PetscCUPMInitialized = PETSC_TRUE;
    }
  }
  PetscFunctionReturn(0);
}

/*
  The routine works as a driver to initialize and view the device

  Input Parameter:
    initDevice: True if user explicitly has -cuda/hip_set_device xxx
    device:     Significant when <initDeivce>. Basically, it is the integer presentation of the xxx above
    logView:    True if -log_view or -log_summary
    devView:    True if -{cuda,hip}_view
 */
static PetscErrorCode PetscCUPMInitializeAndView(PetscBool initDevice,PetscInt device,PetscBool logView,PetscBool devView)
{
  PetscErrorCode        ierr;
  cupmError_t           cerr;
  PetscMPIInt           rank;
  int                   devId,devCount;
  cupmDeviceProp        prop;

  PetscFunctionBegin;
  PetscCUPMSynchronize = logView;
  if (initDevice) {ierr = PetscCUPMInitialize(PETSC_COMM_WORLD,device);CHKERRQ(ierr);}
  else if (logView) { /* With -log_view, we want to do costly gpu runtime initialization early so that not to distort the timing later. */
    devCount = 0;
    cerr = cupmGetDeviceCount(&devCount);
    cupmGetLastError(); /* Reset the last error */
    if (cerr == cupmSuccess && devCount >= 1) { /* There are devices */
      devId = 0;
      if (devCount > 1) { /* Decide which device to init when there are multiple */
        cerr = cupmSetDeviceFlags(cupmDeviceMapHost);
        cupmGetLastError(); /* Reset the last error */
        if (cerr == cupmSuccess) { /* It implies gpu runtime has not been initialized */
          ierr  = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
          devId = rank % devCount;
          cerr  = cupmSetDevice(devId);CHKERRCUPM(cerr);
        } else if (cerr == cupmErrorSetOnActiveProcess) {
          /* It means user initialized gpu runtime outside of petsc. We respect the device choice. */
          cerr = cupmGetDevice(&devId);CHKERRCUPM(cerr);
        }
      }
      ierr = PetscCUPMInitialize(PETSC_COMM_WORLD,(PetscInt)devId);CHKERRQ(ierr);
    }
  }

  if (devView) {
    ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
    cerr = cupmGetDeviceCount(&devCount);CHKERRCUPM(cerr);
    for (devId = 0; devId < devCount; ++devId) {
      cerr = cupmGetDeviceProperties(&prop,devId);CHKERRCUPM(cerr);
      ierr = PetscPrintf(PETSC_COMM_WORLD, "device %d: %s\n", devId, prop.name);CHKERRQ(ierr);
    }
    cerr = cupmGetDevice(&devId);CHKERRCUPM(cerr);
    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d] Using device %d.\n",rank,devId);CHKERRQ(ierr);
    ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*
  The routine checks user's device related options and initializes the device if instructed.

  Input Parameter:
    logView:    True if -log_view or -log_summary
 */
static PetscErrorCode PetscOptionsCheckCUPM(PetscBool logView)
{
  PetscErrorCode ierr;
  PetscBool      initDevice = PETSC_FALSE,devView = PETSC_FALSE,devNone = PETSC_FALSE;
  PetscInt       device;
  char           devStr[32]={0};
#if defined(PETSC_HAVE_KOKKOS)
  PetscBool      set,kinited,devDefault;
#endif

  PetscFunctionBegin;
#if defined(PETSC_HAVE_KOKKOS)
  ierr = PetscKokkosIsInitialized_Private(&kinited);CHKERRQ(ierr);
  if (kinited) { /* Check if Petsc device options conform with Kokkos' device */
    ierr = PetscOptionsGetString(NULL,NULL,cupmSetDeviceStr,devStr,sizeof(devStr),&set);CHKERRQ(ierr);
    if (set) { /* If users have initialized Kokkos themselves, but also had e.g., -cuda_set_device XXX, for simplicity, make sure XXX is DEFAULT */
      ierr = PetscStrcasecmp("DEFAULT",devStr,&devDefault);CHKERRQ(ierr);
      if (!devDefault) {ierr = PetscStrcasecmp("PETSC_DEFAULT",devStr,&devDefault);CHKERRQ(ierr);}
      if (!devDefault)  SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Kokkos was initialized before PetscInitialize(), but you have "cupmSetDeviceStr" %s. Remove the option or use "cupmSetDeviceStr" default.",devStr);
    } else { /* If users did not have e.g., '-cuda_set_device XXX', insert one here so that petsc can continue its own device initialization */
      ierr = PetscOptionsSetValue(NULL,cupmSetDeviceStr,"DEFAULT");CHKERRQ(ierr);
    }
  }
#endif

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,cupmOptionsStr,"Sys");CHKERRQ(ierr);
  ierr = PetscOptionsString(cupmSetDeviceStr,NULL,PetscCUPMInitializeStr,devStr,devStr,sizeof(devStr),&initDevice);CHKERRQ(ierr);
  ierr = PetscStrcasecmp("none",devStr,&devNone);CHKERRQ(ierr);
  if (devNone) device = -3; /* -3 is the locally used PETSC_NONE in Petsc{CUDA/HIP}Initialize() */
  else {ierr = PetscOptionsInt(cupmSetDeviceStr,"Set which MPI ranks to use which devices",PetscCUPMInitializeStr,device,&device,&initDevice);CHKERRQ(ierr);}
  ierr = PetscOptionsBool(cupmSynchronizeStr,"Wait for the device to complete operations before returning to the CPU (on by default with -log_summary or -log_view)",NULL,PetscCUPMSynchronize,&PetscCUPMSynchronize,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsName(cupmViewStr,"Display device information and assignments",NULL,&devView);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  ierr = PetscCUPMInitializeAndView(initDevice,device,logView,devView);CHKERRQ(ierr);

#if defined(PETSC_HAVE_KOKKOS)
  if (PetscCUPMInitialized && !kinited) {
    ierr = PetscKokkosInitialize_Private();CHKERRQ(ierr);
    PetscBeganKokkos = PETSC_TRUE;
  }
#endif
  PetscFunctionReturn(0);
}
