
#include <petscsys.h> /*I "petscsys.h" I*/

#if defined(PETSC_HAVE_SSE)

  #include PETSC_HAVE_SSE
  #define SSE_FEATURE_FLAG 0x2000000 /* Mask for bit 25 (from bit 0) */

PetscErrorCode PetscSSEHardwareTest(PetscBool *flag)
{
  char      vendor[13];
  char      Intel[13] = "GenuineIntel";
  char      AMD[13]   = "AuthenticAMD";
  char      Hygon[13] = "HygonGenuine";
  PetscBool flg;

  PetscFunctionBegin;
  PetscCall(PetscStrncpy(vendor, "************", sizeof(vendor)));
  CPUID_GET_VENDOR(vendor);
  PetscCall(PetscStrcmp(vendor, Intel, &flg));
  if (!flg) PetscCall(PetscStrcmp(vendor, AMD, &flg));
  if (!flg) {
    PetscCall(PetscStrcmp(vendor, Hygon, &flg));
    if (flg) {
      /* Intel, AMD, and Hygon use bit 25 of CPUID_FEATURES */
      /* to denote availability of SSE Support */
      unsigned long myeax, myebx, myecx, myedx;
      CPUID(CPUID_FEATURES, &myeax, &myebx, &myecx, &myedx);
      if (myedx & SSE_FEATURE_FLAG) *flag = PETSC_TRUE;
      else *flag = PETSC_FALSE;
    }
    PetscFunctionReturn(PETSC_SUCCESS);
  }
}

  #if defined(PETSC_HAVE_FORK)
    #include <signal.h>
    /*
   Early versions of the Linux kernel disables SSE hardware because
   it does not know how to preserve the SSE state at a context switch.
   To detect this feature, try an sse instruction in another process.
   If it works, great!  If not, an illegal instruction signal will be thrown,
   so catch it and return an error code.
*/
    #define PetscSSEOSEnabledTest(arg) PetscSSEOSEnabledTest_Linux(arg)

static void PetscSSEDisabledHandler(int sig)
{
  signal(SIGILL, SIG_IGN);
  exit(-1);
}

PetscErrorCode PetscSSEOSEnabledTest_Linux(PetscBool *flag)
{
  int status, pid = 0;

  PetscFunctionBegin;
  signal(SIGILL, PetscSSEDisabledHandler);
  pid = fork();
  if (pid == 0) {
    SSE_SCOPE_BEGIN;
    XOR_PS(XMM0, XMM0);
    SSE_SCOPE_END;
    exit(0);
  } else wait(&status);
  if (!status) *flag = PETSC_TRUE;
  else *flag = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

  #else
    /*
   Windows 95/98/NT4 should have a Windows Update/Service Patch which enables this hardware.
   Windows ME/2000 doesn't disable SSE Hardware
*/
    #define PetscSSEOSEnabledTest(arg) PetscSSEOSEnabledTest_TRUE(arg)
  #endif

PetscErrorCode PetscSSEOSEnabledTest_TRUE(PetscBool *flag)
{
  PetscFunctionBegin;
  if (flag) *flag = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#else /* Not defined PETSC_HAVE_SSE */

  #define PetscSSEHardwareTest(arg)  PetscSSEEnabledTest_FALSE(arg)
  #define PetscSSEOSEnabledTest(arg) PetscSSEEnabledTest_FALSE(arg)

PetscErrorCode PetscSSEEnabledTest_FALSE(PetscBool *flag)
{
  PetscFunctionBegin;
  if (flag) *flag = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#endif /* defined PETSC_HAVE_SSE */

static PetscBool petsc_sse_local_is_untested  = PETSC_TRUE;
static PetscBool petsc_sse_enabled_local      = PETSC_FALSE;
static PetscBool petsc_sse_global_is_untested = PETSC_TRUE;
static PetscBool petsc_sse_enabled_global     = PETSC_FALSE;
/*@C
     PetscSSEIsEnabled - Determines if Intel Streaming SIMD Extensions (SSE) to the x86 instruction
     set can be used.  Some operating systems do not allow the use of these instructions despite
     hardware availability.

     Collective

     Input Parameter:
.    comm - the MPI Communicator

     Output Parameters:
+    lflag - Local Flag  `PETSC_TRUE` if enabled in this process
-    gflag - Global Flag `PETSC_TRUE` if enabled for all processes in comm

     Options Database Key:
.    -disable_sse - Disable use of hand tuned Intel SSE implementations

     Level: developer

     Note:
     `NULL` can be specified for `lflag` or `gflag` if either of these values are not desired.
@*/
PetscErrorCode PetscSSEIsEnabled(MPI_Comm comm, PetscBool *lflag, PetscBool *gflag)
{
  PetscBool disabled_option;

  PetscFunctionBegin;
  if (petsc_sse_local_is_untested && petsc_sse_global_is_untested) {
    disabled_option = PETSC_FALSE;

    PetscCall(PetscOptionsGetBool(NULL, NULL, "-disable_sse", &disabled_option, NULL));
    if (disabled_option) {
      petsc_sse_local_is_untested  = PETSC_FALSE;
      petsc_sse_enabled_local      = PETSC_FALSE;
      petsc_sse_global_is_untested = PETSC_FALSE;
      petsc_sse_enabled_global     = PETSC_FALSE;
    }

    if (petsc_sse_local_is_untested) {
      PetscCall(PetscSSEHardwareTest(&petsc_sse_enabled_local));
      if (petsc_sse_enabled_local) { PetscCall(PetscSSEOSEnabledTest(&petsc_sse_enabled_local)); }
      petsc_sse_local_is_untested = PETSC_FALSE;
    }

    if (gflag && petsc_sse_global_is_untested) {
      PetscCall(MPIU_Allreduce(&petsc_sse_enabled_local, &petsc_sse_enabled_global, 1, MPIU_BOOL, MPI_LAND, comm));

      petsc_sse_global_is_untested = PETSC_FALSE;
    }
  }

  if (lflag) *lflag = petsc_sse_enabled_local;
  if (gflag) *gflag = petsc_sse_enabled_global;
  PetscFunctionReturn(PETSC_SUCCESS);
}
