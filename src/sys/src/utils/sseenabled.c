/* $Id: sseenabled.c,v 1.8 2001/06/21 18:29:04 curfman Exp buschelm $ */
#include "petsc.h"

#ifdef PETSC_HAVE_SSE

#include PETSC_HAVE_SSE
#define SSE_FEATURE_FLAG 0x2000000 /* Mask for bit 25 (from bit 0) */

#include <string.h>
int PetscSSEHardwareTest(PetscTruth *flag) {
  char *vendor="************";

  *flag = PETSC_FALSE;
  CPUID_GET_VENDOR(vendor);
  if (!strcmp(vendor,"GenuineIntel")) { 
    /* If Genuine Intel ... */
    unsigned long eax,ebx,ecx,edx;
    CPUID(CPUID_FEATURES,&eax,&ebx,&ecx,&edx);
    /* SSE Feature is indicated by Bit 25 of the EDX register */
    if (edx & SSE_FEATURE_FLAG) {
      *flag = PETSC_TRUE;
    }
  }
  return(0);
}

#ifdef PARCH_linux
#include <signal.h>
/* 
   Early versions of the Linux kernel disables SSE hardware because
   it does not know how to preserve the SSE state at a context switch.
   To detect this feature, try an sse instruction in another process.  
   If it works, great!  If not, an illegal instruction signal will be thrown,
   so catch it and return an error code. 
*/
#define PetscSSEOSEnabledTest(arg) PetscSSEOSEnabledTest_Linux(arg)

static void SSEEnabledHandler(int sig) {
  signal(SIGILL,SIG_IGN);
  exit(-1);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSSEOSEnabledTest"
int PetscSSEOSEnabledTest_Linux(PetscTruth *flag) {
  int status,pid =0;
  PetscFunctionBegin;
  signal(SIGILL,SSEEnabledHandler);
  pid = fork();
  if (pid==0) {
    SSE_SCOPE_BEGIN;
      XOR_PS(XMM0,XMM0);
    SSE_SCOPE_END;
    exit(0);
  } else {
    wait(&status);
  }
  if (!status) {
    *flag = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

#endif
#ifdef PARCH_win32
/* 
   Windows 95/98/NT4 should have a Windows Update/Service Patch which enables this hardware.
   Windows ME/2000 doesn't disable SSE Hardware 
*/
#define PetscSSEOSEnabledTest(arg) PetscSSEOSEnabledTest_TRUE(arg)

int PetscSSEOSEnabledTest_TRUE(PetscTruth *flag) {
  *flag = PETSC_TRUE;
  return(0);
}

#endif 
#else  /* Not defined PETSC_HAVE_SSE */

#define PetscSSEHardwareTest(arg) 0
#define PetscSSEOSEnabledTest(arg) 0

#endif /* defined PETSC_HAVE_SSE */

#undef __FUNCT__
#define __FUNCT__ "PetscSSEIsEnabled"
int PetscSSEIsEnabled(PetscTruth *flag) {
  int ierr;
  PetscFunctionBegin;
  *flag = PETSC_FALSE;
  ierr = PetscSSEHardwareTest(flag);CHKERRQ(ierr);
  if (*flag) {
    ierr = PetscSSEOSEnabledTest(flag);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


