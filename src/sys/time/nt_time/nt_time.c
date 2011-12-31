
#include <petscsys.h>
#include <Windows.h>
#define FACTOR   4294967296.0 /* pow(2,32) */

EXTERN_C_BEGIN

#undef __FUNCT__  
#define __FUNCT__ "nt_time"
PetscLogDouble  nt_time(void)
{
  static PetscBool      flag = PETSC_TRUE;
  PetscErrorCode ierr;

  static LARGE_INTEGER  StartTime,PerfFreq,CurTime; 
  static PetscLogDouble SecInTick=0.0;
  
  DWORD                 dwStartHigh,dwCurHigh;
  PetscLogDouble        dTime,dHigh;
  PetscLogDouble        ptime;
  
  
  PetscFunctionBegin;
  if (flag) {
    ierr = QueryPerformanceCounter(&StartTime);CHKERRQ(!ierr);
    ierr = QueryPerformanceFrequency(&PerfFreq);CHKERRQ(!ierr);
    /* Explicitly convert the higher 32 bits, and add the lower 32 bits from the counter */
    /* works on non-pentium CPUs ? */
#if defined(PETSC_HAVE_LARGE_INTEGER_U)
    SecInTick = 1.0/((double)PerfFreq.u.HighPart*FACTOR+(double)PerfFreq.u.LowPart);
#else
    SecInTick = 1.0/((double)PerfFreq.HighPart*FACTOR+(double)PerfFreq.LowPart);
#endif
    flag = PETSC_FALSE;
  }		
  
  ierr        = QueryPerformanceCounter(&CurTime);CHKERRQ(!ierr);
#if defined(PETSC_HAVE_LARGE_INTEGER_U)
  dwCurHigh   = (DWORD)CurTime.u.HighPart;
  dwStartHigh = (DWORD)StartTime.u.HighPart;
#else
  dwCurHigh   = (DWORD)CurTime.HighPart;
  dwStartHigh = (DWORD)StartTime.HighPart;
#endif
  dHigh       = (signed)(dwCurHigh - dwStartHigh);

#if defined(PETSC_HAVE_LARGE_INTEGER_U)
  dTime = dHigh*(double)FACTOR + (double)CurTime.u.LowPart - (double)StartTime.u.LowPart;
#else
  dTime = dHigh*(double)FACTOR + (double)CurTime.LowPart - (double)StartTime.LowPart;
#endif
  /* Use the following with older versions of the Borland compiler
  dTime = dHigh*(double)FACTOR + (double)CurTime.u.LowPart - (double)StartTime.u.LowPart;
  */
  ptime = (double)SecInTick*dTime;

  PetscFunctionReturn(ptime);
}

EXTERN_C_END

