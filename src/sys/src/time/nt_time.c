/*$Id: nt_time.c,v 1.25 2001/04/05 21:06:51 balay Exp $*/

#include <petsc.h>
#if defined (PARCH_win32_gnu) || defined (PARCH_win32)
#include <Windows.h>
#define FACTOR   4294967296.0 /* pow(2,32) */

#undef __FUNCT__  
#define __FUNCT__ "nt_time"
PetscLogDouble nt_time(void) 
{
  static PetscTruth     flag = PETSC_TRUE;
  int                   ierr;

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
    SecInTick = 1.0/((double)PerfFreq.HighPart*FACTOR+(double)PerfFreq.LowPart);
    flag = PETSC_FALSE;
  }		
  
  ierr        = QueryPerformanceCounter(&CurTime);CHKERRQ(!ierr);
  dwCurHigh   = (DWORD)CurTime.HighPart;
  dwStartHigh = (DWORD)StartTime.HighPart;
  dHigh       = (signed)(dwCurHigh - dwStartHigh);

  dTime = dHigh*(double)FACTOR + (double)CurTime.LowPart - (double)StartTime.LowPart;
  /* Use the following with older versions of the Borland compiler
  dTime = dHigh*(double)FACTOR + (double)CurTime.u.LowPart - (double)StartTime.u.LowPart;
  */
  ptime = (double)SecInTick*dTime;

  PetscFunctionReturn(ptime);
}
#endif
