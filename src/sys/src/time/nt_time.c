#ifndef lint
static char vcid[] = "$Id: nt_gnu_time.c,v 1.1 1997/02/21 17:30:05 balay Exp balay $";
#endif

#include <petsc.h>
#if defined (PARCH_nt_gnu)
#include <Windows32/Base.h>
#include <Windows32/Defines.h>
#include <Windows32/Structures.h>
#include <Windows32/Functions.h>
#define FACTOR   4294967296.0

double nt_gnu_time()
{
  static int    flag = 1;
  int           ierr;

  LARGE_INTEGER StartTime,PerfFreq,CurTime; 
  static double SecInTick=0.0;
  
  DWORD dwStartHigh, dwCurHigh;
  double dTime, dHigh;
  double time;
  
  
  if (flag) {
  ierr = QueryPerformanceCounter( &StartTime ); CHKERRQ(ierr);
  ierr = QueryPerformanceFrequency( &PerfFreq ); CHKERRQ(ierr);
  SecInTick = 1.0/((double)PerfFreq.HighPart*FACTOR+(double)PerfFreq.LowPart);
  }		
  
  ierr        = QueryPerformanceCounter( &CurTime ); CHKERRQ(ierr);
  dwCurHigh   = (DWORD)CurTime.HighPart;
  dwStartHigh = (DWORD)StartTime.HighPart;
  dHigh       = (signed)(dwCurHigh - dwStartHigh);

  dTime = dHigh*(double)FACTOR + (double)CurTime.LowPart - (double)StartTime.LowPart;
  time  = (double)SecInTick*dTime;

  return time;
}

#else
double PetscTime_nt_gnu()
{
  SETERRQ(1,0,"Wrong Architecture");
}
#endif
