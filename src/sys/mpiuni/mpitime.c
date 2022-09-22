#include <petscsys.h>
#ifndef MPIUNI_H
  #error "Wrong mpi.h included! require mpi.h from MPIUNI"
#endif

#if defined(__cplusplus)
extern "C" {
#endif
/* ------------------------------------------------------------------
   Microsoft Windows has its own time routines
*/
#if defined(PETSC_USE_MICROSOFT_TIME)
  #include <windows.h>
  #define FACTOR 4294967296.0 /* pow(2,32) */

double MPI_Wtime(void)
{
  static int           flag = 1;
  static LARGE_INTEGER StartTime, PerfFreq, CurTime;
  static double        SecInTick = 0.0;

  DWORD  dwStartHigh, dwCurHigh;
  double dTime, dHigh;
  double ptime;

  if (flag) {
    if (!QueryPerformanceCounter(&StartTime)) PETSCABORT(MPI_COMM_WORLD, PETSC_ERR_LIB);
    if (!QueryPerformanceFrequency(&PerfFreq)) PETSCABORT(MPI_COMM_WORLD, PETSC_ERR_LIB);
      /* Explicitly convert the higher 32 bits, and add the lower 32 bits from the counter */
      /* works on non-pentium CPUs ? */
  #if defined(PETSC_HAVE_LARGE_INTEGER_U)
    SecInTick = 1.0 / ((double)PerfFreq.u.HighPart * FACTOR + (double)PerfFreq.u.LowPart);
  #else
    SecInTick = 1.0 / ((double)PerfFreq.HighPart * FACTOR + (double)PerfFreq.LowPart);
  #endif
    flag = 0;
  }

  if (!QueryPerformanceCounter(&CurTime)) PETSCABORT(MPI_COMM_WORLD, PETSC_ERR_LIB);
  #if defined(PETSC_HAVE_LARGE_INTEGER_U)
  dwCurHigh   = (DWORD)CurTime.u.HighPart;
  dwStartHigh = (DWORD)StartTime.u.HighPart;
  #else
  dwCurHigh = (DWORD)CurTime.HighPart;
  dwStartHigh = (DWORD)StartTime.HighPart;
  #endif
  dHigh = (signed)(dwCurHigh - dwStartHigh);

  #if defined(PETSC_HAVE_LARGE_INTEGER_U)
  dTime = dHigh * (double)FACTOR + (double)CurTime.u.LowPart - (double)StartTime.u.LowPart;
  #else
  dTime = dHigh * (double)FACTOR + (double)CurTime.LowPart - (double)StartTime.LowPart;
  #endif
  /* Use the following with older versions of the Borland compiler
  dTime = dHigh*(double)FACTOR + (double)CurTime.u.LowPart - (double)StartTime.u.LowPart;
  */
  ptime = (double)SecInTick * dTime;
  return (ptime);
}

/* ------------------------------------------------------------------
    The usual Unix time routines.
*/
#else

  #if defined(PETSC_HAVE_SYS_TIME_H)
    #include <sys/time.h>
  #endif

  #if defined(PETSC_NEEDS_GETTIMEOFDAY_PROTO)
extern int gettimeofday(struct timeval *, struct timezone *);
  #endif

double MPI_Wtime(void)
{
  static struct timeval _tp;
  gettimeofday(&_tp, (struct timezone *)0);
  return ((double)_tp.tv_sec) + (1.0e-6) * (_tp.tv_usec);
}
#endif

#if defined(__cplusplus)
}
#endif
