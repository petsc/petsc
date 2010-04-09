/*
       Low cost access to system time. This, in general, should not
     be included in user programs.
*/

#if !defined(__PTIME_H)
#define __PTIME_H

#include "petscsys.h"
#if defined(PETSC_HAVE_SYS_TIME_H)
#include <sys/types.h>
#include <sys/time.h>
#endif
#if defined(PETSC_NEEDS_GETTIMEOFDAY_PROTO)
EXTERN_C_BEGIN
EXTERN int gettimeofday(struct timeval *,struct timezone *);
EXTERN_C_END
#endif

/* Global counters */
extern PetscLogDouble PETSC_DLLEXPORT BaseTime;

/*
   PetscTime - Returns the current time of day in seconds.  

   Synopsis:
   PetscTime(PetscLogDouble v)

   Not Collective

   Output Parameter:
.  v - time counter


   Usage: 
     PetscLogDouble v;
     PetscTime(v);
     .... perform some calculation ...
     printf("Time for operation %g\n",v);

   Notes:
   Since the PETSc libraries incorporate timing of phases and operations, 
   PetscTime() is intended only for timing of application codes.  
   The options database commands -log, -log_summary, and -log_all activate
   PETSc library timing.  See the users manual for further details.

.seealso:  PetscTimeSubtract(), PetscTimeAdd()

.keywords:  Petsc, time
*/

/*
   PetscTimeSubtract - Subtracts the current time of day (in seconds) from
   the value v.  

   Synopsis:
   PetscTimeSubtract(PetscLogDouble v)

   Not Collective

   Input Parameter:
.  v - time counter

   Output Parameter:
.  v - time counter (v = v - current time)


   Notes:
   Since the PETSc libraries incorporate timing of phases and operations, 
   PetscTimeSubtract() is intended only for timing of application codes.  
   The options database commands -log, -log_summary, and -log_all activate
   PETSc library timing.  See the users manual for further details.

.seealso:  PetscTime(), PetscTimeAdd()

.keywords:  Petsc, time, subtract
*/

/*
   PetscTimeAdd - Adds the current time of day (in seconds) to the value v.  

   Synopsis:
   PetscTimeAdd(PetscLogDouble v)

   Not Collective

   Input Parameter:
.  v - time counter

   Output Parameter:
.  v - time counter (v = v + current time)

   Notes:
   Since the PETSc libraries incorporate timing of phases and operations, 
   PetscTimeAdd() is intended only for timing of application codes.  
   The options database commands -log, -log_summary, and -log_all activate
   PETSc library timing.  See the users manual for further details.

.seealso:  PetscTime(), PetscTimeSubtract()

.keywords:  Petsc, time, add
*/

/* ------------------------------------------------------------------
    Some machines have very fast MPI_Wtime()
*/
#if (defined(PETSC_HAVE_FAST_MPI_WTIME) && !defined(__MPIUNI_H))
#define PetscTime(v)         (v)=MPI_Wtime();

#define PetscTimeSubtract(v) (v)-=MPI_Wtime();

#define PetscTimeAdd(v)      (v)+=MPI_Wtime();

/* ------------------------------------------------------------------
   Power1,2,3,PC machines have a fast clock read_real_time()
*/ 
#elif defined(PETSC_USE_READ_REAL_TIME)
EXTERN PetscLogDouble rs6000_time(void);
#define PetscTime(v)         (v)=rs6000_time();

#define PetscTimeSubtract(v) (v)-=rs6000_time();

#define PetscTimeAdd(v)      (v)+=rs6000_time();

/* ------------------------------------------------------------------
    Dec Alpha has a very fast system clock accessible through getclock()
    getclock() doesn't seem to have a prototype for C++
*/
#elif defined(PETSC_USE_GETCLOCK)
EXTERN_C_BEGIN
EXTERN int getclock(int clock_type,struct timespec *tp);
EXTERN_C_END


#define PetscTime(v)         {static struct  timespec _tp; \
                             getclock(TIMEOFDAY,&_tp); \
                             (v)=((PetscLogDouble)_tp.tv_sec)+(1.0e-9)*(_tp.tv_nsec);}

#define PetscTimeSubtract(v) {static struct timespec  _tp; \
                             getclock(TIMEOFDAY,&_tp); \
                             (v)-=((PetscLogDouble)_tp.tv_sec)+(1.0e-9)*(_tp.tv_nsec);}

#define PetscTimeAdd(v)      {static struct timespec  _tp; \
                             getclock(TIMEOFDAY,&_tp); \
                             (v)+=((PetscLogDouble)_tp.tv_sec)+(1.0e-9)*(_tp.tv_nsec);}

/* ------------------------------------------------------------------
   ASCI RED machine has a fast clock accessiable through dclock() 
*/
#elif defined (PETSC_USE_DCLOCK)
EXTERN_C_BEGIN
EXTERN PetscLogDouble dclock();
EXTERN_C_END

#define PetscTime(v)         (v)=dclock();

#define PetscTimeSubtract(v) (v)-=dclock();

#define PetscTimeAdd(v)      (v)+=dclock();


/* ------------------------------------------------------------------
   Windows uses a special time code
*/
#elif defined (PETSC_USE_NT_TIME)
#include <time.h>
EXTERN_C_BEGIN
EXTERN PetscLogDouble nt_time(void);
EXTERN_C_END
#define PetscTime(v)         (v)=nt_time();

#define PetscTimeSubtract(v) (v)-=nt_time();

#define PetscTimeAdd(v)      (v)+=nt_time();

/* ------------------------------------------------------------------
    The usual Unix time routines.
*/
#else
#define PetscTime(v)         {static struct timeval _tp; \
                             gettimeofday(&_tp,(struct timezone *)0);\
                             (v)=((PetscLogDouble)_tp.tv_sec)+(1.0e-6)*(_tp.tv_usec);}

#define PetscTimeSubtract(v) {static struct timeval _tp; \
                             gettimeofday(&_tp,(struct timezone *)0);\
                             (v)-=((PetscLogDouble)_tp.tv_sec)+(1.0e-6)*(_tp.tv_usec);}

#define PetscTimeAdd(v)      {static struct timeval _tp; \
                             gettimeofday(&_tp,(struct timezone *)0);\
                             (v)+=((PetscLogDouble)_tp.tv_sec)+(1.0e-6)*(_tp.tv_usec);}
#endif

#endif





