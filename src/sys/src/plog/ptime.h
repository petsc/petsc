/* $Id: ptime.h,v 1.49 1998/04/25 16:11:07 balay Exp balay $ */
/*
       Low cost access to system time. This, in general, should not
     be included in user programs.
*/

#if !defined(__PTIME_PACKAGE)
#define __PTIME_PACKAGE

#include "petsc.h"

/*
   PetscTime - Returns the current time of day in seconds.  

   Output Parameter:
.  v - time counter

   Synopsis:
   PetscTime(PLogDouble v)

   Usage: 
     PLogDouble v;
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

   Input Parameter:
.  v - time counter

   Output Parameter:
.  v - time counter (v = v - current time)

   Synopsis:
   PetscTimeSubtract(PLogDouble v)

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

   Input Parameter:
.  v - time counter

   Output Parameter:
.  v - time counter (v = v + current time)

   Synopsis:
   PetscTimeAdd(PLogDouble v)

   Notes:
   Since the PETSc libraries incorporate timing of phases and operations, 
   PetscTimeAdd() is intended only for timing of application codes.  
   The options database commands -log, -log_summary, and -log_all activate
   PETSc library timing.  See the users manual for further details.

.seealso:  PetscTime(), PetscTimeSubtract()

.keywords:  Petsc, time, add
*/

/*
    Defines the interface to the IBM rs6000 high accuracy clock. The 
  routine used is defined in petsc/src/sys/src/rs6000_time.h.
*/ 
#if defined(PARCH_rs6000)
#include <sys/time.h>
struct my_timestruc_t {
  unsigned long tv_sec;/* seconds*/
  long          tv_nsec;/* and nanoseconds*/
};
#if defined (__cplusplus)
extern "C" { extern rs6000_time(struct my_timestruc_t *);}
#else
extern rs6000_time(struct my_timestruc_t *);
#endif
#define PetscTime(v)         {static struct  my_timestruc_t _tp; \
                             rs6000_time(&_tp); \
                             (v)=((PLogDouble)_tp.tv_sec)+(1.0e-9)*(_tp.tv_nsec);}

#define PetscTimeSubtract(v) {static struct my_timestruc_t  _tp; \
                             rs6000_time(&_tp); \
                             (v)-=((PLogDouble)_tp.tv_sec)+(1.0e-9)*(_tp.tv_nsec);}

#define PetscTimeAdd(v)      {static struct my_timestruc_t  _tp; \
                             rs6000_time(&_tp); \
                             (v)+=((PLogDouble)_tp.tv_sec)+(1.0e-9)*(_tp.tv_nsec);}
/*
    Dec Alpha has a very fast system clock accessible through getclock()
    the Clock is not accessible from gcc/g++
*/
#elif defined(PARCH_alpha) && !defined (__cplusplus)
#include <sys/types.h>
#include <sys/time.h>

#define PetscTime(v)         {static struct  timespec _tp; \
                             getclock(TIMEOFDAY,&_tp); \
                             (v)=((PLogDouble)_tp.tv_sec)+(1.0e-9)*(_tp.tv_nsec);}

#define PetscTimeSubtract(v) {static struct timespec  _tp; \
                             getclock(TIMEOFDAY,&_tp); \
                             (v)-=((PLogDouble)_tp.tv_sec)+(1.0e-9)*(_tp.tv_nsec);}

#define PetscTimeAdd(v)      {static struct timespec  _tp; \
                             getclock(TIMEOFDAY,&_tp); \
                             (v)+=((PLogDouble)_tp.tv_sec)+(1.0e-9)*(_tp.tv_nsec);}

/*
    Some machines have very fast MPI_Wtime()
*/
#elif (defined(HAVE_FAST_MPI_WTIME) && !defined(PETSC_USING_MPIUNI))
#include <sys/time.h>
#define PetscTime(v)         (v)=MPI_Wtime();

#define PetscTimeSubtract(v) (v)-=MPI_Wtime();

#define PetscTimeAdd(v)      (v)+=MPI_Wtime();

#elif defined(PARCH_nt_gnu) || defined (PARCH_nt)
#include <time.h>
extern PLogDouble nt_time();
#define PetscTime(v)         (v)=nt_time();

#define PetscTimeSubtract(v) (v)-=nt_time();

#define PetscTimeAdd(v)      (v)+=nt_time();

#elif defined(HAVE_SYS_TIME_H)
/*
    The usual Unix time routines.
*/
#if (defined(PARCH_IRIX5_oldcode) && defined(__cplusplus))
struct timeval {
  long tv_sec;         /* seconds */
  long tv_usec;        /* and microseconds */
};

struct timezone {
  int tz_minuteswest; /* minutes west of Greenwich */
  int tz_dsttime;     /* type of dst correction */
};
extern "C" {
/*
extern int gettimeofday(struct timeval *, struct timezone *);
*/
extern int gettimeofday(struct timeval *,...);
}
#elif ( defined (PARCH_IRIX64) ||  defined (PARCH_IRIX) || defined(PARCH_IRIX5))
#include <sys/resource.h>
#if defined(__cplusplus)
extern "C" {
/* 
extern int gettimeofday(struct timeval *, struct timezone *);
*/
extern int gettimeofday(struct timeval *,...);
}
#endif

#else
#include <sys/types.h>
#include <sys/time.h>
#endif

#if defined(PARCH_sun4) && !defined(__cplusplus)
extern int gettimeofday(struct timeval *, struct timezone *);
#endif
/*
   With Solaris 5.3 (and maybe 5.4) you should add the   
   || defined(PARCH_solaris) below
*/
#if defined(PARCH_sun4) && defined(__cplusplus)
extern "C" {
extern int gettimeofday(struct timeval *, struct timezone *);
}
#endif

#define PetscTime(v)         {static struct timeval _tp; \
                             gettimeofday(&_tp,(struct timezone *)0);\
                             (v)=((PLogDouble)_tp.tv_sec)+(1.0e-6)*(_tp.tv_usec);}

#define PetscTimeSubtract(v) {static struct timeval _tp; \
                             gettimeofday(&_tp,(struct timezone *)0);\
                             (v)-=((PLogDouble)_tp.tv_sec)+(1.0e-6)*(_tp.tv_usec);}

#define PetscTimeAdd(v)      {static struct timeval _tp; \
                             gettimeofday(&_tp,(struct timezone *)0);\
                             (v)+=((PLogDouble)_tp.tv_sec)+(1.0e-6)*(_tp.tv_usec);}
#endif

#endif





