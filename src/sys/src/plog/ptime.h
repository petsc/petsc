/* $Id: ptime.h,v 1.22 1996/09/24 20:36:07 balay Exp balay $ */
/*
       Low cost access to system time. This, in general, should not
     be included in user programs.
*/

#if !defined(__PTIME_PACKAGE)
#define __PTIME_PACKAGE

/*
   PetscTime - Returns the current time of day in seconds.  

   Output Parameter:
.  v - time counter

   Synopsis:
   PetscTime(double v)

   Usage: 
     double v;
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
   PetscTimeSubtract(double v)

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
   PetscTimeAdd(double v)

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
#include <sys/types.h>
#include <sys/time.h>
#if defined(__cplusplus) 
extern "C" { extern UTP_readTime(struct timestruc_t *);}
#else 
extern UTP_readTime(struct timestruc_t *);
#endif

#define PetscTime(v)         {static struct  timestruc_t _tp; \
                             UTP_readTime(&_tp); \
                             (v)=((double)_tp.tv_sec)+(1.0e-9)*(_tp.tv_nsec);}

#define PetscTimeSubtract(v) {static struct timestruc_t  _tp; \
                             UTP_readTime(&_tp); \
                             (v)-=((double)_tp.tv_sec)+(1.0e-9)*(_tp.tv_nsec);}

#define PetscTimeAdd(v)      {static struct timestruc_t  _tp; \
                             UTP_readTime(&_tp); \
                             (v)+=((double)_tp.tv_sec)+(1.0e-9)*(_tp.tv_nsec);}
/*
    Dec Alpha has a very fast system clock accessible through getclock()
*/
#elif defined(PARCH_alpha)
#include <sys/timers.h>

#define PetscTime(v)         {static struct  timespec _tp; \
                             UTP_readTime(&_tp); \
                             (v)=((double)_tp.tv_sec)+(1.0e-9)*(_tp.tv_nsec);}

#define PetscTimeSubtract(v) {static struct timespec  _tp; \
                             UTP_readTime(&_tp); \
                             (v)-=((double)_tp.tv_sec)+(1.0e-9)*(_tp.tv_nsec);}

#define PetscTimeAdd(v)      {static struct timespec  _tp; \
                             UTP_readTime(&_tp); \
                             (v)+=((double)_tp.tv_sec)+(1.0e-9)*(_tp.tv_nsec);}

/*
    Cray MPI implementation has very fast MPI_Wtime()
*/
#elif defined(PARCH_t3d)
#define PetscTime(v)         (v)=MPI_Wtime();

#define PetscTimeSubtract(v) (v)-=MPI_Wtime();

#define PetscTimeAdd(v)      (v)+=MPI_Wtime();

#elif defined(HAVE_SYS_TIME_H)
/*
    The usual Unix time routines.
*/
#if (defined(PARCH_IRIX)  || defined(PARCH_IRIX64)) && defined(__cplusplus)
struct timeval {
        long    tv_sec;         /* seconds */
        long    tv_usec;        /* and microseconds */
};

struct timezone {
        int     tz_minuteswest; /* minutes west of Greenwich */
        int     tz_dsttime;     /* type of dst correction */
};
extern "C" {
extern int gettimeofday(struct timeval *, struct timezone *);
}
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
                             (v)=((double)_tp.tv_sec)+(1.0e-6)*(_tp.tv_usec);}

#define PetscTimeSubtract(v) {static struct timeval _tp; \
                             gettimeofday(&_tp,(struct timezone *)0);\
                             (v)-=((double)_tp.tv_sec)+(1.0e-6)*(_tp.tv_usec);}

#define PetscTimeAdd(v)      {static struct timeval _tp; \
                             gettimeofday(&_tp,(struct timezone *)0);\
                             (v)+=((double)_tp.tv_sec)+(1.0e-6)*(_tp.tv_usec);}
#else
/*
    The time on Windows NT systems. 
*/
#define PetscTime(v)
#define PetscTimeSubtract(v)
#define PetscTimeAdd(v)      {static struct timeval _tp; \

#endif

#endif


