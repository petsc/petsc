/* $Id: ptime.h,v 1.61 1999/04/01 22:38:41 balay Exp balay $ */
/*
       Low cost access to system time. This, in general, should not
     be included in user programs.
*/

#if !defined(__PTIME_H)
#define __PTIME_H

#include "petsc.h"
#include "pinclude/petscfix.h"
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

/* ------------------------------------------------------------------
    Some machines have very fast MPI_Wtime()
*/
#if (defined(HAVE_FAST_MPI_WTIME) && !defined(USING_MPIUNI))
#define PetscTime(v)         (v)=MPI_Wtime();

#define PetscTimeSubtract(v) (v)-=MPI_Wtime();

#define PetscTimeAdd(v)      (v)+=MPI_Wtime();
/* ------------------------------------------------------------------

    Defines the interface to the IBM rs6000 high accuracy clock. The 
  routine used is defined in petsc/src/sys/src/time/rs6000_time.s
*/ 
#elif defined(USE_IBM_ASM_CLOCK)
#include <sys/time.h>
struct my_timestruc_t {
  unsigned long tv_sec;/* seconds*/
  long          tv_nsec;/* and nanoseconds*/
};
EXTERN_C_BEGIN
extern void rs6000_time(struct my_timestruc_t *);
EXTERN_C_END
#define PetscTime(v)         {static struct  my_timestruc_t _tp; \
                             rs6000_time(&_tp); \
                             (v)=((PLogDouble)_tp.tv_sec)+(1.0e-9)*(_tp.tv_nsec);}

#define PetscTimeSubtract(v) {static struct my_timestruc_t  _tp; \
                             rs6000_time(&_tp); \
                             (v)-=((PLogDouble)_tp.tv_sec)+(1.0e-9)*(_tp.tv_nsec);}

#define PetscTimeAdd(v)      {static struct my_timestruc_t  _tp; \
                             rs6000_time(&_tp); \
                             (v)+=((PLogDouble)_tp.tv_sec)+(1.0e-9)*(_tp.tv_nsec);}

/* ------------------------------------------------------------------
    Dec Alpha has a very fast system clock accessible through getclock()
    the Clock is not accessible from gcc/g++
*/
#elif defined(USE_GETCLOCK)
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

/* ------------------------------------------------------------------
   ASCI RED machine has a fast clock accessiable through dclock() 
*/
#elif defined (USE_DCLOCK)
EXTERN_C_BEGIN
extern PLogDouble dclock();
EXTERN_C_BEGIN

#define PetscTime(v)         (v)=dclock();

#define PetscTimeSubtract(v) (v)-=dclock();

#define PetscTimeAdd(v)      (v)+=dclock();


/* ------------------------------------------------------------------
   NT uses a special time code
*/
#elif defined (USE_NT_TIME)
#include <time.h>
extern PLogDouble nt_time();
#define PetscTime(v)         (v)=nt_time();

#define PetscTimeSubtract(v) (v)-=nt_time();

#define PetscTimeAdd(v)      (v)+=nt_time();

/* ------------------------------------------------------------------
    The usual Unix time routines.
*/
#elif defined(HAVE_SYS_TIME_H)

#include <sys/types.h>
#include <sys/time.h>

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





