/* $Id: snes.h,v 1.17 1995/06/02 21:05:19 bsmith Exp bsmith $ */

#if !defined(__PTIME_PACKAGE)
#defined __PTIME_PACKAGE

#include <sys/types.h>
#include <sys/time.h>

/*
     Macros for timing. In the future some of these may be 
    machine dependent versions
*/


#define PetscTime(v)         {struct timeval _tp; \
                             gettimeofday(&_tp,(struct timezone *)0);\
                             (v)=((double)_tp.tv_sec)+(1.0e-6)*(_tp.tv_usec);}
#define PetscTimeSubtract(v) {struct timeval _tp; \
                             gettimeofday(&_tp,(struct timezone *)0);\
                             (v)-=((double)_tp.tv_sec)+(1.0e-6)*(_tp.tv_usec);}
#define PetscTimeAdd(v)      {struct timeval _tp; \
                             gettimeofday(&_tp,(struct timezone *)0);\
                             (v)+=((double)_tp.tv_sec)+(1.0e-6)*(_tp.tv_usec);}

#endif
