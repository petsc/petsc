#include <sys/types.h>
#include <sys/time.h>
/*
     For timing!
*/


#define PetscTime(v) {struct timeval _tp; \
        gettimeofday(&_tp,(struct timezone *)0);\
        (v) = ((double)_tp.tv_sec) + (1.0e-6)*(_tp.tv_usec);}
#define PetscTimeSubtract(v) {struct timeval _tp; \
        gettimeofday(&_tp,(struct timezone *)0);\
        (v) -= ((double)_tp.tv_sec) + (1.0e-6)*(_tp.tv_usec);}
#define PetscTimeAdd(v) {struct timeval _tp; \
        gettimeofday(&_tp,(struct timezone *)0);\
        (v) += ((double)_tp.tv_sec) + (1.0e-6)*(_tp.tv_usec);}

