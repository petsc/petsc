long MPID_DUMMY = 0;
#include <sys/time.h>
double MPI_Wtime()
{

  struct timeval _tp; 
  
  gettimeofday(&_tp,(struct timezone *)0);
  return ((double)_tp.tv_sec)+(1.0e-6)*(_tp.tv_usec);
}
