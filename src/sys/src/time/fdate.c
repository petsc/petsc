#ifndef lint
static char vcid[] = "$Id: fdate.c,v 1.3 1996/03/19 21:24:22 bsmith Exp bsmith $";
#endif

#include "src/sys/src/files.h"

char *PetscGetDate()
{
  struct timeval tp;
  gettimeofday( &tp, (struct timezone *)0 );
  return asctime(localtime((time_t *) &tp.tv_sec));
}
