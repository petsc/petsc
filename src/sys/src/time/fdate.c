#ifndef lint
static char vcid[] = "$Id: fdate.c,v 1.2 1996/02/08 18:26:06 bsmith Exp bsmith $";
#endif

#include "files.h"

char *PetscGetDate()
{
  struct timeval tp;
  gettimeofday( &tp, (struct timezone *)0 );
  return asctime(localtime((time_t *) &tp.tv_sec));
}
