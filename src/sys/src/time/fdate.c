#ifndef lint
static char vcid[] = "$Id: fdate.c,v 1.1 1996/01/30 18:27:25 bsmith Exp bsmith $";
#endif

#include "files.h"

char *SYGetDate()
{
  struct timeval tp;
  gettimeofday( &tp, (struct timezone *)0 );
  return asctime(localtime((time_t *) &tp.tv_sec));
}
