#ifndef lint
static char vcid[] = "$Id: file.c,v 1.27 1995/12/31 17:17:34 curfman Exp bsmith $";
#endif
/*
      Code for opening and closing files.
*/
#include "file.h"

char *SYGetDate()
{
  struct timeval tp;
  gettimeofday( &tp, (struct timezone *)0 );
  return asctime(localtime((time_t *) &tp.tv_sec));
}
