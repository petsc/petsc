
/*
       Tests the ESI_MapAlgebraic interface
*/
#include "ESI.h"

extern int ESI_Map_test(ESI_Map *);

int ESI_MapAlgebraic_test(ESI_MapAlgebraic *map)
{
  int ierr,length,offset,*offsets;

  ierr = ESI_Map_test((ESI_Map*) map); if (ierr) return ierr;
  if (ierr) {printf("error calling ESI_Map_test\n");return ierr;}

  ierr = map->getLocalInfo(length,offset); if (ierr) return ierr;
  if (ierr) {printf("error calling mapalgebraic->getLocalInfo\n");return ierr;}
  printf("ESI_MapAlgebraic_test: local length %d offset %d\n",length,offset);
  ierr = map->getGlobalInfo(length,offsets); if (ierr) return ierr;
  if (ierr) {printf("error calling mapalgebraic->getGlobalInfo\n");return ierr;}
  printf("ESI_MapAlgebraic_test: total length %d first offset %d\n",length,offsets[0]);
  return 0;
}







