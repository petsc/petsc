/*
      Data structure used for DM based Multigrid preconditioner.
*/
#if !defined(__MG_IMPL)
#define __MG_IMPL
#include "private/pcimpl.h"
#include "petscmg.h"
#include "petscksp.h"


/*

*/
typedef struct
{
  PC      mg;
  DM      *dm;
} PCDMMG_MG;


#endif

