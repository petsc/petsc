

/*
   Some utilities written in C. Just to demonstrate that it can be done,
 these could equally well be written in F90
*/

#include "esmfgrid.h"

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define cesmfgridview   CESMFGRIDVIEW
#define cesmfgridviewsr CESMFGRIDVIEWSR
#define cesmfgridviewuq CESMFGRIDVIEWUQ
#elif defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define cesmfgridview   cesmfgridview_
#define cesmfgridviewsr cesmfgridviewsr_
#define cesmfgridviewuq cesmfgridviewuq_
#endif



EXTERN_C_BEGIN
void cesmfgridview(ESMFGrid grid)
{
  
  PetscPrintf(grid->base.fcomm,"Base ESMFGrid grid viewer\n typename %s dimension %d\n",grid->base.type_name,grid->dimension);
}

void cesmfgridviewsr(ESMFGridStructuredRectangular grid)
{
  int rank;
  MPI_Comm comm = grid->grid.base.fcomm;
  MPI_Comm_rank(comm,&rank);
  PetscPrintf(comm,"Global grid size %d %d\n",grid->globalsizes[0],grid->globalsizes[1]);
  PetscSynchronizedPrintf(comm,"[%d] local grid logical corners %d %d %d %d\n",rank,grid->localrange[0][0],
                          grid->localrange[1][0],grid->localrange[0][1],grid->localrange[1][1]);
  PetscSynchronizedPrintf(comm,"[%d] ghost grid logical corners %d %d %d %d\n",rank,grid->ghostrange[0][0],
                          grid->ghostrange[1][0],grid->ghostrange[0][1],grid->ghostrange[1][1]);
  PetscSynchronizedFlush(comm);
}

void cesmfgridviewuq(ESMFGridUnStructuredQuads grid)
{
  
}
EXTERN_C_END

