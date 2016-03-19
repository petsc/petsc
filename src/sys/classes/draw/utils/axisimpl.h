
#include <petscdraw.h>              /*I "petscdraw.h" I*/
#include <petsc/private/petscimpl.h>              /*I "petscsys.h" I*/

struct _p_PetscDrawAxis {
  PETSCHEADER(int);
  PetscReal      xlow,ylow,xhigh,yhigh;                    /* User - coord limits */
  PetscErrorCode (*ylabelstr)(PetscReal,PetscReal,char**);/* routines to generate labels */
  PetscErrorCode (*xlabelstr)(PetscReal,PetscReal,char**);
  PetscErrorCode (*xticks)(PetscReal,PetscReal,int,int*,PetscReal*,int);
  PetscErrorCode (*yticks)(PetscReal,PetscReal,int,int*,PetscReal*,int);
                                           /* location and size of ticks */
  PetscDraw win;
  int       ac,tc,cc;                     /* axis,tick, character color */
  char      *xlabel,*ylabel,*toplabel;
  PetscBool hold;
};

#define MAXSEGS 20

PETSC_INTERN PetscErrorCode PetscADefTicks(PetscReal,PetscReal,int,int*,PetscReal*,int);
PETSC_INTERN PetscErrorCode PetscADefLabel(PetscReal,PetscReal,char**);
PETSC_INTERN PetscErrorCode PetscAGetNice(PetscReal,PetscReal,int,PetscReal*);
PETSC_INTERN PetscErrorCode PetscAGetBase(PetscReal,PetscReal,int,PetscReal*,int*);

PETSC_INTERN PetscErrorCode PetscStripe0(char*);
PETSC_INTERN PetscErrorCode PetscStripAllZeros(char*);
PETSC_INTERN PetscErrorCode PetscStripTrailingZeros(char*);
PETSC_INTERN PetscErrorCode PetscStripInitialZero(char*);
PETSC_INTERN PetscErrorCode PetscStripZeros(char*);
PETSC_INTERN PetscErrorCode PetscStripZerosPlus(char*);
