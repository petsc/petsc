#define PETSC_DLL

#include "petscsys.h"              /*I "petscsys.h" I*/

struct _p_DrawAxis {
  PETSCHEADER(int);
    PetscReal      xlow,ylow,xhigh,yhigh;     /* User - coord limits */
    PetscErrorCode (*ylabelstr)(PetscReal,PetscReal,char **);/* routines to generate labels */ 
    PetscErrorCode (*xlabelstr)(PetscReal,PetscReal,char **);
    PetscErrorCode (*xticks)(PetscReal,PetscReal,int,int*,PetscReal*,int);
    PetscErrorCode (*yticks)(PetscReal,PetscReal,int,int*,PetscReal*,int);  
                                          /* location and size of ticks */
    PetscDraw  win;
    int        ac,tc,cc;                     /* axis,tick, character color */
    char       *xlabel,*ylabel,*toplabel;
    PetscBool  hold;
};

#define MAXSEGS 20

EXTERN PetscErrorCode PetscADefTicks(PetscReal,PetscReal,int,int*,PetscReal*,int);
EXTERN PetscErrorCode PetscADefLabel(PetscReal,PetscReal,char**);
EXTERN PetscErrorCode PetscAGetNice(PetscReal,PetscReal,int,PetscReal*);
EXTERN PetscErrorCode PetscAGetBase(PetscReal,PetscReal,int,PetscReal*,int*);

EXTERN PetscErrorCode PetscStripAllZeros(char*);
EXTERN PetscErrorCode PetscStripTrailingZeros(char*);
EXTERN PetscErrorCode PetscStripInitialZero(char*);
EXTERN PetscErrorCode PetscStripZeros(char*);
EXTERN PetscErrorCode PetscStripZerosPlus(char*);
