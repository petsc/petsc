
#include <petscsys.h>              /*I "petscsys.h" I*/

struct _p_PetscDrawAxis {
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

extern PetscErrorCode PetscADefTicks(PetscReal,PetscReal,int,int*,PetscReal*,int);
extern PetscErrorCode PetscADefLabel(PetscReal,PetscReal,char**);
extern PetscErrorCode PetscAGetNice(PetscReal,PetscReal,int,PetscReal*);
extern PetscErrorCode PetscAGetBase(PetscReal,PetscReal,int,PetscReal*,int*);

extern PetscErrorCode PetscStripAllZeros(char*);
extern PetscErrorCode PetscStripTrailingZeros(char*);
extern PetscErrorCode PetscStripInitialZero(char*);
extern PetscErrorCode PetscStripZeros(char*);
extern PetscErrorCode PetscStripZerosPlus(char*);
