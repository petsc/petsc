/*
      Structure used for Multigrid code 
*/
#if !defined(__MG_PACKAGE)
#define __MG_PACKAGE

#define Multiplicative 0
#define Additive       1

typedef struct _MG* MG;


extern int MGCreate(MG *);
extern int MGDestroy(MG);
extern int MGCheck(MG);
extern int MGSetNumberSmoothUp(MG,int);
extern int MGSetNumberSmoothDown(MG,int);
extern int MGSetCycles(MG,int);
extern int MGACycle(MG);
extern int MGMCycle(MG);
extern int MGFCycle(MG);


#endif

