/*
      Structure used for Multigrid code 
*/
#if !defined(__MG)
#define __MG

#define Multiplicative 0
#define Additive       1

typedef void Vector;

typedef struct 
{
    int    pre, post;              /* Number smooths*/
    int    cycles;                 /* Number cycles to run */
    int    level;                  /* level = 0 coarsest level */
    Vector *b;                     /* Right hand side */ 
    Vector *x;                     /* Solution */
    Vector *r;                     /* Residual */
    void   (*residual)(); 
    void   (*smoothd)(); 
    void   (*smoothu)(); 
    void   (*interpolate)(); 
    void   (*restrict)(); 
    void   (*coarse_solve)();
    void   (*zerovector)(); 
    void   *rp, *sdp, *sup, *ip, *rtp, *cp, *zp;
} MG;

MG **MGCreate();
void MGDestroy();
int  MGCheck();
void MGSetNumberSmoothUp();
void MGSetNumberSmoothDown();
void MGSetCycles();
void MGACycle();
void MGMCycle();
void MGFCycle();

/*MC
      MGCycle - Runs either an additive or multiplicative cycle of multigrid. 

  Input Parameters:
.   mg - the multigrid context 
.   am - either Multiplicative or Additive 

   Synopsis:
    void  MGCycle(mg,am)
    MG   **mg;
    int  am;

  Note: a simple macro wrapper which calls MGMCycle() or MGACycle(). 
M*/ 
#define MGCycle(mg,am) if (am == Multiplicative) MGMCycle(mg); \
                       else MGACycle(mg);

#define MGERR(l) if (l > (mg)[0]->level) {SETERR(1); return;}

/*MC
      MGSetCoarseSolve - Sets the solver function to be used on the 


  Input Parameters:
.   mg - the multigrid context 
.   f - the solver function
.   c - the solver context which is passed to solver function 

   Synopsis:
    void  MGSetCoarseSolve(mg,f,c)
    MG   **mg;
    void (*f)();
    void *c;
M*/ 
#define MGSetCoarseSolve(mg,f,c)  { \
              (mg)[(mg)[0]->level]->coarse_solve = f;  \
              (mg)[(mg)[0]->level]->cp           = c;}

/*MC
      MGSetResidual - Sets the function to be used to calculate the 
                      residual on the lth level. 

  Input Parameters:
.   mg - the multigrid context 
.   f - the residual function
.   c - the context which is passed to residual function 
.   l - the level this is to be used for

   Synopsis:
    void  MGSetResidual(mg,l,f,c)
    MG   **mg;
    void (*f)();
    void *c;
    int  l;
M*/
#define MGSetResidual(mg,l,f,c)  { MGERR(l);\
              (mg)[(mg)[0]->level - (l)]->residual = f;  \
              (mg)[(mg)[0]->level - (l)]->rp       = c;}

/*MC
      MGSetInterpolate - Sets the function to be used to calculate the 
                      interpolation on the lth level. 

  Input Parameters:
.   mg - the multigrid context 
.   f - the interpolation function
.   c - the context which is passed to interpolation function 
.   l - the level this is to be used for

   Synopsis:
    void  MGSetInterpolate(mg,l,f,c)
    MG   **mg;
    void (*f)();
    void *c;
    int  l;
M*/
#define MGSetInterpolate(mg,l,f,c)  { MGERR(l);\
              (mg)[(mg)[0]->level - (l)]->interpolate = f;  \
              (mg)[(mg)[0]->level - (l)]->ip          = c;}

/*MC
      MGSetZeroVector - Sets the function to be used to zero a vector
                        on the lth level. 

  Input Parameters:
.   mg - the multigrid context 
.   f - the function
.   c - the context which is passed to zero function 
.   l - the level this is to be used for

   Synopsis:
    void  MGSetZeroVector(mg,l,f,c)
    MG   **mg;
    void (*f)();
    void *c;
    int  l;
M*/
#define MGSetZeroVector(mg,l,f,c)  { MGERR(l);\
              (mg)[(mg)[0]->level - (l)]->zerovector = f;  \
              (mg)[(mg)[0]->level - (l)]->zp        = c;}

/*MC
      MGSetRestriction - Sets the function to be used to restrict vector
                        from lth level to l-1. 

  Input Parameters:
.   mg - the multigrid context 
.   f - the function
.   c - the context which is passed to function 
.   l - the level this is to be used for

   Synopsis:
    void  MGSetRestriction(mg,l,f,c)
    MG   **mg;
    void (*f)();
    void *c;
    int  l;
M*/
#define MGSetRestriction(mg,l,f,c)  { MGERR(l);\
              (mg)[(mg)[0]->level - (l)]->restrict  = f;  \
              (mg)[(mg)[0]->level - (l)]->rtp       = c;}

/*MC
      MGSetSmootherUp - Sets the function to be used as smoother after 
                        coarse grid correction (post-smoother). 

  Input Parameters:
.   mg - the multigrid context 
.   f - the function
.   c - the context which is passed to function 
.   l - the level this is to be used for
.   d - the number of smoothing steps to make

   Synopsis:
    void  MGSetSmootherUp(mg,l,f,c,d)
    MG   **mg;
    void (*f)();
    void *c;
    int  l,d;
M*/
#define MGSetSmootherUp(mg,l,f,c,d)  { MGERR(l);\
              (mg)[(mg)[0]->level - (l)]->smoothu  = f;  \
              (mg)[(mg)[0]->level - (l)]->sup       = c; \
              (mg)[(mg)[0]->level - (l)]->post      = d;}

/*MC
      MGSetSmootherDown - Sets the function to be used as smoother before 
                        coarse grid correction (post-smoother). 

  Input Parameters:
.   mg - the multigrid context 
.   f - the function
.   c - the context which is passed to function 
.   l - the level this is to be used for
.   d - the number of smoothing steps to make

   Synopsis:
    void  MGSetSmootherDown(mg,l,f,c,d)
    MG   **mg;
    void (*f)();
    void *c;
    int  l,d;
M*/
#define MGSetSmootherDown(mg,l,f,c,d)  { MGERR(l); \
              (mg)[(mg)[0]->level - (l)]->smoothd  = f;  \
              (mg)[(mg)[0]->level - (l)]->sdp      = c; \
              (mg)[(mg)[0]->level - (l)]->pre      = d;}

/*MC
      MGSetCyclesOnLevel - Sets the number of cycles to run on this level. 

  Input Parameters:
.   mg - the multigrid context 
.   l - the level this is to be used for
.   n - the number of cycles

   Synopsis:
    void  MGSetCyclesOnLevel(mg,l,n)
    MG   **mg;
    int  l,n;
M*/
#define MGSetCyclesOnLevel(mg,l,c)  { MGERR(l); \
              (mg)[(mg)[0]->level - (l)]->cycles  = c;}

/*MC
      MGSetRhs - Sets the vector space to be used to store right hand 
                 side on a particular level. User should free this 
                 space at conclusion of multigrid use. 

  Input Parameters:
.   mg - the multigrid context 
.   l - the level this is to be used for
.   c - the space

   Synopsis:
    void  MGSetRhs(mg,l,c)
    MG   **mg;
    void *c;
    int  l;
M*/
#define MGSetRhs(mg,l,c)  { MGERR(l); \
              (mg)[(mg)[0]->level - (l)]->b  = c;}

/*MC
      MGSetX - Sets the vector space to be used to store solution 
                 on a particular level.User should free this 
                 space at conclusion of multigrid use.

  Input Parameters:
.   mg - the multigrid context 
.   l - the level this is to be used for
.   c - the space

   Synopsis:
    void  MGSetX(mg,l,c)
    MG   **mg;
    void *c;
    int  l;
M*/
#define MGSetX(mg,l,c)  { MGERR(l); \
              (mg)[(mg)[0]->level - (l)]->x  = c;}

/*MC
      MGSetR - Sets the vector space to be used to store residual 
                 on a particular level. User should free this 
                 space at conclusion of multigrid use.

  Input Parameters:
.   mg - the multigrid context 
.   l - the level this is to be used for
.   c - the space

   Synopsis:
    void  MGSetR(mg,l,c)
    MG   **mg;
    void *c;
    int  l;
M*/
#define MGSetR(mg,l,c)  { MGERR(l); \
              (mg)[(mg)[0]->level - (l)]->r  = c;}

#endif

