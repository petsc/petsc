/* $Id: ecimpl.h,v 1.4 1997/09/12 00:06:30 curfman Exp bsmith $ */

#ifndef _ECIMPL
#define _ECIMPL

#include "ec.h"

/*
     Maximum number of monitors that can be run with a single EC
*/
#define MAXECMONITORS 5 

/*
   Defines the EC data structure.
*/
struct _p_EC {
  PETSCHEADER(int)

  ECProblemType     problemtype;       /* generalized or plain eigenvalue problem */
  ECSpectrumPortion spectrumportion;   /* largest, smallest, etc. */
  Scalar            location;

  /* --------Monitor routines (most return -1 on error) --------*/
  /*
        Called to monitor the convergence of the spectrum 
     in iterative eigenvalue computations.
  */
  int  (*monitor[MAXECMONITORS])(EC,int,double,void*); 
  void *monitorcontext[MAXECMONITORS];            
  int  numbermonitors;                  

  /* -------Convergence test routine ---------------------------*/
  int (*converged)(EC,int,double,void*);
  void *cnvP; 

  /*------------ Major routines that act on EC-----------------*/
  int  (*setup)(EC);
  int  (*solve)(EC);   
  int  (*solveeigenvectors)(EC);   
  int  (*setfromoptions)(EC);
  int  (*printhelp)(EC,char*);
  int  (*destroy)(EC);
  int  (*view)(EC,Viewer);
  void *data;

  /* ----------------Default work-area management -------------------- */
  int    nwork;
  Vec    *work;

  int    setupcalled;

  int    computeeigenvectors;
  int    n;  /* number of eigenvalues requested or default value */
  Mat    A,B;

  double *realpart,*imagpart;
  Vec    evecs;
};

#define ECMonitor(ec,it,rnorm) \
        { int _ierr,_i,_im = ec->numbermonitors; \
          for ( _i=0; _i<_im; _i++ ) {\
            _ierr = (*ec->monitor[_i])(ec,it,rnorm,ec->monitorcontext[_i]); \
            CHKERRQ(_ierr); \
	  } \
	}

#endif
