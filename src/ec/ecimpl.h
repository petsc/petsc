/* $Id: ecimpl.h,v 1.1 1997/01/07 17:15:13 bsmith Exp bsmith $ */

#ifndef _ECIMPL
#define _ECIMPL

#include "ec.h"


/*
     Maximum number of monitors you can run with a single EC
*/
#define MAXECMONITORS 5 

/*
   Defines the KSP data structure.
*/
struct _EC {
  PETSCHEADER

  ECProblemType     problemtype;       /* generalized or plain eigenvalue problem */
  ECSpectrumPortion spectrumportion;   /* largest, smallest etc */
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


  /*------------ Major routines which act on EC-----------------*/
  int  (*setup)(EC);
  int  (*solve)(EC);   
  int  (*solveeigenvectors)(EC);   
  int  (*setfromoptions)(EC);
  int  (*printhelp)(EC,char*);
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
