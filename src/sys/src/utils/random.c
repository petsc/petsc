#ifndef lint
static char vcid[] = "$Id: rndm.c,v 1.1 1994/03/18 00:21:49 gropp Exp $";
#endif

/*
    This file contains routines for interfacing to random number generators.
    This provides more than just an interface to some system random number
    generator:

    Numbers can be shuffled for use as random tuples

    Multiple random number generators may be used

    I'm still not sure what interface I want here.  There should be
    one to reinitialize and set the seed.
 */

#include "tools.h"

/* Private data */
typedef struct {
    unsigned long seed;
    /* array for shuffling ??? */
    } SYRndm;

#if defined(sun4) || defined(rs6000)
extern double drand48();
void *SYCreateRndm()
{
srand48(0x12345678);
return 0;
}

void SYFreeRndm( r )
SYRndm *r;
{
FREE( r );
}

double SYDRndm( r )
SYRndm *r;
{
return drand48();
}

/* Question:  should there be a routine for a random vector */	
#else
/* Should put a simple, portable random number generator here */
void *SYCreateRndm()
{
return 0;
}

void SYFreeRndm( r )
SYRndm *r;
{
FREE( r );
}

double SYDRndm( r )
SYRndm *r;
{
return 0.5;
}
#endif
