/* $Id: dvecimpl.h,v 1.13 1999/01/27 21:19:59 balay Exp bsmith $ */
/* 
   This should not be included in users code.

  Includes definition of structure for seqential double precision vectors

  These are shared by dvec1.c dvec2.c dvec3.c bvec1.c bvec2.c 
  pvectors/pvec.c pvectors/pbvec.c 
*/

#ifndef __DVECIMPL 
#define __DVECIMPL

#include "src/vec/vecimpl.h"

typedef struct { 
  VECHEADER
} Vec_Seq;

extern int VecMDot_Seq(int ,Vec ,const Vec[], Scalar *);
extern int VecMTDot_Seq(int ,Vec ,const Vec[], Scalar *);
extern int VecMin_Seq(Vec ,int* ,double * );
extern int VecSet_Seq(const Scalar* ,Vec );
extern int VecSetRandom_Seq(PetscRandom ,Vec );
extern int VecMAXPY_Seq( int,const Scalar *, Vec, Vec *);
extern int VecAYPX_Seq(const Scalar *, Vec , Vec);
extern int VecWAXPY_Seq(const Scalar*,Vec,Vec,Vec);
extern int VecPointwiseMult_Seq( Vec, Vec, Vec);
extern int VecPointwiseDivide_Seq(Vec,Vec,Vec);
extern int VecGetArray_Seq(Vec,Scalar *[]);
extern int VecRestoreArray_Seq(Vec,Scalar *[]);
extern int VecPlaceArray_Seq(Vec,const Scalar *);
extern int VecReplaceArray_Seq(Vec,const Scalar *);
extern int VecGetSize_Seq(Vec,int *);
extern int VecDot_Seq(Vec, Vec,Scalar *);
extern int VecTDot_Seq(Vec, Vec,Scalar *);
extern int VecScale_Seq(const Scalar *,Vec);
extern int VecCopy_Seq(Vec, Vec);
extern int VecSwap_Seq(Vec,Vec);
extern int VecAXPY_Seq(const Scalar *, Vec, Vec);
extern int VecAXPBY_Seq(const Scalar *,const Scalar *,Vec, Vec);
extern int VecMax_Seq(Vec,int*,double *);
extern int VecDuplicate_Seq(Vec,Vec *);
extern int VecGetMap_Seq(Vec,Map *);

#endif
