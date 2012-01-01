#ifndef ad_utils_H_seen
#define ad_utils_H_seen

#undef ADUTILS_EXTERN
#if defined(__cplusplus)
#define ADUTILS_EXTERN extern "C"
#else
#define ADUTILS_EXTERN extern
#endif

typedef void DERIV_TYPE;

ADUTILS_EXTERN void PetscADResetIndep(void);
ADUTILS_EXTERN void PetscADSetValueAndColor(DERIV_TYPE *,int,ISColoringValue*,double *);
ADUTILS_EXTERN void PetscADSetValArray(DERIV_TYPE *,int,double *);
ADUTILS_EXTERN void PetscADSetIndepVector(DERIV_TYPE *,int,double *);
ADUTILS_EXTERN void PetscADSetIndepArrayColored(DERIV_TYPE *,int,int *);
ADUTILS_EXTERN int PetscADIncrementTotalGradSize(int);
ADUTILS_EXTERN void PetscADSetIndepDone(void);
ADUTILS_EXTERN void PetscADExtractGrad(double *,DERIV_TYPE *);
ADUTILS_EXTERN int  PetscADGetDerivTypeSize();
ADUTILS_EXTERN double *PetscADGetGradArray(DERIV_TYPE *);
#endif

