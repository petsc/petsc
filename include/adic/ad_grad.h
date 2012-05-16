/*
  THIS PROGRAM DISCLOSES MATERIAL PROTECTABLE UNDER COPYRIGHT
  LAWS OF THE UNITED STATES.  FOR LICENSING INFORMATION CONTACT:

  Paul Hovland and Boyana Norris, Mathematics and Computer Science Division,
  Argonne National Laboratory, 9700 S. Cass Avenue, Argonne IL 60439, 
  {hovland,norris}@mcs.anl.gov.
*/

#include "ad_grad_macro_axpys.h"

#if !defined(AD_GRAD_H)
#define AD_GRAD_H

#include <string.h>

PETSC_EXTERN int ad_grad_size;
PETSC_EXTERN int ad_total_grad_size;   /*only used when generating MPI programs*/
PETSC_EXTERN int ad_grad_size_shadow;

#   if defined(__cplusplus)
        extern "C" {
#   endif
    
#define ad_AD_GradInitMPI(pargc, pargv) \
    { \
        ad_mpi_init(pargc, pargv, &ad_total_grad_size); \
    }


#define ad_AD_GradFinalMPI() \
    { \
        ad_mpi_final(); \
    }

#define ad_AD_GradFinal() \
    { \
        ad_grad_size = 0; \
    }



#define ad_AD_GradInit(n) \
    { \
	if (n == -1) \
	   ad_grad_size = ad_GRAD_MAX; \
	else \
           ad_grad_size = n; \
        ad_grad_size_shadow = 0; \
    }


#define ad_AD_ClearGrad(gz) memset((char*)gz, 0, ad_GRAD_MAX*sizeof(double)); 

#define ad_AD_ClearGrad2(gz)\
    {\
        int iWiLlNeVeRCoNfLiCt0;\
        for (iWiLlNeVeRCoNfLiCt0 = 0 ; iWiLlNeVeRCoNfLiCt0 < ad_GRAD_MAX; \
             iWiLlNeVeRCoNfLiCt0++) {\
            gz[iWiLlNeVeRCoNfLiCt0] = 0.0;\
        }\
    }

#define ad_AD_ClearGradArray(ggz,size)\
    {\
        int iWiLlNeVeRCoNfLiCt0;\
        for (iWiLlNeVeRCoNfLiCt0 = 0 ; iWiLlNeVeRCoNfLiCt0 < size; \
             iWiLlNeVeRCoNfLiCt0++) {\
            ad_AD_ClearGrad(DERIV_grad((ggz)[iWiLlNeVeRCoNfLiCt0])); \
        }\
    }

#define ad_AD_CopyGrad(gz,gx) \
    {\
        int iWiLlNeVeRCoNfLiCt0;\
        for (iWiLlNeVeRCoNfLiCt0 = 0 ; iWiLlNeVeRCoNfLiCt0 < ad_GRAD_MAX;\
	     iWiLlNeVeRCoNfLiCt0++) {\
            gz[iWiLlNeVeRCoNfLiCt0] = gx[iWiLlNeVeRCoNfLiCt0];\
        }\
    }

#   define ad_AD_GetTotalGradSize() ad_grad_size
#   define ad_AD_SetTotalGradSize(x) ad_grad_size = x

#   define ad_AD_IncrementTotalGradSize(x) \
    { \
         if (x + ad_grad_size_shadow > ad_GRAD_MAX) {\
                SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER, "The number of independent variables exceeds the maximum compiled for!\n Edit your program and change Process adiC(%d) to Process adiC(%d)", x + ad_grad_size_shadow,ad_GRAD_MAX);\
        }\
        ad_grad_size_shadow += x;\
    }

#   define ad_AD_ExtractGrad(a, var) \
    { \
        int pOsItIoN; \
        for (pOsItIoN = 0; pOsItIoN < ad_grad_size; pOsItIoN++) {\
            (a)[pOsItIoN] = DERIV_grad(var)[pOsItIoN];  \
        }\
    }
#   define ad_AD_ExtractVal(a, var) \
    { \
	a = DERIV_val(var); \
    }
#   define ad_AD_SetGrad(a, var) \
    { \
        int pOsItIoN; \
        for (pOsItIoN = 0; pOsItIoN < ad_grad_size; pOsItIoN++) {\
            DERIV_grad(var)[pOsItIoN] = (a)[pOsItIoN];  \
        }\
    }

#   define ad_AD_SetIndepDone() ad_AD_CommitShadowVar()
#   define ad_AD_ResetIndep() ad_AD_ResetShadowVar()
#   define ad_AD_SetIndep(var) \
    { \
        int pOsItIoN = ad_AD_IncrShadowVar(); \
        if (pOsItIoN > ad_GRAD_MAX) {\
          SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER, "The number of independent variables exceeds the maximum compiled for!\n Edit your program and change Process adiC(%d) to Process adiC(%d)",pOsItIoN ,ad_GRAD_MAX);\
        }\
        ad_AD_ClearGrad(DERIV_grad(var)); \
        DERIV_grad(var)[pOsItIoN] = 1; \
    }
#   define ad_AD_SetIndepArray(vars, size) \
    { \
        int iWiLlNeVeRCoNfLiCt; \
        for (iWiLlNeVeRCoNfLiCt = 0; iWiLlNeVeRCoNfLiCt < size; \
	       iWiLlNeVeRCoNfLiCt++) { \
            ad_AD_ClearGrad(DERIV_grad((vars)[iWiLlNeVeRCoNfLiCt])); \
            DERIV_grad((vars)[iWiLlNeVeRCoNfLiCt])[ad_AD_IncrShadowVar()] = 1; \
        } \
    }

#   define ad_AD_SetIndepArrayElement(var, index) \
    { \
       ad_AD_ClearGrad(DERIV_grad((var)[index])); \
       DERIV_grad((var)[index])[ad_AD_IncrShadowVar()] = 1; \
    }

#   define ad_AD_SetIndepArrayColored(vars, size, colors) \
    { \
        int iWiLlNeVeRCoNfLiCt; \
        for (iWiLlNeVeRCoNfLiCt = 0; iWiLlNeVeRCoNfLiCt < size; \
               iWiLlNeVeRCoNfLiCt++) { \
            ad_AD_ClearGrad2(DERIV_grad((vars)[iWiLlNeVeRCoNfLiCt])); \
            DERIV_grad((vars)[iWiLlNeVeRCoNfLiCt])[ad_grad_size_shadow+(colors)[iWiLlNeVeRCoNfLiCt]] = 1; \
        } \
    }

/* values array is the same length as vars */
#   define ad_AD_SetIndepVector(vars, size, values) \
    { \
        int iWiLlNeVeRCoNfLiCt; \
        for (iWiLlNeVeRCoNfLiCt = 0; iWiLlNeVeRCoNfLiCt < size; \
               iWiLlNeVeRCoNfLiCt++) { \
            ad_AD_ClearGrad(DERIV_grad((vars)[iWiLlNeVeRCoNfLiCt])); \
            DERIV_grad((vars)[iWiLlNeVeRCoNfLiCt])[ad_grad_size_shadow] = (values)[iWiLlNeVeRCoNfLiCt]; \
        } \
    }

#define ad_AD_SetValArray(vars, size, values) \
    { \
        int iWiLlNeVeRCoNfLiCt; \
        for (iWiLlNeVeRCoNfLiCt = 0; iWiLlNeVeRCoNfLiCt < size; \
	       iWiLlNeVeRCoNfLiCt++) { \
            DERIV_val((vars)[iWiLlNeVeRCoNfLiCt]) = (values)[iWiLlNeVeRCoNfLiCt]; \
        } \
    }

PETSC_EXTERN int ad_AD_IncrShadowVar(void);
PETSC_EXTERN void  ad_AD_CommitShadowVar(void);
PETSC_EXTERN void  ad_AD_ResetShadowVar(void);


#   if defined(__cplusplus)
        }
#  endif
#endif /*AD_GRAD_H*/


