/*
  THIS PROGRAM DISCLOSES MATERIAL PROTECTABLE UNDER COPYRIGHT
  LAWS OF THE UNITED STATES.  FOR LICENSING INFORMATION CONTACT:

  Christian Bischof or Lucas Roh, Mathematics and Computer Science Division,
  Argonne National Laboratory, 9700 S. Cass Avenue, Argonne IL 60439,
  {bischof,roh}@mcs.anl.gov.
*/

/*
   This for PETSc where we KNOW that ad_grad_size is ad_GRAD_MAX thus we KNOW at COMPILE time the 
 size of the loops; by passing this information to the compiler it may compile faster code?
*/

#define ad_grad_size_dynamic ad_GRAD_MAX

#ifdef __cplusplus
extern "C" {
#endif

void ad_grad_axpy_n(int, void*, ...);

#ifdef __cplusplus
}
#endif

#define ad_grad_axpy_0(ggz) \
                { int iWiLlNeVeRCoNfLiCt; double *gz = DERIV_grad(*(ggz)); \
                for (iWiLlNeVeRCoNfLiCt = 0 ; iWiLlNeVeRCoNfLiCt < ad_grad_size_dynamic ; iWiLlNeVeRCoNfLiCt++) {\
                    gz[iWiLlNeVeRCoNfLiCt] = 0.0;\
                } }
            
#define ad_grad_axpy_zero(gz) ad_grad_axpy_0(gz)
            
#define ad_grad_axpy_copy(ggz,ggx) \
                { int iWiLlNeVeRCoNfLiCt; double *gz = DERIV_grad(*(ggz)),  *gx = DERIV_grad(*(ggx)); \
                for (iWiLlNeVeRCoNfLiCt = 0 ; iWiLlNeVeRCoNfLiCt < ad_grad_size_dynamic ; iWiLlNeVeRCoNfLiCt++) {\
                    gz[iWiLlNeVeRCoNfLiCt] = gx[iWiLlNeVeRCoNfLiCt];\
                } }
            
#define ad_grad_axpy_1(ggz, ca, gga)\
                { int iWiLlNeVeRCoNfLiCt; double *gz = DERIV_grad(*(ggz)) , *ga = DERIV_grad(*(gga));\
                for (iWiLlNeVeRCoNfLiCt = 0; iWiLlNeVeRCoNfLiCt < ad_grad_size_dynamic; iWiLlNeVeRCoNfLiCt++) {\
                    gz[iWiLlNeVeRCoNfLiCt] =  +(ca)*ga[iWiLlNeVeRCoNfLiCt];\
                } }

#define ad_grad_axpy_2(ggz, ca, gga, cb, ggb)\
                { int iWiLlNeVeRCoNfLiCt; double *gz = DERIV_grad(*(ggz)) , *ga = DERIV_grad(*(gga)), *gb = DERIV_grad(*(ggb));\
                for (iWiLlNeVeRCoNfLiCt = 0; iWiLlNeVeRCoNfLiCt < ad_grad_size_dynamic; iWiLlNeVeRCoNfLiCt++) {\
                    gz[iWiLlNeVeRCoNfLiCt] =  +(ca)*ga[iWiLlNeVeRCoNfLiCt] +(cb)*gb[iWiLlNeVeRCoNfLiCt];\
                } }

#define ad_grad_axpy_3(ggz, ca, gga, cb, ggb, cc, ggc)\
                { int iWiLlNeVeRCoNfLiCt; double *gz = DERIV_grad(*(ggz)) , *ga = DERIV_grad(*(gga)), *gb = DERIV_grad(*(ggb)), *gc = DERIV_grad(*(ggc));\
                for (iWiLlNeVeRCoNfLiCt = 0; iWiLlNeVeRCoNfLiCt < ad_grad_size_dynamic; iWiLlNeVeRCoNfLiCt++) {\
                    gz[iWiLlNeVeRCoNfLiCt] =  +(ca)*ga[iWiLlNeVeRCoNfLiCt] +(cb)*gb[iWiLlNeVeRCoNfLiCt] +(cc)*gc[iWiLlNeVeRCoNfLiCt];\
                } }

#define ad_grad_axpy_4(ggz, ca, gga, cb, ggb, cc, ggc, cd, ggd)\
                { int iWiLlNeVeRCoNfLiCt; double *gz = DERIV_grad(*(ggz)) , *ga = DERIV_grad(*(gga)), *gb = DERIV_grad(*(ggb)), *gc = DERIV_grad(*(ggc)), *gd = DERIV_grad(*(ggd));\
                for (iWiLlNeVeRCoNfLiCt = 0; iWiLlNeVeRCoNfLiCt < ad_grad_size_dynamic; iWiLlNeVeRCoNfLiCt++) {\
                    gz[iWiLlNeVeRCoNfLiCt] =  +(ca)*ga[iWiLlNeVeRCoNfLiCt] +(cb)*gb[iWiLlNeVeRCoNfLiCt] +(cc)*gc[iWiLlNeVeRCoNfLiCt] +(cd)*gd[iWiLlNeVeRCoNfLiCt];\
                } }

#define ad_grad_axpy_5(ggz, ca, gga, cb, ggb, cc, ggc, cd, ggd, ce, gge)\
                { int iWiLlNeVeRCoNfLiCt; double *gz = DERIV_grad(*(ggz)) , *ga = DERIV_grad(*(gga)), *gb = DERIV_grad(*(ggb)), *gc = DERIV_grad(*(ggc)), *gd = DERIV_grad(*(ggd)), *ge = DERIV_grad(*(gge));\
                for (iWiLlNeVeRCoNfLiCt = 0; iWiLlNeVeRCoNfLiCt < ad_grad_size_dynamic; iWiLlNeVeRCoNfLiCt++) {\
                    gz[iWiLlNeVeRCoNfLiCt] =  +(ca)*ga[iWiLlNeVeRCoNfLiCt] +(cb)*gb[iWiLlNeVeRCoNfLiCt] +(cc)*gc[iWiLlNeVeRCoNfLiCt] +(cd)*gd[iWiLlNeVeRCoNfLiCt] +(ce)*ge[iWiLlNeVeRCoNfLiCt];\
                } }

#define ad_grad_axpy_6(ggz, ca, gga, cb, ggb, cc, ggc, cd, ggd, ce, gge, cf, ggf)\
                { int iWiLlNeVeRCoNfLiCt; double *gz = DERIV_grad(*(ggz)) , *ga = DERIV_grad(*(gga)), *gb = DERIV_grad(*(ggb)), *gc = DERIV_grad(*(ggc)), *gd = DERIV_grad(*(ggd)), *ge = DERIV_grad(*(gge)), *gf = DERIV_grad(*(ggf));\
                for (iWiLlNeVeRCoNfLiCt = 0; iWiLlNeVeRCoNfLiCt < ad_grad_size_dynamic; iWiLlNeVeRCoNfLiCt++) {\
                    gz[iWiLlNeVeRCoNfLiCt] =  +(ca)*ga[iWiLlNeVeRCoNfLiCt] +(cb)*gb[iWiLlNeVeRCoNfLiCt] +(cc)*gc[iWiLlNeVeRCoNfLiCt] +(cd)*gd[iWiLlNeVeRCoNfLiCt] +(ce)*ge[iWiLlNeVeRCoNfLiCt] +(cf)*gf[iWiLlNeVeRCoNfLiCt];\
                } }

#define ad_grad_axpy_7(ggz, ca, gga, cb, ggb, cc, ggc, cd, ggd, ce, gge, cf, ggf, cg, ggg)\
                { int iWiLlNeVeRCoNfLiCt; double *gz = DERIV_grad(*(ggz)) , *ga = DERIV_grad(*(gga)), *gb = DERIV_grad(*(ggb)), *gc = DERIV_grad(*(ggc)), *gd = DERIV_grad(*(ggd)), *ge = DERIV_grad(*(gge)), *gf = DERIV_grad(*(ggf)), *gg = DERIV_grad(*(ggg));\
                for (iWiLlNeVeRCoNfLiCt = 0; iWiLlNeVeRCoNfLiCt < ad_grad_size_dynamic; iWiLlNeVeRCoNfLiCt++) {\
                    gz[iWiLlNeVeRCoNfLiCt] =  +(ca)*ga[iWiLlNeVeRCoNfLiCt] +(cb)*gb[iWiLlNeVeRCoNfLiCt] +(cc)*gc[iWiLlNeVeRCoNfLiCt] +(cd)*gd[iWiLlNeVeRCoNfLiCt] +(ce)*ge[iWiLlNeVeRCoNfLiCt] +(cf)*gf[iWiLlNeVeRCoNfLiCt] +(cg)*gg[iWiLlNeVeRCoNfLiCt];\
                } }

#define ad_grad_axpy_8(ggz, ca, gga, cb, ggb, cc, ggc, cd, ggd, ce, gge, cf, ggf, cg, ggg, ch, ggh)\
                { int iWiLlNeVeRCoNfLiCt; double *gz = DERIV_grad(*(ggz)) , *ga = DERIV_grad(*(gga)), *gb = DERIV_grad(*(ggb)), *gc = DERIV_grad(*(ggc)), *gd = DERIV_grad(*(ggd)), *ge = DERIV_grad(*(gge)), *gf = DERIV_grad(*(ggf)), *gg = DERIV_grad(*(ggg)), *gh = DERIV_grad(*(ggh)); \
		for (iWiLlNeVeRCoNfLiCt = 0; iWiLlNeVeRCoNfLiCt < ad_grad_size_dynamic; iWiLlNeVeRCoNfLiCt++) {\
                    gz[iWiLlNeVeRCoNfLiCt] =  +(ca)*ga[iWiLlNeVeRCoNfLiCt] +(cb)*gb[iWiLlNeVeRCoNfLiCt] +(cc)*gc[iWiLlNeVeRCoNfLiCt] +(cd)*gd[iWiLlNeVeRCoNfLiCt] +(ce)*ge[iWiLlNeVeRCoNfLiCt] +(cf)*gf[iWiLlNeVeRCoNfLiCt] +(cg)*gg[iWiLlNeVeRCoNfLiCt] +(ch)*gh[iWiLlNeVeRCoNfLiCt];\
                } }

#define ad_grad_axpy_9(ggz, ca, gga, cb, ggb, cc, ggc, cd, ggd, ce, gge, cf, ggf, cg, ggg, ch, ggh, ci, ggi)\
                { int iWiLlNeVeRCoNfLiCt; double *gz = DERIV_grad(*(ggz)) , *ga = DERIV_grad(*(gga)), *gb = DERIV_grad(*(ggb)), *gc = DERIV_grad(*(ggc)), *gd = DERIV_grad(*(ggd)), *ge = DERIV_grad(*(gge)), *gf = DERIV_grad(*(ggf)), *gg = DERIV_grad(*(ggg)), *gh = DERIV_grad(*(ggh)), *gi = DERIV_grad(*(ggi));\
                for (iWiLlNeVeRCoNfLiCt = 0; iWiLlNeVeRCoNfLiCt < ad_grad_size_dynamic; iWiLlNeVeRCoNfLiCt++) {\
                    gz[iWiLlNeVeRCoNfLiCt] =  +(ca)*ga[iWiLlNeVeRCoNfLiCt] +(cb)*gb[iWiLlNeVeRCoNfLiCt] +(cc)*gc[iWiLlNeVeRCoNfLiCt] +(cd)*gd[iWiLlNeVeRCoNfLiCt] +(ce)*ge[iWiLlNeVeRCoNfLiCt] +(cf)*gf[iWiLlNeVeRCoNfLiCt] +(cg)*gg[iWiLlNeVeRCoNfLiCt] +(ch)*gh[iWiLlNeVeRCoNfLiCt] +(ci)*gi[iWiLlNeVeRCoNfLiCt];\
                } }

#define ad_grad_axpy_10(ggz, ca, gga, cb, ggb, cc, ggc, cd, ggd, ce, gge, cf, ggf, cg, ggg, ch, ggh, ci, ggi, cj, ggj)\
                { int iWiLlNeVeRCoNfLiCt; double *gz = DERIV_grad(*(ggz)) , *ga = DERIV_grad(*(gga)), *gb = DERIV_grad(*(ggb)), *gc = DERIV_grad(*(ggc)), *gd = DERIV_grad(*(ggd)), *ge = DERIV_grad(*(gge)), *gf = DERIV_grad(*(ggf)), *gg = DERIV_grad(*(ggg)), *gh = DERIV_grad(*(ggh)), *gi = DERIV_grad(*(ggi)), *gj = DERIV_grad(*(ggj));\
                for (iWiLlNeVeRCoNfLiCt = 0; iWiLlNeVeRCoNfLiCt < ad_grad_size_dynamic; iWiLlNeVeRCoNfLiCt++) {\
                    gz[iWiLlNeVeRCoNfLiCt] =  +(ca)*ga[iWiLlNeVeRCoNfLiCt] +(cb)*gb[iWiLlNeVeRCoNfLiCt] +(cc)*gc[iWiLlNeVeRCoNfLiCt] +(cd)*gd[iWiLlNeVeRCoNfLiCt] +(ce)*ge[iWiLlNeVeRCoNfLiCt] +(cf)*gf[iWiLlNeVeRCoNfLiCt] +(cg)*gg[iWiLlNeVeRCoNfLiCt] +(ch)*gh[iWiLlNeVeRCoNfLiCt] +(ci)*gi[iWiLlNeVeRCoNfLiCt] +(cj)*gj[iWiLlNeVeRCoNfLiCt];\
                } }










