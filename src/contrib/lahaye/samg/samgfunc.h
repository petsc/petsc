#ifndef SAMGFUNC_H
#define SAMGFUNC_H

struct SAMG_PARAM{
  /*..Class 0 SAMG parameters..*/
                 int    MATRIX; 
  /*..Class 1 SAMG parameters..*/
                 int    NSOLVE; 
                 int    IFIRST;
                 double EPS;
                 int    NCYC; 
                 int    ISWTCH; 
                 double A_CMPLX; 
                 double G_CMPLX; 
                 double P_CMPLX; 
                 double W_AVRGE; 
                 double CHKTOL; 
                 int    IDUMP; 
                 int    IOUT; 
  /*..Class 2 SAMG parameters..*/
                 int    NRD; 
                 int    NRC; 
                 int    NRU; 
                 int    LEVELX; 
                 int    NPTMN; 
  /*..Class 3 SAMG parameters..*/
                 int    NCG; 
                 int    NWT;
                 int    NTR;  
                 double ECG; 
                 double EWT; 
                 double ETR; 
               };

/*..Driver of SAMG..*/ 
extern "C" void drvsamg_(int* nnu, double* a, int* ia, int* ja, double* u,
                     double* f, int* matrix, int* nsolve, int* ifirst, 
                     double* eps, int* ncyc, int* iswtch, double* a_cmplx, 
                     double* g_cmplx, double* p_cmplx, double* w_avrge, 
                     double* chktol, int* idump, int* iout, int* nrd, 
                     int* nrc, int* nru, int* levelx, int* nptmn, int* ncg, 
                     int* nwt, int* ntr, double* ecg, double* etw,
                     double* etr, int* levels, 
                     int* debug); 

/*..Memory cleanup after SAMG run..*/ 
extern "C" void samg_cleanup_(void); 

/*..Level 1 routines to access coarse grid Galerkin operators..*/ 
extern "C" void samggetdimmat_(int* k, int* nnu, int* nna);
extern "C" void samggetmat_(int* k, double* aout, int* iaout, int* jaout);

/*..Level 1 routines to access interpolation operator..*/ 
extern "C" void samggetdimint_(int* k, int* nnu, int* nna);
extern "C" void samggetint_(int* k, double* wout, int* iwout, int* jwout); 

/*..Aux function to pass SAMG hierarchy to PETSc..*/
  void apply_shift(int*ia, int nnu, int ia_shift, int* ja, int nna, 
                     int ja_shift);

#endif//SAMGFUNC_H
