#ifndef SAMGFUNC_H
#define SAMGFUNC_H


/*..Structure to hold SAMG parameters..*/ 
struct SAMG_PARAM{
  /*....Primary parameters....*/
                 int    MATRIX; 
                 int    IFIRST;
                 double EPS;
                 int    NSOLVE; 
                 int    NCYC; 
                 int    ISWTCH; 
                 double A_CMPLX; 
                 double G_CMPLX; 
                 double P_CMPLX; 
                 double W_AVRGE; 
                 double CHKTOL; 
                 int    IDUMP; 
                 int    IOUT; 
  /*....Secundary parameters....*/
                 int    LEVELX; 
                 int    NPTMN; 
                 double ECG; 
                 double EWT; 
                 int    NCG; 
                 int    NWT;
                 double ETR; 
                 int    NTR;
                 int    NRD; 
                 int    NRU; 
                 int    NRC; 
               };

/*..We'll use underscores in the name giving of the routines as the 
  Fortran compiler does not distinguish between lower and upper case..*/ 

#ifdef PETSC_HAVE_FORTRAN_CAPS
#elif defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define SAMGPETSC_apply_shift            samgpetsc_apply_shift_ 
#define SAMGPETSC_get_levels             samgpetsc_get_levels_
#define SAMGPETSC_get_dim_operator       samgpetsc_get_dim_operator_
#define SAMGPETSC_get_operator           samgpetsc_get_operator_
#define SAMGPETSC_get_dim_interpol       samgpetsc_get_dim_interpol_
#define SAMGPETSC_get_interpol           samgpetsc_get_interpol_
#else
#define SAMGPETSC_apply_shift            samgpetsc_apply_shift
#define SAMGPETSC_get_levels             samgpetsc_get_levels
#define SAMGPETSC_get_dim_operator       samgpetsc_get_dim_operator
#define SAMGPETSC_get_operator           samgpetsc_get_operator
#define SAMGPETSC_get_dim_interpol       samgpetsc_get_dim_interpol
#define SAMGPETSC_get_interpol           samgpetsc_get_interpol
#endif 

/*..Routine to apply ia_shift on all elements of ia and ja_shift on all 
  elements of ja..*/ 
extern "C" 
void SAMGPETSC_apply_shift(int* ia, int* nnu, int* ia_shift, 
                           int* ja, int* nna, int* ja_shift); 

/*..Level 1 routine to get number of levels created..*/ 
extern "C" 
void SAMGPETSC_get_levels(int* levelscp); 

/*..Level 1 routines to access grid operators..*/ 
extern "C" 
void SAMGPETSC_get_dim_operator(int* k, int* nnu, int* nna);
extern "C"
void SAMGPETSC_get_operator(int* k, double* aout, int* iaout, int* jaout);

/*..Level 1 routines to access interpolation operator..*/ 
extern "C" 
void SAMGPETSC_get_dim_interpol(int* k, int* nnu, int* nna);
extern "C" 
void SAMGPETSC_get_interpol(int* k, double* wout, int* iwout, int* jwout); 

/*..Aux function to pass SAMG hierarchy to PETSc..*/
void apply_shift(int*ia, int nnu, int ia_shift, int* ja, int nna, 
                     int ja_shift);

/*..Functions for verifying intermediate results: printing interpolation 
  and coarser grid matrices to file..*/ 
int SamgPetscWriteOperator(const int numnodes, const double* Asky, 
                           const int* ia, const int* ja, int extension);      
int SamgPetscWriteInterpol(const int numrows, const double* weights, 
                  const int* iweights, const int* jweights, int extension); 
 
#endif//SAMGFUNC_H
