
/* INTERFACE TO FORTRAN90 SAMG : amg subroutines have 
   to be call as SAMG_xxxx resp. USER_coo when this interface 
   is used */


#ifdef PETSC_HAVE_FORTRAN_CAPS
#elif defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define  SAMG                         samg_
#define  SAMG_simple                  samg_simple_
#define  USER_coo                     user_coo_
#define  SAMG_ctime                   samg_ctime_
#define  SAMG_set_ecg                 samg_set_ecg_
#define  SAMG_set_ewt                 samg_set_ewt_
#define  SAMG_set_etr                 samg_set_etr_
#define  SAMG_set_densx               samg_set_densx_
#define  SAMG_set_eps_dd              samg_set_eps_dd_
#define  SAMG_set_slow_coarsening     samg_set_slow_coarsening_
#define  SAMG_set_term_coarsening     samg_set_term_coarsening_
#define  SAMG_set_droptol             samg_set_droptol_
#define  SAMG_set_droptol_smo         samg_set_droptol_smo_ 
#define  SAMG_set_droptol_cl          samg_set_droptol_cl_
#define  SAMG_set_stability           samg_set_stability_
#define  SAMG_set_eps_diag            samg_set_eps_diag_
#define  SAMG_set_eps_lsq             samg_set_eps_lsq_
#define  SAMG_set_rcondx              samg_set_rcondx_
#define  SAMG_set_a_cmplx_default     samg_set_a_cmplx_default_
#define  SAMG_set_g_cmplx_default     samg_set_g_cmplx_default_
#define  SAMG_set_p_cmplx_default     samg_set_p_cmplx_default_
#define  SAMG_set_w_avrge_default     samg_set_w_avrge_default_
#define  SAMG_set_ecg_default         samg_set_ecg_default_
#define  SAMG_set_ewt_default         samg_set_ewt_default_
#define  SAMG_set_etr_default         samg_set_etr_default_
#define  SAMG_set_conv_stop_default   samg_set_conv_stop_default_
#define  SAMG_set_logio               samg_set_logio_
#define  SAMG_set_mode_debug          samg_set_mode_debug_
#define  SAMG_set_iter_pre            samg_set_iter_pre_
#define  SAMG_set_levelx              samg_set_levelx_
#define  SAMG_set_nptmn               samg_set_nptmn_
#define  SAMG_set_ncg                 samg_set_ncg_
#define  SAMG_set_nwt                 samg_set_nwt_
#define  SAMG_set_ntr                 samg_set_ntr_
#define  SAMG_set_nrd                 samg_set_nrd_
#define  SAMG_set_nru                 samg_set_nru_
#define  SAMG_set_nrc                 samg_set_nrc_
#define  SAMG_set_np_opt              samg_set_np_opt_
#define  SAMG_set_np_mod1             samg_set_np_mod1_
#define  SAMG_set_np_mod2             samg_set_np_mod2_
#define  SAMG_set_max_level           samg_set_max_level_
#define  SAMG_set_nptmax              samg_set_nptmax_
#define  SAMG_set_ncyc_start          samg_set_ncyc_start_
#define  SAMG_set_lfil_smo            samg_set_lfil_smo_
#define  SAMG_set_nrc_emergency       samg_set_nrc_emergency_
#define  SAMG_set_iter_check          samg_set_iter_check_
#define  SAMG_set_neg_diag            samg_set_neg_diag_
#define  SAMG_set_maxop_restart       samg_set_maxop_restart_
#define  SAMG_set_nsolve_default      samg_set_nsolve_default_
#define  SAMG_set_ncyc_default        samg_set_ncyc_default_
#define  SAMG_set_ncgrad_default      samg_set_ncgrad_default_
#define  SAMG_set_nkdim_default       samg_set_nkdim_default_
#define  SAMG_set_nrc_default         samg_set_nrc_default_
#define  SAMG_set_itmax_conv_default  samg_set_itmax_conv_default_
#define  SAMG_set_lfil_cl_default     samg_set_lfil_cl_default_
#define  SAMG_set_full_pivoting       samg_set_full_pivoting_
#define  SAMG_set_iodump              samg_set_iodump_
#define  SAMG_set_iogrid              samg_set_iogrid_
#define  SAMG_set_iomovie             samg_set_iomovie_
#define  SAMG_set_lastgrid            samg_set_lastgrid_
#define  SAMG_set_ncframes            samg_set_ncframes_
#define  SAMG_cleanup                 samg_cleanup_
#define  SAMG_reset_secondary         samg_reset_secondary_
#else
#define  SAMG                         samg 
#define  SAMG_simple                  samg_simple
#define  USER_coo                     user_coo 
#define  SAMG_ctime                   samg_ctime 
#define  SAMG_set_ecg                 samg_set_ecg 
#define  SAMG_set_ewt                 samg_set_ewt 
#define  SAMG_set_etr                 samg_set_etr 
#define  SAMG_set_densx               samg_set_densx 
#define  SAMG_set_eps_dd              samg_set_eps_dd 
#define  SAMG_set_slow_coarsening     samg_set_slow_coarsening 
#define  SAMG_set_term_coarsening     samg_set_term_coarsening 
#define  SAMG_set_droptol             samg_set_droptol 
#define  SAMG_set_droptol_smo         samg_set_droptol_smo 
#define  SAMG_set_droptol_cl          samg_set_droptol_cl 
#define  SAMG_set_stability           samg_set_stability 
#define  SAMG_set_eps_diag            samg_set_eps_diag 
#define  SAMG_set_eps_lsq             samg_set_eps_lsq 
#define  SAMG_set_rcondx              samg_set_rcondx
#define  SAMG_set_a_cmplx_default     samg_set_a_cmplx_default 
#define  SAMG_set_g_cmplx_default     samg_set_g_cmplx_default 
#define  SAMG_set_p_cmplx_default     samg_set_p_cmplx_default 
#define  SAMG_set_w_avrge_default     samg_set_w_avrge_default 
#define  SAMG_set_ecg_default         samg_set_ecg_default 
#define  SAMG_set_ewt_default         samg_set_ewt_default 
#define  SAMG_set_etr_default         samg_set_etr_default 
#define  SAMG_set_conv_stop_default   samg_set_conv_stop_default 
#define  SAMG_set_logio               samg_set_logio 
#define  SAMG_set_mode_debug          samg_set_mode_debug 
#define  SAMG_set_iter_pre            samg_set_iter_pre 
#define  SAMG_set_levelx              samg_set_levelx 
#define  SAMG_set_nptmn               samg_set_nptmn 
#define  SAMG_set_ncg                 samg_set_ncg 
#define  SAMG_set_nwt                 samg_set_nwt 
#define  SAMG_set_ntr                 samg_set_ntr 
#define  SAMG_set_nrd                 samg_set_nrd 
#define  SAMG_set_nru                 samg_set_nru 
#define  SAMG_set_nrc                 samg_set_nrc 
#define  SAMG_set_np_opt              samg_set_np_opt 
#define  SAMG_set_np_mod1             samg_set_np_mod1 
#define  SAMG_set_np_mod2             samg_set_np_mod2 
#define  SAMG_set_max_level           samg_set_max_level 
#define  SAMG_set_nptmax              samg_set_nptmax 
#define  SAMG_set_ncyc_start          samg_set_ncyc_start
#define  SAMG_set_lfil_smo            samg_set_lfil_smo 
#define  SAMG_set_nrc_emergency       samg_set_nrc_emergency 
#define  SAMG_set_iter_check          samg_set_iter_check 
#define  SAMG_set_neg_diag            samg_set_neg_diag
#define  SAMG_set_maxop_restart       samg_set_maxop_restart
#define  SAMG_set_nsolve_default      samg_set_nsolve_default
#define  SAMG_set_ncyc_default        samg_set_ncyc_default 
#define  SAMG_set_ncgrad_default      samg_set_ncgrad_default
#define  SAMG_set_nkdim_default       samg_set_nkdim_default 
#define  SAMG_set_nrc_default         samg_set_nrc_default 
#define  SAMG_set_itmax_conv_default  samg_set_itmax_conv_default 
#define  SAMG_set_lfil_cl_default     samg_set_lfil_cl_default 
#define  SAMG_set_full_pivoting       samg_set_full_pivoting 
#define  SAMG_set_iodump              samg_set_iodump
#define  SAMG_set_iogrid              samg_set_iogrid
#define  SAMG_set_iomovie             samg_set_iomovie
#define  SAMG_set_lastgrid            samg_set_lastgrid
#define  SAMG_set_ncframes            samg_set_ncframes
#define  SAMG_cleanup                 samg_cleanup
#define  SAMG_reset_secondary         samg_reset_secondary
#endif


extern "C"                                  // declaration of amg internal timing routine
void SAMG_ctime(float * tim);

extern "C" 
void SAMG(int * nnu, int * nna, int * nsys,
          int * ia, int * ja, double * a, double * f, double * u,
          int * iu, int * ndiu, int * ip, int * ndip, int * matrix, int * iscale,
          double * res_in, double * res_out, int * ncyc_done, int * ierr,
          int * nsolve, int * ifirst, double * eps, int * ncyc, int * iswtch,
          double * a_cmplx, double * g_cmplx, double * p_cmplx, double * w_avrge,
          double * chktol, int * idump, int * iout);

extern "C" 
void SAMG_simple(int * iounit, int * nnu, int * nna, int * nsys,
          int * ia, int * ja, double * a, double * f, double * u,
          int * iu, int * ndiu, int * ip, int * ndip, int * matrix, int * iscale,
          double * res_in, double * res_out, int * ncyc_done, int * ierr);

extern "C" 
void USER_coo(int * i,int * ndim, double * x, double * y, double * z); 

// set double precision values
extern "C"
void SAMG_set_ecg(double * val);
extern "C"
void SAMG_set_ewt(double * val);
extern "C"
void SAMG_set_etr(double * val);
extern "C"
void SAMG_set_densx(double * val);
extern "C"
void SAMG_set_eps_dd(double * val);
extern "C"
void SAMG_set_slow_coarsening(double * val);
extern "C"
void SAMG_set_term_coarsening(double * val);
extern "C"
void SAMG_set_droptol(double * val);
extern "C"
void SAMG_set_droptol_smo(double * val);
extern "C"
void SAMG_set_droptol_cl(double * val);
extern "C"
void SAMG_set_stability(double * val);
extern "C"
void SAMG_set_eps_diag(double * val);
extern "C"
void SAMG_set_eps_lsq(double * val);
extern "C"
void SAMG_set_rcondx(double * val);
extern "C"
void SAMG_set_a_cmplx_default(double * val);
extern "C"
void SAMG_set_g_cmplx_default(double * val);
extern "C"
void SAMG_set_p_cmplx_default(double * val);
extern "C"
void SAMG_set_w_avrge_default(double * val);
extern "C"
void SAMG_set_ecg_default(double * val);
extern "C"
void SAMG_set_ewt_default(double * val);
extern "C"
void SAMG_set_etr_default(double * val);
extern "C"
void SAMG_set_conv_stop_default(double * val);

// set integer values
extern "C"
void SAMG_set_logio(int * ival);
extern "C"
void SAMG_set_mode_debug(int * ival);
extern "C"
void SAMG_set_iter_pre(int * ival);
extern "C"
void SAMG_set_levelx(int * ival);
extern "C"
void SAMG_set_nptmn(int * ival);
extern "C"
void SAMG_set_ncg(int * ival);
extern "C"
void SAMG_set_nwt(int * ival);
extern "C"
void SAMG_set_ntr(int * ival);
extern "C"
void SAMG_set_nrd(int * val);
extern "C"
void SAMG_set_nru(int * ival);
extern "C"
void SAMG_set_nrc(int * ival);
extern "C"
void SAMG_set_np_opt(int * ival);
extern "C"
void SAMG_set_np_mod1(int * ival);
extern "C"
void SAMG_set_np_mod2(int * ival);
extern "C"
void SAMG_set_max_level(int * ival);
extern "C"
void SAMG_set_nptmax(int * ival);
extern "C"
void SAMG_set_ncyc_start(int * ival);
extern "C"
void SAMG_set_lfil_smo(int * ival);
extern "C"
void SAMG_set_nrc_emergency(int * ival);
extern "C"
void SAMG_set_iter_check(int * ival);
extern "C"
void SAMG_set_neg_diag(int * ival);
extern "C"
void SAMG_set_maxop_restart(int * ival);
extern "C"
void SAMG_set_nsolve_default(int * ival);
extern "C"
void SAMG_set_ncyc_default(int * ival);
extern "C"
void SAMG_set_ncgrad_default(int * ival);
extern "C"
void SAMG_set_nkdim_default(int * ival);
extern "C"
void SAMG_set_nrc_default(int * ival);
extern "C"
void SAMG_set_itmax_conv_default(int * ival);
extern "C"
void SAMG_set_lfil_cl_default(int * ival);
extern "C"
void SAMG_set_iodump(int * ival);
extern "C"
void SAMG_set_iogrid(int * ival);
extern "C"
void SAMG_set_iomovie(int * ival);
extern "C"
void SAMG_set_lastgrid(int * ival);
extern "C"
void SAMG_set_ncframes(int * ival);

// set logicals (Fortran subroutine will set the logical to lvale.ne.0)
extern "C"
void SAMG_set_full_pivoting(int * lval);

// reset amg secondary variables
extern "C"
void SAMG_reset_secondary(void);

// release all amg memory
extern "C"
void SAMG_cleanup(void);
