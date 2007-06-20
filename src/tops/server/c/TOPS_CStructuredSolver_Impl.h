/*
 * File:          TOPS_CStructuredSolver_Impl.h
 * Symbol:        TOPS.CStructuredSolver-v0.0.0
 * Symbol Type:   class
 * Babel Version: 1.0.4
 * Description:   Server-side implementation for TOPS.CStructuredSolver
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 */

#ifndef included_TOPS_CStructuredSolver_Impl_h
#define included_TOPS_CStructuredSolver_Impl_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_TOPS_CStructuredSolver_h
#include "TOPS_CStructuredSolver.h"
#endif
#ifndef included_TOPS_Solver_h
#include "TOPS_Solver.h"
#endif
#ifndef included_TOPS_Structured_Solver_h
#include "TOPS_Structured_Solver.h"
#endif
#ifndef included_gov_cca_CCAException_h
#include "gov_cca_CCAException.h"
#endif
#ifndef included_gov_cca_Component_h
#include "gov_cca_Component.h"
#endif
#ifndef included_gov_cca_Port_h
#include "gov_cca_Port.h"
#endif
#ifndef included_gov_cca_Services_h
#include "gov_cca_Services.h"
#endif
#ifndef included_gov_cca_ports_ParameterGetListener_h
#include "gov_cca_ports_ParameterGetListener.h"
#endif
#ifndef included_gov_cca_ports_ParameterSetListener_h
#include "gov_cca_ports_ParameterSetListener.h"
#endif
#ifndef included_sidl_BaseClass_h
#include "sidl_BaseClass.h"
#endif
#ifndef included_sidl_BaseInterface_h
#include "sidl_BaseInterface.h"
#endif
#ifndef included_sidl_ClassInfo_h
#include "sidl_ClassInfo.h"
#endif
#ifndef included_sidl_RuntimeException_h
#include "sidl_RuntimeException.h"
#endif

/* DO-NOT-DELETE splicer.begin(TOPS.CStructuredSolver._includes) */
/* Insert-Code-Here {TOPS.CStructuredSolver._includes} (include files) */

#if defined(HAVE_LONG_LONG)
#undef HAVE_LONG_LONG
#endif
#include "petscdmmg.h"
#include "TOPS.hxx"
#include "gov_cca_ports_ParameterPortFactory.h"
#include "gov_cca_ports_ParameterPort.h"

/* DO-NOT-DELETE splicer.end(TOPS.CStructuredSolver._includes) */

/*
 * Private data for class TOPS.CStructuredSolver
 */

struct TOPS_CStructuredSolver__data {
  /* DO-NOT-DELETE splicer.begin(TOPS.CStructuredSolver._data) */
  /* Insert-Code-Here {TOPS.CStructuredSolver._data} (private data members) */
  int ignore; /* dummy to force non-empty struct; remove if you add data */

  DMMG                                  *dmmg;
  DA                                    da;
  int                                   lengths[4],m,n,p,dim,s,levels,bs;
  DAStencilType                         stencil_type;
  DAPeriodicType                        wrap;
  int                                   startedpetsc;
  gov_cca_Services			myServices;
  gov_cca_ports_ParameterPortFactory 	ppf;
  gov_cca_ports_ParameterPort        	params;
		      
  
  /* DO-NOT-DELETE splicer.end(TOPS.CStructuredSolver._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct TOPS_CStructuredSolver__data*
TOPS_CStructuredSolver__get_data(
  TOPS_CStructuredSolver);

extern void
TOPS_CStructuredSolver__set_data(
  TOPS_CStructuredSolver,
  struct TOPS_CStructuredSolver__data*);

extern
void
impl_TOPS_CStructuredSolver__load(
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_TOPS_CStructuredSolver__ctor(
  /* in */ TOPS_CStructuredSolver self,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_TOPS_CStructuredSolver__ctor2(
  /* in */ TOPS_CStructuredSolver self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_TOPS_CStructuredSolver__dtor(
  /* in */ TOPS_CStructuredSolver self,
  /* out */ sidl_BaseInterface *_ex);

/*
 * User-defined object methods
 */

extern struct TOPS_CStructuredSolver__object* 
  impl_TOPS_CStructuredSolver_fconnect_TOPS_CStructuredSolver(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct TOPS_CStructuredSolver__object* 
  impl_TOPS_CStructuredSolver_fcast_TOPS_CStructuredSolver(void* bi, 
  sidl_BaseInterface* _ex);
extern struct TOPS_Solver__object* 
  impl_TOPS_CStructuredSolver_fconnect_TOPS_Solver(const char* url, sidl_bool 
  ar, sidl_BaseInterface *_ex);
extern struct TOPS_Solver__object* 
  impl_TOPS_CStructuredSolver_fcast_TOPS_Solver(void* bi, sidl_BaseInterface* 
  _ex);
extern struct TOPS_Structured_Solver__object* 
  impl_TOPS_CStructuredSolver_fconnect_TOPS_Structured_Solver(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct TOPS_Structured_Solver__object* 
  impl_TOPS_CStructuredSolver_fcast_TOPS_Structured_Solver(void* bi, 
  sidl_BaseInterface* _ex);
extern struct gov_cca_CCAException__object* 
  impl_TOPS_CStructuredSolver_fconnect_gov_cca_CCAException(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct gov_cca_CCAException__object* 
  impl_TOPS_CStructuredSolver_fcast_gov_cca_CCAException(void* bi, 
  sidl_BaseInterface* _ex);
extern struct gov_cca_Component__object* 
  impl_TOPS_CStructuredSolver_fconnect_gov_cca_Component(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct gov_cca_Component__object* 
  impl_TOPS_CStructuredSolver_fcast_gov_cca_Component(void* bi, 
  sidl_BaseInterface* _ex);
extern struct gov_cca_Port__object* 
  impl_TOPS_CStructuredSolver_fconnect_gov_cca_Port(const char* url, sidl_bool 
  ar, sidl_BaseInterface *_ex);
extern struct gov_cca_Port__object* 
  impl_TOPS_CStructuredSolver_fcast_gov_cca_Port(void* bi, sidl_BaseInterface* 
  _ex);
extern struct gov_cca_Services__object* 
  impl_TOPS_CStructuredSolver_fconnect_gov_cca_Services(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct gov_cca_Services__object* 
  impl_TOPS_CStructuredSolver_fcast_gov_cca_Services(void* bi, 
  sidl_BaseInterface* _ex);
extern struct gov_cca_ports_ParameterGetListener__object* 
  impl_TOPS_CStructuredSolver_fconnect_gov_cca_ports_ParameterGetListener(const 
  char* url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct gov_cca_ports_ParameterGetListener__object* 
  impl_TOPS_CStructuredSolver_fcast_gov_cca_ports_ParameterGetListener(void* bi,
  sidl_BaseInterface* _ex);
extern struct gov_cca_ports_ParameterSetListener__object* 
  impl_TOPS_CStructuredSolver_fconnect_gov_cca_ports_ParameterSetListener(const 
  char* url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct gov_cca_ports_ParameterSetListener__object* 
  impl_TOPS_CStructuredSolver_fcast_gov_cca_ports_ParameterSetListener(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_BaseClass__object* 
  impl_TOPS_CStructuredSolver_fconnect_sidl_BaseClass(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseClass__object* 
  impl_TOPS_CStructuredSolver_fcast_sidl_BaseClass(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidl_BaseInterface__object* 
  impl_TOPS_CStructuredSolver_fconnect_sidl_BaseInterface(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_TOPS_CStructuredSolver_fcast_sidl_BaseInterface(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidl_ClassInfo__object* 
  impl_TOPS_CStructuredSolver_fconnect_sidl_ClassInfo(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_ClassInfo__object* 
  impl_TOPS_CStructuredSolver_fcast_sidl_ClassInfo(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidl_RuntimeException__object* 
  impl_TOPS_CStructuredSolver_fconnect_sidl_RuntimeException(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_RuntimeException__object* 
  impl_TOPS_CStructuredSolver_fcast_sidl_RuntimeException(void* bi, 
  sidl_BaseInterface* _ex);
extern
gov_cca_Services
impl_TOPS_CStructuredSolver_getServices(
  /* in */ TOPS_CStructuredSolver self,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_TOPS_CStructuredSolver_dimen(
  /* in */ TOPS_CStructuredSolver self,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_TOPS_CStructuredSolver_length(
  /* in */ TOPS_CStructuredSolver self,
  /* in */ int32_t a,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_TOPS_CStructuredSolver_setDimen(
  /* in */ TOPS_CStructuredSolver self,
  /* in */ int32_t dim,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_TOPS_CStructuredSolver_setLength(
  /* in */ TOPS_CStructuredSolver self,
  /* in */ int32_t a,
  /* in */ int32_t l,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_TOPS_CStructuredSolver_setStencilWidth(
  /* in */ TOPS_CStructuredSolver self,
  /* in */ int32_t width,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_TOPS_CStructuredSolver_getStencilWidth(
  /* in */ TOPS_CStructuredSolver self,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_TOPS_CStructuredSolver_setLevels(
  /* in */ TOPS_CStructuredSolver self,
  /* in */ int32_t levels,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_TOPS_CStructuredSolver_Initialize(
  /* in */ TOPS_CStructuredSolver self,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_TOPS_CStructuredSolver_solve(
  /* in */ TOPS_CStructuredSolver self,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_TOPS_CStructuredSolver_setBlockSize(
  /* in */ TOPS_CStructuredSolver self,
  /* in */ int32_t bs,
  /* out */ sidl_BaseInterface *_ex);

extern
struct sidl_double__array*
impl_TOPS_CStructuredSolver_getSolution(
  /* in */ TOPS_CStructuredSolver self,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_TOPS_CStructuredSolver_setSolution(
  /* in */ TOPS_CStructuredSolver self,
  /* in array<double> */ struct sidl_double__array* location,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_TOPS_CStructuredSolver_setServices(
  /* in */ TOPS_CStructuredSolver self,
  /* in */ gov_cca_Services services,
  /* out */ sidl_BaseInterface *_ex);

extern
sidl_bool
impl_TOPS_CStructuredSolver_updateParameterPort(
  /* in */ TOPS_CStructuredSolver self,
  /* in */ const char* portName,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_TOPS_CStructuredSolver_updatedParameterValue(
  /* in */ TOPS_CStructuredSolver self,
  /* in */ const char* portName,
  /* in */ const char* fieldName,
  /* out */ sidl_BaseInterface *_ex);

extern struct TOPS_CStructuredSolver__object* 
  impl_TOPS_CStructuredSolver_fconnect_TOPS_CStructuredSolver(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct TOPS_CStructuredSolver__object* 
  impl_TOPS_CStructuredSolver_fcast_TOPS_CStructuredSolver(void* bi, 
  sidl_BaseInterface* _ex);
extern struct TOPS_Solver__object* 
  impl_TOPS_CStructuredSolver_fconnect_TOPS_Solver(const char* url, sidl_bool 
  ar, sidl_BaseInterface *_ex);
extern struct TOPS_Solver__object* 
  impl_TOPS_CStructuredSolver_fcast_TOPS_Solver(void* bi, sidl_BaseInterface* 
  _ex);
extern struct TOPS_Structured_Solver__object* 
  impl_TOPS_CStructuredSolver_fconnect_TOPS_Structured_Solver(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct TOPS_Structured_Solver__object* 
  impl_TOPS_CStructuredSolver_fcast_TOPS_Structured_Solver(void* bi, 
  sidl_BaseInterface* _ex);
extern struct gov_cca_CCAException__object* 
  impl_TOPS_CStructuredSolver_fconnect_gov_cca_CCAException(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct gov_cca_CCAException__object* 
  impl_TOPS_CStructuredSolver_fcast_gov_cca_CCAException(void* bi, 
  sidl_BaseInterface* _ex);
extern struct gov_cca_Component__object* 
  impl_TOPS_CStructuredSolver_fconnect_gov_cca_Component(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct gov_cca_Component__object* 
  impl_TOPS_CStructuredSolver_fcast_gov_cca_Component(void* bi, 
  sidl_BaseInterface* _ex);
extern struct gov_cca_Port__object* 
  impl_TOPS_CStructuredSolver_fconnect_gov_cca_Port(const char* url, sidl_bool 
  ar, sidl_BaseInterface *_ex);
extern struct gov_cca_Port__object* 
  impl_TOPS_CStructuredSolver_fcast_gov_cca_Port(void* bi, sidl_BaseInterface* 
  _ex);
extern struct gov_cca_Services__object* 
  impl_TOPS_CStructuredSolver_fconnect_gov_cca_Services(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct gov_cca_Services__object* 
  impl_TOPS_CStructuredSolver_fcast_gov_cca_Services(void* bi, 
  sidl_BaseInterface* _ex);
extern struct gov_cca_ports_ParameterGetListener__object* 
  impl_TOPS_CStructuredSolver_fconnect_gov_cca_ports_ParameterGetListener(const 
  char* url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct gov_cca_ports_ParameterGetListener__object* 
  impl_TOPS_CStructuredSolver_fcast_gov_cca_ports_ParameterGetListener(void* bi,
  sidl_BaseInterface* _ex);
extern struct gov_cca_ports_ParameterSetListener__object* 
  impl_TOPS_CStructuredSolver_fconnect_gov_cca_ports_ParameterSetListener(const 
  char* url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct gov_cca_ports_ParameterSetListener__object* 
  impl_TOPS_CStructuredSolver_fcast_gov_cca_ports_ParameterSetListener(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_BaseClass__object* 
  impl_TOPS_CStructuredSolver_fconnect_sidl_BaseClass(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseClass__object* 
  impl_TOPS_CStructuredSolver_fcast_sidl_BaseClass(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidl_BaseInterface__object* 
  impl_TOPS_CStructuredSolver_fconnect_sidl_BaseInterface(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_TOPS_CStructuredSolver_fcast_sidl_BaseInterface(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidl_ClassInfo__object* 
  impl_TOPS_CStructuredSolver_fconnect_sidl_ClassInfo(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_ClassInfo__object* 
  impl_TOPS_CStructuredSolver_fcast_sidl_ClassInfo(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidl_RuntimeException__object* 
  impl_TOPS_CStructuredSolver_fconnect_sidl_RuntimeException(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_RuntimeException__object* 
  impl_TOPS_CStructuredSolver_fcast_sidl_RuntimeException(void* bi, 
  sidl_BaseInterface* _ex);
#ifdef __cplusplus
}
#endif
#endif
