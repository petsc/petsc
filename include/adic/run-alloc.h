#if !defined(RUN_ALLOC_H)
#define RUN_ALLOC_H

#if defined(__cplusplus)
PETSC_EXTERN "C" {
#endif

void* ad_adic_deriv_init(int dsize, int bsize);
void ad_adic_deriv_final(void);
void* ad_adic_deriv_alloc(void);
void ad_adic_deriv_free(void*);

#if defined(__cplusplus)
}
#endif

#endif /*RUN_ALLOC_H*/


