#line 181 "run-alloc.w"
#if !defined(RUN_ALLOC_H)
#define RUN_ALLOC_H

#if defined(__cplusplus)
extern "C" {
#endif

void* ad_adic_deriv_init(int dsize, int bsize);
void ad_adic_deriv_final();
void* ad_adic_deriv_alloc();
void ad_adic_deriv_free(void*);

#if defined(__cplusplus)
}
#endif

#endif /*RUN_ALLOC_H*/


