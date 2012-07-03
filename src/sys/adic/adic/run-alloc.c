#include <petscconf.h>
#if defined(PETSC_HAVE_STDLIB_H)
#include <stdlib.h>
#endif
#if defined(PETSC_HAVE_MALLOC_H)
#include <malloc.h>
#endif

#include "adic/run-alloc.h"

#if defined(__cplusplus)
extern "C" {
#endif
  
#define DEFAULT_BUCKET_SIZE		1000
#define DEFAULT_DERIV_SIZE		sizeof(void*)
  
static int bucket_size = 0;
static int deriv_size = 0;

typedef struct genlist {
  struct genlist *next;
  struct genlist *prev;
  char data[1];
} genlist_t;
static genlist_t* freeList = 0;
static genlist_t* bucketList = 0;
static genlist_t* curBucket = 0;
static int nCount = 0;



void* ad_adic_deriv_init(int dsize, int bsize)
{ 
  if (!dsize) {
    deriv_size = DEFAULT_DERIV_SIZE;
  } else {
    deriv_size = dsize;
  }
  if (!bsize) {
    bucket_size = DEFAULT_BUCKET_SIZE;
  } else {
    bucket_size = bsize;
  }
  
  curBucket = (genlist_t*)malloc(deriv_size * bucket_size);
  curBucket->next = 0;
  curBucket->prev = 0;
  
  freeList = 0;
  bucketList = curBucket;
  nCount = 0;
  return(bucketList);
}

void ad_adic_deriv_final(void)
{
  if (bucketList) {
    genlist_t* block = bucketList;
    genlist_t* tmp;
    while ((tmp = block->next)) {
      free(block);
      block = tmp;
    }
    free(block);
    bucketList = 0;
  }
}

void* ad_adic_deriv_alloc(void)
{
  
#if defined(DEBUG)
  static 	count = 0;
  if (++count >= gdebug.nTokens) {
    msg("Allocated %d-th deriv obj", count);
    count = 0;
  }
#endif
  
  if (freeList) {
    void* pobj = freeList;
    freeList = freeList->next;
    return pobj;
  }
  else if (nCount >= bucket_size) {
    curBucket = (genlist_t*)malloc(deriv_size * bucket_size);
    curBucket->next = bucketList;
    bucketList->prev = curBucket;
    bucketList = curBucket;
    nCount = 1;
    return curBucket->data;
  }
  else {
    return curBucket->data + (deriv_size * nCount++);
  }
}

void ad_adic_deriv_free(void* ptr)
{
#if defined(DEBUG)
  static 	count = 0;
  if (++count >= gdebug.nTokens) {
    msg("Freed %d tokens", count);
    count = 0;
  }
#endif
  
  genlist_t*	list = freeList;
  freeList = (genlist_t*)ptr;
  freeList->next = list;
}

#if defined(__cplusplus)
}
#endif

