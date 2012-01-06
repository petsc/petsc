#include <petscconf.h>
#if defined(PETSC_HAVE_STDLIB_H)
#include <stdlib.h>
#endif
#if defined(PETSC_HAVE_MALLOC_H)
#include <malloc.h>
#endif

#include "adic/run-map.h"
    
#if defined(__cplusplus)
extern "C" {
#endif

void* ad_map_init(int dsize, int msize, int bsize, int asize)
{ 
  int 	i;
  MapEntry*	entry;
  char*	pblock;
  
  desc_size = dsize;
  entry_size = dsize + sizeof(Pair) - sizeof(double);
  if (!asize) {
    buckets_per_block = DEFAULT_BUCKETS_PER_BLOCK;
  } else {
    buckets_per_block = asize;
  }	  
  
  if (!msize) {
      map_size = DEFAULT_MAP_SIZE;
  } else {
    map_size = msize;
  }
  if (!bsize) {
    entries_per_bucket = DEFAULT_BUCKET_SIZE;
  } else {
    entries_per_bucket = bsize;
  }
  bucket_size = entries_per_bucket*entry_size;
  if (map_def) {
    free(map_def);
  }
  map_def = (MapEntry*)calloc(map_size, sizeof(MapEntry) + bucket_size);
  entry = map_def;
  pblock = (char*)(map_def + map_size);
  for (i = 0; i < map_size; i++) {
    entry->next = (Pair*)pblock;
    pblock += bucket_size;
    entry++;
  }
  
  freeList = 0;
  blockList = 0;
  curBlock = 0;
  return(map_def);
}
  
void ad_map_cleanup()
{
  if (blockList) {
    genlist_t* block = blockList;
    genlist_t* tmp;
    while ((tmp = block->next)) {
      free(block);
      block = tmp;
    }
    free(block);
    blockList = 0;
  }
  
  free(map_def);
  map_def = 0;
}

/*
void* ad_map_reg_array_d(double* base, int size)
{
  assert(!array.base);
  array.base = base;
  array.top = base + size;
  array.desc = calloc(size, desc_size);
  array.isSingle = 0;
}
*/

/*
void* ad_map_reg_array_s(float* base, int size)
{
  assert(!array.base);
  array.base = base;
  array.top = base + size/2;
  array.desc = calloc(size, desc_size);
  array.isSingle = 1;
}
*/


void* ad_map_get(void* key)
{
  Pair *pa;
  /*
  if (key < array.top && key >= array.base) {
    if (array.isSingle) {
      return array.desc + ((single*)key - (single*)array.base);
    }
    else {
      return array.desc + ((double*)key - array.base);
    }
  }
  */
  MapEntry*	entry = map_def + (((long)key>>3) % map_size);
  if (entry->cache && entry->cache->key == key) {
    return entry->cache->val;
  }
  while (1) {
    int 	i = 0;
    pa = entry->next;
    while (++i < entries_per_bucket) {
      
      if (pa->key == key) {
        entry->cache = pa;
        return pa->val;
      }
      else if (!pa->key) {
        pa->key = key;
        entry->cache = pa;
        return pa->val;
      }
      else {
        pa = (Pair*)((char*)pa + entry_size);
      }
    }
    if (pa->key) {
      pa = (Pair*)pa->key;
    }
    else {
      Pair* tmp = (Pair*)ad_map_alloc_bucket();
      pa->key = tmp;
      tmp->key = key;
      entry->cache = tmp;
      return tmp->val;
    }
  }
}


static void* ad_map_alloc_bucket(void)
{
#if defined(DEBUG)
  static 	count = 0;
  if (++count >= gdebug.nTokens) {
    msg("Allocated %d tokens", count);
    count = 0;
  }
#endif
  static int	nBlocks;
  static int	nCurBucket;
  
  if (!curBlock || nCurBucket >= buckets_per_block) {
    if (freeList) {
      curBlock = freeList;
      freeList = freeList->next;
    }
    else {
      curBlock = (genlist_t*)calloc(sizeof(genlist_t) - sizeof(double) + 
                                    buckets_per_block * bucket_size, 1);
      curBlock->next = blockList;
      blockList = curBlock;
      nBlocks++;
    }
    nCurBucket = 0;
  }
  return (char*)curBlock->data + (nCurBucket++)*bucket_size;
}


void* ad_map_free_bucket(void* ptr)
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
  return(freeList);
}

void* ad_map_free(void* key)
{
  void** p = (void**)ad_map_get(key);
  *(p-1) = (void*)1;
  return(*p);
}


#if defined(__cplusplus)
}
#endif


