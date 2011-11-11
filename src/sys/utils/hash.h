#ifndef _HASH_H
#define _HASH_H

/* 
 This code is adapted from the khash library, version 0.2.4. 
 It has been modified to fit into PETSc.
 Original copyright notice follows. 
*/

/* The MIT License

   Copyright (c) 2008, 2009 by attractor <attractor@live.co.uk>

   Permission is hereby granted, free of charge, to any person obtaining
   a copy of this software and associated documentation files (the
   "Software"), to deal in the Software without restriction, including
   without limitation the rights to use, copy, modify, merge, publish,
   distribute, sublicense, and/or sell copies of the Software, and to
   permit persons to whom the Software is furnished to do so, subject to
   the following conditions:

   The above copyright notice and this permission notice shall be
   included in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
   EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
   MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
   NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
   BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
   ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
   CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
   SOFTWARE.
*/

/*
  An example:

#include "khash.h"
KHASH_MAP_INIT_INT(32, char)
int main() {
	int ret, is_missing;
	khiter_t k;
	khash_t(32) *h = kh_init(32);
	k = kh_put(32, h, 5, &ret);
	if (!ret) kh_del(32, h, k);
	kh_value(h, k) = 10;
	k = kh_get(32, h, 10);
	is_missing = (k == kh_end(h));
	k = kh_get(32, h, 5);
	kh_del(32, h, k);
	for (k = kh_begin(h); k != kh_end(h); ++k)
		if (kh_exist(h, k)) kh_value(h, k) = 1;
	kh_destroy(32, h);
	return 0;
}
*/

/*
  2009-09-26 (0.2.4):

    * Improve portability

  2008-09-19 (0.2.3):

	* Corrected the example
	* Improved interfaces

  2008-09-11 (0.2.2):

	* Improved speed a little in kh_put()

  2008-09-10 (0.2.1):

	* Added kh_clear()
	* Fixed a compiling error

  2008-09-02 (0.2.0):

	* Changed to token concatenation which increases flexibility.

  2008-08-31 (0.1.2):

	* Fixed a bug in kh_get(), which has not been tested previously.

  2008-08-31 (0.1.1):

	* Added destructor
*/


/*!
  @header

  Generic hash table library.

  @copyright Heng Li
 */


#include <stdlib.h>
#include <string.h>
#include <limits.h>

#include <petscsys.h>

/* compiler-specific configuration */
#if UINT_MAX == 0xffffffffu
typedef unsigned int khint32_t;
#elif ULONG_MAX == 0xffffffffu
typedef unsigned long khint32_t;
#endif

#if ULONG_MAX == ULLONG_MAX
typedef unsigned long khint64_t;
#elif defined(_MSC_VER)
typedef unsigned _int64 khint64_t;
#else
typedef unsigned long long khint64_t;
#endif

typedef khint32_t khint_t;
typedef khint_t khiter_t;


/* Will need to expand this list, if on-the-node hash table size grows beyond 2^32. */
#define __ac_HASH_PRIME_SIZE 32
static const khint_t __ac_prime_list[__ac_HASH_PRIME_SIZE] =
{
  0ul,          3ul,          11ul,         23ul,         53ul,
  97ul,         193ul,        389ul,        769ul,        1543ul,
  3079ul,       6151ul,       12289ul,      24593ul,      49157ul,
  98317ul,      196613ul,     393241ul,     786433ul,     1572869ul,
  3145739ul,    6291469ul,    12582917ul,   25165843ul,   50331653ul,
  100663319ul,  201326611ul,  402653189ul,  805306457ul,  1610612741ul,
  3221225473ul, 4294967291ul
};

#define __ac_isempty(flag, i) ((flag[i>>4]>>((i&0xfU)<<1))&2)
#define __ac_isdel(flag, i) ((flag[i>>4]>>((i&0xfU)<<1))&1)
#define __ac_iseither(flag, i) ((flag[i>>4]>>((i&0xfU)<<1))&3)
#define __ac_set_isdel_false(flag, i) (flag[i>>4]&=~(1ul<<((i&0xfU)<<1)))
#define __ac_set_isempty_false(flag, i) (flag[i>>4]&=~(2ul<<((i&0xfU)<<1)))
#define __ac_set_isboth_false(flag, i) (flag[i>>4]&=~(3ul<<((i&0xfU)<<1)))
#define __ac_set_isdel_true(flag, i) (flag[i>>4]|=1ul<<((i&0xfU)<<1))

static const double __ac_HASH_UPPER = 0.77;

#define KHASH_INIT(name, khkey_t, khval_t, kh_is_map, __hash_func, __hash_equal) \
	typedef struct {													\
		khint_t n_buckets, size, n_occupied, upper_bound;				\
		khint32_t *flags;												\
		khkey_t *keys;													\
		khval_t *vals;													\
	} kh_##name##_t;													\
	PETSC_STATIC_INLINE kh_##name##_t *kh_init_##name() {						\
		return (kh_##name##_t*)calloc(1, sizeof(kh_##name##_t));		\
	}																	\
	PETSC_STATIC_INLINE void kh_destroy_##name(kh_##name##_t *h)				\
	{																	\
		if (h) {														\
			free(h->keys); free(h->flags);								\
			free(h->vals);												\
			free(h);													\
		}																\
	}																	\
	PETSC_STATIC_INLINE void kh_clear_##name(kh_##name##_t *h)				\
	{																	\
		if (h && h->flags) {											\
			memset(h->flags, 0xaa, ((h->n_buckets>>4) + 1) * sizeof(khint32_t)); \
			h->size = h->n_occupied = 0;								\
		}																\
	}																	\
	PETSC_STATIC_INLINE khint_t kh_get_##name(const kh_##name##_t *h, khkey_t key) \
	{																	\
		if (h->n_buckets) {												\
			khint_t inc, k, i, last;									\
			k = __hash_func(key); i = k % h->n_buckets;					\
			inc = 1 + k % (h->n_buckets - 1); last = i;					\
			while (!__ac_isempty(h->flags, i) && (__ac_isdel(h->flags, i) || !__hash_equal(h->keys[i], key))) { \
				if (i + inc >= h->n_buckets) i = i + inc - h->n_buckets; \
				else i += inc;											\
				if (i == last) return h->n_buckets;						\
			}															\
			return __ac_iseither(h->flags, i)? h->n_buckets : i;		\
		} else return 0;												\
	}																	\
	PETSC_STATIC_INLINE void kh_resize_##name(kh_##name##_t *h, khint_t new_n_buckets) \
	{																	\
		khint32_t *new_flags = 0;										\
		khint_t j = 1;													\
		{																\
			khint_t t = __ac_HASH_PRIME_SIZE - 1;						\
			while (__ac_prime_list[t] > new_n_buckets) --t;				\
			new_n_buckets = __ac_prime_list[t+1];						\
			if (h->size >= (khint_t)(new_n_buckets * __ac_HASH_UPPER + 0.5)) j = 0;	\
			else {														\
				new_flags = (khint32_t*)malloc(((new_n_buckets>>4) + 1) * sizeof(khint32_t));	\
				memset(new_flags, 0xaa, ((new_n_buckets>>4) + 1) * sizeof(khint32_t)); \
				if (h->n_buckets < new_n_buckets) {						\
					h->keys = (khkey_t*)realloc(h->keys, new_n_buckets * sizeof(khkey_t)); \
					if (kh_is_map)										\
						h->vals = (khval_t*)realloc(h->vals, new_n_buckets * sizeof(khval_t)); \
				}														\
			}															\
		}																\
		if (j) {														\
			for (j = 0; j != h->n_buckets; ++j) {						\
				if (__ac_iseither(h->flags, j) == 0) {					\
					khkey_t key = h->keys[j];							\
					khval_t val;										\
					if (kh_is_map) val = h->vals[j];					\
					__ac_set_isdel_true(h->flags, j);					\
					while (1) {											\
						khint_t inc, k, i;								\
						k = __hash_func(key);							\
						i = k % new_n_buckets;							\
						inc = 1 + k % (new_n_buckets - 1);				\
						while (!__ac_isempty(new_flags, i)) {			\
							if (i + inc >= new_n_buckets) i = i + inc - new_n_buckets; \
							else i += inc;								\
						}												\
						__ac_set_isempty_false(new_flags, i);			\
						if (i < h->n_buckets && __ac_iseither(h->flags, i) == 0) { \
							{ khkey_t tmp = h->keys[i]; h->keys[i] = key; key = tmp; } \
							if (kh_is_map) { khval_t tmp = h->vals[i]; h->vals[i] = val; val = tmp; } \
							__ac_set_isdel_true(h->flags, i);			\
						} else {										\
							h->keys[i] = key;							\
							if (kh_is_map) h->vals[i] = val;			\
							break;										\
						}												\
					}													\
				}														\
			}															\
			if (h->n_buckets > new_n_buckets) {							\
				h->keys = (khkey_t*)realloc(h->keys, new_n_buckets * sizeof(khkey_t)); \
				if (kh_is_map)											\
					h->vals = (khval_t*)realloc(h->vals, new_n_buckets * sizeof(khval_t)); \
			}															\
			free(h->flags);												\
			h->flags = new_flags;										\
			h->n_buckets = new_n_buckets;								\
			h->n_occupied = h->size;									\
			h->upper_bound = (khint_t)(h->n_buckets * __ac_HASH_UPPER + 0.5); \
		}																\
	}																	\
	PETSC_STATIC_INLINE khint_t kh_put_##name(kh_##name##_t *h, khkey_t key, khint_t *ret) \
	{																	\
		khint_t x;														\
		if (h->n_occupied >= h->upper_bound) {							\
			if (h->n_buckets > (h->size<<1)) kh_resize_##name(h, h->n_buckets - 1); \
			else kh_resize_##name(h, h->n_buckets + 1);					\
		}																\
		{																\
			khint_t inc, k, i, site, last;								\
			x = site = h->n_buckets; k = __hash_func(key); i = k % h->n_buckets; \
			if (__ac_isempty(h->flags, i)) x = i;						\
			else {														\
				inc = 1 + k % (h->n_buckets - 1); last = i;				\
				while (!__ac_isempty(h->flags, i) && (__ac_isdel(h->flags, i) || !__hash_equal(h->keys[i], key))) { \
					if (__ac_isdel(h->flags, i)) site = i;				\
					if (i + inc >= h->n_buckets) i = i + inc - h->n_buckets; \
					else i += inc;										\
					if (i == last) { x = site; break; }					\
				}														\
				if (x == h->n_buckets) {								\
					if (__ac_isempty(h->flags, i) && site != h->n_buckets) x = site; \
					else x = i;											\
				}														\
			}															\
		}																\
		if (__ac_isempty(h->flags, x)) {								\
			h->keys[x] = key;											\
			__ac_set_isboth_false(h->flags, x);							\
			++h->size; ++h->n_occupied;									\
			*ret = 1;													\
		} else if (__ac_isdel(h->flags, x)) {							\
			h->keys[x] = key;											\
			__ac_set_isboth_false(h->flags, x);							\
			++h->size;													\
			*ret = 2;													\
		} else *ret = 0;												\
		return x;														\
	}																	\
	PETSC_STATIC_INLINE void kh_del_##name(kh_##name##_t *h, khint_t x)		\
	{																	\
		if (x != h->n_buckets && !__ac_iseither(h->flags, x)) {			\
			__ac_set_isdel_true(h->flags, x);							\
			--h->size;													\
		}																\
	}

/* --- BEGIN OF HASH FUNCTIONS --- */

/*! @function
  @abstract     Integer hash function
  @param  key   The integer [khint32_t]
  @return       The hash value [khint_t]
 */
#define kh_int_hash_func(key) (khint32_t)(key)
/*! @function
  @abstract     Integer comparison function
 */
#define kh_int_hash_equal(a, b) ((a) == (b))
/*! @function
  @abstract     64-bit integer hash function
  @param  key   The integer [khint64_t]
  @return       The hash value [khint_t]
 */
#define kh_int64_hash_func(key) (khint32_t)((key)>>33^(key)^(key)<<11)
/*! @function
  @abstract     64-bit integer comparison function
 */
#define kh_int64_hash_equal(a, b) ((a) == (b))
/*! @function
  @abstract     const char* hash function
  @param  s     Pointer to a null terminated string
  @return       The hash value
 */
PETSC_STATIC_INLINE khint_t __ac_X31_hash_string(const char *s)
{
	khint_t h = *s;
	if (h) for (++s ; *s; ++s) h = (h << 5) - h + *s;
	return h;
}
/*! @function
  @abstract     Another interface to const char* hash function
  @param  key   Pointer to a null terminated string [const char*]
  @return       The hash value [khint_t]
 */
#define kh_str_hash_func(key) __ac_X31_hash_string(key)
/*! @function
  @abstract     Const char* comparison function
 */
#define kh_str_hash_equal(a, b) (strcmp(a, b) == 0)

/* --- END OF HASH FUNCTIONS --- */

/* Other necessary macros... */

/*!
  @abstract Type of the hash table.
  @param  name  Name of the hash table [symbol]
 */
#define khash_t(name) kh_##name##_t

/*! @function
  @abstract     Initiate a hash table.
  @param  name  Name of the hash table [symbol]
  @return       Pointer to the hash table [khash_t(name)*]
 */
#define kh_init(name) kh_init_##name()

/*! @function
  @abstract     Destroy a hash table.
  @param  name  Name of the hash table [symbol]
  @param  h     Pointer to the hash table [khash_t(name)*]
 */
#define kh_destroy(name, h) kh_destroy_##name(h)

/*! @function
  @abstract     Reset a hash table without deallocating memory.
  @param  name  Name of the hash table [symbol]
  @param  h     Pointer to the hash table [khash_t(name)*]
 */
#define kh_clear(name, h) kh_clear_##name(h)

/*! @function
  @abstract     Resize a hash table.
  @param  name  Name of the hash table [symbol]
  @param  h     Pointer to the hash table [khash_t(name)*]
  @param  s     New size [khint_t]
 */
#define kh_resize(name, h, s) kh_resize_##name(h, s)

/*! @function
  @abstract     Insert a key to the hash table.
  @param  name  Name of the hash table [symbol]
  @param  h     Pointer to the hash table [khash_t(name)*]
  @param  k     Key [type of keys]
  @param  r     Extra return code: 0 if the key is present in the hash table;
                1 if the bucket is empty (never used); 2 if the element in
				the bucket has been deleted [int*]
  @return       Iterator to the inserted element [khint_t]
 */
#define kh_put(name, h, k, r) kh_put_##name(h, k, r)

/*! @function
  @abstract     Retrieve a key from the hash table.
  @param  name  Name of the hash table [symbol]
  @param  h     Pointer to the hash table [khash_t(name)*]
  @param  k     Key [type of keys]
  @return       Iterator to the found element, or kh_end(h) is the element is absent [khint_t]
 */
#define kh_get(name, h, k) kh_get_##name(h, k)

/*! @function
  @abstract     Remove a key from the hash table.
  @param  name  Name of the hash table [symbol]
  @param  h     Pointer to the hash table [khash_t(name)*]
  @param  k     Iterator to the element to be deleted [khint_t]
 */
#define kh_del(name, h, k) kh_del_##name(h, k)

/*! @function
  @abstract     Test whether a bucket contains data.
  @param  h     Pointer to the hash table [khash_t(name)*]
  @param  x     Iterator to the bucket [khint_t]
  @return       1 if containing data; 0 otherwise [int]
 */
#define kh_exist(h, x) (!__ac_iseither((h)->flags, (x)))

/*! @function
  @abstract     Get key given an iterator
  @param  h     Pointer to the hash table [khash_t(name)*]
  @param  x     Iterator to the bucket [khint_t]
  @return       Key [type of keys]
 */
#define kh_key(h, x) ((h)->keys[x])

/*! @function
  @abstract     Get value given an iterator
  @param  h     Pointer to the hash table [khash_t(name)*]
  @param  x     Iterator to the bucket [khint_t]
  @return       Value [type of values]
  @discussion   For hash sets, calling this results in segfault.
 */
#define kh_val(h, x) ((h)->vals[x])

/*! @function
  @abstract     Alias of kh_val()
 */
#define kh_value(h, x) ((h)->vals[x])

/*! @function
  @abstract     Get the start iterator
  @param  h     Pointer to the hash table [khash_t(name)*]
  @return       The start iterator [khint_t]
 */
#define kh_begin(h) (khint_t)(0)

/*! @function
  @abstract     Get the end iterator
  @param  h     Pointer to the hash table [khash_t(name)*]
  @return       The end iterator [khint_t]
 */
#define kh_end(h) ((h)->n_buckets)

/*! @function
  @abstract     Get the number of elements in the hash table
  @param  h     Pointer to the hash table [khash_t(name)*]
  @return       Number of elements in the hash table [khint_t]
 */
#define kh_size(h) ((h)->size)

/*! @function
  @abstract     Get the number of buckets in the hash table
  @param  h     Pointer to the hash table [khash_t(name)*]
  @return       Number of buckets in the hash table [khint_t]
 */
#define kh_n_buckets(h) ((h)->n_buckets)

/* More conenient interfaces */

/*! @function
  @abstract     Instantiate a hash set containing integer keys
  @param  name  Name of the hash table [symbol]
 */
#define KHASH_SET_INIT_INT(name)										\
	KHASH_INIT(name, khint32_t, char, 0, kh_int_hash_func, kh_int_hash_equal)

/*! @function
  @abstract     Instantiate a hash map containing integer keys
  @param  name  Name of the hash table [symbol]
  @param  khval_t  Type of values [type]
 */
#define KHASH_MAP_INIT_INT(name, khval_t)								\
	KHASH_INIT(name, khint32_t, khval_t, 1, kh_int_hash_func, kh_int_hash_equal)

/*! @function
  @abstract     Instantiate a hash map containing 64-bit integer keys
  @param  name  Name of the hash table [symbol]
 */
#define KHASH_SET_INIT_INT64(name)										\
	KHASH_INIT(name, khint64_t, char, 0, kh_int64_hash_func, kh_int64_hash_equal)

/*! @function
  @abstract     Instantiate a hash map containing 64-bit integer keys
  @param  name  Name of the hash table [symbol]
  @param  khval_t  Type of values [type]
 */
#define KHASH_MAP_INIT_INT64(name, khval_t)								\
	KHASH_INIT(name, khint64_t, khval_t, 1, kh_int64_hash_func, kh_int64_hash_equal)

#define KHASH_SET_INIT_STR(name)										\
	KHASH_INIT(name, kh_cstr_t, char, 0, kh_str_hash_func, kh_str_hash_equal)

/*! @function
  @abstract     Instantiate a hash map containing const char* keys
  @param  name  Name of the hash table [symbol]
  @param  khval_t  Type of values [type]
 */
#define KHASH_MAP_INIT_STR(name, khval_t)								\
	KHASH_INIT(name, kh_cstr_t, khval_t, 1, kh_str_hash_func, kh_str_hash_equal)

/* HASHI */
#if PETSC_SIZEOF_INT == 8
KHASH_MAP_INIT_INT64(HASHI,PetscInt)
#elif PETSC_SIZEOF_INT == 4
KHASH_MAP_INIT_INT(HASHI,PetscInt)
#else
#error "This value of PETSC_SIZEOF_INT is not supported by petsckhash."
#endif

typedef khash_t(HASHI) *PetscHashI;

typedef khiter_t   PetscHashIIter;

#define PetscHashICreate(ht) ((ht) = kh_init(HASHI))

#define PetscHashIClear(ht)    (kh_clear(HASHI,(ht)))

#define PetscHashIDestroy(ht) if((ht)){kh_destroy(HASHI,(ht));(ht)=0;}

#define PetscHashIResize(ht,n) (kh_resize(HASHI,(ht),(n)))

#define PetscHashISize(ht,n)     ((n)=kh_size((ht)))

#define PetscHashIIterNext(ht,hi)  do{++(hi);} while(!kh_exist((ht),(hi)) && (hi) != kh_end((ht)))

#define PetscHashIIterBegin(ht,hi) {(hi) = kh_begin((ht));if(!kh_exist((ht),(hi))){PetscHashIIterNext((ht),(hi));}}


#define PetscHashIIterAtEnd(ht,hi) ((hi) == kh_end((ht)))

#define PetscHashIIterGetKeyVal(ht,hi,i,ii) if(kh_exist((ht),(hi)))((i) = kh_key((ht),(hi)),(ii) = kh_val((ht),(hi))); else ((i) = -1, (ii) = -1);
#define PetscHashIIterGetKey(ht,hi,i) if(kh_exist((ht),(hi)))((i) = kh_key((ht),(hi))); else ((i) = -1);
#define PetscHashIIterGetVal(ht,hi,ii) if(kh_exist((ht),(hi)))((ii) = kh_val((ht),(hi))); else ((ii) = -1);

#define PetscHashIAdd(ht,i,ii)                                          \
{                                                                       \
 khiter_t _11_hi;                                                       \
 khint_t  _11_hr;                                                       \
 _11_hi = kh_put(HASHI,(ht),(i),&_11_hr);                               \
 kh_val((ht),_11_hi) = (ii);                                            \
}
/*
  arr is the integer array to put the indices to, n is the offset into arr to start putting the indices at.
  n is updated as the indices are put into arr, so n must be an lvalue.
 */
#define PetscHashIGetKeys(ht,n,arr)                                     \
{                                                                       \
  PetscHashIIter _12_hi;                                                \
  PetscInt _12_i;                                                       \
  PetscHashIIterBegin((ht),_12_hi);                                     \
  while(!PetscHashIIterAtEnd((ht),_12_hi)) {                            \
    PetscHashIIterGetKey((ht),_12_hi,_12_i);                            \
    (arr)[(n)++] = _12_i;                                               \
    PetscHashIIterNext((ht),_12_hi);                                    \
  }                                                                     \
}

#define PetscHashIGetVals(ht,n,arr)                                    \
{                                                                       \
  PetscHashIIter _12_hi;                                                \
  PetscInt _12_ii;                                                      \
  PetscHashIBegin((ht),_12_hi);                                         \
  while(!PetscHashIIterAtEnd((ht),_12_hi)){                                 \
    PetscHashIIterGetVal((ht),_12_hi,_12_ii);                           \
    (arr)[(n)++] = _12_ii;                                              \
    PetscHashIIterNext((ht),_12_hi);                                        \
  }                                                                     \
}

#define PetscHashIDuplicate(ht,hd)                                      \
{                                                                       \
  PetscHashIIter  _14_hi;                                               \
  PetscInt   _14_i, _14_ii;                                             \
  PetscHashICreate((hd));                                               \
  PetscHashIIterBegin((ht),_14_hi);                                     \
  while(!PetscHashIIterAtEnd((ht),_14_hi)){                             \
    PetscHashIIterGetKeyVal((ht),_14_hi,_14_i,_14_ii);                  \
    PetscHashIAdd((hd), _14_i,_14_ii);                                  \
    PetscHashIIterNext((ht),_14_hi);                                    \
  }                                                                     \
}

/*
 Locate index i in the hash table ht. If i is found in table, ii is its index, 
 between 0 and kh_size(ht)-1 (inclusive); otherwise, ii == -1. 
 */
#define PetscHashIMap(ht,i,ii)  \
{                                          \
  khiter_t _9_hi;                          \
  _9_hi = kh_get(HASHI,(ht),(i));              \
  if(_9_hi != kh_end((ht)))                \
    (ii) = kh_val((ht),_9_hi);             \
  else                                     \
    (ii) = -1;                             \
}                                          \

/* 
 Locate all integers from array iarr of length len in hash table ht.
 Their images -- their local numbering -- are stored in iiarr of length len.  
 If drop == PETSC_TRUE:
  - if an integer is not found in table, it is omitted and upon completion 
    iilen has the number of located indices; iilen <= ilen in this case. 
 If drop == PETSC_FALSE:
  - if an integer is not found in table, it is replaced by -1; iilen == ilen
    upon completion.
 */
#define PetscHashIMapArray(ht,ilen,iarr,iilen,iiarr)                   \
  do {                                                                 \
    PetscInt _10_i;                                                    \
    (iilen) = 0;                                                       \
    for(_10_i = 0, (iilen) = 0; _10_i < (ilen); ++_10_i) {             \
      PetscHashIMap(ht,(iarr)[_10_i],(iiarr)[(iilen)]);                \
      if((iiarr)[(iilen)] != -1) ++(iilen);                            \
    }                                                                  \
} while(0)

/* HASHIJ */
/* Linked list of values in a bucket. */
struct _IJNode {
  PetscInt k;
  struct _IJNode *next;
};
typedef struct _IJNode IJNode;

/* Value (holds a linked list of nodes) in the bucket. */
struct _IJVal {
  PetscInt n;
  IJNode *head, *tail;
};
typedef struct _IJVal IJVal;

/* Key (a pair of integers). */
struct _PetscHashIJKey {
  PetscInt i, j;
};
typedef struct _PetscHashIJKey PetscHashIJKey;

/* Hash function: mix two integers into one. 
   Shift by half the number of bits in PetscInt to the left and then XOR.  If the indices fit into the lowest half part of PetscInt, this is a bijection.
   We should shift by (8/2)*sizeof(PetscInt): sizeof(PetscInt) is the number of bytes in PetscInt, with 8 bits per byte.
 */
#define IJKeyHash(key) ((((key).i) << (4*sizeof(PetscInt)))^((key).j))

/* Compare two keys (integer pairs). */
#define IJKeyEqual(k1,k2) (((k1).i==(k2).i)?((k1).j==(k2).j):0)

KHASH_INIT(HASHIJ,PetscHashIJKey,IJVal,1,IJKeyHash,IJKeyEqual)

struct _PetscHashIJ {
  PetscBool multivalued;
  PetscInt size;
  khash_t(HASHIJ) *ht;
};


typedef struct _PetscHashIJ *PetscHashIJ;

typedef khiter_t             PetscHashIJIter;

typedef IJNode              *PetscHashIJValIter;

#undef  __FUNCT__
#define __FUNCT__ "PetscHashIJCreate"
PETSC_STATIC_INLINE PetscErrorCode PetscHashIJCreate(PetscHashIJ *h){
  PetscErrorCode _15_ierr;
  PetscFunctionBegin;
  PetscValidPointer(h,1);
  _15_ierr = PetscNew(struct _PetscHashIJ, (h)); CHKERRQ(_15_ierr);
  (*h)->ht = kh_init(HASHIJ);
  (*h)->multivalued = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "PetscHashIJGetMultivalued"
PETSC_STATIC_INLINE PetscErrorCode PetscHashIJGetMultivalued(PetscHashIJ h, PetscBool *m) {
  PetscFunctionBegin;
  *m = (h)->multivalued;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "PetscHashIJSetMultivalued"
PETSC_STATIC_INLINE PetscErrorCode PetscHashIJSetMultivalued(PetscHashIJ h, PetscBool m) {
  PetscFunctionBegin;
  (h)->multivalued = m;
  PetscFunctionReturn(0);
}



#undef  __FUNCT__
#define __FUNCT__ "PetscHashIJResize"
PETSC_STATIC_INLINE PetscErrorCode PetscHashIJResize(PetscHashIJ h, PetscInt n){
  PetscFunctionBegin;
  (kh_resize(HASHIJ,(h)->ht,(n)));
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "PetscHashIJKeySize"
PETSC_STATIC_INLINE PetscErrorCode PetscHashIJKeySize(PetscHashIJ h, PetscInt *n){
  PetscFunctionBegin;
  ((*n)=kh_size((h)->ht));
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "PetscHashIJSize"
PETSC_STATIC_INLINE PetscErrorCode PetscHashIJSize(PetscHashIJ h, PetscInt *m){
  PetscFunctionBegin;
  (*m)=h->size;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "PetscHashIJIterNext"
PETSC_STATIC_INLINE PetscErrorCode PetscHashIJIterNext(PetscHashIJ h, PetscHashIJIter hi, PetscHashIJIter *hn)  {
  PetscFunctionBegin;
  *hn = hi;
  do{++(*hn);} while(!kh_exist((h)->ht,(*hn)) && (*hn) != kh_end((h)->ht));
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "PetscHashIJIterBegin"
PETSC_STATIC_INLINE PetscErrorCode PetscHashIJIterBegin(PetscHashIJ h, PetscHashIJIter *hi) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  (*hi) = kh_begin((h)->ht);if(*hi != kh_end((h)->ht) && !kh_exist((h)->ht,(*hi))){ierr = PetscHashIJIterNext((h),(*hi),(hi)); CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#define PetscHashIJIterAtEnd(h,hi) ((hi) == kh_end((h)->ht))

#undef  __FUNCT__
#define __FUNCT__ "PetscHashIJGetKey"
PETSC_STATIC_INLINE PetscErrorCode PetscHashIJIterGetKey(PetscHashIJ h, PetscHashIJIter hi, PetscHashIJKey *key) {
  PetscFunctionBegin;
  (*key) = kh_key((h)->ht,(hi));
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "PetscHashIJGetValIter"
PETSC_STATIC_INLINE PetscErrorCode PetscHashIJIterGetValIter(PetscHashIJ h, PetscHashIJIter hi, PetscHashIJValIter *vi) {
  PetscFunctionBegin;
  if(hi != kh_end(h->ht) && kh_exist((h)->ht,(hi)))((*vi) = kh_val((h)->ht,(hi)).head); else ((*vi) = 0);
  PetscFunctionReturn(0);
}

#define PetscHashIJValIterAtEnd(h, vi) ((vi) == 0)

#undef  __FUNCT__
#define __FUNCT__ "PetscHashIJValIterNext"
PETSC_STATIC_INLINE PetscErrorCode PetscHashIJValIterNext(PetscHashIJ h, PetscHashIJValIter vi, PetscHashIJValIter *vn) {
  PetscFunctionBegin;
  ((*vn) = (vi)->next);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "PetscHashIJValIterGetVal"
PETSC_STATIC_INLINE PetscErrorCode PetscHashIJValIterGetVal(PetscHashIJ h, PetscHashIJValIter vi, PetscInt *v) {
  PetscFunctionBegin;
  ((*v) = (vi)->k);
  PetscFunctionReturn(0);
}


#undef  __FUNCT__
#define __FUNCT__ "PetscHashIJAdd"
PETSC_STATIC_INLINE PetscErrorCode PetscHashIJAdd(PetscHashIJ h,PetscHashIJKey i, PetscInt ii){
  khiter_t _11_hi;
  khint_t  _11_r;
  IJNode   *_11_ijnode;
  IJVal    *_11_ijval;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  _11_hi = kh_put(HASHIJ,(h)->ht,(i),&_11_r);
  _11_ijval = &(kh_val((h)->ht,_11_hi));
  if(_11_r) {
    _11_ijval->head = _11_ijval->tail = 0;
    _11_ijval->n = 0;
  }
  if(!_11_r && !(h)->multivalued) {
    _11_ijval->head->k = (ii);
  }
  else {
    ierr = PetscNew(IJNode, &_11_ijnode); CHKERRQ(ierr);
    _11_ijnode->k = (ii);
    _11_ijval = &(kh_val((h)->ht,_11_hi));
    if(!_11_ijval->tail) {
      _11_ijval->tail = _11_ijnode;
      _11_ijval->head = _11_ijnode;
    }
    else {
      _11_ijval->tail->next = _11_ijnode;
      _11_ijval->tail = _11_ijnode;
    }
    ++(_11_ijval->n);
    ++((h)->size);
  }
  PetscFunctionReturn(0);
}

/*
  arr is the key array to put the key to, and must be big enough to accommodate all keys.
 */
#undef  __FUNCT__
#define __FUNCT__ "PetscHashIJGetKeys"
PETSC_STATIC_INLINE PetscErrorCode PetscHashIJGetKeys(PetscHashIJ h,PetscHashIJKey *arr) {
  PetscHashIJIter _12_hi;
  PetscHashIJKey _12_key;
  PetscInt n;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  n = 0;
  ierr = PetscHashIJIterBegin((h),&_12_hi); CHKERRQ(ierr);
  while(!PetscHashIJIterAtEnd((h),_12_hi)) {
    ierr = PetscHashIJIterGetKey((h),_12_hi,&_12_key); CHKERRQ(ierr);
    (arr)[(n)++] = _12_key;
    ierr = PetscHashIJIterNext((h),_12_hi, &_12_hi); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*
  iarr,jarr,karr are integer arrays to put the indices into, and must be allocated to the right size.
 */
#undef  __FUNCT__
#define __FUNCT__ "PetscHashIJGetIndices"
PETSC_STATIC_INLINE PetscErrorCode PetscHashIJGetIndices(PetscHashIJ h, PetscInt *iarr, PetscInt *jarr, PetscInt *karr){
  PetscErrorCode ierr;
  PetscHashIJIter _12_hi;
  PetscHashIJValIter _12_vi;
  PetscHashIJKey _12_key;
  PetscInt n = 0;
  PetscFunctionBegin;
  ierr = PetscHashIJIterBegin((h),&_12_hi); CHKERRQ(ierr);
  while(!PetscHashIJIterAtEnd((h),_12_hi)) {
    ierr = PetscHashIJIterGetKey((h),_12_hi,&_12_key);    CHKERRQ(ierr);
    ierr = PetscHashIJIterGetValIter((h),_12_hi,&_12_vi); CHKERRQ(ierr);
    while(!PetscHashIJValIterAtEnd((h),_12_vi)) {
      (iarr)[(n)] = _12_key.i;
      (jarr)[(n)] = _12_key.j;
      ierr = PetscHashIJValIterGetVal((h),_12_vi,&(karr)[(n)]); CHKERRQ(ierr);
      ++(n);
      ierr = PetscHashIJValIterNext((h),_12_vi, &_12_vi); CHKERRQ(ierr);
    }
    ierr = PetscHashIJIterNext((h),_12_hi, &_12_hi); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "PetscHashIJDuplicate"
PETSC_STATIC_INLINE PetscErrorCode PetscHashIJDuplicate(PetscHashIJ h, PetscHashIJ *hd) {
  PetscHashIJIter  _14_hi;
  PetscHashIJValIter _14_vi;
  PetscHashIJKey   _14_key;
  PetscInt         _14_val;
  PetscErrorCode   ierr;
  PetscFunctionBegin;
  ierr = PetscHashIJCreate((hd)); CHKERRQ(ierr);
  ierr = PetscHashIJIterBegin((h),&_14_hi); CHKERRQ(ierr);
  while(!PetscHashIJIterAtEnd((h),_14_hi)){
    ierr = PetscHashIJIterGetKey((h),_14_hi,&_14_key); CHKERRQ(ierr);
    ierr = PetscHashIJIterGetValIter((h),_14_hi,&_14_vi); CHKERRQ(ierr);
    while(!PetscHashIJValIterAtEnd((h),_14_vi)) {
      ierr = PetscHashIJValIterNext((h),_14_vi,&_14_vi); CHKERRQ(ierr);
      ierr = PetscHashIJValIterGetVal((h),_14_vi,&_14_val); CHKERRQ(ierr);
      ierr = PetscHashIJAdd((*hd), _14_key,_14_val); CHKERRQ(ierr);
      ierr = PetscHashIJValIterNext((h),_14_vi,&_14_vi); CHKERRQ(ierr);
    }
    ierr = PetscHashIJIterNext((h),_14_hi, &_14_hi); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "PetscHashIJClearValues"
PETSC_STATIC_INLINE PetscErrorCode PetscHashIJClearValues(PetscHashIJ h) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if((h) && (h)->ht) {
    PetscHashIJIter _15_hi;
    PetscHashIJValIter  _15_vi, _15_vid;
    PetscErrorCode _15_ierr;
    ierr = PetscHashIJIterBegin((h),&_15_hi); CHKERRQ(ierr);
    while(!PetscHashIJIterAtEnd((h),_15_hi)) {
      ierr = PetscHashIJIterGetValIter((h),_15_hi,&_15_vi); CHKERRQ(ierr);
      while(!PetscHashIJValIterAtEnd((h),_15_vi)){ 
        _15_vid = _15_vi;
        ierr = PetscHashIJValIterNext((h),_15_vi,&_15_vi); CHKERRQ(ierr);
        _15_vid->next = 0;
        _15_ierr = PetscFree(_15_vid); CHKERRQ(_15_ierr);
      }
      ierr = PetscHashIJIterNext((h),_15_hi,&_15_hi); CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "PetscHashIJClear"
PETSC_STATIC_INLINE PetscErrorCode PetscHashIJClear(PetscHashIJ h) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscHashIJClearValues((h)); CHKERRQ(ierr);
  kh_clear(HASHIJ,(h)->ht);   
  (h)->size = 0;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__ 
#define __FUNCT__ "PetscHashIJDestroy"
PETSC_STATIC_INLINE PetscErrorCode PetscHashIJDestroy(PetscHashIJ *h){
  PetscFunctionBegin;
  PetscValidPointer(h,1);
  if((*h)) {
    PetscErrorCode _16_ierr;
    PetscHashIJClearValues((*h));
    if((*h)->ht) {
      kh_destroy(HASHIJ,(*h)->ht);
      (*h)->ht=0;
    }
    _16_ierr = PetscFree((*h)); CHKERRQ(_16_ierr);  
  }
  PetscFunctionReturn(0);
}





#endif /* _KHASH_H */

