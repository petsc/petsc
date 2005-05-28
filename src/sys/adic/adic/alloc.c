\title{Runtime Library: Descriptor Object Mapping}
\author{Lucas Roh}
\date{\today}
\maketitle
\begin{abstract}
\end{abstract}
\tableofcontents

@ \section{Introduction}

We need to provide a mapping between an active variable with its
derivative object.  Ideally the derivative object has the same lifetime
as its active variable.  We must be able to
support the derivative object being a complex object that may contain
pointers.  The pointers may point to some variable sized objects, which
ideally should also have the same lifetimes as the active variable.
Overall, there is a complex object memory management issues that must be
resolved.  

Due to the fact that aliasing in general cannot be resolved, we use
the address of an object as its identifier.  Two variables can have
the same address but with different lifetimes.

We can assume the following preconditions: 
\begin{itemize} 
%
\item The lifetime of the derivative object can be the superset of that
of its active variable.  
%
\item An active variable is always initialized before used, except for
the static variables.  
%
\item When an active variable is set, all the previous values of the
associated derivative object is irrelevant.
\end{itemize}

The above preconditions means that the same derivative object may be
used for different active variables with the same address.  However,
this requires one to know the status of derivative objects.  For
example, if a derivative object contains a pointer, we need to know
whether it points to a valid object so that we may allocate a new
object, use the existing object, or free it.  The upshot is that we
need to initialize the derivative object to some known state before we
can write new values into the object.

Without additional descriptions about derivative objects, and the
knowledge of the lifetimes of variables, it is difficult to manage
derivative object memory management.

Hence, we use two-tier derivative object handling.  At the bottom tier,
we provide a basic mapping between a variable address and the associated
derivative object.  The derivative object is viewed as a blackbox object
of a given size.  The first time a variable address is encountered, it
allocates the derivative object and maps it to the variable address.
The user has the option of always initializing the entire derivative
object to zero.  Also, the bottom tier provides an explicit deallocate
operation that frees the derivative object.  The caller must be sure
that the derivative object itself does not point to any additional
objects; otherwise, we have garbage objects.  The top tier provides the
management of derivative objects.  It understands the makeup of a
derivative object and provides the management of additional objects
pointed by entries in the derivative objects.  For each type of
derivative objects, there is a unique top tier.  Typically, the top
tiers are provided as part of AIF modules.  In this library, we provide
the bottom tier services.

The following low-level services are available:
\begin{itemize}
\item AD_INIT_MAP
\item AD_CLEANUP_MAP
\item void* AD_GET_DERIV_OBJ(x)
\item AD_FREE_DERIV_OBJ(x)
\end{itemize}

The following high-level services and declarations are required.
\begin{verbatim}
\item DERIV_name(x) defined for each field {\tt name} of the derivative
object.
\item DERIV_val(x)
\item DERIV_TYPE
\end{verbatim}


In order to associate a derivative object with an active variable, we
use an associative map.  Each entry maps the address of an active
variable to the derivative object.  The map is maintained as a hash
table indexed by the variable address.  It is organized as an array of
linked buckets.  Each bucket is a fixed size array of map entries.
Additional buckets are linked.  A map entry contains a pointer to the
active variable with the associated derivative object occupying the
space immediately after the pointer.

The desc_size is the size of the derivative object.  

<<assoc header>>=

static int entry_size = 0;
static int bucket_size = 0;
static int entries_per_bucket = 0;
static int map_size = 0;


<<assoc header>>=

#define DEFAULT_MAP_SIZE		1000
#define DEFAULT_BUCKET_SIZE		10
#define DEFAULT_BUCKETS_PER_BLOCK	100
    typedef struct {
        void* key;
        double val[1];
    } Pair;
    typedef struct {
        Pair* cache;
        Pair* next;
    } MapEntry;
    static MapEntry* map = 0;

    typedef struct genlist {
        struct genlist *next;
        double data[1];
    } genlist_t;
    static genlist_t* freeList;
    static genlist_t* blockList;
    static genlist_t* curBlock;

    typedef struct {
        int isSingle;
        double* base;
        double* top;
        void* desc;
    } ArrayEntry;
    static ArrayEntry	array;


@ \section{Initializing the Map}

The associative map is first initialized by calling an initialization
routine.  This routine allocates/initializes a table of the given size.
The initialization routine takes several parameters:
\begin{itemize}
\item dsize: the derivative object size (in bytes).  This value should
not be zero.
\item msize: the main table size in terms of number of entries
\item bsize: the bucket size in terms of entries.  This value must be at 
least 2.
\item asize: the number of buckets per allocation block.  
\end{itemize}

If a parameter value is set to zero, then a default value is chosen.  

We compute entry_size = dsize + sizeof(Pair) - sizeof(double) to
account for alignment problems.
We allocate for each entry of the map an initial bucket from the
same storage block.

<<assoc header>>=
    void* ad_map_init(int dsize, int msize, int bsize);

<<assoc function definitions>>=
    void* ad_map_init(int dsize, int msize, int bsize)
    { 
        int 	i;
        Pair*	pa;
        MapEntry*	entry;
	char*	pblock;

	desc_size = dsize;
	entry_size = dsize + sizeof(Pair) - sizeof(double);

	if (msize == 0) {
	    map_size = DEFAULT_MAP_SIZE;
	}
	else {
	    map_size = msize;
	}
	if (bsize == 0) {
	    entries_per_bucket = DEFAULT_BUCKET_SIZE;
	}
	else {
	    entries_per_bucket = bsize;
	}
	bucket_size = entries_per_bucket*entry_size;
	if (map) {
	    free(map);
	}
        map = calloc(map_size, sizeof(MapEntry) + bucket_size);
	entry = map;
        pblock = (char*)(map + map_size);
	for (i = 0; i < map_size; i++) {
	    entry->next = (Pair*)pblock;
	    pblock += bucket_size;
	    entry++;
	}

        freeList = 0;
	blockList = 0;
	curBlock = 0;
    }



@ The cleanup phase deallocates the map and all allocated buckets.

<<assoc header>>=
    void ad_map_cleanup();

<<assoc function definitions>>=
    void ad_map_cleanup()
    {
        int 	i;
	<<delete all blocks>>
        free(map);
	map = 0;
    }


<<delete all blocks>>=

if (blockList) {
    genlist_t* block = blockList;
    genlist_t* tmp;
    while (tmp = block->next) {
        free(block);
	block = tmp;
    }
    free(block);
    blockList = 0;
}

@ 
<<assoc header>>=
    void* ad_map_reg_array_d(double* base, int size);

<<assoc function definitions>>=
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

<<assoc header>>=
    void* ad_map_reg_array_s(float* base, int size);

<<assoc function definitions>>=
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


@ \subsection{Get}

When the address of the associated derivative object is desired, a get
function can be called with the address of the active variable as the
parameter.  When the variable address does not exist in the map, we need
to create a new entry.  Depending on whether eager or lazy allocation
scheme is used, we do not necessarily allocate a new descriptor object.
In fact, we provide two different versions of the get functions: one
version allocates a descriptor object and the other does not.  In both
cases, an appropriate entry is created.

<<assoc header>>=
    void* ad_map_get(void* key);

<<assoc function definitions>>=
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
        MapEntry*	entry = map + <<hash value>>;
	if (entry->cache && entry->cache->key == key) {
	    return entry->cache->val;
	}
        while (1) {
	    int 	i = 0;
	    pa = entry->next;
	    while (++i < entries_per_bucket) {

@	        /*go through the bucket for the match or an empty slot.
                  the last entry of the bucket is used as a link to
                  next block.*/

<<assoc function definitions>>=
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
@		//go to the next bucket.
<<assoc function definitions>>=
		pa = (Pair*)pa->key;
	    }
	    else {
@		/*no more bucket. allocate a new bucket.*/

<<assoc function definitions>>=
		Pair* tmp = <<alloc new bucket>>;
		pa->key = tmp;
		tmp->key = key;
		entry->cache = tmp;
		return tmp->val;
	    }
	}
    }


<<hash value>>=
    (((int)key>>3) % map_size)

<<alloc new bucket>>=
    (Pair*)ad_map_alloc_bucket()


@ Buckets are linked together whose head
is stored in blockList.  If buckets are freed, they are chained 
to freeList.  Each block contains a pointer in the beginning which
contains the link to the next block.

<<assoc header>>=
    static void* ad_map_alloc_bucket();

<<assoc function definitions>>=
    static void* ad_map_alloc_bucket()
    {
#if defined(DEBUG)
	static 	count = 0;
	    if (++count >= gdebug.nTokens) {
		msg("Allocated %d tokens", count);
		count = 0;
	    }
	}
#endif
	static int	nBlocks;
        static int	nCurBucket;

	if (!curBlock || nCurBucket >= buckets_per_block) {
	    if (freeList) {
@	        //check the freelist and allocate from it if available
<<assoc function definitions>>=
	        curBlock = freeList;
	        freeList = freeList->next;
	    }
	    else {
@	        //allocate a new block and add to a linked list.
The figure: sizeof(genlist_t) - sizeof(double) accounts for the
alignment problem.

<<assoc function definitions>>=
		curBlock = (genlist_t*)calloc(
                                sizeof(genlist_t) - sizeof(double) + 
				           buckets_per_block * bucket_size, 1);
	        curBlock->next = blockList;
		blockList = curBlock;
                nBlocks++;
            }
	    nCurBucket = 0;
	}
	return (char*)curBlock->data + (nCurBucket++)*bucket_size;
    }


<<assoc header>>=
   void* ad_map_free_bucket(void* ptr);

<<assoc function definitions>>= 
   void* ad_map_free_bucket(void* ptr)
   {
@    //debugging
<<assoc function definitions>>=
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



@ \section{Free Map Entry}

<<assoc header>>=

void* ad_map_free(void* key);

<<assoc function definitions>>=

void* ad_map_free(void* key)
{
    void** p = (void**)ad_map_get(key);
    *(p-1) = (void*)1;
}


@ \section{Miscellaneous}

<<assoc macros>>=






@ \section{Structure}

The overall structure for the source files are outlined below:

<<run-map.h>>=
#if !defined(RUN_MAP_H)
#define RUN_MAP_H

    <<assoc macros>>
    <<assoc header>>

#endif /*RUN_MAP_H*/


<<run-map.c>>=
    #include <malloc.h>
    #include "run-map.h"
    <<assoc function definitions>>


