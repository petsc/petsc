#if !defined(RUN_MAP_H)
#define RUN_MAP_H

#if defined(__cplusplus)
PETSC_EXTERN "C" {
#endif

    static int desc_size = 0;
    static int entry_size = 0;
    static int bucket_size = 0;
    static int entries_per_bucket = 0;
    static int buckets_per_block = 0;
    static int map_size = 0;

#   define DEFAULT_MAP_SIZE		1000
#   define DEFAULT_BUCKET_SIZE		10
#   define DEFAULT_BUCKETS_PER_BLOCK	100
    typedef struct {
        void* key;
        double val[1];
    } Pair;
    typedef struct {
        Pair* cache;
        Pair* next;
    } MapEntry;
    static MapEntry* map_def = 0;

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


    void* ad_map_init(int dsize, int msize, int bsize, int asize);
    void ad_map_cleanup();
    void* ad_map_reg_array_d(double* base, int size);
    void* ad_map_reg_array_s(float* base, int size);
    void* ad_map_get(void* key);
    static void* ad_map_alloc_bucket(void);
    void* ad_map_free_bucket(void* ptr);
    void* ad_map_free(void* key);

#if defined(__cplusplus)
}
#endif

#endif /*RUN_MAP_H*/


