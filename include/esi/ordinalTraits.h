#ifndef __ESI_ordinalTraits_h
#define __ESI_ordinalTraits_h

namespace esi {

template<class T>
struct ordinalTraits {
   static inline const char* name() {
     cout << "esi::ordinalTraits: unsupported ordinal type." << endl; abort(); 
     return(NULL);
   };
};

template<>
struct ordinalTraits<int4> {
   typedef int4 ordinal_type;
   static inline const char* name() { return("esi::int4"); };
};

template<>
struct ordinalTraits<int8> {
   typedef int8 ordinal_type;
   static inline const char* name() {return("esi::int8");};
};


#if defined(__SUNPRO_CC) && __SUNPRO_CC < 0x500
#define TYPENAME
#else
#define TYPENAME typename
#endif

};     // esi namespace
#endif

