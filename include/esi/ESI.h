#ifndef __ESI_h_seen
#define __ESI_h_seen

/* ESI.h for C++
 * A library header for the equation solver interface.
 *
 */

// cruft
#include "../esi/basicTypes.h"
#include "../esi/ordinalTraits.h"
#include "../esi/scalarTraits.h"
#include "../esi/Argv.h"

//
// Following are some #defines that will provide bool support
// when using a compiler that doesn't have a native 'bool' type.
//
// These defines can be turned on explicitly by supplying
// ' -DBOOL_NOT_SUPPORTED ' on the compile line if necessary.
//

#if defined(__SUNPRO_CC) && __SUNPRO_CC < 0x500
//SUNWspro 4.2 C++ compiler doesn't have 'bool'.
#define BOOL_NOT_SUPPORTED
#endif

#ifdef BOOL_NOT_SUPPORTED

#ifdef bool
#undef bool
#endif
#ifdef true
#undef true
#endif
#ifdef false
#undef false
#endif

#define bool int
#define true 1
#define false 0

#endif

// core ESI interfaces
#include "../esi/Object.h"

#include "../esi/IndexSpace.h"

#include "../esi/Vector.h"
#include "../esi/VectorReplaceAccess.h"

#include "../esi/Operator.h"
#include "../esi/OperatorTranspose.h"

#include "../esi/MatrixData.h"
#include "../esi/MatrixRowReadAccess.h"
#include "../esi/MatrixRowWriteAccess.h"
#include "../esi/MatrixRowPointerAccess.h"

#include "../esi/Preconditioner.h"
#include "../esi/PreconditionerTranspose.h"

#include "../esi/Solver.h"
#include "../esi/SolverIterative.h"

#endif /* __ESI_h_seen */
