static char help[] = "Overlap Tests.\n\n";

#include <petscdmmesh.h>
//#include "overlapTest.hh"

using ALE::Obj;

typedef struct {
  int debug; // The debugging level
} Options;

#undef __FUNCT__
#define __FUNCT__ "InsertionTest"
PetscErrorCode InsertionTest(Options *options)
{
  typedef PetscInt point_type;
  typedef short    rank_type;
  typedef PETSc::SendOverlap<point_type,rank_type> send_overlap_type;
  MPI_Comm          comm          = PETSC_COMM_WORLD;
  const char       *stageName     = "Overlap Insertion Test";
  const char       *eventName     = "Insert";
  ALE::LogStage     stage         = ALE::LogStageRegister(stageName);
  const PetscInt    insertRounds  = 6;
  const PetscInt    baseExp       = 2;
  const PetscInt    baseInserts   = pow(10, baseExp);
  PetscLogEvent     insertEvents[insertRounds];
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  // Test fast assembly
  for(PetscInt i = 0; i < insertRounds; ++i) {
    char name[1024];

    ierr = PetscSNPrintf(name, 1023, "%sE%d", eventName, i+baseExp);CHKERRQ(ierr);
    ierr = PetscLogEventRegister(name, PETSC_OBJECT_CLASSID, &insertEvents[i]);CHKERRQ(ierr);
  }
  ALE::LogStagePush(stage);
  // Use some number of ranks, start with 2
  for(PetscInt numRanks = 2; numRanks < 3; ++numRanks) {
    for(PetscInt i = 0, numInsertions = baseInserts; i < insertRounds; ++i, numInsertions *= 10) {
      send_overlap_type sendOverlap(comm, options->debug);

      sendOverlap.setNumRanks(numRanks);
      ierr = PetscPrintf(comm, "Insertion Round %d for %d points\n", i, numInsertions);
      for(rank_type rank = 0; rank < numRanks; ++rank) {
        sendOverlap.setNumPoints(rank, numInsertions);
      }
      sendOverlap.assemble();
      // Insert some number of points per rank, look at time/insertion, I think its growing quadratically
      ierr = PetscLogEventBegin(insertEvents[i],0,0,0,0);CHKERRQ(ierr);
      for(rank_type rank = 0; rank < numRanks; ++rank) {
        for(PetscInt p = 0; p < numInsertions; ++p) {
          const point_type localPoint  = p;
          const point_type remotePoint = numInsertions - p - 1;

          sendOverlap.addArrow(localPoint, rank, remotePoint);
        }
      }
      sendOverlap.assemblePoints();
      ierr = PetscLogEventEnd(insertEvents[i],0,0,0,0);CHKERRQ(ierr);
    }
  }
  ALE::LogStagePop(stage);
  PetscStageLog     stageLog;
  PetscEventPerfLog eventLog;

  ierr = PetscLogGetStageLog(&stageLog);
  ierr = PetscStageLogGetEventPerfLog(stageLog, stage, &eventLog);
  for(PetscInt i = 0, numInsertions = baseInserts; i < insertRounds; ++i, numInsertions *= 10) {
    PetscEventPerfInfo eventInfo = eventLog->eventInfo[insertEvents[i]];

    //CPPUNIT_ASSERT_EQUAL(eventInfo.count, 1);
    //CPPUNIT_ASSERT_EQUAL((int) eventInfo.flops, 0);
    if (options->debug) {
      ierr = PetscPrintf(comm, " %d: Average time per insertion for 1e%d points: %gs\n", eventInfo.count, i+baseExp, eventInfo.time/(numInsertions));
    }
    //CPPUNIT_ASSERT((eventInfo.time <  maxTimePerInsertion * numInsertions));
  }
  ierr = PetscPrintf(comm, "times = [");
  for(PetscInt i = 0, numInsertions = baseInserts; i < insertRounds; ++i, numInsertions *= 10) {
    PetscEventPerfInfo eventInfo = eventLog->eventInfo[insertEvents[i]];
    ierr = PetscPrintf(comm, "%g, ", eventInfo.time/numInsertions);
  }
  ierr = PetscPrintf(comm, "]\n");
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ProcessOptions"
PetscErrorCode ProcessOptions(MPI_Comm comm, Options *options)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  options->debug = 0;

  ierr = PetscOptionsBegin(comm, "", "Options for overlap stress test", "Sieve");CHKERRQ(ierr);
    ierr = PetscOptionsInt("-debug", "The debugging level", "overlap1.c", options->debug, &options->debug, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char *argv[])
{
  MPI_Comm       comm;
  Options        options;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscInitialize(&argc, &argv, (char *) 0, help);CHKERRQ(ierr);
  comm = PETSC_COMM_WORLD;
  ierr = PetscLogBegin();CHKERRQ(ierr);
  ierr = ProcessOptions(comm, &options);CHKERRQ(ierr);
  try {
    ierr = InsertionTest(&options);CHKERRQ(ierr);
  } catch (ALE::Exception e) {
    std::cout << e << std::endl;
  }
  ierr = PetscFinalize();
  PetscFunctionReturn(0);
}
