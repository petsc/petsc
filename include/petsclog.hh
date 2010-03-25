#if !defined(__PETSCLOG_HH)
#define __PETSCLOG_HH

#if defined(PETSC_CLANGUAGE_CXX) && !defined(PETSC_USE_EXTERN_CXX)
#include<map>

namespace PETSc {
  class LogStage {
  protected:
    std::string   name;
    PetscLogEvent id;
  public:
    LogStage() : name(), id() {};
    LogStage(const std::string& name, PetscLogStage id) : name(name), id(id) {};
    LogStage(const LogStage& stage) : name(stage.name), id(stage.id) {};
    void push() {
      PetscErrorCode ierr = PetscLogStagePush(this->id); CHKERRXX(ierr);
    };
    void pop() {
      PetscErrorCode ierr = PetscLogStagePop(); CHKERRXX(ierr);
    };
    PetscLogEvent getId() {return this->id;};
  };

  class LogEvent {
  protected:
    std::string   name;
    PetscLogEvent id;
  public:
    LogEvent() : name(), id() {};
    LogEvent(const std::string& name, PetscLogEvent id) : name(name), id(id) {};
    LogEvent(const LogEvent& event) : name(event.name), id(event.id) {};
    void begin(PetscObject o1 = PETSC_NULL, PetscObject o2 = PETSC_NULL, PetscObject o3 = PETSC_NULL, PetscObject o4 = PETSC_NULL) {
      PetscErrorCode ierr = PetscLogEventBegin(this->id, o1, o2, o3, o4); CHKERRXX(ierr);
    };
    void end(PetscObject o1 = PETSC_NULL, PetscObject o2 = PETSC_NULL, PetscObject o3 = PETSC_NULL, PetscObject o4 = PETSC_NULL) {
      PetscErrorCode ierr = PetscLogEventEnd(this->id, o1, o2, o3, o4); CHKERRXX(ierr);
    };
    void barrierBegin(PetscObject o1 = PETSC_NULL, PetscObject o2 = PETSC_NULL, PetscObject o3 = PETSC_NULL, PetscObject o4 = PETSC_NULL, MPI_Comm comm = MPI_COMM_NULL) {
      if (comm == MPI_COMM_NULL) comm = PETSC_COMM_WORLD;
      PetscErrorCode ierr = PetscLogEventBarrierBegin(this->id, o1, o2, o3, o4, comm); CHKERRXX(ierr);
    };
    void barrierEnd(PetscObject o1 = PETSC_NULL, PetscObject o2 = PETSC_NULL, PetscObject o3 = PETSC_NULL, PetscObject o4 = PETSC_NULL, MPI_Comm comm = MPI_COMM_NULL) {
      if (comm == MPI_COMM_NULL) comm = PETSC_COMM_WORLD;
      PetscErrorCode ierr = PetscLogEventBarrierEnd(this->id, o1, o2, o3, o4, comm); CHKERRXX(ierr);
    };
    PetscLogEvent getId() {return this->id;};
  };

  class Log {
  public:
    static std::map<std::string,LogEvent> event_registry;
    static std::map<std::string,LogStage> stage_registry;

    static LogEvent& Event(const std::string& name, PetscCookie cookie = PETSC_OBJECT_COOKIE) {
      if (event_registry.find(name) == event_registry.end()) {
        PetscLogEvent  id;
        PetscErrorCode ierr;

        /* Should check for already registered events */
        ierr = PetscLogEventRegister(name.c_str(), cookie, &id);CHKERRXX(ierr);
        event_registry[name] = LogEvent(name, id);
      }
      return event_registry[name];
    };

    static LogStage& Stage(const std::string& name) {
      if (stage_registry.find(name) == stage_registry.end()) {
        PetscLogStage  id;
        PetscErrorCode ierr;

        /* Should check for already registered stages */
        ierr = PetscLogStageRegister(name.c_str(), &id);CHKERRXX(ierr);
        stage_registry[name] = LogStage(name, id);
      }
      return stage_registry[name];
    };
  };
}
#endif /* PETSC_CLANGUAGE_CXX */

#endif /* __PETSCLOG_HH */
