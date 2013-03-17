#ifndef included_ALE_args_hh
#define included_ALE_args_hh
// This should be included indirectly -- only by including ALE.hh

#include <vector>
#include <string>
#include <boost/program_options.hpp>



namespace ALE {
  //
  struct AnyArg {
    typedef ::boost::program_options::value_semantic value_semantic;
    typedef value_semantic*                          value_semantic_ptr;
    // cast operator
    virtual operator value_semantic_ptr() const = 0;
    virtual ~AnyArg(){};
  };
  //
  // Arg<T> is the type of object that can be added to ArgDB and
  // ultimately holds an argument of type T.
  //
  template<typename T>
  struct Arg : public AnyArg {
    typedef typename AnyArg::value_semantic value_semantic;
    typedef value_semantic*                 value_semantic_ptr;
  protected:
    ::boost::program_options::typed_value<T>* _dtor;
  public:
    Arg(T* storage = NULL) : _dtor(new ::boost::program_options::typed_value<T>(storage)){};
    virtual ~Arg() {} // we do not delete _dtor since it's destroyed
    //when the ::boost::program_options::options_description container is destroyed
    //
    // cast operator
    virtual operator value_semantic_ptr() const {
      return this->_dtor;
    };
    // forwarding methods
    Arg& DEFAULT(const T& v) {
      this->_dtor->default_value(v);
      return *this;
    };
    Arg& IS_MULTIPLACED() {// may be defined in multiple places on the command line
      this->_dtor->composing();
      return *this;
    };
    Arg& IS_A_FLAG() { // no value expected
      this->_dtor->zero_token();
      return *this;
    };
    Arg& IS_A_LIST() {// multiple tokens per value
      this->_dtor->multi_token();
      return *this;
    };
  };// struct Arg
  //
  // The return type of ArgDB dereference:
  //   ArgValue val = argDB["arg"];
  // ArgValue val can be cast to the type compatible with Arg<T>,
  // if the following description of "arg" was used:
  //   argDB("arg", "arg help", Arg<T>);
  //
  struct ArgValue : ::boost::program_options::variable_value {
    typedef ::boost::program_options::variable_value super;
  public:
    ArgValue(const super& val) : super(val) {};
    // cast
    template<typename T>
    operator const T&() {
      return super::as<T>();
    }
    //
    template<typename T>
    operator T& () {
      return super::as<T>();
    }
  };// struct ArgValue

    //
  class ArgDB : public ::boost::program_options::variables_map {
  protected:
    typedef ::boost::program_options::variables_map super;
    string _name;
    ALE::Obj< ::boost::program_options::options_description> _descs;
  public:
    // Basic
    ArgDB(const string& name)                        :
      _name(name), _descs(new ::boost::program_options::options_description(name))
    {};
    //
    ArgDB(const ArgDB& argDB, int argc, char **argv) :
      _name(argDB.name()),_descs(new ::boost::program_options::options_description(_name))
    {
      (*this)(argDB);
      this->parse(argc,argv);
    };
    // Printing
    friend std::ostream& operator<<(std::ostream& os, const ArgDB& argDB) {
      os << *(argDB._descs) << "\n";
      return os;
    }
    // Main
    //
    ArgDB& operator()(const ArgDB& argDB) {
      this->_descs->add(*(argDB._descs));
      return *this;
    };
    //
    ArgDB& operator()(const string& name, const string& helpLine) {
      this->_descs->add_options()(name.c_str(), helpLine.c_str());
      return *this;
    };
    ArgDB& operator()(const string& name, const string& helpLine, const AnyArg& descriptor) {
      this->_descs->add_options()(name.c_str(), descriptor, helpLine.c_str());
      return *this;
    };
    ArgDB& operator()(const string& name, const AnyArg& descriptor) {
      this->_descs->add_options()(name.c_str(), descriptor);
      return *this;
    };
    //
    ArgDB& parse(int argc, char **argv) {
      ::boost::program_options::basic_command_line_parser<char> parser(argc, argv);
#if BOOST_VERSION >= 103300   // works beginning from Boost V1.33.0
      parser.allow_unregistered().options(*(this->_descs));
#endif
      ::boost::program_options::store(parser.run(), *this);
      return *this;
    };
    //
    ArgValue operator[](const string& str) const {return super::operator[](str);};
    //
    // Aux
    //
    const string& name() const {return this->_name;};
    //
    ArgDB& rename(const string& name) {
      this->_name = name;
      Obj< ::boost::program_options::options_description> tmp = this->_descs;
      this->_descs = new ::boost::program_options::options_description(name);
      this->_descs->add(tmp);
      return *this;
    };
  };// class ArgDB

} // namespace ALE


#endif
