#ifndef PETSC_CPP_CRTP_HPP
#define PETSC_CPP_CRTP_HPP

#if defined(__cplusplus)

namespace Petsc
{

namespace util
{

// A useful crtp helper class to abstract away all the static_cast<Derived *>(this) nonsense
template <typename T, template <typename> class CRTPType>
class crtp {
protected:
  T       &underlying() noexcept { return static_cast<T &>(*this); }
  const T &underlying() const noexcept { return static_cast<const T &>(*this); }

private:
  // private constructor + friend decl preempts any diamond dependency problems
  // https://www.fluentcpp.com/2017/05/19/crtp-helper/
  constexpr crtp() noexcept = default;
  friend CRTPType<T>;
};

} // namespace util

} // namespace Petsc

#endif // __cplusplus

#endif // PETSC_CPP_CRTP_HPP
