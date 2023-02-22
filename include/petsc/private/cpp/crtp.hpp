#ifndef PETSC_CPP_CRTP_HPP
#define PETSC_CPP_CRTP_HPP

#if defined(__cplusplus)

namespace Petsc
{

namespace util
{

// A useful crtp helper class to abstract away all the static_cast<Derived *>(this) nonsense
template <template <typename, typename...> class CRTPType, typename Derived, typename... T>
class crtp {
protected:
  Derived       &underlying() noexcept { return static_cast<Derived &>(*this); }
  const Derived &underlying() const noexcept { return static_cast<const Derived &>(*this); }

private:
  // private constructor + friend decl preempts any diamond dependency problems
  // https://www.fluentcpp.com/2017/05/19/crtp-helper/
  constexpr crtp() noexcept = default;
  friend CRTPType<Derived, T...>;
};

} // namespace util

} // namespace Petsc

#endif // __cplusplus

#endif // PETSC_CPP_CRTP_HPP
