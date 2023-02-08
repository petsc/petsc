#ifndef IMPLDEVICEBASE_HPP
#define IMPLDEVICEBASE_HPP

#if defined(__cplusplus)
  #include <petsc/private/deviceimpl.h>
  #include <petsc/private/viewerimpl.h>

  #include <petsc/private/cpp/crtp.hpp>
  #include <petsc/private/cpp/type_traits.hpp>
  #include <petsc/private/cpp/utility.hpp>
  #include <petsc/private/cpp/array.hpp>

  #include <cstring> // for std::strlen

namespace Petsc
{

namespace device
{

namespace impl
{

template <typename Derived> // CRTP
class DeviceBase : public util::crtp<DeviceBase, Derived> {
public:
  using derived_type            = Derived;
  using createContextFunction_t = PetscErrorCode (*)(PetscDeviceContext);

  // default constructor
  constexpr DeviceBase(createContextFunction_t f) noexcept : create_(f) { }

  template <typename T = derived_type>
  PETSC_NODISCARD static constexpr PetscDeviceType PETSC_DEVICE_IMPL() noexcept
  {
    return T::PETSC_DEVICE_IMPL_();
  }

  PetscErrorCode        getDevice(PetscDevice, PetscInt) noexcept;
  static PetscErrorCode configureDevice(PetscDevice) noexcept;
  static PetscErrorCode viewDevice(PetscDevice, PetscViewer) noexcept;
  static PetscErrorCode getAttribute(PetscDevice, PetscDeviceAttribute, void *) noexcept;

protected:
  // function to create a PetscDeviceContext (the (*create) function pointer usually set
  // via XXXSetType() for other PETSc objects)
  const createContextFunction_t create_;

  // if you want the base class to handle the entire options query, has the same arguments as
  // PetscOptionDeviceBasic
  static PetscErrorCode PetscOptionDeviceAll(MPI_Comm, std::pair<PetscDeviceInitType, PetscBool> &, std::pair<PetscInt, PetscBool> &, std::pair<PetscBool, PetscBool> &) noexcept;

  // if you want to start and end the options query yourself, but still want all the default
  // options
  static PetscErrorCode PetscOptionDeviceBasic(PetscOptionItems *, std::pair<PetscDeviceInitType, PetscBool> &, std::pair<PetscInt, PetscBool> &, std::pair<PetscBool, PetscBool> &) noexcept;

  // option templates to follow, each one has two forms:
  // - A simple form returning only the value and flag. This gives no control over the message,
  //   arguments to the options query or otherwise
  // - A complex form, which allows you to pass most of the options query arguments *EXCEPT*
  //   - The options query function called
  //   - The option string

  // option template for initializing the device
  static PetscErrorCode PetscOptionDeviceInitialize(PetscOptionItems *, PetscDeviceInitType *, PetscBool *) noexcept;
  template <typename... T, util::enable_if_t<sizeof...(T) >= 3, int> = 0>
  static PetscErrorCode PetscOptionDeviceInitialize(PetscOptionItems *, T &&...) noexcept;
  // option template for selecting the default device
  static PetscErrorCode PetscOptionDeviceSelect(PetscOptionItems *, PetscInt *, PetscBool *) noexcept;
  template <typename... T, util::enable_if_t<sizeof...(T) >= 3, int> = 0>
  static PetscErrorCode PetscOptionDeviceSelect(PetscOptionItems *, T &&...) noexcept;
  // option templates for viewing a device
  static PetscErrorCode PetscOptionDeviceView(PetscOptionItems *, PetscBool *, PetscBool *) noexcept;
  template <typename... T, util::enable_if_t<sizeof...(T) >= 3, int> = 0>
  static PetscErrorCode PetscOptionDeviceView(PetscOptionItems *, T &&...) noexcept;

private:
  // base function for all options templates above, they basically just reformat the arguments,
  // create the option string and pass it off to this function
  template <typename... T, typename F = PetscErrorCode (*)(PetscOptionItems *, const char *, T &&...)>
  static PetscErrorCode PetscOptionDevice(F &&, PetscOptionItems *, const char[], T &&...) noexcept;

  // default crtp implementations
  static PetscErrorCode init_device_id_(PetscInt *id) noexcept
  {
    PetscFunctionBegin;
    *id = 0;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  static constexpr PetscErrorCode configure_device_(PetscDevice) noexcept { return PETSC_SUCCESS; }
  static constexpr PetscErrorCode view_device_(PetscDevice, PetscViewer) noexcept { return PETSC_SUCCESS; }
};

template <typename D>
inline PetscErrorCode DeviceBase<D>::getDevice(PetscDevice device, PetscInt id) noexcept
{
  PetscFunctionBegin;
  PetscCall(this->underlying().init_device_id_(&id));
  device->deviceId           = id;
  device->ops->createcontext = this->underlying().create_;
  device->ops->configure     = this->underlying().configureDevice;
  device->ops->view          = this->underlying().viewDevice;
  device->ops->getattribute  = this->underlying().getAttribute;
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <typename D>
inline PetscErrorCode DeviceBase<D>::configureDevice(PetscDevice device) noexcept
{
  PetscFunctionBegin;
  PetscCall(derived_type::configure_device_(device));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <typename D>
inline PetscErrorCode DeviceBase<D>::viewDevice(PetscDevice device, PetscViewer viewer) noexcept
{
  PetscFunctionBegin;
  PetscCall(derived_type::view_device_(device, viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <typename D>
inline PetscErrorCode DeviceBase<D>::getAttribute(PetscDevice device, PetscDeviceAttribute attr, void *value) noexcept
{
  PetscFunctionBegin;
  PetscCall(derived_type::get_attribute_(device->deviceId, attr, value));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <typename D>
template <typename... T, typename F>
inline PetscErrorCode DeviceBase<D>::PetscOptionDevice(F &&OptionsFunction, PetscOptionItems *PetscOptionsObject, const char optstub[], T &&...args) noexcept
{
  constexpr auto dtype    = PETSC_DEVICE_IMPL();
  const auto     implname = PetscDeviceTypes[dtype];
  auto           buf      = std::array<char, 128>{};
  constexpr auto buflen   = buf.size() - 1;

  PetscFunctionBegin;
  if (PetscDefined(USE_DEBUG)) {
    const auto len = std::strlen(optstub) + std::strlen(implname);

    PetscCheck(len < buflen, PetscOptionsObject->comm, PETSC_ERR_PLIB, "char buffer is not large enough to hold '%s%s'; have %zu need %zu", optstub, implname, buflen, len);
  }
  PetscCall(PetscSNPrintf(buf.data(), buflen, "%s%s", optstub, implname));
  PetscCall(OptionsFunction(PetscOptionsObject, buf.data(), std::forward<T>(args)...));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <typename D>
template <typename... T, util::enable_if_t<sizeof...(T) >= 3, int>>
inline PetscErrorCode DeviceBase<D>::PetscOptionDeviceInitialize(PetscOptionItems *PetscOptionsObject, T &&...args) noexcept
{
  PetscFunctionBegin;
  PetscCall(PetscOptionDevice(PetscOptionsEList_Private, PetscOptionsObject, "-device_enable_", std::forward<T>(args)...));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <typename D>
inline PetscErrorCode DeviceBase<D>::PetscOptionDeviceInitialize(PetscOptionItems *PetscOptionsObject, PetscDeviceInitType *inittype, PetscBool *flag) noexcept
{
  auto type = static_cast<PetscInt>(util::to_underlying(*inittype));

  PetscFunctionBegin;
  PetscCall(PetscOptionDeviceInitialize(PetscOptionsObject, "How (or whether) to initialize a device", "PetscDeviceInitialize()", PetscDeviceInitTypes, 3, PetscDeviceInitTypes[type], &type, flag));
  *inittype = static_cast<PetscDeviceInitType>(type);
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <typename D>
template <typename... T, util::enable_if_t<sizeof...(T) >= 3, int>>
inline PetscErrorCode DeviceBase<D>::PetscOptionDeviceSelect(PetscOptionItems *PetscOptionsObject, T &&...args) noexcept
{
  PetscFunctionBegin;
  PetscCall(PetscOptionDevice(PetscOptionsInt_Private, PetscOptionsObject, "-device_select_", std::forward<T>(args)...));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <typename D>
inline PetscErrorCode DeviceBase<D>::PetscOptionDeviceSelect(PetscOptionItems *PetscOptionsObject, PetscInt *id, PetscBool *flag) noexcept
{
  PetscFunctionBegin;
  PetscCall(PetscOptionDeviceSelect(PetscOptionsObject, "Which device to use. Pass " PetscStringize(PETSC_DECIDE) " to have PETSc decide or (given they exist) [0-" PetscStringize(PETSC_DEVICE_MAX_DEVICES) ") for a specific device", "PetscDeviceCreate()", *id, id, flag, PETSC_DECIDE, PETSC_DEVICE_MAX_DEVICES));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <typename D>
template <typename... T, util::enable_if_t<sizeof...(T) >= 3, int>>
inline PetscErrorCode DeviceBase<D>::PetscOptionDeviceView(PetscOptionItems *PetscOptionsObject, T &&...args) noexcept
{
  PetscFunctionBegin;
  PetscCall(PetscOptionDevice(PetscOptionsBool_Private, PetscOptionsObject, "-device_view_", std::forward<T>(args)...));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <typename D>
inline PetscErrorCode DeviceBase<D>::PetscOptionDeviceView(PetscOptionItems *PetscOptionsObject, PetscBool *view, PetscBool *flag) noexcept
{
  PetscFunctionBegin;
  PetscCall(PetscOptionDeviceView(PetscOptionsObject, "Display device information and assignments (forces eager initialization)", "PetscDeviceView()", *view, view, flag));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <typename D>
inline PetscErrorCode DeviceBase<D>::PetscOptionDeviceBasic(PetscOptionItems *PetscOptionsObject, std::pair<PetscDeviceInitType, PetscBool> &initType, std::pair<PetscInt, PetscBool> &initId, std::pair<PetscBool, PetscBool> &initView) noexcept
{
  PetscFunctionBegin;
  PetscCall(PetscOptionDeviceInitialize(PetscOptionsObject, &initType.first, &initType.second));
  PetscCall(PetscOptionDeviceSelect(PetscOptionsObject, &initId.first, &initId.second));
  PetscCall(PetscOptionDeviceView(PetscOptionsObject, &initView.first, &initView.second));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <typename D>
inline PetscErrorCode DeviceBase<D>::PetscOptionDeviceAll(MPI_Comm comm, std::pair<PetscDeviceInitType, PetscBool> &initType, std::pair<PetscInt, PetscBool> &initId, std::pair<PetscBool, PetscBool> &initView) noexcept
{
  constexpr char optname[] = "PetscDevice %s Options";
  constexpr auto dtype     = PETSC_DEVICE_IMPL();
  const auto     implname  = PetscDeviceTypes[dtype];
  auto           buf       = std::array<char, 128>{};
  constexpr auto buflen    = buf.size() - 1; // -1 to leave room for null

  PetscFunctionBegin;
  if (PetscDefined(USE_DEBUG)) {
    // -3 since '%s' is replaced and dont count null char for optname
    const auto len = std::strlen(implname) + PETSC_STATIC_ARRAY_LENGTH(optname) - 3;

    PetscCheck(len < buflen, comm, PETSC_ERR_PLIB, "char buffer is not large enough to hold 'PetscDevice %s Options'; have %zu need %zu", implname, buflen, len);
  }
  PetscCall(PetscSNPrintf(buf.data(), buflen, optname, implname));
  PetscOptionsBegin(comm, nullptr, buf.data(), "Sys");
  PetscCall(PetscOptionDeviceBasic(PetscOptionsObject, initType, initId, initView));
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace impl

} // namespace device

} // namespace Petsc

  #define PETSC_DEVICE_IMPL_BASE_CLASS_HEADER(base_name, T) \
    using base_name = ::Petsc::device::impl::DeviceBase<T>; \
    friend base_name; \
    using base_name::base_name

#endif // __cplusplus

#endif // IMPLDEVICEBASE_HPP
