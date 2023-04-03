#ifndef PETSC_CPP_UNORDERED_MAP_HPP
#define PETSC_CPP_UNORDERED_MAP_HPP

#if defined(__cplusplus)
  #include <petsc/private/cpp/type_traits.hpp>
  #include <petsc/private/cpp/utility.hpp>    // std ::pair
  #include <petsc/private/cpp/functional.hpp> // std::hash, std::equal_to

  #include <unordered_map>

  #include <cstdint>   // std::uint32_t
  #include <climits>   // CHAR_BIT
  #include <iterator>  // std::inserter
  #include <limits>    // std::numeric_limits
  #include <algorithm> // std::fill
  #include <vector>

namespace Petsc
{

namespace khash
{

// ==========================================================================================
// KHashTable - The hash table implementation which underpins UnorderedMap (and possibly
// UnorderedSet in the future).
//
// This class serves to implement the majority of functionality for both classes. In fact, it
// is possible to use -- without modification -- as a khash_unordered_set already.
//
// Template parameters are as follows:
//
// Value:
// The value type of the hash table, i.e. the set of unique items it stores
//
// Hash:
// The hasher type, provides a std::size_t operator()(const Value&) to produce a hash of Value
//
// Eq:
// The comparison type, provides a bool operator()(const Value&, const Value&) to compare two
// different Value's
// ==========================================================================================

template <typename Value, typename Hash, typename Eq>
class KHashTable : util::compressed_pair<Hash, Eq> {
  // Note we derive from compressed_pair<Hash, Eq>! This is to enable us to efficiently
  // implement hash_function() and key_eq() since -- if Hash and Eq are empty -- we do not have
  // to pay to store them due to empty base-class optimization.
  template <bool>
  class table_iterator;

  template <bool>
  friend class table_iterator;

public:
  using value_type      = Value;
  using hasher          = Hash;
  using key_equal       = Eq;
  using size_type       = std::size_t;
  using reference       = value_type &;
  using const_reference = const value_type &;
  using difference_type = std::ptrdiff_t;
  using iterator        = table_iterator</* is const = */ false>;
  using const_iterator  = table_iterator</* is const = */ true>;
  using khash_int       = std::uint32_t; // used as the internal iterator type
  using flags_type      = std::uint32_t;

  using flag_bucket_width     = std::integral_constant<unsigned long, sizeof(flags_type) * CHAR_BIT>;
  using flag_pairs_per_bucket = std::integral_constant<unsigned long, flag_bucket_width::value / 2>;

  static_assert(std::numeric_limits<flags_type>::is_integer && std::is_unsigned<flags_type>::value, "");
  static_assert(flag_bucket_width::value % 2 == 0, "");

  KHashTable()  = default;
  ~KHashTable() = default;

  KHashTable(const KHashTable &)            = default;
  KHashTable &operator=(const KHashTable &) = default;

  KHashTable(KHashTable &&) noexcept;
  KHashTable &operator=(KHashTable &&) noexcept;

  template <typename Iter>
  KHashTable(Iter, Iter) noexcept;

  PETSC_NODISCARD iterator       begin() noexcept;
  PETSC_NODISCARD const_iterator cbegin() const noexcept;
  PETSC_NODISCARD const_iterator begin() const noexcept;

  PETSC_NODISCARD iterator       end() noexcept;
  PETSC_NODISCARD const_iterator cend() const noexcept;
  PETSC_NODISCARD const_iterator end() const noexcept;

  PETSC_NODISCARD size_type bucket_count() const noexcept;
  PETSC_NODISCARD size_type size() const noexcept;
  PETSC_NODISCARD size_type capacity() const noexcept;
  PETSC_NODISCARD bool      empty() const noexcept;

  PetscErrorCode reserve(size_type) noexcept;
  PetscErrorCode resize(size_type) noexcept;
  PetscErrorCode clear() noexcept;

  PETSC_NODISCARD bool occupied(khash_int) const noexcept;
  PETSC_NODISCARD bool occupied(const_iterator) const noexcept;

  iterator erase(iterator) noexcept;
  iterator erase(const_iterator) noexcept;
  iterator erase(const_iterator, const_iterator) noexcept;

  template <typename... Args>
  std::pair<iterator, bool> emplace(Args &&...) noexcept;

  std::pair<iterator, bool> insert(const value_type &) noexcept;
  std::pair<iterator, bool> insert(value_type &&) noexcept;
  iterator                  insert(const_iterator, const value_type &) noexcept;
  iterator                  insert(const_iterator, value_type &&) noexcept;

  hasher    hash_function() const noexcept;
  key_equal key_eq() const noexcept;

  void swap(KHashTable &) noexcept;

protected:
  PETSC_NODISCARD iterator       make_iterator_(khash_int) noexcept;
  PETSC_NODISCARD const_iterator make_iterator_(khash_int) const noexcept;

  template <typename T>
  PetscErrorCode khash_find_(T &&, khash_int *) const noexcept;

  // emplacement for the hash map, where key and value are constructed separately
  template <typename KeyType, typename... ValueTypeArgs>
  PETSC_NODISCARD std::pair<iterator, bool> find_and_emplace_(KeyType &&, ValueTypeArgs &&...) noexcept;
  template <typename KeyValueType>
  PETSC_NODISCARD std::pair<iterator, bool> find_and_emplace_(KeyValueType &&) noexcept;

private:
  template <typename Iter>
  KHashTable(Iter, Iter, std::input_iterator_tag) noexcept;
  template <typename Iter>
  KHashTable(Iter, Iter, std::random_access_iterator_tag) noexcept;

  // Every element in the table has a pair of 2 flags that describe its current state. These
  // are addressed via the *index* into values_ for the element. These flags are:
  //
  // 1. Empty: has the element *ever* been constructed? Note empty of yes implies deleted no.
  // 2. Deleted: has the element been constructed and marked as deleted?
  //
  // Since these flags are combineable we can store them in compressed form in a bit-table,
  // where each pair of consecutive 2*i and 2*i+1 bits denote the flags for element i.
  //
  // Thus if we use a vector of std::bitset's (which are each N-bits wide) we can effectively
  // store N / 2 flags per index, for example:
  //
  // std::vector<std::bitset<32>> flags;
  //
  // int flags_per_idx = 32 / 2; (16 pairs of flags)
  // int value_index = 24;
  // int flags_index = value_index / flags_per_idx; (index into the right bucket in flags)
  // int flags_bucket_index = (value_index % flags_per_idx) << 1; (within that bucket grab the
  //                                                               right entry)
  //
  // or visually speaking:
  //
  // flags = [
  //   [00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00],
  //   [00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00], <--- flags_index (1)
  //   ...                   ^--- flags_bucket_index (16)
  // ]
  //
  // Thus to access a particular flag pair, one must right-shift flags[flags_index] by
  // flags_bucket_index. Then the desired flag pair will be the first and second bits of the
  // result.
  PETSC_NODISCARD static constexpr khash_int flag_bucket_index_(khash_int) noexcept;

  PETSC_NODISCARD static flags_type       &flag_bucket_at_(khash_int, std::vector<flags_type> &) noexcept;
  PETSC_NODISCARD static const flags_type &flag_bucket_at_(khash_int, const std::vector<flags_type> &) noexcept;
  PETSC_NODISCARD flags_type              &flag_bucket_at_(khash_int) noexcept;
  PETSC_NODISCARD const flags_type        &flag_bucket_at_(khash_int) const noexcept;

  template <unsigned>
  PETSC_NODISCARD static bool khash_test_flag_(khash_int, const std::vector<flags_type> &) noexcept;
  PETSC_NODISCARD static bool khash_is_del_(khash_int, const std::vector<flags_type> &) noexcept;
  PETSC_NODISCARD static bool khash_is_empty_(khash_int, const std::vector<flags_type> &) noexcept;
  PETSC_NODISCARD static bool khash_is_either_(khash_int, const std::vector<flags_type> &) noexcept;
  PETSC_NODISCARD static bool khash_occupied_(khash_int, const std::vector<flags_type> &) noexcept;
  PETSC_NODISCARD bool        khash_is_del_(khash_int) const noexcept;
  PETSC_NODISCARD bool        khash_is_empty_(khash_int) const noexcept;
  PETSC_NODISCARD bool        khash_is_either_(khash_int) const noexcept;

  template <unsigned, bool>
  static PetscErrorCode khash_set_flag_(khash_int, std::vector<flags_type> &) noexcept;
  template <bool>
  static PetscErrorCode khash_set_deleted_(khash_int, std::vector<flags_type> &) noexcept;
  template <bool>
  static PetscErrorCode khash_set_empty_(khash_int, std::vector<flags_type> &) noexcept;
  template <bool>
  static PetscErrorCode khash_set_both_(khash_int, std::vector<flags_type> &) noexcept;
  template <bool>
  PetscErrorCode khash_set_deleted_(khash_int) noexcept;
  template <bool>
  PetscErrorCode khash_set_empty_(khash_int) noexcept;
  template <bool>
  PetscErrorCode khash_set_both_(khash_int) noexcept;

  // produce the default bit pattern:
  //
  //               v--- deleted: no
  // 0b101010 ... 10
  //              ^---- empty: yes
  template <std::size_t mask_width>
  static PETSC_CONSTEXPR_14 flags_type default_bit_pattern_impl_() noexcept
  {
    flags_type x{};

    for (std::size_t i = 0; i < mask_width; ++i) {
      if (i % 2) {
        // odd,
        x |= 1ULL << i;
      } else {
        // even
        x &= ~(1UL << i);
      }
    }
    return x;
  }

public:
  PETSC_NODISCARD static PETSC_CONSTEXPR_14 flags_type default_bit_pattern() noexcept
  {
    // forces constexpr evaluation, which may not be guaranteed. Note that after GCC 6.1+
    // tries to constexpr-evaluate _any_ function marked constexpr and will inline evaluate
    // default_bit_mask_impl_() at any optimization level > 0.
    //
    // clang constexpr evaluates this at 3.7 but is inconsistent between versions at which
    // optimization level the call is fully unraveled.
    PETSC_CONSTEXPR_14 auto ret = default_bit_pattern_impl_<flag_bucket_width::value>();
    return ret;
  }

private:
  template <typename KeyType, typename ValueConstructor>
  PETSC_NODISCARD std::pair<iterator, bool> find_and_emplace_final_(KeyType &&, ValueConstructor &&) noexcept;

  PetscErrorCode khash_maybe_rehash_() noexcept;
  PetscErrorCode khash_erase_(khash_int) noexcept;

  std::vector<value_type> values_{};
  std::vector<flags_type> flags_{};
  size_type               count_       = 0;
  size_type               n_occupied_  = 0;
  size_type               upper_bound_ = 0;
};

// ==========================================================================================
// KHashTable::table_iterator
// ==========================================================================================

template <typename V, typename H, typename KE>
template <bool is_const_it>
class KHashTable<V, H, KE>::table_iterator {
  template <typename U>
  using conditional_const = util::conditional_t<is_const_it, util::add_const_t<U>, U>;

  template <bool>
  friend class table_iterator;

  friend class KHashTable;

public:
  // internal typedef
  using table_type = conditional_const<KHashTable>;
  using khash_int  = typename table_type::khash_int;

  // iterator-related typedefs
  using iterator_category = std::bidirectional_iterator_tag;
  using difference_type   = typename table_type::difference_type;
  using value_type        = conditional_const<typename table_type::value_type>;
  using reference         = value_type &;
  using pointer           = value_type *;

  table_iterator() noexcept = default;
  table_iterator(table_type *map, khash_int it) noexcept : map_(map), it_(it) { }

  table_iterator(const table_iterator &) noexcept            = default;
  table_iterator &operator=(const table_iterator &) noexcept = default;

  table_iterator(table_iterator &&) noexcept            = default;
  table_iterator &operator=(table_iterator &&) noexcept = default;

  template <bool other_is_const_it, util::enable_if_t<is_const_it && !other_is_const_it> * = nullptr>
  table_iterator(const table_iterator<other_is_const_it> &other) noexcept : table_iterator(other.map_, other.it_)
  {
  }

  template <bool other_is_const_it, util::enable_if_t<is_const_it && !other_is_const_it> * = nullptr>
  table_iterator &operator=(const table_iterator<other_is_const_it> &other) noexcept
  {
    // self assignment is OK here
    PetscFunctionBegin;
    map_ = other.map_;
    it_  = other.it_;
    PetscFunctionReturn(*this);
  }

  // prefix
  table_iterator &operator--() noexcept
  {
    constexpr khash_int map_begin = 0;

    PetscFunctionBegin;
    // Use of map_begin + 1 instead of map_begin (like in operator++()) is deliberate. We do
    // not want it_ == map_begin here since that would mean that the while-loop decrements it
    // out of bounds!
    // Likewise we are allowed to be 1 past the bucket size, otherwise backwards iteration
    // would not work!
    PetscCallAbort(PETSC_COMM_SELF, check_iterator_inbounds_(1, 1));
    do {
      --it_;
    } while (it_ > map_begin && !map_->occupied(it_));
    PetscFunctionReturn(*this);
  }

  // postfix
  table_iterator operator--(int) noexcept
  {
    table_iterator old(*this);

    PetscFunctionBegin;
    --(*this);
    PetscFunctionReturn(old);
  }

  // prefix
  table_iterator &operator++() noexcept
  {
    khash_int map_end = 0;

    PetscFunctionBegin;
    PetscCallAbort(PETSC_COMM_SELF, check_iterator_inbounds_());
    map_end = map_->bucket_count();
    do {
      ++it_;
    } while (it_ != map_end && !map_->occupied(it_));
    PetscFunctionReturn(*this);
  }

  // postfix
  table_iterator operator++(int) noexcept
  {
    table_iterator old(*this);

    PetscFunctionBegin;
    ++(*this);
    PetscFunctionReturn(old);
  }

  PETSC_NODISCARD reference operator*() const noexcept
  {
    PetscFunctionBegin;
    PetscCallAbort(PETSC_COMM_SELF, check_iterator_inbounds_());
    PetscFunctionReturn(map_->values_[it_]);
  }

  PETSC_NODISCARD pointer operator->() const noexcept
  {
    PetscFunctionBegin;
    PetscCallAbort(PETSC_COMM_SELF, check_iterator_inbounds_());
    PetscFunctionReturn(std::addressof(map_->values_[it_]));
  }

  template <bool rc>
  PETSC_NODISCARD bool operator==(const table_iterator<rc> &r) const noexcept
  {
    return std::tie(map_, it_) == std::tie(r.map_, r.it_);
  }

  template <bool rc>
  PETSC_NODISCARD bool operator!=(const table_iterator<rc> &r) const noexcept
  {
    return !(*this == r);
  }

private:
  table_type *map_ = nullptr;
  khash_int   it_  = 0;

  PetscErrorCode check_iterator_inbounds_(int map_begin_offset = 0, int map_end_offset = 0) const noexcept
  {
    PetscFunctionBegin;
    if (PetscDefined(USE_DEBUG)) {
      std::int64_t map_begin = map_begin_offset;
      std::int64_t map_end   = map_end_offset;

      PetscCheck(map_, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Iterator has a NULL map pointer");
      map_end += map_->bucket_count();
      PetscCheck((it_ >= map_begin) && (it_ < map_end), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Iterator index value %" PRId32 " is out of range for map (%p): [%" PRId64 ", %" PRId64 ")", it_, (void *)map_, map_begin, map_end);
    } else {
      static_cast<void>(map_begin_offset);
      static_cast<void>(map_end_offset);
    }
    PetscFunctionReturn(PETSC_SUCCESS);
  }
};

// ==========================================================================================
// KHashTable - Private API
// ==========================================================================================

// Generic iterator constructor
template <typename V, typename H, typename KE>
template <typename Iter>
inline KHashTable<V, H, KE>::KHashTable(Iter first, Iter last, std::input_iterator_tag) noexcept
{
  PetscFunctionBegin;
  std::copy(std::move(first), std::move(last), std::inserter(*this, begin()));
  PetscFunctionReturnVoid();
}

// An optimization for random_access_iterators. Since these mandate that std::distance() is
// equivalent to end-begin, we can use this to pre-allocate the hashmap for free before we
// insert
template <typename V, typename H, typename KE>
template <typename Iter>
inline KHashTable<V, H, KE>::KHashTable(Iter first, Iter last, std::random_access_iterator_tag) noexcept
{
  PetscFunctionBegin;
  PetscCallAbort(PETSC_COMM_SELF, reserve(static_cast<size_type>(std::distance(first, last))));
  std::copy(std::move(first), std::move(last), std::inserter(*this, begin()));
  PetscFunctionReturnVoid();
}

// ------------------------------------------------------------------------------------------
// KHashTable - Private API - flag bucket API - accessors
// ------------------------------------------------------------------------------------------

template <typename V, typename H, typename KE>
inline constexpr typename KHashTable<V, H, KE>::khash_int KHashTable<V, H, KE>::flag_bucket_index_(khash_int it) noexcept
{
  return (it % flag_pairs_per_bucket::value) << 1;
}

template <typename V, typename H, typename KE>
inline typename KHashTable<V, H, KE>::flags_type &KHashTable<V, H, KE>::flag_bucket_at_(khash_int it, std::vector<flags_type> &flags) noexcept
{
  return flags[it / flag_pairs_per_bucket::value];
}

template <typename V, typename H, typename KE>
inline const typename KHashTable<V, H, KE>::flags_type &KHashTable<V, H, KE>::flag_bucket_at_(khash_int it, const std::vector<flags_type> &flags) noexcept
{
  return flags[it / flag_pairs_per_bucket::value];
}

template <typename V, typename H, typename KE>
inline typename KHashTable<V, H, KE>::flags_type &KHashTable<V, H, KE>::flag_bucket_at_(khash_int it) noexcept
{
  return flag_bucket_at_(it, flags_);
}

template <typename V, typename H, typename KE>
inline const typename KHashTable<V, H, KE>::flags_type &KHashTable<V, H, KE>::flag_bucket_at_(khash_int it) const noexcept
{
  return flag_bucket_at_(it, flags_);
}

// ------------------------------------------------------------------------------------------
// KHashTable - Private API - flag bucket API - query
// ------------------------------------------------------------------------------------------

template <typename V, typename H, typename KE>
template <unsigned selector>
inline bool KHashTable<V, H, KE>::khash_test_flag_(khash_int it, const std::vector<flags_type> &flags) noexcept
{
  static_assert(selector > 0 || selector <= 3, "");
  return (flag_bucket_at_(it, flags) >> flag_bucket_index_(it)) & selector;
}

template <typename V, typename H, typename KE>
inline bool KHashTable<V, H, KE>::khash_is_del_(khash_int it, const std::vector<flags_type> &flags) noexcept
{
  return khash_test_flag_<1>(it, flags);
}

template <typename V, typename H, typename KE>
inline bool KHashTable<V, H, KE>::khash_is_empty_(khash_int it, const std::vector<flags_type> &flags) noexcept
{
  return khash_test_flag_<2>(it, flags);
}

template <typename V, typename H, typename KE>
inline bool KHashTable<V, H, KE>::khash_is_either_(khash_int it, const std::vector<flags_type> &flags) noexcept
{
  return khash_test_flag_<3>(it, flags);
}

template <typename V, typename H, typename KE>
inline bool KHashTable<V, H, KE>::khash_occupied_(khash_int it, const std::vector<flags_type> &flags) noexcept
{
  return !khash_is_either_(it, flags);
}

template <typename V, typename H, typename KE>
inline bool KHashTable<V, H, KE>::khash_is_del_(khash_int it) const noexcept
{
  return khash_is_del_(it, flags_);
}

template <typename V, typename H, typename KE>
inline bool KHashTable<V, H, KE>::khash_is_empty_(khash_int it) const noexcept
{
  return khash_is_empty_(it, flags_);
}

template <typename V, typename H, typename KE>
inline bool KHashTable<V, H, KE>::khash_is_either_(khash_int it) const noexcept
{
  return khash_is_either_(it, flags_);
}

// ------------------------------------------------------------------------------------------
// KHashTable - Private API - flag bucket API - set
// ------------------------------------------------------------------------------------------

template <typename V, typename H, typename KE>
template <unsigned flag_selector, bool set>
inline PetscErrorCode KHashTable<V, H, KE>::khash_set_flag_(khash_int it, std::vector<flags_type> &flags) noexcept
{
  static_assert(flag_selector > 0U && flag_selector <= 3U, "");

  PetscFunctionBegin;
  if (set) {
    flag_bucket_at_(it, flags) |= flag_selector << flag_bucket_index_(it);
  } else {
    flag_bucket_at_(it, flags) &= ~(flag_selector << flag_bucket_index_(it));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <typename V, typename H, typename KE>
template <bool b>
inline PetscErrorCode KHashTable<V, H, KE>::khash_set_deleted_(khash_int it, std::vector<flags_type> &flags) noexcept
{
  PetscFunctionBegin;
  PetscCall(khash_set_flag_<1, b>(it, flags));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <typename V, typename H, typename KE>
template <bool b>
inline PetscErrorCode KHashTable<V, H, KE>::khash_set_empty_(khash_int it, std::vector<flags_type> &flags) noexcept
{
  PetscFunctionBegin;
  PetscCall(khash_set_flag_<2, b>(it, flags));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <typename V, typename H, typename KE>
template <bool b>
inline PetscErrorCode KHashTable<V, H, KE>::khash_set_both_(khash_int it, std::vector<flags_type> &flags) noexcept
{
  PetscFunctionBegin;
  PetscCall(khash_set_flag_<3, b>(it, flags));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <typename V, typename H, typename KE>
template <bool b>
inline PetscErrorCode KHashTable<V, H, KE>::khash_set_deleted_(khash_int it) noexcept
{
  PetscFunctionBegin;
  PetscCall(khash_set_deleted_<b>(it, flags_));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <typename V, typename H, typename KE>
template <bool b>
inline PetscErrorCode KHashTable<V, H, KE>::khash_set_empty_(khash_int it) noexcept
{
  PetscFunctionBegin;
  PetscCall(khash_set_empty_<b>(it, flags_));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <typename V, typename H, typename KE>
template <bool b>
inline PetscErrorCode KHashTable<V, H, KE>::khash_set_both_(khash_int it) noexcept
{
  PetscFunctionBegin;
  PetscCall(khash_set_both_<b>(it, flags_));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <typename V, typename H, typename KE>
template <typename KeyType, typename ValueConstructor>
inline std::pair<typename KHashTable<V, H, KE>::iterator, bool> KHashTable<V, H, KE>::find_and_emplace_final_(KeyType &&key, ValueConstructor &&constructor) noexcept
{
  khash_int it = 0;

  PetscFunctionBegin;
  PetscCallAbort(PETSC_COMM_SELF, khash_maybe_rehash_());
  {
    const auto nb   = bucket_count();
    const auto mask = nb - 1;
    const auto hash = hash_function()(key);
    auto       i    = hash & mask;

    PetscAssertAbort(nb > 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Have %zu bucket count after rehash", nb);
    if (khash_is_empty_(i)) {
      it = i; // for speed up
    } else {
      const auto last = i;
      auto       site = nb;
      khash_int  step = 0;

      it = nb;
      while (!khash_is_empty_(i) && (khash_is_del_(i) || !key_eq()(values_[i], key))) {
        if (khash_is_del_(i)) site = i;
        i = (i + (++step)) & mask;
        if (i == last) {
          it = site;
          break;
        }
      }
      if (it == nb) {
        // didn't find a completely empty place to put it, see if we can reuse an existing
        // bucket
        if (khash_is_empty_(i) && site != nb) {
          // reuse a deleted element (I think)
          it = site;
        } else {
          it = i;
        }
      }
    }
  }
  if (occupied(it)) PetscFunctionReturn({make_iterator_(it), false});
  // not present at all or deleted, so create (or replace) the element using the constructor
  // lambda
  PetscCallCXXAbort(PETSC_COMM_SELF, values_[it] = constructor());
  ++count_;
  if (khash_is_empty_(it)) ++n_occupied_;
  // order matters, must do this _after_ we check is_empty() since this call sets is_empty to
  // false!
  PetscCallAbort(PETSC_COMM_SELF, khash_set_both_<false>(it));
  PetscFunctionReturn({make_iterator_(it), true});
}

template <typename V, typename H, typename KE>
inline PetscErrorCode KHashTable<V, H, KE>::khash_maybe_rehash_() noexcept
{
  PetscFunctionBegin;
  if (n_occupied_ >= upper_bound_) {
    auto target_size = bucket_count();

    if (target_size > (size() << 1)) {
      // clear "deleted" elements
      --target_size;
    } else {
      // expand the hash table
      ++target_size;
    }
    PetscCall(resize(target_size));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <typename V, typename H, typename KE>
inline PetscErrorCode KHashTable<V, H, KE>::khash_erase_(khash_int it) noexcept
{
  PetscFunctionBegin;
  PetscAssert(it < bucket_count(), PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Attempting to erase iterator (index %d) which did not exist in the map", it);
  PetscAssert(count_ > 0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Attempting to erase iterator (index %d) which did not exist in the map, have element count %zu", it, count_);
  PetscAssert(occupied(it), PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Attempting to erase iterator (index %d) which exists in the map but is not occupied", it);
  --count_;
  PetscCall(khash_set_deleted_<true>(it));
  PetscCallCXX(values_[it] = value_type{});
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ==========================================================================================
// KHashTable - Protected API
// ==========================================================================================

template <typename V, typename H, typename KE>
inline typename KHashTable<V, H, KE>::iterator KHashTable<V, H, KE>::make_iterator_(khash_int it) noexcept
{
  return {this, it};
}

template <typename V, typename H, typename KE>
inline typename KHashTable<V, H, KE>::const_iterator KHashTable<V, H, KE>::make_iterator_(khash_int it) const noexcept
{
  return {this, it};
}

template <typename V, typename H, typename KE>
template <typename T>
inline PetscErrorCode KHashTable<V, H, KE>::khash_find_(T &&key, khash_int *it) const noexcept
{
  const auto nb  = bucket_count();
  auto       ret = nb;

  PetscFunctionBegin;
  if (nb) {
    const auto mask = nb - 1;
    const auto hash = hash_function()(key);
    auto       i    = hash & mask;
    const auto last = i;
    khash_int  step = 0;

    while (!khash_is_empty_(i) && (khash_is_del_(i) || !key_equal()(values_[i], key))) {
      i = (i + (++step)) & mask;
      if (i == last) {
        *it = ret;
        PetscFunctionReturn(PETSC_SUCCESS);
      }
    }
    if (occupied(i)) ret = i;
  }
  *it = ret;
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <typename V, typename H, typename KE>
template <typename KeyType, typename... ValueTypeArgs>
inline std::pair<typename KHashTable<V, H, KE>::iterator, bool> KHashTable<V, H, KE>::find_and_emplace_(KeyType &&key, ValueTypeArgs &&...value_ctor_args) noexcept
{
  return find_and_emplace_final_(std::forward<KeyType>(key), [&] { return value_type{std::forward<ValueTypeArgs>(value_ctor_args)...}; });
}

template <typename V, typename H, typename KE>
template <typename KeyValueType>
inline std::pair<typename KHashTable<V, H, KE>::iterator, bool> KHashTable<V, H, KE>::find_and_emplace_(KeyValueType &&key_value) noexcept
{
  return find_and_emplace_final_(std::forward<KeyValueType>(key_value), [&] { return std::forward<KeyValueType>(key_value); });
}

// ==========================================================================================
// KHashTable - Public API
// ==========================================================================================

// Generic iterator constructor
template <typename V, typename H, typename KE>
template <typename Iter>
inline KHashTable<V, H, KE>::KHashTable(Iter first, Iter last) noexcept : KHashTable(std::move(first), std::move(last), typename std::iterator_traits<Iter>::iterator_category{})
{
}

template <typename V, typename H, typename KE>
inline KHashTable<V, H, KE>::KHashTable(KHashTable &&other) noexcept :
  values_(std::move(other.values_)), flags_(std::move(other.flags_)), count_(util::exchange(other.count_, 0)), n_occupied_(util::exchange(other.n_occupied_, 0)), upper_bound_(util::exchange(other.upper_bound_, 0))
{
}

template <typename V, typename H, typename KE>
inline KHashTable<V, H, KE> &KHashTable<V, H, KE>::operator=(KHashTable &&other) noexcept
{
  PetscFunctionBegin;
  if (this != &other) {
    PetscCallCXXAbort(PETSC_COMM_SELF, values_ = std::move(other.values_));
    PetscCallCXXAbort(PETSC_COMM_SELF, flags_ = std::move(other.flags_));
    count_       = util::exchange(other.count_, 0);
    n_occupied_  = util::exchange(other.n_occupied_, 0);
    upper_bound_ = util::exchange(other.upper_bound_, 0);
  }
  PetscFunctionReturn(*this);
}

template <typename V, typename H, typename KE>
inline typename KHashTable<V, H, KE>::iterator KHashTable<V, H, KE>::begin() noexcept
{
  khash_int it = 0;

  PetscFunctionBegin;
  for (; it < bucket_count(); ++it) {
    if (occupied(it)) break;
  }
  PetscFunctionReturn(make_iterator_(it));
}

template <typename V, typename H, typename KE>
inline typename KHashTable<V, H, KE>::const_iterator KHashTable<V, H, KE>::cbegin() const noexcept
{
  khash_int it = 0;

  PetscFunctionBegin;
  for (; it < bucket_count(); ++it) {
    if (occupied(it)) break;
  }
  PetscFunctionReturn(make_iterator_(it));
}

template <typename V, typename H, typename KE>
inline typename KHashTable<V, H, KE>::const_iterator KHashTable<V, H, KE>::begin() const noexcept
{
  return cbegin();
}

template <typename V, typename H, typename KE>
inline typename KHashTable<V, H, KE>::iterator KHashTable<V, H, KE>::end() noexcept
{
  return make_iterator_(bucket_count());
}

template <typename V, typename H, typename KE>
inline typename KHashTable<V, H, KE>::const_iterator KHashTable<V, H, KE>::cend() const noexcept
{
  return make_iterator_(bucket_count());
}

template <typename V, typename H, typename KE>
inline typename KHashTable<V, H, KE>::const_iterator KHashTable<V, H, KE>::end() const noexcept
{
  return cend();
}

template <typename V, typename H, typename KE>
inline typename KHashTable<V, H, KE>::size_type KHashTable<V, H, KE>::bucket_count() const noexcept
{
  return values_.size();
}

template <typename V, typename H, typename KE>
inline typename KHashTable<V, H, KE>::size_type KHashTable<V, H, KE>::size() const noexcept
{
  return count_;
}

template <typename V, typename H, typename KE>
inline typename KHashTable<V, H, KE>::size_type KHashTable<V, H, KE>::capacity() const noexcept
{
  return flags_.size() * flag_pairs_per_bucket::value;
}

template <typename V, typename H, typename KE>
inline bool KHashTable<V, H, KE>::empty() const noexcept
{
  return size() == 0;
}

// REVIEW ME: should really be called rehash()
template <typename V, typename H, typename KE>
inline PetscErrorCode KHashTable<V, H, KE>::reserve(size_type req_size) noexcept
{
  PetscFunctionBegin;
  if (size() < req_size) PetscCall(resize(req_size));
  PetscFunctionReturn(PETSC_SUCCESS);
}

namespace detail
{

// templated version of
// https://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2. See also
// https://stackoverflow.com/questions/466204/rounding-up-to-next-power-of-2
template <typename T>
static inline PETSC_CONSTEXPR_14 T round_up_to_next_pow2(T v) noexcept
{
  static_assert(std::numeric_limits<T>::is_integer && std::is_unsigned<T>::value, "");
  if (v <= 1) return 1;
  --v;
  for (std::size_t i = 1; i < (sizeof(v) * CHAR_BIT); i *= 2) v |= v >> i;
  ++v;
  return v;
}

  // compilers sadly don't yet recognize that the above is just searching for the next nonzero
  // bit (https://godbolt.org/z/3q1qxqK4a) and won't emit the versions below, which usually
  // boil down to a single tailor-made instruction.
  //
  // __builtin_clz():
  // Returns the number of leading 0-bits in x, starting at the most significant bit
  // position. If x is 0, the result is undefined.
  //
  // see https://gcc.gnu.org/onlinedocs/gcc/Other-Builtins.html

  #if PetscHasBuiltin(__builtin_clz)
template <>
inline constexpr unsigned int round_up_to_next_pow2(unsigned int v) noexcept
{
  return v <= 1 ? 1 : 1 << ((sizeof(v) * CHAR_BIT) - __builtin_clz(v - 1));
}
  #endif

  #if PetscHasBuiltin(__builtin_clzl)
template <>
inline constexpr unsigned long round_up_to_next_pow2(unsigned long v) noexcept
{
  return v <= 1 ? 1 : 1 << ((sizeof(v) * CHAR_BIT) - __builtin_clzl(v - 1));
}
  #endif

  // both MSVC and Intel compilers lie about having __builtin_clzll so just disable this
  #if PetscHasBuiltin(__builtin_clzll) && !PetscDefined(HAVE_WINDOWS_COMPILERS)
template <>
inline constexpr unsigned long long round_up_to_next_pow2(unsigned long long v) noexcept
{
  return v <= 1 ? 1 : 1 << ((sizeof(v) * CHAR_BIT) - __builtin_clzll(v - 1));
}
  #endif

template <typename T>
static inline constexpr unsigned integer_log2(T x) noexcept
{
  static_assert(std::numeric_limits<T>::is_integer && std::is_unsigned<T>::value, "");
  return x ? 1 + integer_log2(x >> 1) : std::numeric_limits<unsigned>::max();
}

} // namespace detail

template <typename V, typename H, typename KE>
inline PetscErrorCode KHashTable<V, H, KE>::resize(size_type req_size) noexcept
{
  constexpr size_type min_n_buckets = 4;
  const auto          new_n_buckets = std::max(detail::round_up_to_next_pow2(req_size), min_n_buckets);
  const auto          new_size      = (new_n_buckets >> 1) + (new_n_buckets >> 2);

  PetscFunctionBegin;
  // hash table count to be changed (shrink or expand); rehash
  if (size() < new_size) {
    const auto old_n_buckets = bucket_count();
    const auto new_mask      = new_n_buckets - 1;
    const auto khash_fsize   = [](size_type size) -> size_type {
      if (size >= flag_pairs_per_bucket::value) {
        // use constexpr here to force compiler to evaluate this at all optimization levels
        constexpr auto shift_val = detail::integer_log2(flag_pairs_per_bucket::value);

        return size >> shift_val;
      }
      return 1;
    };
    std::vector<flags_type> new_flags(khash_fsize(new_n_buckets), default_bit_pattern());

    // grow the hash table, note order is important! we cannot just call
    // values_.resize(new_n_buckets) because that might drop buckets containing data. The loop
    // below (if new_n_buckets < bucket_count()) will compress the table, such that we can
    // shrink afterwards
    if (old_n_buckets < new_n_buckets) PetscCallCXX(values_.resize(new_n_buckets));
    for (size_type i = 0; i < old_n_buckets; ++i) {
      if (!occupied(i)) continue;
      // kick-out process; sort of like in Cuckoo hashing
      PetscCall(khash_set_deleted_<true>(i));
      while (true) {
        // key is updated every loop from the swap so need to recompute the hash function each
        // time... could possibly consider stashing the hash value in the key-value pair
        auto      &key  = values_[i];
        const auto hash = hash_function()(key);
        auto       j    = hash & new_mask;
        khash_int  step = 0;

        while (!khash_is_empty_(j, new_flags)) j = (j + (++step)) & new_mask;
        PetscCall(khash_set_empty_<false>(j, new_flags));
        if (j < old_n_buckets && occupied(j)) {
          using std::swap;

          // i == j should never reach this point since occupied(j) (in this case equivalent
          // to occupied(i)) should never be true because we call khash_set_deleted_<true>(i)
          // above!
          PetscAssert(i != j, PETSC_COMM_SELF, PETSC_ERR_PLIB, "i %zu = j %zu. About to swap the same element!", static_cast<std::size_t>(i), static_cast<std::size_t>(j));
          // kick out the existing element
          PetscCallCXX(swap(values_[j], key));
          // mark it as deleted in the old hash table
          PetscCall(khash_set_deleted_<true>(j));
        } else {
          // write the element and jump out of the loop but check that we don't self
          // move-assign
          if (i != j) PetscCallCXX(values_[j] = std::move(key));
          break;
        }
      }
    }

    // shrink the hash table
    if (old_n_buckets > new_n_buckets) PetscCallCXX(values_.resize(new_n_buckets));
    PetscCallCXX(flags_ = std::move(new_flags));
    n_occupied_  = count_;
    upper_bound_ = new_size;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <typename V, typename H, typename KE>
inline PetscErrorCode KHashTable<V, H, KE>::clear() noexcept
{
  PetscFunctionBegin;
  PetscCallCXX(values_.clear());
  PetscCallCXX(std::fill(flags_.begin(), flags_.end(), default_bit_pattern()));
  count_       = 0;
  n_occupied_  = 0;
  upper_bound_ = 0;
  PetscAssert(size() == 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "clear() did not set size (%zu) to 0", size());
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <typename V, typename H, typename KE>
inline bool KHashTable<V, H, KE>::occupied(khash_int it) const noexcept
{
  return khash_occupied_(it, flags_);
}

template <typename V, typename H, typename KE>
inline bool KHashTable<V, H, KE>::occupied(const_iterator it) const noexcept
{
  return occupied(it.it_);
}

template <typename V, typename H, typename KE>
inline typename KHashTable<V, H, KE>::iterator KHashTable<V, H, KE>::erase(iterator pos) noexcept
{
  iterator ret(pos);

  PetscFunctionBegin;
  ++ret;
  PetscCallAbort(PETSC_COMM_SELF, khash_erase_(pos.it_));
  PetscFunctionReturn(ret);
}

template <typename V, typename H, typename KE>
inline typename KHashTable<V, H, KE>::iterator KHashTable<V, H, KE>::erase(const_iterator pos) noexcept
{
  iterator ret(pos);

  PetscFunctionBegin;
  ++ret;
  PetscCallAbort(PETSC_COMM_SELF, khash_erase_(pos.it_));
  PetscFunctionReturn(ret);
}

template <typename V, typename H, typename KE>
inline typename KHashTable<V, H, KE>::iterator KHashTable<V, H, KE>::erase(const_iterator begin_pos, const_iterator end_pos) noexcept
{
  PetscFunctionBegin;
  for (; begin_pos != end_pos; ++begin_pos) PetscCallAbort(PETSC_COMM_SELF, khash_erase_(begin_pos.it_));
  PetscFunctionReturn(make_iterator_(begin_pos.it_));
}

template <typename V, typename H, typename KE>
template <typename... Args>
inline std::pair<typename KHashTable<V, H, KE>::iterator, bool> KHashTable<V, H, KE>::emplace(Args &&...args) noexcept
{
  return find_and_emplace_(value_type{std::forward<Args>(args)...});
}

template <typename V, typename H, typename KE>
inline std::pair<typename KHashTable<V, H, KE>::iterator, bool> KHashTable<V, H, KE>::insert(const value_type &val) noexcept
{
  return find_and_emplace_(val);
}

template <typename V, typename H, typename KE>
inline typename KHashTable<V, H, KE>::iterator KHashTable<V, H, KE>::insert(const_iterator, const value_type &val) noexcept
{
  return insert(val).first;
}

template <typename V, typename H, typename KE>
inline std::pair<typename KHashTable<V, H, KE>::iterator, bool> KHashTable<V, H, KE>::insert(value_type &&val) noexcept
{
  return find_and_emplace_(std::move(val));
}

template <typename V, typename H, typename KE>
inline typename KHashTable<V, H, KE>::iterator KHashTable<V, H, KE>::insert(const_iterator, value_type &&val) noexcept
{
  return insert(std::move(val)).first;
}

template <typename V, typename H, typename KE>
inline typename KHashTable<V, H, KE>::hasher KHashTable<V, H, KE>::hash_function() const noexcept
{
  return this->first();
}

template <typename V, typename H, typename KE>
inline typename KHashTable<V, H, KE>::key_equal KHashTable<V, H, KE>::key_eq() const noexcept
{
  return this->second();
}

template <typename V, typename H, typename KE>
inline void KHashTable<V, H, KE>::swap(KHashTable &other) noexcept
{
  PetscFunctionBegin;
  if (PetscUnlikely(this != &other)) {
    using std::swap;

    swap(values_, other.values_);
    swap(flags_, other.flags_);
    swap(count_, other.count_);
    swap(n_occupied_, other.n_occupied_);
    swap(upper_bound_, other.upper_bound_);
  }
  PetscFunctionReturnVoid();
}

namespace detail
{

template <typename KeyType, typename Hasher>
struct indirect_hasher : Hasher {
  using nested_value_type = Hasher;
  using key_type          = KeyType;

  template <typename T>
  PETSC_NODISCARD std::size_t operator()(const std::pair<key_type, T> &kv) const noexcept
  {
    return static_cast<const nested_value_type &>(*this)(kv.first);
  }

  template <typename T>
  PETSC_NODISCARD std::size_t operator()(const std::pair<key_type, T> &kv) noexcept
  {
    return static_cast<nested_value_type &>(*this)(kv.first);
  }

  using nested_value_type::operator();
};

template <typename KeyType, typename KeyEqual>
struct indirect_equal : KeyEqual {
  using nested_value_type = KeyEqual;
  using key_type          = KeyType;

  template <typename T>
  PETSC_NODISCARD bool operator()(const std::pair<key_type, T> &lhs, const std::pair<key_type, T> &rhs) const noexcept
  {
    return static_cast<const nested_value_type &>(*this)(lhs.first, rhs.first);
  }

  template <typename T>
  PETSC_NODISCARD bool operator()(const std::pair<key_type, T> &lhs, const key_type &rhs) const noexcept
  {
    return static_cast<const nested_value_type &>(*this)(lhs.first, rhs);
  }
};

} // namespace detail

} // namespace khash

// ==========================================================================================
// UnorderedMap - A drop-in replacement for std::unordered_map that is more memory efficient
// and performant.
//
// Has identical API to a C++17 conformant std::unordered_map, and behaves identically to
// it. The only exception is iterator invalidation:
//
//  Operation                                  |  std::unorderd_map    | Petsc::UnorderedMap
// --------------------------------------------|-----------------------|---------------------
// - All read only operations, swap, std::swap | Never                 | Never
// - clear, operator=                          | Always                | Always
// - rehash, reserve                           | Always                | Only if causes
//                                             |                       | resizing
// - insert, emplace, emplace_hint, operator[] | Only if causes rehash | Only if it causes
//                                             |                       | rehash, in which case
//                                             |                       | rehash will ALWAYS
//                                             |                       | resize
// - erase                                     | Only to the element   | Only to the element
//                                             | erased                | erased
// ==========================================================================================
template <typename K, typename T, typename H = std::hash<K>,
  #if PETSC_CPP_VERSION >= 14
          typename KE = std::equal_to<>
  #else
          typename KE = std::equal_to<K>
  #endif
          >
class UnorderedMap;

template <typename KeyType, typename T, typename Hash, typename KeyEqual>
class UnorderedMap : public khash::KHashTable<std::pair<KeyType, T>, khash::detail::indirect_hasher<KeyType, Hash>, khash::detail::indirect_equal<KeyType, KeyEqual>> {
  using table_type = khash::KHashTable<std::pair<KeyType, T>, khash::detail::indirect_hasher<KeyType, Hash>, khash::detail::indirect_equal<KeyType, KeyEqual>>;
  using typename table_type::khash_int;

public:
  // workaround for MSVC bug
  // https://developercommunity.visualstudio.com/t/error-c2244-unable-to-match-function-definition-to/225941
  using value_type      = typename table_type::value_type;
  using key_type        = typename value_type::first_type;
  using mapped_type     = typename value_type::second_type;
  using hasher          = typename table_type::hasher::nested_value_type;
  using key_equal       = typename table_type::key_equal::nested_value_type;
  using size_type       = typename table_type::size_type;
  using reference       = typename table_type::reference;
  using const_reference = typename table_type::const_reference;
  using iterator        = typename table_type::iterator;
  using const_iterator  = typename table_type::const_iterator;

  using table_type::table_type; // inherit constructors

  PETSC_NODISCARD iterator       find(const key_type &) noexcept;
  PETSC_NODISCARD const_iterator find(const key_type &) const noexcept;

  template <typename... Args>
  std::pair<iterator, bool> emplace(Args &&...) noexcept;

  using table_type::erase; // inherit erase overloads
  size_type erase(const key_type &) noexcept;

  mapped_type &operator[](const key_type &) noexcept;

  PETSC_NODISCARD std::pair<iterator, iterator>             equal_range(const key_type &) noexcept;
  PETSC_NODISCARD std::pair<const_iterator, const_iterator> equal_range(const key_type &) const noexcept;

  PETSC_NODISCARD size_type count(const key_type &) const noexcept;
  PETSC_NODISCARD bool      contains(const key_type &) const noexcept;

  // must be declared in class definition...
  friend void swap(UnorderedMap &lhs, UnorderedMap &rhs) noexcept
  {
    PetscFunctionBegin;
    PetscCallCXXAbort(PETSC_COMM_SELF, lhs.swap(rhs));
    PetscFunctionReturnVoid();
  }

private:
  template <typename KeyTuple, typename ArgTuple>
  PETSC_NODISCARD std::pair<iterator, bool> emplace_(std::piecewise_construct_t, KeyTuple &&, ArgTuple &&) noexcept;
  template <typename Key, typename... Args>
  PETSC_NODISCARD std::pair<iterator, bool> emplace_(Key &&, Args &&...) noexcept;
};

// ==========================================================================================
// UnorderedMap - Private API
// ==========================================================================================

template <typename K, typename T, typename H, typename KE>
template <typename KeyTuple, typename MappedTuple>
inline std::pair<typename UnorderedMap<K, T, H, KE>::iterator, bool> UnorderedMap<K, T, H, KE>::emplace_(std::piecewise_construct_t pcw, KeyTuple &&key_tuple, MappedTuple &&mapped_type_constructor_args) noexcept
{
  // clang-format off
  return this->find_and_emplace_(
    std::get<0>(key_tuple),
    pcw,
    std::forward<KeyTuple>(key_tuple),
    std::forward<MappedTuple>(mapped_type_constructor_args)
  );
  // clang-format on
}

template <typename K, typename T, typename H, typename KE>
template <typename Key, typename... Args>
inline std::pair<typename UnorderedMap<K, T, H, KE>::iterator, bool> UnorderedMap<K, T, H, KE>::emplace_(Key &&key, Args &&...mapped_type_constructor_args) noexcept
{
  return this->emplace_(std::piecewise_construct, std::forward_as_tuple(std::forward<Key>(key)), std::forward_as_tuple(std::forward<Args>(mapped_type_constructor_args)...));
}

// ==========================================================================================
// UnorderedMap - Public API
// ==========================================================================================

template <typename K, typename T, typename H, typename KE>
inline typename UnorderedMap<K, T, H, KE>::iterator UnorderedMap<K, T, H, KE>::find(const key_type &key) noexcept
{
  khash_int it = 0;

  PetscFunctionBegin;
  PetscCallAbort(PETSC_COMM_SELF, this->khash_find_(key, &it));
  PetscFunctionReturn(this->make_iterator_(it));
}

template <typename K, typename T, typename H, typename KE>
inline typename UnorderedMap<K, T, H, KE>::const_iterator UnorderedMap<K, T, H, KE>::find(const key_type &key) const noexcept
{
  khash_int it = 0;

  PetscFunctionBegin;
  PetscCallAbort(PETSC_COMM_SELF, this->khash_find_(key, &it));
  PetscFunctionReturn(this->make_iterator_(it));
}

template <typename K, typename T, typename H, typename KE>
template <typename... Args>
inline std::pair<typename UnorderedMap<K, T, H, KE>::iterator, bool> UnorderedMap<K, T, H, KE>::emplace(Args &&...args) noexcept
{
  return this->emplace_(std::forward<Args>(args)...);
}

template <typename K, typename T, typename H, typename KE>
inline typename UnorderedMap<K, T, H, KE>::mapped_type &UnorderedMap<K, T, H, KE>::operator[](const key_type &key) noexcept
{
  return this->emplace(key).first->second;
}

template <typename K, typename T, typename H, typename KE>
inline typename UnorderedMap<K, T, H, KE>::size_type UnorderedMap<K, T, H, KE>::erase(const key_type &key) noexcept
{
  PetscFunctionBegin;
  {
    auto it = this->find(key);

    if (it == this->end()) PetscFunctionReturn(PETSC_SUCCESS);
    PetscCallCXX(this->erase(it));
  }
  PetscFunctionReturn(1);
}

template <typename K, typename T, typename H, typename KE>
inline std::pair<typename UnorderedMap<K, T, H, KE>::iterator, typename UnorderedMap<K, T, H, KE>::iterator> UnorderedMap<K, T, H, KE>::equal_range(const key_type &key) noexcept
{
  auto it = this->find(key);
  return {it, it == this->end() ? it : std::next(it)};
}

template <typename K, typename T, typename H, typename KE>
inline std::pair<typename UnorderedMap<K, T, H, KE>::const_iterator, typename UnorderedMap<K, T, H, KE>::const_iterator> UnorderedMap<K, T, H, KE>::equal_range(const key_type &key) const noexcept
{
  auto it = this->find(key);
  return {it, it == this->end() ? it : std::next(it)};
}

template <typename K, typename T, typename H, typename KE>
inline typename UnorderedMap<K, T, H, KE>::size_type UnorderedMap<K, T, H, KE>::count(const key_type &key) const noexcept
{
  return this->contains(key);
}

template <typename K, typename T, typename H, typename KE>
inline bool UnorderedMap<K, T, H, KE>::contains(const key_type &key) const noexcept
{
  return this->find(key) != this->end();
}

// ==========================================================================================
// UnorderedMap - Global functions
// ==========================================================================================

template <typename K, typename T, typename H, typename KE>
PETSC_NODISCARD bool operator==(const UnorderedMap<K, T, H, KE> &lhs, const UnorderedMap<K, T, H, KE> &rhs) noexcept
{
  PetscFunctionBegin;
  if (lhs.size() != rhs.size()) PetscFunctionReturn(false);
  for (auto it = lhs.begin(), lhs_end = lhs.end(), rhs_end = rhs.end(); it != lhs_end; ++it) {
    const auto rhs_it = rhs.find(it->first);

    if (rhs_it == rhs_end || !(*it == *rhs_it)) PetscFunctionReturn(false);
  }
  PetscFunctionReturn(true);
}

template <typename K, typename T, typename H, typename KE>
PETSC_NODISCARD bool operator!=(const UnorderedMap<K, T, H, KE> &lhs, const UnorderedMap<K, T, H, KE> &rhs) noexcept
{
  return !(lhs == rhs);
}

} // namespace Petsc

#endif // __cplusplus

#endif // PETSC_CPP_UNORDERED_MAP_HPP
