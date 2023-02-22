static const char help[] = "Tests UnorderedMap.\n";

#include <petsc/private/cpp/unordered_map.hpp>
#include <petscviewer.h>

#include <sstream> // std::ostringstream
#include <string>
#include <vector>
#include <algorithm> // std::sort

// ==========================================================================================
// Setup
// ==========================================================================================

// see https://stackoverflow.com/questions/2590677/how-do-i-combine-hash-values-in-c0x
static inline void hash_combine(std::size_t &) noexcept { }

template <typename T, typename... Rest>
static inline void hash_combine(std::size_t &seed, const T &v, Rest &&...rest) noexcept
{
  std::hash<T> hasher;
  seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  hash_combine(seed, std::forward<Rest>(rest)...);
}

#define MAKE_HASHABLE(type, ...) \
  namespace std \
  { \
  template <> \
  struct hash<type> { \
    std::size_t operator()(const type &t) const noexcept \
    { \
      std::size_t ret = 0; \
      hash_combine(ret, __VA_ARGS__); \
      return ret; \
    } \
  }; \
  }

using pair_type = std::pair<int, double>;
MAKE_HASHABLE(pair_type, t.first, t.second);

using namespace Petsc::util;

struct Foo {
  int    x{};
  double y{};

  constexpr Foo() noexcept = default;
  constexpr Foo(int x, double y) noexcept : x(x), y(y) { }

  bool operator==(const Foo &other) const noexcept { return x == other.x && y == other.y; }
  bool operator!=(const Foo &other) const noexcept { return !(*this == other); }
  bool operator<(const Foo &other) const noexcept { return std::tie(x, y) < std::tie(other.x, other.y); }

  PetscErrorCode to_string(std::string &buf) const noexcept
  {
    PetscFunctionBegin;
    PetscCallCXX(buf = std::to_string(x) + ", " + std::to_string(y));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  friend std::ostream &operator<<(std::ostream &oss, const Foo &f) noexcept
  {
    std::string ret;

    PetscFunctionBegin;
    PetscCallAbort(PETSC_COMM_SELF, f.to_string(ret));
    oss << ret;
    PetscFunctionReturn(oss);
  }
};

MAKE_HASHABLE(Foo, t.x, t.y);

struct Bar {
  std::vector<int> x{};
  std::string      y{};

  Bar() noexcept = default;
  Bar(std::vector<int> x, std::string y) noexcept : x(std::move(x)), y(std::move(y)) { }

  bool operator==(const Bar &other) const noexcept { return x == other.x && y == other.y; }
  bool operator<(const Bar &other) const noexcept { return std::tie(x, y) < std::tie(other.x, other.y); }

  PetscErrorCode to_string(std::string &buf) const noexcept
  {
    PetscFunctionBegin;
    PetscCallCXX(buf = '<');
    for (std::size_t i = 0; i < x.size(); ++i) {
      PetscCallCXX(buf += std::to_string(x[i]));
      if (i + 1 != x.size()) PetscCallCXX(buf += ", ");
    }
    PetscCallCXX(buf += ">, <" + y + '>');
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  friend std::ostream &operator<<(std::ostream &oss, const Bar &b) noexcept
  {
    std::string ret;

    PetscFunctionBegin;
    PetscCallAbort(PETSC_COMM_SELF, b.to_string(ret));
    oss << ret;
    PetscFunctionReturn(oss);
  }
};

struct BadHash {
  template <typename T>
  constexpr std::size_t operator()(const T &) const noexcept
  {
    return 1;
  }
};

template <typename T>
struct Printer {
  using signature = PetscErrorCode(const T &, std::string &);

  mutable std::string      buffer;
  std::function<signature> printer;

  template <typename F>
  Printer(F &&printer) noexcept : printer(std::forward<F>(printer))
  {
  }

  PETSC_NODISCARD const char *operator()(const T &value) const noexcept
  {
    PetscFunctionBegin;
    PetscCallAbort(PETSC_COMM_SELF, this->printer(value, this->buffer));
    PetscFunctionReturn(this->buffer.c_str());
  }
};

#if defined(__GNUC__)
// gcc 6.4 through 7.5 have a visibility bug:
//
// error: 'MapTester<T>::test_insert()::<lambda(MapTester<T>::value_type&)> [with T =
// ...]::<lambda(...)>' declared with greater visibility than the type of its field
// 'MapTester<T>::test_insert()::<lambda(MapTester<T>::value_type&)> [with T =
// ...]::<lambda(const char*, const insert_return_type&)
//
// Error message implies that the visibility of the lambda in question is  greater than the
// visibility of the capture list value "this".
//
// Since lambdas are translated into the classes with the operator()(...) and (it seems like)
// captured values are translated into the fields of this class it looks like for some reason
// the visibility of that class is higher than the one of those fields.
//
// see https://gcc.gnu.org/bugzilla/show_bug.cgi?id=80947
  #if ((__GNUC__ == 6) && (__GNUC_MINOR__ >= 4)) || ((__GNUC__ == 7) && (__GNUC_MINOR__ <= 5))
    #define PETSC_GCC_LAMBDA_VISIBILITY_WORKAROUND 1
  #endif
#endif

#ifdef PETSC_GCC_LAMBDA_VISIBILITY_WORKAROUND
  #pragma GCC visibility push(hidden)
#endif
template <typename... T>
class MapTester {
public:
  using map_type    = Petsc::UnorderedMap<T...>;
  using key_type    = typename map_type::key_type;
  using value_type  = typename map_type::value_type;
  using mapped_type = typename map_type::mapped_type;

  const PetscViewer               vwr;
  const std::string               map_name;
  Printer<key_type>               key_printer;
  Printer<mapped_type>            value_printer;
  std::function<value_type(void)> generator;

  PetscErrorCode view_map(const map_type &map) const noexcept
  {
    std::ostringstream oss;

    PetscFunctionBegin;
    PetscCallCXX(oss << std::boolalpha);
    PetscCallCXX(oss << "map: '" << this->map_name << "'\n");
    PetscCallCXX(oss << "  size: " << map.size() << '\n');
    PetscCallCXX(oss << "  capacity: " << map.capacity() << '\n');
    PetscCallCXX(oss << "  bucket count: " << map.bucket_count() << '\n');
    PetscCallCXX(oss << "  empty: " << map.empty() << '\n');
    PetscCallCXX(oss << "  flag bucket width: " << map_type::flag_bucket_width::value << '\n');
    PetscCallCXX(oss << "  flag pairs per bucket: " << map_type::flag_pairs_per_bucket::value << '\n');
    PetscCallCXX(oss << "  {\n");
    for (auto &&entry : map) PetscCallCXX(oss << "    key: [" << this->key_printer(entry.first) << "] -> [" << this->value_printer(entry.second) << "]\n");
    PetscCallCXX(oss << "  }\n");
    PetscCall(PetscViewerASCIIPrintf(vwr, "%s", oss.str().c_str()));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

#define MapCheck(map__, cond__, comm__, ierr__, base_mess__, ...) \
  do { \
    if (PetscUnlikely(!(cond__))) { \
      PetscCall(this->view_map(map__)); \
      SETERRQ(comm__, ierr__, "%s: " base_mess__, this->map_name.c_str(), __VA_ARGS__); \
    } \
  } while (0)

  PetscErrorCode check_size_capacity_coherent(map_type &map) const noexcept
  {
    const auto msize = map.size();
    const auto mcap  = map.capacity();

    PetscFunctionBegin;
    MapCheck(map, msize == map.size(), PETSC_COMM_SELF, PETSC_ERR_PLIB, "Map size appears to change each time it is called! first call: %zu, second call %zu", msize, map.size());
    MapCheck(map, mcap == map.capacity(), PETSC_COMM_SELF, PETSC_ERR_PLIB, "Map capacity appears to change each time it is called! first call: %zu, second call %zu", mcap, map.capacity());
    MapCheck(map, msize >= 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Map size %zu unexpected!", msize);
    MapCheck(map, mcap >= 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Map capacity %zu unexpected!", mcap);
    MapCheck(map, mcap >= msize, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Map capacity %zu < map size %zu!", mcap, msize);
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscErrorCode check_size_capacity_coherent(map_type &map, std::size_t expected_size, std::size_t expected_min_capacity) const noexcept
  {
    PetscFunctionBegin;
    PetscCall(check_size_capacity_coherent(map));
    MapCheck(map, map.size() == expected_size, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Map size %zu did not increase (from %zu) after insertion!", map.size(), expected_size);
    MapCheck(map, map.capacity() >= expected_min_capacity, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Map capacity %zu did not increase (from %zu)!", map.capacity(), expected_min_capacity);
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscErrorCode test_insert(map_type &map) noexcept
  {
    auto key             = key_type{};
    auto value           = mapped_type{};
    auto size_before     = map.size();
    auto capacity_before = map.capacity();

    const auto check_all_reinsert = [&](value_type &key_value) {
      using insert_return_type  = std::pair<typename map_type::iterator, bool>;
      auto      &key            = key_value.first;
      auto      &value          = key_value.second;
      const auto key_const      = key;
      const auto value_const    = value;
      const auto pair           = std::make_pair(key_const, value_const);
      const auto check_reinsert = [&](const char op[], const insert_return_type &ret) {
        PetscFunctionBegin;
        MapCheck(map, !ret.second, PETSC_COMM_SELF, PETSC_ERR_PLIB, "%s reinserted key '%s'", op, this->key_printer(key));
        MapCheck(map, ret.first->first == key, PETSC_COMM_SELF, PETSC_ERR_PLIB, "%s returned iterator key '%s' != expected '%s'", this->key_printer(ret.first->first), op, this->key_printer(key));
        MapCheck(map, ret.first->second == value, PETSC_COMM_SELF, PETSC_ERR_PLIB, "%s returned iterator value '%s' != expected '%s'", op, this->value_printer(ret.first->second), this->value_printer(value));
        MapCheck(map, map[key] == value, PETSC_COMM_SELF, PETSC_ERR_PLIB, "%s map[%s] '%s' != '%s'", op, this->key_printer(key), this->value_printer(map[key]), this->value_printer(value));
        MapCheck(map, map[key_const] == value_const, PETSC_COMM_SELF, PETSC_ERR_PLIB, "%s changed value '%s' != expected '%s'", op, this->value_printer(map[key_const]), this->value_printer(value_const));
        PetscFunctionReturn(PETSC_SUCCESS);
      };

      PetscFunctionBegin;
#define CHECK_REINSERT(...) check_reinsert(PetscStringize(__VA_ARGS__), __VA_ARGS__)
      // check the following operations don't clobber values
      PetscCall(CHECK_REINSERT(map.emplace(key, value)));
      PetscCall(CHECK_REINSERT(map.emplace(std::piecewise_construct, std::make_tuple(key), std::make_tuple(value))));
      PetscCall(CHECK_REINSERT(map.insert(std::make_pair(key, value))));
      PetscCall(CHECK_REINSERT(map.insert(pair)));
#undef CHECK_REINSERT
      PetscFunctionReturn(PETSC_SUCCESS);
    };

    PetscFunctionBegin;
    PetscCall(this->check_size_capacity_coherent(map));
    // put key in map
    PetscCallCXX(map[key] = value);
    // check we properly sized up
    PetscCall(this->check_size_capacity_coherent(map, size_before + 1, capacity_before));
    // and that the value matches
    MapCheck(map, map[key] == value, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Map default key %s != map value %s", this->key_printer(key), this->value_printer(value));
    // and that the following operations don't clobber the value
    {
      value_type kv{key, value};

      PetscCall(check_all_reinsert(kv));
    }

    // test that clearing workings
    capacity_before = map.capacity();
    PetscCall(map.clear());
    // should have size = 0 (but capacity unchanged)
    PetscCall(this->check_size_capacity_coherent(map, 0, capacity_before));

    // test that all inserted values are found in the map
    const auto test_map_contains_expected_items = [&](std::function<PetscErrorCode(std::vector<value_type> &)> fill_map, std::size_t kv_size) {
      auto                     key_value_pairs = this->make_key_values(kv_size);
      std::vector<std::size_t> found_key_value(key_value_pairs.size());

      PetscFunctionBegin;
      PetscCall(map.clear());
      PetscCall(this->check_size_capacity_coherent(map, 0, 0));
      PetscCall(fill_map(key_value_pairs));
      // map size should exactly match the size of the vector, but we don't care about capacity
      PetscCall(this->check_size_capacity_coherent(map, key_value_pairs.size(), 0));

      // sort the vector so we can use std::binary_search on it
      PetscCallCXX(std::sort(key_value_pairs.begin(), key_value_pairs.end()));
      for (auto it = map.cbegin(); it != map.cend(); ++it) {
        const auto kv_begin = key_value_pairs.cbegin();
        const auto found    = std::lower_bound(kv_begin, key_value_pairs.cend(), *it);
        const auto dist     = std::distance(kv_begin, found);

        // check that the value returned exists in our expected range
        MapCheck(map, found != key_value_pairs.cend(), PETSC_COMM_SELF, PETSC_ERR_PLIB, "Map contained key-value pair (%s, %s) not present in input range!", this->key_printer(it->first), this->value_printer(it->second));
        MapCheck(map, dist >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Index of found key-value pair (%s -> %s) %td is < 0", this->key_printer(it->first), this->value_printer(it->second), static_cast<std::ptrdiff_t>(dist));
        // record that we found this particular entry
        PetscCallCXX(++found_key_value.at(static_cast<std::size_t>(dist)));
      }

      // there should only be 1 instance of each key-value in the map
      for (std::size_t i = 0; i < found_key_value.size(); ++i) {
        MapCheck(map, found_key_value[i] == 1, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Map failed to insert key %s (value %s), have find count %zu", this->key_printer(key_value_pairs[i].first), this->value_printer(key_value_pairs[i].second), found_key_value[i]);
      }
      PetscFunctionReturn(PETSC_SUCCESS);
    };

    // clang-format off
    PetscCall(
      test_map_contains_expected_items(
        [&](std::vector<value_type> &key_value_pairs)
        {
          PetscFunctionBegin;
          for (auto &&key_value : key_value_pairs) {
            PetscCallCXX(map[key_value.first] = key_value.second);
            PetscCall(check_all_reinsert(key_value));
          }
          PetscFunctionReturn(PETSC_SUCCESS);
        },
        108
      )
    );
    // clang-format on

    // test that inserting using std algorithms work
    {
      value_type saved_value;

      // clang-format off
      PetscCall(
        test_map_contains_expected_items(
          [&](std::vector<value_type> &key_value_pairs)
          {
            PetscFunctionBegin;
            // save this for later
            PetscCallCXX(saved_value = key_value_pairs.front());
            // test the algorithm insert works as expected
            PetscCallCXX(std::copy(key_value_pairs.cbegin(), key_value_pairs.cend(), std::inserter(map, map.begin())));
            PetscFunctionReturn(PETSC_SUCCESS);
          },
          179
        )
      );
      // clang-format on
      auto it = map.find(saved_value.first);

      // can't use map[] since that might inadvertently insert it
      MapCheck(map, it != map.end(), PETSC_COMM_SELF, PETSC_ERR_PLIB, "Map failed no longer contains key-value pair (%s -> %s) after std::copy() and container went out of scope", this->key_printer(saved_value.first), this->value_printer(saved_value.second));
      MapCheck(map, it->first == saved_value.first, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Map founnd iterator key (%s) does not match expected key (%s) after std::copy() insertion", this->key_printer(it->first), this->key_printer(saved_value.first));
      MapCheck(map, it->second == saved_value.second, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Map founnd iterator value (%s) does not match expected value (%s) after std::copy() insertion", this->value_printer(it->second), this->value_printer(saved_value.second));
    }
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscErrorCode test_insert() noexcept
  {
    map_type map;

    PetscFunctionBegin;
    PetscCall(test_insert(map));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscErrorCode test_find(map_type &map) noexcept
  {
    PetscFunctionBegin;
    {
      const auto sample_values = this->make_key_values(145);

      map = map_type(sample_values.begin(), sample_values.end());
      for (auto &&kv : sample_values) {
        auto &&key   = kv.first;
        auto &&value = kv.second;
        auto   it    = map.find(key);

        MapCheck(map, it != map.end(), PETSC_COMM_SELF, PETSC_ERR_PLIB, "Failed to find %s in map", this->key_printer(key));
        MapCheck(map, it->first == key, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Find iterator key %s != expected %s", this->key_printer(it->first), this->key_printer(key));
        MapCheck(map, it->second == value, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Find iterator value %s != expected %s", this->value_printer(it->second), this->value_printer(value));
        MapCheck(map, map.contains(key), PETSC_COMM_SELF, PETSC_ERR_PLIB, "map.contains(key) reports false, even though map.find(key) successfully found it! key: %s", this->key_printer(key));
        MapCheck(map, map.count(key) == 1, PETSC_COMM_SELF, PETSC_ERR_PLIB, "map.count(%s) %zu != 1", this->key_printer(key), map.count(key));

        {
          const auto  range       = map.equal_range(key);
          const auto &range_begin = range.first;
          const auto  range_size  = std::distance(range_begin, range.second);

          MapCheck(map, range_size == 1, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Map equal_range() returned a range of size %zu != 1", range_size);
          MapCheck(map, range_begin->first == key, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Equal range iterator key %s != expected %s", this->key_printer(range_begin->first), this->key_printer(key));
          MapCheck(map, range_begin->second == value, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Equal range iterator value %s != expected %s", this->value_printer(range_begin->second), this->value_printer(value));
        }
      }
    }
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscErrorCode test_find() noexcept
  {
    map_type map;

    PetscFunctionBegin;
    PetscCall(test_find(map));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscErrorCode test_erase(map_type &map) noexcept
  {
    auto           sample_values = this->make_key_values(57);
    const map_type backup(sample_values.cbegin(), sample_values.cend());
    const auto     check_map_is_truly_empty = [&](map_type &map) {
      PetscFunctionBegin;
      MapCheck(map, map.size() == 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Erasing map via iterator range didn't work, map has size %zu", map.size());
      MapCheck(map, map.empty(), PETSC_COMM_SELF, PETSC_ERR_PLIB, "Erasing map via iterators didn't work, map is not empty, has size %zu", map.size());
      // this loop should never actually fire!
      for (auto it = map.begin(); it != map.end(); ++it) MapCheck(map, false, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Erasing via iterator range did not work, map.begin() != map.end()%s", "");
      PetscFunctionReturn(PETSC_SUCCESS);
    };

    PetscFunctionBegin;
    PetscCallCXX(map = backup);
    // test single erase from iterator works
    {
      const auto it        = map.begin();
      const auto begin_key = it->first;
      const auto begin_val = it->second;

      PetscCallCXX(map.erase(it));
      for (auto &&kv : map) MapCheck(map, kv.first != begin_key, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Erasing %s did not work, found again in map", this->key_printer(begin_key));
      // reinsert the value
      PetscCallCXX(map[begin_key] = begin_val);
    }

    // test erase from iterator
    for (auto it = map.begin(); it != map.end(); ++it) {
      const auto before = it;

      PetscCallCXX(map.erase(it));
      MapCheck(map, before == it, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Iterator changed during erase%s", "");
      MapCheck(map, map.occupied(before) == false, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Iterator (%s -> %s) occupied after erase", this->key_printer(before->first), this->value_printer(before->second));
    }

    // test erase from iterator range
    PetscCall(check_map_is_truly_empty(map));
    PetscCallCXX(map = backup);
    PetscCallCXX(map.erase(map.begin(), map.end()));
    PetscCall(check_map_is_truly_empty(map));

    // test erase by clear
    PetscCallCXX(map = backup);
    PetscCall(map.clear());
    PetscCall(check_map_is_truly_empty(map));

    // test that clear works OK (used to be a bug when inserting after clear)
    PetscCallCXX(map.insert(generator()));
    PetscCallCXX(map.insert(generator()));
    PetscCallCXX(map.insert(generator()));
    PetscCallCXX(map.insert(generator()));
    PetscCallCXX(map.erase(map.begin(), map.end()));
    PetscCall(check_map_is_truly_empty(map));

    // test erase by member function swapping with empty map
    for (auto &&kv : sample_values) PetscCallCXX(map.emplace(kv.first, kv.second));
    {
      map_type alt;

      // has the effect of clearing the map
      PetscCallCXX(map.swap(alt));
    }
    PetscCall(check_map_is_truly_empty(map));

    // test erase by std::swap with empty map
    PetscCallCXX(map = backup);
    {
      using std::swap;
      map_type alt;

      // has the effect of clearing the map
      PetscCallCXX(swap(map, alt));
    }
    PetscCall(check_map_is_truly_empty(map));

    // test erase by key, use new values to change it up
    sample_values = this->make_key_values();
    std::copy(sample_values.cbegin(), sample_values.cend(), std::inserter(map, map.begin()));
    for (auto &&kv : sample_values) PetscCallCXX(map.erase(kv.first));
    PetscCall(check_map_is_truly_empty(map));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscErrorCode test_erase() noexcept
  {
    map_type map;

    PetscFunctionBegin;
    PetscCall(test_erase(map));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  // stupid dummy function because auto-lambdas are C++14
  template <typename It>
  PetscErrorCode test_iterators(map_type &map, It it, It it2) noexcept
  {
    constexpr std::size_t max_iter  = 10000;
    constexpr auto        is_normal = std::is_same<It, typename map_type::iterator>::value;
    constexpr auto        is_const  = std::is_same<It, typename map_type::const_iterator>::value;
    static_assert(is_normal || is_const, "");
    constexpr const char *it_name = is_normal ? "Non-const" : "Const";

    PetscFunctionBegin;
    MapCheck(map, it == it2, PETSC_COMM_SELF, PETSC_ERR_PLIB, "%s iterator does not equal itself?", it_name);
    PetscCallCXX(++it);
    PetscCallCXX(it2++);
    MapCheck(map, it == it2, PETSC_COMM_SELF, PETSC_ERR_PLIB, "%s iterator does not equal itself after ++it, and it2++", it_name);
    PetscCallCXX(--it);
    PetscCallCXX(it2--);
    MapCheck(map, it == it2, PETSC_COMM_SELF, PETSC_ERR_PLIB, "%s iterator does not equal itself after --it, and it2--", it_name);
    MapCheck(map, map.size() < max_iter, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Forward progress test only works properly if the map size (%zu) < %zu", map.size(), max_iter);
    // check that the prefix and postfix increment and decerement make forward progress
    {
      std::size_t i;

      // increment
      PetscCallCXX(it = map.begin());
      for (i = 0; i < max_iter; ++i) {
        if (it == map.end()) break;
        PetscCallCXX(++it);
      }
      MapCheck(map, i < max_iter, PETSC_COMM_SELF, PETSC_ERR_PLIB, "%s iterator did not appear to make forward progress using prefix increment! Reached maximum iteration count %zu for map of size %zu", it_name, max_iter, map.size());
      PetscCallCXX(it = map.begin());
      for (i = 0; i < max_iter; ++i) {
        if (it == map.end()) break;
        PetscCallCXX(it++);
      }
      MapCheck(map, i < max_iter, PETSC_COMM_SELF, PETSC_ERR_PLIB, "%s iterator did not appear to make forward progress using postfix increment! Reached maximum iteration count %zu for map of size %zu", it_name, max_iter, map.size());

      // decrement
      PetscCallCXX(it = std::prev(map.end()));
      for (i = 0; i < max_iter; ++i) {
        if (it == map.begin()) break;
        PetscCallCXX(--it);
      }
      MapCheck(map, i < max_iter, PETSC_COMM_SELF, PETSC_ERR_PLIB, "%s iterator did not appear to make forward progress using prefix decrement! Reached maximum iteration count %zu for map of size %zu", it_name, max_iter, map.size());
      PetscCallCXX(it = std::prev(map.end()));
      for (i = 0; i < max_iter; ++i) {
        if (it == map.begin()) break;
        PetscCallCXX(it--);
      }
      MapCheck(map, i < max_iter, PETSC_COMM_SELF, PETSC_ERR_PLIB, "%s iterator did not appear to make forward progress using postfix decrement! Reached maximum iteration count %zu for map of size %zu", it_name, max_iter, map.size());
    }

    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscErrorCode test_misc() noexcept
  {
    const auto sample_values = this->make_key_values(97);
    map_type   map(sample_values.begin(), sample_values.end());

    PetscFunctionBegin;
    PetscCall(this->test_iterators(map, map.begin(), map.begin()));
    PetscCall(this->test_iterators(map, map.cbegin(), map.cbegin()));
    {
      const auto backup                            = map;
      auto       map_copy                          = map;
      const auto check_original_map_did_not_change = [&](const char op[]) {
        PetscFunctionBegin;
        // the original map should not have changed at all
        MapCheck(map, map == backup, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Map does not equal the original map after %s", op);
        PetscFunctionReturn(PETSC_SUCCESS);
      };

      MapCheck(map_copy, map == map_copy, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Copy of map does not equal the original map%s", "");
      PetscCall(check_original_map_did_not_change("move assign"));
      // test that the copied map works OK
      PetscCall(this->test_insert(map_copy));
      PetscCall(check_original_map_did_not_change("test_insert()"));
      PetscCall(this->test_find(map_copy));
      PetscCall(check_original_map_did_not_change("test_find()"));
      PetscCall(this->test_erase(map_copy));
      PetscCall(check_original_map_did_not_change("test_erase()"));
      PetscCallCXX(map_copy = map);

      auto moved_copy = std::move(map_copy);

      MapCheck(moved_copy, map == moved_copy, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Moved copy of map does not equal the original map%s", "");
      PetscCall(check_original_map_did_not_change("move assign"));
      PetscCall(this->test_insert(moved_copy));
      PetscCall(check_original_map_did_not_change("test_insert()"));
      PetscCall(this->test_find(moved_copy));
      PetscCall(check_original_map_did_not_change("test_find()"));
      PetscCall(this->test_erase(moved_copy));
      PetscCall(check_original_map_did_not_change("test_erase()"));
    }
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscErrorCode test() noexcept
  {
    PetscFunctionBegin;
    PetscCall(this->test_insert());
    PetscCall(this->test_find());
    PetscCall(this->test_erase());
    PetscCall(this->test_misc());
    PetscFunctionReturn(PETSC_SUCCESS);
  }

private:
  PETSC_NODISCARD std::vector<value_type> make_key_values(std::size_t size = 100) const noexcept
  {
    std::vector<value_type> v(size);

    std::generate(v.begin(), v.end(), this->generator);
    return v;
  }
};
#ifdef PETSC_GCC_LAMBDA_VISIBILITY_WORKAROUND
  #pragma GCC visibility pop
#endif

template <typename... T, typename... Args>
PETSC_NODISCARD static MapTester<T...> make_tester(PetscViewer vwr, const char name[], Args &&...args)
{
  return {vwr, name, std::forward<Args>(args)...};
}

int main(int argc, char *argv[])
{
  PetscViewer vwr;
  PetscRandom rand;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));
  PetscCall(PetscRandomCreate(PETSC_COMM_SELF, &rand));
  PetscCall(PetscRandomSetInterval(rand, INT_MIN, INT_MAX));
  PetscCall(PetscRandomSetFromOptions(rand));
  PetscCall(PetscViewerASCIIGetStdout(PETSC_COMM_WORLD, &vwr));

  {
    // printer functions
    const auto int_printer = [](int key, std::string &buf) {
      PetscFunctionBegin;
      PetscCallCXX(buf = std::to_string(key));
      PetscFunctionReturn(PETSC_SUCCESS);
    };
    const auto double_printer = [](double value, std::string &buf) {
      PetscFunctionBegin;
      PetscCallCXX(buf = std::to_string(value));
      PetscFunctionReturn(PETSC_SUCCESS);
    };
    const auto foo_printer = [](const Foo &key, std::string &buf) {
      PetscFunctionBegin;
      PetscCall(key.to_string(buf));
      PetscFunctionReturn(PETSC_SUCCESS);
    };
    const auto bar_printer = [](const Bar &value, std::string &buf) {
      PetscFunctionBegin;
      PetscCall(value.to_string(buf));
      PetscFunctionReturn(PETSC_SUCCESS);
    };
    const auto pair_printer = [](const std::pair<int, double> &value, std::string &buf) {
      PetscFunctionBegin;
      PetscCallCXX(buf = '<' + std::to_string(value.first) + ", " + std::to_string(value.second) + '>');
      PetscFunctionReturn(PETSC_SUCCESS);
    };

    // generator functions
    const auto make_int = [&] {
      PetscReal x = 0.;

      PetscFunctionBegin;
      PetscCallAbort(PETSC_COMM_SELF, PetscRandomGetValueReal(rand, &x));
      PetscFunctionReturn(static_cast<int>(x));
    };
    const auto make_double = [&] {
      PetscReal x = 0.;

      PetscFunctionBegin;
      PetscCallAbort(PETSC_COMM_SELF, PetscRandomGetValueReal(rand, &x));
      PetscFunctionReturn(static_cast<double>(x));
    };
    const auto make_foo = [&] {
      PetscFunctionBegin;
      auto ret = Foo{make_int(), make_double()};
      PetscFunctionReturn(ret);
    };
    const auto make_bar = [&] {
      constexpr std::size_t max_size = 14, min_size = 1;
      const auto            isize = std::abs(make_int());
      std::vector<int>      x(std::max(static_cast<std::size_t>(isize) % max_size, min_size));

      PetscFunctionBegin;
      PetscCallCXXAbort(PETSC_COMM_SELF, std::generate(x.begin(), x.end(), make_int));
      auto ret = Bar{std::move(x), std::to_string(isize)};
      PetscFunctionReturn(ret);
    };

    const auto int_double_generator = [&] { return std::make_pair(make_int(), make_double()); };
    PetscCall(make_tester<int, double>(vwr, "int-double basic map", int_printer, double_printer, int_double_generator).test());
    PetscCall(make_tester<int, double, BadHash>(vwr, "int-double bad hash map", int_printer, double_printer, int_double_generator).test());

    const auto int_foo_generator = [&] { return std::make_pair(make_int(), make_foo()); };
    PetscCall(make_tester<int, Foo, BadHash>(vwr, "int-foo bad hash map", int_printer, foo_printer, int_foo_generator).test());

    const auto foo_bar_generator = [&] { return std::make_pair(make_foo(), make_bar()); };
    PetscCall(make_tester<Foo, Bar>(vwr, "foo-bar basic map", foo_printer, bar_printer, foo_bar_generator).test());
    PetscCall(make_tester<Foo, Bar, BadHash>(vwr, "foo-bar bad hash map", foo_printer, bar_printer, foo_bar_generator).test());

    // these test that the indirect_hasher and indirect_equals classes don't barf, since the
    // value_type of the map and hashers is both the same thing
    const auto pair_pair_generator = [&] {
      auto pair = std::make_pair(make_int(), make_double());
      return std::make_pair(pair, pair);
    };
    PetscCall(make_tester<std::pair<int, double>, std::pair<int, double>>(vwr, "pair<int, double>-pair<int, double> basic map", pair_printer, pair_printer, pair_pair_generator).test());
    PetscCall(make_tester<std::pair<int, double>, std::pair<int, double>, BadHash>(vwr, "pair<int, double>-pair<int, double> bad hash map", pair_printer, pair_printer, pair_pair_generator).test());
  }

  PetscCall(PetscRandomDestroy(&rand));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: umap_0

TEST*/
