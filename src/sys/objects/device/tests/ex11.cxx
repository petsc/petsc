static const char help[] = "Tests PetscDeviceContextMarkIntentFromID().\n\n";

#include "petscdevicetestcommon.h"
#include <petscviewer.h>

#include <petsc/private/cpp/type_traits.hpp>
#include <petsc/private/cpp/array.hpp>

#include <cstdarg>       // std::va_list
#include <vector>        // std:vector
#include <unordered_map> // std::take_a_wild_guess
#include <algorithm>     // std::find
#include <iterator>      // std::distance, std::next

struct Marker {
  PetscMemoryAccessMode mode{};

  PetscErrorCode operator()(PetscDeviceContext dctx, PetscContainer cont) const noexcept
  {
    const auto    obj  = reinterpret_cast<PetscObject>(cont);
    PetscObjectId id   = 0;
    const char   *name = nullptr;

    PetscFunctionBegin;
    PetscCall(PetscObjectGetId(obj, &id));
    PetscCall(PetscObjectGetName(obj, &name));
    PetscCall(PetscDeviceContextMarkIntentFromID(dctx, id, this->mode, name));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
};

static constexpr auto mem_read       = Marker{PETSC_MEMORY_ACCESS_READ};
static constexpr auto mem_write      = Marker{PETSC_MEMORY_ACCESS_WRITE};
static constexpr auto mem_read_write = Marker{PETSC_MEMORY_ACCESS_READ_WRITE};
static constexpr auto mark_funcs     = Petsc::util::make_array(mem_read, mem_write, mem_read_write);

static PetscErrorCode MarkedObjectMapView(PetscViewer vwr, std::size_t nkeys, const PetscObjectId *keys, const PetscMemoryAccessMode *modes, const std::size_t *ndeps, const PetscEvent **dependencies)
{
  PetscFunctionBegin;
  if (!vwr) PetscCall(PetscViewerASCIIGetStdout(PETSC_COMM_WORLD, &vwr));
  PetscCall(PetscViewerFlush(vwr));
  PetscCall(PetscViewerASCIIPushSynchronized(vwr));
  PetscCall(PetscViewerASCIISynchronizedPrintf(vwr, "Marked Object Map:\n"));
  PetscCall(PetscViewerASCIIPushTab(vwr));
  PetscCall(PetscViewerASCIISynchronizedPrintf(vwr, "size: %zu\n", nkeys));
  PetscCall(PetscViewerASCIISynchronizedPrintf(vwr, "entries:\n"));
  PetscCall(PetscViewerASCIIPushTab(vwr));
  for (std::size_t i = 0; i < nkeys; ++i) {
    PetscCall(PetscViewerASCIISynchronizedPrintf(vwr, "id %" PetscInt64_FMT " -> {\n", keys[i]));
    PetscCall(PetscViewerASCIIPushTab(vwr));
    PetscCall(PetscViewerASCIISynchronizedPrintf(vwr, "mode: %s\n", PetscMemoryAccessModeToString(modes[i])));
    PetscCall(PetscViewerASCIISynchronizedPrintf(vwr, "dependencies:\n"));
    PetscCall(PetscViewerASCIIPushTab(vwr));
    for (std::size_t j = 0; j < ndeps[i]; ++j) {
      const auto event = dependencies[i][j];

      PetscCall(PetscViewerASCIISynchronizedPrintf(vwr, "event %zu {dtype: %s, dctx_id: %" PetscInt64_FMT ", dctx_state: %" PetscInt64_FMT ", data: %p, destroy: %p}\n", j, PetscDeviceTypes[event->dtype], event->dctx_id, event->dctx_state, event->data,
                                                   reinterpret_cast<void *>(event->destroy)));
    }
    PetscCall(PetscViewerASCIIPopTab(vwr));
    PetscCall(PetscViewerASCIIPopTab(vwr));
    PetscCall(PetscViewerASCIISynchronizedPrintf(vwr, "}\n"));
  }
  PetscCall(PetscViewerASCIIPopTab(vwr));
  PetscCall(PetscViewerASCIIPopTab(vwr));
  PetscCall(PetscViewerFlush(vwr));
  PetscCall(PetscViewerASCIIPopSynchronized(vwr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_ATTRIBUTE_FORMAT(10, 11)
static PetscErrorCode CheckMarkedObjectMap_Private(PetscBool cond, const char cond_str[], MPI_Comm comm, PetscDeviceContext dctx, std::size_t nkeys, const PetscObjectId *keys, const PetscMemoryAccessMode *modes, const std::size_t *ndeps, const PetscEvent **dependencies, const char *format, ...)
{
  PetscFunctionBegin;
  if (PetscUnlikely(!cond)) {
    std::array<char, 2048> buf;
    std::va_list           argp;
    std::size_t            len;
    PetscViewer            vwr;

    PetscCallCXX(buf.fill(0));
    va_start(argp, format);
    PetscCall(PetscVSNPrintf(buf.data(), buf.size(), format, &len, argp));
    va_end(argp);
    PetscCall(PetscViewerASCIIGetStdout(comm, &vwr));
    if (dctx) PetscCall(PetscDeviceContextView(dctx, vwr));
    PetscCall(MarkedObjectMapView(vwr, nkeys, keys, modes, ndeps, dependencies));
    SETERRQ(comm, PETSC_ERR_PLIB, "Condition '%s' failed, marked object map in corrupt state: %s", cond_str, buf.data());
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
#define CheckMarkedObjectMap(__cond__, ...) CheckMarkedObjectMap_Private((PetscBool)(!!(__cond__)), PetscStringize(__cond__), PETSC_COMM_SELF, dctx, nkeys, keys, modes, ndeps, const_cast<const PetscEvent **>(dependencies), __VA_ARGS__);

static PetscErrorCode TestAllCombinations(PetscDeviceContext dctx, const std::vector<PetscContainer> &cont)
{
  std::vector<PetscObjectId> cont_ids;
  PetscObjectId              dctx_id;
  PetscDeviceType            dtype;

  PetscFunctionBegin;
  PetscCallCXX(cont_ids.reserve(cont.size()));
  for (auto &&c : cont) {
    PetscObjectId id;

    PetscCall(PetscObjectGetId((PetscObject)c, &id));
    PetscCallCXX(cont_ids.emplace_back(id));
  }
  PetscCall(PetscObjectGetId(PetscObjectCast(dctx), &dctx_id));
  PetscCall(PetscDeviceContextGetDeviceType(dctx, &dtype));
  for (auto &&func_i : mark_funcs) {
    for (auto &&func_j : mark_funcs) {
      for (auto it = cont.cbegin(), next = std::next(it); it != cont.cend(); ++it, ++next) {
        std::vector<int>       found_keys;
        std::size_t            nkeys;
        PetscObjectId         *keys;
        PetscMemoryAccessMode *modes;
        std::size_t           *ndeps;
        PetscEvent           **dependencies;

        if (next >= cont.cend()) next = cont.cbegin();
        PetscCall(func_i(dctx, *it));
        PetscCall(func_j(dctx, *next));
        PetscCall(PetscGetMarkedObjectMap_Internal(&nkeys, &keys, &modes, &ndeps, &dependencies));
        PetscCallCXX(found_keys.resize(nkeys));
        {
          // The underlying marked object map is *unordered*, and hence the order in which we
          // get the keys is not necessarily the same as the order of operations. This is
          // confounded by the fact that k and knext are not necessarily "linear", i.e. k could
          // be 2 while knext is 0. So we need to map these back to linear space so we can loop
          // over them.
          const auto keys_end           = keys + nkeys;
          const auto num_expected_keys  = std::min(cont.size(), static_cast<std::size_t>(2));
          const auto check_applied_mode = [&](PetscContainer container, PetscMemoryAccessMode mode) {
            std::ptrdiff_t key_idx = 0;
            PetscObjectId  actual_key;

            PetscFunctionBegin;
            PetscCall(PetscObjectGetId((PetscObject)container, &actual_key));
            // search the list of keys from the map for the selected key
            key_idx = std::distance(keys, std::find(keys, keys_end, actual_key));
            PetscCheck(key_idx >= 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Key index %" PetscCount_FMT " < 0, this indicates keys_begin > keys_end?", key_idx);
            found_keys[key_idx]++;
            PetscCall(CheckMarkedObjectMap(key_idx < std::distance(keys, keys_end), "marked object map could not find expected key %" PetscInt64_FMT, actual_key));
            // OK found it, now check the rest of the entries are as we expect them to be
            PetscCall(CheckMarkedObjectMap(modes[key_idx] == mode, "unexpected mode %s, expected %s", PetscMemoryAccessModeToString(modes[key_idx]), PetscMemoryAccessModeToString(mode)));
            PetscCall(CheckMarkedObjectMap(ndeps[key_idx] == 1, "unexpected number of dependencies %zu, expected 1", ndeps[key_idx]));
            PetscCall(CheckMarkedObjectMap(dependencies[key_idx][0]->dtype == dtype, "unexpected device type on event: %s, expected %s", PetscDeviceTypes[dependencies[key_idx][0]->dtype], PetscDeviceTypes[dtype]));
            PetscFunctionReturn(PETSC_SUCCESS);
          };

          // if it == next, then even though we might num_expected_keys keys we never "look
          // for" the missing key
          PetscCheck(cont.size() == 1 || it != next, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Test assumes different inputs, otherwise key check may fail (cont.size(): %zu, it != next: %s)", cont.size(), it != next ? "true" : "false");
          PetscCall(CheckMarkedObjectMap(nkeys == num_expected_keys, "marked object map has %zu keys expected %zu", nkeys, num_expected_keys));
          // check that each function properly applied its mode, it == next if cont.size() = 1,
          // i.e. testing identity
          if (it != next) PetscCall(check_applied_mode(*it, func_i.mode));
          PetscCall(check_applied_mode(*next, func_j.mode));
        }
        // Check that the map contained only keys we were looking for. Any extra keys will have
        // zero find count
        for (auto it = found_keys.cbegin(); it != found_keys.cend(); ++it) PetscCall(CheckMarkedObjectMap(*it > 0, "Marked Object Map has extra object entry: id %" PetscInt64_FMT, keys[std::distance(found_keys.cbegin(), it)]));

        PetscCall(PetscRestoreMarkedObjectMap_Internal(nkeys, &keys, &modes, &ndeps, &dependencies));

        PetscCall(PetscDeviceContextSynchronize(dctx));
        PetscCall(PetscGetMarkedObjectMap_Internal(&nkeys, &keys, &modes, &ndeps, &dependencies));
        PetscCall(CheckMarkedObjectMap(nkeys == 0, "synchronizing device context did not empty dependency map, have %zu keys", nkeys));
        PetscCall(PetscRestoreMarkedObjectMap_Internal(nkeys, &keys, &modes, &ndeps, &dependencies));
      }
    }
  }
  PetscCall(PetscDeviceContextSynchronize(dctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <typename... T>
PETSC_NODISCARD static std::pair<PetscObjectId, std::pair<PetscMemoryAccessMode, std::vector<PetscDeviceContext>>> make_map_entry(PetscObjectId id, PetscMemoryAccessMode mode, T &&...dctxs)
{
  return {
    id, {mode, {std::forward<T>(dctxs)...}}
  };
}

static PetscErrorCode CheckMapEqual(std::unordered_map<PetscObjectId, std::pair<PetscMemoryAccessMode, std::vector<PetscDeviceContext>>> expected_map)
{
  std::size_t            nkeys;
  PetscObjectId         *keys;
  PetscMemoryAccessMode *modes;
  std::size_t           *ndeps;
  PetscEvent           **dependencies;
  PetscDeviceContext     dctx = nullptr;

  PetscFunctionBegin;
  PetscCall(PetscGetMarkedObjectMap_Internal(&nkeys, &keys, &modes, &ndeps, &dependencies));
  {
    const auto key_end = keys + nkeys;
    auto       mode_it = modes;
    auto       ndep_it = ndeps;
    auto       dep_it  = dependencies;

    for (auto key_it = keys; key_it != key_end; ++key_it, ++mode_it, ++ndep_it, ++dep_it) {
      const auto found_it = expected_map.find(*key_it);

      PetscCall(CheckMarkedObjectMap(found_it != expected_map.cend(), "marked object map did not contain key %" PetscInt64_FMT, *key_it));
      {
        // must do these here since found_it may be expected_map.cend()
        const auto &expected_mode  = found_it->second.first;
        const auto &expected_dctxs = found_it->second.second;
        auto        sub_dep_it     = *dep_it;

        PetscCall(CheckMarkedObjectMap(expected_mode == *mode_it, "unexpected mode %s, expected %s", PetscMemoryAccessModeToString(expected_mode), PetscMemoryAccessModeToString(*mode_it)));
        PetscCall(CheckMarkedObjectMap(expected_dctxs.size() == *ndep_it, "unexpected number of dependencies %zu, expected %zu", *ndep_it, expected_dctxs.size()));
        // purposefully hide "dctx" with the loop variable, so we get more detailed output in
        // the error message
        for (auto &&dctx : expected_dctxs) {
          const auto      event = *sub_dep_it;
          PetscDeviceType dtype;
          PetscObjectId   id;

          PetscCall(PetscDeviceContextGetDeviceType(dctx, &dtype));
          PetscCall(PetscObjectGetId(PetscObjectCast(dctx), &id));
          PetscCall(CheckMarkedObjectMap(event->dtype == dtype, "unexpected device type on event: %s, expected %s", PetscDeviceTypes[event->dtype], PetscDeviceTypes[dtype]));
          PetscCall(CheckMarkedObjectMap(event->dctx_id == id, "unexpected dctx id on event: %" PetscInt64_FMT ", expected %" PetscInt64_FMT, event->dctx_id, id));
          ++sub_dep_it;
        }
      }
      // remove the found iterator from the map, this ensure we either run out of map (which is
      // caught by the first check in the loop), or we run out of keys to check, which is
      // caught in the end of the loop
      PetscCallCXX(expected_map.erase(found_it));
    }
  }
  PetscCall(CheckMarkedObjectMap(expected_map.empty(), "Not all keys in marked object map accounted for!"));
  PetscCall(PetscRestoreMarkedObjectMap_Internal(nkeys, &keys, &modes, &ndeps, &dependencies));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char *argv[])
{
  PetscContainer     x, y, z;
  PetscObjectId      x_id, y_id, z_id;
  PetscDeviceContext dctx_a, dctx_b, dctx_c;
  auto               container_view   = PETSC_FALSE;
  const auto         create_container = [&](PetscContainer *c, const char name[], PetscObjectId *id) {
    PetscFunctionBegin;
    PetscCall(PetscContainerCreate(PETSC_COMM_WORLD, c));
    PetscCall(PetscObjectSetName((PetscObject)(*c), name));
    PetscCall(PetscObjectGetId((PetscObject)(*c), id));
    if (container_view) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Container '%s' -> id %" PetscInt64_FMT "\n", name, *id));
    PetscFunctionReturn(PETSC_SUCCESS);
  };
  const auto sync_all = [&] {
    PetscFunctionBegin;
    for (auto &&ctx : {dctx_a, dctx_b, dctx_c}) PetscCall(PetscDeviceContextSynchronize(ctx));
    PetscFunctionReturn(PETSC_SUCCESS);
  };

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));

  PetscOptionsBegin(PETSC_COMM_WORLD, nullptr, "Test Options", "Sys");
  PetscCall(PetscOptionsBool("-container_view", "View container names and ID's", nullptr, container_view, &container_view, nullptr));
  PetscOptionsEnd();

  PetscCall(create_container(&x, "x", &x_id));
  PetscCall(create_container(&y, "y", &y_id));
  PetscCall(create_container(&z, "z", &z_id));

  PetscCall(PetscDeviceContextCreate(&dctx_a));
  PetscCall(PetscObjectSetName(PetscObjectCast(dctx_a), "dctx_a"));
  PetscCall(PetscDeviceContextSetStreamType(dctx_a, PETSC_STREAM_DEFAULT_BLOCKING));
  PetscCall(PetscDeviceContextSetFromOptions(PETSC_COMM_WORLD, dctx_a));
  PetscCall(PetscDeviceContextDuplicate(dctx_a, &dctx_b));
  PetscCall(PetscObjectSetName(PetscObjectCast(dctx_b), "dctx_b"));
  PetscCall(PetscDeviceContextDuplicate(dctx_a, &dctx_c));
  PetscCall(PetscObjectSetName(PetscObjectCast(dctx_c), "dctx_c"));
  PetscCall(PetscDeviceContextViewFromOptions(dctx_a, nullptr, "-dctx_a_view"));
  PetscCall(PetscDeviceContextViewFromOptions(dctx_b, nullptr, "-dctx_b_view"));
  PetscCall(PetscDeviceContextViewFromOptions(dctx_c, nullptr, "-dctx_c_view"));

  // ensure they are all idle
  PetscCall(sync_all());
  PetscCall(CheckMapEqual({}));

  // do the bulk combination tests, these test only the very basic combinations for simple
  // correctness
  PetscCall(TestAllCombinations(dctx_a, {x}));
  PetscCall(TestAllCombinations(dctx_a, {x, y, z}));

  // Now do some specific tests, these should test more complicated scenarios. First and
  // foremost, ensure they are all idle, and that it does not change the map
  PetscCall(sync_all());
  // Map should be empty
  PetscCall(CheckMapEqual({}));

  // Syncing again shouldn't magically fill the map back up
  PetscCall(sync_all());
  PetscCall(CheckMapEqual({}));

  const auto test_multiple_readers = [&](std::array<PetscDeviceContext, 2> readers, std::size_t sync_idx) {
    // the reader which synchronizes
    const auto sync_reader = readers[sync_idx];
    // the reader that will remain in the map after sync_reader synchronizes
    const auto remain_idx    = sync_idx + 1 >= readers.size() ? 0 : sync_idx + 1;
    const auto remain_reader = readers[remain_idx];

    PetscFunctionBegin;
    for (auto &&ctx : readers) PetscCall(mem_read(ctx, x));
    for (auto &&ctx : readers) PetscCall(mem_read(ctx, y));
    PetscCall(CheckMapEqual({
      make_map_entry(x_id, PETSC_MEMORY_ACCESS_READ, readers[0], readers[1]),
      make_map_entry(y_id, PETSC_MEMORY_ACCESS_READ, readers[0], readers[1]),
    }));
    // synchronizing sync_reader should remove it from the dependency list -- but leave remain_reader
    // intact
    PetscCall(PetscDeviceContextSynchronize(sync_reader));
    PetscCall(CheckMapEqual({
      make_map_entry(x_id, PETSC_MEMORY_ACCESS_READ, remain_reader),
      make_map_entry(y_id, PETSC_MEMORY_ACCESS_READ, remain_reader),
    }));
    PetscCall(PetscDeviceContextSynchronize(remain_reader));
    PetscCall(CheckMapEqual({}));
    PetscFunctionReturn(PETSC_SUCCESS);
  };

  // Test that multiple readers can simultaneously read -- even if one of them is synchronized
  PetscCall(test_multiple_readers({dctx_a, dctx_b}, 0));
  PetscCall(test_multiple_readers({dctx_a, dctx_b}, 1));

  // Test that sync of unrelated ctx does not affect the map
  PetscCall(mem_read(dctx_a, x));
  PetscCall(mem_read(dctx_b, y));
  PetscCall(PetscDeviceContextSynchronize(dctx_c));
  // clang-format off
  PetscCall(CheckMapEqual({
    make_map_entry(x_id, PETSC_MEMORY_ACCESS_READ, dctx_a),
    make_map_entry(y_id, PETSC_MEMORY_ACCESS_READ, dctx_b)
  }));
  // clang-format on
  PetscCall(PetscDeviceContextSynchronize(dctx_a));
  PetscCall(PetscDeviceContextSynchronize(dctx_b));
  // Now the map is empty again
  PetscCall(CheckMapEqual({}));

  // Test another context writing over two reads
  PetscCall(mem_read(dctx_a, x));
  PetscCall(mem_read(dctx_b, x));
  // C writing should kick out both A and B
  PetscCall(mem_write(dctx_c, x));
  PetscCall(CheckMapEqual({make_map_entry(x_id, PETSC_MEMORY_ACCESS_WRITE, dctx_c)}));
  PetscCall(PetscDeviceContextSynchronize(dctx_c));
  PetscCall(CheckMapEqual({}));

  // Test that write and synchronize does not interfere with unrelated read
  PetscCall(mem_read_write(dctx_a, x));
  PetscCall(mem_read(dctx_a, y));
  PetscCall(mem_read_write(dctx_b, x));
  PetscCall(mem_read(dctx_b, y));
  // Synchronizing B here must clear everything *but* A's read on Y!
  PetscCall(PetscDeviceContextSynchronize(dctx_b));
  PetscCall(CheckMapEqual({make_map_entry(y_id, PETSC_MEMORY_ACCESS_READ, dctx_a)}));
  PetscCall(PetscDeviceContextSynchronize(dctx_a));
  // Now the map is empty again
  PetscCall(CheckMapEqual({}));

  // Test that implicit stream-dependencies are properly tracked
  PetscCall(mem_read(dctx_a, x));
  PetscCall(mem_read(dctx_b, y));
  // A waits for B
  PetscCall(PetscDeviceContextWaitForContext(dctx_a, dctx_b));
  // Because A waits on B, synchronizing A implicitly implies B read must have finished so the
  // map must be empty
  PetscCall(PetscDeviceContextSynchronize(dctx_a));
  PetscCall(CheckMapEqual({}));

  PetscCall(mem_write(dctx_a, x));
  PetscCall(CheckMapEqual({make_map_entry(x_id, PETSC_MEMORY_ACCESS_WRITE, dctx_a)}));
  PetscCall(PetscDeviceContextWaitForContext(dctx_b, dctx_a));
  PetscCall(PetscDeviceContextWaitForContext(dctx_c, dctx_b));
  // We have created the chain C -> B -> A, so synchronizing C should trickle down to synchronize and
  // remove A from the map
  PetscCall(PetscDeviceContextSynchronize(dctx_c));
  PetscCall(CheckMapEqual({}));

  // Test that superfluous stream-dependencies are properly ignored
  PetscCall(mem_read(dctx_a, x));
  PetscCall(mem_read(dctx_b, y));
  PetscCall(PetscDeviceContextWaitForContext(dctx_c, dctx_b));
  // C waited on B, so synchronizing C should remove B from the map but *not* remove A
  PetscCall(PetscDeviceContextSynchronize(dctx_c));
  PetscCall(CheckMapEqual({make_map_entry(x_id, PETSC_MEMORY_ACCESS_READ, dctx_a)}));
  PetscCall(PetscDeviceContextSynchronize(dctx_a));
  PetscCall(CheckMapEqual({}));

  // Test that read->write correctly wipes out the map
  PetscCall(mem_read(dctx_a, x));
  PetscCall(mem_read(dctx_b, x));
  PetscCall(mem_read(dctx_c, x));
  PetscCall(CheckMapEqual({make_map_entry(x_id, PETSC_MEMORY_ACCESS_READ, dctx_a, dctx_b, dctx_c)}));
  PetscCall(mem_write(dctx_a, x));
  PetscCall(CheckMapEqual({make_map_entry(x_id, PETSC_MEMORY_ACCESS_WRITE, dctx_a)}));
  PetscCall(PetscDeviceContextSynchronize(dctx_a));
  PetscCall(CheckMapEqual({}));

  PetscCall(PetscDeviceContextDestroy(&dctx_a));
  PetscCall(PetscDeviceContextDestroy(&dctx_b));
  PetscCall(PetscDeviceContextDestroy(&dctx_c));

  PetscCall(PetscContainerDestroy(&x));
  PetscCall(PetscContainerDestroy(&y));
  PetscCall(PetscContainerDestroy(&z));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "EXIT_SUCCESS\n"));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  testset:
    requires: cxx
    output_file: ./output/ExitSuccess.out
    test:
      requires: !device
      suffix: host_no_device
    test:
      requires: device
      args: -default_device_type host
      suffix: host_with_device
    test:
      requires: cuda
      args: -default_device_type cuda
      suffix: cuda
    test:
      requires: hip
      args: -default_device_type hip
      suffix: hip
    test:
      requires: sycl
      args: -default_device_type sycl
      suffix: sycl

TEST*/
