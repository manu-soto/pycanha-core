# Third-party licenses

pycanha-core is released under the [MIT License](../LICENSE).

The libraries below are compiled or linked into pycanha-core, so their license
files must travel with any binary built from it (for example the `pycanha-core`
Python wheel). Each folder holds the project's own license files, copied
unmodified from its repository at the version listed.

| Library | Version | License | Files |
|---|---|---|---|
| [Eigen](https://gitlab.com/libeigen/eigen) | 5.0.1 | MPL-2.0 (some files Apache-2.0, BSD-3-Clause, Minpack) | [eigen/](eigen/) |
| [HDF5](https://github.com/HDFGroup/hdf5) | 1.14.6 | BSD-3-Clause | [hdf5/](hdf5/) |
| [zlib](https://github.com/madler/zlib) (used by HDF5) | 1.3.1 | Zlib | [zlib/](zlib/) |
| [SymEngine](https://github.com/symengine/symengine) | 0.15.0 | MIT, plus the BSD/MIT code it bundles | [symengine/](symengine/) |
| [Boost](https://github.com/boostorg/boost) (Multiprecision, used by SymEngine) | 1.86.0 | BSL-1.0 | [boost/](boost/) |
| [fast_float](https://github.com/fastfloat/fast_float) (used by SymEngine) | 6.1.5 | MIT or Apache-2.0 or BSL-1.0 | [fast_float/](fast_float/) |
| [spdlog](https://github.com/gabime/spdlog) | 1.17.0 | MIT | [spdlog/](spdlog/) |
| [Manifold](https://github.com/elalish/manifold) | 3.5.1 | Apache-2.0 | [manifold/](manifold/) |
| [Clipper2](https://github.com/AngusJohnson/Clipper2) (used by Manifold) | 2.0.1 | BSL-1.0 | [clipper2/](clipper2/) |
| [Vulkan-Headers](https://github.com/KhronosGroup/Vulkan-Headers) | 1.4.357.0 | Apache-2.0 or MIT | [vulkan-headers/](vulkan-headers/) |
| [volk](https://github.com/zeux/volk) | 1.4.357.0 | MIT | [volk/](volk/) |
| [Vulkan Memory Allocator](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator) | 3.3.0 | MIT | [vulkan-memory-allocator/](vulkan-memory-allocator/) |

Eigen's source is available at the link above, as MPL-2.0 asks.

## Keeping this up to date

When a dependency version changes in `conanfile.py`:

1. Update its row above.
2. Download its license files again from the repository at the new tag and
   replace the ones in its folder. If the text changed, read the diff: a new
   license or a new bundled component may add an obligation.
3. Check whether the bump added or removed a library compiled into
   pycanha-core (`conan graph info .` lists the host dependencies; Boost,
   fast_float, zlib and Clipper2 come in through other packages). Add or
   remove the row and the folder to match.

Only code that ends up inside pycanha-core is listed. Tools used to build,
test or document it, and libraries loaded from the user's system at run time
(the Vulkan driver, MKL from its own wheel, macOS frameworks), are not.
