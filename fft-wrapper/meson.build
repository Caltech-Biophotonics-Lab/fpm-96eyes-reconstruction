cuda_runtime_dep = dependency('cuda',
    modules: [
        'cuda',
        'cufft',
        'nvToolsExt',
    ],
    required: false,
)

if not cuda_runtime_dep.found()
    error('CUDA toolkit is not installed.')
endif

cufft_wrapper_lib = static_library('cufft_wrapper',
    include_directories: 'inc',
    sources: [
        'src/cuda_batch_fft2d.cpp',
        'src/cuda-context.cpp',
    ],
    dependencies: [
        cuda_runtime_dep,
    ],
)

cufft_wrapper_dep = declare_dependency(
    include_directories: 'inc',
    link_with: cufft_wrapper_lib,
    dependencies: [
        cuda_runtime_dep,
    ],
)

test_cufft_exe = executable('test-cufft',
    sources: [
        'tests/test-fft2.cpp',
    ],
    include_directories: common_inc,
    dependencies: [
        cufft_wrapper_dep,
        catch2_dep,
        armadillo_dep,
    ],
)

test('CuFFT wrapper', test_cufft_exe,
    args: ['-r', 'tap'],
    protocol: 'tap',
)
