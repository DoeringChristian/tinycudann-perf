{
  description = "A very basic flake";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        system = system;
        config.allowUnfree = true;
        config.cudaSupport = true;
        config.cudaVersion = "12";
        overlays = [
          (final: prev: {
            matplotlib = prev.matplotlib.override { enableGtk3 = true; };
          })
        ];
      };

      cuda-common-redist = with pkgs.cudaPackages; [
        cuda_cudart.dev # cuda_runtime.h
        cuda_cudart.lib
        cuda_cccl.dev # <nv/target>
        libcublas.dev # cublas_v2.h
        libcublas.lib
        libcusolver.dev # cusolverDn.h
        libcusolver.lib
        libcusparse.dev # cusparse.h
        libcusparse.lib
        cudatoolkit
      ];

      cuda-native-redist = pkgs.symlinkJoin {
        name = "cuda-redist";
        paths = with pkgs.cudaPackages;
          [ cuda_nvcc ]
          ++ cuda-common-redist;
      };
      backendStdenv = pkgs.cudaPackages.backendStdenv;
    in
    {
      # devShells.${system}.default = fhs.env;
      devShells.${system}.default = pkgs.mkShell {
        packages = [
          backendStdenv
          pkgs.glslang
          pkgs.vulkan-headers
          pkgs.vulkan-loader
          pkgs.vulkan-validation-layers

          pkgs.gobject-introspection
          pkgs.gtk3

          pkgs.texlive.combined.scheme-medium
          pkgs.dejavu_fonts
          # pkgs.texliveSmall

          (pkgs.python3.withPackages (ps: with ps;[
            pip
            virtualenv
            setuptools
            ipython
            pygobject3
            # pygobject3
            # matplotlib
          ]))
        ] ++ [ cuda-native-redist ];
        shellHook = ''
          export CUDA_HOME="${cuda-native-redist}"
          export LIBRARY_PATH="${cuda-native-redist}/lib/stubs:$LIBRARY_PATH"
          export CC="${backendStdenv.cc}/bin/cc"
          export CXX="${backendStdenv.cc}/bin/c++"
          export LD_LIBRARY_PATH="${backendStdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH"
          export LD_LIBRARY_PATH="${cuda-native-redist}/lib:$LD_LIBRARY_PATH"
          export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/lib64"
        '';
      };
    };

}
