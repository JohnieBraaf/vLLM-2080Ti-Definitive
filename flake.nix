{
  description = "vLLM for SM70/SM75 GPUs (GTX 1080 Ti, RTX 2080 Ti) — NixOS/nixpkgs package";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-26.05";

  outputs =
    { self, nixpkgs }:
    let
      system = "x86_64-linux";

      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };

      cudaPkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
        config.cudaSupport = true;
      };

      cudaPackages = cudaPkgs.cudaPackages_13_0;
      python3 = cudaPkgs.python312;

      # ── CUDA home: merged tree used as CUDA_HOME during builds ──────────────

      cudaHome = pkgs.symlinkJoin {
        name = "cuda-home-13_0";
        paths = with cudaPackages; [
          cuda_cudart
          cuda_crt
          cuda_nvcc
          libcublas
          libcublas.include
          cuda_cccl
          cuda_nvtx
          cuda_nvrtc
          libcusparse
          libcusparse.include
          libcusolver
          libcusolver.include
          libcurand
          libcurand.include
          libcufft
          libcufft.include
          cudnn
          libnvjitlink
          nccl
        ];
        postBuild = ''
          ln -sf ${cudaPackages.cuda_cudart}/lib/stubs/libcuda.so $out/lib/libcuda.so
          ln -sf ${cudaPackages.cuda_cudart}/lib/stubs/libcuda.so.1 $out/lib/libcuda.so.1
          ln -sf $out/lib $out/lib64
        '';
      };

      # ── Shared wheel builder ─────────────────────────────────────────────────

      mkWheelPkg =
        {
          pname,
          version,
          url,
          hash,
          propagatedBuildInputs ? [ ],
          extraBuildInputs ? [ ],
          extraPostInstall ? "",
          py ? python3,
        }:
        py.pkgs.buildPythonPackage {
          inherit pname version propagatedBuildInputs;
          format = "wheel";
          src = pkgs.fetchurl { inherit url hash; };
          nativeBuildInputs = [
            pkgs.addDriverRunpath
            pkgs.autoPatchelfHook
            pkgs.patchelf
          ];
          buildInputs =
            extraBuildInputs
            ++ (with cudaPackages; [
              pkgs.gcc15.cc.lib
              cuda_cudart
              libcublas
              libcufft
              libcurand
              libcusolver
              libcusparse
              cudnn
              libcufile
              cuda_nvrtc
              libnvjitlink
              cuda_cupti
            ]);
          autoPatchelfIgnoreMissingDeps = true;
          postInstall = ''
            find $out -name "*.so*" -exec addDriverRunpath {} \; || true
            ${extraPostInstall}
          '';
          doCheck = false;
          doInstallCheck = false;
          dontCheckRuntimeDeps = true;
        };

      # ── NCCL / cuSPARSELt / NVSHMEM companion wheels (torch 2.13 requires) ──

      nvidia-nccl-cu13 = python3.pkgs.buildPythonPackage {
        pname = "nvidia-nccl-cu13";
        version = "2.29.7";
        format = "wheel";
        src = pkgs.fetchurl {
          url = "https://files.pythonhosted.org/packages/67/f4/58e4e91b6919367c7aafb8e36fce9aad1a3047e536bf7e2fd560927d3a4c/nvidia_nccl_cu13-2.29.7-py3-none-manylinux_2_18_x86_64.whl";
          hash = "sha256-7dgVOERnhuw7c5clQ+U7tDvK8L/I73bLZ5/MOQ/+E20=";
        };
        nativeBuildInputs = [
          pkgs.autoPatchelfHook
          pkgs.addDriverRunpath
        ];
        buildInputs = [
          pkgs.gcc15.cc.lib
          cudaPackages.cuda_cudart
        ];
        autoPatchelfIgnoreMissingDeps = true;
        postInstall = ''
          SP=$out/lib/python3.12/site-packages
          mkdir -p $out/lib
          find $SP/nvidia/nccl/lib -name "*.so*" -exec cp -P {} $out/lib/ \;
          find $out/lib -name "*.so*" -exec addDriverRunpath {} \; || true
        '';
        doCheck = false;
        doInstallCheck = false;
        dontCheckRuntimeDeps = true;
      };

      nvidia-cusparselt-cu13 = python3.pkgs.buildPythonPackage {
        pname = "nvidia-cusparselt-cu13";
        version = "0.8.1";
        format = "wheel";
        src = pkgs.fetchurl {
          url = "https://files.pythonhosted.org/packages/34/7d/2661f2fb3ac4302f3a246f5fc030213ac60c1fe0bce84f9783dbd831dbb7/nvidia_cusparselt_cu13-0.8.1-py3-none-manylinux2014_x86_64.whl";
          hash = "sha256-eGzodWjDA/rbWvzHEC1FTNMEDXX2+GJvXbRg0YcfTdA=";
        };
        nativeBuildInputs = [
          pkgs.autoPatchelfHook
          pkgs.addDriverRunpath
        ];
        buildInputs = with cudaPackages; [
          pkgs.gcc15.cc.lib
          cuda_cudart
          libcusparse
        ];
        autoPatchelfIgnoreMissingDeps = true;
        postInstall = ''
          SP=$out/lib/python3.12/site-packages
          mkdir -p $out/lib
          find $SP/nvidia/cusparselt/lib -name "*.so*" -exec cp -P {} $out/lib/ \;
          find $out/lib -name "*.so*" -exec addDriverRunpath {} \; || true
        '';
        doCheck = false;
        doInstallCheck = false;
        dontCheckRuntimeDeps = true;
      };

      nvidia-nvshmem-cu13 = python3.pkgs.buildPythonPackage {
        pname = "nvidia-nvshmem-cu13";
        version = "3.4.5";
        format = "wheel";
        src = pkgs.fetchurl {
          url = "https://files.pythonhosted.org/packages/3c/35/a9bf80a609e74e3b000fef598933235c908fcefcef9026042b8e6dfde2a9/nvidia_nvshmem_cu13-3.4.5-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl";
          hash = "sha256-KQ8KLulMnzaHoCUC87kpmp+f6Cbm0Ch+4YSC541JW4A=";
        };
        nativeBuildInputs = [
          pkgs.autoPatchelfHook
          pkgs.addDriverRunpath
        ];
        buildInputs = with cudaPackages; [
          pkgs.gcc15.cc.lib
          cuda_cudart
          nccl
        ];
        autoPatchelfIgnoreMissingDeps = true;
        postInstall = ''
          SP=$out/lib/python3.12/site-packages
          mkdir -p $out/lib
          find $SP/nvidia/nvshmem/lib -name "*.so*" -exec cp -P {} $out/lib/ \; 2>/dev/null || true
          find $out/lib -name "*.so*" -exec addDriverRunpath {} \; || true
        '';
        doCheck = false;
        doInstallCheck = false;
        dontCheckRuntimeDeps = true;
      };

      # ── PyTorch 2.13+cu130 ───────────────────────────────────────────────────

      torchInitPatch = pkgs.writeText "torch-init-patch.py" ''
        import sys
        path = sys.argv[1]
        txt = open(path).read()
        txt = txt.replace(
            '        _load_global_deps()',
            '        try:\n            _load_global_deps()\n        except Exception:\n            pass'
        )
        open(path, 'w').write(txt)
        print('Patched torch __init__.py')
      '';

      torch-cu130 = mkWheelPkg {
        pname = "torch";
        version = "2.13.0+cu130";
        url = "https://download-r2.pytorch.org/whl/cu130/torch-2.13.0%2Bcu130-cp312-cp312-manylinux_2_28_x86_64.whl";
        hash = "sha256-jbczjmiVw9S9iaAv9CCVB9Hwzy/+s7iYU4taB9HqjB4=";
        extraBuildInputs = [
          nvidia-nccl-cu13
          nvidia-cusparselt-cu13
          nvidia-nvshmem-cu13
        ];
        propagatedBuildInputs = [
          nvidia-nccl-cu13
          nvidia-cusparselt-cu13
          nvidia-nvshmem-cu13
          python3.pkgs.sympy
          python3.pkgs.networkx
          python3.pkgs.filelock
          python3.pkgs.jinja2
          python3.pkgs.typing-extensions
          python3.pkgs.fsspec
        ];
        extraPostInstall = ''
          python3 ${torchInitPatch} $out/lib/python3.12/site-packages/torch/__init__.py
          TORCHLIB=$out/lib/python3.12/site-packages/torch/lib
          cp -P ${nvidia-nccl-cu13}/lib/libnccl.so.2 $TORCHLIB/libnccl.so.2
          find ${nvidia-cusparselt-cu13}/lib -name "libcusparseLt*" -exec cp -P {} $TORCHLIB/ \;
          find ${nvidia-nvshmem-cu13}/lib -name "libnvshmem*" -exec cp -P {} $TORCHLIB/ \; 2>/dev/null || true
          METADATA=$out/lib/python3.12/site-packages/torch-2.13.0+cu130.dist-info/METADATA
          if [ -f "$METADATA" ]; then sed -i '/^Requires-Dist:/d' $METADATA; fi
        '';
      };

      torchaudio-cu130 = mkWheelPkg {
        pname = "torchaudio";
        version = "2.11.0+cu130";
        url = "https://download-r2.pytorch.org/whl/cu130/torchaudio-2.11.0%2Bcu130-cp312-cp312-manylinux_2_28_x86_64.whl";
        hash = "sha256-P7qYj0MB/hNUf+XpnHbZrjaifhne2C7v/tnSRW4S7e8=";
        propagatedBuildInputs = [ torch-cu130 ];
      };

      torchvision-cu130 = mkWheelPkg {
        pname = "torchvision";
        version = "0.28.0+cu130";
        url = "https://download-r2.pytorch.org/whl/cu130/torchvision-0.28.0%2Bcu130-cp312-cp312-manylinux_2_28_x86_64.whl";
        hash = "sha256-igAI00zMToEGa5f/CuWjTGdr/fNGS69AwBsyDcmkXOA=";
        propagatedBuildInputs = [ torch-cu130 ];
      };

      # ── vLLM extra wheels ────────────────────────────────────────────────────

      apache-tvm-ffi = mkWheelPkg {
        pname = "apache-tvm-ffi";
        version = "0.1.11";
        url = "https://files.pythonhosted.org/packages/4d/18/95569107ee83619d61a3bb0d28743a0599f85c5161981e3e098c82c2b185/apache_tvm_ffi-0.1.11-cp312-abi3-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl";
        hash = "sha256-KEPwhM3JTe2s2LJXo5WitxuKPcf8mXEbFIvx0WGYMSg=";
      };

      tilelang = mkWheelPkg {
        pname = "tilelang";
        version = "0.1.12";
        url = "https://files.pythonhosted.org/packages/d1/53/f281a0bd9ee7e03d6a97828fc0e443321ed26ea2b0bd74bf9f1d9451d30f/tilelang-0.1.12-cp38-abi3-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl";
        hash = "sha256-u+tVc8viVEpRpcmmzHN8XjtcLZVBHKo2h/2bCOudX5c=";
        propagatedBuildInputs = [ apache-tvm-ffi ];
      };

      humming-kernels = mkWheelPkg {
        pname = "humming-kernels";
        version = "0.1.13";
        url = "https://files.pythonhosted.org/packages/fe/47/756faf6a9e100e511e7e8fc11e48ea7d8fd5d6a9c0962f69eca7b7d4dfce/humming_kernels-0.1.13-py3-none-manylinux_2_28_x86_64.whl";
        hash = "sha256-KrWbL50Z7kN8jazuWj5vGgFOIm39E8nXx9YdSN0zqsw=";
      };

      quack-kernels = mkWheelPkg {
        pname = "quack-kernels";
        version = "0.6.1";
        url = "https://files.pythonhosted.org/packages/2b/65/a38a30a6ac96a757363a5be9d09cef799640bb143a64ba5a2f4d400d95d9/quack_kernels-0.6.1-py3-none-any.whl";
        hash = "sha256-JmcF6oIRfpscip5E1opFhRnySY2WbA7//WgSEgw5la0=";
      };

      fastsafetensors = mkWheelPkg {
        pname = "fastsafetensors";
        version = "0.3.3";
        url = "https://files.pythonhosted.org/packages/92/8c/e3347b2a44a8ab9aced94fa450df4f309baa21f7f2981a8a7bd6a977f4d3/fastsafetensors-0.3.3-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl";
        hash = "sha256-NYe8ZrjexWCtkDvs+VQIiQE9TUfw4Qz0Xxm+3Ht7/6c=";
      };

      tokenspeed-mla = mkWheelPkg {
        pname = "tokenspeed-mla";
        version = "0.1.8";
        url = "https://files.pythonhosted.org/packages/27/df/0037ade72b165ac97859040919e006aa3d80cb8cc3a79420fb6c03eb16a0/tokenspeed_mla-0.1.8-py3-none-manylinux_2_28_x86_64.whl";
        hash = "sha256-anUm1zJ3Rok/jCDSSqY7pbihI9Df1uZjiOE7dotkUsY=";
      };

      cutlass-dsl-libs-core = mkWheelPkg {
        pname = "nvidia-cutlass-dsl-libs-core";
        version = "4.6.0";
        url = "https://files.pythonhosted.org/packages/84/94/e4e2404ac06a477096ccf8127bf5d391510d36cafb4be86c8c15b4873b0d/nvidia_cutlass_dsl_libs_core-4.6.0-py3-none-any.whl";
        hash = "sha256-+eptMToDyxH6F32jLodHrQysUTWIUIEPNqpsRzYZLCc=";
      };

      cutlass-dsl-libs-base = mkWheelPkg {
        pname = "nvidia-cutlass-dsl-libs-base";
        version = "4.6.0";
        url = "https://files.pythonhosted.org/packages/ce/38/e91f66739d2f8711d1a2457e68cd86d6fbae307ce66ce270a405d4dc6dc7/nvidia_cutlass_dsl_libs_base-4.6.0-cp312-cp312-manylinux_2_28_x86_64.whl";
        hash = "sha256-5BzV203ktTXDCunKRBK5V4AKYlYAGa6R+lHPPqib8lQ=";
      };

      cutlass-dsl-libs-cu13 = mkWheelPkg {
        pname = "nvidia-cutlass-dsl-libs-cu13";
        version = "4.6.0";
        url = "https://files.pythonhosted.org/packages/b1/0f/bd8b25e6307764a7bfefa519241d6e417f3ba1c75ce548aa76b712a4fd15/nvidia_cutlass_dsl_libs_cu13-4.6.0-cp312-cp312-manylinux_2_28_x86_64.whl";
        hash = "sha256-R5n6vEvR94Jf8AEBrgBUxc+2dprv1tlpT22owPB+EsM=";
      };

      nvidia-cutlass-dsl = mkWheelPkg {
        pname = "nvidia-cutlass-dsl";
        version = "4.6.0";
        url = "https://files.pythonhosted.org/packages/8b/1c/fbddb760a0228df87a9e9d1e60b76ecbe6e18035f5853efe0b4563651b2b/nvidia_cutlass_dsl-4.6.0-py3-none-any.whl";
        hash = "sha256-4+Dk2N8g2CyEAfoBP02CAh9B2qX8o9JLVdSmd/IwjKg=";
        propagatedBuildInputs = [
          cutlass-dsl-libs-core
          cutlass-dsl-libs-base
          cutlass-dsl-libs-cu13
        ];
      };

      triton = mkWheelPkg {
        pname = "triton";
        version = "3.7.1";
        url = "https://files.pythonhosted.org/packages/c4/6f/fb96d15db6f36d6eae4cafb998c2e0353bf59d7c4ea1662d7497f269134a/triton-3.7.1-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl";
        hash = "sha256-fkCGmTemggbscNfyW7fsZDPLCD+RNeHzbb0xjcRJpyg=";
      };

      # ── flashinfer ───────────────────────────────────────────────────────────

      flashinfer-cubin = python3.pkgs.buildPythonPackage {
        pname = "flashinfer-cubin";
        version = "0.6.16.post3";
        format = "wheel";
        src = pkgs.fetchurl {
          url = "https://github.com/flashinfer-ai/flashinfer/releases/download/v0.6.16.post3/flashinfer_cubin-0.6.16.post3-py3-none-any.whl";
          hash = "sha256-x5+6mQruKnx+9kIIu2WQDkX+I8OiI/Pfwh7vIl9Dy6I=";
        };
        dontCheckRuntimeDeps = true;
        doCheck = false;
        doInstallCheck = false;
      };

      flashinfer-python = python3.pkgs.buildPythonPackage {
        pname = "flashinfer-python";
        version = "0.6.16.post3";
        format = "wheel";
        src = pkgs.fetchurl {
          url = "https://github.com/flashinfer-ai/flashinfer/releases/download/v0.6.16.post3/flashinfer_python-0.6.16.post3-py3-none-any.whl";
          hash = "sha256-yvaGubB5q+HJ1lq1BWmL0yXoBy3kCv2CLyx08qw7xgE=";
        };
        propagatedBuildInputs = [
          flashinfer-cubin
          python3.pkgs.pynvml
        ];
        dontCheckRuntimeDeps = true;
        doCheck = false;
        doInstallCheck = false;
        postInstall = ''
          python3 - "$out" <<'PYEOF'
          import sys, pathlib
          f = pathlib.Path(sys.argv[1]) / 'lib/python3.12/site-packages/flashinfer/jit/cpp_ext.py'
          if f.exists():
              txt = f.read_text()
              txt = txt.replace(
                  '        subprocess.run(\n            command,',
                  '        import os as _os_fi_; _ninja_env_ = {"PATH": _os_fi_.environ.get("PATH", "/usr/bin:/bin")}\n        subprocess.run(\n            command,'
              ).replace(
                  '            text=True,\n        )\n    except subprocess.CalledProcessError',
                  '            text=True,\n            env=_ninja_env_,\n        )\n    except subprocess.CalledProcessError'
              )
              f.write_text(txt)
              print('Patched flashinfer run_ninja to strip LD_PRELOAD')
          PYEOF
        '';
      };

      # ── compressed-tensors ───────────────────────────────────────────────────

      compressed-tensors = python3.pkgs.buildPythonPackage {
        pname = "compressed-tensors";
        version = "0.17.0";
        format = "wheel";
        src = pkgs.fetchurl {
          url = "https://files.pythonhosted.org/packages/35/63/6edf0415b072fff0bf8b546074dea3f0f9b148e49b601ac98bdc60a76c68/compressed_tensors-0.17.0-py3-none-any.whl";
          hash = "sha256-ShuJtQj377j/tO7opuaeBFLZsIDK4TAUYCXGT76fqao=";
        };
        propagatedBuildInputs = with python3.pkgs; [
          torch-cu130
          transformers
          pydantic
          loguru
        ];
        doCheck = false;
        dontCheckRuntimeDeps = true;
      };

      # ── xgrammar ─────────────────────────────────────────────────────────────

      xgrammar = python3.pkgs.buildPythonPackage {
        pname = "xgrammar";
        version = "0.2.3";
        format = "wheel";
        src = pkgs.fetchurl {
          url = "https://files.pythonhosted.org/packages/d4/fd/5ebd5d14b8993cb225151bbb8f2011742fc7a7d94a3bdbc3ec3954b9b62d/xgrammar-0.2.3-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl";
          hash = "sha256-/fCB+rKWlDAtQdYdz1L619JTh5pxi8avxo2woNq9fxk=";
        };
        propagatedBuildInputs = [
          torch-cu130
          apache-tvm-ffi
          python3.pkgs.pydantic
          python3.pkgs.transformers
        ];
        dontCheckRuntimeDeps = true;
        doCheck = false;
        doInstallCheck = false;
      };

      # ── mistral-common / nvtx / prometheus-fastapi-instrumentator ────────────

      mistral-common = python3.pkgs.buildPythonPackage {
        pname = "mistral-common";
        version = "1.11.7";
        pyproject = true;
        src = pkgs.fetchPypi {
          pname = "mistral_common";
          version = "1.11.7";
          hash = "sha256-07eVg1lc9tlqKrM+QsuESXaDgxR7jFbKxaTxk74Z0g0=";
        };
        build-system = [ python3.pkgs.setuptools ];
        propagatedBuildInputs = with python3.pkgs; [
          pillow
          pydantic
          requests
          sentencepiece
          tiktoken
          jsonschema
          numpy
          pydantic-extra-types
        ];
        doCheck = false;
        dontCheckRuntimeDeps = true;
      };

      nvtx = python3.pkgs.buildPythonPackage {
        pname = "nvtx";
        version = "0.2.15";
        format = "wheel";
        src = pkgs.fetchurl {
          url = "https://files.pythonhosted.org/packages/c0/d1/08f22448d83481408d663065764ba583df091a7de629ed38fc97e522f1af/nvtx-0.2.15-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl";
          hash = "sha256-PKgDCm0ZeVIxgBPdHBLCLaHUuf63a6cuD81EmWEYPCw=";
        };
        buildPhase = "true";
        doCheck = false;
      };

      prometheus-fastapi-instrumentator = python3.pkgs.buildPythonPackage {
        pname = "prometheus-fastapi-instrumentator";
        version = "8.1.0";
        pyproject = true;
        src = pkgs.fetchurl {
          url = "https://files.pythonhosted.org/packages/95/f4/cdcebf7094b03b99fba71ac8f56bd6f227973642662f49d272332d8419b3/prometheus_fastapi_instrumentator-8.1.0.tar.gz";
          hash = "sha256-t38wQ2ZejSjiu9IQF1BhlaQ9mt8dQC0Bv5W0lLflYOE=";
        };
        build-system = [ python3.pkgs.poetry-core ];
        propagatedBuildInputs = with python3.pkgs; [
          prometheus-client
          starlette
        ];
        doCheck = false;
      };

      # ── flash-qla (SM70/SM75 GDN kernel) ────────────────────────────────────

      flashQlaSrc = pkgs.fetchFromGitHub {
        owner = "weicj";
        repo = "FlashQLA-SM70-SM75";
        rev = "3ab27d77d8ca01d7a4718903b726add1a8886c0e";
        hash = "sha256-Pq/S9fBgpfKUH5d7WaRB0ri4tqycjGX1obl4OTeAwpw=";
      };

      buildFlashQlaScript = pkgs.writeText "build-flash-qla-legacy.py" ''
        import os, sys, shutil, glob
        os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5"
        os.environ["TORCH_EXTENSIONS_DIR"] = os.getcwd()
        os.environ["CUDA_HOME"] = "${cudaHome}"
        sys.path.insert(0, "${torch-cu130}/lib/python3.12/site-packages")
        import torch.utils.cpp_extension as ext
        so = ext.load(
            name="flash_qla_legacy_gdn",
            sources=["${./nix/gdn_forward.cu}"],
            extra_cuda_cflags=["-O3"],
            extra_cflags=["-O3"],
            verbose=True,
        )
        built = so.__file__
        print("Built:", built)
        dest = "flash_qla_legacy_gdn.so"
        shutil.copy(built, dest)
        print("Copied to:", dest)
      '';



      flash-qla = python3.pkgs.buildPythonPackage {
        pname = "flash-qla";
        version = "0.1.0";
        format = "other";
        src = flashQlaSrc;
        nativeBuildInputs = [
          cudaPackages.cuda_nvcc
          pkgs.gcc15
          python3
          python3.pkgs.setuptools
          pkgs.ninja
        ];
        buildInputs = [
          torch-cu130
          cudaPackages.cuda_cudart
        ];
        buildPhase = "python3 ${buildFlashQlaScript}";
        installPhase = ''
          FQLA="$out/lib/python3.12/site-packages/flash_qla"
          mkdir -p $FQLA/ops/gated_delta_rule/legacy/csrc
          cp -r ${flashQlaSrc}/flash_qla/* $FQLA/
          chmod -R u+w $FQLA
          cat > $FQLA/__init__.py << 'EOF'
          __version__ = "0.1.0"
          try:
              from flash_qla.ops.gated_delta_rule.chunk import (chunk_gated_delta_rule_fwd, chunk_gated_delta_rule_bwd, chunk_gated_delta_rule)
          except (ImportError, OSError, RuntimeError, ValueError):
              chunk_gated_delta_rule_fwd = None
              chunk_gated_delta_rule_bwd = None
              chunk_gated_delta_rule = None
          __all__ = ["chunk_gated_delta_rule_fwd", "chunk_gated_delta_rule_bwd", "chunk_gated_delta_rule"]
          EOF
          cat > $FQLA/ops/__init__.py << 'EOF'
          try:
              from .gated_delta_rule import chunk_gated_delta_rule
          except (ImportError, OSError, RuntimeError, ValueError):
              chunk_gated_delta_rule = None
          __all__ = ["chunk_gated_delta_rule"]
          EOF
          cat > $FQLA/ops/gated_delta_rule/__init__.py << 'EOF'
          try:
              from .chunk import chunk_gated_delta_rule
          except (ImportError, OSError, RuntimeError, ValueError):
              chunk_gated_delta_rule = None
          __all__ = ["chunk_gated_delta_rule"]
          EOF
          echo "from .sm_legacy import chunk_gated_delta_rule_fwd_legacy" > $FQLA/ops/gated_delta_rule/legacy/__init__.py
          cp ${./nix/sm_legacy.py} $FQLA/ops/gated_delta_rule/legacy/sm_legacy.py
          cp ${./nix/gdn_forward.cu} $FQLA/ops/gated_delta_rule/legacy/csrc/gdn_forward.cu
          cp flash_qla_legacy_gdn.so $FQLA/ops/gated_delta_rule/legacy/
        '';
        propagatedBuildInputs = [ ];
        doCheck = false;
        doInstallCheck = false;
      };

      # ── vLLM sources (upstream forks / deps) ─────────────────────────────────

      cutlass = pkgs.fetchFromGitHub {
        owner = "nvidia";
        repo = "cutlass";
        rev = "v4.4.2";
        hash = "sha256-0q9Ad0Z6E/rO2PdM4uQc8H0E0qs9uKc3reHepiHhjEc=";
      };

      triton-src = pkgs.fetchFromGitHub {
        owner = "triton-lang";
        repo = "triton";
        rev = "v3.5.1";
        hash = "sha256-dyNRtS1qtU8C/iAf0Udt/1VgtKGSvng1+r2BtvT9RB4=";
      };

      deepgemm = pkgs.fetchFromGitHub {
        owner = "vllm-project";
        repo = "DeepGEMM";
        rev = "e21c821f39a2056d68067a466c64ddc942200106";
        hash = "sha256-u3yf88KdH2vf1aROsuS0m9JDf9FpD48f1rBtsSijYCk=";
      };

      flashmla = pkgs.fetchFromGitHub {
        owner = "vllm-project";
        repo = "FlashMLA";
        rev = "a8f794d1251cbfd88a5011445dd5582289c727e4";
        hash = "sha256-k/Mbc70U8wbP4BHnxZ/I607Dc2EnIkhWYd9iKUG740Y=";
      };

      fmha-sm100 = pkgs.fetchFromGitHub {
        owner = "vllm-project";
        repo = "MSA";
        rev = "087c161814d4d9c735b46c21212a09e5f8eb92fa";
        hash = "sha256-ZYdyRMU/fbpVoPz6p4j4Mb84dcw67hTOvXa6BZqWAdk=";
      };

      qutlass = pkgs.fetchFromGitHub {
        owner = "IST-DASLab";
        repo = "qutlass";
        rev = "e74319e3405ce6d71965732880f5dc1f52371f64";
        hash = "sha256-Gzl3KuYXXLXMrVciEYrBPu1FH2cplGUPTFpWzFfUmMo=";
      };

      fmha-sm100-patched = pkgs.runCommand "fmha-sm100-patched" { } ''
        cp -rT ${fmha-sm100} $out
        chmod -R u+w $out
        mkdir -p $out/python/fmha_sm100
        cp -rT ${cutlass} $out/python/fmha_sm100/cutlass
      '';

      vllm-flash-attn-src = pkgs.fetchFromGitHub {
        owner = "vllm-project";
        repo = "flash-attention";
        rev = "28e862d21806bc3580207aa0ad4e2759151e9827";
        hash = "sha256-dt2Gct+E+BwSkvAYu1swww2hbqDBrkmzVz5CWS47Bh0=";
      };

      vllm-flash-attn-patched = pkgs.runCommand "vllm-flash-attn-patched" { } ''
        cp -rT ${vllm-flash-attn-src} $out
        chmod -R u+w $out
        sed -i 's/''${CMAKE_CUDA_FLAGS})/"''${CMAKE_CUDA_FLAGS}")/g' $out/cmake/utils.cmake
        sed -i '/^project(/a find_package(CUDAToolkit REQUIRED)' $out/CMakeLists.txt
        mkdir -p $out/nix-torch-extra/c10/cuda/impl
        echo "#pragma once" > $out/nix-torch-extra/c10/cuda/impl/cuda_cmake_macros.h
        mkdir -p $out/csrc/cutlass
        cp -rT ${cutlass}/include $out/csrc/cutlass/include
      '';

      # ── vLLM ─────────────────────────────────────────────────────────────────

      vllm = python3.pkgs.buildPythonPackage rec {
        pname = "vllm";
        version = "0.2.1-pre3";
        pyproject = true;

        src = ./.;

        build-system = [
          python3.pkgs.setuptools
          python3.pkgs.setuptools-scm
          python3.pkgs.cmake
          python3.pkgs.ninja
          torch-cu130
        ];

        nativeBuildInputs = [
          cudaPackages.cuda_nvcc
          pkgs.cmake
          pkgs.ninja
          pkgs.addDriverRunpath
          pkgs.autoPatchelfHook
          pkgs.gcc15
          python3.pkgs.jinja2
        ];

        buildInputs =
          (with cudaPackages; [
            cuda_cudart
            libcublas
            cuda_cccl
            cuda_nvrtc
            libcusparse
            libcusolver
          ])
          ++ [ torch-cu130 ];

        propagatedBuildInputs =
          (with python3.pkgs; [
            torchaudio-cu130
            torchvision-cu130
            numba
            transformers
            tokenizers
            safetensors
            numpy
            fastapi
            starlette
            uvicorn
            aiohttp
            openai
            pydantic
            prometheus-client
            protobuf
            tiktoken
            diskcache
            lark
            typing-extensions
            filelock
            partial-json-parser
            jsonschema
            pyzmq
            msgspec
            pyyaml
            six
            setuptools
            einops
            cloudpickle
            watchfiles
            python-json-logger
            ninja
            pybase64
            cbor2
            ijson
            setproctitle
            sentencepiece
            regex
            cachetools
            psutil
            requests
            tqdm
            gguf
            pycountry
            blake3
            py-cpuinfo
            packaging
            anthropic
            ray
            lm-format-enforcer
            outlines-core
            depyf
            openai-harmony
            opentelemetry-sdk
            opentelemetry-api
            opentelemetry-exporter-otlp
            uvloop
            llguidance
            model-hosting-container-standards
            (mcp.overrideAttrs (_: { doCheck = false; }))
            opencv-python-headless
          ])
          ++ [
            torch-cu130
            flash-qla
            flashinfer-python
            flashinfer-cubin
            compressed-tensors
            mistral-common
            nvtx
            prometheus-fastapi-instrumentator
            xgrammar
            apache-tvm-ffi
            tilelang
            humming-kernels
            quack-kernels
            fastsafetensors
            tokenspeed-mla
            nvidia-cutlass-dsl
            triton
          ];

        postPatch = ''
          sed -i 's/project(vllm_extensions LANGUAGES CXX)/project(vllm_extensions LANGUAGES CXX CUDA)/' CMakeLists.txt
          sed -i '/project(vllm_extensions LANGUAGES CXX CUDA)/a set(CMAKE_CUDA_ARCHITECTURES 75 CACHE STRING "" FORCE)' CMakeLists.txt
          sed -i '/project(vllm_extensions LANGUAGES CXX CUDA)/a find_package(CUDAToolkit REQUIRED)' CMakeLists.txt
          sed -i '/project(vllm_extensions LANGUAGES CXX CUDA)/a set(CMAKE_CUDA_FLAGS "-gencode arch=compute_75,code=sm_75 --extended-lambda -isystem ${vllm-flash-attn-patched}/nix-torch-extra" CACHE STRING "CUDA flags" FORCE)' CMakeLists.txt
          sed -i '/model-hosting-container-standards/d; /^mcp/d; /opentelemetry-semantic-conventions-ai/d; /llguidance/d; /opencv-python-headless/d' requirements/common.txt
          sed -i 's/lark == 1.2.2/lark/' requirements/common.txt
          sed -i '/quack-kernels/d; /tokenspeed-mla/d; /humming-kernels/d; /nvidia-cutlass-dsl/d; /tilelang/d; /fastsafetensors/d; /flashinfer/d; /apache-tvm-ffi/d; /torchcodec/d; /PyNvVideoCodec/d' requirements/cuda.txt
          sed -i 's/numba == 0.65.0/numba/; s/apache-tvm-ffi==0.1.9/apache-tvm-ffi/' requirements/cuda.txt
          : > vllm/models/minimax_m3/common/ops/__init__.py
          echo 'def minimax_m3_msa_warmup(worker): pass' > vllm/model_executor/warmup/minimax_m3_msa_warmup.py
          sed -i 's/^CUTLASS_FP8_SUPPORTED = cutlass_fp8_supported()/try:\n    CUTLASS_FP8_SUPPORTED = cutlass_fp8_supported()\nexcept Exception:\n    CUTLASS_FP8_SUPPORTED = False/' \
            vllm/model_executor/layers/quantization/utils/w8a8_utils.py
          sed -i 's/^CUTLASS_BLOCK_FP8_SUPPORTED = cutlass_block_fp8_supported()/try:\n    CUTLASS_BLOCK_FP8_SUPPORTED = cutlass_block_fp8_supported()\nexcept Exception:\n    CUTLASS_BLOCK_FP8_SUPPORTED = False/' \
            vllm/model_executor/layers/quantization/utils/w8a8_utils.py
          sed -i 's/    return torch.ops._C.cutlass_scaled_mm_supports_block_fp8(cuda_device_capability)/    try:\n        return torch.ops._C.cutlass_scaled_mm_supports_block_fp8(cuda_device_capability)\n    except (RuntimeError, NotImplementedError):\n        return False/' \
            vllm/_custom_ops.py
          sed -i 's/elif not is_distributed_env and len(active_drivers) != 1:/elif False:/' \
            vllm/triton_utils/importing.py
          sed -i 's|from setuptools_rust.build import build_rust|from setuptools.command.build_ext import build_ext as build_rust|' setup.py
          sed -i '/"setuptools-rust/d' pyproject.toml
          sed -i '/^def get_vllm_version/a\    return "${version}"' setup.py
          sed -i 's|append_cmake_prefix_path("torch" "torch.utils.cmake_prefix_path")|list(INSERT CMAKE_PREFIX_PATH 0 "${torch-cu130}/lib/python3.12/site-packages/torch/share/cmake")|' CMakeLists.txt
          python3 - cmake/utils.cmake << 'PYEOF'
          import re, sys
          path = sys.argv[1]
          txt = open(path).read()
          txt = re.sub(
              r'run_python\(GPU_FLAGS\s+"from torch\.utils\.cpp_extension import COMMON_NVCC_FLAGS.*?Failed to determine torch nvcc compiler flags"\)',
              'set(GPU_FLAGS "-D__CUDA_NO_HALF_OPERATORS__;-D__CUDA_NO_HALF_CONVERSIONS__;-D__CUDA_NO_BFLOAT16_CONVERSIONS__;-D__CUDA_NO_HALF2_OPERATORS__;--expt-relaxed-constexpr")',
              txt, flags=re.DOTALL)
          txt = re.sub(
              r'run_python\(_VLLM_TORCH_GOMP_PATH.*?Failed to find gomp"\)',
              'set(_VLLM_TORCH_GOMP_PATH "")',
              txt, flags=re.DOTALL)
          open(path, 'w').write(txt)
          print('Patched cmake/utils.cmake: hardcoded GPU_FLAGS and gomp path')
          PYEOF
          sed -i 's/return VLLM_TARGET_DEVICE == "cuda" and has_cuda and not _is_tpu()/return VLLM_TARGET_DEVICE == "cuda" and not _is_tpu()/' setup.py
          sed -i 's/torch\.version\.cuda\.split/(torch.version.cuda or "13.0").split/g' setup.py
          python3 -c "import re,glob;r=re.compile('triton[.]next_power_of_2[(]([^()]*(?:[(][^()]*[)][^()]*)*)[)]');repl=lambda m:'(1<<(max('+m.group(1)+',1)-1).bit_length())';[open(f,'w').write(r.sub(repl,t)) for f in glob.glob('vllm/**/*.py',recursive=True) if 'triton.next_power_of_2' in (t:=open(f).read())]"
          sed -i 's/except (RuntimeError, AttributeError):/except Exception:/' vllm/model_executor/layers/fla/ops/utils.py 2>/dev/null || true
          python3 -c "p='vllm/model_executor/layers/quantization/__init__.py';c=open(p).read();open(p,'w').write(c.replace('    from vllm.model_executor.layers.quantization.quark.quark import QuarkConfig','    try:\n        from vllm.model_executor.layers.quantization.quark.quark import QuarkConfig\n    except ImportError:\n        QuarkConfig = None'))"
          cat > tools/build_rust.py << 'RUST_STUB'
          def rust_extensions(*args, **kwargs):
              return []
          def rust_py_extension_module_names(*args, **kwargs):
              return []
          RUST_STUB
        '';

        preBuild = ''cd "$NIX_BUILD_TOP/source"'';

        cmakeFlags = [
          "-DVLLM_PYTHON_EXECUTABLE=${python3}/bin/python3.12"
          "-DVLLM_TARGET_DEVICE=cuda"
          "-DCUDA_TOOLKIT_ROOT_DIR=${cudaHome}"
          "-DCUDAToolkit_INCLUDE_DIR=${cudaHome}/include"
          "-DCUDA_HOME=${cudaHome}"
          "-DCMAKE_CUDA_COMPILER=${cudaPackages.cuda_nvcc}/bin/nvcc"
          "-DCUDA_FOUND:BOOL=TRUE"
          "-DCUDA_VERSION=13.0"
          "-DCMAKE_CXX_COMPILER=${pkgs.gcc15}/bin/g++"
          "-DCMAKE_C_COMPILER=${pkgs.gcc15}/bin/gcc"
        ];

        env = {
          VLLM_CUTLASS_SRC_DIR = "${cutlass}";
          SETUPTOOLS_SCM_PRETEND_VERSION = version;
          TRITON_KERNELS_SRC_DIR = "${triton-src}/python/triton_kernels/triton_kernels";
          DEEPGEMM_SRC_DIR = "${deepgemm}";
          FLASH_MLA_SRC_DIR = "${flashmla}";
          FMHA_SM100_SRC_DIR = "${fmha-sm100-patched}";
          QUTLASS_SRC_DIR = "${qutlass}";
          VLLM_FLASH_ATTN_SRC_DIR = "${vllm-flash-attn-patched}";
          CMAKE_ARGS = "-DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_CUDA_COMPILER=${cudaPackages.cuda_nvcc}/bin/nvcc -DCMAKE_CUDA_COMPILER_WORKS=TRUE -DCUDA_FOUND:BOOL=TRUE -DCUDA_TOOLKIT_ROOT_DIR=${cudaHome} -DCUDAToolkit_INCLUDE_DIR=${cudaHome}/include -DCUDA_VERSION=13.0 -DCMAKE_CXX_FLAGS=-I${vllm-flash-attn-patched}/nix-torch-extra -DCMAKE_CXX_COMPILER=${pkgs.gcc15}/bin/g++ -DCMAKE_SKIP_RPATH=TRUE";
          VLLM_PYTHON_EXECUTABLE = "${python3}/bin/python3.12";
          CUDA_HOME = "${cudaHome}";
          TORCH_CUDA_ARCH_LIST = "7.5";
          VLLM_TARGET_DEVICE = "cuda";
          MAX_JOBS = "8";
          NVCC_THREADS = "4";
          LD_LIBRARY_PATH = "${torch-cu130}/lib/python3.12/site-packages/torch/lib:${cudaHome}/lib:${cudaPackages.cudnn.lib}/lib:${cudaPackages.libnvjitlink.lib}/lib:${pkgs.gcc15.cc.lib}/lib";
        };

        NIX_LDFLAGS = "-L${cudaPackages.cuda_cudart}/lib/stubs";

        postInstall = ''
          python3 -c "import re,glob;r=re.compile('triton[.]next_power_of_2[(]([^()]*(?:[(][^()]*[)][^()]*)*)[)]');repl=lambda m:'(1<<(max('+m.group(1)+',1)-1).bit_length())';[open(f,'w').write(r.sub(repl,open(f).read())) for f in glob.glob('$out/lib/python3.12/site-packages/vllm/**/*.py',recursive=True) if 'triton.next_power_of_2' in open(f).read()]"
          find $out -name "*.so" -exec addDriverRunpath {} \;
        '';

        doCheck = false;
        dontCheckRuntimeDeps = true;
        doInstallCheck = false;
      };

    in
    {
      packages.${system} = {
        default = vllm;
        inherit
          vllm
          torch-cu130
          torchaudio-cu130
          torchvision-cu130
          flash-qla
          flashinfer-python
          flashinfer-cubin
          compressed-tensors
          xgrammar
          mistral-common
          nvtx
          prometheus-fastapi-instrumentator
          apache-tvm-ffi
          tilelang
          humming-kernels
          quack-kernels
          fastsafetensors
          tokenspeed-mla
          nvidia-cutlass-dsl
          triton
          nvidia-nccl-cu13
          nvidia-cusparselt-cu13
          nvidia-nvshmem-cu13
          ;
        cuda-home = cudaHome;
      };
    };
}
