# Running the unit worker on the Windows 5090 box (<WINDOWS-HOST>)

Tailscale IP: `<WORKER_TAILNET_IP>`. The worker transcribes NACA pages with the
campaign engine (Qwen3-VL-8B lora-v2_1 Q8_0, config hash `6ecf7d0dbeb2`).
Needs about 10 GB VRAM free (model 8.5 GB + mmproj 0.9 GB + KV for 2 slots).

## 1. One-time setup
1. Update the NVIDIA Windows driver (CUDA-on-WSL needs a recent Game Ready/Studio driver). Do NOT install a Linux driver inside WSL.
2. In an admin PowerShell: `wsl --install -d Ubuntu` then reboot. Check `wsl -l -v` shows VERSION 2.
3. Install Docker Desktop, Settings > General > "Use the WSL 2 based engine"; Settings > Resources > WSL integration > enable Ubuntu.
4. Verify the GPU is visible to containers (in Ubuntu):
   `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi`
5. Join the tailnet (Tailscale already installed on this box) and confirm the project host answers: `curl http://<project-host>:<port>/` (tailnet URL supplied by the operator).

## 2. Model files (about 9.4 GB, once)
In Ubuntu (keep them on the WSL ext4 disk, not /mnt/c, for load speed):

    mkdir -p ~/models/qwen3-vl-8b-lora-v2_1
    # copy from the operator box over the tailnet:
    scp operator@<operator-tailnet-name>:~/data_normalizing/models/qwen3-vl-8b-lora-v2_1/{Qwen3-VL-8B-lora-v2_1-Q8_0.gguf,mmproj-Qwen3VL-8B-Instruct-F16.gguf} ~/models/qwen3-vl-8b-lora-v2_1/

## 3. Get the image
Build from a checkout of the repo (`git clone` over the tailnet) :

    cd data_normalizing && docker build -f worker/Dockerfile -t naca-worker .

## 4. Run
Standalone (fallback queue; units file and page images from a local data tree):

    docker run --rm --gpus all \
      -v ~/models/qwen3-vl-8b-lora-v2_1:/models:ro \
      -v ~/naca-work:/work \
      naca-worker standalone

BOINC mode (Task 9; project URL is the tailnet address):

    docker run -d --restart unless-stopped --gpus all \
      -v ~/models/qwen3-vl-8b-lora-v2_1:/models:ro \
      -e BOINC_PROJECT_URL=http://<project-host-tailnet>:<port>/<project> \
      -e BOINC_AUTHENTICATOR=<account-key-from-project-admin> \
      naca-worker boinc

Do not run other GPU-heavy programs while it runs; it holds about 10 GB of the 5090.

> **Native path now exists:** volunteers on Windows no longer need WSL2/Docker —
> see `docs/worker-windows-native.md` (signed BOINC app `naca_vision_b`). This
> WSL2 doc remains for the trusted/early Docker worker.
