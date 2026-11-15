# Contestant setup

Two scripts at the repo root install everything you need to train an agent
against the simulator and connect it to the tournament server. Run the one for
your system **from the repo root**:

```powershell
# Windows (PowerShell)
powershell -ExecutionPolicy Bypass -File .\install.ps1
powershell -ExecutionPolicy Bypass -File .\install.ps1 -WithTorch   # also install PyTorch
```

```bash
# macOS / Linux
bash install.sh
bash install.sh --torch                                            # also install PyTorch
```

The first run takes a few minutes, mostly compiling the simulator. When it
finishes it prints `All set.` and the commands to try next.

**Re-run the script after every `git pull`.** It skips whatever is already
installed and rebuilds the simulator, so your local `cr_sim` matches the one the
tournament server runs.

## What it installs

| | Windows (`install.ps1`) | macOS / Linux (`install.sh`) |
|---|---|---|
| Python 3.10–3.13 | Uses one it finds; otherwise installs Python 3.12 for your user with `winget` | Uses one it finds; otherwise stops and tells you how to install it |
| Rust | Installs `rustup` with `winget` if missing. Uses the MSVC toolchain if Visual C++ Build Tools are present, otherwise the GNU toolchain (no Visual Studio needed) | Installs `rustup` into `~/.cargo` if missing |
| Virtual environment | `.venv` at the repo root | `.venv` at the repo root |
| Simulator | `cr_sim`, built from [`gym/`](gym/) with `maturin develop --release` | same |
| Python packages | `numpy`, `websockets`, `pytest` (and `maturin` to build) | same |
| PyTorch (optional) | CPU build, with `-WithTorch` | with `--torch` (CPU build on Linux) |

Nothing is installed system-wide except, on Windows, Python and rustup for
your user account if they were missing.

At the end the script runs [`scripts/check_install.py`](scripts/check_install.py),
which plays one full game between two random agents through the same per-seat
view the tournament server uses and reports which packages are present.

## Requirements

- **Python 3.10 or newer** (3.10–3.13). On Windows the script can install it
  for you.
- **macOS / Linux: a C compiler**, which Rust needs to link:
  - macOS: `xcode-select --install`
  - Ubuntu/Debian: `sudo apt install build-essential python3-venv`
  - Fedora: `sudo dnf install gcc`
- **Windows: `winget`** (App Installer), only if Python or Rust is missing.

## After installing

```powershell
# Windows
.\.venv\Scripts\Activate.ps1
python server\examples\random_agent.py ws://localhost:8765 my-team
```

```bash
# macOS / Linux
source .venv/bin/activate
python server/examples/random_agent.py ws://localhost:8765 my-team
```

Then read [`gym/README.md`](gym/README.md) for the environment API and
[`server/CONNECTING.md`](server/CONNECTING.md) for connecting to the tournament
server.

## Windows: Smart App Control

If **Smart App Control** is on (Windows Security → App & browser control →
Smart App Control), Windows blocks the small helper programs Rust compiles while
building the simulator, and the build fails with *"An Application Control policy
has blocked this file"*. The script detects this and lists your options:

1. Ask the organizers for a prebuilt `cr_sim` wheel (`.whl`) and install it with
   `.\.venv\Scripts\python -m pip install <path-to-wheel>`.
2. Build on another computer, or in WSL / Linux / macOS.
3. Turn Smart App Control off. This is your decision: on some Windows versions
   it cannot be turned back on without reinstalling Windows.
