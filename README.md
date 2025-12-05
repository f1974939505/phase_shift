## Interferometric Phase Unwrapping Toolkit (GUI)

A small toolkit for four-step interferometric phase processing with simple GUIs:
- **Phase unwrapping** from four interferograms `I1.txt`–`I4.txt` with optional denoising.
- **Zernike background removal** from an unwrapped phase map.
- **Phase map comparison** utility for visual QA.

The GUIs are in Chinese (labels/messages), but usage is straightforward and documented below.

### Repository layout
- `phase_unwrap.py`: GUI to load `I1.txt`–`I4.txt`, denoise (optional), compute wrapped phase, unwrap, and save `phase.txt`.
- `zernike_fit.py`: GUI to fit Zernike polynomials to an unwrapped `phase.txt`, save `background.txt` and `surface.txt`.
- `read_phase.py`: Script to visualize two phase maps and their difference.
- `raw/`: Example set of four interferograms (`I1.txt`–`I4.txt`).
- `noised/`: Another example set plus a sample `Phase.txt` (note the capital P).

### Requirements
- Python 3.8+
- Packages:
  - `numpy`
  - `matplotlib`
  - `scipy`
  - `scikit-image`
  - Tkinter (Python’s `tk` GUI library)

On Debian/Ubuntu:
```bash
sudo apt-get update
sudo apt-get install -y python3-tk  # Tkinter runtime for the GUIs
# Optional for Chinese font rendering in plots
sudo apt-get install -y fonts-noto-cjk
```
Install Python packages (use a virtual env if desired):
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy matplotlib scipy scikit-image
```

### Display requirements
The GUIs use the `TkAgg` backend. You must run on a machine with a graphical desktop (or use X forwarding). On headless servers, consider:
```bash
xvfb-run -a python phase_unwrap.py
```

### Data conventions
- Input interferograms: `I1.txt`, `I2.txt`, `I3.txt`, `I4.txt` (2D arrays of floats).
- During loading, values > 1.0 are treated as saturated, set to 0, and added to a mask.
- Unwrapped phase output `phase.txt` uses `100000.0` to encode invalid (masked) pixels instead of `NaN` for portability.
- Zernike processing preserves/uses the original validity mask; outputs also encode invalid pixels as `100000.0`.

### 1) Phase unwrapping GUI (`phase_unwrap.py`)
Run:
```bash
python phase_unwrap.py
```
Steps in the window:
- 点击“浏览…” and select a folder containing `I1.txt`–`I4.txt`.
- Optionally enable 降噪 (denoising) and choose the filter:
  - 中值滤波 = median filter
  - 高斯滤波 = Gaussian filter
  - 非局部均值滤波 = non-local means
- Click 运行分析 to process. The app will:
  - Load the four interferograms and build a saturation mask (>1.0 → invalid)
  - Compute wrapped phase via four-step formula and unwrap it (`skimage.restoration.unwrap_phase`)
  - Save the result to `phase.txt` in the chosen output directory (or alongside the script if not specified)
  - Display a figure window with one interferogram, wrapped phase, mask, and final unwrapped phase

Output:
- `phase.txt` — 2D float array; invalid pixels encoded as `100000.0`.

### 2) Zernike background removal GUI (`zernike_fit.py`)
Run:
```bash
python zernike_fit.py
```
Steps in the window:
- 选择 `phase.txt` produced by the unwrapping step.
- Optionally pick an output folder (defaults to the script’s directory).
- Set:
  - 总拟合项数 = total number of Zernike terms to fit (e.g., 15)
  - 背景定义项数 = how many leading terms constitute “background” to subtract (e.g., 4). Must be ≥1 and ≤ total terms.
- Click 运行分析. The app will fit Zernike coefficients on valid pixels and subtract the background.

Outputs:
- `background.txt` — reconstructed background phase using the first N terms; invalid pixels encoded as `100000.0`.
- `surface.txt` — residual phase (device surface error); invalid pixels encoded as `100000.0`.
- A figure window visualizing original, background, and residual maps.

### 3) Phase comparison utility (`read_phase.py`)
This script plots two phase files and their difference.

Edit the file paths near the bottom of `read_phase.py` before running:
```python
file_a = r'phase.txt'
file_b = r'noised/Phase.txt'  # Example: note the capital P in the repo sample
```
Run:
```bash
python read_phase.py
```
Behavior:
- Loads two phase maps (`NaN` or values > 1000 are treated as invalid for visualization).
- Displays three panels: first map, second map, and symmetric-range difference map.

### Tips and troubleshooting
- Tkinter not found: install `python3-tk` (see above).
- GUI fails to start on remote server: ensure a display, use SSH X11 forwarding, or `xvfb-run`.
- Chinese fonts in plots render as squares: install CJK fonts (e.g., `fonts-noto-cjk`).
- Case-sensitive paths: sample under `noised/` is `Phase.txt` (capital P); adjust `read_phase.py` accordingly.
- Performance: Non-local means denoising can be slow on large arrays; try median or Gaussian first.

### License
No license file is provided. Add one if you plan to distribute.

### Acknowledgments
- Phase unwrapping powered by `skimage.restoration.unwrap_phase`.
- Zernike fitting implemented directly (Noll indexing) with `numpy`/`scipy`.
