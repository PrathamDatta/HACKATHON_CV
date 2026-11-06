### How it works
1. **Clean-up** – fills missing values (normalised convolution), estimates sensor noise, removes flying-pixel spikes, applies noise-adaptive bilateral smoothing.
2. **Discontinuity detection** – a second-difference *step* detector on the low-res depth (steps are flagged, sloped floors/walls are not).
3. **Gate** – RGB is only allowed to influence depth near detected depth discontinuities. Elsewhere the RGB range kernel is switched off, which blocks texture-copying.
4. **Gated tri-lateral JBU** – joint bilateral upsampling with a spatially varying colour tolerance and a depth-consistency term (expert E1).
5. **Learned blend + residual** – a 0.22 M-parameter U-Net sees RGB, the classical outputs and the gate, and predicts a residual δ and a per-pixel blend β: `out = E0 + β(E1 − E0) + δ` (E0 = bicubic, E1 = classical). β = 0 is exactly the deep-only model.
6. **Auto** – pristine input (no holes, low estimated noise) uses the deep-only network; noisy or incomplete input uses the hybrid.

### Using the app
* **Simulate sensor** – pick a sample (NYU Depth V2 *test* frames) or upload RGB + dense depth. The app degrades the depth (resolution, noise, spikes, holes) and compares methods against the ground truth.
* **Upsample my depth map** – upload an RGB image and a low-resolution depth map (1/2, 1/4 or 1/8 of the RGB size). No ground truth is needed.
* Accepted depth formats: `.npy`/`.npz` (metres), 16-bit PNG (millimetres), 8-bit PNG (relative depth, rescaled to 0.5–5 m).

### Limitations
Trained only on indoor Kinect scenes (NYU Depth V2); outdoor LiDAR-style sparse depth is out of distribution. A depth map is required – the app does not estimate depth from RGB alone.
