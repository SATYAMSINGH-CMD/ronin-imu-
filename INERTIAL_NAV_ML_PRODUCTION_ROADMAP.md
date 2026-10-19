# Robust Neural Inertial Navigation (RoNIN) — Production ML Lifecycle Roadmap

> **System Objective**: Transform noisy 6-axis IMU sensor data (Accelerometer + Gyroscope) into drift-resistant, meter-accurate 2D/3D positioning trajectories by combining physics-based AHRS attitude filtering with deep temporal neural velocity estimation.

---

## 1. Why Classical Navigation Fails vs. Why This Pipeline Wins

### The Classical Problem (Double Integration Disaster)
In traditional inertial navigation, you calculate position by integrating raw acceleration twice:
```
Acceleration ──[ Integrate 1 ]──► Velocity ──[ Integrate 2 ]──► Position
```
- A tiny accelerometer error (ε_a) grows with time squared:
  Position Drift ≈ (1/2) × ε_a × t²
- A tiny gyroscope angle error (ε_ω) tilts your gravity vector, leaking 9.81 m/s² into horizontal motion:
  Position Drift ≈ (1/6) × 9.81 × ε_ω × t³

Within 15 to 30 seconds, the estimated position drifts hundreds of meters away from reality!

---

### The RoNIN Solution (Single Integration Magic)
Instead of double-integrating noisy acceleration, we train a neural network to look at a 1-second pattern of motion and predict the current **Velocity vector [v_x, v_y] directly**:
```
[Raw Accel + Gyro] ──► Madgwick Filter ──► Physics Prep ──► Neural Net ──► Velocity [v_x, v_y]
                                                                                │
                                                                   [ Integrate ONCE ]
                                                                                │
                                                                                ▼
                                                                        Accurate Position
```
- Since we only integrate Velocity **once**:
  Position(t) = Position(t - Δt) + Velocity × Δt
- Position drift grows linearly with time:
  Position Drift ∝ t   (NOT t³!)
- Result: The path stays accurate over long walking trajectories with minimal drift.

---

## 2. End-to-End System Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. RAW SENSOR INPUT (200 Hz)                                                │
│    • Accelerometer: [a_x, a_y, a_z] in m/s² (Device Body Frame)             │
│    • Gyroscope:     [ω_x, ω_y, ω_z] in rad/s (Device Body Frame)            │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. STEP 1: MADGWICK AHRS FILTER                                             │
│    • Tracks 3D tilt as a 4D Quaternion: q = [q_w, q_x, q_y, q_z]            │
│    • Fuses gyro speed with gravity vector [0, 0, 1] using gradient descent  │
│    • Tracks and nulls gyroscope bias drift over time                        │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. STEP 2: PHYSICS PRE-PROCESSING                                           │
│    • Rotate to World Coordinates:                                           │
│        a_world = RotationMatrix(q) × a_body                                 │
│    • Clean Gravity Subtraction:                                             │
│        a_linear = a_world - [0, 0, 9.80665]                                 │
│    • Heading-Independent Frame (HIF):                                       │
│        Rotate horizontal X/Y so window starts pointing forward (Yaw = 0°)   │
│    • Static Motion / Zero-Velocity Update (ZUPT):                           │
│        If motion variance < threshold, clamp velocity to exactly [0, 0]     │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 4. STEP 3: PYTORCH NEURAL NETWORK (1D-ResNet / Dilated TCN / Bi-LSTM)       │
│    • Input Window: 200 time steps (1.0 second of motion at 200 Hz)          │
│    • Channels (6): [a_linear_x, a_linear_y, a_linear_z, ω_x, ω_y, ω_z]      │
│    • Dual Output Heads:                                                     │
│        1. Velocity Vector: [v_x, v_y] in m/s                                │
│        2. Uncertainty:     [σ_x, σ_y] (Confidence bounds)                   │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 5. STEP 4: TRAJECTORY INTEGRATION & METRICS                                 │
│    • Distance = Distance + Velocity × Δt                                    │
│    • Reconstruct full 2D walking trajectory [X, Y]                          │
│    • Evaluate against ground truth (Vicon Motion Capture)                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. The Complete Production ML Lifecycle

### PHASE 1: Contract & Performance Targets
- **Input**: 6-axis IMU stream at 200 Hz (`float32`).
- **Output**: 2D ground velocity `[v_x, v_y]` in m/s + Uncertainty `[σ_x, σ_y]`.
- **Target Metrics**:
  * **ATE (Absolute Trajectory Error)**: ≤ 1.50 meters per 100 meters walked.
  * **RTE (Relative Trajectory Error)**: ≤ 1.0% drift over total distance.
  * **Velocity RMSE**: ≤ 0.12 m/s against ground truth.
  * **Inference Latency**: ≤ 5.0 milliseconds per window on mobile/edge CPU.

---

### PHASE 2: RoNIN Dataset Ingestion & Validation
The **RoNIN** (Robust Neural Inertial Navigation) dataset contains >40 hours of 200 Hz IMU recordings across 100+ human subjects with high-precision 3D Vicon optical tracking:
- **Phone carrying poses**: Handheld (typing/talking), In Pocket, In Bag, Free-Swinging.
- **Data Quality Gates (Automated Validation)**:
  1. Sampling frequency must be 200 Hz ± 5 Hz (reject erratic clock jitter).
  2. Maximum gap between packets ≤ 15 milliseconds (no dropped frames).
  3. Accelerometer range check: values must stay within [-16g, +16g] (reject clipped signals).
  4. Gyroscope range check: values must stay within [-2000°/s, +2000°/s].
  5. Check zero NaN / Infinite values across all channels.

---

### PHASE 3: Physics Pre-Processing Details

#### 1. Madgwick Attitude Estimation
Calculates the orientation quaternion `q = [q_w, q_x, q_y, q_z]`:
```
Orientation Rate: dq/dt = (1/2) × q ⊗ ω - β × (∇f / ‖∇f‖)
```
- `ω`: Gyroscope angular rate `[ω_x, ω_y, ω_z]`.
- `∇f`: Gradient descent step pointing directly toward Earth's gravity vector `[0, 0, 1]`.
- `β`: Filter gain balancing gyro integration and accelerometer gravity correction.

#### 2. Rotate to World Coordinates & Strip Gravity
Convert sensor-body acceleration `a_body` to world coordinates `a_world`:
```
a_world = RotationMatrix(q) × a_body
a_linear = a_world - [0, 0, 9.80665]
```
Now, `a_linear_z` represents true vertical acceleration (jumping/elevators), while `a_linear_x` and `a_linear_y` represent horizontal floor acceleration.

#### 3. Heading-Independent Frame (HIF) Alignment
- The world coordinate system has an arbitrary Compass Yaw direction.
- If you train directly on world coordinates, the network wastes learning capacity trying to figure out which way the room is facing.
- **HIF Solution**: For every 1-second window, rotate horizontal acceleration so that the window's starting heading angle is aligned with 0° (Facing forward along X):
  ```
  a_hif_x =  cos(Yaw_anchor) × a_linear_x + sin(Yaw_anchor) × a_linear_y
  a_hif_y = -sin(Yaw_anchor) × a_linear_x + cos(Yaw_anchor) × a_linear_y
  ```
- The network predicts velocity in this clean HIF frame, and we rotate it back to world coordinates using `+Yaw_anchor`.

#### 4. Stationary Detection (ZUPT — Zero Velocity Update)
Compute moving variance over the window:
```
Motion_Energy = Variance(Accel) / σ_a² + Variance(Gyro) / σ_ω²
```
- If `Motion_Energy < Threshold`:
  The person has paused or stopped.
  Force predicted velocity to exactly `[0.0, 0.0]`.
  Eliminates stationary drift completely!

---

### PHASE 4: Leakage-Safe Splitting & Classical Baselines

#### Preventing Data Leakage
- **Never split by random time steps!** (Adjacent windows are 99% identical).
- **Group-K-Fold by Subject ID**:
  * **Train Set**: 80% of human subjects.
  * **Validation Set**: 10% of subjects.
  * **Test Set (Frozen)**: 10% of subjects (never seen during training).
  * Also hold out unseen carrying modes (e.g., test on Free-Swinging when trained on Pocket/Handheld).

#### Classical Benchmark Baselines
Before evaluating deep learning, build 2 reference baselines:
1. **Classical Strapdown INS**: Double integration of rotated acceleration. (Drifts > 50 meters in 30 seconds).
2. **Pedestrian Dead Reckoning (PDR)**: Step counting + step length model (Weinberg formula) + integrated compass heading. (Works for handheld walking, fails completely in bags or running).

---

### PHASE 5: PyTorch Deep Learning Architecture

#### 1. Input Tensor Shape
```
Shape = (Batch_Size, 6 Channels, 200 Time_Steps)
Channels: [a_hif_x, a_hif_y, a_hif_z, ω_hif_x, ω_hif_y, ω_hif_z]
```

#### 2. Architecture Candidates
- **1D-ResNet (Recommended)**: 4 to 6 residual blocks with 1D convolutions, Batch Normalization, and LeakyReLU. Very fast on edge CPUs (< 3 ms).
- **Dilated TCN (Temporal Convolutional Network)**: Exponentially growing receptive field (dilation = 1, 2, 4, 8, 16, 32). Captures long-range stride patterns over 3+ seconds.
- **ConvBiLSTM**: 1D-CNN feature extractor followed by Bidirectional LSTM layers.

#### 3. Physics-Informed Multi-Task Loss Function
Do not use simple Mean Squared Error (MSE) alone. Use a composite loss:
```
Total_Loss = Loss_velocity 
           + 0.5 × Loss_direction 
           + 0.2 × Loss_smoothness 
           + 1.0 × Loss_stationary
```
- **Loss_velocity**: Huber Loss between predicted and true velocity (resilient to sudden sensor spikes).
- **Loss_direction**: `1 - CosineSimilarity(v_pred, v_true)` (forces the model to point in the exact travel direction).
- **Loss_smoothness**: Penalizes non-physical velocity jumps: `‖(v_t - v_{t-1}) / Δt - a_horizontal‖²`.
- **Loss_stationary**: When ZUPT indicates no motion, heavily penalizes any non-zero velocity output.

#### 4. Sensor Data Augmentation
To make the model robust to cheap or noisy sensors:
1. **Random Yaw Rotation**: Rotate the window randomly between 0° and 360°.
2. **Gaussian Noise Jitter**: Inject sensor noise based on real IMU Allan variance.
3. **Scale Factor Perturbation**: Randomly scale accel/gyro axes by ±3% to simulate sensor calibration errors.
4. **Gyro Bias Drift Simulation**: Add a slow random-walk bias drift across the sequence.

---

### PHASE 6: Hyperparameter Optimization (Optuna) & Uncertainty Head

#### 1. Optuna Search Knobs
- Window size: `[100, 200, 300]` samples (0.5s, 1.0s, 1.5s).
- Convolution kernel size: `[3, 5, 7]`.
- ResNet base channels: `[32, 64, 128]`.
- Learning rate: `1e-4` to `5e-3` (Cosine Annealing schedule).
- Madgwick filter gain `β`: `0.01` to `0.40`.

#### 2. Uncertainty Head (Heteroscedastic Loss)
The model outputs mean velocity `[μ_x, μ_y]` and predicted variance `[σ_x², σ_y²]`:
```
Loss_uncertainty = (1/2) × exp(-s) × ‖v_true - μ‖² + (1/2) × s
where s = ln(σ²)
```
- When the motion is ambiguous or chaotic (e.g., phone tumbling inside a purse), the model automatically increases `σ`.
- This variance feed directly into an Extended Kalman Filter (EKF) to balance IMU with GPS/WiFi positions!

---

### PHASE 7: Edge Deployment & Model Quantization

```
PyTorch Model (FP32, 15 MB) ──► 12 ms Latency on CPU
             │
             ▼
Export to ONNX Runtime      ──► 4.2 ms Latency on CPU
             │
             ▼
INT8 Post-Training Quant    ──► 1.6 ms Latency, 3.8 MB Size
```
- **Quantization Gate**: Trajectory error (ATE) on the INT8 model must degrade by less than 3% compared to full FP32 precision.

---

### PHASE 8: Production Serving & MLOps

#### 1. Dual Serving Architecture
- **Real-Time Streaming Engine (WebSocket / gRPC)**:
  * Maintains a 200-sample ring buffer per active user.
  * Streams real-time `[X, Y]` coordinates at 20 Hz.
- **Offline Batch API (FastAPI)**:
  * Accepts uploaded `.csv` / `.parquet` IMU files.
  * Computes smoothed trajectory and generates ATE benchmark plots.

#### 2. Production Telemetry & Drift Monitoring
- **Sensor Clipping Alarm**: Alerts if >0.1% of accelerometer readings hit saturation limits (±16g).
- **Filter Divergence Alarm**: Checks if the gravity vector magnitude drifts away from 9.81 m/s² by >10%.
- **Uncertainty Drift**: Monitors if average predicted variance `σ` is steadily rising (signaling hardware sensor degradation).
- **Automated Trajectory Regression in CI/CD**: Every pull request runs the model on 5 standard benchmark walking paths; fails if ATE exceeds 1.50 meters.

---

## 4. Implementation Sprints

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ SPRINT 1: Physics Engine & Data Foundation                                  │
│ • Download RoNIN dataset subsets.                                           │
│ • Implement Madgwick AHRS filter in Python/NumPy.                           │
│ • Implement World Frame rotation and exact gravity subtraction.             │
│ • Implement Classical INS & PDR baselines.                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│ SPRINT 2: PyTorch Neural Pipeline & HIF Coordinate System                   │
│ • Build PyTorch Dataset with sliding window and HIF alignment.              │
│ • Implement 1D-ResNet and TCN models.                                       │
│ • Implement composite physics loss (Huber + Direction + ZUPT).              │
│ • Build trajectory integrator and evaluate ATE / RTE drift metrics.         │
├─────────────────────────────────────────────────────────────────────────────┤
│ SPRINT 3: Advanced Optimization & Model Regularization                      │
│ • Add IMU data augmentations (Yaw rotation, noise jitter, bias drift).      │
│ • Add Heteroscedastic uncertainty head [σ_x, σ_y].                          │
│ • Run Optuna Bayesian HPO sweep.                                            │
│ • Error analysis across carrying modes (pocket vs handheld vs bag).         │
├─────────────────────────────────────────────────────────────────────────────┤
│ SPRINT 4: Edge Optimization & Production Serving                            │
│ • Export to ONNX and perform INT8 quantization.                             │
│ • Build real-time WebSocket streaming server with ring buffer.              │
│ • Package with Docker and write CI/CD regression tests.                     │
│ • Connect MLflow for experiment tracking and model versioning.              │
└─────────────────────────────────────────────────────────────────────────────┘
```
