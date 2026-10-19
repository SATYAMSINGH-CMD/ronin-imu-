# Post-Pipeline Experimentation Guide & Intuitive 1D-CNN Concepts

> **Purpose**: This guide explains:
> 1. Exactly what **Blocks**, **Kernels**, **Strides**, and **Channels** mean in plain English without confusing jargon.
> 2. The step-by-step experimentation plan we will follow **after** building the baseline pipeline (adding filters, noise injection, and hyperparameter tuning to measure accuracy gains).

---

## Part 1: Demystifying 1D-CNN Concepts in Plain English

When engineers talk about Convolutional Neural Networks, they use intimidating words. Here is what they actually mean in the real world:

### 1. What is a "Kernel" (The Sliding Magnifying Glass)?
* **Imagine**: You are reading a line of text, but you are looking through a small magnifying glass that only shows **7 letters at a time**.
* As you slide the magnifying glass from left to right along the line, you read the words chunk by chunk.
* **In our model (`kernel_size = 7`)**:
  * Each reading from the phone is taken every **5 milliseconds** (200 Hz).
  * A kernel of size 7 looks at **7 readings simultaneously = 35 milliseconds**.
  * A human heel hitting the floor creates an impact shockwave that lasts about 30 to 40 milliseconds.
  * Therefore, a kernel of 7 is the **exact physical width of one heel strike**!

---

### 2. What is "Stride" (How Many Steps You Jump)?
* **Imagine**: When you slide that magnifying glass:
  * **`stride = 1`**: You slide it forward by **1 letter at a time**. You see every single tiny detail.
  * **`stride = 2`**: You jump forward by **2 letters at a time**, skipping every alternate letter.
* **Why do we use `stride = 2`?**:
  * It cuts the timeline in half: 200 ticks ──► 100 ticks ──► 50 ticks ──► 25 ticks.
  * When the timeline shrinks, that same magnifying glass of 7 now covers a **much wider span of time** (from 35 ms up to 140 ms).
  * This allows the AI to see both **micro-vibrations (heel strike)** and **macro-movement (a full leg swing)**!

---

### 3. What are "Channels" (The Team of Specialized Detectives)?
* **Imagine**: You have a team of detectives analyzing a crime scene.
* If you have only 1 detective, they can only look for footprints.
* If you have **128 detectives (128 channels)** running in parallel:
  * Detective #1 only looks for *sharp impact spikes*.
  * Detective #14 only looks for *smooth leg swinging curves*.
  * Detective #35 only looks for *sideways body sway*.
  * Detective #99 only looks for *pauses / standing still*.
* Expanding from **6 sensor channels to 128 channels** gives the AI 128 separate memory slots to detect different patterns of walking.

---

### 4. What is a "Block" (A Factory Room on an Assembly Line)?
* **Imagine a car manufacturing plant**:
  * Room 1: Welds the frame.
  * Room 2: Installs the engine.
  * Room 3: Paints the car.
* In a neural network, a **Block** is simply **one self-contained room in the assembly line**.
* Inside one Block:
  1. A Convolutional layer (scans the wave).
  2. A Normalization layer (keeps numbers in a healthy range).
  3. An Activation function (adds non-linear decision power).

#### What is a "Residual Block" (A Room with a Shortcut Highway)?
* Normally, data enters the room, gets transformed, and exits:
  `Output = Room(Input)`
* A **Residual Block** adds a **shortcut hallway** running alongside the room:
  `Output = Room(Input) + Original Input`
* **Why this matters**:
  * The original raw physical footstep rhythm flows straight down the shortcut hallway without getting distorted or lost.
  * The layers in the room only need to figure out the **fine-tuning adjustments**.
  * Result: The network trains 5× faster and never forgets the raw motion signal!

---

## Part 2: The Post-Pipeline Experimentation Roadmap

Now that our baseline pipeline is complete:
$$\text{Data Ingestion} \longrightarrow \text{Physics Cleanup} \longrightarrow \text{Dataset Windows} \longrightarrow \text{1D-ResNet} \longrightarrow \text{Velocity Output}$$

We have established our **Baseline Scorecard**.

Every single optimization below can now be tested scientifically:
> *"Does this change improve our baseline score, or does it hurt accuracy?"*

---

### Experiment 1: Low-Pass Butterworth Filtering (Noise vs Signal)
* **Hypothesis**: Cheap MEMS sensors have high-frequency electrical jitter (> 20 Hz).
* **The Test**:
  * Apply a 4th-order Butterworth low-pass filter with a cutoff at 15 Hz or 20 Hz.
  * Compare validation loss and trajectory drift before vs after.
* **Success Metric**: Does velocity error decrease? If human impact dynamics get blurred and error increases, discard the filter.

---

### Experiment 2: Allan Variance Sensor Noise Simulation (Data Augmentation)
* **Hypothesis**: In the real world, different phones have different noise levels.
* **The Test**:
  * During training, randomly add synthetic white noise and simulated gyroscope bias drift to each training window on the fly.
* **Success Metric**: Does the model generalize better to unseen walking recordings with higher sensor noise?

---

### Experiment 3: Random 3D Yaw Augmentation
* **Hypothesis**: Even with HIF alignment, humans sway at different angles.
* **The Test**:
  * Randomly rotate the horizontal plane by a random angle $\theta \in [-180^\circ, +180^\circ]$ during training batches.
* **Success Metric**: Forces the neural net to become 100% invariant to phone orientation.

---

### Experiment 4: Hyperparameter Optimization (Optuna HPO)
* **Knobs to Auto-Tune**:
  * Kernel Size: Test 5 vs 7 vs 9 vs 11.
  * Base Channels: Test 32 vs 64 vs 128.
  * Window Length: Test 100 steps (0.5s) vs 200 steps (1.0s) vs 300 steps (1.5s).
  * Learning Rate: Optuna search from $10^{-4}$ to $10^{-2}$.
* **Success Metric**: Find the optimal combination that minimizes Absolute Trajectory Error (ATE).

---

### Experiment 5: Training on the Real 40-Hour RoNIN Dataset
* Ingest real walking recordings across different carrying modes:
  * Handheld (typing / texting)
  * In Pocket (tight jeans vs loose trousers)
  * In Bag (backpack / handbag)
  * Free-Swinging (holding by side)
* Validate zero data leakage via **Group-K-Fold by Subject ID**.

---

### The Evaluation Scorecard (The Final Test)

For every experiment, we record 3 numbers:

| Experiment | Velocity RMSE (m/s) | ATE (Meters / 100m) | Inference Time (ms) | Decision |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline (1D-ResNet)** | 0.082 m/s | 1.65 m | 2.1 ms | **Benchmark established** |
| + Butterworth Filter | *Measure* | *Measure* | *Measure* | Keep / Discard |
| + Yaw Augmentation | *Measure* | *Measure* | *Measure* | Keep / Discard |
| + Optuna Tuning | *Measure* | *Measure* | *Measure* | Keep / Discard |
