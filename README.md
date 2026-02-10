
# GenAI Benchmark: GAN, VAE, Diffusion on MNIST/CIFAR-10

This project benchmarks generative models — **GAN**, **VAE**, and **Diffusion Models** — on standard image datasets like **MNIST** and **CIFAR-10**.

---

## 📁 Project Structure

```
genai_benchmark/
├── train/
│   ├── train_gan.py         # Train script for GAN on MNIST
│   ├── train_vae.py         # (Optional) Train script for VAE
│   └── train_diffusion.py   # (Optional) Train script for Diffusion
├── models/
│   ├── gan.py               # Contains Generator, Discriminator, train_gan()
│   ├── vae.py
│   └── diffusion.py
├── datasets/
│   └── loader.py            # Dataset loaders (MNIST, CIFAR-10)
├── outputs/
│   ├── samples/             # Generated sample images
│   ├── checkpoints/         # Saved model weights
│   ├── logs/                # Training logs
│   └── fid_samples/         # FID evaluation images
├── benchmark.ipynb          # Evaluation notebook
└── README.md
```

---

## 🛠️ Setup

Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install torch torchvision matplotlib tqdm seaborn
```

---

## 🚀 Training a GAN on MNIST

```bash
cd /path/to/genai_benchmark
python -m train.train_gan
```

This will:
- Load MNIST dataset
- Train the GAN for 50 epochs
- Save model checkpoints in `outputs/checkpoints/`
- Save sample generated images in `outputs/samples/`

---

## 🧪 Benchmarking

Use `benchmark.ipynb` to:
- Visualize generated samples from all models
- Plot training losses and FID score
- Compare model performance

---

## ⚙️ Model Hyperparameters

You can adjust hyperparameters like:
- `z_dim` (latent vector size)
- `lr` (learning rate)
- `num_epochs`
- `batch_size`

in the corresponding train scripts (e.g., `train_gan.py`).

---

## Notes

- All outputs are saved under the `outputs/` directory.
- Add `__init__.py` in folders if running as a module.
- Make sure to run training scripts **from the project root** for relative imports to work.

