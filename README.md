Here is the professionally redesigned `README.md` in English. I've structured it to highlight the end-to-end deep learning architecture, the training pipeline, and the comprehensive evaluation metrics, making it highly attractive for an ML portfolio or GitHub repository.

***

# 🔍 License Plate OCR via TrOCR Fine-Tuning

This repository provides an end-to-end Optical Character Recognition (OCR) pipeline for license plates. By replacing brittle, multi-stage traditional OCR systems (which require handcrafted character segmentation) with a modern **Transformer-based Vision-Encoder/Decoder model**, this project achieves robust, sequence-to-sequence plate transcription.

The model is fine-tuned entirely on a diverse mix of synthetic and AI-generated datasets to bridge the domain gap and handle variations in font styles, illumination, blur, and rendering quality.

---

## 🧠 Architecture & Methodology

The core engine relies on a Vision-to-Text Transformer architecture, specifically leveraging transfer learning from a strong pretrained baseline.

* **Base Model:** `microsoft/trocr-small-printed`
* **Architecture (`VisionEncoderDecoder`):**
  * **Encoder:** A Vision Transformer (ViT) that processes the RGB license plate image and converts image patches into visual embeddings.
  * **Decoder:** An autoregressive text decoder that reads the embeddings and generates the plate characters token-by-token.
* **Objective:** Directly predict the full text sequence from the image crop, avoiding the need for individual character bounding boxes.

---

## 📊 Dataset & Preprocessing

The model is trained on a robust, multi-domain dataset to ensure generalization. Labels are automatically extracted from the filenames (e.g., `ABC1234.jpg` $\rightarrow$ Target: `"ABC1234"`).

### Data Domains
Images are aggregated from three distinct sources to simulate real-world variance:
1. `Dataset/synthetic/`: Procedurally generated plates (Python/OpenCV).
2. `Dataset/blender/`: 3D rendered plates with realistic lighting/shadows.
3. `Dataset/Z-Image_AI_generated/`: Diffusion-model generated plates.

### Pipeline Logic
* **Validation:** Filters for valid image extensions (`.jpg`, `.png`, `.jpeg`, `.bmp`, `.webp`).
* **Splitting:** Stratified Train/Test split (`test_size=0.1`) ensuring all three data domains are equally represented in both sets.
* **Tokenization:** Uses `TrOCRProcessor` to extract `pixel_values`. Text targets are tokenized up to `max_target_length=32`, with padding tokens masked using `-100` for loss calculation.

---

## 🚀 Training Pipeline

Training is orchestrated using the Hugging Face `Seq2SeqTrainer` within the `Encoder_Decoder_TrOCR_SynteticData.ipynb` notebook.

**Hyperparameters:**
* **Epochs:** `10`
* **Batch Size:** `55` per device
* **Learning Rate:** `5e-5`
* **Weight Decay:** `0.01`
* **Mixed Precision:** Enabled (`fp16=True`) for VRAM efficiency and speed.
* **Evaluation Strategy:** Step-based (Every `380` steps).
* **Final Checkpoint Output:** `./trocr-plates`

> **Baseline Results:** After 10 epochs (approx. 3183 seconds of runtime), the model reached a `98% accuracy`. Inference tests confirm the sequence-to-sequence pipeline successfully structures outputs, though minor confusion (e.g., predicting `7679799` vs Ground Truth `7809705`) highlights the need for domain-specific fine-tuning.

---

## 🔬 Evaluation Protocol

A standout feature of this repository is its robust custom evaluation suite. Instead of just relying on loss, the notebook implements practical OCR metrics to guide model improvements:

* **Character Error Rate (CER) & Edit Distance:** Measures the minimum number of operations required to change the predicted string into the ground truth.
* **Full-Plate Exact Match:** Strict accuracy measuring perfectly predicted plates.
* **Character-Level Accuracy:** Granular performance tracking.
* **Tolerant Equivalences:** Custom logic to forgive visually identical characters based on domain rules (e.g., `0` vs `O`, `1` vs `I`).
* **Per-Domain Breakdown:** Analyzes accuracy isolated by source folder to detect domain-specific weaknesses.

---

## 🛠️ Quick Start

1. **Setup:** Ensure you have the datasets organized in the `Dataset/` directories as specified above.
2. **Install Dependencies:** `transformers`, `torch`, `Pillow`, `scikit-learn`.
3. **Run Notebook:** Open `Encoder_Decoder_TrOCR_SynteticData.ipynb` and execute the cells sequentially.
4. **Inference:** Once trained, the notebook automatically loads the latest checkpoint from `./trocr-plates` for rapid inference on new images.

---

## 🛤️ Roadmap & Next Improvements

While the current pipeline proves the viability of Seq2Seq plate recognition, future updates will focus on driving down the Character Error Rate:

* **Real-World Integration:** Mix captured, real-world plate crops into the training set to close the synthetic-to-real domain gap.
* **Advanced Decoding:** Implement Beam Search and constrained decoding (enforcing specific license plate regex patterns) during inference.
* **Dynamic Checkpointing:** Track CER during evaluation steps and automatically save the best-performing model, rather than the final step.
* **Error Dashboards:** Build visual confusion matrices to target specific recurrent errors (e.g., `B` vs `8`, `Q` vs `O`).