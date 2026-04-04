# Deepfake Image Detection — Project Context

## Course
CSC 4850 / 6850 — Machine Learning  
Georgia State University  
Instructor: Dong Hye Ye

## Team
- Issac Lara
- Ronald Thorpe
- Kashish Rikhi

---

## Project Summary
Build a machine learning model that classifies a human face image as either **authentic** or **deepfake/manipulated**. Deepfakes are increasingly realistic due to advances in GANs, making automated detection an important problem in computer vision and digital forensics.

---

## Dataset
Primary: **FaceForensics++ dataset** — thousands of labeled real and manipulated facial images generated with various deepfake techniques.  
Supplementary (optional): **Celeb-DF**, **DeepFake Detection Challenge (DFDC)**, or **Kaggle's DeepFake Set** for added diversity and robustness.

---

## Approach

1. **Preprocessing** — Extract facial regions using face detection methods.
2. **Deep Learning Models** — Train CNNs (e.g., ResNet, EfficientNet) to classify real vs. fake.
3. **Traditional ML** — Extract handcrafted features (texture patterns, frequency artifacts, lighting/facial structure inconsistencies) and test classifiers like SVM, Decision Tree, or AdaBoost.
4. **Evaluation Metrics** — Accuracy, Precision, Recall, F1-score.

---

## Milestones
| Week | Task |
|------|------|
| 1 | Identify dataset and perform data preprocessing |
| 2 | Implement face detection and extract facial regions |
| 3 | Train baseline classifiers (CNN or other models) |
| 4 | Evaluate performance and compare models |
| 5 | Prepare presentation and write final report |

---

## Deliverables
- **Oral Presentation** — 15–20 min + 5–10 min Q&A on 4/9, 4/14, 4/16, 4/21, or 4/23 at 5:30 PM
- **Final Report** — Conference-paper style (IEEE template via Overleaf), 4-page limit including title, abstract, and references, due **5/5**
  - Sections: Title, Abstract (≤250 words), Introduction + Literature Review, Methods (with equations), Experimental Results (tables + figures), Conclusion
- **Supplementary** — All source code

---

## Software / Tools
- Python
- PyTorch or TensorFlow
- OpenCV
- NumPy
- Scikit-learn
- Matplotlib

---

## Grading Criteria (40% of final grade)
- **Technical Depth** — How challenging and rigorous is the implementation?
- **Scope** — How many approaches, angles, and variations were explored?
- **Presentation** — Clarity of explanation, quality of visualizations, and writing.