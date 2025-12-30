# Rule-Based Vision Algorithm Platform

## 📋 Overview
This project is a scalable **C++ based Computer Vision Platform** designed for industrial inspection tasks. It utilizes **OpenCV** and efficient rule-based logic to perform high-precision image analysis.

The platform is architected to be **modular**, allowing easy integration of various inspection algorithms (e.g., Contour Inspection, Barcode Reading, OCR) into a single execution pipeline.

---

## 🏗️ Project Structure
The project follows a modular design pattern to separate common utilities, specific algorithm logic, and the execution entry point.

```text
root/
├── CommonUtils.h        # Shared utilities (Logger, Image I/O, Spline Interpolation)
├── KBRulebase.h         # Header for KB Inspection Algorithm
├── KBRulebase.cpp       # Implementation of KB Inspection Logic (Encapsulated)
├── main.cpp             # Main entry point (Controller)
└── README.md            # Project documentation
