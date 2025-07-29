# NLP-driven-E-Invoice-Management
## Disclaimer:
This project is inspired by a research paper [FATURA: A Multi-Layout Invoice Image Dataset for Document Analysis and Understanding](https://arxiv.org/abs/2311.11856).

The project aims to automate the entire ETL pipeline such that whenever a user uploads an invoice image, the system automatically extracts the meaningful information, stores it and updates the existing csv which can be used to integrate in the SQL database as per user needs.
### Optical Character Recognition (Phase 1):
- OCR is used in Phase 1 of the project to extract machine-encoded text from images in the form of words and bounding boxes.
* The input data consists of 2000 images which I made available on my [Google Drive](https://drive.google.com/drive/folders/1yqt-ZLTuOulB_pD0jDSiM6N-LlfayzAz?usp=drive_link).
+ The python library used for OCR is [Pytesseract](https://pypi.org/project/pytesseract/).
### Text-to-Text Transfer Transformer (Phase 2):
- The "invoice_text.csv" file was obtained as an output from Phase 1.
* A "t5-base" model was fine-tuned as per the use case to organize the unstructured OCR text from "invoice_text.csv" into a structured manner using the "labelled data sample.csv".
+ The output from fine-tuning the "t5-base" model was parsed using regular expressions (re) to cover different labels (such as Invoice No., Date, Total Amount, and Vendor).
### Model Deployment using Flask API(Phase 3):
