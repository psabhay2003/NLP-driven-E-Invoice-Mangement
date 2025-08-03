# NLP-driven-E-Invoice-Management
## Disclaimer:
This project is inspired by a research paper [FATURA: A Multi-Layout Invoice Image Dataset for Document Analysis and Understanding](https://arxiv.org/abs/2311.11856).

With this project, I aim to automate the entire pipeline so that whenever a user uploads an invoice image, the system automatically extracts the meaningful information, stores it, and updates the existing CSV, which can be used to integrate into the SQL database as per the user needs.
The project reduces manual data entry by 95%, considering it took me 60 minutes to manually enter data for 200 invoice images in "labelled data sample.csv" and it took around 30 minutes for the model to enter the data for 2000 invoice images, which computes to a 95% reduction in data entry time.
### Optical Character Recognition (Phase 1):
- OCR is used in Phase 1 of the project to extract machine-encoded text from images in the form of words and bounding boxes.
* The input data consists of 2000 images, which I made available on my [Google Drive](https://drive.google.com/drive/folders/1yqt-ZLTuOulB_pD0jDSiM6N-LlfayzAz?usp=drive_link).
+ The python library used for OCR is [Pytesseract](https://pypi.org/project/pytesseract/).
### Text-to-Text Transfer Transformer (Phase 2):
- The "invoice_text.csv" file was obtained as an output from Phase 1.
* A "t5-base" model was fine-tuned as per the use case to organize the unstructured OCR text from "invoice_text.csv" into a structured manner using the "labelled data sample.csv".
+ The output from fine-tuning the "t5-base" model was parsed using regular expressions (re) to cover different labels (such as Invoice No., Date, Total Amount, and Vendor).
### Model Deployment (Phase 3):
-In the deployment phase, the model was deployed to the cloud using a platform named [Railway](https://railway.com/).
+ The app.py file stores the Flask API code, which calls the application package and the required dependencies stored in a virtual machine as a standardized unit called a container. Dockerization is the process of packaging these application files and the requirements.txt into a container for cloud-based hosting. 
* The link to the website is: [NLP-driven E-Invoice Management](https://nlp-driven-e-invoice-mangement-production-6f45.up.railway.app/)
