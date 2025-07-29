from flask import Flask, request, render_template, flash, redirect, url_for, send_file
import os
import csv
import uuid
from ocr_extract import extract_text_from_image     # OCR Phase
from field_extractor import extract_invoice_fields  # Model Phase

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "your-secret-key")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    all_results = []
    csv_filename = None

    if request.method == "POST":
        images = request.files.getlist("image")
        if not images:
            flash("No files uploaded.", "error")
            return redirect(url_for("index"))

        batch_id = uuid.uuid4().hex
        csv_filename = f"invoices_{batch_id}.csv"
        csv_path = os.path.join(UPLOAD_FOLDER, csv_filename)

        with open(csv_path, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["filename", "Invoice No", "Date", "Total Amount", "Vendor"])

            for image in images:
                image_path = os.path.join(UPLOAD_FOLDER, image.filename)
                image.save(image_path)

                raw_text = extract_text_from_image(image_path)
                result = extract_invoice_fields(raw_text)

                all_results.append({"filename": image.filename, "fields": result})

                writer.writerow([
                    image.filename,
                    result.get("Invoice No", ""),
                    result.get("Date", ""),
                    result.get("Total Amount", ""),
                    result.get("Vendor", "")
                ])

                os.remove(image_path)

        return render_template("index.html", all_results=all_results, csv_filename=csv_filename)

    return render_template("index.html")


@app.route("/download/<filename>")
def download_csv(filename):
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    if os.path.exists(filepath):
        return send_file(filepath, as_attachment=True)
    return "File not found.", 404


@app.route("/append-to-existing", methods=["POST"])
def append_to_existing_csv():
    csv_filename = request.form.get("csv_filename")
    new_csv_path = os.path.join(UPLOAD_FOLDER, csv_filename)
    existing_csv_path = os.path.join(UPLOAD_FOLDER, "invoices.csv")

    if not os.path.exists(new_csv_path):
        flash("❌ No generated CSV to append.", "error")
        return redirect(url_for("index"))

    # Create base CSV file if not exists
    if not os.path.exists(existing_csv_path):
        with open(existing_csv_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["filename", "Invoice No", "Date", "Total Amount", "Vendor"])

    # Append content
    with open(new_csv_path, 'r') as new_file, open(existing_csv_path, 'a', newline='') as existing_file:
        reader = csv.reader(new_file)
        writer = csv.writer(existing_file)
        next(reader)  # Skip header
        for row in reader:
            writer.writerow(row)

    flash("✅ Appended to invoices.csv successfully!", "success")
    return send_file(existing_csv_path, as_attachment=True)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Required for Render
    app.run(debug=False, host="0.0.0.0", port=port)
