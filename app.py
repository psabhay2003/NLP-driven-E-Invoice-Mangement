from flask import Flask, request, render_template, flash, redirect, url_for, send_file
import os
import csv
import uuid
import threading
import time
from werkzeug.utils import secure_filename
from ocr_extract import extract_text_from_image
from field_extractor import extract_invoice_fields

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "your-secret-key")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def delayed_delete(path, delay=1200):
    def delete():
        time.sleep(delay)
        if os.path.exists(path):
            try:
                os.remove(path)
            except Exception as e:
                print(f"Failed to delete {path}: {e}")
    threading.Thread(target=delete, daemon=True).start()


@app.route("/", methods=["GET", "POST"])
def index():
    all_results = []
    csv_filename = None

    if request.method == "POST":
        images = request.files.getlist("image")
        if not images:
            flash("❌ No files uploaded.", "error")
            return redirect(url_for("index"))

        batch_id = uuid.uuid4().hex
        csv_filename = f"invoices_{batch_id}.csv"
        csv_path = os.path.join(UPLOAD_FOLDER, csv_filename)

        with open(csv_path, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["filename", "Invoice No", "Date", "Total Amount", "Vendor"])

            for image in images:
                filename = secure_filename(image.filename)
                if not filename:
                    continue

                image_path = os.path.join(UPLOAD_FOLDER, filename)
                image.save(image_path)

                try:
                    raw_text = extract_text_from_image(image_path)
                    result = extract_invoice_fields(raw_text)

                    all_results.append({"filename": filename, "fields": result})

                    writer.writerow([
                        filename,
                        result.get("Invoice No", ""),
                        result.get("Date", ""),
                        result.get("Total Amount", ""),
                        result.get("Vendor", "")
                    ])
                except Exception as e:
                    flash(f"❌ Error processing {filename}: {str(e)}", "error")
                finally:
                    # Delete image after 5 minutes
                    delayed_delete(image_path)

        return render_template("index.html", all_results=all_results, csv_filename=csv_filename)

    return render_template("index.html")


@app.route("/download/<filename>")
def download_csv(filename):
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    if os.path.exists(filepath):
        return send_file(filepath, as_attachment=True)
    flash("❌ File not found.", "error")
    return redirect(url_for("index"))


@app.route("/append-to-existing", methods=["POST"])
def append_to_existing_csv():
    csv_filename = request.form.get("csv_filename")
    new_csv_path = os.path.join(UPLOAD_FOLDER, csv_filename)
    existing_csv_path = os.path.join(UPLOAD_FOLDER, "invoices.csv")

    if not csv_filename or not os.path.exists(new_csv_path):
        flash("❌ Cannot append. File does not exist.", "error")
        return redirect(url_for("index"))

    # Create the base file if not exists
    if not os.path.exists(existing_csv_path):
        with open(existing_csv_path, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["filename", "Invoice No", "Date", "Total Amount", "Vendor"])

    try:
        with open(new_csv_path, "r") as new_file, open(existing_csv_path, "a", newline="") as existing_file:
            reader = csv.reader(new_file)
            writer = csv.writer(existing_file)
            next(reader, None)  # skip header
            for row in reader:
                writer.writerow(row)

        flash("✅ Appended to invoices.csv successfully!", "success")
        return send_file(existing_csv_path, as_attachment=True)

    except Exception as e:
        flash(f"❌ Failed to append: {str(e)}", "error")
        return redirect(url_for("index"))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
