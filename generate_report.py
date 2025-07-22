import openai
import os
from dotenv import load_dotenv
import json
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas

# Load API Key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Read events
def load_events(file_path):
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print("Error reading log file:", e)
        return []

# Generate report from events
def generate_report(events):
    timeline = "\n".join([f'{e["timestamp"]}: {e["event"]}' for e in events])
    prompt = (
        "You are an exam proctor. Analyze the following timeline of exam events "
        "and write a short report:\n\n" + timeline + "\n\nReport:"
    )

    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You generate concise, professional exam monitoring reports."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.4
    )

    return response.choices[0].message.content

# Save report to text and PDF
def save_report(report_text):
    os.makedirs("report", exist_ok=True)

    # Save as text
    with open("report/final_report.txt", "w") as f:
        f.write(report_text)

    # Save as PDF
    pdf_path = "report/final_report.pdf"
    c = canvas.Canvas(pdf_path, pagesize=LETTER)
    width, height = LETTER

    lines = report_text.split("\n")
    y = height - 72  # Start from top (1 inch margin)

    for line in lines:
        c.drawString(72, y, line)
        y -= 18  # Move down for next line

        if y < 72:  # Create a new page if needed
            c.showPage()
            y = height - 72

    c.save()

# Main
if __name__ == "__main__":
    events = load_events("logs/events.json")
    if events:
        report = generate_report(events)
        print("\n=== Final Report ===\n")
        print(report)

        save_report(report)
        print("\nReport saved to:")
        print(" - report/final_report.txt")
        print(" - report/final_report.pdf")
    else:
        print("No events found in logs.")
