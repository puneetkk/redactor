import fitz  # PyMuPDF
import re
import spacy
import pytesseract
from PIL import Image
import io

# Load the spacy model for NER
nlp = spacy.load("en_core_web_md")  # Use a larger model for better accuracy

# List of common words that should not be redacted
common_words = set([
    "Solution", "Drink", "Electrolyte", "Water", "Product", "Ingredient", "Email", "Phone"
])

def is_valid_name(name):
    # Check if the name contains only alphabetic characters and is not a common word
    return all(word.isalpha() and word not in common_words for word in name.split())

def is_dob_context(text, match):
    # Check if the date is mentioned in the context of a DOB
    context_keywords = ["birth", "born", "dob", "date of birth"]
    context_window = 20  # Number of characters to check around the match
    start = max(0, match.start() - context_window)
    end = min(len(text), match.end() + context_window)
    context = text[start:end].lower()
    return any(keyword in context for keyword in context_keywords)

def extract_text_with_ocr(document, page):
    # Extract images from the page
    images = page.get_images(full=True)
    text = ""
    for img_index, img in enumerate(images):
        xref = img[0]
        base_image = document.extract_image(xref)
        image_bytes = base_image["image"]
        image = Image.open(io.BytesIO(image_bytes))
        text += pytesseract.image_to_string(image)
    return text

def redact_pii_from_pdf(input_pdf_path, output_pdf_path):
    # Define regex patterns for PII
    patterns = {
        'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
        'phone': re.compile(r'\b\d{3}[-.\s]??\d{3}[-.\s]??\d{4}\b'),
        'ssn': re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
        'dob': re.compile(
            r'\b(\d{1,2}(?:st|nd|rd|th)?[-/\s]?(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[-/\s]?\d{2,4})\b|'  # Alphanumeric dates like 24th Jan 1982
            r'\b(\d{1,2}[-/\s]\d{1,2}[-/\s]\d{2,4})\b|'  # Dates like 01/01/2000 or 1-1-2000
            r'\b(\d{4}[-/\s]\d{1,2}[-/\s]\d{1,2})\b',  # Dates like 2000-01-01
            re.IGNORECASE
        ),
        'nhs': re.compile(r'\b\d{3}[-\s]?\d{3}[-\s]?\d{4}\b')  # NHS numbers like 123 456 7890
    }

    # Open the PDF
    document = fitz.open(input_pdf_path)

    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text = page.get_text("text")

        # Extract text using OCR for handwritten text
        ocr_text = extract_text_with_ocr(document, page)
        text += ocr_text

        # Redact based on regex patterns
        for label, pattern in patterns.items():
            matches = pattern.finditer(text)
            for match in matches:
                if label == 'dob' and not is_dob_context(text, match):
                    continue  # Skip dates that are not in the context of DOB
                redaction = "REDACTED"
                text_instances = page.search_for(match.group())
                for inst in text_instances:
                    page.add_redact_annot(inst, redaction, fill=(0, 0, 0))
        
        # Redact names using spacy
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                # Post-process the identified name
                name = ent.text.strip()
                if is_valid_name(name):
                    text_instances = page.search_for(name)
                    for inst in text_instances:
                        page.add_redact_annot(inst, "REDACTED", fill=(0, 0, 0))
        
        page.apply_redactions()

    # Save the redacted PDF
    document.save(output_pdf_path)

# Example usage
input_pdf_path = "ocrinput.pdf"
output_pdf_path = "ocrredacted_output.pdf"
redact_pii_from_pdf(input_pdf_path, output_pdf_path)