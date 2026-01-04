import json
import csv
from io import StringIO, BytesIO
from datetime import datetime
import os

def export_as_json(data):
    """Export data as JSON string"""
    return json.dumps(data, indent=2)

def export_as_markdown(segments, summary, metadata=None):
    """Export as formatted Markdown"""
    md = "# Meeting Summary\n\n"

    if metadata:
        md += f"**Date:** {metadata.get('date', 'N/A')}\n"
        md += f"**Duration:** {metadata.get('duration', 'N/A')}\n"
        md += f"**Speakers:** {metadata.get('num_speakers', 'N/A')}\n\n"

    md += "---\n\n## Summary\n\n"
    md += summary + "\n\n"
    md += "---\n\n## Full Transcript\n\n"

    for s in segments:
        md += f"### {s['speaker']} ({s['start']:.1f}s - {s['end']:.1f}s)\n\n"
        md += f"{s['text']}\n\n"

    return md

def export_as_pdf(segments, summary, metadata=None):
    """Export as PDF using reportlab"""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
        from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER

        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter,
                              rightMargin=72, leftMargin=72,
                              topMargin=72, bottomMargin=18)

        # Container for PDF elements
        elements = []
        styles = getSampleStyleSheet()

        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor='#0066cc',
            spaceAfter=30,
            alignment=TA_CENTER
        )

        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            textColor='#0066cc',
            spaceAfter=12,
        )

        # Title
        elements.append(Paragraph("Meeting Summary", title_style))
        elements.append(Spacer(1, 12))

        # Metadata
        if metadata:
            meta_text = f"""
            <b>Date:</b> {metadata.get('date', 'N/A')}<br/>
            <b>Duration:</b> {metadata.get('duration', 'N/A')}<br/>
            <b>Speakers:</b> {metadata.get('num_speakers', 'N/A')}<br/>
            <b>Model:</b> {metadata.get('model', 'N/A')}
            """
            elements.append(Paragraph(meta_text, styles['Normal']))
            elements.append(Spacer(1, 20))

        # Summary section
        elements.append(Paragraph("Executive Summary", heading_style))
        elements.append(Paragraph(summary, styles['BodyText']))
        elements.append(Spacer(1, 20))

        # Transcript section
        elements.append(PageBreak())
        elements.append(Paragraph("Full Transcript", heading_style))
        elements.append(Spacer(1, 12))

        for seg in segments:
            speaker_text = f"<b>{seg['speaker']}</b> ({seg['start']:.1f}s - {seg['end']:.1f}s)"
            elements.append(Paragraph(speaker_text, styles['Heading3']))
            elements.append(Paragraph(seg['text'], styles['BodyText']))
            elements.append(Spacer(1, 12))

        # Build PDF
        doc.build(elements)
        pdf_data = buffer.getvalue()
        buffer.close()

        return pdf_data

    except ImportError:
        print("ERROR: Install reportlab with: pip install reportlab")
        return None
    except Exception as e:
        print(f"PDF generation error: {e}")
        return None

def export_as_csv(segments):
    """Export segments as CSV"""
    buf = StringIO()
    writer = csv.DictWriter(buf, fieldnames=["speaker", "start", "end", "text"])
    writer.writeheader()
    for s in segments:
        writer.writerow(s)
    return buf.getvalue()

def send_email_summary(recipient, subject, body, attachments=None, smtp_config=None):
    """Send meeting summary via email using smtplib"""
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    from email.mime.application import MIMEApplication

    if smtp_config is None:
        smtp_config = {
            'host': os.getenv('SMTP_HOST', 'smtp.gmail.com'),
            'port': int(os.getenv('SMTP_PORT', 587)),
            'username': os.getenv('SMTP_USERNAME'),
            'password': os.getenv('SMTP_PASSWORD')
        }

    try:
        # Create message
        msg = MIMEMultipart()
        msg['From'] = smtp_config['username']
        msg['To'] = recipient
        msg['Subject'] = subject

        # Add body
        msg.attach(MIMEText(body, 'html'))

        # Add attachments
        if attachments:
            for filename, data in attachments.items():
                attachment = MIMEApplication(data)
                attachment.add_header('Content-Disposition', 'attachment', filename=filename)
                msg.attach(attachment)

        # Send email
        with smtplib.SMTP(smtp_config['host'], smtp_config['port']) as server:
            server.starttls()
            server.login(smtp_config['username'], smtp_config['password'])
            server.send_message(msg)

        return True, "Email sent successfully"

    except Exception as e:
        return False, f"Email error: {str(e)}"

def create_email_body(summary, segments, metadata):
    """Create HTML email body"""
    html = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; }}
            .header {{ background-color: #0066cc; color: white; padding: 20px; }}
            .summary {{ background-color: #f0f0f0; padding: 15px; margin: 20px 0; }}
            .speaker {{ color: #0066cc; font-weight: bold; }}
            .timestamp {{ color: #666; font-size: 0.9em; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Meeting Summary</h1>
            <p>Date: {metadata.get('date', 'N/A')}</p>
        </div>

        <div class="summary">
            <h2>Executive Summary</h2>
            <p>{summary}</p>
        </div>

        <h2>Transcript Highlights</h2>
        {''.join([f'<p><span class="speaker">{s["speaker"]}</span> <span class="timestamp">({s["start"]:.1f}s)</span>: {s["text"][:200]}...</p>' for s in segments[:5]])}

        <p><i>Full transcript and detailed summary are attached.</i></p>
    </body>
    </html>
    """
    return html
