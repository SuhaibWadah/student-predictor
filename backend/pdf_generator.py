"""
PDF Generation Module
Generates PDF documents for student improvement plans.
"""

import os
from datetime import datetime
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
import logging

logger = logging.getLogger(__name__)


class ImprovementPlanPDFGenerator:
    """
    Generates PDF documents for student improvement plans.
    Creates professional, formatted PDF files with student information and improvement strategies.
    """
    
    # PDF Configuration
    PAGE_SIZE = letter
    LEFT_MARGIN = 0.75 * inch
    RIGHT_MARGIN = 0.75 * inch
    TOP_MARGIN = 0.75 * inch
    BOTTOM_MARGIN = 0.75 * inch
    
    def __init__(self, output_dir: str = 'pdfs'):
        """
        Initialize PDF generator.
        
        Args:
            output_dir: Directory to save generated PDFs
        """
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    @staticmethod
    def _get_styles():
        """
        Get custom paragraph styles for the PDF.
        
        Returns:
            dict: Dictionary of custom styles
        """
        styles = getSampleStyleSheet()
        
        # Title style
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#667eea'),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        
        # Heading style
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#667eea'),
            spaceAfter=12,
            spaceBefore=12,
            fontName='Helvetica-Bold'
        )
        
        # Body text style
        body_style = ParagraphStyle(
            'CustomBody',
            parent=styles['BodyText'],
            fontSize=11,
            alignment=TA_JUSTIFY,
            spaceAfter=12,
            leading=14
        )
        
        # Subheading style
        subheading_style = ParagraphStyle(
            'CustomSubHeading',
            parent=styles['Heading3'],
            fontSize=12,
            textColor=colors.HexColor('#333333'),
            spaceAfter=10,
            spaceBefore=10,
            fontName='Helvetica-Bold'
        )
        
        return {
            'title': title_style,
            'heading': heading_style,
            'body': body_style,
            'subheading': subheading_style
        }
    
    def generate_improvement_plan_pdf(
        self,
        student_name: str,
        year: int,
        semester: int,
        predicted_score: float,
        improvement_plan: str,
        features: dict = None
    ) -> tuple[str, str]:
        """
        Generate a PDF document for the improvement plan.
        
        Args:
            student_name: Name of the student
            year: Academic year
            semester: Semester number
            predicted_score: Predicted performance score
            improvement_plan: Text content of the improvement plan
            features: Dictionary of student features (optional)
            
        Returns:
            tuple: (pdf_filename, error_message)
        """
        try:
            # Generate filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            safe_name = student_name.replace(' ', '_').lower()
            filename = f"{safe_name}_{timestamp}_improvement_plan.pdf"
            filepath = os.path.join(self.output_dir, filename)
            
            # Create PDF document
            doc = SimpleDocTemplate(
                filepath,
                pagesize=self.PAGE_SIZE,
                leftMargin=self.LEFT_MARGIN,
                rightMargin=self.RIGHT_MARGIN,
                topMargin=self.TOP_MARGIN,
                bottomMargin=self.BOTTOM_MARGIN
            )
            
            # Get styles
            styles = self._get_styles()
            
            # Build document content
            content = []
            
            # Title
            content.append(Paragraph("Student Improvement Plan", styles['title']))
            content.append(Spacer(1, 0.3 * inch))
            
            # Student Information Section
            content.append(Paragraph("Student Information", styles['heading']))
            
            # Create student info table
            student_info_data = [
                ['Name:', student_name],
                ['Academic Year:', str(year)],
                ['Semester:', str(semester)],
                ['Predicted Performance Score:', f"{predicted_score:.2f}/100"],
                ['Generated Date:', datetime.now().strftime('%B %d, %Y')]
            ]
            
            student_table = Table(student_info_data, colWidths=[2 * inch, 4 * inch])
            student_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f0f0f0')),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 11),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey)
            ]))
            
            content.append(student_table)
            content.append(Spacer(1, 0.3 * inch))
            
            # Student Features Section (if provided)
            if features:
                content.append(Paragraph("Current Performance Indicators", styles['heading']))
                
                features_data = [['Indicator', 'Value']]
                for key, value in features.items():
                    # Format feature names (replace underscores with spaces, capitalize)
                    formatted_key = key.replace('_', ' ').title()
                    features_data.append([formatted_key, str(value)])
                
                features_table = Table(features_data, colWidths=[3 * inch, 3 * inch])
                features_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
                    ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f9f9f9')])
                ]))
                
                content.append(features_table)
                content.append(Spacer(1, 0.3 * inch))
            
            # Improvement Plan Section
            content.append(Paragraph("Personalized Improvement Plan", styles['heading']))
            content.append(Spacer(1, 0.1 * inch))
            
            # Format improvement plan text
            plan_paragraphs = improvement_plan.split('\n\n')
            for para in plan_paragraphs:
                if para.strip():
                    # Check if it's a heading (ends with colon)
                    if para.strip().endswith(':'):
                        content.append(Paragraph(para.strip(), styles['subheading']))
                    else:
                        content.append(Paragraph(para.strip(), styles['body']))
            
            content.append(Spacer(1, 0.3 * inch))
            
            # Footer
            content.append(Spacer(1, 0.2 * inch))
            footer_text = "This improvement plan was generated using the Student Performance Predictor system."
            content.append(Paragraph(
                f"<i>{footer_text}</i>",
                ParagraphStyle(
                    'Footer',
                    parent=getSampleStyleSheet()['Normal'],
                    fontSize=9,
                    textColor=colors.grey,
                    alignment=TA_CENTER
                )
            ))
            
            # Build PDF
            doc.build(content)
            
            logger.info(f"PDF generated successfully: {filename}")
            return filename, None
            
        except Exception as e:
            error_msg = f"Error generating PDF: {str(e)}"
            logger.error(error_msg)
            return None, error_msg
    
    @staticmethod
    def get_pdf_path(pdf_filename: str, output_dir: str = 'pdfs') -> str:
        """
        Get the full path to a generated PDF file.
        
        Args:
            pdf_filename: Name of the PDF file
            output_dir: Output directory
            
        Returns:
            str: Full path to the PDF file
        """
        return os.path.join(output_dir, pdf_filename)


def generate_batch_pdf_report(
    predictions: list,
    output_filename: str = None
) -> tuple[str, str]:
    """
    Generate a comprehensive PDF report for batch predictions.
    
    Args:
        predictions: List of prediction dictionaries
        output_filename: Custom output filename (optional)
        
    Returns:
        tuple: (pdf_filename, error_message)
    """
    try:
        # Generate filename
        if not output_filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_filename = f"batch_predictions_{timestamp}.pdf"
        
        filepath = os.path.join('pdfs', output_filename)
        
        # Create output directory if needed
        os.makedirs('pdfs', exist_ok=True)
        
        # Create PDF document
        doc = SimpleDocTemplate(
            filepath,
            pagesize=letter,
            leftMargin=0.5 * inch,
            rightMargin=0.5 * inch,
            topMargin=0.5 * inch,
            bottomMargin=0.5 * inch
        )
        
        # Get styles
        styles = getSampleStyleSheet()
        
        # Build content
        content = []
        
        # Title
        title_style = ParagraphStyle(
            'Title',
            parent=styles['Heading1'],
            fontSize=20,
            textColor=colors.HexColor('#667eea'),
            spaceAfter=20,
            alignment=TA_CENTER
        )
        content.append(Paragraph("Batch Prediction Report", title_style))
        content.append(Spacer(1, 0.2 * inch))
        
        # Report metadata
        meta_style = ParagraphStyle(
            'Meta',
            parent=styles['Normal'],
            fontSize=10,
            textColor=colors.grey
        )
        content.append(Paragraph(
            f"Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}<br/>Total Records: {len(predictions)}",
            meta_style
        ))
        content.append(Spacer(1, 0.2 * inch))
        
        # Results table
        table_data = [['Name', 'Year', 'Semester', 'Predicted Score']]
        
        for pred in predictions:
            table_data.append([
                pred.get('name', ''),
                str(pred.get('year', '')),
                str(pred.get('semester', '')),
                f"{pred.get('predicted_performance', 0):.2f}"
            ])
        
        results_table = Table(table_data, colWidths=[2.5 * inch, 1 * inch, 1 * inch, 1.5 * inch])
        results_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f9f9f9')])
        ]))
        
        content.append(results_table)
        
        # Build PDF
        doc.build(content)
        
        logger.info(f"Batch report PDF generated: {output_filename}")
        return output_filename, None
        
    except Exception as e:
        error_msg = f"Error generating batch report: {str(e)}"
        logger.error(error_msg)
        return None, error_msg
