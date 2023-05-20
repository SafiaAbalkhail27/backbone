# forms.py
from django import forms

class UploadFileForm(forms.Form):
    ALLOWED_EXTENSIONS = ['pdf', 'docx', 'txt', 'csv']  # List of allowed extensions
    
    file = forms.FileField()
    
    def clean_file(self):
        uploaded_file = self.cleaned_data['file']
        extension = uploaded_file.name.split('.')[-1].lower()  # Get the file extension
        
        if extension not in self.ALLOWED_EXTENSIONS:
            raise forms.ValidationError(f'Only files with extensions {", ".join(self.ALLOWED_EXTENSIONS)} are allowed.')
        
        return uploaded_file