import os

from django.http import HttpResponse, Http404


def open_text_file(file_path, file_name):
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        raise Http404(f"Text file '{file_name}' not found")

    if not file_name.endswith('.txt'):
        raise Http404("Only text files are supported")

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        response = HttpResponse(content, content_type='text/plain')

        # Set Content-Disposition header for displaying in browser
        response['Content-Disposition'] = f'inline; filename="{file_name}"'

        return response
    except Exception as e:
        raise Http404(f"Error opening text file: {str(e)}")
