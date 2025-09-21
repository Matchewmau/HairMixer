import logging
import json
import uuid
from django.utils.deprecation import MiddlewareMixin


class RequestIDMiddleware(MiddlewareMixin):
    header_name = 'HTTP_X_REQUEST_ID'
    attr_name = 'request_id'

    def process_request(self, request):
        rid = request.META.get(self.header_name) or str(uuid.uuid4())
        request.request_id = rid
        return None

    def process_response(self, request, response):
        rid = getattr(request, 'request_id', None)
        if rid:
            response['X-Request-ID'] = rid
        return response


class RequestIDFilter(logging.Filter):
    def filter(self, record):
        # Ensure every log record has request_id to satisfy the formatter
        if not hasattr(record, 'request_id') or record.request_id is None:
            record.request_id = '-'
        return True


class JSONLogFormatter(logging.Formatter):
    def format(self, record):
        base = {
            'time': self.formatTime(record, self.datefmt),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
        }
        # Optional extras
        rid = getattr(record, 'request_id', None)
        if rid:
            base['request_id'] = rid
        pathname = getattr(record, 'pathname', None)
        if pathname:
            base['path'] = pathname
        lineno = getattr(record, 'lineno', None)
        if lineno:
            base['line'] = lineno
        return json.dumps(base)
