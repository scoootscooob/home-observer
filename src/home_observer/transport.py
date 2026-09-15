"""Bounded inline media transport for camera capture on a different machine."""
from __future__ import annotations

import base64
import copy
from pathlib import Path

MAX_MEDIA_BYTES = 16 * 1024 * 1024
MAX_BODY_BYTES = 24 * 1024 * 1024
SUFFIXES = {'.jpg', '.jpeg', '.png', '.webp', '.wav', '.flac', '.ogg'}


def media_items(request):
    window = request['window']
    return [*window.get('frames', []), *window.get('audio', [])]


def pack_media(request):
    media, seen, total = [], set(), 0
    for item in media_items(request):
        path = item['path']
        if path in seen:
            continue
        seen.add(path)
        source = Path(path)
        if source.suffix.lower() not in SUFFIXES:
            raise ValueError('unsupported uploaded media extension')
        if not source.is_file() or source.stat().st_size > MAX_MEDIA_BYTES - total:
            raise ValueError('inline media exceeds total byte budget or is not a file')
        with source.open('rb') as stream:
            data = stream.read(MAX_MEDIA_BYTES - total + 1)
        total += len(data)
        if total > MAX_MEDIA_BYTES:
            raise ValueError('inline media exceeds total byte budget')
        media.append({'path': path, 'data': base64.b64encode(data).decode('ascii')})
    return {'request': request, 'media': media}, total


def unpack_media(payload, destination):
    """Map only referenced media to safe temporary filenames, never supplied paths."""
    if not isinstance(payload, dict) or set(payload) != {'request', 'media'}:
        raise ValueError('expected request and media')
    request = copy.deepcopy(payload['request'])
    items = media_items(request)
    expected = {item['path'] for item in items}
    if not isinstance(payload['media'], list) or len(payload['media']) != len(expected):
        raise ValueError('media must exactly cover the request paths')
    paths, total = {}, 0
    for index, media in enumerate(payload['media']):
        if not isinstance(media, dict) or set(media) != {'path', 'data'}:
            raise ValueError('invalid media entry')
        name = media['path']
        if name not in expected or name in paths:
            raise ValueError('unexpected or repeated media path')
        suffix = Path(name).suffix.lower()
        if suffix not in SUFFIXES or not isinstance(media['data'], str):
            raise ValueError('unsupported media extension or encoding')
        data = base64.b64decode(media['data'], validate=True)
        total += len(data)
        if total > MAX_MEDIA_BYTES:
            raise ValueError('inline media exceeds total byte budget')
        path = Path(destination) / f'media-{index}{suffix}'
        path.write_bytes(data)
        paths[name] = str(path)
    for item in items:
        item['path'] = paths[item['path']]
    return request, total
